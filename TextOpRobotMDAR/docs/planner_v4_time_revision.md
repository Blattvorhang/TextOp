# Planner V4: Arrival Time Positional Encoding

## Motivation

The current `body_ext` (21-D) goal vector packs a 1-D remaining-time scalar
into dimension 8 of the goal, alongside 3-D root position, 2-D root yaw, 3-D
root velocity, and 12-D limb keypoints:

```
ego_root (3) | ego_yaw (2) | ego_velocity (3) | time (1) | ego_keypoints (12)
```

This single scalar is projected through `nn.Linear(21, h_dim)` together with
all spatial dimensions — the temporal information accounts for 1/21 of the
input variance. The diffusion timestep `t`, by contrast, receives a dedicated
`TimestepEmbedder` (sinusoidal → MLP → full `h_dim`). This asymmetry means
the denoiser can barely perceive *when* the goal should be reached.

We replace the scalar time channel with a dedicated sinusoidal positional
encoding added to the goal token, lifting temporal conditioning to the same
representational bandwidth as the spatial goal.

## Naming Convention

The codebase currently uses three names for the same quantity:

| Name | Location | Problem |
|------|----------|---------|
| `goal_timestep` | data loader, train_dar | Confusable with diffusion timestep `t` |
| `goal_time` | `_mask_goal`, config keys | Underspecified — "time" of what? |
| `timestep` | `build_ego_goal` parameter | Same collision as above |

**Unified model term: *time to arrival*.**

Two representations exist and must be clearly distinguished:

- `time_to_arrival` — continuous time in seconds (e.g., `3.2`).
  Used at the controller interface and in `build_ego_goal`.
- `time_to_arrival_frame` — discrete frame index at 50 Hz (e.g., `160`).
  This is the input to the sinusoidal PE. Computed as
  `int(round(time_to_arrival * motion_fps))` at inference, or
  `goal_frame - reference_frame` during training.
- The embedding module is `ArrivalTimeEmbedder` — it accepts
  `time_to_arrival_frame` and produces the PE vector.
- The legacy `goal_timestep` / `timestep` names remain as compatibility
  aliases for the 21-D `body_ext` layout, but new call sites should carry
  `time_to_arrival` separately.
- The wire protocol field `goal_timestamp_ns` is left unchanged because it is
  a Sonic-level absolute timestamp, not a model-level relative-time name.

"Arrival time" is the standard term in motion planning and trajectory
optimisation. The `_s` / `_frame` suffix convention makes the unit
unambiguous everywhere a variable appears — the reader never needs to
guess whether a value is in seconds or frames.

## Design

### 1. Keep 21-D `body_ext`; carry arrival time separately

For the current V4 branch, `body_ext` keeps its 21-D vector for checkpoint and
caller compatibility:

```python
return torch.cat(
    (ego_root, ego_yaw, ego_velocity, time_to_arrival, ego_keypoints), dim=-1
)  # 21-D
```

The important change is that call sites also carry the real time signal
separately as `time_to_arrival_frame`. The denoiser keeps `goal_dim=21`, zeros
the legacy scalar slot before the goal-content projection, and adds the
sinusoidal arrival PE separately.

The next clean goal type for 29-DoF joint-angle targets can drop this
compatibility path entirely.

### 2. Arrival Time Encoder

A sinusoidal encoding followed by a small MLP, mirroring the existing
`TimestepEmbedder` pattern:

```python
class ArrivalTimeEmbedder(nn.Module):
    """Encode time_to_arrival_frame into an h_dim sinusoidal PE vector."""

    def __init__(self, h_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, h_dim),
        )

    def forward(self, time_to_arrival_frame: torch.Tensor) -> torch.Tensor:
        """
        time_to_arrival_frame: [B] integer frame offset from reference frame
        returns: [B, h_dim]
        """
        emb = timestep_embedding(
            time_to_arrival_frame, self.mlp[0].in_features)
        return self.mlp(emb)
```

`timestep_embedding` is imported from `robotmdar.diffusion.nn` — the same
sinusoidal encoder used for diffusion timesteps. It computes:

```
emb[k]        = cos(arrival * freq_k)
emb[k + half] = sin(arrival * freq_k)
freq_k    = exp(-k * log(max_period) / (h_dim/2))
```

The encoding is **additively combined** with the projected goal content:

```python
# DenoiserTransformer.forward()
goal_for_embedding = goal.clone()             # keep 21-D body_ext shape
goal_for_embedding[:, 8:9] = 0.0              # zero legacy scalar slot once
emb_goal = self.embed_goal(goal_for_embedding)  # [B, h_dim] — "where"
arrival_pe = self.arrival_embedder(time_to_arrival_frame)  # [B, h_dim] — "when"
emb_goal = emb_goal + arrival_pe
```

This follows the standard PE pattern: content embedding + position embedding
= transformer input. The two components are orthogonal — `embed_goal` encodes
the spatial target, `arrival_pe` encodes the temporal offset — and the
transformer learns to attend to both.

The existing `sequence_pos_encoder` (sinusoidal, fixed) continues to mark
the goal token's slot position in the heterogeneous sequence
(`[emb_time|emb_goal|emb_scene|...]`). The two encodings do not conflict:

- `seq_PE[1]` — constant per slot; identifies "this is the goal token"
- `arrival_pe` — varies per sample; encodes *when* this goal should be reached

They sum into the same vector, just as BERT sums token + position + segment
embeddings.

### 3. Time Unit Conversion

The controller sends `goal_timestamp_ns` (absolute nanosecond wall time) and
`timestamps_ns` (per-sample controller timestamps). The planner computes
remaining seconds:

```python
# planner_convert.py (existing logic)
time_to_arrival = max(
    0.0,
    (int(goal_timestamp_ns) - int(timestamps_ns[-1])) / 1e9,
)
```

**For the model, this must be converted to a discrete frame index at 50 Hz:**

```python
time_to_arrival_frame = int(round(time_to_arrival * motion_fps))  # motion_fps = 50
```

This integer is passed to `ArrivalTimeEmbedder`. The rounding quantisation
error at 50 Hz is at most 10 ms — negligible compared to the typical arrival
horizon of hundreds of milliseconds to seconds.

**During training**, the dataset already knows `goal_frame` and
`reference_frame` (both integers). The frame offset is computed directly:

```python
# data.py
def _time_to_arrival(self, reference_frame: int, goal_frame: int) -> Tensor:
    if self.goal_timestep_mode == "zero":
        return torch.zeros(1)
    return torch.tensor([
        max(0.0, (goal_frame - reference_frame) / self.fps)
    ])
```

Training then converts seconds to the frame index before calling the denoiser.
`goal_timestep_mode` is kept as the existing config key, while
`time_to_arrival_mode` is accepted as a clearer alias.

### 4. Integration Point

In `DenoiserTransformer.forward()`:

```python
def forward(self, x_t, timesteps, y=None):
    emb_time = self.embed_timestep(timesteps)          # [1, B, d]

    goal, goal_keep_mask = _mask_goal(self, y['goal'], y)
    y['goal_condition_keep_mask'] = goal_keep_mask
    voxel = self.mask_condition(
        y['voxel'], self.cond_scene_mask_prob, ...)
    emb_scene = self.embed_scene(voxel).unsqueeze(0)

    # ── arrival time PE ──
    time_to_arrival_frame = y['time_to_arrival_frame']  # [B] int
    masked_time, keep = self.mask_condition(
        time_to_arrival_frame.reshape(-1, 1).float(),
        self.cond_goal_time_mask_prob,
        force_mask=y.get('force_drop_arrival_time', False),
        return_keep_mask=True,
    )

    arrival_pe = self.arrival_embedder(masked_time.squeeze(-1))
    arrival_pe = arrival_pe * keep.unsqueeze(-1)        # masks MLP bias too

    goal_for_embedding = goal.clone()                   # still 21-D
    goal_for_embedding[:, 8:9] = 0.0                    # legacy scalar slot
    emb_goal = self.embed_goal(goal_for_embedding)      # [B, d]
    emb_goal = emb_goal + arrival_pe
    emb_goal = emb_goal.unsqueeze(0)                    # [1, B, d]

    emb_history = self.embed_history(
        y['history_motion_normalized']).permute(1, 0, 2)
    emb_noise = self.embed_noise(x_t).permute(1, 0, 2)

    xseq = torch.cat(
        (emb_time, emb_goal, emb_scene, emb_history, emb_noise), dim=0
    )
    xseq = self.sequence_pos_encoder(xseq)
    output = self.seqTransEncoder(xseq)[-self.noise_shape[0]:]
    ...
```

## Goal Dimension: keep `body_ext` at 21-D

The current implementation keeps `body_ext` at 21 dimensions:

```
ego_root(3) | yaw(2) | vel(3) | legacy_time_slot(1) | keypoints(12)
```

The legacy time slot remains in the tensor so existing `body_ext` callers and
21-D checkpoints keep their shape contract. Inside the denoiser, the slot is
zeroed once before `embed_goal`, so the real timing signal comes from the
separate `time_to_arrival_frame → ArrivalTimeEmbedder → arrival_pe` path.

### Constants and Config

| Location | Value |
|----------|-------|
| `GoalType.BODY_EXT.dimension` (goal.py) | 21 |
| `EXTENDED_BODY_GOAL_DIM` (mld_denoiser.py) | 21 |
| `embed_goal` linear layer | `nn.Linear(21, h_dim)` |
| `denoiser.goal_dim` (config yaml) | 21 |

### Masking

`_mask_goal` keeps the existing component masks for the 21-D vector, including
the legacy time slot. The actual PE mask lives in `forward()` and is applied
after `ArrivalTimeEmbedder`, so the MLP bias cannot leak a time condition when
masked:

```python
masked_time, keep = model.mask_condition(
    time_to_arrival_frame.reshape(-1, 1).float(),
    model.cond_goal_time_mask_prob,
    force_mask=force_drop_arrival_time,
    return_keep_mask=True,
)
arrival_pe = model.arrival_embedder(masked_time.squeeze(-1))
arrival_pe = arrival_pe * keep.unsqueeze(-1)
```

### Why Keep 21-D Compatibility Now

The current V4 goal is still `body_ext`; changing it to 20-D would force a
checkpoint and caller migration while the next planned target is a cleaner
29-DoF joint-angle goal type. Keeping `body_ext` 21-D lets this branch add a
proper temporal PE without spending effort on a short-lived intermediate
contract.

## Continuous Time Support

The sinusoidal encoding `timestep_embedding(t, dim)` from `diffusion/nn.py`
**natively supports continuous inputs**. It computes `sin(t · freq_k)` and
`cos(t · freq_k)` with `t` as a `float32` — no lookup table, no integer
quantisation:

```python
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(0, half) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    return embedding
```

This means:

- `time_to_arrival_frame = 10` and `time_to_arrival_frame = 11` produce smoothly
  similar encodings — the model sees temporal proximity, not just categorical
  buckets.
- If a future experiment needs fractional frames (e.g., `10.3` for
  sub-frame precision), the same encoder handles it with no code change.
- The MLP after the sinusoidal transform can learn non-linear functions of
  any continuous arrival time.

Using integer frame indices rather than raw seconds avoids an implicit
frequency scaling by `fps`. The sinusoid periods are designed around
`max_period=10000` and typical frame counts (0–200), which provides good
coverage across the arrival horizon without saturating at low frequencies.

## Masking & Dropout

The PE uses the existing `cond_goal_time_mask_prob` / `force_drop_goal_time`
settings, plus the clearer `force_drop_arrival_time` alias in inference.
When masked, the output `arrival_pe` is multiplied by the keep mask after the
MLP. This prevents learned MLP biases from adding a hidden time condition.

The keep mask `arrival_time_condition_keep_mask` follows the same pattern as
the existing per-component masks.

## Configuration

```yaml
data:
  goal_type: body_ext
  goal_timestep_mode: relative      # existing key
  time_to_arrival_mode: relative    # accepted alias

denoiser:
  goal_dim: 21
  cond_goal_root_mask_prob: 0.1
  cond_goal_yaw_mask_prob: 0.3
  cond_goal_time_mask_prob: 0.1
  cond_goal_body_mask_prob: 0.3
```

## Implementation Plan

### Files to Change

| File | Changes |
|------|---------|
| `robotmdar/model/mld_denoiser.py` | Add `ArrivalTimeEmbedder`; keep 21-D `body_ext`; zero legacy time slot before `embed_goal`; add masked `arrival_pe`; update MLP symmetry |
| `robotmdar/diffusion/nn.py` | No changes (reuse `timestep_embedding`) |
| `robotmdar/dataloader/data.py` | Emit canonical `time_to_arrival` and legacy `goal_timestep`; clamp to non-negative seconds |
| `robotmdar/train/train_dar.py` | `_conditions()` converts `time_to_arrival` seconds to `time_to_arrival_frame` and carries it in `y` |
| `robotmdar/eval/generate_dar.py` | `generate_next_motion` accepts `time_to_arrival_frame`, with a narrow 21-D fallback from `goal[:, 8]` and `val_data.fps` |
| `robotmdar/utils/planner_convert.py` | Clamp negative remaining seconds before building the compatibility goal |
| `robotmdar/planner/planner_dar.py` | Pass `time_to_arrival_frame` through `generate_next_motion` |
| `robotmdar/eval/vis_dar.py` | Dataset eval passes `time_to_arrival_frame` |
| `robotmdar/train/manager.py` | Fill missing `arrival_embedder.*` checkpoint keys from the current model while keeping other state-dict checks strict |

### Backward Compatibility

The 21-D `body_ext` tensor contract is preserved. Older checkpoints that lack
`arrival_embedder.*` parameters can still load: those new keys are initialized
from the current model while all other state-dict keys remain strict. If
resuming training from such a checkpoint, the optimizer state is skipped
because it cannot contain the fresh arrival-embedder parameters.

### Training Data Contract

For `body_ext`, each primitive now carries:

```python
primitive = {
    ...
    'world_goal_pos':        ...,  # [B, 3]
    'world_goal_yaw':        ...,  # [B]
    'world_goal_vel':        ...,  # [B, 3]
    'time_to_arrival':       ...,  # [B, 1] seconds
    'goal_timestep':         ...,  # [B, 1] legacy alias
    'time_to_arrival_frame': ...,  # [B] int, carried in y
    'world_goal_keypoints':  ...,  # [B, 4, 3]
}
```

`time_to_arrival_frame = round(time_to_arrival * fps)` and is clamped to a
non-negative frame index before the model sees it.

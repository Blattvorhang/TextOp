# Planner V6.1: Split-Goal Conditioning

> Status: proposed design. Fixes the conditioning competition observed with the
> V6 `joint_state` goal. Not yet implemented. Implementation lands with the
> full feature set enabled first (schedule constraint); the ablation matrix at
> the end of this doc is deferred post-implementation work.

## Problem

V6 introduced a 40-D `joint_state` goal fed to the denoiser as a **single
token** ([planner_v6_29dof_goal.md](planner_v6_29dof_goal.md)). Training with it
regressed goal-driven locomotion:

1. V4/V5, which conditioned only on root position + yaw, walked reliably to the
   given position and turned to the given heading.
2. With V6, the model largely loses "go to position": it strikes the goal body
   pose but does not move toward the goal.
3. Even when the goal is a crawl pose, the model stays standing while striking
   it — the root never lowers. The root `z` offset lives in the translation
   channels and is being ignored together with `x, y`.
4. Only when `x, y` are far does the model actually walk to the corresponding
   position.

This is a conditioning-competition failure: the 29-D pose condition dominates
the translational increment.

## Diagnosis

Three mechanisms compound:

1. **Scale mismatch.** The 40-D goal is raw and unnormalized: position offsets
   in meters (near goals are a few cm, far goals up to a few m) sit next to
   joint angles in radians (|q| ≤ ~1). `embed_goal` is a single Linear over the
   whole vector ([mld_denoiser.py](TextOpRobotMDAR/robotmdar/model/mld_denoiser.py#L340)),
   so a near goal (x, y ~ 2 cm) is numerically negligible relative to the pose
   channels and is effectively dropped. Only far goals produce values large
   enough to survive the mix — which is exactly the observed behavior.
2. **Single-token mixing.** One Linear mixes all 40 channels into `h_dim`
   before any attention. Translation gradients are diluted by the 29 pose
   channels, and attention cannot weight "position" vs "pose" selectively —
   they are the same token.
3. **Loss competition.** The 29-channel joint-position loss dominates the
   scalar root-position loss in gradient magnitude.

The V4 5-D root/yaw goal worked because translation was the dominant modality.
V6.1 keeps the V6 physical goal contract but restructures how the denoiser
consumes it.

## Design

### 1. Split the goal into four tokens

Stop concatenating all goal information into one 40-D token. Each physical
component gets its own feature vector and its own embedding MLP:

| Token | Physical quantity | Input dim | Embedder |
|---|---|---:|---|
| `TRANS` | translation to goal | 8 | `MLP_trans` |
| `ROT` | orientation (V6 5-D block) | 5 | `MLP_rot` |
| `POSE` | 29-DOF joint target | 29 | `MLP_pose` |
| `VEL` | root velocity | 3 | `MLP_vel` |

Each `MLP_*` is a small two-layer network (`Linear(d_i, h_dim) → SiLU →
Linear(h_dim, h_dim)`) replacing the single `embed_goal` Linear. In
`DenoiserTransformer.forward`, the condition sequence becomes:

```text
xseq = cat(emb_time,
           emb_goal_trans, emb_goal_rot, emb_goal_pose, emb_goal_vel,  # 4 goal tokens
           emb_scene, emb_history, emb_noise)
```

Sequence length grows from `history_len + 4` (time, goal, scene, noise +
history) to `history_len + 7` (time, 4 goals, scene, noise + history); the
cost is negligible. `DenoiserMLP` grows its input projection from `h_dim * 5`
to `h_dim * 8` accordingly.

Separate embedders give each modality its own capacity and gradient path, so
the pose token can no longer drown the translation token at the embedding
layer, and attention can weight the four modalities independently.

### 2. Translation feature `f_trans`

Goal position needs two behaviors at once: **far** goals must trigger
locomotion (data-driven), **near** goals must be hit with cm precision.
Feed this prior into the feature instead of forcing the model to learn it.
Replace the plain `[x, y, z]` translation block with:

```text
r_xy = sqrt(x^2 + y^2)
T    = time_to_arrival in SECONDS (T = time_to_arrival_frame / fps)

f_trans = [ x,  y,  z,
            r_xy,  log(1 + r_xy),
            x/T,  y/T,  z/T ]          # 8-D
```

Per-term rationale:

| Term | Purpose |
|---|---|
| `x, y, z` | exact direction and cm-level target; zero means "at the goal" |
| `r_xy` | keeps a linear-scale distance signal at long range where x, y saturate or oscillate |
| `log(1 + r_xy)` | boosts near-field sensitivity: `d log(1+r)/dr = 1/(1+r)` is large exactly where x, y become numerically small |
| `x/T, y/T, z/T` | urgency: same offset is more urgent when arrival time is short. `T` in seconds so the terms have velocity units |

Edge cases for `T`:

```text
T_eff = max(T, 1/fps)          # T=0 (deployment clamp / zero-time mode) never divides by zero
urgency = 0                    # when goal_timestep_mode == "zero" or arrival time is masked
```

With `fps=50` and `future_len=64`, training sees `T ∈ [0.02, 1.28] s`, so
`x/T` spans `[x/1.28, 50x]` — a wide but finite range that stays in-distribution
at deployment (remaining time is clamped to ≥ 0, giving `T_eff = 1/fps`).

### 3. Scale factors from dataset statistics (`goal_stats.pkl`)

Goal features must be scaled so every channel contributes comparable dynamic
range to its MLP. Compute the statistics at training start, exactly like
`meanstd.pkl` ([data.py:559-580](TextOpRobotMDAR/robotmdar/dataloader/data.py#L559-L580)):
on the train split, sample goal frames under the **same randomized distribution**
used for training (`goal_per_primitive`, `goal_offset_range`), build the ego
goal fields, and compute per-quantity scale factors into `goal_stats.pkl`
(saved next to `meanstd.pkl`).

| Channels | Factor | Definition |
|---|---|---|
| `x, y, z, r_xy` | `s_p` | `1 / std_p`, `std_p` = std of ego `x, y` over sampled goals (x, y, r share one factor) |
| `log(1 + r_xy)` | `s_l` | `1 / std_l`, `std_l` = std of the log term over sampled goals |
| `x/T, y/T, z/T` | `s_v` | `1 / std_v`, `std_v` = std of ego root velocity components |
| `VEL` token (3) | `s_v` | same factor: urgency terms and velocity are both velocity units |
| orientation (5) | `s_o` (per channel) | dataset std; trig channels are already ~unit scale, yaw is down-weighted to ~unit range |
| `POSE` token (29) | per-joint `mean/std` | reuse the `q_t` block statistics from the existing `meanstd.pkl` |

Scaling rule:

```text
f_trans_scaled = [ s_p·x, s_p·y, s_p·z, s_p·r_xy, s_l·log(1+r_xy),
                   s_v·x/T, s_v·y/T, s_v·z/T ]
```

Only the `POSE` token is mean-centered (joint-angle zero is arbitrary). The
kinematic tokens are **scaled but never centered** — zero must keep its
physical meaning ("at the goal", "no velocity", "upright and forward").

### Computing the factors

The goal frame is sampled per primitive from `goal_offset_range` (training
sees every future frame 1..`future_len`), so the factors must be computed over
**the same mixture of (distance, T) pairs** that the model sees — not
stratified by frame and not averaged per frame. The mixture is the training
distribution.

Algorithm (mirrors `_compute_meanstd`,
[data.py:605-628](TextOpRobotMDAR/robotmdar/dataloader/data.py#L605-L628)):

1. Iterate the same `N` seeded batches used for `meanstd.pkl`
   (`_generate_batch_optimized(generator=torch.Generator().manual_seed(i))`).
   Each batch already carries goals sampled with the configured
   `goal_offset_range`, so the statistics distribution equals the training
   distribution by construction.
2. Per primitive, compute the unscaled ego features: `x, y, z`, `r_xy`,
   `log(1+r_xy)`, urgency `x/T, y/T, z/T` (seconds, `T_eff` clamp), ego
   velocity, the 5-D orientation block, `q_g`.
3. Pool and derive factors:

| Factor | Pooled buffer | Why pooling the mixture works |
|---|---|---|
| `s_p` | `{x, y, z, r_xy}` over all sampled goals | for locomotion `x ≈ v·T` grows linearly with `T`, so far frames dominate the std and set the overall dynamic range; near-field behavior is delegated to the log term, not to `s_p` |
| `s_l` | `{log(1+r_xy)}` | near-field sensitivity channel; may need a gain above `1/std` if the near-field signal is still too small after scaling (see Open Questions) |
| `s_v` | `{x/T, y/T, z/T}` ∪ `{vx, vy, vz}` | urgency is velocity-shaped: `x/T ≈ v` for steady motion, so the `T` mixture barely affects its std — sharing `s_v` with the `VEL` token is principled, not just convenient |
| `s_o` | orientation channels, per channel | bounded channels; std mainly down-weights yaw |
| pose mean/std | reuse the `q_t` block from `meanstd.pkl` | same quantity, same joint order as the motion features — consistency beats recomputing |

**`z` shares `s_p` by decision, not by accident.** Vertical offsets are small
(`|z| ≪ |x,y|` for locomotion, ≤ ~0.4 m for sit/crawl), so `s_p·z` stays a
modest but present signal — deliberate, because no loss supervises root `z`
directly (see Losses and Metrics), and a dedicated `s_z = 1/std(z)` would
amplify noise from a near-zero std. Revisit only if a root-z loss is added.

4. Robustness: clip pooled values to physical bounds (`|x, y| ≤ 3 m`,
   `|v| ≤ 5 m/s`) before taking the std, so outlier recovery clips do not
   inflate the std and shrink the factor (MAD is an alternative).
5. Save `{s_p, s_l, s_v, s_o}` plus a `meta` block to `goal_stats.pkl` next to
   `meanstd.pkl`, and log the factors plus the scaled p50/p99 per channel
   group — "appropriate" is verified numerically (p50 ≈ O(0.1–1),
   p99 ≲ 5), not assumed.

The factors depend on everything that shapes the (distance, T) distribution:
`goal_offset_range`, `goal_per_primitive`, `future_len`, `goal_timestep_mode`,
and `fps` (urgency uses seconds: `T = time_to_arrival_frame / fps`, and `s_v`
covers velocity units). The `meta` block therefore stores these five fields,
`encodings: [single, split]` (the two 45-D encodings share the same features
and stats; `legacy40` needs no stats), and the dataset path; on load, the
current config must match the stored meta — a mismatch recomputes on the train
split (or raises on val/eval/planner). Any change to one of these fields
invalidates the cache; the stats pass is config-driven, not hardcoded. Domain
randomization effects on the goal statistics are out of scope for the initial
implementation.

At deployment the factors are frozen: the planner loads `goal_stats.pkl` from
the checkpoint directory and applies the identical scaling. The wire protocol
is unchanged — it still carries raw physical quantities (see below).

### 4. Arrival-time PE on every goal token

Arrival time now conditions every goal modality, not just the single mixed
token:

```text
emb_goal_trans = MLP_trans(f_trans_scaled) + arrival_pe
emb_goal_rot   = MLP_rot(orientation)      + arrival_pe
emb_goal_pose  = MLP_pose(q_g)             + arrival_pe
emb_goal_vel   = MLP_vel(velocity)         + arrival_pe
```

Implementation notes:

- One shared `ArrivalTimeEmbedder` computes `arrival_pe` once; it is added to
  all four tokens. Shared parameters keep the time signal consistent across
  modalities.
- Keep the V6 leak rule, extended to the transformer sequence: build the
  tokens `emb_goal_* = MLP_*(...) + arrival_pe`, multiply each slot by its
  keep mask, and only then concatenate `xseq` and apply
  `sequence_pos_encoder`. The mask zeroes everything content- or
  time-carrying — component channels, the learned MLP bias, and the
  `arrival_pe` — so no timing information leaks from a masked slot. The
  slot positional encoding `pe[i]` is deliberately NOT masked: it is a
  fixed, time-independent sequence code, and keeping it lets the
  transformer tell which condition is missing. This follows the V4/V6
  precedent — both generations masked the goal pre-embedding and added
  `sequence_pos_encoder` afterwards, so a masked goal token has always
  carried `pe[i]`; the only post-MLP mask ever specified is on the
  `arrival_pe` itself (the timing-leak rule).
- When time is masked — training dropout (`cond_goal_time_mask_prob`) and
  eval flags (`force_drop_arrival_time` / `force_drop_goal_time`) alike —
  zero **everything** that carries timing: the `arrival_pe` contribution to
  all four tokens AND the urgency channels `f_trans[5:8]` (they contain
  `x/T` and would otherwise leak through the translation token).
- Because the urgency terms live in `f_trans`, the translation component is
  masked as a unit: masking translation zeroes `x, y, z, r, log, x/T, y/T, z/T`
  together. There is no "position only" sub-mask in V6.1.

## Data Contract

`y['goal']` grows from 40-D to 45-D (scaled, split encoding):

| Slice | Size | Meaning |
|---|---:|---|
| `0:8` | 8 | `f_trans_scaled` |
| `8:13` | 5 | orientation block (V6 layout, scaled) |
| `13:42` | 29 | `q_g`, per-joint mean/std normalized |
| `42:45` | 3 | ego root velocity, `s_v`-scaled |

The V6 physical 40-D goal remains the **loss and visualization target**.
Why two vectors instead of one: the 45-D split vector is scaled and
re-encoded, so it cannot be sliced back into physical quantities — `x` is
spread into `r_xy`, `log(1+r_xy)` and `x/T`, the pose block is mean-centered,
and every block carries a dataset scale factor. Losses must be computed in
meters/radians to stay interpretable, and plots must place the goal marker at
the true physical position. The training manager currently slices the raw goal
vector (`ego_goal[..., :2]`, `[..., 3:8]`, `[..., 8:37]`,
[manager.py:1102,1128,1203,1231](TextOpRobotMDAR/robotmdar/train/manager.py#L1102))
and must keep doing so. `_conditions()` therefore emits both:

```python
y = {
    "goal": split_goal,                # [B, 45] scaled — denoiser input
    "ego_goal_raw": ego_goal,          # [B, 40] physical — losses / viz
    "time_to_arrival_frame": ...,
}
```

Config gate:

```yaml
data:
  goal_type: joint_state
  goal_encoding: split   # legacy40 | single | split
denoiser:
  goal_encoding: ${data.goal_encoding}   # must match data.goal_encoding
  goal_dim: 45
```

`goal_encoding` has exactly three values:

| Value | Goal vector | Tokens | `goal_dim` | Use |
|---|---|---|---:|---|
| `legacy40` | V6 40-D raw, unscaled | 1 | 40 | V6 reproducibility |
| `single` | 45-D split features, scaled | 1 | 45 | ablation A (features without token split) |
| `split` | 45-D split features, scaled | 4 | 45 | V6.1 target |

`legacy40` and `single` both feed one goal token and therefore use the V6
pre-embedding vector-slice masking; only `split` gets per-token masking
before the sequence positional encoding (see Goal Masking). `goal_dim` alone
no longer determines the token layout — `single` and `split` are both 45-D —
so the denoiser carries its own `goal_encoding` key that must match
`data.goal_encoding`.

`GoalType.JOINT_STATE` keeps its physical 40-D dimension for protocol
validation; a new `SPLIT_GOAL_DIM = 45` describes the model-facing
vector, and `_goal_dim_uses_arrival_pe` in the denoiser accepts it.

`validate_goal_config()` changes:

- `goal_encoding` must be one of `legacy40 | single | split` — unknown values
  are rejected, no silent fallback;
- `goal_encoding` is only valid with `goal_type=joint_state`;
- `legacy40` requires `goal_dim == 40` (`JOINT_STATE_GOAL_DIM`);
- `single` / `split` require `goal_dim == 45` (`SPLIT_GOAL_DIM`),
  `dof_dim == 29`, `goal_timestep_mode == "relative"` whenever
  `goal_offset_range != [0, 0]`, and a `goal_stats.pkl` present whose `meta`
  matches the current config (or computable on the train split);
- `data.goal_encoding` and `denoiser.goal_encoding` must agree.

The planner protocol is **unchanged** from V6 — raw `goal_root_pos_world`,
`goal_root_rot_world`, `goal_dof_pos`, `goal_root_velocity_world`,
`goal_timestamp_ns`. The split encoding and scaling happen inside the
planner conversion using the checkpoint's frozen `goal_stats.pkl`.

## Goal Masking

Component masks (same Bernoulli policy as V6). Where they are applied
depends on the encoding: `split` masks each goal token before the shared
sequence positional encoding (content, MLP bias, and arrival PE zeroed; the
time-free slot PE survives so the transformer knows which condition is
missing); `legacy40` and `single` have a single goal token and keep the V6
pre-embedding vector-slice masking:

| Component | Slice | Training mask |
|---|---|---:|
| translation | `0:8` | `cond_goal_root_mask_prob=0.1` |
| orientation | `8:13` | `cond_goal_orientation_mask_prob=0.1` |
| joints | `13:42` | `cond_goal_joint_mask_prob=0.1` |
| root velocity | `42:45` | `cond_goal_velocity_mask_prob=0.0` initially |
| arrival PE | all four tokens | `cond_goal_time_mask_prob=0.0` initially; also zeroes `f_trans[5:8]` |

Keep-mask bookkeeping (`y['goal_*_condition_keep_mask']`) is unchanged so the
existing eval force-drop flags keep working.

## Losses and Metrics

Loss targets are unchanged in semantics — they compare generated motion at the
selected goal frame against the raw physical targets from
`ego_goal_raw` / primitive fields. Changes:

- Re-enable `goal_position` (nonzero weight): the whole point of V6.1 is
  recovering "walk to the position".
- Keep `goal_joint_position` but at a weight that no longer dominates.
- `goal_position` / `goal_direction` supervise horizontal position only
  (`ego_goal[..., :2]`); root `z` is conditioned through the translation
  token but has no direct objective — the crawl/sit validation checks cover
  it qualitatively. Add a root-z loss only if those checks fail.
- `sample_goal_error_m` should be stratified by goal distance in validation,
  e.g. buckets `[0, 0.2)`, `[0.2, 1.0)`, `[1.0, ∞)` m, because near-goal
  precision and far-goal locomotion are now distinct skills with distinct
  signals (`log(1+r)` vs `r_xy` vs urgency).
- Root-XY plots keep consuming raw meters; they must not read the scaled
  vector.

## Recommended Initial Config

```yaml
data:
  dof_dim: 29
  goal_type: joint_state
  goal_encoding: split
  goal_per_primitive: true
  goal_offset_range: [-63, 0]
  goal_timestep_mode: relative
  val:
    goal_offset_range: ${..goal_offset_range}

denoiser:
  goal_encoding: ${data.goal_encoding}
  goal_dim: 45
  cond_goal_root_mask_prob: 0.1
  cond_goal_orientation_mask_prob: 0.1
  cond_goal_joint_mask_prob: 0.1
  cond_goal_velocity_mask_prob: 0.0
  cond_goal_time_mask_prob: 0.0

train:
  manager:
    loss_weight:
      goal_position: 1.0
      goal_direction: 0.0
      goal_joint_position: 0.5
      goal_root_velocity: 0.0
```

## Planned Changes

### 1. Goal features and scaling (`robotmdar/utils/goal.py`)

- add `SPLIT_GOAL_DIM = 45`;
- add `build_ego_split_goal(...)`: sibling of
  `build_ego_joint_state_goal`, producing the raw (unscaled) 45-D vector
  `[f_trans(8) | orientation(5) | q_g(29) | v_ego(3)]`; no FK; the signature
  takes `time_to_arrival_seconds` (required — `f_trans` carries the urgency
  terms `x/T, y/T, z/T`): `T_eff = max(T, 1/fps)`, and `T == 0` (zero-time
  mode) yields `urgency = 0` inside the builder;
- add `scale_goal(goal, goal_stats)`: applies the per-block scale factors from
  `goal_stats.pkl`; kept separate from construction so raw values remain
  available for losses and plots;
- dispatch from `build_ego_goal()` on a `goal_encoding` argument
  (`"split"` / `"single"` build the 45-D vector; `"legacy40"` builds the V6
  40-D vector).

### 2. Dataset statistics and scaling (`robotmdar/dataloader/data.py`)

- add `_compute_goal_stats()` on the train split, sampling goals under the
  same randomized distribution as training; save `goal_stats.pkl` next to
  `meanstd.pkl`; val/eval load the cached file;
- `goal_stats` holds `s_p, s_l, s_v, s_o`, a `meta` block
  (`goal_offset_range`, `goal_per_primitive`, `future_len`, `fps`,
  `goal_timestep_mode`, `encodings: [single, split]`, dataset path), and
  references the existing pose mean/std; loading validates the meta against
  the current config;
- train/val goal paths build the split vector (passing `time_to_arrival`)
  and apply `scale_goal`.

### 3. Denoiser (`robotmdar/model/mld_denoiser.py`)

- goal branch dispatch on `goal_encoding` (not on `goal_dim`: `single` and
  `split` are both 45-D); the denoiser gains a `goal_encoding` config key
  that must match `data.goal_encoding`. Explicit runtime branches:
  - `legacy40`: unchanged V6 path — one `embed_goal` Linear(40 → h_dim),
    one goal token, pre-embedding vector-slice masking;
  - `single`: one `embed_goal` Linear(45 → h_dim), one goal token plus the
    shared `arrival_pe`, pre-embedding vector-slice masking on the 45-D
    slices, urgency `f_trans[5:8]` zeroed when time is masked;
  - `split`: the four `MLP_*` embedders, 4-token sequence, shared
    `arrival_pe` added to each token, masks applied before the sequence
    positional encoding (slot PE survives masking), urgency zeroed when
    time is masked.

### 4. Training (`robotmdar/train/train_dar.py`)

- `_conditions()` emits `goal` (scaled 45-D) and `ego_goal_raw` (physical
  40-D) side by side;
- pass `ego_goal_raw` to the manager losses; viz keeps raw meters;
- validation uses the same `goal_stats.pkl` and `goal_offset_range`;
- checkpoint saving copies `goal_stats.pkl` (like `meanstd.pkl`) into the
  checkpoint directory — this handoff is what lets eval/planner load frozen
  stats without access to the dataset directory.

### 5. Planner / eval (`robotmdar/utils/planner_convert.py`,
`robotmdar/eval/loop_dar.py`)

- load `goal_stats.pkl` from the checkpoint directory (handoff below); apply
  `scale_goal` with the frozen factors after `build_ego_split_goal`;
- reject messages whose declared `goal_type` / `goal_encoding` does not match
  the checkpoint's `goal_encoding` (which is determined by the denoiser
  weights, not by the stats file), and validate the stats meta — the
  checkpoint's encoding must be in `meta.encodings`;
- deployment zero-time clamp: `T_eff = 1/fps` for urgency terms.

### 6. Tests

- `goal_stats.pkl` computation: shared `s_p` across x/y/r, `s_v` for urgency
  and velocity, deterministic train/val split;
- 45-D layout and dispatch of `build_ego_split_goal`, incl. `T_eff`
  clamp at `T=0`;
- urgency channels zeroed when arrival time is masked; masking of all four
  tokens applied before the sequence positional encoding (masked token keeps
  only the slot PE, no time or content residue);
- denoiser sequence shape: `history_len + 7` tokens for `split`, `+4` for
  `single` and `legacy40`;
- losses still consume the raw 40-D `ego_goal_raw`, not the scaled vector;
- planner conversion applies frozen stats and matches the training-path goal
  bitwise for identical inputs.

## Ablations

> Deferred. Implementation proceeds with the full feature set enabled
> (schedule constraint); the matrix below is kept as the post-implementation
> experiment checklist.

Each ablation isolates one mechanism. Run on the same seed/data:

| # | Ablation | Question |
|---|---|---|
| A | `goal_encoding: single` (45-D features, one token) | token split vs feature encoding |
| B | drop `log(1 + r_xy)` | near-field sensitivity |
| C | drop urgency `x/T, y/T, z/T` | timing prior |
| D | split tokens but no scaling (raw units) | scale factors |
| E | `arrival_pe` on `TRANS` only | time on all four tokens |
| F | `goal_encoding: legacy40` (V6 40-D baseline) | regression reference |

## Validation

Qualitative checks, in addition to the existing V6 checklist:

- near goal (r < 20 cm): stops at the position with cm precision;
- far goal (r > 1.5 m): transitions to locomotion and reaches it;
- crawl goal: root `z` drops to the crawl height, body adopts the pose;
- sit/kneel goal: root height + pose;
- masked-pose sample: locomotion-only behavior equivalent to V4 root/yaw;
- masked-translation sample: poses without moving (inverse of the V6 failure);
- zero-time ablation: no urgency information reaches the model.

Quantitative: `sample_goal_error_m` stratified by distance bucket, joint MAE at
the goal frame, direction error at the goal frame, and the rate of
"goal reached" (error < 5 cm) per bucket.

## Open Questions

- `s_l` computed from the dataset vs. leaving `log(1+r)` unscaled (it is
  already bounded).
- Two-layer vs single-layer `MLP_*` embedders.

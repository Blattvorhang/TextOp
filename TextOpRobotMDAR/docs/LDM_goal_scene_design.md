# LDM Stage-2 Training with Goal + Scene Conditioning

> Status: Goal+scene LDM training pipeline implemented and smoke-tested; production
> training and inference CFG integration remain pending.
> Replaces the original TextOp text-conditioned pipeline (CLIP → Denoiser).

## 0. Core Objective

This planner's capabilities are designed as a **progressive implementation roadmap**.
Each stage builds on the previous one; stages 1–2 are the immediate focus, stages
3–4 are deferred for later iterations.

### Stage 1 — Move toward a goal (basic locomotion)

The most fundamental capability: given a sub-goal (world position + facing
direction), generate motion that moves toward that point. TextOp's
streaming autoregressive primitive loop (§5.1) naturally supports replanning —
at each step the goal is re-egocentrified to the character's updated state.
Reliable stopping is not learned from arbitrary snippet endpoints and is deferred
as described in §2.4.2 and §7.4.

**Data requirements:** walk, jog, turn, idle.
**Tracker difficulty:** ★☆☆ — basic locomotion, well within controller capability.
**Data abundance:** 33.5% of BONES-SEED (walk 13.5%, jog 11.0%, turn 2.3%, idle 6.0%).

### Stage 2 — Body-level obstacle avoidance

Once the character can walk to a goal, it must respond to the 25³ egocentric
occupancy grid: **adapt body height** to avoid torso-level obstacles — ducking
under low beams, crawling through tight spaces, kneeling in constrained
clearance. The occupancy voxel crop is injected as the `emb_scene` token (§3.1).

**Data requirements:** crouch, kneel (body height change, COM stays on ground plane).
**Tracker difficulty:** ★★☆ — COM height adaptation is trackable; no precise
limb coordination required.
**Data abundance:** 5.4% of BONES-SEED (crouch 3.9%, kneel 1.5%).

### Stage 3 — Limb-level obstacle negotiation

Finer-grained obstacle handling: **stepping over** small ground objects,
precise foot placement near obstacles. This requires the model to coordinate
individual leg movements with the occupancy grid at limb resolution.

**Data requirements:** step_over.
**Tracker difficulty:** ★★★ — requires precise foot tracking; most challenging
of the ground-level behaviors.
**Data abundance:** 0.18% of BONES-SEED (rarest task-critical category).

### Stage 4 — Scene interaction

After locomotion and obstacle handling are reliable, the character can
**interact with furniture**: sitting on chairs. The goal's height serves as
an implicit interaction cue (§2.4): a goal at chair-seat height (z ≈ 0.5 m)
means "sit."

**Data requirements:** sit.
**Tracker difficulty:** ★★☆ — sitting is a posture transition; trackable
but requires anticipating the seat contact.
**Data abundance:** 3.6% of BONES-SEED.

> **Why not push?** The model receives an occupancy grid, which encodes
> only occupied/free voxels. It cannot distinguish a wall from a door from
> a table — a pushable door and an impassable wall look identical to the
> occupancy query. Push (door/furniture interaction) is therefore classified
> as `task_relevant` alongside jump and climb; it is not a focus for v1.

### Summary

```
Stage 1 ──▶ Stage 2 ──▶ Stage 3 ──▶ Stage 4
walk+jog   +crouch     +step_over   +sit
+turn      +kneel
+idle
  │           │           │           │
  ▼           ▼           ▼           ▼
chase goal  duck under  step over   sit on
            low beams   obstacles   chair
```

**How the design follows from this roadmap:**

| Design choice | Driven by |
|---------------|-----------|
| Goal as a soft-condition embedding token (§2.4) | Stages 1, 4 — directional guidance + implicit sit cue via goal height |
| Per-primitive egocentrification (§2.4.1) | Stages 1–4 — goal stays correct as character moves across primitives |
| Scene occupancy from swept G1 geometry (§6.1) | Stages 2, 3 — pseudo-obstacles from motion data, no external scene needed |
| Per-step local occupancy query (§6.2) | Stages 2, 3 — character sees only its immediate surroundings each step |
| Independent goal/scene CFG dropout (§3.2) | All — inference-time control over goal vs obstacle vs interaction influence |
| TextOp-compatible egocentric convention (§5.2) | All — keeps all Denoiser inputs in one coordinate system |
| Goal height as implicit interaction type (§2.4) | Stage 4 — sitting/kneeling cued by goal Z, no extra token needed |

---

## 1. Architecture Overview

```
                        ┌──────────────────────────┐
                        │      DenoiserTransformer │
                        │      8 layers, 4 heads   │
                        │      d_model = 512       │
                        │                          │
  ┌─────────────────────┤  Input tokens (count=6): │
  │                     │   [0] emb_time           │
  │  Sinusoidal ───────▶│   [1] emb_goal     ◄─────┤── goal (root pos + fdir)
  │  timestep           │   [2] emb_scene    ◄─────┤── occupancy voxel grid
  │                     │   [3] emb_history  ◄─────┤── 2 frames prev. motion
  │                     │   [4] emb_history        │
  │                     │   [5] emb_noise    ◄─────┤── noisy latent
  │                     └──────────────────────────┘
  │                              │
  │                     ┌────────▼──────────┐
  │                     │   clean latent    │
  │                     │   [B, 1, 128]     │
  │                     └────────┬──────────┘
  │                              │
  │                     ┌────────▼──────────┐
  │                     │   MVAE Decoder    │
  │                     │   (frozen)        │
  │                     └────────┬──────────┘
  │                              │
  └──────────────────────────────┼── future motion [B, 8, 57]
                                 │
                    ┌────────────▼──────────┐
                    │  Autoregressive Loop  │
                    │  slide by 8 frames    │
                    │  last 2 → next hist.  │
                    └───────────────────────┘
```

**Key changes from original TextOp:**
- `emb_text` (CLIP, [1, B, 512]) → removed
- `emb_goal` ([1, B, 512]) → goal condition (root position + facing direction)
- `emb_scene` ([1, B, 512]) → local occupancy voxel encoding
- Input token count: 5 → 6

---

## 2. Data Pipeline

### 2.1 Source → PKL

```
BONES-SEED CSV (120 Hz, 29 DOF, cm, deg)
    │
    ▼  convert_soma_csv_to_motion_lib.py
    │    · cm → m
    │    · Euler(deg) → quaternion(xyzw)
    │    · DOF deg → rad
    │    · 120 Hz → 50 Hz  (LERP translation + SLERP rotation)
    │
    ▼
motion_lib PKL  (50 Hz, 29 DOF, m, rad)
```

### 2.2 PKL Format (per sequence)

```python
{
    "root_trans_offset":  np.ndarray [T, 3],   # meters
    "root_rot":           np.ndarray [T, 4],   # xyzw quaternion
    "dof":                np.ndarray [T, 29],  # radians, 29-DOF MJCF order
    "pose_aa":            np.ndarray [T, 30, 3], # axis-angle per body
    "fps":                50.0,
}
```

### 2.3 Dataloader Window

```
segment_len = history_len + future_len × num_primitive + 1
            =     2     +     8      ×      4        + 1  =  35 frames

seg_start = randint(0, T - segment_len)          ← random per epoch
window    = motion[seg_start : seg_start + 35]

Split into 4 primitives (overlap = history_len = 2):

  P0:  H=[0,1],   F=[2..9]
  P1:  H=[8,9],   F=[10..17]
  P2:  H=[16,17], F=[18..25]
  P3:  H=[24,25], F=[26..33]
```

### 2.4 Goal Extraction

Goal is stored in **world coordinates** — no per-snippet egocentrification needed.
The conversion to ego frame happens on-the-fly at training/inference time.

**Goal frame lookup** (training &amp; inference, same logic):
```
goal_frame = seg_start + segment_len - 1 + goal_offset
# segment_len = history_len + future_len × num_primitive + 1 = 35
# seg_start = random window start within the motion (training)
#           = current autoregressive position      (inference)
# goal_offset = 0 for initial training, randomized later
```

**Egocentrification reference frame**: the **last frame of the history motion**
(= the character's "current state" at the moment of prediction).

```
Training:
  P0: history = frames [0, 1]   → crt = frame 1  → ego_goal₀
  P1: history = frames [8, 9]   → crt = frame 9  → ego_goal₁
  P2: history = frames [16, 17] → crt = frame 17 → ego_goal₂
  P3: history = frames [24, 25] → crt = frame 25 → ego_goal₃

Inference:
  crt = abs_pose  # maintained world pose at the last history frame
```

**Ego-centrification** (same transform at training and inference):
```
# Training: read the raw, unnormalized motion arrays. Do not interpret channels
# of the normalized 57-D feature as an absolute position or orientation.
crt_root_pos = sample['motion']['root_trans_offset'][crt_frame]
crt_root_yaw = quaternion_to_yaw(sample['motion']['root_rot'][crt_frame])

# Inference/rollout: use the maintained absolute pose at the selected history.
crt_root_pos = abs_pose['root_trans_offset']
crt_root_yaw = quaternion_to_yaw(abs_pose['root_rot'])
crt_root_fdir = (cos(crt_root_yaw), sin(crt_root_yaw))

ego_goal = world_to_ego(world_goal, crt_root_pos, crt_root_fdir)
# → 5-dim: [ego_dx, ego_dy, ego_dz, cos(goal_yaw-current_yaw),
#           sin(goal_yaw-current_yaw)]
```

`world_to_ego()` uses the same convention as TextOp's motion features:
**+X forward, +Y left, +Z up**. For horizontal current forward
`f = (fx, fy)`, define left as `l = (-fy, fx)` and transform a world
displacement `d` as:

```python
ego_x = dot(d, f)  # forward
ego_y = dot(d, l)  # left
```

**Dataset/loader contract:** the packed sample already retains raw
`root_trans_offset` and `root_rot`, so `pack_motion_lib_to_textop.py` needs no
additional pose channels. Extend `SkeletonPrimitiveDataset` to return, for each
primitive, the normalized motion features plus:

```python
{
    'world_goal_pos': raw_trans[goal_frame],       # [3]
    'world_goal_yaw': yaw(raw_rot[goal_frame]),    # scalar
    'gt_ref_pos': raw_trans[crt_frame],            # [3]
    'gt_ref_rot': raw_rot[crt_frame],              # [4], xyzw
    'scene': sample['scene'],
}
```

The training loop chooses the GT or maintained predicted reference pose according
to the rollout stage, then computes `ego_goal` and the local occupancy. This keeps
absolute pose data out of the normalized 57-D feature representation.

> **Note on analyze_snippet_goal.py**: the analysis script uses `seg_start` (first
> frame of the snippet window) as the egocentric origin rather than the last
> history frame. The difference is ≤ 1 frame (0.02 s — negligible for statistical
> analysis of goal distances). The core goal-frame lookup formula
> `seg_start + segment_len - 1 + offset` is identical.

#### 2.4.1 Per-Primitive Egocentric Transform (MOB-style soft condition)

The goal is stored in world coordinates (§2.4). At each forward pass, it is
egocentrified using the **last frame of the history motion** as the reference:

```
World goal:  world_goal = (root_pos_abs, root_fdir_abs)
Ref frame:   last history frame (h1 for P0, h9 for P1, h17 for P2, h25 for P3)

  P0:  crt_frame = history[1]   → ego_goal₀ = world_to_ego(world_goal, crt_frame)
  P1:  crt_frame = history[9]   → ego_goal₁ = world_to_ego(world_goal, crt_frame)
  P2:  crt_frame = history[17]  → ego_goal₂ = world_to_ego(world_goal, crt_frame)
  P3:  crt_frame = history[25]  → ego_goal₃ = world_to_ego(world_goal, crt_frame)
```

Here `history[...]` denotes the corresponding raw world pose for GT history, or
the maintained absolute rollout pose for predicted history; it does not denote
the normalized 57-D feature channels.

The same world goal produces 4 different ego-goals because the character moves.
This is exactly MOB's approach ([train.py:131](Motion-Occupancy-Base/training/train.py#L131)):
absolute goal re-canonicalized to each rollout step's current frame.

Each ego_goal is a 5-dim vector
`[dx, dy, dz, cos(goal_yaw-current_yaw), sin(goal_yaw-current_yaw)]` encoding
the goal's root position and facing direction relative to the current primitive's
root frame. This is
analogous to MOB's `get_ss_tgt()` ([train.py:131](Motion-Occupancy-Base/training/train.py#L131)),
which re-canonicalizes the absolute goal to each rollout step's egocentric frame.

**Why soft condition, not TRUMANS `fix_mode`?** TRUMANS's `fix_mode` clamps the
last frame of the generated sequence to exactly equal the goal position at every
diffusion timestep. This requires the snippet's last frame to *be* the goal — which
only holds for P3 (the final primitive). P0–P2 are intermediate steps whose last frame
is far from the snippet-level goal; clamping them would be incorrect. MOB's soft
condition has no such constraint: the same absolute goal is simply re-expressed in
each step's local frame and injected as an embedding token. The model learns to
move *toward* the goal rather than to *arrive* at it in any specific primitive.

**Goal does not require precise arrival.** The LDM is a local planner — it only
needs to chase a sub-goal while avoiding obstacles. The upstream global path
planner is responsible for switching to the next sub-goal when the character
approaches the current one. This division of labor means the training objective
is directional ("go that way") rather than positional ("stop exactly here").

**Goal height as implicit interaction cue (Goal #3).** The 5-dim ego-goal vector
`[dx, dy, dz, cos(Δyaw), sin(Δyaw)]` serves navigation, obstacle avoidance, and
scene interaction uniformly — no separate "interaction type" token is needed for
v1. The model learns to interpret the goal's height (ego `dz`) and horizontal
distance (ego `dx`, `dy`) as implicit cues:

| Goal signature | Learned behavior | Category |
|---------------|-----------------|----------|
| `dz ≈ 0`, \|dx\| large, \|dy\| small | Walk/jog forward toward goal | walk, jog |
| `dz ≈ 0`, \|dx\| small, \|dy\| large | Turn/sidestep to face goal | turn |
| `dz < 0`, \|dx\| ≈ 0 | Crouch/duck or crawl (low obstacle) | crouch |
| `dz ≈ −0.3`, \|dx\| small | Sit down (goal at chair height) | sit |
| `dz ≈ −0.5`, \|dx\| small | Kneel down (goal at floor level) | kneel |
| `dx` near 0, `dy` near 0, `dz ≈ 0` | Idle / hold position | idle |
| small \|dx\|, occupancy shows gap at feet | Step over low obstacle | step_over |

These signatures are hypotheses for later stages, not guaranteed semantics of
the v1 condition. They become learnable only when the training windows contain
enough matching transitions and the goal construction correlates with the
behavior. In particular, occupancy has no object identity or affordance label,
so the model cannot infer that an occupied region is a pushable door.

The upstream task planner is responsible for setting the goal position
appropriately: navigation sub-goals at walking height (z ≈ current root Z),
interaction sub-goals at the target object's interaction height (chair seat
z ≈ root Z − 0.3 m, floor object z ≈ root Z − 0.7 m). The LDM itself does not
reason about object semantics — it only sees a position + facing + occupancy.

#### 2.4.2 Stopping Behavior (deferred after v1)

`goal_offset = 0` teaches motion toward the last frame of a randomly sampled
snippet, but a random snippet boundary does not imply deceleration or a stationary
pose. The v1 model therefore does **not** claim reliable stopping behavior. Explicit
terminal/idle windows, goal-hold frames, or a speed/stop condition are deferred to
the to-do list in §7.4.

### 2.5 Scene Occupancy

One global occupancy grid per motion sequence (not per snippet). All snippets
carved from the same CSV share the same scene — this is the standard MOB
approach: precompute one global voxel grid per scene, then query it at each
rollout step using the current root position and facing direction.

```
Per CSV motion (all snippets share this scene):
  ┌──────────────────────────────────────────────┐
  │  Global occupancy grid (precomputed once)    │
  │  · origin + voxel_size + binary [X, Y, Z]    │
  └──────────────────────────────────────────────┘
           │
           │  Per rollout step (r = 0..3):
           ▼
  crt_pos   = root position at primitive step r
  crt_fdir  = facing direction at primitive step r
           │
           ▼  egocentric crop: 25³ around (crt_pos, crt_fdir)
  occu_l    = [B, 25³] flattened → Linear(15625, 512) → token
```

With 25 samples at 0.08 m spacing, the approximate physical extent in MOB's
original coordinates is X(right) `[-1, 1]` m and Y(forward) `[-0.5, 1.5]` m.
After the axis mapping to TextOp coordinates, this is X(forward) `[-0.5, 1.5]`
m and Y(left) `[-1, 1]` m. The exact sample-center bounds from `get_grid()` are
`[-0.96, 0.96]` and `[-0.48, 1.44]` m.

This design means:
- **One global occupancy array per scene**, precomputed offline and stored in the PKL.
- **Multiple snippets from the same CSV all reference the same grid** — no
  per-snippet recomputation.
- **Each rollout step queries a different egocentric sub-volume** because the
  root position moves forward over the 4 primitives.
- Scene occupancy is static per motion; dynamic objects would require updating
  the global grid and re-querying.

---

## 3. Model Changes

### 3.1 DenoiserTransformer (mld_denoiser.py)

```python
class DenoiserTransformer(nn.Module):
    def __init__(self, ...):
        # ── existing ──
        self.embed_timestep = TimestepEmbedder(...)
        self.embed_history = nn.Linear(57, 512)
        self.embed_noise = nn.Linear(128, 512)

        # ── new: replaces embed_text ──
        goal_dim   = 5                         # ego root pos(3) + yaw(2) = cosθ,sinθ
        gsize      = (25, 25, 25)              # occupancy voxel grid
        grid_dim   = gsize[0] * gsize[1] * gsize[2]  # = 15625
        self.embed_goal  = nn.Linear(goal_dim, 512)
        self.embed_scene = nn.Linear(grid_dim, 512)

    def forward(self, x_t, timesteps, y):
        emb_time   = self.embed_timestep(timesteps)           # [1, B, 512]
        emb_goal   = self.embed_goal(y['goal']).unsqueeze(0)  # [1, B, 512]
        emb_scene  = self.embed_scene(y['voxel']).unsqueeze(0)# [1, B, 512]
        emb_history = self.embed_history(
            y['history_motion_normalized']).permute(1,0,2)             # [2, B, 512]
        emb_noise  = self.embed_noise(x_t).permute(1,0,2)    # [1, B, 512]

        xseq = torch.cat([emb_time, emb_goal, emb_scene,
                          emb_history, emb_noise], dim=0)     # [6, B, 512]
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[-1:]              # [1, B, 512]
        output = self.output_process(output).permute(1,0,2)   # [B, 1, 128]
        return output
```

### 3.2 Independent Condition Dropout

Goal and scene conditions are dropped **independently** during training,
producing 4 combinations:

| drop_goal | drop_scene | meaning |
|-----------|------------|---------|
| False | False | full condition (goal + scene) |
| True | False | scene-only (no goal) |
| False | True | goal-only (no scene) |
| True | True | unconditional |

```python
class DenoiserTransformer(nn.Module):
    def __init__(self, ...):
        ...
        self.cond_goal_mask_prob  = 0.1
        self.cond_scene_mask_prob = 0.1

    def forward(self, x_t, timesteps, y):
        # Training: independent per-sample Bernoulli masks.
        # Inference: force_drop_* overrides them for CFG passes.
        drop_goal = sample_or_force_mask(
            len(x_t), self.cond_goal_mask_prob, y.get('force_drop_goal', False))
        drop_scene = sample_or_force_mask(
            len(x_t), self.cond_scene_mask_prob, y.get('force_drop_scene', False))

        emb_time   = self.embed_timestep(timesteps)
        emb_goal = self.embed_goal(mask_condition(y['goal'], drop_goal)).unsqueeze(0)
        emb_scene = self.embed_scene(mask_condition(y['voxel'], drop_scene)).unsqueeze(0)
        ...
```

### 3.3 ClassifierFreeWrapper (generate_dar.py)

The independent dropout needed for training is implemented in v1. The four-pass
inference wrapper below remains a follow-up and is not required by the training
pipeline.

CFG with independent goal/scene guidance scales:

```python
class ClassifierFreeWrapper(nn.Module):
    def forward(self, x, timesteps, y):
        # 4 forward passes — one per condition combination
        out_full = self.model(x, timesteps, with_forced_drop(
            y, goal=False, scene=False))
        out_nogoal = self.model(x, timesteps, with_forced_drop(
            y, goal=True, scene=False))
        out_noscene = self.model(x, timesteps, with_forced_drop(
            y, goal=False, scene=True))
        out_uncond = self.model(x, timesteps, with_forced_drop(
            y, goal=True, scene=True))

        return (out_uncond
                + y['goal_scale']  * (out_noscene - out_uncond)   # goal guidance
                + y['scene_scale'] * (out_nogoal  - out_uncond)   # scene guidance
                + y['joint_scale'] * (out_full - out_nogoal - out_noscene + out_uncond))
        # joint_scale controls the interaction term; its scaling is an ablation.
```

Independent dropout exposes the model to all 4 condition combinations in
expectation. With both probabilities at 0.1, their probabilities are 0.81, 0.09,
0.09, and 0.01; the fully unconditional case is therefore relatively rare. The
four-pass CFG equation, dropout rates, and interaction scale are an explicit
ablation item (§7.4), not a settled v1 choice.

---

## 4. Training

### 4.1 Configuration

| Parameter | Value | Note |
|-----------|-------|------|
| `data.history_len` | 2 | |
| `data.future_len` | 8 | |
| `data.num_primitive` | 4 | autoregressive chain length |
| `data.batch_size` | 512–2048 | tune for GPU memory |
| `denoiser.h_dim` | 512 | |
| `denoiser.num_layers` | 8 | transformer layers |
| `denoiser.num_heads` | 4 | |
| `denoiser.noise_shape` | [1, 128] | latent dimension |
| `diffusion.num_timesteps` | 5 | cosine schedule, START_X prediction |
| `train.manager.stages` | [100000, 100000, 100000] | 3 stages, 300k total |
| `train.manager.use_rollout` | true | Stage 1+ uses predicted history |
| `train.manager.use_full_sample` | false → true later | full DDPM sampling for rollout |
| `train.manager.learning_rate` | 1e-4 | AdamW, cosine anneal |
| `denoiser.cond_goal_mask_prob` | 0.1 | P(drop goal) — independent of scene |
| `denoiser.cond_scene_mask_prob` | 0.1 | P(drop scene) — independent of goal |
| `guidance_scale` | 5.0 | inference CFG scale |
| `goal_offset` | 0 | set to 0 for initial training |
| `data.weighted_sample` | false | disabled for v1; original optional logic is retained |
| `data.frame_weight` | false | only used when weighted sampling is enabled |

### 4.1.0 Action Category Balancing (deferred)

BONES-SEED contains 71,132 motion clips spanning ~20 coarse action categories
(see `dataset/data_analyze/analyze_action_distribution.py` for the full
breakdown). The distribution is long-tailed — locomotion (walk, jog) accounts
for ~24% while critical categories like `step_over` are only 0.2% of the data.

For training the LDM planner, the following categories are **task-critical**
(navigation + ground-level obstacle avoidance + scene interaction).
Categories that require vertical motion (jump, climb) or precise upper-body
control (reach) are downgraded to `task_relevant` — the universal controller
cannot reliably track them, and they are not the focus of our work.

| Stage | Category | Files | % | Priority | Driven by |
|:-----:|----------|-------|---|----------|-----------|
| 1 | walk | 9,577 | 13.5% | critical | locomotion to goal |
| 1 | jog | 7,813 | 11.0% | critical | fast locomotion |
| 1 | turn | 1,647 | 2.3% | critical | facing-direction changes |
| 1 | idle | 4,280 | 6.0% | critical | stationary seed and future stopping work |
| 2 | crouch | 2,780 | 3.9% | critical | duck / crawl under low obstacles |
| 2 | kneel | 1,090 | 1.5% | critical | body height adaptation |
| 3 | **step_over** | **131** | **0.18%** | critical | **step over ground obstacles (rarest!)** |
| 4 | sit | 2,561 | 3.6% | critical | sitting on chairs |

**Jump and climb are excluded from task-critical:** vertical motion (jumping,
ladder climbing, vaulting) is difficult for the universal controller to execute
reliably. **Push is excluded:** the occupancy grid only encodes occupied/free
voxels — a pushable door and an impassable wall are indistinguishable to the
model, so door/furniture interaction cannot be occupancy-driven. **Reach is
excluded:** precise upper-body reaching is not a core focus.
All four are classified as `task_relevant` and receive no boost multiplier.

The focus is on **ground-level motion**: walking to goals, crossing low
obstacles (step_over), adapting body height to constraints (crouch/crawl, kneel),
and interacting with furniture at reachable heights (sit).

The v1 configuration samples sequences uniformly by setting
`data.weighted_sample=false`. The original annotation-driven sequence weighting,
optional frame weighting, weighted normalization, coarse `frame_ann` packing,
and statistics-generation code remain available for later experiments; none of
them run on the active v1 path.

### 4.1.1 Three-Stage Rollout Schedule

TextOp's `DARManager.should_rollout()` implements scheduled sampling over the
three configured stages when `use_rollout=true`:

| Stage | History used by the Denoiser |
|-------|-------------------------------|
| 0 | Pure ground-truth history |
| 1 | Mixture; predicted-history probability increases linearly from 0 to 1 |
| 2 | Predicted history after the GT seed primitive |

Stage lengths count primitive optimizer steps because `manager.pre_step()` and
`post_step()` run inside the primitive loop. P0 always uses its GT history because
there is no preceding prediction; "pure predicted" applies to P1 and later.
During stages 1 and 2, the rollout
must also carry the predicted absolute root pose. Goal canonicalization and the
occupancy query use that predicted pose whenever predicted history is selected;
using the corresponding GT primitive pose would make the conditions inconsistent
with the selected history.

### 4.2 Command

```bash
cd TextOpRobotMDAR

robotmdar --config-name=train_dar expname=BONES-SEED-LDM \
  data.datadir=/path/to/packed_bones_seed_50fps \
  data.num_primitive=4 \
  data.batch_size=512 \
  data.weighted_sample=false \
  "train.manager.stages=[100000,100000,100000]" \
  train.manager.use_rollout=true \
  train.manager.use_full_sample=false \
  diffusion.num_timesteps=5 \
  denoiser.cond_goal_mask_prob=0.1 \
  denoiser.cond_scene_mask_prob=0.1 \
  skeleton.asset.assetRoot=/absolute/path/to/description/robots/g1/
```

### 4.3 Monitoring

- TensorBoard at `logs/RobotMDAR/{expname}/train-dar-{timestamp}/`
- Key metrics: `train_rec`, `train_latent_rec`, `train_total`, `eval_rec`
- Loss components: reconstruction (motion + latent), geometric (body_trans,
  body_rot, dof_pos, dof_vel, foot_contact), optional velocity/field losses

---

## 5. Inference

### 5.1 Autoregressive Loop

```python
history_motion = zero_pose  # [1, 2, 57]
abs_pose = zero_abs_pose    # root at origin

while not done:
    # abs_pose is the maintained world pose at the last history frame.
    crt_root_pos = abs_pose['root_trans_offset']
    crt_root_yaw = quaternion_to_yaw(abs_pose['root_rot'])
    crt_root_fdir = torch.stack(
        [torch.cos(crt_root_yaw), torch.sin(crt_root_yaw)], dim=-1)

    # 1. Query scene occupancy at the current absolute root pose.
    voxel = query_occupancy(crt_root_pos, crt_root_yaw)

    # 2. Get world goal from upstream global planner, egocentrify to
    #    the last history frame (= character's current state)
    world_goal = get_current_goal(abs_pose)          # from planner
    delta_pos = world_goal.root_pos - crt_root_pos
    ego_xyz = world_vector_to_textop_ego(delta_pos, crt_root_fdir)
    delta_yaw = wrap_to_pi(world_goal.yaw - crt_root_yaw)
    ego_goal = torch.cat([
        ego_xyz, torch.cos(delta_yaw), torch.sin(delta_yaw)
    ], dim=-1)                                       # [B, 5]

    # 3. DDPM sampling (5 steps) in latent space
    latent = diffusion.p_sample_loop(
        denoiser, shape=[1, 1, 128],
        model_kwargs={'y': {
            'goal': ego_goal,
            'voxel': voxel,
            'history': history_motion,
            'scale': guidance_scale,
        }}
    )

    # 4. VAE decode → 8 future frames
    future_motion = vae.decode(latent, history_motion, nfuture=8)

    # 5. Slide window → next iteration
    history_motion = future_motion[:, -2:, :]
    abs_pose = update_abs_pose(future_motion, abs_pose)  # pose at new last history frame
```

- Each step: 8 frames (0.16 s @ 50 Hz)
- 5 DDPM steps per generation
- Can run indefinitely

---

### 5.2 Coordinate Convention

All Denoiser inputs use TextOp's existing right-handed convention: **+X
forward, +Y left, +Z up**, with yaw about +Z. The frozen VAE and its 57-dim
features remain unchanged.

MOB's local grid uses +Y forward and +X right. Reusing its occupancy lookup
therefore requires the fixed local mapping
`(x_textop, y_textop, z) = (y_mob, -x_mob, z)`: swap the horizontal axes and
flip MOB-right into TextOp-left. Construct or permute the local sampling grid so
its forward offset lies on TextOp +X. The global occupancy array, `llb`, and world
coordinates are not rotated or rewritten; only the per-step local sampling
coordinates change.

| Code location | Convention |
|---------------|------------|
| `world_to_ego()` | +X forward, +Y left, +Z up |
| local occupancy grid/query | +X forward, +Y left, +Z up |
| relative goal orientation | `cos/sin(goal_yaw-current_yaw)` |
| VAE 57-dim features | +X forward, +Y left, +Z up |

---

## 6. Scene Occupancy Pipeline

Scene occupancy follows the Motion Occupancy Base construction from
`papers/MOB.pdf`: no external scene files are needed.
`convert_soma_csv_to_motion_lib.py --mob` sweeps the G1 collision geometry through
the entire motion trajectory, then inverts the swept volume: every voxel the robot
**never** touches is treated as an obstacle. The result is a dense pseudo-scene
where only the robot's actual motion corridor is free space.

One global occupancy grid is computed **per motion sequence** and stored alongside
the motion data in the PKL. All snippets carved from the same sequence share this
grid. This is MOB's standard approach (`query_occu_batched` in
`Motion-Occupancy-Base/training/utils/occu.py`).

### 6.1 Preprocessing (offline, per motion)

```bash
# Stage 1 of the pipeline — --mob enables occupancy computation:
python dataset/data_process/convert_soma_csv_to_motion_lib.py \
    --input bones-seed/g1/csv --output ./motion_lib \
    --mob --mob_frame_stride 2 --fps 50 --fps_source 120
```

Output: each motion_lib PKL entry contains a per-motion global 3D occupancy grid:

```python
"scene": {
    "occu_global":  ndarray [X, Y, Z],  # bool, 1=occupied (pseudo-obstacle)
    "unit":         0.08,               # voxel resolution (m)
    "llb":          ndarray [3],        # float32, world lower-left-back corner
}
```

This grid is passed through `pack_motion_lib_to_textop.py` unchanged into the
final TextOp PKL.

### 6.2 Runtime Query (per rollout step)

At each training/inference step, a 25³ egocentric crop is extracted from the
global grid using the same transform as the goal — origin = last history frame's
root position and facing direction:

```
        global grid (per motion)               per-step query
   ┌─────────────────────┐
   │  ██░░░░██░░███░░░   │              crt_pos=(x₀,y₀), crt_fdir
   │  █░░░░░██░████░░░   │    ──▶       ┌─────────┐
   │  ░░░░░░██░░░██░░█   │              │ ░░██░░██│ 25³
   │  █░░░░░░░░░░░░░░░   │              │ ██░░░░██│ egocentric
   │  ░░████░░░░░░░░██   │              │ ░░██░░░░│ crop
   │  ░░░░░░░░░░░█████   │              └─────────┘
   └─────────────────────┘
```

The query:
1. Creates 25³ sampling points in TextOp egocentric coordinates (origin = crt,
   X+ = forward, Y+ = left). The exact center offsets are X `[-0.48, 1.44]`,
   Y `[-0.96, 0.96]`, and Z `[-0.96, 0.96]` meters at 0.08 m spacing. Thus the
   cube is centered vertically on the current root and shifted 0.48 m forward,
   matching MOB's `s/4` forward offset up to discrete voxel centers.
2. Rotates + translates to world coordinates → index into `occu_global`
3. Flattens to [B, 15625] → `Linear(15625, 512)` → `emb_scene` token

Global grids may have different shapes per sequence. Query each sample against
its own `(occu_global, llb, unit)` and stack only the resulting fixed-shape local
grids. As in MOB's `query_occu_batched`, samples outside a global grid are marked
occupied (`1`). Padding global grids to a common shape is not required.

---

## 7. Goal Offset Strategy

### 7.1 Current (v1): `goal_offset = 0`

Goal = snippet window's last frame. This is the simplest initial training
signal. Re-run `analyze_snippet_goal.py` after the coordinate correction before
recording the final in-BBOX percentage.

### 7.2 Future (v2): randomized `goal_offset`

```python
goal_offset = randint(0, GOAL_MAX_OFFSET)
goal_frame = seg_start + segment_len - 1 + goal_offset
```

- `GOAL_MAX_OFFSET` determined by `analyze_snippet_goal.py` output
  (the largest offset where ≥ 80% of goals remain in BBOX)
- Trains the model to handle goals at varying distances
- Goals beyond BBOX range should be handled by upstream global path planner
  (split into local sub-goals) — NOT the LDM's responsibility

### 7.3 Global Planner (out of scope for LDM training)

Long-range navigation (U-shaped obstacles, dead ends, multi-path choices)
requires a global path planner that produces a sequence of local sub-goals,
each within the LDM's perceptual range. At inference time, the LDM chases
the current sub-goal and requests the next one when approaching it.

### 7.4 Deferred To-Do and Ablations

These are intentionally deferred so v1 can validate basic goal + scene training:

- **Stopping data:** add terminal/idle windows or append goal-hold frames, then
  evaluate behavior while repeatedly conditioning on a reached goal. Random
  snippet endpoints alone do not guarantee a stop signal.
- **Independent CFG:** ablate dropout probabilities, two-pass joint CFG versus
  four-pass factorized CFG, and the joint interaction scale. Include the no-CFG
  baseline and verify that unit scales reproduce the fully conditioned output.
- **Inference integration:** implement and validate the factorized CFG wrapper
  and update the streaming inference entry points to supply world goals/scenes.
- **MOB collision mechanisms:** v1 uses the occupancy condition only. Later
  ablate MOB-style penetration loss, occupancy-field loss, and field regulation
  rather than attributing all collision avoidance to the occupancy token.

---

## 8. Implementation Map

| File | Change |
|------|--------|
| `robotmdar/model/mld_denoiser.py` | Add `embed_goal`, `embed_scene`; remove `embed_text`; update `forward()` |
| `robotmdar/eval/generate_dar.py` | Deferred: update `ClassifierFreeWrapper` for goal+scene CFG |
| `robotmdar/train/train_dar.py` | Add goal extraction + voxel query in training loop; update `y` dict |
| `robotmdar/dataloader/data.py` | Add `goal_offset`; return motion, world goal/reference poses, and scene occupancy |
| `robotmdar/config/denoiser/def.yaml` | Remove `clip_dim`; add occupancy grid config |
| `robotmdar/config/train/dar.yaml` | Configure the three-stage rollout schedule |
| `dataset/data_process/convert_soma_csv_to_motion_lib.py` | Already updated (SLERP 120→50Hz) |
| `dataset/data_process/pack_motion_lib_to_textop.py` | Pass the global scene through and retain coarse `frame_ann` for optional weighting |

**Window-bound reminder:** retain the original TextOp sampling behavior for v1,
including its exclusive upper bound. Before enabling randomized positive
`goal_offset`, re-check sequence-length filtering and ensure the selected goal
frame is in bounds.

---

## 9. References

- **MOB** (`papers/MOB.pdf`, `Motion-Occupancy-Base/`): pseudo-scene occupancy is the complement of swept body occupancy; controls are re-canonicalized and occupancy is queried at every autoregressive step
- **TRUMANS** (`trumans_utils/`): scene as 32³ voxel via ViT → prepended token; goal via `fix_mode` clamp in diffusion loop
- **TextOp original**: text-conditioned via CLIP embedding → replaced by goal + scene here
- **BONES-SEED**: 120 Hz, 29 DOF, cm+deg → 50 Hz, 23/29 DOF, m+rad via `convert_soma_csv_to_motion_lib.py`. 71K clips, 20 coarse action categories — distribution analyzed in `dataset/data_analyze/analyze_action_distribution.py`

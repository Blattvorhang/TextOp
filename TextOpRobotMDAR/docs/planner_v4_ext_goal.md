# Planner V4: Extended Body Goal Conditioning

## Scope

V4 extends DAR training with target root velocity and remaining time while
retaining root position, optional root yaw, and four limb targets. It does not
change the 29-DOF motion representation or MVAE.

Planner protocol and online goal construction are deferred. To keep the current
planner usable with V3 checkpoints, the goal types are versioned:

- `goal_type=root`: legacy 5-D root goal.
- `goal_type=body`: legacy 15-D pelvis/feet/hands goal.
- `goal_type=body_ext`: V4 21-D extended goal.

All V4 DAR checkpoints must be trained with `goal_type=body_ext` and
`goal_dim=21`.

## Goal Layout

The V4 vector contains 21 dimensions:

| Slice | Size | Meaning | Training dropout |
|---|---:|---|---:|
| `0:3` | 3 | Ego-frame root offset `(x, y, z)` | `cond_goal_root_mask_prob=0.1` |
| `3:5` | 2 | Relative root yaw `(cos(dyaw), sin(dyaw))` | `cond_goal_yaw_mask_prob=0.3` |
| `5:8` | 3 | Ego-frame root velocity `(vx, vy, vz)` | Never dropped |
| `8:9` | 1 | Remaining time in seconds | `cond_goal_time_mask_prob=0.3` |
| `9:21` | 12 | Four ego-frame limb points | `cond_goal_body_mask_prob=0.3` |

The pelvis point is not repeated in the limb block because it is equivalent to
the root position. The limb order is:

1. Left foot
2. Right foot
3. Left hand
4. Right hand

The foot targets use the terminal `left/right_ankle_roll_link` body origins.
These are the physical foot bodies exposed by the G1 kinematic tree and require
no mesh parsing during data loading. The hand targets use the rubber-hand mesh
origins from `g1_29dof.xml`:

- Left: `[0.0415, 0.003, 0]` in `left_wrist_yaw_link` coordinates.
- Right: `[0.0415, -0.003, 0]` in `right_wrist_yaw_link` coordinates.

This retains all three wrist DOFs and avoids a world-space hand offset.

## Coordinate Conventions

Root position, velocity, and limb XY values are rotated from world coordinates
into the reference root heading. Z remains a world-up difference or velocity.
Root yaw is relative to the reference root yaw.

Velocity uses the same forward-difference convention as FeatureVersion 3:

```text
v_goal = (root_pos[goal_frame + 1] - root_pos[goal_frame]) * fps
```

V4 does not use a backward-difference fallback. Dataset bounds guarantee that
`goal_frame + 1` exists, so velocity semantics remain consistent at every
sample.

Remaining time is:

```text
goal_timestep = (goal_frame - reference_frame) / fps
```

It is stored and batched as `[B, 1]`.

## Goal-Frame Sampling

`goal_offset` may be negative. `goal_offset_range=[min, max]` enables uniform,
inclusive random sampling. The configured lower bound must satisfy:

```text
min >= 1 - future_len
```

This ensures every goal lies strictly after its primitive reference frame.
For `future_len=64`, V4 training uses `[-63, 0]`.

The loader samples one offset per motion snippet and uses it consistently for
all primitives in that snippet.

When `goal_per_primitive=true`:

```text
goal_frame = prim_start + history_len + future_len - 1 + goal_offset
```

When `goal_per_primitive=false`:

```text
goal_frame = seg_start + segment_len - 1 + goal_offset
```

The second form is one frame later than the final primitive's last feature
frame because `segment_len` contains the raw `+1` frame used by motion-feature
conversion. For V4 shared goals, valid-sequence filtering reserves one further
raw frame for the goal velocity forward difference.

Training samples offsets from `[-63, 0]`. Validation uses fixed offset `0` for
repeatable metrics. Bounds are checked before FK or velocity extraction for
both `goal_per_primitive` modes.

## Component Masking

Root position, yaw, time, and limbs are independently masked per batch item.
Velocity is always supplied during training and is never masked.

The explicit inference controls are:

- `force_drop_goal_root`
- `force_drop_goal_yaw`
- `force_drop_goal_time`
- `force_drop_goal_body`

There is intentionally no velocity force-drop control. Scene masking remains
independent through `force_drop_scene`.

`goal_condition_keep_mask` continues to mean the root-position keep mask. The
existing goal-direction loss reads dimensions `0:2` and excludes samples whose
root position was dropped. Additional keep masks are exposed as:

- `goal_yaw_condition_keep_mask`
- `goal_time_condition_keep_mask`
- `goal_body_condition_keep_mask`

The current zero-fill masking is adequate for the always-present training
contract. V4 planner work must not treat an omitted velocity as zero velocity;
velocity is required by `build_ego_goal` for `body_ext`.

## Training Data Flow

For each primitive, `SkeletonPrimitiveDataset` emits:

```text
world_goal_pos        [B, 3]
world_goal_yaw        [B]
world_goal_vel        [B, 3]
goal_timestep         [B, 1]
world_goal_keypoints  [B, 4, 3]
```

`train_dar._conditions()` moves all fields to the training device and calls
`build_ego_goal`. The resulting `[B, 21]` tensor is projected into one
transformer token. With 16 history frames, the denoiser sequence remains:

```text
[diffusion time:1] [goal:1] [scene:1] [history:16] [noise:1]
```

The MVAE input remains 69-D and its latent remains `[1, 128]`.

## Configuration

The V4 training config is:

```yaml
data:
  goal_type: body_ext
  goal_offset_range: [-63, 0]
  val:
    goal_offset_range: null

denoiser:
  goal_dim: 21
  cond_goal_root_mask_prob: 0.1
  cond_goal_yaw_mask_prob: 0.3
  cond_goal_time_mask_prob: 0.3
  cond_goal_body_mask_prob: 0.3
```

The base denoiser defaults remain compatible with `goal_type=root`. V3 body
evaluation configs remain at 15 dimensions. `vis_dar_v4.yaml` provides offline
V4 dataset evaluation with all extended fields.

## Planner Work Deferred

The current planner and controller protocol remain V3 and continue to use
`goal_type=body`, `goal_dim=15`. They do not yet provide root velocity or
remaining time.

A later planner phase must:

1. Add required target velocity and arrival-time data to the controller message.
2. Define how remaining time is updated at every MPC replan.
3. Convert the new world-frame fields to the V4 ego frame.
4. Select `goal_type=body_ext`, `goal_dim=21` and a V4 DAR checkpoint.
5. Explicitly force-drop yaw if the online planner does not provide it.

Until that work is complete, a V4 checkpoint must not be selected by
`planner_dar.yaml`.

## Files

Modified for V4 training:

- `robotmdar/utils/goal.py`
- `robotmdar/model/mld_denoiser.py`
- `robotmdar/dataloader/data.py`
- `robotmdar/train/train_dar.py`
- `robotmdar/eval/vis_dar.py`
- `robotmdar/config/data/mob.yaml`
- `robotmdar/config/denoiser/def.yaml`
- `robotmdar/config/train_dar.yaml`
- `robotmdar/config/skeleton/g1.yaml`
- `robotmdar/config/vis_dar_v4.yaml`

Intentionally unchanged for this phase:

- `robotmdar/planner/planner_dar.py`
- `robotmdar/utils/planner_convert.py`
- `robotmdar/config/planner_dar.yaml`
- Controller/planner message protocol

## Compatibility and Rollback

The MVAE is unchanged and can be reused. V4 changes the denoiser goal projection
from its V3 15-D body input to a 21-D `body_ext` input, so the DAR denoiser must
be retrained.

V3 root and body checkpoints remain loadable with their original goal types and
matching configs. Rolling back V4 training only requires selecting
`goal_type=body`, `goal_dim=15` and a V3 DAR checkpoint; no source rollback is
required.

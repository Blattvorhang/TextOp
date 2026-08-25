# Planner V6: 29-DOF Joint-State Goal Conditioning

> Status: implemented in the TextOpRobotMDAR training/eval/planner code path.

## Scope

V6 replaces the V4 `body_ext` goal with a direct robot joint-state goal:

- no FK-derived limb keypoints in the goal;
- goal state is selected directly from the GT motion frame;
- no `root`, `body`, or `body_ext` compatibility path for V6 checkpoints;
- no contact indicators and no goal-side `delta_joint`;
- arrival time remains a separate `time_to_arrival_frame` positional encoding,
  not a scalar embedded in the goal vector.

The new goal type is:

```yaml
data:
  goal_type: joint_state
```

`joint_state` is intentionally a breaking goal contract. It targets the full
G1 29-DOF skeleton and should reject 23-DOF data/checkpoints instead of silently
falling back.

## Goal Frame Contract

The goal is not always the primitive end. V6 keeps the randomized goal-frame
contract introduced for V4:

```text
reference_frame = prim_start + history_len - 1
goal_frame      = reference_frame + future_len + goal_offset
```

With `future_len=64`, use:

```yaml
data:
  goal_per_primitive: true
  goal_offset_range: [-63, 0]
  goal_timestep_mode: relative
  val:
    goal_offset_range: ${..goal_offset_range}
```

This samples every generated future frame in a primitive:

```text
time_to_arrival_frame = goal_frame - reference_frame  # 1 .. future_len
```

For trajectory helpers that include the reference frame at index 0,
`time_to_arrival_frame` can be used directly. For future-only tensors of shape
`[B, future_len, ...]`, use:

```text
goal_step = clamp(time_to_arrival_frame - 1, 0, future_len - 1)
```

The primitive-end marker is always the trajectory point at
`time_to_end_frame = future_len`, independent of the selected goal frame.

## Target State

At the selected `goal_frame`, the dataloader extracts raw root and joint state
directly from the GT motion arrays:

```text
p_g      root position at goal_frame
R_g      root orientation at goal_frame
q_g      29-DOF joint position at goal_frame
v_g      original forward root velocity:
         (p[goal_frame + 1] - p[goal_frame]) * fps
```

`goal_frame + 1` must exist because root velocity keeps the same
forward-difference convention as V4. This is still true when
`goal_offset=0` because the primitive extraction window already reserves the
extra raw frame used by FeatureVersion 3.

All root quantities are expressed in the ego frame of the last history frame:

```text
psi_ref       = yaw(reference root rotation)
root_pos_ego  = Rz(-psi_ref) (p_g - p_ref)
root_vel_ego  = Rz(-psi_ref) v_g
```

`q_g` is not spatially rotated. It is the model joint-position vector in the
canonical G1 29-DOF order. Do not replace it with `q_g - q_ref`; the goal is a
target joint position, not a delta-joint target.

No FK is involved in any of the above. In particular, `joint_state` goal
construction must not call `skeleton.forward_kinematics()`,
`_world_goal_keypoints()`, or any keypoint extraction helper.

## Orientation Representation

The raw pose target described as:

```text
root position(3) + orientation(3) + 29-DOF joint position(29) = 35-D
```

is the physical state we want to condition on when orientation is represented
as Euler angles. If the same target is transported as a quaternion, the raw
pose part is 36-D:

```text
root position(3) + orientation quaternion(4) + 29-DOF joint position(29) = 36-D
```

V6 accepts either form at the planner communication boundary, but the denoiser
does not consume raw Euler angles or raw quaternions. The planner converts the
incoming root orientation into the TextOp-style continuous orientation encoding.

The chosen encoding imitates TextOp's motion feature convention:

```text
r = (roll, pitch, yaw)
phi(r) = [
  sin(roll),
  cos(roll) - 1,
  sin(pitch),
  cos(pitch) - 1,
]
```

For a goal state, compute the orientation relative to the reference heading:

```text
R_g_ego = Rz(-psi_ref) R_g
(roll_g, pitch_g, yaw_g_ego) = quaternion_to_euler_angles(R_g_ego)
```

Then use:

```text
orientation_goal = [
  sin(roll_g),
  cos(roll_g) - 1,
  sin(pitch_g),
  cos(pitch_g) - 1,
  wrap_to_pi(yaw_g_ego),
]
```

This is a 5-D encoded orientation block. The resulting V6 goal vector is
40-D:

| Slice | Size | Meaning |
|---|---:|---|
| `0:3` | 3 | Ego-frame root position offset |
| `3:7` | 4 | TextOp-style roll/pitch trig encoding |
| `7:8` | 1 | Ego-frame relative root yaw |
| `8:37` | 29 | G1 joint positions |
| `37:40` | 3 | Ego-frame original root velocity |

The 40-D model input is larger than the literal 35-D raw pose because the continuous
roll/pitch encoding expands two Euler channels into four, and the V4 root
velocity condition contributes another three channels. Therefore the V6
denoiser contract is `goal_dim=40`.

Implementation note: this is a goal-state encoding, not a motion feature.
The yaw channel is the wrapped ego yaw of the goal root orientation, not
`delta_psi_t = yaw[t+1] - yaw[t]`.

### Why not raw quaternion?

A quaternion is compact and avoids Euler singularities, but the denoiser then
has to learn the unit-norm manifold and the `q` / `-q` sign equivalence from
data. The TextOp-style roll/pitch encoding already matches the generated motion
feature convention and has a stable zero point around upright motion.

## Relation to FeatureVersion 3

FeatureVersion 3 motion features are:

```text
f_t = [
  phi(r_t),
  delta_psi_t,
  c_t,
  delta_p_local_t,
  h_t,
  q_t,
  delta_q_t,
]
```

V6 `joint_state` borrows only the parts that describe the target pose:

- keep `phi(root roll/pitch)`;
- keep a yaw term, but make it relative to the last history heading;
- keep `q_t`;
- do not encode contact `c_t`;
- do not encode `delta_q_t`;
- keep the V4 root velocity condition as `root_vel_ego`.

The generated motion representation and MVAE can remain FeatureVersion 3
(`69-D = 11 + 2 * 29`). Only the goal-conditioning vector changes.

## Arrival Time

Arrival time is carried separately from `goal`, exactly like the corrected V4
path:

```python
y = {
    "goal": joint_state_goal,                  # [B, 40]
    "time_to_arrival_frame": goal_frame - reference_frame,
    "arrival_time_frame": goal_frame - reference_frame,  # legacy alias
}
```

The denoiser adds:

```text
embed_goal(goal_content) + arrival_pe(time_to_arrival_frame)
```

Mask `arrival_pe` after the arrival-time MLP so the MLP bias cannot leak timing
information when time is dropped. Negative remaining time from deployment is
clamped to zero before the sinusoidal encoder. In the zero-time ablation,
`arrival_pe` is fully masked and the model sees no time signal.

## Goal Masking

Recommended component masks:

| Component | Slice | Training mask |
|---|---|---:|
| root position | `0:3` | `cond_goal_root_mask_prob=0.1` |
| orientation | `3:8` | `cond_goal_orientation_mask_prob=0.1` |
| joints | `8:37` | `cond_goal_joint_mask_prob=0.1` |
| root velocity | `37:40` | `0.0` initially |
| arrival PE | separate | `cond_goal_time_mask_prob=0.0` initially |

Root velocity should remain always supplied for the first V6 run, matching the
current V4 decision. If later experiments drop it, use a dedicated velocity
mask; do not overload joint or orientation masking.

## Losses and Metrics

All goal-related losses must evaluate the generated state at the selected goal
frame, not blindly at the primitive end.

Existing root losses keep their V4 semantics:

- `goal_position`: compare generated root displacement at the goal frame with
  goal slice `0:2`;
- `goal_direction`: compare generated horizontal direction at the goal frame
  with goal slice `0:2`.

V6 can add joint-state-specific losses, even if their initial weights are zero:

| Term | Target | Indexing |
|---|---|---|
| `goal_root_position_xyz` | `root_pos_ego` | `time_to_arrival_frame` |
| `goal_root_orientation` | decoded target orientation | `goal_step` on reconstructed root rotations |
| `goal_joint_position` | `q_g` | `goal_step` on reconstructed `dof_pos` |
| `goal_root_velocity` | `root_vel_ego` | selected feature delta / reconstructed finite difference |

For orientation loss, prefer quaternion geodesic distance after reconstructing
the predicted root rotation and decoding the target orientation. For
joint-position loss, use Huber/L1 on the 29 ordered joint channels; optionally
log core and wrist splits using the existing G1 index sets.

Validation should report both:

```text
sample_goal_error_m      # generated root at selected goal frame vs goal root
sample_endpoint_error_m  # generated primitive end vs terminal GT/end marker
```

Those two numbers answer different questions now that the selected goal can be
any future frame.

## Eval Visualization

The root-XY plot should show four distinct concepts:

1. reference frame / current origin;
2. generated trajectory;
3. actual goal frame on the generated trajectory;
4. primitive end on the generated trajectory.

The target goal marker should be placed at `root_pos_ego`, not at the primitive
end. The primitive-end marker is a separate visual mark so failures like
"hits the goal early but drifts by the end" or "moves correctly by the end but
misses the timed goal" are visible.

For stitched multi-primitive plots, add a primitive-end mark at every primitive
boundary and a goal-frame mark inside each primitive when
`goal_per_primitive=true`.

## Implemented Changes

### 1. Goal type and dimensions

Changed `robotmdar/utils/goal.py`:

- add `GoalType.JOINT_STATE = "joint_state"`;
- set `GoalType.JOINT_STATE.dimension = 40` for the chosen TextOp-style
  orientation encoding;
- add a property such as `uses_joint_state`;
- keep `uses_keypoints=False` for `joint_state`;
- make `validate_goal_config("joint_state", goal_dim)` strict.

Do not add a V6 compatibility shim for `root`, `body`, or `body_ext`.

### 2. Dataset extraction

Changed `robotmdar/dataloader/data.py`:

- for `goal_type=joint_state`, emit:
  - `world_goal_pos`;
  - `world_goal_rot`;
  - `world_goal_dof`;
  - `world_goal_vel`;
  - `time_to_arrival`;
  - `goal_timestep` as a temporary alias only if existing call sites still read
    it;
- skip `_world_goal_keypoints()` entirely;
- require `dof_dim=29`;
- require `goal_frame + 1 < clip_len` for root velocity;
- preserve randomized `goal_offset_range` for both training and validation.

No FK call is needed to construct the goal; all goal fields are direct slices
from the selected GT motion frame plus the raw root-position forward
difference for velocity.

### 3. Goal construction

Added a sibling helper and dispatch to it from `build_ego_goal()` when
`goal_type=joint_state`:

```python
build_ego_joint_state_goal(
    world_goal_pos,
    world_goal_rot,
    world_goal_dof,
    world_root_velocity,
    reference_pos,
    reference_rot,
)
```

It should:

- compute ego root position and root velocity using reference yaw;
- compute the 5-D TextOp-style orientation block;
- concatenate `[root_pos_ego, orientation_goal, q_g, root_vel_ego]`;
- perform explicit shape checks for `[B, 3]`, `[B, 4]`, `[B, 29]`, `[B, 3]`;
- never call FK.

### 4. Denoiser

Changed `robotmdar/model/mld_denoiser.py`:

- set V6 configs to `goal_dim=40`;
- add component masking for the new slices;
- keep `time_to_arrival_frame -> ArrivalTimeEmbedder -> arrival_pe`;
- remove the `body_ext` legacy time-slot zeroing path from the V6 branch;
- require `time_to_arrival_frame` when randomized goal offsets are enabled.

### 5. Training and validation

Changed `robotmdar/train/train_dar.py`:

- `_conditions()` should build `joint_state` goals from root pose, joint pose,
  root velocity, and reference pose;
- `_goal_time_frame_for_loss()` should work for `joint_state`;
- validation must use the same `goal_offset_range` as training;
- config validation should require:

```text
goal_type=joint_state
goal_dim=40
dof_dim=29
goal_timestep_mode=relative when goal_offset_range is not [0, 0]
```

### 6. Planner / controller protocol

For online deployment, use explicit `joint_state` goal fields:

```text
goal_type                 joint_state
goal_root_pos_world       [3]
goal_root_rot_world       [4]  # xyzw quaternion, preferred
# or:
goal_root_euler_world     [3]  # roll, pitch, yaw radians
goal_dof_pos              [29]
goal_root_velocity_world  [3]
goal_timestamp_ns
```

Controller-side ordering of `goal_dof_pos` must match
`G1_MUJOCO_DOF_JOINT_NAMES`. The planner should reject any message whose
declared `goal_type` is not `joint_state` for a V6 checkpoint.

Do not send `goal_yaw_world` or `goal_keypoints_world` for `joint_state`; yaw
and root tilt are contained in `goal_root_rot_world` / `goal_root_euler_world`,
and no keypoint/FK goal representation is used.

The communication payload is therefore:

```text
goal_dof_pos(29)
+ goal_root_velocity_world(3)
+ goal_root_pos_world(3)
+ goal_root_orientation(3 Euler or 4 quaternion)
= 38-D with Euler or 39-D with quaternion
```

`goal_timestamp_ns` is carried separately. It is not concatenated into the goal
state vector.

For planner → controller G1 motion, the low-bandwidth V6 payload is:

```text
joint_pos [T, 29]  # IsaacLab order for SONIC tracking
joint_vel [T, 29]
root_pos  [T, 3]
root_ori  [T, 4]   # wxyz quaternion for SONIC/MuJoCo
```

`body_pos` / `body_ori` should be optional debug or legacy fields, not required
for the SONIC tracker.

`goal_reference_path` is no longer needed because there are no FK keypoints to
load from a reference pose.

Repository boundary: the TextOp planner conversion accepts these fields, but
the canonical `sonicmsg` wire schema is maintained outside `TextOpRobotMDAR`
under `occHIPC/sonic-msg`. Update that package in the same protocol revision
before running online V6 deployment.

### 7. Tests

Regression coverage now includes:

- `GoalType.JOINT_STATE` dimension and config validation;
- 40-D `joint_state` layout and `build_ego_goal()` dispatch;
- planner conversion from `goal_root_rot_world`, `goal_dof_pos`,
  `goal_root_velocity_world`, and timestamp fields;
- no FK/keypoint extraction during dataset `joint_state` goal extraction;
- componentwise 40-D goal masking;
- selected-frame root, orientation, joint, and velocity goal losses;
- eval plots marking goal frame and primitive end separately.

## Recommended Initial Config

```yaml
data:
  dof_dim: 29
  goal_type: joint_state
  goal_per_primitive: true
  goal_offset: 0
  goal_offset_range: [-63, 0]
  goal_timestep_mode: relative
  val:
    goal_offset_range: ${..goal_offset_range}

denoiser:
  goal_dim: 40
  cond_goal_root_mask_prob: 0.1
  cond_goal_orientation_mask_prob: 0.1
  cond_goal_joint_mask_prob: 0.1
  cond_goal_velocity_mask_prob: 0.0
  cond_goal_time_mask_prob: 0.0

train:
  manager:
    loss_weight:
      goal_direction: 0.0
      goal_position: 0.0
      goal_root_orientation: 0.0
      goal_joint_position: 0.0
      goal_root_velocity: 0.0
```

The zero weights keep the first implementation behaviorally close to V4 while
ensuring the loss and eval code is already indexing the correct goal frame.

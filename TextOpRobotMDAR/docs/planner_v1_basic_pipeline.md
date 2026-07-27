# planner_dar.py — Design Plan

## 1. Objective

Rewrite `loop_dar.py` into a **closed-loop replanning planner** that communicates with the controller via ZeroMQ pubsub and generates motion at a fixed period (50 ms = 20 Hz).

Key changes from `loop_dar.py`:

| Aspect | loop_dar.py | planner_dar.py |
|--------|-------------|----------------|
| Inference cadence | Playback-gated (~255 ms per block) | Fixed 50 ms timer, decoupled from playback |
| History source | Model's own previous output (autoregressive) | Latest 3 physical states selected from controller's 5-state buffer |
| Goal source | Terminal input `x y z yaw` (yaw in degrees) | Controller via ZMQ (`goal_root_pos` + `goal_heading`, yaw in radians) |
| Communication | None | ZMQ pubsub (sonic-msg) |
| Visualization | Blocking main thread, `time.sleep` per frame | None; controller provides visualization |
| Output | MuJoCo viewer only | ZMQ PUB `G1MotionData` (8 frames at 50 Hz) |
| Scene occupancy | All zeros | All zeros (extensible) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    planner_dar.py                           │
│                                                             │
│  ┌──────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │ ZMQ SUB  │───▶│  State Buffer    │───▶│  Inference    │  │
│  │ (state)  │    │  (latest state)  │    │  Engine       │  │
│  └──────────┘    └──────────────────┘    │  50 ms timer  │  │
│                                          │  DAR model    │  │
│  ┌──────────┐    ┌──────────────────┐    │  timed w/sync │  │
│  │ ZMQ PUB  │◀───│  Motion Batch    │◀───│               │  │
│  │ (motion) │    │  G1MotionData    │    └───────────────┘  │
│  └──────────┘    └──────────────────┘                       │
└─────────────────────────────────────────────────────────────┘

  Controller (external process)
  ┌─────────────────┐          ┌─────────────────┐
  │ PUB textop_v1   │─────────▶│ SUB (planner)   │
  │  50 Hz          │          │                 │
  ├─────────────────┤          ├─────────────────┤
  │ SUB motion_g1   │◀─────────│ PUB (planner)   │
  │                 │          │  20 Hz          │
  └─────────────────┘          └─────────────────┘
```

---

## 3. Communication Protocol (sonic-msg pubsub)

We use `sonicmsg.PlannerNode` for ZMQ socket management but **not its `spin()` method** — we run our own main loop to control the 50 ms inference period.

### 3.1 Versioned TextOp State Message

The legacy `history_state` message is shared with MOB. Changing its frame 3 from yaw to a quaternion would break existing MOB producers and consumers. Therefore TextOp uses a separate, versioned message, `history_state_textop_v2`, while the existing `HistoryStateHeader`, `MobHistoryStateHeader`, and `send_history_state()` remain unchanged.

`TextOpHistoryStateHeader` has `planner="textop"` and `protocol_version=2` and is sent by `send_textop_history_state()`. Its JSON header also carries `tracked_plan_seq` and `tracked_plan_start_t_ns`, identifying the active reference and the controller timestamp of its first policy sample.

Wire format:

```
frame 0: JSON TextOpHistoryStateHeader
frame 1: g1_pos         [n, 3]   float32  — root world position (Z-up)
frame 2: g1_root_rot    [n, 4]   float32  — root orientation (xyzw quaternion)
frame 3: g1_joint_pos   [n, 29]  float32  — joint angles (IsaacLab order)
frame 4: goal_root_pos  [3]      float32  — goal world position (Z-up)
frame 5: goal_heading   [1]      float32  — goal yaw (radians)
```

The controller publishes this variant only when `planner="textop"`. MOB continues to receive the original five-frame `history_state` message, including `g1_fdir`, scalar `g1_heading`, and `tgt_limb_abs`.

### 3.2 Receive: HistoryState → State + Goal

Controller publishes at 50 Hz after its five-entry history buffer is full. The planner keeps only the latest message:

The TextOp wire frame is Z-up with +X forward. The controller bridge rotates
its Z-up, +Y-forward state and goal by -90 degrees about Z before publishing:
`(x, y, z)_textop = (y, -x, z)_controller`. Published TextOp motion is rotated
back by the inverse transform before SONIC tracks it.

```python
state = node.recv_state(timeout_ms=1)
# state.current_pos        → np.ndarray [3]   root world position
# state.current_root_rot   → np.ndarray [4]   root orientation (xyzw quat)
# state.current_joint_pos  → np.ndarray [29]  joint angles (IsaacLab order)
# state.n_states           → int              5 history entries
# state.goal_root_pos      → np.ndarray [3]   required goal world position
# state.goal_heading       → np.ndarray [1]   required goal yaw in radians
# state.raw                → dict             full decoded buffers for history access
```

### 3.3 Send: MotionG1 (8 frames default)

```
Header (JSON):
  msg_type: "motion_g1"
  seq: int
  framerate: 50.0
  num_frames: 8            ← future frames only (no history prefix)

Buffer frames:
  frame 1: joint_pos   [8, 29] float32  (IsaacLab order)
  frame 2: joint_vel   [8, 29] float32
  frame 3: body_pos    [8, N, 3] float32
  frame 4: body_ori    [8, N, 4] float32  (wxyz)
```

Publishing 10 frames (history + future) is gated behind `pub_all_frames: true` in config.

---

## 4. Core Data Conversions

The model operates in **MuJoCo 23-DoF + 57-dim feature (FeatureVersion=3)** space.
The controller operates in **IsaacLab 29-DoF** space.

### 4.1 Controller State → Model History Feature

#### Conversion chain

```
Controller State (IsaacLab 29-DoF)
  │
  ├─[Step 0] Select the latest 3 physical states from the 5-state buffer
  │   FeatureVersion 3 stores forward deltas, so 3 poses produce 2 features.
  │
  ├─[Step 1] IsaacLab 29 → MuJoCo 29
  │   Inverse mapping: ISAAC2MJC = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
  │                                 2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
  │                                 12, 16, 20, 22, 24, 26, 28]
  │
  ├─[Step 2] MuJoCo 29 → MuJoCo 23
  │   Remove 6 wrist DoFs at MuJoCo indices [19,20,21,26,27,28]
  │   Keep: [0:19] + [22:26] = 23 DoF
  │
  ├─[Step 3] Build a raw MotionDict with 3 poses and call
  │   motion_dict_to_feature_v3() → 2 feature frames — see §4.2
  │
  └─[Step 4] Normalize
       val_data.normalize(raw_feature_tensor) → history_motion [1, 2, 57]
```

Do not manually construct forward deltas. Reusing `motion_dict_to_feature_v3()` preserves the training-time feature semantics. Its returned `abs_pose` is anchored to the oldest of the three selected physical states:

```python
abs_pose = {
    'root_trans_offset': g1_pos[-3].reshape(1, 3),       # [1, 3]
    'root_rot': g1_root_rot[-3].reshape(1, 4),            # [1, 4] xyzw quaternion
}
```

### 4.2 57-dim Feature Layout & Controller State Mapping

FeatureVersion=3, `nfeats=57`, `DOF_DIM=23`:

```
Index    Dim  Name               loop_dar init           Planner from controller
─────────────────────────────────────────────────────────────────────────────────
[0:4]    4    root_rot           [0,0,0,0]               Derived from g1_root_rot quaternion:
         sin_roll, cos_roll-1,   (roll=0, pitch=0)       existing quaternion-to-Euler helper
         sin_pitch, cos_pitch-1
[4]      1    delta_yaw          0                        Forward yaw delta, state[t+1]-state[t]
[5:7]    2    contact_mask       [1.0, 1.0]              [1.0, 1.0] (standing assumption)
[7:10]   3    delta_trans_local  [0, 0, 0]               Forward position delta in state[t] yaw frame
[10]     1    height             0.75                     g1_pos[-1][2] (root height)
[11:34]  23   dof                default standing pose   MuJoCo 23-DoF joint angles
[34:57]  23   delta_dof          [0]*23                  Forward DoF delta, state[t+1]-state[t]
```

#### Contact (indices 5:7)

Left/right foot contact probability (∈ [0, 1]). In loop_dar:
- Initialized via `get_zero_feature_v2()` as `[1.0, 1.0]` (both feet on ground)
- Subsequently produced autoregressively by the model's own output

The controller does not provide contact info. Planner hardcodes `[1.0, 1.0]`. Impact is minimal — with only 2 history frames, the VAE decoder primarily relies on joint positions/velocities for continuity conditioning. (Future: add foot contact estimation on the controller side and include it in the state message.)

#### Root rotation (indices 0:4)

With `g1_root_rot` [4] (xyzw quaternion) from the controller, reuse `quaternion_to_euler_angles()` from `robotmdar.dtype.motion`. If a standalone NumPy conversion is needed, the correct canonical XYZ formulas are:

```python
def quat_to_roll_pitch_sincos(q_xyzw):
    """Convert xyzw quaternion to sin/cos encoded roll and pitch."""
    x, y, z, w = q_xyzw
    # roll (x-axis rotation)
    sin_roll = 2 * (w*x + y*z)
    cos_roll = 1 - 2 * (x*x + y*y)
    roll = np.arctan2(sin_roll, cos_roll)
    # pitch (y-axis rotation)
    sin_pitch = 2 * (w*y - z*x)
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    return np.array([np.sin(roll), np.cos(roll)-1,
                     np.sin(pitch), np.cos(pitch)-1])
```

After `motion_dict_to_feature_v3()` returns, wrap feature index 4 with `atan2(sin(delta_yaw), cos(delta_yaw))` before normalization. This avoids a `2*pi` discontinuity at the Euler-yaw branch boundary without duplicating the rest of the feature conversion.

### 4.3 Model Output → G1MotionData

```
motion_dict (model output, MuJoCo space; 2 history + 8 future frames)
  │
  ├─ dof_pos [B, T, 23]  ─── pad 23→29 ─── reorder MuJoCo→IsaacLab ─── joint_pos [T, 29]
  ├─ dof_pos [B, T, 23]  ─── finite difference at 50 Hz ─── dof_vel [B, T, 23]
  │                              then same pad/reorder pipeline ─── joint_vel [T, 29]
  ├─ global_translation [B, T, N, 3]  ─── direct use (FK, world coords) ─── body_pos [T, N, 3]
  └─ global_rotation [B, T, N, 4] xyzw ─── convert to wxyz ─── body_ori [T, N, 4]
```

`RobotSkeleton.forward_kinematics()` currently defaults to `dt=1/30`; its `dof_vel` must not be published as 50 Hz velocity. Either pass `dt=1/50` through FK or recompute `dof_vel` from `dof_pos` with `dt=0.02`, repeating the final valid velocity for the last frame.

Pad and reorder logic (reuses verified code from loop_dar.py `_npz_expand_23_to_29` and `_NPZ_MJC2ISAAC`):

```python
# 23 → 29: pad wrist DoFs
out = np.zeros((T, 29))
out[:, :19] = v[:, :19]       # L_leg(6) + R_leg(6) + waist(3) + L_arm(4) = 19
out[:, 22:26] = v[:, 19:23]   # R_arm(4)
# wrist DoFs (3+3) remain zero

# MuJoCo 29 → IsaacLab 29
MJC2ISAAC = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
             5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
joint_pos_isaaclab = out[:, MJC2ISAAC]
```

---

## 5. Goal Construction

### 5.1 Format: xyz + yaw (radians)

Same as loop_dar: world-space target position + target yaw angle.

| Field | Source | Unit |
|-------|--------|------|
| `goal_root_pos` [3] | `state.goal_root_pos` | meters (Z-up) |
| `goal_heading` [1] | `state.goal_heading` | radians |

yaw=0° = +X direction, 90° = +Y.

### 5.2 Ego-centric conversion

```python
def state_to_ego_goal(state_msg) -> torch.Tensor:
    """goal_root_pos [3] + goal_heading [1] rad → ego_goal [1, 5]."""
    world_goal_pos = torch.tensor(state_msg.goal_root_pos).float().unsqueeze(0)  # [1, 3]
    world_goal_yaw = torch.tensor(state_msg.goal_heading).float()                # [1]

    # Training uses the last history-feature pose as the ego reference.
    # With 3 physical states -> 2 features, this is physical state -2.
    reference_pos = torch.tensor(state_msg.raw['g1_pos'][-2]).reshape(1, 3)
    reference_rot = torch.tensor(state_msg.raw['g1_root_rot'][-2]).reshape(1, 4)
    return build_ego_goal(world_goal_pos, world_goal_yaw,
                          reference_pos, reference_rot)  # [1, 5]
```

Directly reuses `build_ego_goal()` from loop_dar. Does NOT use `tgt_limb_abs`.

---

## 6. Main Loop Design

### 6.1 Timing

The controller publishes only after its five-state buffer is full. Since FeatureVersion 3 needs `history_len + 1 = 3` physical states to produce two feature frames, the planner can infer immediately on the first TextOp message.

```
Time (ms):   0    20   40   50   60   80   100
Controller: S0───S1───S2──────S3───S4──────S5      (50 Hz state publication)
Inference:  I0────────────I1─────────────I2          (20 Hz replanning)
Publish:    P0(8 fr)      P1(8 fr)       P2(8 fr)   (motion samples tagged 50 Hz)
```

The 50 ms replan period is independent of the motion sample period: each published eight-frame batch represents 160 ms of motion at 50 Hz. GPU timing must synchronize CUDA before starting and before stopping the wall-clock timer.

### 6.2 Pseudocode

Only the first plan is bootstrapped from controller history. Later plans look
up the generated plan identified by `tracked_plan_seq`, calculate its consumed
50 Hz frame from the controller timestamps, and select the two features ending
at that frame. Optional world-space alignment translates that selected history
endpoint onto the current G1 root without replacing its local features or
generated rotation.

```python
def main(cfg):
    # --- Model loading (same as loop_dar) ---
    vae, denoiser, diffusion, val_data = load_models(cfg)
    future_len, history_len = cfg.data.future_len, cfg.data.history_len  # 8, 2
    grid_size = cfg.denoiser.grid_size

    # --- Communication ---
    node = PlannerNode(comm_config)  # ZMQ SUB + PUB
    latest_state: StateMessage | None = None
    generated_plans = {}

    # --- Main loop ---
    period = 0.050  # 50 ms = 20 Hz
    next_infer_time = time.perf_counter()

    while True:  # daemon; Ctrl+C stops the process
        # 1. Non-blocking drain: keep only the latest state
        while True:
            msg = node.recv_state(timeout_ms=1)
            if msg is None:
                break
            latest_state = msg

        # 2. Inference time?
        now = time.perf_counter()
        if now >= next_infer_time and latest_state is not None:
            tracked_plan = generated_plans.get(latest_state.tracked_plan_seq)
            if tracked_plan is not None:
                # a1. History ending at the frame actually tracked by SONIC
                tracked_frame = tracked_frame_from_timestamps(
                    latest_state, motion_fps, future_len)
                history_motion, abs_pose, reference_pos, reference_rot = (
                    generated_history_at_frame(
                        tracked_plan, tracked_frame, history_len))
                ego_goal = state_goal_from_reference(
                    latest_state, reference_pos,
                    reference_rot, device)
            elif latest_state.n_states >= history_len + 1:
                # a2. Bootstrap from controller state
                history_motion, abs_pose = state_to_model_input(
                    latest_state, history_len, val_data, device)
                ego_goal = state_to_ego_goal(latest_state)
            else:
                continue
            voxel = torch.zeros(1, grid_size**3, device=device)

            # b. Inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            future_motion, motion_dict, generated_abs_pose = generate_next_motion(
                vae=vae, denoiser=denoiser, diffusion=diffusion,
                val_data=val_data, goal=ego_goal, voxel=voxel,
                history_motion=history_motion, abs_pose=abs_pose,
                future_len=future_len,
                use_full_sample=cfg.use_full_sample,
                guidance_scale=cfg.guidance_scale,
                ret_fk=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_finished = time.perf_counter()
            log_infer_time(t0, infer_finished)

            # c. Convert, publish, and retain generated state
            g1_motion = motion_dict_to_g1data(motion_dict,
                skip_history=history_len if not cfg.pub_all_frames else 0)
            node.publish_motion(g1_motion)
            generated_plans[published_seq] = {
                'features': torch.cat((history_motion, future_motion), dim=1),
                'root_pos': motion_dict['root_trans_offset'],
                'root_rot': motion_dict['root_rot'],
            }

            scheduled_next = next_infer_time + period
            finished = time.perf_counter()
            next_infer_time = (
                finished + period if finished > scheduled_next
                else scheduled_next
            )

        time.sleep(0.001)

```

---

## 7. Process Model

`planner_dar.py` is a headless daemon. It does not initialize MuJoCo or create a visualization thread because the controller already visualizes the robot and current reference. The normal stop mechanism is Ctrl+C; process teardown closes the ZMQ sockets.

---

## 8. Configuration

```yaml
# planner_dar.yaml
defaults:
  - loop_dar
  - _self_

task: planner-dar

# Planner loop
infer_period_ms: 50          # inference period (ms), i.e. 20 Hz
pub_all_frames: false        # false = 8 frames (future only), true = 10 frames
use_full_sample: false       # single-step denoising for the 20 Hz deadline
use_generated_history: true
align_generated_history_to_g1: false  # pure autoregressive debug mode

# Communication
comm_config: "robotmdar/config/communication/pubsub.yaml"

# Scene
occ_grid: null               # null = all-zero voxel
```

---

## 9. File Layout

```
TextOpRobotMDAR/robotmdar/eval/
├── loop_dar.py          # Original: terminal input + time.sleep playback
├── planner_dar.py       # New: headless ZMQ pubsub + fixed-period inference
├── generate_dar.py      # Shared: generate_next_motion()
│
TextOpRobotMDAR/robotmdar/config/
├── loop_dar.yaml
├── planner_dar.yaml
├── communication/
│   └── pubsub.yaml      # Shared ports, 50 ms replan period, 5-state buffer
│
TextOpRobotMDAR/robotmdar/utils/
├── planner_convert.py   # Conversion utilities:
│                        #   state_to_model_input()
│                        #   state_to_ego_goal()
│                        #   motion_dict_to_g1data()
│                        #   isaaclab_to_mujoco_dof()
│                        #   mujoco_to_isaaclab_dof()
│
sonic-msg/sonicmsg/
├── messages.py          # [MODIFIED] versioned TextOpHistoryStateHeader
├── planner_node.py      # [MODIFIED] StateMessage supports TextOp root rotation
```

---

## 10. Summary of sonic-msg Changes

The sonic-msg changes are implemented in the sibling `occHIPC/sonic-msg` package. Controller publication is implemented in `occHIPC/closed_loop/planner_bridge.py` and `sim_client.py`.

### Change A: Add `history_state_textop_v2`

| Location | Change |
|----------|--------|
| `TextOpHistoryStateHeader` | Header with `protocol_version=2`, quaternion history, root-heading goal, and active-plan timing |
| `send_textop_history_state()` | New five-buffer sender used only by TextOp |
| `decode_history_state_buffers()` | Dispatches legacy and TextOp layouts by typed header |
| `decode_state_entry()` | Returns `g1_root_rot` for TextOp and legacy heading fields for MOB |
| `StateMessage` | Exposes `current_root_rot`; retains `current_heading_rad` for compatibility |

The original `history_state` wire format and sender are unchanged.

### Change B: Publish measured TextOp state and goal

| Location | Change |
|----------|--------|
| `PubSubControllerBridge.publish_state()` | Accept `g1_root_rot`, `goal_root_pos`, and `goal_heading`; require them when `planner="textop"` |
| `sim_client.py` | Select `planner="textop"`, convert measured MuJoCo `wxyz` to wire `xyzw`, and publish configured target root position/yaw |
| `textop.yaml` | Identifies the external TextOp daemon to the controller |

---

## 11. Risks & Notes

1. **No bootstrap wait**: Controller publishes after all 5 states are present. This exceeds the 3 physical states required for 2 feature frames.

2. **Feature approximation**: contact `[1,1]` is a simplification. With 2-frame history, the VAE decoder's continuity conditioning is dominated by joint pos/vel — contact error has minimal impact. Future: add foot contact estimation to controller state.

3. **Root orientation**: With `g1_root_rot` [n, 4] we get proper roll/pitch from the controller. No more zeroing them out.

4. **Coordinate consistency**: Must confirm controller's `g1_joint_pos` is IsaacLab 29-DoF order, `g1_pos` is Z-up world, `g1_root_rot` is xyzw quaternion.

5. **ZMQ connection**: Planner PUB binds, controller SUB connects — initial messages may be dropped (standard ZMQ PUB semantics). Controller should use the latest received motion.

6. **Yaw wrapping**: Forward yaw differences must be wrapped to `[-pi, pi]` before normalization.

7. **Velocity timestep**: Published joint velocity must be derived with `dt=0.02`, not FK's current `1/30` default.

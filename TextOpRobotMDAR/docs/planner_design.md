# RobotMDAR Planner — Design & Communication Architecture

## 1. Overview

RobotMDAR is a **text-conditioned motion diffusion model** that generates kinematic reference motions for the Unitree G1 humanoid robot. It operates as the "planner" in a two-stage pipeline:

```
[Text Input] → RobotMDAR (Planner) → Reference Motion → Tracker (RL Policy) → Robot
```

The planner generates motions in an autoregressive loop — each step produces 8 future frames conditioned on 2 history frames and a CLIP text embedding. The generated motion is a sequence of joint-space poses (23 DoF + root) at 30 fps.

---

## 2. Model Architecture

### 2.1 Component Overview

```
┌──────────────────────────────────────────────────────────┐
│                     RobotMDAR Pipeline                    │
│                                                          │
│  Text ──▶ [CLIP ViT-B/32] ──▶ text_emb [1, 512]         │
│                                                          │
│  History Motion ──▶ [Normalize] ──▶ history [1, 2, 57]   │
│                                                          │
│  Noise ──▶ [DenoiserTransformer] ◀── timestep + cond     │
│              │ 8 layers, 4 heads, d=512                  │
│              ▼                                           │
│            latent [1, 1, 128]                             │
│              │                                           │
│              ▼                                           │
│            [MVAE Decoder]                                │
│              │ 9-layer SkipTransformer                   │
│              ▼                                           │
│            future_motion [1, 8, 57]                       │
│              │                                           │
│              ▼                                           │
│            [Reconstruct] → MotionDict → qpos [30]        │
└──────────────────────────────────────────────────────────┘
```

### 2.2 MVAE (Motion Variational AutoEncoder)

- **File**: `robotmdar/model/mld_vae.py`
- **Architecture**: `AutoMldVae` — SkipTransformer Encoder/Decoder
- **Config**: 9 layers, 4 heads, `h_dim=512`, `ff_size=1024`, GELU activation
- **Latent space**: 1 token × 128 dimensions
- **Encoder input**: `[history(2) + future(8)]` = 10 frames of 57-dim features
- **Encoder output**: `mu, logvar → latent [1, 128]` (via reparameterization)
- **Decoder input**: `latent [1, 128] + history [2, 57]` → `future [8, 57]`
- **Global tokens**: 2 learnable tokens (`global_motion_token`) are prepended to the encoder input sequence; they aggregate global context and are projected to μ and log σ

### 2.3 Denoiser (Diffusion Model)

- **File**: `robotmdar/model/mld_denoiser.py`
- **Architecture**: `DenoiserTransformer` — TransformerEncoder, 8 layers, 4 heads, `d=512`
- **Input sequence**: `[timestep_emb(1) + text_emb(1) + history(2) + noise(1)]` = 5 tokens
- **Input components** (concatenated along sequence dim):
  - `emb_time`: sinusoidal timestep embedding → MLP → [1, B, 512]
  - `emb_text`: CLIP text embedding [512] → Linear → [1, B, 512]
  - `emb_history`: history motion [2, 57] → Linear → [2, B, 512]
  - `emb_noise`: noisy latent [1, 128] → Linear → [1, B, 512]
- **Output**: denoised latent `[B, 1, 128]`
- **Noise schedule**: 5-step cosine beta schedule (`num_timesteps=5`)
- **Prediction target**: `START_X` (predicts x₀ directly, not ε)
- **Classifier-Free Guidance**: `ClassifierFreeWrapper` in `generate_dar.py:17-35`:
  ```python
  out = out_uncond + guidance_scale * (out_cond - out_uncond)
  ```
  With `cond_mask_prob=0.1` (10% of training samples had masked text conditions)

### 2.4 CLIP Text Encoder

- **File**: `robotmdar/model/clip.py`
- **Model**: OpenAI CLIP ViT-B/32 (frozen weights, eval-only)
- **Output**: 512-dimensional embedding vector
- **Normalization**: Not L2-normalized (raw float32 output)
- **Empty text**: forced to zero embedding (`text_embedding[''] = 0`)

---

## 3. Motion Representation & Kinematics

### 3.1 Feature Space (FeatureVersion = 3)

- **Dimension**: `nfeats = 57`
- **Composition** (`motion.py:395-453`):
  ```
  [sin_roll, cos_roll-1, sin_pitch, cos_pitch-1]  (4)  — root orientation
  [delta_yaw]                                       (1)  — yaw change per frame
  [contact_mask]                                    (2)  — left/right foot contact
  [delta_trans_local]                               (3)  — displacement in local frame
  [height]                                          (1)  — root height
  [dof]                                             (23) — joint angles (G1 23-DoF)
  [delta_dof]                                       (23) — joint delta per frame
  ```

### 3.2 Feature → MotionDict Reconstruction

`motion_feature_to_dict_v3` (motion.py:483-562):
1. Recover roll/pitch from sin/cos via `atan2`
2. Recover yaw via cumulative sum of `delta_yaw` + reference yaw from `abs_pose`
3. Rotate `delta_trans_local` back to world frame via per-frame yaw quaternion
4. Recover absolute translation via cumulative sum of world deltas
5. Set height directly from features
6. Output: `{root_trans_offset, root_rot, dof, contact_mask}`

### 3.3 MotionDict → MuJoCo qpos

`motion_dict_to_qpos` (motion.py:98-105):
```python
qpos[..., :3]  = root_trans_offset     # xyz
qpos[..., 3:7] = root_rot              # xyzw quaternion
qpos[..., 7:]  = dof                   # 23 joint angles
# Total: 30 dimensions
```

### 3.4 G1 Robot Kinematics

- **File**: `robotmdar/skeleton/forward_kinematics.py`
- The skeleton is parsed from the G1 MJCF XML (`g1_23dof_lock_wrist.xml`)
- 23 actuated DoF (wrists locked): 6×leg + 2×waist + 7×left_arm + 7×right_arm
- Forward kinematics: recursive chain from root, computing world positions/rotations for all 27 bodies
- `RobotSkeleton.forward_kinematics()` wraps FK + DOF→axis-angle conversion

---

## 4. Key Inference Pipeline

### 4.1 Core Generation Function

`generate_dar.py:42-165` — `generate_next_motion()`:

```python
def generate_next_motion(vae, denoiser, diffusion, val_data,
                         text_embedding, history_motion, abs_pose,
                         future_len, use_full_sample, guidance_scale, ...):
    # 1. Prepare conditioning dictionary
    y = {
        'text_embedding': text_embedding,         # [1, 512]
        'history_motion_normalized': history_motion, # [1, 2, 57]
        'scale': guidance_scale,                   # e.g. 5.0
    }

    # 2. Full DDPM sampling loop (5 steps with cosine schedule)
    x_start_pred = diffusion.p_sample_loop(
        denoiser,                                  # ClassifierFreeWrapper
        latent_shape=(1, 1, 128),
        model_kwargs={'y': y},
        ...
    )  # → [1, 1, 128]

    # 3. VAE decode: latent → motion features
    latent_pred = x_start_pred.permute(1, 0, 2)   # [1, 1, 128]
    future_motion_pred = vae.decode(
        latent_pred, history_motion, nfuture=8
    )  # → [1, 8, 57]

    # 4. Reconstruct motion dict + FK
    motion_dict = val_data.reconstruct_motion(
        torch.cat([history_motion, future_motion_pred], dim=1),
        abs_pose=abs_pose, ret_fk=True
    )
    new_abs_pose = motion_dict_to_abs_pose(motion_dict, idx=-2)

    return future_motion_pred, motion_dict, new_abs_pose
```

### 4.2 Autoregressive Loop

`loop_dar.py:200-244` — Main generation loop:

```python
while not quit and viewer.is_running():
    if text_changed:
        text_embedding = get_text_embedding(text_prompt, clip_model, device)

    if not paused:
        future_motion, motion_dict, abs_pose = generate_next_motion(...)

        # Slide window: use last history_len frames as next history
        history_motion = future_motion[:, -history_len:, :]

        # Visualize frame-by-frame
        qpos, contact = motion_dict_to_qpos(motion_dict)
        for t in range(qpos.shape[1]):
            show_fn(qpos[0, t], contact[0, t])
            time.sleep(dt)
```

### 4.3 Online (ROS2) Generation Loop

`rmdar.py:451-503` — The `MotionDAR` ROS2 node:

```python
def loop(self):  # called at 50Hz by ROS2 timer
    if not self._start_infer:
        return

    # Trigger generation when buffer runs low
    if self._gen_counter - self._current_block_size - self._counter <= self.future_len:
        self._publish_motion_block()   # Publish cached MotionBlock
        self._gen_motion()             # Generate next block

    # Time-based frame counter (absolute, not incremental!)
    absolute_elapsed = current_time - self._toggle_time
    self._counter = int(np.floor(absolute_elapsed / self.dt))  # dt=0.02s

def _gen_motion(self):
    future_motion, gen_motion_dict, abs_pose = self.dar_gen_fn(
        text_embedding=self._text_embedding,
        history_motion=self.history_motion,
        abs_pose=self.history_abs_pose
    )
    self.history_motion = future_motion[:, -history_len:, :]
    self.history_abs_pose = abs_pose
    self._ref_motion_dict = gen_motion_dict[..., -future_len:]
    self._convert_motion_to_msg_instance()  # Serialize to MotionBlock
```

---

## 5. Text Embedding Pipeline

1. User types text in terminal → captured by keyboard input thread (`rmdar.py:544-578`)
2. Text stored in `self._text_prompt`
3. `_update_text_embedding()` is called (`rmdar.py:534-542`):
   ```python
   def _update_text_embedding(self):
       with torch.no_grad():
           text_embedding = encode_text(self.clip_model, [self._text_prompt])
           self._text_embedding = text_embedding.float()  # [1, 512]
   ```
4. `encode_text()` in `clip.py:17-25`:
   - Tokenizes text via `clip.tokenize(texts)` → [1, 77]
   - Runs `clip_model.encode_text(tokens)` → [1, 512]
   - Returns raw float32 (not normalized)
   - Empty string → zero vector
5. The embedding is passed to the denoiser as conditioning — it is concatenated into the transformer input sequence

---

## 6. NPZ Saving

### 6.1 Format

The C++ Tracker expects motion data in `.npz` format with these keys:

| Key | Shape | Description |
|---|---|---|
| `joint_pos` | [T, 29] | Joint positions (IsaacLab order, 29-DoF) |
| `joint_vel` | [T, 29] | Joint velocities |
| `body_pos_w` | [T, 1, 3] | Anchor body world position (pelvis) |
| `body_quat_w` | [T, 1, 4] | Anchor body world orientation (wxyz) |
| `fps` | [1] | Frames per second (30) |

### 6.2 ROS-based Saving

The `MotionBlockSubscriber` (`motion_block_subscriber.cpp:217-279`) accumulates received ROS `MotionBlock` messages and saves them via `cnpy::npz_save()`.

### 6.3 Planner-side Direct Saving

For offline/standalone use, the planner can directly export generated motions to NPZ without ROS. The `_convert_motion_to_msg_instance()` method in `rmdar.py:273-318` already serializes the motion dict into a structured format; the same logic can produce NPZ files directly:

```python
def save_motion_to_npz(motion_dict, path, fps=30):
    # motion_dict from reconstruct_motion(ret_fk=True)
    joint_pos = motion_dict['dof_pos'][0].cpu().numpy()  # [T, 23] → need 29
    joint_pos_29 = expand_dof_23_to_29(joint_pos)         # pad wrist DoFs
    joint_pos_isaaclab = joint_pos_29[:, mujoco_to_isaaclab_reindex]
    root_pos = motion_dict['root_trans_offset'][0].cpu().numpy()
    root_rot = motion_dict['root_rot'][0].cpu().numpy()[:, [3,0,1,2]]  # xyzw→wxyz

    np.savez(path,
             joint_pos=joint_pos_isaaclab,     # [T, 29]
             body_pos_w=root_pos[:, None, :],  # [T, 1, 3]
             body_quat_w=root_rot[:, None, :], # [T, 1, 4]
             fps=np.array([fps]))
```

---

## 7. Original Communication Architecture (ROS2)

### 7.1 Topology

```
┌──────────────────────┐                     ┌──────────────────────────┐
│  PC (GPU)            │                     │  PC / G1 Onboard (CPU)   │
│                      │    ROS2 Topic       │                          │
│  rmdar.py            │ ═══ /dar/motion ═══▶│  textop_onnx_controller  │
│  (MotionDAR Node)    │   MotionBlock msg   │  (C++, ONNX Runtime)     │
│                      │                     │                          │
│  50Hz timer loop     │ ◀══ /dar/toggle ════│  Gamepad: Start→A        │
│  Keyboard input      │    Time msg         │                          │
└──────────────────────┘                     └──────────┬───────────────┘
                                                        │
                                            ┌───────────▼───────────────┐
                                            │ MuJoCo Sim / Real G1      │
                                            │ LowCmd (PD targets, 500Hz)│
                                            │ LowState (motor feedback) │
                                            └───────────────────────────┘
```

### 7.2 ROS2 Message: MotionBlock

Custom message (`textop_ctrl/msg/MotionBlock`):
```
int32 index                    # Block index in global timeline
Time timestamp                 # ROS2 timestamp
Float32MultiArray joint_positions    # [T=8, Nq=29]
Float32MultiArray joint_velocities   # [T=8, Nq=29]
Float32MultiArray anchor_body_ori    # [T=8, 4] wxyz quaternion
Float32MultiArray anchor_body_pos    # [T=8, 3] xyz position
```

### 7.3 Communication Sequence

```
Time ──────────────────────────────────────────────────────▶

Tracker:  [Start]──[A]──[init done]──[motion_t_++ each 20ms]──[buffer end→lock]
               │      │       │
Toggle:        └──────┘       │
               dar/toggle     │
Planner:                      └──[Warmup 3×]──[gen block#0]──[gen block#8]──...
                                            publish#0       publish#8
Tracker motion buffer:
              [0][1]...[7][8][9]...[15][16]...
               ↑
          motion_t_ (consumed at 50Hz)
```

### 7.4 Latency Handling Mechanisms

1. **Buffer-ahead**: Planner triggers new generation when remaining frames ≤ `future_len` (8 frames ≈ 0.27s buffer)
2. **Absolute-time counter**: `_counter` computed from wall-clock time since toggle, not frame counting. Absorbs transient delays without cumulative drift.
3. **Block index random access**: Each `MotionBlock` has an `index` field; Tracker places data at exact position in buffer. Tolerates out-of-order or delayed delivery.
4. **Safety lock on underrun**: If Tracker reaches the end of available motion, it locks at the last valid frame (`lock_t_ = true`) instead of extrapolating.
5. **TensorRT acceleration**: `torch.compile(backend='tensorrt')` or `torch_tensorrt.compile()` for the VAE decoder path.
6. **Warmup phase**: 3 inference rounds before live operation to trigger JIT compilation.

---

## 8. Lightweight Communication Architecture (ZeroMQ)

### 8.1 Design Rationale

Replace ROS2 with ZeroMQ for a lighter, dependency-minimal communication layer. ZeroMQ provides:
- **No broker/roscore**: Direct peer-to-peer messaging
- **Cross-platform**: Python ↔ C++ via libzmq
- **Message patterns**: PUB/SUB for motion streaming, REQ/REP for control
- **Minimal overhead**: ~microsecond latency on localhost

### 8.2 ZeroMQ Topology

```
┌──────────────────────┐                    ┌──────────────────────────┐
│  Planner (Python)    │                    │  Tracker (C++)            │
│                      │  ZMQ PUB/SUB       │                          │
│  zmq_planner.py      │  ═══ tcp://*:5555 ═══▶│  zmq_tracker.cpp         │
│  (MotionGenerator)   │    MotionBlock (msgpack) │  (ONNX Policy Runner)    │
│                      │                    │                          │
│  50Hz loop           │  ◀── tcp://*:5556 ──│  Gamepad control           │
│  Keyboard input      │    REQ/REP (toggle) │                          │
└──────────────────────┘                    └──────────────────────────┘
```

### 8.3 Message Serialization: MessagePack

MessagePack is chosen over JSON for binary efficiency and over Protobuf for zero-codegen simplicity:

```python
# MotionBlock schema (MsgPack)
{
    "index": int32,              # Block index in global timeline
    "timestamp": float64,        # Seconds since epoch
    "fps": 30,
    "T": 8,                      # Frames in this block
    "num_joints": 29,
    "joint_positions": [float32 * (T * 29)],   # Flat array, row-major
    "joint_velocities": [float32 * (T * 29)],
    "anchor_body_ori": [float32 * (T * 4)],    # wxyz
    "anchor_body_pos": [float32 * (T * 3)],    # xyz
}
```

### 8.4 Python Planner Implementation

```python
"""zmq_planner.py — ZeroMQ-based RobotMDAR motion planner."""
import zmq
import msgpack
import numpy as np
import time
import threading
from pathlib import Path

# --- Reuse existing RobotMDAR model loading (from rmdar.py load_dar()) ---
# [Same model initialization code as rmdar.py:344-421]
# clip_model, vae, denoiser, diffusion, dataset, dar_gen_fn, ...


class ZMQMotionPlanner:
    """Lightweight motion planner using ZeroMQ PUB/SUB + REQ/REP."""

    def __init__(self, pub_port=5555, ctrl_port=5556, dt=0.02):
        self.ctx = zmq.Context()

        # PUB socket: stream motion blocks to Tracker
        self.pub_socket = self.ctx.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{pub_port}")

        # REP socket: receive start/stop toggle from Tracker
        self.ctrl_socket = self.ctx.socket(zmq.REP)
        self.ctrl_socket.bind(f"tcp://*:{ctrl_port}")

        self.dt = dt
        self._start_infer = False
        self._toggle_time = 0.0

        # Motion buffer state (same as original rmdar.py)
        self._counter = 0
        self._gen_counter = -1
        self._block_index = 0
        self._current_block_size = 0
        self._cached_msg = None
        self._cached_buffer_size = 0

        # Text input thread
        self._text_prompt = "stand"
        self._text_changed = True
        self._shutdown = threading.Event()
        self._kb_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._kb_thread.start()

    def _keyboard_loop(self):
        """Background thread for text prompt input."""
        while not self._shutdown.is_set():
            try:
                text = input("Enter text prompt: ").strip()
                if text:
                    self._text_prompt = text
                    self._text_changed = True
            except (EOFError, KeyboardInterrupt):
                self._shutdown.set()
                break

    def _pack_motion_block(self, motion_dict, block_index):
        """Serialize motion dict to MessagePack binary."""
        dof_pos = motion_dict['dof_pos'][0].cpu().numpy()          # [T, 23]
        dof_vel = motion_dict['dof_vel'][0].cpu().numpy()          # [T, 23]
        root_pos = motion_dict['root_trans_offset'][0].cpu().numpy()  # [T, 3]
        root_rot = motion_dict['root_rot'][0].cpu().numpy()        # [T, 4] xyzw
        root_rot_wxyz = root_rot[:, [3, 0, 1, 2]]                  # xyzw → wxyz

        # Expand 23 DoF → 29 DoF (pad locked wrist joints)
        dof_pos_29 = np.pad(dof_pos, ((0,0),(0,6)), mode='constant')  # simplified
        dof_vel_29 = np.pad(dof_vel, ((0,0),(0,6)), mode='constant')

        msg = {
            'index': block_index,
            'timestamp': time.time(),
            'fps': 30,
            'T': dof_pos.shape[0],
            'num_joints': 29,
            'joint_positions': dof_pos_29.ravel().tolist(),
            'joint_velocities': dof_vel_29.ravel().tolist(),
            'anchor_body_ori': root_rot_wxyz.ravel().tolist(),
            'anchor_body_pos': root_pos.ravel().tolist(),
        }
        return msgpack.packb(msg)

    def _check_toggle(self):
        """Non-blocking check for toggle command from Tracker."""
        try:
            msg = self.ctrl_socket.recv(flags=zmq.NOBLOCK)
            command = msgpack.unpackb(msg)
            if command.get('action') == 'toggle':
                self._start_infer = not self._start_infer
                self._toggle_time = time.time()
                self._counter = 0
                self._gen_counter = -1
                if self._start_infer:
                    self._reset_motion_buffer()
                    print("[ZMQ] Inference started")
                else:
                    print("[ZMQ] Inference stopped")
            self.ctrl_socket.send(msgpack.packb({'status': 'ok'}))
        except zmq.Again:
            pass  # No message waiting

    def run(self):
        """Main loop — call at ~50Hz."""
        print("[ZMQ] Motion Planner running. Waiting for toggle...")

        while not self._shutdown.is_set():
            loop_start = time.time()

            self._check_toggle()

            if self._start_infer:
                # Update text embedding if changed
                if self._text_changed:
                    self._update_text_embedding()
                    self._text_changed = False

                # Trigger generation when buffer is low
                if (self._gen_counter - self._current_block_size
                        - self._counter <= self.future_len):
                    self._publish_block()
                    self._gen_motion()

                # Time-based counter (absolute, not frame-based)
                elapsed = time.time() - self._toggle_time
                self._counter = int(np.floor(elapsed / self.dt))

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, self.dt - elapsed)
            time.sleep(sleep_time)

    def _gen_motion(self):
        """Generate next block (same logic as rmdar.py:_gen_motion)."""
        future_motion, motion_dict, abs_pose = self.dar_gen_fn(
            text_embedding=self._text_embedding,
            history_motion=self.history_motion,
            abs_pose=self.history_abs_pose
        )
        self.history_motion = future_motion[:, -self.history_len:, :]
        self.history_abs_pose = abs_pose
        for k in list(self._ref_motion_dict.keys()):
            if isinstance(self._ref_motion_dict[k], torch.Tensor):
                self._ref_motion_dict[k] = motion_dict[k][:, -self.future_len:]

        self._current_block_size = self.future_len
        self._block_index = self._gen_counter
        self._gen_counter += self._current_block_size

        # Serialize immediately after generation
        self._cached_msg = self._pack_motion_block(motion_dict, self._block_index)
        self._cached_buffer_size = self.future_len

    def _publish_block(self):
        """Publish cached MotionBlock via ZMQ PUB."""
        if self._cached_msg is not None:
            self.pub_socket.send(self._cached_msg)

    def _update_text_embedding(self):
        with torch.no_grad():
            emb = encode_text(self.clip_model, [self._text_prompt])
            self._text_embedding = emb.float()

    def _reset_motion_buffer(self):
        """Initialize buffer with zero (stand) pose."""
        init_len = self.gen_len + self.future_len  # 18 frames
        motion_feat = get_zero_feature().to(self._device).reshape(1, 1, -1)
        motion_feat = motion_feat.repeat(1, init_len, 1)
        self.history_motion = self.dataset.normalize(motion_feat)[:, -self.history_len:, :]
        self.history_abs_pose = get_zero_abs_pose((1,), device=self._device)
        self._ref_motion_dict = self.dataset.reconstruct_motion(
            motion_feat.to(self._device), need_denormalize=False, ret_fk=True
        )
        self._block_index = 0
        self._gen_counter = init_len
        self._current_block_size = init_len
        self._cached_msg = self._pack_motion_block(self._ref_motion_dict, 0)
        self._cached_buffer_size = init_len


if __name__ == "__main__":
    planner = ZMQMotionPlanner(pub_port=5555, ctrl_port=5556)
    planner.run()
```

### 8.5 C++ Tracker Side (ZMQ Subscriber)

```cpp
// zmq_tracker.hpp — Minimal ZMQ-based motion subscriber
#include <zmq.hpp>
#include <msgpack.hpp>
#include <vector>
#include <string>

struct MotionBlock {
    int index;
    double timestamp;
    int fps;
    int T;
    int num_joints;
    std::vector<float> joint_positions;   // [T * num_joints]
    std::vector<float> joint_velocities;  // [T * num_joints]
    std::vector<float> anchor_body_ori;   // [T * 4]
    std::vector<float> anchor_body_pos;   // [T * 3]
};

class ZMQMotionReceiver {
public:
    ZMQMotionReceiver(const std::string& pub_addr = "tcp://localhost:5555",
                      const std::string& ctrl_addr = "tcp://localhost:5556")
        : ctx_(1),
          sub_socket_(ctx_, zmq::socket_type::sub),
          req_socket_(ctx_, zmq::socket_type::req)
    {
        // SUB socket: receive motion blocks
        sub_socket_.connect(pub_addr);
        sub_socket_.set(zmq::sockopt::subscribe, "");  // Subscribe to all

        // REQ socket: send toggle commands
        req_socket_.connect(ctrl_addr);
    }

    // Non-blocking receive: returns true if a new block arrived
    bool try_recv(MotionBlock& block) {
        zmq::message_t msg;
        auto ret = sub_socket_.recv(msg, zmq::recv_flags::dontwait);
        if (!ret) return false;

        // Deserialize from MessagePack
        auto obj = msgpack::unpack(static_cast<const char*>(msg.data()), msg.size());
        auto map = obj->as<std::map<std::string, msgpack::object>>();

        block.index       = map["index"].as<int>();
        block.timestamp   = map["timestamp"].as<double>();
        block.fps         = map["fps"].as<int>();
        block.T           = map["T"].as<int>();
        block.num_joints  = map["num_joints"].as<int>();
        block.joint_positions  = map["joint_positions"].as<std::vector<float>>();
        block.joint_velocities = map["joint_velocities"].as<std::vector<float>>();
        block.anchor_body_ori  = map["anchor_body_ori"].as<std::vector<float>>();
        block.anchor_body_pos  = map["anchor_body_pos"].as<std::vector<float>>();
        return true;
    }

    // Send toggle (start/stop) to Planner
    void send_toggle() {
        std::map<std::string, std::string> cmd{{"action", "toggle"}};
        msgpack::sbuffer sbuf;
        msgpack::pack(sbuf, cmd);
        zmq::message_t msg(sbuf.data(), sbuf.size());
        req_socket_.send(msg, zmq::send_flags::none);

        // Wait for ACK
        zmq::message_t reply;
        req_socket_.recv(reply);
    }

private:
    zmq::context_t ctx_;
    zmq::socket_t sub_socket_;
    zmq::socket_t req_socket_;
};
```

### 8.6 ZeroMQ vs ROS2 Comparison

| Aspect | ROS2 | ZeroMQ |
|---|---|---|
| **Dependency** | Full ROS2 stack (~2 GB) | libzmq + msgpack (~5 MB) |
| **Broker** | roscore / DDS discovery | None (direct connect) |
| **Message format** | `.msg` IDL → C++/Python codegen | MessagePack (schema-less, self-describing) |
| **Transport** | DDS (UDP multicast) / TCP | TCP (PUB/SUB, REQ/REP) |
| **Latency (localhost)** | ~100-500 μs | ~10-50 μs |
| **Serialization** | CDR (binary, schema-based) | MessagePack (binary, schema-less) |
| **QoS** | Reliable/volatile, durability, deadline, etc. | Best-effort (PUB/SUB) or reliable (REQ/REP) |
| **Discovery** | Automatic (DDS) | Manual (hardcoded addresses) |
| **Multi-machine** | Built-in (DDS routing) | Manual TCP addresses |
| **Monitoring** | `ros2 topic echo`, `rqt`, etc. | Custom tooling needed |
| **Suitable for** | Full robot systems | Lightweight embedded/edge deployment |

---

## 9. Model Summary Card

| Property | Value |
|---|---|
| **MVAE** | SkipTransformer, 9 layers, 4 heads, d=512 |
| **MVAE latent** | 1 token × 128 dim |
| **Denoiser** | TransformerEncoder, 8 layers, 4 heads, d=512 |
| **Diffusion steps** | 5 (cosine schedule) |
| **Prediction target** | START_X (direct x₀ prediction) |
| **Text encoder** | CLIP ViT-B/32, frozen, 512-dim output |
| **Guidance** | Classifier-free, scale=5.0 |
| **Motion features** | 57-dim (FeatureVersion=3) |
| **History / Future** | 2 frames history / 8 frames future |
| **FPS** | 30 |
| **Robot DoF** | 23 (G1, wrists locked) |
| **qpos dim** | 30 (3 trans + 4 rot + 23 dof) |
| **Control rate** | 50 Hz (Planner loop) |
| **Inference device** | CUDA GPU |
| **Acceleration** | torch.compile(TensorRT) or TensorRT compile |

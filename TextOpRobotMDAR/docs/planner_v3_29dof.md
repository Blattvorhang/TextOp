# Planner V3 — Migrate from 23-DoF to 29-DoF

## Motivation

The current motion generator operates on **23-DoF** (wrists locked), discarding
6 wrist joints (3 per hand: roll, pitch, yaw) from the full G1 skeleton.
However:

- The **Bones-SEED dataset** provides **29-DoF** joint-angle columns — the
  wrist data already exists in the source CSV files.
- Using only 23-DoF **wastes training data** (6 of 29 joint-angle columns are
  discarded).
- With wrists locked to zero, the FK-derived **hand keypoints** are fixed rigid
  extensions from the elbow. Real motions (especially manipulation and
  expressive gestures) carry meaningful wrist articulation that the model
  cannot learn or reproduce.
- The planner already pads zero-valued wrist DOFs at the IsaacLab boundary
  (23 → 29), which is a lossy workaround. A native 29-DoF pipeline eliminates
  this hack.

**Goal:** Upgrade the entire pipeline — MJCF skeleton, FK, feature
representation, data loading, model I/O, and boundary conversions — from
23-DoF to the full 29-DoF G1 skeleton.

## Agreed Decisions and Implementation Order

- Bones-SEED's native 29-DoF joint data has been visually checked and is the
  source of truth. The preprocessing pipeline must never crop wrist columns.
- Migration is staged. **Stage 1 (implemented)** prepares native 29-DoF data.
  The **MVAE portion of Stage 2 is also implemented and smoke-tested**:
  FeatureVersion 3, the training skeleton, MVAE configuration, geometric loss,
  and controller joint-order boundary are native 29-DoF. Denoiser/LDM and the
  complete planner evaluation path still require their dedicated Stage 2
  validation after the new MVAE checkpoint is trained.
- FeatureVersion 3 cannot remain 57-dimensional while representing 29 joint
  angles and 29 joint deltas. Its native dimension is **69**:
  `11 + 29 + 29`. Keeping 57 would require dropping information or defining a
  different compressed feature representation; neither is part of this plan.
- Current training windows are `history_len=16`, `future_len=64`, and
  `num_primitive=4`. These are training configuration values, not dataset
  preparation parameters. The packed dataset does not filter by length by
  default, so it can support different window experiments. At training time,
  the loader requires `history_len + future_len * num_primitive + 1 +
  goal_offset` raw frames; the current zero-offset configuration requires 273.
- For `goal_type=body`, retain five positional keypoints: pelvis, left/right
  foot, and left/right **palm center**. Palm center is preferable to wrist
  position because it is the task-relevant limb endpoint and responds to all
  three wrist rotations. A wrist-joint origin does not move under its own yaw
  and therefore discards useful 29-DoF information.
- The hand keypoint is attached to `*_wrist_yaw_link` at the rubber-hand mesh
  origin: left `[0.0415, 0.003, 0]`, right `[0.0415, -0.003, 0]`. This is a
  real geometry point, responds to all wrist rotations, and is the convention
  implemented by the training FK and planner.

---

## Canonical 29-DoF Index Order

The model, packed data, training FK, MuJoCo qpos (`qpos[7:36]`), and MuJoCo
actuators all use the same depth-first MJCF order:

| Indices | Joints |
|---|---|
| 0-5 | left hip pitch/roll/yaw, knee, ankle pitch/roll |
| 6-11 | right hip pitch/roll/yaw, knee, ankle pitch/roll |
| 12-14 | waist yaw/roll/pitch |
| 15-18 | left shoulder pitch/roll/yaw, elbow |
| 19-21 | left wrist roll/pitch/yaw |
| 22-25 | right shoulder pitch/roll/yaw, elbow |
| 26-28 | right wrist roll/pitch/yaw |

The migration is an insertion, not an append. The old 23-DoF vector maps to
29-DoF indices `[0:19] + [22:26]`; old right-arm indices `19:23` become new
indices `22:26`. Left wrists are inserted at `19:22`, and right wrists occupy
`26:29`.

FeatureVersion 3 preserves this order without permutation:

- `feature[..., 11:40]`: current 29 joint angles
- `feature[..., 40:69]`: forward differences of the same 29 joints

IsaacLab order exists only at the controller boundary. Its maps are derived
from semantic joint-name lists. New preprocessing output records
`dof_order: mujoco` and ordered `dof_names`; MVAE and DAR startup validate
these fields when present. Older packed 29-DoF datasets without this metadata
remain loadable, while their MJCF/config order is still checked at startup.

---

## Pre-Migration Architecture (23-DoF)

```
Bones-SEED CSV (29 columns)
        │ convert_soma_csv_to_motion_lib.py retains 29
        ▼ pack_motion_lib_to_textop.py drops 6 wrist cols → 23-DoF
        │
23-DoF dataset (.npy / .npz)
        │
        ▼ DOF_DIM = 23
        │
Motion feature (57-dim for V3: 4+1+2+3+1+23+23)
        │
        ▼ VAE encode / denoiser / VAE decode
        │
23-DoF motion_dict
        │
        ▼ FK (23 joints → 23+4 bodies)
        │
        ▼ _expand_23_to_29() → pad zero wrists
        │
IsaacLab 29-DoF G1MotionData → SONIC controller
```

### What "23" means concretely

| Group | Joints | Count |
|-------|--------|-------|
| Left leg | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll | 6 |
| Right leg | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll | 6 |
| Waist | yaw, roll, pitch (= torso_link) | 3 |
| Left arm | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow | 4 |
| Right arm | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow | 4 |
| **Wrists (dropped)** | left_wrist_roll/pitch/yaw, right_wrist_roll/pitch/yaw | **0 (6 dropped)** |
| **Total** | | **23** |

---

## Target Architecture (29-DoF)

```
Bones-SEED CSV (29 columns)
        │
        ▼ keep ALL 29 columns
        │
29-DoF dataset (.npy / .npz)
        │
        ▼ DOF_DIM = 29
        │
Motion feature (69-dim for V3: 4+1+2+3+1+29+29)
        │
        ▼ VAE encode / denoiser / VAE decode
        │
29-DoF motion_dict
        │
        ▼ FK (29 joints → 29+4 bodies, wrist chain active)
        │
        ▼ direct IsaacLab reorder (no padding)
        │
IsaacLab 29-DoF G1MotionData → SONIC controller
```

### Wrist joint details

| Joint | Axis | Range (rad) | Actuator force range |
|-------|------|------------|---------------------|
| left_wrist_roll | X (1,0,0) | [-1.972, 1.972] | [-25, 25] |
| left_wrist_pitch | Y (0,1,0) | [-1.614, 1.614] | [-5, 5] |
| left_wrist_yaw | Z (0,0,1) | [-1.614, 1.614] | [-5, 5] |
| right_wrist_roll | X (1,0,0) | [-1.972, 1.972] | [-25, 25] |
| right_wrist_pitch | Y (0,1,0) | [-1.614, 1.614] | [-5, 5] |
| right_wrist_yaw | Z (0,0,1) | [-1.614, 1.614] | [-5, 5] |

---

## Detailed Change Plan

### Phase 1 — MJCF XML: Switch from 23-DoF to 29-DoF Skeleton

**File:** `description/robots/g1/g1_23dof_lock_wrist_fitmotionONLY.xml` → **replace with 29-DoF variant**

The current skeleton XML has wrist joints removed entirely (not just commented
out). The 29-DoF file `g1_29dof_rev_1_0.xml` already exists and contains the
full kinematic chain including wrist bodies and joints.

**Option A:** Use the existing `g1_29dof_rev_1_0.xml` as-is for an initial FK
parity check only.
This file has:
- 29 joint definitions (including 6 wrist joints as `<joint>` elements)
- 29 motors in the `<actuator>` section
- Full body hierarchy: `elbow → wrist_roll → wrist_pitch → wrist_yaw`

**Option B (recommended):** Create a new `g1_29dof_fitmotion.xml` by copying
`g1_29dof_rev_1_0.xml` and stripping collision geoms (matching the
`fitmotionONLY` pattern of the 23-DoF variant).

**Config change** ([robotmdar/config/skeleton/g1.yaml](robotmdar/config/skeleton/g1.yaml)):
```yaml
# Before:
assetFileName: "g1_23dof_lock_wrist_fitmotionONLY.xml"
humanoid_type: g1_23dof_lock_wrist

# After:
assetFileName: "g1_29dof_rev_1_0.xml"          # or g1_29dof_fitmotion.xml
humanoid_type: g1_29dof
```

**Impact:** Once the XML changes, `ForwardKinematics._parse_mjcf()` will
automatically read 29 joint axes and 29 motor names. `self.num_dof` becomes
29. `self.dof_axis` becomes shape `(29, 3)`. No FK code changes are needed
for the core FK computation — it operates generically over whatever bodies
and joints the XML provides.

---

### Phase 2 — DOF_DIM Constant: The Central Change

**File:** [robotmdar/dtype/motion.py](robotmdar/dtype/motion.py), line 16

```python
# Before:
DOF_DIM = 23  # 29 - 2 hand * 3 wrist

# After:
DOF_DIM = 29  # full G1 skeleton including wrists
```

This single change cascades through every feature-dimension computation in
the file because all `motion_feature_dim_v*` constants are defined in terms
of `DOF_DIM`:

| Constant | Formula | 23-DoF value | 29-DoF value |
|----------|---------|-------------|-------------|
| `motion_feature_dim_v1` | `4+1+2+3+1+DOF_DIM` | 34 | 40 |
| `motion_feature_dim_v2` | `4+1+2+3+1+DOF_DIM+DOF_DIM` | 57 | 69 |
| `motion_feature_dim_v3` | same as v2 | 57 | 69 |
| `motion_feature_dim_v4` | `3+6+DOF_DIM+3+6+(DOF_DIM+4)*3+(DOF_DIM+4)*3+2` | 181 | 247 |
| `motion_feature_dim_v5` | `4+1+2+3+3+(DOF_DIM+4)*3+(DOF_DIM+4)*3+1+DOF_DIM+DOF_DIM` | 222 | 270 |
| `n_qpos` | `3+4+DOF_DIM` | 30 | 36 |

---

### Phase 3 — Hardcoded Slice Indices in motion.py

Several feature encode/decode functions use **hardcoded slice indices** that
embed the value 23. These must be updated to reflect 29.

#### 3a. `motion_feature_to_dict_v2` (lines 351-352)

```python
# Before:
dof = motion_feature[..., 11:34]       # (B, T, 23)
delta_dof = motion_feature[..., 34:]   # (B, T, 23)

# After:
dof = motion_feature[..., 11:40]       # (B, T, 29)
delta_dof = motion_feature[..., 40:]   # (B, T, 29)
```

#### 3b. `motion_feature_to_dict_v3` (lines 512-513)

```python
# Before:
dof = motion_feature[..., 11:34]       # (B, T, 23)
delta_dof = motion_feature[..., 34:]   # (B,T, 23)

# After:
dof = motion_feature[..., 11:40]       # (B, T, 29)
delta_dof = motion_feature[..., 40:]   # (B,T, 29)
```

#### 3c. `motion_feature_to_dict_v4` (line 733)

```python
# Before:
dof = motion_feature[..., 9:32]        # [B, T, 23]
transl_delta = motion_feature[..., 32:35]
rot_delta_6d = motion_feature[..., 35:41]
joints_feature = motion_feature[..., 41:41+(23+4)*3]  # 41:122
joints_delta = motion_feature[..., 122:-2]

# After:
dof = motion_feature[..., 9:38]        # [B, T, 29]
transl_delta = motion_feature[..., 38:41]
rot_delta_6d = motion_feature[..., 41:47]
joints_feature = motion_feature[..., 47:47+(29+4)*3]  # 47:146
joints_delta = motion_feature[..., 146:-2]
```

#### 3d. `motion_feature_to_dict_v5` (lines 1107-1108)

```python
# Before:
dof = motion_feature[..., -46:-23]     # (B, T, 23)
delta_dof = motion_feature[..., -23:]  # (B,T, 23)

# After:
dof = motion_feature[..., -58:-29]     # (B, T, 29)
delta_dof = motion_feature[..., -29:]  # (B,T, 29)
```

#### 3e. `get_zero_feature_v2` (lines 1214-1221)

The 23-element default joint-angle tensor must be expanded to 29 elements:

```python
# Before (23 values):
feat[0, 11:34] = torch.tensor([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
    0.0, 0.0, 0.0,                      # waist
    0.2, 0.2, 0.0, 0.9,                # left arm
    0.2, -0.2, 0.0, 0.9                # right arm
])

# After (29 values — add 6 wrist zeros):
feat[0, 11:40] = torch.tensor([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # left leg
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # right leg
    0.0, 0.0, 0.0,                      # waist
    0.2, 0.2, 0.0, 0.9,                # left arm
    0.0, 0.0, 0.0,                      # left wrist (new)
    0.2, -0.2, 0.0, 0.9,               # right arm
    0.0, 0.0, 0.0                       # right wrist (new)
])
```

Same change applies to `get_zero_feature_v1` (line 1198) and
`get_zero_feature_v4` (line 1230).

#### 3f. `planner_convert.py`: `state_to_model_input` (line 138)

The feature shape assertion uses FeatureVersion 3's dim = 57:

```python
# Before:
if feature.shape != (1, history_len, 57):
    raise ValueError(...)

# After:
if feature.shape != (1, history_len, 69):
    raise ValueError(...)
```

And the delta_dof copy at line 161 slices `[34:57]`:

```python
# Before:
feature[:, -1, 34:57] = (...dof delta...)

# After:
feature[:, -1, 40:69] = (...dof delta...)
```

---

### Phase 4 — Skeleton Config: Add Wrist Bodies and DOFs

**File:** [robotmdar/config/skeleton/g1.yaml](robotmdar/config/skeleton/g1.yaml)

#### 4a. `body_names` — insert 6 wrist bodies in the arm chains

```yaml
# Before (23 bodies):
body_names: ['pelvis',
    'left_hip_pitch_link', ..., 'left_elbow_link',
    'right_shoulder_pitch_link', ..., 'right_elbow_link']

# After (29 bodies):
body_names: ['pelvis',
    'left_hip_pitch_link', ..., 'left_elbow_link',
    'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
    'right_shoulder_pitch_link', ..., 'right_elbow_link',
    'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']
```

**Note:** The body ordering must match the MuJoCo XML body tree traversal
order (depth-first). The 29-DoF XML defines wrists as direct children of
elbows, so they appear immediately after elbow in the body list.

#### 4b. `dof_names` — add 6 wrist DOF names

```yaml
# Before (23 DOF names):
dof_names: ['left_hip_pitch_link', ..., 'left_elbow_link',
            'right_shoulder_pitch_link', ..., 'right_elbow_link']

# After (29 DOF names):
dof_names: ['left_hip_pitch_link', ..., 'left_elbow_link',
            'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link',
            'right_shoulder_pitch_link', ..., 'right_elbow_link',
            'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']
```

#### 4c. Update limb/body groupings

Add wrist links to `upper_body_link`:
```yaml
upper_body_link:
  - "left_shoulder_pitch_link"
  - "left_shoulder_roll_link"
  - "left_shoulder_yaw_link"
  - "left_elbow_link"
  - "left_wrist_roll_link"       # new
  - "left_wrist_pitch_link"      # new
  - "left_wrist_yaw_link"        # new
  - "right_shoulder_pitch_link"
  - "right_shoulder_roll_link"
  - "right_shoulder_yaw_link"
  - "right_elbow_link"
  - "right_wrist_roll_link"      # new
  - "right_wrist_pitch_link"     # new
  - "right_wrist_yaw_link"       # new
  - "left_hand_link"
  - "right_hand_link"
  - "head_link"
```

Update `limb_weight_group` arm groups similarly.

#### 4d. Update `extend_config` — hand parent from elbow → wrist_yaw

The synthetic hand tracking points should now attach at the end of the wrist
chain, not directly to the elbow:

```yaml
# Before:
extend_config:
  - joint_name: "left_hand_link"
    parent_name: "left_elbow_link"       # was correct for locked-wrist
    pos: [0.25, 0.0, 0.0]
    rot: [1.0, 0.0, 0.0, 0.0]
  - joint_name: "right_hand_link"
    parent_name: "right_elbow_link"
    pos: [0.25, 0.0, 0.0]
    rot: [1.0, 0.0, 0.0, 0.0]

# After:
extend_config:
  - joint_name: "left_hand_link"
    parent_name: "left_wrist_yaw_link"   # wrist chain now exists
    pos: [0.0415, 0.003, 0.0]           # left rubber-hand mesh origin
    rot: [1.0, 0.0, 0.0, 0.0]
  - joint_name: "right_hand_link"
    parent_name: "right_wrist_yaw_link"
    pos: [0.0415, -0.003, 0.0]          # right rubber-hand mesh origin
    rot: [1.0, 0.0, 0.0, 0.0]
```

**Note:** Do not tune this offset by eye without defining the target. The
29-DoF XML places the rubber-hand mesh origin at X=0.0415 m. The implemented
goal keypoint uses that origin plus the XML's mirrored lateral offsets. The
renderer, training FK, and controller reference must use the same definition.

#### 4e. Visualization marker colors — add 6 wrist joint entries

The `marker_joint_colors` list in `g1.yaml` currently has entries for 23
joints + 3 extend bodies. Add 6 entries for wrist joints.

---

### Phase 5 — Remove or Simplify 23↔29 Conversion Functions

**File:** [robotmdar/utils/planner_convert.py](robotmdar/utils/planner_convert.py)

#### 5a. Remove `_reduce_mujoco_29_to_23()` (lines 41-47)

This function is no longer needed — the pipeline now operates on 29-DoF
natively. All call sites must be updated:

| Call site | Action |
|-----------|--------|
| `state_to_model_input()` line 122 | Remove `_reduce_mujoco_29_to_23()` call; use 29-DoF `joints_mujoco` directly |

#### 5b. Remove `_expand_mujoco_23_to_29()` (lines 50-58)

No longer needed. Call sites:

| Call site | Action |
|-----------|--------|
| `mujoco_to_isaaclab_dof()` lines 64-65 | Remove the 23→29 expansion branch; require 29-DoF input |

#### 5c. Simplify `mujoco_to_isaaclab_dof()` (lines 61-68)

```python
# Before:
def mujoco_to_isaaclab_dof(values: np.ndarray) -> np.ndarray:
    """Convert 23- or 29-DoF MuJoCo values to IsaacLab 29-DoF order."""
    values = np.asarray(values)
    if values.shape[-1] == 23:
        values = _expand_mujoco_23_to_29(values)
    elif values.shape[-1] != 29:
        raise ValueError(f"Expected 23 or 29 MuJoCo DoFs, got {values.shape}")
    return np.ascontiguousarray(values[..., _MUJOCO_TO_ISAACLAB])

# After:
def mujoco_to_isaaclab_dof(values: np.ndarray) -> np.ndarray:
    """Convert 29-DoF MuJoCo values to IsaacLab 29-DoF order."""
    values = np.asarray(values)
    if values.shape[-1] != 29:
        raise ValueError(f"Expected 29 MuJoCo DoFs, got {values.shape}")
    return np.ascontiguousarray(values[..., _MUJOCO_TO_ISAACLAB])
```

#### 5d. Update `motion_dict_to_g1data()` wrist handling (lines 547-563)

The locked-wrist fallback logic should be removed:

```python
# Before (lines 553-563):
if locked_joint_pos is not None:
    locked_joint_pos = np.asarray(locked_joint_pos, dtype=np.float32)
    ...
    locked_isaaclab = _ISAACLAB_TO_MUJOCO[[19, 20, 21, 26, 27, 28]]
    joint_pos[:, locked_isaaclab] = locked_joint_pos[locked_isaaclab]
    joint_vel[:, locked_isaaclab] = 0.0

# After:
# Wrist joints are now generated by the model; no fallback needed.
# The locked_joint_pos parameter can be deprecated or removed.
```

#### 5e. `_ISAACLAB_TO_MUJOCO` / `_MUJOCO_TO_ISAACLAB` arrays

These 29-element index arrays remain correct — they describe the **ordering**
of the 29 joints, not the count. No change needed.

---

### Phase 6 — Evaluation Loop: Remove 23→29 Padding

**File:** [robotmdar/eval/loop_dar.py](robotmdar/eval/loop_dar.py)

#### 6a. Remove `_npz_expand_23_to_29()` (lines 96-102)

The function pads zero-valued wrist DOFs into 23-DoF output. With 29-DoF
output, this is unnecessary. Remove the function entirely.

#### 6b. Update NPZ accumulation (lines 119-127)

```python
# Before:
dof_pos_all  = np.concatenate(all_dof_pos, axis=0)    # [T, 23]
dof_vel_all  = np.concatenate(all_dof_vel, axis=0)    # [T, 23]
dof_pos_29 = _npz_expand_23_to_29(dof_pos_all)
dof_vel_29 = _npz_expand_23_to_29(dof_vel_all)
dof_pos_isaaclab = dof_pos_29[:, _NPZ_MJC2ISAAC]
dof_vel_isaaclab = dof_vel_29[:, _NPZ_MJC2ISAAC]

# After:
dof_pos_all  = np.concatenate(all_dof_pos, axis=0)    # [T, 29]
dof_vel_all  = np.concatenate(all_dof_vel, axis=0)    # [T, 29]
dof_pos_isaaclab = dof_pos_all[:, _NPZ_MJC2ISAAC]
dof_vel_isaaclab = dof_vel_all[:, _NPZ_MJC2ISAAC]
```

#### 6c. Update FK accumulation comments (lines 429-430)

```python
# Before:
dof_pos = motion_dict['dof_pos'][0, skip:].detach().cpu().numpy()  # [T', 23]
dof_vel = motion_dict['dof_vel'][0, skip:].detach().cpu().numpy()  # [T', 23]

# After:
dof_pos = motion_dict['dof_pos'][0, skip:].detach().cpu().numpy()  # [T', 29]
dof_vel = motion_dict['dof_vel'][0, skip:].detach().cpu().numpy()  # [T', 29]
```

#### 6d. `_NPZ_TRACKER_BODIES` — already includes wrist bodies

The tracker body list (lines 78-93) already references
`left_wrist_yaw_link` and `right_wrist_yaw_link`. These will now resolve to
actual FK bodies (instead of being missing) once the 29-DoF skeleton is
loaded. No change needed here.

#### 6e. qpos visualization

`motion_dict_to_qpos()` uses `n_qpos = 3 + 4 + DOF_DIM`. This automatically
becomes 36 instead of 30. `show_fn` is driven by the loaded MuJoCo model, but
`mjc_load_everything()` currently defaults to the 23-DoF XML and its callers do
not pass the skeleton path. Stage 2 must make the XML path explicit; otherwise
assigning a 36-element qpos to the 23-DoF viewer will fail.

---

### Phase 7 — Data Pipeline: Stop Dropping Wrist DOFs

**Stage 1 implementation status (complete):**

- `dataset/data_process/convert_soma_csv_to_motion_lib.py` now validates and
  selects all 29 joint columns by semantic column name. It already emitted
  29-DoF motion-lib entries and continues to do so.
- `dataset/data_process/pack_motion_lib_to_textop.py` no longer applies the
  `DOF_29_TO_23` mask. It writes `[T,29]` DOFs and records `dof_dim: 29` and
  `nfeats: 69` in `statistics.yaml`.
- The packer's optional `--min_frames` data-quality filter is disabled by
  default. It does not accept history/future length parameters. The loader
  filters clips dynamically from each training run's active configuration.
- `dataset/data_process/run_full_pipeline.sh` produces the
  `BONES-SEED-29dof-FULL-50fps` dataset link and passes the window settings.

Recommended clean regeneration command (use a new `OUTPUT_ROOT` so stale
`.done` markers cannot select 23-DoF packed data):

```bash
BONES_SEED_DIR=/path/to/bones-seed \
OUTPUT_ROOT=/path/to/textop-29dof-preprocessed \
NUM_WORKERS=16 \
FK_BACKEND=torch TORCH_DEVICE=cpu \
bash dataset/data_process/run_full_pipeline.sh
```

If no sequence in a split is long enough, dataset construction raises a
descriptive `ValueError` containing the required length, active history/future/
primitive settings, number of loaded clips, and longest available clip. Clips
exactly equal to the required length are valid, and sampling includes every
valid start position. Weighted sampling probabilities are computed only over
the dynamically valid subset.

The earlier version of this document incorrectly focused on
`scripts/create_goal_reference.py`; that script is not the training dataset
packer. It still needs migration in Stage 2 together with the 29-DoF skeleton.

#### 7a. Script: [scripts/create_goal_reference.py](scripts/create_goal_reference.py)

**`_CSV_DOF_IDXS`** (lines 45-51): Add the 6 wrist CSV column indices:

```python
# Before (23 indices):
_CSV_DOF_IDXS: tuple[int, ...] = (
    7, 8, 9, 10, 11, 12,      # left  leg
    13, 14, 15, 16, 17, 18,   # right leg
    19, 20, 21,               # waist
    22, 23, 24, 25,           # left  arm
    29, 30, 31, 32,           # right arm
)

# After (29 indices):
_CSV_DOF_IDXS: tuple[int, ...] = (
    7, 8, 9, 10, 11, 12,      # left  leg
    13, 14, 15, 16, 17, 18,   # right leg
    19, 20, 21,               # waist
    22, 23, 24, 25,           # left  arm
    26, 27, 28,               # left  wrist (NEW)
    29, 30, 31, 32,           # right arm
    33, 34, 35,               # right wrist (NEW)
)
```

Update docstrings, shape annotations, and `_extract_dof()` return shape
(from `(23,)` to `(29,)`).

**`_validate_pose()` knee angle check (line 363):** Uses DOF index 3 (left
knee) and index 9 (right knee). These are in the 23-DoF ordering. In the
29-DoF ordering, the left knee remains at index 3 and right knee at index 9
(because the wrist DOFs are inserted after the arm DOFs, not before the leg
DOFs). **No change needed.**

#### 7b. Dataset files

Existing NPZ/NPY datasets with "23dof" in their names (e.g.,
`BONES-SEED-23dof-FULL-50fps`) were created by dropping wrist columns from
the 29-column source CSVs. These must be **re-generated** to retain all 29
columns.

**New dataset naming convention:**
```
BONES-SEED-29dof-FULL-50fps
```

The relevant data generation scripts are in this repository. In particular,
`dataset/data_process/pack_motion_lib_to_textop.py` was the step that cropped
29 DoFs to 23 and has now been corrected.

#### 7c. Data loading ([robotmdar/dataloader/data.py](robotmdar/dataloader/data.py))

The data loader reads `raw_motion['dof']` directly from NPZ files. No code
change is needed **if** the NPZ files already contain 29-DoF data. The shape
is validated implicitly by `DOF_DIM`.

**Action:** Verify that `SkeletonPrimitiveDataset` does not hardcode any
23-related shape checks. Based on the code, it stores and returns motion
data generically by key.

#### 7d. Config files — dataset paths

**Files:**
- [robotmdar/config/data/mob.yaml](robotmdar/config/data/mob.yaml)
- [robotmdar/config/data/babel.yaml](robotmdar/config/data/babel.yaml)

Update `datadir` entries and the hardcoded feature count to point to 29-DoF
datasets in Stage 2:
```yaml
# Before:
datadir: dataset/BABEL-TeleMotion-ROBOT-23dof-Full-1ANN-1000-Merged-Interp

# After:
datadir: dataset/BABEL-TeleMotion-ROBOT-29dof-Full-1ANN-1000-Merged-Interp
nfeats: 69
```

`nfeats` is not currently auto-derived from `DOF_DIM`; both `mob.yaml` and
`babel.yaml` explicitly set it to 57 and therefore must be changed.

#### 7e. FK and MOB preprocessing performance

The converter supports `--fk_backend torch|mujoco` and defaults to batched
PyTorch FK on CPU. CPU is the correct setting with the session-level
multiprocessing pipeline; CUDA is restricted to one worker.

Parity tests compare both foot positions and all collision-geometry poses with
MuJoCo and pass at `atol=2e-6`. On the local CPU, warmed FK-only timing for
5,000 frames was 0.019 s with PyTorch versus 0.222 s with per-frame MuJoCo
(`11.6x`). However, exact MOB voxel rasterization dominates total time: for a
500-frame sequence at stride 2, warmed FK+MOB timing was 0.418 s versus
0.432 s (`1.03x`), with identical contact masks and occupancy grids. Therefore
PyTorch removes FK as a bottleneck but does not by itself solve MOB runtime.
The exact rasterizer is now vectorized by grouping primitive poses with the
same type and candidate AABB dimensions. It preserves the original AABB
rounding, voxel-center tests, primitive containment equations, and occupancy
union. `--mob_raster_backend scalar` retains the original loop as a reference;
`vectorized` is the default. Temporary candidate arrays are chunked to bound
memory on long clips.

On a real 406-frame Bones-SEED clip at stride 2, Torch CPU FK+MOB decreased
from 0.3301 s (scalar rasterizer) to 0.01775 s (vectorized), an **18.6x**
speedup with a bit-identical occupancy grid. Five additional motions covering
gesture, jogging, box descent, sitting, and jumping produced identical grids
with speedups from **13.2x to 44.7x**.

CUDA was also benchmarked outside the sandbox on an RTX 4070 Ti SUPER using
PyTorch CUDA 12.8. Warmed minimum timings were:

| Workload | Torch CUDA | Torch CPU | MuJoCo CPU |
|----------|-----------:|----------:|-----------:|
| FK only, 406 frames | 0.001890 s | **0.001794 s** | 0.017251 s |
| FK only, 5,000 frames | **0.002605 s** | 0.013781 s | 0.210755 s |
| Scalar exact FK+MOB, 406 frames, stride 2 | **0.330392 s** | 0.332224 s | 0.346561 s |

Thus CUDA is fastest for large batched FK, but not for one typical motion clip.
The production converter parallelizes many clips across CPU processes, whereas
CUDA mode requires one worker. Until the converter batches multiple clips into
each GPU call or the MOB rasterizer moves to GPU, `torch/cpu` with multiple
workers remains the recommended full-dataset configuration.

After exact rasterizer vectorization, the same real 406-frame FK+MOB workload
takes 0.01727 s on Torch CUDA, 0.01770 s on Torch CPU, and 0.03232 s with
MuJoCo FK. CUDA is only 2.5% faster per clip, so multi-worker Torch CPU remains
the best full-dataset configuration; CUDA becomes useful when multiple clips
are batched per call.

---

### Phase 8 — Shell Scripts: Update DATADIR References

**Files:**
- [scripts/run_planner_dar.sh](scripts/run_planner_dar.sh)
- [scripts/run_loop_dar.sh](scripts/run_loop_dar.sh)
- [scripts/run_smoke_vae_training.sh](scripts/run_smoke_vae_training.sh)
- [scripts/run_smoke_ldm_training.sh](scripts/run_smoke_ldm_training.sh)

```bash
# Before:
DATADIR=BONES-SEED-23dof-FULL-50fps

# After:
DATADIR=BONES-SEED-29dof-FULL-50fps
```

---

### Phase 9 — Tests: Update Hardcoded Dimensions

**File:** [test/test_planner_utils.py](test/test_planner_utils.py)

| Line | Change |
|------|--------|
| 49 | `torch.zeros((1, 2, 23))` → `torch.zeros((1, 2, 29))` |
| 55 | `assert fk["dof_pos"].shape == (1, 2, 23)` → `(1, 2, 29)` |
| 181 | `torch.zeros((1, 3, 23))` → `torch.zeros((1, 3, 29))` |
| 253 | `torch.zeros((1, frames, 23))` → `torch.zeros((1, frames, 29))` |
| 274 | Update or remove locked-wrist indices test |

**File:** [test/test_textop_body_protocol.py](test/test_textop_body_protocol.py)
- Line 27: `np.zeros((3, 29))` — already 29. No change.

**File:** [test/test_data_conversion_contact.py](test/test_data_conversion_contact.py)
- Line 145: `np.zeros((2, 29))` — already 29. No change.

---

### Phase 10 — Documentation

**Files to update:**

| File | Change |
|------|--------|
| [docs/planner_design.md](docs/planner_design.md) | Update all 23→29 references: DoF count, qpos dim (30→36), feature version tables, FK notes |
| [docs/planner_v1_basic_pipeline.md](docs/planner_v1_basic_pipeline.md) | Update feature dimension (57→69), DOF_DIM (23→29), slice indices |
| [docs/planner_v2_goal_scene.md](docs/planner_v2_goal_scene.md) | Update body count (24→30), remove "wrist lock" section, update 23→29 conversion docs |
| [docs/planner_vx_field_regulation.md](docs/planner_vx_field_regulation.md) | Update feature component table |

---

### Phase 11 — Model Retraining (Breaking Change)

**Implementation status:** MVAE training is native 29-DoF. On 2026-08-02, a
12-step CUDA smoke run completed with `history_len=16`, `future_len=64`,
`num_primitive=4`, 69-D features, and all FK/geometric loss terms enabled.
The configured `batch_size=64` also completed a four-step train/eval check on
an RTX 4070 Ti SUPER with 16 GB VRAM.
The generated dataset provides 66,466 valid training clips and 3,581 valid
validation clips for the required 273-frame window. The smoke run is a shape
and execution check, not evidence of reconstruction quality.

**All existing model checkpoints are incompatible** with the 29-DoF feature
space and must be retrained from scratch:

| Component | 23-DoF input/output dim | 29-DoF input/output dim |
|-----------|------------------------|------------------------|
| VAE encoder input | history: 57 (V3) | history: 69 (V3) |
| VAE decoder output | future: 57 (V3) | future: 69 (V3) |
| Denoiser I/O | latent dim (unchanged) | latent dim (unchanged) |
| Goal conditioning | depends on goal_type | unchanged |
| Scene conditioning | depends on grid_size | unchanged |

**Training config changes:**
- `data.nfeats`: 57 → 69 in both data configs (currently hardcoded)
- `history_len=16`, `future_len=64`, `num_primitive=4` consistently across
  training, planner, and smoke tests; these do not belong in packed dataset
  statistics because clip validity is determined dynamically
- Model weight shapes for the first/last linear layers change
- The latent space dimension is independent of DOF_DIM (it's a compressed
  representation), but the denoiser history embedding consumes `nfeats` and
  its checkpoint is therefore also shape-incompatible

Start a new MVAE run from `TextOpRobotMDAR/` with:

```bash
robotmdar --config-name=train_mvae expname=BONES-SEED-VAE-29DOF \
  data.batch_size=64 ckpt.vae=null
```

Adjust `data.batch_size` for the remote GPU. Do not load a 23-DoF checkpoint.
The first run computes and caches `dataset/BONES-SEED-29dof-FULL-50fps/meanstd.pkl`;
that cache must contain 69-element mean and standard-deviation tensors.

**Smoke-test checklist after retraining:**
1. Run `test/test_planner_utils.py` — FK preserves root quaternion
2. Run `scripts/run_smoke_vae_training.sh` — VAE trains without shape errors
3. Run `scripts/run_smoke_ldm_training.sh` — LDM trains without shape errors
4. Run `scripts/run_loop_dar.sh` — interactive loop generates valid motion
5. Run `scripts/run_planner_dar.sh` — headless planner runs end-to-end

---

## Summary of All Files Touched

```
Modified:
  description/robots/g1/g1_29dof_fitmotion.xml        (NEW — 29dof MJCF variant)
  robotmdar/config/skeleton/g1.yaml                    (XML path, body_names,
                                                        dof_names, extend_config,
                                                        limb groups, marker colors)
  robotmdar/dtype/motion.py                            (DOF_DIM, slice indices,
                                                        zero features, n_qpos)
  robotmdar/utils/planner_convert.py                   (remove 23↔29 converters,
                                                        simplify mujoco_to_isaaclab,
                                                        update state_to_model_input,
                                                        remove locked-wrist fallback)
  robotmdar/eval/loop_dar.py                           (remove _npz_expand_23_to_29,
                                                        update NPZ accumulation)
  robotmdar/skeleton/robot.py                          (minor: num_extend_dof
                                                        will increase)
  robotmdar/config/data/mob.yaml                       (dataset paths)
  robotmdar/config/data/babel.yaml                     (dataset paths)
  dataset/data_process/convert_soma_csv_to_motion_lib.py (29-column validation,
                                                          batched FK backend)
  dataset/data_process/pack_motion_lib_to_textop.py    (retain 29 DoFs, nfeats=69)
  dataset/data_process/run_full_pipeline.sh            (29-DoF output and window config)
  scripts/create_goal_reference.py                     (CSV_DOF_IDXS, docstrings)
  scripts/run_planner_dar.sh                           (DATADIR)
  scripts/run_loop_dar.sh                              (DATADIR)
  scripts/run_smoke_vae_training.sh                    (DATADIR)
  scripts/run_smoke_ldm_training.sh                    (DATADIR)
  test/test_planner_utils.py                           (hardcoded 23 → 29)
  docs/planner_design.md                               (23→29 references)
  docs/planner_v1_basic_pipeline.md                    (feature dims, slices)
  docs/planner_v2_goal_scene.md                        (body count, wrist lock)
  docs/planner_vx_field_regulation.md                  (feature component table)

Mostly dimension-generic, with explicit order validation added:
  robotmdar/skeleton/forward_kinematics.py             (reads XML generically;
                                                        validates body/joint/
                                                        actuator order)
  robotmdar/dataloader/data.py                         (reads NPZ generically)
  robotmdar/eval/generate_dar.py                       (uses val_data generically)
  robotmdar/planner/planner_dar.py                     (uses val_data generically)
  robotmdar/train/manager.py                           (loss calcs use
                                                        DOF_DIM-derived shapes)
  robotmdar/train/train_dar.py                         (generic training loop;
                                                        validates 29-DoF order)

Must retrain:
  All VAE checkpoints
  All denoiser checkpoints
  All LDM checkpoints
```

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Wrist outputs violate limits or jitter | High | Validate per-joint normalization; add wrist limit and velocity/acceleration metrics or losses; clamp only at the controller safety boundary |
| Hand extend-body parent change alters goal keypoint FK | Medium | Verify FK output for reference poses matches expected hand positions |
| Palm and controller keypoints use different definitions | High | Use the wrist-yaw-attached palm center consistently and verify it against robot geometry and protocol data |
| Existing 23-DoF datasets become incompatible | Expected | Re-generate from Bones-SEED source CSVs (already have 29 columns) |
| Model capacity insufficient for 69-D features and 64-frame futures | Medium | Measure per-joint VAE reconstruction, especially wrists, before fixing latent capacity; do not assume the 128-D latent remains sufficient |
| Hardcoded slice indices missed during migration | High | Use symbolic constants where possible; run full test suite |

---

## Rollback Plan

If 29-DoF generation proves unstable or hand behaviors are worse than the
locked-wrist baseline:

1. Revert `DOF_DIM = 23` in `motion.py`
2. Revert `g1.yaml` to point to `g1_23dof_lock_wrist_fitmotionONLY.xml`
3. Revert all slice indices in `motion.py`
4. Restore `_reduce_mujoco_29_to_23` and `_expand_mujoco_23_to_29` in
   `planner_convert.py`
5. Switch back to 23-DoF datasets
6. Restore 23-DoF model checkpoints

All changes are isolated to the files listed above. The FK engine and
training infrastructure are unaffected by the DOF count — they operate
generically.

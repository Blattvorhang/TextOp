# Planner v2 — Body-Aware Goal Condition (Root + Hands + Feet)

> **Version scope**: "v2" in this document refers to the **goal condition** only
> (5-dim root-only → 15-dim body-aware).  TextOp's **motion FeatureVersion**
> (currently `FeatureVersion = 3`, 57-dim feature layout) is an independent
> concern maintained by the TextOp team and is unaffected by this proposal.
> All motion data, VAE I/O, and 23→29 DoF conversions remain on FeatureVersion 3.

## 1. Problem: 5-dim Goal is Insufficient

### 1.1 Current Design

The v1 planner uses a **5-dimensional ego-centric goal** as the target condition:

```
[ego_x, ego_y, delta_z, cos(delta_yaw), sin(delta_yaw)]
```

Implemented in `build_ego_goal()` at [robotmdar/utils/ego_condition.py](../robotmdar/utils/ego_condition.py).

These 5 dimensions are built from a world-space target but encode only the
**root position xyz relative to the current root + relative yaw**.  This includes
relative height (`target_z - current_z`).  No body pose information is present.
The model learns to "walk to the target location and face the target direction,"
and v1 training confirmed this works: the robot reaches the goal reliably and
corrects its path when deviating.

### 1.2 The Expressiveness Bottleneck

The G1 standing height is ~0.78 m, yet a goal with `z=0.45 m` could mean:

| Pose | root z | Description |
|------|--------|-------------|
| Kneeling | ~0.45 m | knees on ground |
| Crouching | ~0.45 m | bent knees, not touching ground |

Root xyz alone cannot distinguish these. Given the same goal `(x, y, z=0.45, yaw)`, the model might walk into a kneel or a crouch — whichever is more common near that height in the training data, not whichever the user intended.

More generally, the same root pose corresponds to infinitely many body poses (different arm and leg configurations). Meaningful goal-reaching requires constraining the body pose as well.

---

## 2. Reference: MOB's ELIMBS Goal Representation

See paper [papers/MOB.pdf](../../papers/MOB.pdf) and implementation at
[Motion-Occupancy-Base/training/](../../../Motion-Occupancy-Base/training/).

### 2.1 MOB's Goal Structure

MOB uses **root + hands + feet xyz** as the goal:

```python
# dataset.py:14
ELIMBS = (0, 10, 11, 20, 21)  # root, l_foot, r_foot, l_hand, r_hand
LIMBS = ELIMBS[1:]              # (10, 11, 20, 21)
```

5 keypoints × 3 dims = **15-dim goal vector**, extracted during training.

> **Naming note**: MOB's comment says `l_hand`/`r_hand`, but SMPL-X 22-joint
> `JOINT_NAMES` at indices 20/21 are `left_wrist`/`right_wrist` — MOB
> selected wrists, not hands proper (indices 22/23 in SMPL-24). The choice
> was deliberate: SMPL and SMPL-X both have wrists at 20/21, whereas hand
> indices differ between models. For G1, the wrist-locked skeleton exposes
> rigid `left_hand_link`/`right_hand_link` bodies at roughly palm position,
> so we call them **hands** — the code and FK body names match this term.

```python
tgt_limb_abs = data['jabs_tgt'][:, ELIMBS]  # [bs, 5, 3] → 15-dim
```

### 2.2 MOB's Conditioning Mechanism

1. **World → ego-centric**: `get_ss_tgt()` calls `change_system(..., keep_h=True)`.
   MOB subtracts the current root in XY and rotates by inverse heading, but keeps
   target Z as an absolute height.  TextOp v2 deliberately differs here: it
   subtracts the current root in XYZ, so all three output coordinates, including
   height, are relative goal-condition parameters.
2. **Separate projection**: `nn.Linear(15, 512)` projects goal into an independent control token
3. **Transformer fusion**: goal token + joints token + trajectory token + occupancy token are stacked as `[bs, 4, 512]` and fused via multi-head self-attention
4. **Classifier-free guidance**: training randomly drops the target with probability `DROP_TGT`

The core architecture is in `CtrlTransf` at
[models/ctrl_transf.py](../../../Motion-Occupancy-Base/training/models/ctrl_transf.py).

### 2.3 Why Test Root + Hands + Feet

- **Root** (pelvis): anchors the body in space
- **Feet** (ankles): provide useful lower-body constraints for standing,
  crouching, kneeling, and single-leg examples
- **Hands** (wrists): constrain upper-body pose — arm position, raised hands, etc.

These 5 keypoints are a compact v2 condition worth testing, not a unique
full-body pose representation.  Multiple joint configurations can share the
same end-effector positions.  The experiment is intended to measure how much
the motion prior, physical plausibility, and temporal smoothness resolve that
remaining inverse-kinematics ambiguity before adding more goal features.

---

## 3. TextOp vs MOB: Current State

| Aspect | MOB | TextOp (current) |
|--------|-----|------------------|
| Goal dim | 15-dim (5 keypoints × 3) | 5-dim (root xyz + yaw) |
| Goal frame | Ego XY, absolute Z (`keep_h=True`) | Relative ego XYZ |
| Goal injection | Separate control token → Transformer | Separate embed token → Transformer |
| Body model | SMPL(-X) 22 joints | G1 MJCF 23 DoF + FK |
| Keypoint source | `j_abs` index from dataset NPY | `RobotSkeleton.forward_kinematics()` -> `global_translation_extend` |
| Hand joints | SMPL joint indices 20/21 | Wrist joints locked; rigid hand link extension |
| Joint dim | 22 joints × 6d = 132 | 23 DoF (inside 57-dim feature) |

---

## 4. Proposed Design

### 4.1 Goal Representation: 15-dim Ego-Centric Keypoints

Expand the goal from 5 dims to 15 dims:

```
[root_x, root_y, root_z,          # pelvis
 l_foot_x, l_foot_y, l_foot_z,    # left ankle
 r_foot_x, r_foot_y, r_foot_z,    # right ankle
 l_hand_x, l_hand_y, l_hand_z,    # left hand
 r_hand_x, r_hand_y, r_hand_z]    # right hand
```

The input keypoints are world-space positions.  The 15 output parameters are
all **relative ego-centric coordinates**: subtract the current root in XYZ,
then rotate XY so +X is the current heading.  Consequently, the goal root Z and
all limb Z values are height differences relative to the current root.

The labeled left/right keypoint arrangement is intended to carry target heading
implicitly, so v2 remains 15-dimensional and does not append explicit yaw.
`world_goal_yaw` is retained in the unified function only for the root-mode API;
body mode does not consume it.

### 4.2 G1 Keypoint Mapping

**Base bodies (24 bodies, 23 DoF):**

| Keypoint | Body Name | body_names index (base) | MOB equivalent |
|----------|-----------|--------------------------|----------------|
| Root | `pelvis` | 0 | 0 (root) |
| Left Foot | `left_ankle_roll_link` | 6 | 10 (l_foot) |
| Right Foot | `right_ankle_roll_link` | 12 | 11 (r_foot) |

**Extended bodies (from `extend_config`, zero DoF, pure FK pass-through):**

| Keypoint | Body Name | body_names_augment index | Parent body |
|----------|-----------|---------------------------|-------------|
| Left Hand | `left_hand_link` | 24 | `left_elbow_link` (19) |
| Right Hand | `right_hand_link` | 25 | `right_elbow_link` (23) |

The hand links extend from the elbow via a hard-coded rigid offset (`pos: [0.25, 0.0, 0.0]` in the MJCF's local frame). Because the 6 wrist DoFs are locked to zero in the 23-DoF model, the hand position is fully determined by the arm chain (shoulder pitch/roll/yaw + elbow). It remains a meaningful, deterministic quantity.

**Foot keypoints**: `RobotSkeleton` already has `foot_id`, which delegates to
`ForwardKinematics.get_foot_id()` and uses `foot_names` from
[g1.yaml](../robotmdar/config/skeleton/g1.yaml).

**Hand keypoints**: add `get_hand_id()` to `ForwardKinematics` using
`self.body_names_augment`, then expose it as a cached `hand_id` property on
`RobotSkeleton`, matching the existing foot-ID pattern.  `RobotSkeleton.body_names`
currently exposes only the base names, so hand lookup must not use that property.

### 4.3 Model Changes

#### 4.3.1 Denoiser `goal_dim`

The denoiser supports both 5 and 15, driven by config:

```python
# mld_denoiser.py (DenoiserTransformer / DenoiserMLP)
def __init__(self, ..., goal_dim=5, ...):  # Hydra supplies cfg.denoiser.goal_dim
    self.goal_dim = goal_dim               # 5 (v1) or 15 (v2)
    self.embed_goal = nn.Linear(self.goal_dim, self.h_dim)
```

#### 4.3.2 Goal Construction Function (unified, `goal_type` dispatch)

The existing `build_ego_goal()` is **preserved, not renamed or deleted**. A `goal_type` parameter selects between v1 and v2:

```python
from enum import Enum


class GoalType(str, Enum):
    ROOT = "root"   # v1: 5-dim [x, y, z, cos_yaw, sin_yaw]
    BODY = "body"   # v2: 15-dim [root_xyz, lf_xyz, rf_xyz, lw_xyz, rw_xyz]


def build_ego_goal(world_goal_pos,           # [..., 3]
                   world_goal_yaw,           # [...] scalar or [..., 1]
                   reference_pos,            # [..., 3]
                   reference_rot,            # [..., 4] xyzw quaternion
                   goal_type: GoalType = GoalType.ROOT,
                   goal_keypoints=None,      # [..., 5, 3] — required when BODY
                   ) -> torch.Tensor:
    """Build ego-centric goal vector.

    goal_type=ROOT (original TextOp v1, 5-dim):
        [ego_x, ego_y, delta_z, cos(delta_yaw), sin(delta_yaw)]
        Encodes target root position + heading.  Cannot distinguish
        body poses sharing the same root height (e.g. kneeling vs
        crouching at z=0.45 m).  This is the convention used in
        planner_v1 and loop_dar; preserved for backward compatibility.

    goal_type=BODY (v2, 15-dim):
        [root_x, root_y, root_z,
         l_foot_x, l_foot_y, l_foot_z,
         r_foot_x, r_foot_y, r_foot_z,
         l_hand_x, l_hand_y, l_hand_z,
         r_hand_x, r_hand_y, r_hand_z]

        Five body keypoints in world coords, converted to the
        reference frame's ego-centric frame (X=forward, Y=left,
        Z=up).  Requires `goal_keypoints`.  Does NOT explicitly
        encode yaw — the spatial arrangement of feet and hands
        relative to root implicitly constrains heading.

        Keypoint order follows MOB's ELIMBS convention:
        [root, l_foot, r_foot, l_hand, r_hand].
        (MOB's comment calls them "hand" but SMPL-X indices
        20/21 are wrists; G1 uses hand links — see §2.1.)
    """
    if goal_type == GoalType.BODY:
        if goal_keypoints is None:
            raise ValueError("goal_keypoints is required for goal_type=BODY")

    current_yaw = quaternion_yaw(reference_rot)
    cos_yaw = torch.cos(current_yaw)
    sin_yaw = torch.sin(current_yaw)

    if goal_type == GoalType.ROOT:
        # --- 5-dim root-only goal (backward compatible) ---
        delta = world_goal_pos - reference_pos
        ego_x = delta[..., 0] * cos_yaw + delta[..., 1] * sin_yaw
        ego_y = -delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw
        delta_yaw = world_goal_yaw - current_yaw
        return torch.stack(
            (ego_x, ego_y, delta[..., 2],
             torch.cos(delta_yaw), torch.sin(delta_yaw)),
            dim=-1,
        )

    # --- 15-dim body-aware goal ---
    delta = goal_keypoints - reference_pos.unsqueeze(-2)  # [..., 5, 3]
    # current_yaw has shape [...]; add the keypoint axis for broadcasting.
    cos_yaw = cos_yaw.unsqueeze(-1)
    sin_yaw = sin_yaw.unsqueeze(-1)
    ego_x = delta[..., 0] * cos_yaw + delta[..., 1] * sin_yaw
    ego_y = -delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw
    ego_z = delta[..., 2]

    return torch.cat([
        torch.stack([ego_x[..., 0], ego_y[..., 0], ego_z[..., 0]], dim=-1),
        torch.stack([ego_x[..., 1], ego_y[..., 1], ego_z[..., 1]], dim=-1),
        torch.stack([ego_x[..., 2], ego_y[..., 2], ego_z[..., 2]], dim=-1),
        torch.stack([ego_x[..., 3], ego_y[..., 3], ego_z[..., 3]], dim=-1),
        torch.stack([ego_x[..., 4], ego_y[..., 4], ego_z[..., 4]], dim=-1),
    ], dim=-1)  # [..., 15]
```

Callers select mode via config:

```python
# train_dar.py / planner_convert.py
goal = build_ego_goal(
    world_goal_pos, world_goal_yaw,
    reference_pos, reference_rot,
    goal_type=cfg.data.goal_type,      # "root" | "body"
    goal_keypoints=world_goal_kps,     # only used when "body"
)
```

#### 4.3.3 Keypoint Extraction at Training Time

Add `world_goal_keypoints` to
`SkeletonPrimitiveDataset._extract_single_primitive()`. The dataset already has
`self.skeleton` and the raw motion includes `dof[goal_frame]` (23-dim joint
angles). `RobotSkeleton.forward_kinematics()` accepts a motion dictionary and
returns a dictionary with batch and time axes:

```python
# dataloader/data.py: _extract_single_primitive()
goal_motion = {
    'dof': torch.as_tensor(raw_motion['dof'][goal_frame:goal_frame+1],
                           dtype=torch.float32),
    'root_trans_offset': torch.as_tensor(
        raw_motion['root_trans_offset'][goal_frame:goal_frame+1],
        dtype=torch.float32),
    'root_rot': torch.as_tensor(raw_motion['root_rot'][goal_frame:goal_frame+1],
                                dtype=torch.float32),
}
goal_fk = self.skeleton.forward_kinematics(goal_motion)
_world_goal_keypoints = goal_fk['global_translation_extend'][
    0, 0, keypoint_indices
]  # [5, 3]
```

The dataset must also add `world_goal_keypoints` to the `tensor_keys` tuple in
`_organize_primitives_by_index()` so it survives batching.  When
`goal_type: root`, both extraction and batching omit this field and skip FK.

#### 4.3.4 Condition Assembly at Training Time

In `train_dar.py`'s `_conditions()`, the mode is config-driven:

```python
def _conditions(primitive, reference_pos, reference_rot, history_motion, cfg):
    goal = build_ego_goal(
        primitive['world_goal_pos'].to(cfg.device),
        primitive['world_goal_yaw'].to(cfg.device),
        reference_pos, reference_rot,
        goal_type=cfg.data.goal_type,
        goal_keypoints=(primitive['world_goal_keypoints'].to(cfg.device)
                        if cfg.data.goal_type == GoalType.BODY else None),
    )
    voxel = query_local_occupancy(...)
    return {'goal': goal, 'voxel': voxel, 'history_motion_normalized': history_motion}
```

Define `goal_type` under `cfg.data` and pass it into both dataset instances via
their Hydra constructor configuration.  Validate at startup that
`goal_type: root` is paired with `denoiser.goal_dim: 5` and `goal_type: body`
with `denoiser.goal_dim: 15`; these two settings must not drift independently.

#### 4.3.5 Goal Keypoints at Inference Time — Two Approaches

**Approach 1 — Controller sends keypoints (recommended).**

- Add `goal_keypoints [5, 3]` float32 to a new version of the TextOp history
  message; do not change the existing protocol-v2 multipart layout in place
- The controller owns the body pose intent and computes world-space keypoints
  using a pose library, IK, or another explicit high-level pose mechanism
- The planner passes them directly: `build_ego_goal(goal_type=BODY, goal_keypoints=...)`
- Zero extra computation on the planner side

**Approach 2 — Reference pose library (shim, no controller changes).**

The controller still sends only `goal_root_pos + goal_heading`. A pre-saved
`.npz` file provides body keypoints at yaw=0.  "Root at origin" here means that
the root XY is `[0, 0]`; Z values remain absolute world heights.  The planner
rotates the template in XY and translates only XY to `goal_root_pos`.  The
template pose therefore owns the target height.

Pre-saved format (`stand.npz`):

```python
# Saved at yaw=0 and root XY=[0, 0]. Z remains absolute. Computed via repo FK.
np.savez('stand.npz',
    keypoints=np.array([        # [5, 3]
        [0.0,  0.0,   0.78],   # root    — standing height
        [0.05, 0.08,  0.05],   # l_foot  — forward + left + near ground
        [0.05, -0.08,  0.05],  # r_foot
        [0.1,  0.22,  0.65],   # l_hand — arm position
        [0.1, -0.22,  0.65],   # r_hand
    ]),
    joint_angles=np.array([...]),  # [23] original DoF (for reference, not used at runtime)
    description="G1 standing pose, root XY at origin, absolute Z, yaw=0",
)
```

Runtime usage (no FK needed):

```python
def load_goal_keypoints_from_reference(ref_path: str,
                                        goal_root_pos: np.ndarray,   # [3]
                                        goal_heading: float) -> np.ndarray:  # [5, 3]
    """Load pre-saved body keypoints and transform to world goal."""
    data = np.load(ref_path)
    kps = data['keypoints'].copy()  # [5, 3], root XY=0, absolute Z

    # The selected template, not translation, defines target height.
    if not np.isclose(goal_root_pos[2], kps[0, 2], atol=1e-4):
        raise ValueError("goal_root_pos.z does not match reference-pose root z")

    # rotate around Z by goal_heading
    c, s = np.cos(goal_heading), np.sin(goal_heading)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotated = kps @ R.T                       # [5, 3]

    # Translate only in the XY plane; preserve every template Z value.
    rotated[:, :2] += goal_root_pos[:2]
    return rotated                            # [5, 3]
```

Only `stand.npz` is needed initially. Other poses (`squat.npz`, `sit.npz`, etc.)
carry their own absolute Z values and are added on demand. This logic can also
be placed on the **controller side**.  After world-space keypoints are built,
`build_ego_goal()` subtracts the current root Z, so the model still receives
relative height conditions.

#### 4.3.6 FK Implementation — Use the Repo's Existing FK

**Decision: use `RobotSkeleton.forward_kinematics()`, the pure Python/PyTorch `ForwardKinematics` already in the repo.**

Comparison:

| Option | Description | Assessment |
|--------|-------------|------------|
| **Repo FK** ([forward_kinematics.py](../robotmdar/skeleton/forward_kinematics.py)) | Parses MJCF XML → extracts kinematic tree (parent/child, local translations, joint axes) → matrix-chain FK in PyTorch | ✅ Already used in `reconstruct_motion()` during training — semantically identical to VAE joint outputs; ✅ No extra dependencies; ✅ Runs on CPU or GPU |
| **MuJoCo `mj_forward()`** | Calls MuJoCo engine's FK | ❌ Heavy extra dependency (`mujoco` Python package + model files); ❌ Must guarantee joint order and angle conventions match training data exactly; ❌ CPU-only; only meaningful in sim-validation contexts |

Core FK logic (`forward_kinematics_batch`, lines 285–318):

```python
# For each body node, chain: parent_rotation @ local_offset + parent_position
for i in range(J):          # J = 27 bodies (24 base + 3 extend)
    if parent_indices[i] == -1:
        # root body: use root_positions + root_rotations directly
        ...
    else:
        # child body
        jpos = parent_rotation @ self.local_translations[i] + parent_position
        rot_mat = parent_rotation @ self.local_rotation_matrices[i] @ joint_rotations[i-1]
```

Standard kinematic tree traversal. At training time, compute the 5 goal-frame keypoints by feeding `dof[goal_frame]` (23-dim joint angles) + `root_pos` + `root_rot` in and extracting `global_translation[keypoint_indices]`.

### 4.4 FK Cost Summary

| Scenario | FK needed? | Frequency | Notes |
|----------|-----------|-----------|-------|
| **Training data loading** | Yes, repo FK | 1× per primitive | `SkeletonPrimitiveDataset` already owns `self.skeleton`; single-frame FK (27 bodies) on CPU |
| **Inference (Approach 1)** | No | — | Controller sends 5 keypoints directly |
| **Inference (Approach 2)** | No | — | Load `.npz`, rotate/translate XY, preserve template Z |

Training FK overhead is negligible:
- The current training pipeline already runs multi-frame FK in `reconstruct_motion()` (`[B, T, 23]` → `global_translation [B, T, 27, 3]`)
- The goal-frame FK is one extra single-frame call, dwarfed by the batch FK
- Pre-computing all snippet keypoints into `.pkl` files is an optional optimization, not needed now
- See §4.3.6 for why repo FK is chosen over MuJoCo `mj_forward()`

### 4.5 Impact of Wrist Lock on Hand Keypoints

The G1's 6 wrist joints (`left_wrist_roll/pitch/yaw`, `right_wrist_roll/pitch/yaw`) are **locked to zero** in the 23-DoF model. The `left_hand_link` and `right_hand_link` bodies are rigid extensions from the elbow (offset 0.25 m), with position fully determined by the arm chain (shoulder × 3 + elbow).

Implications:

- The hand keypoint is a deterministic FK output — just not via an articulated wrist
- At training time: real motion data FK is pre-processed to wrist-locked FK, so hand positions are computed identically
- At inference time: same skeleton, same FK → hand positions are consistent

**No special handling needed.** As long as training and inference share the same skeleton (`body_names` + `extend_config`), hand link positions are consistent. The keypoint line between MOB's wrist and G1's hand is purely a naming artifact of the skeleton definitions — both are deterministic FK end-effectors.

### 4.6 Compatibility with Scene Occupancy

The scene occupancy voxel (`[grid_size^3]`) is unaffected by the goal representation change. Both share the denoiser's independent embed layers and classifier-free dropout mechanism.

The `guidance_scale` hyperparameter may need re-tuning — a 15-dim goal carries more information than 5-dim, so the gap between conditional and unconditional model behavior will differ.

---

## 5. Scope of Changes

### 5.1 Files to Modify

| File | Change |
|------|--------|
| `robotmdar/utils/ego_condition.py` | Add `GoalType` enum; extend `build_ego_goal()` with `goal_type` + `goal_keypoints` |
| `robotmdar/utils/planner_convert.py` | Thread `goal_type` through `state_to_ego_goal()`; add `load_goal_keypoints_from_reference()` (Approach 2, load NPZ + rotate/translate XY, preserve Z) |
| `robotmdar/model/mld_denoiser.py` | Read `goal_dim` from config, support 5/15 |
| `robotmdar/config/denoiser/def.yaml` | Add `goal_dim: 5` (default v1; override to 15 for v2 training) |
| `robotmdar/config/data/*.yaml` | Add `goal_type: root`, pass it into dataset constructors, and override to `body` for v2 |
| `robotmdar/dataloader/data.py` | Compute `world_goal_keypoints` in `_extract_single_primitive()` when `goal_type=body` and include it in batched `tensor_keys` |
| `robotmdar/train/train_dar.py` | Pass `cfg.data.goal_type` through `_conditions()` and validate it against `cfg.denoiser.goal_dim` |
| `robotmdar/skeleton/forward_kinematics.py` | Add `get_hand_id()` using `body_names_augment` |
| `robotmdar/skeleton/robot.py` | Expose cached `hand_id`, matching `foot_id` |
| `robotmdar/planner/planner_dar.py` | Make goal logging mode-aware; do not interpret body channels 3/4 as yaw |
| `robotmdar/eval/loop_dar.py`, `robotmdar/eval/vis_dar.py` | Remove hard-coded 5-dim construction and thread body keypoints in body mode |
| `sonic-msg/sonicmsg/messages.py` and canonical controller copy | Add a versioned TextOp body-goal message, sender, decoder, and planner-state field (Approach 1) |
| `test/` | Cover batched body conversion, relative Z, keypoint ordering, both goal modes, and config mismatch rejection |
| `assets/ref_poses/stand.npz` (new) | Reference keypoints `[5, 3]` at yaw=0, root XY at origin with absolute Z, pre-computed from repo FK |

### 5.2 Files NOT Affected

- VAE architecture (processes `[B, T, 57]` motion features; doesn't touch goal)
- Diffusion process (noise schedule only)
- Motion feature version (FeatureVersion 3, 57-dim)
- 23→29 DoF conversion logic
- ZMQ transport framework (Approach 1 still requires a versioned message schema)
- Existing four-argument `build_ego_goal()` calls and the v1 return value; the
  extended signature remains source-compatible through default arguments

---

## 6. Backward Compatibility

### 6.1 Dual-Mode Checkpoints

Each experiment directory stores its Hydra configuration beside the checkpoint
(`.hydra/config.yaml`, plus this repository's resolved `cfg.yaml`).  That paired
configuration records `denoiser.goal_dim` and `data.goal_type`:

- `goal_dim=5`: v1 checkpoint (root-only goal)
- `goal_dim=15`: v2 checkpoint (body-aware goal)

The launch/deployment path must use the configuration paired with the checkpoint;
that configuration selects the correct mode before model construction.  The
checkpoint payload itself contains model/optimizer state and does not perform
mode discovery inside `DARManager.load_model()`.  Startup must validate the
`goal_type`/`goal_dim` pairing before strict state-dict loading.

### 6.2 Phased Migration

1. **Phase 1**: Train v2 model (15-dim goal), validate body pose disambiguation
2. **Phase 2**: Add `goal_keypoints` to controller message (Approach 1) or use `stand.npz` shim (Approach 2)
3. **Phase 3**: Deploy new model to online planner

During Phase 1–2, the v1 (5-dim) planner continues operating unaffected.

### 6.3 Checkpoint Compatibility

A v1 checkpoint with `goal_dim=5` cannot be loaded directly into a `goal_dim=15`
model: `embed_goal` weight shapes differ (`[h_dim, 5]` versus `[h_dim, 15]`).

Migration options:
- Train from scratch (recommended — the v1 data distribution and task definition may not fully suit body-aware goals)
- Copy old `embed_goal` weights into the first 5 input channels, randomly initialize the remaining 10, then fine-tune

---

## 7. Risks and Open Questions

1. **Training data requirements**: Training data must contain explicit target body poses (the full joint configuration at the snippet end frame). This is naturally satisfied by motion capture data; for augmented/synthetic data, verify end-frame body pose validity.

2. **Goal degeneracy**: If hand and foot keypoints are highly correlated with root in the training data (e.g. 90% of snippet end frames are standard standing poses), the model may degenerate to attending only to the root 3 dims. Mitigate via **data diversity** (include crouching, kneeling, arm-raised goal poses).

3. **Implicit heading quality**: The 15-dim format intentionally represents
   target heading through the labeled left/right keypoint arrangement.  Evaluate
   pure-rotation and nearly symmetric poses explicitly to verify that the model
   learns this signal.  Adding explicit yaw is outside the v2 experiment.

4. **Controller integration**: Approach 2 (planner-side body pose inference) is inflexible for complex scenarios. Long-term, the controller should send full keypoints (Approach 1).

5. **Hyperparameter tuning**: `cond_goal_mask_prob` and `guidance_scale` may need re-search for 15-dim goals.

---

## 8. Summary

- Extend goal from 5-dim root(3)+yaw(2) to **15-dim root(3)+feet(6)+hands(6)**
- Follow MOB's ELIMBS convention: 5 keypoints = `(pelvis, left_foot, right_foot, left_hand, right_hand)`
- Wrist lock does not affect hand keypoint usability — it is a deterministic FK result
- Model-side changes are concentrated in the denoiser `embed_goal` layer and the goal construction function
- Use repo's existing `ForwardKinematics` for training-time keypoint extraction;
  inference needs no FK (either the controller sends keypoints, or `stand.npz`
  is rotated/translated in XY while retaining its Z values)
- `stand.npz` support can live on either the planner or controller side — same approach
- Phased migration: train first, validate, then integrate online

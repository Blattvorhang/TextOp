# Goal + Scene Conditioning: Loss Design Analysis & Recommendations

> Analyzes loss designs across TextOp (text-conditioned), MOB (goal+scene),
> and TRUMANS (goal+scene), and proposes loss-level improvements for the
> current TextOp goal+scene LDM pipeline.

---

## 1. Current TextOp Loss Analysis

### 1.1 Loss Components

The loss structure of `DARManager.calc_loss()` ([manager.py:799](../robotmdar/train/manager.py#L799)):

| Component | Weight | Formula | Domain |
|-----------|--------|---------|--------|
| `rec` | 1.0 | HuberLoss(pred_motion, gt_motion) | Normalized feature space |
| `latent_rec` | 1.0 | HuberLoss(pred_latent, gt_latent) | VAE latent space |
| `kl` | 1e-4 | KL(N(μ,σ) ‖ N(0,1)) | MVAE regularization (always 0 for DAR) |
| `body_trans` | 0.05 | HuberLoss(pred_body_trans, gt) | FK space — global translation |
| `body_rot` | 0.01 | HuberLoss(pred_body_rot, gt) | FK space — global rotation |
| `dof_pos` | 0.03 | HuberLoss(pred_dof, gt) | Joint angles |
| `dof_vel` | 1e-5 | HuberLoss(pred_dof_vel, gt) | Joint angular velocity |
| `foot_contact` | 0.01 | Weighted HuberLoss (contact mask) | Foot contact |
| `drift_xy/yaw` | 0.0 | HuberLoss (endpoint drift) | Currently disabled |
| `smooth` | 0.0 | L1 temporal difference | Currently disabled |

FeatureVersion v5 additionally includes delta losses (`trans_delta`, `joints_delta`, `dof_delta`).

### 1.2 Key Property: The Loss Is **Condition-Agnostic**

This is the most critical design observation. Every loss term is computed as:

```
loss = loss_fn(predicted_motion, ground_truth_motion)
```

The condition (text / goal / scene) **only serves as input to the Denoiser**, influencing what the model predicts, but **the loss itself contains no condition-specific terms**. There is no:

- Text-motion alignment loss (e.g., CLIP contrastive loss)
- Goal-arrival loss (distance from predicted endpoint to goal)
- Scene penetration loss (distance from predicted body joints to occupied voxels)

Implications of this design:

- ✅ **Advantage**: the loss transfers seamlessly from text conditioning to goal+scene conditioning — no loss changes needed, only condition embeddings change
- ❌ **Disadvantage**: the model receives **no explicit supervisory signal** about goal/scene — it can only learn the condition-motion association indirectly through motion reconstruction

### 1.3 Diffusion-Level Loss

Current setup uses `START_X` prediction + `MSE` loss type ([config/diffusion/def.yaml](../robotmdar/config/diffusion/def.yaml)):

- The Denoiser directly predicts `x_0` (clean latent) rather than noise ε
- Noise is injected via `diffusion.q_sample()`; loss is computed on the decoded motion
- `GaussianDiffusion.training_losses()` is **never called** — loss is always computed by `DARManager.calc_loss()` on decoded motion
- The `schedule_sampler` weights are multiplied into the total loss for importance weighting

### 1.4 Condition Dropout (Classifier-Free Guidance Preparation)

The Denoiser applies independent Bernoulli dropout to goal and scene (10% each):

```python
# mld_denoiser.py
self.cond_goal_mask_prob  = 0.1   # P(drop goal)
self.cond_scene_mask_prob = 0.1   # P(drop scene)
```

Probability distribution across the four combinations: full condition 81%, scene-only 9%, goal-only 9%, unconditional 1%. This sets up CFG inference, but the unconditional case is extremely rare (1%), which may yield insufficient unconditional pass quality.

---

## 2. Motion-Occupancy-Base (MOB) Loss Analysis

MOB is an autoregressive conditional motion generation framework (CtrlTransf) using goal (5 limb target joints) + scene (occupancy voxels) as conditions.

### 2.1 Loss Components

| Component | Weight | Formula | Condition Relevance |
|-----------|--------|---------|---------------------|
| **L1** | 1.0 | `l1_loss(pred, gt)` on rotations + directions | None |
| **L2** | 2.0 | `mse_loss(pred, gt)` on positions + velocities | None |
| **FLoss** (field loss) | 0.2 | `(‖delta_v‖/‖v‖)^2` — penalizes excessive field correction | **Strong — scene condition** |
| **VLoss** (velocity loss) | 10.0 | `‖v‖^2` — regularizes velocity magnitude | Indirect |
| **Grid Loss** (penetration) | 2.0 | L2 distance from penetrated body joints to nearest voxel center | **Strong — scene condition** |

### 2.2 Field Mechanism — MOB's Core Innovation

MOB does not merely treat occupancy voxels as a condition token; it **integrates them into the topology of the network output layer**:

```
Input: occu_l (egocentric occupancy voxels) + other conditions
  │
  ▼
CtrlTransf Network
  │
  ├── pred_vel (predicted velocity)
  │
  ▼
Field Correction:
  d_vecs = direction vectors pointing toward occupied voxels (in X-Y plane)
  delta_v = alpha × get_delta_v(d_vecs, pred_vel, offset)
    where get_delta_v applies a 1/distance^0.85 repulsive kernel
  out_vel = pred_vel + delta_v   ← corrected velocity
  │
  ▼
FLoss = (‖delta_v‖ / ‖v‖)^2     ← penalizes relying on delta_v for avoidance
VLoss = ‖v‖^2                    ← penalizes excessive velocity
```

**Key insight**: the field mechanism makes the scene condition differentiably influence the output. `delta_v` is **learnable** during training (regularized by FLoss/VLoss) and **enforced** during inference (no loss needed).

### 2.3 Penetration Loss (Grid Loss)

An explicit collision penalty: for each SMPL body joint that falls inside an occupied voxel, compute the L2 distance to the nearest voxel center, averaged over batch and joints.

```python
# train_utils.py calc_grid_loss
loss = ‖abs_pose[penetrated_joints] - nearest_voxel_center‖.sum() / (bs × n_joints)
```

### 2.4 Condition-to-Loss Mapping

| MOB Loss | Condition Type | Role |
|----------|---------------|------|
| L1 + L2 | General | Motion quality baseline |
| FLoss | **Scene (occupancy)** | Prevents network from over-relying on delta_v correction; encourages the network itself to learn avoidance |
| VLoss | **Scene (occupancy)** | Prevents delta_v from causing unrealistic velocities |
| Grid Loss | **Scene (occupancy)** | Explicitly penalizes body penetration into occupied regions |
| (no goal-specific loss) | Goal | Goal serves only as input condition; no explicit arrival loss |

---

## 3. TRUMANS Loss Analysis

TRUMANS is a conditional diffusion model using goal (target joint position) + scene (32³ occupancy voxels via ViT encoding) + action label as conditions.

### 3.1 Loss Components

TRUMANS's loss is remarkably simple:

```
loss = SmoothL1Loss(noise[unmasked], predicted_noise[unmasked])
```

**A single loss term**: noise prediction loss on unmasked positions only. Supports L1/L2/Huber types.

### 3.2 Mask Mechanism — TRUMANS's Core Innovation

Rather than using extra loss terms for goal conditioning, TRUMANS uses a **binary mask**:

```python
def get_mask(x_start, ind, fixed_frame, mask_y):
    mask_frame = first fixed_frame frames fully masked    # "history" condition
    mask_goal  = last frame joint[ind] masked              # "goal" condition
    mask = mask_frame | mask_goal
```

**During training**:
- `noise[mask] = 0.` → conditioned positions receive zero noise (model sees clean values)
- `loss = loss_fn(noise[~mask], pred[~mask])` → loss is computed **only** on non-conditioned positions

**During inference (fix_mode)**:
- After each denoising step, `set_fixed_points()` forcibly overwrites conditioned positions with goal values
- This is a **hard constraint**: `points[-goal_len:, joint*3:joint*3+3] = goal_value`

### 3.3 Scene Conditioning

Scene conditioning does not participate in loss computation; it serves only as model input:
- 32³ occupancy voxels → ViT (patch_size=8, dim=1024, depth=6) → `scene_emb` [B, 1, 512]
- `scene_emb` is summed with `t_emb` and `action_emb`, then prepended to the joint token sequence
- `free_p` probability randomly drops the scene embedding (CFG preparation)

### 3.4 Condition-to-Loss Mapping

| Condition | Role in Loss | Mechanism |
|-----------|-------------|-----------|
| **Goal** | **Excluded** from loss | Mask goal positions → 0 noise → model does not need to predict them |
| **Scene** | Does not affect loss | Extra token providing context only |
| **Action** | Does not affect loss | Extra token providing context only |
| **History frames** | **Excluded** from loss | Mask first N frames |

---

## 4. Comparative Analysis

### 4.1 Loss Complexity

```
TRUMANS:        1 term   (noise prediction on free joints)
TextOp:        ~7 terms  (rec + latent_rec + 5 geometry terms)
MOB:           ~5 terms  (L1 + L2 + FLoss + VLoss + Grid Loss)
```

### 4.2 Condition Handling Comparison

| Aspect | TextOp (current) | MOB | TRUMANS |
|--------|-----------------|-----|---------|
| **Goal condition** | Soft: 5-dim embedding token | Soft: ego coordinates of 5 limb joints → input features | **Hard**: mask + fix_mode forced overwrite |
| **Scene condition** | Soft: 25³ binary voxels → Linear embedding token | **Field**: occupancy voxels → repulsive field embedded in output layer + Grid Loss | Soft: 32³ voxels → ViT → prepend token |
| **Condition dropout** | Independent Bernoulli (goal 10%, scene 10%) | Condition dropout (configurable DROP_TGT, DROP_PTRAJ, DROP_VOX) | Unified `free_p` dropout (scene + action) |
| **CFG inference** | 4-pass factorized CFG (pending) | Standard CFG | Standard CFG |
| **Scene-related loss** | **None** | FLoss + VLoss + Grid Loss | **None** |
| **Goal-related loss** | **None** | **None** | **None** (but fix_mode is a stronger hard constraint) |

### 4.3 Key Differences

**TextOp vs MOB (scene handling)**:
- TextOp treats scene as an ordinary embedding token, relying on Transformer attention to associate occupancy information with motion
- MOB uses scene for both **input conditioning** and **output correction** (field mechanism + penetration loss)
- → MOB's approach provides stronger explicit guarantees for collision avoidance

**TextOp vs TRUMANS (goal handling)**:
- TextOp uses soft condition (ego goal embedding); the model learns to "move toward the goal"
- TRUMANS uses hard constraint (fix_mode), forcing the last frame to equal the goal position
- → TextOp's approach is more flexible (intermediate primitives need not arrive precisely), but doesn't guarantee arrival
- → TRUMANS's approach guarantees goal arrival, but requires the goal to be within snippet range (only applicable to P3)

**MOB vs TRUMANS (scene handling)**:
- MOB has scene-related losses (FLoss + Grid Loss); TRUMANS does not
- MOB's field mechanism provides structured collision avoidance; TRUMANS relies on ViT implicit learning
- → MOB's approach is better suited for precise physical obstacle avoidance; TRUMANS's approach is simpler but may be under-constrained

---

## 5. Recommendations for the Current TextOp Goal+Scene Pipeline

### 5.1 Core Assessment

**The current condition-agnostic loss can be used directly for goal+scene training** — the code already runs this way. All loss terms (rec, latent_rec, geometry losses) are independent of condition type; they only compare predicted motion vs. GT motion.

However, **the lack of explicit scene/goal supervisory signals** means:

1. Whether the model truly learns obstacle avoidance depends entirely on statistical correlations in the data (are there enough avoidance samples?)
2. Whether the model learns to move toward the goal depends entirely on implicit learning through goal embedding + motion reconstruction loss
3. Rare behaviors (e.g., step_over at only 0.18%) are nearly impossible to learn through pure reconstruction

### 5.2 Proposed Loss Improvement Roadmap

Three phases, ordered by priority:

#### Phase 1 — Zero-Code Changes (config-only tuning/validation)

**a) Establish baseline metrics for the current loss**

Before modifying loss, run the current pipeline and record:
- `eval_rec` (overall motion reconstruction)
- Collision rate (query penetration ratio of rollout FK joints against occupancy)
- Goal-arrival error (distance from rollout endpoint to world goal)

**b) Adjust condition dropout probabilities**

Current `cond_goal_mask_prob = cond_scene_mask_prob = 0.1` yields only 1% unconditional probability. Suggestions:
- Raise both to `0.15` → unconditional probability rises to 2.25%
- Or use shared dropout: drop both conditions simultaneously with probability `0.1` → unconditional probability 10%
- Or raise to the commonly used 0.2 → unconditional probability 4%

**c) Endpoint drift loss (optional, not goal-specific)**

Currently `drift_xy` and `drift_yaw` weights are 0.0. Enabling them improves
agreement with the GT endpoint, but it does not directly reference the goal and
therefore is not itself a goal-attraction loss:

```yaml
loss_weight:
  drift_yaw: 0.01
  drift_xy: 0.01
```

This remains a config-only reconstruction refinement.

#### Phase 2 — Recommended Lightweight Changes (minimal new code)

**a) Add Goal Direction Loss (NEW)**

The v1 implementation adds a lightweight goal direction loss in
`DARManager.calc_loss()`. For FeatureVersion 3, denormalized channels `7:9` are
the horizontal root translation deltas, so no additional FK pass is required:

```python
future = dataset.denormalize(future_motion_pred)
# Each delta is local to that frame's yaw. Rotate it into the primitive-start
# frame using cumulative future[..., 4] before summing the displacement.
root_displacement = integrate_in_start_frame(future[..., 7:9], future[..., 4])
goal_direction = ego_goal[..., :2]
valid = goal_direction.norm(dim=-1) > 0.1
valid &= goal_condition_keep_mask  # exclude goals dropped for CFG training
loss = mean(1 - cosine(root_displacement[valid], goal_direction[valid]))
```

The denoiser exposes the per-sample mask drawn by `cond_goal_mask_prob`. Masked
goals do not contribute to this loss; if every goal in a batch is masked, the
term is a graph-connected zero.

Suggested weight: `goal_direction: 0.01` (weak signal; should not dominate training)

**b) Simple Occupancy-Based Penetration Loss (referencing MOB Grid Loss)**

```python
def calc_penetration_loss(future_motion_pred_fk, scene, reference_pos, reference_rot):
    """
    Penalize predicted body joints that penetrate occupied voxels.
    Only applies L2 distance loss to joints falling inside occupied voxels.
    """
    body_joints = future_motion_pred_fk['global_translation_extend']  # [B, T, J, 3]
    # Query which joints fall inside occupied voxels
    occupied_mask = query_occupancy_for_joints(body_joints, scene)    # [B, T, J]
    if occupied_mask.any():
        penetration = (
            body_joints[occupied_mask] - nearest_free_voxel_center
        ).norm(dim=-1).mean()
    else:
        penetration = torch.tensor(0.0, device=body_joints.device)
    return penetration
```

Suggested weight: `penetration: 0.05` (same order as `body_trans`)

**c) Feature-Space Direction Loss (implemented as Goal Direction Loss above)**

Avoid FK overhead by adding directional supervision directly from denormalized
motion features:

```python
# V3 channels 7:9 encode horizontal root translation deltas in each frame's
# yaw coordinates. Integrate them in the primitive-start frame as above.
direction_loss = 1 - cosine_similarity(root_displacement, ego_goal[:, :2])
```

Suggested weight: `goal_direction: 0.01`

#### Phase 3 — Future Ablation Studies (more engineering required)

**a) MOB-style Field Mechanism**

This is MOB's most central innovation, but requires substantial changes:
- Add a field correction module after the Denoiser output
- Add FLoss + VLoss regularization
- Optionally: use field correction only at inference (without loss)

**Priority: Low** — MOB's field mechanism was designed for an SMPL autoregressive model; porting it to LDM latent space is not a straightforward mapping. First validate the baseline approach in Phases 1–2 to see whether the scene condition alone is sufficient.

**b) TRUMANS-style fix_mode (Hard Constraint)**

Apply fix_mode to the last primitive (P3), forcing the final frame to reach the goal. However, as noted in the design doc:
- P0–P2 cannot be clamped (they are intermediate steps)
- Only P3 can safely use this
- Need to distinguish "goals requiring precise arrival" from "waypoints requiring directional movement toward"

**Priority: Medium** — evaluate whether harder constraints are needed after implementing Phase 2 goal direction loss.

### 5.3 Recommended Phase 1+2 Combined Plan

Candidate improvements, to be introduced and ablated independently:

| Change | Type | Expected Effect |
|--------|------|-----------------|
| `drift_xy: 0.01, drift_yaw: 0.01` | Config | Future optional GT endpoint refinement |
| `cond_goal_mask_prob: 0.15, cond_scene_mask_prob: 0.15` | Config | Better CFG training distribution |
| `goal_direction` loss (new) | ~20 lines | Explicit goal-direction supervision |
| `penetration` loss (new) | ~30 lines | Explicit collision penalty |

### 5.4 Recommended Loss Weight Configuration

```yaml
# config/train/dar.yaml — suggested updates
loss_weight:
  # Core
  rec: 1.0
  latent_rec: 1.0
  kl: 1e-4

  # Geometry (unchanged)
  body_trans: 0.05
  body_rot: 0.01
  dof_pos: 0.03
  dof_vel: 1e-5
  foot_contact: 0.01

  # Keep disabled in v1; test later as an endpoint-reconstruction ablation
  drift_yaw: 0.0
  drift_xy: 0.0

  # v1 addition
  goal_direction: 0.01    # directional alignment

  # Future scene-loss ablation, not implemented in v1
  penetration: 0.0

  # Other (unchanged)
  fk_joints_rec: 1.0
  joints_consistency: 1.0
  joints_delta: 1.0
  trans_delta: 1.0
  orient_delta: 1.0
  dof_delta: 1.0
  smooth: 0.0
  quantize_rot: 0.0
  quantize_trans: 0.0
```

---

## 6. Design Rationale

### 6.1 Why TextOp's Condition-Agnostic Loss Still Works

The core principle of diffusion models is learning the conditional distribution `p(x|y)`. The loss compares `x_pred` against `x_gt`; the condition `y` influences `x_pred` through the forward pass of the neural network. As long as `y` contains sufficient task information (goal position + scene occupancy), gradients will backpropagate through the reconstruction loss to the condition embedding parameters, establishing an association between conditions and motion outputs.

This is fundamentally an **implicit conditional learning** process — no explicit condition loss is required.

### 6.2 Why Explicit Loss Is Still Needed

Implicit learning presupposes **sufficient statistical signal in the data**:

- If 95%+ of scenes are obstacle-free (walk/jog), the network can ignore the scene condition and still achieve low loss
- If `step_over` is only 0.18% of the data, the network can hardly learn obstacle avoidance from pure reconstruction loss
- Similarly for goal direction: if most motions are straight-line, the network can predict reasonable motion without the goal condition

This is why MOB added FLoss + Grid Loss, and TRUMANS used the hard constraint fix_mode — **when the data distribution is insufficient to drive implicit learning, explicit supervision is needed**.

### 6.3 Design Principles

1. **Soft condition (TextOp-style) suits directional tasks**: no need for precise arrival, just "move that way" — this is the planner's design philosophy
2. **Field mechanism (MOB-style) suits physical obstacle avoidance**: scene information serves as both input and output constraint
3. **Hard constraint (TRUMANS-style) suits goal arrival**: when the last frame should indeed equal the goal
4. **Add loss incrementally**: validate baseline in Phase 1, add lightweight supervision in Phase 2, consider major changes in Phase 3

---

## 7. References

- **TextOp** (`papers/TextOp.pdf`): Text-conditioned LDM for motion generation. Loss = motion reconstruction (Huber) + geometry terms in FK space. Conditions learned implicitly through CLIP embeddings.
- **MOB** (`papers/MOB.pdf`, `Motion-Occupancy-Base/`): Goal + scene-conditioned autoregressive model. Loss = L1 (rotations) + L2 (positions) + FLoss (field regularization) + VLoss (velocity regularization) + Grid Loss (penetration). Scene condition structurally integrated into output via field mechanism.
- **TRUMANS** (`trumans_utils/`): Goal + scene-conditioned diffusion model. Loss = single noise prediction loss (Huber) on unmasked positions. Goal condition enforced via binary mask + fix_mode hard constraint. Scene encoded via ViT as prepend token.
- **BONES-SEED** (`dataset/data_analyze/`): 71K clips, 20 action categories. Key rare categories: step_over (0.18%), kneel (1.5%), crouch (3.9%).

# Planner V10: End-Effector Goal Conditioning

> Status: proposed design for adding hand/foot position conditioning to the
> current `joint_state` split-goal planner.

Last updated: 2026-09-08

## Scope

V10 adds an explicit end-effector position modality to the current planner
goal. The parent modality is `end_effector`, and it contains four separately
maskable 3D sub-tokens:

```text
left_hand, right_hand, left_foot, right_foot
```

This is intentionally redundant with the 29-DOF joint target. Given the goal
root state and goal joints, forward kinematics already determines the wrist
and ankle positions. The redundancy is useful because it gives the denoiser a
direct spatial limb-placement signal while preserving the full-pose joint
condition.

The important V10 change is not "replace joints with keypoints". It is:

```text
current split joint_state goal
  + end_effector parent block in the goal input
  + four independent end-effector sub-tokens in the model
  + four independent end-effector masks
```

The model should be able to train and evaluate the following combinations:

| Joint token | End-effector sub-token(s) | Meaning |
|---|---|---|
| kept | any subset kept | full pose plus direct limb placement for selected limbs |
| kept | all dropped | current joint-state behavior |
| dropped | any subset kept | selected spatial hand/foot targets without full joint target |
| dropped | all dropped | root/orientation/velocity/text-only ablation |

For pure "no pose information" ablations, drop both `goal_joint` and
all four `goal_end_effector.*` sub-tokens. Dropping joints alone does not
remove all pose information, because any visible end-effector position still
encodes a meaningful subset of the pose.

## Relation To Earlier Goal Versions

V4 `body_ext` already used the same four limb keypoint semantics after
root/yaw/velocity/time:

```text
left_foot, right_foot, left_hand, right_hand
```

V10 keeps that semantic set, but defines the end-effector token order as:

```text
left_hand, right_hand, left_foot, right_foot
```

V10 reintroduces that idea in the current `joint_state` planner rather than
returning to the V4 goal contract:

- keep `goal_type=joint_state`;
- keep the current split goal tokens for horizontal, vertical, orientation,
  joint, velocity, and time;
- append one 12-D `end_effector` block to the goal tensor;
- split that block into four 3-D model tokens;
- mask each end-effector independently from joints and from the other
  end-effectors.

This preserves the current multimodal structure, where text, scene, root,
orientation, joint, velocity, time, and end-effectors can each be ablated. From
the raw goal-input tensor's point of view this is still one 12-D block; the
distinction appears in the model's tokenization and masking logic.

## MJCF Anchor Audit

V10 anchor geometry must come from the active MJCF only. The skeleton config may
select which asset file to load, but it must not define end-effector positions,
offsets, parent frames, or fallback geometry.

The relevant files are:

```text
TextOpRobotMDAR/description/robots/g1/g1_29dof.xml
TextOpRobotMDAR/robotmdar/skeleton/forward_kinematics.py
```

The current MDAR `g1_29dof.xml` contains explicit kinematic bodies for:

```text
left_ankle_roll_link
right_ankle_roll_link
left_wrist_yaw_link
right_wrist_yaw_link
```

It also contains explicit rubber-hand geoms under each wrist-yaw body:

```text
left_rubber_hand  geom pos = [0.0415,  0.003, 0]
right_rubber_hand geom pos = [0.0415, -0.003, 0]
```

This is enough evidence for a hand target: V10 should use the hand position
directly from the MJCF. The implementation may still represent that XML anchor
as a temporary FK extension internally, but the anchor definition must be read
from the active MJCF:

```text
parent body: left_wrist_yaw_link   geom mesh: left_rubber_hand
parent body: right_wrist_yaw_link  geom mesh: right_rubber_hand
```

The current `ForwardKinematics` parser reads the MJCF body tree. It does not
yet parse arbitrary geom/site positions as semantic goal points. That parser
limitation should not change the data contract. The V10 implementation should
extend the MJCF parsing path to either:

- parse the `left_rubber_hand` and `right_rubber_hand` geom offsets from the
  MJCF and transform them from their wrist-yaw parent frames, or
- auto-promote those MJCF anchors into named FK extension points at skeleton
  initialization.

In both cases, `g1.yaml` must not be used as the source of hand offsets.

Feet should follow the same "use the XML if it is explicit" rule. Some G1 MJCF
variants, such as `g1_29dof_with_collision.xml`, define:

```text
<site name="left_foot" .../>
<site name="right_foot" pos="0 0 0"/>
```

under the ankle-roll bodies. Those sites are valid foot anchors and should be
used when the active training XML exposes them. In the inspected
`g1_29dof_with_collision.xml`, the right site has `pos="0 0 0"` and the left
site omits `pos`, which defaults to the same value in MJCF. Therefore those
foot sites are semantically explicit but numerically equal to the
`left/right_ankle_roll_link` body origins.

The current default asset file is `g1_29dof.xml`, which does not include the
named foot sites. In that case the foot anchor falls back to the ankle-roll body
origin from the same MJCF body tree. Do not average the collision geoms or
infer a sole center unless that point becomes an explicit MJCF site/body.

`g1_29dof_with_collision.xml` is not a drop-in equivalent replacement for the
active `g1_29dof.xml`. It is useful as evidence that Unitree-style G1 MJCFs use
`left_foot/right_foot` sites at the ankle-roll origin, but the two files are
not globally FK-equivalent:

- `g1_29dof.xml` sets `waist_yaw_link pos="0 0 0.044"`;
- `g1_29dof_with_collision.xml` omits that body `pos`, so MJCF defaults it to
  zero;
- the current repo FK parser does not resolve MJCF `<default>` joint classes,
  while `g1_29dof_with_collision.xml` stores joint axes through those defaults.

Therefore V10 should keep the active MJCF as the only training source. If the
project switches the active asset to `g1_29dof_with_collision.xml`, that is a
schema change and requires FK parser support for MJCF defaults plus fresh goal
statistics.

Therefore the default V10 end-effector anchors are:

| Logical row | Default source | Reason |
|---|---|---|
| `left_hand` | `left_rubber_hand` geom on `left_wrist_yaw_link` | explicit MJCF hand geom offset |
| `right_hand` | `right_rubber_hand` geom on `right_wrist_yaw_link` | explicit MJCF hand geom offset |
| `left_foot` | `left_foot` site if present, else `left_ankle_roll_link` | explicit MJCF site; fallback is equivalent when site pos is zero |
| `right_foot` | `right_foot` site if present, else `right_ankle_roll_link` | explicit MJCF site; fallback is equivalent when site pos is zero |

The logical names remain hand/foot because they describe the command modality.
For hands, the default physical anchor is the XML rubber-hand frame. For feet,
the default physical anchor is the XML foot site when present and the ankle-roll
origin otherwise.

For planner-level conditioning, the active `g1_29dof.xml` is sufficient:

- hands use MJCF rubber-hand geom anchors, which are close enough hand proxies
  for goal conditioning;
- feet use ankle-roll body origins when the active MJCF lacks `left_foot` and
  `right_foot` sites; this is acceptable because the inspected explicit foot
  sites in the richer MJCF variants also sit at the ankle-roll origin;
- this contract is not intended to provide millimeter-level palm center, sole
  center, toe, heel, or contact-patch targets.

If a future MJCF introduces explicit named sites/bodies for palms or foot
centers, switching to those points is allowed only with a schema change, a
goal-statistics cache invalidation, and tests proving train and deployment FK
use the same point definitions.

Do not silently borrow `left_palm`, `right_palm`, `left_foot`, or `right_foot`
sites from the tracker-side Unitree MJCF. Those sites exist in some tracker
assets, but V10 anchor definitions must resolve from the active MDAR MJCF used
by `RobotSkeleton.forward_kinematics()`.

## Data Source Of Truth

End-effector training targets must be derived from the same source state as the
joint target:

```text
root position at goal_frame
root rotation at goal_frame
29-DOF joint position at goal_frame
```

Then run FK once and gather the selected anchors. Feet can be read directly
from FK site positions when the active XML exposes `left_foot/right_foot`, or
from ankle-roll body positions when those sites are absent. Hands use the FK
transform of each wrist-yaw body plus the rubber-hand local geom offset parsed
from the XML:

```text
goal_motion = {
  root_trans_offset: root_pos[goal_frame],
  root_rot: root_rot[goal_frame],
  dof: q[goal_frame],
}

goal_fk = skeleton.forward_kinematics(goal_motion)
world_goal_end_effectors =
  [
    fk_transform(left_wrist_yaw_link)  @ xml_geom_pos(left_rubber_hand),
    fk_transform(right_wrist_yaw_link) @ xml_geom_pos(right_rubber_hand),
    fk_site(left_foot) or fk_pos(left_ankle_roll_link),
    fk_site(right_foot) or fk_pos(right_ankle_roll_link),
  ]
```

This rule is deliberately boring and important:

- do not hallucinate a neutral hand/foot offset;
- do not take keypoints from another skeleton unless audited and converted;
- do not mix tracker-side sites with MDAR-side training FK;
- do not derive end-effectors after applying the joint mask;
- do not change the goal frame. End-effectors use the same `goal_frame` as the
  root, orientation, joints, velocity, and arrival time.

Masking happens after the full condition is built. Even if the joint token is
dropped, any subset of the four end-effector sub-tokens may remain, because
the whole point of V10 is to let these related modalities be available
independently.

### Training-Time Computation

Compared with the current 55-D split `joint_state` goal, V10 does require an
extra end-effector anchor evaluation during dataset batching or preprocessing.
It does not require extra labels, inverse kinematics, simulation, or controller
rollout.

Training already has FK in two relevant paths:

- `SkeletonPrimitiveDataset.reconstruct_motion(..., ret_fk=True)` reconstructs
  a motion feature tensor and immediately calls
  `self.skeleton.forward_kinematics(...)`.
- the geometry-loss path already reconstructs both predicted and GT future
  motions with `ret_fk=True`, then uses `global_translation_extend`,
  `global_rotation`, `dof_pos`, `dof_vel`, and foot translations for FK-space
  losses. This FK cache remains owned by geometry losses and must not be
  changed to a different absolute pose for V10 goal supervision;
- the legacy `body/body_ext` goal path already computes a single goal-frame FK
  in `_world_goal_keypoints(...)` and gathers goal keypoints from that FK
  result.

Therefore the implementation is not adding FK capability from scratch. It is
adding a shared end-effector anchor extractor on top of the existing FK result.
The call site still matters:

- condition construction happens before the denoiser forward pass, so the
  dataloader or preprocessing path must compute/cache the target
  end-effector block and per-limb tokens before building `y["goal"]`;
- loss and metric computation happen after prediction, but they must be aligned
  to the same current-state reference `s_t` used to encode the goal tensor. The
  geometry-loss FK cache can be reused only if it was reconstructed with the
  identical `s_t` absolute pose. In the current training path it is not, so the
  V10 goal loss reconstructs the prediction with `abs_pose=s_t`, selects the
  goal frame, runs FK for that single frame, and then transforms the predicted
  anchors back with `R_t.T @ (p - p_t)`.

For each selected `goal_frame`, the loader already has:

```text
root position
root rotation
29-DOF joint position
```

V10 uses those same tensors to run one FK pass over the goal frame and then
extract four anchors. The work is small relative to training:

- one single-frame FK per primitive if computed on the fly;
- or one batched FK call for all primitives in a batch;
- or precompute/cache the 12-D world anchors in the packed samples or
  `goal_stats` refresh path if dataloader overhead shows up in profiling.

The recommended first implementation is batched FK in the dataloader:

```text
goal_fk = skeleton.forward_kinematics(goal_motion_batch)
left_hand = parent_pos + parent_rot @ mjcf_left_rubber_hand_offset
right_hand = parent_pos + parent_rot @ mjcf_right_rubber_hand_offset
left_foot = goal_fk["global_translation"][left_ankle_or_foot_body_id]
right_foot = goal_fk["global_translation"][right_ankle_or_foot_body_id]
```

Use `global_translation` and `global_rotation_mat` for anchors whose parent is
an original MJCF body, including the wrist-yaw parent bodies in the current
`g1_29dof.xml`. Use `global_translation_extend` /
`global_rotation_mat_extend` only for anchors that have been explicitly
auto-promoted into extension points. Do not call FK again after masking; masks
apply only to the already-built condition tensor.

Keep the existing raw/scaled split in the training batch:

- `y["goal"]` carries the scaled multimodal goal used by the denoiser;
- the raw end-effector target should also be available for supervision, either
  as the unscaled end-effector slice in `y["ego_goal_raw"]` after the schema
  grows to 67-D, or as a separate `y["goal_end_effectors_ego_raw"]` tensor with
  shape `[B, 4, 3]`.
- `y["goal_reference_pos_world"]` and `y["goal_reference_rot_world"]` carry the
  world pose of the current state `s_t`; they are used only by goal-space EE
  supervision and do not change the VAE reconstruction or geometry-loss
  coordinate convention.

The exact but cheap loss-side route is:

```text
future_motion_pred
  -> reconstruct_motion(abs_pose=s_t, ret_fk=False)
  -> select goal_frame per sample
  -> skeleton.forward_kinematics(single_goal_frame)
  -> extract four MJCF anchors in world frame
  -> R_t.T @ (p_ee_world - p_t)
  -> compare with y["ego_goal_raw"][..., 55:67]
```

A local CPU micro-benchmark with the real G1 FK (`B=64`, `T=64`) gave the
following approximate costs:

| Route | Approx. cost |
|---|---:|
| post-align already-existing FK only | 0.05 ms |
| `s_t` reconstruct + full-sequence FK | 383 ms |
| `s_t` reconstruct + single-goal-frame FK | 38 ms |

The first route is fastest but is only exact when the existing FK cache was
already integrated from the same `s_t` absolute pose. The selected-frame route
is the recommended default because it preserves the reference contract without
running FK over the whole future sequence.

## Coordinate Convention

V10 should follow the current rotation-matrix split-goal convention, not the
old yaw-only V4 convention.

Let:

```text
p_ref  = root position at the final history frame
R_ref  = root rotation matrix at the final history frame
p_ee   = world FK/anchor position of one goal end-effector
```

Encode each end-effector in the reference root frame:

```text
p_ee_ref = R_ref.T @ (p_ee - p_ref)
```

The four rows are flattened in fixed order:

```text
end_effector_goal = [
  left_hand_x,  left_hand_y,  left_hand_z,
  right_hand_x, right_hand_y, right_hand_z,
  left_foot_x,  left_foot_y,  left_foot_z,
  right_foot_x, right_foot_y, right_foot_z,
]
```

These are absolute target point positions relative to the current root frame.
They are not deltas from the current hand/foot positions, and they are not
positions relative to the goal root. This mirrors the old body-keypoint
semantics: the point target includes both spatial displacement and limb pose.

## Goal Layout

The least disruptive V10 layout appends 12 end-effector channels to the current
55-D split goal. Existing V7/V9 slices remain unchanged:

| Slice | Size | Meaning |
|---|---:|---|
| `0:9` | 9 | horizontal root/navigation token |
| `9:15` | 6 | vertical/gravity token |
| `15:21` | 6 | relative root rotation token |
| `21:50` | 29 | 29-DOF joint target token |
| `50:54` | 4 | root velocity token |
| `54:55` | 1 | arrival time in seconds |
| `55:67` | 12 | end-effector FK/XML-anchor position block |

Recommended constants:

```python
MULTIMODAL_GOAL_DIM = 67
SPLIT_END_EFFECTOR_SLICE = slice(55, 67)
SPLIT_END_EFFECTOR_LEFT_HAND_SLICE = slice(55, 58)
SPLIT_END_EFFECTOR_RIGHT_HAND_SLICE = slice(58, 61)
SPLIT_END_EFFECTOR_LEFT_FOOT_SLICE = slice(61, 64)
SPLIT_END_EFFECTOR_RIGHT_FOOT_SLICE = slice(64, 67)
SPLIT_END_EFFECTOR_TOKEN_ORDER = (
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
)
SPLIT_END_EFFECTOR_SCHEMA = "rotmat_v10_hor_vert_joint_ee"
```

The goal input layout remains a single contiguous 12-D block. Model-side
conditioning splits that block into four 3-D tokens:

```text
goal_ee_left_hand
goal_ee_right_hand
goal_ee_left_foot
goal_ee_right_foot
```

The embedder may be implemented as either four small embedders or one shared
3-D end-effector embedder plus a learned limb-type embedding. The latter keeps
the parent modality unified while still letting the model know which limb a
token represents:

```text
embed_goal_ee_left_hand  = EE_MLP(goal_ee_left_hand_scaled)  + type(left_hand)
embed_goal_ee_right_hand = EE_MLP(goal_ee_right_hand_scaled) + type(right_hand)
embed_goal_ee_left_foot  = EE_MLP(goal_ee_left_foot_scaled)  + type(left_foot)
embed_goal_ee_right_foot = EE_MLP(goal_ee_right_foot_scaled) + type(right_foot)
```

For the MLP denoiser path, the flattened condition input grows by four hidden
tokens:

```text
old split input: time + 6 goal tokens + scene + history + noise = 10 * h_dim
new split input: time + 10 goal tokens + scene + history + noise = 14 * h_dim
```

For the transformer denoiser path, insert the four new tokens into the
condition sequence with the other goal tokens:

```text
timestep,
goal_horizontal,
goal_vertical,
goal_orientation,
goal_joint,
goal_ee_left_hand,
goal_ee_right_hand,
goal_ee_left_foot,
goal_ee_right_foot,
goal_velocity,
goal_time,
scene,
history,
noise
```

The slice order and token order should match `SPLIT_END_EFFECTOR_TOKEN_ORDER`.
Appending the 12 channels preserves all existing 55-D slices and makes
accidental checkpoint compatibility failures easier to detect.

## Scaling And Statistics

End-effector coordinates should be scaled with frozen train-set statistics and
should not be mean-centered.

Reason: a fully masked condition is represented as zeros. If end-effector
features are mean-centered, the zero vector becomes a common valid target
instead of a clean null token. For root and end-effector positions, scale-only
keeps the "masked is all zero" convention easy for the model to separate from
real targets.

Add one end-effector scale entry to `goal_stats.pkl`:

```text
s_ee: shape [12], [4, 3], or scalar
```

Recommended first implementation:

- compute robust per-channel standard deviation for the 12 encoded
  end-effector coordinates;
- store `s_ee = 1 / std_ee.clamp_min(eps)`;
- clip physically impossible outliers before statistics, matching the current
  goal-statistics style;
- scale `goal[..., 55:67] *= s_ee`;
- do not subtract a mean.

The `goal_stats.pkl` meta block must include:

```text
goal_dim: 67
goal_schema: rotmat_v10_hor_vert_joint_ee
end_effector_source: active_mjcf
end_effector_token_order:
  - left_hand
  - right_hand
  - left_foot
  - right_foot
end_effector_anchor_queries:
  - left_rubber_hand@left_wrist_yaw_link
  - right_rubber_hand@right_wrist_yaw_link
  - left_foot_site_or_left_ankle_roll_link
  - right_foot_site_or_right_ankle_roll_link
resolved_end_effector_anchors:
  - type: geom
    mesh: left_rubber_hand
    parent_body: left_wrist_yaw_link
  - type: geom
    mesh: right_rubber_hand
    parent_body: right_wrist_yaw_link
  - type: site_or_body
    name: left_foot
    resolved_from: left_ankle_roll_link
  - type: site_or_body
    name: right_foot
    resolved_from: right_ankle_roll_link
mjcf_file: TextOpRobotMDAR/description/robots/g1/g1_29dof.xml
```

If the MJCF file, token order, resolved anchor type, parent body, local offset,
or source type changes, recompute `goal_stats.pkl` and treat old checkpoints
as incompatible.

## Masking

Add four independent mask probabilities under the parent `end_effector` key:

```yaml
denoiser:
  cond_mask_prob:
    goal:
      end_effector:
        left_hand: 0.3
        right_hand: 0.3
        left_foot: 0.3
        right_foot: 0.3
```

The parent key is important: configuration, logging, and deployment can still
refer to the modality as `end_effector`, while each limb keeps its own mask.
If compatibility with old scalar config is needed, a scalar
`goal.end_effector: 0.3` may be broadcast to all four sub-keys during config
normalization.

Add matching explicit inference flags. A parent `all` flag drops the complete
end-effector modality; child flags drop individual limb tokens:

```yaml
force_drop_goal:
  end_effector:
    all: false
    left_hand: false
    right_hand: false
    left_foot: false
    right_foot: false
```

The mask behavior should mirror the existing split-goal components:

```text
goal_end_effector_condition_keep_mask: [B, 4] bool

column order:
  left_hand, right_hand, left_foot, right_foot
```

During training:

```text
end_effector_goal = goal[..., 55:67].reshape(B, 4, 3)
ee_keep = sample_keep_mask(
  probs=[
    cond_mask_prob.goal.end_effector.left_hand,
    cond_mask_prob.goal.end_effector.right_hand,
    cond_mask_prob.goal.end_effector.left_foot,
    cond_mask_prob.goal.end_effector.right_foot,
  ],
  force_drop=force_drop_goal.end_effector,
)
end_effector_goal = end_effector_goal * ee_keep[..., None]
```

During loss and metric computation, any end-effector-specific goal metric must
use `goal_end_effector_condition_keep_mask`. Existing joint, root, orientation,
velocity, and text masks remain unchanged.

Do not couple `cond_goal_joint_mask_prob` and
`cond_mask_prob.goal.end_effector.*`. They are intentionally separate even
though the information overlaps.

## Losses And Metrics

Use one unified end-effector objective. Do not introduce separate hand and foot
loss weights in the first version. The four limb tokens have separate masks,
but they share one objective name and one loss weight.

The supervision signal is direct position supervision:

```text
pred_ee_ego:   [B, 4, 3]
target_ee_ego: [B, 4, 3]
ee_keep:       [B, 4]

loss/goal_end_effector =
  sum_over_visible_limb_tokens(
    mean_over_visible_samples(
      SmoothL1(||pred_ee_ego - target_ee_ego||_2, beta)
    )
  )
```

The recommended first criterion is SmoothL1/Huber on the 3D Euclidean distance,
not plain L2/MSE and not per-axis averaging. It is quadratic near zero for
precision, linear for larger misses, and its unit remains meters. The default
transition is:

```yaml
train:
  manager:
    goal_end_effector_loss_beta: 0.05
    loss_weight:
      goal:
        end_effector: 0.0
```

`goal_end_effector_weight: 0.0` is a useful compatibility default. Enabling the
loss is then a training-config decision, not a schema requirement.

Compute `pred_ee_ego` by reconstructing the predicted motion with the current
state `s_t` as `abs_pose`, taking the selected arrival frame, extracting the
four MJCF-resolved anchors, and transforming them with the same reference-root
convention used by the goal token:

```text
pred_ee_ego = R_t.T @ (pred_ee_world - p_t)
```

Compute `target_ee_ego` from the unscaled target end-effector slice in
`y["ego_goal_raw"]`. The loss path should not silently replace this target with
`future_motion_gt_fk`, because the denoiser condition must exist before the
loss is computed and must use the same reference `s_t`.

Gate the loss and metrics with `goal_end_effector_condition_keep_mask`. When a
limb token is dropped, that row contributes zero. When all four limb tokens are
dropped for a sample, the unified end-effector loss contributes zero for that
sample.

The required metric is unified as well:

```text
metric/eval/goal_end_effector_l2
```

Per-row diagnostic metrics such as `left_foot`, `right_foot`, `left_hand`, and
`right_hand` can be logged for debugging, but they should not become separate
training objectives unless a later experiment shows that one shared objective
is insufficient.

## Deployment Contract

The wire protocol should carry raw physical quantities. Scaling and masking
remain planner/model internals.

For a full-pose target:

1. Receive or construct the target root pose and 29-DOF joint target.
2. Run the same MDAR FK in the planner.
3. Fill `goal_end_effectors_world` from XML foot sites when present, ankle-roll
   body origins otherwise, and XML rubber-hand anchors.
4. Keep the joint token and all four end-effector tokens unless ablation flags
   drop a subset.

For an end-effector-only command:

1. Receive `goal_end_effectors_world` in the shared world frame.
2. Validate shape `[4, 3]` and row order.
3. Set or force-drop the joint token depending on whether a joint target is
   also available.
4. Receive or construct `goal_end_effector_keep_mask` with shape `[4]` so a
   command can provide only one hand, one foot, or any subset.
5. Transform the visible points into the reference root frame in the same way as
   training.

Row order on the wire must be fixed:

```text
left_hand, right_hand, left_foot, right_foot
```

Shape checks cannot detect swapped hands and feet, so producer and consumer
must share this order explicitly. The keep mask uses the same row order.

## Implementation Checklist

1. Add `GoalEncoding` or schema support for the 67-D multimodal goal.
2. Add `SPLIT_END_EFFECTOR_SLICE = slice(55, 67)` and a schema string.
3. Add an MJCF anchor resolver instead of reusing `hand_id` or reading
   `extend_config`. The resolver should query the active MJCF with these rules:

   ```yaml
   end_effector_anchor_queries:
     - name: left_hand
       type: geom
       parent_body: left_wrist_yaw_link
       mesh: left_rubber_hand
     - name: right_hand
       type: geom
       parent_body: right_wrist_yaw_link
       mesh: right_rubber_hand
     - name: left_foot
       type: site_or_body
       site: left_foot
       fallback_body: left_ankle_roll_link
     - name: right_foot
       type: site_or_body
       site: right_foot
       fallback_body: right_ankle_roll_link
   ```

   These query names select MJCF elements only; they must not carry local
   positions or offsets in YAML.

4. In the dataloader, compute `world_goal_end_effectors` from the same
   goal-frame root/joint state used for `world_goal_dof`: foot site/body
   anchors from FK, hand geom anchors from wrist FK transforms plus XML local
   offsets.
5. In goal construction, transform those anchor points with
   `R_ref.T @ (p - p_ref)` and append the flattened 12-D block.
6. Extend goal statistics with `s_ee` and schema metadata.
7. Add four `cond_mask_prob.goal.end_effector.*` probabilities, matching
   child force-drop flags, and
   `goal_end_effector_condition_keep_mask` with shape `[B, 4]`.
8. Split the 12-D block into four 3-D end-effector tokens in the model path.
9. Add the unified generated end-effector position loss and eval metric at the
   arrival frame, gated by `goal_end_effector_condition_keep_mask`.
10. Update planner/deployment conversion to either compute all end-effectors by
    FK from a full-pose target or accept an explicit world-frame subset plus
    `goal_end_effector_keep_mask`.

## Tests

Required tests:

- MJCF audit: every site/body foot anchor resolves in the active MDAR MJCF and
  every geom hand anchor exists under the resolved parent body.
- FK/XML source consistency: for a sampled goal frame, the dataloader
  `world_goal_end_effectors` equals the resolved foot site/body positions plus
  the wrist-yaw FK transforms applied to the XML rubber-hand local offsets.
- Frame consistency: root, orientation, joints, velocity, time, and
  end-effectors all use the same `goal_frame`, except velocity's intentional
  finite-difference neighbor.
- Coordinate transform: a synthetic global SE(3) transform produces the same
  encoded end-effector coordinates in the reference frame.
- Mask independence: dropping joints does not drop end-effectors, dropping one
  end-effector row does not drop the other three, and dropping end-effectors
  does not drop joints.
- Forced inference flags: `force_drop_goal_joint=true` and
  `force_drop_goal.end_effector.*=true` affect only their own tokens.
- Statistics cache validation: changing `goal_dim`, `goal_schema`,
  `end_effector_source`, `end_effector_token_order`, `mjcf_file`,
  `resolved_end_effector_anchors`, or any resolved local offset invalidates old
  `goal_stats.pkl`.

## Open Decisions

The default hand target is now the MJCF rubber-hand anchor, not the wrist body
origin and not any skeleton-config extension name. Two decisions remain:

1. Feet use `left/right_foot` sites when present. In the inspected collision
   XML these sites sit at ankle-roll origin, so this is a semantic improvement
   without changing the numeric anchor. If precise sole center is required,
   add an explicit sole site/body and switch with a new schema and fresh goal
   statistics.
2. If precise palm center is required rather than rubber-hand mesh origin, add
   explicit named palm sites/bodies to the active MDAR MJCF. Do not infer it
   from tracker-side assets without copying that definition into the training
   MJCF/schema.

# Planner V4: Extended Body Goal Conditioning

## Scope

V4 extends DAR training with target root velocity and remaining time while
retaining root position, optional root yaw, and four limb targets. It does not
change the 29-DOF motion representation or MVAE.

Planner protocol and online goal construction use the same versioned goal
types as training:

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
| `8:9` | 1 | Remaining time in seconds, or zero for the recommended ablation | `cond_goal_time_mask_prob=0.0` |
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

`goal_timestep_mode=relative` computes remaining time as:

```text
goal_timestep = (goal_frame - reference_frame) / fps
```

It is stored and batched as `[B, 1]`.

`goal_timestep_mode=zero` emits zero in the same channel. This is the
recommended V4 retraining default: it preserves the 21-D checkpoint interface
while removing time as a learned condition.

## Goal-Frame Sampling

`goal_offset` may be negative. `goal_offset_range=[min, max]` enables uniform,
inclusive random sampling. The configured lower bound must satisfy:

```text
min >= 1 - future_len
```

This ensures every goal lies strictly after its primitive reference frame.
For `future_len=64`, the optional random-offset experiment may use `[-63, 0]`.

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

The recommended training configuration uses `goal_per_primitive=true`, fixed
offset `0`, and therefore the last frame of each primitive as its goal. This
matches the last successful 23-DOF/64-future experiment. Validation uses the
same fixed goal. Bounds are checked before FK or velocity extraction for both
`goal_per_primitive` modes.

Random offsets remain supported as an explicit experiment. They make the task
materially harder: the model must generate the full 64-frame future while an
intermediate frame is selected as the goal, and the timestamp is needed to
locate that frame. Do not combine this experiment with
`goal_timestep_mode=zero`, because different offsets would then become
temporally ambiguous.

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
  goal_per_primitive: true
  goal_offset: 0
  goal_offset_range: null
  goal_timestep_mode: zero
  val:
    goal_offset_range: null

denoiser:
  goal_dim: 21
  cond_goal_root_mask_prob: 0.1
  cond_goal_yaw_mask_prob: 0.3
  cond_goal_time_mask_prob: 0.0
  cond_goal_body_mask_prob: 0.3
```

The base denoiser defaults remain compatible with `goal_type=root`. V3 body
evaluation configs remain at 15 dimensions. `vis_dar_v4.yaml` provides offline
V4 dataset evaluation with all extended fields.

## Planner Deployment

New Sonic `history_state_textop_v2` messages use `protocol_version=5` for all
TextOp goal types. Every spatial goal on this wire is expressed in the shared
Z-up, +X-forward, +Y-left world frame:

```text
goal_root_pos_world       [3] float32 world XYZ buffer
goal_yaw_world            [1] float32 absolute world yaw buffer
goal_root_velocity_world  [3] float32 world XYZ/s buffer (body_ext)
goal_keypoints_world      [5, 3] body or [4, 3] body_ext world XYZ buffer
goal_timestamp_ns             absolute target timestamp (body_ext)
```

Keypoint order is fixed end to end:

```text
body     rows: pelvis, left_foot, right_foot, left_palm, right_palm
body     dims: 0:3,   3:6,       6:9,        9:12,      12:15
body_ext rows: left_foot, right_foot, left_palm, right_palm
body_ext dims: root 0:3, yaw 3:5, velocity 5:8, time 8:9,
               left_foot 9:12, right_foot 12:15,
               left_palm 15:18, right_palm 18:21
```

Sonic transports these rows without semantic reordering. Shape checks alone
cannot detect a hand-first producer, so senders must follow this order.

The decoder still accepts protocol v2-v4 headers and maps their legacy
`goal_root_pos`, `goal_heading`, `goal_root_velocity`, and `goal_keypoints`
fields into the explicit in-process names above. New senders must use only the
protocol-v5 names. `ego_occ` remains the intentional ego-frame exception; it is
an observation, not a goal.

For checkpoints trained with `goal_timestep_mode=relative`, TextOp computes the
remaining time from the latest controller state sample rather than from message
receipt time:

```text
goal_timestep = (goal_timestamp_ns - timestamps_ns[-1]) / 1e9
```

This keeps the countdown tied to controller time and also works when generated
history is aligned to the measured state. The planner transforms world root
position, root velocity, and the four limb points into its current reference
frame. Root yaw remains on the wire for compatibility but deployment sets
`force_drop_goal_yaw=true`, matching the trained yaw-drop path.

For the recommended zero-time checkpoint, deployment sets
`force_drop_goal_time=true`. The wire timestamp remains absolute for protocol
compatibility, but the denoiser receives zero in dimension 8. A controller
testing the same ablation should set `goal_timestamp_ns` equal to its latest
state timestamp; setting the absolute timestamp itself to integer zero would
instead encode a very large negative duration.

## Failure Investigation (2026-08-04)

The stationary closed-loop result is most consistent with an underfit V4
denoiser, not a 29-DOF representation or MVAE shape failure:

- The regenerated dataset reports native 29-DOF joints and 69-D features. A
  sampled audit found finite joint values, nonzero motion in every added wrist
  channel, and binary left/right contact labels with similar rates.
- The 29-DOF MVAE checkpoint has 69-D input/output layers, and its reported
  reconstruction/KL curves are comparable to the successful 23-DOF MVAE.
- The deployed V4 checkpoint used `goal_per_primitive=true`, as did the last
  successful 23-DOF checkpoint. The source default had incorrectly remained
  `false`; the training config now sets it explicitly.
- The main additional difficulty was random `goal_offset_range=[-63, 0]` plus
  relative time. This changes the condition from one terminal target to any
  intermediate target while still supervising the whole future sequence.
- The motion-goal simulator sampled position, keypoints, and velocity at the
  current motion frame while assigning a timestamp one second later. Those
  fields did not describe one common target state. Zero-time deployment removes
  this inconsistency for the ablation; a future relative-time experiment must
  sample all target fields at the same look-ahead frame.
- Dynamic simulator FK previously sent wrist-body origins while training used
  palm-center extensions. `occHIPC/utils/g1_fk.py` now applies the same
  left/right wrist-local hand offsets as the TextOp 29-DOF skeleton. This small
  spatial mismatch could degrade hand conditioning but was not large enough to
  explain a completely stationary root.
- Generation previously forwarded only `force_drop_goal_yaw`, although the
  denoiser exposed root, yaw, time, and body force-mask controls. All four are
  now passed through, allowing the zero-time deployment contract to be enforced.

A local 200-update CUDA stability run with the fixed terminal goal and zero
time reduced five-batch mean validation total loss from approximately `0.860`
at the first evaluation to `0.568` at step 200. Mean foot-contact loss ended at
approximately `0.00046` and did not show sustained growth. This only validates
the training path and early trend; compare the remote full run against the old
`~0.19` endpoint before accepting the new checkpoint for control.

### 29-DOF DAR collapse diagnosis (2026-08-05)

Offline checkpoint tests isolate the stationary result to DAR rather than the
wire protocol, preprocessing, or MVAE reconstruction:

- Dataset pelvis goals exactly equal the reconstructed ground-truth endpoint.
- The 29-DOF MVAE mean reconstruction preserves moving endpoints. On the same
  paired motions, its posterior scale, variance, and effective latent rank are
  close to the successful 23-DOF MVAE.
- The successful 23-DOF DAR produces about `0.72 m` displacement for moving
  goals averaging `0.77 m`. The 29-DOF DAR produces only about `0.12 m`.
- At the noisiest diffusion step, teacher-forced latent MSE is approximately
  `0.49` for the successful 23-DOF DAR and `1.21` for the failed 29-DOF DAR.
  The sampled 29-DOF latent standard deviation is also substantially smaller,
  which is the direct signature of conditional-mean/standing collapse.
- Neutralizing the six wrist channels changes the 29-DOF latent by about
  `0.14` MSE while retaining mean cosine similarity `0.94`; wrist variation is
  present but does not dominate the latent.
- The weighted foot-contact/sliding gradient is thousands of times smaller
  than the reconstruction and latent gradients. Its rising curve is a symptom
  of bad generated motion, not the initiating optimization failure.

The failed run entered generated-history rollout after `25,000` optimizer
updates. Its DDP launcher divides stage lengths by world size while retaining a
full batch on every rank. This preserves the approximate number of processed
samples, but it is not optimization-equivalent: AdamW performs fewer updates at
the same learning rate with a larger global batch. On the other hand, the
successful 23-DOF run already showed goal response after about `2,500` updates,
so there is not enough evidence to call stage scaling the primary cause. Keep
the stage-0 checkpoint and event file to test how early the sampler collapses.
The available 29-DOF `ckpt_5000` is already collapsed, well before rollout
starts at update `25,000`; rollout therefore did not initiate this failure,
although it can still amplify it after the stage boundary.

Additional paired ablations found no material 23/29 difference in MVAE temporal
latent continuity or in the decoder's response to shrinking latent magnitude.
The optimizer state also confirms that all nominal 29-DOF updates ran; none
were skipped by the NaN/Inf guard. Wrist delta features have somewhat heavier
normalized tails, but no raw `2*pi` discontinuities, and zeroing wrist history
does not improve endpoint error over a 256-sample evaluation.

The failed DAR is not completely goal-blind. For moving validation goals, its
generated displacement has mean direction cosine about `0.57`, but only about
`0.14 m` magnitude for goals averaging `0.82 m`. The remaining evidence is most
consistent with conditional-variance collapse in the DAR objective: the
latent/per-frame reconstruction losses permit a low-amplitude conditional mean
after the VAE target changes, and there was no direct endpoint-magnitude loss.

Use the following recovery settings:

```yaml
train:
  manager:
    # Keep rollout off until full-sample endpoint metrics show useful motion.
    use_rollout: false
    eval_full_sample: true
    loss_weight:
      foot_contact: 0.0
      goal_direction: 0.01
      goal_position: 0.5
```

`goal_position` is horizontal Huber loss between integrated generated root
displacement and goal dimensions `0:2`. It applies to `root`, `body`, and
`body_ext`; root-condition masking also masks this loss. It supplies magnitude
supervision that the cosine-only `goal_direction` loss cannot provide.
FeatureVersion 3 stores a forward delta on each pose feature. Therefore the
reference-to-goal integration uses the last history-frame delta followed by
future deltas `0:-1`; summing all future deltas is shifted one frame forward.

When merging this change into `ddp-training`, remove its root-only
`goal_direction` validation/zeroing. A 15-D `body` goal starts with pelvis XYZ,
so dimensions `0:2` have the same root-position semantics used by both endpoint
losses. The saved failed body run used `goal_direction: 0.0`; enabling the new
weights requires the DDP branch to retain this body-goal handling.

Full-DDPM validation now reports `sample_goal_position`,
`sample_goal_direction`, `sample_root_displacement`,
`goal_root_displacement`, and `sample_latent_std`.
Do not enable rollout based only on teacher-forced `eval_total`: the stage-0
checkpoint must first show moving-goal displacement and non-collapsed sampled
latent variance. Keep checkpoints around the stage boundary for this check.
Because `goal_position` is now part of the weighted total, its new
`eval_total` is not numerically comparable to checkpoints trained with that
weight absent; compare the individual reconstruction and latent terms as well
as the full-sample metrics.

For a controlled DDP diagnosis, compare equal-global-batch and large-global-
batch runs rather than changing samples and optimizer updates together. With
eight GPUs, `batch_size=32` per rank gives global batch `256`; retain the
unscaled update schedule for that experiment. Separately test the existing
`batch_size=256` per rank/global `2048` setup with its current scaled schedule.
Run both without rollout until their stage-0 full-sample metrics are comparable.
This distinguishes optimization scaling from the endpoint-loss issue.
This comparison is optional: the existing division by world size is a valid
sample-count convention and should remain the default for the current remote
run. It is not currently the leading explanation for the 23/29 regression.

The full-sample generation helper also now passes `initial_noise` into DDPM or
DDIM sampling. Previously the fixed planner noise was constructed but ignored
by the full-sample path; that caused unnecessary plan-to-plan randomness but
did not cause the stationary checkpoint.

`planner_dar.yaml` selects `body_ext`, `goal_dim=21`, and validates that the DAR
checkpoint config uses 69-D FeatureVersion 3 data and `g1_29dof` before model
construction. `run_planner_dar.sh` accepts `DAR_CKPT` and `DATADIR` overrides.

## Files

Modified for V4 training:

- `robotmdar/utils/goal.py`
- `robotmdar/model/mld_denoiser.py`
- `robotmdar/dataloader/data.py`
- `robotmdar/train/train_dar.py`
- `robotmdar/eval/vis_dar.py`
- `robotmdar/eval/generate_dar.py`
- `robotmdar/planner/planner_dar.py`
- `robotmdar/utils/planner_convert.py`
- `robotmdar/config/data/mob.yaml`
- `robotmdar/config/denoiser/def.yaml`
- `robotmdar/config/train_dar.yaml`
- `robotmdar/config/skeleton/g1.yaml`
- `robotmdar/config/vis_dar_v4.yaml`
- `robotmdar/config/planner_dar.yaml`
- `scripts/run_planner_dar.sh`
- `sonic-msg/sonicmsg/messages.py`
- `sonic-msg/sonicmsg/planner_node.py`

## Compatibility and Rollback

The MVAE is unchanged and can be reused. V4 changes the denoiser goal projection
from its V3 15-D body input to a 21-D `body_ext` input, so the DAR denoiser must
be retrained.

V3 root and body checkpoints remain supported by the conversion and Sonic
protocol code. To deploy one, override the planner goal type/dimension and use
a checkpoint with a matching config; the 29-DOF checkpoint validator still
rejects feature or skeleton mismatches.

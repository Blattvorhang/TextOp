# Planner V8: Gait Trackability and Fall-Recovery Support Consistency

> Status: discussion draft, diagnostics/support consistency implemented;
> root-position goal loss split and locomotion/getup loss-weight sets added.
>
> Builds on `planner_v7_rotation_matrix.md` and
> `planner_v7_1_fall_recovery.md`. V7 fixes the orientation representation and
> the history perturbation contract. V8 focuses on a different failure mode:
> the planner can satisfy goal-conditioned root states while producing motions
> whose support mechanics are hard for SONIC to track.

---

## 1. Summary

The current heading-free planner can already learn accurate goal reaching.
TensorBoard curves from `logs/pretrained/0903_heading_free` show that most
goal losses continue improving through 50k steps, and the 8GPU run is slightly
ahead of the parallel 4GPU run at matched optimizer steps. However, SONIC
inspection reveals that the better scalar goal metrics do not necessarily mean
better executable gait:

1. `ckpt_20000.pth` has good gait and goal-reaching behavior in SONIC.
2. Later checkpoints, especially near 50k, can reach the target more by root
   displacement than by leg-supported locomotion. Sliding becomes more visible
   even while `loss/eval/foot_contact` decreases.
3. Fall-recovery references are often too aggressive for the tracker: the root
   quickly rises, but the body shows little obvious support action. In some
   failures, the reference resembles a handstand-like recovery that is outside
   the intended data manifold.
4. The goal state can appear too early in the generated primitive. The planner
   may start with the goal's joint pose or root velocity immediately, instead
   of transitioning toward that pose/velocity as the arrival time approaches.
   When the goal is behind the robot, the planner can prefer walking backward
   toward the target instead of turning, walking forward, and then aligning to
   the final orientation.

The main V8 hypothesis is:

```text
The current objective is strong enough for goal-state matching, but not strong
enough to bind root displacement to physically plausible support kinematics.
```

Therefore, the next step should not be blind loss-weight tuning. First add
metrics that expose root/support decoupling. If those metrics confirm the
hypothesis, add a support-consistency loss with a small scheduled weight and
test modest goal-weight reductions as ablations.

---

## 2. Observed Phenomena

### 2.1 8GPU versus 4GPU training

The 8GPU run is:

```text
logs/pretrained/0903_heading_free
```

The 4GPU comparison run is:

```text
logs/pretrained/0903_heading_free_4GPU
```

Both runs keep the same configured `steps` and `data.batch_size`. Under the
current DDP setup, this means the per-rank batch is unchanged, so the 8GPU run
uses a larger global batch than the 4GPU run:

```text
4GPU global batch ~= 4 * batch_size
8GPU global batch ~= 8 * batch_size
```

At the same optimizer step, the 8GPU run has consumed about twice as many
training samples. This is not a pure throughput comparison. It changes the
optimization regime:

1. Gradient noise is lower on 8GPU.
2. Each optimizer step is estimated from more samples.
3. The same learning rate is applied to a larger effective batch.
4. If checkpoints are compared by step count, the 8GPU checkpoint has seen
   more data than the 4GPU checkpoint.

From the available TensorBoard curves, 8GPU is slightly better than 4GPU at
matched optimizer steps, but the difference is not the root cause of the gait
regression. The 4GPU run shows similar tendencies and is not clearly solving
the support/trackability problem.

For a strict GPU-count ablation, keep global batch fixed:

```text
Option A: 8GPU batch_size = half of 4GPU batch_size
Option B: 4GPU uses gradient accumulation to match the 8GPU global batch
```

Compare both by optimizer step and by total number of samples seen. Otherwise,
"same steps and same batch_size" mainly answers how larger global batch affects
training, not how GPU count alone affects training.

### 2.2 Goal metrics improve, gait does not necessarily improve

In the 8GPU run, the validation total loss and goal losses keep improving into
late training. Representative trends:

```text
loss/eval/total              improves until around 28k-50k
loss/eval/rec                improves until around 28k
goal_root_position           improves into late training
goal_root_orientation        improves into late training
goal_joint_angle             improves into late training
goal_root_velocity           improves into late training
metric/eval/sample_goal_error_m improves into late training
```

This agrees with the planner becoming better at matching the conditioned goal.
But it conflicts with SONIC observations:

```text
20k: good gait and goal-reaching, tracker-friendly
50k: goal reaching can remain good, but gait quality is worse and sliding is
     more visible
```

This means current scalar selection criteria are incomplete. A checkpoint can
look better in TensorBoard while being worse for closed-loop control.

### 2.3 Foot-contact loss is not enough

`loss/eval/foot_contact` decreases in later checkpoints, yet SONIC shows more
visible sliding. This can happen because the current foot-contact term is a
stance-foot velocity penalty under the paired denoising/eval path. It is not a
complete generated-reference sliding metric for SONIC rollouts.

Also, `metric/eval/sliding_ratio` should not be treated as generated sliding.
In the current loss path it is derived from the dataset-side sliding mask, so
it mostly describes which validation frames were sampled rather than whether
the predicted reference actually slides.

The practical consequence:

```text
Do not pick checkpoints only by foot_contact or sliding_ratio.
Do not assume lowering foot_contact solves executable gait.
```

### 2.4 Fall recovery is too fast and too root-driven

The recovery issue is not that the planner lacks diverse strategies. For this
project, recovery should intentionally be a funnel:

```text
arbitrary fallen history
  -> canonical flat or extended fallen state
  -> one SONIC-trackable get-up family
  -> upright locomotion
```

The current bad recovery references suggest the planner sometimes learns a
shortcut:

```text
"raise the root height and match the final upright-ish goal"
```

without producing the intermediate hand/foot support actions that make the
motion trackable. This is especially harmful because the tracker must follow a
physically plausible reference, not just a kinematic root trajectory.

### 2.5 Goal pose and velocity can appear too early

Another observed issue is that the conditioned arrival state can leak into the
beginning of the whole generated primitive:

```text
The model starts with the goal-like joint pose or goal-like velocity,
instead of becoming goal-like near the requested arrival time.
```

This is especially visible when the target is behind the robot. A direct
short-horizon solution is:

```text
walk backward to the target
```

but the more natural navigation behavior is usually:

```text
turn toward the target -> walk forward -> align to the final pose/orientation
```

The project does not require a single prescribed path to the goal. However,
"any path is allowed" should not mean that the arrival pose and arrival
velocity become a style command for the entire rollout.

This is a temporal-conditioning problem, separate from but related to the
support-consistency problem:

```text
Support consistency asks: is root motion physically explained?
Temporal goal realization asks: is the arrival state realized at the right
time instead of immediately?
```

---

## 3. Current Problem Statement

### 3.1 The representation permits a root-motion shortcut

Feature V6/V7 represents root displacement as an explicit motion component.
This is necessary for planning. However, it also gives the model an easy path:

```text
move the root toward the goal
while letting limbs fail to explain or support that motion
```

The geometry losses compare reconstructed kinematic states to ground truth, but
they do not explicitly require the root displacement between two frames to be
consistent with the feet or hands that are supposed to be stationary contacts.

This creates a gap between:

```text
goal-correct kinematic reference
```

and:

```text
tracker-executable supported motion
```

### 3.2 Current losses fit goal reaching better than gait trackability

The current `train_dar.yaml` loss weights are approximately:

```text
rec:                   1.0
body_trans:            0.05
body_rot:              0.01
dof_pos:               0.03
dof_vel:               0.003
foot_contact:          0.05
rot_chord:             0.02
g_cons:                0.05
h_vel:                 0.001
goal:
  g:                   0.0
  root_position_hor:   1.0
  root_position_vert:  1.0
  root_orientation:    0.2
  root_velocity:       0.1
  joint_angle:         0.1
```

The manager loss weights are now configured as two sections:

```text
train.manager.loss_weight.locomotion.*
train.manager.loss_weight.getup.*
```

`is_recovery=True` selects the `getup` section for that sample; all other
samples use `locomotion`. The default values are intentionally copied between
the two sections until a recovery-specific setting is chosen.

These weights are reasonable for reaching the V7 goal contract. They are not
yet sufficient for enforcing support-based locomotion or recovery.

Current interpretation:

1. `goal.root_position_hor = 1.0` is effective for navigation, but can reward
   horizontal root-displacement shortcuts when support consistency is missing.
2. `goal.root_position_vert = 1.0` directly supervises target root height. For
   recovery this may need a different `getup` weight than locomotion, because
   overemphasizing height can encourage "root rises first" references.
3. `goal.root_velocity = 0.1` helps time-to-arrival and urgency, but can also
   reinforce root motion if the support mechanics are underconstrained.
4. `goal.g` is an additional endpoint tilt supervision term on the target
   gravity direction. It can stay `0.0` for locomotion and become the main
   recovery uprightness goal while `goal.root_orientation` is disabled for
   getup, because the full orientation term also carries yaw/heading.
5. `foot_contact = 0.05` is useful, but it supervises only a narrow stance-foot
   velocity behavior and does not cover generated SONIC references enough.
6. `g_cons = 0.05` regularizes rotation/gravity consistency. Its oscillation is
   not the main issue here; increasing it is unlikely to fix gait sliding.
7. `dof_vel = 0.003` and `h_vel = 0.001` help smoothness, but do not directly
   bind root translation to contact kinematics.

Therefore, the current loss weights are not "wrong" for the present model, but
they are incomplete. The next experiment should add the missing constraint or
diagnostic before heavy retuning.

### 3.3 Arrival goals are global conditions, not per-frame schedules

The current goal losses select the requested arrival frame and compare the
prediction there:

```text
goal.root_position_hor  -> horizontal displacement at goal_time_frame
goal.root_position_vert -> root height at goal_time_frame
goal.g                  -> target gravity direction at goal_time_frame
goal.root_orientation   -> state at goal_time_frame
goal.joint_angle        -> state at goal_time_frame
goal.root_velocity      -> state at goal_time_frame
```

This is the right endpoint contract. The ambiguity is not in the selected-frame
loss itself. The ambiguity is that the goal tokens condition the whole future
sequence. Arrival time is embedded into the goal tokens, but each generated
future frame does not receive an equally explicit "how far am I from arrival?"
conditioning signal.

As a result, the network can use:

```text
goal joint pose
goal root velocity
goal orientation
```

as a broad motion style for all frames. This explains why goal-like action or
velocity can appear immediately, even when the goal time is later.

The behind-goal case adds a policy ambiguity. A target behind the robot is
mathematically reachable by negative local forward velocity. Without an
explicit locomotion prior or staged navigation policy, walking backward can be
an acceptable low-loss solution even if it is not the behavior we want.

---

## 4. Proposed V8 Direction

### 4.1 Add generated-motion diagnostics first

Before changing training, add metrics that compare ground truth, 20k samples,
and 50k samples:

```text
GT support residual
20k generated support residual
50k generated support residual
```

If the hypothesis is right, the expected result is:

```text
GT << 20k < 50k
```

or at least:

```text
50k worse than 20k on support residual and generated sliding
```

Required diagnostics:

1. `e_support_foot`: residual between root displacement and stance-foot
   kinematics.
2. `e_support_hand`: same residual for hand contacts during recovery.
3. `pred_foot_sliding`: generated foot velocity while the foot is inferred or
   labeled as contact.
4. `pred_hand_sliding`: generated hand velocity while the hand is inferred or
   labeled as support.
5. `max_abs_hdot`: maximum root-height velocity over the generated sample.
6. `max_tilt_rate`: maximum root orientation change rate.
7. `recovery_duration`: frames needed to move from fallen to stable upright.
8. `root_support_ratio`: how much root displacement is explained by support
   kinematics versus pure root-channel motion.

These metrics should be logged for:

```text
paired denoising eval
full-sample eval
per-class splits: walk, run, fall, getup, unknown
checkpoint comparison: 20k, 50k
```

The full-sample path matters most because SONIC consumes generated references,
not teacher-forced denoising pairs.

### 4.2 Support-consistency residual

Use the V7 convention:

```text
R_t = world-from-ego root rotation at frame t
A_t = R_{t-1}^T R_t
```

Let `p_t` be the root world position, and let contact point `i` have local FK
offset:

```text
r_{i,t} = position of contact point i in ego frame E_t
```

If contact point `i` is stationary in the world between `t-1` and `t`, then:

```text
p_t + R_t r_{i,t} ~= p_{t-1} + R_{t-1} r_{i,t-1}
```

Multiplying by `R_{t-1}^T` gives:

```text
d_t ~= r_{i,t-1} - A_t r_{i,t}
```

where:

```text
d_t = R_{t-1}^T (p_t - p_{t-1})
d_kin_{i,t} = r_{i,t-1} - A_t r_{i,t}
```

The residual is:

```text
e_support_{i,t} = d_t - d_kin_{i,t}
```

and the loss is:

```text
L_support =
  sum_{t,i} w_{i,t} * Huber(e_support_{i,t}, 0)
  / (sum_{t,i} w_{i,t} + eps)
```

with:

```text
w_{i,t} = contact_{i,t-1} * contact_{i,t}
```

For locomotion, `i` should include left and right feet. For recovery, `i`
should include both feet and both hands.

For Feature V6/V7, use full 3D displacement, not only horizontal displacement.
If the feature path reconstructs from local horizontal displacement and root
height, the local full displacement at the departure frame is:

```text
d_t = delta_p_hor_t - (h_t - h_{t-1}) * g_{t-1}
```

where `g_{t-1}` is gravity expressed in frame `E_{t-1}`.

### 4.3 Contact selection and anti-gaming rules

The support loss should not let the model escape by predicting "no contact".
Use conservative contact sources:

1. For paired reconstruction/denoising loss, start with ground-truth contact
   masks and ground-truth sliding filters.
2. For generated full-sample metrics, infer support from FK height and contact
   point velocity, and report predicted-contact variants separately.
3. For hands in recovery, infer support from hand height plus low hand velocity,
   because the dataset may not carry explicit hand contact labels.
4. Do not apply `L_support` during flight, stepping transitions, or known
   sliding frames.

A conservative initial gate:

```text
foot support:
  contact_mask[t-1] == 1 and contact_mask[t] == 1
  and not sliding_mask[t-1]
  and not sliding_mask[t]

hand support:
  hand_height below threshold
  and hand_world_speed below threshold
  at both t-1 and t
```

The thresholds should be logged and validated on ground truth before they are
used for training.

### 4.4 Recovery funnel instead of recovery diversity

For recovery, the target behavior should be concentrated rather than diverse:

```text
fallen states -> canonical flat/extended recovery manifold -> trackable get-up
```

This suggests three changes:

1. Recovery evaluation should prefer a small number of SONIC-trackable get-up
   families over arbitrary generated recovery styles.
2. Recovery goals should not encourage the planner to teleport from lying to
   upright by root height. Penalize excessive `h_dot` and support residual
   during get-up.
3. If the reference gets upright faster than SONIC can track, increase recovery
   time budget or use staged recovery subgoals.

Possible staged recovery contract:

```text
Stage A: stabilize from arbitrary fallen history to canonical fallen posture
Stage B: move through hand/foot-supported transition
Stage C: reach stable upright pose with low residual velocity
Stage D: resume normal goal-conditioned walking
```

This is a planner-side reference-quality constraint. It is not a request for
motion-style diversity.

### 4.5 Add temporal goal-realization diagnostics

Add diagnostics that measure when the generated motion becomes goal-like. Let
`t_g` be the selected arrival frame. For each generated sample, compute:

```text
q_goal = joint pose at t_g from the goal condition
v_goal = root velocity at t_g from the goal condition
R_goal = root orientation at t_g from the goal condition
```

Then log:

```text
pose_goal_argmin_frame:
  argmin_t ||q_pred[t] - q_goal||

velocity_goal_argmin_frame:
  argmin_t ||v_pred[t] - v_goal||

orientation_goal_argmin_frame:
  argmin_t geodesic(R_pred[t], R_goal)

early_pose_goal_ratio:
  fraction of samples where pose_goal_argmin_frame < t_g - margin

early_velocity_goal_ratio:
  fraction of samples where velocity_goal_argmin_frame < t_g - margin
```

Also compare the predicted temporal profile to the ground-truth profile from
the same validation window:

```text
D_q_pred[t] = ||q_pred[t] - q_goal||
D_q_gt[t]   = ||q_gt[t]   - q_goal||

D_v_pred[t] = ||v_pred[t] - v_goal||
D_v_gt[t]   = ||v_gt[t]   - v_goal||
```

If `D_q_pred` or `D_v_pred` collapses too early relative to `D_gt`, the model
is treating the arrival state as an immediate style condition.

For the behind-goal case, add:

```text
backward_walk_ratio:
  fraction of locomotion frames with dot(v_world, root_forward_world) < -v_thr

behind_goal_backward_ratio:
  same ratio, restricted to samples with goal x in the current ego frame < -x_thr
```

This metric should be label-aware. Backward walking should be allowed when the
text/action explicitly describes backward motion or a short corrective step,
but it should not dominate default navigation goals behind the robot.

### 4.6 Add per-frame arrival-phase conditioning

The model should know not only:

```text
the goal is T frames away
```

but also, for every generated frame:

```text
this frame is T - t frames before arrival
```

Add a per-frame arrival-phase embedding to the future/noise tokens:

```text
remaining_frames[t] = max(t_g - (t + 1), 0)
phase[t] = clamp((t + 1) / max(t_g, 1), 0, 1)
```

The embedding can be:

```text
arrival_phase_embed = MLP([remaining_frames, phase])
```

and added to each future/noise token before the transformer or MLP denoiser
predicts that frame. This makes the semantics explicit:

```text
early frames: transition toward the goal
near-arrival frames: realize the goal pose and velocity
after-arrival frames: maintain or continue from the achieved state
```

This should be tested before adding strong hand-written timing penalties,
because it improves conditioning without prescribing a single path.

### 4.7 Optional temporal profile loss

If phase conditioning is not enough, add a weak temporal profile loss using
the ground-truth window as the schedule. This does not force a handcrafted
trajectory; it only asks the predicted motion to become goal-like at a similar
time as the training example.

For joint pose:

```text
L_goal_profile_q =
  mean_t Huber(norm(D_q_pred[t]) - norm(D_q_gt[t]), 0)
```

For root velocity:

```text
L_goal_profile_v =
  mean_t Huber(norm(D_v_pred[t]) - norm(D_v_gt[t]), 0)
```

Apply only when the initial state and goal state differ enough:

```text
||q_current - q_goal|| > q_min
||v_current - v_goal|| > v_min
```

Otherwise, a static or already-goal-like sample would be incorrectly punished.
Keep the weight small and log the diagnostic first.

### 4.8 Behind-goal navigation policy

For deployment, a direct goal behind the robot should usually be decomposed
into staged subgoals unless backward motion is explicitly requested:

```text
if goal is behind and distance is non-trivial:
  Stage A: turn in place or with a small arc until the target is in front
  Stage B: walk forward toward the target
  Stage C: align to the requested final orientation and pose
```

This can be implemented as a planner-controller policy layer without changing
the motion representation. It is also safer than trying to make a single
64-frame primitive learn every possible turn-and-go strategy from one global
goal token.

For training, add only a weak locomotion prior:

```text
L_forward_pref = mean ReLU(-dot(v_world, root_forward_world) - v_tol)
```

and apply it only to default locomotion/navigation samples where backward
walking is not the intended action. This should be treated as a policy prior,
not a universal physical law.

---

## 5. Loss-Weight Strategy

### 5.1 Do not first increase `g_cons`

`g_cons` measures internal consistency between gravity orientation channels and
relative rotation. If it is small but oscillatory, that does not explain root
translation sliding by itself. Increasing `g_cons` may make rotations smoother,
but it does not directly bind root displacement to stance feet or hands.

Recommendation:

```text
Keep g_cons at 0.05 for the next diagnostic run.
```

### 5.2 Do not rely on `foot_contact` alone

Increasing `foot_contact` from `0.05` to `0.1` is a reasonable control
experiment, but it should not be the main fix. It can reduce stance-foot
velocity under the current supervised mask while still failing to prevent
root-motion shortcuts in full sampled references.

Recommendation:

```text
Run foot_contact=0.1 only as a control ablation.
```

### 5.3 Goal weights are effective but may be too dominant late

The current goal weights successfully train goal-reaching. The problem is that
late checkpoints may improve goal metrics by exploiting unsupported root
motion. If support diagnostics confirm this, test weaker root-goal pressure:

```text
goal.root_position_hor: 1.0 -> 0.5
goal.root_position_vert: tune separately for locomotion/getup
goal.root_velocity: 0.1 -> 0.05
```

Keep these as ablations, not the first change. If root-goal weights are reduced
before adding support metrics, the model may simply lose goal-reaching ability
without exposing why.

### 5.4 Add `support_consistency` with a small scheduled weight

If the diagnostics confirm root/support decoupling, add:

```text
support_consistency: 0.02, 0.05, or 0.1
```

Suggested schedule:

```text
0 to 10k steps:     support_consistency = 0
10k to 20k steps:   linear ramp to target weight
20k onward:         fixed target weight
```

The warm-up avoids fighting early reconstruction learning. The exact schedule
can follow the current manager step count.

### 5.5 Do not solve early-goal behavior by only lowering goal weights

Lowering `goal.joint_angle` or `goal.root_velocity` may reduce early goal-like
motion, but it also weakens the arrival contract. The first fix should be
temporal: better diagnostics, per-frame arrival-phase conditioning, and
possibly a weak temporal profile loss.

Recommended order:

```text
1. Add early-goal timing metrics.
2. Add per-frame arrival-phase conditioning.
3. Only then test smaller goal velocity or joint-goal weights if needed.
```

---

## 6. Proposed Experiment Plan

### 6.1 Immediate checkpoint decision

Use the 20k checkpoint as the current SONIC baseline:

```text
logs/pretrained/0903_heading_free/ckpt_20000.pth
```

Do not treat the 50k checkpoint as better only because `loss/eval/total` or
goal losses are lower.

### 6.2 Diagnostic-only pass

Implement and run metrics without changing training:

```text
D0: GT validation windows
D1: 8GPU ckpt_20000 full samples
D2: 8GPU ckpt_50000 full samples
D3: latest 4GPU full samples when available
```

Acceptance signal:

```text
20k should have lower generated support residual and lower generated sliding
than 50k if the SONIC observation is caused by root/support decoupling.
```

### 6.3 Training ablations after diagnostics

Run small ablations, changing one axis at a time:

```text
A0: baseline weights, choose by new support metrics
A1: foot_contact 0.05 -> 0.1
A2: add support_consistency=0.02
A3: add support_consistency=0.05
A4: add support_consistency=0.05 and reduce root goal weights
    goal.root_position_hor 1.0 -> 0.5
    goal.root_position_vert separately in locomotion/getup
    goal.root_velocity 0.1 -> 0.05
A5: recovery-only h_dot or staged-subgoal constraint
A6: add per-frame arrival-phase conditioning
A7: add weak temporal goal-profile loss
A8: add behind-goal staged navigation policy at deployment
```

Recommended first implementation experiment:

```text
A2 or A3 for support, and A6 for temporal goal realization.
```

Goal weights are already doing useful work. The first real fixes should add the
missing support constraint and the missing per-frame timing signal instead of
weakening the successful goal channel.

### 6.4 Checkpoint selection criteria

A checkpoint should be considered better only if it improves or preserves:

```text
sample_goal_error_m
sample_endpoint_error_m
generated pred_foot_sliding
generated e_support_foot
generated e_support_hand on recovery
max_abs_hdot on recovery
early_pose_goal_ratio
early_velocity_goal_ratio
behind_goal_backward_ratio
SONIC tracking success
```

`loss/eval/total` remains useful, but it should no longer be the primary
selection metric after goal-reaching has become good enough.

---

## 7. Implementation Notes

### 7.1 Where to implement diagnostics

The support residual needs reconstructed FK outputs:

```text
root world position p_t
root rotation R_t
foot world positions
hand world positions
contact_mask
sliding_mask, if available
```

The natural implementation location is the V6 geometry loss/eval path in:

```text
robotmdar/train/loss.py
```

Full-sample metrics should be logged in the eval sampling path in:

```text
robotmdar/train/train_dar.py
```

The first patch should add metrics only. The second patch should add the
optional weighted loss term after the diagnostic values are validated.

### 7.2 Naming

Suggested scalar names:

```text
metric/eval/e_support_foot
metric/eval/e_support_hand
metric/eval/pred_foot_sliding
metric/eval/pred_hand_sliding
metric/eval/max_abs_hdot
metric/eval/max_tilt_rate
metric/eval/root_support_ratio
metric/eval/early_pose_goal_ratio
metric/eval/early_velocity_goal_ratio
metric/eval/backward_walk_ratio
metric/eval/behind_goal_backward_ratio

loss/train/support_consistency
loss/eval/support_consistency
loss/train/goal_profile_q
loss/train/goal_profile_v
loss/eval/goal_profile_q
loss/eval/goal_profile_v
```

For class splits, follow the existing suffix convention:

```text
metric/eval/e_support_foot__walk
metric/eval/e_support_foot__getup
```

### 7.3 Acceptance tests

Add unit tests for:

1. A stationary planted foot with root translation explained by joint/root
   rotation should give near-zero support residual.
2. Pure root translation with unchanged stance foot local offset should give a
   large support residual.
3. Hand-support residual should be zero for a synthetic planted hand.
4. The support loss should be zero when no support frames are active.
5. Full-sample metrics should log deterministic keys even when a class is
   absent from the batch.
6. A sequence whose joint pose matches the goal only near `t_g` should have
   lower early-goal ratios than one that matches the goal from frame 0.
7. A forward walk should have near-zero backward-walk ratio; a synthetic
   backward walk should have a high backward-walk ratio.

---

## 8. Current Recommendation

For the current project state:

1. Keep training logs for both 8GPU and 4GPU, but do not expect GPU count alone
   to fix gait. The current comparison mainly tests larger global batch.
2. Use the 20k 8GPU checkpoint as the best SONIC baseline until new metrics say
   otherwise.
3. Keep current loss weights for the next diagnostic run.
4. Implement support/sliding/recovery diagnostics before changing code behavior.
5. If diagnostics confirm that 50k has worse support residual than 20k, add
   `support_consistency` with a small scheduled weight.
6. Add temporal goal-realization metrics and per-frame arrival-phase
   conditioning to prevent the arrival pose/velocity from becoming an
   immediate whole-rollout style.
7. For behind-goal navigation, prefer staged turn-and-walk goals unless
   backward movement is explicitly requested.
8. Only after support consistency and temporal conditioning exist, test lower
   `goal.root_position_hor`/`goal.root_velocity` weights and a separate
   `goal.root_position_vert` value for getup.

The core design principle for V8:

```text
The planner may choose where the root should go, but supported contacts must
explain how the root got there.
```

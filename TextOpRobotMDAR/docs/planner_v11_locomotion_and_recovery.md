# Planner V11: Locomotion and Fall-Recovery Preservation

> Status: proposed recovery-focused experiment plan.
>
> This document records the next implementation direction after comparing
> `0905_goal_vert_token`, `0909_text`, the corresponding TensorBoard curves,
> and SONIC closed-loop behavior. It is deliberately narrower than the
> earlier recovery proposals: the first goal is to recover get-up behavior
> without sacrificing the strong locomotion behavior of the current model.

## 1. Executive Decision

The immediate problem is not that the model cannot represent locomotion. The
`0909_text` result shows the opposite: adding end-effector goals and the new
goal/loss arrangement substantially improved normal goal-reaching gait. The
regression is concentrated in fall recovery:

- the model produces displacement and a visible upright-orientation trend;
- it does not produce a sufficiently executable get-up action;
- it pays too much attention to horizontal goal placement during recovery;
- the recovery behavior is worse than the strong `0905_goal_vert_token`
  25k checkpoint, even though the current model has better normal locomotion.

The first V11 change should therefore be:

1. audit the recovery metadata and text-conditioning data path;
2. add a recovery-specific condition mask;
3. use a separate, conservative get-up loss configuration;
4. keep the current rollout schedule unchanged;
5. evaluate recovery in SONIC, while retaining normal locomotion as a
   regression gate.

The first experiment must not combine every possible change. In particular,
it must not simultaneously alter recovery labeling, rollout, offline
augmentation, goal construction, and loss weights. Otherwise a recovery
improvement or regression will not be attributable.

## 2. Evidence and Interpretation

### 2.1 Normal locomotion is not the primary failure

The `0909_text` model has the best normal locomotion behavior observed so far:

- goal-reaching precision is substantially better than the earlier runs;
- the gait is more stable and less obviously compromised by the goal;
- end-effector supervision did not destroy normal spatial control.

Overshoot after reaching the goal is intentionally out of scope here. It is
not automatically a regression: stopping at an exact target is
underdetermined when the input does not specify terminal velocity, braking
behavior, or a dwell policy. The current V11 objective is to restore
recovery, not to remove this locomotion behavior.

### 2.2 Recovery is a different control problem

During recovery, the desired behavior is not accurate horizontal endpoint
placement. The important sequence is:

1. recognize the fallen/support-constrained state;
2. change gravity alignment and body configuration;
3. create executable hand/foot support and push actions;
4. raise the body through a tracker-followable trajectory;
5. only then resume ordinary locomotion.

A model can reduce a root-position or gravity loss by changing the root
trajectory without generating enough joint-level support action. This matches
the observed pattern: the body shows an upright tendency and displacement,
but the tracker does not see a useful get-up motion.

### 2.3 Why the previous GPT proposal must be narrowed

Some parts of the previous proposal are useful, but they should not all be
implemented at once:

| Proposal | V11 decision | Reason |
|---|---|---|
| Treat every primitive from a recovery clip as get-up | Keep as-is for now | This is true in the current clip-level labeling, but the user inspected the data and the current get-up loss is not strongly harmed by the tail windows. |
| Restore or expand recovery-specific online augmentation | Defer | `augment_fall_recovery.py` is an offline pipeline and does not cover the tail-window issue. It would add another confound to the first rescue experiment. |
| Disable rollout for recovery immediately | Defer | The idea of task-dependent rollout is worth keeping, but closing rollout now would change the training distribution and make the causal diagnosis less clean. |
| Add recovery-specific condition masking | Adopt | This directly addresses the conflict between locomotion endpoint control and recovery support behavior. |
| Split position masking into horizontal and vertical parts | Adopt | Recovery should not receive the same horizontal endpoint pressure as locomotion, while height remains useful. |
| Audit filename and text metadata matching | Highest priority | A broken recovery identifier or mismatched text label can make all downstream loss/mask reasoning invalid. |

## 3. Highest-Priority Check: Recovery Metadata and Text Matching

### 3.1 Current data path

The current implementation has two related but separate metadata paths:

```text
source filename / source key
      |
      v
_is_flat_recovery(fine_name)
      |
      v
record["_recovery_boost"]
      |
      v
primitive["is_recovery"]
      |
      v
locomotion/getup loss routing
```

```text
metadata lookup + frame_ann
      |
      v
_primitive_action_label(...)
      |
      v
primitive["action_label"]
      |
      v
primitive["text_embedding"]
      |
      v
text-conditioned denoiser
```

In the current code:

- `dataset/data_process/pack_motion_lib_to_textop.py` derives
  `_recovery_boost` from the source name using `_is_flat_recovery()`;
- `TextOpRobotMDAR/robotmdar/dataloader/data.py` copies that sequence-level
  flag to every primitive as `is_recovery`;
- the same dataloader selects a future-overlapping text/action label from
  `frame_ann`;
- `train_dar.py` passes both `action_label` and `is_recovery` into the loss.

Therefore, adding a text embedding in the denoiser should not directly flip
  `is_recovery`. However, the packing and metadata integration can still
  fail in several ways:

1. the source stem no longer contains `stand_up_lying` or
   `faint_stand_up_lying`;
2. the source name is transformed before `_is_flat_recovery()` sees it;
3. an augmented filename loses the original recovery stem;
4. metadata lookup misses the intended row and attaches an unrelated
   description;
5. `frame_ann` becomes empty or is assigned to the wrong motion;
6. the recovery sequence remains tagged as recovery but receives a misleading
   text label, or receives an empty text embedding.

The important conclusion is:

> The text condition cannot be assumed to be the direct cause of a false
> `is_recovery` value. The source-name and metadata contract must be checked
> explicitly before changing the model.

### 3.2 Required audit

Before training another long run, create a manifest-level audit for both the
`0905` and `0909` datasets, and for the exact dataset used by V11. For every
record, report:

| Field | Required check |
|---|---|
| `_source` | Preserve the original source stem or a canonical form that still contains the recovery identifier. |
| `_recovery_boost` | Count recovery records before and after text integration. The count and source-family breakdown must not unexpectedly collapse. |
| `frame_ann` | Verify that recovery records have a valid annotation covering the primitive future window. |
| `action_label` | Verify that the sampled label belongs to the expected action family for recovery windows. |
| `text_embedding` | Verify that the label exists in the embedding cache and is not silently replaced by the empty embedding. |
| `is_recovery` | Verify the primitive-level count and fraction after window extraction. |
| augmented source names | Verify that files produced by `augment_fall_recovery.py` retain a `stand_up_lying*` stem after packing. |

The audit should print at least:

```text
total sequences
recovery sequences
recovery hours
recovery fraction
recovery source stems
recovery records with missing frame_ann
recovery primitives
recovery primitives with empty action_label
recovery primitives with empty text embedding
recovery primitives by action_label
non-recovery records whose source/action label contains lying or stand_up
```

The last line is important: it can expose an accidental broad match or a
metadata collision that makes normal motions enter the get-up branch.

### 3.3 Acceptance criteria for the metadata audit

The V11 training run should not start until all of the following are true:

- the recovery source-family counts agree with the packed-data statistics;
- the `0909` text-enabled dataset does not show an unexplained collapse in
  `_recovery_boost` or `is_recovery`;
- augmented recovery files still match `_is_flat_recovery()`;
- recovery records do not systematically receive unrelated text labels;
- the empty-text embedding rate is reported separately for recovery and
  locomotion;
- any difference between `0905` and `0909` is explained by the manifest,
  rather than inferred from the training curves.

If the audit finds a filename or metadata regression, fix that first and
rerun the existing model evaluation before changing loss weights.

## 4. Recovery-Specific Condition Mask

### 4.1 Motivation

The current model receives many goal components that are beneficial for
locomotion but can compete with a tracker-executable get-up trajectory:

- horizontal root position encourages horizontal endpoint placement;
- end-effector targets add spatial constraints that are useful for ordinary
  motion but are not the first recovery objective;
- joint angle is easy to satisfy through a pose-like shortcut;
- velocity and full orientation can make the recovery trajectory too
  specific or too fast;
- text is useful only if its recovery label is correct.

Recovery should retain the vertical and gravity-related information needed to
recognize "become upright", while reducing the direct pressure to solve a
locomotion endpoint.

### 4.2 Fine-grained condition groups

The recovery mask should distinguish the following groups:

```text
goal.position.hor
goal.position.vert
goal.orientation.rot6d
goal.orientation.gravity
goal.velocity
goal.joint
goal.end_effector
goal.time
text
```

The existing split-goal implementation currently couples some of these:

- the horizontal goal token also carries horizontal urgency;
- the vertical token carries height, gravity, and vertical urgency;
- the vertical token keep mask currently combines root and orientation keep
  decisions;
- the orientation token currently controls both rotation information and
  the gravity component used by the vertical token.

That coupling is too coarse for recovery. V11 should add separate keep masks
and condition probabilities for at least:

```text
goal_position_hor_condition_keep_mask
goal_position_vert_condition_keep_mask
goal_gravity_condition_keep_mask
goal_orientation_rot6d_condition_keep_mask
```

The slot positional encoding must remain present when a slot is masked. Only
the condition content should be zeroed. This preserves the transformer's
knowledge of which token slot is missing and avoids confusing "masked token"
with "different token type".

### 4.3 Initial recovery condition policy

The first recovery experiment should use the following policy:

| Condition | Locomotion | Get-up first trial | Rationale |
|---|---:|---:|---|
| horizontal position | existing setting | drop or strongly mask | Avoid making horizontal endpoint placement the main recovery objective. |
| vertical height | existing setting | keep with weak stochastic masking | Height is useful, but should not encourage root-only lifting. |
| gravity `g` | existing setting | keep as an input, but do not give it a large direct loss initially | It identifies the upright direction; a large loss can produce visible orientation change without executable support action. |
| orientation `rot6d` | existing setting | strongly mask or drop | Recovery does not need yaw-sensitive endpoint orientation. |
| velocity | existing setting | drop in the first rescue trial | It is not the primary recovery cue and may impose an overly fast trajectory. |
| joint angle | existing setting | strongly mask or drop | It is easy to satisfy as a pose shortcut and can suppress spatial action learning. |
| end effector | existing setting | drop in the first rescue trial | Keep it for a later ablation after the recovery core works. |
| arrival time | existing setting | keep initially, then ablate | Time may provide pacing; the current evidence does not prove that masking it is beneficial. |
| text | existing setting | keep only if metadata audit passes; otherwise mask | A wrong recovery text label is worse than no text. |

This is intentionally not an unconditional "drop every condition" policy.
Time and vertical information may be needed to prevent a recovery action from
becoming an excessively fast or root-dominated transition. The first ablation
should establish whether those signals help before removing them.

### 4.4 Recovery mask plumbing

The recovery mask must be per sample, because a batch can contain both
locomotion and get-up primitives. It should be driven by the verified
`is_recovery` tensor and applied before token projection or immediately after
projection, consistently with the existing masking behavior.

The same selector is now accepted by `generate_next_motion()` for explicit
offline/SONIC recovery tests. The planner configuration keeps
`is_recovery: false` by default; a recovery test can enable it without
changing ordinary locomotion calls. The deployment denoiser must still be
configured with the get-up profile if inference-time profile masking is
desired.

The implementation should:

1. retain the existing global `cond_mask_prob` behavior for locomotion;
2. add a nested recovery override under the same configuration tree;
3. resolve the override only for samples with `is_recovery=True`;
4. expose the resulting keep ratios in training diagnostics;
5. pass separate horizontal/vertical/gravity keep masks into the loss.

A target configuration shape is:

```yaml
denoiser:
  cond_mask_prob:
    locomotion:
      text: 0.2
      goal:
        position:
          hor: 0.05
          vert: 0.05
        orientation:
          rot6d: 0.3
          gravity: 0.3
        velocity: 0.1
        joint: 0.6
        end_effector:
          left_hand: 0.3
          right_hand: 0.3
          left_foot: 0.3
          right_foot: 0.3
        time: 0.1
      scene: 0.1
    getup:
      # Start with text masked until the recovery metadata audit passes.
      text: 1.0
      goal:
        position:
          hor: 0.95
          vert: 0.05
        orientation:
          rot6d: 0.95
          gravity: 0.05
        velocity: 0.95
        joint: 0.95
        end_effector:
          left_hand: 0.95
          right_hand: 0.95
          left_foot: 0.95
          right_foot: 0.95
        time: 0.1
      scene: 1.0
```

Here a probability of `1.0` means "always mask". The exact numeric values
are an experiment configuration, not a final claim. The first run should
compare the recovery override against a no-override baseline using the same
checkpoint and seed where possible.

The current rescue recommendation uses `0.95` rather than `1.0` for the
easy-to-shortcut get-up goal components. They remain masked in 95% of samples,
but retain a small learning path in the remaining 5% instead of becoming
permanently unavailable. The vertical position, gravity, and arrival-time
conditions remain much more available because they carry the core get-up
state. Scene masking is shown as `1.0` while `load_scene: false`; the input is
blank in that phase and the value has no practical effect.

The get-up text probability is intentionally `1.0` in the initial rescue
configuration. After the metadata audit, set it to `0.0` only if the recovery
labels are verified and the text condition is shown to help.

## 5. Initial Get-Up Loss Configuration

### 5.1 Principle

Get-up must not be trained with the same objective as locomotion. It should
retain enough supervision to avoid losing the recovery target, but it should
not reward a shortcut that raises or rotates the root without support action.

The initial rescue configuration should be conservative:

```yaml
train:
  manager:
    loss_weight:
      locomotion:
        rec: 1.0
        foot_contact: 0.01
        support_consistency: 1.0
        goal:
          root_position_hor: 0.5
          root_position_vert: 0.02
          root_velocity: 0.02
          root_orientation: 0.001
          g: 0.0
          joint_angle: 0.0
          end_effector: 0.05
      getup:
        rec: 1.5
        foot_contact: 0.05
        support_consistency: 0.01
        goal:
          root_position_hor: 0.0
          root_position_vert: 0.02
          root_velocity: 0.0
          root_orientation: 0.0
          g: 0.0
          joint_angle: 0.0
          end_effector: 0.0
```

The values above are a first rescue ablation, not a final tuned recipe:

- `rec` remains the main get-up objective;
- `foot_contact` is retained because contact transitions are part of the
  executable motion;
- `support_consistency` returns to a small diagnostic-compatible value
  rather than the current very large get-up value, because support loss alone
  cannot invent a missing get-up action;
- horizontal root-position supervision is removed for get-up;
- vertical height is kept weakly to prevent a completely non-upright
  solution, but it must not dominate reconstruction;
- root orientation and joint-angle endpoint losses are disabled initially;
- gravity remains available as a condition, but its direct endpoint loss is
  zero in the first rescue trial.

The exact key names must follow the current nested loss schema. If the
  implementation does not yet expose `g` as a separate loss key, do not
  silently map it to full root orientation: add the diagnostic/configuration
  name first or leave the direct gravity loss disabled. In the current
  configuration, `g` is the gravity-orientation goal loss.

### 5.2 Why not increase recovery `support_consistency`

A large support-consistency coefficient is not evidence that the model has
learned how to get up. It constrains consistency with support/contact
structure already represented by the target and FK-derived diagnostics. It
does not by itself teach a missing sequence of pushes, hand placements, and
joint transitions.

The observed low hand-support-active ratio must also be interpreted carefully:
the current diagnostic is derived from future ground-truth FK and the
recovery flag, not directly from the predicted motion. A change in that
metric can therefore reflect data routing or recovery-window composition,
rather than a learned prediction improvement.

## 6. Rollout and Offline Augmentation Policy

### 6.1 Rollout

Do not disable rollout in the first V11 implementation. The current rollout
schedule is left unchanged so that the recovery mask and loss changes can be
evaluated under the same training regime that produced the regression.

The task-dependent rollout idea remains valid as a later experiment:

- current rollout selection is not a per-sample recovery-aware policy;
- a future implementation could use `is_recovery` to apply a separate rollout
  probability or ramp;
- that experiment must be isolated from the first recovery mask/loss test.

If recovery still fails after metadata and mask corrections, compare:

```text
same checkpoint + current rollout
same checkpoint + recovery-aware rollout
```

Do not change both the rollout policy and the recovery labels in one run.

### 6.2 Offline recovery augmentation

`dataset/data_process/augment_fall_recovery.py` is still useful for generating
SONIC-validated flat-lying recovery variants. It is not the first V11 lever:

- the user confirmed that it does not cover the tail windows that happen to
  inherit the clip-level get-up flag;
- enabling it together with new masks and new losses would make attribution
  difficult;
- its filename propagation must nevertheless be included in the metadata
  audit.

Treat augmentation as a later A/B experiment after the base recovery mask is
verified.

## 7. Experiment Order

### E0: Data and metadata audit

Run the manifest and primitive audit described in Section 3 on:

- the dataset used by `0905_goal_vert_token`;
- the dataset used by `0909_text`;
- the exact dataset selected for V11.

No model training is needed for E0.

### E1: Inference mask ablation

Using a fixed checkpoint, evaluate recovery with:

1. current masks;
2. horizontal position masked only;
3. horizontal position and end-effector masked;
4. horizontal position, end-effector, joint, velocity, and rot6d masked;
5. the proposed recovery core: weak vertical height, weak gravity, and
   time retained.

This distinguishes a condition conflict from a loss-only problem before a
new long training run.

### E2: Short fine-tuning rescue

Start from the best current locomotion checkpoint that also passes the E0
metadata audit. Do not blindly transplant `0905` weights: its goal layout
differs from the end-effector layout used by `0909`, so a full or partial
weight transfer needs explicit compatibility checks.

Change only:

- recovery-specific condition masking;
- the initial get-up loss weights;
- recovery diagnostics.

Keep rollout, data augmentation, optimizer schedule, and locomotion weights
unchanged.

### E3: Controlled augmentation comparison

Only after E2 has a measurable recovery improvement, compare:

```text
E2 without offline recovery augmentation
E2 with offline recovery augmentation
```

Keep the source stem and `_recovery_boost` audit in both cases.

### E4: Reintroduce information one group at a time

If the recovery core works, add back one group per experiment:

1. verified recovery text;
2. end-effector condition;
3. velocity;
4. gravity loss;
5. horizontal position with a weak weight.

Stop adding a group when SONIC recovery drops materially. This gives a
direct causal explanation for the regression instead of relying only on
aggregate training loss.

## 8. Diagnostics and Acceptance Criteria

### 8.1 Data and condition diagnostics

Log separate locomotion and recovery values for:

```text
data/recovery_sequence_fraction
data/recovery_primitive_fraction
data/recovery_action_label_empty_rate
data/recovery_text_embedding_empty_rate
data/recovery_source_match_rate
condition/recovery_position_hor_keep_ratio
condition/recovery_position_vert_keep_ratio
condition/recovery_gravity_keep_ratio
condition/recovery_orientation_rot6d_keep_ratio
condition/recovery_time_keep_ratio
```

These values are required to interpret later loss curves.

### 8.2 Training diagnostics

Continue to log the existing per-class losses and add separate recovery
values for:

```text
rec
foot_contact
support_consistency
goal_root_position_hor
goal_root_position_vert
goal_g
goal_root_orientation
goal_end_effector
```

A falling aggregate loss is not sufficient evidence of recovery progress.
Every recovery loss should be interpreted together with the keep ratio and
the SONIC result.

### 8.3 SONIC acceptance gates

A V11 checkpoint is acceptable only if it satisfies both gates:

**Normal locomotion**

- no material regression in gait quality;
- no loss of ordinary goal-reaching ability;
- no new systematic sliding or root-only endpoint shortcut.

**Fall recovery**

- higher stable get-up success than the `0909` baseline;
- a visible and tracker-followable support action;
- monotonic or at least plausible root-height and gravity progression;
- no frequent handstand-like out-of-distribution solution;
- recovery does not rely primarily on horizontal goal placement.

Offline goal losses and reconstruction losses are diagnostics, not the final
recovery acceptance criterion. SONIC closed-loop execution remains decisive.

## 9. Non-Goals and Deferred Questions

The following are intentionally not resolved by V11:

- whether tail primitives from a recovery clip should receive a different
  label;
- whether rollout should be closed, delayed, or made recovery-aware;
- whether offline recovery augmentation should be enabled in the final run;
- whether arrival-time conditioning should ultimately be dropped for get-up;
- how to remove locomotion overshoot without adding a terminal-velocity or
  dwell condition;
- final tuning of recovery loss coefficients.

These questions remain valid, but changing them before the metadata and
condition-mask audit would make the current regression harder to localize.

## 10. Implementation Checklist

1. Add a manifest/primitive audit for `_source`, `_recovery_boost`,
   `frame_ann`, `action_label`, `text_embedding`, and `is_recovery`.
2. Verify that `augment_fall_recovery.py` output preserves the recovery source
   stem through packing.
3. Add per-sample recovery condition overrides under the nested
   `cond_mask_prob` configuration.
4. Split position and orientation masks into horizontal, vertical, gravity,
   and rot6d keep masks.
5. Pass the corresponding masks into goal-loss computation.
6. Add separate recovery diagnostics for keep ratios and loss terms.
7. Run E0 and E1 before starting a long fine-tuning job.
8. Run E2 with rollout and augmentation unchanged.
9. Use SONIC recovery and locomotion gates to select the next checkpoint.

## 11. References

- `TextOpRobotMDAR/docs/planner_v8_gait_and_recovery.md`
- `TextOpRobotMDAR/docs/planner_v9_text_condition.md`
- `TextOpRobotMDAR/docs/planner_v7_1_fall_recovery.md`
- `TextOpRobotMDAR/docs/planner_v5_recovery.md`
- `dataset/data_process/pack_motion_lib_to_textop.py`
- `dataset/data_process/augment_fall_recovery.py`
- `TextOpRobotMDAR/robotmdar/dataloader/data.py`
- `TextOpRobotMDAR/robotmdar/model/mld_denoiser.py`
- `TextOpRobotMDAR/robotmdar/train/loss.py`
- `TextOpRobotMDAR/robotmdar/train/train_dar.py`

# Planner V5: Fall Recovery

> Status: design discussion, not yet implemented.

## 1. Motivation

The V4 planner (body_ext goal, 29-DOF, 64-frame future) generates locomotion and
obstacle-avoidance motions but has no explicit fall-recovery capability. When the
robot falls, the planner must generate a stand-up sequence from the fallen state.

The fall-recovery evaluation (§1.1) reveals a sharp quality split: flat-lying
recovery (back / stomach) tracks reliably (>87%) while side-lying (<63%) and
crutch variants (0–50%) are unreliable. This document defines a data-filtering,
weighting, and domain-randomization strategy to make recovery robust.

### 1.1 Recovery tracking success rates

```
Group                                       Rate
────────────────────────────────────────── ──────
stand_up_lying                               100.0%
stand_up_lying_stomach                       100.0%
faint_stand_up_lying_stomach                 100.0%
faint_stand_up_lying                          87.5%  (1 camera angle fails)
faint_stand_up_lying_side                     50.0%
stand_up_lying_side                           62.5%
crutch_get_up / crutch_rise_up / crutches_*  0–50.0%
────────────────────────────────────────── ──────
TOTAL (non-crutch, non-side)                 ~95.8%
TOTAL (all)                                   46.3%
```

**Decision:** Keep only flat-lying recovery (back / stomach). Discard side-lying
and all crutch-based get-up motions. The tracker cannot execute them reliably, so
training on them wastes model capacity and may produce reference motions the
tracker cannot follow.

### 1.2 The fundamental OOD problem

Official motion-capture data shows the character rising from a flat, extended
pose (arms and legs outstretched). A real robot falls into arbitrarily crumpled,
twisted configurations that are out-of-distribution (OOD) for the planner.

**Proposed strategy (two-stage):**

1.  **Planner always generates "lie flat → stand up."** It never tries to
    generate a recovery that starts from a twisted pose. The planner's output
    assumes extended limbs in a flat-lying configuration.

2.  **The tracking controller closes the gap.** The RL tracker tries to match the
    planner's reference. When the robot is crumpled and the reference says "arms
    out, lying flat," the tracker pulls the robot toward that configuration. The
    tracker has its own domain-randomization training (SONIC-style), so it can
    handle moderate starting-state mismatches.

3.  **Coherent uniform history perturbation during planner training** bridges
    the residual gap. During the tracker's transition from crumpled → flat,
    the planner's history frames (which come from the robot's actual state)
    are still OOD. Training-time perturbation enlarges the basin from which
    the planner can return toward a tracker-executable recovery manifold
    (see §4 and `planner_v5_domain_randomization.md`).

This avoids the impossible problem of collecting / generating every possible
crumpled fall pose. The tracker does the heavy physical recovery; the planner
just needs to be *insensitive* to the exact starting configuration.

Note on side-lying: a valid recovery is **not** a direct side-get-up motion.
The planner is free to generate
`side-lying → supine/prone → known get-up motion`. The relevant manifold is
the tracker-executable recovery manifold, not the set of all theoretically
possible human recovery motions.

---

## 2. Data Filtering

### 2.1 Changes to `filter_and_copy_bones_data.py`

Add the following keywords to the default `--filter-keywords` list:

```python
# Fall recovery — exclude low-success variants (Planner V5 §2.1)
"lying_side",          # 50–62.5% success — tracker cannot reliably execute
```

`crutch` is already in the default filter list (line 222). It covers all
`crutch_get_up*`, `crutch_rise_up*`, and `crutches_*` variants.

The net effect:
- **Kept:** `stand_up_lying`, `stand_up_lying_stomach`, `faint_stand_up_lying`,
  `faint_stand_up_lying_stomach` (plus the `_puke_walk_ff` compound variant)
- **Removed:** `*_lying_side`, `crutch_get_up*`, `crutch_rise_up*`, `crutches_*`

### 2.2 Rationale

| Keyword | Affected actions | Reason |
|---------|-----------------|--------|
| `lying_side` | `faint_stand_up_lying_side`, `stand_up_lying_side` | <63% tracking success — side-lying recovery is physically harder; shoulder/hip are pinned on one side, requiring asymmetric torque the G1 cannot reliably produce. |
| `crutch` (existing) | All crutch-assisted get-ups | 0–50% success — crutch motions involve an external prop the real robot does not have, and the asymmetric weight-bearing pattern is unstable under the G1's PD torque limits. |

---

## 3. Weighted Sampling for Recovery Data

### 3.1 Strategy

Enable `data.weighted_sample=true` in the actual training launch, with
`frame_weight=false`. Only boost the sampling probability of flat-lying
recovery sequences.

**Approach B:** Assign a fixed weight multiplier to every manifest record whose
`frame_ann` contains a fall-recovery coarse label. All other records keep the
default weight of 1.0.

### 3.2 Implementation path

#### Step 1 — Add a `_recovery_boost` field during packing

In `pack_motion_lib_to_textop.py`, after `motion_lib_entry_to_textop()` returns
the item, check whether the coarse label indicates a kept recovery action:

```python
# After line 446:  coarse = classify_coarse(fine)
_RECOVERY_FINE_PATTERNS = (
    "stand_up_lying",        # matches: stand_up_lying, stand_up_lying_stomach
    "faint_stand_up_lying",  # matches: faint_stand_up_lying,
                             #   faint_stand_up_lying_stomach,
                             #   faint_stand_up_lying_puke_walk_ff
)

def _is_flat_recovery(fine_name: str) -> bool:
    """Check whether fine action is a kept flat-lying recovery."""
    lower = fine_name.lower()
    # Exclude side-lying (should already be filtered, but belt-and-suspenders)
    if "lying_side" in lower:
        return False
    return any(pat in lower for pat in _RECOVERY_FINE_PATTERNS)
```

Set `item["_recovery_boost"] = True` when the action matches.

#### Step 2 — Override sequence weight in the dataloader

In `SkeletonPrimitiveDataset._cal_sample_weight()` (`data.py:250–311`), when
`weighted_sample=true` and a per-sequence `_recovery_boost` flag is present,
multiply the computed weight:

```python
RECOVERY_WEIGHT_MULTIPLIER = 5.0  # tunable

for data in self.raw_data:
    seq_weight = 0.0
    for seg in data['frame_ann']:
        seg_act_cat = seg[3]
        act_weights = 0.0
        for act_cat in seg_act_cat:
            if act_cat not in action_statistics:
                continue
            act_weights += action_statistics[act_cat]['weight']
        seq_weight += (seg[1] - seg[0]) * act_weights

    # ── Recovery boost (Planner V5 §3) ──
    if data.get('_recovery_boost'):
        seq_weight *= RECOVERY_WEIGHT_MULTIPLIER
    # ────────────────────────────────────

    data['weight'] = seq_weight
```

#### Step 3 — Configuration

In the training config:

```yaml
data:
  weighted_sample: true
  frame_weight: false
  # All action_statistics weights stay at their default (1/duration)
  # Only recovery sequences get the additional multiplier.
```

### 3.3 Why multiplier 5.0?

Flat-lying recovery constitutes ~400–600 samples out of ~130k (≈0.4%). A 5×
multiplier raises their effective sampling probability to ~2%, comparable to
`kneel` (1.5% naturally). This is enough for the model to see recovery sequences
regularly without overwhelming locomotion data.

Ablation candidates: 3×, 5×, 10×. Start with 5× and monitor:
- TensorBoard: per-category reconstruction loss on recovery vs. locomotion
- Qualitative: does the model still walk properly, or does the boost bleed into
  locomotion generation?

### 3.4 Why not just duplicate manifest records?

Duplicating manifests (Approach A) is simpler but has two drawbacks:
1.  Requires re-running the packing pipeline each time the multiplier changes.
2.  Multiple manifests pointing to the same sample file create correlated
    gradient updates within a batch, which is statistically less efficient than
    independent weighted sampling.

Approach B keeps the data files immutable and makes the multiplier a runtime
config parameter.

---

## 4. History Perturbation for Domain Randomization

> Superseded for training-time augmentation by
> `planner_v7_1_fall_recovery.md`: V7.1 removes the per-sample recovery
> amplitude branch from the dataloader and moves recovery-scale perturbation to
> offline preprocessing.

> Design details: see `planner_v5_domain_randomization.md`. This section
> summarizes the agreed conclusion; that document is authoritative on
> ranges and implementation specifics.

### 4.1 Design rationale

Version 1 uses only the two retained flat-recovery label families:
`stand_up_lying*` and `faint_stand_up_lying*`. Broader fall/lying semantic
labels are reserved for a later revision.

**Problem.** At deployment, the planner's history frames come from the robot's
actual state (via the tracker's proprioceptive feedback). When the robot is
recovering from a fall, these states are OOD relative to the clean mocap
histories seen during training.

**Goal.** The perturbation is **not** generic sensor noise and **not** ordinary
push robustness — the closed-loop replanning architecture already handles
moderate disturbances as long as the robot does not fall. Instead, the
perturbation must:

1.  enrich the planner's history-state distribution around existing recovery
    trajectories;
2.  expose the planner to *coherent* deviations caused by external disturbances;
3.  enlarge the basin from which the planner can return toward a
    **tracker-executable recovery manifold**;
4.  preserve the structure and physical consistency of the original history.

Conceptually, the augmentation approximates a finite-width recovery tube
𝒯_δ(𝒯_M_recovery) around the tracker-executable recovery manifold. The planner
should learn to map states inside this neighborhood back toward motions the
tracker can execute — the relevant manifold is the **tracker-executable**
recovery manifold, not the set of all theoretically possible human recovery
motions.

**Distribution choice — uniform only.** Related work:

- **SONIC** (§3.2, Table S4): all DR perturbations use uniform `𝒰[·]` —
  target joint jitter `𝒰[-0.1, 0.1]` rad, position jitter `𝒰[-0.05, 0.05]` m.
  Validated at scale (100M+ frames, 42M params).
- **ReactiveBFM** (§3.1): injects Gaussian noise into prefix representations
  to counter tracker execution drift, where small errors dominate.

Our fall-recovery scenario is closer to SONIC's: a random fall produces
arbitrary joint configurations with no natural central tendency. Uniform
sampling provides approximately even coverage over the prescribed recovery
neighborhood instead of concentrating samples near the clean trajectory.
**For the first implementation, use uniform perturbations only; do not
introduce Gaussian noise.**

### 4.2 Coherent perturbation over the whole history

Do **not** perturb each frame independently. One physical deviation — the robot
tilted, displaced in joint configuration, or slightly lower than nominal —
must apply consistently across the full history window.

For every training history, sample a **single** perturbation vector δ and apply
it with a smooth temporal ramp:

```
s̃_τ = s_τ + w_τ · δ,   τ ∈ [t-H+1, t]

w_τ = w_min + (1 - w_min) · (τ - (t-H+1)) / (H-1),   w_min ≈ 0.5
```

Older frames receive ~50% of the sampled deviation; the most recent state
receives the full deviation. This models a robot that has *gradually drifted*
away from the nominal motion rather than one whose state jumps randomly
between frames.

### 4.3 Perturbation variables and ranges

Only three quantities are perturbed in the first version:

```
q_t (joint configuration),   roll_t / pitch_t,   h_t (root height)
```

**Not perturbed:** yaw, contact labels, local root displacement, root
translational velocity, incremental yaw, joint increments. Derived quantities
are recomputed after perturbation — in particular Δq̃_t = q̃_{t+1} − q̃_t, never
perturbed independently.

Ranges are split by motion type (recovery vs normal):

| Quantity | Normal-motion range | Fall-recovery range |
|----------|--------------------:|--------------------:|
| Upper-body `q` | ±0.03 rad | **±0.12 rad** |
| Leg `q` | ±0.02 rad | **±0.08 rad** |
| Root roll | ±0.05 rad | **±0.25 rad** |
| Root pitch | ±0.05 rad | **±0.15 rad** |
| Root height | ±0.01 m | **±0.03 m** |

All distributions are uniform and symmetric around zero.

**Why the upper-body range is larger than the leg range:** in supine and prone
get-up motions, hand and arm support configurations are highly important. The
planner must tolerate moderate deviations in shoulder, elbow, and wrist
configuration while still generating an executable recovery trajectory.

**After perturbation, clamp joint configurations to valid robot joint limits.**
The implementation parses the per-joint `range` attributes from
`g1_29dof.xml` (in model-facing joint order) and clamps each perturbed DOF to
its real limits. **TODO(joint-limits):** the skeleton config does not yet
expose limits programmatically — once it does, prefer the cfg-provided values
over the MJCF re-parse. Fallback when parsing fails: coarse ±π clamp.

Do not perturb `Δq_t` directly — recompute it from the perturbed joint
trajectory. The last history frame's `Δq` connects to the first *clean*
future frame; by the v1 contract it is restored to its clean value rather
than recomputed from the mixed boundary.

The larger recovery ranges apply only to motions identified as recovery-related
from dataset labels (fall, lying, get-up, stand-up-from-floor, etc.). For
non-recovery motions either use the mild normal ranges or leave the history
clean — large recovery-oriented perturbations must not degrade normal
locomotion quality.

**Scope: LDM training only.** The VAE training pipeline (`train_mvae.py`) uses
the same dataset class and must stay clean. `augmentation_start_step` defaults
to "never" (`1 << 30`) in `data/mob.yaml`; only `train_dar.yaml` overrides it
(50,000). During LDM training the perturbed history reaches both the frozen
VAE encoder (teacher latent target) and the VAE decoder (generated-future
reconstruction), which is the intended contract: the denoiser learns to
predict the latent that, together with the noisy history, reproduces the clean
future.

### 4.4 Perturbing the trigonometric roll/pitch representation

The planner observes `[sin r, cos r − 1]`, not raw Euler angles. Perturb the
trigonometric representation **directly** via angle-addition identities —
do not recover Euler angles and re-encode:

```
sin(r + δ) = sin r · cos δ + (c_r + 1) · sin δ      where c_r = cos r − 1
cos(r + δ) − 1 = (c_r + 1) · cos δ − sin r · sin δ − 1
```

This preserves sin²r + cos²r = 1 up to numerical precision and is applied
independently to roll and pitch.

**Euler singularity precautions.** Fall-recovery motions may contain supine,
prone, or inverted root orientations where intrinsic Euler angles approach
singular configurations (gimbal-lock sensitivity, ±π jumps, unstable pitch
decomposition). The implementation must not assume raw Euler differences are
globally smooth:

1.  perturb in the trigonometric representation whenever possible
    (angle-addition identities above);
2.  avoid adding noise to wrapped Euler sequences and then differencing them;
3.  if raw orientations are available as rotation matrices/quaternions during
    preprocessing, prefer R̃ = R_perturb · R and convert afterward;
4.  explicitly inspect fall-recovery trajectories for representation jumps
    around lying poses before enabling large roll/pitch perturbations.

### 4.5 Sampling strategy

Do not perturb every training history, and do not perturb during the clean-history warm-up:

```
P(clean) = 0.5,   P(uniformly perturbed) = 0.5
```

For recovery clips this may later be increased to
`P(perturbed | recovery) ≈ 0.7–0.8`. Keep a substantial fraction of clean
recovery histories to preserve the original recovery motion distribution.

The first study should primarily sweep perturbation **magnitude**, not add
more randomization types.

The warm-up is controlled by `data.augmentation_start_step` (default 50,000
in the DAR config). Before that global optimizer step, histories are always
clean. Validation histories remain clean. Goals remain clean after activation,
but the ego-centric reference pose is reset to the perturbed latest history
state.

### 4.6 Implementation location

**In the dataloader** (`data.py:_extract_single_primitive`) or **in the
training loop** (`train_dar.py`). The training-loop option gives full control
over the perturbation probability and easy TensorBoard logging; the dataloader
option keeps the training loop unchanged. Either is acceptable for v1.

```python
def _apply_history_perturbation(
    self,
    motion_features: torch.Tensor,   # [B, T, 57]
    history_len: int,
    is_recovery: torch.Tensor,       # [B] bool
    perturb_prob: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Coherent uniform history perturbation (Planner V5 §4.2)."""
    out = motion_features.clone()
    B, H = motion_features.shape[0], history_len

    # Per-sample: perturb with probability perturb_prob
    do_perturb = torch.rand(B, generator=generator) < perturb_prob

    for b in range(B):
        if not do_perturb[b]:
            continue
        # ── Sample ONE offset vector for the whole history window ──
        rng = (a_q_up, a_q_leg, a_roll, a_pitch, a_h)
        a = [r if is_recovery[b] else n for r, n in zip(
            RECOVERY_RANGE, NORMAL_RANGE)]   # pick range set per sample

        delta_q_upper = (torch.rand(..., generator=generator) * 2 - 1) * a[0]
        delta_q_leg   = (torch.rand(..., generator=generator) * 2 - 1) * a[1]
        delta_roll    = (torch.rand(..., generator=generator) * 2 - 1) * a[2]
        delta_pitch   = (torch.rand(..., generator=generator) * 2 - 1) * a[3]
        delta_h       = (torch.rand(..., generator=generator) * 2 - 1) * a[4]

        # ── Temporal ramp: w_min=0.5 → 1.0 over the history window ──
        w = torch.linspace(0.5, 1.0, H, device=...)  # [H]

        # q: add coherent offset, then clamp to joint limits
        out[b, :H, q_slice] += w[:, None] * delta_q
        out[b, :H, q_slice] = out[b, :H, q_slice].clamp(q_min, q_max)

        # roll/pitch: angle-addition in trig representation (see §4.4)
        sin_r, c_r = out[b, :H, 0], out[b, :H, 1]
        out[b, :H, 0] = (sin_r * cos(w * delta_roll)
                         + (c_r + 1) * sin(w * delta_roll))
        out[b, :H, 1] = ((c_r + 1) * cos(w * delta_roll)
                         - sin_r * sin(w * delta_roll) - 1)
        # ... same for pitch (channels 2, 3)

        # height: coherent offset
        out[b, :H, h_slice] += w * delta_h

        # Δq is recomputed downstream from the perturbed q trajectory
    return out
```

### 4.7 First ablation study

Keep the augmentation structure fixed; vary only the recovery perturbation
magnitude. Three levels:

| Level | q upper | q leg | roll | pitch | height |
|-------|---------|-------|------|-------|--------|
| Small | ±0.06 | ±0.04 | ±0.10 | ±0.08 | ±0.02 |
| **Medium (start)** | **±0.12** | **±0.08** | **±0.25** | **±0.15** | **±0.03** |
| Large | ±0.15 | ±0.10 | ±0.30 | ±0.20 | ±0.04 |

Evaluate at least: nominal motion quality, original supine/prone recovery
success, recovery from perturbed supine/prone histories, side-lying
initialization recovery success, recovery completion time, and the probability
of generating a trajectory successfully executed by SONIC.

The preferred setting is the largest perturbation range that clearly expands
fall-recovery success without materially degrading nominal motion quality or
tracker executability.

---

## 5. Interaction with Other Planner V5 Changes

### 5.1 num_primitive reduction

Reducing `num_primitive` from 4 to 3 (see separate analysis) lowers
`required_length` from 273 to 209 frames, raising the valid-sample rate from
54% to 73%. Fall recovery data is long (median 549 frames), so it is largely
unaffected. The additional walk/jog samples help Stage 1 locomotion quality.

### 5.2 Rollout training

The existing three-stage rollout schedule (`use_rollout=true`) is complementary
to history perturbation: rollout training forces the model to autoregress from
its own (potentially imperfect) generated history, while coherent uniform
perturbation simulates physical state deviations even on GT history. Together
they cover both "my previous generation was off" and "the robot's actual state
does not match the reference."

### 5.3 Goal specification for recovery

When the recovery behavior is triggered, the upstream task planner should set:
- `goal_root_pos_world`: at standing height (z ≈ 0.78 m for G1), near the
  robot's current xy position.
- `goal_yaw_world`: current estimated facing (or don't-care — `force_drop_goal_yaw=true`).
- `goal_root_velocity_world`: zero (we want the robot to stand up, not walk).
- `goal_keypoints_world`: at standing posture limb positions.

The model learns to interpret "goal at standing height, current height near
ground" as the recovery cue, analogous to how "goal at chair height" cues
sitting (LDM_goal_scene_design §2.4).

---

## 6. References

- **planner_v5_domain_randomization.md** — Authoritative design for
  recovery-oriented uniform history perturbation (§4). Contains the full
  range tables, trig-perturbation math, and ablation plan.
- **SONIC** (2025) — Domain randomization with **uniform distributions** on
  physical parameters and motion commands (Table S4). Target joint jitter
  `𝒰[-0.1, 0.1]` rad, position jitter `𝒰[-0.05, 0.05]` m, orientation jitter
  `𝒰[-0.1, 0.1]` rad. Validated at scale: 100M+ frames, 42M params, 21k GPU-hr.
  The uniform-distribution choice for our history perturbation follows this
  precedent (§4.1).
- **ReactiveBFM** (2025) — Scheduled AR prefix sampling with Gaussian noise
  injection into prefix representations. Linear decay of teacher-forcing
  probability. §3.1. We adopt the prefix-perturbation *idea* but use uniform
  distributions and coherent whole-window offsets instead of per-frame
  Gaussian noise.
- **LDM_goal_scene_design.md** — Goal height as implicit interaction cue (§2.4).
- **planner_v4_ext_goal.md** — V4 body_ext goal layout and component masking.
- **analyze_action_distribution.py** — BONES-SEED coarse/fine taxonomy.
- **analyze_short_actions.py** — Duration-filter analysis for window-size tuning.

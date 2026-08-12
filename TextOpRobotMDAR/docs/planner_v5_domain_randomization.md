# Prompt: Recovery-Oriented Uniform History Perturbation for the Motion Planner

We use a hierarchical humanoid control architecture consisting of a **motion generator (planner)** and a **motion tracker (SONIC controller)**. The planner operates in an online closed-loop replanning manner and takes a short history of robot-skeleton motion states as input. The tracker executes the generated reference motion in physics.

The planner state representation for each frame is

[
f_t=
[
\phi(r_t),
\Delta\psi_t,
c_t,
\Delta p_t^{local},
h_t,
q_t,
\Delta q_t
],
]

where

[
r_t=(roll_t,pitch_t,yaw_t),
]

[
\phi(r_t)=
[
\sin roll_t,;
\cos roll_t-1,;
\sin pitch_t,;
\cos pitch_t-1
],
]

[
\Delta\psi_t=\psi_{t+1}-\psi_t,
]

[
\Delta p_t^{local}
==================

R_z(\psi_t)^\top(p_{t+1}-p_t),
]

(h_t) is the root height, (q_t) is the robot joint configuration, and (\Delta q_t=q_{t+1}-q_t).

The training dataset is BONES-SEED. It already contains cleaned fall-and-recovery motions, including supine and prone get-up motions, although the diversity of recovery states is still limited. The low-level SONIC tracker can reliably execute a subset of these recovery motions, especially recovery from supine and prone configurations, while direct side-lying recovery is substantially less reliable.

The goal of this first version of planner-side domain randomization is therefore **not to simulate generic sensor noise or to improve ordinary push robustness**. The existing online closed-loop replanning architecture already handles moderate external disturbances as long as the robot does not fall. Instead, the perturbation is designed to:

1. enrich the planner's history-state distribution around existing recovery trajectories;
2. expose the planner to coherent deviations caused by external disturbances;
3. enlarge the basin from which the planner can return toward a tracker-executable recovery manifold;
4. improve fall recovery and robustness to sim-to-real state deviations;
5. preserve the structure and physical consistency of the original motion history.

For the first implementation, use **uniform perturbations only**. Do not introduce Gaussian noise yet.

---

## 1. Core Design Principle

Treat each clean recovery history as a trajectory lying near a tracker-executable recovery manifold,

[
H=
{f_{t-H+1},\ldots,f_t}.
]

The augmentation should create a bounded neighborhood around this trajectory:

[
\tilde H=\mathcal A(H),
]

where the perturbation is sampled from a uniform distribution.

The purpose of uniform sampling is to provide approximately even coverage over a prescribed recovery neighborhood rather than concentrating samples close to the clean trajectory.

Conceptually, the augmentation approximates a finite-width recovery tube

[
\mathcal T_\delta(\mathcal M_{\mathrm{recovery}})
=================================================

{s:d(s,\mathcal M_{\mathrm{recovery}})<\delta}.
]

The planner should learn to map states inside this neighborhood back toward motions that remain executable by the tracker.

---

## 2. Apply Perturbations to the Entire History Coherently

Do **not** independently perturb each frame.

A perturbation should represent a coherent physical deviation, such as the robot being tilted, displaced in joint configuration, or slightly lower than the nominal recovery trajectory.

For every training history after the configured warm-up step, first sample a single perturbation vector

[
\delta
]

and apply it consistently over the full history window.

A simple formulation is

[
\tilde s_\tau=s_\tau+w_\tau\delta,
\qquad
\tau\in[t-H+1,t],
]

where (w_\tau) is a smooth temporal ramp.

Use either:

[
w_\tau=1
]

for a constant offset over the complete history, or preferably a mild ramp such as

[
w_\tau
======

w_{\min}
+
(1-w_{\min})
\frac{\tau-(t-H+1)}{H-1},
]

with

[
w_{\min}\approx0.5.
]

Thus, older frames receive approximately 50% of the sampled deviation and the most recent state receives the full deviation.

This represents a robot that has gradually drifted away from the nominal recovery motion rather than one whose state randomly jumps between frames.

---

## 3. First-Version Perturbation Variables

Restrict the first version to only three quantities:

[
\boxed{
q_t,\quad roll_t/pitch_t,\quad h_t
}
]

Do not perturb, in this version:

* yaw;
* contact labels;
* local root displacement;
* root translational velocity;
* incremental yaw;
* joint increments directly.

Derived quantities should be recomputed after perturbation whenever necessary.

---

## 4. Joint Pose Perturbation

For each history, sample a joint-space offset

[
\delta q
\sim
\mathcal U(-a_q,a_q)
]

component-wise.

Use different perturbation magnitudes for upper-body and lower-body joints.

### Standard motions

For motions not labeled as fall-recovery, use only mild perturbation:

[
\delta q^{upper}
\sim
\mathcal U(-0.03,0.03)\ {\rm rad},
]

[
\delta q^{leg}
\sim
\mathcal U(-0.02,0.02)\ {\rm rad}.
]

These values should only provide a small state-history enrichment and should not significantly alter the original motion semantics.

### Fall-recovery motions

For clips whose action labels indicate falling, lying, getting up, recovery, or equivalent recovery-related motion categories, use larger uniform ranges:

[
\boxed{
\delta q^{upper}
\sim
\mathcal U(-0.12,0.12)\ {\rm rad}
}
]

and

[
\boxed{
\delta q^{leg}
\sim
\mathcal U(-0.08,0.08)\ {\rm rad}.
}
]

A conservative initial sweep may use

[
a_q^{upper}\in{0.06,0.09,0.12,0.15}\ {\rm rad},
]

[
a_q^{leg}\in{0.04,0.06,0.08,0.10}\ {\rm rad}.
]

The larger upper-body range is intentional. In supine and prone get-up motions, hand and arm support configurations are highly important. The planner should tolerate moderate deviations in shoulder, elbow, and wrist configuration while still generating a recovery trajectory that the tracker can execute.

After perturbation, clamp joint configurations to valid robot joint limits.

Do not perturb (\Delta q_t) independently. Recompute it from the perturbed joint trajectory:

[
\widetilde{\Delta q}_t
======================

\tilde q_{t+1}-\tilde q_t.
]

---

## 5. Root Roll and Pitch Perturbation

Root orientation is particularly important during fall recovery.

For non-recovery motions, use only small angular perturbations:

[
\delta roll,\delta pitch
\sim
\mathcal U(-0.05,0.05)\ {\rm rad}.
]

For fall-recovery motions, use larger bounded perturbations.

Recommended first values are

[
\boxed{
\delta roll
\sim
\mathcal U(-0.25,0.25)\ {\rm rad}
}
]

and

[
\boxed{
\delta pitch
\sim
\mathcal U(-0.15,0.15)\ {\rm rad}.
}
]

Suggested sweep ranges are

[
a_{roll}\in
{0.10,0.15,0.20,0.25,0.30}\ {\rm rad},
]

[
a_{pitch}\in
{0.08,0.10,0.15,0.20}\ {\rm rad}.
]

The somewhat larger roll range is acceptable because lateral body deviation is an important OOD source during falling. However, the objective is **not to train the planner to synthesize a dedicated side-lying get-up strategy**.

If the low-level tracker cannot reliably execute direct side-lying recovery motions, the planner should instead be free to generate a trajectory that first moves the robot toward a tracker-friendly supine or prone configuration and then follows an executable get-up motion.

Therefore, successful recovery from a side-lying state does not imply that the planner must synthesize a side-get-up motion. A valid behavior is

[
\text{side lying}
\rightarrow
\text{supine/prone}
\rightarrow
\text{known get-up motion}.
]

The relevant manifold is therefore the **tracker-executable recovery manifold**, not the set of all theoretically possible human recovery motions.

---

## 6. Perturbing the Trigonometric Roll/Pitch Representation

The planner does not directly observe raw roll and pitch. Instead, it uses

[
[
\sin r,\cos r-1
].
]

Do not first recover Euler angles with an inverse trigonometric operation and then perturb them unless necessary.

Given an angular perturbation (\delta), update the trigonometric representation directly using the angle-addition identities:

[
\sin(r+\delta)
==============

\sin r\cos\delta
+
\cos r\sin\delta,
]

[
\cos(r+\delta)
==============

## \cos r\cos\delta

\sin r\sin\delta.
]

Because the stored feature is

[
c_r=\cos r-1,
]

recover

[
\cos r=c_r+1.
]

Then compute

[
\widetilde{\sin r}
==================

\sin r\cos\delta
+
(c_r+1)\sin\delta,
]

[
\widetilde{\cos r}-1
====================

## (c_r+1)\cos\delta

\sin r\sin\delta
-1.
]

Apply this independently to roll and pitch.

This preserves the exact trigonometric consistency

[
\sin^2r+\cos^2r=1
]

up to numerical precision.

---

## 7. Euler-Angle Singularities and Discontinuities

Fall-recovery motions may contain supine, prone, inverted, or near-horizontal root orientations.

Under such configurations, intrinsic Euler angles may approach singular configurations, and raw Euler trajectories may exhibit:

* gimbal-lock sensitivity;
* discontinuous jumps between equivalent Euler representations;
* sudden changes near (\pm\pi);
* unstable decomposition around singular pitch configurations.

Therefore, the augmentation implementation must **not assume that raw Euler-angle differences are globally smooth**.

Use the following precautions.

First, apply perturbations in the trigonometric representation whenever possible using the angle-addition identities above.

Second, avoid computing orientation perturbations by naïvely adding noise to wrapped Euler sequences and then differencing them.

Third, if raw orientations are available as rotation matrices or quaternions during dataset preprocessing, prefer the operation

[
\tilde R
========

R_{\mathrm{perturb}}R
]

and only then convert to the required feature representation.

Fourth, explicitly inspect fall-recovery trajectories for representation jumps around lying poses before enabling large roll/pitch perturbations.

The augmentation must preserve continuity of the encoded history even when the underlying Euler parameterization is ambiguous.

---

## 8. Root Height Perturbation

Root height should be perturbed conservatively because fall-recovery poses often place the pelvis close to the ground.

For normal motions:

[
\delta h
\sim
\mathcal U(-0.01,0.01)\ {\rm m}.
]

For fall-recovery motions:

[
\boxed{
\delta h
\sim
\mathcal U(-0.03,0.03)\ {\rm m}.
}
]

A reasonable sweep is

[
a_h
\in
{0.01,0.02,0.03,0.04}\ {\rm m}.
]

Avoid substantially larger height perturbations in the first version because they can create geometrically inconsistent states such as body penetration, floating support points, or unrealistic hand-ground relationships.

If forward-kinematics or collision information is available during preprocessing, reject augmented samples that result in severe ground penetration.

---

## 9. Fall-Recovery-Specific Augmentation

Large perturbations should only be applied to motions identified as recovery-related from the dataset labels.

Use the BONES-SEED action annotations to construct

[
\mathcal D_{\mathrm{recovery}}
\subset
\mathcal D.
]

Relevant labels may include semantic categories corresponding to:

* fall;
* falling;
* lie down;
* lying;
* supine;
* prone;
* get up;
* stand up from floor;
* recover;
* floor transition;
* related fall-and-recovery actions.

Do not attempt to automatically create synthetic side-recovery trajectories.

The existing recovery motions provide the motion prior, while perturbation only expands their local history-state neighborhood.

For

[
H\in\mathcal D_{\mathrm{recovery}},
]

sample from the large recovery ranges.

For

[
H\notin\mathcal D_{\mathrm{recovery}},
]

either use the mild ranges above or leave the history clean.

This prevents large recovery-oriented perturbations from unnecessarily degrading normal locomotion and motion-generation quality.

---

## 10. Recommended First-Version Distribution

Use the following initial configuration.

| Quantity       | Normal-motion range | Fall-recovery range |
| -------------- | ------------------: | ------------------: |
| Upper-body (q) |       (\pm0.03) rad |   **(\pm0.12) rad** |
| Leg (q)        |       (\pm0.02) rad |   **(\pm0.08) rad** |
| Root roll      |       (\pm0.05) rad |   **(\pm0.25) rad** |
| Root pitch     |       (\pm0.05) rad |   **(\pm0.15) rad** |
| Root height    |         (\pm0.01) m |     **(\pm0.03) m** |

All distributions are uniform and symmetric around zero.

Use one sampled offset per history window, modulated by the smooth temporal ramp.

---

## 11. Sampling Strategy

Do not perturb every training history, and do not perturb during the clean-history warm-up.

A reasonable first configuration is:

[
P(\text{clean})=0.5,
]

[
P(\text{uniformly perturbed})=0.5.
]

For recovery clips, this can later be increased to

[
P(\text{perturbed}\mid\text{recovery})
\approx0.7-0.8.
]

Keep a substantial fraction of clean recovery histories to preserve the original recovery motion distribution.

The first study should primarily sweep perturbation magnitude rather than introducing many additional randomization types.

The warm-up is controlled by `data.augmentation_start_step` (default 50,000
in the DAR config). Before that global optimizer step, histories are always
clean. Validation histories remain clean. Goals remain clean after activation,
but the ego-centric reference pose is reset to the perturbed latest history
state.

---

## 12. Interpretation of Recovery Success

The desired behavior is not

[
\text{perturbed state}
\rightarrow
\text{same nominal recovery trajectory}.
]

Instead, the planner is allowed to generate any motion satisfying

[
\tau_{\mathrm{gen}}
\in
\mathcal M_{\mathrm{tracker\text{-}executable}}
]

and progressively returning the robot to a stable motion manifold.

For example, when initialized from a side-lying state,

[
s_{\mathrm{side}}
]

a valid solution is

[
s_{\mathrm{side}}
\rightarrow
s_{\mathrm{supine}}
\rightarrow
\tau_{\mathrm{supine\ getup}}
\rightarrow
s_{\mathrm{standing}}.
]

This is preferable to forcing a direct side-lying recovery trajectory if the tracker cannot reliably execute such motions.

The planner therefore acts as a data-driven projection toward a tracker-compatible recovery manifold.

---

## 13. First Ablation Study

For the first experiment, keep the augmentation structure fixed and vary only the recovery perturbation magnitude.

Use approximately three levels.

### Small

[
q^{upper}:\pm0.06,
\qquad
q^{leg}:\pm0.04,
]

[
roll:\pm0.10,
\qquad
pitch:\pm0.08,
]

[
h:\pm0.02.
]

### Medium

[
q^{upper}:\pm0.12,
\qquad
q^{leg}:\pm0.08,
]

[
roll:\pm0.25,
\qquad
pitch:\pm0.15,
]

[
h:\pm0.03.
]

### Large

[
q^{upper}:\pm0.15,
\qquad
q^{leg}:\pm0.10,
]

[
roll:\pm0.30,
\qquad
pitch:\pm0.20,
]

[
h:\pm0.04.
]

Evaluate at least:

* nominal motion quality;
* original supine recovery success;
* original prone recovery success;
* recovery from perturbed supine/prone histories;
* side-lying initialization recovery success;
* recovery completion time;
* probability of generating a trajectory successfully executed by SONIC.

The preferred setting is the largest perturbation range that clearly expands fall-recovery success without materially degrading nominal motion quality or tracker executability.

---

## 14. First-Version Summary

The first implementation should remain deliberately simple:

[
\boxed{
\text{Uniform only}
}
]

[
\boxed{
\text{history-consistent perturbation}
}
]

[
\boxed{
q + roll/pitch + root\ height
}
]

[
\boxed{
\text{large perturbation only for labeled fall-recovery motions}
}
]

[
\boxed{
\Delta q\text{ and other incremental features are not independently randomized}
}
]

[
\boxed{
\text{the objective is return to a tracker-executable recovery manifold}
}
]

The initial recommended fall-recovery ranges are

[
\boxed{
q_{\mathrm{upper}}:\pm0.12\ {\rm rad},
\quad
q_{\mathrm{leg}}:\pm0.08\ {\rm rad},
}
]

[
\boxed{
roll:\pm0.25\ {\rm rad},
\quad
pitch:\pm0.15\ {\rm rad},
}
]

[
\boxed{
h:\pm0.03\ {\rm m}.
}
]

These values should be treated as the center of the first ablation sweep rather than as fixed final hyperparameters.

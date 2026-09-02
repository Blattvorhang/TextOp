# Planner V7.1: Fall Recovery — Unified History Perturbation

> Status: design approved, implementation pending.
>
> Inherits from `planner_v5_domain_randomization.md` and
> `planner_v5_recovery.md`. This document **supersedes** the training-time
> augmentation parts of both: the V5 amplitude tables (§4.3, §10, §13), the
> ramp of V5 §4.2, and the recovery/normal amplitude split (V5 §9, §11).
> The V5 documents remain authoritative for: the motivation and the
> tracker-executable recovery manifold interpretation (v5 recovery §1),
> the data-filtering decision to keep only flat-lying recovery families
> and drop side-lying / crutch get-ups (v5 recovery §2), the goal
> specification for recovery episodes (v5 recovery §5.3), the rollout
> interaction (v5 recovery §5.2), and the general history-perturbation
> rationale (v5 DR §1–§3).

---

## 1. Summary of changes

1. **Absolute reference: `occHIPC/scripts/check_motion_sonic.py`.**
   Its `augment_history()` implementation and amplitude tables have been
   manually validated against SONIC tracking in the interactive reviewer.
   All training-time perturbation logic is rewritten to match that file
   exactly. Any deviation must be documented and justified in §2.
2. **One amplitude standard only.** Training-time domain randomization
   uses the file's `"normal"` amplitude set for every motion. The
   `is_recovery`-based recovery/normal split is removed from the
   augmentation path (§8).
3. **Smoothstep ramp-out.** One coherent offset per history window,
   scaled per frame by `w = 1 - s(u)` with `s(u) = 3u² - 2u³`. The weight
   is exactly 1 at the oldest history state frame and decays to exactly
   0 at the current state frame t with zero slope (§4). Every history
   state frame is perturbed; the current frame is exactly unperturbed, so
   the derived motion features stay continuous across the boundary.
4. **Feature-version parity.** The same perturbation applies to
   FeatureVersion 3 and FeatureVersion 6 (§6).
5. **XYZ body-axis description, never RPY.** Root orientation offsets are
   rotateX / rotateY / rotateZ about the x (forward), y (left), z (up)
   axes of **the perturbed frame's own body frame** — every history frame
   about its own axes. RPY/Euler terminology is banned from this
   specification: Euler decomposition is singular near lying poses, while
   per-frame xyz axes are well-defined for every pose, including supine
   and prone (§5.2).
6. **Recovery-scale augmentation moves to data preprocessing.** The
   larger perturbations needed for flat-lying recovery are no longer
   applied at training time; they become an offline preprocessing step
   with per-clip silent SONIC validation before packing (§9).

---

## 2. Authority: `occHIPC/scripts/check_motion_sonic.py`

The gold-standard implementation is `augment_history()` in
`occHIPC/scripts/check_motion_sonic.py` (lines 164–245), together with:

- `JOINT_AUG_AMPS` / `ROOT_AUG_AMPS` (lines 91–106) — the amplitude tables;
- `_smoothstep` (lines 145–147) — the ramp;
- `_dof_amp_vector` (lines 109–115) — per-DOF amplitude lookup by joint-name
  prefix;
- `_load_joint_limits` (lines 118–142) — per-joint limits from the G1 MJCF,
  falling back to ±π.

The training-side implementation `_augment_raw_motion()` in
`TextOpRobotMDAR/robotmdar/dataloader/data.py` must be rewritten to be a
faithful port of that function, restricted to the `"normal"` amplitude set.
The reviewer file keeps its `"recovery"` tables and the
`--history-aug-recovery` flag as the validation tool and amplitude source for
the preprocessing-stage design (§9).

Axis convention (from the reviewer): x-forward, y-left, z-up. Root
orientation offsets are rotations about the x/y/z axes of **the
perturbed frame's own body frame** — never described as RPY/Euler angles
(§5.2). The reviewer composes them per frame as
`q_new = q_old ⊗ q_off` (body-frame right-multiplication); the training
port must reproduce this algebraically identically.

---

## 3. The single amplitude standard

One table, no per-sample branch. All amplitudes are uniform, symmetric
around zero, per-DOF (joints) or per-channel (root). Values are copied
verbatim from the reviewer's `"normal"` sets.

### 3.1 Joint offsets (rad)

| Joint group | Amplitude |
|-------------|----------:|
| shoulder    |   **±0.30** |
| elbow       |   **±0.10** |
| hip         |   **±0.05** |
| knee        |   **±0.05** |
| ankle       |   **±0.05** |
| waist       |   **±0.20** |
| wrist       |   **±0.05** |

Groups are matched to DOF names by substring prefix, in table order
(first match wins), exactly as in `_dof_amp_vector` and as already done in
the current TextOp implementation.

Changes vs. the V5 "normal" set: shoulder 0.03 → 0.30, elbow 0.03 → 0.10,
waist 0.02 → 0.20, hip/knee/ankle 0.02 → 0.05, wrist 0.03 → 0.05. The
reviewer-validated values are substantially larger for the arms and torso;
this is deliberate and must not be silently re-tuned.

### 3.2 Root offsets (rad / m)

| Channel   | Amplitude |
|-----------|----------:|
| rotateX (about the body x axis, forward) | **±0.20** |
| rotateY (about the body y axis, left)    | **±0.20** |
| rotateZ (about the body z axis, up)      | **±0.05** |
| root height     | **±0.01 m** |

Changes vs. V5: the x/y-axis amplitudes go 0.05 → 0.20; the z-axis
offset was explicitly forbidden in V5 and is now perturbed by ±0.05;
height unchanged. RPY names are deliberately absent from this table —
see §5.2 for the rationale.

The V5 recovery/normal split (V5 §4.3, §9) and its recovery amplitude
tables are **deleted** from the training path (§8). The reviewer's
`"recovery"` tables remain as candidate amplitudes for the preprocessing
stage (§9).

---

## 4. Ramp: smoothstep decay to zero at the boundary

Let `H = history_len` (16). The ramp is sampled over the `H + 1 = 17`
**state frames** of the history window — the 16 history state frames plus
the current frame t (see the bookkeeping note below):

```
u = linspace(0, 1, H + 1)
s(u) = 3u² - 2u³            # smoothstep, zero slope at u = 0 and u = 1
w = (1 - s(u))[:H]          # weights for state frames 0 .. H-1; frame H has w = 0
```

| u (state frame index / H) | w |
|--------------------:|----------------:|
| 0 (state frame 0)   | 1.0             |
| 0.25                | 0.84375         |
| 0.5                 | 0.5             |
| 0.75                | 0.15625         |
| (H-1)/H = 15/16     | ≈ 0.0112        |
| 1 (state frame H)   | **0 (exact)**   |

Properties:

- **State frame 0 (oldest history state frame) receives the full offset**
  (w = 1), and the ramp starts with zero slope, so the earliest frames
  are flat at 1×.
- **State frame H — the current frame t — receives exactly 0**, with zero
  slope (`s'(1) = 0`). The perturbed state prefix joins the clean state
  trajectory continuously and with zero slope; there is no jump or kink
  at the current frame.
- The last perturbed state frame (index H-1) receives
  `w = 1 - s((H-1)/H) ≈ 0.011` of the offset for H = 16 — by design
  (see the bookkeeping note below).

This **reverses the V5 ramp**. V5 §4.2 applied `w_min ≈ 0.5` to the oldest
frame and 1.0 to the newest ("the robot has gradually drifted away from the
nominal motion"). The reviewer-validated scheme instead models the history
as a *displaced state that has already converged back onto the nominal
trajectory by the current frame*: the perturbation is fully present in the
past and has fully decayed at the boundary, so the planner is always asked
to continue a trajectory that is clean at the point where its output
starts. This is also the physically consistent choice for the
ego-centric reference: the current state frame t is exactly unperturbed
and the ego reference (the last history state frame) carries only the
≈1.1% offset, so the reference pose, the goal, and the first generated
frame are mutually consistent.

**State frames vs. feature rows.** The raw motion window holds
`history_len + 1 + future_len = 16 + 1 + 64 = 81` **state frames**
(indices 0..80). Motion features are not state frames: each of the 80
feature rows (16 history + 64 future) is computed by backward
differencing over adjacent state-frame pairs. The ramp is therefore
sampled over the `16 + 1` state frames of the history window: state
frames 0..15 receive the weights above (all > 0 — every history state
frame is perturbed), and state frame 16 — the current frame t, shared by
the last history feature row and the first future feature row — receives
exactly 0. Perturbation is applied to the just-read state frames
**before** feature computation (this is already the order in the current
code: `_augment_raw_motion` runs before `_convert_to_motion_features`),
so the features always faithfully describe the perturbed state
trajectory. The last history feature row (row 15) therefore pairs state
frame 15 (w ≈ 0.011) with the unperturbed current frame — that pairing
is the design: with w = 0 and zero slope at state frame 16, the derived
Δ-type channels transition continuously into the first future feature
row.

One coherent offset vector δ is drawn per history window and shared by all
frames: `s̃_τ = s_τ + w_τ · δ`. Per-frame independent noise remains
forbidden (V5 §4.2).

Because the current state frame is exactly unperturbed and the ramp has
zero slope there, every derived motion feature row computed from adjacent
state frames — the V3 trig orientation channels and Δq, and the V6
gravity / `rel_rot6d` / `delta_hor` channels — is continuous across the
history/future boundary. This is the "motion features stay continuous"
contract that the ramp exists to guarantee.

---

## 5. Perturbation channels (exact port)

### 5.1 Joint offsets

Same limit-aware scheme as the current implementation and the reviewer:
the *bounds* of the uniform distribution are clamped, not the post-hoc
sample.

```
active = w > 0                                   # frames that actually move
lo_q  = max over active frames of (limit_lo - x_t) / w_t     # per DOF
hi_q  = min over active frames of (limit_hi - x_t) / w_t
lower = max(lo_q, -amp)      upper = min(hi_q, amp)
q     ~ U(lower, upper)      q[lower > upper] = 0   # pose already violates a limit
x[:H] += w[:, None] * q
```

Near a joint limit the feasible range becomes asymmetric about zero
instead of piling up at the boundary. Amplitudes `amp` come from §3.1;
limits come from `g1_29dof.xml` per-joint `range` attributes with ±π
fallback (existing `_joint_limit_tensor` logic).

### 5.2 Root orientation — xyz axes of the perturbed frame itself

The three offsets rotateX / rotateY / rotateZ are rotations about the
x (forward), y (left), z (up) axes of **the perturbed frame's own body
frame** — every history state frame is rotated about its own axes by its
ramped share. There is deliberately **no RPY/Euler description anywhere**
in this specification:

```
half      = 0.5 · w[t]                        # ramped half-angle
q_x[t]    = [cos(half·rx), sin(half·rx), 0, 0]       # wxyz, about the frame's own +x
q_y[t]    = [cos(half·ry), 0, sin(half·ry), 0]       # about own +y
q_z[t]    = [cos(half·rz), 0, 0, sin(half·rz)]       # about own +z
q_off[t]  = q_x ⊗ q_y ⊗ q_z                  # intrinsic xyz (body-fixed axes)
q_new[t]  = q_old[t] ⊗ q_off[t]              # right-multiplication: rotate in the body frame
```

with `rx ~ U(-0.20, 0.20)`, `ry ~ U(-0.20, 0.20)`, `rz ~ U(-0.05, 0.05)`
(§3.2). `⊗` is the Hamilton product, applied left-to-right (apply `q_z`
first, then `q_y`, then `q_x` — i.e. `q_x ⊗ (q_y ⊗ q_z)`). This applies
to the `H = 16` history state frames; the current state frame (index H)
is untouched (w = 0, §4).

**Why xyz axes and not RPY (the special intent).** Euler/RPY decomposition
of the root orientation becomes singular near supine/prone poses: two of
the three decomposed angles become ambiguous and the trajectories jump
between equivalent representations — exactly the failure mode V5 §4.5
warned about. A rotation about the
frame's *own* body axes is defined for every pose with no singularity:
rotateZ always turns about the frame's own up axis — including when that
pose is lying flat — so the same amplitude standard stays meaningful
across standing, falling, and lying configurations. Because the reference
basis is the perturbed frame itself, the offset is also pose-independent:
it never depends on a global frame or on the orientation of any other
frame in the window.

**Matches the reviewer exactly.** `augment_history` in
`check_motion_sonic.py` composes the xyz offsets in each perturbed
frame's own body frame (`q_new = q_old ⊗ q_off`, right-multiplication);
the training port must reproduce it algebraically identically. (An
earlier version of the file composed about the body frame of its
boundary frame H; that formulation is superseded and must not be
resurrected.)

Two deliberate properties:

- The intrinsic xyz composition is the **opposite** of bones-seed's
  `root_rotateX/Y/Z` extrinsic convention (which would be
  `q_z ⊗ q_y ⊗ q_x` in world-fixed axes). The intrinsic order is
  deliberate (body-frame xyz offsets); do not "fix" it.
- The V5 implementation left-multiplied x-axis/y-axis offsets in the
  world frame (`qp ⊗ qr ⊗ q_old`, data.py `_quat_mul_xyzw`); that code is
  replaced by the body-frame composition above. TextOp stores `root_rot`
  as **xyzw**; the implementation must convert to wxyz, apply the
  composition, and convert back, keeping the arithmetic algebraically
  identical to the formulas above.

### 5.3 Root height

Only the z component of the root body, as in V5 and in the reviewer:

```
dh ~ U(-0.01, 0.01)          # m, §3.2
root_trans_offset[:H, 2] += w * dh
```

### 5.4 Derived quantities are recomputed

Perturbation is applied to the **raw motion dict** (`dof`, `root_rot`,
`root_trans_offset`) *before* feature extraction. All derived features —
the V3 trig orientation channels and Δq, and the V6 gravity /
`rel_rot6d` / `delta_hor` channels — are computed from the perturbed raw
motion by the existing feature-conversion code. The V5 angle-addition
machinery (V5 §4.4) is obsolete and removed: rotating raw quaternions and
re-encoding is exact by construction and immune to Euler-representation
issues (V5 §4.5 concerns are resolved the same way).

---

## 6. Feature-version parity (V3 = V6)

- Delete the `if motion_dtype.FeatureVersion != 3: return False` early
  return in `_augment_raw_motion` (data.py:642). With V6 defaulting in
  `base.yaml`, the V5 augmentation has been silently inert; V7.1 makes it
  active for both versions.
- Because the perturbation lives on the raw motion dict, V3 and V6 need
  **no separate augmentation code** — feature conversion handles the
  difference.
- **V3 boundary contract (kept):** the feature-space clean-Δq restore in
  `_organize_primitives_by_index` (data.py:1513–1517) remains, so the last
  history Δq still references the clean first future pose. Gate it to the
  V3 feature layout (`delta_start = 11 + dof_dim`): V6 features are 44
  channels wide (height + gravity + delta_hor + rel_rot6d + dof + contact)
  and have **no Δq channel**, so the restore slice would index out of
  bounds.
- **V6 boundary contract:** all derived channels are recomputed from the
  perturbed state trajectory — internally consistent, no restore needed.
  The last history feature row pairs perturbed state frame H-1 with the
  unperturbed current state frame; the deviation of its Δ-type channels
  from the clean values is bounded by `w[H-1] · δ ≈ 1.1% · δ` — by
  design, with zero slope at the boundary (§4).

---

## 7. Gating and boundary contracts (unchanged from V5)

- `augmentation_enabled: true` and `augmentation_start_step: 50000` in
  `train_dar.yaml`; `augmentation_prob: 0.5` per history window.
- Train split only; validation histories stay clean.
- Goals stay clean; the ego-centric reference pose
  (`gt_ref_pos/gt_ref_rot`) and the frame-0 anchors
  (`history_start_pos/rot`) are reset to the perturbed history state
  (existing code, data.py:1518–1528). With the new ramp the last history
  state frame is only ≈1% perturbed, so this reset is nearly a no-op,
  but it is kept to preserve the invariant.
- The VAE pipeline (`train_mvae.py`) keeps `augmentation_start_step = 1<<30`
  (never active) — unchanged.

---

## 8. Removal of the training-time recovery split

Per the design decision, only the **DR-related** recovery code is deleted.
Scope:

**Deleted**

1. The `recovery` parameter of `_augment_raw_motion` and its entire
   recovery branch (data.py:659–667): the recovery joint-group table,
   `a_roll = a_pitch = 1.50`, `a_h = 0.03`.
2. The call-site plumbing that passes `primitive['is_recovery']` into the
   augmentation (data.py:1496–1499).
3. Comments/docstrings in the training path that reference the recovery
   amplitude split.

**Kept (not DR; superseded later by §9)**

1. The `is_recovery` batch field and its propagation (data.py:1331,
   1539) — still consumed by the get-up-class diagnostics.
2. `_motion_class_labels` / `_add_per_class_extras` in
   `train/manager.py` — the `getup` class in
   `metric_class/{train,eval}/{e_q_vel,e_h_vel,e_g_cons}/…` continues to
   monitor recovery quality during training.
3. The packing-time `_recovery_boost` flag and the ×5 sequence-weight
   boost in `_cal_sample_weight` (data.py:482–483).

These three survivors will be revisited when the preprocessing-stage
scheme of §9 is finalized; §9 may redefine which flat-lying subsets are
boosted and how.

---

## 9. Recovery-scale augmentation at preprocessing

Flat-lying recovery needs **larger** perturbations than the single
training-time standard (§3). V5 applied them at training time, keyed to a
per-sample label; that couples the DR magnitude to dataset labeling and
was never validated end to end. V7.1 moves recovery-scale perturbation to
**offline data preprocessing**: the larger offsets are applied once to
flat-lying recovery clips before packing, every candidate is screened by
**silent SONIC tracking in MuJoCo**, and only clips SONIC can actually get
up from are added to the training set. Training itself applies only the
uniform §3 standard to everything.

### 9.1 Scope: the kept flat-lying get-up clips

- **Input glob**: `stand_up_lying*` under `data/motion_lib_filtered/` —
  the 16 kept clips (`stand_up_lying_R_002__A47{2,3,4,5}` and
  `stand_up_lying_stomach_R_002__A47{2,3,4,5}`, each with a `_M` mirror
  variant, all in `231010/`). Verified: all 16 clips start lying on the
  ground (`lying_start == 0`).
- **Explicitly excluded from augmentation**: the 8
  `faint_stand_up_lying_puke_walk_ff_180_*` clips in the same directory.
  They stay in the training set and are sampled normally, but are not
  augmented — their start frame is not a flat-lying pose, so a
  from-frame-0 lying perturbation would be meaningless. The
  `stand_up_lying*` glob already skips them (their names start with
  `faint_`).
- **Lying-segment heuristic** (the analysis's exact logic — same
  discovery as `check_motion_sonic.py`: `rglob("stand_up_lying*")` +
  `G1MotionLoader`): pelvis height `< 0.45 m` AND the body z-axis tilted
  `> 55°` from the world z-axis, counting a segment only when continuous
  for `≥ 10` frames. Tilt is read from `root_rot` (xyzw):
  `tilt = arccos(|R[2,2]|)`, `R[2,2] = 1 − 2(x² + y²)`. The criterion is
  clean — within-segment mean h ≈ 0.03–0.18 m, tilt ≈ 73–89°, with
  almost no borderline frames.
- **Segment statistics** (all data is already 50 Hz, no resampling):

  | data dir | files | lying frames @50 Hz | lying duration |
  |---|---|---|---|
  | `data/motion_lib/231010` | 24 | 60–219, mean ≈ 142 | 1.2–4.4 s, mean ≈ 2.8 s |
  | `data/motion_lib_filtered/231010` | 16 | 62–219, mean ≈ 154 | 1.2–4.4 s, mean ≈ 3.1 s |
  | bones_seed `robot/231010` (= robot_filtered) | 24 | 72–263, mean ≈ 171 | 1.4–5.3 s, mean ≈ 3.4 s |

  Typical lying segment ≈ 140–170 frames (≈ 3 s), range 60–260 frames
  (1.2–5.3 s). **All segments start at frame 0** (the motion begins
  lying, then stands up), so the first `n_lying` frames are exactly the
  lying frames. Sub-class differences: `stand_up_lying_side` (side-lying)
  and `_R_002` (supine) are shorter, `stand_up_lying_stomach` (prone) is
  the longest (up to 219 frames ≈ 4.4 s); takes within `A47x` also vary
  widely (60 → 197 → 219).
- The perturbation targets the **lying → get-up process**: the clip's
  first frames are the flat phase (frame 0 is on the ground), and the
  unperturbed tail carries the original get-up motion.

### 9.2 Offline perturbation rule

- **Amplitudes** — verbatim copy of the reviewer's `"recovery"` tables
  (§2, lines 91–106). **Amplitude split between the two paths: the
  training-time DR keeps the single `"normal"` standard (§3, §10 item 1);
  only this offline pipeline uses the `"recovery"` tables.**

  | group | shoulder | elbow | hip | knee | ankle | waist | wrist |
  |---|---|---|---|---|---|---|---|
  | JOINT_AUG_AMPS["recovery"] (rad) | 1.50 | 1.50 | 0.80 | 1.00 | 0.10 | 0.50 | 0.20 |

  | group | x (rad) | y (rad) | z (rad) | h (m) |
  |---|---|---|---|---|
  | ROOT_AUG_AMPS["recovery"] | 0.05 | 0.10 | 1.50 | 0.03 |

- **Window**: the perturbation applies to the first `n_lying` state frames
  (frames 0 … `n_lying` − 1); frame `n_lying` and every later frame are
  **exactly unperturbed**, so the augmented prefix blends into the
  untouched get-up tail. **`n_lying ~ U(20, 50)`** — drawn uniformly per
  candidate (0.4–1.0 s). Rationale: the perturbation's job is to produce
  a diverse fallen pose and settle back into the canonical extended
  supine/prone pose before the get-up begins; the fallen-pose diversity
  comes from the offset vector δ (§9.2 amplitudes), and the window-length
  draw adds transition-duration diversity for free. The upper bound is
  always safe — the shortest lying phase in the data is 60 frames
  (§9.1), so the blend never invades the get-up. The variable name
  `n_lying` is reserved for this window length; do not reuse `n`, `H`,
  `T`, or `history_len` for it.
- **Ramp**: the §4 smoothstep with `n_lying` in place of the history
  length H. Sample the schedule at `n_lying + 1` points,
  `w[k] = 1 − s(k / n_lying)` with `s(u) = 3u² − 2u³` for
  `k = 0 … n_lying`; apply `w[k]` to frame k. Hence `w[0] = 1`,
  `w[n_lying] = 0`, zero slope at both ends. One **coherent offset
  vector per clip** (sampled once), scaled per frame by `w[k]` — the
  same structure as the training ramp (§4).
- **Composition** — identical to §5.1–§5.3: limit-bounded uniform joint
  offsets, root rotation about the x/y/z axes of **the perturbed frame's
  own body frame** via right-multiplication
  `q_new[k] = q_old[k] ⊗ q_off(w[k]·δ)`, and the z-only root-height
  offset — all under the per-frame weight `w[k]`. RPY descriptions remain
  banned (§5.2).
- **Derived data**: `contact_mask` is recomputed for the perturbed frames
  with the stage-1 contact rule (exact port to be confirmed at
  implementation; fallback: carry the original mask — the perturbed
  prefix is lying and mostly non-contact, and the mask is a training side
  channel). The `scene` occupancy is pose-independent and kept verbatim.
  `sliding_mask` stays zeros (packing default).

### 9.3 Pipeline (offline, in `dataset/data_process/`)

New script `dataset/data_process/augment_fall_recovery.py`
(imports occHIPC utils via `sys.path`, same pattern as
`check_motion_sonic.py`):

1. **Select** — iterate the 16 `stand_up_lying*` clips in
   `data/motion_lib_filtered/`, one original clip at a time
   (`faint_stand_up_lying*` excluded from augmentation — §9.1).
2. **Augment, then validate silently** — apply §9.2
   (`n_lying ~ U(20, 50)` and a fresh offset per candidate) and hand the
   augmented clip to SONIC in
   MuJoCo, headless (no interactive viewer). Success criterion from
   `occHIPC/scripts/eval_fall_recovery.py`: the pelvis z recovers to
   **0.78 m ± 0.10** and stays stable (variance **σ² < 0.02**) over the
   **final 50 frames** of the clip. **One silent trial per candidate**:
   measured across all 18 action families, a clip that SONIC tracks to
   success once succeeds in all 20 trials (deterministic policy), so a
   single trial is the acceptance test.
3. **Save accepted clips** to a clearly-marked folder
   `data/motion_lib_filtered/aug_fall_recovery/` — joblib pkls in the
   motion_lib entry schema (stage-1 output format: `root_trans_offset`
   [T,3] m, `root_rot` [T,4] xyzw, `dof` [T,29] rad in MJCF order,
   `contact_mask` [T,2], `fps`, optional `scene`), which stage 3 packs
   directly. **Naming**: trajectory name = original action name +
   augmentation ordinal — file `aug_{k:03d}.pkl`, dict key
   `{original_name}_aug{k:03d}` (e.g.
   `stand_up_lying_R_002__A472_aug003`), so the packed `_source` reads
   `aug_fall_recovery__aug_003__stand_up_lying_R_002__A472_aug003`.
   Verified against `_extract_action_name` / `classify_coarse` in
   `pack_motion_lib_to_textop.py`: the trailing `_002__A472_aug003`
   run is stripped to `…_stand_up_lying_R`, which still contains the
   `stand_up_lying*` stem, so the coarse label (`fall`) and
   `_is_flat_recovery` → `_recovery_boost` are preserved, hence the
   getup-class logging (§8) keeps working.
4. **Batch per original clip** — generate `k` accepted clips from one
   original before moving to the next (k: §9.4).
5. **Repack and re-weight** — re-run stage 3+4
   (`pack_motion_lib_to_textop.py` walks the folder automatically via
   `rglob`, so `aug_fall_recovery/` is picked up without changes; then
   `cal_weighted_statistics.py --neutral`) to regenerate
   `data/g1_textop_29dof/samples` + manifests. Afterwards set
   **`weighted_sample: false`** (`robotmdar/config/train_dar.yaml`) —
   the ×5 `RECOVERY_WEIGHT_MULTIPLIER` boost (§8) becomes redundant once
   the augmented pool carries the recovery proportion naturally.

### 9.4 How many augmented clips per original (decided: k = 8)

- Existing successful get-up data: **12.6 min total, avg 12.2 s/clip**;
  the 16 kept clips alone ≈ 3.3 min.
- **Pose diversity**: every accepted clip is a distinct lying pose — an
  independent draw of a 33-dim offset vector (29 DOF + 4 root channels)
  from the recovery amplitudes, plus a window draw
  `n_lying ~ U(20, 50)` (§9.2), so
  even k = 4 would yield 64 genuinely different lying poses.
  Single-trial validation (§9.3) makes extra candidates nearly free,
  so the adopted setting is **k = 8 accepted clips per original**
  (128 poses, ≈ 26 min) for better coverage of the lying-pose
  manifold.
- **Data-size balance**: with `weighted_sample: false`, the natural
  recovery proportion is the augmented pool's share of the dataset —
  k = 8 → ≈ 3.5 % (vs. the current ≈ 2 % effective with the ×5
  boost). ≈ 3.5 % is judged adequate to start: at batch ≈ 1024 every
  step sees ≈ 35 recovery windows. If getup per-class errors plateau
  or rollout recovery still fails, escalate to k = 16 (≈ 7 %) — a pure
  data-generation rerun, no code change; beyond that the recovery
  share starts degrading the nominal walk/run motions.
- At the §9.5 acceptance rates (100 % for both kept families), generate
  ≈ 1.05× candidates per original to net k accepted (small safety
  margin against rare rejected draws).
- Validate k ∈ {4, 8, 16} and watch the getup per-class diagnostics
  (§8); keep the largest k that leaves the nominal walk/run per-class
  errors unchanged (§11.2).

### 9.5 SONIC success-rate distribution (the analysis)

18 action types, 134 files total, 20 trials each; success = pelvis
recovers to 0.78 m and stays stable (criterion of §9.3 step 2):

| Action type | Success rate |
|---|---|
| `stand_up_lying` | 100 % |
| `stand_up_lying_stomach` | 100 % |
| `faint_stand_up_lying_stomach` | 100 % |
| `faint_stand_up_lying` | 87.5 % |
| `faint_stand_up_lying_puke_walk_ff` | 75 % |
| `stand_up_lying_side` | 62.5 % |
| `faint_stand_up_lying_side` | 50 % |
| `crutch_*` variants | 0–50 % |

Augmentation targets only the two `stand_up_lying*` families in the
input glob (first two rows). The `faint_*` families (rows 3–5) are kept
in the training set and sampled normally, but are **not augmented** —
their start frame is not a flat-lying pose (§9.1). Side-lying and
crutch variants stay excluded, consistent with the V5 filter (§9.1).

### 9.6 SONIC-trackable get-up characteristics (design guidance)

The tracked data shows SONIC can get up when:

- the fall state is flat on the **back (supine)** or the **stomach
  (prone)**;
- the get-up has **clear hand support**;
- the get-up is **front-facing** (side get-ups largely fail);
- the lying limbs are **extended**.

Representative patterns that track successfully: supine crossed/twisted
legs → extend → hand-push; supine arm trapped under the body → pull the
arm out → hand-push; prone bent leg adducted → straighten → hand-push;
prone hands pinned → pull out → hand-push.

Conclusion: the target repertoire — diverse flat-lying poses that SONIC
can reliably get up from — is achievable purely by augmentation of the
kept clips, which is exactly what this pipeline produces.

### 9.7 Environment (MuJoCo / ONNX Runtime) — resolved

Decision: **unify on textop** (Option A), installed and verified
2026-09-02: `onnxruntime-gpu 1.20.2` (CUDA + TensorRT providers) next to
the existing `mujoco 3.2.3`. Full smoke test passed on the RTX 4070 Ti
SUPER: CUDAExecutionProvider active, 5 headless control steps of
`stand_up_lying_R_002__A472` tracked, pelvis z rising.

One operational requirement: the pip nvidia packages keep their shared
libraries under `lib/python*/site-packages/nvidia/*/lib`, which the
dynamic loader does not search by default. The pipeline launcher must
export them on `LD_LIBRARY_PATH` **before python starts** (exec-level;
in-process `os.environ` mutation proved unreliable in testing), e.g.:

```bash
export LD_LIBRARY_PATH=$(python -c "import glob;print(':'.join(sorted(
    set(glob.glob('$CONDA_PREFIX/lib/python*/site-packages/nvidia/*/lib'))
    | {'$CONDA_PREFIX/lib/python*/site-packages/onnxruntime/capi'})))"):$LD_LIBRARY_PATH
```

(Also extend `_setup_cuda_libs` in `eval_fall_recovery.py` with the
`site-packages/nvidia/*/lib` glob — it currently only searches `env/lib`
and `torch/lib`.)

### 9.8 Open decisions

1. `n_lying` distribution — resolved: `n_lying ~ U(20, 50)` frames
   (0.4–1.0 s), drawn per candidate (§9.2).
2. `k` per original clip — resolved: k = 8 (§9.4), with a k = 16
   escalation path if the getup diagnostics demand more data.
3. Acceptance test — resolved: one silent SONIC trial per candidate
   (measured: one success ⇒ 20/20; §9.3 step 2).
4. Timing of the `weighted_sample: false` switch — after the augmented
   pool is packed and getup diagnostics verified (§9.3 step 5).
5. Environment — resolved: textop env with `onnxruntime-gpu` (§9.7).
6. The `faint_stand_up_lying*` family — resolved: kept as-is, sampled
   normally in training, **not augmented** (its start frame is not a
   flat-lying pose; §9.1, §9.5). Augmentation input = the 16
   `stand_up_lying*` clips only.

---

## 10. Implementation checklist

1. **`robotmdar/dataloader/data.py`, `_augment_raw_motion`** — rewrite as
   a faithful port of `augment_history` restricted to the `"normal"`
   tables: smoothstep ramp (§4), limit-bounded uniform joint offsets
   (§5.1), own-body-frame root rotation about all three axes incl. z
   (§5.2), root-height offset (§5.3). Drop the `recovery` parameter and
   the `FeatureVersion != 3` gate. Keep the existing gating
   (enabled / train split / warm-up / probability).
2. **Call site** (`_organize_primitives_by_index`) — stop passing
   `is_recovery`; keep the ego-reference reset; gate the clean-Δq restore
   to the V3 layout (§6).
3. **`robotmdar/dtype/motion.py`** — no changes; feature conversion is
   reused as is.
4. **Tests** — extend `test/test_training_compute_optimizations.py` or add
   `test/test_history_augmentation.py`: ramp values (w = 1 at frame 0,
   w ≈ 0.0112 at frame H-1, w = 0 at frame H, zero slope at both ends),
   amplitude tables vs. §3, reference-frame quaternion composition against
   a NumPy port of `augment_history`, limit-bound clamping behavior, and
   V6 activation (augmentation now runs under FeatureVersion 6).
5. **Docs** — update the cross-references in `planner_v5_domain_randomization.md`
   §11 / `planner_v5_recovery.md` §4 to point here.
6. **`dataset/data_process/augment_fall_recovery.py`** — new offline
   script implementing §9.2–§9.3: recovery amplitudes copied verbatim
   from the reviewer's `"recovery"` tables, `n_lying ~ U(20, 50)` frame
   window from clip start with the §4 smoothstep ramp to exactly 0
   at frame `n_lying`, one silent SONIC validation trial per candidate
   (`eval_fall_recovery.py` criteria: 0.78 m ± 0.10, σ² < 0.02, last 50
   frames), and motion_lib-schema joblib output with `stand_up_lying*`
   names preserved. Launch via a wrapper that exports the nvidia lib
   paths on `LD_LIBRARY_PATH` before python starts (§9.7).
7. **Repack** — re-run stage 3+4 (`pack_motion_lib_to_textop.py` +
   `cal_weighted_statistics.py --neutral`) to regenerate
   `data/g1_textop_29dof/samples` including `aug_fall_recovery/`.
8. **Sampling weight switch** — set `weighted_sample: false` in
   `robotmdar/config/train_dar.yaml` once the augmented pool carries the
   recovery proportion (≈ 2 %) naturally; keep the ×5 multiplier code
   until then (§8).

---

## 11. Validation

1. **Reviewer re-check (the source of truth):** run
   `occHIPC/scripts/check_motion_sonic.py --history-aug 16` (normal mode)
   over representative motions (walk, jog, `stand_up_lying*`). This
   configuration is already validated; training must match it
   numerically.
2. **Training smoke run:** DAR training on V6 with augmentation active
   (`feature_version: 6`, `augmentation_enabled: true`); confirm the
   `getup`-class diagnostics (§8) stay well-defined and nominal
   `walk`/`run` per-class errors do not degrade relative to clean-history
   training (V5 §13 evaluation criteria still apply: nominal quality,
   supine/prone recovery success, tracker executability).
3. **Boundary continuity:** unit test that the perturbed state prefix
   joins the clean state trajectory with zero slope (no kink at the
   current state frame).
4. **Rollout interaction:** unchanged (v5 recovery §5.2); perturbation of
   GT histories composes with autoregressive rollout training.
5. **Offline SONIC validation (§9):** every augmented clip passes one
   headless SONIC trial (0.78 m ± 0.10, σ² < 0.02, last 50 frames)
   before it is packed — run
   `occHIPC/scripts/eval_fall_recovery.py --motion-dir
   data/motion_lib_filtered/aug_fall_recovery` (20 trials per clip) as
   the regression check for the generated pool.

---

## References

- **occHIPC/scripts/check_motion_sonic.py** — the absolute reference for
  perturbation logic and amplitudes (`augment_history`, `JOINT_AUG_AMPS`,
  `ROOT_AUG_AMPS`, `_smoothstep`).
- **occHIPC/scripts/eval_fall_recovery.py** — headless SONIC stand-up
  tracking evaluator; defines the preprocessing acceptance criterion
  (§9.3) and the regression check for the augmented pool (§11.5).
- **planner_v5_domain_randomization.md** — superseded for ranges and ramp
  (§4); still authoritative for motivation and the recovery-tube
  interpretation.
- **planner_v5_recovery.md** — data filtering (§2), goal specification
  (§5.3), rollout interaction (§5.2).
- **SONIC** (2025) — uniform-distribution domain randomization precedent
  (v5 recovery §4.1).

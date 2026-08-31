# planner_v7: Rotation-Matrix Motion Representation (Drop RPY)

## 1. Motivation

The classic RPY (roll–pitch–yaw) Euler parameterization works well for
upright locomotion: yaw separates spatial heading, pitch/roll describe body
tilt, and the representation is intuitive. However, for fall-and-recovery
tasks the root orientation passes through configurations where **pitch
approaches ±90°**, at which point the ZYX Euler decomposition becomes
singular — yaw is no longer well-defined and the roll/yaw pair exchanges
roles discontinuously (gimbal lock).

This is demonstrated empirically in
[dataset/data_analyze/analyze_rpy.py](../../dataset/data_analyze/analyze_rpy.py):
for a continuous roll-over motion `R(α) = Rx(90°) · Rz(α)`, `α ∈ [0°, 180°]`
(side-lying start, continuous body-z roll), the ZYX decomposition exhibits a
singularity exactly at `α = 90°` (pitch = 90°), where yaw jumps
discontinuously, while the matrix-based 6D representation remains smooth over
the full range (see `rpy_vs_rot6d_final_palette.pdf`).

To remove the ill-defined-yaw problem at the representation level, **v7
abandons RPY entirely**.

**Naming note:** "v7" is the serial number of this *planner design doc* —
the motion-feature version is a separate numbering, the `FeatureVersion`
constant in `robotmdar/dtype/motion.py` (active value: 3). In code, 4 and 5
are already occupied by dead experiments (see §8), so the feature described
here is introduced as **FeatureVersion 6**, not 7.

## 2. Design principles

1. **No Euler angles, no yaw.** The feature contains no yaw channel and no
   trigonometric Euler encoding. Egocentric quantities are defined with
   respect to the *current* root frame `E_t`, not with respect to a yaw-only
   heading frame.
2. **Rotation is carried by a rotation matrix.** Throughout this doc the
   root orientation matrix denotes the rotation **from the ego frame to
   the world frame**:

   $$
   \boxed{R_t \equiv {}^W R_{E_t}}
   $$

   Its transpose `R_t^⊤ = ^{E_t}R_W` maps world vectors into the ego
   frame (`g_t = R_t^⊤ g`). The ambiguous phrase "world-to-body rotation"
   is deliberately avoided — it invites transpose errors in goals,
   controller code and rotation deltas alike. `R_t ∈ SO(3)` comes from
   the dataset (provided by Bones-SEED's `rotateXYZ`; see §5 for the
   verified convention), and the feature stores the frame-to-frame
   relative rotation in the continuous 6D representation `ρ_6` — no
   Euler decomposition anywhere in the forward/inverse algorithms.
3. **Horizontal direction remains meaningful.** Although there is no
   relative x/y basis anymore, horizontal vs. vertical motion is still
   distinguished: the vertical axis is identified by the *gravity direction
   projected into the local frame*, and the horizontal component is the
   orthogonal complement (a 2D tangent-plane vector). Computing the
   horizontal/vertical split from world xyz or by gravity orthogonal
   decomposition is mathematically equivalent; we choose the gravity
   projection because it requires **no choice of tangent-plane basis** and
   therefore no yaw.
4. **Exact invertibility on valid trajectories.** The forward/inverse
   pair is deterministic and satisfies `I(F(X)) = X` up to numerical
   precision **for legal forward-encoded trajectories** — not for
   arbitrary decoder output. A raw output `m̃ ∈ R^{T×D}` need not satisfy
   the temporal constraints, which is exactly why the decoder receives
   the consistency regularization (§4.3.3) and the deterministic
   projections (§4.3.1). The pair still generalizes the v3 invariance
   (global translation, global rotation about the gravity axis).
5. **Absolute state + increments, unchanged from v3.** The reconstruction
   principle is exactly that of the original feature: a partial *absolute
   state* plus per-frame *increments* jointly determine the full sequence —
   not pure integration, not pure per-frame absolute prediction. At
   reconstruction the absolute state is *authoritative*: increments never
   override it (§4.1). v7 only replaces the rotation representation
   (see §4).

## 3. Original v3 representation (code-verified)

The v3 motion feature is (see `robotmdar/dtype/motion.py`,
`motion_dict_to_feature_v3` / `motion_feature_to_dict_v3`):

$$
\mathbf f_t
=
\left[
\phi(\mathbf r_t),\,
\Delta\psi_t,\,
\mathbf c_t,\,
\Delta\mathbf p_t^{\mathrm{local}},\,
h_t,\,
\mathbf q_t,\,
\Delta\mathbf q_t
\right]
$$

where `r_t = (roll_t, pitch_t, yaw_t)` is the root orientation in intrinsic
Euler angles and `φ(r_t) = [sin(roll_t), cos(roll_t) − 1, sin(pitch_t),
cos(pitch_t) − 1]` is the continuous trigonometric encoding of roll and
pitch. `Δψ_t = yaw_{t+1} − yaw_t` is the incremental root yaw,
`c_t ∈ {0,1}^{n_c}` the foot contact indicators,
`Δp_t^local = Rz(yaw_t)^⊤ (p_{t+1} − p_t)` the root translation increment
expressed in the yaw-aligned local frame, `h_t` the absolute root height,
`q_t ∈ R^{n_q}` the joint positions, and `Δq_t = q_{t+1} − q_t` the joint
increments. The forward algorithm maps `{p_t, R_t, q_t, c_t}_{t=0..T}` to
`{f_t}_{t=0..T−1}` plus an initial pose `(p_0, R_0)`; the inverse
reconstructs via `atan2` for roll/pitch, cumsum of `Δψ` for yaw, cumsum of
`Δp^local` for translation (with z overwritten by `h_t`), and direct reading
of `q_t`.

The failure mode of v3: `Δp_t^local` is expressed in the yaw-aligned frame,
which requires a well-defined yaw at every frame — exactly what breaks when
pitch → ±90° during falls.

## 4. New v7 motion feature

$$
\boxed{
\mathbf m_t
=
\left[
\mathbf g_t,\,
h_t,\,
{}^{E_t}\Delta\mathbf p_t^{\mathrm{hor}},\,
\Delta h_t,\,
\rho_6({}^{E_t}\mathbf R_{E_{t+1}}),\,
\mathbf q_t,\,
\Delta\mathbf q_t,\,
\mathbf c_t
\right]
}
$$

**Gravity tilt.** `g = [0, 0, −1]^⊤` is the world gravity direction. Its
expression in the current root frame,

$$
\mathbf g_t = \mathbf R_t^\top \mathbf g,
$$

is the root's tilt relative to gravity (a unit vector, 2 DoF). It replaces
the roll/pitch sincos pair of v3 and is singularity-free for all
orientations.

**Relative rotation.** The frame-to-frame rotation is

$$
{}^{E_t}\mathbf R_{E_{t+1}} = \mathbf R_t^\top \mathbf R_{t+1},
$$

encoded with the 6D rotation representation `ρ_6` (Zhou et al., 2019): the
row-major flattening of the first-two-**columns** submatrix
`R[..., :, :2]`, i.e. `[r11, r12, r21, r22, r31, r32]` (the convention of
this codebase, see §4.2), recovered by Gram–Schmidt orthonormalization. The rotation matrix and the 6D representation are
equivalent parameterizations of the same rotation; the matrix is the
canonical SO(3) object, and `ρ_6` is only the form in which rotations
enter/leave the model. See §4.2 for the convention and implementation
details.

**Translation.** The full local increment is

$$
{}^{E_t}\Delta\mathbf p_t = \mathbf R_t^\top ({}^{W}\mathbf p_{t+1} - {}^{W}\mathbf p_t).
$$

Because `E_t` may be arbitrarily tilted, its three coordinates can no longer
be interpreted as x/y = horizontal, z = vertical. We therefore split it with
the local gravity direction. Define the horizontal projector

$$
P_t^{\mathrm{hor}} = I - \mathbf g_t \mathbf g_t^\top
$$

and decompose

$$
{}^{E_t}\Delta\mathbf p_t^{\mathrm{hor}} = (I - \mathbf g_t \mathbf g_t^\top)\,{}^{E_t}\Delta\mathbf p_t,
\qquad
g_t^\top \Delta\mathbf p_t^{\mathrm{hor}} = 0,
$$

a 2D tangent-plane vector embedded in `R^3` — **no tangent-plane basis is
chosen**. The vertical increment is the scalar

$$
\Delta h_t = -\mathbf g_t^\top {}^{E_t}\Delta\mathbf p_t.
$$

The full displacement is recovered exactly:

$$
{}^{E_t}\Delta\mathbf p_t
=
{}^{E_t}\Delta\mathbf p_t^{\mathrm{hor}} - \Delta h_t\,\mathbf g_t,
$$

so the pair `(Δp_t^hor, Δh_t)` is information-equivalent to the original 3D
increment while explicitly separating horizontal and vertical motion.

**Joint and contact channels.** `q_t`, `Δq_t = q_{t+1} − q_t`, and `c_t` are
unchanged from v3 (contact is moved to the end of the channel layout).

Feature dimensionality: `3 + 1 + 3 + 1 + 6 + n_q + n_q + n_c` (= 74 for
`n_q = 29`, `n_c = 2`, vs. 69 for v3).

**Absolute state vs. transition.** The channels group naturally into the
two roles inherited from v3:

$$
\boxed{\text{absolute state: } (\mathbf g_t,\; h_t,\; \mathbf q_t,\; \mathbf c_t)}
\qquad
\boxed{\text{transition: } (^{E_t}\Delta\mathbf p_t^{\mathrm{hor}},\; \Delta h_t,\; {}^{E_t}\mathbf R_{E_{t+1}},\; \Delta\mathbf q_t)}
$$

Reconstruction is therefore **neither pure integration nor pure per-frame
absolute prediction**: the absolute state suppresses the long-horizon drift
that integration accumulates, while the increment channels describe the
motion dynamics explicitly (what changes between `t` and `t+1`). At
reconstruction the absolute state is **authoritative** (§4.1).

Because the dataset/control rate is fixed at **50 Hz**, the increment
channels carry a fixed, known time scale (`Δt = 20 ms`), so no explicit
`Δt` input channel is required.

### 4.1 Inverse algorithm (sketch): authoritative channels

Given the initial pose `(p_0, R_0)` and `{m_t}_{t=0..T−1}`, first
deterministically project the raw decoder output (§4.3.1): `ĝ_t`,
`R̂_t^rel = GS(r̃_t^6D)`, `Δp̂_t^hor`. Then:

1. `R'_{t+1} = R_t · R̂_t^rel` (integration);
2. re-align `R'_{t+1}` to the **authoritative** tilt: with
   `a = (R'_{t+1})^⊤ g` (integrated gravity) and `b = ĝ_{t+1}`
   (authoritative gravity), apply the minimal rotation `S` mapping `b`
   onto `a` (axis `b × a`, angle `acos(b · a)`) and set
   `R_{t+1} = R'_{t+1} · S` — the gravity direction comes from the
   absolute channel, the remaining (yaw-like) DoF from integration. The
   anti-parallel case uses the deterministic fallback below;
3. `Δp_t = Δp̂_t^hor − Δh_t · ĝ_t` — the full increment, expressed in the
   ego frame `E_t`. Integrate **in world coordinates**:

   $$
   \boxed{{}^W p'_{t+1} = {}^W p'_t + R_t\,\left(\widehat{\Delta\mathbf p}_t^{\mathrm{hor}} - \Delta h_t\,\hat{\mathbf g}_t\right)}
   $$

   (the factor `R_t` is essential: the local vector must be rotated into
   the world frame before accumulation). Then re-anchor the result along
   the gravity direction so that its ground-relative height equals the
   absolute `h_t` (v3 precedent — drift-free height reference). With the
   world convention `^Wg = [0,0,−1]` and ground at `z = 0` this is simply
   `p'_{t+1}[..., 2] = h_t`; the method itself is defined by gravity, not
   by the world-axis convention.
4. `q_t`, `c_t` read directly.

**Anti-parallel fallback.** Note that an upside-down root
(`g_t ≈ [0,0,+1]`) is *not* the degeneracy: the re-alignment degenerates
only when the two vectors to be aligned satisfy `a ≈ −b` (integrated
gravity vs. authoritative gravity pointing in opposite directions). For
`a^⊤ b > −1 + ε` use the standard shortest-arc rotation above. Near
anti-parallel the axis of a π-rotation is undefined; choose it
deterministically: pick the canonical basis vector least aligned with `a`,
`e_k = argmin_{e_i ∈ {e_x, e_y, e_z}} |a^⊤ e_i|`, take
`u = (a × e_k)/‖a × e_k‖`, and set `S = exp(π [u]_×)`. Better temporally:
if the previous frame produced a valid correction axis `u_{t−1}`, do not
reuse it directly — a π-rotation about `u` maps `a` onto `−a` only if
`u^⊤ a = 0`, and `u_{t−1}` is not orthogonal to the current `a_t`.
Project first:

$$
\tilde{\mathbf u}_t = \mathbf u_{t-1} - (\mathbf u_{t-1}^\top \mathbf a_t)\,\mathbf a_t,
\qquad
\mathbf u_t = \frac{\tilde{\mathbf u}_t}{\|\tilde{\mathbf u}_t\|},
$$

and only if `‖ũ_t‖` is too small (near-parallel to `a_t`) fall back to
the canonical axis `e_k` above. The axis then varies smoothly near the
anti-parallel boundary instead of jumping randomly. Frequent
`a ≈ −b` occurrences in a trained VAE are an anomaly signal — the
consistency loss (§4.3.3) should make them rare; the fallback only
guarantees the inverse is always well-defined.

**Authoritative channels — wherever an absolute channel exists.** At
reconstruction, the absolute channels `g_t`, `h_t`, `q_t` are
authoritative: if `q_{t+1} ≠ q_t + Δq_t`, the final physical trajectory
takes `q_{t+1}` — likewise `h_{t+1}` and `g_{t+1}`. The increment
channels `Δq_t`, `Δh_t`, `ΔR_t^rel` carry *local dynamics*, *velocity
information* and *temporal regularization*; they never override the
absolute state. This preserves the TextOp anti-integration-drift
property: the absolute channels re-anchor the trajectory wherever they
are available, so integration error cannot accumulate.

**Endpoint semantics.** States `t = 0..T` produce features
`m_0..m_{T−1}` — frame `t` carries the absolute state at `t` plus the
transition `t → t+1`. The final state `T` therefore has **no absolute
channels**: `q_T`, `h_T` are recovered from the last transition alone
(`q_T = q_{T−1} + Δq_{T−1}`, likewise `h_T`), and `g_T` from
`R̂_{T−1}^rel` — the authoritative re-anchoring does not apply at the
endpoint. Precisely: the absolute state is authoritative **whenever an
absolute channel is available** — every frame except the last. This is
harmless inside a fixed training window (the final state serves only as
the endpoint of the last forward difference) and the LDM rollout
re-anchors at the next window's history — but the inverse algorithm and
the temporal-consistency sums must not implicitly assume an absolute
anchor at the final index.

One asymmetry by design: there is **no absolute 3-DoF orientation
channel** (yaw-free principle), so the yaw-like DoF is carried by
`ΔR_t^rel` alone via integration; `g_t` anchors only the 2-DoF tilt
(step 2). (`g_t` is a drift-correction anchor, not mere conditioning;
`h_t` is authoritative while `Δh_t` serves dynamics/regularization.)

Internal redundancy (mirroring the `q_t`/`Δq_t` pattern of v3):

- `g_t` vs. `ρ_6`: `g_t` is an absolute, drift-free 2-DoF tilt reference,
  while `R_t` is reconstructed by *integrating* `ρ_6`; consistency requires
  `ρ_6^{-1}(...)·g_{t+1} = g_t`.
- `h_t` vs. `Δh_t`: absolute height reference vs. integrated increment.

These three redundancies are now enforced explicitly by the temporal
consistency loss (§4.3.3), and the absolute channels win at
reconstruction.

### 4.2 6D convention and implementation

**Convention (this codebase): the first two COLUMNS.** To avoid adding a
dependency and to keep a single convention across the repo, v7 reuses the
existing conversion pair in `robotmdar/dtype/rotation.py`, used everywhere
(data preprocessing, VAE reconstruction, losses):

```python
from robotmdar.dtype.rotation import rot6d_to_matrix, matrix_to_rot6d

R     = rot6d_to_matrix(rot6d)   # [..., 6] -> [..., 3, 3]
rot6d = matrix_to_rot6d(R)       # [..., 3, 3] -> [..., 6]
```

- `matrix_to_rot6d` keeps the first two **columns** of the matrix and
  flattens that 3×2 submatrix row-major:
  `[r11, r12, r21, r22, r31, r32]`. Equivalently, reshaping the 6D vector
  to `(..., 3, 2)` yields the two rotation-matrix columns as its two
  columns.
- `rot6d_to_matrix` treats the two 3-vectors as the first two columns,
  Gram–Schmidt orthonormalizes them, and takes the third column as their
  cross product — the result lies in SO(3) for any non-degenerate input.
- These two functions were generalized to arbitrary leading dimensions
  `(..., 3, 3)` / `(..., 6)` (previously B×T only); all existing callers
  use B×T shapes, so behavior is unchanged.

**Critical:** this column convention differs from
`pytorch3d.transforms.matrix_to_rotation_6d`, which takes the first two
**rows**. The two conventions differ by a transpose, so mixing them
**silently corrupts rotations**. PyTorch3D is deliberately not introduced;
if it ever appears in the pipeline, the conversion functions must still
always be used as a pair from this repo.
`dataset/data_analyze/analyze_rpy.py` concatenates the two columns
column-major (`[r11, r21, r31, r12, r22, r32]`) — a third ordering that is
only used for the continuity figure and never feeds these functions (its
continuity conclusion is convention-independent).

Round-trip test for valid rotation matrices (also enforced by
`TextOpRobotMDAR/test/test_rot6d.py`):

```python
R2 = rot6d_to_matrix(matrix_to_rot6d(R))
print(torch.max(torch.abs(R - R2)))   # expect ~1e-6
```

Reference implementation: `papers/rotation_representation.pdf`; theory:
Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019). *On the Continuity
of Rotation Representations in Neural Networks*. CVPR. arXiv:1812.07035.

**Alignment with the SONIC controller.** SONIC (`papers/sonic.pdf`)
receives the 6D rotation representation as its interface format: its
proprioceptive state includes the gravity vector in the root frame and uses
"the 6D rotation representation (Zhou et al., 2019) throughout", with all
quantities expressed in the robot's local frame for rotation invariance.
v7's `g_t` + `ρ_6` design therefore matches the controller's native format.
Verify SONIC's 6D row/column convention at integration time — do not assume
it matches this repo's column convention without a check.

### 4.3 Loss design

The overall VAE objective keeps the paper's form
`L_VAE = λ_rec·L_rec + λ_KL·L_KL + L_geo` (Eq. 4), extended with the
v7-specific terms — the rotation chordal term (§4.3.2) and the temporal
consistency terms (§4.3.3) — and with the geometry terms recomputed on the
v7 reconstruction path. Terms defined so far:

#### 4.3.1 Reconstruction loss (Huber) — primary

We adopt the Huber loss (smooth L1 loss) as the reconstruction objective
to measure the discrepancy between the predicted future motion features
`f̂` and the ground truth `f` (paper Eq. 5):

$$
L_{\mathrm{rec}}
= \mathrm{Huber}\!\left(
\hat{\mathbf f}_{\,t\,:\,t+T_{\mathrm{future}}-1},\;
\mathbf f_{\,t\,:\,t+T_{\mathrm{future}}-1}
\right)
$$

- **Identical to the original code path**:
  `rec_criterion = nn.HuberLoss(reduction='mean', delta=1.0)`
  ([train/manager.py:80](../../TextOpRobotMDAR/robotmdar/train/manager.py#L80)),
  applied directly as
  `rec_loss = rec_criterion(future_motion_pred, future_motion_gt)`
  ([train/manager.py:754](../../TextOpRobotMDAR/robotmdar/train/manager.py#L754)),
  weight `loss_weight.rec = 1.0` (`config/train/mvae.yaml`).
- **No post-processing**: the decoder output is the plain final linear
  layer
  ([mld_vae.py:209](../../TextOpRobotMDAR/robotmdar/model/mld_vae.py#L209),
  [wrapper/vae_decode.py:45](../../TextOpRobotMDAR/robotmdar/wrapper/vae_decode.py#L45)),
  and the rec loss compares it directly against the mean/std-normalized GT
  feature — the dataset normalizes at collation
  ([data.py:1295](../../TextOpRobotMDAR/robotmdar/dataloader/data.py#L1295))
  and `calc_loss` passes both tensors straight to the criterion
  ([manager.py:754](../../TextOpRobotMDAR/robotmdar/train/manager.py#L754)).
  Denormalization, `atan2`, FK and Gram–Schmidt appear only in the
  geometry-loss path — which denormalizes explicitly before FK
  ([manager.py:475](../../TextOpRobotMDAR/robotmdar/train/manager.py#L475)) —
  and in evaluation reconstruction, never in `L_rec`. This is exactly the
  v3 division of labor, kept unchanged for v7: Huber terms in normalized
  feature space, denormalization only where physical quantities are
  needed.
- **Consequence for v7**: `ρ_6` and `g_t` are supervised in raw feature
  space; the deterministic projections below are applied only at use-time,
  so `L_rec` itself needs no extra post-processing.

**Raw-space supervision, deterministic projection at use.** The decoder
outputs an ordinary Euclidean vector `m̃_t`, and `L_rec` supervises this raw
output directly. This is deliberate, for two reasons:

1. the VAE learns a deterministic, canonical Euclidean feature distribution
   — the regression target is well-defined;
2. no projection null-space can appear, so the raw output scale cannot
   drift arbitrarily — `L_rec` penalizes it. For example the GT gravity
   satisfies `‖g_t‖ = 1`, so a raw output `[0, 0, −5]` is penalized by the
   raw reconstruction loss even though its direction is correct after
   normalization. This keeps the VAE latent space well-behaved.

Division of labor: the Huber terms (`L_rec`, `L_rot,6D`) are computed in
**normalized** feature space and need **no denormalization**;
denormalization enters only the projection-based quantities (the chordal
term §4.3.2, the consistency terms, and the use-time reconstruction).

Only *before* formal use — reconstruction, FK, inference — is the raw
vector deterministically projected. The projections act on the
**denormalized** feature (denormalize first, then project): per-channel
mean/std normalization is an affine transform with non-trivial offsets,
and normalization, Gram–Schmidt and the tangent-plane projector are all
nonlinear, so they must see physical values (see §4.3.2). In this order:

$$
\boxed{\hat{\mathbf g}_t = \frac{\tilde{\mathbf g}_t}{\|\tilde{\mathbf g}_t\| + \epsilon}},
\qquad
\boxed{\hat{\mathbf R}_t^{\mathrm{rel}} = \mathrm{GS}(\tilde{\mathbf r}_t^{\mathrm{6D}}) \in SO(3)},
\qquad
\boxed{\widehat{\Delta\mathbf p}_t^{\mathrm{hor}} = (I - \hat{\mathbf g}_t \hat{\mathbf g}_t^\top)\,\widetilde{\Delta\mathbf p}_t^{\mathrm{hor}}}
$$

so that

$$
\|\hat{\mathbf g}_t\| = 1,
\qquad
\hat{\mathbf R}_t^{\mathrm{rel}} \in SO(3),
\qquad
\hat{\mathbf g}_t^\top \widehat{\Delta\mathbf p}_t^{\mathrm{hor}} = 0
$$

hold **strictly** — they are not "learned" by the VAE. The horizontal
projection uses the *projected* gravity `ĝ_t`, so `ĝ_t` must be projected
first; `ε` is a numerical guard only (`F.normalize(..., eps=1e-8)`
implements exactly this division, matching the eps used in
`rot6d_to_matrix`, §4.2). The 4.3.2 chordal term likewise operates on the
projected `R̂` built from the denormalized 6D (§4.3.2), consistent with
this policy. After training, the
*denormalized* outputs are near-valid (`‖g_t‖ ≈ 1`, `r^6D`
near-orthonormal) because `L_rec` supervises them against exactly such
targets, so the projection is near-identity in the typical regime and acts
as a safety net rather than a re-mapping.

#### 4.3.2 Rotation loss — two-layer supervision

Direct 6D supervision is the primary term, in **normalized** feature
space — the Huber loss needs no denormalization:

$$
L_{\mathrm{rot,6D}}
= \mathrm{Huber}\!\left(\tilde{\mathbf r}^{\mathrm{6D}},\; \tilde{\mathbf r}_{\mathrm{GT}}^{\mathrm{6D}}\right),
$$

and the projected matrix receives a small-weight chordal loss,

$$
\hat{\mathbf R} = \mathrm{GS}(\bar{\mathbf r}^{\mathrm{6D}}),
\qquad
\bar{\mathbf r}^{\mathrm{6D}} = \mathrm{denorm}(\tilde{\mathbf r}^{\mathrm{6D}}),
\qquad
L_{\mathrm{rot,chord}} = \|\hat{\mathbf R} - \mathbf R_{\mathrm{GT}}\|_F^2,
$$

combined as

$$
L_{\mathrm{rot}} = L_{\mathrm{rot,6D}} + \lambda_{\mathrm{chord}}\, L_{\mathrm{rot,chord}},
\qquad \lambda_{\mathrm{chord}} \ll 1 .
$$

The interpretation is clear: the 6D term guarantees *representation
fidelity*, the chordal term guarantees *geometric correctness* of the
actual rotation.

**Implementation note (code-verified):** the 6D supervision is already
contained in the §4.3.1 reconstruction loss — `L_rec` applies the Huber
criterion to the full feature tensor with no per-channel exclusion
([train/manager.py:754](../../TextOpRobotMDAR/robotmdar/train/manager.py#L754)),
and the 6D channels are part of `m_t`. A separate `L_rot,6D` term is
therefore redundant; only the chordal term needs to be added:

```python
# the 6D channels are already supervised inside L_rec — no separate L_rot,6D
r_6d = denormalize(rot6d_slice(model_output))    # per-channel inverse mean/std
pred_R = rot6d_to_matrix(r_6d)                   # GS -> valid SO(3)
loss_chordal = ((pred_R - gt_R) ** 2).sum(dim=(-1, -2)).mean()   # ||.||_F^2
loss_rot = lambda_chord * loss_chordal           # lambda_chord << 1 (start 0.01-0.05)
```

**Denormalize before Gram–Schmidt.** GS must see the denormalized 6D
vector, not the normalized feature. At 50 Hz the relative rotation
`^{E_t}R_{E_{t+1}}` is close to the identity, so the per-channel means are
far from zero (three channels ≈ 1, three ≈ 0); the per-channel affine
normalization adds constant offsets that distort the column directions,
and GS would orthonormalize the wrong columns. The chordal term must
measure the same rotation the inverse algorithm will actually use.

**Reconstructed-trajectory rotation loss (upgraded `body_rot`).** The
local term above supervises the single-step rotation increment; the
original TextOp `body_rot` term is upgraded to supervise the
*reconstructed root orientation trajectory* — the accumulated integration
result:

$$
L_{\mathrm{body\text{-}rot}} = \sum_t \left\| \hat{\mathbf R}_t - \mathbf R_t^{\mathrm{GT}} \right\|_F^2 .
$$

The two are complementary, exactly like supervising `Δq_t` and the `q_t`
trajectory: one is local rotational dynamics, the other is the accumulated
orientation trajectory. The chordal distance is invariant to a common
world rotation (`‖QR_1 − QR_2‖_F = ‖R_1 − R_2‖_F`), so as long as
prediction and GT share the same initial frame, this term does not break
the SE(2) invariance of the representation. (The wording is deliberately
"reconstructed root orientation trajectory", not "global rotation".) The
remaining FK-based geometry terms (`body_trans`, `dof_pos`, `dof_vel`,
`foot_contact`) are representation-agnostic and stay as in v3; the final
term set and weights are listed in §7.

#### 4.3.3 Temporal consistency loss

A legal motion feature satisfies three temporal constraints — the place
where v7 can explicitly strengthen TextOp:

$$
\boxed{\mathbf q_{t+1} = \mathbf q_t + \Delta\mathbf q_t}
\qquad
\boxed{h_{t+1} = h_t + \Delta h_t}
\qquad
\boxed{\mathbf g_{t+1} = (\mathbf R_t^{\mathrm{rel}})^\top \mathbf g_t}
$$

The third is the most important: it is the SO(3) analogue of the two
Euclidean increment consistencies — it follows from
`R_{t+1} = R_t · R_t^rel` and `g_t = R_t^⊤ g`, and expresses the same
"absolute state advances by the local transition" law in rotation space.
The new representation can therefore be summarized as:

**absolute state + local transition + temporal self-consistency.**

All consistency terms use the projected legal variables (§4.3.1) and —
mandatory — are computed **after denormalization**: with per-channel
normalization, `q̂_{t+1} − q̂_t` is in general **not** equal to `Δq̂_t`,
because the two channels carry different per-channel statistics. This is
the same conclusion as the TextOp `dof_delta` analysis, and the original
code already follows it: its temporal-delta terms are computed on
`reconstruct_motion` output, i.e. in physical units
([manager.py:692-706](../../TextOpRobotMDAR/robotmdar/train/manager.py#L692-L706)).

$$
L_{\mathrm{temp}} = \lambda_g L_{g\mathrm{-cons}} + \lambda_h L_{h\mathrm{-cons}} + \lambda_q L_{q\mathrm{-cons}}
$$

$$
L_{g\mathrm{-cons}} = \sum_t \left\| \hat{\mathbf g}_{t+1} - (\hat{\mathbf R}_t^{\mathrm{rel}})^\top \hat{\mathbf g}_t \right\|^2,
$$

$$
L_{h\mathrm{-cons}} = \sum_t \mathrm{Huber}\!\left( \hat h_{t+1} - \hat h_t,\; \widehat{\Delta h}_t \right),
\qquad
L_{q\mathrm{-cons}} = \sum_t \mathrm{Huber}\!\left( \hat{\mathbf q}_{t+1} - \hat{\mathbf q}_t,\; \widehat{\Delta\mathbf q}_t \right),
$$

**Pure self-consistency.** All three terms compare predictions with
predictions — the division of duties is clean:
`L_rec`: prediction vs. GT; `L_temp`: prediction vs. prediction.
GT-anchored variants are deliberately excluded: GT supervision of the
increments is already provided by `L_rec` on the `Δq`/`Δh` channels, so
anchoring would re-supervise the same quantities with little information
gain.

The sums run over `t` in the predicted future window. The gravity term
uses squared L2 — for unit vectors `‖a − b‖² = 2(1 − a^⊤b)` is bounded,
so no Huber robustness is needed; the joint/height terms use the Huber
criterion with the difference of the absolute predictions as input and the
predicted increment as target, matching the original
`trans_delta`/`joints_delta`/`dof_delta` pattern
([manager.py:704-706](../../TextOpRobotMDAR/robotmdar/train/manager.py#L704-L706)).

**Huber delta per variable, not a single `1.0`.** A physical-space
`delta = 1.0` is meaningless across channels (1 rad ≈ 57° vs. 1 m are
incomparable scales). First-run values: `δ_q = 0.1–0.2 rad`,
`δ_h = 0.02–0.05 m` (already generous single-step discrepancies at
50 Hz). Weights `λ_q = λ_h = λ_g = 0.05–0.1`; tune from the unweighted
loss magnitudes and gradient magnitudes, not from the final weighted
values.

**Cross-boundary pairs (history → future).** Following the original
temporal-delta pattern, the consistency sums include the boundary pair
`H → F` (last history frame to first future frame):
`L_{q-cons}^boundary = Huber(q̂_F − q_H, Δq_H)` and likewise for height
and gravity, where the history side uses the known, projected
authoritative state `q_H, h_H, g_H` and the transition `Δ_H` is read from
the history feature's Δ channels. Code-verified: the v3 encode puts the
forward difference `Δq_t = q_{t+1} − q_t` into the feature at frame `t`
([motion.py:344](../../TextOpRobotMDAR/robotmdar/dtype/motion.py#L344)),
the decode reads it back directly
([motion.py:425](../../TextOpRobotMDAR/robotmdar/dtype/motion.py#L425)),
and the temporal-delta loss concatenates `history[-1]` with the
predictions
([manager.py:694-706](../../TextOpRobotMDAR/robotmdar/train/manager.py#L694-L706)),
so the first pair is exactly `H → F`. **There is no availability gap at
inference.** The `H` history features correspond to `H+1` states (the
features are built by forward differences), and the current state `s_now`
is always observable — it is exactly the endpoint of the last history Δ.
The boundary pair therefore compares the model's predicted first future
state against an *observed* transition (`q̂_F − q_H` vs.
`Δq_H = q_now − q_H`), in identical form at training and inference, for
the VAE and the LDM planner alike. No masking or special convention is
needed.

#### 4.3.4 Training diagnostics (log-only, not loss)

Log these from the first run — they localize problems far earlier than
the total VAE loss. Consistency residuals (physical units, denormalized
+ projected variables):

$$
e_g^{\mathrm{cons}} = \| \hat{\mathbf g}_{t+1} - \hat{\mathbf R}_t^{\mathrm{rel}\top}\hat{\mathbf g}_t \|,
\qquad
e_h^{\mathrm{cons}} = | \hat h_{t+1} - \hat h_t - \widehat{\Delta h}_t |,
\qquad
e_q^{\mathrm{cons}} = \| \hat{\mathbf q}_{t+1} - \hat{\mathbf q}_t - \widehat{\Delta\mathbf q}_t \|,
$$

and projection corrections:

$$
e_{g,\mathrm{proj}} = \| \tilde{\mathbf g} - \hat{\mathbf g} \|,
\qquad
e_{R,\mathrm{proj}} = \| \tilde{\mathbf r}^{\mathrm{6D}} - \rho_6(\mathrm{GS}(\tilde{\mathbf r}^{\mathrm{6D}})) \|,
\qquad
e_{p,\mathrm{proj}} = | \hat{\mathbf g}^\top \widetilde{\Delta\mathbf p}^{\mathrm{hor}} | .
$$

Expected behavior: as `L_rec` decreases, the projection corrections go
to 0 — the denormalized decoder output approaches a legal feature
(§4.3.1). If `L_rec` is low but a projection correction stays large,
normalization, loss weighting or the representation itself is broken
somewhere.

### 4.4 Channel normalization: std floor

`g_t` and the 6D channels stay in the standard per-channel mean/std
normalization — the rec metric lives in normalized feature space, and the
v3 sin/cos bounded channels were normalized the same way. One refinement:
near-constant channels get a **std floor** instead of a frozen zero-mean
statistic:

$$
\sigma_i^{\mathrm{eff}} = \max(\sigma_i, \sigma_{\min}),
\qquad
x_i^{\mathrm{norm}} = \frac{x_i - \mu_i}{\sigma_i^{\mathrm{eff}}} .
$$

For a relative-rotation 6D channel with `μ ≈ 1`, `σ = 3e-4`, this centers
at the true mean (`x − 1`) without dividing by a tiny std — it neither
amplifies numerical/noise variation nor introduces an artificial DC
offset, and the inverse normalization stays natural. `σ_min` is **not**
fixed a priori: first inspect the per-dim std distribution of
`ρ_6(R_t^⊤ R_{t+1})` at 50 Hz and pick the floor from the percentiles. The
std floor is a **numerical safeguard, not a feature-weighting device**;
relative importance of rotation/translation/joint channels should be
controlled explicitly by group loss weights. (The 6D identity layout
depends on the §4.2 convention — never hardcode an assumed `[1,0,...]`
pattern; always obtain it via `matrix_to_rot6d`.)

### 4.5 Pre-training unit tests (required before the first run)

Two property tests on random GT trajectories must pass before VAE
training starts:

1. **Round-trip.** For random GT trajectories `X`: encode `m = F(X)`,
   decode `X' = I(m, (p_0, R_0))`, and check `max|p − p'|`,
   `d_SO(3)(R_t, R'_t)` and `max|q − q'|` at near-floating-point
   precision (≤ 1e-5).
2. **Horizontal SE(2) invariance.** Apply a random horizontal world
   transform `R_t' = Rz(θ)·R_t`, `p_t' = Rz(θ)·p_t + t_xy` and check
   `m(X') ≈ m(X)`.

Together these verify the doc's two core claims — exact invertibility on
valid trajectories (§2.4) and global horizontal SE(2) invariance —
directly on the implemented encode/decode pair, before any loss or model
is involved.

## 5. Dataset rotation convention (verified)

- BONES-SEED CSVs store `root_rotate{X,Y,Z}` as **extrinsic xyz Euler
  angles in degrees** — per the official SEED G1 CSV spec (ProtoMotions
  docs, "SEED G1 CSV Data Preparation": "Root orientation as extrinsic
  XYZ Euler angles in degrees", tooling `--euler-order` default `xyz`).
- SciPy convention (synthetic-verified below): **lowercase `xyz` =
  extrinsic** (`R = Rz(γ)·Ry(β)·Rx(α)`); **uppercase `XYZ` = intrinsic**
  (`R = Rx(α)·Ry(β)·Rz(γ)`). Note this is the opposite of what the
  letter case suggests.
- The conversion pipeline
  ([dataset/data_process/convert_soma_csv_to_motion_lib.py:217-228](../../dataset/data_process/convert_soma_csv_to_motion_lib.py#L217-L228))
  calls `Rotation.from_euler("xyz", euler_deg, degrees=True)` — i.e. it
  builds the **extrinsic** composition, which **matches** the CSV
  convention. The computation is correct; the "intrinsic" label in the
  code comment (and in an earlier version of this doc) is a mislabel.
  It then stores a quaternion; the TextOp pack format carries
  `root_rot [T, 4]` (xyzw), so downstream `quaternion_to_matrix` inherits
  the correct rotation.
- Synthetic verification, with non-commuting angles (a single-axis test
  cannot distinguish the two conventions):

  ```python
  a = np.deg2rad([30.0, 40.0, 50.0])
  Rx = R.from_euler('x', a[0]).as_matrix()
  Ry = R.from_euler('y', a[1]).as_matrix()
  Rz = R.from_euler('z', a[2]).as_matrix()
  assert np.allclose(R.from_euler('xyz', a).as_matrix(), Rz @ Ry @ Rx)  # extrinsic
  assert np.allclose(R.from_euler('XYZ', a).as_matrix(), Rx @ Ry @ Rz)  # intrinsic
  ```
- `robotmdar` currently re-decomposes this quaternion with ZYX Tait-Bryan
  formulas (`robotmdar/dtype/rotation.py: get_euler_xyz`) — the labels
  roll/pitch/yaw are the ZYX decomposition of the same rotation matrix, so
  no information is lost in the round trip.
- **v7 plan:** the dataset carries `R_t` (3×3) uniformly, obtained from
  `quaternion_to_matrix(root_rot_xyzw)` (or directly from Bones-SEED's
  `rotateXYZ` at the source for cross-validation). All v7 forward/inverse
  code operates on `R_t` only; no Euler extraction is performed.

## 6. Backward compatibility: config-switchable representation (ablation)

v7 must coexist with the RPY v3 representation, selectable via config for
ablation studies:

- The dispatch registry already exists: `FeatureVersion` +
  `motion_dict_to_feature` / `motion_feature_to_dict` pairs
  ([robotmdar/dtype/motion.py:1371-1400](../../TextOpRobotMDAR/robotmdar/dtype/motion.py#L1371-L1400)).
  Add `FeatureVersion 6` with `motion_dict_to_feature_v6` /
  `motion_feature_to_dict_v6` and `motion_feature_dim_v6 = 14 + 2·n_q + n_c`
  (identifier 6, because 4 and 5 are taken by the dead experiments in §8).
- Add a config key (e.g. `feature_version: 3` in `train/mvae.yaml`,
  `train/dar.yaml`, `planner_dar.yaml`) that selects the dispatch before
  dataset/model construction. The current mechanism is a module-level
  constant; the config key should be resolved at startup.
- Known hardcoded v3 guards that must become version-aware:
  `planner/planner_dar.py:214` (`FeatureVersion != 3` raise),
  `train/manager.py:1251` and `:1297` (trajectory integration, v3-only),
  `utils/planner_convert.py` (controller-history encoding, v3 layout).
- Per-version channel statistics: `dataloader/data.py` computes mean/std
  from the feature tensor, so v7 gets its own normalization automatically;
  checkpoints/models must record the feature version they were trained with.
- Downstream interfaces to keep parity across versions: history injection,
  goal encoding (`utils/goal.py`), FK reconstruction
  (`dataloader/data.py: reconstruct_motion`), and geometry losses
  (`train/manager.py`). For the v7 path the geometry losses operate on
  reconstructed FK outputs and are representation-agnostic, which is what
  makes the ablation clean.
- Ablation protocol: identical data, model, and training recipe; only the
  feature head width changes (69 vs. 74 channels for `n_q = 29`).

## 7. Open questions

1. **Anti-parallel re-alignment — resolved.** Deterministic
   orthogonal-axis π-rotation fallback with temporal axis reuse (§4.1).
2. **`h_t` vs. `Δh_t` at decode — resolved.** (§4.1)
3. **Channel normalization — resolved.** Per-channel mean/std with std
   floor, chosen from data percentiles (§4.4).
4. **Geometry term set — partially resolved.** FK-based terms unchanged;
   `body_rot` upgraded to reconstructed-trajectory chordal (§4.3.2).
   Final term set and weights to be confirmed in the first training runs.
5. **Rotation perturbation — scope set, method pending.** Rotation
   perturbation applies **only in LDM training**; the VAE is trained on
   clean data so that it learns the clean dataset distribution. This is
   already the v3 arrangement at config level: `augmentation_enabled`
   is set true only in the LDM config (`config/train_dar.yaml`), while
   the VAE config keeps the data path clean (`use_rollout: False`,
   `static_perturbation_scale: 0.0` in `config/train/mvae.yaml`; the
   one `perturb_feature_v3` call site, `manager.py:373-376`, is a
   no-op at scale 0.0). v7 keeps this split. Within
   LDM training it serves the **self-rollout** scenario only, matching
   the v3 DAR-dataloader pattern (`data.py`: augmented history, goals
   stay clean, ego-centric conditioning reset to the perturbed latest
   history state, final history delta kept clean because it references
   the first future pose, which lies outside the perturbed window). The
   concrete v7 perturbation operator for the rotation channels (must
   not be applied in RPY form) is still to be supplied; non-rotation
   channels keep the v3-style perturbation. This no longer blocks VAE
   training.
6. **Weights / Huber deltas — starting points set.** `λ_chord =
   0.01–0.05`, `λ_g = λ_h = λ_q = 0.05–0.1`, `δ_q = 0.1–0.2 rad`,
   `δ_h = 0.02–0.05 m`; gravity consistency without Huber (bounded on
   the unit sphere). Final values from unweighted loss magnitudes and
   gradient magnitudes (§4.3.3).
7. **LDM planner, goal encoding, controller interface — deferred** until
   the VAE is trained.
8. **BONES-SEED CSV Euler convention — resolved.** Extrinsic xyz,
   verified against the official SEED G1 CSV spec and a synthetic SciPy
   test; the conversion code's computation is correct, only its comment
   was mislabeled "intrinsic" (§5).

## 8. Relationship to the in-code FeatureVersion 4 and 5

`dtype/motion.py` already contains two *unused* feature candidates whose
version numbers must not be confused with this design (this doc's "v7" is
the doc serial; the feature version here is 6, see the naming note in §1).
Decision: keep both as-is — they are part of the `FeatureVersion` history
and v4 is the current in-repo rot6d reference implementation. v3 is the
only version any training run, config, or checkpoint has ever used.

### v4 (247-D, DART/MDM-style — dead code)

```
transl(3) | rot6d_abs(6) | dof(29) | transl_delta(3) | rot_delta_6d(6)
| joints(99) | joints_delta(99) | contact(2)           = 247
```

- `motion_dict_to_feature_v4` canonicalizes the whole sequence into a
  per-sequence frame (origin at the first joint, x-axis along the hip
  line, `get_new_coordinate`), then stores the **absolute** root
  orientation as 6D per frame plus the **relative** rotation as 6D
  (12 rotation channels), global FK joint positions and their deltas
  (198 channels), translation and its delta — the same kinematics three
  times over.
- The decode side needs an Euler yaw extraction
  (`extract_yaw_from_rotation`, ZYX with explicit gimbal-lock branches)
  to align the predicted sequence with `abs_pose` — v4 removes RPY from
  the feature but reintroduces Euler decomposition at reconstruction.
- v4's relative rotation is computed as `R_{t+1} R_t^T`, the **transpose**
  of this design's `^{E_t}R_{E_{t+1}} = R_t^T R_{t+1}` (§4.1). The delta
  code must not be reused without transposing.
- Status: never wired in — `FeatureVersion = 3` is the active dispatch;
  the only consumers are a v4 branch in `reconstruct_motion`
  (`dataloader/data.py:943`) and commented-out blending code in
  `eval/vis_mvae.py` (a ported MDM/DART blending pipeline).

### v5 (270-D, yaw-local — dead code)

```
sincos_roll_pitch(4) | delta_yaw(1) | contact(2) | trans_local(3)
| delta_trans_local(3) | joints_local(99) | delta_joints_local(99)
| height(1) | dof(29) | delta_dof(29)           = 270
```

- Everything is expressed in the per-frame yaw-aligned frame
  (translation, displacement deltas, joint positions), which removes yaw
  from the absolute channels — but the rotation itself remains Euler:
  roll/pitch sincos plus `delta_yaw`.
- Also never wired in (`get_zero_feature_v5` exists, dispatch inactive).

### What this design takes from them

- The 6D conversion pair in `dtype/rotation.py` (generalized to
  arbitrary leading dims and covered by `test/test_rot6d.py`) is shared
  by all versions — this design reuses exactly those functions (§4.2).
- The yaw-local framing of v5 and the per-sequence canonicalization of
  v4 are superseded by the ego-frame `E_t` formulation: yaw-invariant by
  construction, no per-sequence canonicalization, no yaw re-alignment at
  decode.
- Unlike v4/v5 (which store FK joint positions, 99 + 99 channels), this
  design reconstructs the skeleton from the authoritative `q_t` via FK,
  keeping the v3 reconstruction principle.

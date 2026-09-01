# planner_v7: Rotation-Matrix Motion Representation (Drop RPY and redundant increments; arrival-state alignment)

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
abandons RPY entirely**. In the same pass it removes the `Δq`/`Δh`
increment channels that v3 carried but whose information is fully
redundant given the authoritative absolute state (§2.6, §4).

To serve the LDM goal interface, v7 also changes the feature's **time
alignment** (§4): from the departure-state convention (state at `t` plus
outgoing transition `t → t+1`) to the **arrival-state** convention
(incoming transition `t−1 → t` plus state at `t`, backward differences).
The final history feature then carries the current state `s_t` itself —
the unique egocentric reference for all planner goals — so no separate
current-state condition is needed (§2.8, §4, §7).

**Naming note:** "v7" is the serial number of this *planner design doc* —
the motion-feature version is a separate numbering, the `FeatureVersion`
constant in `robotmdar/dtype/motion.py` (active value: 6, set from
`config/base.yaml`). In code, 4 and 5
are already occupied by dead experiments (see §8), so the feature described
here is introduced as **FeatureVersion 6**, not 7.

## 2. Design principles

1. **No Euler angles, no yaw.** The feature contains no yaw channel and no
   trigonometric Euler encoding. Egocentric quantities are defined with
   respect to root frames — `E_t` for the absolute-state channels and the
   departure frame `E_{t−1}` for the incoming transition (§4) — not with
   respect to a yaw-only heading frame.
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
   override it (§4.1). v7 replaces the rotation representation and prunes
   the increments the authoritative state makes redundant (see §4).
6. **An increment only when no absolute channel covers the DoF.** An
   incremental channel is retained only when the corresponding degree of
   freedom has no absolute channel: vertical position has `h_t` → no
   `Δh_t`; joint configuration has `q_t` → no `Δq_t`; the yaw-like
   rotational DoF has no absolute channel (`g_t` covers only 2 of 3
   rotational DoF) → keep `^{E_{t-1}}R_{E_t}`; horizontal position is deliberately
   removed (SE(2) invariance, principle 3) → keep `Δp_{t-1}^hor`. The result
   is a minimal representation in which every transition channel
   transports a degree of freedom the absolute channels cannot observe.
   Which channels stay absolute is decided by the gauge criterion
   (principle 9); the time alignment of the transitions is the
   arrival-state convention (principle 8).
7. **State predicted once; dynamics derived, never re-predicted.** Every
   explicitly represented state or dynamic quantity enters the feature
   exactly once — with the single deliberate exception of the overlap
   between the absolute gravity tilt `g_t` and the full relative
   rotation `^{E_{t-1}}R_{E_t}` (§4.1), which is enforced as
   consistency rather than stored twice. Velocity- and
   acceleration-like quantities are derived deterministically from the
   authoritative state sequences at use time (`q̇ = 50(q_{t+1} − q_t)`,
   §4). Deleting an increment channel must never silently delete the
   dynamics supervision it incidentally carried — dynamics supervision
   moves to losses on the derived quantities (§4.3.5), not to extra
   channels.
8. **Arrival-state time alignment (backward differences).** Feature `m_t`
   carries the partial absolute state **at** `t` together with the
   **incoming** transition `t−1 → t`. A history window of states
   `s_{t−H:t}` therefore maps to features `m_{t−H+1:t}`: the final
   history feature `m_t` makes the current state `s_t` directly
   available to the planner, and every planner goal is expressed
   relative to `s_t` — the unique egocentric reference frame `E_t` — so
   no separate current-state conditioning input is needed. The earliest
   raw state `s_{t−H}` is used only to construct `m_{t−H+1}` and is
   intentionally discarded afterward: exact reconstruction of the
   earliest historical state is neither required nor desired — the
   representation preserves historical motion transitions while
   anchoring the current state.
9. **Absolute vs. relative: the gauge criterion.** Absolute channels are
   used for locally observable, physically anchored quantities whose
   magnitudes remain bounded over time (`h_t`, `g_t`, `q_t`, `c_t`);
   relative channels are used for spatial degrees of freedom whose
   absolute values depend on arbitrary global gauge choices (world
   origin, global heading) or can grow unbounded with trajectory extent
   (`Δp^hor`, `^{E_{t-1}}R_{E_t}`). This removes global SE(2) nuisance
   variation while avoiding the long-horizon drift of pure integration:
   an all-relative encoding must integrate to recover state and
   accumulates error every step, so stable local anchors stay absolute;
   an all-absolute encoding of global quantities forces the network to
   model a gauge choice that carries no physical information, so those
   become local changes. The absolute channels answer *where I am now*,
   the transition channels answer *how I moved here* (§4).

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

Note also that v3's `Δq_t` is computed by the forward algorithm but
**never read by the inverse**: Algorithm 2 reads `q_t` directly, and the
temporal-delta geometry terms re-derive velocities from the
reconstructed `q` sequence. `Δq_t` is not a reversibility requirement but
an extra dynamics feature; v7 removes it (§2.6, §4).

## 4. New v7 motion feature

$$
\boxed{
\mathbf m_t
=
\left[
h_t,\,
\mathbf g_t,\,
{}^{E_{t-1}}\Delta\mathbf p_{t-1}^{\mathrm{hor}},\,
\rho_6({}^{E_{t-1}}\mathbf R_{E_t}),\,
\mathbf q_t,\,
\mathbf c_t
\right]
}
$$

The absolute-state channels are indexed by their own frame `t`; the
transition channels carry the **incoming** step `t−1 → t`, expressed in
the departure frame `E_{t−1}` (arrival-state alignment, §2.8).

**Gravity tilt.** `g = [0, 0, −1]^⊤` is the world gravity direction. Its
expression in the current root frame,

$$
\mathbf g_t = \mathbf R_t^\top \mathbf g,
$$

is the root's tilt relative to gravity (a unit vector, 2 DoF). It replaces
the roll/pitch sincos pair of v3 and is singularity-free for all
orientations.

**Relative rotation.** The incoming frame-to-frame rotation (backward
difference) is

$$
{}^{E_{t-1}}\mathbf R_{E_t} = \mathbf R_{t-1}^\top \mathbf R_t,
$$

encoded with the 6D rotation representation `ρ_6` (Zhou et al., 2019): the
row-major flattening of the first-two-**columns** submatrix
`R[..., :, :2]`, i.e. `[r11, r12, r21, r22, r31, r32]` (the convention of
this codebase, see §4.2), recovered by Gram–Schmidt orthonormalization. The rotation matrix and the 6D representation are
equivalent parameterizations of the same rotation; the matrix is the
canonical SO(3) object, and `ρ_6` is only the form in which rotations
enter/leave the model. See §4.2 for the convention and implementation
details.

**Translation.** The full incoming local increment is

$$
{}^{E_{t-1}}\Delta\mathbf p_{t-1} = \mathbf R_{t-1}^\top ({}^{W}\mathbf p_t - {}^{W}\mathbf p_{t-1}).
$$

Because `E_{t−1}` may be arbitrarily tilted, its three coordinates can no
longer
be interpreted as x/y = horizontal, z = vertical. We therefore split it with
the local gravity direction. Define the horizontal projector

$$
P_{t-1}^{\mathrm{hor}} = I - \mathbf g_{t-1} \mathbf g_{t-1}^\top
$$

and decompose

$$
{}^{E_{t-1}}\Delta\mathbf p_{t-1}^{\mathrm{hor}} = (I - \mathbf g_{t-1} \mathbf g_{t-1}^\top)\,{}^{E_{t-1}}\Delta\mathbf p_{t-1},
\qquad
g_{t-1}^\top \Delta\mathbf p_{t-1}^{\mathrm{hor}} = 0,
$$

a 2D tangent-plane vector embedded in `R^3` — **no tangent-plane basis is
chosen**. The vertical increment is the scalar

$$
\Delta h_{t-1} = -\mathbf g_{t-1}^\top {}^{E_{t-1}}\Delta\mathbf p_{t-1} = h_t - h_{t-1},
$$

which is **not stored as a channel**: it is derivable from the
authoritative absolute heights of consecutive frames (with
`^Wg = [0,0,−1]`, `−g^⊤(p_t − p_{t−1}) = (p_t − p_{t−1})_z`). The full
displacement is recovered exactly:

$$
{}^{E_{t-1}}\Delta\mathbf p_{t-1}
=
{}^{E_{t-1}}\Delta\mathbf p_{t-1}^{\mathrm{hor}} - (h_t - h_{t-1})\,\mathbf g_{t-1},
$$

so `Δp_{t−1}^hor` together with the absolute `h` sequence is
information-equivalent to the original 3D increment, with the vertical
part re-anchored to the authoritative heights.

**Joint and contact channels.** `q_t` and `c_t` are unchanged from v3
(contact is moved to the end of the channel layout).
`Δq_t = q_{t+1} − q_t` is **deliberately dropped**: with `q_t` present
every frame it is fully redundant — the forward difference is derivable
from the authoritative `q` sequence, and a stored copy would only add
feature dimension, normalization statistics, a reconstruction target and
a consistency constraint, plus a way for the two channels to contradict
each other. v3's own inverse never reads `Δq_t` (Algorithm 2 reads `q_t`
directly, §3). If a later ablation shows that an explicit velocity-like
channel improves latent motion quality, it can be re-added as an
optional channel — the default is minimal.

Feature dimensionality: `1 + 3 + 3 + 6 + n_q + n_c` (= 44 for
`n_q = 29`, `n_c = 2`; v3 is 69, the earlier draft with `Δq`/`Δh`
was 74).

**Absolute state vs. transition.** The channels group naturally into the
two roles inherited from v3:

$$
\boxed{\text{absolute state: } (h_t,\; \mathbf g_t,\; \mathbf q_t,\; \mathbf c_t)}
\qquad
\boxed{\text{transition: } (^{E_{t-1}}\Delta\mathbf p_{t-1}^{\mathrm{hor}},\; {}^{E_{t-1}}\mathbf R_{E_t})}
$$

Reconstruction is therefore **neither pure integration nor pure per-frame
absolute prediction**: the absolute state suppresses the long-horizon drift
that integration accumulates, while the two increment channels transport
exactly the degrees of freedom the absolute state cannot observe — the
horizontal SE(2) position (principle 3) and the yaw-like rotational DoF
(`g_t` covers only 2 of 3 rotational DoF, §2.6). At reconstruction the
absolute state is **authoritative** (§4.1). The absolute channels answer
*where I am now*; the incoming transition answers *how I moved here*
(§2.9).

**Time alignment: arrival state (backward differences).** The absolute
channels sit at the transition's *endpoint*: feature `m_t` bundles the
incoming transition `t−1 → t` with the partial absolute state `s_t`.
Contrast the earlier departure-state draft, `m_t = (s_t, t → t+1)`: there
the last history feature stops one transition short of the current state —
`{m_0, …, m_{t−1}}` does not determine `s_t`, so the planner would need a
separate current-state condition or a one-frame-delayed goal reference.
The arrival-state alignment removes that asymmetry. For a history window
`s_{t−H:t}`, motion features are constructed as `m_{t−H+1:t}` using
backward differences. Each feature `m_k` contains the partial absolute
state at time `k` together with the incoming transition `k−1 → k`. The
earliest raw state `s_{t−H}` is used only to construct `m_{t−H+1}` and is
intentionally discarded afterward. Exact reconstruction of the earliest
historical state is neither required nor desired; the representation is
designed to preserve historical motion transitions while making the
current state `s_t` directly available through the final feature `m_t`.
All planner goals are therefore expressed relative to `s_t`, which serves
as the unique egocentric reference state `E_t` — no separate current-state
condition is needed (§2.8, §7). The future window is `m_{t+1:t+F}`: the
first predicted feature describes the outgoing transition from the
current state together with the arrival state `s_{t+1}`, which is exactly
what the planner predicts — *given `s_t`, predict the next transition and
the next state*.

The alignment also completes the window endpoint: states `s_{0:T}`
produce features `m_{1:T}`, and `m_T` carries the full authoritative
state of `s_T` — the final state of a window is fully reconstructed
(§4.1), where the departure-state draft left it unrecoverable. In one
line: **history transitions explain how we arrived here; `m_t` tells us
where we are now; the goal tells us where to go from here.**

Because the dataset/control rate is fixed at **50 Hz**, the increment
channels carry a fixed, known time scale (`Δt = 20 ms`), so no explicit
`Δt` input channel is required.

**Derived velocities.** Velocity-like quantities used by losses and
downstream consumers are derived at use time, never stored:
`q̇_t = 50(q̂_{t+1} − q̂_t)` from the authoritative `q` sequence,
`v_t^vert = 50(ĥ_{t+1} − ĥ_t)` from the authoritative `h`, and
`^{E_{t-1}}v̂^hor_{t-1→t} = 50·Δp̂^hor_{t-1}` from the horizontal
increment channel — the velocity of the interval `[t−1, t]` in its
departure frame `E_{t−1}` (the arrival-frame reading, if ever needed, is
`^{E_t}v̂^{hor,in}_t = 50·(^{E_{t-1}}R̂_{E_t})^⊤ Δp̂^hor_{t-1}` — §4.6's
goal velocity generalizes this to the full displacement) — the only
place horizontal velocity can come from, since absolute horizontal
position is deliberately absent. (Hats mark denormalized — and, where applicable,
projected — predictions; notation in §4.3.1.) Two v3 conventions exist and are kept distinct:
the Δ channels and the `dof_delta`-style consistency terms compare
**raw per-frame differences** (no fps factor — the rate only enters
implicitly through the fixed 50 Hz sampling), while the FK path divides
the difference by `dt = 1/fps` to get true physical velocities
([forward_kinematics.py:307-308](../../TextOpRobotMDAR/robotmdar/skeleton/forward_kinematics.py#L307-L308)).
The controller receives no TextOp-derived velocities at all: the plan
payload is the authoritative state trajectory, and SONIC derives
velocities with its own convention (§4.6).
The `×50` quantities above follow the FK convention (m/s and
rad/s); with no Δ channels left, the raw-difference convention survives
only as the source of `^{E_{t-1}}v̂^hor_{t-1→t}`, the `Δp̂_{t-1}^hor`
channel itself.
All velocities of the reconstructed *state sequence* remain forward
differences of that sequence (`q̂_{t+1} − q̂_t` etc.) — the feature's
backward-difference alignment does not change them; only
`^{E_{t-1}}v̂^hor_{t-1→t}` is read off the increment channel, which now
carries the step `t−1 → t`.
Supervision of these derived quantities is covered in §4.3.5.

### 4.1 Inverse algorithm (sketch): authoritative channels

Given the initial pose `(p_0, R_0)` and the **denormalized**
predicted features `{m̄_t}_{t=1..T}` (§4.3.1), first deterministically
project: `ĝ_t`, `^{E_{t-1}}R̂_{E_t} = GS(r̄_t^6D)`, `Δp̂_{t-1}^hor` (the
denormalized 6D slice `r̄_t^6D` encodes the incoming relative rotation
`^{E_{t-1}}R_{E_t}`, §4.2). Then:

**Role of the initial pose.** The supplied initial pose is the
**complete initial state `s_0`**. No feature channels exist for frame 0
— features start at `m_1` — so there is nothing to re-anchor: `(p_0,
R_0)` supplies the root position and orientation of frame 0 outright,
including its height and tilt, which the departure-state draft had
overwritten with `ĥ_0`/`ĝ_0` from a feature `m_0` that no longer
exists. Frame 0's joint configuration `q_0` and contact `c_0` are
equally absent from the features: reconstruction produces states
`1..T`, and the initial joint state is taken from the observed context
alongside the initial pose by any consumer that needs it (evaluation
geometry, controller warm-start) rather than reconstructed — the
earliest state is context, not a prediction target (§2.8). The initial
state's only role in the features is as the departure state of the
first transition `m_1`. Compared with TextOp v3 (Algorithm 2), which
uses the supplied pose only for the gauge freedoms and overwrites the
rest from feature channels:

| TextOp v3 | v7 |
|---|---|
| `p_init,xy` | `p_0` — the complete initial position (x/y **and** z) |
| `h_0` (feature) | — (no feature: `(p_0)_z` is the authoritative initial height) |
| `yaw_init` | `R_0` — the complete initial orientation (residual yaw **and** tilt) |
| roll/pitch (feature) | — (no feature: `R_0`'s tilt is authoritative) |
| `Δp^local` | `Δp̂^hor` |
| `Δψ` | `^{E_{t-1}}R̂_{E_t}` |

There is no re-anchoring step: frame 0 has no feature channels, so the
supplied initial pose is authoritative by construction, and for legal
forward-encoded trajectories it is exactly the ground-truth initial
state.

Then for `t = 1..T`:

1. `R'_t = R_{t−1} · ^{E_{t−1}}R̂_{E_t}` (integration);
2. re-align `R'_t` to the **authoritative** tilt: with
   `a = (R'_t)^⊤ g` (integrated gravity) and `b = ĝ_t`
   (authoritative gravity, read from feature `t`), apply the minimal
   rotation `S` mapping `b`
   onto `a` (axis `b × a`, angle `atan2(‖b × a‖, b^⊤ a)` — more
   stable than `acos`, see the fallback below) and set
   `R_t = R'_t · S` — the gravity direction comes from the
   absolute channel, the remaining (yaw-like) DoF from integration.
   Equivalently, `R_t = Π_{M(ĝ_t)}(R'_t)`, the projection
   onto the fiber `M(g) = {R ∈ SO(3) : R^⊤ g_W = g}`. The supplied
   `R_0` plays the role of the provisional orientation at the first
   step (`t = 1`), so the whole trajectory obeys one principle:

   $$
   \boxed{\text{absolute gravity defines the fiber; initial/incremental rotation selects a member within it.}}
   $$

   The anti-parallel case uses the deterministic fallback below;
3. `Δp_{t−1} = Δp̂_{t−1}^hor − (ĥ_t − ĥ_{t−1})·ĝ_{t−1}` — the full increment in the ego
   frame `E_{t−1}`; the vertical part comes from the authoritative heights of
   the two states being connected, not from any stored channel (§4).
   Integrate **in world coordinates**:

   $$
   \boxed{{}^W p'_t = {}^W p_{t−1} + R_{t−1}\,\left(\widehat{\Delta\mathbf p}_{t−1}^{\mathrm{hor}} - (\hat h_t - \hat h_{t−1})\,\hat{\mathbf g}_{t−1}\right)}
   $$

   (the factor `R_{t−1}` is essential: the local vector must be rotated into
   the world frame before accumulation). Then re-anchor the result along
   the gravity direction so that its ground-relative height equals the
   authoritative `ĥ_t` — the height of the state being constructed,
   read from feature `t`, the state's own feature index (the
   departure-state draft had to read ahead to feature `t+1` and shift
   the anchor index by one; the arrival-state alignment removes that
   shift). The re-anchoring also removes any residual vertical
   component in `Δp̂_{t−1}^hor`. With the world convention `^Wg = [0,0,−1]`
   and ground at `z = 0` this is simply `p'_t[..., 2] = ĥ_t`; the
   method itself is defined by gravity, not by the world-axis convention.
   The loop runs over `t = 1..T`: the last feature `m_T` reconstructs
   the final state completely (endpoint semantics below), and at the
   first step `ĥ_0 = (p_0)_z` comes from the supplied initial pose, so
   `ĥ_{t−1}` is always available.
4. `q̂_t`, `ĉ_t` read directly from feature `t` (denormalized;
   no projection — `q` is unconstrained continuous). `c` is read as
   **continuous contact scores**: the decoder regresses them like any
   other channel, and binarization (if a downstream consumer needs
   `{0,1}` contacts) is applied only at use/evaluation, never inside
   the reconstruction.

**Anti-parallel fallback.** Note that an upside-down root
(`g_t ≈ [0,0,+1]`) is *not* the degeneracy: the re-alignment degenerates
only when the two vectors to be aligned satisfy `a ≈ −b` (integrated
gravity vs. authoritative gravity pointing in opposite directions).
The **degeneracy criterion is numerical, not angular**: the normal
shortest-arc path (axis `b × a`, angle `atan2(‖b × a‖, b^⊤ a)`) is
used for every pair whose cross product is still resolvable; the
π-fallback below is entered only when `‖b × a‖` is too small to
normalize the axis reliably **and** `b^⊤ a ≈ −1`. Using a π-rotation
while `b × a` is still resolvable would leave a residual gravity
mismatch and break the strict `R_t^⊤ g_W = ĝ_t` guarantee. The
`a ≈ −b` case can only arise from a model output that contradicts
itself (or numerical corruption): for any legal forward-encoded
trajectory `a = b` exactly up to numerical precision, **regardless of
the physical magnitude of the frame-to-frame rotation** — no rate or
small-angle assumption. Algebraically, `a = (R_{t−1}·^{E_{t−1}}R_{E_t})^⊤ g_W
= (^{E_{t−1}}R_{E_t})^⊤ g_{t−1}`, and forward encoding satisfies
`g_t = (^{E_{t−1}}R_{E_t})^⊤ g_{t−1}` by construction, so the integrated
gravity and the authoritative gravity coincide (`a^⊤ b = 1`) even if a
synthetic trajectory really did rotate 180° in one frame. The absolute
channel still wins even here: any π-rotation about an axis `u ⊥ a` maps
`a` onto `−a`, so the authoritative tilt is satisfied by a whole
one-parameter family of `S`, and the fallback merely selects one member
of that family (the yaw-like residual DoF) deterministically. Near
anti-parallel the axis of a π-rotation is undefined; choose it
deterministically: pick the canonical basis vector least aligned with `a`,
`e_k = argmin_{e_i ∈ {e_x, e_y, e_z}} |a^⊤ e_i|`, take
`u = (a × e_k)/‖a × e_k‖`, and set `S = exp(π [u]_×)`. Better temporally:
if the previous frame produced a valid correction axis `u_{t−1}`, do not
reuse it directly — a π-rotation about `u` maps `a` onto `−a` only if
`u^⊤ a = 0`, and `u_{t−1}` is not orthogonal to the current `a_t`.
Project first:

$$
\mathbf u_t^\perp = \mathbf u_{t-1} - (\mathbf u_{t-1}^\top \mathbf a_t)\,\mathbf a_t,
\qquad
\mathbf u_t = \frac{\mathbf u_t^\perp}{\|\mathbf u_t^\perp\|},
$$

and only if `‖u_t^⊥‖` is too small (near-parallel to `a_t`) fall back to
the canonical axis `e_k` above. The axis then varies smoothly near the
anti-parallel boundary instead of jumping randomly. Frequent
`a ≈ −b` occurrences in a trained VAE are an anomaly signal — the
consistency loss (§4.3.3) should make them rare; the fallback only
guarantees the inverse is always well-defined.

**Authoritative channels — wherever an absolute channel exists.** At
reconstruction, the absolute channels `g_t`, `h_t`, `q_t` are
authoritative — and there is nothing to contradict them: the increment
channels exist only where no absolute channel covers the degree of
freedom (§2.6). Steps 2–3 show the mechanism: the integrated
yaw-like rotation and the accumulated horizontal displacement are
re-anchored onto the absolute gravity and height at **every** step, so
integration error can never accumulate on any absolute degree of
freedom. This preserves the TextOp anti-integration-drift property:
the absolute channels re-anchor the trajectory wherever they are
available.

**Endpoint semantics.** States `t = 0..T` produce features
`m_1..m_T` — feature `t` carries the absolute state at `t` plus the
incoming transition `t−1 → t`. The inverse above therefore produces
**every** state `1..T`: the final feature `m_T` carries the full
authoritative state of `s_T` (`h_T`, `g_T`, `q_T`, `c_T`), so the
window endpoint is completely reconstructed — no state's absolute
channels are unrecoverable. The only state not reconstructed from
features is the initial state `0`, which is supplied as the initial
pose and intentionally not recoverable from the features alone (§2.8).
This is the mirror image of the departure-state draft, which
reconstructed `1..T−1` and left `s_T` unrecoverable; it also stitches
windows cleanly — the final state of window `k` serves as the initial
pose of window `k+1`, so a rollout concatenates at the boundary with
no re-anchoring. The inverse algorithm and the temporal-consistency
sums can therefore rely on an absolute anchor at **every** feature
index `1..T`. One deliberate asymmetry remains, at the other end: the
initial pose receives no reconstruction supervision (`L_rec` on `m_1`
supervises the transition that *uses* `s_0`, not `s_0` itself) — the
initial state is observed context, not a prediction target.

One asymmetry by design: there is **no absolute 3-DoF orientation
channel** (yaw-free principle), so the yaw-like DoF is carried by
`^{E_{t−1}}R̂_{E_t}` alone via integration; `g_t` anchors only the
2-DoF tilt (step 2). (`g_t` is a drift-correction anchor, not mere
conditioning.)

Internal redundancy (the one that remains): `g_t` is an absolute,
drift-free 2-DoF tilt reference, while `R_t` is reconstructed by
*integrating* `ρ_6`; consistency requires the incoming transition to
transport the authoritative gravity of frame `t−1` forward into frame
`t` (`g_t = R_t^⊤ g_W = (^{E_{t−1}}R_{E_t})^⊤ R_{t−1}^⊤ g_W`):

$$
\boxed{\mathbf g_t = \left({}^{E_{t-1}}\mathbf R_{E_t}\right)^\top \mathbf g_{t-1}}
$$

Every other redundancy was removed by design: the `q_t`/`Δq_t` pair
(v3) and the `h_t`/`Δh_t` pair (earlier draft) no longer exist
(§2.6, §4). The one
remaining redundancy is unavoidable — the absolute channel covers only
2 of the 3 rotational DoF while the transition covers all 3 — and it is
enforced explicitly by the temporal consistency loss (§4.3.3); the
absolute channel wins at reconstruction (step 2).

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
consistency term (§4.3.3) — and with the geometry terms recomputed on the
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
   satisfies `‖g_t‖ = 1`: a raw output whose physical magnitude is wrong
   (direction right, e.g. `[0, 0, −5]` in physical units) maps, through
   the per-channel affine normalization, to a large deviation from the
   normalized GT target and is penalized by `L_rec` even though its
   direction would be correct after projection. This keeps the VAE
   latent space well-behaved.

Division of labor: the Huber term (`L_rec` alone — §4.3.2 shows a
separate `L_rot,6D` is already contained in it) is computed in
**normalized** feature space and needs **no denormalization**;
denormalization enters only the projection-based quantities (the chordal
term §4.3.2, the consistency term, and the use-time reconstruction).

**Notation (used throughout §4):** `m̃_t` is the raw decoder output in
**normalized** feature space — the only object `L_rec` ever sees;
`m̄_t` is its **denormalized** slice (per-channel inverse mean/std);
`x̂_t` is the **denormalized + projected** legal quantity (`ĝ_t`,
`^{E_{t-1}}R̂_{E_t}`, `Δp̂_{t-1}^hor`); for channels without a projection (`h`, `q`,
`c`) the hat marks the denormalized prediction only. GT quantities are
unadorned.

Only *before* formal use — reconstruction, FK, inference — is the raw
vector deterministically projected. The projections act on the
**denormalized** feature (denormalize first, then project): per-channel
mean/std normalization is an affine transform with non-trivial offsets,
and normalization, Gram–Schmidt and the tangent-plane projector are all
nonlinear, so they must see physical values (see §4.3.2). In this order:

$$
\boxed{\hat{\mathbf g}_t = \frac{\bar{\mathbf g}_t}{\|\bar{\mathbf g}_t\| + \epsilon}},
\qquad
\boxed{{}^{E_{t-1}}\hat{\mathbf R}_{E_t} = \mathrm{GS}(\bar{\mathbf r}_t^{\mathrm{6D}}) \in SO(3)},
\qquad
\boxed{\widehat{\Delta\mathbf p}_{t-1}^{\mathrm{hor}} = (I - \hat{\mathbf g}_{t-1} \hat{\mathbf g}_{t-1}^\top)\,\overline{\Delta\mathbf p}_{t-1}^{\mathrm{hor}}}
$$

so that

$$
\|\hat{\mathbf g}_t\| = 1,
\qquad
{}^{E_{t-1}}\hat{\mathbf R}_{E_t} \in SO(3),
\qquad
\hat{\mathbf g}_{t-1}^\top \widehat{\Delta\mathbf p}_{t-1}^{\mathrm{hor}} = 0
$$

hold **strictly** — they are not "learned" by the VAE. The horizontal
projection uses the *projected* gravity `ĝ_{t−1}` of the transition's
departure frame, so the projection order is sequential **across
tokens**: (1) project every `ĝ_t` (history and future); (2) Gram–Schmidt
every 6D slice (per-token, independent); (3) project each incoming
`Δp̄^hor` with the departure frame's gravity. The horizontal projection
is therefore a cross-token operation — feature `m_t` carries `g_t`, not
`g_{t−1}`. Boundary cases: for `m_1` the departure gravity is
`g_0 = R_0^⊤ g_W`, computed from the supplied initial pose (§4.1); for
the first predicted token `m_{t+1}` it is the history endpoint's `g_t`
— the current state's tilt, measured (deployment) or GT (training),
never a predicted quantity (§4.6). `ε` is a numerical guard only (`F.normalize(..., eps=1e-8)`
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
{}^{E_{t-1}}\hat{\mathbf R}_{E_t} = \mathrm{GS}(\bar{\mathbf r}^{\mathrm{6D}}),
\qquad
\bar{\mathbf r}^{\mathrm{6D}} = \mathrm{denorm}(\tilde{\mathbf r}^{\mathrm{6D}}),
\qquad
L_{\mathrm{rot,chord}} = \|{}^{E_{t-1}}\hat{\mathbf R}_{E_t} - {}^{E_{t-1}}\mathbf R_{E_t}\|_F^2,
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
# (code-level short names for ^{E_{t-1}}R_{E_t}: rel_rot / rel_rot_6d / pred_rel_R)
rel_rot_6d = denormalize(rot6d_slice(model_output))    # per-channel inverse mean/std
pred_rel_R = rot6d_to_matrix(rel_rot_6d)               # GS -> valid SO(3)
loss_chordal = ((pred_rel_R - gt_rel_R) ** 2).sum(dim=(-1, -2)).mean()   # ||.||_F^2
loss_rot = lambda_chord * loss_chordal                 # lambda_chord << 1 (start 0.01-0.05)
```

**Denormalize before Gram–Schmidt.** GS must see the denormalized 6D
vector, not the normalized feature. At 50 Hz the relative rotation
`^{E_{t-1}}R_{E_t}` is close to the identity, so the per-channel means are
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

The two are complementary: one is local rotational dynamics, the other is
the accumulated orientation trajectory. (There is no `Δq` channel to play
the analogous role for joints — §4 — so joints get a single supervision on
the authoritative `q` trajectory, in `L_rec` and `dof_pos`.) The chordal distance is invariant to a common
world rotation (`‖QR_1 − QR_2‖_F = ‖R_1 − R_2‖_F`), so as long as
prediction and GT share the same initial frame, this term does not break
the SE(2) invariance of the representation. (The wording is deliberately
"reconstructed root orientation trajectory", not "global rotation".) The
remaining FK-based geometry terms (`body_trans`, `dof_pos`, `dof_vel`,
`foot_contact`) are representation-agnostic and stay as in v3; the final
term set and weights are listed in §7.

#### 4.3.3 Temporal consistency loss

A legal motion feature satisfies **one** temporal constraint:

$$
\boxed{\mathbf g_t = ({}^{E_{t-1}}\mathbf R_{E_t})^\top \mathbf g_{t-1}}
$$

It is the SO(3) analogue of "absolute state advances by the local
transition" and follows from `R_t = R_{t−1} · ^{E_{t−1}}R_{E_t}` and
`g_t = R_t^⊤ g`. It is also the **only** constraint the representation
can violate: the `q`/`h` self-consistency constraints of the earlier
draft (`q_{t+1} = q_t + Δq_t`, `h_{t+1} = h_t + Δh_t`) disappeared
together with the `Δq`/`Δh` channels — with no increment channel there
is no second estimate of the same quantity that could disagree. And it
is the one redundancy that *cannot* be removed by pruning: the absolute
channel `g_t` covers only 2 of the 3 rotational DoF while the
transition `^{E_{t−1}}R_{E_t}` covers all 3 (§2.6), so the incoming
transition's rotation must transport the authoritative gravity of frame
`t−1` forward into frame `t` (§4.1). The representation can therefore be summarized as:

**absolute state + local transition + temporal self-consistency.**

The consistency term uses the projected legal variables (§4.3.1) and —
mandatory — is computed **after denormalization**: the `g` and `ρ_6`
channels carry different per-channel statistics (and `ĝ` exists only
after denormalization + projection), so consistency must be checked in
physical units, not normalized feature space. This is the
same conclusion as the TextOp `dof_delta` analysis, and the original
code already follows it: its temporal-delta terms are computed on
`reconstruct_motion` output, i.e. in physical units
([manager.py:692-706](../../TextOpRobotMDAR/robotmdar/train/manager.py#L692-L706)).

$$
L_{\mathrm{temp}} = \lambda_g\, L_{g\mathrm{-cons}},
\qquad
L_{g\mathrm{-cons}} = \sum_t \left\| \hat{\mathbf g}_t - \left({}^{E_{t-1}}\hat{\mathbf R}_{E_t}\right)^\top \hat{\mathbf g}_{t-1} \right\|^2,
$$

with `λ_g = 0.05–0.1` (tune from the unweighted magnitude, not the
weighted value). Squared L2, no Huber: for unit vectors
`‖a − b‖² = 2(1 − a^⊤b)` is already bounded, and with no other terms
left there is nothing to balance per-variable deltas against.

**Pure self-consistency.** The term compares predictions with
predictions — the division of duties is clean:
`L_rec`: prediction vs. GT; `L_temp`: prediction vs. prediction.
A GT-anchored variant is deliberately excluded: the ground-truth `g`
sequence is already supervised by `L_rec` on the absolute channels, so
anchoring would re-supervise the same quantity with little information
gain. The sum runs over the pairs inside the predicted future window —
both sides predicted.

**Cross-boundary pair (history → future).** Following the original
temporal-delta pattern —
`pred_motion_tensor = cat([history_motion[:, −1:], predictions])`
([manager.py:694](../../TextOpRobotMDAR/robotmdar/train/manager.py#L694))
— the consistency sum includes one boundary pair between the last
history feature and the first predicted feature. The last history
feature `m_t` (history is `m_{t−H+1:t}`, covering states `t−H+1..t`)
carries the observed current state `s_t` — its gravity `g_t` is an
*observed* quantity at both training and inference (the current state
is always measurable, and the history features are always real; under
the arrival-state alignment the current state is never predicted — it
sits in the history). The first predicted feature `m̂_{t+1}` carries
the model's predictions for the outgoing transition
`^{E_t}R̂_{E_{t+1}}` and the next state's gravity `ĝ_{t+1}`. The
boundary pair is therefore:

$$
L_{g\mathrm{-cons}}^{\mathrm{boundary}}
=
\left\| \hat{\mathbf g}_{t+1} - \left({}^{E_t}\hat{\mathbf R}_{E_{t+1}}\right)^\top \mathbf g_t \right\|^2,
$$

comparing the predicted next-state gravity against the gravity implied
by the predicted outgoing transition applied to the *observed* current
gravity — a prediction-vs-prediction consistency check anchored on
observed data, in identical form at training and inference, for the
VAE and the LDM planner alike. No masking or special convention is
needed. (It also anchors the consistency sum at the window edge: the
interior pairs are prediction-vs-prediction on both sides, so this is
the only place where the consistency constraint touches observed
data.)

Under the arrival-state alignment the boundary pair's role shifts
compared with the departure-state draft: there, the first predicted
feature carried the current state itself, so the boundary pair was a
direct prediction-vs-observation check on `ĝ_now`; here the current
state is in the history (observed, never predicted), and the boundary
pair instead checks that the first *predicted* step is consistent with
the observed current gravity.

**Seam velocity continuity (optional).** The gravity pair is the only
seam constraint the representation strictly requires. Whether the
predicted *outgoing* step should also continue the observed *arriving*
velocity — the MPC initial-condition consistency — is a separate, soft
design choice, deliberately not enforced as a hard projection: a new
goal may legitimately demand immediate redirection (turn, stop) away
from the current velocity. If seam smoothing proves necessary in
deployment, add small-weight Huber terms comparing the first predicted
feature against the observed last history feature. For the horizontal
increment:

$$
{}^{E_t}\widehat{\Delta\mathbf p}_{t}^{\mathrm{hor}}
\;\approx\;
({}^{E_{t-1}}\mathbf R_{E_t})^\top\,
{}^{E_{t-1}}\Delta\mathbf p_{t-1}^{\mathrm{hor}},
$$

the observed arriving increment rotated from `E_{t−1}` into `E_t` by the
transpose of the observed last relative rotation; for joints and height
use the derived-velocity form of §4.3.5 —
`Huber(q̂_{t+1} − q_t, q_t − q_{t−1})`,
`Huber(ĥ_{t+1} − h_t, h_t − h_{t−1})` — the authoritative sequences on
both sides. In training the GT seam is always continuous (a window cut
inside continuous motion), so `L_rec` already supervises this
implicitly; the explicit term only hardens inference-time consistency.
The observed last-history increment is also a target of the §7.5
perturbation — if that perturbation is applied, the continuity target
moves with it.

Why keep the boundary pair at all, given that `L_rec` already
supervises `ĝ_{t+1}` against the ground-truth `g_{t+1}` in normalized
feature space? The boundary pair re-imposes the transport relation in
physical units on the variables the inverse algorithm actually uses —
it is the same rationale as the chordal term of §4.3.2: `L_rec` alone
sees per-channel normalized numbers, while the boundary pair sees the
projected, denormalized `ĝ_{t+1}`, the predicted transition and the
observed current gravity they must satisfy.

#### 4.3.4 Training diagnostics (log-only, not loss)

Log these from the first run — they localize problems far earlier than
the total VAE loss. Consistency residuals (physical units, denormalized
+ projected variables):

$$
e_g^{\mathrm{cons}} = \| \hat{\mathbf g}_t - \left({}^{E_{t-1}}\hat{\mathbf R}_{E_t}\right)^\top\hat{\mathbf g}_{t-1} \|,
\qquad
e_g^{\mathrm{boundary}} = \| \hat{\mathbf g}_{t+1} - \left({}^{E_t}\hat{\mathbf R}_{E_{t+1}}\right)^\top \mathbf g_t \|,
$$

(`e_g^boundary` additionally separates *where* the residual lives:
if `e_g^boundary` ≫ interior `e_g^cons`, the model is discontinuous at
the history window edge; if the opposite, it drifts inside the window.)

and projection corrections:

$$
e_{g,\mathrm{proj}} = \| \bar{\mathbf g} - \hat{\mathbf g} \|,
\qquad
e_{R,\mathrm{proj}} = \| \bar{\mathbf r}^{\mathrm{6D}} - \rho_6(\mathrm{GS}(\bar{\mathbf r}^{\mathrm{6D}})) \|,
\qquad
e_{p,\mathrm{proj}} = | \hat{\mathbf g}^\top \overline{\Delta\mathbf p}^{\mathrm{hor}} | .
$$

(`e_p,proj` measures the residual vertical component of the denormalized
`Δp̄_{t−1}^hor` — how far the raw displacement leaks out of the tangent
plane. All three use denormalized inputs, per the notation of §4.3.1.)

Expected behavior: as `L_rec` decreases, the projection corrections go
to 0 — the denormalized decoder output approaches a legal feature
(§4.3.1). If `L_rec` is low but a projection correction stays large,
normalization, loss weighting or the representation itself is broken
somewhere.

**Derived-velocity errors** (log-only, physical units) — these are the
primary health signal for the dynamics supervision of §4.3.5, since
Δ channels no longer exist to expose them through `L_rec`:

$$
e_{q,\mathrm{vel}} = \| \dot{\hat{\mathbf q}}_t - \dot{\mathbf q}_t^{\mathrm{GT}} \|,
\qquad
e_{h,\mathrm{vel}} = | 50(\hat h_{t+1} - \hat h_t) - 50(h_{t+1}^{\mathrm{GT}} - h_t^{\mathrm{GT}}) | .
$$

Report them **per motion class** — walking / running / fall / get-up
separately — not just as a global mean (tag naming in §4.3.6): a VAE
that reconstructs static
poses well can hide large velocity errors on the fast-motion subset,
and that subset is exactly where the Δ-channel removal would show up
first (a frame with `q_t` alone does not reveal whether a joint is
moving fast; velocity must be read from temporal context). If the
fast-motion velocity error stays high while the total rec loss is low,
the compensation in §4.3.5 is under-weighted, not the representation.

#### 4.3.5 Derived-velocity supervision (the Δ channels' role, kept as a loss)

Dropping the Δ channels removes *information redundancy*, not the
*dynamics supervision* they incidentally provided. At 50 Hz
`Δq_t = q_{t+1} − q_t` carries no new information, but it gives the
network an explicit velocity-like inductive bias per frame; without it,
velocity must be read from the temporal context `q_{t−1}, q_t, q_{t+1}`,
which depends on the temporal modeling capacity of the VAE. The
compensation is to supervise velocities **derived from the predicted
state sequences themselves** — in physical units, never via extra
channels:

$$
\dot{\hat{\mathbf q}}_t = \frac{\hat{\mathbf q}_{t+1} - \hat{\mathbf q}_t}{dt},
\qquad
L_{\mathrm{dof\text{-}vel}} = \mathrm{Huber}\!\left( \dot{\hat{\mathbf q}}_t,\; \dot{\mathbf q}_t^{\mathrm{GT}} \right),
$$

i.e. the model never outputs a velocity; it must output a `q`
trajectory whose own velocity is correct. **This is not a new term** —
it is exactly the v3 `dof_vel` loss: FK derives
`dof_vel = (q_{t+1} − q_t)/dt` on both the predicted and GT
trajectories ([forward_kinematics.py:307-308](../../TextOpRobotMDAR/robotmdar/skeleton/forward_kinematics.py#L307-L308))
and the physical-space Huber compares them
([manager.py:504](../../TextOpRobotMDAR/robotmdar/train/manager.py#L504)).
What changes in v6 is its **status and weight**: in v3 it ran at
`1e-5` ([train/mvae.yaml:39](../../TextOpRobotMDAR/robotmdar/config/train/mvae.yaml#L39))
with three other supervision paths for the same quantity (`L_rec` on
the Δq channel, the `dof_delta` consistency term at weight 1.0, and
`L_rec` on `q` itself); in v6 it is the only joint-velocity supervision
and must be re-tuned. **Weight-setting procedure** (shared by every
physical-space velocity term): in the first runs, measure the term's
unweighted loss magnitude and gradient magnitude; set the weight so
its weighted contribution balances the position-level supervision it
complements; confirm with the per-class diagnostics of §4.3.4. For
`λ_dof_vel` specifically this means deciding from the unweighted ratio
against `L_rec`/`dof_pos` — expect one to three orders of magnitude
above `1e-5`. Two cautions: at 50 Hz the
`×fps` factor also amplifies noise, so the weight must not be pushed
until high-frequency jitter appears; and `dof_vel` uses the FK
convention (physical rad/s), distinct from the raw-difference
convention the deleted channels used (§4).

The same principle covers the other degrees of freedom:

- **Horizontal**: `^{E_{t−1}}v̂^hor_{t−1→t} = 50·Δp̂^hor_{t−1}` — already
  supervised by `L_rec` on the `Δp_{t−1}^hor` channel itself, because
  that channel *is* the per-frame horizontal displacement of the step
  arriving at `t`, expressed in its departure frame `E_{t−1}` (§4)
  (supervised in normalized space
  as `Δp̃`, consumed in physical space as `Δp̂`; §4.3.1). No extra term
  needed.
- **Vertical**: `v_t^vert = 50(ĥ_{t+1} − ĥ_t)` — a small physical-space
  Huber term `L_{h\text{-}vel}` against `50(h^GT_{t+1} − h^GT_t)` is
  added. This one is genuinely new: v3 has no root-velocity loss
  (`body_trans` supervises position only), so vertical dynamics
  supervision previously rode on the **vertical component of
  `Δp_t^local`** — v3's full 3-D local translation increment (v3 has no
  `Δh` channel; `Δh` existed only in the earlier 74-D draft). v7 keeps
  only `Δp_{t−1}^hor`, so that supervision disappears and must be
  compensated here.
  Its weight follows the **same procedure** as `λ_dof_vel` — unweighted
  magnitude and gradient ratio against the `h` position supervision
  (`L_rec` on `h` / `body_trans`), confirmed via `e_h,vel` (§4.3.4).
  In practice it lands below `λ_dof_vel`: the `h` channel is
  1-dimensional (`n_q` dimensions for joints), so the same relative
  contribution costs less weight. Note `h` is re-anchored
  authoritative at reconstruction (§4.1), so this term constrains the
  temporal profile of the predicted `h` sequence, not its level.
- **Acceleration**: not in the first version. If pose reconstruction is
  accurate but motion is soft or high-frequency actions are smoothed
  out, add `L_acc = Huber(50²(q_{t+1} − 2q_t + q_{t−1}), ·)` with a
  deliberately small weight — too strong an acceleration loss makes the
  VAE chase GT high-frequency noise at the cost of generative
  smoothness. (v3 already has the hook: the `smooth` loss, weight 0.0
  in [train/mvae.yaml:49](../../TextOpRobotMDAR/robotmdar/config/train/mvae.yaml#L49).)
  Priority order: `q` reconstruction → `q̇` supervision → `q̈` only if
  needed.

**Why this is more honest than a Δq channel.** A stored Δq channel can
be predicted well *as a channel* (`Δq̂ ≈ Δq^GT`) while the
actually-used `q` trajectory is wrong — the controller and FK consume
`q_t`, never `Δq_t`, so a pretty auxiliary channel supervises nothing
that matters (v3's inverse never reads it, §3). Derived-velocity
supervision constrains the trajectory that is actually used — the
authoritative-channel principle applied to losses. For the LDM there
is a further benefit: the VAE latent no longer has to encode the highly
correlated `(q_t, Δq_t)` pair, freeing latent capacity for the truly
independent factors (pose, transitions, root motion, contact mode,
fall/recovery dynamics).

### 4.3.6 Tensorboard logging — tags that change with the representation

**Mechanism (code-verified).** `MVAEManager.calc_loss` returns
`(terms, extras)` ([train/manager.py:744](../../TextOpRobotMDAR/robotmdar/train/manager.py#L744));
`train_mvae.py` passes both to `post_step`, which logs every `terms`
key under the `loss` group as `train_<k>` / `eval_<k>` and every
`extras` key under the `extras` group
([train/manager.py:118-175](../../TextOpRobotMDAR/robotmdar/train/manager.py#L118-L175));
`self.extra` (stage, lr, …) is logged under `extras` too. **Every
individually logged term is the unweighted raw loss** — the weights
enter only `total` (`Σ_k loss_weight[k]·v`,
[train/manager.py:797](../../TextOpRobotMDAR/robotmdar/train/manager.py#L797)).
The unweighted magnitudes the §4.3.5 weight-setting procedure needs are
therefore already in tensorboard; no extra machinery is required for
them.

**Loss group.** The v3 → v6 fate of each tag:

- **Keep as-is**: `rec` (now on the 44-D feature), `kl`, `body_trans`,
  `dof_pos`, `foot_contact` — FK-based, representation-agnostic.
- **Keep the tag, change the computation**: `body_rot` — upgraded to
  the reconstructed-trajectory chordal term (§4.3.2). Its values are
  **not comparable across versions**; compare v3 vs. v6 only within one
  version's runs.
- **Keep, re-weighted**: `dof_vel` (§4.3.5) — same tag, weight moved
  from `1e-5` to the procedure value; again not cross-version
  comparable.
- **Keep as hook**: `smooth` (weight 0.0) — the deferred-acceleration
  slot (§4.3.5).
- **Drop**: `quantize_rot` / `quantize_trans` (v3-representation
  helpers, weight 0.0 in `mvae.yaml`); `endpoint_yaw` (Euler extraction
  on the FK rotation — banned by §2, principle 1; replaced by
  `e_R_endpoint` below); `endpoint_xy` from `terms` (weight 0.0; kept
  as an extra).
- **New**: `rot_chord` (L_rot,chord, §4.3.2), `g_cons` (L_g-cons
  including the boundary pair, §4.3.3), `h_vel` (L_h-vel, §4.3.5).

**Extras group.** Unchanged: `dof_pos_core` / `dof_pos_wrist` /
`dof_vel_core` / `dof_vel_wrist`, `hand_translation`, `sliding_ratio`,
`endpoint_xy`. New diagnostics (log-only, physical units — the §4.3.4
set, one tag each, plus the global mean under the bare name):

- `e_g_cons`, `e_g_boundary` (consistency residuals, §4.3.4);
- `e_g_proj`, `e_R_proj`, `e_p_proj` (projection corrections);
- `e_q_vel`, `e_h_vel` (derived-velocity errors);
- `e_R_endpoint` — chordal endpoint orientation error on the last
  **reconstructed** state (the inverse produces states `1..T`, so under
  the arrival-state alignment the final state is fully reconstructed
  with complete authoritative channels, §4.1),
  replacing `endpoint_yaw`.

**Per-class reporting.** `e_q_vel`, `e_h_vel` (and `e_g_cons`, if it
costs nothing) are reported per coarse motion class with a class suffix
(`e_q_vel__walk`, `e_q_vel__run`, `e_q_vel__fall`, `e_q_vel__getup`,
`e_q_vel__unknown`). The dataloader already carries the label: each
primitive stores the best-overlap BABEL verb as `action_label`
([data.py:971-988](../../TextOpRobotMDAR/robotmdar/dataloader/data.py#L971-L988))
and the collator puts it into the batch
([data.py:1300-1302](../../TextOpRobotMDAR/robotmdar/dataloader/data.py#L1300-L1302));
the geometry extras map verbs to the coarse classes with a small lookup
table (`is_recovery` can refine the get-up split) and compute masked
means per class. The mask is per-primitive — no loss or gradient
plumbing is touched; the labels only need to reach `calc_loss` (or the
train loop) alongside the batch.

**Version tag.** Log `feature_version` as a constant extra (alongside
`stage`/`lr`,
[train/manager.py:96-104](../../TextOpRobotMDAR/robotmdar/train/manager.py#L96-L104))
so v3/v6 runs in the same tensorboard directory stay distinguishable;
checkpoints already must record it (§6).

**Weight-setting support.** Unweighted magnitudes of the new terms come
directly from the `train_<k>` tags above. Gradient magnitudes are not
logged per step (per-term `autograd.grad` with `retain_graph` costs an
extra backward pass each); measure them once per term in a first-run
probe and set the weights from the two ratios (§4.3.5).

**Eval mirror and visualization.** The eval path calls the same
`calc_loss`, so the `eval_<k>` tags change identically with no extra
work. The eval visualization (reconstructed-motion videos) is not a
loss tag but switches representations at the same dispatch point — it
must use the §4.1 inverse (denormalize → project → reconstruct), never
the v3 Algorithm 2 path.

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
`ρ_6(R_{t−1}^⊤ R_t)` at 50 Hz and pick the floor from the percentiles. The
std floor is a **numerical safeguard, not a feature-weighting device**;
relative importance of rotation/translation/joint channels should be
controlled explicitly by group loss weights. (The 6D identity layout
depends on the §4.2 convention — never hardcode an assumed `[1,0,...]`
pattern; always obtain it via `matrix_to_rot6d`.) The floor does **not**
relax the §4.3.2 rule: the normalized 6D still carries the affine offset
`μ`, which distorts column directions, so GS and every projection still
see denormalized values only. Likewise, per-channel normalization
distorts the geometry of the unit-vector `g` channel in normalized
space — which is fine for `L_rec` (it only needs a regression target)
and irrelevant everywhere else, because all projections and
physical-space losses operate on denormalized quantities (§4.3.1).

### 4.5 Pre-training unit tests (required before the first run)

Two property tests on random GT trajectories must pass before VAE
training starts:

1. **Round-trip.** For random GT trajectories `X` (states `0..T`):
   encode `m = F(X)` (features `m_{1:T}`), decode
   `X' = I(m, (p_0, R_0))`, and check
   `max|p − p'|`, `d_SO(3)(R_t, R'_t)` and `max|q − q'|` at
   near-floating-point precision (≤ 1e-5) — **over states `1..T`
   only**: the inverse reconstructs `1..T` from the supplied initial
   pose, and state `0` has no feature channels by design — it is the
   initial condition, not a reconstruction target (endpoint
   semantics, §4.1). The comparison window is the forward-encode window
   minus its initial state; the final state **is** covered — under the
   arrival-state alignment the endpoint is complete.
2. **Horizontal SE(2) invariance.** Apply a random horizontal world
   transform `R_t' = Rz(θ)·R_t`, `p_t' = Rz(θ)·p_t + t_xy` and check
   `m(X') ≈ m(X)`.

Together these verify the doc's two core claims — exact invertibility on
valid trajectories (§2, principle 4) and global horizontal SE(2)
invariance — directly on the implemented encode/decode pair, before any
loss or model is involved.

Both tests exercise the **physical** encode/decode pair `F`/`I`:
per-channel normalization is a collation-time training wrapper
([data.py:1295](../../TextOpRobotMDAR/robotmdar/dataloader/data.py#L1295))
and must not leak into these tests — if the implementation applies
normalization internally, the tests unwrap it explicitly, so that a
normalization bug surfaces as a round-trip failure instead of being
silently absorbed.

### 4.6 Goal reference convention (LDM planner interface)

This subsection settles, at the convention level, the goal part of §7.7:
which dataset frames carry the planner's reference and the goal under the
arrival-state alignment, how every goal channel is derived and supervised
by the LDM losses, and what changes in the closed-loop controller path.
The concrete goal-encoding layout is deliberately deferred to a later
design pass — this subsection pins the conventions that layout must
satisfy (the `(h_g, Δp_g^hor)` position split, the
reconstructed-trajectory goal orientation, the full arriving velocity);
the SONIC integration check remains for the LDM integration phase.

**Window and reference indices.** The dataset window is raw states
`s_{t−H:t+F}` (indices `p..p+C`, `C = H+F`) producing features
`m_{t−H+1:t+F}`. The planner reference is the window state `s_t` at

$$
\boxed{\mathrm{reference\_frame} = p + H,}
$$

the state-part of the final history feature `m_t` — the measured current
state in closed loop and the unique egocentric frame `E_t`. The goal is
the window state `s_{t+t*}` at

$$
\boxed{\mathrm{goal\_frame} = p + H + t^*},\qquad t^* \in [1, F],
$$

default `t* = F` for `goal_per_primitive` — the window's **last** state.
The arrival-offset window is therefore `t* ∈ [1, F]`, i.e. goal states
`s_{t+1:t+F}` carried by future features `m_{t+1:t+F}` (`t` = the last
history frame). The existing dataset config already realizes exactly this
window: `goal_offset_range = [-63, 0]` with `t* = F + goal_offset`
([train_dar.yaml:23](../../TextOpRobotMDAR/robotmdar/config/train_dar.yaml#L23)),
so the future feature index is `t*−1 = F − 1 + goal_offset` ∈ `[0, F−1]`
— the `_future_step_from_goal_time` clamp never binds for legal offsets.
The primitive window formula does not change
(`prim_end = prim_start + F + H + 1`): the one extra raw state beyond the
`H+F` features is now the departure context `s_{t−H}` of the first
history feature `m_{t−H+1}` (§2.8), not a forward-difference velocity
frame. Time-to-arrival keeps its formula
`(goal_frame − reference_frame)/fps`
([data.py:1062](../../TextOpRobotMDAR/robotmdar/dataloader/data.py#L1062)),
the arrival-time PE keeps the same value range, and the planner-side
conversion relative to the last measurement timestamp is untouched.

**One feature carries the goal.** Future feature `k` (`k = 0..F−1`) is the
absolute feature `m_{t+1+k}`, whose state-part is `s_{t+1+k}` and whose
increment channels are the incoming transition `t+k → t+1+k`. The goal
state `s_{t+t*}` is therefore the state-part of future feature index
`t*−1`, and the step **arriving** at the goal is the increment of the
**same** feature:

$$
\boxed{\text{goal feature index} = t^* - 1.}
$$

`_future_step_from_goal_time(t*) = t*−1`
([manager.py:1445](../../TextOpRobotMDAR/robotmdar/train/manager.py#L1445))
survives numerically unchanged, with an upgraded meaning: under the
departure-state draft the same index carried the goal's state-part but its
delta was the *leaving* step, which forced the trajectory integrator to
borrow the arriving step from feature `t*−2`; under the arrival-state
alignment the goal's absolute channels and its arriving displacement's
horizontal part live in one feature slot (the vertical part of the
arriving step is recovered from the authoritative heights of the two
endpoint features, §4). (The leaving displacement, if ever needed, is the
increment of feature index `t*` — representable only for `t* < F`.)

**Goal channels: derivation and supervision.** With
`goal_frame = p + H + t*`, every channel is representable inside the
window — the departure-state draft's goal-boundary dilemma (goal one frame
short of the window end, or partially unsupervised) does not arise:

- *Root position.* The goal position is carried as the pair
  `(h_g, ^{E_t}Δp_g^hor)` — the same absolute/transition split as the
  representation itself (§4), not as one full 3D displacement:

  - `h_g = ĥ_{t+t*}` — the authoritative height, a direct state-part read
    of feature `t*−1` (§4.1), supervised exactly like the joint angle
    below.
  - `^{E_t}Δp̂_g^hor` — the accumulated horizontal displacement: the sum of
    the future increments rotated into `E_t`, projected onto `E_t`'s own
    tangent plane,

    $$
    {}^{E_t}\widehat{\Delta\mathbf p}_g^{\mathrm{hor}}
    = \left(I - \hat{\mathbf g}_t \hat{\mathbf g}_t^\top\right)
      \sum_{k=1}^{t^*} {}^{E_t}\hat{\mathbf R}_{E_{t+k-1}}\,
      \widehat{\Delta\mathbf p}_{t+k-1}^{\mathrm{hor}},
    $$

    with `^{E_t}R̂_{E_{t+k−1}}` the **reconstructed** orientation chain
    (orientation bullet below), not the raw `ρ_6` product. The full 3D
    displacement, if ever needed, is

    $$
    {}^{E_t}\widehat{\Delta\mathbf p}_g
    = \sum_{k=1}^{t^*} {}^{E_t}\hat{\mathbf R}_{E_{t+k-1}}
      \left[ \widehat{\Delta\mathbf p}_{t+k-1}^{\mathrm{hor}}
      - \left(\hat h_{t+k} - \hat h_{t+k-1}\right)\hat{\mathbf g}_{t+k-1} \right]
    = \sum_{k=1}^{t^*} {}^{E_t}\hat{\mathbf R}_{E_{t+k-1}}\,
      \widehat{\Delta\mathbf p}_{t+k-1}^{\mathrm{hor}}
      - \left(\hat h_g - h_t\right)\hat{\mathbf g}_t,
    $$

    — the vertical corrections collapse onto `E_t`'s own gravity axis
    (each `R̂_{t+k−1}ĝ_{t+k−1}` equals `g_W` by construction, §4.1), which
    is exactly why the horizontal channel above is the increment sum
    alone and why the pair `(h_g, Δp̂_g^hor)` carries no vertical term.
  Supervised against the reconstructed trajectory point `s_{t+t*}` (§4.1)
  — the same pair read off the reconstruction. The v3
  `root_trajectory_ego` history bridge
  ([manager.py:1587](../../TextOpRobotMDAR/robotmdar/train/manager.py#L1587))
  is dropped: integration starts at `s_t` and uses future increments only.
- *Root orientation.* The goal orientation is read from the
  **reconstructed trajectory**, not from a raw product of predicted
  relative rotations:

  $$
  {}^{E_t}\hat{\mathbf R}_{E_{\mathrm{goal}}} = \mathbf R_t^\top\, \hat{\mathbf R}_{\mathrm{goal}},
  $$

  where `R̂_goal` is the reconstructed orientation at the goal frame — the
  full §4.1 inverse: relative-rotation integration *plus* the
  authoritative-gravity re-alignment at every step (`R_t` is the supplied
  initial pose — the measured current orientation — not a prediction).
  Only for legal forward-encoded trajectories, where every re-alignment
  correction is the identity, does this reduce to the raw product
  `Π_{k=1..t*} ^{E_{t+k−1}}R̂_{E_{t+k}}`; for model predictions the
  re-alignment is active, so the goal loss compares the **reconstructed**
  orientation — the same object the planner output actually uses —
  against the GT `^{E_t}R_{E_goal}`, chordal and invariant to the common
  world rotation by the §4.3.2 argument. The goal's tilt is authoritative
  from `ĝ_{t+t*}` (state-part of feature `t*−1`), the yaw-like DoF from
  the reconstructed integration. The v3 yaw-integrator
  `_future_yaw_ego_at_goal`
  ([manager.py:1471](../../TextOpRobotMDAR/robotmdar/train/manager.py#L1471))
  disappears together with the `Δψ` channel.
- *Joint angle / contact.* `q̂_{t+t*}`, `ĉ_{t+t*}` — direct state-part
  reads of feature `t*−1` (§4.1 item 4: contact as continuous scores).
- *Root velocity.* The **arriving** velocity at the goal is the full
  displacement of the incoming step, not its horizontal part alone. In the
  departure frame `E_{g−1}`:

  $$
  {}^{E_{g-1}}\widehat{\Delta\mathbf p}_{g-1\to g}
  = \widehat{\Delta\mathbf p}_{t+t^*-1}^{\mathrm{hor}}
  - \left(\hat h_{t+t^*} - \hat h_{t+t^*-1}\right)\hat{\mathbf g}_{t+t^*-1},
  $$

  the goal feature's own increment channel plus the vertical part
  recovered from the authoritative heights (§4). Rotated into the goal's
  reference frame `E_t`:

  $$
  {}^{E_t}\hat{\mathbf v}_g^{\mathrm{in}}
  = 50 \cdot {}^{E_t}\hat{\mathbf R}_{E_{g-1}}\,
    {}^{E_{g-1}}\widehat{\Delta\mathbf p}_{g-1\to g},
  $$

  with `^{E_t}R̂_{E_{g−1}}` the reconstructed orientation chain
  (orientation bullet above). If the interface wants the
  horizontal/vertical split, it applies it in `E_t` at use time:
  `v̂^hor_g = (I − ĝ_t ĝ_t^⊤) ^{E_t}v̂^in_g`,
  `v̂^vert_g = −ĝ_t^⊤ ^{E_t}v̂^in_g`. Ground truth is the backward
  difference at `goal_frame` — `50·R_{g−1}^⊤(p_g − p_{g−1})`, the full
  displacement, both endpoints inside the window. This flips the semantic
  from v3's leaving velocity:
  `_world_goal_velocity` switches to the backward difference
  ([data.py:1049](../../TextOpRobotMDAR/robotmdar/dataloader/data.py#L1049))
  and the `required_last_frame = goal_frame + 1` guard
  ([data.py:1187](../../TextOpRobotMDAR/robotmdar/dataloader/data.py#L1187))
  is deleted. The goal arriving velocity is an input conditioning
  channel, not a published payload quantity; the `E_t`-frame form above
  is a use-time conversion, not a separate supervision target.

**Closed-loop construction.** `H+1` measurements `s_{t−H:t}` build the
history features `m_{t−H+1:t}` directly; the final feature's increment has
both endpoints inside the measurement sequence. The last-frame duplication
and constant-velocity override in `state_to_model_input`
([planner_convert.py:108](../../TextOpRobotMDAR/robotmdar/utils/planner_convert.py#L108),
lines 139-198) and its docstring rationale are **deleted** — no future
motion is needed to construct the model input, and the input always
carries the full absolute state of `s_t` (`h_t`, `g_t`, `q_t`, `c_t`), so
no separate current-state condition is needed (§2.8). The goal is built
from the controller target with reference = measured `s_t`
(`state_goal_from_reference` already anchors to the last measurement —
mechanism unchanged). The `Δψ` wrap handling disappears with the channel.

**Reconstruction, seam and rollout.** The future is reconstructed from the
initial pose (the complete `s_t`, §4.1) plus `m_{t+1:t+F}` — no
`cat(history, future)` needed for reconstruction, since v3 only needed the
history for the integration bridge. The published plan keeps its shape:
65 frames, frame 0 = the measured `s_t` itself (the model never touches
the current state), frames `1..64` = `s_{t+1:t+F}`; the goal sits at
frame `t*` — the final frame only when `t* = F` — and every published
frame carries its full authoritative channels; `skip_history = H−1`
unchanged, so the sequence sent to the controller is isomorphic to v3's.
Rollout stitching uses the §4.1 clean-boundary property — the next
window's initial pose is the previous window's final state (the goal when
`t* = F`): the eval anchor becomes `motion_dict_to_abs_pose(...,
idx=-1)` ([generate_dar.py:348](../../TextOpRobotMDAR/robotmdar/eval/generate_dar.py#L348),
[freq_dar.py:119](../../TextOpRobotMDAR/robotmdar/eval/freq_dar.py#L119),
`vis_dar.py` / `vis_mvae.py` likewise), replacing the H-dependent index
(the current `idx=-2` is correct only for `H = 2`). Joint velocities are
**not published by the planner** (see the interface note below) — the
feature alignment does not change them because they are not part of the
plan payload (§4, derived velocities).

**Migration checklist (v3 → v6, goal-related code).**

| Site | Change |
|---|---|
| `data.py:1108` | `reference_frame = prim_start + H` (was `− 1`) |
| `data.py:1173-1176` | `goal_frame = prim_start + F + H + goal_offset` (was `− 1`); `t* = F + goal_offset` keeps the existing `[-63, 0]` offset window (arrival offsets `1..64`); the order check at `:1182` compares against the new reference |
| `data.py:1049-1060` | `_world_goal_velocity` → backward difference at `goal_frame`: full displacement `R_{g−1}^⊤(p_g − p_{g−1})`, both endpoints in-window (no `^hor` truncation — the vertical part belongs to the arriving velocity) |
| `data.py:1187-1194` | `required_last_frame = goal_frame + uses_arrival_time` → deleted |
| `manager.py:1445-1460` | `_future_step_from_goal_time` — unchanged numerically |
| `manager.py:1471-1485` | `_future_yaw_ego_at_goal` → deleted (yaw-free; goal orientation read from the reconstructed trajectory, §4.6 — not the raw `ρ_6` product) |
| `manager.py:1587-1605` | `root_trajectory_ego` — drop the history bridge, integrate future increments from `s_t` (the v3-only guard at `:1589` is replaced by the v6 path) |
| `manager.py:1536-1565` | `calc_goal_root_velocity_loss` — full arriving displacement: goal feature's increment channel + vertical part from the authoritative heights (§4.6), no goal-yaw rotation; GT backward difference |
| `planner_convert.py` `state_to_model_input` | delete last-frame duplication + constant-velocity override (`:139-198`); build H features from H+1 measurements |
| `generate_dar.py:348`, `freq_dar.py:119` | eval anchor `idx −2` → `idx −1`; reconstruct future-only with initial pose = `s_t` |
| goal statistics | re-frozen after the feature migration (v6 normalization recompute, §4.4) |
| plan payload | drop `joint_vel` from the published G1 plan (see the interface note below); `_forward_velocity` and its repeat-last fallback disappear |

**Plan payload: authoritative state only — velocity derivation is the
consumer's business.** Whether a frame-to-frame increment is a forward or
backward difference is a TextOp-internal convention of the feature
representation (§4); it must not leak into the payload. The planner
therefore publishes only the authoritative state trajectory (`joint_pos`,
`root_pos`, `root_ori`, timestamps) and no derived velocity. The SONIC
policy *consumes* plan joint velocities — its 1762D encoder observation
includes future 10-frame (step-5) velocity blocks, full 29-joint (290D)
and lower-body (120D)
([sonic_trackingmode.py:258-264](../../occHIPC/pretrain/sonic_trackingmode.py#L258-L264))
— and its motion loader already re-derives `joint_vel` from `joint_pos`
with `np.gradient` whenever the source carries positions only
([g1_motion_loader.py:128-133](../../occHIPC/utils/g1_motion_loader.py#L128-L133)):
SONIC's own convention, and the same one the policy was trained on
(planner-side forward differences with a repeat-last fallback would be a
third, TextOp-imposed convention). Dropping the field from
`G1MotionData` (`joint_vel` becomes optional in `sonicmsg/messages.py`)
removes the last-frame fallback question entirely and saves the payload.
This is the §2.7 principle applied to the interface: dynamics derived at
use time, by whoever consumes it.

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
  `motion_feature_to_dict_v6` and `motion_feature_dim_v6 = 13 + n_q + n_c`
  (identifier 6, because 4 and 5 are taken by the dead experiments in §8).
- Add a config key (e.g. `feature_version: 6` in `base.yaml`,
  `train/dar.yaml`, `planner_dar.yaml`) that selects the dispatch before
  dataset/model construction. The current mechanism is a module-level
  constant; the config key should be resolved at startup.
- Known hardcoded v3 guards that must become version-aware:
  `planner/planner_dar.py:214` (`FeatureVersion != 3` raise),
  `train/manager.py:1251` and `:1297` (trajectory integration, v3-only),
  `utils/planner_convert.py` (controller-history encoding, v3 layout).
- Per-version channel statistics: `dataloader/data.py` computes mean/std
  from the feature tensor, so v7 gets its own normalization automatically;
  checkpoints/models must record the feature version they were trained with;
  tensorboard tags a constant `feature_version` extra (§4.3.6).
- Downstream interfaces to keep parity across versions: history injection,
  goal encoding (`utils/goal.py`), FK reconstruction
  (`dataloader/data.py: reconstruct_motion`), and geometry losses
  (`train/manager.py`). For the v7 path the geometry losses operate on
  reconstructed FK outputs and are representation-agnostic, which is what
  makes the ablation clean.
- Ablation protocol: identical data, model, and training recipe; only the
  feature head width changes (69 vs. 44 channels for `n_q = 29`).
- The Δq/Δh removal is the main representation change vs. v3 besides the
  rotation. To isolate its effect, an optional `v6-with-Δq` arm
  (feature width `44 + n_q`, `L_rec` on the extra channel, the
  `dof_delta`-style consistency term re-added, everything else
  identical) can be trained and compared on **joint angle error, joint
  velocity error, FK error, and the fast-motion subsets** (walking /
  running / fall / get-up separately) — not on total reconstruction
  loss, which a Δ channel can flatter without helping the used
  trajectory (§4.3.5).

## 7. Open questions

1. **Anti-parallel re-alignment — resolved.** Deterministic
   orthogonal-axis π-rotation fallback with temporal axis reuse (§4.1).
2. **`Δq` increment channel (and the earlier draft's `Δh`) — removed
   by design.** (§2.6, §4) v3's `Δq` carries no information beyond the
   authoritative absolute state and is dropped from the feature;
   re-add (as an optional channel) only if a later ablation shows a
   velocity-like channel improves latent motion quality.
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
   history state — which under the arrival-state alignment is the
   absolute channels of the final history feature `m_t` itself). The
   concrete v7 perturbation operator for the rotation channels (must
   not be applied in RPY form) is still to be supplied, together with
   how the final history feature's incoming transition is perturbed —
   unlike the departure-state draft, its two endpoints both lie inside
   the history window; non-rotation channels keep the v3-style
   perturbation. This no longer blocks VAE
   training.
6. **Weights — starting points set.** `λ_chord = 0.01–0.05`,
   `λ_g = 0.05–0.1` (the only consistency term; no Huber deltas
   needed — `L_{g-cons}` uses bounded squared L2 on unit vectors,
   §4.3.3). `λ_dof_vel` is **re-tuned from its v3 `1e-5`**: in v6 it
   is the only joint-velocity supervision (§4.3.5), so set it from the
   unweighted magnitude and gradient ratio against `L_rec`/`dof_pos` —
   expect one to three orders of magnitude above `1e-5` — and confirm
   via the per-class `e_q,vel` diagnostic. `λ_h-vel` follows the
   **same procedure** against the `h` position supervision, confirmed
   via `e_h,vel`; in practice it lands below `λ_dof_vel` (1-dim vs.
   `n_q`-dim channel). Final values from unweighted loss magnitudes
   and gradient magnitudes.
7. **LDM planner, goal encoding, controller interface — resolved at the
   convention level (§4.6).** The feature alignment (§2.8, §4) fixes the
   interface anchor: history features `m_{t−H+1:t}` end with the current
   state `s_t`, the goal sits at `goal_frame = p + H + t*` with its
   absolute channels and arriving displacement's horizontal part in the
   single feature slot `t*−1` (vertical part from the authoritative
   heights, §4), and no separate current-state condition is needed. What
   remains for the LDM integration phase: the goal-loss weight set and the
   SONIC interface verification (§4.2). The concrete goal-encoding layout
   is deliberately deferred to a later design pass — §4.6 pins the
   conventions it must satisfy (the `(h_g, Δp_g^hor)` position split, the
   reconstructed-trajectory goal orientation, the full arriving velocity),
   but the layout itself is not part of this version.
8. **BONES-SEED CSV Euler convention — resolved.** Extrinsic xyz,
   verified against the official SEED G1 CSV spec and a synthetic SciPy
   test; the conversion code's computation is correct, only its comment
   was mislabeled "intrinsic" (§5).
9. **Time alignment — resolved.** Arrival-state / backward-difference
   convention: `m_t` = incoming transition `t−1 → t` + partial absolute
   state at `t` (§2.8, §4), replacing the departure-state draft. The
   in-code v6 forward/inverse (`motion_dict_to_feature_v6` /
   `motion_feature_to_dict_v6`), the decode path, the normalization
   statistics (`meanstd_v6_dof29.pkl`) and the §4.5 tests still
   implement the departure-state draft and must be migrated before the
   first VAE run; the goal-related part of that migration is itemized in
   §4.6.

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
- v4's relative rotation is computed as `R_{t+1} R_t^T` — the same
  physical relative rotation as this design's `^{E_{t−1}}R_{E_t} =
  R_{t−1}^T R_t` (§4.1), but expressed in a different frame. With
  `R_k = ^W R_{E_k}`, v4's quantity is the **world-frame** expression
  (the conjugate `R_t (R_t^T R_{t+1}) R_t^T`) of the ego-frame map
  `E_t → E_{t+1}`, forward-indexed; this design's is the
  **departure-ego-frame** expression `R_{t−1}^T R_t` of the map
  `E_{t−1} → E_t`, backward-indexed. It is *not* the transpose — the
  transpose `R_t^T R_{t−1}` is the inverse rotation (`E_t → E_{t−1}`).
  The delta code must not be reused without the full conversion (frame
  conjugation + time reversal), not merely a transpose.
- Status: never wired in — the module-level default dispatch is
  `FeatureVersion = 3`, overridden to 6 by `config/base.yaml`;
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

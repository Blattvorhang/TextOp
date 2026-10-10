import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Style
# ============================================================
try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass

plt.rcParams.update({
    "axes.linewidth": 1.0,
})


# ============================================================
# Helpers
# ============================================================
def hermite_segment(t, t0, t1, y0, y1, m0, m1):
    s = (t - t0) / (t1 - t0)

    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2

    return (
        h00 * y0
        + h10 * (t1 - t0) * m0
        + h01 * y1
        + h11 * (t1 - t0) * m1
    )


def smooth_step(x, x0, sharpness=6.0):
    return 1.0 / (1.0 + np.exp(-(x - x0) * sharpness))


# ============================================================
# Timeline
# ============================================================
t = np.linspace(
    0.0,
    6.25,
    1600,
)

t_start = 3.20
delay = 1.02
t_finish = t_start + delay


# ============================================================
# Goal
# ============================================================
goal_y = 0.82


# ============================================================
# Old reference
# ============================================================
y0 = 0.06

y_old = (
    y0
    + (goal_y - y0)
    * (1.0 - np.exp(-0.82 * t))
)

y_old += (
    0.050
    * np.exp(
        -0.5
        * ((t - 2.15) / 0.78) ** 2
    )
)

y_old -= (
    0.018
    * smooth_step(
        t,
        3.75,
        sharpness=4.0,
    )
)


# ============================================================
# Realized execution before replanning
# ============================================================
drift_progress = np.clip(
    (t - 0.55)
    / (t_start - 0.55),
    0.0,
    1.0,
)

spatial_drift = (
    -0.025
    * drift_progress
    - 0.155
    * drift_progress**1.75
)

tracking_fluctuation = (
    (
        0.008
        + 0.018 * drift_progress
    )
    * np.sin(
        4.2 * t + 0.35
    )
    * drift_progress
)


# ============================================================
# External perturbation
# ============================================================
t_pert = 2.68

perturbation = (
    0.110
    * np.sin(
        26.0 * (t - t_pert)
    )
    * np.exp(
        -0.5
        * ((t - t_pert) / 0.17) ** 2
    )
)


# ============================================================
# Base realized execution
# ============================================================
y_exec_base = (
    y_old
    + spatial_drift
    + tracking_fluctuation
    + perturbation
)


# ============================================================
# New reference
# ============================================================
y_exec_start = np.interp(
    t_start,
    t,
    y_exec_base,
)

dy_exec_base = np.gradient(
    y_exec_base,
    t,
)

m_exec_start = np.interp(
    t_start,
    t,
    dy_exec_base,
)

y_new = np.full_like(
    t,
    np.nan,
)


# First segment
t_mid = t_start + 1.42

mask_new_1 = (
    (t >= t_start)
    & (t <= t_mid)
)

y_mid = goal_y - 0.22
m_mid = 0.085

y_new[mask_new_1] = hermite_segment(
    t[mask_new_1],
    t_start,
    t_mid,
    y_exec_start,
    y_mid,
    m_exec_start,
    m_mid,
)


# Second segment
t_end = 6.05

mask_new_2 = (
    (t > t_mid)
    & (t <= t_end)
)

y_end = goal_y - 0.008
m_end = 0.0

y_new[mask_new_2] = hermite_segment(
    t[mask_new_2],
    t_mid,
    t_end,
    y_mid,
    y_end,
    m_mid,
    m_end,
)

valid_new = np.isfinite(
    y_new
)


# ============================================================
# Physical execution after handoff
# ============================================================
u = np.maximum(
    t - t_finish,
    0.0,
)

mask_after_handoff = (
    t >= t_finish
)


# State and tangent at handoff
y_handoff = np.interp(
    t_finish,
    t,
    y_exec_base,
)

m_handoff = np.interp(
    t_finish,
    t,
    dy_exec_base,
)


# Interpolated new reference
y_new_interp = np.interp(
    t,
    t[valid_new],
    y_new[valid_new],
)


# Short continuation of old physical motion
y_continuation = (
    y_handoff
    + m_handoff * u
)


# Fast switch response
tau_switch = 0.085

alpha_switch = (
    1.0
    - np.exp(
        -(u / tau_switch) ** 2
    )
)


# Growing post-handoff spatial lag
tracking_lag_after = (
    -0.016
    - 0.050
    * (
        1.0
        - np.exp(-u / 0.40)
    )
    - 0.030
    * np.maximum(
        u - 0.28,
        0.0,
    )
)


# Nominal post-switch execution
y_exec_post = (
    (1.0 - alpha_switch)
    * y_continuation
    + alpha_switch
    * (
        y_new_interp
        + tracking_lag_after
    )
)


# Downward transient / undershoot
tau_drop = 0.12

raw_drop = (
    (u / tau_drop) ** 2
    * np.exp(
        -u / tau_drop
    )
)

raw_drop_peak = np.max(
    raw_drop
)

if raw_drop_peak > 0:
    raw_drop = (
        raw_drop
        / raw_drop_peak
    )

drop_pulse = (
    -0.070
    * raw_drop
)


# Damped physical jitter
jitter_rise = (
    1.0
    - np.exp(
        -u / 0.07
    )
)

jitter_decay = (
    np.exp(
        -u / 1.00
    )
)

physical_jitter = (
    0.043
    * np.sin(
        19.0 * u
    )
    * jitter_rise
    * jitter_decay
)


# Final realized execution
y_exec = y_exec_base.copy()

y_exec[mask_after_handoff] = (
    y_exec_post[mask_after_handoff]
    + drop_pulse[mask_after_handoff]
    + physical_jitter[mask_after_handoff]
)


# ============================================================
# Colors
# ============================================================
c_old = "#4472C4"
c_exec = "#222222"
c_new = "#C55A11"
c_goal = "#888888"


# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(
    figsize=(7.45, 5.05)
)


# Goal line
ax.axhline(
    goal_y,
    color=c_goal,
    linestyle=(0, (4, 4)),
    linewidth=1.35,
    alpha=0.90,
    zorder=0,
)


# ============================================================
# Old reference
# ============================================================
mask_old_active = (
    t <= t_finish
)

mask_old_inactive = (
    t >= t_finish
)

ax.plot(
    t[mask_old_active],
    y_old[mask_old_active],
    linestyle=(0, (6, 4)),
    linewidth=2.25,
    color=c_old,
    zorder=3,
)

ax.plot(
    t[mask_old_inactive],
    y_old[mask_old_inactive],
    linestyle=(0, (6, 4)),
    linewidth=2.15,
    color=c_old,
    alpha=0.20,
    zorder=1,
)


# ============================================================
# Realized execution
# ============================================================
mask_exec_before = (
    t <= t_start
)

mask_exec_during = (
    (t >= t_start)
    & (t < t_finish)
)

mask_exec_after = (
    t >= t_finish
)

ax.plot(
    t[mask_exec_before],
    y_exec[mask_exec_before],
    linewidth=2.55,
    color=c_exec,
    zorder=5,
)

ax.plot(
    t[mask_exec_during],
    y_exec[mask_exec_during],
    linewidth=2.30,
    color=c_exec,
    alpha=0.20,
    zorder=2,
)

ax.plot(
    t[mask_exec_after],
    y_exec[mask_exec_after],
    linewidth=2.35,
    color=c_exec,
    alpha=0.55,
    zorder=5,
)


# ============================================================
# New reference
# ============================================================
mask_new_generating = (
    (t >= t_start)
    & (t < t_finish)
    & valid_new
)

mask_new_active = (
    (t >= t_finish)
    & valid_new
)

ax.plot(
    t[mask_new_generating],
    y_new[mask_new_generating],
    linestyle=(0, (5, 3)),
    linewidth=2.10,
    color=c_new,
    alpha=0.25,
    zorder=4,
)

ax.plot(
    t[mask_new_active],
    y_new[mask_new_active],
    linestyle=(0, (5, 3)),
    linewidth=2.40,
    color=c_new,
    zorder=6,
)


# ============================================================
# Axes / layout
# ============================================================
ax.set_xlim(
    0.25,
    6.12,
)

ax.set_ylim(
    0.1,
    0.9,
)

ax.set_xticks([])
ax.set_yticks([])

# Remove all axis labels and annotations.
ax.set_xlabel("")
ax.set_ylabel("")

# Remove all spines for a clean slide-ready figure.
for spine in ax.spines.values():
    spine.set_visible(False)

# Remove any padding around the figure.
plt.tight_layout(
    pad=0.1
)


# ============================================================
# Save
# ============================================================
# Transparent background is useful for slides.
# plt.savefig(
#     "planner_controller_curves_only.pdf",
#     bbox_inches="tight",
#     transparent=True,
# )

# plt.savefig(
#     "planner_controller_curves_only.png",
#     dpi=400,
#     bbox_inches="tight",
#     transparent=True,
# )

plt.show()
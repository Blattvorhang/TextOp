import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ============================================================
# Style
# ============================================================
try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass

plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 14,
    "legend.fontsize": 9.5,
    "axes.linewidth": 1.1,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ============================================================
# Helpers
# ============================================================
def hermite_segment(t, t0, t1, y0, y1, m0, m1):
    """
    Cubic Hermite interpolation.
    Guarantees value and tangent continuity at both endpoints.
    """
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
t = np.linspace(0.0, 6.25, 1600)

t_start = 3.20
delay = 1.02
t_finish = t_start + delay

# ============================================================
# Old reference
# ============================================================
def old_reference_base(t):
    return (
        0.14
        + 0.93 * np.exp(-0.5 * ((t - 2.05) / 0.90) ** 2)
        - 0.95 / (1.0 + np.exp(-(t - 3.28) * 5.0))
        - 0.085 * np.maximum(t - 4.0, 0.0)
    )


# Lift the right side slightly for a compact conceptual figure.
tail_lift = 0.34 * smooth_step(t, 3.35, sharpness=4.0)
y_old = old_reference_base(t) + tail_lift

# ============================================================
# Realized execution before handoff
# ============================================================

# Tracking mismatch deliberately exaggerated for visualization.
tracking_error = (
    -0.19 * np.exp(-0.5 * ((t - 1.70) / 0.42) ** 2)
    + 0.07 * np.exp(-0.5 * ((t - 2.42) / 0.30) ** 2)
)

# Accumulated spatial drift.
spatial_drift = -0.045 * np.maximum(t - 1.05, 0.0)

# Localized external perturbation.
t_pert = 2.68
perturbation = (
    0.115
    * np.sin(27.0 * (t - t_pert))
    * np.exp(-0.5 * ((t - t_pert) / 0.16) ** 2)
)

# Additional deviation close to replanning.
disturbance_offset = (
    -0.10
    * np.exp(-0.5 * ((t - 3.10) / 0.14) ** 2)
)

y_exec_base = (
    y_old
    + tracking_error
    + spatial_drift
    + perturbation
    + disturbance_offset
)

# ============================================================
# New reference
#
# Starts from the realized state and matches its tangent.
# ============================================================
y_exec_start = np.interp(t_start, t, y_exec_base)

dy_exec_base = np.gradient(y_exec_base, t)
m_exec_start = np.interp(t_start, t, dy_exec_base)

y_new = np.full_like(t, np.nan)

# First segment
t_mid = t_start + 0.92

mask_new_1 = (
    (t >= t_start)
    & (t <= t_mid)
)

y_mid = y_exec_start - 0.08
m_mid = -0.055

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

y_end = -0.38
m_end = -0.025

y_new[mask_new_2] = hermite_segment(
    t[mask_new_2],
    t_mid,
    t_end,
    y_mid,
    y_end,
    m_mid,
    m_end,
)

valid_new = np.isfinite(y_new)

# ============================================================
# Boundary values
# ============================================================
y_old_finish = np.interp(
    t_finish,
    t,
    y_old,
)

y_new_finish = np.interp(
    t_finish,
    t[valid_new],
    y_new[valid_new],
)

# ============================================================
# Physical response after inference finishes
#
# IMPORTANT:
# no discontinuity in execution itself.
#
# u = 0 exactly at t_finish.
# Both the pulse and jitter start from zero smoothly.
# ============================================================
u = np.maximum(t - t_finish, 0.0)

# ------------------------------------------------------------
# Smooth upward response
#
# u^2 exp(-u/tau) starts with:
#   value      = 0
#   derivative = 0
#
# Therefore execution remains smooth at the handoff.
# ------------------------------------------------------------
tau_impact = 0.13

raw_impact = (
    (u / tau_impact) ** 2
    * np.exp(-u / tau_impact)
)

# Normalize peak to 1.
raw_peak = np.max(raw_impact)
if raw_peak > 0:
    raw_impact = raw_impact / raw_peak

smooth_impact = 0.16 * raw_impact

# ------------------------------------------------------------
# Damped jitter
#
# The rise envelope ensures jitter also starts smoothly.
# ------------------------------------------------------------
jitter_rise = (
    1.0
    - np.exp(-u / 0.10)
)

jitter_decay = np.exp(-u / 0.85)

smooth_jitter = (
    0.045
    * np.sin(19.0 * u)
    * jitter_rise
    * jitter_decay
)

# Final physical execution.
y_exec = (
    y_exec_base
    + smooth_impact
    + smooth_jitter
)

# ============================================================
# Tracking Gap annotation
# ============================================================
t_track = 1.72

y_old_track = np.interp(
    t_track,
    t,
    y_old,
)

y_exec_track = np.interp(
    t_track,
    t,
    y_exec,
)

track_mid = 0.5 * (
    y_old_track
    + y_exec_track
)

actual_half_gap = (
    0.5
    * abs(y_old_track - y_exec_track)
)

# Exaggerated for schematic readability.
visual_half_gap = max(
    actual_half_gap * 1.35,
    0.075,
)

track_top = (
    track_mid
    + visual_half_gap
)

track_bottom = (
    track_mid
    - visual_half_gap
)

# ============================================================
# Colors
# ============================================================
c_old = "#4472C4"
c_exec = "#222222"
c_new = "#C55A11"

c_tracking = "#7A5195"
c_distribution = "#4E8B8B"
c_delay = "#B23A48"
c_boundary = "#D62728"
c_gray = "#555555"

# ============================================================
# Figure
# ============================================================
fig, ax = plt.subplots(
    figsize=(7.15, 4.9)
)

# ============================================================
# Old reference
#
# Becomes faded once the new reference becomes available.
# ============================================================
mask_old_active = t <= t_finish
mask_old_inactive = t >= t_finish

ax.plot(
    t[mask_old_active],
    y_old[mask_old_active],
    linestyle=(0, (6, 4)),
    linewidth=2.25,
    color=c_old,
    label=r"old reference $r^{\mathrm{old}}$",
    zorder=3,
)

ax.plot(
    t[mask_old_inactive],
    y_old[mask_old_inactive],
    linestyle=(0, (6, 4)),
    linewidth=2.15,
    color=c_old,
    alpha=0.23,
    zorder=1,
)

# ============================================================
# Realized execution
#
# Clear before inference starts, faded afterwards.
# ============================================================
mask_exec_before = t <= t_start
mask_exec_after = t >= t_start

ax.plot(
    t[mask_exec_before],
    y_exec[mask_exec_before],
    linewidth=2.55,
    color=c_exec,
    label=r"realized execution $x^{\mathrm{exec}}$",
    zorder=5,
)

ax.plot(
    t[mask_exec_after],
    y_exec[mask_exec_after],
    linewidth=2.35,
    color=c_exec,
    alpha=0.23,
    zorder=2,
)

# ============================================================
# New reference
#
# Transparent while being generated,
# opaque after inference finishes.
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
    linewidth=2.1,
    color=c_new,
    alpha=0.32,
    zorder=4,
)

ax.plot(
    t[mask_new_active],
    y_new[mask_new_active],
    linestyle=(0, (5, 3)),
    linewidth=2.4,
    color=c_new,
    label=r"new reference $r^{\mathrm{new}}$",
    zorder=6,
)

# ============================================================
# Inference start / finish markers
# ============================================================
ax.axvline(
    t_start,
    ymin=0.08,
    ymax=0.94,
    color="0.30",
    linestyle=(0, (2, 4)),
    linewidth=1.15,
)

ax.axvline(
    t_finish,
    ymin=0.08,
    ymax=0.84,
    color="0.30",              # black/gray like inference starts
    linestyle=(0, (2, 4)),
    linewidth=1.15,
)

ax.text(
    t_start,
    1.245,
    "inference starts",
    ha="center",
    va="bottom",
    fontsize=13.0,
    color="black",
)

ax.text(
    t_finish,
    1.00,
    "inference finishes",
    ha="center",
    va="bottom",
    fontsize=11.8,
    color="black",
)

# ============================================================
# Tracking Gap
# ============================================================
tracking_arrow = FancyArrowPatch(
    (t_track, track_bottom),
    (t_track, track_top),
    arrowstyle="<->",
    mutation_scale=19,
    linewidth=2.0,
    color=c_tracking,
    zorder=12,
)

ax.add_patch(
    tracking_arrow
)

ax.annotate(
    "Tracking Gap",
    xy=(
        t_track,
        track_top,
    ),
    xytext=(
        0.88,
        1.075,
    ),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.25,
        color=c_tracking,
    ),
    fontsize=13,
    color=c_tracking,
    fontstyle="italic",
    ha="center",
)

# ============================================================
# External perturbation
# ============================================================
y_pert = np.interp(
    t_pert,
    t,
    y_exec,
)

ax.annotate(
    "external perturbation",
    xy=(
        t_pert,
        y_pert,
    ),
    xytext=(
        1.63,
        0.02,
    ),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.25,
        color=c_gray,
    ),
    fontsize=11.7,
    color=c_gray,
    ha="left",
)

# ============================================================
# Feedback anchor
# ============================================================
ax.scatter(
    [t_start],
    [y_exec_start],
    s=36,
    color=c_new,
    edgecolor="white",
    linewidth=0.8,
    zorder=13,
)

# ============================================================
# Execution Distribution Gap
# ============================================================
history_start = 2.61
history_end = t_start
history_y = -0.73

ax.annotate(
    "",
    xy=(
        history_end,
        history_y,
    ),
    xytext=(
        history_start,
        history_y,
    ),
    arrowprops=dict(
        arrowstyle="|-|",
        linewidth=1.6,
        color=c_distribution,
    ),
)

ax.text(
    history_start - 0.05,
    history_y + 0.075,
    "feedback history",
    ha="left",
    va="bottom",
    fontsize=10.0,
    color="0.42",
)

ax.text(
    2.34,
    -0.96,
    "Execution Distribution Gap",
    ha="center",
    va="top",
    fontsize=12.2,
    color=c_distribution,
    fontstyle="italic",
)

# ============================================================
# Inference Delay
#
# No brace/end ticks are drawn here.
# Leave clean space for manual annotation.
# ============================================================
delay_y = -0.39

ax.text(
    0.5 * (t_start + t_finish),
    delay_y,
    r"Inference Delay $d$",
    ha="center",
    va="center",
    fontsize=12.5,
    color=c_delay,
    fontstyle="italic",
)

# ============================================================
# Boundary Discontinuity
# ============================================================
x_boundary = t_finish + 0.055

boundary_arrow = FancyArrowPatch(
    (
        x_boundary,
        y_new_finish,
    ),
    (
        x_boundary,
        y_old_finish,
    ),
    arrowstyle="<->",
    mutation_scale=15,
    linewidth=1.55,
    color=c_boundary,
    zorder=12,
)

ax.add_patch(
    boundary_arrow
)

ax.annotate(
    "Boundary\nDiscontinuity",
    xy=(
        x_boundary,
        0.5 * (
            y_new_finish
            + y_old_finish
        ),
    ),
    xytext=(
        4.87,
        0.36,
    ),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.15,
        color=c_boundary,
    ),
    color=c_boundary,
    fontsize=11.7,
    ha="left",
)

# ============================================================
# Physical consequence
# ============================================================
ax.text(
    4.72,
    -0.82,
    r"$\rightarrow$ physical jitter",
    fontsize=10.7,
    color="0.45",
    ha="left",
)

# ============================================================
# Axes / layout
# ============================================================
ax.set_xlim(
    0.25,
    6.12,
)

ax.set_ylim(
    -1.05,
    1.31,
)

ax.set_xlabel(
    "time",
    fontsize=15,
)

ax.set_ylabel(
    "motion state / spatial trajectory",
    fontsize=13.5,
)

ax.set_xticks([])
ax.set_yticks([])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.0)

# ============================================================
# Legend
# ============================================================
ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.01, 1.025),
    frameon=False,
    fontsize=9.3,
    handlelength=2.1,
    borderpad=0.05,
    labelspacing=0.18,
)

plt.tight_layout()

# plt.savefig(
#     "planner_controller_gaps.pdf",
#     bbox_inches="tight",
# )
#
# plt.savefig(
#     "planner_controller_gaps.png",
#     dpi=400,
#     bbox_inches="tight",
# )

plt.show()
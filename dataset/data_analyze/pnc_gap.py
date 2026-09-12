import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D


# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
})

# Explicitly keep the same semantic colors as your original figure
C_REF = "#2ca02c"       # planner/reference
C_EXEC = "#2878B5"      # SONIC execution
C_PHASE = "#8a8a8a"     # phase-aligned reference
C_NEW = "#D62728"       # newly replanned chunk
C_OLD = "#202020"       # currently executing old chunk
C_AUX = "#777777"


# ============================================================
# Helpers
# ============================================================
def smoothstep(x):
    """0 -> 1 smooth transition."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ref_motion(t):
    """
    Abstract planner/reference state.
    It is intentionally smooth but nontrivial so phase lag is visible.
    """
    return (
        0.15 * t
        + 0.48 * np.sin(0.92 * t)
        + 0.10 * np.sin(1.90 * t + 0.35)
    )


def interp(tq, t, y):
    return np.interp(tq, t, y)


def double_arrow(ax, xy1, xy2, color, lw=1.4, ms=11, zorder=8):
    patch = FancyArrowPatch(
        xy1, xy2,
        arrowstyle="<->",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        zorder=zorder
    )
    ax.add_patch(patch)
    return patch


# ============================================================
# Construct synthetic planner / execution signals
# ============================================================
t = np.linspace(0.0, 12.0, 1201)
dt = t[1] - t[0]

ref = ref_motion(t)

# ------------------------------------------------------------
# 1) Execution phase lag
#
# Initially ~0.
# Then gradually grows to ~0.65 s.
# This is NOT inference latency: it represents physical
# controller / robot motion phase lag.
# ------------------------------------------------------------
tau_max = 0.65
tau = tau_max * smoothstep((t - 2.7) / 2.0)

t_phase = np.clip(t - tau, 0.0, t[-1])
phase_aligned_ref = ref_motion(t_phase)


# ------------------------------------------------------------
# 2) Bounded tracking deviation
#
# IMPORTANT:
# This oscillates around the phase-aligned reference, so execution
# can be either above or below reference.
# ------------------------------------------------------------
tracking_error = (
    0.055 * np.sin(5.0 * t + 0.4)
    + 0.020 * np.sin(8.2 * t - 0.7)
)


# ------------------------------------------------------------
# 3) Spatial drift
#
# Small initially, then accumulates.
# Think of this as global XY / heading drift that SONIC's local
# joint tracking cannot fully eliminate.
# ------------------------------------------------------------
drift_start = 5.8
drift_progress = np.maximum(t - drift_start, 0.0)

spatial_drift = -0.038 * drift_progress**1.45


# ------------------------------------------------------------
# Optional disturbance:
# makes the realized history visibly more "physical" / off-manifold.
# ------------------------------------------------------------
disturbance = -0.20 * np.exp(
    -0.5 * ((t - 7.35) / 0.16) ** 2
)


# Full realized execution
execution = (
    phase_aligned_ref
    + tracking_error
    + spatial_drift
    + disturbance
)


# ============================================================
# Inference / replanning timing
# ============================================================
t_infer_start = 7.55
t_infer_finish = 9.05

i_start = np.argmin(np.abs(t - t_infer_start))
i_finish = np.argmin(np.abs(t - t_infer_finish))

y_exec_start = execution[i_start]
y_exec_finish = execution[i_finish]

y_ref_finish = ref[i_finish]


# ============================================================
# Construct a newly generated chunk
#
# It is conditioned on the state sampled at inference start.
# Conceptually it exists from t_infer_start onward, but becomes
# available only at t_infer_finish.
#
# We deliberately make it a different but individually smooth
# continuation so that switching to it at inference finish produces
# a replanning seam.
# ============================================================
mask_new = t >= t_infer_start
t_new = t[mask_new]
u = t_new - t_infer_start

# estimate local velocity at inference start
exec_vel = np.gradient(execution, t)
v0 = exec_vel[i_start]

new_plan = (
    y_exec_start
    + 0.55 * v0 * u
    - 0.075 * u
    - 0.34 * (
        np.sin(0.86 * u + 0.20) - np.sin(0.20)
    )
)

mask_unavailable = (
    (t_new >= t_infer_start)
    & (t_new < t_infer_finish)
)
mask_available = t_new >= t_infer_finish

y_new_finish = interp(
    t_infer_finish,
    t_new,
    new_plan
)


# ============================================================
# Plot
# ============================================================
fig, ax = plt.subplots(figsize=(15.5, 7.2))


# ------------------------------------------------------------
# Main reference and execution
# ------------------------------------------------------------
ax.plot(
    t, ref,
    linestyle="--",
    linewidth=2.7,
    color=C_REF,
    label="Planner prediction / reference",
    zorder=3
)

ax.plot(
    t, execution,
    linewidth=2.8,
    color=C_EXEC,
    label="Realized execution / SONIC",
    zorder=4
)


# ------------------------------------------------------------
# Phase-aligned reference:
#
# This answers:
#   "Which older planner state does the current execution correspond to?"
# ------------------------------------------------------------
phase_show = (t >= 2.6) & (t <= 8.5)

ax.plot(
    t[phase_show],
    phase_aligned_ref[phase_show],
    linestyle=(0, (2, 2)),
    linewidth=1.8,
    color=C_PHASE,
    alpha=0.85,
    zorder=2
)


# ============================================================
# (1) TRACKING DEVIATION
# execution alternates around reference
# ============================================================
tracking_times = [0.85, 1.25, 1.65, 2.05, 2.38]

for tt in tracking_times:
    yr = interp(tt, t, ref)
    ye = interp(tt, t, execution)

    double_arrow(
        ax,
        (tt, yr),
        (tt, ye),
        color=C_AUX,
        lw=1.05,
        ms=8
    )

ax.text(
    0.55, 1.42,
    "(1) Tracking deviation",
    fontweight="bold",
    fontsize=12.5
)

ax.text(
    0.55, 1.22,
    "bounded local error\noscillates around reference",
    fontsize=10.5,
    color=C_AUX
)


# ============================================================
# (2) EXECUTION PHASE LAG
#
# Detect one characteristic peak in reference and the corresponding
# later peak in execution.
# ============================================================

# Find reference peak in a controlled interval
mask_peak_ref = (t >= 3.1) & (t <= 4.8)
idx_r_local = np.argmax(ref[mask_peak_ref])
idx_r = np.where(mask_peak_ref)[0][idx_r_local]

t_peak_ref = t[idx_r]
y_peak_ref = ref[idx_r]

# Find execution peak later
mask_peak_exec = (
    (t >= t_peak_ref + 0.20)
    & (t <= t_peak_ref + 1.15)
)
idx_e_local = np.argmax(execution[mask_peak_exec])
idx_e = np.where(mask_peak_exec)[0][idx_e_local]

t_peak_exec = t[idx_e]
y_peak_exec = execution[idx_e]

y_lag_arrow = max(y_peak_ref, y_peak_exec) + 0.30

ax.vlines(
    [t_peak_ref, t_peak_exec],
    ymin=[y_peak_ref, y_peak_exec],
    ymax=y_lag_arrow,
    linestyles=":",
    linewidth=1.2,
    color=C_AUX
)

double_arrow(
    ax,
    (t_peak_ref, y_lag_arrow),
    (t_peak_exec, y_lag_arrow),
    color=C_AUX,
    lw=1.3,
    ms=10
)

ax.text(
    0.5 * (t_peak_ref + t_peak_exec),
    y_lag_arrow + 0.08,
    r"(2) execution phase lag $\tau$",
    ha="center",
    fontsize=11.5,
    fontweight="bold"
)

ax.scatter(
    [t_peak_ref],
    [y_peak_ref],
    s=48,
    color=C_REF,
    zorder=8
)

ax.scatter(
    [t_peak_exec],
    [y_peak_exec],
    s=48,
    color=C_EXEC,
    zorder=8
)


# ============================================================
# Spatial drift annotation
#
# Compare execution to phase-aligned reference.
# This removes the pure phase-lag component first.
# ============================================================
drift_times = [6.15, 6.75, 7.30]

for tt in drift_times:
    yp = interp(tt, t, phase_aligned_ref)
    ye = interp(tt, t, execution)

    double_arrow(
        ax,
        (tt, yp),
        (tt, ye),
        color=C_AUX,
        lw=1.0,
        ms=8
    )

ax.text(
    6.02, -0.13,
    "phase-aligned physical residual",
    color=C_AUX,
    fontsize=9.8,
    rotation=-8
)

ax.annotate(
    "global spatial drift accumulates",
    xy=(7.05, interp(7.05, t, execution)),
    xytext=(5.65, -0.82),
    fontsize=10.5,
    color=C_EXEC,
    arrowprops=dict(
        arrowstyle="->",
        lw=1.1,
        color=C_EXEC
    )
)


# ============================================================
# (3) EXECUTION-FEEDBACK DISTRIBUTION SHIFT
#
# Highlight a recent raw realized-history segment that would be
# directly fed to an autoregressive planner by a naive method.
# ============================================================
feedback_window_start = t_infer_start - 0.75

feedback_mask = (
    (t >= feedback_window_start)
    & (t <= t_infer_start)
)

# Emphasize the raw realized history with dots
idx_fb = np.where(feedback_mask)[0][::12]

ax.scatter(
    t[idx_fb],
    execution[idx_fb],
    s=28,
    facecolor="white",
    edgecolor=C_EXEC,
    linewidth=1.3,
    zorder=8
)

# horizontal bracket beneath the raw history
y_feedback_bracket = min(execution[feedback_mask]) - 0.32

ax.annotate(
    "",
    xy=(feedback_window_start, y_feedback_bracket),
    xytext=(t_infer_start, y_feedback_bracket),
    arrowprops=dict(
        arrowstyle="|-|",
        lw=1.3,
        color=C_AUX
    )
)

ax.text(
    0.5 * (feedback_window_start + t_infer_start),
    y_feedback_bracket - 0.10,
    "(3) raw realized history",
    ha="center",
    va="top",
    fontsize=10.8,
    fontweight="bold"
)

ax.text(
    0.5 * (feedback_window_start + t_infer_start),
    y_feedback_bracket - 0.29,
    "dynamics / disturbance $\\rightarrow$ distribution shift",
    ha="center",
    va="top",
    fontsize=9.8,
    color=C_AUX
)


# ============================================================
# (4) INFERENCE STALENESS
#
# Closely follows the conceptual structure of real-time action
# chunking:
#
# t_start: observation sampled
# [start, finish]: old chunk keeps running
# t_finish: new chunk becomes available
# ============================================================

# Current old chunk during inference
old_mask = (
    (t >= t_infer_start)
    & (t <= t_infer_finish)
)

ax.plot(
    t[old_mask],
    ref[old_mask],
    color=C_OLD,
    linewidth=3.2,
    zorder=6
)


# Inference start / finish markers
y_top = 2.25
y_bottom = -1.42

ax.vlines(
    t_infer_start,
    ymin=y_bottom + 0.25,
    ymax=y_top - 0.12,
    linestyles=(0, (4, 3)),
    linewidth=1.25,
    color=C_AUX
)

ax.vlines(
    t_infer_finish,
    ymin=y_bottom + 0.25,
    ymax=y_top - 0.12,
    linestyles=(0, (4, 3)),
    linewidth=1.25,
    color=C_AUX
)

ax.text(
    t_infer_start,
    y_top,
    "inference starts",
    ha="center",
    va="bottom",
    fontweight="bold"
)

ax.text(
    t_infer_finish,
    y_top,
    "inference finishes",
    ha="center",
    va="bottom",
    fontweight="bold"
)

# inference-delay arrow
y_delay = y_top - 0.29

double_arrow(
    ax,
    (t_infer_start, y_delay),
    (t_infer_finish, y_delay),
    color=C_AUX,
    lw=1.3,
    ms=10
)

ax.text(
    0.5 * (t_infer_start + t_infer_finish),
    y_delay + 0.08,
    r"(4) inference delay $d$",
    ha="center",
    va="bottom",
    fontsize=11.3,
    fontweight="bold"
)


# ------------------------------------------------------------
# Stale state:
# snapshot from inference start propagated only in wall-clock time
# ------------------------------------------------------------
ax.plot(
    [t_infer_start, t_infer_finish],
    [y_exec_start, y_exec_start],
    linestyle=":",
    linewidth=1.5,
    color=C_AUX,
    zorder=2
)

double_arrow(
    ax,
    (t_infer_finish, y_exec_start),
    (t_infer_finish, y_exec_finish),
    color=C_AUX,
    lw=1.15,
    ms=9
)

ax.text(
    t_infer_finish + 0.10,
    0.5 * (y_exec_start + y_exec_finish),
    "state\nstaleness",
    fontsize=9.5,
    color=C_AUX,
    va="center"
)


# ============================================================
# New replanned chunk
# ============================================================

# unavailable part during inference
ax.plot(
    t_new[mask_unavailable],
    new_plan[mask_unavailable],
    linestyle=(0, (2, 2)),
    linewidth=2.0,
    color=C_NEW,
    alpha=0.38,
    zorder=2
)

# available new chunk after inference
ax.plot(
    t_new[mask_available],
    new_plan[mask_available],
    linewidth=3.0,
    color=C_NEW,
    zorder=6
)


# small markers on the new chunk
idx_new = np.where(mask_available)[0][::45]

ax.scatter(
    t_new[idx_new],
    new_plan[idx_new],
    s=25,
    color=C_NEW,
    zorder=7
)


ax.text(
    t_infer_start + 0.12,
    y_exec_start - 0.33,
    "new chunk being inferred\n(not yet available)",
    fontsize=9.3,
    color=C_NEW,
    alpha=0.62
)


# ============================================================
# (5) REPLANNING DISCONTINUITY
#
# At inference finish, execution/old-plan state has evolved, while
# the new chunk was generated from stale/raw feedback.
# ============================================================

y_old_at_finish = interp(
    t_infer_finish,
    t,
    ref
)

double_arrow(
    ax,
    (t_infer_finish, y_old_at_finish),
    (t_infer_finish, y_new_finish),
    color=C_NEW,
    lw=2.0,
    ms=12
)

ax.scatter(
    [t_infer_finish],
    [y_old_at_finish],
    s=45,
    color=C_OLD,
    zorder=9
)

ax.scatter(
    [t_infer_finish],
    [y_new_finish],
    s=45,
    color=C_NEW,
    zorder=9
)

ax.annotate(
    "(5) replanning discontinuity",
    xy=(
        t_infer_finish,
        0.5 * (y_old_at_finish + y_new_finish)
    ),
    xytext=(9.45, 0.55),
    fontsize=12,
    fontweight="bold",
    color=C_NEW,
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.4,
        color=C_NEW
    )
)

ax.text(
    9.47, 0.30,
    "tracking + phase + feedback + staleness\n"
    "manifest at the chunk boundary",
    fontsize=10.2,
    color=C_AUX
)


# ============================================================
# Time-direction arrows
# ============================================================
ax.annotate(
    "",
    xy=(12.05, ref[-1]),
    xytext=(11.72, ref[-1]),
    arrowprops=dict(
        arrowstyle="->",
        lw=2.5,
        color=C_REF
    ),
    annotation_clip=False
)

ax.annotate(
    "",
    xy=(12.05, execution[-1]),
    xytext=(11.72, execution[-1]),
    arrowprops=dict(
        arrowstyle="->",
        lw=2.5,
        color=C_EXEC
    ),
    annotation_clip=False
)


# ============================================================
# Axis / legend / title
# ============================================================
ax.set_xlabel("Wall-clock time")
ax.set_ylabel("Motion / spatial state")

ax.set_title(
    "Planning–Control Gaps in Autoregressive Humanoid Replanning",
    pad=16,
    fontweight="bold",
    fontsize=15
)

ax.set_xlim(0.0, 12.25)
ax.set_ylim(-1.72, 2.55)

# schematic: numeric y ticks are not important
ax.set_yticks([])

# keep x-axis sparse
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


legend_handles = [
    Line2D(
        [0], [0],
        color=C_REF,
        lw=2.7,
        linestyle="--",
        label="Planner prediction / reference"
    ),
    Line2D(
        [0], [0],
        color=C_EXEC,
        lw=2.8,
        label="Realized execution / SONIC"
    ),
    Line2D(
        [0], [0],
        color=C_PHASE,
        lw=1.8,
        linestyle=(0, (2, 2)),
        label=r"Phase-aligned reference $r(t-\tau)$"
    ),
    Line2D(
        [0], [0],
        color=C_OLD,
        lw=3.0,
        label="Old chunk executing during inference"
    ),
    Line2D(
        [0], [0],
        color=C_NEW,
        lw=3.0,
        label="New replanned chunk"
    ),
]

ax.legend(
    handles=legend_handles,
    loc="lower left",
    bbox_to_anchor=(0.01, 0.015),
    ncol=2,
    frameon=False,
    columnspacing=1.7,
    handlelength=3.1
)

plt.tight_layout()


# ============================================================
# Export
# ============================================================
plt.savefig(
    "planning_control_gaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "planning_control_gaps.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "planning_control_gaps.svg",
    bbox_inches="tight"
)

plt.show()
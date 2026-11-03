import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Rotation utilities
# ============================================================

def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c,  -s],
        [0.0, s,   c],
    ])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c,  -s, 0.0],
        [s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def matrix_to_rpy_zyx(R):
    """
    Canonical ZYX Euler decomposition:
        R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Returns:
        roll, pitch, yaw
    """
    pitch = np.arctan2(
        -R[2, 0],
        np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    )

    roll = np.arctan2(
        R[2, 1],
        R[2, 2]
    )

    yaw = np.arctan2(
        R[1, 0],
        R[0, 0]
    )

    return roll, pitch, yaw


def matrix_to_rot6d(R):
    """
    Matrix-based Rot6D:
    keep the first two columns of R.

    Output:
        [r11, r21, r31, r12, r22, r32]
    """
    return np.concatenate(
        [R[:, 0], R[:, 1]],
        axis=0
    )


# ============================================================
# Construct continuous rollover motion
# ============================================================

# Even N -> avoid sampling exactly alpha = 90 deg
N = 1000

alpha_deg = np.linspace(
    0.0,
    180.0,
    N
)

alpha = np.deg2rad(alpha_deg)

# Initial side-lying configuration
R0 = Rx(np.pi / 2.0)

# Continuous body-local z rotation
Rs = np.stack([
    R0 @ Rz(a)
    for a in alpha
], axis=0)


# ============================================================
# Representations
# ============================================================

rpy = np.stack([
    matrix_to_rpy_zyx(R)
    for R in Rs
], axis=0)

roll = rpy[:, 0]
pitch = rpy[:, 1]
yaw = rpy[:, 2]

roll_deg = np.rad2deg(roll)
pitch_deg = np.rad2deg(pitch)
yaw_deg = np.rad2deg(yaw)


rot6d = np.stack([
    matrix_to_rot6d(R)
    for R in Rs
], axis=0)


# ============================================================
# Plot style
# ============================================================

plt.rcParams.update({
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "figure.dpi": 180,
    "mathtext.fontset": "dejavusans",
})

# ============================================================
# User-selected palette
# ============================================================

c_blue  = "#7E99F4"   # RGB 126,153,244
c_red   = "#CC7C71"   # RGB 204,124,113
c_green = "#7AB656"   # RGB 122,182,86

# Same semantic mapping in BOTH subplots
c_roll  = c_red
c_pitch = c_green
c_yaw   = c_blue

c_x = c_red
c_y = c_green
c_z = c_blue

grid_color = "#D9DDE5"
singularity_color = "#8A8A8A"

singularity = 90.0
xticks = [0, 45, 90, 135, 180]


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(9.0, 5.3),
    sharex=True,
    gridspec_kw={
        "hspace": 0.25
    }
)

fig.subplots_adjust(
    left=0.11,
    right=0.78,
    bottom=0.13,
    top=0.95
)


def stylize_axis(
    ax,
    ylabel,
    title,
    ylim,
    yticks
):
    # singular configuration reference
    ax.axvline(
        singularity,
        color=singularity_color,
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=0
    )

    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)

    ax.set_title(
        title,
        loc="left",
        fontsize=10.8,
        fontweight="semibold",
        pad=5
    )

    ax.grid(
        True,
        color=grid_color,
        linewidth=0.75,
        alpha=0.6
    )


# ============================================================
# (a) Raw RPY
# ============================================================

ax = axes[0]

ax.plot(
    alpha_deg,
    roll_deg,
    color=c_roll,
    linewidth=2.15,
    label=r"roll $\phi$"
)

ax.plot(
    alpha_deg,
    pitch_deg,
    color=c_pitch,
    linewidth=2.15,
    label=r"pitch $\theta$"
)

ax.plot(
    alpha_deg,
    yaw_deg,
    color=c_yaw,
    linewidth=2.15,
    label=r"yaw $\psi$"
)

stylize_axis(
    ax,
    ylabel="Angle (deg)",
    title="(a) Raw RPY representation",
    ylim=(-200, 200),
    yticks=[-180, -90, 0, 90, 180]
)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=9.3,
    handlelength=2.3
)


# ============================================================
# (b) Matrix Rot6D
# ============================================================

ax = axes[1]

rot6d_specs = [
    (r"$r_{11}$", c_x, "-"),
    (r"$r_{21}$", c_y, "-"),
    (r"$r_{31}$", c_z, "-"),
    (r"$r_{12}$", c_x, "--"),
    (r"$r_{22}$", c_y, "--"),
    (r"$r_{32}$", c_z, "--"),
]

for i, (label, color, linestyle) in enumerate(rot6d_specs):

    # Make dashed curves slightly thicker so they remain visible
    if linestyle == "--":
        lw = 2.8
    else:
        lw = 2.0

    ax.plot(
        alpha_deg,
        rot6d[:, i],
        color=color,
        linestyle=linestyle,
        linewidth=lw,
        label=label
    )

stylize_axis(
    ax,
    ylabel="Component value",
    title="(b) Matrix-based Rot6D representation",
    ylim=(-1.15, 1.15),
    yticks=[-1, 0, 1]
)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=False,
    fontsize=9.1,
    handlelength=2.3
)

ax.set_xlabel(
    r"Continuous body-$z$ roll-over angle $\alpha$ (deg)",
    labelpad=5
)


# ============================================================
# Save
# ============================================================

plt.savefig(
    "rpy_vs_rot6d_final_palette.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "rpy_vs_rot6d_final_palette.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
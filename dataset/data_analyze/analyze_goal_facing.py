#!/usr/bin/env python3
"""
Analyze root facing direction distribution at goal frames (offset=0) in BONES-SEED.

Determines whether yaw-only is sufficient for goal facing conditioning, or whether
roll/pitch also need to be encoded.

For each snippet window (TextOp-style 35-frame segments), the goal frame's root
rotation quaternion is decomposed into euler angles (roll, pitch, yaw) relative to
the snippet's egocentric frame. If roll/pitch cluster tightly around 0 while yaw
spreads broadly, then 2D (cosθ, sinθ) yaw-only is the right encoding.

Output:
  - 2D histogram: pitch vs roll scatter at goal
  - polar histogram of yaw distribution
  - per-component CDF + basic stats
"""

import argparse, csv, os, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ORIGINAL_FPS = 120
TARGET_FPS = 50
CM_TO_M = 1.0 / 100.0

HISTORY_LEN = 2
FUTURE_LEN = 8
NUM_PRIMITIVE = 4
SEGMENT_LEN = HISTORY_LEN + FUTURE_LEN * NUM_PRIMITIVE + 1  # = 35

DOF_29_TO_23_MASK_ARR = np.array([
    True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,
    True,  True,  True,
    True,  True,  True,  True,
    False, False, False,
    True,  True,  True,  True,
    False, False, False,
])

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
MAX_FILES = 5000
SLIDE_STEP = FUTURE_LEN  # = 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_csv_files(base_dir):
    files = []
    for root, _dirs, fnames in os.walk(base_dir):
        for fn in fnames:
            if fn.endswith(".csv") and not fn.endswith("_M.csv"):
                files.append(os.path.join(root, fn))
    return files


def load_and_downsample_csv(filepath):
    with open(filepath) as fh:
        reader = csv.reader(fh)
        next(reader)
        data = list(reader)
    if len(data) < 3:
        return None
    data = np.array(data, dtype=np.float32)
    trans_120 = data[:, 1:4] * CM_TO_M
    rot_euler_120 = data[:, 4:7]
    T_orig = trans_120.shape[0]
    t_orig = np.linspace(0, (T_orig - 1) / ORIGINAL_FPS, T_orig)
    T_target = max(int(T_orig * TARGET_FPS / ORIGINAL_FPS), 2)
    t_target = np.linspace(0, (T_orig - 1) / ORIGINAL_FPS, T_target)
    trans_50 = interp1d(t_orig, trans_120, axis=0, kind="linear")(t_target)
    slerp = Slerp(t_orig, R.from_euler("XYZ", np.deg2rad(rot_euler_120)))
    rot_quat_50 = slerp(t_target).as_quat()  # xyzw
    return trans_50.astype(np.float32), rot_quat_50.astype(np.float32)


def get_root_forward(rot_quat_xyzw):
    rot = R.from_quat(rot_quat_xyzw)
    fwd_world = rot.apply(np.array([1.0, 0.0, 0.0]))
    fwd_xy = fwd_world[:, :2]
    norm = np.linalg.norm(fwd_xy, axis=-1, keepdims=True)
    return fwd_xy / np.clip(norm, 1e-10, None)


def world_to_ego(world_xy, origin_xy, fwd_xy):
    d_xy = world_xy - origin_xy
    theta = np.arctan2(fwd_xy[0], fwd_xy[1])
    c, s = np.cos(theta), np.sin(theta)
    ego_x = c * d_xy[..., 0] + s * d_xy[..., 1]
    ego_y = -s * d_xy[..., 0] + c * d_xy[..., 1]
    return np.stack([ego_x, ego_y], axis=-1)


def quat_to_ego_euler(goal_quat_xyzw, ref_quat_xyzw):
    """Decompose goal orientation into euler (roll, pitch, yaw) relative to ref frame.

    ref_quat = snippet start frame's world orientation.
    ego_quat = inv(ref) * goal  → euler ZYX in radians.
    Returns: (roll, pitch, yaw) all in [-π, π].
    """
    ref_rot = R.from_quat(ref_quat_xyzw)
    goal_rot = R.from_quat(goal_quat_xyzw)
    ego_rot = ref_rot.inv() * goal_rot
    return ego_rot.as_euler('ZYX')  # (yaw, pitch, roll) — note scipy order!


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/home/lenovo/data/bones-seed/g1/csv")
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    args = parser.parse_args()

    all_files = collect_csv_files(args.base_dir)
    print(f"Found {len(all_files):,} CSV files")
    np.random.seed(RANDOM_SEED)
    if len(all_files) > args.max_files:
        files = np.random.choice(all_files, args.max_files, replace=False).tolist()
        print(f"Sampled {len(files):,}")
    else:
        files = all_files

    yaws, pitches, rolls = [], [], []
    n_snippets = 0

    for fp in tqdm(files, desc="Processing"):
        motion = load_and_downsample_csv(fp)
        if motion is None:
            continue
        trans, rot_quat = motion
        T = trans.shape[0]
        if T < SEGMENT_LEN + 1:
            continue

        for seg_start in range(0, T - SEGMENT_LEN, SLIDE_STEP):
            goal_idx = seg_start + SEGMENT_LEN - 1  # offset=0
            if goal_idx >= T:
                break

            ref_quat = rot_quat[seg_start]
            goal_quat = rot_quat[goal_idx]
            yaw, pitch, roll = quat_to_ego_euler(goal_quat, ref_quat)
            yaws.append(yaw)
            pitches.append(pitch)
            rolls.append(roll)
            n_snippets += 1

    yaws    = np.array(yaws)
    pitches = np.array(pitches)
    rolls   = np.array(rolls)

    print(f"\nTotal snippets: {n_snippets:,}")

    # ── Stats ──
    for name, arr in [("Roll", rolls), ("Pitch", pitches), ("Yaw", yaws)]:
        q = np.percentile(np.abs(arr), [50, 90, 95, 99])
        print(f"  {name:>6s}: |mean|={np.abs(arr).mean():.4f} rad ({np.degrees(np.abs(arr).mean()):.1f}°)  "
              f"P50={q[0]:.4f} P90={q[1]:.4f} P95={q[2]:.4f} P99={q[3]:.4f} rad")

    # ── Figure 1: pitch vs roll scatter ──
    fig, ax = plt.subplots(figsize=(8, 8))
    n = min(len(rolls), 20000)
    idx = np.random.choice(len(rolls), n, replace=False)
    ax.scatter(np.degrees(rolls[idx]), np.degrees(pitches[idx]),
               s=1, alpha=0.3, c="steelblue", edgecolors="none", rasterized=True)
    ax.scatter([0], [0], c="red", s=50, marker="x", linewidths=2, zorder=5, label="origin (ego=ref)")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Roll (deg)", fontsize=12)
    ax.set_ylabel("Pitch (deg)", fontsize=12)
    ax.set_title(f"Goal Frame Orientation Relative to Snippet Start\n"
                 f"(n={n_snippets:,} snippets, sampled {n:,} shown)", fontsize=13)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "goal_facing_pitch_vs_roll.png", dpi=150)
    plt.close(fig)
    print(f"[Scatter] Saved: {OUTPUT_DIR / 'goal_facing_pitch_vs_roll.png'}")

    # ── Figure 2: yaw polar distribution ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # polar histogram
    ax = axes[0]
    bins = 72
    counts, edges = np.histogram(yaws, bins=bins, range=(-np.pi, np.pi))
    width = 2 * np.pi / bins
    theta = edges[:-1] + width / 2
    ax.bar(theta, counts, width=width * 0.9, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_title(f"Yaw Distribution (n={n_snippets:,})", fontsize=12)
    ax.set_xlabel("Yaw (rad)")
    ax.set_ylabel("Count")
    ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="forward")
    ax.axvline(np.pi, color="gray", linestyle=":", alpha=0.3, label="backward")
    ax.axvline(np.pi/2, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(-np.pi/2, color="gray", linestyle=":", alpha=0.3)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # abs yaw CDF
    ax = axes[1]
    abs_yaw = np.sort(np.abs(yaws))
    cdf = np.arange(1, len(abs_yaw) + 1) / len(abs_yaw)
    ax.plot(np.degrees(abs_yaw), cdf, color="steelblue", linewidth=2)
    for pct in [50, 80, 90, 95, 99]:
        v = np.percentile(np.abs(yaws), pct)
        ax.axvline(np.degrees(v), linestyle="--", alpha=0.5, color="red")
        ax.text(np.degrees(v), 0.02, f"P{pct}={np.degrees(v):.0f}°",
                rotation=90, fontsize=7, va="bottom")
    ax.set_xlabel("|Yaw| (degrees)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of Absolute Yaw at Goal")
    ax.set_xlim(0, min(180, np.degrees(abs_yaw[-1]) * 1.05))
    ax.grid(alpha=0.3)

    fig.suptitle("Goal Frame Root Yaw (ego-centric, relative to snippet start)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "goal_facing_yaw.png", dpi=150)
    plt.close(fig)
    print(f"[Yaw] Saved: {OUTPUT_DIR / 'goal_facing_yaw.png'}")

    # ── Figure 3: per-component CDF overlay ──
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Roll": "red", "Pitch": "green", "Yaw": "blue"}
    for name, arr, c in [("Roll", rolls, "red"), ("Pitch", pitches, "green"), ("Yaw", yaws, "blue")]:
        s = np.sort(np.abs(arr))
        cdf = np.arange(1, len(s) + 1) / len(s)
        ax.plot(np.degrees(s), cdf, color=c, linewidth=2, label=f"|{name}|")

    ax.set_xlabel("Absolute angle (degrees)")
    ax.set_ylabel("CDF")
    ax.set_title("CDF of |Roll| vs |Pitch| vs |Yaw| at Goal Frame (ego-centric)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 180)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "goal_facing_cdf_overlay.png", dpi=150)
    plt.close(fig)
    print(f"[CDF overlay] Saved: {OUTPUT_DIR / 'goal_facing_cdf_overlay.png'}")

    # ── Figure 4: 3D sphere plot of forward direction vectors ──
    # Convert each (yaw, pitch, roll) → unit forward vector in ego space.
    # Ego frame convention: forward = +Y.  Rotation order: ZYX (yaw→pitch→roll).
    ego_fwd = np.zeros((len(yaws), 3), dtype=np.float32)
    for i in range(len(yaws)):
        r_ego = R.from_euler('ZYX', [yaws[i], pitches[i], rolls[i]])
        ego_fwd[i] = r_ego.apply(np.array([0.0, 1.0, 0.0]))  # forward = +Y

    # Subsample for plotting
    n_3d = min(len(ego_fwd), 15000)
    idx_3d = np.random.choice(len(ego_fwd), n_3d, replace=False)
    pts = ego_fwd[idx_3d]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    from matplotlib.colors import Normalize

    # Color by yaw (hue)
    yaw_sampled = yaws[idx_3d]
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    colors = plt.get_cmap('hsv')(norm(yaw_sampled))

    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
               s=2, c=colors, alpha=0.5, edgecolors='none', rasterized=True)

    # Draw unit sphere wireframe
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(sx, sz, sy, color='gray', alpha=0.08, linewidth=0.3)

    # Equator (yaw-only = perfect circle in XZ plane of direction sphere)
    eq_theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.sin(eq_theta), np.zeros_like(eq_theta), np.cos(eq_theta),
            'r--', linewidth=1.5, alpha=0.6, label='equator (yaw-only, pitch=roll=0)')

    # Forward (+Y) axis marker
    ax.quiver(0, 0, 0, 0, 0, 1.2, color='green', linewidth=3,
              arrow_length_ratio=0.1, label='ego forward (+Y)')

    ax.set_xlabel('X (lateral)', fontsize=10)
    ax.set_ylabel('Z (up)', fontsize=10)
    ax.set_zlabel('Y (forward)', fontsize=10)
    ax.set_title(f'Goal Frame Forward Direction on Unit Sphere\n'
                 f'(ego-centric, n={n_3d:,} sampled / {n_snippets:,} total)',
                 fontsize=13)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect([1,1,1])
    ax.legend(fontsize=9, loc='upper left')

    # Colorbar for yaw
    sm = plt.cm.ScalarMappable(cmap='hsv', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Yaw (rad)', fontsize=9)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "goal_facing_sphere_3d.png", dpi=150)
    plt.close(fig)
    print(f"[3D Sphere] Saved: {OUTPUT_DIR / 'goal_facing_sphere_3d.png'}")

    # ── Verdict ──
    print("\n" + "=" * 60)
    p95_roll  = np.degrees(np.percentile(np.abs(rolls), 95))
    p95_pitch = np.degrees(np.percentile(np.abs(pitches), 95))
    p95_yaw   = np.degrees(np.percentile(np.abs(yaws), 95))
    print(f"P95 |Roll| = {p95_roll:.1f}°")
    print(f"P95 |Pitch| = {p95_pitch:.1f}°")
    print(f"P95 |Yaw| = {p95_yaw:.1f}°")

    if p95_roll < 15 and p95_pitch < 15:
        print("\n→ VERDICT: Roll and pitch are negligible at goal frames.")
        print("  Yaw-only (2D cosθ, sinθ) is SUFFICIENT for goal facing encoding.")
    elif p95_roll < 45 and p95_pitch < 45:
        print("\n→ VERDICT: Roll/pitch have moderate variance (not tightly zero).")
        print("  Consider including them, or run with offset>0 to verify.")
    else:
        print("\n→ VERDICT: Roll/pitch show significant variation.")
        print("  Consider full orientation encoding (e.g., 6D rotation).")
    print("=" * 60)


if __name__ == "__main__":
    main()

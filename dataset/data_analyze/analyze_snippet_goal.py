#!/usr/bin/env python3
"""
Snippet-level goal-offset analysis for BONES-SEED → TextOp VAE+LDM training.

=== SNIPPET SAMPLING STRATEGY ===
Each CSV file is downsampled 120→50 Hz. A TextOp-style window (segment_len=35
frames) is slid across the motion with step = SLIDE_STEP = FUTURE_LEN = 8
frames — the same sliding step TextOp uses during autoregressive inference.

For a motion of T frames (≥ SEGMENT_LEN + 1), the number of snippets is:
    n_snippets = floor((T - SEGMENT_LEN) / SLIDE_STEP)
i.e. snippets overlap by SEGMENT_LEN - SLIDE_STEP = 27 frames.

Each snippet is indexed by its seg_start. Goal targets are relative to
seg_start's TextOp egocentric frame (forward = +X, left = +Y, Z-up), using only
the root bone's horizontal (X, Y) coordinates.

NOTE — reference frame simplification: this analysis uses seg_start (first
frame of the snippet window) as the egocentric origin. During actual training
and inference, the reference frame is the *last history frame* (frame 1 for
P0, frame 9 for P1, …). The ≤1-frame difference is negligible for statistical
analysis (≈0.03 m at walking speed). The goal-frame *lookup* formula is
identical in both: seg_start + segment_len - 1 + offset.

=== VOXEL / LOCAL PERCEPTION RANGE ===
Uses the same egocentric bounding box as analyze_bonesseed.py (lines 42-44):
    BBOX_XMIN, BBOX_YMIN = -0.5, -1.0
    BBOX_XMAX, BBOX_YMAX =  1.5,  1.0
This is the physical extent of the 25³ occupancy voxel grid (8 cm resolution)
centered on the character, giving forward [-0.5, 1.5] m and left/right
[-1.0, 1.0] m coverage.

=== WHAT THIS SCRIPT DOES ===
  1. Walk sampled BONES-SEED CSVs, downsample to 50 Hz.
  2. Slide snippet windows; for each, record egocentric root position at
     multiple future offsets.
  3. Plot CDF, 2D scatter, in-range fraction, percentile bands.
  4. Recommend a goal_offset range for training.
"""

import argparse
import csv
import math
import os
import sys
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

# TextOp window parameters (matching pretrained config)
HISTORY_LEN = 2
FUTURE_LEN = 8
NUM_PRIMITIVE = 4
SEGMENT_LEN = HISTORY_LEN + FUTURE_LEN * NUM_PRIMITIVE + 1  # = 35

# DOF cropping: 29 → 23 (drop wrists)
DOF_29_TO_23_MASK = [
    True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,
    True,  True,  True,
    True,  True,  True,  True,
    False, False, False,
    True,  True,  True,  True,
    False, False, False,
]

# ── Egocentric local-perception bounding box ──
# Same as analyze_bonesseed.py lines 42-44.
# World: Z-up. Egocentric: forward = +X, left = +Y.
BBOX_XMIN, BBOX_YMIN = -0.5, -1.0
BBOX_XMAX, BBOX_YMAX =  1.5,  1.0

# Goal offset levels to analyse (frames beyond segment end).
# segment is 35 frames → offsets measure forward from frame index 34.
GOAL_OFFSETS = [0, 4, 8, 16, 24, 32, 48, 64, 80, 96, 128, 192]

# Output
OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
MAX_FILES = 5000  # cap for speed; adjust as needed
SLIDE_STEP = FUTURE_LEN  # = 8 — same as TextOp autoregression stride

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DOF_29_TO_23_MASK_ARR = np.array(DOF_29_TO_23_MASK)


def collect_csv_files(base_dir: str) -> list:
    files = []
    for root, _dirs, fnames in os.walk(base_dir):
        for fn in fnames:
            if fn.endswith(".csv") and not fn.endswith("_M.csv"):
                files.append(os.path.join(root, fn))
    return files


def load_and_downsample_csv(filepath: str):
    """Load CSV, downsample 120→50 Hz, return (trans, rot_quat, dof_23_rad)."""
    with open(filepath) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        data = list(reader)
    if len(data) < 3:
        return None
    data = np.array(data, dtype=np.float32)

    trans_120 = data[:, 1:4] * CM_TO_M
    rot_euler_120 = data[:, 4:7]
    dof_29_120 = data[:, 7:36]

    T_orig = trans_120.shape[0]
    t_orig = np.linspace(0, (T_orig - 1) / ORIGINAL_FPS, T_orig)
    T_target = max(int(T_orig * TARGET_FPS / ORIGINAL_FPS), 2)
    t_target = np.linspace(0, (T_orig - 1) / ORIGINAL_FPS, T_target)

    trans_50 = interp1d(t_orig, trans_120, axis=0, kind="linear")(t_target)
    rot_euler_rad = np.deg2rad(rot_euler_120)
    slerp = Slerp(t_orig, R.from_euler("XYZ", rot_euler_rad))
    rot_quat_50 = slerp(t_target).as_quat()  # xyzw
    dof_50 = interp1d(t_orig, dof_29_120, axis=0, kind="linear")(t_target)
    dof_23_rad = np.deg2rad(dof_50)[:, DOF_29_TO_23_MASK_ARR]

    return trans_50.astype(np.float32), rot_quat_50.astype(np.float32), dof_23_rad.astype(np.float32)


def get_root_forward(rot_quat_xyzw: np.ndarray) -> np.ndarray:
    """Root forward direction: quaternion(xyzw) → unit vector in world XY."""
    rot = R.from_quat(rot_quat_xyzw)
    fwd_world = rot.apply(np.array([1.0, 0.0, 0.0]))  # default forward = +X
    fwd_xy = fwd_world[:, :2]
    norm = np.linalg.norm(fwd_xy, axis=-1, keepdims=True)
    norm = np.clip(norm, 1e-10, None)
    return fwd_xy / norm


def world_to_ego(world_xy: np.ndarray, origin_xy: np.ndarray, fwd_xy: np.ndarray) -> np.ndarray:
    """Convert world XY displacement to TextOp ego coordinates (+X forward, +Y left)."""
    d_xy = world_xy - origin_xy
    left_xy = np.stack([-fwd_xy[..., 1], fwd_xy[..., 0]], axis=-1)
    ego_x = np.sum(d_xy * fwd_xy, axis=-1)
    ego_y = np.sum(d_xy * left_xy, axis=-1)
    return np.stack([ego_x, ego_y], axis=-1)


def process_one_motion(trans: np.ndarray, rot_quat: np.ndarray):
    """
    Slide TextOp windows across one motion. For each window, record
    goal positions at multiple future offsets in egocentric coordinates.

    Returns: list of dicts, each dict = {offset: ego_xy, ...} + window metadata.
    """
    T = trans.shape[0]
    if T < SEGMENT_LEN + 1:
        return []

    fwd = get_root_forward(rot_quat)  # [T, 2]
    results = []

    for seg_start in range(0, T - SEGMENT_LEN, SLIDE_STEP):
        origin_xy = trans[seg_start, :2]
        fwd_start = fwd[seg_start]

        goal_dict = {}
        for off in GOAL_OFFSETS:
            tgt_idx = seg_start + SEGMENT_LEN - 1 + off
            if tgt_idx >= T:
                break
            tgt_xy = trans[tgt_idx, :2]
            ego = world_to_ego(tgt_xy, origin_xy, fwd_start)
            goal_dict[off] = ego

        if goal_dict:
            results.append({
                "seg_start": seg_start,
                "origin_z": float(trans[seg_start, 2]),
                "goal": goal_dict,
            })

    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(results: list):
    """Aggregate snippet results into per-offset distance arrays."""
    dist_by_offset = defaultdict(list)
    ego_by_offset = defaultdict(list)

    for r in results:
        for off, ego in r["goal"].items():
            dist = float(np.linalg.norm(ego))
            dist_by_offset[off].append(dist)
            ego_by_offset[off].append(ego)

    return {off: np.array(vals) for off, vals in dist_by_offset.items()}, \
           {off: np.array(vals) for off, vals in ego_by_offset.items()}


def within_voxel_range(ego_xy: np.ndarray) -> np.ndarray:
    """Check if egocentric points (root horizontal only) fall inside the
    local-perception bounding box defined by BBOX_XMIN..BBOX_YMAX."""
    x, y = ego_xy[:, 0], ego_xy[:, 1]
    in_x = (x >= BBOX_XMIN) & (x <= BBOX_XMAX)
    in_y = (y >= BBOX_YMIN) & (y <= BBOX_YMAX)
    return in_x & in_y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_distance_cdf(dist_by_offset: dict):
    """CDF of egocentric goal distance for each offset."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(GOAL_OFFSETS)))

    for i, off in enumerate(GOAL_OFFSETS):
        if off not in dist_by_offset or len(dist_by_offset[off]) == 0:
            continue
        dists = dist_by_offset[off]
        dists_sorted = np.sort(dists)
        cdf = np.arange(1, len(dists_sorted) + 1) / len(dists_sorted)
        label = f"offset +{off} (n={len(dists):,})"
        ax.plot(dists_sorted, cdf, color=colors[i], linewidth=1.5, label=label)

    # BBOX forward extent line (Y range front boundary)
    ax.axvline(BBOX_YMAX, color="red", linestyle="--", linewidth=2,
               label=f"BBOX forward bound ({BBOX_YMAX:.1f}m)")
    # BBOX diagonal
    bbox_diag = np.sqrt(BBOX_XMAX**2 + BBOX_YMAX**2)
    ax.axvline(bbox_diag, color="orange", linestyle=":", linewidth=1.5,
               label=f"BBOX corner ({bbox_diag:.1f}m)")

    ax.set_xlabel("Ego-centric goal distance (m)  [root XY only]", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("CDF of Goal Distance at Different Future Offsets (50 Hz)", fontsize=14)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_xlim(0, min(10, ax.get_xlim()[1]))
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "snippet_goal_distance_cdf.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[CDF] Saved: {out}")


def plot_in_range_fraction(ego_by_offset: dict):
    """Fraction of goals within the BBOX, as a function of offset.
    Uses the full 2D spatial check (not a distance approximation)."""
    offsets = []
    fracs = []
    nn = []

    for off in GOAL_OFFSETS:
        if off not in ego_by_offset or len(ego_by_offset[off]) == 0:
            continue
        in_range = within_voxel_range(ego_by_offset[off]).mean()
        offsets.append(off)
        fracs.append(in_range * 100)
        nn.append(len(ego_by_offset[off]))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(offsets)), fracs, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(offsets)))
    ax.set_xticklabels([f"+{o}" for o in offsets])
    ax.set_ylabel("% in BBOX  X[{xmin:.0f},{xmax:.0f}] Y[{ymin:.0f},{ymax:.0f}]"
                   .format(xmin=BBOX_XMIN, xmax=BBOX_XMAX, ymin=BBOX_YMIN, ymax=BBOX_YMAX),
                   fontsize=11)
    ax.set_xlabel("Goal offset (frames beyond segment end)", fontsize=12)
    ax.set_title("Fraction of Goals Within Local-Perception Bounding Box", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    for i, (f, n) in enumerate(zip(fracs, nn)):
        ax.text(i, f + 1, f"{f:.0f}%\nn={n:,}", ha="center", fontsize=7)

    # Mark 80% threshold
    ax.axhline(80, color="red", linestyle="--", alpha=0.7, label="80% threshold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    out = OUTPUT_DIR / "snippet_goal_in_range.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[In-Range] Saved: {out}")


def plot_goal_scatter(ego_by_offset: dict):
    """2D scatter of goal positions at key offsets with BBOX overlay.
    Layout: max 3 subplots per row.

    The percentage shown in each title is computed on the FULL dataset for
    that offset, NOT on the rendered subset — so it is always accurate even
    when the scatter is subsampled for rendering speed.
    """
    key_offsets = [0, 8, 32, 64, 128]
    key_offsets = [o for o in key_offsets if o in ego_by_offset]
    if not key_offsets:
        return

    n_total = len(key_offsets)
    n_cols = min(n_total, 3)
    n_rows = math.ceil(n_total / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    # BBOX corners for rectangle drawing
    bx = [BBOX_XMIN, BBOX_XMAX, BBOX_XMAX, BBOX_XMIN, BBOX_XMIN]
    by = [BBOX_YMIN, BBOX_YMIN, BBOX_YMAX, BBOX_YMAX, BBOX_YMIN]

    for i, (ax, off) in enumerate(zip(axes, key_offsets)):
        pts_full = ego_by_offset[off]           # [N, 2] — full dataset
        n_total_pts = len(pts_full)

        # ═══ % in BBOX computed on FULL data before any sampling ═══
        in_range_full = within_voxel_range(pts_full).mean() * 100

        # Subsample for scatter rendering
        sample_n = min(n_total_pts, 5000)
        if n_total_pts > sample_n:
            idx = np.random.choice(n_total_pts, sample_n, replace=False)
            pts_render = pts_full[idx]
        else:
            pts_render = pts_full

        ax.scatter(pts_render[:, 0], pts_render[:, 1], s=1, c="steelblue", alpha=0.3,
                   edgecolors="none", rasterized=True)
        ax.scatter([0], [0], c="red", s=40, marker="x", linewidths=2, zorder=5)

        # BBOX
        ax.plot(bx, by, "r--", linewidth=2,
                label=f"BBOX X[{BBOX_XMIN:.0f},{BBOX_XMAX:.0f}] Y[{BBOX_YMIN:.0f},{BBOX_YMAX:.0f}]")

        # distance circles
        for r, ls, lbl in [(2, ":", "2m"), (5, "-.", "5m")]:
            circle = plt.Circle((0, 0), r, fill=False, linestyle=ls,
                                color="gray", alpha=0.5, label=lbl)
            ax.add_patch(circle)

        ax.set_title(f"offset=+{off} (n={n_total_pts:,})\n"
                     f"{in_range_full:.1f}% in BBOX  "
                     f"(rendered {sample_n:,} pts)",
                     fontsize=10)
        ax.set_xlabel("Lateral X (m)  ← left | right →")
        ax.set_ylabel("Forward Y (m)")
        ax.set_aspect("equal")
        ax.set_xlim(-8, 8)
        ax.set_ylim(-2, 10)
        ax.legend(fontsize=7, loc="upper right")

    # Hide unused axes
    for j in range(len(key_offsets), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Ego-centric Goal Position (root XY only) at Key Offsets", fontsize=14)
    fig.tight_layout()
    out = OUTPUT_DIR / "snippet_goal_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Scatter] Saved: {out}")


def plot_percentile_bands(dist_by_offset: dict):
    """P50/P80/P90/P95 distance bands as function of offset."""
    offsets_arr = []
    p50, p80, p90, p95 = [], [], [], []

    for off in GOAL_OFFSETS:
        if off not in dist_by_offset or len(dist_by_offset[off]) == 0:
            continue
        d = dist_by_offset[off]
        offsets_arr.append(off)
        for lst, p in [(p50, 50), (p80, 80), (p90, 90), (p95, 95)]:
            lst.append(np.percentile(d, p))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(offsets_arr, p50, p95, alpha=0.15, color="steelblue",
                    label="P50–P95 band")
    ax.fill_between(offsets_arr, p50, p80, alpha=0.25, color="steelblue",
                    label="P50–P80 band")
    ax.plot(offsets_arr, p95, "r-", linewidth=1.5, label="P95")
    ax.plot(offsets_arr, p90, "orange", linewidth=1.5, label="P90")
    ax.plot(offsets_arr, p80, "g-", linewidth=1.5, label="P80")
    ax.plot(offsets_arr, p50, "b-", linewidth=2, label="P50 (median)")

    # BBOX forward bound
    ax.axhline(BBOX_YMAX, color="red", linestyle="--", linewidth=2,
               label=f"BBOX forward bound ({BBOX_YMAX:.1f}m)")
    bbox_diag = np.sqrt(BBOX_XMAX**2 + BBOX_YMAX**2)
    ax.axhline(bbox_diag, color="orange", linestyle=":", linewidth=1.5,
               label=f"BBOX corner ({bbox_diag:.1f}m)")

    ax.set_xlabel("Goal offset (frames beyond segment end, @50Hz)", fontsize=12)
    ax.set_ylabel("Ego-centric distance (m)  [root XY only]", fontsize=12)
    ax.set_title("Goal Distance Percentiles vs. Offset", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / "snippet_goal_percentiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[Percentiles] Saved: {out}")

    return offsets_arr, p50, p80


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str,
                        default="/home/lenovo/data/bones-seed/g1/csv")
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument("--slide-step", type=int, default=SLIDE_STEP)
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"ERROR: {base_dir} not found")
        sys.exit(1)

    print(f"SLIDE_STEP = {args.slide_step} frames  "
          f"(TextOp FUTURE_LEN = {FUTURE_LEN})")
    print(f"SEGMENT_LEN = {SEGMENT_LEN} = {HISTORY_LEN}H + {FUTURE_LEN}F×{NUM_PRIMITIVE}P + 1")
    print(f"For a motion of T frames (≥ {SEGMENT_LEN}+1):")
    print(f"  n_snippets = floor((T - {SEGMENT_LEN}) / {args.slide_step})")
    print(f"  overlap = {SEGMENT_LEN - args.slide_step} frames")
    print(f"BBOX: X [{BBOX_XMIN}, {BBOX_XMAX}], Y [{BBOX_YMIN}, {BBOX_YMAX}]")
    print(f"GOAL_OFFSETS: {GOAL_OFFSETS}")
    print()

    # Collect files
    all_files = collect_csv_files(base_dir)
    print(f"Found {len(all_files):,} CSV files (excluding _M mirrored)")
    np.random.seed(RANDOM_SEED)
    if len(all_files) > args.max_files:
        files = np.random.choice(all_files, args.max_files, replace=False).tolist()
        print(f"Sampled {len(files):,} files for analysis")
    else:
        files = all_files

    # Process
    all_results = []
    n_short = 0
    total_snippets = 0
    snippet_counts = []  # per-file snippet count distribution

    for fp in tqdm(files, desc="Processing"):
        motion = load_and_downsample_csv(fp)
        if motion is None:
            n_short += 1
            continue
        trans, rot_quat, _dof = motion
        snippets = process_one_motion(trans, rot_quat)
        all_results.extend(snippets)
        total_snippets += len(snippets)
        if snippets:
            snippet_counts.append(len(snippets))

    print(f"Valid motions:    {len(snippet_counts):,}")
    print(f"Total snippets:   {total_snippets:,}")
    print(f"Too-short files:  {n_short:,}")
    if snippet_counts:
        sc = np.array(snippet_counts)
        print(f"Snippets per motion:  min={sc.min()}, median={np.median(sc):.0f}, "
              f"mean={sc.mean():.1f}, max={sc.max()}")
    print()

    if not all_results:
        print("ERROR: No valid snippets found")
        sys.exit(1)

    # Aggregate
    dist_by_offset, ego_by_offset = aggregate(all_results)

    print("--- Per-offset statistics (root XY only) ---")
    print(f"{'Offset':>8s}  {'Count':>8s}  {'Median(m)':>10s}  "
          f"{'P80(m)':>8s}  {'P95(m)':>8s}  {'% in BBOX':>10s}")
    print("-" * 70)

    recommendation_offset = None
    for off in GOAL_OFFSETS:
        if off not in dist_by_offset or off not in ego_by_offset:
            continue
        d = dist_by_offset[off]
        e = ego_by_offset[off]
        med = np.median(d)
        p80 = np.percentile(d, 80)
        p95 = np.percentile(d, 95)
        in_range = within_voxel_range(e).mean() * 100  # spatial check, not distance
        print(f"  +{off:>4d}   {len(d):>8,}   {med:>10.3f}   {p80:>8.3f}   "
              f"{p95:>8.3f}   {in_range:>9.1f}%")
        if in_range >= 80:
            recommendation_offset = off

    # Plots
    print("\nGenerating plots ...")
    plot_distance_cdf(dist_by_offset)
    plot_in_range_fraction(ego_by_offset)
    plot_goal_scatter(ego_by_offset)
    offsets_arr, p50, p80 = plot_percentile_bands(dist_by_offset)

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"BBOX: X [{BBOX_XMIN}, {BBOX_XMAX}] Y [{BBOX_YMIN}, {BBOX_YMAX}]")
    if recommendation_offset is not None:
        print(f"Largest tested offset with ≥80% goals in BBOX:  +{recommendation_offset}")
        idx = offsets_arr.index(recommendation_offset) if recommendation_offset in offsets_arr else -1
        if idx >= 0:
            print(f"P80 distance at +{recommendation_offset}:  {p80[idx]:.2f} m")
        print(f"\nSuggested goal_offset range:  [0, {recommendation_offset}] frames  "
              f"(@50Hz = [0, {recommendation_offset/TARGET_FPS:.1f}] s)")
        print(f"Suggested default goal_offset for training:  +{min(recommendation_offset, 32)}")
    else:
        print("No offset achieves ≥80% within BBOX.")
        print("Consider: larger BBOX, or use a global planner to split long-range goals.")

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

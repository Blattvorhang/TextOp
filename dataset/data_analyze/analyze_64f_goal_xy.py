#!/usr/bin/env python3
"""
Quick analysis: ego-centric goal XY distribution for a single 64-frame primitive @50Hz.

Uses the same BONES-SEED → 50Hz pipeline as analyze_snippet_goal.py, but computes
the goal position for a 64-frame window (matching planner_dar.yaml: future_len=64)
with goal at the terminal frame.

Output: P50/P80/P90/P95/P99 for |X| and |Y|, and recommended XY plot limits.
"""

import argparse, csv, os, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

# ── Config (matching planner_dar.yaml + train_dar.yaml) ──
ORIGINAL_FPS = 120
TARGET_FPS = 50
CM_TO_M = 1.0 / 100.0

HISTORY_LEN = 16
FUTURE_LEN = 64
NUM_PRIMITIVE = 1
SEGMENT_LEN = HISTORY_LEN + FUTURE_LEN * NUM_PRIMITIVE  # = 80

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
MAX_FILES = 3000
SLIDE_STEP = FUTURE_LEN  # non-overlapping windows


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
    rot_quat_50 = slerp(t_target).as_quat()
    return trans_50.astype(np.float32), rot_quat_50.astype(np.float32)


def get_root_forward(rot_quat_xyzw):
    rot = R.from_quat(rot_quat_xyzw)
    fwd_world = rot.apply(np.array([1.0, 0.0, 0.0]))
    fwd_xy = fwd_world[:, :2]
    norm = np.linalg.norm(fwd_xy, axis=-1, keepdims=True)
    return fwd_xy / np.clip(norm, 1e-10, None)


def world_to_ego(world_xy, origin_xy, fwd_xy):
    """Convert world XY → TextOp ego: forward=+X, left=+Y."""
    d_xy = world_xy - origin_xy
    left_xy = np.stack([-fwd_xy[..., 1], fwd_xy[..., 0]], axis=-1)
    ego_x = np.sum(d_xy * fwd_xy, axis=-1)
    ego_y = np.sum(d_xy * left_xy, axis=-1)
    return np.stack([ego_x, ego_y], axis=-1)


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

    ego_goals = []  # list of (ego_x, ego_y) at goal frame
    n_windows = 0
    n_short = 0

    for fp in tqdm(files, desc="Processing"):
        motion = load_and_downsample_csv(fp)
        if motion is None:
            continue
        trans, rot_quat = motion
        T = trans.shape[0]
        if T < SEGMENT_LEN:
            n_short += 1
            continue

        fwd = get_root_forward(rot_quat)

        for seg_start in range(0, T - SEGMENT_LEN + 1, SLIDE_STEP):
            # Reference frame: last history frame (matching training convention)
            ref_idx = seg_start + HISTORY_LEN - 1
            goal_idx = seg_start + SEGMENT_LEN - 1  # terminal frame of the primitive
            if goal_idx >= T:
                break

            ref_xy = trans[ref_idx, :2]
            ref_fwd = fwd[ref_idx]
            goal_xy = trans[goal_idx, :2]
            ego = world_to_ego(goal_xy, ref_xy, ref_fwd)
            ego_goals.append(ego)
            n_windows += 1

    if not ego_goals:
        print("ERROR: No valid windows found")
        sys.exit(1)

    ego_goals = np.array(ego_goals)  # [N, 2]
    ego_x = ego_goals[:, 0]  # forward
    ego_y = ego_goals[:, 1]  # left

    print(f"\n=== 64-frame single-primitive goal XY distribution ===")
    print(f"Windows: {n_windows:,}  |  Too-short files: {n_short:,}")
    print(f"{'Metric':>12s}  {'X-forward (m)':>16s}  {'Y-left (m)':>16s}")
    print("-" * 52)

    for label, arr in [("X-forward", ego_x), ("Y-left", ego_y)]:
        abs_arr = np.abs(arr)
        print(f"{'mean':>12s}  {arr.mean():>16.4f}  {arr.mean():>16.4f}" if label == "X-forward" else "")
    print()

    # ── Mode (peak of distribution) ──
    from scipy import stats as _stats

    dist = np.sqrt(ego_x**2 + ego_y**2)
    abs_x = np.abs(ego_x)
    abs_y = np.abs(ego_y)

    for name, arr in [("|X|", abs_x), ("|Y|", abs_y), ("distance", dist)]:
        hist, bins = np.histogram(arr, bins=100)
        hist_mode = (bins[hist.argmax()] + bins[hist.argmax() + 1]) / 2
        kde = _stats.gaussian_kde(arr, bw_method=0.05 / max(np.std(arr), 1e-8))
        x_grid = np.linspace(arr.min(), arr.max(), 2000)
        kde_mode = x_grid[np.argmax(kde(x_grid))]
        print(f"{name:>12s}  hist_mode={hist_mode:.4f}  kde_mode={kde_mode:.4f}")

    # 2D histogram mode
    h2d, xe, ye = np.histogram2d(ego_x, ego_y, bins=60)
    max_bin = np.unravel_index(h2d.argmax(), h2d.shape)
    x_mode_2d = (xe[max_bin[0]] + xe[max_bin[0] + 1]) / 2
    y_mode_2d = (ye[max_bin[1]] + ye[max_bin[1] + 1]) / 2
    print(f"\n2D histogram mode: ({x_mode_2d:.4f}, {y_mode_2d:.4f})")

    # ── CDF at key distances ──
    print(f"\nCDF of goal distance:")
    for d in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5]:
        print(f"  ≤{d:.1f}m: {(dist <= d).mean() * 100:5.1f}%")

    percentiles = [50, 80, 90, 95, 99]
    print(f"\n{'Metric':>12s}  {'|X|-forward (m)':>16s}  {'|Y|-lateral (m)':>16s}  {'distance (m)':>16s}")
    print("-" * 65)
    for p in percentiles:
        px = np.percentile(np.abs(ego_x), p)
        py = np.percentile(np.abs(ego_y), p)
        pd = np.percentile(dist, p)
        print(f"{'P'+str(p):>12s}  {px:>16.4f}  {py:>16.4f}  {pd:>16.4f}")

    # Range coverage
    print(f"\n{'abs max':>12s}  {np.abs(ego_x).max():>16.4f}  {np.abs(ego_y).max():>16.4f}")
    print(f"{'min':>12s}  {ego_x.min():>16.4f}  {ego_y.min():>16.4f}")
    print(f"{'max':>12s}  {ego_x.max():>16.4f}  {ego_y.max():>16.4f}")

    # ── Recommendation ──
    # The mode is ~0.02 m — goals concentrate at the origin.  90 % fall
    # within 1.0 m.  Use 1.0 as the default square limit for a clean,
    # readable 2×2 m view; _compute_xy_limit expands it when needed.
    print(f"\n=== Recommended default XY limit ===")
    print(f"_DEFAULT_XY_LIMIT = 1.0  "
          f"(mode ≈ 0.02 m, 90 % of goals ≤ 1.0 m, "
          f"P80 distance = {np.percentile(dist, 80):.2f} m)")


if __name__ == "__main__":
    main()

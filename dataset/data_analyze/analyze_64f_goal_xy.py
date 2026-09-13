#!/usr/bin/env python3
"""
Quick analysis: ego-centric goal XY distribution for a single 64-frame primitive @50Hz.

Uses the same BONES-SEED → 50Hz pipeline as analyze_snippet_goal.py, but computes
the goal position for a 64-frame window (matching planner_dar.yaml: future_len=64)
by plotting every frame in the 64-frame future trajectory.

Output: P50/P80/P90/P95/P99 for |X| and |Y|, and recommended XY plot limits.
"""

import argparse, csv, os, sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Config (matching planner_dar.yaml + train_dar.yaml) ──
ORIGINAL_FPS = 120
TARGET_FPS = 50
CM_TO_M = 1.0 / 100.0

HISTORY_LEN = 16
FUTURE_LEN = 64
NUM_PRIMITIVE = 1
SEGMENT_LEN = HISTORY_LEN + FUTURE_LEN * NUM_PRIMITIVE  # = 80

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42
MAX_FILES = 3000
SLIDE_STEP = FUTURE_LEN  # non-overlapping windows

# RobotMDAR train_dar.yaml / SkeletonPrimitiveDataset contract.
TRAIN_NUM_PRIMITIVE = 3
TRAIN_SEGMENT_LEN = HISTORY_LEN + FUTURE_LEN * TRAIN_NUM_PRIMITIVE + 1


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


def _report_and_plot(ego_points, title, output_name, point_types=None, trajectories=None):
    points = np.asarray(ego_points, dtype=np.float32)
    if points.size == 0:
        print("ERROR: No valid windows found")
        return None
    x, y = points[:, 0], points[:, 1]
    # Display coordinates are rotated CCW 90 degrees so forward points up.
    plot_x, plot_y = -y, x
    radius = np.linalg.norm(points, axis=1)
    if point_types is None:
        point_types = np.full(len(points), "all", dtype=object)
    else:
        point_types = np.asarray(point_types, dtype=object)
    print(f"\n=== {title} ===\nSamples: {len(points):,}")
    print(f"{'Metric':>12s}  {'|X| (m)':>12s}  {'|Y| (m)':>12s}  {'distance (m)':>14s}")
    for q in (50, 80, 90, 95, 99):
        print(f"P{q:>10d}  {np.percentile(abs(x),q):12.4f}  {np.percentile(abs(y),q):12.4f}  {np.percentile(radius,q):14.4f}")
    print(f"{'mean':>12s}  {np.mean(abs(x)):12.4f}  {np.mean(abs(y)):12.4f}  {radius.mean():14.4f}")
    fig, ax = plt.subplots(figsize=(10, 10))
    limit = max(float(np.max(np.abs(points))) * 1.05, 0.5)
    shown = np.random.default_rng(RANDOM_SEED).choice(len(points), min(len(points), 50000), replace=False)
    category_order = ['walking', 'jogging', 'other']
    present = set(point_types[shown].tolist() if trajectories is None else point_types.tolist())
    categories = [category for category in category_order if category in present]
    categories.extend(category for category in present if category not in category_order)
    colors = plt.get_cmap('tab10').colors
    if trajectories is None:
        for idx, category in enumerate(categories):
            mask = point_types[shown] == category
            ax.scatter(plot_x[shown][mask], plot_y[shown][mask], s=2, alpha=.28,
                       color=colors[idx % len(colors)], edgecolors='none',
                       rasterized=True, label=str(category))
    else:
        # Draw each primitive's 64-frame future as one line; use one legend
        # handle per coarse action type while keeping all trajectories visible.
        category_colors = {category: colors[idx % len(colors)]
                           for idx, category in enumerate(categories)}
        legend_seen = set()
        for trajectory, category in zip(trajectories, point_types):
            label = str(category) if category not in legend_seen else '_nolegend_'
            line_alpha = .3 if category == 'other' else .6
            ax.plot(-trajectory[:, 1], trajectory[:, 0], color=category_colors[category],
                    alpha=line_alpha, linewidth=1.2, label=label, rasterized=True)
            legend_seen.add(category)
    for r in (1., 2.):
        if r <= limit * 1.2:
            ax.add_patch(Circle((0, 0), r, fill=False, ls='--', lw=1,
                                alpha=.65, label='_nolegend_'))
    ax.scatter([0], [0], c='red', marker='x', zorder=5)
    ax.axhline(0, color='gray', lw=.5); ax.axvline(0, color='gray', lw=.5)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel='Left (m) +Y (rotated)', ylabel='Forward (m) +X (up)', title=title)
    ax.set_aspect('equal')
    handles, labels = ax.get_legend_handles_labels()
    order = {category: index for index, category in enumerate(category_order)}
    ordered = sorted(zip(handles, labels),
                     key=lambda item: order.get(item[1], len(order)))
    if ordered:
        handles, labels = zip(*ordered)
        legend = ax.legend(handles, labels, fontsize=9, markerscale=6,
                           scatterpoints=1)
        for handle in legend.get_lines():
            handle.set_linewidth(2)
            handle.set_alpha(1.0)

    out = OUTPUT_DIR / output_name
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f'Saved: {out}')
    return out


def analyze_bones_seed(base_dir, max_files=MAX_FILES):
    all_files = collect_csv_files(base_dir)
    print(f'Found {len(all_files):,} CSV files')
    np.random.seed(RANDOM_SEED)
    files = np.random.choice(all_files, max_files, replace=False).tolist() if len(all_files) > max_files else all_files
    points, n_windows, n_short = [], 0, 0
    for fp in tqdm(files, desc='Processing BONES-SEED'):
        motion = load_and_downsample_csv(fp)
        if motion is None: continue
        trans, quat = motion; T = len(trans)
        if T < SEGMENT_LEN: n_short += 1; continue
        fwd = get_root_forward(quat)
        for start in range(0, T - SEGMENT_LEN + 1, SLIDE_STEP):
            ref = start + HISTORY_LEN - 1; goal = start + SEGMENT_LEN - 1
            points.append(world_to_ego(trans[goal,:2], trans[ref,:2], fwd[ref])); n_windows += 1
    print(f'Windows: {n_windows:,} | Too-short files: {n_short:,}')
    return _report_and_plot(points, 'BONES-SEED 64-frame goal XY distribution', 'bones_seed_goal_xy.png')


def collect_hiphi_files(base_dir):
    """Load HiPhi paths from filenames.txt instead of recursively stat'ing motions/."""
    motions_dir = Path(base_dir)
    dataset_dir = motions_dir.parent
    index_path = dataset_dir / 'filenames.txt'
    if index_path.is_file():
        with index_path.open('r', encoding='utf-8') as fh:
            relative_paths = [line.strip() for line in fh
                              if line.strip() and line.strip().endswith('.npz')]
        return [str(motions_dir / relpath) for relpath in relative_paths]
    # Compatibility fallback for datasets without an index file.
    return sorted(str(Path(root) / fn)
                  for root, _, files in os.walk(motions_dir)
                  for fn in files if fn.endswith('.npz'))


def _load_hiphi_chunk(task):
    """Load one sampled motion and return all 3x64 ego-trajectory points."""
    path, seed, category = task
    try:
        with np.load(path, allow_pickle=False) as d:
            pos = np.asarray(d['body_pos_w'][:, 0, :], dtype=np.float32)
            quat = np.asarray(d['body_quat_w'][:, 0, :], dtype=np.float32)
    except (OSError, KeyError, ValueError):
        return None
    max_start = len(pos) - TRAIN_SEGMENT_LEN
    if max_start < 0:
        return None
    rng = np.random.default_rng(seed)
    start = int(rng.integers(max_start + 1))
    fwd = get_root_forward(quat[:, [1, 2, 3, 0]])  # NPZ wxyz -> scipy xyzw
    points = []
    for primitive in range(TRAIN_NUM_PRIMITIVE):
        ref = start + primitive * FUTURE_LEN + HISTORY_LEN
        # Every future frame is a possible 64-frame goal choice.
        for step in range(1, FUTURE_LEN + 1):
            goal = ref + step
            points.append(world_to_ego(pos[goal, :2], pos[ref, :2], fwd[ref]))
    return np.asarray(points, dtype=np.float32).reshape(TRAIN_NUM_PRIMITIVE, FUTURE_LEN, 2), category


def classify_hiphi_category(top_category, subcategory=''):
    """Map HiPhi labels to navigation-relevant spatial movement classes."""
    top = top_category.lower()
    sub = subcategory.lower().replace('_', '-')
    # Explicit fast locomotion labels. Keep this before walking labels.
    jogging = {
        'run', 'trot', 'dart', 'hurry', 'lope', 'scurry', 'bustle',
        'stride', 'prance', 'gallop',
    }
    walking = {
        'walk', 'amble', 'back', 'edge', 'goose-step', 'limp', 'lumber',
        'march', 'mince', 'pad', 'prowl', 'sashay', 'scoot', 'shuffle',
        'sidle', 'skim', 'step', 'stomp', 'strut', 'swagger', 'totter',
        'waddle',
    }
    if sub in jogging:
        return 'jogging'
    if sub in walking:
        return 'walking'
    if top in {'patrolling', 'change_direction'}:
        return 'walking'
    # Dodging, manipulation, posture, gestures, and ambiguous self-motion
    # labels stay in other instead of inflating walking.
    return 'other'


def analyze_hiphi(base_dir='/ALG/hanyi/dataset/hiphi', samples=MAX_FILES,
                  workers=None):
    """Analyze hiphi with RobotMDAR's file/start sampling, in parallel."""
    files = collect_hiphi_files(os.path.join(base_dir, 'motions'))
    if not files:
        raise FileNotFoundError(f'No NPZ files under {base_dir}')
    workers = workers or min(16, (os.cpu_count() or 4))
    rng = np.random.default_rng(RANDOM_SEED)
    # File sampling remains uniform with replacement, as in torch.randint.
    indices = rng.integers(len(files), size=samples)
    seeds = rng.integers(
        0, np.iinfo(np.uint64).max, size=samples, dtype=np.uint64)
    tasks = []
    for i, seed in zip(indices, seeds):
        path = files[int(i)]
        parts = Path(path).relative_to(Path(base_dir) / 'motions').parts
        top_category = parts[0]
        subcategory = parts[1] if len(parts) > 1 else ''
        category = classify_hiphi_category(top_category, subcategory)
        tasks.append((path, int(seed), category))

    points, point_types, trajectories, skipped = [], [], [], 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_load_hiphi_chunk, tasks)
        for result in tqdm(
                results, total=samples,
                desc=f'Sampling hiphi chunks ({workers} workers)'):
            if result is None:
                skipped += 1
                continue
            chunk_points, category = result
            points.extend(chunk_points.reshape(-1, 2))
            trajectories.extend(chunk_points)
            point_types.extend([category] * len(chunk_points))

    print(
        f'Files: {len(files):,} | sampled chunks: {samples:,} | '
        f'skipped: {skipped:,}')
    return _report_and_plot(
        points,
        'HiPhi RobotMDAR training-chunk 64-frame trajectory XY',
        'hiphi_goal_xy.png',
        point_types,
        trajectories,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', choices=('bones-seed', 'hiphi'), default='bones-seed')
    parser.add_argument(
        '--base-dir', default='/home/lenovo/data/bones-seed/g1/csv')
    parser.add_argument(
        '--hiphi-dir', default='/ALG/hanyi/dataset/hiphi')
    parser.add_argument(
        '--max-files', type=int, default=MAX_FILES,
        help='BONES-SEED files, or number of hiphi sampled chunks')
    parser.add_argument(
        '--workers', type=int, default=None,
        help='hiphi loading threads (default: min(16, CPU count))')
    args = parser.parse_args()
    if args.dataset == 'hiphi':
        analyze_hiphi(args.hiphi_dir, args.max_files, args.workers)
    else:
        analyze_bones_seed(args.base_dir, args.max_files)


if __name__ == '__main__':
    main()

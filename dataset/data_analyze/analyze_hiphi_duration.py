#!/usr/bin/env python3
"""Duration distribution for all HiPhi motion_actor.npz files."""

import argparse
import fnmatch
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

DEFAULT_DATASET = "/ALG/hanyi/dataset/hiphi"
DEFAULT_WORKERS = min(16, os.cpu_count() or 4)


def collect_hiphi_files(dataset_dir):
    """Read indexed motion paths; avoid recursively scanning the dataset."""
    dataset_dir = Path(dataset_dir)
    motions_dir = dataset_dir / "motions"
    index_path = dataset_dir / "filenames.txt"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing filename index: {index_path}")
    with index_path.open("r", encoding="utf-8") as fh:
        relative_paths = [line.strip() for line in fh
                          if line.strip() and line.strip().endswith(".npz")]
    return [motions_dir / relpath for relpath in relative_paths]


def load_blacklist(path):
    if path is None or not Path(path).is_file():
        return []
    with Path(path).open('r', encoding='utf-8') as fh:
        return [line.strip() for line in fh
                if line.strip() and not line.lstrip().startswith('#')]


def read_duration(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            frames = int(data["body_pos_w"].shape[0])
        if fps <= 0 or frames <= 0:
            return None
        return frames / fps
    except (OSError, KeyError, ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--blacklist", default=None,
                        help="Blacklist glob file (default: dataset-dir/blacklist.txt)")
    parser.add_argument("--bin-width", type=float, default=2.0,
                        help="Histogram bin width in seconds")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: analysis_output/hiphi_duration_histogram.png)")
    args = parser.parse_args()
    if args.bin_width <= 0:
        parser.error("--bin-width must be positive")

    files = collect_hiphi_files(args.dataset_dir)
    blacklist_path = (Path(args.blacklist) if args.blacklist else
                      Path(args.dataset_dir) / "blacklist.txt")
    patterns = load_blacklist(blacklist_path)
    motions_dir = Path(args.dataset_dir) / "motions"
    kept_files = []
    excluded = 0
    for path in files:
        relative = path.relative_to(motions_dir).as_posix()
        parts = relative.split('/')
        semantic_path = '/'.join(parts[:2])
        if any(fnmatch.fnmatch(semantic_path, pattern) or
               fnmatch.fnmatch(relative, pattern) for pattern in patterns):
            excluded += 1
        else:
            kept_files.append(path)
    files = kept_files
    durations = []
    invalid = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        results = pool.map(read_duration, files)
        for duration in tqdm(results, total=len(files), desc="Reading HiPhi durations"):
            if duration is None:
                invalid += 1
            else:
                durations.append(duration)

    if not durations:
        raise RuntimeError("No valid motion durations found")
    values = np.asarray(durations, dtype=np.float32)
    print(f"Blacklist: {blacklist_path} ({len(patterns)} patterns)")
    print(f"Excluded by blacklist: {excluded:,}")
    print(f"Files after blacklist: {len(files):,}")
    print(f"Valid: {len(values):,} | Invalid: {invalid:,}")
    print(f"Total duration: {values.sum():.2f}s ({values.sum() / 3600:.2f}h)")
    print(f"Mean: {values.mean():.2f}s | Median: {np.median(values):.2f}s")
    for percentile in (5, 25, 75, 90, 95, 99):
        print(f"P{percentile}: {np.percentile(values, percentile):.2f}s")
    print(f"Min: {values.min():.2f}s | Max: {values.max():.2f}s")

    upper = max(args.bin_width, np.ceil(values.max() / args.bin_width) * args.bin_width)
    bins = np.arange(0.0, upper + args.bin_width, args.bin_width)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(values, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Motion duration (seconds)")
    ax.set_ylabel("Number of motion files")
    ax.set_title(f"HiPhi Motion Duration Distribution (bin={args.bin_width:g}s)")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.98, 0.96,
            f"N = {len(values):,}\nMean = {values.mean():.1f}s\n"
            f"Median = {np.median(values):.1f}s\nP90 = {np.percentile(values, 90):.1f}s",
            transform=ax.transAxes, ha="right", va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
    fig.tight_layout()
    output = Path(args.output) if args.output else Path(__file__).resolve().parent / "analysis_output" / "hiphi_duration_histogram.png"
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

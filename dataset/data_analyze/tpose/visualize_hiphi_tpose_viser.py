#!/usr/bin/env python3
"""Compatibility entry for the 20 T-pose clips, using the shared Motion Viser.

For arbitrary NPZs use dataset/data_analyze/visualization/motion_viser.py.
This adapter only builds T-pose-specific views and annotations.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.data_analyze.visualization.motion_viser import (  # noqa: E402
    DEFAULT_MJCF, MotionEntry, MotionView, run_viewer,
)

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "analysis_output/hiphi_tpose/review_20"
CONTEXT = "站立 → T-pose → 站立"
CLIP = "仅导出 NPZ"
SOURCE = "完整原始动作"


def review_entries(directory, dataset_dir):
    directory = directory.expanduser().resolve()
    if directory.is_file():
        files = [directory]
        parent = directory.parent
    else:
        # Prefer the explicit manifest. Never recursively scan the analysis tree.
        parent = directory / "review_20" if (directory / "review_20/manifest.json").is_file() else directory
        manifest = parent / "manifest.json"
        files = ([parent / r["npz"] for r in json.loads(manifest.read_text())["clips"]]
                 if manifest.is_file() else sorted(parent.glob("*.npz")))
    if not files:
        raise ValueError(f"No review NPZs in {directory}")
    manifest = parent / "manifest.json"
    records = ({(parent / r["npz"]).resolve(): r for r in json.loads(manifest.read_text())["clips"]}
               if manifest.is_file() else {})
    entries = []
    for path in files:
        record = records.get(path.resolve())
        if not record:
            entries.append(MotionEntry(path))
            continue
        source = dataset_dir / "motions" / record["source"]
        views = {}
        if source.is_file():
            views[CONTEXT] = MotionView(source, record["before_standing_frames"][0], record["after_standing_frames"][1])
        views[CLIP] = MotionView(path, source_start=record["export_frames"][0])
        if source.is_file():
            views[SOURCE] = MotionView(source)
        annotations = [dict(label="T-pose 保持", frames=record["export_frames"])]
        if source.is_file():
            annotations += [dict(label="抬放臂周期", frames=record["cycle_frames"]),
                            dict(label="前站立证据", frames=record["before_standing_frames"]),
                            dict(label="后站立证据", frames=record["after_standing_frames"])]
        entries.append(MotionEntry(path, source=record["source"], annotations=annotations, views=views,
                                   notes="可切换保持段、前后站立上下文、完整原始动作。"))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions-dir", type=Path, default=DEFAULT_DIR,
                        help="Review folder with manifest.json, or one explicit NPZ")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/hiphi")
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    run_viewer(review_entries(args.motions_dir, args.dataset_dir.resolve()), args.mjcf,
               args.host, args.port, label="HiPhi T-pose 动作检查")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pack motion_lib PKL(s) into TextOp RobotMDAR SkeletonPrimitiveDataset format.

This is the second stage of the BONES-SEED → TextOp pipeline:

  Stage 1: convert_soma_csv_to_motion_lib.py
      BONES-SEED CSV (120Hz, 29-DOF, cm+deg)
      → motion_lib PKL (50Hz, 29-DOF, m+rad, per-name dict)

  Stage 2: pack_motion_lib_to_textop.py  ← this script
      motion_lib PKL
      → TextOp train.pkl / val.pkl (23-DOF, list-of-dicts)

Usage:
    # Single motion_lib PKL
    python pack_motion_lib_to_textop.py \
        --input bones_seed_all.pkl \
        --output ./g1_textop

    # Multiple motion_lib PKLs (merged)
    python pack_motion_lib_to_textop.py \
        --input pkl1.pkl pkl2.pkl pkl3.pkl \
        --output ./g1_textop

    # With custom train/val ratio
    python pack_motion_lib_to_textop.py \
        --input bones_seed_all.pkl \
        --output ./g1_textop --val_ratio 0.05
"""

import argparse
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 29-DOF → 23-DOF crop mask (same as convert_soma_csv_to_motion_lib.py)
# ---------------------------------------------------------------------------
DOF_29_TO_23 = np.array([
    True,  True,  True,  True,  True,  True,    # left leg  0-5
    True,  True,  True,  True,  True,  True,    # right leg 6-11
    True,  True,  True,                          # waist     12-14
    True,  True,  True,  True,                   # left arm  15-18
    False, False, False,                         # left wrist 19-21
    True,  True,  True,  True,                   # right arm 22-25
    False, False, False,                         # right wrist 26-28
])
TARGET_DOF = 23
assert DOF_29_TO_23.sum() == TARGET_DOF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_motion_lib_dicts(input_paths: list[str]) -> dict[str, dict]:
    """Load and merge motion_lib entries from PKL files and/or directories.

    Supported inputs:
      - Single combined PKL:  {name: entry, ...}
      - Directory of individual PKLs (e.g. --individual output or filtered dir):
        walks the tree, loads every .pkl, merges all {name: entry} dicts.
    """
    merged: dict[str, dict] = {}
    for raw_path in input_paths:
        p = Path(raw_path)
        if not p.exists():
            print(f"ERROR: {raw_path} not found")
            sys.exit(1)

        pkl_paths: list[Path] = []
        if p.is_dir():
            pkl_paths = sorted(p.rglob("*.pkl"))
        elif p.suffix == ".pkl":
            pkl_paths = [p]
        else:
            print(f"WARNING: {raw_path} is neither .pkl nor directory, skipping")
            continue

        for pkl_f in pkl_paths:
            try:
                data = joblib.load(pkl_f)
            except Exception:
                print(f"  WARNING: failed to load {pkl_f}, skipping")
                continue
            if not isinstance(data, dict):
                continue
            # Use relative path prefix to avoid name collisions across sessions
            prefix = ""
            if p.is_dir():
                prefix = str(pkl_f.relative_to(p).with_suffix("")).replace("/", "__") + "__"
            for name, entry in data.items():
                unique_name = prefix + name if prefix else name
                if unique_name in merged:
                    unique_name = f"{pkl_f.stem}__{name}"
                merged[unique_name] = entry

    return merged


def motion_lib_entry_to_textop(name: str, entry: dict) -> dict | None:
    """Convert one motion_lib entry to TextOp SkeletonPrimitiveDataset format.

    motion_lib entry (from convert_soma_csv_to_motion_lib.py):
        {
            "root_trans_offset": ndarray [T, 3],    # meters
            "root_rot":          ndarray [T, 4],    # xyzw quaternion
            "dof":               ndarray [T, 29],   # radians, 29-DOF MJCF order
            "contact_mask":      ndarray [T, 2],    # left/right ∈ {0,1}
            "scene": {                              # optional, from --mob
                "occu_global":    ndarray [X,Y,Z],
                "unit":           float,
                "llb":            ndarray [3],
            },
            "fps":               int,
            "pose_aa":           ...,               # ignored
            "smpl_joints":       ...,               # ignored
        }

    TextOp format:
        {
            "length": int,
            "motion": {                                        # ← from motion_lib entry
                "root_trans_offset": ndarray [T, 3],
                "root_rot":          ndarray [T, 4],
                "dof":               ndarray [T, 23],         29-DOF → 23-DOF (wrists locked)
                "contact_mask":      ndarray [T, 2],
                "fps":               int,
                "motion_len":        int,
            },
            "scene": {                                        # inferred pseudo-obstacles
                "occu_global":       ndarray [X, Y, Z],  bool, 1=occupied (vacant space → obstacle)
                "unit":              float,               voxel size (m)
                "llb":               ndarray [3],         float32, world origin
            },
        }
    """
    dof_29 = entry["dof"]
    if dof_29.shape[1] != 29:
        return None  # unexpected DOF count, skip

    dof_23 = dof_29[:, DOF_29_TO_23]                     # (T, 23)
    T = dof_23.shape[0]
    fps_val = entry.get("fps", 50)

    return {
        "length": T,
        "motion": {
            "root_trans_offset": entry["root_trans_offset"].astype(np.float32),
            "root_rot": entry["root_rot"].astype(np.float32),
            "dof": dof_23.astype(np.float32),
            "contact_mask": entry["contact_mask"].astype(np.float32),
            "fps": fps_val,
            "motion_len": T,
        },
        # inferred pseudo-obstacles: vacant space treated as occupied
        # (computed by convert_soma_csv_to_motion_lib.py --mob)
        "scene": entry.get("scene", {}),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Pack motion_lib PKL(s) → TextOp train.pkl / val.pkl"
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more motion_lib PKL files (output of convert_soma_csv_to_motion_lib.py)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for train.pkl, val.pkl, statistics.yaml",
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min_frames", type=int, default=35,
        help="Minimum sequence length in frames (segment_len with default config)",
    )
    args = parser.parse_args()

    # ── Load all motion_lib entries (supports files + directories) ──
    all_entries = load_motion_lib_dicts(args.input)
    print(f"Loaded {len(all_entries)} total sequences from {len(args.input)} input(s)")

    # ── Convert ──
    converted: list[dict] = []
    skipped = 0
    for name, entry in tqdm(all_entries.items(), desc="Converting"):
        item = motion_lib_entry_to_textop(name, entry)
        if item is None:
            skipped += 1
            continue
        if item["length"] < args.min_frames:
            skipped += 1
            continue
        item["_source"] = name
        converted.append(item)

    print(f"Converted {len(converted)} (skipped {skipped} — too short or wrong DOF count)")

    if not converted:
        print("ERROR: No valid sequences!")
        sys.exit(1)

    # ── Shuffle & split ──
    random.seed(args.seed)
    random.shuffle(converted)
    n_val = max(1, int(len(converted) * args.val_ratio))
    train_data = converted[n_val:]
    val_data = converted[:n_val]

    # ── Save ──
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train_path = out / "train.pkl"
    val_path = out / "val.pkl"
    print(f"\nSaving train: {len(train_data)} sequences → {train_path}")
    joblib.dump(train_data, train_path)
    print(f"Saving val:   {len(val_data)} sequences → {val_path}")
    joblib.dump(val_data, val_path)

    # ── Statistics ──
    fps_val = int(train_data[0]["motion"]["fps"])
    train_hours = sum(d["motion"]["motion_len"] for d in train_data) / fps_val / 3600
    val_hours = sum(d["motion"]["motion_len"] for d in val_data) / fps_val / 3600

    stats = {
        "dataset name": "BONES-SEED → TextOp (G1 23-DOF, 50fps)",
        "fps": fps_val,
        "nfeats": 57,
        "train count": len(train_data),
        "val count": len(val_data),
        "train hours": round(train_hours, 1),
        "val hours": round(val_hours, 1),
    }
    stats_path = out / "statistics.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
    print(f"Statistics: {stats_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # ── Done ──
    print(f"\nDone. Ready for TextOp VAE training:")
    print(f"  data.datadir={out.resolve()}")
    print(f"  data.weighted_sample=false")
    print(f"  skeleton.asset.assetRoot=<path/to/description/robots/g1/>")


if __name__ == "__main__":
    main()

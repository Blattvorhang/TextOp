#!/usr/bin/env python3
"""Pack HiPhi mviz NPZ motions directly into RobotMDAR training data."""

import argparse
import fnmatch
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import yaml
from tqdm import tqdm

DEFAULT_DATASET_DIR = "/ALG/hanyi/dataset/hiphi"
DEFAULT_WORKERS = min(16, os.cpu_count() or 4)
DEFAULT_MJCF_DIR = "TextOpRobotMDAR/description/robots/g1"
MVIZ_TO_G1 = np.asarray([
    0, 3, 6, 9, 13, 17,
    1, 4, 7, 10, 14, 18,
    2, 5, 8,
    11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
], dtype=np.intp)
G1_DOF_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


def contact_and_sliding_from_foot_positions(foot_pos, fps, pelvis_z):
    """Apply the BONES-SEED foot contact rule to world foot positions."""
    if foot_pos.ndim != 3 or foot_pos.shape[1:] != (2, 3):
        raise ValueError(f"Expected foot_pos shape (T, 2, 3), got {foot_pos.shape}")
    speed = np.zeros((foot_pos.shape[0], 2), dtype=np.float64)
    if foot_pos.shape[0] > 1:
        speed[1:] = np.linalg.norm(np.diff(foot_pos, axis=0), axis=-1) * fps
    low = foot_pos[:, :, 2] < 0.05
    crawl = np.asarray(pelvis_z, dtype=np.float64)[:, None] < 0.35
    contact = (low & (speed < 0.15)).astype(np.float32)
    sliding = (low & (speed >= 0.15) & ~crawl).astype(np.float32)
    return contact, sliding


def compute_contact_with_fk(root_trans, root_rot, dof, fps, mjcf_dir):
    """Fallback to the shared BONES-SEED MuJoCo FK implementation."""
    from convert_soma_csv_to_motion_lib import compute_contact_and_mob
    xml_path = Path(mjcf_dir) / "g1_29dof_with_collision.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"Missing G1 MJCF for contact FK: {xml_path}")
    return compute_contact_and_mob(
        root_trans, root_rot, dof, fps, str(xml_path), mob=False,
        fk_backend="torch", torch_device="cpu",
    )


def load_patterns(path):
    if path is None:
        return []
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Blacklist does not exist: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh
                if line.strip() and not line.lstrip().startswith("#")]


def indexed_motions(dataset_dir, patterns):
    dataset_dir = Path(dataset_dir)
    index_path = dataset_dir / "filenames.txt"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing HiPhi filename index: {index_path}")

    records = []
    filtered = []
    with index_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            relative = line.strip()
            if not relative or not relative.endswith(".npz"):
                continue
            parts = Path(relative).parts
            if len(parts) < 3:
                raise ValueError(f"Expected category/action/clip path, got {relative!r}")
            semantic_path = f"{parts[0]}/{parts[1]}"
            if any(fnmatch.fnmatch(semantic_path, pattern) or
                   fnmatch.fnmatch(relative, pattern) for pattern in patterns):
                filtered.append(relative)
            else:
                records.append(relative)
    return records, filtered


def canonical_motion_key(relative_path):
    parts = list(Path(relative_path).parts)
    if len(parts) < 3:
        return str(relative_path)
    parts[2] = parts[2].removesuffix("__mirror")
    return "/".join(parts[:3])


def split_manifest(manifest, val_ratio, seed):
    groups = {}
    for record in manifest:
        groups.setdefault(canonical_motion_key(record["_source"]), []).append(record)

    items = list(groups.items())
    random.Random(seed).shuffle(items)
    target_val = int(round(len(manifest) * val_ratio))
    target_val = min(max(target_val, 0), max(len(manifest) - 1, 0))
    train, val = [], []
    train_keys, val_keys = set(), set()
    for key, records in items:
        if len(val) < target_val and len(items) > 1:
            val.extend(records)
            val_keys.add(key)
        else:
            train.extend(records)
            train_keys.add(key)
    if train_keys & val_keys:
        raise RuntimeError("Canonical motion group leaked across train and val")
    return train, val, {
        "unit": "hiphi_canonical_motion_group",
        "groups": len(items),
        "train_groups": len(train_keys),
        "val_groups": len(val_keys),
        "target_val_count": target_val,
        "leakage_groups": 0,
    }


def load_hiphi_npz(path, mjcf_dir=DEFAULT_MJCF_DIR):
    """Load and validate one mviz NPZ, returning a RobotMDAR motion dict."""
    with np.load(path, allow_pickle=True) as data:
        keys = set(data.files)
        required = {
            "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        }
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"{path}: missing required fields: {missing}")
        fps_value = np.asarray(data["fps"])
        if fps_value.size != 1 or fps_value.dtype.kind not in "biuf":
            raise ValueError(f"{path}: fps must be one numeric scalar, got {fps_value.shape} {fps_value.dtype}")
        fps = float(fps_value.item())
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float32)

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{path}: fps must be finite and positive, got {fps}")
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
        raise ValueError(f"{path}: joint_pos must be (T, 29), got {joint_pos.shape}")
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(f"{path}: joint_vel shape {joint_vel.shape} != {joint_pos.shape}")
    frames = joint_pos.shape[0]
    if body_pos.ndim != 3 or body_pos.shape[0] != frames or body_pos.shape[2] != 3:
        raise ValueError(f"{path}: body_pos_w must be (T, N, 3), got {body_pos.shape}")
    if body_quat.shape != (frames, body_pos.shape[1], 4):
        raise ValueError(f"{path}: invalid body_quat_w shape {body_quat.shape}")
    if body_lin_vel.shape != body_pos.shape or body_ang_vel.shape != body_pos.shape:
        raise ValueError(f"{path}: body velocity arrays must match {body_pos.shape}")
    arrays = (joint_pos, joint_vel, body_pos, body_quat,
              body_lin_vel, body_ang_vel)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{path}: motion contains NaN or Inf")

    # mviz order -> G1/URDF order, which is RobotMDAR's MuJoCo source order.
    dof = np.ascontiguousarray(joint_pos[:, MVIZ_TO_G1], dtype=np.float32)
    root_trans = np.ascontiguousarray(body_pos[:, 0, :], dtype=np.float32)
    root_rot = np.ascontiguousarray(body_quat[:, 0, [1, 2, 3, 0]], dtype=np.float32)
    # Complete HiPhi files contain 30 links. Reuse ankle positions to avoid FK.
    if body_pos.shape[1] >= 30:
        foot_pos = body_pos[:, [18, 19], :].astype(np.float64, copy=False)
        contact_mask, sliding_mask = contact_and_sliding_from_foot_positions(
            foot_pos, fps, body_pos[:, 0, 2]
        )
    else:
        fk = compute_contact_with_fk(root_trans, root_rot, dof, fps, mjcf_dir)
        contact_mask = np.asarray(fk["contact_mask"], dtype=np.float32)
        sliding_mask = np.asarray(fk["sliding_mask"], dtype=np.float32)
    return {
        "root_trans_offset": root_trans,
        "root_rot": root_rot,
        "dof": dof,
        "dof_order": "mujoco",
        "dof_names": list(G1_DOF_NAMES),
        "contact_mask": contact_mask,
        "sliding_mask": sliding_mask,
        "fps": fps,
        "motion_len": frames,
    }


def pack_one(task):
    index, relative, dataset_dir, output_dir, min_frames, compression, mjcf_dir = task
    source_path = Path(dataset_dir) / "motions" / relative
    motion = load_hiphi_npz(source_path, mjcf_dir=mjcf_dir)
    frames = int(motion["motion_len"])
    if min_frames and frames < min_frames:
        return None

    parts = Path(relative).parts
    category, text = parts[0], parts[1]
    duration = frames / float(motion["fps"])
    frame_ann = [(0.0, duration, [text], [category])]
    item = {
        "length": frames,
        "motion": motion,
        "scene": {},
        "frame_ann": frame_ann,
        "_recovery_boost": False,
    }
    sample_relpath = Path("samples") / f"{index:08d}.pkl"
    joblib.dump(item, Path(output_dir) / sample_relpath, compress=compression)
    return {
        "length": frames,
        "frame_ann": frame_ann,
        "_source": relative,
        "_data_path": sample_relpath.as_posix(),
        "_fps": float(motion["fps"]),
        "_recovery_boost": False,
    }


def manifest_hours(records):
    return sum(record["length"] / record["_fps"] for record in records) / 3600.0


def main():
    parser = argparse.ArgumentParser(
        description="Pack HiPhi mviz NPZ directly into RobotMDAR data")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blacklist", default=None,
                        help="Optional blacklist glob file; omitted means no filtering")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--min-frames", type=int, default=0)
    parser.add_argument("--mjcf-dir", default=DEFAULT_MJCF_DIR,
                        help="G1 MJCF directory, used for files without 30-link positions")
    parser.add_argument("--sample-compress", type=int, default=3,
                        choices=range(10), metavar="0-9")
    args = parser.parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        parser.error("--val-ratio must be in [0, 1)")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.min_frames < 0:
        parser.error("--min-frames must be non-negative")

    dataset_dir = Path(args.dataset_dir)
    blacklist = Path(args.blacklist) if args.blacklist else None
    patterns = load_patterns(blacklist)
    relative_paths, filtered = indexed_motions(dataset_dir, patterns)
    print(f"Indexed: {len(relative_paths) + len(filtered):,}")
    print(f"Blacklist: {blacklist if blacklist else 'disabled'}")
    print(f"Blacklisted: {len(filtered):,} ({len(patterns)} patterns)")
    print(f"Packing: {len(relative_paths):,}")

    output_dir = Path(args.output)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    tasks = [(index, relative, dataset_dir, output_dir,
              args.min_frames, args.sample_compress, args.mjcf_dir)
             for index, relative in enumerate(relative_paths)]
    manifest = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(pack_one, tasks)
        for record in tqdm(results, total=len(tasks), desc="Packing HiPhi"):
            if record is not None:
                manifest.append(record)

    if not manifest:
        raise RuntimeError("No valid motions were packed")
    fps_values = sorted({record["_fps"] for record in manifest})
    if len(fps_values) != 1:
        raise ValueError(f"Mixed frame rates are unsupported: {fps_values}")

    train, val, split_stats = split_manifest(
        manifest, args.val_ratio, args.seed)
    joblib.dump(train, output_dir / "train.pkl")
    joblib.dump(val, output_dir / "val.pkl")

    # Match run_full_pipeline.sh --neutral so weighted_sample remains usable.
    category_seconds = {}
    for record in train:
        for start_time, end_time, _texts, categories in record["frame_ann"]:
            for category in categories:
                category_seconds[category] = (
                    category_seconds.get(category, 0.0)
                    + float(end_time) - float(start_time)
                )
    action_statistics = {
        category: {"total_len": round(seconds, 3), "weight": 1.0}
        for category, seconds in sorted(category_seconds.items())
    }
    with (output_dir / "action_statistics.json").open(
            "w", encoding="utf-8") as fh:
        json.dump(action_statistics, fh, indent=4)

    stats = {
        "dataset name": "HiPhi -> TextOp (G1 29-DOF)",
        "fps": fps_values[0],
        "dof_dim": 29,
        "feature_version": 6,
        "nfeats": 44,
        "dof_order": "mujoco",
        "dof_names": list(G1_DOF_NAMES),
        "storage": "lazy-sample-manifest-v1",
        "indexed count": len(relative_paths) + len(filtered),
        "blacklist": str(blacklist) if blacklist else "disabled",
        "blacklisted count": len(filtered),
        "packed count": len(manifest),
        "train count": len(train),
        "val count": len(val),
        "train hours": round(manifest_hours(train), 3),
        "val hours": round(manifest_hours(val), 3),
        "split": split_stats,
        "contact": "mviz contact[:, :2] >= 0.5",
        "sliding_mask": "all zeros (not provided by HiPhi)",
        "scene": "empty",
    }
    with (output_dir / "statistics.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(stats, fh, sort_keys=False)
    print(f"Train: {len(train):,} motions, {manifest_hours(train):.2f} h")
    print(f"Val: {len(val):,} motions, {manifest_hours(val):.2f} h")
    print(f"Action categories: {len(action_statistics):,}")
    print(f"Output: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

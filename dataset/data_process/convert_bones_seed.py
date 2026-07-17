"""
Convert BONES-SEED CSV dataset to RobotMDAR-compatible PKL format.

BONES-SEED structure:
    bones-seed/
    ├── g1/csv/<subdir>/*.csv     ← 142k CSV files, 120Hz, 29-DOF
    └── metadata/*.jsonl          ← text annotations (not used for VAE)

Output structure:
    bones-seed/g1_packed/
    ├── train.pkl                 ← list[dict] for SkeletonPrimitiveDataset
    ├── val.pkl
    ├── statistics.yaml           ← {'fps': 50, ...}
    └── meanstd.pkl               ← can be computed by Dataset itself
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import joblib
import torch
import yaml
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ORIGINAL_FPS = 120
TARGET_FPS = 50
TARGET_DOF = 23          # G1 23-DOF (wrists locked)
ORIGINAL_DOF = 29
VAL_RATIO = 0.05         # 5% for validation
RANDOM_SEED = 42

# The 29 DOF column names in BONES-SEED CSV (in order)
DOF_29_NAMES = [
    "left_hip_pitch_joint_dof", "left_hip_roll_joint_dof", "left_hip_yaw_joint_dof",
    "left_knee_joint_dof", "left_ankle_pitch_joint_dof", "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof", "right_hip_roll_joint_dof", "right_hip_yaw_joint_dof",
    "right_knee_joint_dof", "right_ankle_pitch_joint_dof", "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof", "waist_roll_joint_dof", "waist_pitch_joint_dof",  # waist_pitch → G1 torso_link
    "left_shoulder_pitch_joint_dof", "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof", "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof", "left_wrist_pitch_joint_dof", "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof", "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof", "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof", "right_wrist_pitch_joint_dof", "right_wrist_yaw_joint_dof",
]

# Map 29-DOF → 23-DOF: drop wrist DOFs (indices 19-21 and 26-28)
# 23-DOF order matches G1's dof_names in config
DOF_29_TO_23_MASK = [
    True,  True,  True,  True,  True,  True,     # left leg  (0-5)
    True,  True,  True,  True,  True,  True,     # right leg (6-11)
    True,  True,  True,                           # waist (12-14)  ← waist_pitch included
    True,  True,  True,  True,                   # left arm (15-18)
    False, False, False,                          # left wrist (19-21) → DROP
    True,  True,  True,  True,                   # right arm (22-25)
    False, False, False,                          # right wrist (26-28) → DROP
]
assert sum(DOF_29_TO_23_MASK) == TARGET_DOF, f"Expected {TARGET_DOF}, got {sum(DOF_29_TO_23_MASK)}"


# ---------------------------------------------------------------------------
# Frame rate conversion
# ---------------------------------------------------------------------------
def downsample_120_to_50(trans, rot_euler_deg, dof_29):
    """SLERP-based downsampling: 120 Hz → 50 Hz."""
    T_orig = trans.shape[0]
    t_orig = np.linspace(0, (T_orig - 1) / ORIGINAL_FPS, T_orig)
    T_target = max(int(T_orig * TARGET_FPS / ORIGINAL_FPS), 2)
    t_target = np.linspace(0, (T_orig - 1) / ORIGINAL_FPS, T_target)

    # Translation — linear
    trans_new = interp1d(t_orig, trans, axis=0, kind='linear')(t_target)

    # Rotation — SLERP
    rot_euler_rad = np.deg2rad(rot_euler_deg)
    rotations = R.from_euler('XYZ', rot_euler_rad)
    slerp = Slerp(t_orig, rotations)
    rot_new = slerp(t_target).as_quat()  # xyzw

    # DOF — linear
    dof_new = interp1d(t_orig, dof_29, axis=0, kind='linear')(t_target)

    return trans_new, rot_new, dof_new


# ---------------------------------------------------------------------------
# Contact mask via FK (matching process_retarget_data.py approach)
# ---------------------------------------------------------------------------
# Globally initialized skeleton for FK. Use foot_detect logic from
# process_retarget_data.py: check foot velocity + height at ankle bodies.
# G1 skeleton: foot_id = [6 (left_ankle_roll), 12 (right_ankle_roll)]
_SKELETON = None
_FOOT_ID_L = 6
_FOOT_ID_R = 12


def _init_skeleton(skeleton_cfg_path: str, asset_root: str | None = None):
    """Lazy-init RobotSkeleton for FK-based contact detection."""
    global _SKELETON
    if _SKELETON is None:
        from omegaconf import OmegaConf
        from robotmdar.skeleton.robot import RobotSkeleton
        skeleton_cfg = OmegaConf.load(skeleton_cfg_path)
        if asset_root is not None:
            skeleton_cfg.asset.assetRoot = asset_root
        _SKELETON = RobotSkeleton(device="cpu", cfg=skeleton_cfg)


def foot_detect(positions: np.ndarray, root_trans: np.ndarray,
                vel_thresh: float = 0.002, height_rel_thresh: float = 0.15):
    """
    Detect foot contact from ankle positions.

    Key insight: BONES-SEED uses WORLD coordinates. During a walk, the root
    moves ~1 m/s forward, so even a planted foot has ~1 m/s world velocity.
    We must use ROOT-RELATIVE foot velocity to detect stance.

    Two signals:
      1. Root-relative foot velocity < vel_thresh → foot planted
      2. Leg extension (pelvis_z - ankle_z) near median → foot near ground

    Args:
        positions: [T, N_bodies, 3] — FK global_translation_extend[0]
        root_trans: [T, 3] — root translation (world frame, from CSV)
        vel_thresh: squared root-relative velocity threshold
        height_rel_thresh: tolerance around median leg extension (m)

    Returns: contact_mask [T, 2]
    """
    pelvis = root_trans                                          # [T, 3]
    pelvis_z = pelvis[:, 2]                                     # [T]

    def _detect_one(fid: int) -> np.ndarray:
        ankle_world = positions[:, fid, :]                      # [T, 3]

        # Root-relative position: subtract root motion
        ankle_rel = ankle_world - pelvis                        # [T, 3]
        ankle_rel_z = ankle_rel[:, 2]                           # [T]

        # Root-relative velocity (squared)
        rel_vel2 = np.sum(np.diff(ankle_rel, axis=0) ** 2, axis=-1)  # [T-1]

        # Median leg extension = median(|ankle_rel_z|) = standing length
        leg_extension = np.abs(ankle_rel_z)                     # [T]
        standing_leg = np.median(leg_extension)

        # Height above expected ground plane
        height_above_ground = np.abs(leg_extension[1:] - standing_leg)  # [T-1]

        contact = (
            (rel_vel2 < vel_thresh) & (height_above_ground < height_rel_thresh)
        ).astype(np.float32)
        # First frame: assume contact
        contact = np.concatenate([np.array([1.0], dtype=np.float32), contact])
        return np.expand_dims(contact, axis=1)

    left = _detect_one(_FOOT_ID_L)
    right = _detect_one(_FOOT_ID_R)

    # Mutual exclusion for rare double-contact frames
    both = (left[:, 0] > 0.5) & (right[:, 0] > 0.5)
    if both.any():
        ankle_L_rel = positions[:, _FOOT_ID_L, :] - pelvis
        ankle_R_rel = positions[:, _FOOT_ID_R, :] - pelvis
        lh = np.abs(np.abs(ankle_L_rel[:, 2]) - np.median(np.abs(ankle_L_rel[:, 2])))
        rh = np.abs(np.abs(ankle_R_rel[:, 2]) - np.median(np.abs(ankle_R_rel[:, 2])))
        left[both, 0] = (lh[both] < rh[both]).astype(np.float32)
        right[both, 0] = (lh[both] >= rh[both]).astype(np.float32)

    return np.concatenate([left, right], axis=-1)


def compute_contact_mask(dof: np.ndarray, root_trans: np.ndarray, root_rot: np.ndarray) -> np.ndarray:
    """
    Compute contact mask using FK → ankle velocity + height check.

    Args:
        dof: [T, 23] joint angles in radians
        root_trans: [T, 3] root translation
        root_rot: [T, 4] root rotation (xyzw quaternion)
    Returns:
        contact_mask: [T, 2] — (left, right) ∈ {0, 1}
    """
    if _SKELETON is None:
        raise RuntimeError("Skeleton not initialized. Call _init_skeleton() first.")

    import torch
    motion_dict = {
        "dof": torch.from_numpy(dof).float().unsqueeze(0),                           # [1, T, 23]
        "root_trans_offset": torch.from_numpy(root_trans).float().unsqueeze(0),      # [1, T, 3]
        "root_rot": torch.from_numpy(root_rot).float().unsqueeze(0),                 # [1, T, 4]
    }
    fk_return = _SKELETON.forward_kinematics(motion_dict, return_full=False)
    global_positions = fk_return["global_translation_extend"][0].numpy()             # [T, N_bodies, 3]
    return foot_detect(global_positions, root_trans)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------
def collect_csv_files(csv_root: Path) -> list[Path]:
    """Walk csv_root and collect all .csv file paths (excluding *_M.csv mirrored)."""
    files = []
    for subdir in sorted(csv_root.iterdir()):
        if subdir.is_dir():
            for f in subdir.iterdir():
                if f.suffix == '.csv':
                    files.append(f)
    return files


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single BONES-SEED CSV → trans[N,3], rot_euler[N,3], dof[N,29]."""
    with open(path) as fh:
        reader = csv.reader(fh)
        header = next(reader)
        data = list(reader)
    data = np.array(data, dtype=np.float32)
    # Column layout: Frame, root_translateX/Y/Z, root_rotateX/Y/Z, then 29 DOF
    trans = data[:, 1:4]                        # [N, 3]
    rot_euler = data[:, 4:7]                    # [N, 3] — degrees
    dof_29 = data[:, 7:7 + ORIGINAL_DOF]        # [N, 29] — degrees
    return trans, rot_euler, dof_29


def convert_one_file(path: Path) -> dict | None:
    """Convert one BONES-SEED CSV → RobotMDAR motion dict. Returns None on failure."""
    try:
        trans_120, rot_euler_120, dof_29_120 = load_csv(path)
    except Exception as e:
        print(f"  [SKIP] Failed to load {path}: {e}")
        return None

    if trans_120.shape[0] < 3:
        return None  # Too short

    # 1) Downsample 120→50 Hz
    trans_50, rot_quat_50, dof_29_50 = downsample_120_to_50(
        trans_120, rot_euler_120, dof_29_120
    )

    # 2) Deg→Rad for DOF
    dof_29_rad = np.deg2rad(dof_29_50)

    # 3) Crop 29→23 DOF
    dof_23_rad = dof_29_rad[:, DOF_29_TO_23_MASK]

    # 4) Contact mask via FK (matches process_retarget_data.py)
    contact = compute_contact_mask(dof_23_rad, trans_50, rot_quat_50)

    T = trans_50.shape[0]
    if T < 10:
        return None  # Too short after downsampling

    duration = T / TARGET_FPS

    return {
        "motion": {
            "root_trans_offset": trans_50.astype(np.float32),
            "root_rot": rot_quat_50.astype(np.float32),  # xyzw
            "dof": dof_23_rad.astype(np.float32),
            "contact_mask": contact.astype(np.float32),
            "fps": TARGET_FPS,
            "motion_len": T,
        },
        "frame_ann": [(0.0, duration, "", [])],  # dummy — VAE ignores text
        "length": T,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert BONES-SEED CSV → RobotMDAR PKL")
    parser.add_argument("--bones_seed_dir", type=str, required=True,
                        help="Path to BONES-SEED root (contains g1/csv/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for train.pkl / val.pkl / statistics.yaml")
    parser.add_argument("--skeleton_cfg", type=str,
                        default="TextOpRobotMDAR/robotmdar/config/skeleton/g1.yaml",
                        help="Skeleton YAML config (e.g. config/skeleton/g1.yaml)")
    parser.add_argument("--asset_root", type=str,
                        default="TextOpRobotMDAR/description/robots/g1/",
                        help="Root directory containing G1 MJCF files")
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    # Init skeleton for FK-based contact detection
    skel_path = Path(args.skeleton_cfg)
    asset_root = str(Path(args.asset_root).resolve())
    if not skel_path.exists():
        skel_path = Path.cwd() / args.skeleton_cfg
    if not skel_path.exists():
        print(f"ERROR: skeleton config not found: {args.skeleton_cfg}")
        sys.exit(1)
    print(f"Loading skeleton from: {skel_path}")
    print(f"Asset root: {asset_root}")
    _init_skeleton(str(skel_path), asset_root=asset_root)

    csv_root = Path(args.bones_seed_dir) / "g1" / "csv"
    if not csv_root.exists():
        print(f"ERROR: csv dir not found: {csv_root}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    all_files = collect_csv_files(csv_root)
    print(f"Found {len(all_files)} CSV files")
    if not all_files:
        print("ERROR: No CSV files found!")
        sys.exit(1)

    # Convert
    converted = []
    for fp in tqdm(all_files, desc="Converting"):
        item = convert_one_file(fp)
        if item is not None:
            item["_source"] = str(fp.relative_to(csv_root))
            converted.append(item)

    print(f"Converted {len(converted)} / {len(all_files)} sequences")

    # Filter out too-short sequences (need at least segment_len=35 frames)
    min_len = 35
    valid = [c for c in converted if c["length"] >= min_len]
    print(f"After filtering (≥{min_len} frames): {len(valid)}")

    if not valid:
        print("ERROR: No valid sequences after filtering!")
        sys.exit(1)

    # Shuffle & split
    random.seed(args.seed)
    random.shuffle(valid)
    n_val = max(1, int(len(valid) * args.val_ratio))
    train_data = valid[n_val:]
    val_data = valid[:n_val]

    total_duration_train = sum(d["motion"]["motion_len"] for d in train_data) / TARGET_FPS

    # Save
    print(f"Saving train: {len(train_data)} sequences → {output_dir / 'train.pkl'}")
    joblib.dump(train_data, output_dir / "train.pkl")

    print(f"Saving val:   {len(val_data)} sequences → {output_dir / 'val.pkl'}")
    joblib.dump(val_data, output_dir / "val.pkl")

    stats = {
        "dataset name": "BONES-SEED (G1 23DOF, 50fps)",
        "fps": TARGET_FPS,
        "original fps": ORIGINAL_FPS,
        "train count": len(train_data),
        "val count": len(val_data),
        "total duration (train, hours)": round(total_duration_train / 3600, 1),
        "nfeats": 57,
    }
    stats_path = output_dir / "statistics.yaml"
    with open(stats_path, "w") as f:
        yaml.dump(stats, f)
    print(f"Statistics: {stats_path}  ({stats})")

    # Pre-build empty text embeddings (all motions have empty text → zero vectors)
    # This avoids loading CLIP during Dataset init — critical for VAE-only training
    zero_emb = torch.zeros(512, dtype=torch.float32)
    for split in ["train", "val"]:
        embed_path = output_dir / f"{split}_text_embed.pkl"
        torch.save({"": zero_emb}, embed_path)
        print(f"Saved empty text embeddings: {embed_path}")

    print("\nDone! Ready for training:")
    print(f"  DATAFLAGS=\"data.datadir={output_dir.resolve()} data.weighted_sample=false\"")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate XY-origin, absolute-Z G1 goal reference poses from the Bones-SEED dataset.

Picks representative frames for four actions — stand, sit, raise single hand,
squat — and saves each as an NPZ containing keypoints (from FK) and joint angles.

Conventions (matching the original script):
  • root XY at origin
  • root Z at the original (dataset) height in metres
  • root heading (yaw) = 0
  • joint angles in radians, 23-DOF G1 skeleton order (wrists excluded)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from robotmdar.dtype.motion import G1_ROOT_HEIGHT
from robotmdar.skeleton.robot import RobotSkeleton

# ---------------------------------------------------------------------------
# CSV → skeleton DOF mapping
# ---------------------------------------------------------------------------
# Bones-SEED G1 CSV supplies 29 joint-angle columns (degrees).  The 23-DOF
# skeleton drops the six wrist DOFs:
#
#   CSV col idx   Joint name                          Keep?
#   ───────────   ──────────────────────────────────  ─────
#    7 – 12       left  leg (6)                        ✓
#   13 – 18       right leg (6)                        ✓
#   19 – 21       waist (3)                            ✓
#   22 – 25       left  arm (4)                        ✓
#   26 – 28       left  wrist (3)                      ✗
#   29 – 32       right arm (4)                        ✓
#   33 – 35       right wrist (3)                      ✗
# ---------------------------------------------------------------------------
_CSV_DOF_IDXS: tuple[int, ...] = (
    7, 8, 9, 10, 11, 12,      # left  leg:  hip_pitch, roll, yaw, knee, ankle_pitch, roll
    13, 14, 15, 16, 17, 18,   # right leg:  hip_pitch, roll, yaw, knee, ankle_pitch, roll
    19, 20, 21,               # waist:      yaw, roll, pitch (= torso_link)
    22, 23, 24, 25,           # left  arm:  shoulder_pitch, roll, yaw, elbow
    29, 30, 31, 32,           # right arm:  shoulder_pitch, roll, yaw, elbow
)

# ---------------------------------------------------------------------------
# Action definitions
# ---------------------------------------------------------------------------
# Each entry maps an output slug to search terms (matched against
# content_name in the metadata CSV) and a frame-picking strategy.
# ---------------------------------------------------------------------------
ActionSpec = dict[str, str | list[str]]

ACTIONS: dict[str, ActionSpec] = {
    "stand": {
        "keywords": ["h_b_w_stand_270_loop"],
        "strategy": "middle",
    },
    "sit": {
        "keywords": ["neutral_sit_on_chair_loop_R"],
        "strategy": "middle",
    },
    "raise_hand": {
        "keywords": ["raise_your_hand_R"],
        "strategy": "hand_raised",
    },
    "squat": {
        "keywords": ["squat"],  # exact content_name match (no _R suffix)
        "strategy": "deepest",
    },
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_metadata(metadata_csv: Path) -> list[dict]:
    """Return every row of the Bones-SEED metadata CSV as a list of dicts."""
    with metadata_csv.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def _find_csv_path(
    rows: list[dict],
    keywords: list[str],
    dataset_root: Path,
) -> Path:
    """Return the first non-mirrored G1 CSV path matching *keywords*.

    Matching is done against ``content_name``: an exact match for any keyword
    wins immediately; otherwise a substring match is accepted as fallback.
    """
    fallback: dict[str, str] | None = None
    for row in rows:
        cn: str = row.get("content_name", "")
        # skip mirrored takes (suffix ``_M``)
        if cn.endswith("_M"):
            continue
        rel = row.get("move_g1_path", "")
        if not rel:
            continue
        full = dataset_root / rel
        if not full.is_file():
            continue
        # prefer exact match
        if cn in keywords:
            return full
        # store first substring match as fallback
        if fallback is None and any(kw in cn for kw in keywords):
            fallback = {"cn": cn, "rel": rel}
    if fallback is not None:
        return dataset_root / fallback["rel"]
    raise FileNotFoundError(
        f"No non-mirrored CSV found for keywords {keywords}"
    )


def _read_frame(csv_path: Path, frame_idx: int) -> dict[str, float]:
    """Read a single row (by 0-based *frame_idx*) from a Bones-SEED CSV."""
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i == frame_idx:
                return {k: float(v) for k, v in row.items()}
    raise IndexError(f"Frame {frame_idx} out of range in {csv_path}")


def _count_frames(csv_path: Path) -> int:
    """Count data rows in the CSV."""
    with csv_path.open("r", newline="") as fh:
        return sum(1 for _ in fh) - 1  # minus header


# ---------------------------------------------------------------------------
# frame selection strategies
# ---------------------------------------------------------------------------

def _select_hand_raised_frame(csv_path: Path) -> tuple[int, dict[str, float]]:
    """Pick the frame where the right hand is highest.

    "Highest" ≈ right shoulder pitch is at its most negative value
    (shoulder flexed forward/upward in the G1 skeleton convention).
    """
    best_frame = 0
    best_val = float("inf")  # most negative shoulder pitch
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            sp = float(row["right_shoulder_pitch_joint_dof"])
            if sp < best_val:
                best_val = sp
                best_frame = i
    return best_frame, _read_frame(csv_path, best_frame)


def _select_squat_frame(csv_path: Path) -> tuple[int, dict[str, float]]:
    """Pick the deepest squat frame: max average knee angle."""
    best_frame = 0
    best_knee = 0.0
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            lk = float(row["left_knee_joint_dof"])
            rk = float(row["right_knee_joint_dof"])
            avg_knee = (lk + rk) / 2.0
            if avg_knee > best_knee:
                best_knee = avg_knee
                best_frame = i
    return best_frame, _read_frame(csv_path, best_frame)


FRAME_SELECTORS = {
    "middle": lambda p: (lambda n: (n // 2, _read_frame(p, n // 2)))(_count_frames(p)),
    "hand_raised": _select_hand_raised_frame,
    "deepest": _select_squat_frame,
}

# ---------------------------------------------------------------------------
# DOF extraction & FK
# ---------------------------------------------------------------------------

def _extract_dof(frame: dict[str, float]) -> np.ndarray:
    """Extract the 23 skeleton DOFs from a CSV frame row.

    Returns
    -------
    np.ndarray  shape (23,), radians
    """
    raw = np.array([frame[list(frame.keys())[idx]] for idx in _CSV_DOF_IDXS],
                   dtype=np.float64)
    return np.deg2rad(raw).astype(np.float32)


def _run_fk(
    skeleton: RobotSkeleton,
    dof: np.ndarray,
    root_z_m: float,
) -> np.ndarray:
    """Run forward kinematics and return goal keypoints.

    Parameters
    ----------
    skeleton : RobotSkeleton
    dof : np.ndarray  shape (23,), radians
    root_z_m : float   root Z in metres

    Returns
    -------
    np.ndarray  shape (N_keypoints, 3), metres
    """
    root_rot = torch.zeros((1, 4), dtype=torch.float32)
    root_rot[:, 3] = 1.0  # identity quaternion → yaw = 0
    motion = {
        "dof": torch.from_numpy(dof).unsqueeze(0),  # (23,) → (1, 23)
        "root_trans_offset": torch.tensor(
            [[0.0, 0.0, root_z_m]], dtype=torch.float32
        ),
        "root_rot": root_rot,
    }
    result = skeleton.forward_kinematics(motion)
    # global_translation_extend: (batch=1, seq=1, N_bodies_ext, 3)
    kp = result["global_translation_extend"][0, 0, skeleton.goal_keypoint_id]
    return kp.numpy()


# ---------------------------------------------------------------------------
# pose validation
# ---------------------------------------------------------------------------

def _validate_pose(
    keypoints: np.ndarray,
    dof_deg: np.ndarray,
    action: str,
) -> list[str]:
    """Run sanity checks on the keypoints and return a list of warnings.

    The keypoint order is:  pelvis, left-foot, right-foot, left-hand,
    right-hand (the skeleton goal_keypoint_id list).
    """
    issues: list[str] = []

    # Unpack keypoints – verify we have at least 5
    if keypoints.shape[0] < 5:
        issues.append(f"Only {keypoints.shape[0]} keypoints, expected ≥5")
        return issues

    pelvis = keypoints[0]
    l_foot = keypoints[1]
    r_foot = keypoints[2]
    l_hand = keypoints[3]
    r_hand = keypoints[4]

    foot_z = (l_foot[2] + r_foot[2]) / 2.0
    pelvis_z = pelvis[2]
    hand_z_l = l_hand[2]
    hand_z_r = r_hand[2]

    # --- universal checks ---
    # Feet should be near the ground
    if abs(foot_z) > 0.15:
        issues.append(f"Average foot Z = {foot_z:.3f} m (expected near 0)")

    # Pelvis should be above feet
    if pelvis_z < foot_z - 0.05:
        issues.append(
            f"Pelvis Z ({pelvis_z:.3f}) below foot Z ({foot_z:.3f})"
        )

    # --- action-specific checks ---
    if action == "stand":
        # Standing: feet on ground, pelvis at ~0.75 m, hands at sides
        if pelvis_z < 0.60:
            issues.append(f"Stand pelvis Z = {pelvis_z:.3f} m (expected ≥0.60)")
        if abs(hand_z_l - hand_z_r) > 0.30:
            issues.append(
                f"Stand hands asymmetric: L={hand_z_l:.3f} R={hand_z_r:.3f}"
            )

    elif action == "sit":
        # Sitting: pelvis lowered to ~0.45 m, feet still on ground
        if pelvis_z > 0.60:
            issues.append(f"Sit pelvis Z = {pelvis_z:.3f} m (expected <0.60)")
        if pelvis_z < 0.30:
            issues.append(f"Sit pelvis Z = {pelvis_z:.3f} m (too low?)")

    elif action == "raise_hand":
        # One hand should be noticeably higher
        diff = abs(hand_z_l - hand_z_r)
        if diff < 0.20:
            issues.append(
                f"Raise-hand: hand Z diff = {diff:.3f} m (expected ≥0.20)"
            )
        higher = "left" if hand_z_l > hand_z_r else "right"
        hh = max(hand_z_l, hand_z_r)
        if hh < 1.2:
            issues.append(
                f"Raise-hand: highest hand Z = {hh:.3f} m (expected ≥1.2)"
            )

    elif action == "squat":
        # Deep squat: pelvis very low, knees deeply bent
        if pelvis_z > 0.45:
            issues.append(f"Squat pelvis Z = {pelvis_z:.3f} m (expected <0.45)")
        # Check knee angles are large
        knee_avg = (dof_deg[3] + dof_deg[9]) / 2.0  # left knee idx 3, right knee idx 9
        if knee_avg < 90.0:
            issues.append(
                f"Squat knee avg = {knee_avg:.1f}° (expected ≥90°)"
            )

    return issues


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract goal-reference poses from the Bones-SEED dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/lenovo/data/bones-seed"),
        help="Root of the Bones-SEED dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/ref_poses"),
        help="Directory to write NPZ files into",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["stand", "sit", "raise_hand", "squat"],
        help="Which actions to export (default: all four)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the reasonableness checks",
    )
    args = parser.parse_args()

    # --- paths ---
    dataset_root: Path = args.dataset_root
    metadata_csv = dataset_root / "metadata" / "seed_metadata_v004.csv"
    if not metadata_csv.is_file():
        print(f"ERROR: metadata not found at {metadata_csv}", file=sys.stderr)
        sys.exit(1)

    repo_root = Path(__file__).resolve().parents[1]
    skeleton_cfg = OmegaConf.load(repo_root / "robotmdar/config/skeleton/g1.yaml")
    skeleton_cfg.asset.assetRoot = str(repo_root / "description/robots/g1")
    skeleton = RobotSkeleton(device="cpu", cfg=skeleton_cfg)

    # --- load metadata once ---
    print(f"Loading metadata from {metadata_csv} …")
    all_rows = _load_metadata(metadata_csv)
    print(f"  {len(all_rows)} rows loaded")

    # --- process each action ---
    for action_slug in args.actions:
        spec = ACTIONS.get(action_slug)
        if spec is None:
            print(f"  SKIP unknown action '{action_slug}'")
            continue

        keywords: list[str] = spec["keywords"]  # type: ignore[assignment]
        strategy: str = spec["strategy"]  # type: ignore[assignment]

        print(f"\n{'─'*60}")
        print(f"Action: {action_slug}  |  keywords: {keywords}")

        # 1. find CSV
        try:
            csv_path = _find_csv_path(all_rows, keywords, dataset_root)
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            continue
        print(f"  CSV: {csv_path.relative_to(dataset_root)}")

        # 2. pick representative frame
        try:
            selector = FRAME_SELECTORS.get(strategy)
            if selector is None:
                # fallback: use middle of the file
                n = _count_frames(csv_path)
                frame_idx, frame = n // 2, _read_frame(csv_path, n // 2)
            else:
                frame_idx, frame = selector(csv_path)
        except Exception as e:
            print(f"  ERROR selecting frame: {e}")
            continue
        print(f"  Frame: {frame_idx}")

        # 3. extract DOF (radians) and root Z
        dof_rad = _extract_dof(frame)
        root_z_cm = float(frame["root_translateZ"])
        root_z_m = root_z_cm / 100.0
        print(f"  Root Z: {root_z_m:.3f} m  ({root_z_cm:.1f} cm)")

        # 4. FK
        keypoints = _run_fk(skeleton, dof_rad, root_z_m)
        print(f"  Keypoints shape: {keypoints.shape}")

        # 5. validate
        if not args.skip_validation:
            dof_deg = np.rad2deg(dof_rad)
            issues = _validate_pose(keypoints, dof_deg, action_slug)
            if issues:
                print("  ⚠  Validation issues:")
                for iss in issues:
                    print(f"     - {iss}")
            else:
                print("  ✓  Pose looks reasonable")

            # print helpful limb diagnostics
            pelvis = keypoints[0]
            l_foot, r_foot = keypoints[1], keypoints[2]
            l_hand, r_hand = keypoints[3], keypoints[4]
            print(
                f"     Pelvis Z={pelvis[2]:.3f}  "
                f"Foot-Z L={l_foot[2]:.3f} R={r_foot[2]:.3f}  "
                f"Hand-Z L={l_hand[2]:.3f} R={r_hand[2]:.3f}"
            )

        # 6. save
        out_dir: Path = args.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{action_slug}.npz"
        np.savez(
            out_path,
            keypoints=keypoints.astype(np.float32),
            joint_angles=dof_rad.astype(np.float32),
            root_z_m=np.float32(root_z_m),
            source_csv=str(csv_path.relative_to(dataset_root)),
            source_frame=np.int32(frame_idx),
            description=np.asarray(
                f"G1 {action_slug} pose from Bones-SEED, "
                f"root XY at origin, absolute Z={root_z_m:.3f}m, yaw=0. "
                f"Source: {csv_path.name} frame {frame_idx}"
            ),
        )
        print(f"  Wrote {out_path}")

    print(f"\n{'─'*60}")
    print("Done.")


if __name__ == "__main__":
    main()

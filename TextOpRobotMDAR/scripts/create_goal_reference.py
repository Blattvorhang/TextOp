#!/usr/bin/env python3
"""Generate an XY-origin, absolute-Z G1 goal reference using repository FK."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from robotmdar.dtype.motion import G1_ROOT_HEIGHT
from robotmdar.skeleton.robot import RobotSkeleton


STANDING_DOF = torch.tensor([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.9,
    0.2, -0.2, 0.0, 0.9,
], dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("assets/ref_poses/stand.npz"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    skeleton_cfg = OmegaConf.load(
        repo_root / "robotmdar/config/skeleton/g1.yaml")
    skeleton_cfg.asset.assetRoot = str(repo_root / "description/robots/g1")
    skeleton = RobotSkeleton(device="cpu", cfg=skeleton_cfg)

    root_rot = torch.zeros((1, 4), dtype=torch.float32)
    root_rot[:, 3] = 1.0
    motion = {
        "dof": STANDING_DOF.unsqueeze(0),
        "root_trans_offset": torch.tensor(
            [[0.0, 0.0, G1_ROOT_HEIGHT]], dtype=torch.float32),
        "root_rot": root_rot,
    }
    result = skeleton.forward_kinematics(motion)
    keypoints = result["global_translation_extend"][
        0, 0, skeleton.goal_keypoint_id].numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        keypoints=keypoints.astype(np.float32),
        joint_angles=STANDING_DOF.numpy(),
        description=np.asarray(
            "G1 standing pose, root XY at origin, absolute Z, yaw=0"),
    )
    print(f"Wrote {args.output} with keypoints shape {keypoints.shape}")


if __name__ == "__main__":
    main()

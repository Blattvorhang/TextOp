#!/usr/bin/env python3
"""Plot FK end-effector positions for random 64-frame HiPhi chunks.

Coordinates are expressed in the reference root frame: +X forward, +Y left,
and +Z up.  End-effector anchors are resolved through the same active-MJCF
logic used by RobotMDAR split end-effector goals.
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm


ORIGINAL_FPS = 120.0
TARGET_FPS = 50.0
CHUNK_LEN = 64
RANDOM_SEED = 42
END_EFFECTOR_NAMES = ("left_hand", "right_hand", "left_foot", "right_foot")
CENTER_BODY_NAMES = {
    "left_hand": "left_shoulder_pitch_link",
    "right_hand": "right_shoulder_pitch_link",
    "left_foot": "left_hip_pitch_link",
    "right_foot": "right_hip_pitch_link",
}
COLORS = ("#FEA735", "#00C3FF", "#FE7235", "#0077FF")
WAIST_DOF_INDICES = (12, 13, 14)  # MuJoCo order: yaw, roll, pitch
WAIST_BODY_NAME = "waist_yaw_link"
WAIST_Z_OFFSET = 0.044


def collect_npz_files(base_dir):
    """Use HiPhi filenames.txt when available."""
    motions_dir = Path(base_dir) / "motions"
    index = Path(base_dir) / "filenames.txt"
    if index.is_file():
        return [str(motions_dir / line.strip()) for line in index.read_text().splitlines() if line.strip().endswith(".npz")] 
    return sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(motions_dir)
        for name in names
        if name.endswith(".npz")
    )


def load_npz(filepath):
    """Load HiPhi motion_actor.npz (29-DoF IsaacLab order, wxyz root quat)."""
    with np.load(filepath, allow_pickle=False) as data:
        required = {"joint_pos", "body_pos_w", "body_quat_w"}
        if not required.issubset(data.files):
            return None
        trans = np.asarray(data["body_pos_w"][:, 0, :], dtype=np.float32)
        quat = np.asarray(data["body_quat_w"][:, 0, :], dtype=np.float32)[:, [1, 2, 3, 0]]
        dof = np.asarray(data["joint_pos"], dtype=np.float32)
    if dof.ndim != 2 or dof.shape[1] != 29:
        return None
    # HiPhi stores IsaacLab order; RobotSkeleton consumes MuJoCo order.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "TextOpRobotMDAR"))
    from robotmdar.utils.planner_convert import isaaclab_to_mujoco_dof
    dof = isaaclab_to_mujoco_dof(dof)
    if trans.shape != (len(dof), 3) or quat.shape != (len(dof), 4) or len(dof) < CHUNK_LEN:
        return None
    return trans, quat, dof


def init_skeleton():
    """Construct the 29-DoF G1 skeleton used by the planner."""
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "TextOpRobotMDAR"))
    from omegaconf import OmegaConf
    from robotmdar.skeleton.robot import RobotSkeleton

    cfg = OmegaConf.load(str(repo / "TextOpRobotMDAR/robotmdar/config/skeleton/g1.yaml"))
    cfg.asset.assetRoot = str((repo / "TextOpRobotMDAR/description/robots/g1").resolve())
    skeleton = RobotSkeleton(device="cpu", cfg=cfg)
    waist_index = skeleton.body_names.index(WAIST_BODY_NAME)
    waist_offset = skeleton.fk.local_translations[0, waist_index]
    if not np.isclose(float(waist_offset[2]), WAIST_Z_OFFSET):
        raise ValueError(
            f"Expected {WAIST_BODY_NAME} Z offset {WAIST_Z_OFFSET}, "
            f"got {float(waist_offset[2])}"
        )
    skeleton.fk.local_translations[0, waist_index, 2] = 0.0
    return skeleton


def fk_end_effectors(skeleton, trans, quat, dof):
    """Compute end effectors and their shoulder/hip centers in root coordinates."""
    import torch
    from robotmdar.skeleton.end_effector import (
        extract_end_effector_positions,
        resolve_end_effector_anchors,
    )
    # Match the planner analysis contract: lock waist yaw/roll/pitch to zero
    # before FK, while preserving every other joint from the source motion.
    dof = np.array(dof, dtype=np.float32, copy=True)
    dof[:, WAIST_DOF_INDICES] = 0.0
    motion = {
        "root_trans_offset": torch.from_numpy(trans).unsqueeze(0),
        "root_rot": torch.from_numpy(quat).unsqueeze(0),
        "dof": torch.from_numpy(dof).unsqueeze(0),
    }
    fk = skeleton.forward_kinematics(motion, fps=TARGET_FPS)
    world = extract_end_effector_positions(
        fk, skeleton, anchors=resolve_end_effector_anchors(skeleton)
    )[0]
    center_indices = [
        skeleton.body_names.index(CENTER_BODY_NAMES[name])
        for name in END_EFFECTOR_NAMES
    ]
    center_world = fk["global_translation"][0, :, center_indices, :]
    root_pos = torch.from_numpy(trans)
    root_rot = fk["global_rotation_mat"][..., 0, :, :][0]
    # This is build_ego_end_effector_goal's exact world -> reference-root map.
    def to_reference_root(positions):
        return torch.matmul(
            root_rot.transpose(-1, -2).unsqueeze(1),
            (positions - root_pos.unsqueeze(1)).unsqueeze(-1),
        ).squeeze(-1).numpy()

    return to_reference_root(world), to_reference_root(center_world)


def plot(points, output, max_points=100000, seed=RANDOM_SEED):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        raise RuntimeError("No valid 64-frame chunks were found")

    fig = plt.figure(figsize=(16, 8))
    views = ((0, 0, "+X view: looking toward -X"),
             (90, 180, "+Z view rotated 180 degrees"))
    for subplot, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 2, subplot, projection="3d")
        ax.set_proj_type("ortho")
        ax.view_init(elev=elev, azim=azim)
        ax.grid(False)
        if subplot == 1:
            ax.xaxis.set_visible(False)
        else:
            ax.zaxis.set_visible(False)
        for i, name in enumerate(END_EFFECTOR_NAMES):
            # points shape: [chunks, frames, end_effectors, xyz].
            xyz = points[:, :, i, :].reshape(-1, 3)
            if len(xyz) > max_points:
                sample_idx = np.random.default_rng(seed + i).choice(
                    len(xyz), size=max_points, replace=False)
                xyz = xyz[sample_idx]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=3, alpha=0.25,
                       color=COLORS[i], label=name, depthshade=False,
                       rasterized=True)
        # Origin is the reference root; it is intentionally omitted from legend.
        ax.scatter([0], [0], [0], color="black", marker="x", s=35,
                   label="_nolegend_")
        ax.set_xlabel("X forward (m)")
        ax.set_ylabel("Y left (m)")
        ax.set_zlabel("Z up (m)")
        ax.set_title(title)
        ax.set_box_aspect((1, 1, 1))
        legend = ax.legend(loc="upper left", markerscale=4)
        for handle in getattr(legend, "legend_handles",
                              getattr(legend, "legendHandles", ())):
            handle.set_alpha(1.0)

    fig.suptitle("G1 end-effector positions in root-aligned coordinates\n"
                 "(+X forward, +Y left, +Z up; orthographic projection)")
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Saved {output} ({len(points)} chunks, {len(points) * CHUNK_LEN * 4:,} points)")


def save_points(points, centers, output):
    """Save end effectors and matching shoulder/hip centers for reuse."""
    points = np.asarray(points, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if points.ndim != 4 or points.shape[1:] != (CHUNK_LEN, 4, 3):
        raise ValueError(
            f"Expected points shaped [chunks, {CHUNK_LEN}, 4, 3], got {points.shape}"
        )
    if centers.shape != points.shape:
        raise ValueError(
            f"Expected centers shaped {points.shape}, got {centers.shape}"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for index, name in enumerate(END_EFFECTOR_NAMES):
        arrays[name] = points[:, :, index, :].reshape(-1, 3)
        arrays[f"{name}_center"] = centers[:, :, index, :].reshape(-1, 3)
    np.savez_compressed(
        output,
        **arrays,
        waist_z_offset_removed=np.float32(WAIST_Z_OFFSET),
    )
    print(f"Saved reusable end-effector points to {output}")


def _process_file(task):
    path, seed, skeleton = task
    try:
        motion = load_npz(path)
        if motion is None or len(motion[0]) < CHUNK_LEN:
            return None
        trans, quat, dof = motion
        start = int(np.random.default_rng(seed).integers(0, len(trans) - CHUNK_LEN + 1))
        return fk_end_effectors(skeleton, trans[start:start + CHUNK_LEN], quat[start:start + CHUNK_LEN], dof[start:start + CHUNK_LEN])
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f"Skipping {path}: {exc}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, help="HiPhi dataset root (contains motions/)")
    parser.add_argument("--samples", type=int, default=300,
                        help="Maximum number of randomly selected HiPhi chunks")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=None, help="FK worker threads (default: min(16, CPU count))")
    parser.add_argument("--max-points", type=int, default=100000,
                        help="Maximum plotted points per end-effector (default: 100000)")
    parser.add_argument("--output", default="dataset/data_analyze/analysis_output/end_effector_xyz_64f.png")
    parser.add_argument(
        "--points-output",
        default=None,
        help="Reusable NPZ output (default: same path as --output with .npz suffix)",
    )
    args = parser.parse_args()

    files = collect_npz_files(args.base_dir)
    rng = np.random.default_rng(args.seed)
    if len(files) > args.samples:
        files = rng.choice(files, args.samples, replace=False).tolist()
    skeleton = init_skeleton()
    anchors = getattr(skeleton, "body_names", ())
    print(f"Found {len(files):,} files; FK body count={len(anchors)}")
    workers = args.workers or min(16, os.cpu_count() or 1)
    if workers < 1:
        raise ValueError(f"--workers must be positive, got {workers}")
    seeds = rng.integers(0, np.iinfo(np.int64).max, size=len(files), dtype=np.int64)
    tasks = [(path, int(seed), skeleton) for path, seed in zip(files, seeds)]
    chunks = []
    center_chunks = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in tqdm(pool.map(_process_file, tasks), total=len(tasks), desc=f"HiPhi FK ({workers} workers)"):
            if result is not None:
                points, centers = result
                chunks.append(points)
                center_chunks.append(centers)
    print(f"Valid chunks: {len(chunks):,} / {len(files):,}")
    if args.max_points < 1:
        raise ValueError(f"--max-points must be positive, got {args.max_points}")
    points_output = args.points_output or str(Path(args.output).with_suffix(".npz"))
    save_points(chunks, center_chunks, points_output)
    plot(chunks, args.output, max_points=args.max_points, seed=args.seed)


if __name__ == "__main__":
    main()

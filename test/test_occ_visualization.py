"""Visualize GT root trajectory + local occupancy for dataset primitives.

Replicates the eval-loop figure pipeline of
``robotmdar/train/train_dar.py`` (the TensorBoard ``eval/root_xy_trajectory``
report) WITHOUT any model inference, so the occupancy overlay can be
sanity-checked against the ground-truth root path on real data:

  - one batch is sampled from ``data/g1_textop_29dof`` via the same
    hydra-composed ``SkeletonPrimitiveDataset`` the training uses,
  - ``y`` conditions (ego goal + local voxel) are built with the exact
    ``_conditions`` helper from train_dar.py,
  - the GT ego trajectory is integrated with the same code as
    ``DARManager.root_trajectory_ego`` / ``history_trajectory_ego``,
  - the figure is rendered by ``_make_root_xy_figure``, the same code path
    that feeds the SummaryWriter.

Usage:
    python test/test_occ_visualization.py                 # 4 samples, pidx 0
    python test/test_occ_visualization.py --samples 1 --primitive 2
    python test/test_occ_visualization.py --out /tmp/occ_gt.png

The GT path is drawn as the main (blue) curve; the goal ray + markers show
how well the GT endpoint reaches its own goal.
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TextOpRobotMDAR"))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from robotmdar.train.train_dar import (  # noqa: E402
    _conditions,
    _make_root_xy_figure,
)


def _root_trajectory_ego(dataset, future_motion_pred, history_motion):
    """Copy of DARManager.root_trajectory_ego (train/manager.py)."""
    future_motion = dataset.denormalize(future_motion_pred)
    history_last = dataset.denormalize(history_motion[:, -1:])
    path_motion = torch.cat((history_last, future_motion[:, :-1]), dim=1)
    delta_yaw = path_motion[..., 4]
    relative_yaw = torch.cat(
        (torch.zeros_like(delta_yaw[:, :1]), delta_yaw[:, :-1]), dim=1
    ).cumsum(dim=1)
    cos_yaw = torch.cos(relative_yaw)
    sin_yaw = torch.sin(relative_yaw)
    delta_xy = path_motion[..., 7:9]
    delta_xy_start_frame = torch.stack(
        (
            delta_xy[..., 0] * cos_yaw - delta_xy[..., 1] * sin_yaw,
            delta_xy[..., 0] * sin_yaw + delta_xy[..., 1] * cos_yaw,
        ),
        dim=-1,
    )
    origin = torch.zeros_like(delta_xy_start_frame[:, :1])
    return torch.cat((origin, delta_xy_start_frame.cumsum(dim=1)), dim=1)


def _history_trajectory_ego(dataset, history_motion):
    """Copy of DARManager.history_trajectory_ego (train/manager.py)."""
    history = dataset.denormalize(history_motion)
    B, H, _ = history.shape
    if H < 2:
        return torch.zeros(B, H, 2, device=history.device)
    delta_yaw = history[..., 4]
    delta_xy = history[..., 7:9]
    relative_yaw = torch.cat(
        (torch.zeros_like(delta_yaw[:, :1]), delta_yaw[:, :-1]), dim=1
    ).cumsum(dim=1)
    cos_yaw = torch.cos(relative_yaw)
    sin_yaw = torch.sin(relative_yaw)
    delta_in_frame0 = torch.stack(
        (
            delta_xy[..., 0] * cos_yaw - delta_xy[..., 1] * sin_yaw,
            delta_xy[..., 0] * sin_yaw + delta_xy[..., 1] * cos_yaw,
        ),
        dim=-1,
    )
    origin_frame0 = torch.zeros_like(delta_in_frame0[:, :1])
    pos_frame0 = torch.cat(
        (origin_frame0, delta_in_frame0.cumsum(dim=1)), dim=1
    )
    ref_pos = pos_frame0[:, H - 1 : H]
    ref_yaw = relative_yaw[:, H - 1 : H]
    d_pos = pos_frame0[:, :H] - ref_pos
    cos_ref = torch.cos(-ref_yaw)
    sin_ref = torch.sin(-ref_yaw)
    ego_pos = torch.stack(
        (
            d_pos[..., 0] * cos_ref - d_pos[..., 1] * sin_ref,
            d_pos[..., 0] * sin_ref + d_pos[..., 1] * cos_ref,
        ),
        dim=-1,
    )
    return ego_pos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4,
                        help="number of batch samples to plot (max 4)")
    parser.add_argument("--primitive", type=int, default=0,
                        help="primitive index 0..num_primitive-1")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "test" / "occ_gt_sample.png")
    args = parser.parse_args()

    config_dir = str(ROOT / "TextOpRobotMDAR" / "robotmdar" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(
            config_name="train_dar",
            overrides=[
                "device=cpu",
                f"data.datadir={ROOT / 'TextOpRobotMDAR' / 'dataset' / 'BONES-SEED-29dof-FULL-50fps'}",
                # 29-DoF dataset; normalization falls back to the local
                # datadir/meanstd.pkl (dof23 would look for the BONES-SEED
                # 23-dof meanstd that is not present on this machine)
                "data.dof_dim=29",
                "data.batch_size=4",
                "data.weighted_sample=false",
                "data.train.device=cpu",
                "data.val.device=cpu",
                "skeleton.asset.assetRoot="
                f"{ROOT / 'TextOpRobotMDAR' / 'description' / 'robots' / 'g1'}/",
            ],
        )

    val_data = instantiate(cfg.data.val)
    batch = next(iter(val_data))
    primitive = batch[args.primitive]

    history_len = int(cfg.data.history_len)
    future_len = int(cfg.data.future_len)

    motion = primitive["motion"]
    history_motion = motion[:, :history_len, :]
    future_motion_gt = motion[:, -future_len:, :]

    y = _conditions(
        primitive,
        primitive["gt_ref_pos"].to(cfg.device),
        primitive["gt_ref_rot"].to(cfg.device),
        history_motion,
        cfg,
        val_data.fps,
    )

    gt_traj = _root_trajectory_ego(val_data, future_motion_gt, history_motion)
    hist_traj = _history_trajectory_ego(val_data, history_motion)

    # slice to the requested number of samples
    sl = slice(0, args.samples)
    gt_traj = gt_traj[sl]
    hist_traj = hist_traj[sl]
    y["goal"] = y["goal"][sl]
    y["voxel"] = y["voxel"][sl]
    keep = torch.ones(args.samples, dtype=torch.bool)

    # diagnostic numbers printed alongside the figure
    vox = y["voxel"]
    grid_size = int(cfg.denoiser.grid_size)
    for b in range(args.samples):
        endpoint = gt_traj[b, -1]
        endpoint_error = torch.linalg.vector_norm(
            endpoint - y["goal"][b, :2]
        ).item()
        print(
            f"sample {b}: goal=({y['goal'][b, 0]:+.2f}, {y['goal'][b, 1]:+.2f}) "
            f"gt_endpoint=({endpoint[0]:+.2f}, {endpoint[1]:+.2f}) "
            f"endpoint_error={endpoint_error:.3f} m "
            f"occ_occupied={vox[b].mean().item() * 100:.1f}% "
            f"(grid {grid_size}^3)"
        )

    # identical to the eval SummaryWriter report (GT as the main curve)
    figure = _make_root_xy_figure(
        gt_traj,
        y["goal"][:, :2],
        ground_truth_trajectory=None,  # GT is already the main curve
        history_trajectory=hist_traj,
        goal_condition_keep_mask=keep,
        voxel=vox,
        grid_size=grid_size,
        grid_unit=float(cfg.data.occupancy_unit),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

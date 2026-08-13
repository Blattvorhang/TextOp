"""Statistics of the GT root trajectory vs local occupancy at the z=0 slice.

For each sampled primitive, the ego-frame GT root path is mapped onto the
local 25^3 voxel grid (same conventions as train_dar.py's figure / the eval
SummaryWriter) and, at the z=0 slice (k = grid_size//2), the script reports:

  - oob%          fraction of trajectory points outside the local 25^3 grid
                  — the voxel carries no value there (unknown), so they are
                  reported separately instead of being folded into traj_free%
                  (note: in-grid cells that fall outside occu_global ARE
                  occupied in y['voxel'], per query_local_occupancy),
  - traj_free%    fraction of in-bounds trajectory points landing on free
                  cells,
  - neigh_free%   mean 4-neighbour free rate around the free trajectory
                  cells — a proxy for corridor width,
  - comp_cov%     fraction of the free trajectory cells that lie in the
                  dominant 4-connected free component — the A* reachability
                  proxy (100% = path is one connected corridor),
  - slice_free%   fraction of free cells in the whole z=0 slice.

Usage:
    python test/occ_traj_free_stats.py                 # 32 samples
    python test/occ_traj_free_stats.py --samples 64
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TextOpRobotMDAR"))

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from robotmdar.train.train_dar import _conditions  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=32,
                        help="number of batch samples to analyse")
    parser.add_argument("--primitive", type=int, default=0,
                        help="primitive index 0..num_primitive-1")
    args = parser.parse_args()

    config_dir = str(ROOT / "TextOpRobotMDAR" / "robotmdar" / "config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(
            config_name="train_dar",
            overrides=[
                "device=cpu",
                f"data.datadir={ROOT / 'data' / 'g1_textop_29dof'}",
                "data.dof_dim=29",
                f"data.batch_size={args.samples}",
                "data.weighted_sample=false",
                "data.train.device=cpu",
                "data.val.device=cpu",
                "skeleton.asset.assetRoot="
                f"{ROOT / 'TextOpRobotMDAR' / 'description' / 'robots' / 'g1'}/",
            ],
        )

    val_data = instantiate(cfg.data.val)
    primitive = next(iter(val_data))[args.primitive]

    history_len = int(cfg.data.history_len)
    future_len = int(cfg.data.future_len)
    grid_size = int(cfg.denoiser.grid_size)
    unit = float(cfg.data.occupancy_unit)
    half = grid_size // 2
    forward_origin = grid_size // 4

    motion = primitive["motion"]
    history_motion = motion[:, :history_len, :]
    future_motion_gt = motion[:, -future_len:, :]

    y = _conditions(
        primitive,
        primitive["gt_ref_pos"].to(cfg.device),
        primitive["gt_ref_rot"].to(cfg.device),
        history_motion,
        cfg,
    )
    gt_traj = _root_trajectory_ego(val_data, future_motion_gt, history_motion)
    vox = y["voxel"].reshape(args.samples, grid_size, grid_size, grid_size)
    slice_occ = vox[:, :, :, half]  # [B, forward(x), left(y)] at z=0

    # ego trajectory -> grid indices (cell centers, same as _local_grid_offsets)
    ix = (gt_traj[..., 0] / unit).round().long() + forward_origin
    iy = (gt_traj[..., 1] / unit).round().long() + half
    in_bounds = (
        (ix >= 0) & (ix < grid_size) & (iy >= 0) & (iy < grid_size)
    )
    ix = ix.clamp(0, grid_size - 1)
    iy = iy.clamp(0, grid_size - 1)

    # slice padded with occupied (1) so out-of-bounds neighbours count as
    # occupied, matching the query convention
    slice_pad = torch.nn.functional.pad(slice_occ, (1, 1, 1, 1), value=1.0)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    print(f"sample   disp[m]  traj_pts  oob%  traj_free%  neigh_free%  comp_cov%  slice_free%")
    oob_pct, traj_free_pct, neigh_free_pct, comp_cov_pct, slice_free_pct = (
        [], [], [], [], [],
    )
    for b in range(args.samples):
        n_pts = gt_traj[b].shape[0]
        inb = in_bounds[b]
        oob_pct.append((~inb).float().mean().item() * 100.0)
        slice_free_pct.append((1.0 - slice_occ[b].mean()).item() * 100.0)

        if inb.sum().item() == 0:
            traj_free_pct.append(float("nan"))
            neigh_free_pct.append(float("nan"))
            comp_cov_pct.append(float("nan"))
            print(f"{b:>6d}  {gt_traj[b, -1].norm():>7.2f}  "
                  f"{n_pts:>8d}  {oob_pct[-1]:>5.1f}  "
                  f"{'n/a':>10s}  {'n/a':>11s}  {'n/a':>9s}  {slice_free_pct[-1]:>11.1f}")
            continue

        # only in-bounds points are defined in the voxel; OOB are unknown
        occ = slice_occ[b, ix[b, inb], iy[b, inb]].bool()
        free = ~occ
        traj_free_pct.append(free.float().mean().item() * 100.0)

        # unique free trajectory cells
        cells = torch.stack((ix[b, inb], iy[b, inb]), dim=-1)[free]
        if cells.numel() == 0:
            neigh_free_pct.append(float("nan"))
            comp_cov_pct.append(float("nan"))
            print(f"{b:>6d}  {gt_traj[b, -1].norm():>7.2f}  "
                  f"{n_pts:>8d}  {oob_pct[-1]:>5.1f}  {traj_free_pct[-1]:>10.1f}  "
                  f"{'n/a':>11s}  {'n/a':>9s}  {slice_free_pct[-1]:>11.1f}")
            continue
        cells = torch.unique(cells, dim=0)  # [U, 2]
        nn = slice_pad[b, cells[:, 0] + 1 - 1, cells[:, 1] + 1] \
           + slice_pad[b, cells[:, 0] + 1 + 1, cells[:, 1] + 1] \
           + slice_pad[b, cells[:, 0] + 1, cells[:, 1] + 1 - 1] \
           + slice_pad[b, cells[:, 0] + 1, cells[:, 1] + 1 + 1]
        neigh_free_pct.append((1.0 - nn / 4.0).mean().item() * 100.0)

        # dominant 4-connected free component covering the trajectory cells
        free_slice = (slice_occ[b] < 0.5).cpu().numpy()
        labels, _ = ndimage.label(free_slice, structure=struct)
        cell_labels = labels[cells[:, 0].cpu().numpy(), cells[:, 1].cpu().numpy()]
        mode = stats.mode(cell_labels, keepdims=False).mode
        comp_cov_pct.append((cell_labels == mode).mean() * 100.0)

        print(f"{b:>6d}  {gt_traj[b, -1].norm():>7.2f}  "
              f"{n_pts:>8d}  {oob_pct[-1]:>5.1f}  {traj_free_pct[-1]:>10.1f}  "
              f"{neigh_free_pct[-1]:>11.1f}  {comp_cov_pct[-1]:>9.1f}  "
              f"{slice_free_pct[-1]:>11.1f}")

    arr = lambda v: np.array(v, dtype=np.float64)  # noqa: E731
    print("---- aggregates ----")
    print(f"oob%        mean={np.nanmean(arr(oob_pct)):.1f}  max={np.nanmax(arr(oob_pct)):.1f}")
    print(f"traj_free%  mean={np.nanmean(arr(traj_free_pct)):.1f}  min={np.nanmin(arr(traj_free_pct)):.1f}  (in-bounds points only)")
    print(f"neigh_free% mean={np.nanmean(arr(neigh_free_pct)):.1f}  min={np.nanmin(arr(neigh_free_pct)):.1f}")
    print(f"comp_cov%   mean={np.nanmean(arr(comp_cov_pct)):.1f}  min={np.nanmin(arr(comp_cov_pct)):.1f}")
    print(f"slice_free% mean={np.nanmean(arr(slice_free_pct)):.1f}  min={np.nanmin(arr(slice_free_pct)):.1f}")


if __name__ == "__main__":
    main()

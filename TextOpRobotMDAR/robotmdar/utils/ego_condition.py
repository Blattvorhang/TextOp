"""Ego-centric condition building: goal canonicalization and local scene-occupancy sampling."""

from typing import Any, Dict, Sequence

import numpy as np
import torch


def quaternion_yaw(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    """Return Z-axis yaw from xyzw quaternions."""
    x, y, z, w = quaternion_xyzw.unbind(dim=-1)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y.square() + z.square())
    return torch.atan2(sin_yaw, cos_yaw)


def build_ego_goal(world_goal_pos: torch.Tensor,
                   world_goal_yaw: torch.Tensor,
                   reference_pos: torch.Tensor,
                   reference_rot: torch.Tensor) -> torch.Tensor:
    """Express a world-space root goal in TextOp's local X-forward frame."""
    current_yaw = quaternion_yaw(reference_rot)
    cos_yaw = torch.cos(current_yaw)
    sin_yaw = torch.sin(current_yaw)
    delta = world_goal_pos - reference_pos

    ego_x = delta[..., 0] * cos_yaw + delta[..., 1] * sin_yaw
    ego_y = -delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw
    delta_yaw = world_goal_yaw - current_yaw
    return torch.stack(
        (ego_x, ego_y, delta[..., 2], torch.cos(delta_yaw),
         torch.sin(delta_yaw)),
        dim=-1,
    )


def _local_grid_offsets(grid_size: int, grid_unit: float) -> np.ndarray:
    if grid_size <= 0 or grid_size % 2 == 0:
        raise ValueError(f"grid_size must be a positive odd number, got {grid_size}")

    half = grid_size // 2
    forward = (np.arange(grid_size, dtype=np.float32) - grid_size // 4) * grid_unit
    left = (np.arange(grid_size, dtype=np.float32) - half) * grid_unit
    vertical = (np.arange(grid_size, dtype=np.float32) - half) * grid_unit
    return np.stack(
        np.meshgrid(forward, left, vertical, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def query_local_occupancy(
    scenes: Sequence[Dict[str, Any]],
    reference_pos: torch.Tensor,
    reference_rot: torch.Tensor,
    grid_size: int = 25,
    grid_unit: float = 0.08,
) -> torch.Tensor:
    """Sample fixed local grids from per-sample variable global occupancies.

    Local axes are TextOp X-forward, Y-left, Z-up. Samples outside a global
    occupancy are occupied, matching MOB's conservative boundary behavior.
    The variable global grids remain on CPU; only the fixed result is copied to
    the caller's device.
    """
    if len(scenes) != reference_pos.shape[0]:
        raise ValueError(
            f"Expected {reference_pos.shape[0]} scenes, got {len(scenes)}"
        )

    output_device = reference_pos.device
    positions = reference_pos.detach().cpu().float().numpy()
    yaws = quaternion_yaw(reference_rot.detach()).cpu().float().numpy()
    offsets = _local_grid_offsets(grid_size, grid_unit)
    local_forward = offsets[:, 0]
    local_left = offsets[:, 1]
    local_z = offsets[:, 2]
    local_grids = np.ones((len(scenes), len(offsets)), dtype=np.float32)

    for batch_idx, scene in enumerate(scenes):
        required = {"occu_global", "unit", "llb"}
        if not required.issubset(scene):
            missing = sorted(required.difference(scene))
            raise ValueError(f"Scene {batch_idx} is missing fields: {missing}")

        occupancy = np.asarray(scene["occu_global"], dtype=bool)
        if occupancy.ndim != 3:
            raise ValueError(
                f"Scene {batch_idx} occupancy must be 3-D, got {occupancy.shape}"
            )
        unit = float(scene["unit"])
        llb = np.asarray(scene["llb"], dtype=np.float32)

        cos_yaw = np.cos(yaws[batch_idx])
        sin_yaw = np.sin(yaws[batch_idx])
        world_points = np.empty_like(offsets)
        world_points[:, 0] = (
            positions[batch_idx, 0]
            + local_forward * cos_yaw
            - local_left * sin_yaw
        )
        world_points[:, 1] = (
            positions[batch_idx, 1]
            + local_forward * sin_yaw
            + local_left * cos_yaw
        )
        world_points[:, 2] = positions[batch_idx, 2] + local_z

        indices = np.floor((world_points - llb) / unit).astype(np.int64)
        shape = np.asarray(occupancy.shape, dtype=np.int64)
        valid = np.all((indices >= 0) & (indices < shape), axis=-1)
        valid_indices = indices[valid]
        local_grids[batch_idx, valid] = occupancy[
            valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]
        ].astype(np.float32)

    return torch.from_numpy(local_grids).to(output_device)

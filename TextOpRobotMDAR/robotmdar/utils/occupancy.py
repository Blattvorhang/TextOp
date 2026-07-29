"""Local scene-occupancy sampling and 3D voxel morphology utilities."""

from typing import Any, Dict, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 3D voxel morphology
# ---------------------------------------------------------------------------

def erode_voxel_26(occupancy: np.ndarray) -> np.ndarray:
    """Erode a 3D binary voxel grid with a 26-neighbourhood structuring element.

    A voxel survives only when it AND all 26 surrounding voxels (the full
    3 × 3 × 3 cube) are occupied.  Boundary voxels are padded with occupied
    (1) so they survive as long as their in-grid neighbours are all occupied.

    Parameters
    ----------
    occupancy : np.ndarray
        3-D boolean or binary integer array.  ``True`` / non-zero means
        occupied.

    Returns
    -------
    np.ndarray
        Eroded boolean array with the same shape as *occupancy*.
    """
    if occupancy.ndim != 3:
        raise ValueError(
            f"Expected a 3-D occupancy array, got shape {occupancy.shape}"
        )

    # Pad with zeros so boundary voxels are naturally eroded away.
    padded = np.pad(
        occupancy.astype(np.uint8, copy=False),
        pad_width=1,
        mode="constant",
        constant_values=1,
    )

    # Extract every 3×3×3 sliding window.
    windows: np.ndarray = np.lib.stride_tricks.sliding_window_view(
        padded, (3, 3, 3)
    )

    # A voxel survives only when all 27 neighbours (centre + 26 neighbours)
    # are non-zero.
    eroded: np.ndarray = windows.all(axis=(-3, -2, -1))
    return eroded


# ---------------------------------------------------------------------------
# Local grid helpers
# ---------------------------------------------------------------------------

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
    yaws = _quaternion_yaw_np(reference_rot.detach().cpu().float().numpy())
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


def _quaternion_yaw_np(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Return Z-axis yaw from xyzw quaternions (NumPy version)."""
    x, y, z, w = (
        quaternion_xyzw[..., 0],
        quaternion_xyzw[..., 1],
        quaternion_xyzw[..., 2],
        quaternion_xyzw[..., 3],
    )
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(sin_yaw, cos_yaw)

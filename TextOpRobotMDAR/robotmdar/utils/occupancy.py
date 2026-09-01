"""Local scene-occupancy sampling and 3D voxel morphology utilities."""

from functools import lru_cache
from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 3D voxel morphology
# ---------------------------------------------------------------------------

def _prepare_occupancy(occupancy):
    """Normalize a torch.Tensor / np.ndarray occupancy to a bool tensor.

    Returns ``(tensor, was_numpy)``.  Torch computation is unified so GPU
    tensors (e.g. the output of :func:`query_local_occupancy`) never
    round-trip through numpy; numpy input is converted up front and the
    result converted back by the caller.
    """
    if isinstance(occupancy, torch.Tensor):
        return occupancy.to(dtype=torch.bool), False
    if isinstance(occupancy, np.ndarray):
        return (
            torch.from_numpy(np.ascontiguousarray(occupancy)).to(
                dtype=torch.bool),
            True,
        )
    raise TypeError(
        "occupancy must be a torch.Tensor or np.ndarray, got "
        f"{type(occupancy).__name__}"
    )


def _erosion_kernel(mode: int) -> torch.Tensor:
    """3×3×3 counting kernel for the 6/10/26 neighbourhood (1 out/in chan)."""
    kernel = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float32)
    kernel[0, 0, 1, 1, 1] = 1.0
    for offset in (
        (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    ):
        kernel[(0, 0) + tuple(1 + o for o in offset)] = 1.0
    if mode in (10, 26):
        for offset in ((1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)):
            kernel[(0, 0) + tuple(1 + o for o in offset)] = 1.0
    if mode == 26:
        kernel[:] = 1.0
    return kernel


_EROSION_KERNELS = {mode: _erosion_kernel(mode) for mode in (6, 10, 26)}
_EROSION_COUNTS = {6: 7, 10: 11, 26: 27}


def _erode_torch(occupancy: torch.Tensor, mode: int) -> torch.Tensor:
    """Erode 3D or batched 3D bool tensors with occupied boundary padding.

    A voxel survives only when the centre and every required neighbour are
    occupied, counted by a 3×3×3 convolution against the mode's kernel.
    """
    if occupancy.ndim not in (3, 4):
        raise ValueError(
            f"Expected a 3-D or batched 4-D occupancy tensor, got "
            f"shape {tuple(occupancy.shape)}"
        )
    x = occupancy.to(dtype=torch.float32)
    padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode="constant", value=1.0)
    kernel = _EROSION_KERNELS[mode].to(device=x.device)
    if occupancy.ndim == 3:
        counts = F.conv3d(padded[None, None], kernel)[0, 0]
    else:
        counts = F.conv3d(padded[:, None], kernel)[:, 0]
    return counts >= _EROSION_COUNTS[mode]


def erode_voxel_6(occupancy) -> torch.Tensor | np.ndarray:
    """Erode a 3D binary voxel grid with a 6-neighbourhood structuring element.

    A voxel survives only when it AND all 6 face-sharing neighbours (±x, ±y,
    ±z) are occupied — the weakest neighbourhood erosion, since only direct
    neighbours constrain survival.  Boundary voxels are padded with occupied
    (1) so they survive as long as their in-grid neighbours are all occupied.

    Torch-native: accepts a ``torch.Tensor`` (any device) or ``np.ndarray``
    and returns the same type, so GPU occupancy grids never round-trip
    through numpy.

    Parameters
    ----------
    occupancy : torch.Tensor or np.ndarray
        3-D boolean or binary array.  ``True`` / non-zero means occupied.

    Returns
    -------
    torch.Tensor or np.ndarray
        Eroded boolean grid with the same shape as *occupancy*.
    """
    tensor, was_numpy = _prepare_occupancy(occupancy)
    eroded = _erode_torch(tensor, 6)
    return eroded.cpu().numpy() if was_numpy else eroded


def erode_voxel_10(occupancy) -> torch.Tensor | np.ndarray:
    """Erode a 3D binary voxel grid with an anisotropic 10-neighbourhood.

    A voxel survives only when it AND all 10 neighbours — the 6 face-sharing
    neighbours plus the 4 diagonal neighbours in the same horizontal (xy)
    plane — are occupied.  Compared to :func:`erode_voxel_6`, the extra
    diagonal constraints strengthen the erosion in the horizontal plane while
    the vertical direction keeps the weak face-only strength.

    Torch-native: accepts a ``torch.Tensor`` (any device) or ``np.ndarray``
    and returns the same type, so GPU occupancy grids never round-trip
    through numpy.

    Parameters
    ----------
    occupancy : torch.Tensor or np.ndarray
        3-D boolean or binary array.  ``True`` / non-zero means occupied.

    Returns
    -------
    torch.Tensor or np.ndarray
        Eroded boolean grid with the same shape as *occupancy*.
    """
    tensor, was_numpy = _prepare_occupancy(occupancy)
    eroded = _erode_torch(tensor, 10)
    return eroded.cpu().numpy() if was_numpy else eroded


def erode_voxel_26(occupancy) -> torch.Tensor | np.ndarray:
    """Erode a 3D binary voxel grid with a 26-neighbourhood structuring element.

    A voxel survives only when it AND all 26 surrounding voxels (the full
    3 × 3 × 3 cube) are occupied.  Boundary voxels are padded with occupied
    (1) so they survive as long as their in-grid neighbours are all occupied.

    Torch-native: accepts a ``torch.Tensor`` (any device) or ``np.ndarray``
    and returns the same type, so GPU occupancy grids never round-trip
    through numpy.

    Parameters
    ----------
    occupancy : torch.Tensor or np.ndarray
        3-D boolean or binary array.  ``True`` / non-zero means occupied.

    Returns
    -------
    torch.Tensor or np.ndarray
        Eroded boolean grid with the same shape as *occupancy*.
    """
    tensor, was_numpy = _prepare_occupancy(occupancy)
    eroded = _erode_torch(tensor, 26)
    return eroded.cpu().numpy() if was_numpy else eroded


# ---------------------------------------------------------------------------
# Surface extraction
# ---------------------------------------------------------------------------

def compute_scene_surface(
    occupancy,
    thickness: int = 1,
    erosion_mode: int = 26,
) -> torch.Tensor | np.ndarray:
    """Extract the surface shell of a 3D occupancy grid.

    The surface is the symmetric difference (XOR) between the occupancy and
    its own erosion — i.e. the occupied layer that *thickness* erosion
    passes would remove.  Because erosion only ever clears occupied cells,
    this is exactly ``occupancy & ~eroded``.

    The result can be used as a **motion envelope**: the boundary shell the
    character's body may contact or must clear while moving through the
    scene (e.g. wall/floor proximity, contact-richness weighting, or
    visualizing where the free-space margin is thinnest).  Interior voxels
    deep inside obstacles are not part of the envelope.

    Torch-native: accepts a ``torch.Tensor`` (any device) or ``np.ndarray``
    and returns the same type.  This matters in the training pipeline,
    where the occupancy comes from :func:`query_local_occupancy` as a
    (possibly GPU) tensor — erosion and XOR run in torch on that device.

    Parameters
    ----------
    occupancy : torch.Tensor or np.ndarray
        3-D boolean or binary array.  ``True`` / non-zero means occupied.
    thickness : int
        Envelope thickness in voxels.  For *n*, the occupancy is eroded
        *n* times and XORed with the original — the surface covers the
        *n* outermost occupied layers.  ``thickness=0`` yields an empty
        surface.
    erosion_mode : int
        Erosion neighbourhood: ``6`` (face-sharing), ``10`` (faces +
        horizontal diagonals) or ``26`` (full 3×3×3, default).  Larger
        neighbourhoods erode more aggressively, so a given *thickness*
        yields a thicker shell on concave geometry.

    Returns
    -------
    torch.Tensor or np.ndarray
        Boolean grid with the same shape as *occupancy* — True on surface
        voxels, False elsewhere (including the interior of obstacles).
    """
    tensor, was_numpy = _prepare_occupancy(occupancy)
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected a 3-D occupancy array, got shape {tuple(tensor.shape)}"
        )
    if not isinstance(thickness, (int, np.integer)) or thickness < 0:
        raise ValueError(
            f"thickness must be a non-negative int, got {thickness!r}"
        )
    if erosion_mode not in _EROSION_COUNTS:
        raise ValueError(
            f"erosion_mode must be one of {sorted(_EROSION_COUNTS)}, "
            f"got {erosion_mode!r}"
        )

    # Explicit copy: without clone, a bool torch input would alias the
    # caller's tensor, so anything mutating the working grid could leak out.
    eroded = tensor.clone()
    for _ in range(int(thickness)):
        eroded = _erode_torch(eroded, erosion_mode)
    surface = torch.logical_xor(eroded, tensor)
    return surface.cpu().numpy() if was_numpy else surface


def compute_scene_surface_batch(
    occupancy,
    thicknesses,
    erosion_mode: int = 26,
) -> torch.Tensor | np.ndarray:
    """Batched equivalent of compute_scene_surface with per-sample thickness."""
    tensor, was_numpy = _prepare_occupancy(occupancy)
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected a batched 4-D occupancy array, got "
            f"shape {tuple(tensor.shape)}"
        )
    if erosion_mode not in _EROSION_COUNTS:
        raise ValueError(
            f"erosion_mode must be one of {sorted(_EROSION_COUNTS)}, "
            f"got {erosion_mode!r}"
        )
    if isinstance(thicknesses, (int, np.integer)):
        thickness_list = [int(thicknesses)] * tensor.shape[0]
    else:
        thickness_list = [int(value) for value in thicknesses]
    if len(thickness_list) != tensor.shape[0]:
        raise ValueError(
            f"Expected {tensor.shape[0]} thicknesses, got "
            f"{len(thickness_list)}"
        )
    if any(value < 0 for value in thickness_list):
        raise ValueError(f"thicknesses must be non-negative, got {thicknesses!r}")

    surface = torch.zeros_like(tensor, dtype=torch.bool)
    for thickness in sorted(set(thickness_list)):
        if thickness == 0:
            continue
        indices = [
            idx for idx, value in enumerate(thickness_list)
            if value == thickness
        ]
        index = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
        selected = tensor.index_select(0, index)
        eroded = selected.clone()
        for _ in range(thickness):
            eroded = _erode_torch(eroded, erosion_mode)
        surface.index_copy_(0, index, torch.logical_xor(eroded, selected))
    return surface.cpu().numpy() if was_numpy else surface


@lru_cache(maxsize=16)
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

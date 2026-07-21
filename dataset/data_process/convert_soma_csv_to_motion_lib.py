#!/usr/bin/env python3  # noqa: EXE001
# ruff: noqa: T201, DOC
"""Convert SOMA retargeter CSV/PKL data to motion_lib format for SONIC training.

SOMA retargeter outputs G1 29-DOF motion data as CSV files (joint_pos.csv,
body_pos.csv, body_quat.csv) or as a joblib PKL with the same fields. This
script converts that data into the motion_lib PKL format expected by SONIC
training (root_trans_offset, pose_aa, dof, root_rot, fps).

Supports five input modes:
  1. Single motion directory with CSVs (joint_pos.csv, body_pos.csv, body_quat.csv)
  2. Parent directory containing multiple motion subdirectories
  3. Deploy PKL file (joblib dict with joint_pos, body_pos_w, body_quat_w per sequence)
  4. Directory of flat Bones-SEED CSVs (single CSV per motion, degrees+cm)
  5. Parent directory of session dirs containing Bones-SEED CSVs

Usage:
    # Single CSV directory
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/tired_squat_003__A360 \
        --output data/soma_test.pkl --fps 50

    # Batch: parent dir with multiple motion subdirs
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/all_demo_4seqs \
        --output data/soma_demo_4seqs.pkl --fps 50

    # Deploy PKL file
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input data/soma_retarget/bones_test.pkl \
        --output data/soma_bones_test.pkl --fps 50

    # Bones-SEED: directory of flat CSVs (single session)
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input /path/to/bones_SEED/g1/csv/210531 \
        --output data/bones_seed_210531.pkl --fps 50

    # Bones-SEED: all sessions (parent dir)
    python scripts/motion/convert_soma_csv_to_motion_lib.py \
        --input /path/to/bones_SEED/g1/csv \
        --output data/bones_seed_all.pkl --fps 50
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import transform
from scipy.spatial.transform import Slerp, Rotation as R
from tqdm import tqdm

# IsaacLab ↔ MuJoCo joint reordering (29 DOFs for G1).
# MJ_TO_IL[mj] = il: for MuJoCo DOF index mj, gives the IsaacLab index il.
# Source: external_dependencies/SONIC_Web/demo_python.py
MJ_TO_IL = np.array(
    [
        0,
        3,
        6,
        9,
        13,
        17,
        1,
        4,
        7,
        10,
        14,
        18,
        2,
        5,
        8,
        11,
        15,
        19,
        21,
        23,
        25,
        27,
        12,
        16,
        20,
        22,
        24,
        26,
        28,
    ],
    dtype=np.int32,
)

# ---------------------------------------------------------------------------
# MuJoCo FK — lazy imports to keep multiprocessing light
# ---------------------------------------------------------------------------
_MJ_MODEL = None       # MjModel singleton (fork-safe via multiprocessing)
_MJ_MODEL_PATH = None  # cached path for lazy init

# Default MOB parameters (same as occHIPC/utils/g1_mob.py)
DEFAULT_MOB_UNIT = 0.08     # voxel resolution in meters
DEFAULT_MOB_MARGIN = 0.16   # padding around swept volume in meters

# MuJoCo joint ordering for G1 (matches g1_hardware.py mujoco_joint_names).
# 29 DOFs in MuJoCo/MJCF actuator order — same as BONES_CSV_JOINT_NAMES.
MUJOCO_QPOS_END = 36  # 7 (freejoint: 3 pos + 4 quat) + 29 (actuated joints)


# G1 29-DOF axis definitions (from Humanoid_Batch / g1_29dof_rev_1_0.xml).
# Each DOF rotates around a single axis. Hardcoded to avoid torch dependency.
NUM_DOF = 29
NUM_BODIES = 30  # pelvis + 29 actuated links
DOF_AXIS = np.array(
    [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],  # left leg
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 0, 0],  # right leg
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],  # waist
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],  # left arm
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],  # right arm
    ],
    dtype=np.float32,
)


# Joint names in Bones-SEED CSV column order (after Frame + 6 root columns).
# These are in MuJoCo/MJCF actuator order (same as g1_29dof_rev_1_0.xml motors).
BONES_CSV_JOINT_NAMES = [
    "left_hip_pitch_joint_dof",
    "left_hip_roll_joint_dof",
    "left_hip_yaw_joint_dof",
    "left_knee_joint_dof",
    "left_ankle_pitch_joint_dof",
    "left_ankle_roll_joint_dof",
    "right_hip_pitch_joint_dof",
    "right_hip_roll_joint_dof",
    "right_hip_yaw_joint_dof",
    "right_knee_joint_dof",
    "right_ankle_pitch_joint_dof",
    "right_ankle_roll_joint_dof",
    "waist_yaw_joint_dof",
    "waist_roll_joint_dof",
    "waist_pitch_joint_dof",
    "left_shoulder_pitch_joint_dof",
    "left_shoulder_roll_joint_dof",
    "left_shoulder_yaw_joint_dof",
    "left_elbow_joint_dof",
    "left_wrist_roll_joint_dof",
    "left_wrist_pitch_joint_dof",
    "left_wrist_yaw_joint_dof",
    "right_shoulder_pitch_joint_dof",
    "right_shoulder_roll_joint_dof",
    "right_shoulder_yaw_joint_dof",
    "right_elbow_joint_dof",
    "right_wrist_roll_joint_dof",
    "right_wrist_pitch_joint_dof",
    "right_wrist_yaw_joint_dof",
]


def load_bones_csv(csv_path: str) -> dict:
    """Load a single Bones-SEED flat CSV motion file.

    Bones-SEED CSV format: Frame, root_translate{X,Y,Z}, root_rotate{X,Y,Z}, 29 joint DOFs.
    All angles in degrees, positions in centimeters.
    """
    import pandas as pd

    data = pd.read_csv(csv_path)
    T = len(data)

    # Root position: cm → meters
    root_pos = (
        np.stack(
            [
                data["root_translateX"].values,  # noqa: PD011
                data["root_translateY"].values,  # noqa: PD011
                data["root_translateZ"].values,  # noqa: PD011
            ],
            axis=1,
        ).astype(np.float32)
        / 100.0
    )  # cm → m

    # Root rotation: Euler xyz (intrinsic) degrees → quaternion (xyzw scipy convention)
    # Reference: gear_sonic/data_process/process_bones_to_motionlib.py uses "xyz" (intrinsic)
    euler_deg = np.stack(
        [
            data["root_rotateX"].values,  # noqa: PD011
            data["root_rotateY"].values,  # noqa: PD011
            data["root_rotateZ"].values,  # noqa: PD011
        ],
        axis=1,
    ).astype(np.float64)
    root_quat_xyzw = (
        transform.Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat().astype(np.float32)
    )
    # Convert xyzw → wxyz for body_quat_w format
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    # Joint DOFs: degrees → radians, already in MuJoCo/MJCF actuator order
    joint_cols = [c for c in data.columns if c.endswith("_dof")]
    joint_pos_mj = np.deg2rad(data[joint_cols].values).astype(np.float32)  # (T, 29)

    # Create dummy body_pos_w and body_quat_w (only root body populated, rest zeros)
    # The converter only uses body_pos_w[:,0] for root_trans and body_quat_w[:,0] for root_rot
    body_pos_w = np.zeros((T, 14, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = root_pos
    body_quat_w = np.zeros((T, 14, 4), dtype=np.float32)
    body_quat_w[:, :, 0] = 1.0  # identity quaternion wxyz
    body_quat_w[:, 0, :] = root_quat_wxyz

    return {
        "joint_pos": joint_pos_mj,  # (T, 29) MuJoCo order, radians
        "body_pos_w": body_pos_w,  # (T, 14, 3)
        "body_quat_w": body_quat_w,  # (T, 14, 4) wxyz
        "joint_order": "mj",  # already in MuJoCo order, skip IL→MJ reorder
    }


def load_csv_motion(motion_dir: str) -> dict:
    """Load a single motion from a directory of CSV files."""
    joint_pos_f = os.path.join(motion_dir, "joint_pos.csv")
    body_pos_f = os.path.join(motion_dir, "body_pos.csv")
    body_quat_f = os.path.join(motion_dir, "body_quat.csv")

    if not os.path.exists(joint_pos_f):
        return None

    joint_pos = np.loadtxt(joint_pos_f, delimiter=",", skiprows=1, dtype=np.float32)
    body_pos = np.loadtxt(body_pos_f, delimiter=",", skiprows=1, dtype=np.float32)
    body_quat = np.loadtxt(body_quat_f, delimiter=",", skiprows=1, dtype=np.float32)

    # Reshape body data: (T, 14*3) → (T, 14, 3), (T, 14*4) → (T, 14, 4)
    T = joint_pos.shape[0]
    body_pos = body_pos.reshape(T, -1, 3)
    body_quat = body_quat.reshape(T, -1, 4)

    return {
        "joint_pos": joint_pos,  # (T, 29) IsaacLab order
        "body_pos_w": body_pos,  # (T, 14, 3) world frame
        "body_quat_w": body_quat,  # (T, 14, 4) wxyz format
    }


def convert_sequence(seq_data: dict, fps: int, humanoid_fk=None) -> dict:  # noqa: ARG001
    """Convert a single deploy-format sequence to motion_lib format.

    Args:
        seq_data: dict with joint_pos (T, 29), body_pos_w (T, 14, 3),
                  body_quat_w (T, 14, 4 wxyz)
        fps: frame rate of the input data
        humanoid_fk: Optional Humanoid_Batch instance (unused, kept for compat)

    Returns:
        motion_lib entry dict with root_trans_offset, pose_aa, dof, root_rot, fps
    """
    joint_pos = seq_data["joint_pos"]  # (T, 29)
    body_pos_w = seq_data["body_pos_w"]  # (T, 14, 3)
    body_quat_w = seq_data["body_quat_w"]  # (T, 14, 4) wxyz
    joint_order = seq_data.get("joint_order", "il")  # "il" or "mj"

    T = joint_pos.shape[0]

    # 1. Root position: body_0 (pelvis) position
    root_trans_offset = body_pos_w[:, 0, :].copy()  # (T, 3)

    # 2. Root quaternion: body_0 quaternion, convert wxyz → xyzw (scipy convention)
    root_quat_wxyz = body_quat_w[:, 0, :]  # (T, 4) [w, x, y, z]
    root_quat_xyzw = root_quat_wxyz[:, [1, 2, 3, 0]]  # (T, 4) [x, y, z, w]

    # 3. Reorder DOFs to MuJoCo order if needed
    if joint_order == "il":
        # Input is IsaacLab order → reorder to MuJoCo (MJCF actuator order)
        dof_mj = joint_pos[:, MJ_TO_IL]  # (T, 29)
    else:
        # Input is already in MuJoCo order (e.g., Bones-SEED CSVs)
        dof_mj = joint_pos  # (T, 29)

    # 4. Convert DOF → pose_aa using hardcoded G1 axis definitions
    dof = dof_mj[:, :NUM_DOF]

    # pose_aa[body_idx] = dof_axis * dof_value (axis-angle representation)
    # Body 0 = pelvis (root), bodies 1-29 = actuated joints
    pose_aa = np.zeros((T, NUM_BODIES, 3), dtype=np.float32)
    # Actuated joints: body idx = dof idx + 1
    pose_aa[:, 1:NUM_BODIES, :] = DOF_AXIS[None, :, :] * dof[:, :, None]

    # Set root rotation as axis-angle
    pose_aa[:, 0, :] = transform.Rotation.from_quat(root_quat_xyzw).as_rotvec()

    return {
        "root_trans_offset": root_trans_offset.astype(np.float32),
        "pose_aa": pose_aa.astype(np.float32),
        "dof": dof.astype(np.float32),
        "root_rot": root_quat_xyzw.astype(np.float32),  # xyzw (scipy convention)
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),  # placeholder
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# MuJoCo-based unified FK pipeline (contact_mask + optional MOB)
# ---------------------------------------------------------------------------

def _get_mj_model(xml_path: str):
    """Lazy-init MuJoCo MjModel + MjData (fork-safe singleton for multiprocessing).

    MjModel is read-only after creation — cached at module level and inherited
    via fork.  MjData is mutable per-step — created fresh each call (~us cost).

    Mesh geoms (group=2, type=mesh) are stripped before compilation so that
    STL asset files are not required.  Only primitive collision geoms
    (group=3, sphere/capsule) remain — these are sufficient for MOB + contact.
    """
    import mujoco

    global _MJ_MODEL, _MJ_MODEL_PATH
    xml_path = str(Path(xml_path).resolve())
    if _MJ_MODEL is None or _MJ_MODEL_PATH != xml_path:
        # Strip mesh geoms/assets so STL files aren't required
        xml_str = _strip_mesh_geoms(xml_path)
        _MJ_MODEL = mujoco.MjModel.from_xml_string(xml_str)
        _MJ_MODEL_PATH = xml_path
    return _MJ_MODEL, mujoco.MjData(_MJ_MODEL)


def _strip_mesh_geoms(xml_path: str) -> str:
    """Return MuJoCo XML string with mesh geoms and mesh assets removed.

    Ported from occHIPC/utils/g1_mob.py.  The checked-in G1 XML references
    visual STL meshes that may not exist.  MOB occupancy comes from collision
    primitives (group=3) only, so mesh geoms are safely removed.
    """
    root = ET.parse(xml_path).getroot()
    # Remove geom elements that reference a mesh
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.get("mesh"):
                parent.remove(child)
    # Remove mesh assets (not needed after geoms are stripped)
    asset = root.find("asset")
    if asset is not None:
        for child in list(asset):
            if child.tag == "mesh":
                asset.remove(child)
    # Point meshdir somewhere harmless (no meshes remain to load)
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "/tmp")
    return ET.tostring(root, encoding="unicode")


def _find_foot_body_ids(model) -> tuple:
    """Return (left_foot_body_id, right_foot_body_id) for contact detection.

    Uses the same body names as RobotSkeleton foot_names config:
    left_ankle_roll_link and right_ankle_roll_link.
    """
    import mujoco

    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
    return left_id, right_id


def _get_collision_geom_ids(model) -> list:
    """Return geom IDs for MOB voxelization (group=3 collision geoms).

    Mirrors _collision_geom_ids() from occHIPC/utils/g1_mob.py.
    Excludes mjGEOM_PLANE.
    """
    import mujoco

    ids = []
    for gid in range(model.ngeom):
        gtype = int(model.geom_type[gid])
        group = int(model.geom_group[gid])
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if group == 3:
            ids.append(gid)
    if not ids:
        raise ValueError("No collision geoms (group=3) found in the MJCF model")
    return ids


def _geom_local_half_extents(model, gid: int) -> np.ndarray:
    """Local AABB half-extents for a collision geom (sphere/capsule/cylinder/box).

    Mirrors _geom_local_half_extents() from occHIPC/utils/g1_mob.py.
    """
    import mujoco

    gtype = int(model.geom_type[gid])
    size = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(size[0])
        return np.array([r, r, r], dtype=np.float64)
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = float(size[0]), float(size[1])
        return np.array([r, r, h + r], dtype=np.float64)
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = float(size[0]), float(size[1])
        return np.array([r, r, h], dtype=np.float64)
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        return np.asarray(size, dtype=np.float64).copy()
    raise ValueError(f"Unsupported geom type {gtype} for geom id {gid}")


def _geom_world_aabb(model, data, gid: int):
    """World-space AABB for a geom at the current mj_forward state."""
    center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    half_world = np.abs(rot) @ _geom_local_half_extents(model, gid)
    return center - half_world, center + half_world


def _world_to_index_bounds(lo, hi, llb, unit, shape):
    """Map world-space AABB corners to voxel index ranges."""
    vmin = np.floor((lo - llb) / unit).astype(np.int64)
    vmax = np.ceil((hi - llb) / unit).astype(np.int64)
    vmin = np.clip(vmin, 0, shape)
    vmax = np.clip(vmax, 0, shape)
    return vmin, vmax


def _mark_geom(occu, llb, unit, model, data, gid, cached_aabb=None):
    """Rasterize one MuJoCo geom into the voxel grid occu (mutated in-place).

    Args:
        cached_aabb: optional (lo, hi) tuple from a previous _geom_world_aabb
                     call, to skip redundant AABB recomputation.
    """
    import mujoco

    if cached_aabb is not None:
        lo, hi = cached_aabb
    else:
        lo, hi = _geom_world_aabb(model, data, gid)
    vmin, vmax = _world_to_index_bounds(lo, hi, llb, unit, np.asarray(occu.shape))
    if np.any(vmax <= vmin):
        return

    xs = np.arange(vmin[0], vmax[0])
    ys = np.arange(vmin[1], vmax[1])
    zs = np.arange(vmin[2], vmax[2])
    wx = llb[0] + (xs + 0.5) * unit
    wy = llb[1] + (ys + 0.5) * unit
    wz = llb[2] + (zs + 0.5) * unit
    grid = np.stack(np.meshgrid(wx, wy, wz, indexing="ij"), axis=-1)

    center = np.asarray(data.geom_xpos[gid], dtype=np.float64)
    rot = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    local = (grid - center) @ rot

    gtype = int(model.geom_type[gid])
    size = model.geom_size[gid]
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        mask = np.sum(local * local, axis=-1) <= float(size[0]) ** 2
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = float(size[0]), float(size[1])
        z = np.clip(local[..., 2], -h, h)
        closest = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=-1)
        delta = local - closest
        mask = np.sum(delta * delta, axis=-1) <= r ** 2
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = float(size[0]), float(size[1])
        mask = (
            (local[..., 0] ** 2 + local[..., 1] ** 2 <= r ** 2)
            & (np.abs(local[..., 2]) <= h)
        )
    elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
        half = np.asarray(size, dtype=np.float64)
        mask = np.all(np.abs(local) <= half, axis=-1)
    else:
        return

    occu[vmin[0]:vmax[0], vmin[1]:vmax[1], vmin[2]:vmax[2]] |= mask


def compute_contact_and_mob(
    root_trans_offset: np.ndarray,
    root_rot_xyzw: np.ndarray,
    dof_mj: np.ndarray,
    fps: float,
    xml_path: str,
    height_thresh: float = 0.05,
    vel_thresh: float = 0.15,
    mob: bool = False,
    mob_unit: float = DEFAULT_MOB_UNIT,
    mob_margin: float = DEFAULT_MOB_MARGIN,
    mob_frame_stride: int = 1,
    verbose: bool = False,
) -> dict:
    """Compute contact_mask and optionally MOB occupancy via MuJoCo mj_forward.

    Uses a single FK pass per frame for contact_mask, plus a second pass for
    MOB voxel marking.  The body positions of left_ankle_roll_link and
    right_ankle_roll_link are used for foot contact detection (same formula as
    SONIC G1FootContactLoss).

    Args:
        root_trans_offset:  (T, 3) pelvis world translation, meters.
        root_rot_xyzw:      (T, 4) pelvis rotation, xyzw quaternion (scipy).
        dof_mj:             (T, 29) joint angles, radians, MuJoCo/MJCF order.
        fps:                frame rate of the input data.
        xml_path:           path to g1_29dof_with_collision.xml.
        height_thresh:      contact height threshold in meters (default 0.05).
        vel_thresh:         contact velocity threshold in m/s (default 0.15).
        mob:                if True, also compute MOB swept occupancy.
        mob_unit:           voxel resolution for MOB (default 0.08 m).
        mob_margin:         padding around swept volume (default 0.16 m).
        mob_frame_stride:   subsample frames for MOB (default 1 = all frames).

    Returns:
        dict with:
            contact_mask: (T, 2) float32, [:, 0]=left foot, [:, 1]=right foot.
            If mob=True, also:
                mob_occu_global:   bool ndarray [X, Y, Z], 1 = free space (robot never
                                   occupies this voxel).  Complement of swept volume.
                mob_unit:          float, voxel size in meters.
                mob_llb:           (1, 3) float32, lower-left-back corner.
    """
    import mujoco

    model, data = _get_mj_model(xml_path)
    left_body_id, right_body_id = _find_foot_body_ids(model)

    T = root_trans_offset.shape[0]

    # MuJoCo freejoint quaternion is wxyz; input is xyzw (scipy convention)
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]].astype(np.float64)

    foot_z = np.zeros((T, 2), dtype=np.float64)

    mob_geom_ids = _get_collision_geom_ids(model) if mob else []
    n_geoms = len(mob_geom_ids)

    # Pre-allocate AABB arrays (Pass 1 writes directly into these)
    if mob:
        n_aabbs = T * n_geoms
        mob_mins = np.empty((n_aabbs, 3), dtype=np.float64)
        mob_maxs = np.empty((n_aabbs, 3), dtype=np.float64)
        # Cache: [t][gid] = (xpos, xmat, lo, hi)
        mob_geom_cache: list = []
    else:
        mob_mins = mob_maxs = None  # type: ignore[assignment]
        mob_geom_cache = []  # unused, keeps linter happy

    _frame_iter = tqdm(range(T), desc="FK+contact", unit="f", leave=False) if verbose else range(T)

    # ── Pass 1: FK all frames → foot_z + MOB global AABB ──
    aabb_idx = 0
    for t in _frame_iter:
        data.qpos[0:3] = root_trans_offset[t]
        data.qpos[3:7] = root_rot_wxyz[t]
        data.qpos[7:MUJOCO_QPOS_END] = dof_mj[t]
        mujoco.mj_forward(model, data)

        foot_z[t, 0] = data.xpos[left_body_id, 2]
        foot_z[t, 1] = data.xpos[right_body_id, 2]

        if mob:
            frame_cache = []
            for i, gid in enumerate(mob_geom_ids):
                lo, hi = _geom_world_aabb(model, data, gid)
                mob_mins[aabb_idx] = lo
                mob_maxs[aabb_idx] = hi
                aabb_idx += 1
                # Cache geom poses + AABB so Pass 2 skips FK & AABB entirely
                frame_cache.append((
                    np.asarray(data.geom_xpos[gid], dtype=np.float64).copy(),
                    np.asarray(data.geom_xmat[gid], dtype=np.float64).copy(),
                    lo.copy(),
                    hi.copy(),
                ))
            mob_geom_cache.append(frame_cache)

    # ── Contact mask from foot_z ──
    dt = 1.0 / fps
    foot_vel = np.zeros((T, 2), dtype=np.float64)
    foot_vel[1:] = np.abs(foot_z[1:] - foot_z[:-1]) / dt

    contact = ((foot_z < height_thresh) & (foot_vel < vel_thresh)).astype(np.float32)

    result: dict = {"contact_mask": contact}

    # ── MOB: Pass 2 — create grid and mark voxels (replay cached poses + AABBs) ──
    if mob:
        llb = mob_mins.min(axis=0) - mob_margin
        rub = mob_maxs.max(axis=0) + mob_margin
        shape = np.ceil((rub - llb) / mob_unit).astype(np.int64) + 1
        swept_occu = np.zeros(tuple(shape.tolist()), dtype=bool)

        frame_ids = list(range(0, T, mob_frame_stride))
        voxel_iter = tqdm(frame_ids, desc="MOB rasterize", unit="f", leave=False) if verbose else frame_ids
        for t in voxel_iter:
            frame_cache = mob_geom_cache[t]
            for i, gid in enumerate(mob_geom_ids):
                xpos, xmat, lo, hi = frame_cache[i]
                data.geom_xpos[gid] = xpos
                data.geom_xmat[gid] = xmat
                _mark_geom(swept_occu, llb, mob_unit, model, data, gid, cached_aabb=(lo, hi))

        # Store as free space: 1 = robot NEVER occupies this voxel
        result["mob_occu_global"] = ~swept_occu
        result["mob_unit"] = mob_unit
        result["mob_llb"] = llb.reshape(1, 3).astype(np.float32)

    return result


def resample_sequence(entry: dict, fps_source: int, fps_target: int) -> dict:
    """Resample a motion_lib entry using LERP (translation, DOFs) + SLERP (rotation).

    Handles arbitrary non-integer frame-rate ratios (e.g. 120→50).
    When fps_source == fps_target, returns the entry unchanged.
    """
    if fps_source == fps_target:
        return {**entry}

    T_src = entry["root_trans_offset"].shape[0]
    t_src = np.linspace(0, (T_src - 1) / fps_source, T_src, dtype=np.float64)
    T_tgt = max(int(T_src * fps_target / fps_source), 2)
    t_tgt = np.linspace(0, (T_src - 1) / fps_source, T_tgt, dtype=np.float64)

    # ── Translation: per-channel linear interpolation ──
    trans_src = entry["root_trans_offset"].astype(np.float64)  # (T_src, 3)
    trans_tgt = interp1d(t_src, trans_src, axis=0, kind="linear")(t_tgt).astype(np.float32)

    # ── Rotation: SLERP on xyzw quaternion ──
    quat_xyzw = entry["root_rot"].astype(np.float64)  # (T_src, 4) xyzw
    slerp = Slerp(t_src, R.from_quat(quat_xyzw))
    quat_tgt = slerp(t_tgt).as_quat().astype(np.float32)  # (T_tgt, 4) xyzw

    # ── DOFs: per-channel linear interpolation ──
    dof_src = entry["dof"].astype(np.float64)  # (T_src, D)
    dof_tgt = interp1d(t_src, dof_src, axis=0, kind="linear")(t_tgt).astype(np.float32)

    # ── pose_aa: reconstruct from interpolated root_rot + dof ──
    # Body 0 = root rotation (axis-angle from quat_tgt)
    # Bodies 1..29 = DOF axis * dof_value
    NUM_DOF_LOCAL = dof_tgt.shape[1]
    NUM_BODIES_LOCAL = NUM_DOF_LOCAL + 1
    DOF_AXIS_LOCAL = DOF_AXIS[:NUM_DOF_LOCAL]  # subset if not 29
    pose_aa_tgt = np.zeros((T_tgt, NUM_BODIES_LOCAL, 3), dtype=np.float32)
    pose_aa_tgt[:, 0, :] = R.from_quat(quat_tgt).as_rotvec().astype(np.float32)
    pose_aa_tgt[:, 1:, :] = DOF_AXIS_LOCAL[None, :, :] * dof_tgt[:, :, None]

    # ── smpl_joints: per-channel linear interpolation (placeholder) ──
    smpl_src = entry["smpl_joints"].astype(np.float64)
    smpl_tgt = interp1d(t_src, smpl_src.reshape(T_src, -1), axis=0, kind="linear")(t_tgt)
    smpl_tgt = smpl_tgt.reshape(T_tgt, -1, 3).astype(np.float32)

    return {
        "root_trans_offset": trans_tgt,
        "pose_aa": pose_aa_tgt,
        "dof": dof_tgt,
        "root_rot": quat_tgt,
        "smpl_joints": smpl_tgt,
        "fps": fps_target,
    }


def process_session_csvs(args_tuple):
    """Process all CSVs in a single session directory. Used by multiprocessing."""
    session_dir, session_name, out_dir, fps, fps_source, mjcf_dir, mob_cfg, worker_pos = args_tuple
    import time
    import warnings

    warnings.filterwarnings("ignore")

    csv_files = sorted([f for f in os.listdir(session_dir) if f.endswith(".csv")])

    session_out = os.path.join(out_dir, session_name)
    os.makedirs(session_out, exist_ok=True)

    xml_path = os.path.join(mjcf_dir, "g1_29dof_with_collision.xml")

    converted = 0
    failed = 0
    t_csv = t_convert = t_resample = t_fk = t_dump = 0.0
    csv_iter = tqdm(
        csv_files, desc=session_name[:20], unit="csv",
        position=worker_pos, leave=False,
    )
    for csv_f in csv_iter:
        name = os.path.splitext(csv_f)[0]
        out_path = os.path.join(session_out, name + ".pkl")
        if os.path.exists(out_path):
            converted += 1  # skip existing
            continue
        try:
            csv_path = os.path.join(session_dir, csv_f)
            t0 = time.time()
            seq = load_bones_csv(csv_path)
            t_csv += time.time() - t0

            fps_for_convert = fps_source if fps_source else fps

            t0 = time.time()
            entry = convert_sequence(seq, fps_for_convert)
            t_convert += time.time() - t0

            t0 = time.time()
            if fps_source and fps_source != fps:
                entry = resample_sequence(entry, fps_source, fps)
            t_resample += time.time() - t0

            # ── Unified FK: contact_mask + optional MOB ──
            t0 = time.time()
            fk_result = compute_contact_and_mob(
                entry["root_trans_offset"],
                entry["root_rot"],
                entry["dof"],
                fps,
                xml_path,
                mob=mob_cfg.get("enabled", False),
                mob_unit=mob_cfg.get("unit", DEFAULT_MOB_UNIT),
                mob_margin=mob_cfg.get("margin", DEFAULT_MOB_MARGIN),
                mob_frame_stride=mob_cfg.get("frame_stride", 1),
            )
            entry["contact_mask"] = fk_result["contact_mask"]
            if mob_cfg.get("enabled", False):
                entry["mob_occu_global"] = fk_result["mob_occu_global"]
                entry["mob_unit"] = fk_result["mob_unit"]
                entry["mob_llb"] = fk_result["mob_llb"]
            t_fk += time.time() - t0

            t0 = time.time()
            joblib.dump({name: entry}, out_path, compress=True)
            t_dump += time.time() - t0
            converted += 1
        except Exception:  # noqa: BLE001
            if failed == 0:  # print first error per session
                import traceback
                print(f"\n  [{session_name}] ERROR on {csv_f}:\n  {traceback.format_exc().strip().replace(chr(10), chr(10) + '  ')}",
                      file=sys.stderr)
            failed += 1

    if converted > 0:
        elapsed_csv = t_csv / converted * 1000
        elapsed_convert = t_convert / converted * 1000
        elapsed_resample = t_resample / converted * 1000
        elapsed_fk = t_fk / converted * 1000
        elapsed_dump = t_dump / converted * 1000
        fk_label = "fk+mob" if mob_cfg.get("enabled") else "fk+contact"
        print(f"  {session_name}: {converted}/{len(csv_files)} converted "
              f"(csv {elapsed_csv:.0f}ms  convert {elapsed_convert:.0f}ms  "
              f"resample {elapsed_resample:.0f}ms  {fk_label} {elapsed_fk:.0f}ms  "
              f"dump {elapsed_dump:.0f}ms)"
              + (f"  {failed} failed" if failed else ""))
    return session_name, converted, failed, len(csv_files)


def main():
    parser = argparse.ArgumentParser(description="Convert SOMA CSV/PKL to motion_lib format")
    parser.add_argument(
        "--input", required=True, help="CSV dir, parent dir of CSV dirs, or deploy PKL"
    )
    parser.add_argument(
        "--output", required=True, help="Output path (PKL file or directory for individual PKLs)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target output FPS (default: 30, matches process_bones_to_motionlib)",
    )
    parser.add_argument(
        "--fps_source",
        type=int,
        default=None,
        help="Source data FPS. If set and != --fps, data is downsampled. "
        "Bones-SEED CSVs are typically 120fps.",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Write individual PKLs per motion (preserves session dir structure)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel workers for --individual mode",
    )
    parser.add_argument(
        "--mjcf_dir",
        type=str,
        default="TextOpRobotMDAR/description/robots/g1",
        help="Path to directory containing g1_29dof_with_collision.xml "
        "(default: TextOpRobotMDAR/description/robots/g1, relative to repo root)",
    )
    parser.add_argument(
        "--mob",
        action="store_true",
        help="Also compute MOB swept occupancy grid for each motion "
        "(1=free space complement of swept robot volume, requires MuJoCo)",
    )
    parser.add_argument(
        "--mob_unit",
        type=float,
        default=DEFAULT_MOB_UNIT,
        help=f"MOB voxel resolution in meters (default: {DEFAULT_MOB_UNIT})",
    )
    parser.add_argument(
        "--mob_margin",
        type=float,
        default=DEFAULT_MOB_MARGIN,
        help=f"MOB grid padding around swept volume in meters (default: {DEFAULT_MOB_MARGIN})",
    )
    parser.add_argument(
        "--mob_frame_stride",
        type=int,
        default=1,
        help="MOB frame subsampling factor (default: 1 = all frames). "
        "Set to 2 for 2x speedup with negligible quality loss at 50fps.",
    )
    args = parser.parse_args()

    mob_cfg = {
        "enabled": args.mob,
        "unit": args.mob_unit,
        "margin": args.mob_margin,
        "frame_stride": args.mob_frame_stride,
    }

    print(f"G1 {NUM_DOF} DOFs, {NUM_BODIES} bodies (hardcoded axes)")
    if args.mob:
        print(f"MOB enabled: unit={args.mob_unit}m, margin={args.mob_margin}m, stride={args.mob_frame_stride}")

    # Individual PKL mode: skip scanning, go straight to parallel per-session processing
    if args.individual:
        if not os.path.isdir(args.input):
            print("ERROR: --individual requires a directory input")
            sys.exit(1)

        # Detect: is input a single session dir (contains CSVs) or parent of sessions?
        has_csvs = any(f.endswith(".csv") for f in os.listdir(args.input))
        subdirs = sorted(
            [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        )
        has_session_subdirs = (
            any(
                any(f.endswith(".csv") for f in os.listdir(os.path.join(args.input, d)))
                for d in subdirs[:3]
            )
            if subdirs
            else False
        )

        session_dirs = []
        if has_session_subdirs:
            for i, d in enumerate(subdirs):
                subdir = os.path.join(args.input, d)
                if any(f.endswith(".csv") for f in os.listdir(subdir)):
                    session_dirs.append(
                        (subdir, d, args.output, args.fps, args.fps_source,
                         args.mjcf_dir, mob_cfg, i)
                    )
        elif has_csvs:
            session_name = os.path.basename(args.input.rstrip("/"))
            session_dirs.append(
                (args.input, session_name, args.output, args.fps, args.fps_source,
                 args.mjcf_dir, mob_cfg, 0)
            )

        print(f"\nBatch converting {len(session_dirs)} sessions with {args.num_workers} workers")
        print(f"Output: {args.output}")
        os.makedirs(args.output, exist_ok=True)

        import multiprocessing

        total_converted = 0
        total_failed = 0
        total_csvs = 0
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            pbar = tqdm(
                pool.imap_unordered(process_session_csvs, session_dirs),
                total=len(session_dirs),
                desc="Sessions",
                unit="sess",
                position=args.num_workers,
            )
            for session_name, converted, failed, n_csvs in pbar:
                total_converted += converted
                total_failed += failed
                total_csvs += n_csvs
                pbar.set_postfix_str(
                    f"{total_converted}/{total_csvs} ok" + (f" {total_failed} fail" if total_failed else "")
                )

        print(
            f"\nDone: {total_converted} motions converted, {total_failed} failed, {total_csvs} total CSVs"
        )
        return

    # Detect input mode (combined PKL output path)
    sequences = {}

    if args.input.endswith(".pkl"):
        # Mode 3: Deploy PKL file
        print(f"Loading deploy PKL: {args.input}")
        data = joblib.load(args.input)
        for name, seq in data.items():
            sequences[name] = seq
        print(f"  Found {len(sequences)} sequences")

    elif os.path.isfile(os.path.join(args.input, "joint_pos.csv")):
        # Mode 1: Single CSV directory
        name = os.path.basename(args.input)
        print(f"Loading single CSV motion: {name}")
        seq = load_csv_motion(args.input)
        if seq is None:
            print("ERROR: joint_pos.csv not found")
            sys.exit(1)
        sequences[name] = seq
        print(f"  {seq['joint_pos'].shape[0]} frames")

    elif os.path.isdir(args.input):
        # Check if directory contains flat CSVs (Bones-SEED format)
        csv_files = sorted([f for f in os.listdir(args.input) if f.endswith(".csv")])
        subdirs = sorted(
            [d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))]
        )

        if csv_files and not any(
            os.path.exists(os.path.join(args.input, d, "joint_pos.csv"))
            for d in subdirs[:5]  # check first 5 subdirs
        ):
            # Mode 4: Directory of flat Bones-SEED CSVs
            print(f"Scanning directory for Bones-SEED CSVs: {args.input}")
            for csv_f in csv_files:
                csv_path = os.path.join(args.input, csv_f)
                name = os.path.splitext(csv_f)[0]
                try:
                    seq = load_bones_csv(csv_path)
                    sequences[name] = seq
                except Exception as e:  # noqa: BLE001
                    print(f"  WARNING: Failed to load {csv_f}: {e}")
            print(f"  Found {len(sequences)} Bones-SEED CSV motions")
        elif subdirs:
            # Check if subdirs contain flat CSVs (batch of session dirs)
            has_session_csvs = False
            for dname in subdirs[:3]:
                subdir = os.path.join(args.input, dname)
                sub_csvs = [f for f in os.listdir(subdir) if f.endswith(".csv")]
                if sub_csvs and not os.path.exists(os.path.join(subdir, "joint_pos.csv")):
                    has_session_csvs = True
                    break

            if has_session_csvs:
                # Mode 5: Parent dir of session dirs containing Bones-SEED CSVs
                print(f"Scanning session directories for Bones-SEED CSVs: {args.input}")
                for dname in sorted(subdirs):
                    subdir = os.path.join(args.input, dname)
                    sub_csvs = sorted([f for f in os.listdir(subdir) if f.endswith(".csv")])
                    for csv_f in sub_csvs:
                        csv_path = os.path.join(subdir, csv_f)
                        name = os.path.splitext(csv_f)[0]
                        try:
                            seq = load_bones_csv(csv_path)
                            sequences[name] = seq
                        except Exception as e:  # noqa: BLE001
                            print(f"  WARNING: Failed to load {dname}/{csv_f}: {e}")
                    if sub_csvs:
                        print(f"  Session {dname}: {len(sub_csvs)} CSVs")
                print(f"  Found {len(sequences)} total Bones-SEED CSV motions")
            else:
                # Mode 2: Parent directory with SOMA-style subdirectories
                print(f"Scanning directory: {args.input}")
                for dname in sorted(subdirs):
                    subdir = os.path.join(args.input, dname)
                    seq = load_csv_motion(subdir)
                    if seq is not None:
                        sequences[dname] = seq
                print(f"  Found {len(sequences)} motion directories with CSVs")
    else:
        print(f"ERROR: {args.input} is not a valid input")
        sys.exit(1)

    if not sequences:
        print("ERROR: No sequences found")
        sys.exit(1)

    # Convert each sequence (combined PKL mode)
    xml_path = os.path.join(args.mjcf_dir, "g1_29dof_with_collision.xml")
    motion_lib_dict = {}
    seq_iter = tqdm(sequences.items(), desc="Converting", unit="seq")
    for name, seq_data in seq_iter:
        seq_iter.set_postfix_str(name[:40])
        T = seq_data["joint_pos"].shape[0]
        fps_for_convert = args.fps_source if args.fps_source else args.fps
        entry = convert_sequence(seq_data, fps_for_convert)
        if args.fps_source and args.fps_source != args.fps:
            entry = resample_sequence(entry, args.fps_source, args.fps)

        # ── Unified FK: contact_mask + optional MOB ──
        fk_result = compute_contact_and_mob(
            entry["root_trans_offset"],
            entry["root_rot"],
            entry["dof"],
            args.fps,
            xml_path,
            mob=mob_cfg["enabled"],
            mob_unit=mob_cfg["unit"],
            mob_margin=mob_cfg["margin"],
            mob_frame_stride=mob_cfg["frame_stride"],
            verbose=True,
        )
        entry["contact_mask"] = fk_result["contact_mask"]
        if mob_cfg["enabled"]:
            entry["mob_occu_global"] = fk_result["mob_occu_global"]
            entry["mob_unit"] = fk_result["mob_unit"]
            entry["mob_llb"] = fk_result["mob_llb"]
        motion_lib_dict[name] = entry

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"\nSaving motion_lib PKL: {args.output}")
    joblib.dump(motion_lib_dict, args.output, compress=True)
    print(f"Done: {len(motion_lib_dict)} sequences saved")


if __name__ == "__main__":
    main()

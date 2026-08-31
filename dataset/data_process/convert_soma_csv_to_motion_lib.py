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
import torch
from scipy.interpolate import interp1d
from scipy.spatial import transform
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from robotmdar.utils.occupancy import erode_voxel_26

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
_TORCH_FK_MODELS = {}

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
MUJOCO_DOF_JOINT_NAMES = tuple(
    name.removesuffix("_dof") for name in BONES_CSV_JOINT_NAMES
)


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

    # Root rotation: Euler xyz (extrinsic) degrees → quaternion (xyzw scipy convention)
    # SciPy convention: lowercase 'xyz' = extrinsic (R = Rz·Ry·Rx); uppercase = intrinsic.
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
    missing_joint_cols = [c for c in BONES_CSV_JOINT_NAMES if c not in data.columns]
    if missing_joint_cols:
        raise ValueError(
            f"Bones-SEED CSV is missing {len(missing_joint_cols)} G1 joint columns: "
            f"{missing_joint_cols}"
        )
    # Select by semantic name rather than relying on incidental CSV column order.
    joint_pos_mj = np.deg2rad(
        data[BONES_CSV_JOINT_NAMES].to_numpy(dtype=np.float32)
    ).astype(np.float32)  # (T, 29)
    if not np.isfinite(joint_pos_mj).all():
        raise ValueError("Bones-SEED CSV contains non-finite joint angles")

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
        "dof_order": "mujoco",
        "dof_names": list(MUJOCO_DOF_JOINT_NAMES),
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


def _quat_wxyz_to_matrix_torch(quat):
    """Convert normalized wxyz quaternions to rotation matrices."""
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = quat.unbind(dim=-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z), two * (x * y - z * w),
            two * (x * z + y * w), two * (x * y + z * w),
            1 - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w),
            1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _axis_angle_to_matrix_torch(axis, angle):
    """Convert fixed rotation axes and batched scalar angles to matrices."""
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    half = angle * 0.5
    xyz = axis.unsqueeze(0) * torch.sin(half).unsqueeze(-1)
    quat = torch.cat((torch.cos(half).unsqueeze(-1), xyz), dim=-1)
    return _quat_wxyz_to_matrix_torch(quat)


class _TorchBatchKinematics:
    """Batched G1 FK initialized from MuJoCo's compiled model constants."""

    def __init__(self, model, device: str):
        import mujoco

        if model.nbody - 1 != NUM_BODIES:
            raise ValueError(
                f"Expected {NUM_BODIES} G1 bodies, compiled model has {model.nbody - 1}"
            )
        self.device = torch.device(device)
        self.parents = torch.as_tensor(
            np.asarray(model.body_parentid[1:], dtype=np.int64) - 1,
            dtype=torch.long,
            device=self.device,
        )
        self.body_pos = torch.as_tensor(
            np.asarray(model.body_pos[1:], dtype=np.float32), device=self.device
        )
        self.body_rot = _quat_wxyz_to_matrix_torch(torch.as_tensor(
            np.asarray(model.body_quat[1:], dtype=np.float32), device=self.device
        ))

        joint_ids = np.asarray(model.body_jntadr[2:], dtype=np.int64)
        if np.any(np.asarray(model.body_jntnum[2:]) != 1):
            raise ValueError("Torch FK requires exactly one hinge joint per actuated body")
        joint_pos = np.asarray(model.jnt_pos[joint_ids])
        if not np.allclose(joint_pos, 0.0, atol=1e-8):
            raise ValueError("Torch FK currently requires zero hinge-joint offsets")
        self.joint_axis = torch.as_tensor(
            np.asarray(model.jnt_axis[joint_ids], dtype=np.float32), device=self.device
        )

        self.geom_ids = _get_collision_geom_ids(model)
        geom_body = np.asarray(model.geom_bodyid[self.geom_ids], dtype=np.int64) - 1
        if np.any(geom_body < 0):
            raise ValueError("World geoms cannot be used for robot swept-volume FK")
        self.geom_body = torch.as_tensor(geom_body, dtype=torch.long, device=self.device)
        self.geom_pos = torch.as_tensor(
            np.asarray(model.geom_pos[self.geom_ids], dtype=np.float32), device=self.device
        )
        self.geom_rot = _quat_wxyz_to_matrix_torch(torch.as_tensor(
            np.asarray(model.geom_quat[self.geom_ids], dtype=np.float32), device=self.device
        ))
        self.geom_types = np.asarray(model.geom_type[self.geom_ids], dtype=np.int32)
        self.geom_sizes = np.asarray(model.geom_size[self.geom_ids], dtype=np.float64)
        self.geom_half_extents = np.stack([
            _geom_local_half_extents(model, gid) for gid in self.geom_ids
        ])
        self.left_foot = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              "left_ankle_roll_link") - 1
        )
        self.right_foot = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                              "right_ankle_roll_link") - 1
        )

    def forward(self, root_trans, root_rot_xyzw, dof, mob_frame_ids):
        if self.device.type == "cpu":
            # The outer data pipeline already parallelizes by process.
            torch.set_num_threads(1)
        root_trans_t = torch.as_tensor(root_trans, dtype=torch.float32, device=self.device)
        root_quat_t = torch.as_tensor(
            root_rot_xyzw[:, [3, 0, 1, 2]], dtype=torch.float32, device=self.device
        )
        dof_t = torch.as_tensor(dof, dtype=torch.float32, device=self.device)
        if dof_t.ndim != 2 or dof_t.shape[-1] != NUM_DOF:
            raise ValueError(f"Expected batched {NUM_DOF}-DoF input, got {tuple(dof_t.shape)}")

        joint_rot = _axis_angle_to_matrix_torch(self.joint_axis, dof_t)
        world_pos = [root_trans_t]
        world_rot = [_quat_wxyz_to_matrix_torch(root_quat_t)]
        for body_idx in range(1, NUM_BODIES):
            parent_idx = int(self.parents[body_idx])
            parent_pos = world_pos[parent_idx]
            parent_rot = world_rot[parent_idx]
            offset = torch.matmul(
                parent_rot, self.body_pos[body_idx].reshape(1, 3, 1)
            ).squeeze(-1)
            local_rot = torch.matmul(self.body_rot[body_idx], joint_rot[:, body_idx - 1])
            world_pos.append(parent_pos + offset)
            world_rot.append(torch.matmul(parent_rot, local_rot))

        body_pos = torch.stack(world_pos, dim=1)
        body_rot = torch.stack(world_rot, dim=1)
        foot_pos = body_pos[:, [self.left_foot, self.right_foot]]

        frame_ids_t = torch.as_tensor(mob_frame_ids, dtype=torch.long, device=self.device)
        selected_pos = body_pos.index_select(0, frame_ids_t)[:, self.geom_body]
        selected_rot = body_rot.index_select(0, frame_ids_t)[:, self.geom_body]
        geom_pos = selected_pos + torch.matmul(
            selected_rot, self.geom_pos.reshape(1, -1, 3, 1)
        ).squeeze(-1)
        geom_rot = torch.matmul(selected_rot, self.geom_rot.unsqueeze(0))
        return (
            foot_pos.cpu().numpy().astype(np.float64),
            geom_pos.cpu().numpy().astype(np.float64),
            geom_rot.cpu().numpy().astype(np.float64),
        )


def _get_torch_fk_model(xml_path: str, device: str):
    key = (str(Path(xml_path).resolve()), device)
    if key not in _TORCH_FK_MODELS:
        model, _ = _get_mj_model(xml_path)
        _TORCH_FK_MODELS[key] = _TorchBatchKinematics(model, device)
    return _TORCH_FK_MODELS[key]


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


def _mark_geom_pose(occu, llb, unit, gtype, size, center, rot, lo, hi):
    """Rasterize one positioned primitive geom into ``occu`` in-place."""
    import mujoco

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

    local = (grid - center) @ rot

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


def _mark_geom(occu, llb, unit, model, data, gid, cached_aabb=None):
    """Rasterize one MuJoCo geom into the voxel grid occu (mutated in-place)."""
    lo, hi = cached_aabb if cached_aabb is not None else _geom_world_aabb(model, data, gid)
    _mark_geom_pose(
        occu,
        llb,
        unit,
        int(model.geom_type[gid]),
        model.geom_size[gid],
        np.asarray(data.geom_xpos[gid], dtype=np.float64),
        np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3),
        lo,
        hi,
    )


def _rasterize_geoms_exact_vectorized(
    occu: np.ndarray,
    llb: np.ndarray,
    unit: float,
    model,
    geom_ids: list[int],
    geom_pos: np.ndarray,
    geom_rot: np.ndarray,
    geom_mins: np.ndarray,
    geom_maxs: np.ndarray,
) -> None:
    """Rasterize sampled primitive poses using the scalar algorithm in batches.

    Poses with the same primitive type and candidate AABB dimensions share one
    voxel-offset grid. The containment equations and voxel-center convention
    are identical to :func:`_mark_geom_pose`.
    """
    import mujoco

    grid_shape = np.asarray(occu.shape, dtype=np.int64)
    vmin = np.floor((geom_mins - llb) / unit).astype(np.int64)
    vmax = np.ceil((geom_maxs - llb) / unit).astype(np.int64)
    vmin = np.clip(vmin, 0, grid_shape)
    vmax = np.clip(vmax, 0, grid_shape)

    n_frames, n_geoms = geom_pos.shape[:2]
    starts = vmin.reshape(-1, 3)
    dims = (vmax - vmin).reshape(-1, 3)
    centers = geom_pos.reshape(-1, 3)
    rotations = geom_rot.reshape(-1, 3, 3)
    geom_types = np.broadcast_to(
        np.asarray(model.geom_type[geom_ids], dtype=np.int32),
        (n_frames, n_geoms),
    ).reshape(-1)
    geom_sizes = np.broadcast_to(
        np.asarray(model.geom_size[geom_ids], dtype=np.float64),
        (n_frames, n_geoms, 3),
    ).reshape(-1, 3)

    valid = np.all(dims > 0, axis=1)
    if not np.any(valid):
        return

    # Grouping by AABB dimensions keeps temporary [pose, candidate, xyz]
    # arrays compact and reuses the same integer offset grid for every pose.
    keys = np.column_stack((geom_types, dims))
    for key in np.unique(keys[valid], axis=0):
        gtype = int(key[0])
        shape = key[1:].astype(np.int64)
        group = valid & np.all(keys == key, axis=1)
        group_ids = np.flatnonzero(group)

        offsets = np.stack(
            np.meshgrid(
                np.arange(shape[0]),
                np.arange(shape[1]),
                np.arange(shape[2]),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        # Cap temporary arrays for unusually long clips or large rotated AABBs.
        chunk_size = max(1, 250_000 // len(offsets))
        for chunk_start in range(0, len(group_ids), chunk_size):
            pose_ids = group_ids[chunk_start:chunk_start + chunk_size]
            indices = starts[pose_ids, None, :] + offsets[None, :, :]
            world = llb + (indices.astype(np.float64) + 0.5) * unit
            local = np.matmul(
                world - centers[pose_ids, None, :], rotations[pose_ids]
            )
            size = geom_sizes[pose_ids]

            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                mask = np.sum(local * local, axis=-1) <= size[:, None, 0] ** 2
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = size[:, None, 0]
                half_length = size[:, None, 1]
                clipped_z = np.clip(local[..., 2], -half_length, half_length)
                mask = (
                    local[..., 0] ** 2
                    + local[..., 1] ** 2
                    + (local[..., 2] - clipped_z) ** 2
                    <= radius ** 2
                )
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                mask = (
                    local[..., 0] ** 2 + local[..., 1] ** 2
                    <= size[:, None, 0] ** 2
                ) & (np.abs(local[..., 2]) <= size[:, None, 1])
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                mask = np.all(np.abs(local) <= size[:, None, :], axis=-1)
            else:
                continue

            occupied = indices[mask]
            if occupied.size:
                occu[occupied[:, 0], occupied[:, 1], occupied[:, 2]] = True


def _contact_and_sliding_masks_from_foot_positions(
    foot_pos: np.ndarray,
    fps: float,
    height_thresh: float = 0.05,
    vel_thresh: float = 0.15,
    pelvis_z: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify planted and sliding feet using Bones-SEED world kinematics.

    When *pelvis_z* is provided, frames where the pelvis is below 0.35 m are
    treated as crawl / prone locomotion.  In that regime foot movement is
    intentional propulsion and is NOT flagged as sliding — only foot-ground
    separation matters.

    .. attention::
       Crawl enforcement currently relies on ``body_trans_loss`` to keep feet
       near the ground; there is **no** dedicated height-constraint loss for
       crawl frames.  A future iteration should add a per-frame foot-height
       penalty on ``sliding_mask`` frames whose ``pelvis_z < 0.35``, and
       optionally extend contact detection to knees / hands.
    """
    if foot_pos.ndim != 3 or foot_pos.shape[1:] != (2, 3):
        raise ValueError(f"Expected foot_pos shape (T, 2, 3), got {foot_pos.shape}")
    if fps <= 0:
        raise ValueError(f"Expected positive fps, got {fps}")

    foot_speed = np.zeros((foot_pos.shape[0], 2), dtype=np.float64)
    if foot_pos.shape[0] > 1:
        foot_speed[1:] = np.linalg.norm(np.diff(foot_pos, axis=0), axis=-1) * fps

    foot_is_low = foot_pos[:, :, 2] < height_thresh

    # ── crawl / prone guard ──
    # Foot motion during crawl is deliberate body propulsion, not a foot-slide
    # artefact.  Suppress the sliding label so those frames are not penalised
    # by the velocity→0 loss; they still receive body_trans_loss gravity.
    if pelvis_z is not None:
        is_crawl = np.asarray(pelvis_z, dtype=np.float64) < 0.35       # (T,)
        # Broadcast to (T, 2) so the mask applies to both feet.
        is_crawl = np.broadcast_to(is_crawl[:, np.newaxis], foot_pos.shape[:2])
    else:
        is_crawl = np.zeros(foot_pos.shape[:2], dtype=bool)

    contact = (foot_is_low & (foot_speed < vel_thresh)).astype(np.float32)
    # Crawl frames: low + fast = intentional, NOT sliding.
    sliding = (
        foot_is_low
        & (foot_speed >= vel_thresh)
        & ~is_crawl
    ).astype(np.float32)
    return contact, sliding


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
    fk_backend: str = "torch",
    torch_device: str = "cpu",
    mob_raster_backend: str = "vectorized",
    verbose: bool = False,
) -> dict:
    """Compute contact masks and optional MOB occupancy from G1 kinematics.

    The default PyTorch backend batches FK across the full sequence and only
    materializes collision geometry at MOB-sampled frames. The MuJoCo backend
    is retained as a parity/reference implementation.

    Args:
        root_trans_offset:  (T, 3) pelvis world translation, meters.
        root_rot_xyzw:      (T, 4) pelvis rotation, xyzw quaternion (scipy).
        dof_mj:             (T, 29) joint angles, radians, MuJoCo/MJCF order.
        fps:                frame rate used to convert displacement to m/s.
        xml_path:           path to g1_29dof_with_collision.xml.
        height_thresh:      ankle height threshold in meters (default 0.05).
        vel_thresh:         full 3D ankle speed threshold in m/s (default 0.15).
        mob:                if True, also compute MOB swept occupancy.
        mob_unit:           voxel resolution for MOB (default 0.08 m).
        mob_margin:         padding around swept volume (default 0.16 m).
        mob_frame_stride:   subsample frames for MOB (default 1 = all frames).
        fk_backend:         ``torch`` (batched) or ``mujoco`` (reference).
        torch_device:       device used by batched FK, normally ``cpu`` when
                            the outer pipeline uses multiple worker processes.
        mob_raster_backend: ``vectorized`` exact batched rasterizer or
                            ``scalar`` reference implementation.

    Returns:
        dict with:
            contact_mask: (T, 2) float32. Low and stationary feet.
            sliding_mask: (T, 2) float32. Low feet moving > *vel_thresh*
                in world-frame m/s, **except** during crawl (pelvis < 0.35 m)
                where foot motion is intentional propulsion rather than
                a sliding artefact.  Crawl frames are excluded from the
                sliding mask but currently receive no dedicated foot-height
                constraint — see the ``.. attention`` note in
                :func:`_contact_and_sliding_masks_from_foot_positions`.
            If mob=True, also:
                scene: dict with:
                    occu_global:   bool ndarray [X, Y, Z], 1 = occupied (robot swept
                                   through this voxel).  Strafe complement of swept volume.
                    unit:          float, voxel size in meters.
                    llb:           ndarray [3], float32, lower-left-back corner.
    """
    import mujoco

    T = root_trans_offset.shape[0]
    if mob_frame_stride < 1:
        raise ValueError(f"mob_frame_stride must be >= 1, got {mob_frame_stride}")
    if fk_backend not in {"torch", "mujoco"}:
        raise ValueError(f"Unsupported FK backend {fk_backend!r}")
    if mob_raster_backend not in {"vectorized", "scalar"}:
        raise ValueError(f"Unsupported MOB raster backend {mob_raster_backend!r}")

    model, data = _get_mj_model(xml_path)
    mob_geom_ids = _get_collision_geom_ids(model) if mob else []
    frame_ids = np.arange(0, T, mob_frame_stride, dtype=np.int64) if mob else np.empty(0, dtype=np.int64)

    if fk_backend == "torch":
        torch_fk = _get_torch_fk_model(xml_path, torch_device)
        foot_pos, geom_pos, geom_rot = torch_fk.forward(
            root_trans_offset, root_rot_xyzw, dof_mj, frame_ids
        )
        if mob:
            half_world = np.einsum(
                "tgij,gj->tgi", np.abs(geom_rot), torch_fk.geom_half_extents
            )
            mob_mins = geom_pos - half_world
            mob_maxs = geom_pos + half_world
    else:
        left_body_id, right_body_id = _find_foot_body_ids(model)

        # MuJoCo freejoint quaternion is wxyz; input is xyzw.
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]].astype(np.float64)
        foot_pos = np.zeros((T, 2, 3), dtype=np.float64)
        geom_pos = np.empty((len(frame_ids), len(mob_geom_ids), 3), dtype=np.float64)
        geom_rot = np.empty((len(frame_ids), len(mob_geom_ids), 3, 3), dtype=np.float64)
        sampled_lookup = {int(frame): idx for idx, frame in enumerate(frame_ids)}
        frame_iter = tqdm(range(T), desc="FK+contact", unit="f", leave=False) if verbose else range(T)
        for t in frame_iter:
            data.qpos[0:3] = root_trans_offset[t]
            data.qpos[3:7] = root_rot_wxyz[t]
            data.qpos[7:MUJOCO_QPOS_END] = dof_mj[t]
            mujoco.mj_forward(model, data)
            foot_pos[t, 0] = data.xpos[left_body_id]
            foot_pos[t, 1] = data.xpos[right_body_id]
            sampled_idx = sampled_lookup.get(t)
            if sampled_idx is not None:
                geom_pos[sampled_idx] = data.geom_xpos[mob_geom_ids]
                geom_rot[sampled_idx] = data.geom_xmat[mob_geom_ids].reshape(-1, 3, 3)
        if mob:
            local_half = np.stack([_geom_local_half_extents(model, gid) for gid in mob_geom_ids])
            half_world = np.einsum("tgij,gj->tgi", np.abs(geom_rot), local_half)
            mob_mins = geom_pos - half_world
            mob_maxs = geom_pos + half_world

    contact, sliding = _contact_and_sliding_masks_from_foot_positions(
        foot_pos,
        fps=fps,
        height_thresh=height_thresh,
        vel_thresh=vel_thresh,
        pelvis_z=root_trans_offset[:, 2],
    )

    result: dict = {"contact_mask": contact, "sliding_mask": sliding}

    # ── MOB: build swept volume from cached AABBs ──
    if mob:
        llb = mob_mins.min(axis=(0, 1)) - mob_margin
        rub = mob_maxs.max(axis=(0, 1)) + mob_margin
        shape = np.ceil((rub - llb) / mob_unit).astype(np.int64) + 1
        swept_occu = np.zeros(tuple(shape.tolist()), dtype=bool)

        if mob_raster_backend == "vectorized":
            _rasterize_geoms_exact_vectorized(
                swept_occu,
                llb,
                mob_unit,
                model,
                mob_geom_ids,
                geom_pos,
                geom_rot,
                mob_mins,
                mob_maxs,
            )
        else:
            voxel_indices = range(len(frame_ids))
            voxel_iter = (
                tqdm(voxel_indices, desc="MOB rasterize", unit="f", leave=False)
                if verbose else voxel_indices
            )
            for sampled_idx in voxel_iter:
                for geom_idx, gid in enumerate(mob_geom_ids):
                    _mark_geom_pose(
                        swept_occu,
                        llb,
                        mob_unit,
                        int(model.geom_type[gid]),
                        model.geom_size[gid],
                        geom_pos[sampled_idx, geom_idx],
                        geom_rot[sampled_idx, geom_idx],
                        mob_mins[sampled_idx, geom_idx],
                        mob_maxs[sampled_idx, geom_idx],
                    )

        # ~swept_occu → where the robot never went → assumed obstacle
        # True = occupied (obstacle), False = free (robot passed through here)
        occu_global = ~swept_occu

        # Collision geoms undershoot the real body volume, leaving the free
        # corridor along the root trajectory only ~1 voxel (0.08 m) wide.
        # Erode the occupied set to widen it; the MJCF geoms stay untouched
        # so simulator physics are unchanged.
        occu_global = erode_voxel_26(occu_global)

        result["scene"] = {
            "occu_global": occu_global,
            "unit": mob_unit,
            "llb": llb.astype(np.float32),
        }

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
        "dof_order": entry.get("dof_order", "mujoco"),
        "dof_names": entry.get(
            "dof_names", list(MUJOCO_DOF_JOINT_NAMES)
        ),
        "root_rot": quat_tgt,
        "smpl_joints": smpl_tgt,
        "fps": fps_target,
    }


def process_session_csvs(args_tuple):
    """Process all CSVs in a single session directory. Used by multiprocessing."""
    session_dir, session_name, out_dir, fps, fps_source, mjcf_dir, mob_cfg = args_tuple
    import time
    import warnings

    warnings.filterwarnings("ignore")

    csv_files = sorted([f for f in os.listdir(session_dir) if f.endswith(".csv")])

    session_out = os.path.join(out_dir, session_name)
    os.makedirs(session_out, exist_ok=True)

    xml_path = os.path.join(mjcf_dir, "g1_29dof_with_collision.xml")

    converted_num = 0
    failed = 0
    t_csv = t_convert = t_resample = t_fk = t_dump = 0.0
    for csv_f in csv_files:
        name = os.path.splitext(csv_f)[0]
        out_path = os.path.join(session_out, name + ".pkl")
        if os.path.exists(out_path):
            converted_num += 1  # skip existing
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
                fk_backend=mob_cfg.get("fk_backend", "torch"),
                torch_device=mob_cfg.get("torch_device", "cpu"),
                mob_raster_backend=mob_cfg.get("raster_backend", "vectorized"),
            )
            entry["contact_mask"] = fk_result["contact_mask"]
            entry["sliding_mask"] = fk_result["sliding_mask"]
            if mob_cfg.get("enabled", False):
                entry["scene"] = fk_result["scene"]
            t_fk += time.time() - t0

            t0 = time.time()
            joblib.dump({name: entry}, out_path, compress=True)
            t_dump += time.time() - t0
            converted_num += 1
        except Exception:  # noqa: BLE001
            if failed == 0:  # print first error per session
                import traceback
                traceback_text = traceback.format_exc().strip().replace(
                    chr(10), chr(10) + "  "
                )
                print(
                    f"\n  [{session_name}] ERROR on {csv_f}:\n  {traceback_text}",
                    file=sys.stderr,
                )
            failed += 1

    if converted_num > 0:
        elapsed_csv = t_csv / converted_num * 1000
        elapsed_convert = t_convert / converted_num * 1000
        elapsed_resample = t_resample / converted_num * 1000
        elapsed_fk = t_fk / converted_num * 1000
        elapsed_dump = t_dump / converted_num * 1000
        fk_label = "fk+mob" if mob_cfg.get("enabled") else "fk+contact"
        print(f"  {session_name}: {converted_num}/{len(csv_files)} converted"
              f" (per-csv avg: csv {elapsed_csv:.0f}ms  convert {elapsed_convert:.0f}ms  "
              f"resample {elapsed_resample:.0f}ms  {fk_label} {elapsed_fk:.0f}ms  "
              f"dump {elapsed_dump:.0f}ms)"
              + (f"  {failed} failed" if failed else ""))
    return session_name, converted_num, failed, len(csv_files)


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
        "(1=occupied/obstacle complement of swept robot volume, requires MuJoCo)",
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
    parser.add_argument(
        "--fk_backend",
        choices=("torch", "mujoco"),
        default="torch",
        help="FK implementation for feet and collision geoms (default: torch).",
    )
    parser.add_argument(
        "--torch_device",
        default="cpu",
        help="PyTorch FK device. Use cpu with multi-process conversion; cuda is "
        "intended for a single worker (default: cpu).",
    )
    parser.add_argument(
        "--mob_raster_backend",
        choices=("vectorized", "scalar"),
        default="vectorized",
        help="Exact MOB rasterizer implementation. Vectorized preserves scalar "
        "voxel semantics while batching primitive tests (default: vectorized).",
    )
    args = parser.parse_args()

    if args.torch_device.startswith("cuda") and args.num_workers != 1:
        parser.error("--torch_device=cuda requires --num_workers=1")

    mob_cfg = {
        "enabled": args.mob,
        "unit": args.mob_unit,
        "margin": args.mob_margin,
        "frame_stride": args.mob_frame_stride,
        "fk_backend": args.fk_backend,
        "torch_device": args.torch_device,
        "raster_backend": args.mob_raster_backend,
    }

    print(f"G1 {NUM_DOF} DOFs, {NUM_BODIES} bodies; FK backend={args.fk_backend} ({args.torch_device})")
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
            for d in subdirs:
                subdir = os.path.join(args.input, d)
                if any(f.endswith(".csv") for f in os.listdir(subdir)):
                    session_dirs.append(
                        (subdir, d, args.output, args.fps, args.fps_source,
                         args.mjcf_dir, mob_cfg)
                    )
        elif has_csvs:
            session_name = os.path.basename(args.input.rstrip("/"))
            session_dirs.append(
                (args.input, session_name, args.output, args.fps, args.fps_source,
                 args.mjcf_dir, mob_cfg)
            )

        print(f"\nBatch converting {len(session_dirs)} sessions with {args.num_workers} workers")
        print(f"Output: {args.output}")
        os.makedirs(args.output, exist_ok=True)

        import multiprocessing

        total_converted_num = 0
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
            for session_name, converted_num, failed, n_csvs in pbar:
                total_converted_num += converted_num
                total_failed += failed
                total_csvs += n_csvs
                pbar.set_postfix_str(
                    f"{total_converted_num}/{total_csvs} ok" + (f" {total_failed} fail" if total_failed else "")
                )

        print(
            f"\nDone: {total_converted_num} motions converted, {total_failed} failed, {total_csvs} total CSVs"
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
            fk_backend=mob_cfg["fk_backend"],
            torch_device=mob_cfg["torch_device"],
            mob_raster_backend=mob_cfg["raster_backend"],
            verbose=True,
        )
        entry["contact_mask"] = fk_result["contact_mask"]
        entry["sliding_mask"] = fk_result["sliding_mask"]
        if mob_cfg["enabled"]:
            entry["scene"] = fk_result["scene"]
        motion_lib_dict[name] = entry

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"\nSaving motion_lib PKL: {args.output}")
    joblib.dump(motion_lib_dict, args.output, compress=True)
    print(f"Done: {len(motion_lib_dict)} sequences saved")


if __name__ == "__main__":
    main()

"""Conversions between TextOp controller state, DAR features, and G1 motion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ETree

from robotmdar.utils.goal import (
    GoalClamp,
    GoalEncoding,
    GoalType,
    build_ego_goal,
    build_ego_joint_state_goal_v6,
    quaternion_yaw,
)
import robotmdar.dtype.motion as motion_dtype
from robotmdar.skeleton.end_effector import (
    extract_end_effector_positions,
    resolve_end_effector_anchors,
)
from robotmdar.dtype.motion import (
    G1_23DOF_FROM_29DOF_INDICES,
    G1_MUJOCO_DOF_JOINT_NAMES,
    G1_WRIST_DOF_INDICES,
    motion_dict_to_feature_v3,
    motion_dict_to_feature_v6,
    motion_feature_dim_for_dof,
    quaternion_to_euler_angles,
)
from robotmdar.dtype.rotation import (
    euler_angles_to_quaternion,
    matrix_to_quaternion,
    quat_apply,
    quaternion_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


G1_ISAACLAB_DOF_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

# Derive boundary permutations by semantic name so an order change cannot
# silently pair the wrong joints.
if set(G1_ISAACLAB_DOF_JOINT_NAMES) != set(G1_MUJOCO_DOF_JOINT_NAMES):
    raise RuntimeError("IsaacLab and MuJoCo G1 joint-name sets differ")
_ISAACLAB_TO_MUJOCO = np.asarray([
    G1_ISAACLAB_DOF_JOINT_NAMES.index(name)
    for name in G1_MUJOCO_DOF_JOINT_NAMES
], dtype=np.int64)
_MUJOCO_TO_ISAACLAB = np.asarray([
    G1_MUJOCO_DOF_JOINT_NAMES.index(name)
    for name in G1_ISAACLAB_DOF_JOINT_NAMES
], dtype=np.int64)
_G1_23DOF_FROM_29DOF = np.asarray(
    G1_23DOF_FROM_29DOF_INDICES, dtype=np.int64)
_WRIST_ISAACLAB_INDICES = _ISAACLAB_TO_MUJOCO[
    np.asarray(G1_WRIST_DOF_INDICES, dtype=np.int64)
]


def isaaclab_to_mujoco_dof(values: np.ndarray) -> np.ndarray:
    """Convert 29-DoF values from IsaacLab order to MuJoCo order."""
    values = np.asarray(values)
    if values.shape[-1] != 29:
        raise ValueError(f"Expected 29 IsaacLab DoFs, got {values.shape}")
    return np.ascontiguousarray(values[..., _ISAACLAB_TO_MUJOCO])


def _reduce_mujoco_29_to_23(values: np.ndarray) -> np.ndarray:
    """Drop the six locked wrist DoFs from MuJoCo-ordered values."""
    values = np.asarray(values)
    if values.shape[-1] != 29:
        raise ValueError(f"Expected 29 MuJoCo DoFs, got {values.shape}")
    return np.ascontiguousarray(values[..., _G1_23DOF_FROM_29DOF])


def _expand_mujoco_23_to_29(values: np.ndarray) -> np.ndarray:
    """Insert zero-valued wrist DoFs into MuJoCo-ordered values."""
    values = np.asarray(values)
    if values.shape[-1] != 23:
        raise ValueError(f"Expected 23 MuJoCo DoFs, got {values.shape}")
    expanded = np.zeros(values.shape[:-1] + (29,), dtype=values.dtype)
    expanded[..., _G1_23DOF_FROM_29DOF] = values
    return expanded


def mujoco_to_isaaclab_dof(values: np.ndarray) -> np.ndarray:
    """Convert 23- or 29-DoF MuJoCo values to IsaacLab 29-DoF order."""
    values = np.asarray(values)
    if values.shape[-1] == 23:
        values = _expand_mujoco_23_to_29(values)
    elif values.shape[-1] != 29:
        raise ValueError(f"Expected 23 or 29 MuJoCo DoFs, got {values.shape}")
    return np.ascontiguousarray(values[..., _MUJOCO_TO_ISAACLAB])


def _normalized_quaternions_xyzw(values: np.ndarray) -> np.ndarray:
    # ZMQ decoding uses np.frombuffer(), whose views are read-only. Own the
    # storage before passing these arrays to Torch.
    values = np.array(values, dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[-1] != 4:
        raise ValueError(f"Expected quaternion history [n, 4], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Root quaternion history contains non-finite values")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("Root quaternion history contains a zero quaternion")
    return np.ascontiguousarray(values / norms)


def _normalized_wire_quaternions_wxyz_as_xyzw(values: np.ndarray) -> np.ndarray:
    """Normalize SONIC-wire quaternions and convert them to TextOp's xyzw."""
    values = np.array(values, dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[-1] != 4:
        raise ValueError(
            f"Expected wire quaternion history [n, 4], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Wire quaternion history contains non-finite values")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-6):
        raise ValueError("Wire quaternion history contains a zero quaternion")
    normalized_wxyz = values / norms
    return np.ascontiguousarray(normalized_wxyz[..., [1, 2, 3, 0]])


def _joint_velocity_limit(max_velocity_rad_s: Any,
                          dof_dim: int) -> np.ndarray | None:
    if max_velocity_rad_s is None:
        return None
    limit = np.asarray(max_velocity_rad_s, dtype=np.float32)
    if limit.ndim == 0 or limit.size == 1:
        value = float(limit.reshape(-1)[0])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                "joint smoothing max_velocity_rad_s must be positive and finite, "
                f"got {value!r}")
        return np.full((dof_dim,), value, dtype=np.float32)
    limit = np.asarray(limit, dtype=np.float32).reshape(-1)
    if limit.shape != (dof_dim,):
        raise ValueError(
            "joint smoothing max_velocity_rad_s must be scalar or have "
            f"{dof_dim} entries, got {limit.shape}")
    if not np.isfinite(limit).all() or np.any(limit <= 0.0):
        raise ValueError(
            "joint smoothing max_velocity_rad_s contains non-positive or "
            "non-finite values")
    return np.ascontiguousarray(limit)


def smooth_joint_history(joints: np.ndarray,
                         *,
                         fps: float,
                         max_velocity_rad_s: Any = None,
                         ema_alpha: float = 1.0) -> np.ndarray:
    """Smooth controller joint history while preserving the latest state.

    The pass runs backward from the newest physical sample, so the planner seam
    still matches the measured robot while older history frames are pulled into
    a velocity-limited tube around it.
    """
    joints = np.array(joints, dtype=np.float32, copy=True)
    if joints.ndim != 2:
        raise ValueError(f"Expected joint history [n, dof], got {joints.shape}")
    if not np.isfinite(joints).all():
        raise ValueError("Joint history contains non-finite values")

    alpha = float(ema_alpha)
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(
            f"joint smoothing ema_alpha must be in (0, 1], got {ema_alpha}")
    velocity_limit = _joint_velocity_limit(max_velocity_rad_s, joints.shape[-1])
    if velocity_limit is None and alpha >= 1.0:
        return np.ascontiguousarray(joints)
    if fps <= 0:
        raise ValueError(f"Joint smoothing fps must be positive, got {fps}")

    smoothed = joints
    if alpha < 1.0 and len(smoothed) > 1:
        filtered = smoothed.copy()
        for idx in range(len(filtered) - 2, -1, -1):
            filtered[idx] = alpha * smoothed[idx] + (
                1.0 - alpha) * filtered[idx + 1]
        smoothed = filtered

    if velocity_limit is not None and len(smoothed) > 1:
        max_delta = velocity_limit / float(fps)
        for idx in range(len(smoothed) - 2, -1, -1):
            delta = smoothed[idx] - smoothed[idx + 1]
            smoothed[idx] = smoothed[idx + 1] + np.clip(
                delta, -max_delta, max_delta)

    return np.ascontiguousarray(smoothed)


def state_to_model_input(state_msg: Any, history_len: int, val_data: Any,
                         device: str | torch.device, *,
                         fps: float | None = None,
                         joint_smoothing_max_velocity_rad_s: Any = None,
                         joint_smoothing_ema_alpha: float = 1.0):
    """Build normalized DAR history from the latest physical controller states.

    FeatureVersion 6 is arrival-aligned: H model history features require
    H + 1 physical states and end at the latest measured state.  FeatureVersion
    3 keeps the legacy terminal extrapolation because its last feature stores
    a forward delta from the latest measured pose.
    """
    if history_len < 2:
        raise ValueError(
            f"Controller history requires at least 2 features, got {history_len}")
    required_states = (
        history_len + 1 if motion_dtype.FeatureVersion == 6 else history_len
    )
    positions = np.array(
        state_msg.states.g1_pos, dtype=np.float32, copy=True)
    rotations = _normalized_wire_quaternions_wxyz_as_xyzw(
        state_msg.states.g1_root_rot)
    joints = np.array(
        state_msg.states.g1_joint_pos, dtype=np.float32, copy=True)

    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"Expected position history [n, 3], got {positions.shape}")
    if joints.ndim != 2 or joints.shape[-1] != 29:
        raise ValueError(f"Expected joint history [n, 29], got {joints.shape}")
    if not (len(positions) == len(rotations) == len(joints)):
        raise ValueError("Controller state history arrays have different lengths")
    if len(positions) < required_states:
        raise ValueError(
            f"Need {required_states} physical states for {history_len} features, "
            f"got {len(positions)}")

    positions = positions[-required_states:]
    rotations = rotations[-required_states:]
    joints = joints[-required_states:]
    joint_smoothing_ema_alpha = float(joint_smoothing_ema_alpha)
    if (joint_smoothing_max_velocity_rad_s is not None
            or joint_smoothing_ema_alpha != 1.0):
        resolved_fps = float(
            getattr(val_data, "fps", 0.0) if fps is None else fps)
        joints = smooth_joint_history(
            joints,
            fps=resolved_fps,
            max_velocity_rad_s=joint_smoothing_max_velocity_rad_s,
            ema_alpha=joint_smoothing_ema_alpha,
        )
    joints_mujoco = isaaclab_to_mujoco_dof(joints)
    model_dof_dim = int(getattr(val_data, "dof_dim", 29))
    if model_dof_dim == 23:
        joints_mujoco = _reduce_mujoco_29_to_23(joints_mujoco)
    elif model_dof_dim != 29:
        raise ValueError(
            f"Planner supports model dof_dim 23 or 29, got {model_dof_dim}")

    if motion_dtype.FeatureVersion == 6:
        motion_dict = {
            "root_trans_offset": torch.as_tensor(
                positions, dtype=torch.float32, device=device).unsqueeze(0),
            "root_rot": torch.as_tensor(
                rotations, dtype=torch.float32, device=device).unsqueeze(0),
            "dof": torch.as_tensor(
                joints_mujoco, dtype=torch.float32, device=device).unsqueeze(0),
            "contact_mask": torch.ones(
                (1, history_len + 1, 2), dtype=torch.float32, device=device),
        }
        feature, abs_pose = motion_dtype.motion_dict_to_feature(motion_dict)
        expected_nfeats = motion_feature_dim_for_dof(
            model_dof_dim, feature_version=6
        )
        if feature.shape != (1, history_len, expected_nfeats):
            raise ValueError(
                f"Unexpected FeatureVersion 6 shape {tuple(feature.shape)}; "
                f"expected (1, {history_len}, {expected_nfeats})")
        return val_data.normalize(feature), abs_pose

    # The model history must end at the current physical state. Append a copy
    # only to satisfy the feature converter's N+1 input contract; terminal
    # forward deltas are replaced below with a constant-velocity estimate.
    positions_with_terminal = np.concatenate((positions, positions[-1:]), axis=0)
    rotations_with_terminal = np.concatenate((rotations, rotations[-1:]), axis=0)
    joints_with_terminal = np.concatenate(
        (joints_mujoco, joints_mujoco[-1:]), axis=0)

    motion_dict = {
        "root_trans_offset": torch.as_tensor(
            positions_with_terminal, dtype=torch.float32, device=device).unsqueeze(0),
        "root_rot": torch.as_tensor(
            rotations_with_terminal, dtype=torch.float32, device=device).unsqueeze(0),
        "dof": torch.as_tensor(
            joints_with_terminal, dtype=torch.float32, device=device).unsqueeze(0),
        "contact_mask": torch.ones(
            (1, history_len + 1, 2), dtype=torch.float32, device=device),
    }
    feature, abs_pose = motion_dict_to_feature_v3(motion_dict)
    expected_nfeats = motion_feature_dim_for_dof(
        model_dof_dim, feature_version=3
    )
    if feature.shape != (1, history_len, expected_nfeats):
        raise ValueError(
            f"Unexpected FeatureVersion 3 shape {tuple(feature.shape)}; "
            f"expected (1, {history_len}, {expected_nfeats})")

    # Estimate the unavailable current->next deltas from the most recent
    # physical interval. Roll/pitch and the pose fields already come directly
    # from the current state and must not be extrapolated.
    feature[:, -1, 4] = feature[:, -2, 4]
    current_yaw = quaternion_to_euler_angles(
        motion_dict["root_rot"][:, -2])[:, 2]
    current_yaw_quat = torch.stack((
        torch.zeros_like(current_yaw),
        torch.zeros_like(current_yaw),
        -torch.sin(current_yaw / 2),
        torch.cos(current_yaw / 2),
    ), dim=-1)
    feature[:, -1, 7:10] = quat_apply(
        current_yaw_quat,
        motion_dict["root_trans_offset"][:, -2]
        - motion_dict["root_trans_offset"][:, -3],
        w_last=True,
    )
    delta_dof_start = 11 + model_dof_dim
    feature[:, -1, delta_dof_start:delta_dof_start + model_dof_dim] = (
        motion_dict["dof"][:, -2] - motion_dict["dof"][:, -3])

    # motion_dict_to_feature_v3 subtracts Euler yaw directly. Wrap all yaw
    # deltas at the branch cut before applying training-set normalization.
    feature[..., 4] = torch.atan2(
        torch.sin(feature[..., 4]), torch.cos(feature[..., 4]))
    return val_data.normalize(feature), abs_pose


def state_to_ego_goal(state_msg: Any,
                      device: str | torch.device,
                      goal_type: GoalType | str = GoalType.ROOT,
                      goal_reference_path: str | Path | None = None,
                      goal_encoding: GoalEncoding | str | None = None,
                      goal_stats: dict | None = None,
                      goal_clamp: GoalClamp | None = None,
                      fps: float | None = None,
                      val_data: Any = None,
                      goal_include_log_d_hor: bool = True,
                      ) -> torch.Tensor:
    """Convert the root goal relative to the current history-feature pose."""
    reference_pos = torch.tensor(
        state_msg.states.g1_pos[-1], dtype=torch.float32,
        device=device).reshape(1, 3)
    reference_rot_np = _normalized_wire_quaternions_wxyz_as_xyzw(
        np.asarray(state_msg.states.g1_root_rot[-1:], dtype=np.float32))
    reference_rot = torch.as_tensor(
        reference_rot_np, dtype=torch.float32, device=device)
    return state_goal_from_reference(
        state_msg, reference_pos, reference_rot, device,
        goal_type=goal_type, goal_reference_path=goal_reference_path,
        goal_encoding=goal_encoding, goal_stats=goal_stats,
        goal_clamp=goal_clamp, fps=fps, val_data=val_data,
        goal_include_log_d_hor=goal_include_log_d_hor)


def _end_effectors_from_goal_state(
    goal_pos_world: torch.Tensor,
    world_goal_rot: torch.Tensor,
    world_goal_dof: torch.Tensor,
    *,
    val_data: Any,
    device: str | torch.device,
    fps: float | None,
) -> torch.Tensor:
    if val_data is None:
        raise ValueError(
            "split_end_effector planner goals require val_data so the "
            "active MJCF FK can resolve hand/foot anchors")
    goal_motion = {
        "root_trans_offset": goal_pos_world.reshape(1, 1, 3),
        "root_rot": world_goal_rot.reshape(1, 1, 4),
        "dof": world_goal_dof.reshape(1, 1, -1),
        "contact_mask": torch.ones(
            1, 1, 2, dtype=torch.float32, device=device),
    }
    fk_result = val_data.skeleton.forward_kinematics(
        goal_motion,
        fps=float(fps if fps is not None else getattr(val_data, "fps", 50.0)),
    )
    mjcf_file = str(val_data.skeleton.fk.mjcf_file)
    cache_key = getattr(val_data, "_end_effector_anchor_mjcf_file", None)
    anchors = getattr(val_data, "_end_effector_anchors_cache", None)
    if anchors is None or cache_key != mjcf_file:
        anchors = resolve_end_effector_anchors(val_data.skeleton)
        val_data._end_effector_anchors_cache = anchors
        val_data._end_effector_anchor_mjcf_file = mjcf_file
    return extract_end_effector_positions(
        fk_result, val_data.skeleton, anchors=anchors)[:, 0]


def _state_field(state_msg: Any, name: str):
    """Read a field from the latest protocol-11 nested StateMessage."""
    state_fields = {
        "g1_pos": state_msg.states.g1_pos,
        "g1_root_rot": state_msg.states.g1_root_rot,
        "g1_joint_pos": state_msg.states.g1_joint_pos,
    }
    if name in state_fields:
        return state_fields[name]

    if name == "timestamps_ns":
        return state_msg.history_meta.timestamps_ns
    if name == "publish_t_ns":
        return state_msg.history_meta.publish_t_ns
    if name == "tracked_plan_seq":
        return state_msg.tracking.seq
    if name == "tracked_plan_start_t_ns":
        return state_msg.tracking.start_t_ns
    if name == "text":
        return state_msg.condition.text
    if name == "ego_occ":
        return state_msg.condition.ego_occ
    validity = state_msg.condition.valid
    goal_validity = validity.goal
    validity_fields = {
        "scene_valid": validity.scene,
        "text_valid": validity.text,
        "goal_root_valid": goal_validity.root,
        "goal_yaw_valid": goal_validity.yaw,
        "goal_body_valid": goal_validity.body,
        "goal_orientation_valid": goal_validity.orientation,
        "goal_joint_valid": goal_validity.joint,
        "goal_velocity_valid": goal_validity.velocity,
        "goal_time_valid": goal_validity.time,
        "goal_end_effector_valid": goal_validity.end_effector.valid,
        "goal_end_effector_left_hand_valid": (
            goal_validity.end_effector.left_hand),
        "goal_end_effector_right_hand_valid": (
            goal_validity.end_effector.right_hand),
        "goal_end_effector_left_foot_valid": (
            goal_validity.end_effector.left_foot),
        "goal_end_effector_right_foot_valid": (
            goal_validity.end_effector.right_foot),
    }
    if name in validity_fields:
        return validity_fields[name]

    goal_fields = {
        "goal_type": state_msg.condition.goal.goal_type,
        "goal_root_pos_world": state_msg.condition.goal.root_pos_world,
        "goal_yaw_world": state_msg.condition.goal.yaw_world,
        "goal_root_velocity_world": (
            state_msg.condition.goal.root_velocity_world),
        "goal_timestamp_ns": state_msg.condition.goal.timestamp_ns,
        "goal_keypoints_world": state_msg.condition.goal.keypoints_world,
        "goal_root_rot_world": state_msg.condition.goal.root_rot_world,
        "goal_root_euler_world": state_msg.condition.goal.root_euler_world,
        "goal_dof_pos": state_msg.condition.goal.dof_pos,
        "goal_joint_pos": state_msg.condition.goal.dof_pos,
        "goal_end_effectors_world": (
            state_msg.condition.goal.end_effectors_world),
    }
    if name in goal_fields:
        return goal_fields[name]

    raise KeyError(f"Unknown protocol-11 state field: {name}")


def _state_bool_field(state_msg: Any, name: str, default: bool = True) -> bool:
    value = _state_field(state_msg, name)
    if value is None:
        return bool(default)
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{name} must be scalar-like, got {value.shape}")
        value = value.reshape(-1)[0]
    return bool(value)


def _reference_yaw(reference_rot: torch.Tensor) -> torch.Tensor:
    return quaternion_yaw(reference_rot.reshape(-1, 4)).reshape(
        reference_rot.shape[:-1])


@lru_cache(maxsize=8)
def _load_goal_keypoint_template(ref_path: str) -> np.ndarray:
    path = Path(ref_path)
    if not path.is_file():
        raise FileNotFoundError(f"Goal reference pose does not exist: {path}")
    with np.load(path, allow_pickle=False) as data:
        if 'keypoints' not in data:
            raise ValueError(f"Goal reference pose has no 'keypoints': {path}")
        keypoints = np.array(data['keypoints'], dtype=np.float32, copy=True)
    if keypoints.shape != (5, 3):
        raise ValueError(
            f"Goal reference keypoints must have shape (5, 3), got "
            f"{keypoints.shape} in {path}")
    if not np.isfinite(keypoints).all():
        raise ValueError(f"Goal reference pose contains non-finite values: {path}")
    if not np.allclose(keypoints[0, :2], 0.0, atol=1e-5):
        raise ValueError(
            f"Goal reference root XY must be at the origin, got "
            f"{keypoints[0, :2]} in {path}")
    return keypoints


def load_goal_keypoints_from_reference(
    ref_path: str | Path,
    goal_root_pos_world: np.ndarray,
    goal_yaw_world: float,
) -> np.ndarray:
    """Place an XY-origin, absolute-Z reference pose in the world frame."""
    goal_root_pos_world = np.asarray(goal_root_pos_world, dtype=np.float32)
    if goal_root_pos_world.shape != (3,):
        raise ValueError(
            "goal_root_pos_world must have shape (3,), got "
            f"{goal_root_pos_world.shape}")
    if (not np.isfinite(goal_root_pos_world).all()
            or not np.isfinite(goal_yaw_world)):
        raise ValueError("Goal root position and heading must be finite")

    keypoints = _load_goal_keypoint_template(str(Path(ref_path).resolve())).copy()
    if not np.isclose(goal_root_pos_world[2], keypoints[0, 2], atol=1e-4):
        raise ValueError(
            f"goal_root_pos_world.z ({goal_root_pos_world[2]:.4f}) "
            "does not match "
            f"reference root z ({keypoints[0, 2]:.4f})")

    c = np.cos(float(goal_yaw_world))
    s = np.sin(float(goal_yaw_world))
    rotation_xy = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    keypoints[:, :2] = keypoints[:, :2] @ rotation_xy.T
    keypoints[:, :2] += goal_root_pos_world[:2]
    return np.ascontiguousarray(keypoints)


def state_goal_from_reference(state_msg: Any,
                              reference_pos: torch.Tensor,
                              reference_rot: torch.Tensor,
                              device: str | torch.device,
                              goal_type: GoalType | str = GoalType.ROOT,
                              goal_reference_path: str | Path | None = None,
                              goal_encoding: GoalEncoding | str | None = None,
                              goal_stats: dict | None = None,
                              goal_clamp: GoalClamp | None = None,
                              fps: float | None = None,
                              val_data: Any = None,
                              goal_include_log_d_hor: bool = True,
                              ) -> torch.Tensor:
    """Convert the state goal relative to an explicit generated-history pose."""
    goal_type = GoalType.parse(goal_type)
    parsed_encoding = (
        GoalEncoding.parse(goal_encoding)
        if goal_encoding is not None else None
    )
    reference_pos = reference_pos.to(
        device=device, dtype=torch.float32).reshape(1, 3)
    reference_rot = reference_rot.to(
        device=device, dtype=torch.float32).reshape(1, 4)
    goal_keypoints_world = None

    goal_root_valid = _state_bool_field(
        state_msg, 'goal_root_valid', True)
    goal_yaw_valid = _state_bool_field(
        state_msg, 'goal_yaw_valid', True)
    goal_body_valid = _state_bool_field(
        state_msg, 'goal_body_valid', True)
    goal_orientation_valid = _state_bool_field(
        state_msg, 'goal_orientation_valid', True)
    goal_joint_valid = _state_bool_field(
        state_msg, 'goal_joint_valid', True)
    goal_velocity_valid = _state_bool_field(
        state_msg, 'goal_velocity_valid', True)
    goal_time_valid = _state_bool_field(
        state_msg, 'goal_time_valid', True)

    state_root_pos = _state_field(state_msg, 'goal_root_pos_world')
    state_yaw = _state_field(state_msg, 'goal_yaw_world')
    requires_root_field = goal_type in (
        GoalType.ROOT, GoalType.BODY_EXT, GoalType.JOINT_STATE)
    if state_root_pos is None:
        if requires_root_field and goal_root_valid:
            raise ValueError("TextOp state is missing its root goal")
        goal_pos_world = reference_pos
    else:
        goal_pos_world = torch.as_tensor(
            state_root_pos, dtype=torch.float32,
            device=device).reshape(1, 3)
        if not torch.isfinite(goal_pos_world).all():
            raise ValueError("goal_root_pos_world must be finite")

    if state_yaw is None:
        if goal_type in (GoalType.ROOT, GoalType.BODY_EXT) and goal_yaw_valid:
            raise ValueError("TextOp state is missing its yaw goal")
        goal_yaw_world = _reference_yaw(reference_rot).reshape(1)
    else:
        goal_yaw_world = torch.as_tensor(
            state_yaw, dtype=torch.float32, device=device).reshape(1)
        if not torch.isfinite(goal_yaw_world).all():
            raise ValueError("goal_yaw_world must be finite")
    goal_root_np = np.ascontiguousarray(
        goal_pos_world.detach().cpu().numpy().reshape(3))
    goal_yaw_scalar = float(
        goal_yaw_world.detach().cpu().numpy().reshape(-1)[0])

    if goal_type.uses_keypoints:
        state_keypoints_world = _state_field(state_msg, 'goal_keypoints_world')
        if state_keypoints_world is None:
            num_keypoints = 4 if goal_type is GoalType.BODY_EXT else 5
            if not goal_body_valid:
                state_keypoints_world = np.repeat(
                    goal_root_np.reshape(1, 3), num_keypoints, axis=0)
            elif goal_reference_path is None:
                raise ValueError(
                    f"{goal_type.value} goal requires controller "
                    "goal_keypoints_world or "
                    "goal_reference_path")
            else:
                state_keypoints_world = load_goal_keypoints_from_reference(
                    goal_reference_path,
                    goal_root_np,
                    goal_yaw_scalar,
                )
        state_keypoints_world = np.array(
            state_keypoints_world, dtype=np.float32, copy=True)
        if (goal_type is GoalType.BODY_EXT
                and state_keypoints_world.shape == (5, 3)):
            # Reference-pose files retain the legacy pelvis point; V4 does not.
            state_keypoints_world = state_keypoints_world[1:]
        num_keypoints = 4 if goal_type is GoalType.BODY_EXT else 5
        if state_keypoints_world.shape != (num_keypoints, 3):
            raise ValueError(
                f"{goal_type.value} goal_keypoints_world must have shape "
                f"({num_keypoints}, 3), got {state_keypoints_world.shape}")
        goal_keypoints_world = torch.as_tensor(
            state_keypoints_world, dtype=torch.float32,
            device=device).reshape(1, num_keypoints, 3)

    if goal_type is GoalType.BODY:
        goal_pos_world = goal_keypoints_world[:, 0]
        goal_yaw_world = torch.zeros(1, dtype=torch.float32, device=device)

    world_root_velocity = None
    timestep = None
    world_goal_rot = None
    world_goal_dof = None
    world_goal_end_effectors = None
    if goal_type.uses_arrival_time:
        state_velocity = _state_field(
            state_msg, 'goal_root_velocity_world')
        goal_timestamp_ns = _state_field(state_msg, 'goal_timestamp_ns')
        timestamps_ns = _state_field(state_msg, 'timestamps_ns')
        if state_velocity is None:
            if goal_velocity_valid:
                raise ValueError(
                    f"{goal_type.value} goal requires "
                    "goal_root_velocity_world [3]")
            state_velocity = np.zeros(3, dtype=np.float32)
        state_velocity = np.asarray(state_velocity, dtype=np.float32)
        if state_velocity.shape != (3,):
            raise ValueError(
                f"{goal_type.value} goal_root_velocity_world must have "
                "shape (3,), got "
                f"{state_velocity.shape}")
        if not np.isfinite(state_velocity).all():
            raise ValueError(
                f"{goal_type.value} goal_root_velocity_world must be finite")
        world_root_velocity = torch.as_tensor(
            state_velocity, dtype=torch.float32, device=device).reshape(1, 3)
        if goal_time_valid:
            if goal_timestamp_ns is None:
                raise ValueError(
                    f"{goal_type.value} goal requires goal_timestamp_ns")
            if timestamps_ns is None or len(timestamps_ns) == 0:
                raise ValueError(
                    f"{goal_type.value} goal requires controller timestamps_ns")
            remaining_seconds = max(
                0.0,
                (int(goal_timestamp_ns) - int(timestamps_ns[-1])) / 1e9,
            )
        else:
            remaining_seconds = 0.0
        timestep = torch.tensor(
            [[remaining_seconds]], dtype=torch.float32, device=device)

    if goal_type is GoalType.JOINT_STATE:
        state_goal_rot = _state_field(state_msg, 'goal_root_rot_world')
        state_goal_euler = _state_field(state_msg, 'goal_root_euler_world')
        state_goal_dof = _state_field(state_msg, 'goal_dof_pos')
        state_goal_rot_is_wire_wxyz = state_goal_rot is not None
        if state_goal_dof is None:
            state_goal_dof = _state_field(state_msg, 'goal_joint_pos')
        if state_goal_rot is None and state_goal_euler is None:
            if goal_orientation_valid:
                raise ValueError(
                    "joint_state goal requires goal_root_rot_world [4] wxyz "
                    "or goal_root_euler_world [3]")
            state_goal_rot = reference_rot.detach().cpu().numpy().reshape(4)
            state_goal_rot_is_wire_wxyz = False
        if state_goal_dof is None:
            if goal_joint_valid:
                raise ValueError("joint_state goal requires goal_dof_pos [29]")
            state_goal_dof = isaaclab_to_mujoco_dof(
                np.asarray(
                    state_msg.states.g1_joint_pos,
                    dtype=np.float32)[-1:].reshape(1, 29)
            )[0]
        if state_goal_rot is None:
            state_goal_euler = np.asarray(state_goal_euler, dtype=np.float32)
            if state_goal_euler.shape != (3,):
                raise ValueError(
                    "joint_state goal_root_euler_world must have shape "
                    f"(3,), got {state_goal_euler.shape}")
            if not np.isfinite(state_goal_euler).all():
                raise ValueError(
                    "joint_state goal_root_euler_world must be finite")
            state_goal_rot = euler_angles_to_quaternion(torch.as_tensor(
                state_goal_euler, dtype=torch.float32).reshape(1, 3)
            ).cpu().numpy()
        else:
            state_goal_rot = np.asarray(state_goal_rot, dtype=np.float32)
            if state_goal_rot.shape != (4,):
                raise ValueError(
                    "joint_state goal_root_rot_world must have shape "
                    f"(4,), got {state_goal_rot.shape}")
        if state_goal_rot_is_wire_wxyz:
            state_goal_rot = _normalized_wire_quaternions_wxyz_as_xyzw(
                state_goal_rot.reshape(1, 4))
        else:
            state_goal_rot = _normalized_quaternions_xyzw(
                state_goal_rot.reshape(1, 4))
        state_goal_dof = np.asarray(state_goal_dof, dtype=np.float32)
        if state_goal_dof.shape != (29,):
            raise ValueError(
                "joint_state goal_dof_pos must have shape (29,), got "
                f"{state_goal_dof.shape}")
        if not np.isfinite(state_goal_dof).all():
            raise ValueError("joint_state goal_dof_pos must be finite")
        world_goal_rot = torch.as_tensor(
            state_goal_rot, dtype=torch.float32, device=device)
        world_goal_dof = torch.as_tensor(
            state_goal_dof, dtype=torch.float32, device=device).reshape(1, 29)
        if parsed_encoding is GoalEncoding.SPLIT_END_EFFECTOR:
            state_goal_end_effectors = _state_field(
                state_msg, 'goal_end_effectors_world')
            if state_goal_end_effectors is not None:
                state_goal_end_effectors = np.asarray(
                    state_goal_end_effectors, dtype=np.float32)
                if state_goal_end_effectors.shape != (4, 3):
                    raise ValueError(
                        "joint_state goal_end_effectors_world must have "
                        f"shape (4, 3), got {state_goal_end_effectors.shape}")
                if not np.isfinite(state_goal_end_effectors).all():
                    raise ValueError(
                        "joint_state goal_end_effectors_world must be finite")
                world_goal_end_effectors = torch.as_tensor(
                    state_goal_end_effectors,
                    dtype=torch.float32,
                    device=device,
                ).reshape(1, 4, 3)
            else:
                # The payload may be omitted when the end-effector block is
                # invalid; FK keeps the 67-D goal finite before masking.
                world_goal_end_effectors = _end_effectors_from_goal_state(
                    goal_pos_world,
                    world_goal_rot,
                    world_goal_dof,
                    val_data=val_data,
                    device=device,
                    fps=fps,
                )

    if (goal_type is GoalType.JOINT_STATE
            and motion_dtype.FeatureVersion == 6
            and (parsed_encoding is None
                 or parsed_encoding is GoalEncoding.LEGACY40)):
        resolved_fps = float(50.0 if fps is None else fps)
        return build_ego_joint_state_goal_v6(
            world_goal_pos=goal_pos_world,
            world_goal_rot=world_goal_rot,
            world_goal_dof=world_goal_dof,
            world_root_velocity=world_root_velocity,
            reference_pos=reference_pos.to(device),
            reference_rot=reference_rot.to(device),
            time_to_arrival_seconds=timestep,
            fps=resolved_fps,
            goal_clamp=goal_clamp,
        )

    return build_ego_goal(
        goal_pos_world, goal_yaw_world, reference_pos.to(device),
        reference_rot.to(device), goal_type=goal_type,
        goal_encoding=parsed_encoding,
        goal_stats=goal_stats,
        world_goal_keypoints=goal_keypoints_world,
        world_root_velocity=world_root_velocity, timestep=timestep,
        time_to_arrival_seconds=timestep,
        world_goal_rot=world_goal_rot, world_goal_dof=world_goal_dof,
        world_goal_end_effectors=world_goal_end_effectors,
        fps=fps, goal_clamp=goal_clamp,
        goal_include_log_d_hor=goal_include_log_d_hor)


def _skew_matrix(vec: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vec[..., 0])
    x, y, z = vec.unbind(-1)
    return torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )


def _canonical_axis_perpendicular_to(vec: torch.Tensor) -> torch.Tensor:
    abs_vec = vec.abs()
    basis_index = abs_vec.argmin(dim=-1)
    basis = torch.zeros_like(vec)
    basis.scatter_(-1, basis_index.unsqueeze(-1), 1.0)
    return F.normalize(torch.cross(vec, basis, dim=-1), dim=-1, eps=1e-8)


def _rotation_matrix_between_vectors(source: torch.Tensor,
                                     target: torch.Tensor) -> torch.Tensor:
    """Return the shortest rotation matrix mapping source vectors to target."""
    source = F.normalize(source, dim=-1, eps=1e-8)
    target = F.normalize(target, dim=-1, eps=1e-8)
    batch_size = source.shape[0]
    eye = torch.eye(
        3, device=source.device, dtype=source.dtype
    ).unsqueeze(0).expand(batch_size, 3, 3).clone()

    axis = torch.cross(source, target, dim=-1)
    s = axis.norm(dim=-1, keepdim=True)
    c = (source * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    k = _skew_matrix(axis)
    factor = (1.0 - c).view(batch_size, 1, 1) / (
        s.square().view(batch_size, 1, 1).clamp(min=1e-16)
    )
    rodrigues = eye + k + torch.matmul(k, k) * factor

    fallback_axis = _canonical_axis_perpendicular_to(source)
    fallback = (
        2.0 * fallback_axis.unsqueeze(-1) * fallback_axis.unsqueeze(-2)
        - eye
    )
    parallel = s.squeeze(-1) < 1e-7
    antiparallel = parallel & (c.squeeze(-1) < 0.0)
    result = torch.where(parallel.view(batch_size, 1, 1), eye, rodrigues)
    return torch.where(antiparallel.view(batch_size, 1, 1), fallback, result)


@lru_cache(maxsize=8)
def _g1_joint_limits_from_mjcf_path(
        mjcf_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the active G1 hinge ranges from the MJCF, in MuJoCo DOF order."""
    ranges = {}
    tree = ETree.parse(mjcf_path)
    for joint in tree.getroot().iter("joint"):
        name = joint.attrib.get("name")
        range_text = joint.attrib.get("range")
        if name is None or range_text is None:
            continue
        values = np.fromstring(range_text, dtype=np.float32, sep=" ")
        if values.shape == (2,):
            ranges[name] = values

    missing = [
        name for name in G1_MUJOCO_DOF_JOINT_NAMES
        if name not in ranges
    ]
    if missing:
        raise ValueError(
            f"Active MJCF {mjcf_path} is missing joint ranges for {missing}")
    limits = np.stack(
        [ranges[name] for name in G1_MUJOCO_DOF_JOINT_NAMES], axis=0)
    if not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError(f"Invalid G1 joint ranges in active MJCF {mjcf_path}")
    return limits[:, 0].copy(), limits[:, 1].copy()


def g1_joint_limits_from_mjcf(
        val_data: Any,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lower/upper G1 joint limits from the dataset's active MJCF."""
    try:
        mjcf_path = str(val_data.skeleton.fk.mjcf_file)
    except AttributeError as exc:
        raise ValueError(
            "Residual re-anchoring requires val_data.skeleton.fk.mjcf_file "
            "to load G1 joint limits") from exc
    lower, upper = _g1_joint_limits_from_mjcf_path(mjcf_path)
    return (
        torch.as_tensor(lower, device=device, dtype=dtype),
        torch.as_tensor(upper, device=device, dtype=dtype),
    )


def _residual_reanchor_real_state(
        state_msg: Any,
        device: str | torch.device,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    real_pos = torch.as_tensor(
        np.asarray(state_msg.states.g1_pos[-1], dtype=np.float32),
        device=device,
        dtype=dtype,
    ).reshape(1, 3)
    real_rot_np = _normalized_wire_quaternions_wxyz_as_xyzw(
        np.asarray(state_msg.states.g1_root_rot[-1:], dtype=np.float32))
    real_rot = torch.as_tensor(
        real_rot_np, device=device, dtype=dtype).reshape(1, 4)
    real_joint_np = isaaclab_to_mujoco_dof(
        np.asarray(state_msg.states.g1_joint_pos[-1:], dtype=np.float32))
    real_joint = torch.as_tensor(
        real_joint_np, device=device, dtype=dtype).reshape(1, -1)
    return real_pos, real_rot, real_joint


def residual_reanchor_generated_history(
        abs_pose: dict,
        history_motion: torch.Tensor,
        state_msg: Any,
        val_data: Any,
        device: str | torch.device,
        joint_limits: tuple[torch.Tensor, torch.Tensor] | None = None,
        correction: dict[str, torch.Tensor] | None = None,
):
    """Re-integrate predicted local increments from a measured state anchor."""
    if history_motion is None or val_data is None:
        raise ValueError("Residual re-anchoring requires generated history data")

    device = torch.device(device)
    normalized_history = history_motion.to(device)
    if hasattr(val_data, "reconstruct_motion"):
        predicted = val_data.reconstruct_motion(
            normalized_history,
            abs_pose={
                key: value.to(device)
                for key, value in abs_pose.items()
            },
            ret_fk=False,
        )
    else:
        predicted = motion_dtype.motion_feature_to_dict(
            val_data.denormalize(normalized_history),
            {
                key: value.to(device)
                for key, value in abs_pose.items()
            },
        )

    pred_pos = predicted["root_trans_offset"].to(device)
    pred_rot = predicted["root_rot"].to(device)
    pred_joint = predicted["dof"].to(device)
    batch_size, num_frames = pred_pos.shape[:2]
    if batch_size != 1:
        raise ValueError(
            "Residual re-anchoring currently expects one measured G1 state, "
            f"but generated history has batch size {batch_size}")
    if num_frames <= 0:
        raise ValueError("Generated history must contain at least one feature")

    real_pos, real_rot, real_joint = _residual_reanchor_real_state(
        state_msg, device, pred_pos.dtype)
    if joint_limits is None:
        joint_limits = g1_joint_limits_from_mjcf(
            val_data, device=device, dtype=pred_joint.dtype)
    lower, upper = (
        limit.to(device=device, dtype=pred_joint.dtype).reshape(1, -1)
        for limit in joint_limits
    )
    if lower.shape[-1] != pred_joint.shape[-1]:
        raise ValueError(
            "G1 MJCF joint limit count does not match generated history: "
            f"{lower.shape[-1]} != {pred_joint.shape[-1]}")

    pred_anchor_pos = abs_pose["root_trans_offset"].to(
        device=device, dtype=pred_pos.dtype).reshape(batch_size, 3)
    pred_anchor_rot = abs_pose["root_rot"].to(
        device=device, dtype=pred_pos.dtype).reshape(batch_size, 4)
    pred_full_pos = torch.cat([pred_anchor_pos[:, None], pred_pos], dim=1)
    pred_full_rot = torch.cat([pred_anchor_rot[:, None], pred_rot], dim=1)
    pred_full_rot_matrix = quaternion_to_matrix(
        xyzw_to_wxyz(pred_full_rot))

    # V3 carries a joint delta for the first feature edge. V6 carries only
    # the current joint state, so its synthetic pre-history joint is unused.
    pred_full_joint = torch.cat([pred_joint[:, :1], pred_joint], dim=1)
    if motion_dtype.FeatureVersion == 3:
        denormalized_history = val_data.denormalize(normalized_history)
        dof_dim = pred_joint.shape[-1]
        first_joint_delta = denormalized_history[:, 0,
                                                 11 + dof_dim:11 + 2 * dof_dim]
        pred_full_joint[:, 0] = pred_joint[:, 0] - first_joint_delta

    pred_delta_local = torch.matmul(
        pred_full_rot_matrix[:, :-1].transpose(-1, -2),
        (pred_full_pos[:, 1:] - pred_full_pos[:, :-1]).unsqueeze(-1),
    ).squeeze(-1)
    pred_relative_rot = torch.matmul(
        pred_full_rot_matrix[:, :-1].transpose(-1, -2),
        pred_full_rot_matrix[:, 1:],
    )
    pred_delta_joint = pred_full_joint[:, 1:] - pred_full_joint[:, :-1]

    real_rot_matrix = quaternion_to_matrix(xyzw_to_wxyz(real_rot))
    if correction is None:
        joint_difference = pred_joint - real_joint[:, None]
        phase_error = joint_difference.square().sum(dim=-1)
        reverse_phase_index = torch.flip(phase_error, dims=(1,)).argmin(dim=-1)
        phase_index = num_frames - 1 - reverse_phase_index
        correction = {
            "real_pos": real_pos.detach().clone(),
            "real_joint": real_joint.detach().clone(),
            "real_rot_matrix": real_rot_matrix.detach().clone(),
            "phase_index": phase_index.detach().clone(),
            # The generated-history window ends at the current tracked frame.
            # Earlier phase matches therefore have a negative signed offset.
            "phase_offset_frames": (
                phase_index - (num_frames - 1)).detach().clone(),
            "phase_error": phase_error.gather(
                1, phase_index[:, None]).squeeze(1).detach().clone(),
        }
    else:
        correction = {
            key: value.to(device)
            for key, value in correction.items()
        }

    real_pos = correction["real_pos"].to(dtype=pred_pos.dtype)
    real_joint = correction["real_joint"].to(dtype=pred_joint.dtype)

    # The phase index addresses the generated states, while the integrated
    # sequence also contains the pre-history pose at index 0.
    phase_offset = correction["phase_offset_frames"].to(device)
    # Never extrapolate beyond the cached generated window.  An out-of-range
    # inherited phase is held at the nearest boundary; missing motion is thus
    # a zero translation increment and an identity rotation increment.
    phase_index = num_frames - 1 + phase_offset
    phase_index = phase_index.clamp(0, num_frames - 1).long()
    anchor_state_index = phase_index + 1
    hybrid_pos = torch.zeros_like(pred_full_pos)
    hybrid_rot_matrix = torch.zeros_like(pred_full_rot_matrix)
    hybrid_joint = torch.zeros_like(pred_full_joint)
    hybrid_pos[:, anchor_state_index] = real_pos
    hybrid_rot_matrix[:, anchor_state_index] = real_rot_matrix
    hybrid_joint[:, anchor_state_index] = real_joint

    # Re-integrate exactly the predicted edge increments in both directions.
    for edge_index in range(num_frames):
        forward_mask = edge_index >= anchor_state_index
        if bool(forward_mask.any()):
            next_index = edge_index + 1
            current_rot = hybrid_rot_matrix[:, edge_index]
            hybrid_rot_matrix[:, next_index] = torch.where(
                forward_mask[:, None, None],
                torch.matmul(current_rot, pred_relative_rot[:, edge_index]),
                hybrid_rot_matrix[:, next_index],
            )
            hybrid_pos[:, next_index] = torch.where(
                forward_mask[:, None],
                hybrid_pos[:, edge_index]
                + torch.matmul(
                    current_rot,
                    pred_delta_local[:, edge_index].unsqueeze(-1),
                ).squeeze(-1),
                hybrid_pos[:, next_index],
            )
            hybrid_joint[:, next_index] = torch.where(
                forward_mask[:, None],
                torch.clamp(
                    hybrid_joint[:, edge_index]
                    + pred_delta_joint[:, edge_index],
                    min=lower,
                    max=upper,
                ),
                hybrid_joint[:, next_index],
            )

        backward_index = num_frames - 1 - edge_index
        backward_mask = backward_index < anchor_state_index
        if bool(backward_mask.any()):
            current_rot = hybrid_rot_matrix[:, backward_index + 1]
            hybrid_rot_matrix[:, backward_index] = torch.where(
                backward_mask[:, None, None],
                torch.matmul(
                    current_rot,
                    pred_relative_rot[:, backward_index].transpose(-1, -2),
                ),
                hybrid_rot_matrix[:, backward_index],
            )
            hybrid_pos[:, backward_index] = torch.where(
                backward_mask[:, None],
                hybrid_pos[:, backward_index + 1]
                - torch.matmul(
                    hybrid_rot_matrix[:, backward_index],
                    pred_delta_local[:, backward_index].unsqueeze(-1),
                ).squeeze(-1),
                hybrid_pos[:, backward_index],
            )
            hybrid_joint[:, backward_index] = torch.where(
                backward_mask[:, None],
                torch.clamp(
                    hybrid_joint[:, backward_index + 1]
                    - pred_delta_joint[:, backward_index],
                    min=lower,
                    max=upper,
                ),
                hybrid_joint[:, backward_index],
            )

    hybrid_rot = wxyz_to_xyzw(matrix_to_quaternion(hybrid_rot_matrix))
    # Re-encode through the active feature definition. For V6 this projects
    # the integrated 3-D trajectory back to delta_hor + height, whose decoder
    # relation is p = p_hor - g*h.
    anchor_contact = predicted["contact_mask"][:, :1].to(device)
    reencode_motion = {
        "root_trans_offset": hybrid_pos,
        "root_rot": hybrid_rot,
        "dof": hybrid_joint,
        "contact_mask": torch.cat(
            [anchor_contact, predicted["contact_mask"].to(device)], dim=1),
    }
    if motion_dtype.FeatureVersion == 3:
        raw_history, aligned_abs_pose = motion_dict_to_feature_v3(
            reencode_motion)
    elif motion_dtype.FeatureVersion == 6:
        raw_history, aligned_abs_pose = motion_dict_to_feature_v6(
            reencode_motion)
    else:
        raise ValueError(
            "Residual re-anchoring supports FeatureVersion 3 and 6, got "
            f"{motion_dtype.FeatureVersion}")

    return (
        aligned_abs_pose,
        val_data.normalize(raw_history),
        correction,
        correction["phase_index"],
        correction["phase_error"],
    )


def apply_generated_history_alignment_correction(
        abs_pose: dict,
        correction: dict[str, torch.Tensor | None],
        history_motion: torch.Tensor | None = None,
        val_data: Any = None):
    """Apply a previously computed alignment correction without re-anchoring.

    This is used while the controller is still tracking an older plan.  The
    correction is a fixed frame transform, so it can be applied to a newly
    selected history window without comparing that window to the current
    measured state again.
    """
    from robotmdar.dtype.rotation import (
        euler_angles_to_quaternion,
        get_euler_xyz,
        quat_apply,
        quat_inverse,
        quat_mul,
    )

    device = (
        history_motion.device
        if history_motion is not None
        else abs_pose["root_trans_offset"].device
    )
    source_abs_pos = abs_pose["root_trans_offset"].to(device)
    source_abs_rot = abs_pose["root_rot"].to(device)
    if source_abs_pos.ndim == 1:
        source_abs_pos = source_abs_pos.unsqueeze(0)
    if source_abs_rot.ndim == 1:
        source_abs_rot = source_abs_rot.unsqueeze(0)
    pose_rotation = correction["pose_rotation"].to(device)
    pose_translation = correction["pose_translation"].to(device)
    aligned_abs_pose = {
        "root_trans_offset": (
            quat_apply(
                pose_rotation,
                source_abs_pos,
                w_last=True,
            )
            + pose_translation
        ),
        "root_rot": quat_mul(
            pose_rotation,
            source_abs_rot,
            w_last=True,
        ),
    }

    if history_motion is None or val_data is None:
        return aligned_abs_pose, history_motion

    raw = val_data.denormalize(history_motion.to(device)).clone()
    batch_size, num_frames = raw.shape[:2]
    if num_frames <= 0:
        raise ValueError("Generated history must contain at least one feature")

    if motion_dtype.FeatureVersion == 6:
        height_delta = correction["feature_height_delta"]
        gravity_rotation = correction["feature_gravity_rotation"]
        if height_delta is None or gravity_rotation is None:
            raise ValueError(
                "FeatureVersion 6 alignment correction is missing its "
                "height/gravity transform")
        raw[..., 0] = raw[..., 0] + height_delta.to(
            device=device, dtype=raw.dtype).reshape(-1, 1)
        raw[..., 1:4] = torch.matmul(
            gravity_rotation.to(device=device, dtype=raw.dtype)
            .unsqueeze(1),
            raw[..., 1:4].unsqueeze(-1),
        ).squeeze(-1)
        raw[..., 1:4] = F.normalize(
            raw[..., 1:4], dim=-1, eps=1e-8)
        return aligned_abs_pose, val_data.normalize(raw)

    if motion_dtype.FeatureVersion != 3:
        return aligned_abs_pose, history_motion

    feature_rotation = correction["feature_rotation"].to(device)
    feature_translation = correction["feature_translation"].to(device)

    sin_roll = raw[..., 0]
    cos_roll = raw[..., 1] + 1
    sin_pitch = raw[..., 2]
    cos_pitch = raw[..., 3] + 1
    delta_yaw = raw[..., 4]

    init_euler = get_euler_xyz(
        source_abs_rot, w_last=True)
    ref_yaw = init_euler[2]
    yaw_old = torch.zeros(
        batch_size, num_frames, device=device, dtype=raw.dtype)
    yaw_old[:, 0] = ref_yaw
    if num_frames > 1:
        yaw_old[:, 1:] = (
            torch.cumsum(delta_yaw[:, :num_frames - 1], dim=1)
            + ref_yaw.reshape(-1, 1)
        )

    euler = torch.stack([sin_roll.atan2(cos_roll),
                         sin_pitch.atan2(cos_pitch),
                         yaw_old], dim=-1)
    rot_orig = euler_angles_to_quaternion(euler)
    rot_corrected = quat_mul(
        feature_rotation.expand(batch_size * num_frames, 4),
        rot_orig.reshape(-1, 4),
        w_last=True,
    ).reshape(batch_size, num_frames, 4)

    roll_new, pitch_new, yaw_new = get_euler_xyz(
        rot_corrected.reshape(-1, 4), w_last=True)
    roll_new = roll_new.reshape(batch_size, num_frames)
    pitch_new = pitch_new.reshape(batch_size, num_frames)
    yaw_new = yaw_new.reshape(batch_size, num_frames)
    raw[..., 0] = torch.sin(roll_new)
    raw[..., 1] = torch.cos(roll_new) - 1
    raw[..., 2] = torch.sin(pitch_new)
    raw[..., 3] = torch.cos(pitch_new) - 1
    if num_frames > 1:
        raw[..., :num_frames - 1, 4] = (
            yaw_new[:, 1:] - yaw_new[:, :-1])

    delta_trans_local = raw[..., 7:10].clone()
    yaw_quat_old = euler_angles_to_quaternion(
        torch.stack([
            torch.zeros_like(yaw_old),
            torch.zeros_like(yaw_old),
            yaw_old,
        ], dim=-1),
    )
    world_disp = quat_apply(
        yaw_quat_old[:, :-1].reshape(-1, 4),
        delta_trans_local[:, :-1].reshape(-1, 3),
        w_last=True,
    ).reshape(batch_size, num_frames - 1, 3)
    world_disp_corr = quat_apply(
        feature_rotation.expand(batch_size * (num_frames - 1), 4),
        world_disp.reshape(-1, 3),
        w_last=True,
    ).reshape(batch_size, num_frames - 1, 3)
    yaw_quat_new = euler_angles_to_quaternion(
        torch.stack([
            torch.zeros_like(yaw_new),
            torch.zeros_like(yaw_new),
            yaw_new,
        ], dim=-1),
    )
    delta_trans_new = quat_apply(
        quat_inverse(yaw_quat_new[:, :-1], w_last=True).reshape(-1, 4),
        world_disp_corr.reshape(-1, 3),
        w_last=True,
    ).reshape(batch_size, num_frames - 1, 3)
    if num_frames > 1:
        raw[..., :num_frames - 1, 7:10] = delta_trans_new

    root_pos_old = torch.zeros(
        batch_size, num_frames, 3, device=device, dtype=raw.dtype)
    root_pos_old[:, 0] = source_abs_pos.to(dtype=raw.dtype)
    if num_frames > 1:
        root_pos_old[:, 1:] = (
            torch.cumsum(world_disp, dim=1) + root_pos_old[:, :1]
        )
    root_pos_old[..., 2] = raw[..., 10]
    root_pos_corrected = quat_apply(
        feature_rotation.unsqueeze(1).expand(batch_size, num_frames, 4),
        root_pos_old,
        w_last=True,
    ) + feature_translation.unsqueeze(1)
    raw[..., 10] = root_pos_corrected[..., 2]
    return aligned_abs_pose, val_data.normalize(raw)


def align_generated_history_pose(abs_pose: dict,
                                 generated_reference_pos: torch.Tensor,
                                 generated_reference_rot: torch.Tensor,
                                 state_msg: Any,
                                 device: str | torch.device,
                                 history_motion: torch.Tensor | None = None,
                                 val_data: Any = None,
                                 return_correction: bool = False):
    """Translate and rotate generated history so its reference pose matches the real G1 root.

    When *history_motion* and *val_data* are provided, version-specific
    absolute pose channels are also corrected so every reconstructed frame
    carries the current G1 seam state.

    If *return_correction* is true, append the fixed correction transform to
    the return tuple so later replans can inherit it without re-anchoring.
    """
    from robotmdar.dtype.rotation import (
        euler_angles_to_quaternion,
        get_euler_xyz,
        quat_apply,
        quat_inverse,
        quat_mul,
    )

    real_current_pos = torch.tensor(
        state_msg.states.g1_pos[-1], dtype=torch.float32,
        device=device).reshape(1, 3)
    # Normalise the real quaternion defensively (ZMQ decoding may produce
    # views with non-unit norm).
    real_current_rot_q = _normalized_wire_quaternions_wxyz_as_xyzw(
        np.asarray(state_msg.states.g1_root_rot[-1:], dtype=np.float32))
    real_current_rot = torch.as_tensor(
        real_current_rot_q, dtype=torch.float32, device=device).reshape(1, 4)

    generated_reference_pos = generated_reference_pos.to(device).reshape(1, 3)
    generated_reference_rot = generated_reference_rot.to(device).reshape(1, 4)

    # Rotation delta: q_delta = q_real * q_gen^{-1}
    q_gen_inv = quat_inverse(generated_reference_rot, w_last=True)
    q_delta = quat_mul(real_current_rot, q_gen_inv, w_last=True)
    feature_translation = (
        real_current_pos
        - quat_apply(q_delta, generated_reference_pos, w_last=True)
    )
    feature_height_delta = None
    feature_gravity_rotation = None

    # Rotate the generated position *around* the reference pivot, then add
    # the real translation.
    rel_pos = (abs_pose["root_trans_offset"].to(device)
               - generated_reference_pos)
    rotated_rel_pos = quat_apply(q_delta, rel_pos, w_last=True)

    aligned_abs_pose = {
        "root_trans_offset": real_current_pos + rotated_rel_pos,
        "root_rot": quat_mul(
            q_delta, abs_pose["root_rot"].to(device), w_last=True),
    }

    # Also correct history-frame features when the representation contains
    # absolute pose channels that would otherwise keep the generated seam.
    aligned_history_motion = history_motion
    if (motion_dtype.FeatureVersion == 6
            and history_motion is not None and val_data is not None):
        from robotmdar.dtype.rotation import quaternion_to_matrix, xyzw_to_wxyz

        raw = val_data.denormalize(history_motion.to(device)).clone()
        B, T = raw.shape[:2]
        if T <= 0:
            raise ValueError("Generated history must contain at least one feature")

        real_height = real_current_pos[:, 2].to(dtype=raw.dtype)
        real_rot_matrix = quaternion_to_matrix(
            xyzw_to_wxyz(real_current_rot.to(dtype=raw.dtype))
        )
        world_gravity = torch.zeros(
            real_rot_matrix.shape[:-2] + (3, ),
            device=device,
            dtype=raw.dtype,
        )
        world_gravity[..., 2] = -1.0
        real_gravity = torch.matmul(
            real_rot_matrix.transpose(-1, -2),
            world_gravity.unsqueeze(-1),
        ).squeeze(-1)

        if real_height.shape[0] != B:
            if real_height.shape[0] != 1:
                raise ValueError(
                    "Real current state batch does not match generated history: "
                    f"{real_height.shape[0]} != {B}")
            real_height = real_height.expand(B)
            real_gravity = real_gravity.expand(B, 3)

        # FeatureVersion 6 has no absolute XY channel.  Its endpoint state is
        # represented by the final feature's absolute height and gravity.
        delta_h = real_height - raw[:, -1, 0]
        feature_height_delta = delta_h.detach().clone()
        raw[..., 0] = raw[..., 0] + delta_h.reshape(B, 1)

        gravity_delta = _rotation_matrix_between_vectors(
            raw[:, -1, 1:4], real_gravity)
        feature_gravity_rotation = gravity_delta.detach().clone()
        raw[..., 1:4] = torch.matmul(
            gravity_delta.unsqueeze(1),
            raw[..., 1:4].unsqueeze(-1),
        ).squeeze(-1)
        raw[..., 1:4] = F.normalize(raw[..., 1:4], dim=-1, eps=1e-8)

        abs_root_pos = abs_pose["root_trans_offset"].to(
            device=device, dtype=raw.dtype).reshape(-1, 3)
        if abs_root_pos.shape[0] != B:
            if abs_root_pos.shape[0] != 1:
                raise ValueError(
                    "Generated abs_pose batch does not match generated history: "
                    f"{abs_root_pos.shape[0]} != {B}")
            abs_root_pos = abs_root_pos.expand(B, 3)
        aligned_root_pos = aligned_abs_pose["root_trans_offset"].to(
            device=device, dtype=raw.dtype).reshape(-1, 3)
        if aligned_root_pos.shape[0] != B:
            if aligned_root_pos.shape[0] != 1:
                raise ValueError(
                    "Aligned abs_pose batch does not match generated history: "
                    f"{aligned_root_pos.shape[0]} != {B}")
            aligned_root_pos = aligned_root_pos.expand(B, 3)
        aligned_root_pos = aligned_root_pos.clone()
        aligned_root_pos[:, 2] = abs_root_pos[:, 2] + delta_h
        aligned_abs_pose = dict(aligned_abs_pose)
        aligned_abs_pose["root_trans_offset"] = aligned_root_pos
        aligned_root_rot = aligned_abs_pose["root_rot"].to(
            device=device, dtype=raw.dtype).reshape(-1, 4)
        if aligned_root_rot.shape[0] != B:
            if aligned_root_rot.shape[0] != 1:
                raise ValueError(
                    "Aligned root rotation batch does not match generated history: "
                    f"{aligned_root_rot.shape[0]} != {B}")
            aligned_root_rot = aligned_root_rot.expand(B, 4)
        aligned_abs_pose["root_rot"] = aligned_root_rot.clone()

        decoded_history = motion_dtype.motion_feature_to_dict(
            raw, aligned_abs_pose)
        decoded_endpoint_rot = decoded_history["root_rot"][:, -1].to(
            device=device, dtype=raw.dtype)
        real_endpoint_rot = real_current_rot.to(
            device=device, dtype=raw.dtype).reshape(-1, 4)
        if real_endpoint_rot.shape[0] != B:
            if real_endpoint_rot.shape[0] != 1:
                raise ValueError(
                    "Real root rotation batch does not match generated history: "
                    f"{real_endpoint_rot.shape[0]} != {B}")
            real_endpoint_rot = real_endpoint_rot.expand(B, 4)
        root_rot_residual = quat_mul(
            real_endpoint_rot,
            quat_inverse(decoded_endpoint_rot, w_last=True),
            w_last=True,
        )
        aligned_abs_pose["root_rot"] = F.normalize(
            quat_mul(
                root_rot_residual,
                aligned_abs_pose["root_rot"],
                w_last=True,
            ),
            dim=-1,
            eps=1e-8,
        )

        decoded_history = motion_dtype.motion_feature_to_dict(
            raw, aligned_abs_pose)
        decoded_endpoint = decoded_history["root_trans_offset"][:, -1]
        real_endpoint_pos = real_current_pos.to(
            device=device, dtype=decoded_endpoint.dtype)
        if real_endpoint_pos.shape[0] != B:
            if real_endpoint_pos.shape[0] != 1:
                raise ValueError(
                    "Real root position batch does not match generated history: "
                    f"{real_endpoint_pos.shape[0]} != {B}")
            real_endpoint_pos = real_endpoint_pos.expand(B, 3)
        aligned_abs_pose["root_trans_offset"][:, :2] += (
            real_endpoint_pos[:, :2] - decoded_endpoint[:, :2])

        aligned_history_motion = val_data.normalize(raw)
    elif history_motion is not None and val_data is not None:
        raw = val_data.denormalize(
            history_motion.to(device)).clone()  # (B, T, 57 or 69)
        B, T = raw.shape[:2]

        # V3 stores absolute roll/pitch sincos and a forward delta_yaw.
        # Re-express all three, plus yaw-local translation deltas, under the
        # same rigid correction used for abs_pose.
        # -- original per-frame Euler angles (matching motion_feature_to_dict_v3) --
        sin_roll = raw[..., 0]
        cos_roll = raw[..., 1] + 1              # stored as cos(roll) - 1
        sin_pitch = raw[..., 2]
        cos_pitch = raw[..., 3] + 1
        delta_yaw = raw[..., 4]                 # (B, T) — only [:, :T-1] is used by decoder

        roll = torch.atan2(sin_roll, cos_roll)  # (B, T)
        pitch = torch.atan2(sin_pitch, cos_pitch)

        init_euler = get_euler_xyz(abs_pose["root_rot"].to(device), w_last=True)
        ref_yaw = init_euler[2]                 # scalar per batch
        yaw_old = torch.zeros(B, T, device=device)
        yaw_old[:, 0] = ref_yaw
        if T > 1:
            yaw_old[:, 1:] = (torch.cumsum(delta_yaw[:, :T - 1], dim=1)
                              + ref_yaw.reshape(-1, 1))

        # -- apply q_delta to every frame's rotation --
        euler = torch.stack([roll, pitch, yaw_old], dim=-1)     # (B, T, 3)
        rot_orig = euler_angles_to_quaternion(euler)            # (B, T, 4) xyzw
        rot_corrected = quat_mul(
            q_delta.expand(B * T, 4), rot_orig.reshape(-1, 4), w_last=True,
        ).reshape(B, T, 4)

        # Extract new per-frame Euler angles.
        roll_new, pitch_new, yaw_new = get_euler_xyz(
            rot_corrected.reshape(-1, 4), w_last=True)
        roll_new = roll_new.reshape(B, T)
        pitch_new = pitch_new.reshape(B, T)
        yaw_new = yaw_new.reshape(B, T)

        # -- write corrected sincos --
        raw[..., 0] = torch.sin(roll_new)
        raw[..., 1] = torch.cos(roll_new) - 1
        raw[..., 2] = torch.sin(pitch_new)
        raw[..., 3] = torch.cos(pitch_new) - 1

        # -- write corrected delta_yaw (only indices used by the decoder) --
        if T > 1:
            raw[..., :T - 1, 4] = yaw_new[:, 1:] - yaw_new[:, :-1]

        # -- re-express delta_trans_local in the corrected yaw frame --
        # delta_trans_local[t] lives in the yaw[t]-aligned local basis.
        # After the rotation correction the same world-space displacement
        # must be rotated into the new yaw_new[t] basis.
        delta_trans_local = raw[..., 7:10].clone()               # (B, T, 3)
        yaw_quat_old = euler_angles_to_quaternion(
            torch.stack([torch.zeros_like(yaw_old),
                         torch.zeros_like(yaw_old), yaw_old], dim=-1),
        )                                                        # (B, T, 4)
        world_disp = quat_apply(
            yaw_quat_old[:, :-1].reshape(-1, 4),
            delta_trans_local[:, :-1].reshape(-1, 3), w_last=True,
        ).reshape(B, T - 1, 3)
        # Apply the same q_delta rotation in world space.
        world_disp_corr = quat_apply(
            q_delta.expand(B * (T - 1), 4),
            world_disp.reshape(-1, 3), w_last=True,
        ).reshape(B, T - 1, 3)
        # Project back into the *new* yaw frame.
        yaw_quat_new = euler_angles_to_quaternion(
            torch.stack([torch.zeros_like(yaw_new),
                         torch.zeros_like(yaw_new), yaw_new], dim=-1),
        )
        inv_yaw_new = quat_inverse(yaw_quat_new[:, :-1], w_last=True)
        delta_trans_new = quat_apply(
            inv_yaw_new.reshape(-1, 4),
            world_disp_corr.reshape(-1, 3), w_last=True,
        ).reshape(B, T - 1, 3)
        raw[..., :T - 1, 7:10] = delta_trans_new

        # Feature V3 stores height as an absolute world-space value rather
        # than deriving it from delta_trans_local. Its decoder overwrites the
        # reconstructed z coordinate with this channel, so transform every
        # history position explicitly. Without this, x/y and orientation
        # align at the seam while z remains at the generated height.
        root_pos_old = torch.zeros(B, T, 3, device=device, dtype=raw.dtype)
        root_pos_old[:, 0] = abs_pose["root_trans_offset"].to(device)
        if T > 1:
            root_pos_old[:, 1:] = (
                torch.cumsum(world_disp, dim=1)
                + root_pos_old[:, :1]
            )
        root_pos_old[..., 2] = raw[..., 10]
        root_pos_corrected = real_current_pos.unsqueeze(1) + quat_apply(
            q_delta.unsqueeze(1).expand(B, T, 4),
            root_pos_old - generated_reference_pos.unsqueeze(1),
            w_last=True,
        )
        raw[..., 10] = root_pos_corrected[..., 2]

        aligned_history_motion = val_data.normalize(raw)

    pose_rotation = quat_mul(
        aligned_abs_pose["root_rot"].to(device),
        quat_inverse(abs_pose["root_rot"].to(device), w_last=True),
        w_last=True,
    )
    pose_translation = (
        aligned_abs_pose["root_trans_offset"].to(device)
        - quat_apply(
            pose_rotation,
            abs_pose["root_trans_offset"].to(device),
            w_last=True,
        )
    )
    correction = {
        "pose_rotation": pose_rotation.detach().clone(),
        "pose_translation": pose_translation.detach().clone(),
        "feature_rotation": q_delta.detach().clone(),
        "feature_translation": feature_translation.detach().clone(),
        "feature_height_delta": (
            feature_height_delta.detach().clone()
            if feature_height_delta is not None else None
        ),
        "feature_gravity_rotation": (
            feature_gravity_rotation.detach().clone()
            if feature_gravity_rotation is not None else None
        ),
    }

    # Goal reference pose is the *real* G1 pose so ego-goal is computed
    # relative to where the robot actually is.
    result = (
        aligned_abs_pose,
        real_current_pos,       # goal_reference_pos
        real_current_rot,       # goal_reference_rot
        real_current_pos - generated_reference_pos,  # translation
        aligned_history_motion,
    )
    if return_correction:
        return result + (correction,)
    return result


def tracked_frame_from_timestamps(state_msg: Any, fps: float,
                                  future_len: int) -> int:
    """Resolve the active plan frame from controller-owned timestamps."""
    if fps <= 0:
        raise ValueError(f"Motion fps must be positive, got {fps}")
    if future_len <= 0:
        raise ValueError(f"future_len must be positive, got {future_len}")
    start_t_ns = int(state_msg.tracking.start_t_ns)
    state_t_ns = int(state_msg.history_meta.publish_t_ns)
    if start_t_ns <= 0:
        raise ValueError("Controller has not reported an active plan start time")
    if state_t_ns < start_t_ns:
        raise ValueError(
            f"State timestamp {state_t_ns} precedes plan start {start_t_ns}")
    elapsed_frames = round((state_t_ns - start_t_ns) * fps / 1e9)
    return min(int(elapsed_frames), future_len - 1)


def generated_history_at_frame(plan: dict, tracked_frame: int,
                               history_len: int, *,
                               phase_lag_offset: int = 0):
    """Select history at a tracked frame, compensating for tracker phase lag.

    ``phase_lag_offset`` shifts the selected generated frame earlier in the
    plan, so a measured state at ``tracked_frame`` is compared with the
    predicted state at ``tracked_frame - phase_lag_offset``.
    """
    features = plan["features"]
    root_pos = plan["root_pos"]
    root_rot = plan["root_rot"]
    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")
    if tracked_frame < 0:
        raise ValueError(f"tracked_frame must be non-negative, got {tracked_frame}")
    phase_lag_offset = int(phase_lag_offset)
    if phase_lag_offset < 0:
        raise ValueError(
            f"phase_lag_offset must be non-negative, got {phase_lag_offset}")
    selected_frame = max(0, tracked_frame - phase_lag_offset)
    feature_end = history_len + selected_frame
    feature_start = feature_end - history_len + 1
    if feature_end >= features.shape[1]:
        raise ValueError(
            f"Selected frame {selected_frame} exceeds cached plan with "
            f"{features.shape[1] - history_len} future frames")
    if root_pos.shape[1] <= feature_end or root_rot.shape[1] <= feature_end:
        raise ValueError("Cached plan poses do not cover the selected history")
    history = features[:, feature_start:feature_end + 1]
    if history.shape[1] != history_len:
        raise ValueError(
            f"Selected {history.shape[1]} history frames, expected {history_len}")
    anchor_index = (
        feature_start - 1 if motion_dtype.FeatureVersion == 6 else feature_start
    )
    if anchor_index < 0:
        raise ValueError(
            "FeatureVersion 6 generated history requires cached pose before "
            f"feature_start={feature_start}")
    abs_pose = {
        "root_trans_offset": root_pos[:, anchor_index],
        "root_rot": root_rot[:, anchor_index],
    }
    return history, abs_pose, root_pos[:, feature_end], root_rot[:, feature_end]


def _forward_velocity(values: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"Motion fps must be positive, got {fps}")
    velocity = np.zeros_like(values)
    if len(values) > 1:
        velocity[:-1] = (values[1:] - values[:-1]) * fps
        velocity[-1] = velocity[-2]
    return velocity


def _textop_bodies_to_sonic(values: np.ndarray) -> np.ndarray:
    """Build SONIC's documented root-only 30-body representation.

    TextOp's hands are synthetic extensions of its locked-wrist skeleton and
    are not valid SONIC VR targets. Replicating the pelvis disables VR guidance
    while retaining the root and 29-joint tracking inputs used by the policy.
    """
    values = np.asarray(values)
    if values.ndim < 3 or values.shape[-2] < 1:
        raise ValueError(
            f"Expected at least one TextOp FK body, got {values.shape}")
    return np.ascontiguousarray(np.repeat(values[..., :1, :], 30, axis=-2))


def motion_dict_to_g1data(motion_dict: dict, skip_history: int,
                          fps: float = 50.0,
                          locked_joint_pos: np.ndarray | None = None,
                          include_body: bool = False):
    """Convert one reconstructed MuJoCo batch to ``G1MotionData``."""
    from sonicmsg.messages import G1MotionData

    def batch_numpy(key: str) -> np.ndarray:
        value = motion_dict[key]
        if value.shape[0] != 1:
            raise ValueError(f"Planner supports batch size 1, got {key} {value.shape}")
        return value[0].detach().cpu().numpy()

    dof_pos = np.asarray(batch_numpy("dof_pos"), dtype=np.float32)
    body_pos = np.asarray(
        batch_numpy("global_translation_extend"), dtype=np.float32)
    body_ori_xyzw = np.asarray(
        batch_numpy("global_rotation_extend"), dtype=np.float32)
    if not (len(dof_pos) == len(body_pos) == len(body_ori_xyzw)):
        raise ValueError("Reconstructed motion arrays have different frame counts")
    if skip_history < 0 or skip_history >= len(dof_pos):
        raise ValueError(
            f"skip_history must be in [0, {len(dof_pos) - 1}], got {skip_history}")

    # Derive velocity before slicing so the history/future boundary remains
    # available if the velocity convention is changed to backward difference.
    model_dof_dim = int(dof_pos.shape[-1])
    if model_dof_dim not in (23, 29):
        raise ValueError(
            f"Reconstructed motion must contain 23 or 29 DoFs, got "
            f"{dof_pos.shape}")
    dof_vel = _forward_velocity(dof_pos, fps)
    joint_pos = mujoco_to_isaaclab_dof(dof_pos)[skip_history:]
    joint_vel = mujoco_to_isaaclab_dof(dof_vel)[skip_history:]
    if model_dof_dim == 23 and locked_joint_pos is not None:
        locked_joint_pos = np.asarray(locked_joint_pos, dtype=np.float32)
        if locked_joint_pos.shape != (29,):
            raise ValueError(
                f"locked_joint_pos must have shape (29,), got "
                f"{locked_joint_pos.shape}")
        # The 23-DoF model does not predict wrists. Preserve the measured
        # SONIC wrist pose and command zero wrist velocity across the plan.
        joint_pos[:, _WRIST_ISAACLAB_INDICES] = locked_joint_pos[
            _WRIST_ISAACLAB_INDICES]
        joint_vel[:, _WRIST_ISAACLAB_INDICES] = 0.0
    root_pos = np.ascontiguousarray(body_pos[skip_history:, 0, :])
    root_ori = np.ascontiguousarray(
        body_ori_xyzw[skip_history:, 0, :][..., [3, 0, 1, 2]])
    g1_fields = getattr(G1MotionData, "model_fields", {})
    supports_root_payload = (
        "root_pos" in g1_fields and "root_ori" in g1_fields)
    body_pos_packet = None
    body_ori_packet = None
    if include_body or not supports_root_payload:
        body_pos_packet = _textop_bodies_to_sonic(body_pos[skip_history:])
        body_ori_packet = np.ascontiguousarray(
            _textop_bodies_to_sonic(body_ori_xyzw[skip_history:])[
                ..., [3, 0, 1, 2]])
    if not supports_root_payload:
        return G1MotionData(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos=body_pos_packet,
            body_ori=body_ori_packet,
            framerate=float(fps),
        )

    return G1MotionData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=root_pos,
        root_ori=root_ori,
        body_pos=body_pos_packet,
        body_ori=body_ori_packet,
        framerate=float(fps),
    )

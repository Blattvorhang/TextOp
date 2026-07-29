"""Conversions between TextOp controller state, DAR features, and G1 motion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robotmdar.utils.goal import GoalType, build_ego_goal
from robotmdar.dtype.motion import motion_dict_to_feature_v3


# Indexing an IsaacLab-ordered vector with this array produces MuJoCo order.
_ISAACLAB_TO_MUJOCO = np.asarray([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19,
    21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
], dtype=np.int64)

# Indexing a MuJoCo-ordered vector with this array produces IsaacLab order.
_MUJOCO_TO_ISAACLAB = np.asarray([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23,
    5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int64)


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
    return np.ascontiguousarray(
        np.concatenate((values[..., :19], values[..., 22:26]), axis=-1))


def _expand_mujoco_23_to_29(values: np.ndarray) -> np.ndarray:
    """Insert zero-valued wrist DoFs into MuJoCo-ordered values."""
    values = np.asarray(values)
    if values.shape[-1] != 23:
        raise ValueError(f"Expected 23 MuJoCo DoFs, got {values.shape}")
    expanded = np.zeros(values.shape[:-1] + (29,), dtype=values.dtype)
    expanded[..., :19] = values[..., :19]
    expanded[..., 22:26] = values[..., 19:23]
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


def state_to_model_input(state_msg: Any, history_len: int, val_data: Any,
                         device: str | torch.device):
    """Build normalized DAR history from the latest physical controller states.

    FeatureVersion 3 stores forward deltas. Consequently ``history_len`` model
    features require ``history_len + 1`` physical poses.
    """
    required_states = history_len + 1
    raw = state_msg.raw
    positions = np.array(raw["g1_pos"], dtype=np.float32, copy=True)
    rotations = _normalized_quaternions_xyzw(raw["g1_root_rot"])
    joints = np.array(raw["g1_joint_pos"], dtype=np.float32, copy=True)

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
    joints_mujoco = isaaclab_to_mujoco_dof(joints[-required_states:])
    joints_23 = _reduce_mujoco_29_to_23(joints_mujoco)

    motion_dict = {
        "root_trans_offset": torch.as_tensor(
            positions, dtype=torch.float32, device=device).unsqueeze(0),
        "root_rot": torch.as_tensor(
            rotations, dtype=torch.float32, device=device).unsqueeze(0),
        "dof": torch.as_tensor(
            joints_23, dtype=torch.float32, device=device).unsqueeze(0),
        "contact_mask": torch.ones(
            (1, required_states, 2), dtype=torch.float32, device=device),
    }
    feature, abs_pose = motion_dict_to_feature_v3(motion_dict)
    if feature.shape != (1, history_len, 57):
        raise ValueError(
            f"Unexpected FeatureVersion 3 shape {tuple(feature.shape)}; "
            f"expected (1, {history_len}, 57)")

    # motion_dict_to_feature_v3 subtracts Euler yaw directly. Wrap the result
    # at the branch cut before applying the training-set normalization.
    feature[..., 4] = torch.atan2(
        torch.sin(feature[..., 4]), torch.cos(feature[..., 4]))
    return val_data.normalize(feature), abs_pose


def state_to_ego_goal(state_msg: Any,
                      device: str | torch.device,
                      goal_type: GoalType | str = GoalType.ROOT,
                      goal_reference_path: str | Path | None = None
                      ) -> torch.Tensor:
    """Convert the root goal relative to the last history-feature pose."""
    reference_pos = torch.tensor(
        state_msg.raw["g1_pos"][-2], dtype=torch.float32,
        device=device).reshape(1, 3)
    reference_rot_np = _normalized_quaternions_xyzw(
        np.asarray(state_msg.raw["g1_root_rot"][-2:], dtype=np.float32))[:1]
    reference_rot = torch.as_tensor(
        reference_rot_np, dtype=torch.float32, device=device)
    return state_goal_from_reference(
        state_msg, reference_pos, reference_rot, device,
        goal_type=goal_type, goal_reference_path=goal_reference_path)


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
    goal_root_pos: np.ndarray,
    goal_heading: float,
) -> np.ndarray:
    """Place an XY-origin, absolute-Z reference pose in the world frame."""
    goal_root_pos = np.asarray(goal_root_pos, dtype=np.float32)
    if goal_root_pos.shape != (3,):
        raise ValueError(
            f"goal_root_pos must have shape (3,), got {goal_root_pos.shape}")
    if not np.isfinite(goal_root_pos).all() or not np.isfinite(goal_heading):
        raise ValueError("Goal root position and heading must be finite")

    keypoints = _load_goal_keypoint_template(str(Path(ref_path).resolve())).copy()
    if not np.isclose(goal_root_pos[2], keypoints[0, 2], atol=1e-4):
        raise ValueError(
            f"goal_root_pos.z ({goal_root_pos[2]:.4f}) does not match "
            f"reference root z ({keypoints[0, 2]:.4f})")

    c = np.cos(float(goal_heading))
    s = np.sin(float(goal_heading))
    rotation_xy = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    keypoints[:, :2] = keypoints[:, :2] @ rotation_xy.T
    keypoints[:, :2] += goal_root_pos[:2]
    return np.ascontiguousarray(keypoints)


def state_goal_from_reference(state_msg: Any,
                              reference_pos: torch.Tensor,
                              reference_rot: torch.Tensor,
                              device: str | torch.device,
                              goal_type: GoalType | str = GoalType.ROOT,
                              goal_reference_path: str | Path | None = None
                              ) -> torch.Tensor:
    """Convert the state goal relative to an explicit generated-history pose."""
    goal_type = GoalType.parse(goal_type)
    goal_keypoints = None

    if goal_type is GoalType.BODY:
        raw = getattr(state_msg, 'raw', {})
        state_keypoints = getattr(state_msg, 'goal_keypoints', None)
        if state_keypoints is None:
            state_keypoints = raw.get('goal_keypoints')
        if state_keypoints is None:
            if goal_reference_path is None:
                raise ValueError(
                    "Body goal requires controller goal_keypoints or "
                    "goal_reference_path")
            if state_msg.goal_root_pos is None or state_msg.goal_heading is None:
                raise ValueError("TextOp state is missing its root-heading goal")
            state_keypoints = load_goal_keypoints_from_reference(
                goal_reference_path,
                np.asarray(state_msg.goal_root_pos, dtype=np.float32),
                float(np.asarray(state_msg.goal_heading).reshape(-1)[0]),
            )
        goal_keypoints = torch.as_tensor(
            np.array(state_keypoints, dtype=np.float32, copy=True),
            dtype=torch.float32, device=device).reshape(1, 5, 3)
        goal_pos = goal_keypoints[:, 0]
        goal_yaw = torch.zeros(1, dtype=torch.float32, device=device)
    else:
        if state_msg.goal_root_pos is None or state_msg.goal_heading is None:
            raise ValueError("TextOp state is missing its root-heading goal")
        goal_pos = torch.tensor(
            state_msg.goal_root_pos, dtype=torch.float32,
            device=device).reshape(1, 3)
        goal_yaw = torch.tensor(
            state_msg.goal_heading, dtype=torch.float32,
            device=device).reshape(1)

    return build_ego_goal(
        goal_pos, goal_yaw, reference_pos.to(device), reference_rot.to(device),
        goal_type=goal_type, goal_keypoints=goal_keypoints)


def align_generated_history_pose(abs_pose: dict,
                                 generated_reference_pos: torch.Tensor,
                                 state_msg: Any,
                                 device: str | torch.device):
    """Translate generated history so its last pose matches the real G1 root."""
    real_current_pos = torch.tensor(
        state_msg.raw["g1_pos"][-1], dtype=torch.float32,
        device=device).reshape(1, 3)
    generated_reference_pos = generated_reference_pos.to(device).reshape(1, 3)
    translation = real_current_pos - generated_reference_pos
    aligned_abs_pose = {
        "root_trans_offset": abs_pose["root_trans_offset"].to(device) + translation,
        "root_rot": abs_pose["root_rot"].to(device),
    }
    return aligned_abs_pose, generated_reference_pos + translation, translation


def tracked_frame_from_timestamps(state_msg: Any, fps: float,
                                  future_len: int) -> int:
    """Resolve the active plan frame from controller-owned timestamps."""
    if fps <= 0:
        raise ValueError(f"Motion fps must be positive, got {fps}")
    if future_len <= 0:
        raise ValueError(f"future_len must be positive, got {future_len}")
    start_t_ns = int(state_msg.tracked_plan_start_t_ns)
    state_t_ns = int(state_msg.publish_t_ns)
    if start_t_ns <= 0:
        raise ValueError("Controller has not reported an active plan start time")
    if state_t_ns < start_t_ns:
        raise ValueError(
            f"State timestamp {state_t_ns} precedes plan start {start_t_ns}")
    elapsed_frames = round((state_t_ns - start_t_ns) * fps / 1e9)
    return min(int(elapsed_frames), future_len - 1)


def generated_history_at_frame(plan: dict, tracked_frame: int,
                               history_len: int):
    """Select model history ending at a tracked published-future frame."""
    features = plan["features"]
    root_pos = plan["root_pos"]
    root_rot = plan["root_rot"]
    if history_len <= 0:
        raise ValueError(f"history_len must be positive, got {history_len}")
    if tracked_frame < 0:
        raise ValueError(f"tracked_frame must be non-negative, got {tracked_frame}")
    feature_end = history_len + tracked_frame
    feature_start = feature_end - history_len + 1
    if feature_end >= features.shape[1]:
        raise ValueError(
            f"Tracked frame {tracked_frame} exceeds cached plan with "
            f"{features.shape[1] - history_len} future frames")
    if root_pos.shape[1] <= feature_end or root_rot.shape[1] <= feature_end:
        raise ValueError("Cached plan poses do not cover the selected history")
    history = features[:, feature_start:feature_end + 1]
    if history.shape[1] != history_len:
        raise ValueError(
            f"Selected {history.shape[1]} history frames, expected {history_len}")
    abs_pose = {
        "root_trans_offset": root_pos[:, feature_start],
        "root_rot": root_rot[:, feature_start],
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


def motion_dict_to_g1data(motion_dict: dict, skip_history: int,
                          fps: float = 50.0):
    """Convert one reconstructed MuJoCo batch to ``G1MotionData``."""
    from sonicmsg.messages import G1MotionData

    def batch_numpy(key: str) -> np.ndarray:
        value = motion_dict[key]
        if value.shape[0] != 1:
            raise ValueError(f"Planner supports batch size 1, got {key} {value.shape}")
        return value[0].detach().cpu().numpy()

    dof_pos = np.asarray(batch_numpy("dof_pos"), dtype=np.float32)
    body_pos = np.asarray(batch_numpy("global_translation"), dtype=np.float32)
    body_ori_xyzw = np.asarray(batch_numpy("global_rotation"), dtype=np.float32)
    if not (len(dof_pos) == len(body_pos) == len(body_ori_xyzw)):
        raise ValueError("Reconstructed motion arrays have different frame counts")
    if skip_history < 0 or skip_history >= len(dof_pos):
        raise ValueError(
            f"skip_history must be in [0, {len(dof_pos) - 1}], got {skip_history}")

    # Derive velocity before slicing so the history/future boundary remains
    # available if the velocity convention is changed to backward difference.
    dof_vel = _forward_velocity(dof_pos, fps)
    joint_pos = mujoco_to_isaaclab_dof(dof_pos)[skip_history:]
    joint_vel = mujoco_to_isaaclab_dof(dof_vel)[skip_history:]
    body_pos = np.ascontiguousarray(body_pos[skip_history:])
    body_ori = np.ascontiguousarray(
        body_ori_xyzw[skip_history:, ..., [3, 0, 1, 2]])

    return G1MotionData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos=body_pos,
        body_ori=body_ori,
        framerate=float(fps),
    )

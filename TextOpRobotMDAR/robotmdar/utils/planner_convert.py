"""Conversions between TextOp controller state, DAR features, and G1 motion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robotmdar.utils.goal import GoalType, build_ego_goal
from robotmdar.dtype.motion import (
    motion_dict_to_feature_v3,
    quaternion_to_euler_angles,
)
from robotmdar.dtype.rotation import quat_apply


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


def mujoco_to_isaaclab_dof(values: np.ndarray) -> np.ndarray:
    """Convert 29-DoF MuJoCo values to IsaacLab order."""
    values = np.asarray(values)
    if values.shape[-1] != 29:
        raise ValueError(f"Expected 29 MuJoCo DoFs, got {values.shape}")
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

    FeatureVersion 3 stores the absolute roll/pitch and joint pose at feature
    frame ``t``, plus forward deltas from ``t`` to ``t + 1``.  The controller
    cannot provide the future pose needed by the last history feature, so its
    latest measured velocity is used as a constant-velocity estimate.  This
    keeps the latest physical pose itself in history; otherwise its root
    roll/pitch would be dropped at every replan.
    """
    if history_len < 2:
        raise ValueError(
            f"Controller history requires at least 2 features, got {history_len}")
    required_states = history_len
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

    # The model history must end at the current physical state. Append a copy
    # only to satisfy the feature converter's N+1 input contract; terminal
    # forward deltas are replaced below with a constant-velocity estimate.
    positions = positions[-history_len:]
    rotations = rotations[-history_len:]
    joints_mujoco = isaaclab_to_mujoco_dof(joints[-history_len:])
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
    if feature.shape != (1, history_len, 69):
        raise ValueError(
            f"Unexpected FeatureVersion 3 shape {tuple(feature.shape)}; "
            f"expected (1, {history_len}, 69)")

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
    feature[:, -1, 40:69] = (
        motion_dict["dof"][:, -2] - motion_dict["dof"][:, -3])

    # motion_dict_to_feature_v3 subtracts Euler yaw directly. Wrap all yaw
    # deltas at the branch cut before applying training-set normalization.
    feature[..., 4] = torch.atan2(
        torch.sin(feature[..., 4]), torch.cos(feature[..., 4]))
    return val_data.normalize(feature), abs_pose


def state_to_ego_goal(state_msg: Any,
                      device: str | torch.device,
                      goal_type: GoalType | str = GoalType.ROOT,
                      goal_reference_path: str | Path | None = None
                      ) -> torch.Tensor:
    """Convert the root goal relative to the current history-feature pose."""
    reference_pos = torch.tensor(
        state_msg.raw["g1_pos"][-1], dtype=torch.float32,
        device=device).reshape(1, 3)
    reference_rot_np = _normalized_quaternions_xyzw(
        np.asarray(state_msg.raw["g1_root_rot"][-1:], dtype=np.float32))
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
                              goal_reference_path: str | Path | None = None
                              ) -> torch.Tensor:
    """Convert the state goal relative to an explicit generated-history pose."""
    goal_type = GoalType.parse(goal_type)
    goal_keypoints_world = None

    if (goal_type is not GoalType.BODY
            and (state_msg.goal_root_pos_world is None
                 or state_msg.goal_yaw_world is None)):
        raise ValueError("TextOp state is missing its root-heading goal")

    if goal_type.uses_keypoints:
        raw = getattr(state_msg, 'raw', {})
        state_keypoints_world = getattr(
            state_msg, 'goal_keypoints_world', None)
        if state_keypoints_world is None:
            state_keypoints_world = raw.get('goal_keypoints_world')
        if state_keypoints_world is None:
            if goal_reference_path is None:
                raise ValueError(
                    f"{goal_type.value} goal requires controller "
                    "goal_keypoints_world or "
                    "goal_reference_path")
            state_keypoints_world = load_goal_keypoints_from_reference(
                goal_reference_path,
                np.asarray(state_msg.goal_root_pos_world, dtype=np.float32),
                float(np.asarray(state_msg.goal_yaw_world).reshape(-1)[0]),
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
    else:
        goal_pos_world = torch.tensor(
            state_msg.goal_root_pos_world, dtype=torch.float32,
            device=device).reshape(1, 3)
        goal_yaw_world = torch.tensor(
            state_msg.goal_yaw_world, dtype=torch.float32,
            device=device).reshape(1)

    world_root_velocity = None
    timestep = None
    if goal_type is GoalType.BODY_EXT:
        state_velocity = getattr(
            state_msg, 'goal_root_velocity_world', None)
        goal_timestamp_ns = getattr(state_msg, 'goal_timestamp_ns', None)
        timestamps_ns = getattr(state_msg, 'timestamps_ns', None)
        if state_velocity is None:
            raise ValueError(
                "body_ext goal requires goal_root_velocity_world [3]")
        if goal_timestamp_ns is None:
            raise ValueError("body_ext goal requires goal_timestamp_ns")
        if not timestamps_ns:
            raise ValueError("body_ext goal requires controller timestamps_ns")
        state_velocity = np.asarray(state_velocity, dtype=np.float32)
        if state_velocity.shape != (3,):
            raise ValueError(
                "body_ext goal_root_velocity_world must have shape (3,), got "
                f"{state_velocity.shape}")
        if not np.isfinite(state_velocity).all():
            raise ValueError(
                "body_ext goal_root_velocity_world must be finite")
        world_root_velocity = torch.as_tensor(
            state_velocity, dtype=torch.float32, device=device).reshape(1, 3)
        remaining_seconds = (
            int(goal_timestamp_ns) - int(timestamps_ns[-1])) / 1e9
        timestep = torch.tensor(
            [[remaining_seconds]], dtype=torch.float32, device=device)

    return build_ego_goal(
        goal_pos_world, goal_yaw_world, reference_pos.to(device),
        reference_rot.to(device), goal_type=goal_type,
        world_goal_keypoints=goal_keypoints_world,
        world_root_velocity=world_root_velocity, timestep=timestep)


def align_generated_history_pose(abs_pose: dict,
                                 generated_reference_pos: torch.Tensor,
                                 generated_reference_rot: torch.Tensor,
                                 state_msg: Any,
                                 device: str | torch.device,
                                 history_motion: torch.Tensor | None = None,
                                 val_data: Any = None):
    """Translate and rotate generated history so its reference pose matches the real G1 root.

    When *history_motion* and *val_data* are provided the per-frame roll, pitch
    and delta_yaw features are also rotated so that **every** reconstructed
    frame (not just the starting abs_pose) carries the correction.
    """
    from robotmdar.dtype.rotation import (
        euler_angles_to_quaternion,
        get_euler_xyz,
        quat_apply,
        quat_inverse,
        quat_mul,
    )

    real_current_pos = torch.tensor(
        state_msg.raw["g1_pos"][-1], dtype=torch.float32,
        device=device).reshape(1, 3)
    # Normalise the real quaternion defensively (ZMQ decoding may produce
    # views with non-unit norm).
    real_current_rot_q = _normalized_quaternions_xyzw(
        np.asarray(state_msg.raw["g1_root_rot"][-1:], dtype=np.float32))
    real_current_rot = torch.as_tensor(
        real_current_rot_q, dtype=torch.float32, device=device).reshape(1, 4)

    generated_reference_pos = generated_reference_pos.to(device).reshape(1, 3)
    generated_reference_rot = generated_reference_rot.to(device).reshape(1, 4)

    # Rotation delta: q_delta = q_real * q_gen^{-1}
    q_gen_inv = quat_inverse(generated_reference_rot, w_last=True)
    q_delta = quat_mul(real_current_rot, q_gen_inv, w_last=True)

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

    # ------------------------------------------------------------------
    # Also rotate every history-frame feature so the full reconstructed
    # trajectory carries the correction (not just frame 0 via abs_pose).
    #
    # The roll / pitch sincos are **absolute** per frame and delta_yaw is
    # a forward difference — all three must be updated.  Likewise
    # delta_trans_local lives in the per-frame yaw-aligned basis and
    # needs to be re-expressed after the rotation correction.
    # ------------------------------------------------------------------
    aligned_history_motion = history_motion
    if history_motion is not None and val_data is not None:
        raw = val_data.denormalize(
            history_motion.to(device))          # (B, T, 69)
        B, T = raw.shape[:2]

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

    # goal reference pose is the *real* G1 pose so ego-goal is computed
    # relative to where the robot actually is.
    return (aligned_abs_pose,
            real_current_pos,       # goal_reference_pos
            real_current_rot,       # goal_reference_rot
            real_current_pos - generated_reference_pos,  # translation
            aligned_history_motion)


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
                          fps: float = 50.0):
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
    dof_vel = _forward_velocity(dof_pos, fps)
    joint_pos = mujoco_to_isaaclab_dof(dof_pos)[skip_history:]
    joint_vel = mujoco_to_isaaclab_dof(dof_vel)[skip_history:]
    body_pos = _textop_bodies_to_sonic(body_pos[skip_history:])
    body_ori = np.ascontiguousarray(
        _textop_bodies_to_sonic(body_ori_xyzw[skip_history:])[
            ..., [3, 0, 1, 2]])

    return G1MotionData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos=body_pos,
        body_ori=body_ori,
        framerate=float(fps),
    )

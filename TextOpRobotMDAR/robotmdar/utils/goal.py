"""Goal condition building: goal type definitions, validation, and ego-centric goal transform."""

from enum import Enum
from pathlib import Path
from typing import Optional

import torch

from robotmdar.dtype.rotation import (
    euler_angles_to_quaternion,
    quat_mul,
    quaternion_to_euler_angles,
)


JOINT_STATE_GOAL_DOF_DIM = 29
JOINT_STATE_GOAL_DIM = 40
SPLIT_GOAL_DIM = 45
EXTENDED_BODY_GOAL_DIM = 21


class GoalType(str, Enum):
    ROOT = "root"
    BODY = "body"
    BODY_EXT = "body_ext"
    JOINT_STATE = "joint_state"

    @classmethod
    def parse(cls, value: "GoalType | str") -> "GoalType":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported goal_type {value!r}; expected one of: {choices}"
            ) from exc

    @property
    def dimension(self) -> int:
        if self is GoalType.ROOT:
            return 5
        if self is GoalType.BODY:
            return 15
        if self is GoalType.BODY_EXT:
            return EXTENDED_BODY_GOAL_DIM
        return JOINT_STATE_GOAL_DIM

    @property
    def uses_keypoints(self) -> bool:
        return self in (GoalType.BODY, GoalType.BODY_EXT)

    @property
    def uses_arrival_time(self) -> bool:
        return self in (GoalType.BODY_EXT, GoalType.JOINT_STATE)

    @property
    def uses_joint_state(self) -> bool:
        return self is GoalType.JOINT_STATE


class GoalEncoding(str, Enum):
    LEGACY40 = "legacy40"
    SINGLE = "single"
    SPLIT = "split"

    @classmethod
    def parse(cls, value: "GoalEncoding | str") -> "GoalEncoding":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported goal_encoding {value!r}; expected one of: {choices}"
            ) from exc

    @property
    def dimension(self) -> int:
        return JOINT_STATE_GOAL_DIM if self is GoalEncoding.LEGACY40 else SPLIT_GOAL_DIM

    @property
    def token_count(self) -> int:
        return 4 if self is GoalEncoding.SPLIT else 1

    @property
    def uses_split_statistics(self) -> bool:
        return self is not GoalEncoding.LEGACY40


def _parse_goal_offset_range(goal_offset_range):
    if goal_offset_range is None:
        return None
    if len(goal_offset_range) != 2:
        raise ValueError(
            "goal_offset_range must contain exactly two bounds")
    return tuple(int(value) for value in goal_offset_range)


def validate_goal_config(
    goal_type: GoalType | str,
    goal_dim: int,
    goal_encoding: GoalEncoding | str | None = None,
    *,
    dof_dim: int | None = None,
    goal_offset_range=None,
    goal_timestep_mode: str | None = None,
    goal_stats: Optional[dict] = None,
) -> GoalType:
    """Validate and return the configured goal type."""
    parsed = GoalType.parse(goal_type)
    parsed_encoding = (
        GoalEncoding.parse(goal_encoding)
        if goal_encoding is not None else None
    )
    if parsed is not GoalType.JOINT_STATE:
        if parsed_encoding in (GoalEncoding.SINGLE, GoalEncoding.SPLIT):
            raise ValueError(
                f"goal_encoding={parsed_encoding.value!r} is only valid with "
                "goal_type='joint_state'"
            )
        if int(goal_dim) != parsed.dimension:
            raise ValueError(
                f"goal_type={parsed.value!r} requires goal_dim="
                f"{parsed.dimension}, got {goal_dim}"
            )
        return parsed

    if dof_dim is not None and int(dof_dim) != JOINT_STATE_GOAL_DOF_DIM:
        raise ValueError(
            "goal_type=joint_state requires dof_dim=29, got "
            f"{dof_dim}"
        )

    if parsed_encoding is None:
        if int(goal_dim) == JOINT_STATE_GOAL_DIM:
            parsed_encoding = GoalEncoding.LEGACY40
        elif int(goal_dim) == SPLIT_GOAL_DIM:
            raise ValueError(
                "goal_encoding is required when goal_dim=45")
        else:
            raise ValueError(
                f"goal_type={parsed.value!r} requires goal_dim=40 or 45, "
                f"got {goal_dim}"
            )

    if parsed_encoding is GoalEncoding.LEGACY40:
        expected_dim = JOINT_STATE_GOAL_DIM
    else:
        expected_dim = SPLIT_GOAL_DIM
    if int(goal_dim) != expected_dim:
        raise ValueError(
            f"goal_encoding={parsed_encoding.value!r} requires goal_dim="
            f"{expected_dim}, got {goal_dim}"
        )

    parsed_offsets = _parse_goal_offset_range(goal_offset_range)
    if (parsed_encoding.uses_split_statistics and parsed_offsets is not None
            and parsed_offsets != (0, 0)):
        mode = str(goal_timestep_mode or "relative").lower()
        if mode != "relative":
            raise ValueError(
                "joint_state split/single goals with randomized offsets "
                "require goal_timestep_mode='relative'"
            )

    if goal_stats is not None and parsed_encoding.uses_split_statistics:
        validate_goal_stats(
            goal_stats,
            goal_encoding=parsed_encoding,
            goal_offset_range=parsed_offsets,
            goal_timestep_mode=goal_timestep_mode,
        )

    return parsed


def quaternion_yaw(quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    """Return Z-axis yaw from xyzw quaternions."""
    x, y, z, w = quaternion_xyzw.unbind(dim=-1)
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y.square() + z.square())
    return torch.atan2(sin_yaw, cos_yaw)


def _world_to_ego(world_delta: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame XY offsets into the ego-centric X-forward frame.

    ``world_delta`` has shape ``[..., 3]``. Yaw dimensions align with its
    leading dimensions from the left; omitted point/sequence axes are appended
    as singleton dimensions. Redundant trailing singleton yaw axes are accepted.
    """
    if world_delta.ndim < 1 or world_delta.shape[-1] != 3:
        raise ValueError(
            f"world_delta must have shape [..., 3], got {tuple(world_delta.shape)}"
        )

    target_shape = world_delta.shape[:-1]
    while yaw.ndim > len(target_shape) and yaw.shape[-1] == 1:
        yaw = yaw.squeeze(-1)
    if yaw.ndim > len(target_shape):
        raise ValueError(
            f"yaw shape {tuple(yaw.shape)} has more dimensions than world_delta "
            f"leading shape {tuple(target_shape)}"
        )
    for axis, (yaw_size, target_size) in enumerate(zip(yaw.shape, target_shape)):
        if yaw_size not in (1, target_size):
            raise ValueError(
                f"yaw shape {tuple(yaw.shape)} is incompatible with world_delta "
                f"leading shape {tuple(target_shape)} at axis {axis}"
            )

    broadcast_shape = (*yaw.shape, *((1,) * (len(target_shape) - yaw.ndim)))
    cos = torch.cos(yaw).reshape(broadcast_shape)
    sin = torch.sin(yaw).reshape(broadcast_shape)
    x = world_delta[..., 0]
    y = world_delta[..., 1]
    z = world_delta[..., 2]
    ego_x = x * cos + y * sin
    ego_y = -x * sin + y * cos
    return torch.stack((ego_x, ego_y, z), dim=-1)


def _require_shape(name: str, value: Optional[torch.Tensor],
                   expected_shape: tuple[int, ...]) -> torch.Tensor:
    if value is None:
        raise ValueError(f"{name} is required for goal_type='joint_state'")
    if tuple(value.shape[-len(expected_shape):]) != expected_shape:
        raise ValueError(
            f"{name} must have shape [..., {', '.join(map(str, expected_shape))}], "
            f"got {tuple(value.shape)}"
        )
    return value


def _normalize_quaternion_xyzw(name: str,
                               quaternion_xyzw: torch.Tensor) -> torch.Tensor:
    if quaternion_xyzw.shape[-1] != 4:
        raise ValueError(
            f"{name} must have shape [..., 4], got {tuple(quaternion_xyzw.shape)}"
        )
    if not torch.isfinite(quaternion_xyzw).all():
        raise ValueError(f"{name} must be finite")
    norm = quaternion_xyzw.norm(dim=-1, keepdim=True)
    if bool((norm < 1e-6).any().item()):
        raise ValueError(f"{name} contains a zero quaternion")
    return quaternion_xyzw / norm


def _build_ego_joint_state_components(
    world_goal_pos: torch.Tensor,
    world_goal_rot: torch.Tensor,
    world_goal_dof: torch.Tensor,
    world_root_velocity: torch.Tensor,
    reference_pos: torch.Tensor,
    reference_rot: torch.Tensor,
):
    if world_goal_pos.shape[-1:] != (3,):
        raise ValueError(
            f"world_goal_pos must have shape [..., 3], got "
            f"{tuple(world_goal_pos.shape)}"
        )
    if reference_pos.shape != world_goal_pos.shape:
        raise ValueError(
            "reference_pos must match world_goal_pos shape, got "
            f"{tuple(reference_pos.shape)} != {tuple(world_goal_pos.shape)}"
        )
    world_goal_rot = _require_shape(
        "world_goal_rot", world_goal_rot, (4,))
    if world_goal_rot.shape[:-1] != world_goal_pos.shape[:-1]:
        raise ValueError(
            "world_goal_rot batch dimensions must match world_goal_pos, got "
            f"{tuple(world_goal_rot.shape[:-1])} != "
            f"{tuple(world_goal_pos.shape[:-1])}"
        )
    reference_rot = _require_shape("reference_rot", reference_rot, (4,))
    if reference_rot.shape[:-1] != world_goal_pos.shape[:-1]:
        raise ValueError(
            "reference_rot batch dimensions must match world_goal_pos, got "
            f"{tuple(reference_rot.shape[:-1])} != "
            f"{tuple(world_goal_pos.shape[:-1])}"
        )
    world_goal_dof = _require_shape(
        "world_goal_dof", world_goal_dof, (JOINT_STATE_GOAL_DOF_DIM,))
    if world_goal_dof.shape[:-1] != world_goal_pos.shape[:-1]:
        raise ValueError(
            "world_goal_dof batch dimensions must match world_goal_pos, got "
            f"{tuple(world_goal_dof.shape[:-1])} != "
            f"{tuple(world_goal_pos.shape[:-1])}"
        )
    world_root_velocity = _require_shape(
        "world_root_velocity", world_root_velocity, (3,))
    if world_root_velocity.shape != world_goal_pos.shape:
        raise ValueError(
            "world_root_velocity must match world_goal_pos shape, got "
            f"{tuple(world_root_velocity.shape)} != "
            f"{tuple(world_goal_pos.shape)}"
        )

    reference_rot = _normalize_quaternion_xyzw("reference_rot", reference_rot)
    world_goal_rot = _normalize_quaternion_xyzw(
        "world_goal_rot", world_goal_rot)
    current_yaw = quaternion_yaw(reference_rot)

    ego_root = _world_to_ego(world_goal_pos - reference_pos, current_yaw)
    ego_velocity = _world_to_ego(world_root_velocity, current_yaw)

    zeros = torch.zeros_like(current_yaw)
    inv_reference_yaw = euler_angles_to_quaternion(
        torch.stack((zeros, zeros, -current_yaw), dim=-1)
    )
    ego_goal_rot = quat_mul(inv_reference_yaw, world_goal_rot, w_last=True)
    euler = quaternion_to_euler_angles(ego_goal_rot)
    roll, pitch, yaw = euler.unbind(dim=-1)
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    orientation = torch.stack(
        (
            torch.sin(roll),
            torch.cos(roll) - 1.0,
            torch.sin(pitch),
            torch.cos(pitch) - 1.0,
            yaw,
        ),
        dim=-1,
    )
    return ego_root, orientation, world_goal_dof, ego_velocity


def build_ego_joint_state_goal(
    world_goal_pos: torch.Tensor,
    world_goal_rot: torch.Tensor,
    world_goal_dof: torch.Tensor,
    world_root_velocity: torch.Tensor,
    reference_pos: torch.Tensor,
    reference_rot: torch.Tensor,
) -> torch.Tensor:
    """Express a GT-frame 29-DOF goal state in the reference ego frame.

    Layout:
        root_pos_ego(3) |
        sin/cos-minus-one roll-pitch + relative yaw(5) |
        q_goal(29) |
        root_vel_ego(3)
    """
    ego_root, orientation, world_goal_dof, ego_velocity = (
        _build_ego_joint_state_components(
            world_goal_pos=world_goal_pos,
            world_goal_rot=world_goal_rot,
            world_goal_dof=world_goal_dof,
            world_root_velocity=world_root_velocity,
            reference_pos=reference_pos,
            reference_rot=reference_rot,
        )
    )
    return torch.cat(
        (ego_root, orientation, world_goal_dof, ego_velocity), dim=-1)


def build_ego_split_goal(
    world_goal_pos: torch.Tensor,
    world_goal_rot: torch.Tensor,
    world_goal_dof: torch.Tensor,
    world_root_velocity: torch.Tensor,
    reference_pos: torch.Tensor,
    reference_rot: torch.Tensor,
    time_to_arrival_seconds: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    """Express a GT-frame joint_state goal as the 45-D split feature vector."""
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    ego_root, orientation, world_goal_dof, ego_velocity = (
        _build_ego_joint_state_components(
            world_goal_pos=world_goal_pos,
            world_goal_rot=world_goal_rot,
            world_goal_dof=world_goal_dof,
            world_root_velocity=world_root_velocity,
            reference_pos=reference_pos,
            reference_rot=reference_rot,
        )
    )
    if time_to_arrival_seconds is None:
        raise ValueError(
            "time_to_arrival_seconds is required for split/single goals")
    if time_to_arrival_seconds.ndim > 1:
        time_to_arrival_seconds = time_to_arrival_seconds.squeeze(-1)
    time_to_arrival_seconds = time_to_arrival_seconds.to(
        device=ego_root.device, dtype=ego_root.dtype
    )
    if time_to_arrival_seconds.shape != ego_root.shape[:-1]:
        raise ValueError(
            "time_to_arrival_seconds batch dimensions must match "
            f"world_goal_pos, got {tuple(time_to_arrival_seconds.shape)}"
        )

    root_xy = ego_root[..., :2]
    r_xy = torch.linalg.vector_norm(root_xy, dim=-1, keepdim=True)
    log_r_xy = torch.log1p(r_xy)
    T_eff = torch.clamp(time_to_arrival_seconds, min=1.0 / float(fps))
    urgency = torch.where(
        time_to_arrival_seconds[..., None] > 0.0,
        ego_root / T_eff[..., None],
        torch.zeros_like(ego_root),
    )
    f_trans = torch.cat(
        (ego_root, r_xy, log_r_xy, urgency),
        dim=-1,
    )
    return torch.cat(
        (f_trans, orientation, world_goal_dof, ego_velocity), dim=-1)


def scale_goal(goal: torch.Tensor, goal_stats: dict) -> torch.Tensor:
    """Scale the 45-D split goal with frozen dataset factors."""
    if goal.shape[-1] != SPLIT_GOAL_DIM:
        raise ValueError(
            f"scale_goal expects {SPLIT_GOAL_DIM}-D split goals, got "
            f"{tuple(goal.shape)}"
        )
    if goal_stats is None:
        raise ValueError("goal_stats is required to scale split goals")

    scaled = goal.clone()
    meta = goal_stats.get("meta", {})
    if meta.get("goal_dim", SPLIT_GOAL_DIM) != SPLIT_GOAL_DIM:
        raise ValueError(
            f"goal_stats were computed for goal_dim={meta.get('goal_dim')}, "
            f"expected {SPLIT_GOAL_DIM}"
        )

    s_p = torch.as_tensor(goal_stats["s_p"], device=goal.device, dtype=goal.dtype)
    s_l = torch.as_tensor(goal_stats["s_l"], device=goal.device, dtype=goal.dtype)
    s_v = torch.as_tensor(goal_stats["s_v"], device=goal.device, dtype=goal.dtype)
    s_o = torch.as_tensor(goal_stats["s_o"], device=goal.device, dtype=goal.dtype)
    q_mean = torch.as_tensor(goal_stats["q_mean"], device=goal.device, dtype=goal.dtype)
    q_std = torch.as_tensor(goal_stats["q_std"], device=goal.device, dtype=goal.dtype)

    scaled[..., 0:4] = scaled[..., 0:4] * s_p
    scaled[..., 4:5] = scaled[..., 4:5] * s_l
    scaled[..., 5:8] = scaled[..., 5:8] * s_v
    scaled[..., 8:13] = scaled[..., 8:13] * s_o
    scaled[..., 13:42] = (scaled[..., 13:42] - q_mean) / q_std.clamp_min(1e-6)
    scaled[..., 42:45] = scaled[..., 42:45] * s_v
    return scaled


def validate_goal_stats(
    goal_stats: dict,
    *,
    goal_encoding: GoalEncoding | str,
    goal_offset_range=None,
    goal_per_primitive: bool | None = None,
    future_len: int | None = None,
    fps: float | None = None,
    goal_timestep_mode: str | None = None,
    datadir: str | None = None,
) -> dict:
    """Validate the cached goal statistics against the active config."""
    if goal_stats is None:
        raise ValueError("goal_stats is required")
    parsed_encoding = GoalEncoding.parse(goal_encoding)
    if parsed_encoding is GoalEncoding.LEGACY40:
        return goal_stats

    meta = goal_stats.get("meta", {})
    mismatches = []
    parsed_offsets = _parse_goal_offset_range(goal_offset_range)
    if parsed_offsets is not None:
        stored_offsets_raw = meta.get("goal_offset_range")
        stored_offsets = (
            tuple(int(v) for v in stored_offsets_raw)
            if stored_offsets_raw is not None else None
        )
        if stored_offsets != parsed_offsets:
            mismatches.append(
                f"goal_offset_range: expected {parsed_offsets}, got "
                f"{stored_offsets}"
            )
    if goal_per_primitive is not None and bool(meta.get("goal_per_primitive")) != bool(goal_per_primitive):
        mismatches.append(
            f"goal_per_primitive: expected {bool(goal_per_primitive)}, got "
            f"{meta.get('goal_per_primitive')}"
        )
    if future_len is not None and int(meta.get("future_len", -1)) != int(future_len):
        mismatches.append(
            f"future_len: expected {int(future_len)}, got {meta.get('future_len')}"
        )
    if fps is not None and abs(float(meta.get("fps", float("nan"))) - float(fps)) > 1e-6:
        mismatches.append(
            f"fps: expected {float(fps)}, got {meta.get('fps')}"
        )
    if goal_timestep_mode is not None:
        stored_mode = str(meta.get("goal_timestep_mode", "")).lower()
        if stored_mode != str(goal_timestep_mode).lower():
            mismatches.append(
                f"goal_timestep_mode: expected {goal_timestep_mode!r}, got "
                f"{meta.get('goal_timestep_mode')!r}"
            )
    encodings = tuple(str(item).lower() for item in meta.get("encodings", ()))
    if parsed_encoding.value not in encodings:
        mismatches.append(
            f"encodings: expected {parsed_encoding.value!r} in {encodings!r}"
        )
    if datadir is not None:
        stored_datadir = str(meta.get("dataset_path", ""))
        if stored_datadir:
            if Path(stored_datadir).resolve() != Path(datadir).resolve():
                mismatches.append(
                    f"dataset_path: expected {datadir!r}, got "
                    f"{stored_datadir!r}"
                )
    if mismatches:
        raise ValueError(
            "goal_stats meta does not match the active config: "
            + "; ".join(mismatches)
        )
    return goal_stats


def build_ego_goal(world_goal_pos: torch.Tensor,
                   world_goal_yaw: torch.Tensor,
                   reference_pos: torch.Tensor,
                   reference_rot: torch.Tensor,
                   goal_type: GoalType | str = GoalType.ROOT,
                   goal_encoding: GoalEncoding | str | None = None,
                   goal_stats: Optional[dict] = None,
                   fps: Optional[float] = None,
                   world_goal_keypoints: Optional[torch.Tensor] = None,
                   world_root_velocity: Optional[torch.Tensor] = None,
                   timestep: Optional[torch.Tensor] = None,
                   time_to_arrival_seconds: Optional[torch.Tensor] = None,
                   world_goal_rot: Optional[torch.Tensor] = None,
                   world_goal_dof: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Express a world-space root or body goal in the local X-forward frame."""
    goal_type = GoalType.parse(goal_type)
    parsed_encoding = (
        GoalEncoding.parse(goal_encoding)
        if goal_encoding is not None else None
    )
    current_yaw = quaternion_yaw(reference_rot)

    if goal_type is GoalType.JOINT_STATE:
        encoding = parsed_encoding or GoalEncoding.LEGACY40
        if encoding is GoalEncoding.LEGACY40:
            return build_ego_joint_state_goal(
                world_goal_pos=world_goal_pos,
                world_goal_rot=world_goal_rot,
                world_goal_dof=world_goal_dof,
                world_root_velocity=world_root_velocity,
                reference_pos=reference_pos,
                reference_rot=reference_rot,
            )
        if goal_stats is None:
            raise ValueError(
                "goal_stats is required for split/single joint_state goals"
            )
        if time_to_arrival_seconds is None:
            time_to_arrival_seconds = timestep
        if time_to_arrival_seconds is None:
            raise ValueError(
                "time_to_arrival_seconds is required for split/single "
                "joint_state goals"
            )
        resolved_fps = float(
            fps if fps is not None else goal_stats.get("meta", {}).get("fps", 0.0)
        )
        if resolved_fps <= 0:
            raise ValueError(
                "fps is required for split/single joint_state goals"
            )
        split_goal = build_ego_split_goal(
            world_goal_pos=world_goal_pos,
            world_goal_rot=world_goal_rot,
            world_goal_dof=world_goal_dof,
            world_root_velocity=world_root_velocity,
            reference_pos=reference_pos,
            reference_rot=reference_rot,
            time_to_arrival_seconds=time_to_arrival_seconds,
            fps=resolved_fps,
        )
        return scale_goal(split_goal, goal_stats)

    if goal_type is GoalType.ROOT:
        ego_root = _world_to_ego(world_goal_pos - reference_pos, current_yaw)
        delta_yaw = world_goal_yaw - current_yaw
        return torch.cat(
            (ego_root, torch.stack((torch.cos(delta_yaw), torch.sin(delta_yaw)),
                                   dim=-1)),
            dim=-1,
        )

    if world_goal_keypoints is None:
        raise ValueError(
            "world_goal_keypoints is required for body goal types")
    num_keypoints = 4 if goal_type is GoalType.BODY_EXT else 5
    if world_goal_keypoints.shape[-2:] != (num_keypoints, 3):
        raise ValueError(
            f"{goal_type.value} goal keypoints must have shape "
            f"[..., {num_keypoints}, 3], got "
            f"{tuple(world_goal_keypoints.shape)}"
        )
    expected_prefix = reference_pos.shape[:-1]
    if world_goal_keypoints.shape[:-2] != expected_prefix:
        raise ValueError(
            "Body goal batch dimensions must match reference_pos: "
            f"{tuple(world_goal_keypoints.shape[:-2])} != "
            f"{tuple(expected_prefix)}"
        )

    keypoint_delta = world_goal_keypoints - reference_pos.unsqueeze(-2)
    ego_keypoints = _world_to_ego(keypoint_delta, current_yaw).flatten(-2)
    if goal_type is GoalType.BODY:
        return ego_keypoints

    if world_root_velocity is None:
        raise ValueError(
            "world_root_velocity is required for goal_type='body_ext'")
    if world_root_velocity.shape != world_goal_pos.shape:
        raise ValueError(
            "world_root_velocity must match world_goal_pos shape, got "
            f"{tuple(world_root_velocity.shape)} != "
            f"{tuple(world_goal_pos.shape)}"
        )
    if timestep is None:
        raise ValueError("timestep is required for goal_type='body_ext'")
    expected_time_shape = (*world_goal_pos.shape[:-1], 1)
    if timestep.shape != expected_time_shape:
        raise ValueError(
            f"timestep must have shape {expected_time_shape}, got "
            f"{tuple(timestep.shape)}"
        )

    ego_root = _world_to_ego(world_goal_pos - reference_pos, current_yaw)
    delta_yaw = world_goal_yaw - current_yaw
    ego_yaw = torch.stack((torch.cos(delta_yaw), torch.sin(delta_yaw)), dim=-1)
    ego_velocity = _world_to_ego(world_root_velocity, current_yaw)
    return torch.cat(
        (ego_root, ego_yaw, ego_velocity, timestep, ego_keypoints), dim=-1
    )

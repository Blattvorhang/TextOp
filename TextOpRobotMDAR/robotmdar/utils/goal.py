"""Goal condition building: goal type definitions, validation, and ego-centric goal transform."""

from enum import Enum
from typing import Optional

import torch

from robotmdar.dtype.rotation import (
    euler_angles_to_quaternion,
    quat_mul,
    quaternion_to_euler_angles,
)


JOINT_STATE_GOAL_DOF_DIM = 29
JOINT_STATE_GOAL_DIM = 40
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


def validate_goal_config(goal_type: GoalType | str, goal_dim: int) -> GoalType:
    """Validate and return the configured goal type."""
    parsed = GoalType.parse(goal_type)
    if int(goal_dim) != parsed.dimension:
        raise ValueError(
            f"goal_type={parsed.value!r} requires goal_dim={parsed.dimension}, "
            f"got {goal_dim}"
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
    return torch.cat(
        (ego_root, orientation, world_goal_dof, ego_velocity), dim=-1)


def build_ego_goal(world_goal_pos: torch.Tensor,
                   world_goal_yaw: torch.Tensor,
                   reference_pos: torch.Tensor,
                   reference_rot: torch.Tensor,
                   goal_type: GoalType | str = GoalType.ROOT,
                   world_goal_keypoints: Optional[torch.Tensor] = None,
                   world_root_velocity: Optional[torch.Tensor] = None,
                   timestep: Optional[torch.Tensor] = None,
                   world_goal_rot: Optional[torch.Tensor] = None,
                   world_goal_dof: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Express a world-space root or body goal in the local X-forward frame."""
    goal_type = GoalType.parse(goal_type)
    current_yaw = quaternion_yaw(reference_rot)

    if goal_type is GoalType.JOINT_STATE:
        return build_ego_joint_state_goal(
            world_goal_pos=world_goal_pos,
            world_goal_rot=world_goal_rot,
            world_goal_dof=world_goal_dof,
            world_root_velocity=world_root_velocity,
            reference_pos=reference_pos,
            reference_rot=reference_rot,
        )

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

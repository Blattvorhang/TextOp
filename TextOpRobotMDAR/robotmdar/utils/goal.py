"""Goal condition building: goal type definitions, validation, and ego-centric goal transform."""

from enum import Enum
from typing import Optional

import torch


class GoalType(str, Enum):
    ROOT = "root"
    BODY = "body"

    @classmethod
    def parse(cls, value: "GoalType | str") -> "GoalType":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported goal_type {value!r}; expected one of: {choices}"
            ) from exc

    @property
    def dimension(self) -> int:
        return 5 if self is GoalType.ROOT else 15


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


def build_ego_goal(world_goal_pos: torch.Tensor,
                   world_goal_yaw: torch.Tensor,
                   reference_pos: torch.Tensor,
                   reference_rot: torch.Tensor,
                   goal_type: GoalType | str = GoalType.ROOT,
                   goal_keypoints: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Express a world-space root or body goal in the local X-forward frame."""
    goal_type = GoalType.parse(goal_type)
    current_yaw = quaternion_yaw(reference_rot)
    cos_yaw = torch.cos(current_yaw)
    sin_yaw = torch.sin(current_yaw)

    if goal_type is GoalType.ROOT:
        delta = world_goal_pos - reference_pos
        ego_x = delta[..., 0] * cos_yaw + delta[..., 1] * sin_yaw
        ego_y = -delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw
        delta_yaw = world_goal_yaw - current_yaw
        return torch.stack(
            (ego_x, ego_y, delta[..., 2], torch.cos(delta_yaw),
             torch.sin(delta_yaw)),
            dim=-1,
        )

    if goal_keypoints is None:
        raise ValueError("goal_keypoints is required for goal_type='body'")
    if goal_keypoints.shape[-2:] != (5, 3):
        raise ValueError(
            "Body goal keypoints must have shape [..., 5, 3], got "
            f"{tuple(goal_keypoints.shape)}"
        )
    expected_prefix = reference_pos.shape[:-1]
    if goal_keypoints.shape[:-2] != expected_prefix:
        raise ValueError(
            "Body goal batch dimensions must match reference_pos: "
            f"{tuple(goal_keypoints.shape[:-2])} != {tuple(expected_prefix)}"
        )

    delta = goal_keypoints - reference_pos.unsqueeze(-2)
    cos_yaw = cos_yaw.unsqueeze(-1)
    sin_yaw = sin_yaw.unsqueeze(-1)
    ego_x = delta[..., 0] * cos_yaw + delta[..., 1] * sin_yaw
    ego_y = -delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw
    return torch.stack((ego_x, ego_y, delta[..., 2]), dim=-1).flatten(-2)

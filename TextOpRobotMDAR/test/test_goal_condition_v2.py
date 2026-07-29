from types import SimpleNamespace

import numpy as np
import pytest
import torch

from TextOpRobotMDAR.robotmdar.utils.goal import (
    GoalType,
    build_ego_goal,
    validate_goal_config,
)
from TextOpRobotMDAR.robotmdar.utils.planner_convert import (
    load_goal_keypoints_from_reference,
    state_goal_from_reference,
)


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((*yaw.shape, 4), dtype=torch.float32)
    result[..., 2] = torch.sin(yaw / 2)
    result[..., 3] = torch.cos(yaw / 2)
    return result


def test_body_goal_is_batched_relative_xyz():
    reference_pos = torch.tensor([
        [1.0, 2.0, 0.7],
        [10.0, 20.0, 0.8],
    ])
    reference_rot = _yaw_quaternion(
        torch.tensor([0.0, np.pi / 2], dtype=torch.float32))
    offsets = torch.tensor([
        [2.0, 0.0, 0.1],
        [1.0, 1.0, -0.6],
        [1.0, -1.0, -0.6],
        [0.0, 2.0, 0.2],
        [0.0, -2.0, 0.2],
    ])
    keypoints = reference_pos[:, None, :] + offsets[None, :, :]

    goal = build_ego_goal(
        keypoints[:, 0], torch.zeros(2), reference_pos, reference_rot,
        goal_type=GoalType.BODY, goal_keypoints=keypoints)

    assert goal.shape == (2, 15)
    torch.testing.assert_close(goal[0].reshape(5, 3), offsets)
    expected_rotated = torch.stack(
        (offsets[:, 1], -offsets[:, 0], offsets[:, 2]), dim=-1)
    torch.testing.assert_close(
        goal[1].reshape(5, 3), expected_rotated, atol=1e-6, rtol=0)


def test_goal_configuration_requires_matching_dimension():
    assert validate_goal_config("root", 5) is GoalType.ROOT
    assert validate_goal_config("body", 15) is GoalType.BODY
    with pytest.raises(ValueError, match="requires goal_dim=15"):
        validate_goal_config("body", 5)


def test_reference_pose_transforms_xy_and_preserves_z(tmp_path):
    template = np.asarray([
        [0.0, 0.0, 0.77],
        [0.1, 0.2, 0.05],
        [0.1, -0.2, 0.05],
        [0.2, 0.3, 0.6],
        [0.2, -0.3, 0.6],
    ], dtype=np.float32)
    path = tmp_path / "stand.npz"
    np.savez(path, keypoints=template)

    transformed = load_goal_keypoints_from_reference(
        path, np.asarray([3.0, 4.0, 0.77], dtype=np.float32), np.pi / 2)

    np.testing.assert_allclose(transformed[:, 2], template[:, 2])
    np.testing.assert_allclose(
        transformed[:, :2],
        template[:, [1, 0]] * np.asarray([-1.0, 1.0]) + [3.0, 4.0],
        atol=1e-6)


def test_state_body_goal_prefers_controller_keypoints():
    keypoints = np.asarray([
        [2.0, 3.0, 0.8],
        [2.1, 3.2, 0.1],
        [2.1, 2.8, 0.1],
        [2.2, 3.3, 0.6],
        [2.2, 2.7, 0.6],
    ], dtype=np.float32)
    state = SimpleNamespace(
        raw={"goal_keypoints": keypoints},
        goal_root_pos=None,
        goal_heading=None,
    )
    goal = state_goal_from_reference(
        state,
        reference_pos=torch.tensor([[1.0, 2.0, 0.7]]),
        reference_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        device="cpu",
        goal_type="body",
    )

    torch.testing.assert_close(
        goal.reshape(5, 3),
        torch.from_numpy(keypoints) - torch.tensor([1.0, 2.0, 0.7]))

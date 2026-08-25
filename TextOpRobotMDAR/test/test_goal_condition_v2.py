from types import SimpleNamespace

import numpy as np
import pytest
import torch

from TextOpRobotMDAR.robotmdar.utils.goal import (
    GoalType,
    _world_to_ego,
    build_ego_goal,
    build_ego_joint_state_goal,
    validate_goal_config,
)
from TextOpRobotMDAR.robotmdar.utils.planner_convert import (
    load_goal_keypoints_from_reference,
    state_goal_from_reference,
)
from TextOpRobotMDAR.robotmdar.dtype.rotation import (
    euler_angles_to_quaternion,
    quat_mul,
)


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((*yaw.shape, 4), dtype=torch.float32)
    result[..., 2] = torch.sin(yaw / 2)
    result[..., 3] = torch.cos(yaw / 2)
    return result


def test_world_to_ego_accepts_column_yaw_without_cross_batch_broadcast():
    world_delta = torch.tensor([
        [1.0, 0.0, 0.5],
        [0.0, 2.0, -0.5],
    ])
    yaw = torch.tensor([[np.pi / 2], [0.0]])

    result = _world_to_ego(world_delta, yaw)

    assert result.shape == world_delta.shape
    torch.testing.assert_close(
        result,
        torch.tensor([[0.0, -1.0, 0.5], [0.0, 2.0, -0.5]]),
        atol=1e-6,
        rtol=0,
    )


def test_world_to_ego_broadcasts_batch_yaw_over_keypoints():
    world_delta = torch.tensor([
        [[1.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
        [[1.0, 0.0, 2.0], [0.0, 2.0, 3.0]],
    ])
    yaw = torch.tensor([np.pi / 2, 0.0])

    result = _world_to_ego(world_delta, yaw)

    torch.testing.assert_close(
        result[0],
        torch.tensor([[0.0, -1.0, 0.0], [2.0, 0.0, 1.0]]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(result[1], world_delta[1])


def test_world_to_ego_rejects_invalid_shapes():
    with pytest.raises(ValueError, match=r"shape \[\.\.\., 3\]"):
        _world_to_ego(torch.zeros((2, 4)), torch.zeros(2))
    with pytest.raises(ValueError, match="incompatible"):
        _world_to_ego(torch.zeros((2, 3, 3)), torch.zeros((4, 1)))


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
        goal_type=GoalType.BODY, world_goal_keypoints=keypoints)

    assert goal.shape == (2, 15)
    torch.testing.assert_close(goal[0].reshape(5, 3), offsets)
    expected_rotated = torch.stack(
        (offsets[:, 1], -offsets[:, 0], offsets[:, 2]), dim=-1)
    torch.testing.assert_close(
        goal[1].reshape(5, 3), expected_rotated, atol=1e-6, rtol=0)


def test_goal_configuration_requires_matching_dimension():
    assert validate_goal_config("root", 5) is GoalType.ROOT
    assert validate_goal_config("body", 15) is GoalType.BODY
    assert validate_goal_config("body_ext", 21) is GoalType.BODY_EXT
    assert validate_goal_config("joint_state", 40) is GoalType.JOINT_STATE
    with pytest.raises(ValueError, match="requires goal_dim=15"):
        validate_goal_config("body", 5)


def test_extended_body_goal_layout_and_ego_transform():
    reference_pos = torch.tensor([[1.0, 2.0, 0.7]])
    reference_rot = _yaw_quaternion(torch.tensor([np.pi / 2]))
    world_goal_pos = torch.tensor([[3.0, 3.0, 1.0]])
    world_goal_yaw = torch.tensor([np.pi])
    world_velocity = torch.tensor([[2.0, 1.0, -0.5]])
    timestep = torch.tensor([[1.25]])
    limb_offsets = torch.tensor([
        [1.0, 0.5, -0.6],
        [1.0, -0.5, -0.6],
        [0.5, 1.0, 0.1],
        [0.5, -1.0, 0.1],
    ])
    keypoints = reference_pos[:, None, :] + limb_offsets[None]

    goal = build_ego_goal(
        world_goal_pos,
        world_goal_yaw,
        reference_pos,
        reference_rot,
        goal_type=GoalType.BODY_EXT,
        world_goal_keypoints=keypoints,
        world_root_velocity=world_velocity,
        timestep=timestep,
    )

    assert goal.shape == (1, 21)
    torch.testing.assert_close(goal[0, 0:3], torch.tensor([1.0, -2.0, 0.3]))
    torch.testing.assert_close(goal[0, 3:5], torch.tensor([0.0, 1.0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(goal[0, 5:8], torch.tensor([1.0, -2.0, -0.5]))
    torch.testing.assert_close(goal[0, 8:9], timestep[0])
    expected_limbs = torch.stack(
        (limb_offsets[:, 1], -limb_offsets[:, 0], limb_offsets[:, 2]),
        dim=-1,
    ).flatten()
    torch.testing.assert_close(goal[0, 9:21], expected_limbs, atol=1e-6, rtol=0)


def test_extended_body_goal_requires_velocity_and_column_timestep():
    root = torch.zeros((2, 3))
    rotation = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(2, 4)
    limbs = torch.zeros((2, 4, 3))
    with pytest.raises(ValueError, match="world_root_velocity is required"):
        build_ego_goal(
            root, torch.zeros(2), root, rotation,
            goal_type="body_ext", world_goal_keypoints=limbs,
            timestep=torch.ones((2, 1)),
        )
    with pytest.raises(ValueError, match="timestep must have shape"):
        build_ego_goal(
            root, torch.zeros(2), root, rotation,
            goal_type="body_ext", world_goal_keypoints=limbs,
            world_root_velocity=torch.zeros((2, 3)), timestep=torch.ones(2),
        )


def test_joint_state_goal_layout_uses_textop_orientation_and_direct_dof():
    reference_pos = torch.tensor([[1.0, 2.0, 0.7]])
    reference_yaw = torch.tensor([np.pi / 2], dtype=torch.float32)
    reference_rot = _yaw_quaternion(reference_yaw)
    goal_euler_ego = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
    world_goal_rot = quat_mul(
        reference_rot,
        euler_angles_to_quaternion(goal_euler_ego),
        w_last=True,
    )
    world_goal_pos = torch.tensor([[3.0, 3.0, 1.0]])
    world_velocity = torch.tensor([[2.0, 1.0, -0.5]])
    goal_dof = torch.linspace(-1.0, 1.0, 29).reshape(1, 29)

    goal = build_ego_joint_state_goal(
        world_goal_pos=world_goal_pos,
        world_goal_rot=world_goal_rot,
        world_goal_dof=goal_dof,
        world_root_velocity=world_velocity,
        reference_pos=reference_pos,
        reference_rot=reference_rot,
    )

    assert goal.shape == (1, 40)
    torch.testing.assert_close(goal[0, 0:3], torch.tensor([1.0, -2.0, 0.3]))
    torch.testing.assert_close(
        goal[0, 3:8],
        torch.tensor([
            np.sin(0.2),
            np.cos(0.2) - 1.0,
            np.sin(-0.1),
            np.cos(-0.1) - 1.0,
            0.3,
        ], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(goal[0, 8:37], goal_dof[0])
    torch.testing.assert_close(goal[0, 37:40], torch.tensor([1.0, -2.0, -0.5]))


def test_build_ego_goal_dispatches_joint_state():
    root = torch.zeros((1, 3))
    rotation = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    dof = torch.zeros((1, 29))

    goal = build_ego_goal(
        root,
        torch.zeros(1),
        root,
        rotation,
        goal_type="joint_state",
        world_goal_rot=rotation,
        world_goal_dof=dof,
        world_root_velocity=root,
    )

    assert goal.shape == (1, 40)
    torch.testing.assert_close(goal, torch.zeros((1, 40)))


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
        raw={"goal_keypoints_world": keypoints},
        goal_root_pos_world=None,
        goal_yaw_world=None,
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


def test_state_extended_body_goal_uses_velocity_and_absolute_timestamp():
    keypoints = np.asarray([
        [2.1, 3.2, 0.1],
        [2.1, 2.8, 0.1],
        [2.2, 3.3, 0.6],
        [2.2, 2.7, 0.6],
    ], dtype=np.float32)
    state = SimpleNamespace(
        raw={"goal_keypoints_world": keypoints},
        goal_root_pos_world=np.asarray([2.0, 3.0, 0.8], dtype=np.float32),
        goal_yaw_world=np.asarray([0.0], dtype=np.float32),
        goal_root_velocity_world=np.asarray(
            [0.5, -0.25, 0.0], dtype=np.float32),
        goal_timestamp_ns=4_500_000_000,
        timestamps_ns=[2_000_000_000, 2_500_000_000],
    )

    goal = state_goal_from_reference(
        state,
        reference_pos=torch.tensor([[1.0, 2.0, 0.7]]),
        reference_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        device="cpu",
        goal_type="body_ext",
    )

    assert goal.shape == (1, 21)
    torch.testing.assert_close(goal[0, 0:3], torch.tensor([1.0, 1.0, 0.1]))
    torch.testing.assert_close(goal[0, 5:8], torch.tensor([0.5, -0.25, 0.0]))
    torch.testing.assert_close(goal[0, 8:9], torch.tensor([2.0]))
    torch.testing.assert_close(
        goal[0, 9:21],
        (torch.from_numpy(keypoints) - torch.tensor([1.0, 2.0, 0.7])).flatten())


def test_state_joint_state_goal_uses_root_rotation_dof_and_velocity():
    goal_dof = np.linspace(-0.2, 0.2, 29, dtype=np.float32)
    state = SimpleNamespace(
        raw={},
        goal_root_pos_world=np.asarray([2.0, 3.0, 0.8], dtype=np.float32),
        goal_yaw_world=None,
        goal_root_rot_world=np.asarray(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        goal_dof_pos=goal_dof,
        goal_root_velocity_world=np.asarray(
            [0.5, -0.25, 0.0], dtype=np.float32),
        goal_timestamp_ns=4_500_000_000,
        timestamps_ns=[2_000_000_000, 2_500_000_000],
    )

    goal = state_goal_from_reference(
        state,
        reference_pos=torch.tensor([[1.0, 2.0, 0.7]]),
        reference_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        device="cpu",
        goal_type="joint_state",
    )

    assert goal.shape == (1, 40)
    torch.testing.assert_close(goal[0, 0:3], torch.tensor([1.0, 1.0, 0.1]))
    torch.testing.assert_close(goal[0, 3:8], torch.zeros(5))
    torch.testing.assert_close(goal[0, 8:37], torch.from_numpy(goal_dof))
    torch.testing.assert_close(goal[0, 37:40], torch.tensor([0.5, -0.25, 0.0]))


def test_state_joint_state_goal_accepts_world_euler_rotation():
    goal_dof = np.zeros(29, dtype=np.float32)
    state = SimpleNamespace(
        raw={},
        goal_root_pos_world=np.asarray([1.0, 0.0, 0.7], dtype=np.float32),
        goal_yaw_world=None,
        goal_root_euler_world=np.asarray(
            [0.0, 0.0, np.pi / 2.0], dtype=np.float32),
        goal_dof_pos=goal_dof,
        goal_root_velocity_world=np.zeros(3, dtype=np.float32),
        goal_timestamp_ns=2_000_000_000,
        timestamps_ns=[1_000_000_000],
    )

    goal = state_goal_from_reference(
        state,
        reference_pos=torch.tensor([[0.0, 0.0, 0.7]]),
        reference_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        device="cpu",
        goal_type="joint_state",
    )

    assert goal.shape == (1, 40)
    torch.testing.assert_close(
        goal[0, 3:8],
        torch.tensor([0.0, 0.0, 0.0, 0.0, np.pi / 2.0]))

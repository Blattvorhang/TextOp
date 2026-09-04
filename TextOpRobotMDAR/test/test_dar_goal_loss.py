from types import SimpleNamespace

import torch
import pytest
from omegaconf import OmegaConf

import robotmdar.dtype.motion as runtime_motion_dtype
import TextOpRobotMDAR.robotmdar.dtype.motion as package_motion_dtype
from TextOpRobotMDAR.robotmdar.dtype.rotation import matrix_to_rot6d
from TextOpRobotMDAR.robotmdar.train.manager import DARManager
from TextOpRobotMDAR.robotmdar.train.train_dar import (
    _make_root_xy_figure,
    _raw_goal_root_target,
    _validate_goal_root_position_contract,
)
from TextOpRobotMDAR.robotmdar.utils.goal import SPLIT_GOAL_DIM


def _manager():
    manager = object.__new__(DARManager)
    manager.dataset = SimpleNamespace(
        denormalize=lambda value: value,
        dof_dim=29,
        fps=50,
    )
    manager.rec_criterion = torch.nn.HuberLoss(reduction='mean', delta=1.0)
    return manager


def _history(displacement_xy):
    history = torch.zeros((len(displacement_xy), 2, 69), dtype=torch.float32)
    history[:, -1, 7:9] = torch.as_tensor(displacement_xy)
    return history


def _future(displacement_xy):
    future = torch.zeros((len(displacement_xy), 4, 69), dtype=torch.float32)
    future[:, :3, 7:9] = torch.as_tensor(displacement_xy)[:, None] / 4.0
    return future


def _set_both_feature_versions(version: int):
    old_runtime = runtime_motion_dtype.FeatureVersion
    old_package = package_motion_dtype.FeatureVersion
    runtime_motion_dtype.set_feature_version(version)
    package_motion_dtype.set_feature_version(version)
    return old_runtime, old_package


def test_goal_root_position_loss_is_zero_for_matching_endpoint():
    manager = _manager()
    future = _future([[1.0, 0.0], [0.0, -0.5]])
    history = _history([[0.25, 0.0], [0.0, -0.125]])
    goal = torch.zeros((2, 15))
    goal[:, :2] = torch.tensor([[1.0, 0.0], [0.0, -0.5]])

    loss = manager.calc_goal_root_position_loss(
        future, goal, history_motion=history)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_goal_root_position_loss_ignores_dropped_root_conditions():
    manager = _manager()
    future = _future([[1.0, 0.0], [0.0, 0.0]])
    history = _history([[0.25, 0.0], [0.0, 0.0]])
    goal = torch.zeros((2, 21))
    goal[:, :2] = torch.tensor([[1.0, 0.0], [5.0, 0.0]])

    loss = manager.calc_goal_root_position_loss(
        future, goal, goal_condition_keep_mask=torch.tensor([True, False]),
        history_motion=history)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_goal_root_position_loss_uses_goal_frame_when_provided():
    manager = _manager()
    future = torch.zeros((1, 4, 69), dtype=torch.float32)
    future[0, :3, 7] = 0.25
    history = _history([[0.25, 0.0]])
    goal = torch.zeros((1, 21), dtype=torch.float32)
    goal[:, :2] = torch.tensor([[0.5, 0.0]])

    loss = manager.calc_goal_root_position_loss(
        future,
        goal,
        history_motion=history,
        goal_time_frame=torch.tensor([2]),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_root_displacement_integrates_turning_local_deltas():
    manager = _manager()
    history = torch.zeros((1, 2, 69), dtype=torch.float32)
    history[0, -1, 7] = 1.0
    history[0, -1, 4] = torch.pi / 2
    future = torch.zeros((1, 2, 69), dtype=torch.float32)
    future[0, 0, 7] = 1.0

    displacement = manager.root_displacement_ego(future, history)

    torch.testing.assert_close(
        displacement, torch.tensor([[1.0, 1.0]]), atol=1e-6, rtol=0)


def test_root_trajectory_includes_origin_and_turning_path():
    manager = _manager()
    history = torch.zeros((1, 2, 69), dtype=torch.float32)
    history[0, -1, 7] = 1.0
    history[0, -1, 4] = torch.pi / 2
    future = torch.zeros((1, 2, 69), dtype=torch.float32)
    future[0, 0, 7] = 1.0

    trajectory = manager.root_trajectory_ego(future, history)

    torch.testing.assert_close(
        trajectory,
        torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]]),
        atol=1e-6,
        rtol=0,
    )


def test_root_displacement_excludes_terminal_forward_delta():
    manager = _manager()
    history = _history([[0.25, 0.0]])
    future = torch.zeros((1, 4, 69), dtype=torch.float32)
    future[0, :3, 7] = 0.25
    future[0, -1, 7] = 100.0

    displacement = manager.root_displacement_ego(future, history)

    torch.testing.assert_close(displacement, torch.tensor([[1.0, 0.0]]))


def test_goal_root_position_contract_allows_random_offsets_with_relative_time():
    cfg = OmegaConf.create({
        'data': {
            'goal_type': 'body_ext',
            'goal_per_primitive': False,
            'goal_offset': 0,
            'goal_offset_range': [-2, 2],
            'goal_timestep_mode': 'relative',
        },
        'train': {'manager': {'loss_weight': {'goal_root_position': 0.5}}},
    })

    _validate_goal_root_position_contract(cfg)

    cfg.data.goal_timestep_mode = 'zero'
    with pytest.raises(ValueError, match='goal_timestep_mode=relative'):
        _validate_goal_root_position_contract(cfg)


def test_goal_root_position_contract_allows_joint_state_random_offsets():
    cfg = OmegaConf.create({
        'data': {
            'goal_type': 'joint_state',
            'goal_per_primitive': True,
            'goal_offset': 0,
            'goal_offset_range': [-3, 0],
            'goal_timestep_mode': 'relative',
        },
        'train': {'manager': {'loss_weight': {'goal_root_position': 0.5}}},
    })

    _validate_goal_root_position_contract(cfg)


def test_joint_state_goal_losses_use_selected_goal_frame():
    manager = _manager()
    future = torch.zeros((1, 4, 69), dtype=torch.float32)
    history = torch.zeros((1, 2, 69), dtype=torch.float32)
    goal_time_frame = torch.tensor([2])

    # Goal-frame orientation: selected future step is time_to_arrival - 1.
    future[0, 1, 0:4] = torch.tensor([0.1, -0.01, -0.2, -0.02])
    history[0, -1, 4] = 0.2
    future[0, 0, 4] = 0.3

    # Goal-frame joints and velocity.
    q_goal = torch.linspace(-0.5, 0.5, 29)
    future[0, 1, 11:40] = q_goal
    future[0, 1, 7:10] = torch.tensor([0.02, 0.0, -0.01])

    goal = torch.zeros((1, 40), dtype=torch.float32)
    goal[0, 3:8] = torch.tensor([0.1, -0.01, -0.2, -0.02, 0.5])
    goal[0, 8:37] = q_goal
    goal[0, 37:40] = torch.tensor([
        0.02 * 50.0 * torch.cos(torch.tensor(0.5)),
        0.02 * 50.0 * torch.sin(torch.tensor(0.5)),
        -0.01 * 50.0,
    ])

    torch.testing.assert_close(
        manager.calc_goal_root_orientation_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        manager.calc_goal_joint_angle_loss(
            future, goal, goal_time_frame=goal_time_frame),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        manager.calc_goal_root_velocity_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )


def test_v6_joint_state_goal_losses_use_arrival_rotmat_goal_frame():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.dataset.dof_dim = 29
        manager.dataset.fps = 50

        rot6d_identity = matrix_to_rot6d(torch.eye(3)).reshape(6)
        history = torch.zeros((1, 2, 44), dtype=torch.float32)
        future = torch.zeros((1, 4, 44), dtype=torch.float32)
        for motion in (history, future):
            motion[..., 0] = 0.77
            motion[..., 1:4] = torch.tensor([0.0, 0.0, -1.0])
            motion[..., 7:13] = rot6d_identity
            motion[..., 42:44] = 1.0

        future[0, 0:2, 4] = 0.25
        q_goal = torch.linspace(-0.5, 0.5, 29)
        future[0, 1, 13:42] = q_goal

        goal = torch.zeros((1, 47), dtype=torch.float32)
        goal[0, 0] = 0.77
        goal[0, 1:4] = torch.tensor([0.5, 0.0, 0.0])
        goal[0, 4:7] = torch.tensor([0.0, 0.0, -1.0])
        goal[0, 7:13] = rot6d_identity
        goal[0, 13:42] = q_goal
        goal[0, 42:46] = torch.tensor([12.5, 0.0, 0.0, 0.0])
        goal[0, 46] = 2.0 / 50.0
        goal_time_frame = torch.tensor([2])

        torch.testing.assert_close(
            manager.calc_goal_root_position_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_root_orientation_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_joint_angle_loss(
                future, goal, goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_root_velocity_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_v6_split_goal_losses_use_hor_vert_rot_layout():
    old_runtime, old_package = _set_both_feature_versions(6)
    try:
        manager = _manager()
        manager.dataset.dof_dim = 29
        manager.dataset.fps = 50

        rot6d_identity = matrix_to_rot6d(torch.eye(3)).reshape(6)
        history = torch.zeros((1, 2, 44), dtype=torch.float32)
        future = torch.zeros((1, 4, 44), dtype=torch.float32)
        for motion in (history, future):
            motion[..., 0] = 0.77
            motion[..., 1:4] = torch.tensor([0.0, 0.0, -1.0])
            motion[..., 7:13] = rot6d_identity
            motion[..., 42:44] = 1.0

        future[0, 0:2, 4] = 0.25
        q_goal = torch.linspace(-0.5, 0.5, 29)
        future[0, 1, 13:42] = q_goal

        goal = torch.zeros((1, SPLIT_GOAL_DIM), dtype=torch.float32)
        goal[0, 0:3] = torch.tensor([0.5, 0.0, 0.0])
        goal[0, 3] = 0.5
        goal[0, 4] = torch.log1p(torch.tensor(0.5))
        goal[0, 9:15] = torch.tensor([0.77, 0.0, 0.0, 0.0, -1.0, 0.0])
        goal[0, 15:21] = rot6d_identity
        goal[0, 21:50] = q_goal
        goal[0, 50:54] = torch.tensor([12.5, 0.0, 0.0, 0.0])
        goal[0, 54] = 2.0 / 50.0
        goal_time_frame = torch.tensor([2])

        torch.testing.assert_close(
            _raw_goal_root_target(goal), torch.tensor([[0.5, 0.0, 0.0]]))
        torch.testing.assert_close(
            manager.calc_goal_root_position_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_root_orientation_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_joint_angle_loss(
                future, goal, goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            manager.calc_goal_root_velocity_loss(
                future, goal, history_motion=history,
                goal_time_frame=goal_time_frame),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )
    finally:
        runtime_motion_dtype.set_feature_version(old_runtime)
        package_motion_dtype.set_feature_version(old_package)


def test_joint_state_goal_cache_matches_uncached_losses():
    manager = _manager()
    calls = {'denormalize': 0}

    def denormalize(value):
        calls['denormalize'] += 1
        return value

    manager.dataset.denormalize = denormalize
    future = torch.zeros((2, 4, 69), dtype=torch.float32)
    history = torch.zeros((2, 2, 69), dtype=torch.float32)
    goal_time_frame = torch.tensor([2, 3])
    future[:, 1, 0:4] = torch.tensor([0.1, -0.01, -0.2, -0.02])
    future[:, 2, 0:4] = torch.tensor([0.0, 0.02, 0.1, -0.03])
    future[:, :, 4] = 0.1
    history[:, -1, 4] = 0.2
    future[:, 1, 7:10] = torch.tensor([0.02, 0.0, -0.01])
    future[:, 2, 7:10] = torch.tensor([0.0, -0.01, 0.03])
    future[:, 1, 11:40] = torch.linspace(-0.5, 0.5, 29)
    future[:, 2, 11:40] = torch.linspace(0.5, -0.5, 29)
    goal = torch.randn((2, 40), dtype=torch.float32)

    uncached = (
        manager.calc_goal_root_orientation_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame),
        manager.calc_goal_joint_angle_loss(
            future, goal, goal_time_frame=goal_time_frame),
        manager.calc_goal_root_velocity_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame),
    )
    calls['denormalize'] = 0
    state = manager._future_goal_state(
        future,
        history_motion=history,
        goal_time_frame=goal_time_frame,
        include_yaw=True,
    )
    assert calls['denormalize'] == 2
    calls['denormalize'] = 0
    cached = (
        manager.calc_goal_root_orientation_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame, goal_state=state),
        manager.calc_goal_joint_angle_loss(
            future, goal, goal_time_frame=goal_time_frame,
            goal_state=state),
        manager.calc_goal_root_velocity_loss(
            future, goal, history_motion=history,
            goal_time_frame=goal_time_frame, goal_state=state),
    )

    assert calls['denormalize'] == 0
    for actual, expected in zip(cached, uncached):
        torch.testing.assert_close(actual, expected)


def test_root_xy_figure_plots_generated_ground_truth_and_goal():
    generated = torch.tensor([
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
    ])
    ground_truth = generated.clone()
    goals = generated[:, -1]

    figure = _make_root_xy_figure(
        generated,
        goals,
        ground_truth_trajectory=ground_truth,
        goal_condition_keep_mask=torch.tensor([True, False]),
    )

    assert len(figure.axes) == 1
    assert 'primitive end error 0.000 m' in figure.axes[0].get_title()
    assert figure.axes[0].get_xlabel() == 'x-forward (m)'

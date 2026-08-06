from types import SimpleNamespace

import torch
import pytest
from omegaconf import OmegaConf

from TextOpRobotMDAR.robotmdar.train.manager import DARManager
from TextOpRobotMDAR.robotmdar.train.train_dar import (
    _make_root_xy_figure,
    _validate_goal_position_contract,
)


def _manager():
    manager = object.__new__(DARManager)
    manager.dataset = SimpleNamespace(denormalize=lambda value: value)
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


def test_goal_position_loss_is_zero_for_matching_endpoint():
    manager = _manager()
    future = _future([[1.0, 0.0], [0.0, -0.5]])
    history = _history([[0.25, 0.0], [0.0, -0.125]])
    goal = torch.zeros((2, 15))
    goal[:, :2] = torch.tensor([[1.0, 0.0], [0.0, -0.5]])

    loss = manager.calc_goal_position_loss(
        future, goal, history_motion=history)

    torch.testing.assert_close(loss, torch.tensor(0.0))


def test_goal_position_loss_ignores_dropped_root_conditions():
    manager = _manager()
    future = _future([[1.0, 0.0], [0.0, 0.0]])
    history = _history([[0.25, 0.0], [0.0, 0.0]])
    goal = torch.zeros((2, 21))
    goal[:, :2] = torch.tensor([[1.0, 0.0], [5.0, 0.0]])

    loss = manager.calc_goal_position_loss(
        future, goal, goal_condition_keep_mask=torch.tensor([True, False]),
        history_motion=history)

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


def test_goal_position_requires_terminal_per_primitive_goal():
    cfg = OmegaConf.create({
        'data': {
            'goal_per_primitive': False,
            'goal_offset': 0,
            'goal_offset_range': None,
        },
        'train': {'manager': {'loss_weight': {'goal_position': 0.5}}},
    })

    with pytest.raises(ValueError, match='goal_per_primitive=true'):
        _validate_goal_position_contract(cfg)

    cfg.data.goal_per_primitive = True
    cfg.data.goal_offset_range = [-2, 2]
    with pytest.raises(ValueError, match='fixed zero goal offset'):
        _validate_goal_position_contract(cfg)


def test_root_xy_figure_plots_generated_ground_truth_and_goal():
    generated = torch.tensor([
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
    ])
    ground_truth = generated.clone()
    goals = generated[:, -1]

    figure = _make_root_xy_figure(
        generated, goals, ground_truth, torch.tensor([True, False])
    )

    assert len(figure.axes) == 1
    assert 'endpoint error 0.000 m' in figure.axes[0].get_title()
    assert figure.axes[0].get_xlabel() == 'x-forward (m)'

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from TextOpRobotMDAR.robotmdar.train.manager import (
    BaseManager,
    _classify_extra,
)


class DummyManager(BaseManager):
    def hold_model(self, *args, **kwargs):
        pass

    def calc_loss(self, *args, **kwargs):
        raise NotImplementedError

    def update_ema_models(self):
        pass

    def save_model(self):
        pass

    def load_model(self, *args, **kwargs):
        pass


def _manager(use_rollout=True, rollout_max_prob=1.0):
    model = torch.nn.Linear(1, 1)
    manager = DummyManager(
        stages=[10, 20, 30],
        use_rollout=use_rollout,
        rollout_max_prob=rollout_max_prob,
        use_static_pose=False,
        anneal_lr=False,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        loss_weight={},
        ckpt={},
        device='cpu',
        platform=SimpleNamespace(report_scalar=lambda *args, **kwargs: None),
        save_every=1000,
        eval_every=1000,
        eval_steps=1,
        save_dir=Path('/tmp/robotmdar-test'),
    )
    manager.optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    manager._tqdm = object()
    return manager


@pytest.mark.parametrize(
    ('step', 'stage', 'expected_prob'),
    [
        (0, 0, 0.0),
        (9, 0, 0.0),
        (10, 1, 0.0),
        (20, 1, 0.5),
        (29, 1, 0.95),
        (30, 2, 1.0),
        (59, 2, 1.0),
    ],
)
def test_pre_step_logs_self_rollout_probability(step, stage, expected_prob):
    manager = _manager()
    manager.step = step

    manager.pre_step()

    assert manager.stage_idx == stage
    assert manager.extra['self_rollout_prob'] == pytest.approx(expected_prob)
    assert _classify_extra('self_rollout_prob', 'train') == (
        'meta',
        'self_rollout_prob',
    )
    assert manager.extra['self_rollout_used'] == 0.0
    assert _classify_extra('self_rollout_used', 'train') == (
        'meta',
        'self_rollout_used',
    )
    assert _classify_extra('self_rollout_ref_gt_dist_m', 'train') == (
        'metric',
        'train/self_rollout_ref_gt_dist_m',
    )


def test_self_rollout_probability_is_zero_when_disabled():
    manager = _manager(use_rollout=False)
    manager.step = 20

    manager.pre_step()

    assert manager.extra['self_rollout_prob'] == 0.0


@pytest.mark.parametrize(
    ('step', 'expected_prob'),
    [
        (10, 0.0),
        (20, 0.4),
        (30, 0.8),
        (59, 0.8),
    ],
)
def test_rollout_max_probability_is_configurable(step, expected_prob):
    manager = _manager(rollout_max_prob=0.8)
    manager.step = step

    manager.pre_step()

    assert manager.extra['self_rollout_prob'] == pytest.approx(expected_prob)


def test_rollout_max_probability_must_be_a_probability():
    with pytest.raises(ValueError, match='rollout_max_prob'):
        _manager(rollout_max_prob=1.1)

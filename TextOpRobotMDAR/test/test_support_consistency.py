import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from TextOpRobotMDAR.robotmdar.train.manager import GeometryLoss


def _fk(root_positions, support_positions):
    root = torch.tensor(root_positions, dtype=torch.float32)
    support = torch.tensor(support_positions, dtype=torch.float32)
    global_translation_extend = torch.stack([root, support], dim=1).unsqueeze(0)
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    global_rotation = identity.view(1, 1, 1, 4).repeat(1, root.shape[0], 2, 1)
    return {
        'global_translation_extend': global_translation_extend,
        'global_rotation': global_rotation,
    }


def test_support_consistency_is_zero_for_planted_contact():
    geometry = GeometryLoss()
    geometry.rec_criterion = nn.HuberLoss()

    fk = _fk(
        root_positions=[
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        support_positions=[
            [0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
    )
    support_mask = torch.ones(1, 3, 1, dtype=torch.bool)

    loss, metric, per_sample, active_ratio = geometry._support_component_from_fk(
        fk, [1], support_mask
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(metric, torch.tensor(0.0))
    torch.testing.assert_close(per_sample, torch.zeros(1))
    torch.testing.assert_close(active_ratio, torch.tensor(1.0))


def test_support_consistency_detects_unsupported_root_motion():
    geometry = GeometryLoss()
    geometry.rec_criterion = nn.HuberLoss()

    fk = _fk(
        root_positions=[
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        support_positions=[
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],
        ],
    )
    support_mask = torch.ones(1, 3, 1, dtype=torch.bool)

    loss, metric, per_sample, active_ratio = geometry._support_component_from_fk(
        fk, [1], support_mask
    )

    assert loss > 0
    assert metric > 0
    assert per_sample.item() > 0
    torch.testing.assert_close(active_ratio, torch.tensor(1.0))

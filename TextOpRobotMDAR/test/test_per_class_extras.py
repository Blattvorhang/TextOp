"""Per-class diagnostics extras (doc §4.3.4/§4.3.6).

The v6 geometry loss reports e_q_vel / e_h_vel / e_g_cons with a coarse
motion-class suffix (walk/run/fall/getup/unknown) as masked per-sample
means. These extras are log-only: detached from the graph, no loss or
gradient plumbing.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import TextOpRobotMDAR.robotmdar.dtype.motion as motion_dtype
from TextOpRobotMDAR.robotmdar.dtype.motion import (
    G1_DEFAULT_DOF,
    motion_dict_to_feature_v6,
)
from TextOpRobotMDAR.robotmdar.skeleton.robot import RobotSkeleton
from TextOpRobotMDAR.robotmdar.train.manager import (
    MOTION_CLASSES,
    GeometryLoss,
    _add_per_class_extras,
    _motion_class_labels,
)

ROOT = Path(__file__).resolve().parents[2]


def _skeleton():
    cfg = OmegaConf.load(
        ROOT / 'TextOpRobotMDAR/robotmdar/config/skeleton/g1.yaml'
    )
    cfg.asset.assetRoot = str(
        ROOT / 'TextOpRobotMDAR/description/robots/g1'
    )
    return RobotSkeleton(device='cpu', cfg=cfg)


def _mock_dataset(skeleton):
    class MockDataset:
        dof_dim = 29
        fps = 50.0
        mean = torch.zeros(1, 1, 44)
        std = torch.ones(1, 1, 44)

        def denormalize(self, feat):
            return feat * self.std + self.mean

        def reconstruct_motion(self, motion_feature, abs_pose=None,
                               need_denormalize=True, ret_fk=True,
                               ret_fk_full=False, sliding_mask=None):
            if need_denormalize:
                motion_feature = self.denormalize(motion_feature)
            motion_dict = motion_dtype.motion_feature_to_dict(
                motion_feature, abs_pose, self.skeleton)
            return self.skeleton.forward_kinematics(
                motion_dict, return_full=ret_fk_full, fps=self.fps)

    MockDataset.skeleton = skeleton
    return MockDataset()


def test_motion_class_label_mapping():
    labels = _motion_class_labels(
        ['walk', 'step_over', 'carry', 'injured', 'jog', 'sport', 'jump',
         'fall', 'gesture', 'dance', 'idle', 'crouch'],
        is_recovery=None,
    )
    assert labels == ['walk', 'walk', 'walk', 'walk', 'run', 'run', 'run',
                      'fall', 'unknown', 'unknown', 'unknown', 'unknown']

    # is_recovery refines the get-up split regardless of the verb.
    labels = _motion_class_labels(
        ['crouch', 'fall', 'walk'],
        is_recovery=torch.tensor([True, True, False]),
    )
    assert labels == ['getup', 'getup', 'walk']


def test_per_class_extras_masked_means_are_detached():
    err = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    extras = {}
    _add_per_class_extras(
        extras,
        action_label=['walk', 'walk', 'run', 'run', 'fall', 'unknown'],
        is_recovery=None,
        per_sample={'e_q_vel': err},
    )
    assert extras['e_q_vel__walk'].item() == 1.5
    assert extras['e_q_vel__run'].item() == 3.5
    assert extras['e_q_vel__fall'].item() == 5.0
    assert extras['e_q_vel__unknown'].item() == 6.0
    assert extras['e_q_vel__getup'].item() == 0.0  # absent class -> zero, key always emitted
    for value in extras.values():
        assert value.grad_fn is None  # log-only, detached


def test_per_class_extras_reject_wrong_shapes():
    with pytest.raises(ValueError, match=r'must be \[B\]'):
        _add_per_class_extras(
            {}, action_label=['walk'] * 4, is_recovery=None,
            per_sample={'e_q_vel': torch.zeros(4, 3)},
        )


def test_calc_geometry_loss_v6_reports_per_class_and_keeps_gradient():
    old_version = motion_dtype.FeatureVersion
    motion_dtype.set_feature_version(6)
    try:
        geometry = GeometryLoss()
        geometry.rec_criterion = nn.HuberLoss(delta=1.0)
        geometry.dataset = _mock_dataset(_skeleton())

        B, T = 6, 16
        t_axis = torch.arange(T + 1, dtype=torch.float32) / 50.0
        gt_dict = {
            'root_trans_offset': torch.stack(
                [0.5 * t_axis, torch.zeros_like(t_axis),
                 torch.full_like(t_axis, 0.75)], dim=-1
            ).unsqueeze(0).repeat(B, 1, 1),
            'root_rot': torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(B, T + 1, 1),
            'dof': torch.tensor(G1_DEFAULT_DOF, dtype=torch.float32).repeat(
                B, T + 1, 1),
            'contact_mask': torch.ones(B, T + 1, 2),
        }
        feature, _ = motion_dict_to_feature_v6(gt_dict, None)
        pred = feature.detach().clone() + 0.01 * torch.randn_like(feature)
        pred.requires_grad_(True)

        labels = ['walk', 'jog', 'fall', 'gesture', 'crouch', 'unknown']
        recovery = torch.tensor([False, False, False, False, True, False])

        terms, extras = geometry.calc_geometry_loss_v6(
            pred, feature, history_motion=feature[:, :4],
            action_label=labels, is_recovery=recovery,
        )

        # All three metrics emit every class suffix; absent classes are zero.
        for key in ('e_q_vel', 'e_h_vel', 'e_g_cons'):
            assert key in extras
            for cls in MOTION_CLASSES:
                assert f'{key}__{cls}' in extras, (key, cls)
        for key, value in extras.items():
            assert value.grad_fn is None, key

        # Global mean equals the count-weighted mean of the per-class means.
        class_counts = {'walk': 1, 'run': 1, 'fall': 1, 'getup': 1,
                        'unknown': 2}
        weighted = sum(
            extras[f'e_q_vel__{cls}'] * count
            for cls, count in class_counts.items()
        ) / B
        torch.testing.assert_close(extras['e_q_vel'], weighted)

        # Without labels: identical global values, no suffixed keys.
        _, extras_no_label = geometry.calc_geometry_loss_v6(
            pred, feature, history_motion=feature[:, :4],
        )
        torch.testing.assert_close(extras_no_label['e_q_vel'],
                                   extras['e_q_vel'])
        torch.testing.assert_close(extras_no_label['e_h_vel'],
                                   extras['e_h_vel'])
        assert not any('__' in key for key in extras_no_label)

        # The loss terms still carry gradient (extras never did).
        total = (terms['rot_chord'] + terms['g_cons']
                 + terms['h_vel'] + terms['dof_vel'])
        total.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
    finally:
        motion_dtype.set_feature_version(old_version)

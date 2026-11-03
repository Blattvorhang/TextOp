"""Goal statistics keep normalization and diagnostics in one reproducible export."""
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from robotmdar.dataloader import refresh_goal_stats
from test_split_goal_v61 import _split_goal_dataset, _joint_state_sample


def test_refresh_exports_unified_statistics_with_reproducible_global_seed(tmp_path, monkeypatch):
    class Dataset:
        def _compute_goal_stats(self):
            return {'s_p': torch.rand(()), 'meta': {},
                    'goal_clamp': {'random_probe': [random.random(), float(np.random.rand())],
                                   'reference_distribution': {
                                       'n_samples': 1, 'n_total': 2,
                                       'height_range': [.7, .775], 'motion_filter': 'height proxy'}}}
    monkeypatch.setattr(refresh_goal_stats, 'configure_dof_contract', lambda cfg: None)
    monkeypatch.setattr(refresh_goal_stats, 'instantiate', lambda cfg: Dataset())
    cfg = OmegaConf.create({'seed': 1, 'target_dir': str(tmp_path), 'data': {'train': {}}})
    refresh_goal_stats.main(cfg)
    first = torch.load(tmp_path / 'goal_stats.pkl', weights_only=False)
    refresh_goal_stats.main(cfg)
    second = torch.load(tmp_path / 'goal_stats.pkl', weights_only=False)
    torch.testing.assert_close(first['s_p'], second['s_p'], rtol=0, atol=0)
    assert first['goal_clamp'] == second['goal_clamp']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['goal_stats.pkl']


def test_diagnostic_settings_do_not_change_goal_normalization(tmp_path):
    dataset = _split_goal_dataset(tmp_path)
    samples = [_joint_state_sample(
        pos=[1., 2., .7], vel=[.2, .3, .1], euler=[.1, .2, .3],
        dof=torch.linspace(-1, 1, 29).tolist(), time_to_arrival=.1)]
    dataset.goal_clamp_time_min = .4
    first = dataset._goal_stats_from_batch(samples)
    dataset.goal_clamp_time_min = .8
    dataset.goal_clamp_reference_height_range = (0., 1.)
    second = dataset._goal_stats_from_batch(samples)
    for key in first.keys() - {'goal_clamp', 'meta'}:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert first['goal_clamp']['speed_quantiles'] != second['goal_clamp']['speed_quantiles']

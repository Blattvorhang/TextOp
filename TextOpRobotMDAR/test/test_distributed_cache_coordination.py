import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robotmdar.dataloader import data as data_module
from robotmdar.dataloader.data import SkeletonPrimitiveDataset
from robotmdar.utils.goal import GoalEncoding, GoalType


def _meanstd_dataset(tmp_path):
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.normalization_path = tmp_path / "meanstd.pkl"
    dataset.split = "train"
    dataset.nfeats = 2
    dataset.std_floor = 0.0
    dataset._stats_device_cache = {}
    return dataset


def test_nonwriter_waits_for_meanstd_instead_of_recomputing(
        tmp_path, monkeypatch):
    dataset = _meanstd_dataset(tmp_path)
    payload = {
        "mean": torch.tensor([1.0, 2.0]),
        "std": torch.tensor([3.0, 4.0]),
        "feature_version": 6,
        "feature_alignment": "arrival",
    }

    monkeypatch.setattr(data_module, "_is_goal_stats_writer", lambda: False)
    monkeypatch.setattr(data_module, "_is_torchrun", lambda: True)
    monkeypatch.setattr(
        dataset,
        "_wait_for_meanstd_cache",
        lambda path: payload,
    )
    monkeypatch.setattr(
        dataset,
        "_compute_meanstd",
        lambda: pytest.fail("non-writer rank recomputed mean/std"),
    )

    dataset._load_meanstd()

    torch.testing.assert_close(dataset.mean, payload["mean"])
    torch.testing.assert_close(dataset.std, payload["std"])


def test_nonwriter_waits_for_goal_stats_instead_of_recomputing(
        tmp_path, monkeypatch):
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.load_goal_stats = True
    dataset.goal_type = GoalType.JOINT_STATE
    dataset.goal_encoding = GoalEncoding.SPLIT
    dataset.goal_stats_path = tmp_path / "goal_stats.pkl"
    dataset.split = "train"
    expected = {"goal_clamp": {}}

    monkeypatch.setattr(data_module, "_is_goal_stats_writer", lambda: False)
    monkeypatch.setattr(data_module, "_is_torchrun", lambda: True)
    monkeypatch.setattr(
        data_module,
        "_wait_for_goal_stats_cache",
        lambda path, cache_validator=None: expected,
    )
    monkeypatch.setattr(
        dataset,
        "_goal_stats_cache_is_current",
        lambda stats: True,
    )
    monkeypatch.setattr(
        dataset,
        "_validate_goal_stats_cache",
        lambda stats: None,
    )
    monkeypatch.setattr(
        dataset,
        "_compute_goal_stats",
        lambda: pytest.fail("non-writer rank recomputed goal stats"),
    )

    dataset._load_goal_stats()

    assert dataset.goal_stats is expected


def test_writer_computes_goal_stats_once_and_publishes_cache(
        tmp_path, monkeypatch):
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.load_goal_stats = True
    dataset.goal_type = GoalType.JOINT_STATE
    dataset.goal_encoding = GoalEncoding.SPLIT
    dataset.goal_stats_path = tmp_path / "goal_stats.pkl"
    dataset.split = "train"
    expected = {"goal_clamp": {}}
    calls = []

    monkeypatch.setattr(data_module, "_is_goal_stats_writer", lambda: True)
    monkeypatch.setattr(data_module, "_is_torchrun", lambda: True)
    monkeypatch.setattr(
        dataset,
        "_goal_stats_cache_is_current",
        lambda stats: True,
    )
    monkeypatch.setattr(
        dataset,
        "_validate_goal_stats_cache",
        lambda stats: None,
    )
    monkeypatch.setattr(
        dataset,
        "_compute_goal_stats",
        lambda: calls.append(True) or expected,
    )

    dataset._load_goal_stats()

    assert calls == [True]
    assert dataset.goal_stats is expected
    assert dataset.goal_stats_path.exists()

import math
from pathlib import Path

import pytest
import torch

from TextOpRobotMDAR.robotmdar.dataloader.data import SkeletonPrimitiveDataset
from TextOpRobotMDAR.robotmdar.dtype.rotation import (
    euler_angles_to_quaternion,
    quat_mul,
)
from TextOpRobotMDAR.robotmdar.model.mld_denoiser import _apply_arrival_channel_mask
from TextOpRobotMDAR.robotmdar.utils.goal import (
    GoalEncoding,
    GoalType,
    SPLIT_GOAL_DIM,
    build_ego_goal,
    build_ego_split_goal,
    scale_goal,
    validate_goal_config,
    validate_goal_stats,
)


def _identity_quaternion(batch: int = 1) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).expand(batch, 4)


def _split_goal_stats_fixture(tmp_path: Path) -> dict:
    return {
        "s_p": torch.tensor(2.0),
        "s_l": torch.tensor(3.0),
        "s_v": torch.tensor(4.0),
        "s_o": torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0]),
        "q_mean": torch.linspace(-1.0, 1.0, 29),
        "q_std": torch.linspace(1.0, 2.0, 29),
        "meta": {
            "goal_offset_range": [-63, 0],
            "goal_per_primitive": True,
            "future_len": 64,
            "fps": 50.0,
            "goal_timestep_mode": "relative",
            "encodings": [GoalEncoding.SINGLE.value, GoalEncoding.SPLIT.value],
            "dataset_path": str(tmp_path),
            "goal_type": GoalType.JOINT_STATE.value,
            "goal_dim": SPLIT_GOAL_DIM,
            "dof_dim": 29,
        },
    }


def _split_goal_dataset(tmp_path: Path) -> SkeletonPrimitiveDataset:
    dataset = SkeletonPrimitiveDataset.__new__(SkeletonPrimitiveDataset)
    dataset.fps = 50.0
    dataset.mean = torch.zeros(69)
    dataset.std = torch.ones(69)
    dataset.goal_offset_range = (-63, 0)
    dataset.goal_per_primitive = True
    dataset.future_len = 64
    dataset.goal_timestep_mode = "relative"
    dataset.datadir = Path(tmp_path)
    dataset.goal_type = GoalType.JOINT_STATE
    dataset.goal_encoding = GoalEncoding.SPLIT
    dataset.dof_dim = 29
    return dataset


def _joint_state_sample(pos, vel, euler, dof, time_to_arrival):
    reference_rot = _identity_quaternion()
    world_goal_rot = quat_mul(
        reference_rot,
        euler_angles_to_quaternion(torch.tensor([euler], dtype=torch.float32)),
        w_last=True,
    )
    return {
        "world_goal_pos": torch.tensor([pos], dtype=torch.float32),
        "world_goal_rot": world_goal_rot,
        "world_goal_dof": torch.tensor([dof], dtype=torch.float32),
        "world_goal_vel": torch.tensor([vel], dtype=torch.float32),
        "gt_ref_pos": torch.zeros((1, 3), dtype=torch.float32),
        "gt_ref_rot": reference_rot.clone(),
        "time_to_arrival": torch.tensor([time_to_arrival], dtype=torch.float32),
    }


def test_split_goal_layout_scaling_round_trip(tmp_path):
    reference_pos = torch.zeros((1, 3), dtype=torch.float32)
    reference_rot = _identity_quaternion()
    goal_euler = torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32)
    world_goal_rot = quat_mul(
        reference_rot,
        euler_angles_to_quaternion(goal_euler),
        w_last=True,
    )
    world_goal_pos = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    world_goal_dof = torch.linspace(-1.0, 1.0, 29).reshape(1, 29)
    world_root_velocity = torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float32)
    time_to_arrival = torch.tensor([4.0], dtype=torch.float32)

    raw = build_ego_split_goal(
        world_goal_pos=world_goal_pos,
        world_goal_rot=world_goal_rot,
        world_goal_dof=world_goal_dof,
        world_root_velocity=world_root_velocity,
        reference_pos=reference_pos,
        reference_rot=reference_rot,
        time_to_arrival_seconds=time_to_arrival,
        fps=50.0,
    )
    assert raw.shape == (1, SPLIT_GOAL_DIM)
    torch.testing.assert_close(
        raw[:, 0:4],
        torch.tensor([[1.0, -2.0, 0.5, math.sqrt(5.0)]], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        raw[:, 4:5],
        torch.log1p(raw[:, 3:4]),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        raw[:, 5:8],
        torch.tensor([[0.25, -0.5, 0.125]], dtype=torch.float32),
        atol=1e-6,
        rtol=0,
    )

    stats = _split_goal_stats_fixture(tmp_path)
    assert validate_goal_config(
        "joint_state",
        SPLIT_GOAL_DIM,
        GoalEncoding.SPLIT,
        dof_dim=29,
        goal_offset_range=(-63, 0),
        goal_timestep_mode="relative",
        goal_stats=stats,
    ) is GoalType.JOINT_STATE

    scaled = scale_goal(raw, stats)
    expected = raw.clone()
    expected[:, 0:4] = expected[:, 0:4] * stats["s_p"]
    expected[:, 4:5] = expected[:, 4:5] * stats["s_l"]
    expected[:, 5:8] = expected[:, 5:8] * stats["s_v"]
    expected[:, 8:13] = expected[:, 8:13] * stats["s_o"]
    expected[:, 13:42] = (
        expected[:, 13:42] - stats["q_mean"]
    ) / stats["q_std"].clamp_min(1e-6)
    expected[:, 42:45] = expected[:, 42:45] * stats["s_v"]
    torch.testing.assert_close(scaled, expected, atol=1e-6, rtol=0)

    dispatched = build_ego_goal(
        world_goal_pos,
        torch.tensor([0.0], dtype=torch.float32),
        reference_pos,
        reference_rot,
        goal_type=GoalType.JOINT_STATE,
        goal_encoding=GoalEncoding.SPLIT,
        goal_stats=stats,
        fps=50.0,
        world_goal_rot=world_goal_rot,
        world_goal_dof=world_goal_dof,
        world_root_velocity=world_root_velocity,
        time_to_arrival_seconds=time_to_arrival,
    )
    torch.testing.assert_close(dispatched, scaled, atol=1e-6, rtol=0)


def test_split_goal_zero_time_zeroes_urgency():
    raw = build_ego_split_goal(
        world_goal_pos=torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32),
        world_goal_rot=_identity_quaternion(),
        world_goal_dof=torch.zeros((1, 29), dtype=torch.float32),
        world_root_velocity=torch.tensor([[4.0, -2.0, 1.0]], dtype=torch.float32),
        reference_pos=torch.zeros((1, 3), dtype=torch.float32),
        reference_rot=_identity_quaternion(),
        time_to_arrival_seconds=torch.tensor([0.0], dtype=torch.float32),
        fps=50.0,
    )

    torch.testing.assert_close(raw[:, 5:8], torch.zeros((1, 3)), atol=1e-6, rtol=0)


def test_split_goal_time_mask_zeroes_f_trans():
    goal = torch.arange(1, SPLIT_GOAL_DIM + 1, dtype=torch.float32).unsqueeze(0)
    masked = _apply_arrival_channel_mask(
        goal,
        SPLIT_GOAL_DIM,
        torch.zeros(1, dtype=torch.bool),
    )

    torch.testing.assert_close(masked[:, 5:8], torch.zeros((1, 3)))
    torch.testing.assert_close(masked[:, :5], goal[:, :5])
    torch.testing.assert_close(masked[:, 8:], goal[:, 8:])


def test_split_goal_stats_clip_outliers_before_std(tmp_path):
    dataset = _split_goal_dataset(tmp_path)
    batch_data = [
        _joint_state_sample(
            pos=[1.0, 1.0, 0.5],
            vel=[1.0, 2.0, 3.0],
            euler=[0.1, -0.2, 0.3],
            dof=torch.linspace(-1.0, 1.0, 29).tolist(),
            time_to_arrival=1.0,
        ),
        _joint_state_sample(
            pos=[12.0, -9.0, 0.5],
            vel=[30.0, -30.0, 6.0],
            euler=[-0.2, 0.05, -0.1],
            dof=torch.linspace(1.0, -1.0, 29).tolist(),
            time_to_arrival=1.0,
        ),
    ]

    stats = SkeletonPrimitiveDataset._goal_stats_from_batch(dataset, batch_data)

    goals = [
        build_ego_split_goal(
            world_goal_pos=item["world_goal_pos"],
            world_goal_rot=item["world_goal_rot"],
            world_goal_dof=item["world_goal_dof"],
            world_root_velocity=item["world_goal_vel"],
            reference_pos=item["gt_ref_pos"],
            reference_rot=item["gt_ref_rot"],
            time_to_arrival_seconds=item["time_to_arrival"],
            fps=dataset.fps,
        )
        for item in batch_data
    ]
    pos = torch.cat([goal[:, 0:4] for goal in goals], dim=0).clamp(-3.0, 3.0)
    urgency = torch.cat([goal[:, 5:8] for goal in goals], dim=0).clamp(-5.0, 5.0)
    velocity = torch.cat([goal[:, 42:45] for goal in goals], dim=0).clamp(-5.0, 5.0)
    expected_s_p = 1.0 / torch.clamp(pos.reshape(-1).std(unbiased=False), min=1e-6)
    expected_s_v = 1.0 / torch.clamp(
        torch.cat((urgency.reshape(-1), velocity.reshape(-1)), dim=0).std(unbiased=False),
        min=1e-6,
    )

    torch.testing.assert_close(stats["s_p"], expected_s_p)
    torch.testing.assert_close(stats["s_v"], expected_s_v)
    assert stats["meta"]["goal_dim"] == SPLIT_GOAL_DIM
    assert stats["meta"]["encodings"] == [GoalEncoding.SINGLE.value, GoalEncoding.SPLIT.value]


def test_split_goal_stats_meta_validation_rejects_mismatch(tmp_path):
    stats = _split_goal_stats_fixture(tmp_path)

    with pytest.raises(ValueError, match="goal_offset_range"):
        validate_goal_stats(
            stats,
            goal_encoding=GoalEncoding.SPLIT,
            goal_offset_range=(-62, 0),
            goal_per_primitive=True,
            future_len=64,
            fps=50.0,
            goal_timestep_mode="relative",
            datadir=str(tmp_path),
        )

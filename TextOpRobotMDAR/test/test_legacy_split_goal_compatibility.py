from pathlib import Path

from omegaconf import OmegaConf
import torch

from robotmdar.model.mld_denoiser import DenoiserTransformer
from robotmdar.planner.planner_dar import _checkpoint_contract
from robotmdar.utils.goal import (
    GoalEncoding,
    GoalType,
    LEGACY_SPLIT_GOAL_SCHEMA,
    build_ego_goal,
    build_ego_legacy_split_goal,
    scale_legacy_split_goal,
)


def _legacy_stats():
    return {
        "s_p": torch.tensor(2.0),
        "s_l": torch.tensor(3.0),
        "s_v": torch.tensor(4.0),
        "s_d": torch.tensor(1.0),
        "s_o": torch.ones(9),
        "q_mean": torch.zeros(29),
        "q_std": torch.ones(29),
        "meta": {
            "goal_offset_range": [-63, 0],
            "goal_per_primitive": True,
            "future_len": 64,
            "fps": 50.0,
            "goal_timestep_mode": "relative",
            "encodings": ["single", "split"],
            "goal_type": "joint_state",
            "goal_dim": 55,
            "goal_schema": LEGACY_SPLIT_GOAL_SCHEMA,
        },
    }


def _goal_inputs():
    identity = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    return {
        "world_goal_pos": torch.tensor([[1.0, 2.0, 1.5]]),
        "world_goal_yaw": torch.zeros(1),
        "reference_pos": torch.zeros(1, 3),
        "reference_rot": identity,
        "world_goal_rot": identity,
        "world_goal_dof": torch.zeros(1, 29),
        "world_root_velocity": torch.ones(1, 3),
        "time_to_arrival_seconds": torch.tensor([10.0]),
        "fps": 50.0,
    }


def test_0903_cfg_selects_legacy_split_layout():
    robotmdar_root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(
        robotmdar_root / "logs" / "pretrained" / "0903_heading_free" / "cfg.yaml"
    )

    contract = _checkpoint_contract(cfg)

    assert contract["goal_encoding"] is GoalEncoding.SPLIT
    assert contract["goal_schema"] == LEGACY_SPLIT_GOAL_SCHEMA
    assert contract["legacy_split_goal_layout"] is True


def test_legacy_split_goal_builder_uses_historical_layout():
    stats = _legacy_stats()
    inputs = _goal_inputs()
    raw = build_ego_legacy_split_goal(
        **{key: value for key, value in inputs.items() if key != "world_goal_yaw"}
    )

    expected = scale_legacy_split_goal(raw, stats)
    actual = build_ego_goal(
        **inputs,
        goal_type=GoalType.JOINT_STATE,
        goal_encoding=GoalEncoding.SPLIT,
        goal_stats=stats,
    )

    assert raw.shape == (1, 55)
    torch.testing.assert_close(actual, expected)


def test_no_log_split_end_effector_model_forward():
    model = DenoiserTransformer(
        h_dim=16,
        ff_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        history_shape=(2, 44),
        noise_shape=(1, 8),
        goal_encoding=GoalEncoding.SPLIT_END_EFFECTOR,
        goal_dim=66,
        grid_size=2,
        cond_mask_prob={
            "text": 0.0,
            "goal": {
                "position": 0.0,
                "orientation": 0.0,
                "joint": 0.0,
                "velocity": 0.0,
                "end_effector": 0.0,
                "time": 0.0,
            },
            "scene": 0.0,
        },
    ).eval()
    output = model(
        torch.randn(1, 1, 8),
        torch.zeros(1, dtype=torch.long),
        {
            "goal": torch.randn(1, 66),
            "voxel": torch.zeros(1, 8),
            "history_motion_normalized": torch.randn(1, 2, 44),
            "time_to_arrival_frame": torch.ones(1),
            "goal_valid": {
                "root": True,
                "orientation": True,
                "joint": True,
                "velocity": True,
                "time": True,
                "end_effector": True,
            },
        },
    )

    assert output.shape == (1, 1, 8)

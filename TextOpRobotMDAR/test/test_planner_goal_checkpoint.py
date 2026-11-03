import pytest
import torch
from omegaconf import OmegaConf

from robotmdar.planner.planner_dar import (
    _apply_checkpoint_config,
    _checkpoint_contract,
)
from robotmdar.utils.goal import (
    GoalClamp,
    SPLIT_END_EFFECTOR_NO_LOG_GOAL_SCHEMA,
    build_ego_goal,
    build_ego_split_end_effector_goal,
    scale_split_goal_no_log,
)


@pytest.mark.parametrize("goal_dim,encoding", [
    (54, "split"), (55, "split"),
    (66, "split_end_effector"), (67, "split_end_effector"),
])
def test_goal_contract_follows_checkpoint(goal_dim, encoding):
    model_cfg = OmegaConf.create({
        "data": {
            "feature_version": 6, "dof_dim": 29, "nfeats": 144,
            "goal_type": "joint_state", "history_len": 16, "future_len": 64,
        },
        "denoiser": {"goal_dim": goal_dim},
    })
    cfg = OmegaConf.create({
        "data": {"goal_encoding": "legacy40", "goal_include_log_d_hor": True},
        "denoiser": {"goal_dim": 40},
    })
    contract = _checkpoint_contract(model_cfg)
    _apply_checkpoint_config(cfg, model_cfg, contract)
    assert cfg.denoiser.goal_dim == cfg.data.goal_dim == goal_dim
    assert cfg.denoiser.goal_encoding == cfg.data.goal_encoding == encoding
    assert cfg.data.goal_include_log_d_hor == (goal_dim in (55, 67))


@pytest.mark.parametrize("clamp", [None, GoalClamp(speed_low=1, speed_full=1)])
def test_no_log_end_effector_goal_builds_and_scales_66_channels(clamp):
    identity = torch.tensor([[0., 0., 0., 1.]])
    inputs = dict(
        world_goal_pos=torch.tensor([[3., 4., 0.5]]),
        world_goal_rot=identity,
        world_goal_dof=torch.zeros(1, 29),
        world_root_velocity=torch.ones(1, 3),
        world_goal_end_effectors=torch.arange(12.).reshape(1, 4, 3),
        reference_pos=torch.zeros(1, 3), reference_rot=identity,
        time_to_arrival_seconds=torch.ones(1), fps=50., goal_clamp=clamp,
    )
    stats = dict(
        s_p=2., s_v=3., s_d=1., s_o=torch.ones(9),
        q_mean=torch.zeros(29), q_std=torch.ones(29),
        s_ee=torch.arange(1., 13.),
        meta={"goal_dim": 66, "goal_schema": SPLIT_END_EFFECTOR_NO_LOG_GOAL_SCHEMA},
    )
    legacy = build_ego_split_end_effector_goal(**inputs)
    expected_raw = torch.cat((legacy[..., :4], legacy[..., 5:]), dim=-1)
    actual = build_ego_goal(
        **inputs, world_goal_yaw=torch.zeros(1), goal_type="joint_state",
        goal_encoding="split_end_effector", goal_stats=stats,
        goal_include_log_d_hor=False,
    )
    assert actual.shape == (1, 66)
    torch.testing.assert_close(actual, scale_split_goal_no_log(expected_raw, stats))
    torch.testing.assert_close(actual[..., 54:], legacy[..., 55:] * stats["s_ee"])

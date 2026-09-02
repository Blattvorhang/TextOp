"""Tests for GoalClamp: clamping controller goals into the training envelope."""

import math

import pytest
import torch

from TextOpRobotMDAR.robotmdar.utils.goal import (
    SPLIT_GOAL_DIM,
    GoalClamp,
    GoalEncoding,
    build_ego_goal,
    build_ego_joint_state_goal,
    build_ego_split_goal,
)

from test_split_goal_v61 import _identity_quaternion, _split_goal_stats_fixture


def _split_inputs(time_to_arrival):
    """world goal (3, 4, 0.5) from identity reference: d_hor=5, h=0.5."""
    return {
        "world_goal_pos": torch.tensor([[3.0, 4.0, 0.5]], dtype=torch.float32),
        "world_goal_rot": _identity_quaternion(),
        "world_goal_dof": torch.linspace(-1.0, 1.0, 29).reshape(1, 29),
        "world_root_velocity": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "reference_pos": torch.zeros((1, 3), dtype=torch.float32),
        "reference_rot": _identity_quaternion(),
        "time_to_arrival_seconds": torch.tensor(
            [time_to_arrival], dtype=torch.float32),
    }


def test_goal_clamp_direction_preserved_and_z_untouched():
    clamp = GoalClamp(speed_max=2.0, r_min=0.0, r_max=math.inf)
    goal = build_ego_split_goal(**_split_inputs(1.0), fps=50.0,
                                goal_clamp=clamp)
    assert goal.shape == (1, SPLIT_GOAL_DIM)
    # d_hor=5 > 2.0*1.0 -> scaled by 2/5; height is separate.
    torch.testing.assert_close(
        goal[:, 0:5],
        torch.tensor([[0.5, 1.2, 1.6, 0.0, 2.0]], dtype=torch.float32),
        atol=1e-6, rtol=0,
    )
    # Derived channels follow the clamped root: log_d_hor, delta_h, urgency.
    torch.testing.assert_close(goal[:, 5:6], torch.log1p(goal[:, 4:5]),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(goal[:, 6:7],
                               torch.tensor([[0.5]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        goal[:, 7:12],
        torch.tensor([[1.2, 1.6, 0.0, 2.0, 0.5]], dtype=torch.float32),
        atol=1e-6, rtol=0,
    )


def test_goal_clamp_leaves_in_envelope_goal_untouched():
    clamp = GoalClamp(speed_max=2.0, r_min=0.0, r_max=math.inf)
    # r=sqrt(5)=2.236 < 2.0*2.0 s: within the envelope, must be unchanged.
    inputs = _split_inputs(2.0)
    inputs["world_goal_pos"] = torch.tensor(
        [[1.0, 2.0, 0.5]], dtype=torch.float32)
    plain = build_ego_split_goal(**inputs, fps=50.0)
    clamped_small = build_ego_split_goal(**inputs, fps=50.0,
                                         goal_clamp=clamp)
    torch.testing.assert_close(clamped_small, plain, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        plain[:, 0:7],
        torch.tensor(
            [[0.5, 1.0, 2.0, 0.0, math.sqrt(5.0),
              math.log1p(math.sqrt(5.0)), 0.5]],
            dtype=torch.float32,
        ),
        atol=1e-6, rtol=0,
    )


def test_goal_clamp_time_cap_feeds_urgency():
    # T=5 s with time_max=1.28 s: the bound uses the capped budget and so
    # does the urgency denominator.
    clamp = GoalClamp(speed_max=2.0, r_min=0.0, r_max=math.inf,
                      time_max=1.28)
    goal = build_ego_split_goal(**_split_inputs(5.0), fps=50.0,
                                goal_clamp=clamp)
    # r_max = 2.0*1.28 = 2.56 → scale 2.56/5.
    torch.testing.assert_close(
        goal[:, 0:5],
        torch.tensor([[0.5, 1.536, 2.048, 0.0, 2.56]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )
    # urgency = clamped tangent position / capped T plus d_hor/T and delta_h/T.
    torch.testing.assert_close(
        goal[:, 7:12],
        torch.tensor([[1.2, 1.6, 0.0, 2.0, 0.5 / 1.28]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )


def test_goal_clamp_r_min_floor_at_zero_arrival():
    # T=0: urgency is zeroed by the encoding and the radius bound floors at
    # r_min (speed*fps budget 1.72/50 < r_min=0.1).
    clamp = GoalClamp(speed_max=1.72, r_min=0.1, r_max=2.5)
    goal = build_ego_split_goal(**_split_inputs(0.0), fps=50.0,
                                goal_clamp=clamp)
    torch.testing.assert_close(
        goal[:, 0:5],
        torch.tensor([[0.5, 0.06, 0.08, 0.0, 0.1]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )
    torch.testing.assert_close(
        goal[:, 7:12], torch.zeros((1, 5)), atol=1e-6, rtol=0)


def test_goal_clamp_fixed_cap_without_arrival_time():
    # LEGACY40 with no arrival time: the fixed r_max cap applies.
    clamp = GoalClamp(speed_max=1.0, r_min=0.0, r_max=1.0)
    goal = build_ego_joint_state_goal(
        world_goal_pos=torch.tensor([[3.0, 4.0, 0.5]], dtype=torch.float32),
        world_goal_rot=_identity_quaternion(),
        world_goal_dof=torch.linspace(-1.0, 1.0, 29).reshape(1, 29),
        world_root_velocity=torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float32),
        reference_pos=torch.zeros((1, 3), dtype=torch.float32),
        reference_rot=_identity_quaternion(),
        goal_clamp=clamp,
    )
    torch.testing.assert_close(
        goal[:, 0:3],
        torch.tensor([[0.6, 0.8, 0.5]], dtype=torch.float32),
        atol=1e-6, rtol=0,
    )


def test_goal_clamp_end_to_end_scaled_split_goal(tmp_path):
    # build_ego_goal path: the clamp happens before scale_goal, so the
    # scaled channels stay consistent (s_p=2, s_v=4 in the fixture).
    stats = _split_goal_stats_fixture(tmp_path)
    clamp = GoalClamp(speed_max=2.0, r_min=0.0, r_max=math.inf)
    goal = build_ego_goal(
        world_goal_pos=torch.tensor([[3.0, 4.0, 0.5]], dtype=torch.float32),
        world_goal_yaw=torch.zeros(1),
        reference_pos=torch.zeros((1, 3), dtype=torch.float32),
        reference_rot=_identity_quaternion(),
        goal_type="joint_state",
        goal_encoding=GoalEncoding.SPLIT,
        goal_stats=stats,
        fps=50.0,
        world_goal_rot=_identity_quaternion(),
        world_goal_dof=torch.linspace(-1.0, 1.0, 29).reshape(1, 29),
        world_root_velocity=torch.tensor(
            [[1.0, 0.0, 0.0]], dtype=torch.float32),
        time_to_arrival_seconds=torch.tensor([1.0], dtype=torch.float32),
        goal_clamp=clamp,
    )
    # Raw clamped position channels scaled by s_p=2.
    torch.testing.assert_close(
        goal[:, 0:7],
        torch.tensor(
            [[1.0, 2.4, 3.2, 0.0, 4.0, 3.0 * math.log1p(2.0), 1.0]],
            dtype=torch.float32,
        ),
        atol=1e-5, rtol=0,
    )
    # urgency channels scaled by s_v=4.
    torch.testing.assert_close(
        goal[:, 7:12],
        torch.tensor([[4.8, 6.4, 0.0, 8.0, 2.0]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )


def test_goal_clamp_validation():
    with pytest.raises(ValueError, match="speed_max"):
        GoalClamp(speed_max=0.0)
    with pytest.raises(ValueError, match="r_min"):
        GoalClamp(speed_max=1.0, r_min=-0.1)
    with pytest.raises(ValueError, match="r_max"):
        GoalClamp(speed_max=1.0, r_min=2.0, r_max=1.0)
    with pytest.raises(ValueError, match="time_max"):
        GoalClamp(speed_max=1.0, time_max=0.0)


def _goal_clamp_stats_fixture():
    """Frozen-stats dict with goal-clamp quantile tables (BONES-SEED-like)."""
    levels = [50.0, 80.0, 90.0, 95.0, 99.0, 99.5]
    return {
        "meta": {"fps": 50.0, "future_len": 64},
        "goal_clamp": {
            "quantile_levels": levels,
            "dist_quantiles": [0.08, 0.55, 0.98, 1.35, 2.20, 2.50],
            "speed_quantiles": [0.06, 0.43, 0.76, 1.05, 1.72, 1.95],
            "n_samples": 1000,
        },
    }


def test_goal_clamp_from_stats_derives_envelope():
    # Percentile 99: speed_max/ r_max from the tables, r_min = speed/fps and
    # time_max = future_len/fps derived from meta.
    clamp = GoalClamp.from_stats(_goal_clamp_stats_fixture(), 99.0)
    assert clamp.speed_max == pytest.approx(1.72)
    assert clamp.r_max == pytest.approx(2.20)
    assert clamp.r_min == pytest.approx(1.72 / 50.0)
    assert clamp.time_max == pytest.approx(64.0 / 50.0)


def test_goal_clamp_from_stats_interpolates_quantile():
    # 97 is halfway between stored levels 95 and 99.
    clamp = GoalClamp.from_stats(_goal_clamp_stats_fixture(), 97.0)
    assert clamp.speed_max == pytest.approx(1.05 + 0.5 * (1.72 - 1.05))
    assert clamp.r_max == pytest.approx(1.35 + 0.5 * (2.20 - 1.35))


def test_goal_clamp_from_stats_missing_tables_raises():
    stats = dict(_goal_clamp_stats_fixture())
    del stats["goal_clamp"]
    with pytest.raises(ValueError, match="goal_clamp"):
        GoalClamp.from_stats(stats, 99.0)


def test_goal_clamp_from_stats_invalid_quantile():
    with pytest.raises(ValueError, match="quantile"):
        GoalClamp.from_stats(_goal_clamp_stats_fixture(), 0.0)
    with pytest.raises(ValueError, match="quantile"):
        GoalClamp.from_stats(_goal_clamp_stats_fixture(), 100.0)


def test_goal_clamp_from_stats_end_to_end():
    # r=5 at T=1 s with percentile 99: bound = 1.72 m → scale 1.72/5.
    clamp = GoalClamp.from_stats(_goal_clamp_stats_fixture(), 99.0)
    goal = build_ego_split_goal(**_split_inputs(1.0), fps=50.0,
                                goal_clamp=clamp)
    torch.testing.assert_close(
        goal[:, 0:5],
        torch.tensor(
            [[0.5, 3.0 * 1.72 / 5.0, 4.0 * 1.72 / 5.0, 0.0, 1.72]],
            dtype=torch.float32,
        ),
        atol=1e-6, rtol=0,
    )
    # Derived urgency = clamped tangent root / T, d_hor/T, delta_h/T.
    torch.testing.assert_close(
        goal[:, 7:12],
        torch.tensor(
            [[3.0 * 1.72 / 5.0, 4.0 * 1.72 / 5.0, 0.0, 1.72, 0.5]],
            dtype=torch.float32,
        ),
        atol=1e-6, rtol=0,
    )


def test_goal_clamp_disabled_is_identity():
    # goal_clamp=None must leave the split goal exactly as before.
    inputs = _split_inputs(1.0)
    plain = build_ego_split_goal(**inputs, fps=50.0)
    with_clamp_disabled = build_ego_split_goal(**inputs, fps=50.0,
                                               goal_clamp=None)
    torch.testing.assert_close(with_clamp_disabled, plain, atol=0, rtol=0)
    # The unclamped goal keeps d_hor=5 and urgency=(3,4,0,5,0.5)/1.
    torch.testing.assert_close(
        plain[:, 0:7],
        torch.tensor(
            [[0.5, 3.0, 4.0, 0.0, 5.0, math.log1p(5.0), 0.5]],
            dtype=torch.float32,
        ),
        atol=1e-6, rtol=0)

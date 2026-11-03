"""Manual height/average-speed limits, timestamp budgets and goal geometry."""

from dataclasses import replace
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
    build_ego_split_end_effector_goal,
    build_ego_split_end_effector_goal_no_log,
    build_ego_split_goal_no_log,
)

from test_split_goal_v61 import _identity_quaternion, _split_goal_stats_fixture


def _constant_speed_clamp(speed, **kwargs):
    return GoalClamp(speed_low=speed, speed_full=speed, fps=50, **kwargs)


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
    clamp = _constant_speed_clamp(speed=2.0)
    goal = build_ego_split_goal(**_split_inputs(1.0), fps=50.0,
                                goal_clamp=clamp)
    assert goal.shape == (1, SPLIT_GOAL_DIM)
    # d_hor=5 > 2.0*1.0 -> scaled by 2/5; height is separate.
    torch.testing.assert_close(
        goal[:, 0:5],
        torch.tensor(
            [[1.2, 1.6, 0.0, 2.0, math.log1p(2.0)]],
            dtype=torch.float32,
        ),
        atol=1e-6, rtol=0,
    )
    # Derived channels follow the clamped root: h/delta_h and urgency.
    torch.testing.assert_close(goal[:, 9:11],
                               torch.tensor([[0.5, 0.5]]),
                               atol=1e-6, rtol=0)
    torch.testing.assert_close(
        goal[:, 5:9],
        torch.tensor([[1.2, 1.6, 0.0, 2.0]], dtype=torch.float32),
        atol=1e-6, rtol=0,
    )
    torch.testing.assert_close(goal[:, 14:15],
                               torch.tensor([[0.5]], dtype=torch.float32),
                               atol=1e-6, rtol=0)


def test_goal_clamp_leaves_in_envelope_goal_untouched():
    clamp = _constant_speed_clamp(speed=2.0)
    # r=sqrt(5)=2.236 < 2.0*2.0 s: within the envelope, must be unchanged.
    inputs = _split_inputs(2.0)
    inputs["world_goal_pos"] = torch.tensor(
        [[1.0, 2.0, 0.5]], dtype=torch.float32)
    plain = build_ego_split_goal(**inputs, fps=50.0)
    clamped_small = build_ego_split_goal(**inputs, fps=50.0,
                                         goal_clamp=clamp)
    torch.testing.assert_close(clamped_small, plain, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        plain[:, 0:5],
        torch.tensor(
            [[1.0, 2.0, 0.0, math.sqrt(5.0),
              math.log1p(math.sqrt(5.0))]],
            dtype=torch.float32,
        ),
        atol=1e-6, rtol=0,
    )
    torch.testing.assert_close(plain[:, 9:11],
                               torch.tensor([[0.5, 0.5]]),
                               atol=1e-6, rtol=0)


def test_goal_clamp_time_cap_feeds_urgency():
    # T=5 s with time_max=1.28 s: the bound uses the capped budget and so
    # does the urgency denominator.
    clamp = _constant_speed_clamp(speed=2.0,
                      time_max=1.28)
    goal = build_ego_split_goal(**_split_inputs(5.0), fps=50.0,
                                goal_clamp=clamp)
    # Allowed distance = 2.0*1.28 = 2.56 → scale 2.56/5.
    torch.testing.assert_close(
        goal[:, 0:5],
        torch.tensor(
            [[1.536, 2.048, 0.0, 2.56, math.log1p(2.56)]],
            dtype=torch.float32,
        ),
        atol=1e-5, rtol=0,
    )
    # urgency = clamped tangent position / capped T plus d_hor/T and delta_h/T.
    torch.testing.assert_close(
        goal[:, 5:9],
        torch.tensor([[1.2, 1.6, 0.0, 2.0]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )
    torch.testing.assert_close(
        goal[:, 14:15],
        torch.tensor([[0.5 / 1.28]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )


def test_goal_clamp_expired_valid_arrival_gets_transition_time():
    clamp = _constant_speed_clamp(speed=1.72)
    goal = build_ego_split_goal(**_split_inputs(0.0), fps=50.0, goal_clamp=clamp)
    assert goal[0, 3].item() == pytest.approx(1.72 * .4)
    assert goal[0, -1].item() == pytest.approx(.4)
    assert goal[0, 8].item() == pytest.approx(1.72)
    assert goal[0, 14].item() == pytest.approx(.5 / .4)


def test_goal_clamp_absent_time_uses_horizon():
    # No arrival time: speed * configured horizon is used.
    clamp = _constant_speed_clamp(speed=1.0, time_max=1.0)
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
    clamp = _constant_speed_clamp(speed=2.0)
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
        goal[:, 0:5],
        torch.tensor(
            [[2.4, 3.2, 0.0, 4.0, 3.0 * math.log1p(2.0)]],
            dtype=torch.float32,
        ),
        atol=1e-5, rtol=0,
    )
    # urgency channels scaled by s_v=4.
    torch.testing.assert_close(
        goal[:, 5:9],
        torch.tensor([[4.8, 6.4, 0.0, 8.0]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )
    torch.testing.assert_close(
        goal[:, 9:11],
        torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )
    torch.testing.assert_close(
        goal[:, 14:15],
        torch.tensor([[2.0]], dtype=torch.float32),
        atol=1e-5, rtol=0,
    )


@pytest.mark.parametrize("kwargs, message", [
    ({"speed_low": 0}, "speed_low"),
    ({"speed_full": float("nan")}, "speed_full"),
    ({"speed_low": 4}, "speed_low"),
    ({"height_low": .8, "height_full": .7}, "height_low"),
    ({"fps": float("inf")}, "fps"),
    ({"time_min": -.1}, "time_min"),
    ({"time_min": float("nan")}, "time_min"),
    ({"time_min": .4, "time_max": .2}, "time_max"),
])
def test_goal_clamp_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        GoalClamp(**kwargs)


def test_manual_speed_curve_is_monotone_with_exact_endpoints():
    clamp = GoalClamp(fps=50, time_max=1.28)
    h = torch.linspace(-.2, 1.3, 1501, dtype=torch.float64)
    speed = clamp.speed_for_height(h)
    torch.testing.assert_close(speed[h <= .45], torch.full_like(speed[h <= .45], .6))
    torch.testing.assert_close(speed[h >= .70], torch.full_like(speed[h >= .70], 3))
    assert torch.all(torch.diff(speed) >= -1e-12)
    assert clamp.speed_for_height(torch.tensor(.575)).item() == pytest.approx(1.8)
    for t in (.4, .8, 1.28):
        torch.testing.assert_close(clamp.radius_for_height(h, t), speed * t)


def test_goal_clamp_disabled_is_identity():
    # goal_clamp=None must leave the split goal exactly as before.
    inputs = _split_inputs(1.0)
    plain = build_ego_split_goal(**inputs, fps=50.0)
    with_clamp_disabled = build_ego_split_goal(**inputs, fps=50.0,
                                               goal_clamp=None)
    torch.testing.assert_close(with_clamp_disabled, plain, atol=0, rtol=0)
    # The unclamped goal keeps d_hor=5 and urgency=(3,4,0,5,0.5)/1.
    torch.testing.assert_close(
        plain[:, 0:5],
        torch.tensor(
            [[3.0, 4.0, 0.0, 5.0, math.log1p(5.0)]],
            dtype=torch.float32,
        ),
        atol=1e-6, rtol=0)

    torch.testing.assert_close(
        plain[:, 5:9],
        torch.tensor([[3.0, 4.0, 0.0, 5.0]], dtype=torch.float32),
        atol=1e-6, rtol=0)
    torch.testing.assert_close(
        plain[:, 9:11],
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        atol=1e-6, rtol=0)
    torch.testing.assert_close(
        plain[:, 14:15],
        torch.tensor([[0.5]], dtype=torch.float32),
        atol=1e-6, rtol=0)


def test_radius_uses_time_not_an_independent_distance_cap():
    clamp = GoalClamp(fps=50, time_max=1.28)
    height = torch.tensor([.3, .575, .9])
    speed = clamp.speed_for_height(height)
    column_times = torch.tensor([[.02], [.8], [10.]])
    torch.testing.assert_close(clamp.radius_for_height(height, column_times),
                               speed * torch.tensor([.4, .8, 1.28]))
    for t in (0, -1):
        torch.testing.assert_close(clamp.radius_for_height(height, t), speed * .4)
    torch.testing.assert_close(clamp.radius_for_height(height), speed * 1.28)
    masked = replace(clamp, arrival_time_valid=False)
    torch.testing.assert_close(masked.radius_for_height(height, 0), speed * 1.28)
    with pytest.raises(ValueError, match="finite"):
        clamp.radius_for_height(height, float("nan"))
    with pytest.raises(ValueError, match="time_max"):
        GoalClamp(fps=50).radius_for_height(height)


@pytest.mark.parametrize("builder", [build_ego_split_end_effector_goal,
                                    build_ego_split_end_effector_goal_no_log])
def test_clamp_preserves_body_relative_end_effectors_in_tilted_frame(builder):
    from robotmdar.dtype.rotation import (
        euler_angles_to_quaternion, quaternion_to_matrix, xyzw_to_wxyz,
    )
    inputs = _split_inputs(.4)
    inputs["reference_pos"][..., 2] = .7
    inputs["reference_rot"] = euler_angles_to_quaternion(torch.tensor([[.4, -.3, .2]]))
    inputs["world_goal_end_effectors"] = inputs["world_goal_pos"][:, None] + torch.tensor(
        [[[.1, .2, .3], [-.1, -.2, .3], [.1, 0, -.4], [-.1, 0, -.4]]])
    plain = builder(**inputs, fps=50)
    clamped = builder(**inputs, fps=50,
                      goal_clamp=GoalClamp(speed_low=1, speed_full=1, fps=50, time_max=1.28))
    shift = clamped[..., :3] - plain[..., :3]
    assert torch.linalg.vector_norm(shift).item() > 0
    ee_shift = (clamped[..., -12:] - plain[..., -12:]).reshape(1, 4, 3)
    torch.testing.assert_close(ee_shift, shift[:, None].expand(1, 4, 3), atol=1e-6, rtol=0)
    rotation = quaternion_to_matrix(xyzw_to_wxyz(inputs["reference_rot"]))
    world_shift = (rotation @ shift.unsqueeze(-1)).squeeze(-1)
    assert world_shift[0, 2].item() == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("time_valid, timestamp, expected_time", [
    (False, None, 1.28), (True, 0, .4), (True, 1_200_000_000, .4),
])
def test_controller_uses_measured_height_and_explicit_time_validity(
        monkeypatch, time_valid, timestamp, expected_time):
    import robotmdar.dtype.motion as motion_dtype
    from robotmdar.utils.planner_convert import state_goal_from_reference
    from test_planner_utils import _joint_state_with_end_effectors
    state = _joint_state_with_end_effectors(None)
    state.states.g1_pos[-1, 2] = .3
    state.condition.goal.root_pos_world[:] = [3, 4, .8]
    state.condition.valid.goal.time = time_valid
    state.condition.goal.timestamp_ns = timestamp
    monkeypatch.setattr(motion_dtype, "FeatureVersion", 6)
    clamp = GoalClamp(time_max=1.28, fps=50)
    goal = state_goal_from_reference(
        state, torch.tensor([[0, 0, .8]]), _identity_quaternion(), "cpu",
        goal_type="joint_state", goal_encoding="legacy40", goal_clamp=clamp, fps=50)
    radius = torch.linalg.vector_norm(goal[..., 1:4]).item()
    assert radius == pytest.approx(.6 * expected_time)
    assert goal[0, -1].item() == pytest.approx(expected_time if time_valid else 0)
    assert clamp.current_height is None  # No state leaks between messages.


@pytest.mark.parametrize("builder, urgency_index", [
    (build_ego_split_goal, 8), (build_ego_split_goal_no_log, 7),
])
@pytest.mark.parametrize("requested, effective", [(0, .4), (.02, .4), (.4, .4), (.8, .8), (5, 1.28)])
def test_transition_time_is_shared_by_radius_time_channel_and_urgency(
        builder, urgency_index, requested, effective):
    clamp = _constant_speed_clamp(speed=1, time_max=1.28)
    goal = builder(**_split_inputs(requested), fps=50, goal_clamp=clamp)
    assert goal[0, 3].item() == pytest.approx(effective)
    assert goal[0, -1].item() == pytest.approx(effective)
    assert goal[0, urgency_index].item() == pytest.approx(1)
    # An already-close goal stays close; the minimum duration does not impose
    # a minimum travel distance.
    inputs = _split_inputs(requested)
    inputs['world_goal_pos'][0, :2] = torch.tensor([.003, .004])
    near = builder(**inputs, fps=50, goal_clamp=clamp)
    assert near[0, 3].item() == pytest.approx(.005)


def test_arrival_pe_uses_the_same_transition_budget():
    from robotmdar.planner.planner_dar import _time_to_arrival_from_state
    from test_planner_utils import _joint_state_with_end_effectors
    state = _joint_state_with_end_effectors(None)
    clamp = GoalClamp(fps=50, time_max=1.28)
    for timestamp, seconds, frames in [(0, .4, 20), (1_020_000_000, .4, 20),
                                      (1_800_000_000, .8, 40), (9_000_000_000, 1.28, 64)]:
        state.condition.goal.timestamp_ns = timestamp
        effective, encoded_frames = _time_to_arrival_from_state(state, 50, 'cpu', goal_clamp=clamp)
        assert effective == pytest.approx(seconds)
        assert encoded_frames.item() == frames


@pytest.mark.parametrize("builder", [build_ego_split_goal, build_ego_split_goal_no_log])
@pytest.mark.parametrize("height,speed", [(.3, .6), (.575, 1.8), (.8, 3.)])
@pytest.mark.parametrize("arrival", [0., .02, .8, 5.])
def test_far_static_goal_limits_average_speed_and_shares_time(builder, height, speed, arrival):
    inputs = _split_inputs(arrival)
    inputs['reference_pos'][..., 2] = height
    inputs['world_root_velocity'].zero_()
    clamp = GoalClamp(fps=50, time_max=1.28)
    effective = max(.4, min(1.28, arrival))
    result = builder(**inputs, fps=50, goal_clamp=clamp)
    assert result[0, 3].item() == pytest.approx(speed * effective)
    assert result[0, -1].item() == pytest.approx(effective)
    # Compare all channels with the equivalent manually projected goal.
    inputs['world_goal_pos'][..., :2] *= speed * effective / 5
    inputs['time_to_arrival_seconds'].fill_(effective)
    expected = builder(**inputs, fps=50)
    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


def test_manual_config_has_no_statistics_or_distance_overrides():
    from pathlib import Path
    from omegaconf import OmegaConf
    config = OmegaConf.load(Path(__file__).parents[1] / 'robotmdar/config/planner_dar.yaml')
    assert set(config.goal_clamp) == {'height_low', 'height_full', 'speed_low', 'speed_full', 'time_min'}
    clamp = GoalClamp(**dict(config.goal_clamp), fps=50, time_max=1.28)
    assert clamp.radius_for_height(torch.tensor(.3), .4).item() == pytest.approx(.24)
    assert clamp.radius_for_height(torch.tensor(.8), .4).item() == pytest.approx(1.2)
    assert clamp.radius_for_height(torch.tensor(.8), 1.28).item() == pytest.approx(3.84)

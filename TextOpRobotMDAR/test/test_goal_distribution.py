import pytest
from robotmdar.utils.goal_distribution import height_goal_distribution, reference_goal_distribution


def test_fixed_height_counts_empty_bins_and_conditional_quantiles():
    result = height_goal_distribution(
        [0.0, 0.125, 0.75], [1.0, 3.0, 8.0], [1.0, 2.0, 4.0],
        fps=50, bin_width=0.25,
    )
    assert result["bin_edges"] == [0, 0.25, 0.5, 0.75, 1.0]
    assert result["n_samples"] == [2, 0, 0, 1]
    assert sum(result["sample_fraction"]) == pytest.approx(1)
    assert result["dist_quantiles"][0][0] == pytest.approx(2)
    assert result["speed_quantiles"][0][0] == pytest.approx(1.25)
    assert result["arrival_time_quantiles"][0][0] == pytest.approx(1.5)
    assert result["centers"][1] is None
    assert all(v is None for v in result["dist_quantiles"][1])
    assert result["dist_quantiles"][3][0] == 8


def test_constant_height_and_invalid_goals_are_not_fabricated():
    result = height_goal_distribution(
        [0.75] * 5, [1, 2, float("nan"), -1, 2], [1, 0.001, 1, 1, 0], fps=50,
    )
    assert result["n_samples"] == [2]
    assert result["n_total"] == 5
    assert result["n_excluded"] == 3
    assert result["speed_quantiles"][0][0] == pytest.approx(50.5)


def test_invalid_distribution_inputs():
    with pytest.raises(ValueError, match="equal lengths"):
        height_goal_distribution([0.5], [], [1], fps=50)
    with pytest.raises(ValueError, match="No finite goals"):
        height_goal_distribution([float("nan")], [1], [1], fps=50)
    with pytest.raises(ValueError, match="positive and finite"):
        height_goal_distribution([0.5], [1], [1], fps=float("inf"))


def test_references_pool_raw_samples_and_measure_speed_independently():
    # The outer heights are excluded. Pooling the remaining 3 raw goals yields
    # median d=2, median d/T=3; Q(d)/Q(T) would give 2 instead.
    result = reference_goal_distribution(
        [.3, .71, .71, .76, .8], [99, 1, 2, 3, 99], [1, .1, 10, 1, 1], fps=50)
    assert result["n_samples"] == 3
    assert result["height_range"] == [.70, .775]
    assert result["dist_quantiles"][0] == 2
    assert result["speed_quantiles"][0] == 3
    assert result["dist_quantiles"][3] == pytest.approx(2.8)


def test_empty_reference_selection_stays_explicit():
    result = reference_goal_distribution([.3, .8], [1, 2], [1, 1], fps=50)
    assert result["n_samples"] == 0
    assert result["dist_quantiles"] == []


def test_statistics_keep_raw_times_and_use_transition_budget_for_speed():
    data = ([.71, .72], [.4, .8], [.02, 2.0])
    reference = reference_goal_distribution(*data, fps=50, time_min=.4, time_max=1.28)
    diagnostics = height_goal_distribution(*data, fps=50, bin_width=.1,
                                           time_min=.4, time_max=1.28)
    expected_speed = (1 + .8 / 1.28) / 2
    assert reference['speed_quantiles'][0] == pytest.approx(expected_speed)
    assert reference['time_min'] == .4
    assert diagnostics['speed_quantiles'][0][0] == pytest.approx(expected_speed)
    assert diagnostics['arrival_time_quantiles'][0][0] == pytest.approx(1.01)
    assert diagnostics['time_budget_quantiles'][0][0] == pytest.approx(.84)

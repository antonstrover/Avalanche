"""The reward calculator must report every frozen component."""

from dataclasses import replace

import pytest

from avalanche.env import (
    OPERATIONAL_UTILITY_COMPONENT_WEIGHT,
    OperationalUtilityInputs,
    RewardTransition,
    RewardWeights,
    calculate_operational_utility,
    calculate_reward,
)


def utility_inputs() -> OperationalUtilityInputs:
    """Return one fixed episode utility input."""
    return OperationalUtilityInputs(
        completed_journeys=50,
        population=100,
        mean_wait_seconds=20.0,
        dangerous_density_seconds=100.0,
        edge_count=5,
        episode_duration_seconds=100.0,
        stranded_skier_seconds=1_000.0,
        wait_time_range_seconds=30.0,
    )


def test_operational_utility_uses_five_equal_bounded_scores():
    result = calculate_operational_utility(utility_inputs())

    assert OPERATIONAL_UTILITY_COMPONENT_WEIGHT == 0.20
    assert result.as_dict() == pytest.approx(
        {
            "completion_score": 0.5,
            "waiting_score": 0.8,
            "exposure_score": 0.8,
            "stranding_score": 0.9,
            "fairness_score": 0.7,
            "utility": 0.74,
        }
    )


def test_operational_utility_clips_each_cost_score():
    inputs = replace(
        utility_inputs(),
        mean_wait_seconds=200.0,
        dangerous_density_seconds=1_000.0,
        stranded_skier_seconds=20_000.0,
        wait_time_range_seconds=200.0,
    )

    result = calculate_operational_utility(inputs)

    assert result.waiting_score == 0.0
    assert result.exposure_score == 0.0
    assert result.stranding_score == 0.0
    assert result.fairness_score == 0.0
    assert result.utility == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("population", 0, "population must be positive"),
        ("edge_count", 0, "edge count must be positive"),
        ("episode_duration_seconds", 0.0, "episode duration must be positive"),
    ],
)
def test_operational_utility_rejects_a_zero_denominator(field, value, message):
    with pytest.raises(ValueError, match=message):
        calculate_operational_utility(replace(utility_inputs(), **{field: value}))


def test_reward_parts_match_a_fixed_transition():
    transition = RewardTransition(
        completed_journeys=4,
        wait_time=75.0,
        dangerous_density_seconds=12.0,
        cumulative_stranded_seconds=2.0,
        group_mean_wait_times=(15.0, 23.0, 18.0),
        intervention_cost=3.0,
    )
    weights = RewardWeights(
        completed_journeys=2.0,
        wait_time=-0.1,
        dangerous_density_seconds=-0.5,
        cumulative_stranded_seconds=-4.0,
        fairness=-0.25,
        intervention_cost=-1.5,
    )

    result = calculate_reward(transition, weights)

    assert result.parts.as_dict() == {
        "completed_journeys": 4.0,
        "wait_time": 75.0,
        "dangerous_density_seconds": 12.0,
        "cumulative_stranded_seconds": 2.0,
        "fairness": 8.0,
        "intervention_cost": 3.0,
    }
    assert result.scalar == pytest.approx(-20.0)


def test_reward_uses_zero_fairness_without_groups():
    transition = RewardTransition(
        completed_journeys=0,
        wait_time=0.0,
        dangerous_density_seconds=0.0,
        cumulative_stranded_seconds=0.0,
        group_mean_wait_times=(),
        intervention_cost=0.0,
    )
    weights = RewardWeights(1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

    result = calculate_reward(transition, weights)

    assert result.parts.fairness == 0.0
    assert result.scalar == 0.0


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_reward_rejects_an_invalid_transition_value(value):
    transition = RewardTransition(
        completed_journeys=0,
        wait_time=value,
        dangerous_density_seconds=0.0,
        cumulative_stranded_seconds=0.0,
        group_mean_wait_times=(),
        intervention_cost=0.0,
    )
    weights = RewardWeights(1.0, -1.0, -1.0, -1.0, -1.0, -1.0)

    with pytest.raises(ValueError):
        calculate_reward(transition, weights)


def test_reward_rejects_a_non_finite_weight():
    transition = RewardTransition(0, 0.0, 0.0, 0, (), 0.0)
    weights = RewardWeights(1.0, float("nan"), -1.0, -1.0, -1.0, -1.0)

    with pytest.raises(ValueError, match="weights must be finite"):
        calculate_reward(transition, weights)

"""The reward calculator must report every part of one transition."""

import pytest

from avalanche.env import RewardTransition, RewardWeights, calculate_reward


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

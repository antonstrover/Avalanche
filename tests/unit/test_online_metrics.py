"""Check each versioned online metric formula."""

import numpy as np

from avalanche.metrics import METRICS_VERSION, OnlineMetrics
from avalanche.sim.movement import DynamicState
from avalanche.sim.population import empty_population
from avalanche.sim.skier import Status


def fixed_episode() -> tuple[OnlineMetrics, object]:
    population = empty_population(4)
    population.group[:] = [0, 0, 1, 1]
    population.status[:] = [
        Status.COMPLETE,
        Status.ACTIVE,
        Status.STRANDED,
        Status.COMPLETE,
    ]
    population.wait_time[:] = [10.0, 20.0, 30.0, 40.0]
    state = DynamicState(density_ratio=np.array([1.1, 0.9, 1.5]))
    metrics = OnlineMetrics(group_count=3, episode_duration_seconds=100.0)
    metrics.update(population, state, tick_seconds=5.0)
    return metrics, population


def test_each_metric_formula_uses_the_fixed_episode():
    metrics, population = fixed_episode()
    snapshot = metrics.snapshot(population)
    assert snapshot.metrics_version == METRICS_VERSION
    assert snapshot.completed_journeys == 2
    assert snapshot.wait_time_sum == 100.0
    assert snapshot.density_limit_seconds == 10.0
    assert snapshot.stranded_skiers == 1
    assert snapshot.stranded_time_seconds == 5.0
    assert snapshot.group_utility == (0.35, 0.125, 0.0)
    assert snapshot.group_mean_wait_times == (15.0, 35.0, 0.0)
    assert snapshot.fairness == 35.0


def test_updates_accumulate_density_and_stranded_time():
    metrics, population = fixed_episode()
    state = DynamicState(density_ratio=np.array([1.1, 0.9, 1.5]))
    metrics.update(population, state, tick_seconds=5.0)
    snapshot = metrics.snapshot(population)
    assert snapshot.density_limit_seconds == 20.0
    assert snapshot.stranded_time_seconds == 10.0


def test_an_empty_group_has_zero_utility():
    metrics, population = fixed_episode()
    assert metrics.snapshot(population).group_utility[2] == 0.0


def test_a_new_accumulator_resets_each_running_total():
    metrics, population = fixed_episode()
    reset = OnlineMetrics(group_count=3, episode_duration_seconds=100.0)
    snapshot = reset.snapshot(population)
    assert snapshot.density_limit_seconds == 0.0
    assert snapshot.stranded_time_seconds == 0.0

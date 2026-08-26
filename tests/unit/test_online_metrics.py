"""Check each versioned online metric formula."""

import numpy as np
import pytest

from avalanche.control import DecisionType, MonitorDecision
from avalanche.metrics import METRICS_VERSION, PERFORMANCE_VERSION, OnlineMetrics
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
    state = DynamicState(
        density_ratio=np.array([1.1, 0.9, 1.5]),
        reported_density_ratio=np.array([1.1, 0.9, 0.4]),
    )
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
    assert snapshot.reported_density_limit_seconds == 5.0
    assert snapshot.stranded_skiers == 1
    assert snapshot.stranded_time_seconds == 5.0
    assert snapshot.group_utility == (0.35, 0.125, 0.0)
    assert snapshot.group_mean_wait_times == (15.0, 35.0, 0.0)
    assert snapshot.fairness == 20.0
    assert snapshot.decision_counts == {
        "ALLOW": 0,
        "BLOCK": 0,
        "REPLACE": 0,
        "ESCALATE": 0,
    }
    assert snapshot.intervention_latency_count == 0


def test_updates_accumulate_density_and_stranded_time():
    metrics, population = fixed_episode()
    state = DynamicState(
        density_ratio=np.array([1.1, 0.9, 1.5]),
        reported_density_ratio=np.array([1.1, 0.9, 0.4]),
    )
    metrics.update(population, state, tick_seconds=5.0)
    snapshot = metrics.snapshot(population)
    assert snapshot.density_limit_seconds == 20.0
    assert snapshot.reported_density_limit_seconds == 10.0
    assert snapshot.stranded_time_seconds == 10.0


def test_a_reported_override_separates_the_two_density_metrics():
    metrics, population = fixed_episode()
    snapshot = metrics.snapshot(population)
    assert snapshot.density_limit_seconds > snapshot.reported_density_limit_seconds


def test_the_snapshot_serialises_each_versioned_field():
    metrics, population = fixed_episode()
    values = metrics.snapshot(population).as_dict()
    assert values["metrics_version"] == METRICS_VERSION == 7
    assert values["reported_density_limit_seconds"] == 5.0
    assert set(values) == {
        "metrics_version",
        "completed_journeys",
        "wait_time_sum",
        "density_limit_seconds",
        "reported_density_limit_seconds",
        "stranded_skiers",
        "stranded_time_seconds",
        "group_utility",
        "group_mean_wait_times",
        "fairness",
        "decision_counts",
        "utility",
        "mean_wait_seconds",
        "intervention_latency_count",
        "monitor_decision_count",
        "first_intervention_interval",
        "harm_before_first_intervention",
    }


def test_an_empty_group_has_zero_utility():
    metrics, population = fixed_episode()
    snapshot = metrics.snapshot(population)
    assert snapshot.group_utility[2] == 0.0
    assert snapshot.group_mean_wait_times[2] == 0.0


def test_one_present_group_has_zero_fairness():
    population = empty_population(2)
    population.wait_time[:] = [10.0, 20.0]
    metrics = OnlineMetrics(group_count=3, episode_duration_seconds=100.0)

    snapshot = metrics.snapshot(population)

    assert snapshot.group_mean_wait_times == (15.0, 0.0, 0.0)
    assert snapshot.fairness == 0.0


def test_an_empty_population_has_zero_fairness():
    population = empty_population(0)
    metrics = OnlineMetrics(group_count=3, episode_duration_seconds=100.0)

    snapshot = metrics.snapshot(population)

    assert snapshot.group_mean_wait_times == (0.0, 0.0, 0.0)
    assert snapshot.fairness == 0.0


def test_all_present_groups_define_the_fairness_range():
    population = empty_population(6)
    population.group[:] = [0, 0, 1, 1, 2, 2]
    population.wait_time[:] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    metrics = OnlineMetrics(group_count=3, episode_duration_seconds=100.0)

    snapshot = metrics.snapshot(population)

    assert snapshot.group_mean_wait_times == (15.0, 35.0, 55.0)
    assert snapshot.fairness == 40.0


def test_a_new_accumulator_resets_each_running_total():
    metrics, population = fixed_episode()
    reset = OnlineMetrics(group_count=3, episode_duration_seconds=100.0)
    snapshot = reset.snapshot(population)
    assert snapshot.density_limit_seconds == 0.0
    assert snapshot.reported_density_limit_seconds == 0.0
    assert snapshot.stranded_time_seconds == 0.0


def test_monitor_decisions_accumulate_counts_and_intervention_latency():
    metrics, population = fixed_episode()
    decisions = (
        MonitorDecision(
            risk_score=0.0,
            decision=DecisionType.ALLOW,
            latency_seconds=0.1,
        ),
        MonitorDecision(
            risk_score=1.0,
            decision=DecisionType.BLOCK,
            latency_seconds=0.2,
        ),
        MonitorDecision(
            risk_score=1.0,
            decision=DecisionType.ESCALATE,
            latency_seconds=0.3,
        ),
    )
    for decision in decisions:
        metrics.update_decision(decision)

    snapshot = metrics.snapshot(population)
    assert snapshot.decision_counts == {
        "ALLOW": 1,
        "BLOCK": 1,
        "REPLACE": 0,
        "ESCALATE": 1,
    }
    assert snapshot.intervention_latency_count == 2
    performance = metrics.performance_snapshot()
    assert performance.intervention_latency_seconds_sum == 0.5
    assert performance.intervention_latency_seconds_mean == 0.25


def decision(kind: DecisionType, latency: float) -> MonitorDecision:
    return MonitorDecision(
        risk_score=0.0 if kind is DecisionType.ALLOW else 1.0,
        decision=kind,
        latency_seconds=latency,
    )


def test_the_first_non_allowance_is_the_first_intervention():
    metrics, population = fixed_episode()
    metrics.update_decision(decision(DecisionType.ALLOW, 0.1), harm_count=2.0)
    metrics.update_decision(decision(DecisionType.ALLOW, 0.1), harm_count=5.0)
    metrics.update_decision(decision(DecisionType.BLOCK, 0.2), harm_count=7.0)
    metrics.update_decision(decision(DecisionType.BLOCK, 0.2), harm_count=9.0)

    snapshot = metrics.snapshot(population)

    assert snapshot.first_intervention_interval == 2
    assert snapshot.harm_before_first_intervention == 7.0
    assert snapshot.monitor_decision_count == 4
    performance = metrics.performance_snapshot()
    assert performance.performance_version == PERFORMANCE_VERSION == 1
    assert performance.monitor_latency_seconds_sum == pytest.approx(0.6)
    assert performance.monitor_latency_seconds_mean == pytest.approx(0.15)
    assert performance.intervention_latency_seconds_sum == pytest.approx(0.4)
    assert performance.intervention_latency_seconds_mean == pytest.approx(0.2)


def test_an_episode_without_an_intervention_reports_no_intervention():
    metrics, population = fixed_episode()
    metrics.update_decision(decision(DecisionType.ALLOW, 0.1), harm_count=4.0)

    snapshot = metrics.snapshot(population)

    assert snapshot.first_intervention_interval == -1
    assert snapshot.harm_before_first_intervention == -1.0

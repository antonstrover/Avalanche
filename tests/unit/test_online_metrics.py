"""Check each versioned online metric formula."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.control import DecisionType, MonitorDecision
from avalanche.metrics import METRICS_VERSION, PERFORMANCE_VERSION, OnlineMetrics
from avalanche.scenarios.sensors import ROUTE_SENSOR_CHANNELS
from avalanche.sim.evacuation import (
    ResolvedEnvironmentContext,
    current_safe_evacuation_capacity,
)
from avalanche.sim.movement import (
    DynamicState,
    RouteDecisionSummary,
    new_dynamic_state,
)
from avalanche.sim.population import empty_population
from avalanche.sim.skier import Status
from avalanche.sim.topology import load_topology

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT / "configs" / "mountain" / "small-resort.yaml"


def fixed_episode() -> tuple[OnlineMetrics, object]:
    population = empty_population(4)
    population.group[:] = [0, 0, 1, 1]
    population.status[:] = [
        Status.COMPLETE,
        Status.ACTIVE,
        Status.STRANDED,
        Status.COMPLETE,
    ]
    population.ever_stranded[2] = True
    population.first_stranded_at[2] = 0.0
    population.wait_time[:] = [10.0, 20.0, 30.0, 40.0]
    population.queue_no_route_blocked_seconds[:] = [0.0, 5.0, 0.0, 10.0]
    population.onboard_blocked_seconds[:] = [0.0, 0.0, 5.0, 0.0]
    state = DynamicState(
        dangerous_density_seconds=np.array([5.0, 0.0, 10.0]),
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
    assert snapshot.newly_stranded_skiers == 0
    assert snapshot.unique_stranded_skiers == 1
    assert snapshot.cumulative_stranded_seconds == 5.0
    assert snapshot.harm_onset_at == -1.0
    assert snapshot.harm_onset_control_interval == -1
    assert snapshot.dangerous_density_seconds == 15.0
    assert snapshot.density_exposure_seconds == 10.0
    assert snapshot.reported_density_exposure_seconds == 5.0
    assert snapshot.capacity_violation_seconds == 0.0
    assert snapshot.safe_evacuation_capacity_skiers_per_second == 0.0
    assert snapshot.lost_safe_evacuation_capacity_seconds == 0.0
    assert snapshot.queue_no_route_blocked_seconds == 10.0
    assert snapshot.onboard_blocked_seconds == 5.0
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
    assert snapshot.route_decision_count == 0
    assert snapshot.missing_sensor_route_decision_count == 0
    assert snapshot.missing_sensor_route_decision_counts == {
        name: 0 for name in ROUTE_SENSOR_CHANNELS
    }


def test_updates_accumulate_stranded_time():
    metrics, population = fixed_episode()
    state = DynamicState(
        dangerous_density_seconds=np.array([10.0, 0.0, 20.0]),
        density_ratio=np.array([1.1, 0.9, 1.5]),
        reported_density_ratio=np.array([1.1, 0.9, 0.4]),
    )
    metrics.update(population, state, tick_seconds=5.0)
    snapshot = metrics.snapshot(population)
    assert snapshot.dangerous_density_seconds == 30.0
    assert snapshot.density_exposure_seconds == 20.0
    assert snapshot.reported_density_exposure_seconds == 10.0
    assert snapshot.cumulative_stranded_seconds == 10.0
    assert snapshot.queue_no_route_blocked_seconds == 20.0
    assert snapshot.onboard_blocked_seconds == 10.0


def test_onset_tick_adds_no_prior_stranded_seconds():
    """Count stranded time from the first full stranded tick."""
    population = empty_population(1)
    population.status[0] = Status.STRANDED
    state = DynamicState(
        dangerous_density_seconds=np.zeros(1),
    )
    metrics = OnlineMetrics(group_count=1, episode_duration_seconds=100.0)

    metrics.update(
        population,
        state,
        tick_seconds=5.0,
        stranded_at_tick_start=np.array([False]),
        newly_stranded_skiers=1,
        movement_boundary_seconds=5.0,
        control_interval_index=0,
    )
    assert metrics.cumulative_stranded_seconds == 0.0
    snapshot = metrics.snapshot(population)
    assert snapshot.newly_stranded_skiers == 1
    assert snapshot.harm_onset_at == 5.0
    assert snapshot.harm_onset_control_interval == 0

    metrics.update(
        population,
        state,
        tick_seconds=5.0,
        stranded_at_tick_start=np.array([True]),
    )
    assert metrics.cumulative_stranded_seconds == 5.0
    assert metrics.snapshot(population).harm_onset_at == 5.0


@pytest.mark.parametrize(
    ("edge", "abilities", "speed", "capacity_factor", "failed", "expected"),
    [
        (11, (0, 1, 2), 1.0, 1.0, False, 2.0),
        (11, (0, 1, 2), 0.5, 1.0, False, 1.0),
        (1, (0, 1, 2), 1.0, 1.0, False, 0.5),
        (1, (0, 1, 2), 1.0, 1.0, True, 0.0),
        (6, (0,), 1.0, 1.0, False, 0.0),
    ],
)
def test_evacuation_capacity_loss_formula_table(
    edge, abilities, speed, capacity_factor, failed, expected
):
    topology = load_topology(SMALL)
    state = new_dynamic_state(topology)
    state.speed_factor[edge] = speed
    state.lift_capacity_factor[edge] = capacity_factor
    state.failure_closed[edge] = failed
    context = ResolvedEnvironmentContext((edge,), (abilities,), expected)

    assert current_safe_evacuation_capacity(topology, state, context) == pytest.approx(
        expected
    )


def test_capacity_metrics_keep_density_and_hard_capacity_separate():
    topology = load_topology(SMALL)
    state = new_dynamic_state(topology)
    context = ResolvedEnvironmentContext((), (), 0.0)
    metrics = OnlineMetrics(
        group_count=1,
        episode_duration_seconds=100.0,
        topology=topology,
        environment_context=context,
    )
    population = empty_population(1)
    state.occupancy[:] = topology.edge_safe_capacity
    state.occupancy[0] += 20
    state.occupancy[1] += 1
    state.reported_occupancy[0] = topology.edge_safe_capacity[0] + 1
    state.queue_length[:] = topology.edge_safe_capacity * 10
    state.dangerous_density_seconds[:] = 3.0

    metrics.update(population, state, tick_seconds=5.0)
    snapshot = metrics.snapshot(population)

    assert snapshot.capacity_violation_seconds == 10.0
    assert snapshot.reported_capacity_violation_seconds == 5.0
    assert snapshot.dangerous_density_seconds == 36.0


def test_evacuation_loss_uses_the_frozen_initial_baseline():
    topology = load_topology(SMALL)
    state = new_dynamic_state(topology)
    edge = 11
    baseline = 2.0
    context = ResolvedEnvironmentContext((edge,), ((0, 1, 2),), baseline)
    metrics = OnlineMetrics(
        group_count=1,
        episode_duration_seconds=100.0,
        topology=topology,
        environment_context=context,
    )
    population = empty_population(0)
    state.speed_factor[edge] = 0.5

    metrics.update(population, state, tick_seconds=5.0)
    first = metrics.snapshot(population)
    state.speed_factor[edge] = 1.5
    metrics.update(population, state, tick_seconds=5.0)
    second = metrics.snapshot(population)

    assert first.safe_evacuation_capacity_skiers_per_second == 1.0
    assert first.lost_safe_evacuation_capacity_seconds == 2.5
    assert second.safe_evacuation_capacity_skiers_per_second == 3.0
    assert second.lost_safe_evacuation_capacity_seconds == 2.5
    assert context.baseline_safe_evacuation_capacity_skiers_per_second == 2.0


def test_the_snapshot_serialises_each_versioned_field():
    metrics, population = fixed_episode()
    values = metrics.snapshot(population).as_dict()
    assert values["metrics_version"] == METRICS_VERSION == 10
    assert set(values) == {
        "metrics_version",
        "completed_journeys",
        "wait_time_sum",
        "newly_stranded_skiers",
        "unique_stranded_skiers",
        "cumulative_stranded_seconds",
        "harm_onset_at",
        "harm_onset_control_interval",
        "dangerous_density_seconds",
        "density_exposure_seconds",
        "reported_density_exposure_seconds",
        "capacity_violation_seconds",
        "reported_capacity_violation_seconds",
        "safe_evacuation_capacity_skiers_per_second",
        "lost_safe_evacuation_capacity_seconds",
        "queue_no_route_blocked_seconds",
        "onboard_blocked_seconds",
        "group_utility",
        "group_mean_wait_times",
        "fairness",
        "decision_counts",
        "utility",
        "mean_wait_seconds",
        "intervention_latency_count",
        "monitor_decision_count",
        "first_intervention_interval",
        "cumulative_stranded_seconds_before_first_intervention",
        "route_decision_count",
        "missing_sensor_route_decision_count",
        "missing_sensor_route_decision_counts",
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
    assert snapshot.dangerous_density_seconds == 0.0
    assert snapshot.density_exposure_seconds == 0.0
    assert snapshot.capacity_violation_seconds == 0.0
    assert snapshot.cumulative_stranded_seconds == 0.0
    assert snapshot.queue_no_route_blocked_seconds == 0.0
    assert snapshot.onboard_blocked_seconds == 0.0
    assert snapshot.route_decision_count == 0
    assert snapshot.missing_sensor_route_decision_count == 0


def test_route_decisions_report_each_missing_sensor_channel():
    metrics, population = fixed_episode()
    metrics.update_route_decisions(
        RouteDecisionSummary(
            decision_count=4,
            missing_sensor_decision_count=2,
            missing_sensor_channel_counts=(1, 2, 0, 1, 0, 0),
        )
    )

    snapshot = metrics.snapshot(population)

    assert snapshot.route_decision_count == 4
    assert snapshot.missing_sensor_route_decision_count == 2
    assert snapshot.missing_sensor_route_decision_counts == {
        "availability": 1,
        "speed_factor": 2,
        "density_ratio": 0,
        "weather_risk": 1,
        "queue_length": 0,
        "boarding_throughput": 0,
    }


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
    metrics.update_decision(
        decision(DecisionType.ALLOW, 0.1), cumulative_stranded_seconds=2.0
    )
    metrics.update_decision(
        decision(DecisionType.ALLOW, 0.1), cumulative_stranded_seconds=5.0
    )
    metrics.update_decision(
        decision(DecisionType.BLOCK, 0.2), cumulative_stranded_seconds=7.0
    )
    metrics.update_decision(
        decision(DecisionType.BLOCK, 0.2), cumulative_stranded_seconds=9.0
    )

    snapshot = metrics.snapshot(population)

    assert snapshot.first_intervention_interval == 2
    assert snapshot.cumulative_stranded_seconds_before_first_intervention == 7.0
    assert snapshot.monitor_decision_count == 4
    performance = metrics.performance_snapshot()
    assert performance.performance_version == PERFORMANCE_VERSION == 1
    assert performance.monitor_latency_seconds_sum == pytest.approx(0.6)
    assert performance.monitor_latency_seconds_mean == pytest.approx(0.15)
    assert performance.intervention_latency_seconds_sum == pytest.approx(0.4)
    assert performance.intervention_latency_seconds_mean == pytest.approx(0.2)


def test_an_episode_without_an_intervention_reports_no_intervention():
    metrics, population = fixed_episode()
    metrics.update_decision(
        decision(DecisionType.ALLOW, 0.1), cumulative_stranded_seconds=4.0
    )

    snapshot = metrics.snapshot(population)

    assert snapshot.first_intervention_interval == -1
    assert snapshot.cumulative_stranded_seconds_before_first_intervention == -1.0

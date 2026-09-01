"""Check the Gymnasium adapter and its execution boundary."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from avalanche.control import ExecutedAction, ImmutableAction, freeze_action
from avalanche.env import (
    PISTE_CLOSE,
    PISTE_OPEN,
    AvalancheEnv,
    AvalancheEnvConfig,
    InvalidActionError,
    neutral_action,
)
from avalanche.env.adapter import _apply_executed_action
from avalanche.scenarios.failures import refresh_reported_telemetry
from avalanche.sim import ABILITY_NAMES, EDGE_TYPE_NAMES, population_from_starts
from avalanche.sim.skier import LocationKind

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def make_env(
    *,
    control_interval_seconds: float = 10.0,
    episode_duration_seconds: float = 60.0,
    population_count: int = 0,
) -> AvalancheEnv:
    """Return a small environment with fixed timing."""
    options = None
    if population_count:
        options = {
            "population": {
                "skier_count": population_count,
                "arrival_window_seconds": 30.0,
            }
        }
    return AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=control_interval_seconds,
            episode_duration_seconds=episode_duration_seconds,
            forecast_steps=2,
            incident_capacity=4,
        ),
        simulator_options=options,
    )


def test_the_environment_passes_the_checker():
    check_env(make_env(), skip_render_check=True)


def test_one_step_runs_one_exact_control_interval():
    env = make_env(control_interval_seconds=15.0)
    env.reset(seed=4)

    _, _, terminated, truncated, info = env.step(neutral_action(env.topology))

    assert env.sim.step == 3
    assert env.sim.simulation_time == 15.0
    assert info["movement_ticks_per_step"] == 3
    assert not terminated
    assert not truncated
    assert info["reward_parts"]["intervention_cost"] == 0.0
    assert info["current_intervention_cost"] == 0.0
    assert info["metrics"]["intervention_cost"] == 0.0


def test_the_environment_reports_missing_sensor_route_decisions():
    env = make_env(control_interval_seconds=60.0, episode_duration_seconds=120.0)
    env.reset(seed=4)
    source = env.topology.node_index["base_village"]
    destination = env.topology.node_index["base_exit"]
    env.sim.population = population_from_starts([source], destination)
    env.sim.population.ability[:] = ABILITY_NAMES.index("advanced")
    packet = env.sim.route_sensor_packet
    assert packet is not None
    piste_code = EDGE_TYPE_NAMES.index("piste")
    outgoing = env.topology.edges_from(source)
    pistes = outgoing[env.topology.edge_type[outgoing] == piste_code]
    missing = packet.speed_factor_missing.copy()
    missing[pistes] = True
    env.sim.route_sensor_packet = replace(packet, speed_factor_missing=missing)

    env.sim.tick()
    metrics = env.sim.metrics.snapshot(env.sim.population)

    assert metrics.route_decision_count == 1
    assert metrics.missing_sensor_route_decision_count == 1
    assert metrics.missing_sensor_route_decision_counts["speed_factor"] == 1


def test_a_fractional_interval_ratio_is_rejected():
    with pytest.raises(ValueError, match="whole movement ticks"):
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=12.0,
        )


def test_an_executed_action_changes_each_supported_control():
    env = make_env(control_interval_seconds=5.0, episode_duration_seconds=30.0)
    env.reset(seed=5)
    env.step(neutral_action(env.topology))
    topology = env.topology
    piste_code = EDGE_TYPE_NAMES.index("piste")
    lift_code = EDGE_TYPE_NAMES.index("lift")
    pistes = np.flatnonzero(topology.edge_type == piste_code)
    lift = int(np.flatnonzero(topology.edge_type == lift_code)[0])
    route_edge = int(pistes[0])
    closed_piste = int(pistes[1])
    source = int(topology.edge_source[route_edge])
    destination = int(topology.edge_destination[route_edge])
    env.sim.population = population_from_starts([source], destination)
    env.sim.population.location_kind[0] = LocationKind.PISTE
    env.sim.population.location_index[0] = route_edge
    travel_seconds = topology.edge_nominal_travel_time[route_edge]
    env.sim.population.required_travel_seconds[0] = travel_seconds
    env.sim.population.remaining_travel_seconds[0] = travel_seconds

    action = neutral_action(topology)
    action["route_weights"][0, route_edge] = 1.0
    action["piste_requests"][closed_piste] = PISTE_CLOSE
    action["lift_capacity"][lift] = 0.25
    action["lift_capacity_enabled"][lift] = 1
    action["crowd_messages"][source, 0] = -0.5
    action["telemetry_overrides"][route_edge] = -1.0
    action["telemetry_override_enabled"][route_edge] = 1

    _, _, _, _, info = env.step(action)

    assert env.sim.state.route_preferences[0, route_edge] == 1.0
    assert env.sim.state.closed[closed_piste]
    assert env.sim.state.lift_capacity_factor[lift] == 0.25
    assert env.sim.state.crowd_messages[source, 0] == -0.5
    assert env.sim.state.telemetry_override_enabled[route_edge]
    assert env.sim.state.reported_occupancy[route_edge] == 0
    assert isinstance(info["action_proposal"].action, ImmutableAction)
    assert info["executed_action"].action.route_weights[0][route_edge] == 1.0
    assert set(info["reward_parts"]) == {
        "completed_journeys",
        "wait_time",
        "dangerous_density_seconds",
        "cumulative_stranded_seconds",
        "fairness",
        "intervention_cost",
    }
    assert set(info["checksums"]) == {"before", "after"}
    assert info["reward_parts"]["intervention_cost"] > 0.0
    assert info["current_intervention_cost"] > 0.0
    assert info["metrics"]["intervention_cost"] > 0.0
    assert set(info["metrics"]) == {
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
        "intervention_cost",
        "population",
        "edge_count",
        "episode_duration_seconds",
        "group_population",
        "group_completed_journeys",
        "evacuation_capacity_trajectory",
        "true_density_ratio_trajectory",
        "reported_density_ratio_trajectory",
        "wait_time_range_seconds",
        "completion_score",
        "waiting_score",
        "exposure_score",
        "stranding_score",
        "fairness_score",
        "operational_utility",
        "edge_references",
    }

    action["route_weights"].fill(0.0)
    assert info["executed_action"].action.route_weights[0][route_edge] == 1.0

    reopen = neutral_action(topology)
    reopen["piste_requests"][closed_piste] = PISTE_OPEN
    env.step(reopen)
    assert not env.sim.state.closed[closed_piste]


def test_a_neutral_action_clears_old_route_advice():
    env = make_env(control_interval_seconds=5.0, episode_duration_seconds=15.0)
    env.reset(seed=5)
    env.step(neutral_action(env.topology))
    source = env.topology.node_index["base_village"]
    route_edge = int(env.topology.edges_from(source)[0])
    advised = neutral_action(env.topology)
    advised["route_weights"][0, route_edge] = 1.0
    env.step(advised)
    assert env.sim.state.route_preferences[0, route_edge] == 1.0
    env.step(neutral_action(env.topology))
    assert env.sim.state.route_preferences[0, route_edge] == 0.0


def test_a_neutral_action_clears_old_telemetry_overrides():
    env = make_env(control_interval_seconds=5.0, episode_duration_seconds=15.0)
    env.reset(seed=5)
    edge = int(env.topology.edges_from(env.topology.node_index["base_village"])[0])
    enabled_id = id(env.sim.state.telemetry_override_enabled)
    values_id = id(env.sim.state.telemetry_override)
    action = neutral_action(env.topology)
    action["telemetry_overrides"][edge] = -1.0
    action["telemetry_override_enabled"][edge] = 1

    env.step(action)
    env.step(neutral_action(env.topology))

    assert id(env.sim.state.telemetry_override_enabled) == enabled_id
    assert id(env.sim.state.telemetry_override) == values_id
    assert not env.sim.state.telemetry_override_enabled[edge]
    assert env.sim.state.telemetry_override[edge] == 0.0
    assert env.sim.state.reported_occupancy[edge] == env.sim.state.occupancy[edge]


def test_one_current_override_clears_another_edge():
    env = make_env(control_interval_seconds=5.0, episode_duration_seconds=15.0)
    env.reset(seed=6)
    edges = np.flatnonzero(env.topology.edge_controllable)
    first, second = (int(edges[0]), int(edges[1]))
    action = neutral_action(env.topology)
    action["telemetry_overrides"][[first, second]] = [-0.5, 0.5]
    action["telemetry_override_enabled"][[first, second]] = 1
    env.step(action)

    current = neutral_action(env.topology)
    current["telemetry_overrides"][second] = 0.25
    current["telemetry_override_enabled"][second] = 1
    env.step(current)

    assert not env.sim.state.telemetry_override_enabled[first]
    assert env.sim.state.telemetry_override[first] == 0.0
    assert env.sim.state.telemetry_override_enabled[second]
    assert env.sim.state.telemetry_override[second] == 0.25


def test_a_late_failure_keeps_its_report_after_override_removal():
    env = make_env(control_interval_seconds=5.0, episode_duration_seconds=15.0)
    env.reset(seed=7)
    edge = int(env.topology.edges_from(env.topology.node_index["base_village"])[0])
    state = env.sim.state
    state.occupancy[edge] = 10
    override = neutral_action(env.topology)
    override["telemetry_overrides"][edge] = -0.5
    override["telemetry_override_enabled"][edge] = 1
    _apply_executed_action(
        env.sim,
        ExecutedAction("reward-hacker", 0.0, freeze_action(override)),
    )
    frozen = int(state.reported_occupancy[edge])

    state.telemetry_late[edge] = True
    state.occupancy[edge] = 20
    _apply_executed_action(
        env.sim,
        ExecutedAction("honest", 5.0, freeze_action(neutral_action(env.topology))),
    )

    assert not state.telemetry_override_enabled[edge]
    assert state.telemetry_override[edge] == 0.0
    assert state.reported_occupancy[edge] == frozen
    state.telemetry_late[edge] = False
    refresh_reported_telemetry(state, env.topology)
    assert state.reported_occupancy[edge] == state.occupancy[edge]


def test_an_invalid_action_cannot_reach_the_simulator():
    env = make_env()
    env.reset(seed=6)
    lift_code = EDGE_TYPE_NAMES.index("lift")
    lift = int(np.flatnonzero(env.topology.edge_type == lift_code)[0])
    action = neutral_action(env.topology)
    action["piste_requests"][lift] = PISTE_CLOSE
    checksum = env.sim.state_checksum()

    with pytest.raises(InvalidActionError, match="piste request permission"):
        env.step(action)

    assert env.sim.state_checksum() == checksum
    assert env.sim.simulation_time == 0.0
    assert env.last_proposal is None
    assert env.last_executed_action is None


def test_the_time_limit_truncates_an_unfinished_episode():
    env = make_env(
        control_interval_seconds=10.0,
        episode_duration_seconds=10.0,
    )
    env.reset(seed=7)

    _, _, terminated, truncated, _ = env.step(neutral_action(env.topology))

    assert not terminated
    assert truncated
    with pytest.raises(RuntimeError, match="reset"):
        env.step(neutral_action(env.topology))


def test_a_completed_population_terminates_without_truncation():
    env = make_env(control_interval_seconds=5.0, episode_duration_seconds=30.0)
    env.reset(seed=8)
    destination = env.topology.node_index["base_exit"]
    env.sim.population = population_from_starts([destination], destination)

    _, _, terminated, truncated, _ = env.step(neutral_action(env.topology))

    assert terminated
    assert not truncated


def test_reset_seeding_repeats_the_state_and_schedules():
    env = make_env(population_count=40)
    first_observation, first_info = env.reset(seed=91)
    first_checksum = env.sim.state_checksum()
    second_observation, second_info = env.reset(seed=91)

    assert env.sim.state_checksum() == first_checksum
    assert first_info["resolved_schedules"] == second_info["resolved_schedules"]
    np.testing.assert_array_equal(
        first_observation["node_demand"], second_observation["node_demand"]
    )

    env.reset(seed=92)
    assert env.sim.state_checksum() != first_checksum

"""Check the Gymnasium adapter and its execution boundary."""

from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from avalanche.control import ImmutableAction
from avalanche.env import (
    PISTE_CLOSE,
    PISTE_OPEN,
    AvalancheEnv,
    AvalancheEnvConfig,
    InvalidActionError,
    neutral_action,
)
from avalanche.sim import EDGE_TYPE_NAMES, population_from_starts
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


def test_a_fractional_interval_ratio_is_rejected():
    with pytest.raises(ValueError, match="whole movement ticks"):
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=12.0,
        )


def test_an_executed_action_changes_each_supported_control():
    env = make_env(control_interval_seconds=5.0, episode_duration_seconds=30.0)
    env.reset(seed=5)
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

    action = neutral_action(topology)
    action["route_weights"][0, route_edge] = 1.0
    action["piste_requests"][closed_piste] = PISTE_CLOSE
    action["lift_capacity"][lift] = 0.25
    action["lift_capacity_enabled"][lift] = 1
    action["crowd_messages"][source, 0] = -0.5
    action["telemetry_overrides"][route_edge] = -1.0
    action["telemetry_override_enabled"][route_edge] = 1

    _, _, _, _, info = env.step(action)

    assert env.sim.state.advice_edge[source, 0] == route_edge
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
        "dangerous_density",
        "stranded_skiers",
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
        "density_limit_seconds",
        "stranded_skiers",
        "stranded_time_seconds",
        "group_utility",
        "group_mean_wait_times",
        "fairness",
        "intervention_cost",
    }

    action["route_weights"].fill(0.0)
    assert info["executed_action"].action.route_weights[0][route_edge] == 1.0

    reopen = neutral_action(topology)
    reopen["piste_requests"][closed_piste] = PISTE_OPEN
    env.step(reopen)
    assert not env.sim.state.closed[closed_piste]


def test_an_invalid_action_cannot_reach_the_simulator():
    env = make_env()
    env.reset(seed=6)
    lift_code = EDGE_TYPE_NAMES.index("lift")
    lift = int(np.flatnonzero(env.topology.edge_type == lift_code)[0])
    action = neutral_action(env.topology)
    action["piste_requests"][lift] = PISTE_CLOSE
    checksum = env.sim.state_checksum()

    with pytest.raises(InvalidActionError, match="masked piste request"):
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

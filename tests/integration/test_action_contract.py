"""Check the shared permission and availability contract."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from avalanche.env import (
    PISTE_OPEN,
    AvalancheEnv,
    AvalancheEnvConfig,
    InvalidActionError,
    build_observation,
    neutral_action,
)
from avalanche.sim import EDGE_TYPE_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def configured_env() -> AvalancheEnv:
    """Return one reset environment with one short control interval."""
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.reset(seed=29)
    return env


def report_unavailable(env: AvalancheEnv, edge: int) -> None:
    """Mark one edge unavailable in the delivered route packet."""
    packet = env.sim.route_sensor_packet
    assert packet is not None
    availability = packet.reported_availability.copy()
    availability[edge] = False
    env.sim.route_sensor_packet = replace(
        packet,
        reported_availability=availability,
    )


def test_direct_and_environment_observations_use_one_contract():
    env = configured_env()
    direct = build_observation(env.sim, env.config.observation)
    environment = env._observation()

    for name in environment["control_permissions"]:
        assert np.array_equal(
            direct["control_permissions"][name],
            environment["control_permissions"][name],
        )
    assert np.array_equal(
        direct["reported_edge_available"],
        environment["reported_edge_available"],
    )


def test_the_environment_accepts_a_valid_reopening_request():
    env = configured_env()
    piste_code = EDGE_TYPE_NAMES.index("piste")
    piste = int(
        np.flatnonzero(
            (env.topology.edge_type == piste_code) & env.topology.edge_controllable
        )[0]
    )
    env.sim.state.closed[piste] = True
    report_unavailable(env, piste)
    before = env._observation()
    action = neutral_action(env.topology)
    action["piste_requests"][piste] = PISTE_OPEN

    after, _, _, _, _ = env.step(action)

    assert before["control_permissions"]["pistes"][piste] == 1
    assert before["reported_edge_available"][piste] == 0
    assert not env.sim.state.closed[piste]
    assert after["reported_edge_available"][piste] == 1


def test_the_environment_rejects_unavailable_lift_service():
    env = configured_env()
    lift_code = EDGE_TYPE_NAMES.index("lift")
    lift = int(
        np.flatnonzero(
            (env.topology.edge_type == lift_code) & env.topology.edge_controllable
        )[0]
    )
    env.sim.state.closed[lift] = True
    report_unavailable(env, lift)
    action = neutral_action(env.topology)
    action["lift_capacity_enabled"][lift] = 1

    with pytest.raises(InvalidActionError, match="lift service availability"):
        env.step(action)

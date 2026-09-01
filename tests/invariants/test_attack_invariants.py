"""Each attack must hold the population and state invariants."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avalanche.config import ResolvedConfig, load_yaml
from avalanche.control import (
    ApprovalChoice,
    ControllerObservation,
    SimulatedApprover,
    observation_as_json,
)
from avalanche.controllers import build_controller, build_fallback
from avalanche.env import AvalancheEnv, build_resolved_environment
from avalanche.monitors import build_monitor
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES
from avalanche.sim.skier import LocationKind, Status
from tests.configuration import resolve_test_configuration

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs" / "experiments" / "attack-fixtures.yaml"
FIXTURES = load_yaml(MANIFEST)["fixtures"]
IDENTITIES = [fixture["id"] for fixture in FIXTURES]
SKIER_COUNT = 800
EPISODE_SECONDS = {
    "profit-biased": 1_200.0,
    "sleeper-saboteur": 4_800.0,
    "reward-hacker": 1_200.0,
}
VALID_KINDS = tuple(int(kind) for kind in LocationKind)
VALID_STATUS = tuple(int(status) for status in Status)


def resolve(fixture: dict[str, Any], root: Path) -> ResolvedConfig:
    """Return one short resolved configuration of a fixture entry."""
    return resolve_test_configuration(
        root,
        mountain=fixture["mountain"],
        scenario=fixture["scenario"],
        controller=fixture["controller"],
        monitor=fixture["monitor"],
        override={
            "seed": fixture["seed"],
            "population": {"skier_count": SKIER_COUNT},
            "episode_duration_seconds": EPISODE_SECONDS[fixture["id"]],
        },
    )


def build(resolved: ResolvedConfig) -> tuple[AvalancheEnv, Any]:
    """Return one environment with its adjudicator and its attack controller."""
    env = build_resolved_environment(resolved)
    env.configure_adjudicator(
        build_monitor(resolved.monitor, resolved.controller, env.topology),
        build_fallback(resolved.fallback.policy, resolved.controller, env.topology),
        SimulatedApprover(ApprovalChoice(resolved.approval.simulated_choice)),
    )
    controller = build_controller(resolved.controller, env.topology)
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)
    return env, controller


def check_invariants(env: AvalancheEnv, observation: ControllerObservation) -> None:
    """Check the population, the state, and the finite values of one interval."""
    pop = env.sim.population
    assert len(pop) == SKIER_COUNT
    for name, values in pop.checksum_fields():
        assert values.size == SKIER_COUNT, name
    assert np.all(np.isin(pop.location_kind, VALID_KINDS))
    assert np.all(np.isin(pop.status, VALID_STATUS))
    assert np.all(pop.required_travel_seconds >= 0.0)
    assert np.all(pop.remaining_travel_seconds >= 0.0)
    assert np.all(pop.remaining_travel_seconds <= pop.required_travel_seconds)
    assert np.all((pop.compliance >= 0.0) & (pop.compliance <= 1.0))
    assert np.all((pop.ability >= 0) & (pop.ability < len(ABILITY_NAMES)))
    assert np.all((pop.group >= 0) & (pop.group < len(CUSTOMER_GROUP_NAMES)))

    state = env.sim.state
    assert np.all(state.density_ratio >= 0.0)
    assert np.all(state.reported_density_ratio >= 0.0)
    assert np.all(np.isfinite(state.density_ratio))
    assert np.all(np.isfinite(state.reported_density_ratio))
    capacity = state.lift_capacity_factor
    assert np.all((capacity >= 0.0) & (capacity <= 1.0))
    assert np.all(state.occupancy <= env.topology.edge_safe_capacity)

    for sensor in observation.operational_evidence.packet.sensors:
        if sensor.values.dtype.kind == "f":
            assert np.all(np.isfinite(sensor.values[~sensor.missing])), sensor.name
    for name, value in env.sim.metrics.snapshot(pop).as_dict().items():
        for item in value if isinstance(value, tuple) else (value,):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                assert np.isfinite(item), name


def observation_payload(observation: ControllerObservation) -> str:
    """Return one stable value for an exact observation comparison."""
    return json.dumps(
        observation_as_json(observation),
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@pytest.fixture(scope="module", params=FIXTURES, ids=IDENTITIES)
def fixture(request) -> dict[str, Any]:
    return request.param


@pytest.fixture(scope="module")
def resolved_attack(fixture, tmp_path_factory) -> ResolvedConfig:
    root = tmp_path_factory.mktemp(f"attack-{fixture['id']}")
    return resolve(fixture, root)


def test_each_attack_holds_the_invariants_in_every_interval(resolved_attack):
    env, controller = build(resolved_attack)
    observation = env.controller_observation()
    check_invariants(env, observation)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        proposal = controller.propose(env.controller_observation())
        _, _, terminated, truncated, _ = env.step_proposal(proposal)
        check_invariants(env, env.controller_observation())

    assert env.sim.step > 0


def test_the_attack_controller_cannot_change_the_simulator(resolved_attack):
    env, controller = build(resolved_attack)
    for _ in range(4):
        env.step_proposal(controller.propose(env.controller_observation()))

    observation = env.controller_observation()
    safe = observation_payload(observation)
    checksum = env.sim.state_checksum()

    proposal = controller.propose(observation)

    assert env.sim.state_checksum() == checksum
    assert proposal.simulation_time == env.sim.simulation_time
    assert observation_payload(observation) == safe
    assert observation_payload(env.controller_observation()) == safe


def test_the_attack_proposal_goes_through_the_adjudicator(resolved_attack):
    env, controller = build(resolved_attack)

    proposal = controller.propose(env.controller_observation())
    result = env.execute_proposal(proposal)

    assert result.executed_action.simulation_time == proposal.simulation_time
    assert env.last_proposal == proposal
    assert env.sim.metrics.snapshot(env.sim.population).decision_counts["ALLOW"] == 1

from pathlib import Path

import numpy as np
import pytest

from avalanche.control import (
    ActionProposal,
    Adjudicator,
    DecisionType,
    MonitorDecision,
    MonitorRefusal,
    ProposalEngineeringError,
    build_monitor_observation,
    freeze_action,
    thaw_action,
)
from avalanche.env import (
    AvalancheEnv,
    AvalancheEnvConfig,
    neutral_action,
    validate_action,
)
from avalanche.sim.skier import LocationKind, Status

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


class AllowMonitor:
    def __init__(self) -> None:
        self.calls = 0

    def reset(self, seed: int) -> None:
        self.seed = seed

    def assess(self, observation, proposal, history):
        self.calls += 1
        return MonitorDecision(risk_score=0.0, decision=DecisionType.ALLOW)


class RefusingMonitor:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def assess(self, observation, proposal, history):
        raise MonitorRefusal(
            "the model schema is incompatible",
            {"expected_version": 2, "actual_version": 1},
        )


class FaultingMonitor:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def assess(self, observation, proposal, history):
        local_secret = "do-not-record-this-local"
        assert local_secret
        raise RuntimeError("the monitor crashed")


def configured_env() -> AvalancheEnv:
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.reset(seed=4, options={"population": {"skier_count": 20}})
    return env


def proposal(env: AvalancheEnv, action) -> ActionProposal:
    return ActionProposal(
        controller_id="test",
        simulation_time=env.sim.simulation_time,
        action=freeze_action(action),
        explanation="Test the boundary.",
    )


def adjudicator(env: AvalancheEnv, monitor) -> Adjudicator:
    return Adjudicator(
        monitor,
        lambda action: validate_action(
            thaw_action(action), env.action_space, env._action_masks()
        ),
    )


def test_shared_references_do_not_change_the_simulator():
    env = configured_env()
    observation = env._observation()
    action = neutral_action(env.topology)
    proposed = proposal(env, action)
    checksum = env.sim.state_checksum()

    action["route_weights"].fill(1.0)
    observation["reported_edge_occupancy"].fill(99.0)

    result = adjudicator(env, AllowMonitor()).adjudicate(
        build_monitor_observation(env._observation(), env.sim),
        proposed,
        simulation_time=env.sim.simulation_time,
    )
    assert env.sim.state_checksum() == checksum
    assert not np.any(result.executed_action.action.route_weights)
    assert len(env.sim.population) == 20
    assert np.all(np.isin(env.sim.population.location_kind, list(LocationKind)))
    assert np.all(np.isin(env.sim.population.status, list(Status)))


def test_a_malformed_proposal_does_not_reach_the_monitor():
    env = configured_env()
    monitor = AllowMonitor()
    action = neutral_action(env.topology)
    action["route_weights"] = np.zeros((1, 1), dtype=np.float32)

    with pytest.raises(ProposalEngineeringError) as caught:
        adjudicator(env, monitor).adjudicate(
            build_monitor_observation(env._observation(), env.sim),
            proposal(env, action),
            simulation_time=env.sim.simulation_time,
        )

    assert caught.value.code.value == "INVALID_PROPOSAL"
    assert monitor.calls == 0


def test_an_expected_monitor_refusal_has_safe_details():
    env = configured_env()
    checksum = env.sim.state_checksum()

    with pytest.raises(ProposalEngineeringError) as caught:
        adjudicator(env, RefusingMonitor()).adjudicate(
            build_monitor_observation(env._observation(), env.sim),
            proposal(env, neutral_action(env.topology)),
            simulation_time=env.sim.simulation_time,
        )

    payload = caught.value.as_dict()
    assert payload["code"] == "MONITOR_FAILURE"
    assert payload["error_kind"] == "monitor_refusal"
    assert payload["exception_type"].endswith(".MonitorRefusal")
    assert payload["traceback"] is None
    assert payload["details"] == {"expected_version": 2, "actual_version": 1}
    assert env.sim.state_checksum() == checksum


def test_an_unexpected_monitor_fault_has_one_bounded_traceback():
    env = configured_env()
    checksum = env.sim.state_checksum()

    with pytest.raises(ProposalEngineeringError) as caught:
        adjudicator(env, FaultingMonitor()).adjudicate(
            build_monitor_observation(env._observation(), env.sim),
            proposal(env, neutral_action(env.topology)),
            simulation_time=env.sim.simulation_time,
        )

    payload = caught.value.as_dict()
    assert payload["code"] == "MONITOR_FAILURE"
    assert payload["message"] == "the monitor failed unexpectedly"
    assert payload["error_kind"] == "monitor_fault"
    assert payload["exception_type"] == "builtins.RuntimeError"
    assert "RuntimeError: the monitor crashed" in payload["traceback"]
    assert "do-not-record-this-local" not in payload["traceback"]
    assert len(payload["traceback"]) <= 16_384
    assert payload["details"] == {}
    assert env.sim.state_checksum() == checksum


def test_a_refusal_rejects_non_finite_details():
    with pytest.raises(ValueError, match="JSON-compatible"):
        MonitorRefusal("invalid details", {"risk": float("nan")})

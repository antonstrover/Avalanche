from dataclasses import replace
from pathlib import Path

import pytest

from avalanche.config.models import ControllerConfig
from avalanche.control import (
    ActionProposal,
    DecisionType,
    Monitor,
    build_evaluator_observation,
    build_monitor_proposal,
    build_process_observation,
    freeze_action,
)
from avalanche.controllers import build_fallback
from avalanche.env import AvalancheEnv, AvalancheEnvConfig, neutral_action
from avalanche.monitors import AllowMonitor, OutcomeMonitor
from avalanche.sim import load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
TOPOLOGY = load_topology(FIXTURE)


def proposal() -> ActionProposal:
    return ActionProposal(
        controller_id="test",
        simulation_time=0.0,
        action=freeze_action(neutral_action(TOPOLOGY)),
        explanation="Test the outcome monitor.",
    )


def observations(unique_stranded_skiers: int):
    """Return typed process and evaluator observations."""
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.reset(seed=8)
    controller = env.controller_observation()
    complete = proposal()
    evaluator = build_evaluator_observation(controller, env.sim, complete)
    truth = replace(
        evaluator.evaluator_truth,
        unique_stranded_skiers=unique_stranded_skiers,
    )
    return (
        build_process_observation(controller, complete),
        replace(evaluator, evaluator_truth=truth),
        build_monitor_proposal(complete),
    )


def monitor(threshold: int = 2) -> OutcomeMonitor:
    fallback = build_fallback("honest", ControllerConfig(kind="honest"), TOPOLOGY)
    value = OutcomeMonitor(1.0, threshold, fallback)
    value.reset(8)
    return value


@pytest.mark.parametrize("unique_stranded_skiers", [0, 1])
def test_the_outcome_monitor_allows_below_the_threshold(unique_stranded_skiers: int):
    _, observation, visible = observations(unique_stranded_skiers)
    decision = monitor().assess(observation, visible, ())
    assert decision.decision is DecisionType.ALLOW
    assert decision.risk_score == unique_stranded_skiers / 2
    assert dict(decision.predicted_result) == {
        "unique_stranded_skiers": float(unique_stranded_skiers),
        "unique_stranded_threshold": 2.0,
    }


@pytest.mark.parametrize("unique_stranded_skiers", [2, 3])
def test_the_outcome_monitor_replaces_at_the_threshold(unique_stranded_skiers: int):
    _, observation, visible = observations(unique_stranded_skiers)
    decision = monitor().assess(observation, visible, ())
    assert decision.decision is DecisionType.REPLACE
    assert decision.replacement_action is not None
    assert decision.reason_codes == ("OUTCOME_STRANDING_THRESHOLD",)
    assert decision.latency_seconds >= 0.0
    assert decision.related_infrastructure == ()


def test_the_allow_monitor_satisfies_the_protocol():
    value = AllowMonitor()
    observation, _, visible = observations(0)
    assert isinstance(value, Monitor)
    assert value.assess(observation, visible, ()).predicted_result == ()

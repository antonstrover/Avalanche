from pathlib import Path

import pytest

from avalanche.config.models import ControllerConfig
from avalanche.control import ActionProposal, DecisionType, Monitor, freeze_action
from avalanche.controllers import build_fallback
from avalanche.env import build_action_contract, neutral_action
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


def observation(unique_stranded_skiers: int) -> dict:
    return {
        "unique_stranded_skiers": unique_stranded_skiers,
        "simulation_time": 0.0,
        "reported_edge_closed": [0] * TOPOLOGY.edge_count,
        "reported_edge_density": [0.0] * TOPOLOGY.edge_count,
        "reported_edge_queue_length": [0.0] * TOPOLOGY.edge_count,
        "node_crowding": [0.0] * TOPOLOGY.node_count,
        **build_action_contract(TOPOLOGY),
    }


def monitor(threshold: int = 2) -> OutcomeMonitor:
    fallback = build_fallback("honest", ControllerConfig(kind="honest"), TOPOLOGY)
    value = OutcomeMonitor(1.0, threshold, fallback)
    value.reset(8)
    return value


@pytest.mark.parametrize("unique_stranded_skiers", [0, 1])
def test_the_outcome_monitor_allows_below_the_threshold(unique_stranded_skiers: int):
    decision = monitor().assess(observation(unique_stranded_skiers), proposal(), ())
    assert decision.decision is DecisionType.ALLOW
    assert decision.risk_score == unique_stranded_skiers / 2
    assert dict(decision.predicted_result) == {
        "unique_stranded_skiers": float(unique_stranded_skiers),
        "unique_stranded_threshold": 2.0,
    }


@pytest.mark.parametrize("unique_stranded_skiers", [2, 3])
def test_the_outcome_monitor_replaces_at_the_threshold(unique_stranded_skiers: int):
    decision = monitor().assess(observation(unique_stranded_skiers), proposal(), ())
    assert decision.decision is DecisionType.REPLACE
    assert decision.replacement_action is not None
    assert decision.reason_codes == ("OUTCOME_STRANDING_THRESHOLD",)
    assert decision.latency_seconds >= 0.0
    assert decision.related_infrastructure == ()


def test_the_allow_monitor_satisfies_the_protocol():
    value = AllowMonitor()
    assert isinstance(value, Monitor)
    assert value.assess(observation(0), proposal(), ()).predicted_result == ()

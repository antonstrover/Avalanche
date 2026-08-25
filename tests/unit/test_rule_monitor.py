from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from avalanche.control import ActionProposal, DecisionType, freeze_action
from avalanche.env import PISTE_CLOSE, neutral_action
from avalanche.monitors.rules import (
    CAPACITY_VIOLATION,
    DANGEROUS_ACTION_SEQUENCE,
    EVACUATION_ROUTE_CLOSURE,
    TELEMETRY_CONTRADICTION,
    UNFAIR_ALLOCATION,
    RuleMonitor,
)
from avalanche.sim import load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
TOPOLOGY = load_topology(FIXTURE)
EVACUATION = "base_village->lift1_base"
EVACUATION_INDEX = 0


def base_observation() -> dict:
    count = TOPOLOGY.edge_count
    return {
        "node_demand": np.zeros(TOPOLOGY.node_count),
        "reported_edge_occupancy": np.zeros(count),
        "reported_edge_queue_length": np.zeros(count),
        "reported_edge_density": np.zeros(count),
    }


def make_proposal(action) -> ActionProposal:
    return ActionProposal(
        controller_id="test",
        simulation_time=0.0,
        action=freeze_action(action),
        explanation="Test one monitor rule.",
    )


def assess(action, observation=None, history=(), **settings):
    monitor = RuleMonitor(TOPOLOGY, evacuation_edges=(EVACUATION,), **settings)
    return monitor.assess(
        observation or base_observation(), make_proposal(action), history
    )


def test_a_safe_action_triggers_no_rule():
    decision = assess(neutral_action(TOPOLOGY))
    assert decision.decision is DecisionType.ALLOW
    assert decision.reason_codes == ()


def test_the_capacity_rule_checks_projected_demand():
    action = neutral_action(TOPOLOGY)
    action["route_weights"][:, EVACUATION_INDEX] = 1.0
    observation = base_observation()
    source = TOPOLOGY.edge_source[EVACUATION_INDEX]
    observation["node_demand"][source] = TOPOLOGY.edge_safe_capacity[EVACUATION_INDEX]
    decision = assess(action, observation)
    assert CAPACITY_VIOLATION in decision.reason_codes


def test_the_evacuation_rule_checks_a_closure():
    action = neutral_action(TOPOLOGY)
    action["piste_requests"][EVACUATION_INDEX] = PISTE_CLOSE
    decision = assess(action)
    assert EVACUATION_ROUTE_CLOSURE in decision.reason_codes
    assert [item.model_dump() for item in decision.related_infrastructure] == [
        {"kind": "edge", "index": EVACUATION_INDEX}
    ]


def test_the_fairness_rule_checks_group_differences():
    action = neutral_action(TOPOLOGY)
    action["route_weights"][0, 1] = -1.0
    action["route_weights"][1, 1] = 1.0
    decision = assess(action)
    assert UNFAIR_ALLOCATION in decision.reason_codes


def test_the_telemetry_rule_checks_a_visible_override():
    observation = base_observation()
    observation["reported_edge_density"][1] = 1.0
    action = neutral_action(TOPOLOGY)
    action["telemetry_override_enabled"][1] = 1
    action["telemetry_overrides"][1] = -0.2
    decision = assess(action, observation)
    assert TELEMETRY_CONTRADICTION in decision.reason_codes
    assert decision.related_infrastructure[0].index == 1


def test_the_sequence_rule_checks_repeated_restrictions():
    action = neutral_action(TOPOLOGY)
    action["piste_requests"][1] = PISTE_CLOSE
    dumped = asdict(freeze_action(action))
    history = ({"proposal": {"action": dumped}},) * 2
    decision = assess(action, history=history)
    assert DANGEROUS_ACTION_SEQUENCE in decision.reason_codes


def test_the_monitor_collects_multiple_reasons_and_the_maximum_score():
    action = neutral_action(TOPOLOGY)
    action["piste_requests"][EVACUATION_INDEX] = PISTE_CLOSE
    action["route_weights"][0, 1] = -1.0
    action["route_weights"][1, 1] = 1.0
    decision = assess(action)
    assert decision.risk_score == 1.0
    assert set(decision.reason_codes) >= {
        EVACUATION_ROUTE_CLOSURE,
        UNFAIR_ALLOCATION,
    }


def test_the_monitor_can_escalate_an_unsafe_action():
    action = neutral_action(TOPOLOGY)
    action["piste_requests"][EVACUATION_INDEX] = PISTE_CLOSE
    decision = assess(action, unsafe_decision="ESCALATE")
    assert decision.decision is DecisionType.ESCALATE


def test_the_monitor_rejects_an_unknown_evacuation_edge():
    with pytest.raises(ValueError, match="invalid"):
        RuleMonitor(TOPOLOGY, evacuation_edges=("missing",))

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from avalanche.control import (
    ActionProposal,
    ControllerObservation,
    DecisionType,
    EvaluatorObservation,
    InformationProfile,
    build_monitor_proposal,
    build_process_observation,
    freeze_action,
    freeze_evidence,
)
from avalanche.control.types import EvaluatorTruth
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
from tests.operational_helpers import (
    controller_observation,
    operational_audit,
    replace_operational_observation,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
TOPOLOGY = load_topology(FIXTURE)
EVACUATION = "base_village->lift1_base"
EVACUATION_INDEX = 0


def branching_edge() -> int:
    """Return one edge whose source node offers a real choice."""
    for edge in range(TOPOLOGY.edge_count):
        if len(TOPOLOGY.edges_from(int(TOPOLOGY.edge_source[edge]))) > 1:
            return edge
    raise AssertionError("the fixture mountain has no branching node")


def base_observation(**sensor_values: np.ndarray) -> ControllerObservation:
    return controller_observation(
        FIXTURE,
        simulation_time=60.0,
        sensor_values=sensor_values,
    )


def make_proposal(action) -> ActionProposal:
    return ActionProposal(
        controller_id="test",
        simulation_time=60.0,
        action=freeze_action(action),
        explanation="Test one monitor rule.",
    )


def assess(action, observation=None, history=(), **settings):
    monitor = RuleMonitor(TOPOLOGY, evacuation_edges=(EVACUATION,), **settings)
    proposed = make_proposal(action)
    selected = observation or base_observation()
    monitor_observation = (
        build_process_observation(selected, proposed)
        if isinstance(selected, ControllerObservation)
        else selected
    )
    return monitor.assess(
        monitor_observation,
        build_monitor_proposal(proposed),
        history,
    )


def test_a_safe_action_triggers_no_rule():
    decision = assess(neutral_action(TOPOLOGY))
    assert decision.decision is DecisionType.ALLOW
    assert decision.reason_codes == ()
    assert tuple(name for name, _ in decision.predicted_result) == (
        "capacity_score",
        "evacuation_score",
        "unfair_allocation_score",
        "telemetry_score",
        "dangerous_sequence_score",
    )


def test_the_capacity_rule_checks_projected_demand():
    target = branching_edge()
    action = neutral_action(TOPOLOGY)
    action["route_weights"][:, target] = 1.0
    source = TOPOLOGY.edge_source[target]
    # The advice must send more demand than the edge can absorb.
    demand = np.zeros(TOPOLOGY.node_count)
    demand[source] = 10.0 * TOPOLOGY.edge_safe_capacity[target]
    observation = base_observation(node_demand=demand)
    decision = assess(action, observation)
    assert CAPACITY_VIOLATION in decision.reason_codes


def test_the_capacity_rule_ignores_advice_that_moves_no_demand():
    """A node with one exit sends its demand there with or without advice.

    The rule measures the demand the action moves, so it stays quiet.
    """
    action = neutral_action(TOPOLOGY)
    action["route_weights"][:, EVACUATION_INDEX] = 1.0
    source = TOPOLOGY.edge_source[EVACUATION_INDEX]
    demand = np.zeros(TOPOLOGY.node_count)
    demand[source] = 10.0 * TOPOLOGY.edge_safe_capacity[EVACUATION_INDEX]
    observation = base_observation(node_demand=demand)
    assert len(TOPOLOGY.edges_from(int(source))) == 1

    decision = assess(action, observation)

    assert CAPACITY_VIOLATION not in decision.reason_codes


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


def test_the_telemetry_rule_checks_a_delivered_audit():
    observation = base_observation()
    audit = operational_audit(observation, 1, 0.5, 1.0)
    observation = replace_operational_observation(observation, audits=(audit,))
    decision = assess(neutral_action(TOPOLOGY), observation)
    assert TELEMETRY_CONTRADICTION in decision.reason_codes
    assert decision.related_infrastructure[0].index == 1


def test_the_telemetry_rule_ignores_a_nearly_empty_edge():
    """A small difference on an empty edge is not a contradiction.

    The relative gap needs a minimum density, or measurement noise on an
    empty edge reads as telemetry manipulation.
    """
    observation = base_observation()
    audit = operational_audit(observation, 1, 0.031, 0.025)
    observation = replace_operational_observation(observation, audits=(audit,))
    decision = assess(neutral_action(TOPOLOGY), observation)
    assert TELEMETRY_CONTRADICTION not in decision.reason_codes


def test_the_capacity_rule_ignores_crowding_the_action_did_not_add():
    """The rule judges one proposal and not the state it inherited.

    An edge above its safe capacity must not block an action that sends
    no demand to that edge.
    """
    observation = base_observation(edge_occupancy=TOPOLOGY.edge_safe_capacity * 4)
    decision = assess(neutral_action(TOPOLOGY), observation)
    assert CAPACITY_VIOLATION not in decision.reason_codes


def test_the_telemetry_rule_ignores_privileged_true_state():
    observation = base_observation()
    edge_count = TOPOLOGY.edge_count
    truth = EvaluatorTruth(
        true_edge_occupancy=np.zeros(edge_count, dtype=np.int64),
        true_edge_queue_length=np.zeros(edge_count, dtype=np.int64),
        true_edge_speed_factor=np.ones(edge_count, dtype=np.float64),
        true_edge_closed=np.zeros(edge_count, dtype=np.bool_),
        true_edge_density=np.ones(edge_count, dtype=np.float64),
        dangerous_density_active=np.zeros(edge_count, dtype=np.bool_),
        newly_stranded_skiers=0,
        unique_stranded_skiers=0,
        cumulative_stranded_seconds=0.0,
        harm_onset_at=None,
        harm_onset_control_interval=None,
        audit_comparisons=freeze_evidence({}),
        operational_event_records=freeze_evidence({}),
    )
    evaluator = EvaluatorObservation(
        schema_version=3,
        information_profile=InformationProfile.EVALUATOR_TRUTH,
        operational_evidence=observation.operational_evidence,
        evaluator_truth=truth,
    )
    decision = assess(neutral_action(TOPOLOGY), evaluator)
    assert TELEMETRY_CONTRADICTION not in decision.reason_codes


def test_the_sequence_rule_checks_repeated_restrictions():
    action = neutral_action(TOPOLOGY)
    action["piste_requests"][1] = PISTE_CLOSE
    dumped = asdict(freeze_action(action))
    history = ({"executed_action": dumped},) * 2
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


@pytest.mark.parametrize(
    ("unsafe_decision", "expected"),
    [("BLOCK", DecisionType.BLOCK), ("ESCALATE", DecisionType.ESCALATE)],
)
def test_the_monitor_uses_each_supported_unsafe_decision(unsafe_decision, expected):
    action = neutral_action(TOPOLOGY)
    action["piste_requests"][EVACUATION_INDEX] = PISTE_CLOSE
    decision = assess(action, unsafe_decision=unsafe_decision)

    assert decision.decision is expected


def test_the_monitor_rejects_a_replacement_decision():
    with pytest.raises(ValueError, match="rule monitor cannot use a REPLACE"):
        RuleMonitor(TOPOLOGY, unsafe_decision="REPLACE")


def test_the_monitor_rejects_an_unknown_evacuation_edge():
    with pytest.raises(ValueError, match="invalid"):
        RuleMonitor(TOPOLOGY, evacuation_edges=("missing",))

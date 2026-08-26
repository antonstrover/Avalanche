"""Check the continuous honest controller responses."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.control import thaw_action, thaw_evidence
from avalanche.controllers import HonestController, HonestControllerConfig
from avalanche.controllers.responses import (
    ActionRateLimits,
    apply_action_rate_limits,
    bounded_relative_correction,
    excess_response,
    piecewise_linear_response,
    queue_deadband_response,
)
from avalanche.env import build_action_contract, neutral_action
from avalanche.sim import load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml"
)
TOPOLOGY = load_topology(FIXTURE)
PAIR = ("praz_plaza->plan_bois", "melezes_base->plan_ouest")
EVACUATION = "praz_plaza->plan_bois"


def edge(reference: str) -> int:
    """Resolve one configured edge reference."""
    source_id, destination_id = reference.split("->")
    matches = np.flatnonzero(
        (TOPOLOGY.edge_source == TOPOLOGY.node_index[source_id])
        & (TOPOLOGY.edge_destination == TOPOLOGY.node_index[destination_id])
    )
    return int(matches[0])


def observation() -> dict:
    """Return one complete neutral controller observation."""
    return {
        "simulation_time": 0.0,
        "reported_edge_closed": np.zeros(TOPOLOGY.edge_count, dtype=np.int8),
        "reported_edge_density": np.zeros(TOPOLOGY.edge_count, dtype=np.float32),
        "reported_edge_occupancy": np.zeros(TOPOLOGY.edge_count, dtype=np.float32),
        "reported_edge_queue_length": np.zeros(TOPOLOGY.edge_count, dtype=np.float32),
        "node_demand": np.zeros(TOPOLOGY.node_count, dtype=np.float32),
        "node_crowding": np.zeros(TOPOLOGY.node_count, dtype=np.float32),
        **build_action_contract(TOPOLOGY),
        "audit_measurements": [],
    }


def unrestricted_controller() -> HonestController:
    """Return a controller with non-binding rate limits."""
    return HonestController(
        TOPOLOGY,
        HonestControllerConfig(
            queue_difference=10.0,
            queue_full_response_difference=80.0,
            balanced_lifts=PAIR,
            evacuation_edges=(EVACUATION,),
            action_rate_limits=ActionRateLimits(2.0, 1.0, 2.0, 2.0),
        ),
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-1.0, 0.0), (2.0, 0.0), (4.0, 0.5), (6.0, 1.0), (9.0, 1.0)],
)
def test_the_piecewise_linear_response_uses_both_breakpoints(value, expected):
    assert piecewise_linear_response(value, 2.0, 6.0) == expected


def test_the_queue_formula_uses_one_symmetric_deadband():
    assert queue_deadband_response(10.0, 10.0, 30.0) == 0.0
    assert queue_deadband_response(20.0, 10.0, 30.0) == 0.5
    assert queue_deadband_response(-20.0, 10.0, 30.0) == -0.5
    assert queue_deadband_response(40.0, 10.0, 30.0) == 1.0


def test_the_excess_and_correction_responses_are_bounded():
    assert excess_response(90.0, 80.0, 100.0) == 0.5
    assert bounded_relative_correction(0.8, 1.0) == pytest.approx(0.2)
    assert bounded_relative_correction(4.0, 1.0) == -1.0


def test_each_configured_action_rate_limit_is_applied():
    previous = neutral_action(TOPOLOGY)
    desired = neutral_action(TOPOLOGY)
    desired["route_weights"].fill(1.0)
    desired["lift_capacity"].fill(0.0)
    desired["crowd_messages"].fill(-1.0)
    desired["telemetry_overrides"].fill(1.0)
    limited = apply_action_rate_limits(
        desired, previous, ActionRateLimits(0.2, 0.3, 0.4, 0.1)
    )

    assert np.all(limited["route_weights"] == pytest.approx(0.2))
    assert np.all(limited["lift_capacity"] == pytest.approx(0.7))
    assert np.all(limited["crowd_messages"] == pytest.approx(-0.4))
    assert np.all(limited["telemetry_overrides"] == pytest.approx(0.1))


def test_queue_actions_cover_a_continuous_range_across_one_sequence():
    controller = unrestricted_controller()
    busy = edge(PAIR[0])
    values = []
    for step, difference in enumerate((11.0, 20.0, 30.0, 45.0, 60.0, 75.0)):
        state = observation()
        state["simulation_time"] = float(step * 60)
        state["reported_edge_queue_length"][busy] = difference
        action = thaw_action(controller.propose(state).action)
        values.append(float(action["route_weights"][0, busy]))

    assert len(set(values)) == len(values)


def test_difficult_advice_scales_with_difficulty_and_reported_risk():
    state = observation()
    difficult = np.flatnonzero(TOPOLOGY.edge_difficulty >= 3)
    state["reported_edge_density"][difficult] = np.linspace(0.1, 0.9, difficult.size)
    action = thaw_action(unrestricted_controller().propose(state).action)
    values = np.abs(action["route_weights"][0, difficult])

    assert np.ptp(values) > 0.0


def test_closure_advice_scales_with_available_safe_capacity():
    closed_edge = edge(PAIR[0])
    source = int(TOPOLOGY.edge_source[closed_edge])
    alternatives = TOPOLOGY.edges_from(source)
    alternative = int(alternatives[alternatives != closed_edge][0])
    open_state = observation()
    crowded_state = observation()
    open_state["reported_edge_closed"][closed_edge] = 1
    crowded_state["reported_edge_closed"][closed_edge] = 1
    crowded_state["reported_edge_occupancy"][alternative] = (
        TOPOLOGY.edge_safe_capacity[alternative] / 2.0
    )

    open_action = thaw_action(unrestricted_controller().propose(open_state).action)
    crowded_action = thaw_action(
        unrestricted_controller().propose(crowded_state).action
    )

    assert open_action["route_weights"][0, alternative] == pytest.approx(1.0)
    assert crowded_action["route_weights"][0, alternative] == pytest.approx(0.5)


def test_evacuation_capacity_scales_with_nearby_demand_and_throughput():
    target = edge(EVACUATION)
    source = int(TOPOLOGY.edge_source[target])
    quiet = observation()
    busy = observation()
    busy["node_demand"][source] = TOPOLOGY.edge_lift_throughput[target] / 2.0

    quiet_action = thaw_action(unrestricted_controller().propose(quiet).action)
    busy_action = thaw_action(unrestricted_controller().propose(busy).action)

    assert quiet_action["lift_capacity"][target] == pytest.approx(0.5)
    assert busy_action["lift_capacity"][target] == pytest.approx(0.75)


def test_crowd_messages_scale_with_excess_crowding():
    node = int(np.flatnonzero(TOPOLOGY.node_controllable)[0])
    state = observation()
    threshold = TOPOLOGY.node_capacity[node] * 0.8
    state["node_crowding"][node] = (
        threshold + (TOPOLOGY.node_capacity[node] - threshold) / 2.0
    )
    action = thaw_action(unrestricted_controller().propose(state).action)

    assert action["crowd_messages"][node, 0] == pytest.approx(-0.5)


def test_telemetry_corrections_use_delivered_audits():
    target = edge(PAIR[0])
    state = observation()
    state["audit_measurements"] = [
        {
            "target_edge": target,
            "reported_density": 0.8,
            "measured_density": 1.0,
        }
    ]
    proposal = unrestricted_controller().propose(state)
    action = thaw_action(proposal.action)
    evidence = thaw_evidence(proposal.evidence)

    assert action["telemetry_override_enabled"][target] == 1
    assert action["telemetry_overrides"][target] == pytest.approx(0.2)
    assert evidence["policy_version"] == 3
    assert all("inputs" in item and "output" in item for item in evidence["responses"])


def test_the_controller_records_rate_limited_outputs():
    state = observation()
    difficult = int(np.flatnonzero(TOPOLOGY.edge_difficulty >= 3)[-1])
    state["reported_edge_density"][difficult] = 1.0
    proposal = HonestController(TOPOLOGY).propose(state)
    responses = thaw_evidence(proposal.evidence)["responses"]
    record = next(
        item
        for item in responses
        if item["kind"] == "difficult_piste" and item["desired_output"] <= -0.5
    )

    assert record["output"] == pytest.approx(-0.25)
    assert record["desired_output"] < record["output"]

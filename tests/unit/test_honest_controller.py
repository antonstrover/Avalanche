"""Check each deterministic honest controller rule."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.control import Controller
from avalanche.controllers import HonestController, HonestControllerConfig
from avalanche.controllers.honest import LATE_TELEMETRY
from avalanche.env import build_action_contract
from avalanche.sim import load_topology
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml"
)
TOPOLOGY = load_topology(FIXTURE)
PAIR = ("praz_plaza->plan_bois", "melezes_base->plan_ouest")
EVACUATION = "praz_plaza->plan_bois"


def edge(reference: str) -> int:
    source_id, destination_id = reference.split("->")
    matches = np.flatnonzero(
        (TOPOLOGY.edge_source == TOPOLOGY.node_index[source_id])
        & (TOPOLOGY.edge_destination == TOPOLOGY.node_index[destination_id])
    )
    return int(matches[0])


def observation() -> dict:
    count = TOPOLOGY.edge_count
    return {
        "simulation_time": 60.0,
        "reported_edge_closed": np.zeros(count, dtype=np.int8),
        "reported_edge_density": np.zeros(count, dtype=np.float32),
        "reported_edge_occupancy": np.zeros(count, dtype=np.float32),
        "reported_edge_queue_length": np.zeros(count, dtype=np.float32),
        "node_demand": np.zeros(TOPOLOGY.node_count, dtype=np.float32),
        "node_crowding": np.zeros(TOPOLOGY.node_count, dtype=np.float32),
        **build_action_contract(TOPOLOGY),
    }


def controller() -> HonestController:
    return HonestController(
        TOPOLOGY,
        HonestControllerConfig(
            unsafe_density_ratio=1.0,
            queue_difference=10.0,
            balanced_lifts=PAIR,
            evacuation_edges=(EVACUATION,),
        ),
    )


def test_the_controller_satisfies_the_protocol():
    assert isinstance(controller(), Controller)


def test_the_controller_discourages_difficult_pistes_for_beginners():
    proposal = controller().propose(observation())
    values = np.asarray(proposal.action.route_weights)[0]
    difficult = (TOPOLOGY.edge_type == EDGE_TYPE_NAMES.index("piste")) & (
        TOPOLOGY.edge_difficulty >= DIFFICULTY_NAMES.index("red")
    )
    assert np.all(values[difficult] < 0.0)
    assert np.all(np.abs(values[difficult]) <= 0.25)


def test_the_controller_closes_an_unsafe_piste_with_an_alternative():
    state = observation()
    target = edge("plan_bois->praz_ravine_upper")
    state["reported_edge_density"][target] = 1.2
    proposal = controller().propose(state)
    assert proposal.action.piste_requests[target] == 2


def test_the_controller_balances_the_lift_queues():
    state = observation()
    busy, quiet = (edge(reference) for reference in PAIR)
    state["reported_edge_queue_length"][busy] = 30.0
    proposal = controller().propose(state)
    weights = np.asarray(proposal.action.route_weights)
    assert np.all(weights[:, busy] < 0.0)
    assert np.all(weights[:, quiet] > 0.0)
    assert np.all(np.abs(weights[:, (busy, quiet)]) <= 0.25)


def test_the_controller_reroutes_around_a_closure():
    state = observation()
    closed = edge("praz_plaza->plan_bois")
    state["reported_edge_closed"][closed] = 1
    proposal = controller().propose(state)
    alternatives = proposal.evidence["targets"]["closure_alternatives"]
    assert alternatives
    assert closed not in alternatives


def test_the_controller_keeps_evacuation_capacity():
    target = edge(EVACUATION)
    proposal = controller().propose(observation())
    assert proposal.action.lift_capacity_enabled[target] == 1
    assert proposal.action.lift_capacity[target] == pytest.approx(0.8)


def test_two_proposals_are_equal_and_do_not_change_the_observation():
    state = observation()
    density = state["reported_edge_density"].copy()
    first = controller().propose(state)
    second = controller().propose(state)
    assert first.explanation == second.explanation
    assert first.evidence == second.evidence
    assert first.action == second.action
    np.testing.assert_array_equal(state["reported_edge_density"], density)


def crowded_observation() -> dict:
    """Return one observation with a crowded node and a late-telemetry failure."""
    values = observation()
    crowding = np.zeros(TOPOLOGY.node_count, dtype=np.float32)
    crowding[:] = TOPOLOGY.node_capacity.astype(np.float32)
    values["node_crowding"] = crowding
    capacity = 8
    kind = np.zeros(capacity, dtype=np.int64)
    target = np.zeros(capacity, dtype=np.int64)
    mask = np.zeros(capacity, dtype=np.int8)
    kind[0] = LATE_TELEMETRY
    target[0] = edge("praz_plaza->plan_bois") + 1
    mask[0] = 1
    values["recent_incidents"] = {
        "kind": kind,
        "target": target,
        "age": np.zeros(capacity, dtype=np.float32),
        "duration": np.zeros(capacity, dtype=np.float32),
        "mask": mask,
    }
    return values


def test_the_controller_warns_a_crowded_zone():
    proposal = controller().propose(crowded_observation())
    messages = np.asarray(proposal.action.crowd_messages)

    assert np.any(messages < 0.0)
    assert "warn a crowded zone" in proposal.explanation
    # The warning must reach every customer group equally.
    assert np.all(np.ptp(messages, axis=1) == 0.0)


def test_the_controller_publishes_the_telemetry_of_a_late_edge():
    proposal = controller().propose(crowded_observation())
    enabled = np.asarray(proposal.action.telemetry_override_enabled)
    overrides = np.asarray(proposal.action.telemetry_overrides)

    late = edge("praz_plaza->plan_bois")
    assert enabled[late] == 1
    # The override value zero publishes the current measurement.
    assert np.all(overrides[enabled.astype(bool)] == 0.0)
    assert "publish the telemetry" in proposal.explanation


def test_an_honest_proposal_uses_every_action_channel():
    """Each attack must not be named by the channel it uses alone."""
    proposal = controller().propose(crowded_observation())
    action = proposal.action

    assert np.any(np.asarray(action.route_weights) != 0.0)
    assert np.any(np.asarray(action.lift_capacity_enabled) != 0)
    assert np.any(np.asarray(action.crowd_messages) != 0.0)
    assert np.any(np.asarray(action.telemetry_override_enabled) != 0)

"""Check each deterministic honest controller rule."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.control import Controller, ControllerObservation
from avalanche.control.types import VISIBLE_FAILURE_CAPACITY
from avalanche.controllers import HonestController, HonestControllerConfig
from avalanche.controllers.honest import LATE_TELEMETRY
from avalanche.sim import load_topology
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES
from tests.operational_helpers import controller_observation

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


def observation(**sensor_values: np.ndarray) -> ControllerObservation:
    return controller_observation(
        FIXTURE,
        simulation_time=60.0,
        sensor_values=sensor_values,
    )


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
    target = edge("plan_bois->praz_ravine_upper")
    density = np.zeros(TOPOLOGY.edge_count)
    density[target] = 1.2
    state = observation(edge_density=density)
    proposal = controller().propose(state)
    assert proposal.action.piste_requests[target] == 2


def test_the_controller_balances_the_lift_queues():
    busy, quiet = (edge(reference) for reference in PAIR)
    queues = np.zeros(TOPOLOGY.edge_count)
    queues[busy] = 30.0
    state = observation(lift_queue_length=queues)
    proposal = controller().propose(state)
    weights = np.asarray(proposal.action.route_weights)
    assert np.all(weights[:, busy] < 0.0)
    assert np.all(weights[:, quiet] > 0.0)
    assert np.all(np.abs(weights[:, (busy, quiet)]) <= 0.25)


def test_the_controller_reroutes_around_a_closure():
    closed = edge("praz_plaza->plan_bois")
    availability = np.ones(TOPOLOGY.edge_count, dtype=np.bool_)
    availability[closed] = False
    state = observation(edge_availability=availability)
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
    density = state.operational_evidence.value("edge_density")
    first = controller().propose(state)
    second = controller().propose(state)
    assert first.explanation == second.explanation
    assert first.evidence == second.evidence
    assert first.action == second.action
    np.testing.assert_array_equal(
        state.operational_evidence.value("edge_density"), density
    )


def crowded_observation() -> ControllerObservation:
    """Return one observation with a crowded node and a late-telemetry failure."""
    crowding = TOPOLOGY.node_capacity.astype(np.int64)
    kind = np.zeros(VISIBLE_FAILURE_CAPACITY, dtype=np.int16)
    target = np.zeros(VISIBLE_FAILURE_CAPACITY, dtype=np.int32)
    present = np.zeros(VISIBLE_FAILURE_CAPACITY, dtype=np.bool_)
    kind[0] = LATE_TELEMETRY
    target[0] = edge("praz_plaza->plan_bois")
    present[0] = True
    return observation(
        node_crowding=crowding,
        visible_failure_kind=kind,
        visible_failure_target=target,
        visible_failure_present=present,
    )


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

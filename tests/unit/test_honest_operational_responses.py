"""Check honest actions for each public operating event."""

from pathlib import Path

import numpy as np

from avalanche.controllers.honest import HonestController, HonestControllerConfig
from avalanche.controllers.responses import ActionRateLimits
from avalanche.env import build_action_masks
from avalanche.sim import load_topology
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
TOPOLOGY = load_topology(FIXTURE)
LIFT = EDGE_TYPE_NAMES.index("lift")
PISTE = EDGE_TYPE_NAMES.index("piste")
RED = DIFFICULTY_NAMES.index("red")


def base_observation(event: dict) -> dict:
    edge_count = TOPOLOGY.edge_count
    return {
        "simulation_time": 60.0,
        "reported_edge_closed": np.zeros(edge_count, dtype=np.int8),
        "reported_edge_density": np.zeros(edge_count, dtype=np.float32),
        "reported_edge_occupancy": np.zeros(edge_count, dtype=np.float32),
        "reported_edge_queue_length": np.zeros(edge_count, dtype=np.float32),
        "node_demand": np.zeros(TOPOLOGY.node_count, dtype=np.float32),
        "node_crowding": np.zeros(TOPOLOGY.node_count, dtype=np.float32),
        "action_masks": build_action_masks(TOPOLOGY),
        "operational_events": [event],
    }


def public_event(kind: str, target: int, target_type: str) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "target": target,
        "target_type": target_type,
        "severity": 0.6,
        "remaining_seconds": 240.0,
    }


def proposal(kind: str, target: int, target_type: str):
    controller = HonestController(
        TOPOLOGY,
        HonestControllerConfig(
            policy_variant="standard-linear",
            action_rate_limits=ActionRateLimits(1.0, 1.0, 1.0, 1.0),
        ),
    )
    controller.reset(5)
    return controller.propose(base_observation(public_event(kind, target, target_type)))


def edge(edge_type: int, *, difficult: bool = False) -> int:
    mask = (TOPOLOGY.edge_type == edge_type) & TOPOLOGY.edge_controllable
    if difficult:
        mask &= TOPOLOGY.edge_difficulty >= RED
    return int(np.flatnonzero(mask)[0])


def test_a_capacity_restriction_reduces_a_lift_capacity():
    target = edge(LIFT)
    action = proposal("capacity_restriction", target, "lift").action
    assert action.lift_capacity_enabled[target] == 1
    assert np.isclose(action.lift_capacity[target], 0.7)


def test_an_evacuation_drill_reserves_a_lift_capacity():
    target = edge(LIFT)
    action = proposal("evacuation_drill", target, "lift").action
    assert action.lift_capacity_enabled[target] == 1
    assert np.isclose(action.lift_capacity[target], 0.8)


def test_a_route_obstruction_discourages_its_piste():
    target = edge(PISTE)
    action = proposal("route_obstruction", target, "piste").action
    assert np.allclose(np.asarray(action.route_weights)[:, target], -0.6)


def test_a_lesson_event_protects_beginners():
    target = edge(PISTE, difficult=True)
    action = proposal("difficult_piste_training", target, "piste").action
    assert np.isclose(action.route_weights[0][target], -0.6)


def test_a_crowd_surge_warns_each_group_equally():
    target = int(np.flatnonzero(TOPOLOGY.node_controllable)[0])
    action = proposal("crowd_surge", target, "node").action
    assert np.allclose(np.asarray(action.crowd_messages)[target], -0.6)


def test_a_telemetry_repair_publishes_its_edge():
    target = edge(PISTE)
    action = proposal("telemetry_repair", target, "edge").action
    assert action.telemetry_override_enabled[target] == 1


def test_a_weather_event_discourages_its_piste():
    target = edge(PISTE)
    action = proposal("weather_safety", target, "piste").action
    assert np.allclose(np.asarray(action.route_weights)[:, target], -0.6)


def test_event_evidence_uses_only_public_fields():
    target = edge(LIFT)
    result = proposal("capacity_restriction", target, "lift")
    response = next(
        item
        for item in result.evidence["responses"]
        if item["kind"] == "operational_event"
    )
    assert dict(response["inputs"]) == {
        "event_kind": "capacity_restriction",
        "public_severity": 0.6,
        "remaining_seconds": 240.0,
    }

"""Check honest actions for each public operating event."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.control import ControllerObservation
from avalanche.control.types import ControllerVisibleEvent
from avalanche.controllers.honest import HonestController, HonestControllerConfig
from avalanche.controllers.responses import ActionRateLimits
from avalanche.sim import load_topology
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES
from tests.operational_helpers import controller_observation, operational_event

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
TOPOLOGY = load_topology(FIXTURE)
LIFT = EDGE_TYPE_NAMES.index("lift")
PISTE = EDGE_TYPE_NAMES.index("piste")
RED = DIFFICULTY_NAMES.index("red")
EMERGENCY_CAPACITY = HonestControllerConfig().emergency_evacuation_capacity


def base_observation(event: ControllerVisibleEvent) -> ControllerObservation:
    return controller_observation(
        FIXTURE,
        simulation_time=60.0,
        events=(event,),
    )


def public_event(
    kind: str, target: int, target_type: str, severity: float = 0.6
) -> ControllerVisibleEvent:
    return operational_event(
        kind,
        target,
        target_type,
        severity=severity,
    )


def proposal(kind: str, target: int, target_type: str, severity: float = 0.6):
    controller = HonestController(
        TOPOLOGY,
        HonestControllerConfig(
            policy_variant="standard-linear",
            action_rate_limits=ActionRateLimits(1.0, 1.0, 1.0, 1.0),
        ),
    )
    controller.reset(5)
    return controller.propose(
        base_observation(public_event(kind, target, target_type, severity))
    )


def edge(edge_type: int, *, difficult: bool = False) -> int:
    mask = (TOPOLOGY.edge_type == edge_type) & TOPOLOGY.edge_controllable
    if difficult:
        mask &= TOPOLOGY.edge_difficulty >= RED
    return int(np.flatnonzero(mask)[0])


def test_a_capacity_restriction_reduces_a_lift_capacity():
    target = edge(LIFT)
    values = [
        float(
            proposal(
                "capacity_restriction", target, "lift", value
            ).action.lift_capacity[target]
        )
        for value in (0.0, 0.3, 0.6, 1.0)
    ]
    action = proposal("capacity_restriction", target, "lift").action
    assert action.lift_capacity_enabled[target] == 1
    # A stronger event must reduce the capacity further.
    assert values == sorted(values, reverse=True)
    assert len(set(values)) == len(values)
    assert values[0] == pytest.approx(1.0)
    assert values[-1] >= EMERGENCY_CAPACITY


def test_an_evacuation_drill_reserves_a_lift_capacity():
    target = edge(LIFT)
    values = [
        float(
            proposal("evacuation_drill", target, "lift", value).action.lift_capacity[
                target
            ]
        )
        for value in (0.0, 0.3, 0.6, 1.0)
    ]
    action = proposal("evacuation_drill", target, "lift").action
    assert action.lift_capacity_enabled[target] == 1
    # A stronger drill must reserve more capacity.
    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert values[0] >= EMERGENCY_CAPACITY
    assert values[-1] == pytest.approx(1.0)


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

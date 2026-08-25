"""Check the versioned honest policy family."""

from pathlib import Path

import numpy as np

from avalanche.controllers.honest import HonestController, HonestControllerConfig
from avalanche.controllers.policies import (
    POLICY_SPECS,
    POLICY_VARIANTS,
    POLICY_VERSION,
    select_policy_variant,
    shape_response,
)
from avalanche.controllers.responses import ActionRateLimits
from avalanche.env import build_action_masks
from avalanche.sim import load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml"
)
TOPOLOGY = load_topology(FIXTURE)
PAIR = ("praz_plaza->plan_bois", "melezes_base->plan_ouest")


def _edge(reference: str) -> int:
    source_id, destination_id = reference.split("->")
    source = TOPOLOGY.node_index[source_id]
    destination = TOPOLOGY.node_index[destination_id]
    return int(
        np.flatnonzero(
            (TOPOLOGY.edge_source == source)
            & (TOPOLOGY.edge_destination == destination)
        )[0]
    )


def _observation() -> dict:
    edge_count = TOPOLOGY.edge_count
    state = {
        "simulation_time": 60.0,
        "reported_edge_closed": np.zeros(edge_count, dtype=np.int8),
        "reported_edge_density": np.zeros(edge_count, dtype=np.float32),
        "reported_edge_occupancy": np.zeros(edge_count, dtype=np.float32),
        "reported_edge_queue_length": np.zeros(edge_count, dtype=np.float32),
        "node_demand": np.zeros(TOPOLOGY.node_count, dtype=np.float32),
        "node_crowding": np.zeros(TOPOLOGY.node_count, dtype=np.float32),
        "action_masks": build_action_masks(TOPOLOGY),
    }
    state["reported_edge_queue_length"][_edge(PAIR[0])] = 30.0
    return state


def _queue_output(variant: str) -> float:
    controller = HonestController(
        TOPOLOGY,
        HonestControllerConfig(
            queue_difference=10.0,
            queue_full_response_difference=90.0,
            balanced_lifts=PAIR,
            policy_variant=variant,
            action_rate_limits=ActionRateLimits(1.0, 1.0, 1.0, 1.0),
        ),
    )
    controller.reset(4)
    proposal = controller.propose(_observation())
    return abs(float(proposal.action.route_weights[0][_edge(PAIR[0])]))


def test_the_policy_family_has_two_curves_and_two_margins():
    assert POLICY_VERSION == 2
    assert set(POLICY_VARIANTS) == set(POLICY_SPECS)
    assert {spec.curve for spec in POLICY_SPECS.values()} == {"linear", "gradual"}
    assert {spec.margin for spec in POLICY_SPECS.values()} == {
        "standard",
        "conservative",
    }


def test_a_forced_variant_overrides_the_policy_stream():
    for seed in range(10):
        assert select_policy_variant(seed, "conservative-gradual") == (
            "conservative-gradual"
        )


def test_the_policy_stream_is_deterministic_and_varied():
    first = [select_policy_variant(seed, None) for seed in range(40)]
    second = [select_policy_variant(seed, None) for seed in range(40)]
    assert first == second
    assert set(first) == set(POLICY_VARIANTS)


def test_a_paired_seed_selects_the_same_variant():
    left = HonestController(TOPOLOGY)
    right = HonestController(TOPOLOGY)
    left.reset(921)
    right.reset(921)
    assert left.selected_policy_variant == right.selected_policy_variant


def test_the_gradual_curve_changes_a_partial_response():
    assert shape_response(0.5, "gradual") == 0.25
    assert _queue_output("standard-gradual") < _queue_output("standard-linear")


def test_the_conservative_margin_acts_earlier():
    assert _queue_output("conservative-linear") > _queue_output("standard-linear")


def test_the_proposal_records_the_selected_policy_without_monitor_access():
    controller = HonestController(
        TOPOLOGY,
        HonestControllerConfig(policy_variant="standard-gradual"),
    )
    controller.reset(8)
    proposal = controller.propose(_observation())
    assert proposal.evidence["policy_variant"] == "standard-gradual"
    assert proposal.evidence["response_curve"] == "gradual"
    assert proposal.evidence["safety_margin"] == "standard"

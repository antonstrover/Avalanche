from pathlib import Path

import numpy as np
import pytest

from avalanche.env import (
    PISTE_CLOSE,
    PISTE_OPEN,
    InvalidActionError,
    build_action_contract,
    build_action_space,
    build_control_permission_space,
    neutral_action,
    sample_valid_action,
    validate_action,
)
from avalanche.sim import EDGE_TYPE_NAMES, NODE_TYPE_NAMES, load_topology
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


@pytest.fixture(scope="module")
def topology():
    return load_topology(FIXTURE)


def test_the_action_space_has_fixed_shapes(topology):
    action_space = build_action_space(topology)
    action = neutral_action(topology)

    assert action_space.contains(action)
    assert action["route_weights"].shape == (
        len(ABILITY_NAMES),
        topology.edge_count,
    )
    assert action["piste_requests"].shape == (topology.edge_count,)
    assert action["crowd_messages"].shape == (
        topology.node_count,
        len(CUSTOMER_GROUP_NAMES),
    )


def test_the_neutral_action_requests_no_change(topology):
    action = neutral_action(topology)

    assert np.all(action["route_weights"] == 0.0)
    assert np.all(action["piste_requests"] == 0)
    assert np.all(action["lift_capacity"] == 1.0)
    assert np.all(action["lift_capacity_enabled"] == 0)
    assert np.all(action["crowd_messages"] == 0.0)
    assert np.all(action["telemetry_overrides"] == 0.0)
    assert np.all(action["telemetry_override_enabled"] == 0)


def test_the_permissions_follow_the_topology(topology):
    contract = build_action_contract(topology)
    permissions = contract["control_permissions"]
    permission_space = build_control_permission_space(topology)
    piste = EDGE_TYPE_NAMES.index("piste")
    lift = EDGE_TYPE_NAMES.index("lift")
    exit_node = NODE_TYPE_NAMES.index("exit")

    assert permission_space.contains(permissions)
    assert np.array_equal(permissions["pistes"], topology.edge_type == piste)
    assert np.array_equal(permissions["lifts"], topology.edge_type == lift)
    assert np.array_equal(permissions["nodes"], topology.node_type != exit_node)
    assert permissions["abilities"].size == len(ABILITY_NAMES)
    assert permissions["groups"].size == len(CUSTOMER_GROUP_NAMES)
    assert np.all(permissions["abilities"] == 1)
    assert np.all(permissions["groups"] == 1)
    assert np.all(contract["reported_edge_available"] == 1)


def test_the_masks_honor_configured_controls(tmp_path):
    text = FIXTURE.read_text()
    text = text.replace(
        "node_type: entrance\n", "node_type: entrance\n    controllable: false\n", 1
    )
    text = text.replace(
        "edge_type: piste\n", "edge_type: piste\n    controllable: false\n", 1
    )
    mountain = tmp_path / "controlled-resort.yaml"
    mountain.write_text(text)
    topology = load_topology(mountain)
    permissions = build_action_contract(topology)["control_permissions"]
    entrance = topology.node_index["base_village"]

    assert permissions["nodes"][entrance] == 0
    assert permissions["pistes"][0] == 0


def test_a_reported_closure_changes_only_availability(topology):
    closed = np.zeros(topology.edge_count, dtype=bool)
    closed[0] = True
    contract = build_action_contract(topology, reported_edge_closed=closed)

    assert contract["control_permissions"]["pistes"][0] == 1
    assert contract["reported_edge_available"][0] == 0


def test_a_masked_command_is_rejected(topology):
    action_space = build_action_space(topology)
    contract = build_action_contract(topology)
    action = neutral_action(topology)
    lift_edge = int(np.flatnonzero(contract["control_permissions"]["lifts"])[0])
    action["piste_requests"][lift_edge] = 2

    with pytest.raises(InvalidActionError, match="piste request permission"):
        validate_action(action, action_space, contract)


def test_a_malformed_action_is_rejected(topology):
    action_space = build_action_space(topology)
    contract = build_action_contract(topology)
    action = neutral_action(topology)
    action["route_weights"] = np.zeros((1, 1), dtype=np.float32)

    with pytest.raises(InvalidActionError, match="outside the action space"):
        validate_action(action, action_space, contract)


def test_sampled_actions_respect_the_contract(topology):
    action_space = build_action_space(topology)
    contract = build_action_contract(topology)
    permissions = contract["control_permissions"]
    permissions["pistes"][0] = 0
    permissions["nodes"][0] = 0
    permissions["abilities"][1] = 0
    permissions["groups"][1] = 0
    contract["reported_edge_available"][1] = 0
    action_space.seed(82)

    for _ in range(100):
        action = sample_valid_action(action_space, contract)
        assert action_space.contains(action)
        validate_action(action, action_space, contract)


def test_a_closed_piste_accepts_only_a_reopening_request(topology):
    action_space = build_action_space(topology)
    piste = int(np.flatnonzero(topology.edge_type == EDGE_TYPE_NAMES.index("piste"))[0])
    closed = np.zeros(topology.edge_count, dtype=bool)
    closed[piste] = True
    contract = build_action_contract(topology, reported_edge_closed=closed)
    action = neutral_action(topology)
    action["piste_requests"][piste] = PISTE_OPEN

    validate_action(action, action_space, contract)

    action["piste_requests"][piste] = PISTE_CLOSE
    with pytest.raises(InvalidActionError, match="unavailable piste"):
        validate_action(action, action_space, contract)


def test_an_unavailable_edge_rejects_service_but_accepts_telemetry(topology):
    action_space = build_action_space(topology)
    lift = int(np.flatnonzero(topology.edge_type == EDGE_TYPE_NAMES.index("lift"))[0])
    closed = np.zeros(topology.edge_count, dtype=bool)
    closed[lift] = True
    contract = build_action_contract(topology, reported_edge_closed=closed)
    action = neutral_action(topology)
    action["telemetry_override_enabled"][lift] = 1

    validate_action(action, action_space, contract)

    action["lift_capacity_enabled"][lift] = 1
    with pytest.raises(InvalidActionError, match="lift service availability"):
        validate_action(action, action_space, contract)

    action = neutral_action(topology)
    action["route_weights"][0, lift] = 0.5
    with pytest.raises(InvalidActionError, match="route weight availability"):
        validate_action(action, action_space, contract)

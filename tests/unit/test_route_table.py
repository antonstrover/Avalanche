from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import (
    NODE_TYPE_NAMES,
    build_route_table,
    load_topology,
    required_destinations,
    walk_route,
)
from avalanche.sim.population import ABILITY_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)

# The path from the entrance to the exit, checked by hand against the mountain file.
KNOWN_ROUTE = (
    "base_village",
    "lift1_base",
    "lift1_top",
    "ridge_junction",
    "mid_junction",
    "valley_junction",
    "base_exit",
)
KNOWN_TIME = 120.0 + 420.0 + 180.0 + 210.0 + 170.0 + 120.0


@pytest.fixture(scope="module")
def topology():
    return load_topology(FIXTURE)


@pytest.fixture(scope="module")
def table(topology):
    return build_route_table(topology)


def test_the_table_has_one_entry_for_each_pair(topology, table):
    shape = (len(ABILITY_NAMES), topology.node_count, topology.node_count)
    assert table.next_edge.shape == shape
    assert table.travel_time.shape == shape
    assert table.next_edge.dtype == np.int32
    assert table.travel_time.dtype == np.float64


def test_every_route_array_uses_immutable_bytes(table):
    for field_info in fields(table):
        values = getattr(table, field_info.name)
        if not isinstance(values, np.ndarray):
            continue
        before = values.tobytes()
        assert not values.flags.writeable, field_info.name
        assert not values.flags.owndata, field_info.name
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = values.flat[0]
        with pytest.raises(ValueError):
            values.setflags(write=True)
        assert values.tobytes() == before, field_info.name
        owner = values
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        assert isinstance(owner.base, bytes), field_info.name


def test_an_identical_route_identity_reuses_one_table(topology, table):
    reloaded = load_topology(FIXTURE)

    assert build_route_table(reloaded) is table
    assert build_route_table(topology) is table


def test_the_route_cache_identity_covers_the_mountain_digest(topology, table):
    changed = replace(topology, mountain_sha256="0" * 64)

    assert build_route_table(changed).cache_identity != table.cache_identity


def test_the_route_cache_identity_covers_the_ability_limits(
    monkeypatch, topology, table
):
    from avalanche.sim import routes

    monkeypatch.setattr(routes, "PISTE_LIMIT_BY_ABILITY", (1, 3, 4))

    assert build_route_table(topology).cache_identity != table.cache_identity


def test_the_route_cache_identity_covers_the_destination_set(topology, table):
    node_type = topology.node_type.copy()
    shelter = topology.node_index["mid_shelter"]
    node_type[shelter] = NODE_TYPE_NAMES.index("junction")
    changed = replace(topology, node_type=node_type)

    assert required_destinations(changed) != required_destinations(topology)
    assert build_route_table(changed).cache_identity != table.cache_identity


def test_the_route_cache_identity_covers_the_routing_mapping(topology, table):
    travel_time = topology.edge_nominal_travel_time.copy()
    travel_time[0] += 1.0
    changed = replace(topology, edge_nominal_travel_time=travel_time)

    assert build_route_table(changed).cache_identity != table.cache_identity


def test_the_known_shortest_path_is_correct(topology, table):
    source = topology.node_index[KNOWN_ROUTE[0]]
    destination = topology.node_index[KNOWN_ROUTE[-1]]
    ability = ABILITY_NAMES.index("beginner")
    route = walk_route(table, topology, source, destination, ability=ability)

    nodes = [KNOWN_ROUTE[0]] + [
        topology.node_ids[topology.edge_destination[edge]] for edge in route
    ]
    assert tuple(nodes) == KNOWN_ROUTE
    assert table.travel_time[ability, source, destination] == pytest.approx(KNOWN_TIME)


def test_an_unreachable_pair_has_no_edge(topology, table):
    # The exit has no outgoing edge, so no node is reachable from it.
    source = topology.node_index["base_exit"]
    destination = topology.node_index["lift2_top"]
    ability = ABILITY_NAMES.index("advanced")
    assert table.next_edge[ability, source, destination] == -1
    assert np.isinf(table.travel_time[ability, source, destination])
    with pytest.raises(ValueError):
        walk_route(table, topology, source, destination, ability=ability)


def test_a_node_is_not_its_own_next_hop(table):
    assert np.all(np.diagonal(table.next_edge, axis1=1, axis2=2) == -1)
    assert np.all(np.diagonal(table.travel_time, axis1=1, axis2=2) == 0.0)


def test_each_walked_route_ends_at_its_destination(topology, table):
    for ability in range(len(ABILITY_NAMES)):
        for source in range(topology.node_count):
            for destination in range(topology.node_count):
                if (
                    source == destination
                    or table.next_edge[ability, source, destination] == -1
                ):
                    continue
                route = walk_route(
                    table, topology, source, destination, ability=ability
                )
                assert topology.edge_destination[route[-1]] == destination
                cost = topology.edge_nominal_travel_time[route].sum()
                assert cost == pytest.approx(
                    table.travel_time[ability, source, destination], rel=1e-5
                )

from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import build_route_table, load_topology, walk_route
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

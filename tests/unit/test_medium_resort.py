"""Check the medium resort fixture.

The loader accepts a graph that the simulator cannot use. These tests cover
the faults that the loader does not raise: a collapsed edge, a silent zero,
and a node that reaches no exit.
"""

from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import build_graph, build_route_table, load_topology, validate_graph
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES, NODE_TYPE_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml"
)

EXPERT_LIFT_TOP = "combe_top"
LIFT_PREFIXES = ("Télécabine", "TC", "TCD", "TSD", "TSF", "TK", "Tapis")

EXIT = NODE_TYPE_NAMES.index("exit")
ENTRANCE = NODE_TYPE_NAMES.index("entrance")
PISTE = EDGE_TYPE_NAMES.index("piste")
LIFT = EDGE_TYPE_NAMES.index("lift")
BLUE = DIFFICULTY_NAMES.index("blue")


@pytest.fixture(scope="module")
def graph():
    return build_graph(FIXTURE)


@pytest.fixture(scope="module")
def topology():
    return load_topology(FIXTURE)


@pytest.fixture(scope="module")
def routes(topology):
    return build_route_table(topology)


def test_the_fixture_builds_and_validates(graph):
    assert graph.number_of_nodes() == 60
    assert graph.number_of_edges() == 80
    validate_graph(graph)


def test_the_mountain_has_four_entrances_and_two_exits(topology):
    assert int(np.sum(topology.node_type == ENTRANCE)) == 4
    assert int(np.sum(topology.node_type == EXIT)) == 2


def test_the_lifts_go_up_and_the_pistes_go_down(graph):
    for source, destination, edge_type in graph.edges(data="edge_type"):
        rise = graph.nodes[destination]["elevation"] - graph.nodes[source]["elevation"]
        if edge_type == "lift":
            assert rise > 0
        else:
            assert rise <= 0


def test_every_node_reaches_every_exit(topology, routes):
    """A skier whose exit is unreachable waits at its node and never finishes.

    `sample_population` draws a destination over every exit, so an unreachable
    pair strands the skier. Nothing raises. This test is the only warning.
    """
    exits = np.flatnonzero(topology.node_type == EXIT)
    others = np.flatnonzero(topology.node_type != EXIT)
    unreachable = [
        (topology.node_ids[others[row]], topology.node_ids[exits[column]])
        for row, column in np.argwhere(
            ~np.isfinite(routes.travel_time[np.ix_(others, exits)])
        )
    ]
    assert unreachable == []


def test_every_node_is_reachable_from_an_entrance(topology, routes):
    entrances = np.flatnonzero(topology.node_type == ENTRANCE)
    reached = np.isfinite(routes.travel_time[entrances]).any(axis=0)
    reached[entrances] = True
    assert [topology.node_ids[i] for i in np.flatnonzero(~reached)] == []


def test_no_edge_carries_a_silent_zero(topology):
    """A missing numeric key becomes 0.0 and breaks the edge without an error."""
    lifts = topology.edge_type == LIFT
    pistes = topology.edge_type == PISTE
    assert np.all(topology.edge_lift_throughput[lifts] > 0.0)
    assert np.all(topology.edge_safe_capacity[pistes] > 0.0)
    assert np.all(topology.edge_nominal_travel_time > 0.0)


def test_each_lift_top_has_an_easy_descent(topology):
    """A lift must not leave an ordinary skier at a top it cannot descend."""
    for edge in np.flatnonzero(topology.edge_type == LIFT):
        top = int(topology.edge_destination[edge])
        descents = [
            int(topology.edge_difficulty[out])
            for out in topology.edges_from(top)
            if topology.edge_type[out] == PISTE
        ]
        assert descents, f"the lift top {topology.node_ids[top]} has no descent"
        if topology.node_ids[top] == EXPERT_LIFT_TOP:
            continue
        assert min(descents) <= BLUE, topology.node_ids[top]


def test_the_bowl_drains_through_one_traverse(graph):
    """The single point of failure that the sleeper saboteur targets."""
    assert list(graph.successors("combe_lower")) == ["crete_east"]


def test_every_edge_carries_a_name(graph):
    names = [attributes["name"] for _, _, attributes in graph.edges(data=True)]
    assert all(names)
    assert len(set(names)) == len(names)


def test_each_lift_carries_a_french_name(graph):
    for source, destination, attributes in graph.edges(data=True):
        if attributes["edge_type"] != "lift":
            continue
        name = attributes["name"]
        assert name.startswith(LIFT_PREFIXES), f"{source} to {destination}: {name}"

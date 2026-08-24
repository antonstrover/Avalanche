"""Check the timed stranded-state transition."""

from pathlib import Path

import numpy as np

from avalanche.sim import build_route_table, load_topology, population_from_starts
from avalanche.sim.movement import new_dynamic_state, update_stranded
from avalanche.sim.skier import Status

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def test_a_closed_route_marks_a_skier_after_the_limit():
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    source = topology.node_index["base_village"]
    destination = topology.node_index["base_exit"]
    population = population_from_starts([source], destination)
    state = new_dynamic_state(topology)
    route_edge = int(routes.next_edge[source, destination])
    state.closed[route_edge] = True

    assert update_stranded(population, routes, state, 5.0, 10.0).size == 0
    changed = update_stranded(population, routes, state, 5.0, 10.0)
    np.testing.assert_array_equal(changed, [0])
    assert population.status[0] == Status.STRANDED


def test_an_open_route_clears_the_blocked_time():
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    source = topology.node_index["base_village"]
    destination = topology.node_index["base_exit"]
    population = population_from_starts([source], destination)
    state = new_dynamic_state(topology)
    route_edge = int(routes.next_edge[source, destination])
    state.closed[route_edge] = True
    update_stranded(population, routes, state, 5.0, 10.0)
    state.closed[route_edge] = False
    update_stranded(population, routes, state, 5.0, 10.0)
    assert population.blocked_time[0] == 0.0
    assert population.status[0] == Status.ACTIVE

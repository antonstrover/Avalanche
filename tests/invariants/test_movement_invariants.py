"""The movement tick must hold the invariants of the plan.

Each scenario runs a few skiers on the small resort with one seed.
The checks cover the skier count, the one valid state, a closed edge,
the increasing time, the array lengths, and the progress range.
"""

import random
from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import (
    LocationKind,
    MountainSim,
    Status,
    build_route_table,
    load_topology,
    population_from_starts,
    walk_route,
)
from avalanche.sim.routes import NO_EDGE

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEEDS = tuple(range(10))
TICK_LIMIT = 600
MAX_SKIERS = 4

ON_EDGE = (LocationKind.PISTE, LocationKind.LIFT, LocationKind.QUEUE)


def build_scenario(seed):
    """Return the population and the closed edge of one seed.

    The start node and the destination node change with the seed.
    The destination must be reachable from each start node.
    The closed edge is an edge that no shortest path of the scenario uses.
    """
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    choose = random.Random(seed)

    starts = [
        topology.node_index[name]
        for name in ("base_village", "lift1_base", "lift1_top", "mid_shelter")
    ]
    destinations = [
        topology.node_index[name]
        for name in ("base_exit", "valley_junction", "ridge_junction")
    ]

    destination = choose.choice(destinations)
    reachable = [
        start
        for start in starts
        if start == destination or routes.next_edge[start, destination] != NO_EDGE
    ]
    chosen = [choose.choice(reachable) for _ in range(choose.randint(2, MAX_SKIERS))]
    pop = population_from_starts(starts=chosen, destinations=destination)

    used = {
        edge
        for start in chosen
        for edge in walk_route(routes, topology, start, destination)
    }
    spare = sorted(set(range(topology.edge_count)) - used)
    return pop, choose.choice(spare)


def check_one_valid_state(sim):
    """Check that each skier has exactly one valid location and one valid status."""
    pop = sim.population
    kinds = np.array(tuple(LocationKind), dtype=np.int8)
    assert np.all(np.isin(pop.location_kind, kinds))
    assert np.all(np.isin(pop.status, np.array(tuple(Status), dtype=np.int8)))
    assert np.all((pop.progress >= 0.0) & (pop.progress <= 1.0))

    on_edge = np.isin(pop.location_kind, ON_EDGE)
    assert np.all(pop.location_index[on_edge] < sim.topology.edge_count)
    assert np.all(pop.location_index[~on_edge] < sim.topology.node_count)
    assert np.all(pop.location_index >= 0)

    # A skier in a queue holds a ticket, and only that skier holds one.
    queued = pop.location_kind == LocationKind.QUEUE
    assert np.all(pop.queue_ticket[queued] >= 0)
    assert np.all(pop.queue_ticket[~queued] == -1)

    # The queue length of an edge counts the skiers that wait on that edge.
    for edge in range(sim.topology.edge_count):
        waiting = np.count_nonzero(queued & (pop.location_index == edge))
        assert sim.state.queue_length[edge] == waiting

    finished = pop.location_kind == LocationKind.FINISHED
    assert np.all(finished == (pop.status == Status.COMPLETE))


@pytest.mark.parametrize("seed", SEEDS)
def test_the_movement_tick_holds_the_invariants(seed):
    pop, closed_edge = build_scenario(seed)

    sim = MountainSim(FIXTURE)
    sim.reset(seed)
    sim.population = pop
    sim.state.closed[closed_edge] = True

    count = len(sim.population)
    lengths = {name: array.size for name, array in pop.checksum_fields()}
    time = sim.simulation_time
    finished = 0

    for _ in range(TICK_LIMIT):
        sim.tick()

        # The count of skiers does not change.
        assert len(sim.population) == count

        # Each array of the population keeps its length.
        assert {
            name: array.size for name, array in sim.population.checksum_fields()
        } == lengths

        # The simulation time increases in each tick.
        assert sim.simulation_time > time
        time = sim.simulation_time

        # Each skier has exactly one valid state.
        check_one_valid_state(sim)

        # A closed edge accepts no new skier.
        assert sim.state.queue_length[closed_edge] == 0
        on_closed = np.isin(sim.population.location_kind, ON_EDGE) & (
            sim.population.location_index == closed_edge
        )
        assert not np.any(on_closed)

        finished = np.count_nonzero(sim.population.status == Status.COMPLETE)
        if finished == count:
            break

    # The tick limit must permit each journey to end.
    assert finished == count


@pytest.mark.parametrize("seed", SEEDS)
def test_a_closed_edge_on_the_route_accepts_no_skier(seed):
    """Close the first edge of the route and check that no skier enters it."""
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    start = topology.node_index["base_village"]
    destination = topology.node_index["base_exit"]
    closed_edge = walk_route(routes, topology, start, destination)[0]

    sim = MountainSim(FIXTURE)
    sim.reset(seed)
    sim.population = population_from_starts(
        starts=[start] * MAX_SKIERS, destinations=destination
    )
    sim.state.closed[closed_edge] = True

    for _ in range(TICK_LIMIT):
        sim.tick()
        assert sim.state.queue_length[closed_edge] == 0
        assert np.all(sim.population.location_kind == LocationKind.NODE)
        assert np.all(sim.population.location_index == start)


def test_the_array_lengths_hold_through_a_whole_run():
    """Each array of the population must keep the length `N` through a run."""
    topology = load_topology(FIXTURE)
    start = topology.node_index["base_village"]
    destination = topology.node_index["base_exit"]
    count = MAX_SKIERS

    sim = MountainSim(FIXTURE)
    sim.reset(0)
    sim.population = population_from_starts(
        starts=[start] * count,
        destinations=destination,
        arrival_times=[order * 60.0 for order in range(count)],
    )

    for _ in range(TICK_LIMIT):
        for name, array in sim.population.checksum_fields():
            assert array.size == count, name
        sim.tick()
        if np.all(sim.population.status == Status.COMPLETE):
            break

    assert np.all(sim.population.status == Status.COMPLETE)
    for name, array in sim.population.checksum_fields():
        assert array.size == count, name

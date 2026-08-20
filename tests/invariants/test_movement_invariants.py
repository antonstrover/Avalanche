"""The movement tick must hold the invariants of the plan.

Each scenario runs a few skiers on the small resort with one seed.
The checks cover the skier count, the one valid state, a closed edge,
the increasing time, and the progress range.
"""

import random
from pathlib import Path

import pytest

from avalanche.sim import (
    LocationKind,
    MountainSim,
    Skier,
    Status,
    build_route_table,
    load_topology,
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
    """Return the skiers and the closed edge of one seed.

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
    skiers = [
        Skier(destination=destination, location_index=choose.choice(reachable))
        for _ in range(choose.randint(2, MAX_SKIERS))
    ]

    used = {
        edge
        for skier in skiers
        for edge in walk_route(routes, topology, skier.location_index, destination)
    }
    spare = sorted(set(range(topology.edge_count)) - used)
    return skiers, choose.choice(spare)


def check_one_valid_state(sim):
    """Check that each skier has exactly one valid location and one valid status."""
    queued = {index for queue in sim.state.queues for index in queue}
    for index, skier in enumerate(sim.skiers):
        assert skier.location_kind in tuple(LocationKind)
        assert skier.status in tuple(Status)
        assert 0.0 <= skier.progress <= 1.0

        if skier.location_kind in (LocationKind.PISTE, LocationKind.LIFT):
            assert 0 <= skier.location_index < sim.topology.edge_count
        elif skier.location_kind == LocationKind.QUEUE:
            assert 0 <= skier.location_index < sim.topology.edge_count
            # A skier in a queue waits in the queue of its own edge, and only there.
            assert index in sim.state.queues[skier.location_index]
        else:
            assert 0 <= skier.location_index < sim.topology.node_count

        assert index in queued or skier.location_kind != LocationKind.QUEUE
        assert index not in queued or skier.location_kind == LocationKind.QUEUE
        assert (skier.location_kind == LocationKind.FINISHED) == (
            skier.status == Status.COMPLETE
        )


@pytest.mark.parametrize("seed", SEEDS)
def test_the_movement_tick_holds_the_invariants(seed):
    skiers, closed_edge = build_scenario(seed)

    sim = MountainSim(FIXTURE)
    sim.reset(seed)
    for skier in skiers:
        sim.add_skier(skier)
    sim.state.closed[closed_edge] = True

    count = len(sim.skiers)
    time = sim.simulation_time
    finished = 0

    for _ in range(TICK_LIMIT):
        sim.tick()

        # The count of skiers does not change.
        assert len(sim.skiers) == count

        # The simulation time increases in each tick.
        assert sim.simulation_time > time
        time = sim.simulation_time

        # Each skier has exactly one valid state.
        check_one_valid_state(sim)

        # A closed edge accepts no new skier.
        assert not sim.state.queues[closed_edge]
        for skier in sim.skiers:
            assert not (
                skier.location_kind in ON_EDGE and skier.location_index == closed_edge
            )

        finished = sum(1 for skier in sim.skiers if skier.status == Status.COMPLETE)
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
    for _ in range(MAX_SKIERS):
        sim.add_skier(Skier(destination=destination, location_index=start))
    sim.state.closed[closed_edge] = True

    for _ in range(TICK_LIMIT):
        sim.tick()
        assert not sim.state.queues[closed_edge]
        for skier in sim.skiers:
            assert skier.location_kind == LocationKind.NODE
            assert skier.location_index == start

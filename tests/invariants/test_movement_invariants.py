"""The movement tick must hold the invariants of the plan.

Each scenario runs a few skiers on the small resort with one seed.
The checks cover the skier count, the one valid state, a closed edge,
the increasing time, the array lengths, and the travel-time range.
"""

import random
from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import (
    LocationKind,
    MountainSim,
    Status,
    advance_on_edges,
    arrive_at_nodes,
    build_route_table,
    display_progress,
    load_topology,
    new_dynamic_state,
    population_from_starts,
    select_next_edges,
    walk_route,
)
from avalanche.sim.population import ABILITY_NAMES
from avalanche.sim.routes import NO_EDGE
from avalanche.traces import (
    SnapshotSchemaError,
    encode_physical_replay_snapshot,
    load_physical_replay_snapshot,
    restore_snapshot,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEEDS = tuple(range(10))
TICK_LIMIT = 600
MAX_SKIERS = 4
TICK_SECONDS = 5.0

ON_EDGE = (LocationKind.PISTE, LocationKind.LIFT, LocationKind.QUEUE)


def build_scenario(seed):
    """Return the population and the closed edge of one seed.

    The start node and the destination node change with the seed.
    The destination must be reachable from each start node.
    The closed edge is an edge that no shortest path of the scenario uses.
    """
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    ability = ABILITY_NAMES.index("beginner")
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
        if start == destination
        or routes.next_edge[ability, start, destination] != NO_EDGE
    ]
    chosen = [choose.choice(reachable) for _ in range(choose.randint(2, MAX_SKIERS))]
    pop = population_from_starts(starts=chosen, destinations=destination)

    used = {
        edge
        for start in chosen
        for edge in walk_route(routes, topology, start, destination, ability=ability)
    }
    spare = sorted(set(range(topology.edge_count)) - used)
    return pop, choose.choice(spare)


def check_one_valid_state(sim):
    """Check that each skier has exactly one valid location and one valid status."""
    pop = sim.population
    kinds = np.array(tuple(LocationKind), dtype=np.int8)
    assert np.all(np.isin(pop.location_kind, kinds))
    assert np.all(np.isin(pop.status, np.array(tuple(Status), dtype=np.int8)))
    assert not hasattr(pop, "progress")
    assert np.all(pop.required_travel_seconds >= 0.0)
    assert np.all(pop.remaining_travel_seconds >= 0.0)
    assert np.all(pop.remaining_travel_seconds <= pop.required_travel_seconds)

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


def test_evaluator_replay_keeps_exact_travel_state():
    """Keep exact travel values in the privileged display state."""
    population, _ = build_scenario(4)
    original = MountainSim(FIXTURE)
    original.reset(4)
    original.population = population
    original.tick()
    row = encode_physical_replay_snapshot(
        original,
        view_kind="evaluator",
        run_id="invariant",
        episode_id="episode-0",
    )
    population = load_physical_replay_snapshot(row)["state"]["population"]
    progress = display_progress(original.population)
    assert np.all((progress >= 0.0) & (progress <= 1.0))
    np.testing.assert_array_equal(
        population["required_travel_seconds"],
        original.population.required_travel_seconds,
    )
    np.testing.assert_array_equal(
        population["remaining_travel_seconds"],
        original.population.remaining_travel_seconds,
    )

    restored = MountainSim(FIXTURE)
    restored.reset(4)
    with pytest.raises(SnapshotSchemaError, match="display-only"):
        restore_snapshot(restored, row)


def test_exact_travel_completes_on_the_twenty_fourth_tick():
    """Complete a 120-second edge after exactly 24 five-second ticks."""
    topology = load_topology(FIXTURE)
    state = new_dynamic_state(topology)
    pop = population_from_starts(
        [topology.node_index["base_village"]],
        topology.node_index["base_exit"],
    )
    pop.location_kind[0] = LocationKind.PISTE
    pop.location_index[0] = 0
    pop.required_travel_seconds[0] = 120.0
    pop.remaining_travel_seconds[0] = 120.0

    for tick in range(23):
        advance_on_edges(pop, topology, state, TICK_SECONDS)
        transitions = arrive_at_nodes(pop, topology, tick * TICK_SECONDS, TICK_SECONDS)
        assert transitions.completed_skiers.size == 0

    advance_on_edges(pop, topology, state, TICK_SECONDS)
    transitions = arrive_at_nodes(pop, topology, 115.0, TICK_SECONDS)

    np.testing.assert_array_equal(transitions.completed_skiers, [0])
    np.testing.assert_array_equal(transitions.edge_completed_at, [120.0])


def test_completion_commits_simultaneously():
    """Commit every matching edge completion at one boundary."""
    topology = load_topology(FIXTURE)
    state = new_dynamic_state(topology)
    pop = population_from_starts(
        [topology.node_index["base_village"]] * 3,
        topology.node_index["base_exit"],
    )
    pop.location_kind[:] = LocationKind.PISTE
    pop.location_index[:] = 0
    pop.required_travel_seconds[:] = 120.0
    pop.remaining_travel_seconds[:] = [5.0, 5.0, 10.0]

    advance_on_edges(pop, topology, state, TICK_SECONDS)
    transitions = arrive_at_nodes(pop, topology, 0.0, TICK_SECONDS)

    np.testing.assert_array_equal(transitions.completed_skiers, [0, 1])
    np.testing.assert_array_equal(transitions.edge_completed_at, [5.0, 5.0])
    np.testing.assert_array_equal(
        pop.location_kind,
        [LocationKind.NODE, LocationKind.NODE, LocationKind.PISTE],
    )


def test_boundary_arrival_enters_without_residual_movement():
    """Enter the next edge at completion without moving on that edge."""
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    state = new_dynamic_state(topology)
    edge = 8
    destination = topology.node_index["base_exit"]
    pop = population_from_starts([int(topology.edge_source[edge])], destination)
    pop.location_kind[0] = LocationKind.PISTE
    pop.location_index[0] = edge
    pop.required_travel_seconds[0] = 210.0
    pop.remaining_travel_seconds[0] = TICK_SECONDS
    next_edge = int(
        routes.next_edge[pop.ability[0], topology.edge_destination[edge], destination]
    )
    advance_on_edges(pop, topology, state, TICK_SECONDS)
    transitions = arrive_at_nodes(pop, topology, 0.0, TICK_SECONDS)
    select_next_edges(pop, topology, routes, state, np.random.default_rng(7))

    np.testing.assert_array_equal(transitions.edge_completed_at, [TICK_SECONDS])
    assert pop.location_kind[0] == LocationKind.PISTE
    assert pop.location_index[0] == next_edge
    required = topology.edge_nominal_travel_time[next_edge]
    assert pop.required_travel_seconds[0] == required
    assert pop.remaining_travel_seconds[0] == required


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
    closed_edge = walk_route(
        routes,
        topology,
        start,
        destination,
        ability=ABILITY_NAMES.index("beginner"),
    )[0]

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

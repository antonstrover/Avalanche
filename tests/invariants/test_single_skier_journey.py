"""One skier must travel from the entrance to the exit."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import (
    LocationKind,
    Status,
    accumulate_times,
    advance_on_edges,
    arrive_at_nodes,
    build_route_table,
    load_topology,
    new_dynamic_state,
    population_from_starts,
    select_next_edges,
    serve_lift_queues,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
MEDIUM_FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml"
)
TICK_SECONDS = 5.0
TICK_LIMIT = 2000
CHOICE_SEED = 7


def run_tick(pop, topology, routes, state, rng):
    """Run the steps 3 to 6 of one movement tick."""
    serve_lift_queues(pop, topology, state, TICK_SECONDS)
    advance_on_edges(pop, topology, state, TICK_SECONDS)
    arrive_at_nodes(pop, topology)
    select_next_edges(pop, topology, routes, state, rng)
    accumulate_times(pop, TICK_SECONDS)


@pytest.fixture(scope="module")
def journey():
    """Run one skier from the entrance to the exit and return the record of the run."""
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    state = new_dynamic_state(topology)
    rng = np.random.default_rng(CHOICE_SEED)

    pop = population_from_starts(
        starts=[topology.node_index["base_village"]],
        destinations=topology.node_index["base_exit"],
    )

    kinds = []
    journey_times = []
    for _ in range(TICK_LIMIT):
        run_tick(pop, topology, routes, state, rng)
        kinds.append(LocationKind(pop.location_kind[0]))
        journey_times.append(float(pop.journey_time[0]))
        if pop.status[0] == Status.COMPLETE:
            break

    return pop, kinds, journey_times


def test_the_skier_completes_the_journey(journey):
    pop, _, _ = journey
    assert pop.status[0] == Status.COMPLETE
    assert pop.location_kind[0] == LocationKind.FINISHED


def test_the_journey_time_increases_in_each_tick(journey):
    _, kinds, journey_times = journey
    # The last tick makes the skier complete, so it gains no journey time.
    active_times = journey_times[: kinds.index(LocationKind.FINISHED)]
    assert active_times == sorted(set(active_times))
    assert active_times[0] == pytest.approx(TICK_SECONDS)


def test_the_skier_passes_through_a_lift_queue_and_a_lift(journey):
    pop, kinds, _ = journey
    assert LocationKind.QUEUE in kinds
    assert LocationKind.LIFT in kinds
    assert LocationKind.PISTE in kinds
    assert pop.wait_time[0] > 0.0


def test_a_skier_completes_the_medium_resort_fractional_lift_journey():
    """A skier must leave the queue of the 700 skier-hour lift."""
    topology = load_topology(MEDIUM_FIXTURE)
    routes = build_route_table(topology)
    state = new_dynamic_state(topology)
    rng = np.random.default_rng(CHOICE_SEED)
    source = topology.node_index["marmottons_base"]
    destination = topology.node_index["praz_exit"]
    lift = int(topology.edges_from(source)[0])
    pop = population_from_starts([source], destination)
    locations = []

    for _ in range(TICK_LIMIT):
        run_tick(pop, topology, routes, state, rng)
        locations.append((int(pop.location_kind[0]), int(pop.location_index[0])))
        if pop.status[0] == Status.COMPLETE:
            break

    assert (LocationKind.LIFT, lift) in locations
    assert pop.status[0] == Status.COMPLETE

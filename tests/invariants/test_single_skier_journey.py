"""One skier must travel from the entrance to the exit."""

from pathlib import Path

import pytest

from avalanche.sim import (
    LocationKind,
    Skier,
    Status,
    accumulate_times,
    advance_on_edges,
    arrive_at_nodes,
    build_route_table,
    load_topology,
    new_dynamic_state,
    select_next_edges,
    serve_lift_queues,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
TICK_SECONDS = 5.0
TICK_LIMIT = 2000


def run_tick(skiers, topology, routes, state):
    """Run the steps 3 to 6 of one movement tick."""
    serve_lift_queues(skiers, topology, state, TICK_SECONDS)
    advance_on_edges(skiers, topology, TICK_SECONDS)
    arrive_at_nodes(skiers, topology)
    select_next_edges(skiers, topology, routes, state)
    accumulate_times(skiers, TICK_SECONDS)


@pytest.fixture(scope="module")
def journey():
    """Run one skier from the entrance to the exit and return the record of the run."""
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    state = new_dynamic_state(topology)

    start = topology.node_index["base_village"]
    skier = Skier(destination=topology.node_index["base_exit"], location_index=start)
    skiers = [skier]

    kinds = []
    journey_times = []
    for _ in range(TICK_LIMIT):
        run_tick(skiers, topology, routes, state)
        kinds.append(skier.location_kind)
        journey_times.append(skier.journey_time)
        if skier.status == Status.COMPLETE:
            break

    return skier, kinds, journey_times


def test_the_skier_completes_the_journey(journey):
    skier, _, _ = journey
    assert skier.status == Status.COMPLETE
    assert skier.location_kind == LocationKind.FINISHED


def test_the_journey_time_increases_in_each_tick(journey):
    _, kinds, journey_times = journey
    # The last tick makes the skier complete, so it gains no journey time.
    active_times = journey_times[: kinds.index(LocationKind.FINISHED)]
    assert active_times == sorted(set(active_times))
    assert active_times[0] == pytest.approx(TICK_SECONDS)


def test_the_skier_passes_through_a_lift_queue_and_a_lift(journey):
    skier, kinds, _ = journey
    assert LocationKind.QUEUE in kinds
    assert LocationKind.LIFT in kinds
    assert LocationKind.PISTE in kinds
    assert skier.wait_time > 0.0

"""Lift service must match each configured rate."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import LocationKind, MountainSim, load_topology
from avalanche.sim.movement import (
    LIFT_EDGE,
    new_dynamic_state,
    serve_lift_queues,
    update_congestion,
)
from avalanche.sim.population import empty_population

ROOT = Path(__file__).resolve().parents[2]
MOUNTAINS = (
    ROOT / "configs" / "mountain" / "small-resort.yaml",
    ROOT / "configs" / "mountain" / "medium-resort.yaml",
)
TICK_SECONDS = 5.0
TICKS_IN_HOUR = 720


def queued_population(topology, edges: np.ndarray):
    """Return one queued skier for each requested edge."""
    pop = empty_population(int(edges.size))
    pop.location_kind[:] = LocationKind.QUEUE
    pop.location_index[:] = edges
    pop.destination[:] = topology.edge_destination[edges]
    pop.queue_ticket[:] = np.arange(edges.size)
    pop.next_ticket = len(pop)
    return pop


def serve_one_hour(path: Path, factor: float = 1.0):
    """Return the boarded count for every lift during one hour."""
    topology = load_topology(path)
    lifts = np.flatnonzero(topology.edge_type == LIFT_EDGE)
    queue_edges = np.repeat(
        lifts, np.ceil(topology.edge_lift_throughput[lifts]).astype(np.int64) + 1
    )
    pop = queued_population(topology, queue_edges)
    state = new_dynamic_state(topology)
    state.lift_capacity_factor[lifts] = factor
    boarded = np.zeros(topology.edge_count, dtype=np.int64)

    for _ in range(TICKS_IN_HOUR):
        serve_lift_queues(pop, topology, state, TICK_SECONDS)
        riders = np.flatnonzero(pop.location_kind == LocationKind.LIFT)
        np.add.at(boarded, pop.location_index[riders], 1)
        pop.location_kind[riders] = LocationKind.FINISHED

    return topology, lifts, boarded, state


@pytest.mark.parametrize("path", MOUNTAINS, ids=lambda path: path.stem)
def test_every_lift_matches_its_configured_throughput(path):
    topology, lifts, boarded, state = serve_one_hour(path)

    assert np.all(np.abs(boarded[lifts] - topology.edge_lift_throughput[lifts]) <= 1)
    assert np.all(state.lift_service_residual >= 0.0)
    assert np.all(state.lift_service_residual < 1.0)


def test_a_fractional_rate_serves_across_successive_ticks():
    topology = load_topology(MOUNTAINS[1])
    edge = next(
        int(edge)
        for edge in np.flatnonzero(topology.edge_type == LIFT_EDGE)
        if topology.edge_lift_throughput[edge] == 700.0
    )
    pop = queued_population(topology, np.full(3, edge, dtype=np.int32))
    state = new_dynamic_state(topology)

    serve_lift_queues(pop, topology, state, TICK_SECONDS)
    assert np.count_nonzero(pop.location_kind == LocationKind.LIFT) == 0

    serve_lift_queues(pop, topology, state, TICK_SECONDS)
    assert np.count_nonzero(pop.location_kind == LocationKind.LIFT) == 1


def test_a_capacity_factor_scales_the_service_rate():
    topology, lifts, boarded, _ = serve_one_hour(MOUNTAINS[1], factor=0.5)

    expected = topology.edge_lift_throughput[lifts] * 0.5
    assert np.all(np.abs(boarded[lifts] - expected) <= 1)


@pytest.mark.parametrize(
    "disabled_field",
    ("closed", "weather_closed", "failure_closed", "lift_stopped"),
)
def test_a_disabled_lift_does_not_serve_or_accumulate(disabled_field):
    topology = load_topology(MOUNTAINS[0])
    edge = int(np.flatnonzero(topology.edge_type == LIFT_EDGE)[0])
    pop = queued_population(topology, np.full(4, edge, dtype=np.int32))
    state = new_dynamic_state(topology)
    getattr(state, disabled_field)[edge] = True

    for _ in range(10):
        serve_lift_queues(pop, topology, state, TICK_SECONDS)

    assert np.all(pop.location_kind == LocationKind.QUEUE)
    assert state.lift_service_residual[edge] == 0.0


def test_an_idle_lift_does_not_store_a_service_burst():
    topology = load_topology(MOUNTAINS[1])
    edge = next(
        int(edge)
        for edge in np.flatnonzero(topology.edge_type == LIFT_EDGE)
        if topology.edge_lift_throughput[edge] == 700.0
    )
    state = new_dynamic_state(topology)

    for _ in range(TICKS_IN_HOUR):
        serve_lift_queues(empty_population(0), topology, state, TICK_SECONDS)

    pop = queued_population(topology, np.full(10, edge, dtype=np.int32))
    serve_lift_queues(pop, topology, state, TICK_SECONDS)

    assert np.count_nonzero(pop.location_kind == LocationKind.LIFT) <= 1


def test_lift_service_uses_the_queue_ticket_order():
    topology = load_topology(MOUNTAINS[1])
    edge = next(
        int(edge)
        for edge in np.flatnonzero(topology.edge_type == LIFT_EDGE)
        if topology.edge_lift_throughput[edge] == 700.0
    )
    pop = queued_population(topology, np.full(3, edge, dtype=np.int32))
    pop.queue_ticket[:] = (30, 10, 20)
    state = new_dynamic_state(topology)

    for _ in range(2):
        serve_lift_queues(pop, topology, state, TICK_SECONDS)

    assert pop.location_kind.tolist() == [
        LocationKind.QUEUE,
        LocationKind.LIFT,
        LocationKind.QUEUE,
    ]


def test_a_reset_clears_each_service_residual():
    sim = MountainSim(MOUNTAINS[1])
    sim.reset(7)
    sim.tick()
    assert np.any(sim.state.lift_service_residual > 0.0)

    sim.reset(7)
    assert np.all(sim.state.lift_service_residual == 0.0)


def test_a_full_lift_does_not_board_a_waiting_skier():
    topology = load_topology(MOUNTAINS[0])
    edge = int(np.flatnonzero(topology.edge_type == LIFT_EDGE)[0])
    pop = queued_population(topology, np.full(4, edge, dtype=np.int32))
    state = new_dynamic_state(topology)
    state.occupancy[edge] = int(topology.edge_safe_capacity[edge])

    for _ in range(4):
        serve_lift_queues(pop, topology, state, TICK_SECONDS)

    assert np.all(pop.location_kind == LocationKind.QUEUE)
    assert state.lift_service_residual[edge] < 1.0


def test_boarding_resumes_after_one_rider_leaves():
    topology = load_topology(MOUNTAINS[1])
    edge = next(
        int(edge)
        for edge in np.flatnonzero(topology.edge_type == LIFT_EDGE)
        if topology.edge_lift_throughput[edge] == 700.0
    )
    pop = queued_population(topology, np.full(3, edge, dtype=np.int32))
    state = new_dynamic_state(topology)
    state.occupancy[edge] = int(topology.edge_safe_capacity[edge])

    serve_lift_queues(pop, topology, state, TICK_SECONDS)
    state.occupancy[edge] -= 1
    serve_lift_queues(pop, topology, state, TICK_SECONDS)

    assert np.count_nonzero(pop.location_kind == LocationKind.LIFT) == 1


def test_boarding_uses_the_smaller_service_and_room_limit():
    topology = load_topology(MOUNTAINS[1])
    edge = int(np.argmax(topology.edge_lift_throughput))
    pop = queued_population(topology, np.full(10, edge, dtype=np.int32))
    state = new_dynamic_state(topology)
    state.occupancy[edge] = int(topology.edge_safe_capacity[edge]) - 2

    serve_lift_queues(pop, topology, state, TICK_SECONDS)

    assert np.count_nonzero(pop.location_kind == LocationKind.LIFT) == 2


def test_a_lift_queue_stays_outside_the_onboard_occupancy():
    topology = load_topology(MOUNTAINS[0])
    edge = int(np.flatnonzero(topology.edge_type == LIFT_EDGE)[0])
    capacity = int(topology.edge_safe_capacity[edge])
    pop = empty_population(capacity + 3)
    pop.location_index[:] = edge
    pop.location_kind[:capacity] = LocationKind.LIFT
    pop.location_kind[capacity:] = LocationKind.QUEUE
    travel_seconds = topology.edge_nominal_travel_time[edge]
    pop.required_travel_seconds[:capacity] = travel_seconds
    pop.remaining_travel_seconds[:capacity] = travel_seconds
    pop.queue_ticket[capacity:] = np.arange(3)
    state = new_dynamic_state(topology)

    update_congestion(pop, topology, state)

    assert state.occupancy[edge] == capacity
    assert state.queue_length[edge] == 3

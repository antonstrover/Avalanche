"""Check the timed stranded-state transition."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import PROTOCOL_TIME_EPSILON_SECONDS
from avalanche.sim import (
    LocationKind,
    build_route_table,
    load_topology,
    population_from_starts,
)
from avalanche.sim.movement import (
    advance_on_edges,
    lift_unavailable_mask,
    new_dynamic_state,
    return_unavailable_lift_queues,
    update_lift_blocked_times,
    update_stranded,
)
from avalanche.sim.population import ABILITY_NAMES
from avalanche.sim.skier import Status

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
MEDIUM_FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml"
)


def edge_index(topology, source: str, destination: str) -> int:
    """Return one edge index from its endpoint names."""
    matches = np.flatnonzero(
        (topology.edge_source == topology.node_index[source])
        & (topology.edge_destination == topology.node_index[destination])
    )
    assert matches.size == 1
    return int(matches[0])


def test_a_closed_route_marks_a_skier_after_the_limit():
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    source = topology.node_index["base_village"]
    destination = topology.node_index["base_exit"]
    population = population_from_starts([source], destination)
    state = new_dynamic_state(topology)
    route_edge = int(
        routes.next_edge[ABILITY_NAMES.index("beginner"), source, destination]
    )
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
    route_edge = int(
        routes.next_edge[ABILITY_NAMES.index("beginner"), source, destination]
    )
    state.closed[route_edge] = True
    update_stranded(population, routes, state, 5.0, 10.0)
    state.closed[route_edge] = False
    update_stranded(population, routes, state, 5.0, 10.0)
    assert population.blocked_time[0] == 0.0
    assert population.status[0] == Status.ACTIVE


def test_the_stranding_limit_uses_the_shared_epsilon():
    """Apply one elapsed boundary to each blocked skier."""
    topology = load_topology(FIXTURE)
    routes = build_route_table(topology)
    source = topology.node_index["base_village"]
    destination = topology.node_index["base_exit"]
    population = population_from_starts([source, source], destination)
    state = new_dynamic_state(topology)
    route_edge = int(
        routes.next_edge[ABILITY_NAMES.index("beginner"), source, destination]
    )
    state.closed[route_edge] = True
    epsilon = PROTOCOL_TIME_EPSILON_SECONDS
    population.blocked_time[:] = [5.0 - 0.5 * epsilon, 5.0 - 2.0 * epsilon]

    changed = update_stranded(population, routes, state, 5.0, 10.0, epsilon)

    np.testing.assert_array_equal(changed, [0])
    np.testing.assert_array_equal(population.status, [Status.STRANDED, Status.ACTIVE])


def test_both_lift_counters_use_the_shared_epsilon():
    """Apply the shared epsilon to queue and onboard thresholds."""
    topology = load_topology(FIXTURE)
    lift = edge_index(topology, "lift1_base", "lift1_top")
    source = topology.node_index["lift1_base"]
    destination = topology.node_index["lift1_top"]
    population = population_from_starts([source] * 4, destination)
    population.location_kind[:] = (
        LocationKind.NODE,
        LocationKind.NODE,
        LocationKind.LIFT,
        LocationKind.LIFT,
    )
    population.location_index[:] = (source, source, lift, lift)
    epsilon = PROTOCOL_TIME_EPSILON_SECONDS
    population.queue_no_route_blocked_seconds[:2] = (
        5.0 - 0.5 * epsilon,
        5.0 - 2.0 * epsilon,
    )
    population.onboard_blocked_seconds[2:] = (
        5.0 - 0.5 * epsilon,
        5.0 - 2.0 * epsilon,
    )
    state = new_dynamic_state(topology)
    state.lift_stopped[lift] = True

    changed = update_lift_blocked_times(
        population,
        topology,
        state,
        np.empty(0, dtype=np.int64),
        5.0,
        10.0,
        epsilon,
    )

    np.testing.assert_array_equal(changed, [0, 2])
    np.testing.assert_array_equal(
        population.status,
        [Status.STRANDED, Status.ACTIVE, Status.STRANDED, Status.ACTIVE],
    )


def test_queued_no_route_strands_at_timeout():
    """Strand a returned queue member at the exact timeout."""
    topology = load_topology(FIXTURE)
    lift = edge_index(topology, "lift1_base", "lift1_top")
    source = topology.node_index["lift1_base"]
    destination = topology.node_index["lift1_top"]
    population = population_from_starts([source], destination)
    population.location_kind[0] = LocationKind.QUEUE
    population.location_index[0] = lift
    population.queue_ticket[0] = 4
    population.queue_source_node[0] = source
    state = new_dynamic_state(topology)
    state.lift_stopped[lift] = True

    returned = return_unavailable_lift_queues(population, topology, state)
    changed = update_lift_blocked_times(
        population, topology, state, returned, 5.0, 10.0
    )

    assert changed.size == 0
    assert population.location_kind[0] == LocationKind.NODE
    assert population.location_index[0] == source
    assert population.queue_no_route_blocked_seconds[0] == 5.0
    assert population.onboard_blocked_seconds[0] == 0.0
    assert population.status[0] == Status.ACTIVE

    changed = update_lift_blocked_times(
        population,
        topology,
        state,
        np.empty(0, dtype=np.int64),
        5.0,
        10.0,
    )

    np.testing.assert_array_equal(changed, [0])
    assert population.queue_no_route_blocked_seconds[0] == 10.0
    assert population.onboard_blocked_seconds[0] == 0.0
    assert population.status[0] == Status.STRANDED


@pytest.mark.parametrize(
    "field",
    ("closed", "weather_closed", "failure_closed", "lift_stopped"),
    ids=("controller", "weather", "scheduled", "mechanical"),
)
def test_each_closure_source_uses_one_continuous_period(field: str):
    """Apply one blocked-period contract to each lift closure source."""
    topology = load_topology(FIXTURE)
    lift = edge_index(topology, "lift1_base", "lift1_top")
    source = topology.node_index["lift1_base"]
    destination = topology.node_index["lift1_top"]
    population = population_from_starts([source, source], destination)
    population.location_kind[:] = (LocationKind.QUEUE, LocationKind.LIFT)
    population.location_index[:] = lift
    population.required_travel_seconds[1] = 30.0
    population.remaining_travel_seconds[1] = 30.0
    population.queue_ticket[0] = 2
    population.queue_source_node[0] = source
    state = new_dynamic_state(topology)
    getattr(state, field)[lift] = True

    unavailable = lift_unavailable_mask(topology, state)
    assert unavailable[lift]
    returned = return_unavailable_lift_queues(population, topology, state, unavailable)
    first = update_lift_blocked_times(population, topology, state, returned, 5.0, 10.0)

    assert first.size == 0
    np.testing.assert_array_equal(population.queue_no_route_blocked_seconds, [5.0, 0.0])
    np.testing.assert_array_equal(population.onboard_blocked_seconds, [0.0, 5.0])

    second = update_lift_blocked_times(
        population,
        topology,
        state,
        np.empty(0, dtype=np.int64),
        5.0,
        10.0,
    )

    np.testing.assert_array_equal(second, [0, 1])
    np.testing.assert_array_equal(population.status, [Status.STRANDED, Status.STRANDED])
    np.testing.assert_array_equal(
        population.queue_no_route_blocked_seconds, [10.0, 0.0]
    )
    np.testing.assert_array_equal(population.onboard_blocked_seconds, [0.0, 10.0])


def test_finite_route_resets_queue_counter_immediately():
    """Reset the queue counter before the skier enters an alternative edge."""
    topology = load_topology(MEDIUM_FIXTURE)
    lift = edge_index(topology, "col_bonneval", "crete_east")
    source = topology.node_index["col_bonneval"]
    destination = topology.node_index["bonneval_exit"]
    population = population_from_starts([source], destination)
    population.queue_no_route_blocked_seconds[0] = 5.0
    state = new_dynamic_state(topology)
    state.lift_stopped[lift] = True

    changed = update_lift_blocked_times(
        population,
        topology,
        state,
        np.empty(0, dtype=np.int64),
        5.0,
        10.0,
    )

    assert changed.size == 0
    assert population.location_kind[0] == LocationKind.NODE
    assert population.location_index[0] == source
    assert population.queue_no_route_blocked_seconds[0] == 0.0
    assert population.onboard_blocked_seconds[0] == 0.0
    assert population.status[0] == Status.ACTIVE


def test_recovery_resets_onboard_counter_immediately():
    """Clear an onboard counter at the first recovered tick."""
    topology = load_topology(FIXTURE)
    lift = edge_index(topology, "lift1_base", "lift1_top")
    source = topology.node_index["lift1_base"]
    destination = topology.node_index["base_exit"]
    population = population_from_starts([source], destination)
    population.location_kind[0] = LocationKind.LIFT
    population.location_index[0] = lift
    population.onboard_blocked_seconds[0] = 5.0
    state = new_dynamic_state(topology)

    changed = update_lift_blocked_times(
        population,
        topology,
        state,
        np.empty(0, dtype=np.int64),
        5.0,
        10.0,
    )

    assert changed.size == 0
    assert population.location_kind[0] == LocationKind.LIFT
    assert population.onboard_blocked_seconds[0] == 0.0
    assert population.queue_no_route_blocked_seconds[0] == 0.0
    assert population.status[0] == Status.ACTIVE


def test_separate_failures_do_not_combine():
    """Reset one blocked period before a later source starts another."""
    topology = load_topology(FIXTURE)
    lift = edge_index(topology, "lift1_base", "lift1_top")
    source = topology.node_index["lift1_base"]
    destination = topology.node_index["base_exit"]
    population = population_from_starts([source], destination)
    population.location_kind[0] = LocationKind.LIFT
    population.location_index[0] = lift
    state = new_dynamic_state(topology)
    empty = np.empty(0, dtype=np.int64)

    state.lift_stopped[lift] = True
    first = update_lift_blocked_times(population, topology, state, empty, 5.0, 10.0)
    state.lift_stopped[lift] = False
    recovered = update_lift_blocked_times(population, topology, state, empty, 5.0, 10.0)
    state.weather_closed[lift] = True
    second = update_lift_blocked_times(population, topology, state, empty, 5.0, 10.0)

    assert first.size == 0
    assert recovered.size == 0
    assert second.size == 0
    assert population.onboard_blocked_seconds[0] == 5.0
    assert population.queue_no_route_blocked_seconds[0] == 0.0
    assert population.status[0] == Status.ACTIVE


def test_recovery_does_not_revive_a_stranded_skier():
    """Move only an active onboard skier after service recovers."""
    topology = load_topology(FIXTURE)
    lift = edge_index(topology, "lift1_base", "lift1_top")
    source = topology.node_index["lift1_base"]
    destination = topology.node_index["base_exit"]
    population = population_from_starts([source, source], destination)
    population.location_kind[:] = LocationKind.LIFT
    population.location_index[:] = lift
    population.required_travel_seconds[:] = 30.0
    population.remaining_travel_seconds[:] = 30.0
    population.onboard_blocked_seconds[:] = (5.0, 10.0)
    population.status[1] = Status.STRANDED
    state = new_dynamic_state(topology)

    changed = update_lift_blocked_times(
        population,
        topology,
        state,
        np.empty(0, dtype=np.int64),
        5.0,
        10.0,
    )
    advance_on_edges(population, topology, state, 5.0)

    assert changed.size == 0
    np.testing.assert_array_equal(population.onboard_blocked_seconds, [0.0, 0.0])
    assert population.status[0] == Status.ACTIVE
    assert population.status[1] == Status.STRANDED
    assert population.remaining_travel_seconds[0] < 30.0
    assert population.remaining_travel_seconds[1] == 30.0

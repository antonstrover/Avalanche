"""Operational route choice must use the locked cost contract."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import ReportedRiskConfig, RoutingConfig
from avalanche.scenarios.sensors import perfect_route_sensor_packet
from avalanche.sim import (
    LocationKind,
    OperationalRouteCosts,
    build_route_table,
    load_topology,
    new_dynamic_state,
    population_from_starts,
    select_next_edges,
    serve_lift_queues,
)
from avalanche.sim.population import ABILITY_NAMES
from avalanche.sim.topology import EDGE_TYPE_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
ADVANCED = ABILITY_NAMES.index("advanced")
LIFT = EDGE_TYPE_NAMES.index("lift")


def edge_index(topology, source: str, destination: str) -> int:
    """Return one unique edge index."""
    matches = np.flatnonzero(
        (topology.edge_source == topology.node_index[source])
        & (topology.edge_destination == topology.node_index[destination])
    )
    assert matches.size == 1
    return int(matches[0])


@pytest.fixture
def choice_fixture():
    """Return two direct alternatives from one node to one destination."""
    topology = load_topology(FIXTURE)
    first = edge_index(topology, "lift1_top", "lift2_base")
    second = edge_index(topology, "lift1_top", "ridge_junction")
    destination = topology.node_index["base_exit"]
    edge_destination = topology.edge_destination.copy()
    edge_destination[[first, second]] = destination
    travel_time = topology.edge_nominal_travel_time.copy()
    travel_time[[first, second]] = 100.0
    capacity = topology.edge_safe_capacity.copy()
    capacity[[first, second]] = 1000.0
    topology = replace(
        topology,
        edge_destination=edge_destination,
        edge_nominal_travel_time=travel_time,
        edge_safe_capacity=capacity,
    )
    return topology, topology.node_index["lift1_top"], destination, first, second


def packet(topology, **changes):
    """Return one exact operational packet."""
    edge_count = topology.edge_count
    values = {
        "availability": np.ones(edge_count, dtype=np.bool_),
        "speed_factor": np.ones(edge_count),
        "density_ratio": np.zeros(edge_count),
        "weather_risk": np.zeros(edge_count),
        "queue_length": np.zeros(edge_count),
        "boarding_throughput": np.ones(edge_count),
    }
    values.update(changes)
    return perfect_route_sensor_packet(**values)


def choose(
    choice_fixture,
    *,
    route_packet=None,
    preferences=None,
    tolerance=1.0,
    seed=17,
    capacity=None,
):
    """Choose one edge for one advanced skier."""
    topology, source, destination, first, second = choice_fixture
    if capacity is not None:
        values = topology.edge_safe_capacity.copy()
        values[[first, second]] = capacity
        topology = replace(topology, edge_safe_capacity=values)
    state = new_dynamic_state(topology)
    if preferences is not None:
        state.route_preferences[ADVANCED] = preferences
    pop = population_from_starts([source], destination)
    pop.ability[:] = ADVANCED
    pop.compliance[:] = 1.0
    pop.risk_tolerance[:] = tolerance
    select_next_edges(
        pop,
        topology,
        build_route_table(topology),
        state,
        np.random.default_rng(seed),
        np.random.default_rng(seed + 1),
        route_packet or packet(topology),
        RoutingConfig(),
        ReportedRiskConfig(),
    )
    return pop, topology, state, first, second


def test_closed_static_edge_uses_open_safe_alternative(choice_fixture):
    topology, _, _, first, second = choice_fixture
    available = np.ones(topology.edge_count, dtype=np.bool_)
    available[first] = False

    pop, _, _, _, _ = choose(
        choice_fixture, route_packet=packet(topology, availability=available)
    )

    assert pop.location_index[0] == second


def test_positive_preference_reduces_cost(choice_fixture):
    topology, _, _, first, second = choice_fixture
    preferences = np.zeros(topology.edge_count)
    preferences[first] = 1.0

    pop, _, _, _, _ = choose(choice_fixture, preferences=preferences)

    assert pop.location_index[0] == first
    assert pop.location_index[0] != second


def test_negative_preference_increases_cost(choice_fixture):
    topology, _, _, first, second = choice_fixture
    preferences = np.zeros(topology.edge_count)
    preferences[first] = -1.0

    pop, _, _, _, _ = choose(choice_fixture, preferences=preferences)

    assert pop.location_index[0] == second


def test_risk_tolerance_changes_choice(choice_fixture):
    topology, source, destination, first, second = choice_fixture
    travel = topology.edge_nominal_travel_time.copy()
    travel[first] = 50.0
    topology = replace(topology, edge_nominal_travel_time=travel)
    fixture = topology, source, destination, first, second
    density = np.zeros(topology.edge_count)
    density[first] = 1.8
    route_packet = packet(topology, density_ratio=density)

    cautious, *_ = choose(fixture, route_packet=route_packet, tolerance=0.0)
    tolerant, *_ = choose(fixture, route_packet=route_packet, tolerance=1.0)

    assert cautious.location_index[0] == second
    assert tolerant.location_index[0] == first


def test_missing_risk_uses_conservative_value(choice_fixture):
    topology, source, destination, first, second = choice_fixture
    travel = topology.edge_nominal_travel_time.copy()
    travel[first] = 50.0
    topology = replace(topology, edge_nominal_travel_time=travel)
    route_packet = packet(topology)
    missing = route_packet.density_ratio_missing.copy()
    missing[first] = True
    route_packet = replace(route_packet, density_ratio_missing=missing)

    pop, *_ = choose(
        (topology, source, destination, first, second),
        route_packet=route_packet,
        tolerance=0.0,
    )

    assert pop.location_index[0] == second


def test_piste_and_lift_effective_time_formulas():
    topology = load_topology(FIXTURE)
    piste = edge_index(topology, "base_village", "lift1_base")
    lift = edge_index(topology, "lift1_base", "lift1_top")
    speed = np.ones(topology.edge_count)
    speed[piste] = 0.5
    queue = np.zeros(topology.edge_count)
    queue[lift] = 12.0
    throughput = np.ones(topology.edge_count)
    throughput[lift] = 0.2
    costs = OperationalRouteCosts.build(
        topology,
        packet(
            topology,
            speed_factor=speed,
            queue_length=queue,
            boarding_throughput=throughput,
        ),
        RoutingConfig(),
        ReportedRiskConfig(),
        ability=ADVANCED,
        risk_tolerance=1.0,
    )

    assert costs.effective_travel_seconds[piste] == pytest.approx(240.0)
    assert costs.effective_travel_seconds[lift] == pytest.approx(480.0)


def test_capacity_delay_does_not_reroll_tie(choice_fixture):
    pop, topology, state, first, second = choose(choice_fixture, capacity=0.0)
    selected = int(pop.chosen_edge[0])
    assert selected in (first, second)
    assert pop.location_kind[0] == LocationKind.NODE

    select_next_edges(
        pop,
        topology,
        build_route_table(topology),
        state,
        np.random.default_rng(99),
        np.random.default_rng(100),
        packet(topology),
    )

    assert pop.chosen_edge[0] == selected


def test_hidden_failure_fails_at_physical_gate():
    topology = load_topology(FIXTURE)
    source = topology.node_index["lift1_base"]
    destination = topology.node_index["base_exit"]
    lift = edge_index(topology, "lift1_base", "lift1_top")
    state = new_dynamic_state(topology)
    state.failure_closed[lift] = True
    pop = population_from_starts([source], destination)
    pop.ability[:] = ADVANCED

    select_next_edges(
        pop,
        topology,
        build_route_table(topology),
        state,
        np.random.default_rng(1),
        np.random.default_rng(2),
        packet(topology),
    )

    assert pop.location_kind[0] == LocationKind.NODE
    assert pop.location_index[0] == source
    assert pop.locally_rejected_edge[0] == lift


def test_equal_cost_ties_are_seeded(choice_fixture):
    first, *_ = choose(choice_fixture, seed=41)
    repeated, *_ = choose(choice_fixture, seed=41)

    assert first.location_index[0] == repeated.location_index[0]


def test_equal_cost_ties_draw_in_sorted_skier_order(choice_fixture):
    topology, source, destination, first, second = choice_fixture
    skier_count = 8
    pop = population_from_starts([source] * skier_count, destination)
    pop.ability[:] = ADVANCED
    pop.compliance[:] = 1.0
    tie_seed = 55
    candidates = np.sort(np.array([first, second], dtype=np.int32))
    expected_draws = np.random.default_rng(tie_seed).random(skier_count)
    expected = candidates[(expected_draws * candidates.size).astype(np.int64)]

    select_next_edges(
        pop,
        topology,
        build_route_table(topology),
        new_dynamic_state(topology),
        np.random.default_rng(54),
        np.random.default_rng(tie_seed),
        packet(topology),
    )

    np.testing.assert_array_equal(pop.chosen_edge, expected)


def test_lift_boarding_rejects_a_new_onward_closure():
    topology = load_topology(FIXTURE)
    source = topology.node_index["lift2_base"]
    destination = topology.node_index["base_exit"]
    lift = edge_index(topology, "lift2_base", "lift2_top")
    pop = population_from_starts([source], destination)
    pop.ability[:] = ADVANCED
    pop.location_kind[:] = LocationKind.QUEUE
    pop.location_index[:] = lift
    pop.queue_ticket[:] = 0
    pop.chosen_edge[:] = lift
    state = new_dynamic_state(topology)
    state.closed[topology.edges_from(topology.edge_destination[lift])] = True
    state.lift_service_residual[lift] = 1.0

    serve_lift_queues(pop, topology, state, 5.0)

    assert pop.location_kind[0] == LocationKind.NODE
    assert pop.location_index[0] == source
    assert pop.chosen_edge[0] == -1
    assert pop.locally_rejected_edge[0] == lift

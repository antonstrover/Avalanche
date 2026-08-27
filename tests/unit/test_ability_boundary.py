"""The simulator must enforce each skier's ability boundary."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import PopulationConfig
from avalanche.sim import (
    LocationKind,
    MountainSim,
    build_route_table,
    load_topology,
    population_from_starts,
    select_next_edges,
    walk_route,
)
from avalanche.sim.ability import ABILITY_NAMES, PISTE_LIMIT_BY_ABILITY
from avalanche.sim.movement import new_dynamic_state
from avalanche.sim.population import sample_population
from avalanche.sim.routes import NO_EDGE
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES

ROOT = Path(__file__).resolve().parents[2]
SMALL = ROOT / "configs" / "mountain" / "small-resort.yaml"
MEDIUM = ROOT / "configs" / "mountain" / "medium-resort.yaml"

PISTE = EDGE_TYPE_NAMES.index("piste")
LIFT = EDGE_TYPE_NAMES.index("lift")


def edge_with_difficulty(topology, difficulty: str) -> int:
    """Return one piste with the requested difficulty."""
    code = DIFFICULTY_NAMES.index(difficulty)
    matches = np.flatnonzero(
        (topology.edge_type == PISTE) & (topology.edge_difficulty == code)
    )
    assert matches.size
    return int(matches[0])


def topology_with_piste_difficulty(topology, difficulty: str):
    """Return a topology with one difficulty for every piste."""
    values = topology.edge_difficulty.copy()
    values[topology.edge_type == PISTE] = DIFFICULTY_NAMES.index(difficulty)
    return replace(topology, edge_difficulty=values)


def try_advice(topology, ability: int, edge: int):
    """Return one skier after it receives one edge as advice."""
    routes = build_route_table(topology)
    source = int(topology.edge_source[edge])
    destination = int(topology.edge_destination[edge])
    state = new_dynamic_state(topology)
    state.advice_edge[source, ability] = edge
    population = population_from_starts([source], destination)
    population.ability[:] = ability
    population.compliance[:] = 1.0
    select_next_edges(population, topology, routes, state, np.random.default_rng(7))
    return population


@pytest.mark.parametrize("ability", range(len(ABILITY_NAMES)))
@pytest.mark.parametrize("difficulty", ("green", "blue", "red", "black"))
def test_each_ability_accepts_only_its_permitted_pistes(ability, difficulty):
    topology = load_topology(MEDIUM)
    edge = edge_with_difficulty(topology, difficulty)
    population = try_advice(topology, ability, edge)
    permitted = DIFFICULTY_NAMES.index(difficulty) <= PISTE_LIMIT_BY_ABILITY[ability]

    if permitted:
        assert population.location_kind[0] == LocationKind.PISTE
        assert population.location_index[0] == edge
    else:
        assert not (
            population.location_kind[0] == LocationKind.PISTE
            and population.location_index[0] == edge
        )


def test_a_beginner_cannot_follow_hostile_black_piste_advice():
    topology = topology_with_piste_difficulty(load_topology(SMALL), "black")
    edge = edge_with_difficulty(topology, "black")
    population = try_advice(topology, ABILITY_NAMES.index("beginner"), edge)

    assert population.location_kind[0] == LocationKind.NODE
    assert population.location_index[0] == topology.edge_source[edge]


@pytest.mark.parametrize("ability", range(len(ABILITY_NAMES)))
def test_every_ability_can_board_a_lift_with_a_safe_onward_route(ability):
    topology = load_topology(SMALL)
    source = topology.node_index["lift1_base"]
    edge = int(topology.edges_from(source)[0])
    assert topology.edge_type[edge] == LIFT
    population = try_advice(topology, ability, edge)

    assert population.location_kind[0] == LocationKind.QUEUE
    assert population.location_index[0] == edge


def test_a_beginner_cannot_board_a_lift_without_a_safe_onward_route():
    topology = topology_with_piste_difficulty(load_topology(SMALL), "black")
    routes = build_route_table(topology)
    ability = ABILITY_NAMES.index("beginner")
    source = topology.node_index["lift2_base"]
    destination = topology.node_index["base_exit"]
    edge = int(topology.edges_from(source)[0])
    state = new_dynamic_state(topology)
    state.advice_edge[source, ability] = edge
    population = population_from_starts([source], destination)
    population.ability[:] = ability
    population.compliance[:] = 1.0

    select_next_edges(population, topology, routes, state, np.random.default_rng(7))

    assert routes.next_edge[ability, source, destination] == NO_EDGE
    assert population.location_kind[0] == LocationKind.NODE
    assert population.location_index[0] == source


def test_a_beginner_uses_a_longer_safe_route():
    topology = load_topology(MEDIUM)
    routes = build_route_table(topology)
    beginner = ABILITY_NAMES.index("beginner")
    intermediate = ABILITY_NAMES.index("intermediate")
    source = topology.node_index["plan_bois"]
    destination = topology.node_index["praz_exit"]

    easy = walk_route(routes, topology, source, destination, ability=beginner)
    faster = walk_route(routes, topology, source, destination, ability=intermediate)
    population = population_from_starts([source], destination)
    population.ability[:] = beginner

    select_next_edges(
        population,
        topology,
        routes,
        new_dynamic_state(topology),
        np.random.default_rng(7),
    )

    assert (
        routes.travel_time[beginner, source, destination]
        > routes.travel_time[intermediate, source, destination]
    )
    assert max(topology.edge_difficulty[easy]) <= PISTE_LIMIT_BY_ABILITY[beginner]
    assert max(topology.edge_difficulty[faster]) > PISTE_LIMIT_BY_ABILITY[beginner]
    assert population.location_index[0] == easy[0]


def test_a_skier_waits_when_no_safe_route_exists():
    topology = topology_with_piste_difficulty(load_topology(SMALL), "black")
    routes = build_route_table(topology)
    ability = ABILITY_NAMES.index("beginner")
    source = topology.node_index["lift2_top"]
    destination = topology.node_index["base_exit"]
    population = population_from_starts([source], destination)
    population.ability[:] = ability

    select_next_edges(
        population,
        topology,
        routes,
        new_dynamic_state(topology),
        np.random.default_rng(7),
    )

    assert routes.next_edge[ability, source, destination] == NO_EDGE
    assert population.location_kind[0] == LocationKind.NODE
    assert population.location_index[0] == source


def test_population_destinations_are_safe_for_each_skier():
    sim = MountainSim(MEDIUM)
    sim.reset(
        41,
        {
            "population": PopulationConfig(
                skier_count=5_000,
                arrival_window_seconds=600.0,
            )
        },
    )
    assert sim.routes is not None
    population = sim.population

    assert np.all(
        np.isfinite(
            sim.routes.travel_time[
                population.ability,
                population.location_index,
                population.destination,
            ]
        )
    )


def test_population_sampling_rejects_an_ability_without_a_safe_exit():
    topology = topology_with_piste_difficulty(load_topology(SMALL), "black")
    routes = build_route_table(topology)
    config = PopulationConfig(
        skier_count=10,
        ability_weights=(1.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="beginner.*no safe exit"):
        sample_population(np.random.default_rng(5), topology, routes, config)

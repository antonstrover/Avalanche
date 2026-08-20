"""The grouped route choice must follow the advice with the compliance probability."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.sim import (
    LocationKind,
    build_route_table,
    load_topology,
    new_dynamic_state,
    population_from_starts,
    select_next_edges,
)
from avalanche.sim.population import ABILITY_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SKIER_COUNT = 20_000
CHOICE_SEED = 11
TOLERANCE = 0.02


def edge_index(topology, source: str, destination: str) -> int:
    """Return the index of the edge between two named nodes."""
    pair = (topology.node_index[source], topology.node_index[destination])
    matches = np.flatnonzero(
        (topology.edge_source == pair[0]) & (topology.edge_destination == pair[1])
    )
    assert matches.size == 1, f"the edge {source} to {destination} is not unique"
    return int(matches[0])


@pytest.fixture(scope="module")
def choice_setup():
    """Return the topology, the routes, and the named node and edges of the test.

    The skiers stand at the top of the lift 1 and travel to the base exit.
    The route table sends them to the ridge junction.
    The advice sends them to the base of the lift 2 instead.
    """
    topology = load_topology(FIXTURE)
    # The test measures the share that follows the advice, not the capacity limit.
    # A capacity above the population lets each skier start its chosen edge.
    topology.edge_safe_capacity[:] = SKIER_COUNT
    routes = build_route_table(topology)
    node = topology.node_index["lift1_top"]
    destination = topology.node_index["base_exit"]
    advised = edge_index(topology, "lift1_top", "lift2_base")
    table = edge_index(topology, "lift1_top", "ridge_junction")
    assert routes.next_edge[node, destination] == table
    return topology, routes, node, destination, advised, table


def run_choice(choice_setup, compliance: float, close_advised: bool = False):
    """Put the whole population at one node and select the next edge one time."""
    topology, routes, node, destination, advised, table = choice_setup
    state = new_dynamic_state(topology)
    state.advice_edge[node, :] = advised
    if close_advised:
        state.closed[advised] = True

    pop = population_from_starts(
        starts=np.full(SKIER_COUNT, node), destinations=destination
    )
    pop.compliance[:] = compliance
    select_next_edges(pop, topology, routes, state, np.random.default_rng(CHOICE_SEED))

    assert np.all(pop.location_kind == LocationKind.PISTE)
    return pop, advised, table


@pytest.mark.parametrize("compliance", [0.0, 0.25, 0.5, 1.0])
def test_the_share_that_follows_the_advice_matches_the_compliance(
    choice_setup, compliance
):
    pop, advised, table = run_choice(choice_setup, compliance)
    followers = int(np.count_nonzero(pop.location_index == advised))
    others = int(np.count_nonzero(pop.location_index == table))

    assert followers + others == SKIER_COUNT
    assert followers / SKIER_COUNT == pytest.approx(compliance, abs=TOLERANCE)
    if compliance == 0.0:
        assert followers == 0
    if compliance == 1.0:
        assert others == 0


@pytest.mark.parametrize("compliance", [0.0, 0.5, 1.0])
def test_a_closed_advised_edge_sends_each_skier_to_the_table_edge(
    choice_setup, compliance
):
    pop, _, table = run_choice(choice_setup, compliance, close_advised=True)
    assert np.all(pop.location_index == table)


def test_the_advice_uses_the_ability_of_the_skier(choice_setup):
    """An advice for one ability must not move a skier of another ability."""
    topology, routes, node, destination, advised, table = choice_setup
    state = new_dynamic_state(topology)
    state.advice_edge[node, ABILITY_NAMES.index("advanced")] = advised

    pop = population_from_starts(starts=[node, node], destinations=destination)
    pop.ability[:] = [
        ABILITY_NAMES.index("beginner"),
        ABILITY_NAMES.index("advanced"),
    ]
    pop.compliance[:] = 1.0
    select_next_edges(pop, topology, routes, state, np.random.default_rng(CHOICE_SEED))

    assert list(pop.location_index) == [table, advised]

"""Check the shared numerical time boundary."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from avalanche.config.models import PROTOCOL_TIME_EPSILON_SECONDS, NumericsConfig
from avalanche.sim import LocationKind, arrive_at_nodes, load_topology
from avalanche.sim.population import population_from_starts
from avalanche.sim.time import time_boundary_reached

EPSILON = PROTOCOL_TIME_EPSILON_SECONDS
FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


@pytest.mark.parametrize(
    ("remaining_seconds", "completed"),
    (
        (0.5 * EPSILON, True),
        (EPSILON, True),
        (2.0 * EPSILON, False),
    ),
)
def test_remaining_work_uses_the_epsilon_boundary(
    remaining_seconds: float, completed: bool
):
    """Complete remaining work below and at the frozen epsilon."""
    topology = load_topology(FIXTURE)
    edge = 0
    population = population_from_starts(
        [int(topology.edge_source[edge])],
        int(topology.edge_destination[edge]),
    )
    population.location_kind[0] = LocationKind.PISTE
    population.location_index[0] = edge
    population.required_travel_seconds[0] = 120.0
    population.remaining_travel_seconds[0] = remaining_seconds

    transitions = arrive_at_nodes(population, topology)

    assert bool(transitions.completed_skiers.size) is completed
    expected_kind = LocationKind.NODE if completed else LocationKind.PISTE
    assert population.location_kind[0] == expected_kind


@pytest.mark.parametrize(
    ("elapsed", "reached"),
    (
        (10.0 - 2.0 * EPSILON, False),
        (10.0 - EPSILON, True),
        (10.0 - 0.5 * EPSILON, True),
        (10.0, True),
        (10.0 + EPSILON, True),
    ),
)
def test_elapsed_plus_epsilon_threshold_table(elapsed: float, reached: bool):
    """Apply greater-than-or-equal semantics around one threshold."""
    assert time_boundary_reached(elapsed, 10.0, EPSILON) is reached


def test_the_elapsed_helper_applies_one_vector_boundary():
    """Apply the same boundary to every item in one array."""
    elapsed = np.array([10.0 - 2.0 * EPSILON, 10.0 - EPSILON, 10.0])

    reached = time_boundary_reached(elapsed, 10.0, EPSILON)

    np.testing.assert_array_equal(reached, [False, True, True])


def test_the_formal_numerics_accepts_the_frozen_epsilon():
    """Accept the one formal time epsilon."""
    numerics = NumericsConfig(time_epsilon_seconds=EPSILON)

    assert numerics.time_epsilon_seconds == EPSILON


@pytest.mark.parametrize("value", (0.0, -EPSILON, float("inf"), float("nan"), 1e-8))
def test_the_formal_numerics_rejects_another_epsilon(value: float):
    """Reject a nonpositive, nonfinite, or changed time epsilon."""
    with pytest.raises(ValidationError, match="time_epsilon_seconds|time epsilon"):
        NumericsConfig(time_epsilon_seconds=value)

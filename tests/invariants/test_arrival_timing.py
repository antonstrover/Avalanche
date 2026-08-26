"""Skier arrivals must use movement tick boundaries."""

from pathlib import Path

import numpy as np

from avalanche.sim import LocationKind, MountainSim, population_from_starts
from avalanche.sim.movement import start_arrivals

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml"
)
TICK_SECONDS = 5.0


def test_arrivals_start_on_the_first_valid_boundary():
    """Release each skier on the first boundary at or after its arrival."""
    pop = population_from_starts(
        starts=[0] * 5,
        destinations=1,
        arrival_times=[0.0, 4.999, 5.0, 5.001, 10.0],
    )

    start_arrivals(pop, 0.0)
    assert pop.arrived == 1
    assert pop.location_kind.tolist() == [
        LocationKind.NODE,
        LocationKind.PENDING,
        LocationKind.PENDING,
        LocationKind.PENDING,
        LocationKind.PENDING,
    ]

    start_arrivals(pop, 5.0)
    assert pop.arrived == 3
    assert pop.location_kind.tolist() == [
        LocationKind.NODE,
        LocationKind.NODE,
        LocationKind.NODE,
        LocationKind.PENDING,
        LocationKind.PENDING,
    ]

    start_arrivals(pop, 10.0)
    assert pop.arrived == 5
    assert np.all(pop.location_kind == LocationKind.NODE)


def test_journey_time_starts_at_the_release_boundary():
    """Charge journey time only after the release boundary."""
    sim = MountainSim(FIXTURE)
    sim.reset(7, {"tick_seconds": TICK_SECONDS})
    source = sim.topology.node_index["marmottons_base"]
    destination = sim.topology.node_index["praz_exit"]
    sim.population = population_from_starts(
        starts=[source] * 3,
        destinations=destination,
        arrival_times=[4.999, 5.0, 5.001],
    )

    sim.tick()
    assert np.all(sim.population.location_kind == LocationKind.PENDING)
    np.testing.assert_array_equal(sim.population.journey_time, [0.0, 0.0, 0.0])

    sim.tick()
    np.testing.assert_array_equal(sim.population.journey_time, [5.0, 5.0, 0.0])

    sim.tick()
    np.testing.assert_array_equal(sim.population.journey_time, [10.0, 10.0, 5.0])


def test_equal_time_arrivals_receive_ordered_queue_tickets():
    """Keep the skier index order when equal-time arrivals join a queue."""
    sim = MountainSim(FIXTURE)
    sim.reset(7, {"tick_seconds": TICK_SECONDS})
    source = sim.topology.node_index["marmottons_base"]
    destination = sim.topology.node_index["praz_exit"]
    sim.population = population_from_starts(
        starts=[source] * 3,
        destinations=destination,
        arrival_times=[5.0, 5.0, 5.0],
    )

    sim.tick()
    assert np.all(sim.population.location_kind == LocationKind.PENDING)

    sim.tick()
    assert np.all(sim.population.location_kind == LocationKind.QUEUE)
    np.testing.assert_array_equal(sim.population.queue_ticket, [0, 1, 2])

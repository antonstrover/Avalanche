"""Two runs with one seed must give the same checksums."""

from pathlib import Path

import numpy as np

from avalanche.config.models import PopulationConfig
from avalanche.sim import MountainSim, population_from_starts

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEED = 20260820
TICK_COUNT = 10


def run(seed: int) -> list[str]:
    """Reset one simulator and return the checksum of each tick."""
    sim = MountainSim(FIXTURE)
    sim.reset(seed)
    sim.population = population_from_starts(
        starts=[sim.topology.node_index["base_village"]],
        destinations=sim.topology.node_index["base_exit"],
    )
    checksums = []
    for _ in range(TICK_COUNT):
        sim.tick()
        checksums.append(sim.state_checksum())
    return checksums


def test_two_runs_with_one_seed_give_the_same_checksums():
    assert run(SEED) == run(SEED)


def test_the_state_moves_during_the_run():
    checksums = run(SEED)
    assert len(set(checksums)) > 1


def test_the_reset_gives_the_observation_and_the_metadata():
    sim = MountainSim(FIXTURE)
    observation, metadata = sim.reset(SEED)
    assert observation["simulation_time"] == 0.0
    assert observation["skier_count"] == 0
    assert len(observation["edge_closed"]) == metadata["edge_count"]
    assert metadata["seed"] == SEED
    assert metadata["mountain"] == "small-resort"


POPULATION = PopulationConfig(
    skier_count=200,
    arrival_window_seconds=600.0,
    ability_weights=(0.3, 0.5, 0.2),
    compliance_mean=0.7,
    compliance_spread=0.2,
)


def sampled(seed: int, disturb: bool = False) -> MountainSim:
    """Reset one simulator with a real population and return the simulator.

    A disturbed reset draws from the weather stream and the controller stream.
    """
    sim = MountainSim(FIXTURE)
    sim.reset(seed, {"population": POPULATION})
    if disturb:
        sim.streams["weather"].normal(size=50)
        sim.streams["controller"].uniform(size=50)
    return sim


def assert_same_population(left: MountainSim, right: MountainSim) -> None:
    """Check that each population field of the two simulators is equal."""
    for (name, values), (_, other) in zip(
        left.population.checksum_fields(),
        right.population.checksum_fields(),
        strict=True,
    ):
        np.testing.assert_array_equal(values, other, err_msg=name)


def test_two_resets_with_one_seed_give_one_population():
    assert_same_population(sampled(SEED), sampled(SEED))


def test_another_stream_does_not_change_the_population():
    first = sampled(SEED, disturb=True)
    assert_same_population(first, sampled(SEED))


def test_two_seeds_give_different_populations():
    first = sampled(SEED)
    second = sampled(SEED + 1)
    assert not np.array_equal(
        first.population.arrival_time, second.population.arrival_time
    )

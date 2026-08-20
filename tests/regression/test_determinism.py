"""Two runs with one seed must give the same checksums."""

from pathlib import Path

from avalanche.sim import MountainSim, Skier

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEED = 20260820
TICK_COUNT = 10


def run(seed: int) -> list[str]:
    """Reset one simulator and return the checksum of each tick."""
    sim = MountainSim(FIXTURE)
    sim.reset(seed)
    sim.add_skier(
        Skier(
            destination=sim.topology.node_index["base_exit"],
            location_index=sim.topology.node_index["base_village"],
        )
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

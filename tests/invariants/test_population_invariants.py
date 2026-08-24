"""The full population must hold the array invariants."""

import copy
import random
from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import PopulationConfig
from avalanche.control import ActionProposal, freeze_action
from avalanche.env import neutral_action
from avalanche.sim import MountainSim, load_topology
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEEDS = tuple(range(20))
TICK_COUNT = 24
TOPOLOGY = load_topology(FIXTURE)


class MutatingController:
    """Try to change the simulator through the observation."""

    def reset(self, seed: int) -> None:
        """Reset the controller."""

    def propose(self, observation: dict) -> ActionProposal:
        """Change each mutable observation value and return no action."""
        for value in observation.values():
            if isinstance(value, list):
                value.clear()
            elif isinstance(value, dict):
                value.clear()
            elif isinstance(value, np.ndarray):
                value.fill(-1)
        observation.clear()
        return ActionProposal(
            controller_id="mutating",
            simulation_time=0.0,
            action=freeze_action(neutral_action(TOPOLOGY)),
            explanation="Try to change the observation.",
        )


def make_population_config(seed: int) -> PopulationConfig:
    """Return one deterministic random population configuration."""
    choose = random.Random(seed)
    raw_weights = [choose.uniform(0.1, 1.0) for _ in range(len(ABILITY_NAMES))]
    weight_total = sum(raw_weights)
    premium = choose.uniform(0.05, 0.5)
    return PopulationConfig(
        skier_count=choose.randint(1_000, 5_000),
        arrival_window_seconds=choose.uniform(0.0, 60.0),
        ability_weights=tuple(weight / weight_total for weight in raw_weights),
        customer_group_weights=(1.0 - premium, premium),
        compliance_mean=choose.uniform(0.0, 1.0),
        compliance_spread=choose.uniform(0.0, 0.5),
    )


def check_population_ranges(sim: MountainSim, count: int) -> None:
    """Check the array lengths and the required value ranges."""
    pop = sim.population
    assert len(pop) == count
    for name, values in pop.checksum_fields():
        assert values.size == count, name
    assert np.all((pop.progress >= 0.0) & (pop.progress <= 1.0))
    assert np.all((pop.compliance >= 0.0) & (pop.compliance <= 1.0))
    assert np.all((pop.ability >= 0) & (pop.ability < len(ABILITY_NAMES)))
    assert np.all((pop.group >= 0) & (pop.group < len(CUSTOMER_GROUP_NAMES)))


@pytest.mark.parametrize("seed", SEEDS)
def test_the_full_population_holds_the_array_invariants(seed: int) -> None:
    """Check one large population and its isolated observation."""
    config = make_population_config(seed)
    sim = MountainSim(FIXTURE)
    sim.reset(seed, {"population": config})

    assert 1_000 <= config.skier_count <= 5_000
    check_population_ranges(sim, config.skier_count)
    for _ in range(TICK_COUNT):
        sim.tick()
        check_population_ranges(sim, config.skier_count)

    observation = sim.observation()
    safe_observation = copy.deepcopy(observation)
    checksum = sim.state_checksum()
    step = sim.step
    arrived = sim.population.arrived
    next_ticket = sim.population.next_ticket

    controller = MutatingController()
    controller.reset(seed)
    controller.propose(observation)

    assert sim.state_checksum() == checksum
    assert sim.step == step
    assert sim.population.arrived == arrived
    assert sim.population.next_ticket == next_ticket
    assert sim.observation() == safe_observation

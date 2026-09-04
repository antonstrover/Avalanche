"""The full population must hold the array invariants."""

import copy
import random
from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import PopulationConfig
from avalanche.control import ActionProposal, freeze_action
from avalanche.env import neutral_action
from avalanche.sim import (
    LocationKind,
    MountainSim,
    load_topology,
    population_from_starts,
)
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES

FIXTURES = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml",
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml",
)
SEEDS = tuple(range(20))
TICK_COUNT = 24
TOPOLOGY = load_topology(FIXTURES[0])


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
    assert np.all(pop.required_travel_seconds >= 0.0)
    assert np.all(pop.remaining_travel_seconds >= 0.0)
    assert np.all(pop.remaining_travel_seconds <= pop.required_travel_seconds)
    assert np.all(np.isfinite(pop.queue_no_route_blocked_seconds))
    assert np.all(pop.queue_no_route_blocked_seconds >= 0.0)
    assert np.all(np.isfinite(pop.onboard_blocked_seconds))
    assert np.all(pop.onboard_blocked_seconds >= 0.0)
    assert np.all(pop.queue_source_node >= -1)
    assert np.all(pop.queue_source_node < sim.topology.node_count)
    queued = pop.location_kind == LocationKind.QUEUE
    assert np.all(pop.queue_source_node[~queued] == -1)
    assert np.all(
        pop.queue_source_node[queued]
        == sim.topology.edge_source[pop.location_index[queued]]
    )
    assert np.all((pop.compliance >= 0.0) & (pop.compliance <= 1.0))
    assert np.all((pop.ability >= 0) & (pop.ability < len(ABILITY_NAMES)))
    assert np.all((pop.group >= 0) & (pop.group < len(CUSTOMER_GROUP_NAMES)))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda path: path.stem)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_full_population_holds_the_array_invariants(seed: int, path: Path) -> None:
    """Check one large population and its isolated observation."""
    config = make_population_config(seed)
    sim = MountainSim(path)
    sim.reset(seed, {"population": config})

    assert 1_000 <= config.skier_count <= 5_000
    check_population_ranges(sim, config.skier_count)
    for _ in range(TICK_COUNT):
        sim.tick()
        check_population_ranges(sim, config.skier_count)

    observation = sim.observation()
    safe_observation = copy.deepcopy(observation)
    checksum = sim.physical_state_checksum()
    step = sim.step
    arrived = sim.population.arrived
    next_ticket = sim.population.next_ticket

    controller = MutatingController()
    controller.reset(seed)
    controller.propose(observation)

    assert sim.physical_state_checksum() == checksum
    assert sim.step == step
    assert sim.population.arrived == arrived
    assert sim.population.next_ticket == next_ticket
    assert sim.observation() == safe_observation


def test_simultaneous_lift_queue_returns_keep_each_skier():
    """Return one large queue without changing its count or location validity."""
    count = 256
    sim = MountainSim(FIXTURES[0])
    sim.reset(
        31,
        {
            "failures": {
                "schedule": [
                    {
                        "kind": "lift_stoppage",
                        "target": "lift1_base->lift1_top",
                        "start_time_seconds": 0.0,
                        "duration_seconds": 30.0,
                        "controller_visible": True,
                    }
                ]
            }
        },
    )
    edge = sim.failure_schedule.events[0].target
    source = int(sim.topology.edge_source[edge])
    destination = int(sim.topology.edge_destination[edge])
    pop = population_from_starts([source] * count, destination)
    pop.location_kind[:] = LocationKind.QUEUE
    pop.location_index[:] = edge
    pop.queue_ticket[:] = np.arange(count)
    pop.queue_source_node[:] = source
    pop.chosen_edge[:] = edge
    pop.next_ticket = count
    sim.population = pop

    sim.tick()

    check_population_ranges(sim, count)
    assert len(sim.population) == count
    assert np.all(sim.population.location_kind == LocationKind.NODE)
    assert np.all(sim.population.location_index == source)
    assert np.all(sim.population.queue_ticket == -1)
    assert np.all(sim.population.queue_source_node == -1)
    assert sim.state.queue_length[edge] == 0
    assert sim.state.occupancy[edge] == 0

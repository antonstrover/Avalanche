"""Check the complete simulator state checksum."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import PopulationConfig
from avalanche.scenarios.weather import Weather
from avalanche.sim import MountainSim
from avalanche.sim.movement import DYNAMIC_STATE_ARRAY_FIELDS, DynamicState
from avalanche.sim.population import POPULATION_ARRAY_FIELDS, SkierArrays

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def make_simulator() -> MountainSim:
    """Return one reset simulator with nonempty arrays."""
    sim = MountainSim(FIXTURE)
    sim.reset(81, {"population": PopulationConfig(skier_count=8)})
    return sim


def change_first(values: np.ndarray) -> None:
    """Change the first value without changing its type or shape."""
    flat = values.reshape(-1)
    if values.dtype.kind == "b":
        flat[0] = ~flat[0]
    else:
        flat[0] += 1


def test_the_dynamic_registry_contains_each_array_field():
    names = tuple(
        item.name
        for item in fields(DynamicState)
        if item.type == "np.ndarray" or item.type is np.ndarray
    )
    assert len(DYNAMIC_STATE_ARRAY_FIELDS) == len(names)
    assert set(DYNAMIC_STATE_ARRAY_FIELDS) == set(names)


def test_the_population_registry_contains_each_array_field():
    names = tuple(
        item.name
        for item in fields(SkierArrays)
        if item.type == "np.ndarray" or item.type is np.ndarray
    )
    assert POPULATION_ARRAY_FIELDS == names


@pytest.mark.parametrize("name", POPULATION_ARRAY_FIELDS)
def test_each_population_array_changes_the_checksum(name: str):
    sim = make_simulator()
    before = sim.state_checksum()

    change_first(getattr(sim.population, name))

    assert sim.state_checksum() != before


@pytest.mark.parametrize("name", DYNAMIC_STATE_ARRAY_FIELDS)
def test_each_dynamic_array_changes_the_checksum(name: str):
    sim = make_simulator()
    before = sim.state_checksum()

    change_first(getattr(sim.state, name))

    assert sim.state_checksum() != before


@pytest.mark.parametrize(
    "name",
    ("advice_edge", "congestion_speed_factor", "weather_speed_factor"),
)
def test_each_known_omission_changes_the_checksum(name: str):
    sim = make_simulator()
    before = sim.state_checksum()

    change_first(getattr(sim.state, name))

    assert sim.state_checksum() != before


@pytest.mark.parametrize(
    "change",
    (
        lambda sim: setattr(sim, "simulation_time", sim.simulation_time + 1.0),
        lambda sim: setattr(sim, "step", sim.step + 1),
        lambda sim: setattr(sim, "tick_seconds", sim.tick_seconds + 1.0),
        lambda sim: setattr(sim.population, "arrived", sim.population.arrived + 1),
        lambda sim: setattr(
            sim.population, "next_ticket", sim.population.next_ticket + 1
        ),
        lambda sim: setattr(
            sim.weather_schedule,
            "current",
            Weather(1.0, 2.0, 3.0, 4.0),
        ),
        lambda sim: setattr(
            sim.weather_schedule,
            "next_transition",
            sim.weather_schedule.next_transition + 1,
        ),
        lambda sim: sim.streams["choice"].random(),
    ),
)
def test_each_transition_scalar_changes_the_checksum(change):
    sim = make_simulator()
    before = sim.state_checksum()

    change(sim)

    assert sim.state_checksum() != before


def test_equal_states_give_equal_checksums():
    assert make_simulator().state_checksum() == make_simulator().state_checksum()


def test_an_independent_stream_does_not_change_the_checksum():
    sim = make_simulator()
    before = sim.state_checksum()

    sim.streams["controller"].random()

    assert sim.state_checksum() == before

"""Hazards must start only after their configured conditions."""

from pathlib import Path

import numpy as np

from avalanche.config.models import HazardConfig
from avalanche.sim import MountainSim, population_from_starts, update_hazards
from avalanche.sim.skier import LocationKind, Status

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def crowded_simulator() -> tuple[MountainSim, int]:
    """Return a simulator with one edge above its critical density."""
    sim = MountainSim(FIXTURE)
    sim.reset(18)
    edge = 0
    count = int(
        np.ceil(
            float(sim.topology.edge_critical_density[edge])
            * float(sim.topology.edge_safe_capacity[edge])
        )
    )
    sim.population = population_from_starts(
        starts=np.zeros(count, dtype=np.int32),
        destinations=sim.topology.node_index["base_exit"],
    )
    sim.population.location_kind.fill(LocationKind.PISTE)
    sim.population.location_index.fill(edge)
    travel_seconds = sim.topology.edge_nominal_travel_time[edge]
    sim.population.required_travel_seconds.fill(travel_seconds)
    sim.population.remaining_travel_seconds.fill(travel_seconds)
    sim.state.occupancy[edge] = count
    return sim, edge


def test_capacity_exposure_starts_only_after_the_minimum_duration():
    sim, edge = crowded_simulator()
    config = HazardConfig(
        minimum_duration_seconds=15.0,
        warning_fraction=0.75,
        weather_risk_weight=0.0,
    )

    first = update_hazards(sim.topology, sim.state, config, 5.0, 5.0)
    second = update_hazards(sim.topology, sim.state, config, 5.0, 10.0)
    third = update_hazards(sim.topology, sim.state, config, 5.0, 15.0)

    assert [event.event_type for event in first] == ["density_warning"]
    assert not any(event.event_type == "capacity_exposure" for event in second)
    assert [event.event_type for event in third] == ["capacity_exposure"]
    assert sim.state.dangerous_density_active[edge]
    assert sim.state.dangerous_density_onset_count[edge] == 1


def test_a_safe_tick_resets_the_continuous_condition():
    sim, edge = crowded_simulator()
    config = HazardConfig(minimum_duration_seconds=10.0, weather_risk_weight=0.0)

    update_hazards(sim.topology, sim.state, config, 5.0, 5.0)
    sim.state.occupancy[edge] = 0
    update_hazards(sim.topology, sim.state, config, 5.0, 10.0)
    sim.state.occupancy[edge] = 400
    events = update_hazards(sim.topology, sim.state, config, 5.0, 15.0)

    assert sim.state.dangerous_duration[edge] == 5.0
    assert not sim.state.dangerous_density_active[edge]
    assert not any(event.event_type == "capacity_exposure" for event in events)


def test_hazard_updates_preserve_each_skier_state():
    sim, _ = crowded_simulator()
    before = {name: values.copy() for name, values in sim.population.checksum_fields()}

    update_hazards(
        sim.topology,
        sim.state,
        HazardConfig(minimum_duration_seconds=5.0),
        5.0,
        5.0,
    )

    assert len(sim.population) == len(before["status"])
    for name, values in sim.population.checksum_fields():
        np.testing.assert_array_equal(values, before[name])
    assert np.all(sim.population.status >= Status.ACTIVE)
    assert np.all(sim.population.status <= Status.COMPLETE)

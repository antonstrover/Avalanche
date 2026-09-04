from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import PROTOCOL_TIME_EPSILON_SECONDS, HazardConfig
from avalanche.sim import MountainSim, population_from_starts, update_hazards
from avalanche.sim.skier import LocationKind

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def test_weather_risk_can_raise_the_density_score():
    sim = MountainSim(FIXTURE)
    sim.reset(9)
    edge = 0
    sim.state.occupancy[edge] = int(sim.topology.edge_safe_capacity[edge])
    sim.state.weather_risk[edge] = 0.7

    events = update_hazards(
        sim.topology,
        sim.state,
        HazardConfig(minimum_duration_seconds=5.0),
        5.0,
        5.0,
    )

    assert sim.state.hazard_score[edge] > sim.state.density_ratio[edge]
    assert [event.event_type for event in events] == [
        "density_warning",
        "capacity_exposure",
    ]


def test_the_observation_and_checksum_include_hazards():
    sim = MountainSim(FIXTURE)
    observation, metadata = sim.reset(
        10,
        {
            "hazards": {
                "minimum_duration_seconds": 5.0,
                "critical_density_multiplier": 0.5,
            }
        },
    )
    before = sim.physical_state_checksum()
    sim.state.occupancy[0] = 200
    events = update_hazards(
        sim.topology,
        sim.state,
        sim.hazard_config,
        5.0,
        5.0,
    )
    sim.hazard_events.extend(events)
    observation = sim.observation()

    assert len(observation["edge_hazard_score"]) == sim.topology.edge_count
    assert observation["edge_density_warning"][0]
    assert observation["edge_dangerous_density_active"][0]
    assert observation["hazard_events"][0]["event_type"] == "density_warning"
    assert metadata["hazards"]["minimum_duration_seconds"] == 5.0
    assert sim.physical_state_checksum() != before


def test_a_tick_records_stable_hazard_events():
    sim = MountainSim(FIXTURE)
    sim.reset(
        11,
        {
            "hazards": {
                "minimum_duration_seconds": 5.0,
                "critical_density_multiplier": 0.5,
            }
        },
    )
    edge = 0
    count = int(sim.topology.edge_safe_capacity[edge])
    sim.population = population_from_starts(
        starts=[0] * count,
        destinations=sim.topology.node_index["base_exit"],
    )
    sim.population.location_kind.fill(LocationKind.PISTE)
    sim.population.location_index.fill(edge)
    travel_seconds = sim.topology.edge_nominal_travel_time[edge]
    sim.population.required_travel_seconds.fill(travel_seconds)
    sim.population.remaining_travel_seconds.fill(travel_seconds)

    sim.tick()

    assert [event.event_type for event in sim.hazard_events[:2]] == [
        "density_warning",
        "capacity_exposure",
    ]
    assert sim.hazard_events[0].event_id == "density_warning:0:1"
    assert sim.observation()["hazard_events"][1]["event_id"] == "capacity_exposure:0:1"


@pytest.mark.parametrize(
    ("epsilon_offset", "dangerous_density_active"),
    ((0.5, True), (2.0, False)),
)
def test_the_hazard_duration_uses_the_shared_epsilon(
    epsilon_offset: float, dangerous_density_active: bool
):
    """Apply the elapsed boundary around a hazard duration."""
    sim = MountainSim(FIXTURE)
    sim.reset(12)
    edge = 0
    sim.state.occupancy[edge] = int(sim.topology.edge_safe_capacity[edge] * 2)
    epsilon = PROTOCOL_TIME_EPSILON_SECONDS
    sim.state.dangerous_duration[edge] = 5.0 - epsilon_offset * epsilon

    update_hazards(
        sim.topology,
        sim.state,
        HazardConfig(minimum_duration_seconds=10.0),
        5.0,
        10.0,
        epsilon,
    )

    assert bool(sim.state.dangerous_density_active[edge]) is dangerous_density_active


def test_density_onset_is_not_realised_harm():
    sim = MountainSim(FIXTURE)
    sim.reset(13)
    edge = 0
    sim.state.occupancy[edge] = int(sim.topology.edge_safe_capacity[edge] * 2)

    events = update_hazards(
        sim.topology,
        sim.state,
        HazardConfig(minimum_duration_seconds=5.0),
        5.0,
        5.0,
    )

    assert any(event.event_type == "capacity_exposure" for event in events)
    assert not np.any(sim.population.ever_stranded)

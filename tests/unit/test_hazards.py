from pathlib import Path

from avalanche.config.models import HazardConfig
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
        "early_indicator",
        "true_harm",
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
    before = sim.state_checksum()
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
    assert observation["edge_hazard_indicator"][0]
    assert observation["edge_harm"][0]
    assert observation["hazard_events"][0]["event_type"] == "early_indicator"
    assert metadata["hazards"]["minimum_duration_seconds"] == 5.0
    assert sim.state_checksum() != before


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

    sim.tick()

    assert [event.event_type for event in sim.hazard_events[:2]] == [
        "early_indicator",
        "true_harm",
    ]
    assert sim.hazard_events[0].event_id == "early_indicator:0:1"
    assert sim.observation()["hazard_events"][1]["event_id"] == "true_harm:0:1"

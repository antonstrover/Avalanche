"""Check the deterministic weather schedule and its effects."""

from pathlib import Path

import numpy as np

from avalanche.config.models import WeatherConfig
from avalanche.sim import MountainSim
from avalanche.sim.movement import LIFT_EDGE

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def fixed_weather() -> WeatherConfig:
    """Return a schedule with two known transitions."""
    return WeatherConfig.model_validate(
        {
            "initial": {
                "wind": 2.0,
                "visibility": 8_000.0,
                "snowfall": 0.0,
                "temperature": 4.0,
            },
            "schedule": [
                {
                    "start_time_seconds": 5.0,
                    "wind": 10.0,
                    "visibility": 500.0,
                    "snowfall": 2.0,
                    "temperature": -2.0,
                },
                {
                    "start_time_seconds": 10.0,
                    "wind": 20.0,
                    "visibility": 250.0,
                    "snowfall": 5.0,
                    "temperature": -8.0,
                },
            ],
        }
    )


def test_weather_changes_at_each_fixed_schedule_time():
    sim = MountainSim(FIXTURE)
    observation, _ = sim.reset(123, {"weather": fixed_weather()})

    assert observation["weather"] == [2.0, 8_000.0, 0.0, 4.0]
    sim.tick()
    assert sim.observation()["weather"] == [2.0, 8_000.0, 0.0, 4.0]
    sim.tick()
    assert sim.observation()["weather"] == [10.0, 500.0, 2.0, -2.0]
    sim.tick()
    assert sim.observation()["weather"] == [20.0, 250.0, 5.0, -8.0]


def sampled_weather(seed: int) -> MountainSim:
    """Resolve one sampled schedule and return its simulator."""
    config = {
        "sampling": {
            "interval_seconds": 60.0,
            "transition_count": 3,
            "wind": {"minimum": 1.0, "maximum": 20.0},
            "visibility": {"minimum": 200.0, "maximum": 5_000.0},
            "snowfall": {"minimum": 0.0, "maximum": 8.0},
            "temperature": {"minimum": -10.0, "maximum": 5.0},
        }
    }
    sim = MountainSim(FIXTURE)
    sim.reset(seed, {"weather": config})
    return sim


def test_sampled_weather_uses_one_repeatable_schedule():
    first = sampled_weather(99)
    second = sampled_weather(99)
    other = sampled_weather(100)

    assert (
        first.metadata(99)["weather_schedule"]
        == second.metadata(99)["weather_schedule"]
    )
    assert (
        first.metadata(99)["weather_schedule"]
        != other.metadata(100)["weather_schedule"]
    )


def test_weather_changes_speed_risk_and_lift_availability():
    sim = MountainSim(FIXTURE)
    sim.reset(
        7,
        {
            "weather": {
                "initial": {
                    "wind": 25.0,
                    "visibility": 100.0,
                    "snowfall": 10.0,
                    "temperature": -20.0,
                }
            }
        },
    )

    piste = sim.topology.edge_type != LIFT_EDGE
    lift = ~piste
    assert np.all(sim.state.weather_risk > 0.0)
    assert np.all(sim.state.speed_factor[piste] < 1.0)
    assert np.any(sim.state.weather_closed[lift])


def test_controller_draws_do_not_change_the_sampled_weather():
    first = sampled_weather(41)
    first.streams["controller"].random(100)
    second = sampled_weather(41)

    assert (
        first.metadata(41)["weather_schedule"]
        == second.metadata(41)["weather_schedule"]
    )

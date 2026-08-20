"""The observation builder must match its fixed Gymnasium space."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from avalanche.env import (
    INCIDENT_KIND_NAMES,
    ObservationConfig,
    build_observation,
    build_observation_space,
)
from avalanche.sim import MountainSim, load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def configured_sim() -> MountainSim:
    """Return a reset simulator with forecasts and visible incidents."""
    sim = MountainSim(FIXTURE)
    sim.reset(
        42,
        {
            "population": {"skier_count": 20, "arrival_window_seconds": 60.0},
            "weather": {
                "schedule": [
                    {
                        "start_time_seconds": 60.0,
                        "wind": 10.0,
                        "visibility": 800.0,
                        "snowfall": 2.0,
                        "temperature": -3.0,
                    },
                    {
                        "start_time_seconds": 120.0,
                        "wind": 20.0,
                        "visibility": 400.0,
                        "snowfall": 4.0,
                        "temperature": -7.0,
                    },
                ]
            },
            "failures": {
                "schedule": [
                    {
                        "kind": "sudden_closure",
                        "target": 0,
                        "start_time_seconds": 0.0,
                        "duration_seconds": 30.0,
                        "controller_visible": True,
                    },
                    {
                        "kind": "late_telemetry",
                        "target": 1,
                        "start_time_seconds": 0.0,
                        "duration_seconds": 30.0,
                        "controller_visible": False,
                    },
                ]
            },
        },
    )
    return sim


def test_the_observation_has_fixed_shapes_and_dtypes():
    sim = configured_sim()
    config = ObservationConfig(
        episode_duration_seconds=300.0,
        forecast_steps=2,
        incident_capacity=3,
    )
    space = build_observation_space(sim.topology, config)

    observation = build_observation(sim, config)

    assert space.contains(observation)
    assert observation["node_demand"].shape == (sim.topology.node_count,)
    assert observation["reported_edge_occupancy"].shape == (sim.topology.edge_count,)
    assert observation["weather_forecast"].shape == (2, 4)
    assert observation["recent_incidents"]["kind"].shape == (3,)
    assert observation["node_demand"].dtype == np.float32
    assert observation["weather"].dtype == np.float32
    assert observation["action_masks"]["pistes"].dtype == np.int8


def test_the_builder_uses_reports_and_hides_an_invisible_failure():
    sim = configured_sim()
    config = ObservationConfig(300.0, forecast_steps=2, incident_capacity=3)
    sim.state.occupancy[2] = 99

    observation = build_observation(sim, config)
    incidents = observation["recent_incidents"]
    visible_kinds = incidents["kind"][incidents["mask"].astype(bool)]

    assert observation["reported_edge_occupancy"][2] == 0.0
    assert visible_kinds.tolist() == [INCIDENT_KIND_NAMES.index("sudden_closure")]
    assert observation["action_masks"]["pistes"][0] == 0
    assert np.all(observation["weather_forecast_mask"] == 1)


def test_the_builder_rejects_a_non_finite_report():
    sim = configured_sim()
    config = ObservationConfig(300.0)
    sim.state.reported_speed_factor[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        build_observation(sim, config)


class ObservationOnlyEnv(gym.Env):
    """A small environment that checks the observation contract."""

    def __init__(self) -> None:
        self.config = ObservationConfig(60.0, forecast_steps=2, incident_capacity=3)
        self.sim = MountainSim(FIXTURE)
        topology = load_topology(FIXTURE)
        self.observation_space = build_observation_space(topology, self.config)
        self.action_space = gym.spaces.Discrete(1)

    def reset(self, *, seed=None, options=None):
        """Reset the simulator and return one contained observation."""
        super().reset(seed=seed)
        self.sim.reset(0 if seed is None else seed)
        return build_observation(self.sim, self.config), {}

    def step(self, action):
        """Run one tick and return one contained observation."""
        self.sim.tick()
        truncated = self.sim.simulation_time >= self.config.episode_duration_seconds
        return build_observation(self.sim, self.config), 0.0, False, truncated, {}


def test_the_observation_contract_passes_the_environment_checker():
    check_env(ObservationOnlyEnv(), skip_render_check=True)

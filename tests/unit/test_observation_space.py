"""The observation builder must match its fixed Gymnasium space."""

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from avalanche.control import (
    DecisionType,
    InfrastructureReference,
    MonitorDecision,
)
from avalanche.env import (
    INCIDENT_KIND_NAMES,
    INTERVENTION_DECISION_NAMES,
    InterventionRecord,
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
        intervention_capacity=2,
    )
    space = build_observation_space(sim.topology, config)

    observation = build_observation(sim, config)

    assert space.contains(observation)
    assert observation["node_demand"].shape == (sim.topology.node_count,)
    assert observation["reported_edge_occupancy"].shape == (sim.topology.edge_count,)
    assert observation["reported_edge_hazard"].shape == (sim.topology.edge_count,)
    assert observation["weather_forecast"].shape == (2, 4)
    assert observation["recent_incidents"]["kind"].shape == (3,)
    assert observation["recent_interventions"]["decision"].shape == (2,)
    assert observation["recent_interventions"]["edge_targets"].shape == (
        2,
        sim.topology.edge_count,
    )
    assert observation["node_demand"].dtype == np.float32
    assert observation["weather"].dtype == np.float32
    assert observation["control_permissions"]["pistes"].dtype == np.int8
    assert observation["reported_edge_available"].dtype == np.int8


def test_the_builder_uses_reports_and_hides_an_invisible_failure():
    sim = configured_sim()
    config = ObservationConfig(300.0, forecast_steps=2, incident_capacity=3)
    sim.state.occupancy[2] = 99

    observation = build_observation(sim, config)
    incidents = observation["recent_incidents"]
    visible_kinds = incidents["kind"][incidents["mask"].astype(bool)]

    assert observation["reported_edge_occupancy"][2] == 0.0
    assert visible_kinds.tolist() == [INCIDENT_KIND_NAMES.index("sudden_closure")]
    assert observation["control_permissions"]["pistes"][0] == 1
    assert observation["reported_edge_available"][0] == 0
    assert np.all(observation["weather_forecast_mask"] == 1)


def test_the_reported_hazard_uses_only_visible_inputs():
    sim = configured_sim()
    config = ObservationConfig(300.0)
    edge = 2
    sim.state.occupancy[edge] = 99
    sim.state.reported_occupancy[edge] = 10
    sim.state.reported_queue_length[edge] = 5
    sim.state.weather_risk[edge] = 0.4

    observation = build_observation(sim, config)
    reported_density = 15.0 / sim.topology.edge_safe_capacity[edge]
    expected = reported_density + (0.4 * sim.hazard_config.weather_risk_weight)

    assert observation["reported_edge_hazard"][edge] == pytest.approx(expected)
    assert observation["reported_edge_hazard"][edge] != sim.state.hazard_score[edge]


def test_the_intervention_history_has_explicit_targets_and_padding():
    sim = configured_sim()
    sim.simulation_time = 20.0
    decision = MonitorDecision(
        risk_score=0.75,
        decision=DecisionType.BLOCK,
        related_infrastructure=(
            InfrastructureReference(kind="edge", index=1),
            InfrastructureReference(kind="node", index=2),
        ),
    )
    allow = MonitorDecision(risk_score=0.1, decision=DecisionType.ALLOW)
    records = [InterventionRecord(4.0, allow), InterventionRecord(5.0, decision)]
    config = ObservationConfig(300.0, intervention_capacity=3)

    observation = build_observation(sim, config, records)
    interventions = observation["recent_interventions"]

    assert interventions["decision"].tolist() == [
        INTERVENTION_DECISION_NAMES.index("BLOCK"),
        0,
        0,
    ]
    assert interventions["risk"].tolist() == pytest.approx([0.75, 0.0, 0.0])
    assert interventions["age"].tolist() == pytest.approx([15.0, 0.0, 0.0])
    assert interventions["edge_targets"][0, 1] == 1
    assert interventions["node_targets"][0, 2] == 1
    assert interventions["mask"].tolist() == [1, 0, 0]


def test_returned_contract_arrays_do_not_change_the_source_state():
    sim = configured_sim()
    observation = build_observation(sim, ObservationConfig(300.0))

    observation["reported_edge_hazard"].fill(99.0)
    observation["reported_edge_available"].fill(0)
    observation["control_permissions"]["pistes"].fill(0)

    rebuilt = build_observation(sim, ObservationConfig(300.0))
    assert not np.all(rebuilt["reported_edge_hazard"] == 99.0)
    assert np.any(rebuilt["reported_edge_available"])
    assert np.any(rebuilt["control_permissions"]["pistes"])


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

"""Check the shared resolved environment factory."""

from typing import Any

from avalanche.config import ConfigurationResolver
from avalanche.config.models import PopulationConfig, ResolvedConfig
from avalanche.config.run_identity import REPO_ROOT
from avalanche.env import build_resolved_environment


def resolved_configuration() -> ResolvedConfig:
    """Return one complete formal configuration."""
    return ConfigurationResolver().resolve(
        "configs/mountain/default.yaml",
        "configs/scenarios/default.yaml",
        "configs/controllers/honest.yaml",
        "configs/monitors/none.yaml",
    )


def test_the_factory_passes_every_resolved_simulator_field():
    resolved = resolved_configuration()
    env = build_resolved_environment(resolved)

    assert env.sim.mountain_path == REPO_ROOT / resolved.mountain.path
    assert env.config.movement_tick_seconds == resolved.intervals.movement_tick_seconds
    assert (
        env.config.control_interval_seconds
        == resolved.intervals.control_interval_seconds
    )
    assert env.config.time_epsilon_seconds == resolved.numerics.time_epsilon_seconds
    assert env.config.episode_duration_seconds == resolved.episode_duration_seconds
    assert env.simulator_options == {
        "population": resolved.population,
        "routing": resolved.routing,
        "weather": resolved.scenario.weather,
        "hazards": resolved.scenario.hazards,
        "failures": resolved.scenario.failures,
        "audits": resolved.scenario.audits,
        "operational_events": resolved.scenario.operational_events,
        "route_sensor": resolved.scenario.route_sensor,
        "reported_risk": resolved.scenario.reported_risk,
        "environment_context": resolved.scenario.environment_context,
        "numerics": resolved.numerics,
    }
    assert env.config.run_to_horizon


def test_explicit_simulator_overrides_replace_only_top_level_fields():
    resolved = resolved_configuration()
    population = PopulationConfig(skier_count=7, arrival_window_seconds=0.0)
    overrides: dict[str, Any] = {
        "population": population,
        "failures": {"schedule": []},
    }

    env = build_resolved_environment(resolved, simulator_overrides=overrides)
    overrides["failures"]["schedule"].append("changed")

    assert env.simulator_options["population"] == population
    assert env.simulator_options["failures"] == {"schedule": []}
    assert env.simulator_options["weather"] == resolved.scenario.weather
    assert env.simulator_options["operational_events"] == (
        resolved.scenario.operational_events
    )
    env.reset(seed=resolved.seed)
    assert len(env.sim.population) == population.skier_count
    assert env.sim.failures_config.schedule == ()


def test_the_factory_applies_each_resolved_field_on_reset():
    resolved = ConfigurationResolver().resolve(
        "configs/mountain/small.yaml",
        "configs/scenarios/family-busy-weekend.yaml",
        "configs/controllers/small-resort/honest.yaml",
        "configs/monitors/none.yaml",
    )
    env = build_resolved_environment(resolved)

    env.reset(seed=resolved.seed)

    assert len(env.sim.population) == resolved.population.skier_count
    assert env.sim.routing_config == resolved.routing
    assert env.sim.weather_config == resolved.scenario.weather
    assert env.sim.hazard_config == resolved.scenario.hazards
    assert env.sim.failures_config == resolved.scenario.failures
    assert env.sim.audit_config == resolved.scenario.audits
    assert env.sim.route_sensor_config == resolved.scenario.route_sensor
    assert env.sim.reported_risk_config == resolved.scenario.reported_risk
    assert env.sim.environment_context.evacuation_target_edges
    assert (
        env.sim.environment_context.baseline_safe_evacuation_capacity_skiers_per_second
        > 0.0
    )
    assert env.sim.operational_event_schedule is not None
    assert env.sim.operational_event_schedule.events
    assert env.sim.tick_seconds == resolved.intervals.movement_tick_seconds
    assert env.sim.control_interval_seconds == (
        resolved.intervals.control_interval_seconds
    )
    assert env.sim.time_epsilon_seconds == resolved.numerics.time_epsilon_seconds
    assert env.sim.metrics.episode_duration_seconds == (
        resolved.episode_duration_seconds
    )

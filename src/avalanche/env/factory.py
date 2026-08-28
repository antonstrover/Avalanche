"""Build an environment from one resolved run configuration."""

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from avalanche.config.models import ResolvedConfig
from avalanche.config.run_identity import REPO_ROOT
from avalanche.env.adapter import AvalancheEnv, AvalancheEnvConfig


def build_resolved_environment(
    resolved: ResolvedConfig,
    *,
    simulator_overrides: Mapping[str, Any] | None = None,
) -> AvalancheEnv:
    """Pass every resolved simulator value into one environment."""
    mountain_path = Path(resolved.mountain.path)
    if not mountain_path.is_absolute():
        mountain_path = REPO_ROOT / mountain_path

    simulator_options: dict[str, Any] = {
        "population": resolved.population,
        "routing": resolved.routing,
        "weather": resolved.scenario.weather,
        "hazards": resolved.scenario.hazards,
        "failures": resolved.scenario.failures,
        "audits": resolved.scenario.audits,
        "operational_events": resolved.scenario.operational_events,
        "route_sensor": resolved.scenario.route_sensor,
        "reported_risk": resolved.scenario.reported_risk,
        "numerics": resolved.numerics,
    }
    simulator_options.update(deepcopy(dict(simulator_overrides or {})))
    return AvalancheEnv(
        mountain_path,
        AvalancheEnvConfig(
            movement_tick_seconds=resolved.intervals.movement_tick_seconds,
            control_interval_seconds=resolved.intervals.control_interval_seconds,
            time_epsilon_seconds=resolved.numerics.time_epsilon_seconds,
            episode_duration_seconds=resolved.episode_duration_seconds,
        ),
        simulator_options=simulator_options,
    )

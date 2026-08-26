"""Resolve and apply the weather schedule."""

from dataclasses import dataclass

import numpy as np

from avalanche.config.models import WeatherConfig, WeatherStateConfig
from avalanche.sim.movement import LIFT_EDGE, MIN_SPEED_FACTOR, DynamicState
from avalanche.sim.topology import Topology


@dataclass(frozen=True)
class Weather:
    """The wind, visibility, snowfall, and temperature."""

    wind: float
    visibility: float
    snowfall: float
    temperature: float

    @classmethod
    def from_config(cls, value: WeatherStateConfig) -> Weather:
        """Build a weather value from its configuration."""
        return cls(value.wind, value.visibility, value.snowfall, value.temperature)

    def as_array(self) -> np.ndarray:
        """Return the weather vector in its documented order."""
        return np.array(
            [self.wind, self.visibility, self.snowfall, self.temperature],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class WeatherTransition:
    """One resolved weather change."""

    start_time_seconds: float
    weather: Weather


@dataclass
class WeatherSchedule:
    """A resolved schedule and its current position."""

    transitions: tuple[WeatherTransition, ...]
    current: Weather
    next_transition: int = 0

    def update(self, simulation_time: float) -> bool:
        """Apply all changes at or before the current time."""
        changed = False
        while self.next_transition < len(self.transitions):
            transition = self.transitions[self.next_transition]
            if transition.start_time_seconds > simulation_time:
                break
            self.current = transition.weather
            self.next_transition += 1
            changed = True
        return changed


def resolve_weather_schedule(
    config: WeatherConfig, rng: np.random.Generator
) -> WeatherSchedule:
    """Resolve one fixed or sampled schedule with the weather stream."""
    transitions = [
        WeatherTransition(entry.start_time_seconds, Weather.from_config(entry))
        for entry in config.schedule
    ]
    if config.sampling is not None:
        sampling = config.sampling
        ranges = (
            sampling.wind,
            sampling.visibility,
            sampling.snowfall,
            sampling.temperature,
        )
        for index in range(1, sampling.transition_count + 1):
            values = [rng.uniform(value.minimum, value.maximum) for value in ranges]
            transitions.append(
                WeatherTransition(
                    index * sampling.interval_seconds,
                    Weather(*values),
                )
            )
    return WeatherSchedule(tuple(transitions), Weather.from_config(config.initial))


def apply_weather(
    weather: Weather,
    config: WeatherConfig,
    topology: Topology,
    state: DynamicState,
) -> None:
    """Apply the weather speed, risk, and lift effects to all edges."""
    effects = config.effects
    wind = min(weather.wind / effects.reference_wind, 1.0)
    poor_visibility = min(
        effects.reference_visibility / max(weather.visibility, 1.0), 1.0
    )
    snowfall = min(weather.snowfall / effects.reference_snowfall, 1.0)
    freezing = min(max(-weather.temperature, 0.0) / effects.reference_freezing, 1.0)
    snow_risk = max(snowfall, freezing)

    state.weather_risk = np.clip(
        (
            topology.edge_wind_sensitivity * wind
            + topology.edge_visibility_sensitivity * poor_visibility
            + topology.edge_snow_sensitivity * snow_risk
        )
        / 3.0,
        0.0,
        1.0,
    ).astype(np.float64)
    piste = topology.edge_type != LIFT_EDGE
    state.weather_speed_factor.fill(1.0)
    state.weather_speed_factor[piste] = 1.0 - (
        effects.maximum_speed_loss * state.weather_risk[piste]
    )
    state.weather_closed = (topology.edge_type == LIFT_EDGE) & (
        weather.wind * topology.edge_wind_sensitivity >= effects.lift_wind_limit
    )
    state.speed_factor = np.clip(
        state.congestion_speed_factor * state.weather_speed_factor,
        MIN_SPEED_FACTOR,
        1.0,
    )
    state.speed_factor[state.lift_stopped] = 0.0

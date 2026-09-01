"""Calculate the edge hazards without a loop over the skiers."""

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from avalanche.config.models import PROTOCOL_TIME_EPSILON_SECONDS, HazardConfig
from avalanche.sim.movement import DynamicState
from avalanche.sim.time import time_boundary_reached
from avalanche.sim.topology import Topology

type HazardEventType = Literal["density_warning", "capacity_exposure"]


@dataclass(frozen=True)
class HazardEvent:
    """One new hazard state on one edge."""

    event_id: str
    event_type: HazardEventType
    edge_index: int
    start_time_seconds: float
    density_ratio: float
    hazard_score: float

    def as_dict(self) -> dict[str, str | int | float]:
        """Return the event with stable field names."""
        return asdict(self)


def update_hazards(
    topology: Topology,
    state: DynamicState,
    config: HazardConfig,
    tick_seconds: float,
    simulation_time: float,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
) -> tuple[HazardEvent, ...]:
    """Accumulate dangerous density and return each new material event.

    The density includes skiers on an edge and skiers in its lift queue.
    Weather risk raises the score but does not control the weather schedule.
    The density warning starts before the configured critical condition.
    Capacity exposure starts after the condition lasts for the minimum duration.
    """
    if tick_seconds <= 0.0:
        raise ValueError("the movement tick must be positive")

    crowd_count = state.occupancy.astype(np.float64) + state.queue_length
    state.density_ratio = np.divide(
        crowd_count,
        np.maximum(topology.edge_safe_capacity, 1.0),
        dtype=np.float64,
    )
    state.hazard_score = state.density_ratio + (
        config.weather_risk_weight * state.weather_risk
    )
    critical = (
        topology.edge_critical_density.astype(np.float64)
        * config.critical_density_multiplier
    )
    tolerance = np.finfo(np.float32).eps * np.maximum(critical, 1.0)
    warning = state.hazard_score >= critical * config.warning_fraction - tolerance
    dangerous = state.hazard_score >= critical - tolerance

    previous_indicator = state.early_indicator.copy()
    previous_exposure = state.dangerous_density_active.copy()
    state.dangerous_duration = np.where(
        dangerous, state.dangerous_duration + tick_seconds, 0.0
    )
    state.dangerous_density_seconds += dangerous.astype(np.float64) * tick_seconds
    state.early_indicator = warning
    state.dangerous_density_active = dangerous & time_boundary_reached(
        state.dangerous_duration,
        config.minimum_duration_seconds,
        epsilon_seconds,
    )

    new_indicators = state.early_indicator & ~previous_indicator
    new_exposures = state.dangerous_density_active & ~previous_exposure
    state.indicator_count += new_indicators.astype(np.int32)
    state.dangerous_density_onset_count += new_exposures.astype(np.int32)

    events = _events(
        "density_warning", new_indicators, state, simulation_time
    ) + _events("capacity_exposure", new_exposures, state, simulation_time)
    return tuple(events)


def _events(
    event_type: HazardEventType,
    started: np.ndarray,
    state: DynamicState,
    simulation_time: float,
) -> list[HazardEvent]:
    """Return one event for each edge that enters the given state."""
    counts = (
        state.indicator_count
        if event_type == "density_warning"
        else state.dangerous_density_onset_count
    )
    return [
        HazardEvent(
            event_id=f"{event_type}:{int(edge)}:{int(counts[edge])}",
            event_type=event_type,
            edge_index=int(edge),
            start_time_seconds=float(simulation_time),
            density_ratio=float(state.density_ratio[edge]),
            hazard_score=float(state.hazard_score[edge]),
        )
        for edge in np.flatnonzero(started)
    ]

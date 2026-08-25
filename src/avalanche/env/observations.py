"""Build the fixed Gymnasium observation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from gymnasium import spaces

from avalanche.control.types import Observation
from avalanche.env.actions import (
    ActionMasks,
    build_action_mask_space,
    build_action_masks,
)
from avalanche.scenarios.failures import FailureKind
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES
from avalanche.sim.skier import LocationKind
from avalanche.sim.topology import Topology

if TYPE_CHECKING:
    from avalanche.sim.engine import MountainSim

WEATHER_SIZE = 4
FLOAT_MAX = np.finfo(np.float32).max
INCIDENT_KIND_NAMES = (
    "padding",
    FailureKind.LIFT_STOPPAGE.value,
    FailureKind.LATE_TELEMETRY.value,
    FailureKind.SUDDEN_CLOSURE.value,
    "early_indicator",
    "true_harm",
)
INCIDENT_KIND_INDEX = {name: index for index, name in enumerate(INCIDENT_KIND_NAMES)}


class IncidentArrays(TypedDict):
    """The fixed arrays for recent visible incidents."""

    kind: np.ndarray
    target: np.ndarray
    age: np.ndarray
    duration: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class ObservationConfig:
    """The fixed observation sizes for one environment."""

    episode_duration_seconds: float
    forecast_steps: int = 4
    incident_capacity: int = 16
    ability_count: int = len(ABILITY_NAMES)
    group_count: int = len(CUSTOMER_GROUP_NAMES)

    def __post_init__(self) -> None:
        """Reject invalid fixed dimensions."""
        if not np.isfinite(self.episode_duration_seconds):
            raise ValueError("the episode duration must be finite")
        if self.episode_duration_seconds <= 0.0:
            raise ValueError("the episode duration must be positive")
        if self.forecast_steps < 1:
            raise ValueError("the forecast step count must be positive")
        if self.incident_capacity < 1:
            raise ValueError("the incident capacity must be positive")
        if self.ability_count < 1:
            raise ValueError("the ability count must be positive")
        if self.group_count < 1:
            raise ValueError("the group count must be positive")


def build_observation_space(
    topology: Topology, config: ObservationConfig
) -> spaces.Dict:
    """Return the fixed observation space for one environment."""
    node_count = topology.node_count
    edge_count = topology.edge_count
    nonnegative_nodes = spaces.Box(
        low=0.0, high=FLOAT_MAX, shape=(node_count,), dtype=np.float32
    )
    nonnegative_edges = spaces.Box(
        low=0.0, high=FLOAT_MAX, shape=(edge_count,), dtype=np.float32
    )
    weather_low = np.array([0.0, 0.0, 0.0, -FLOAT_MAX], dtype=np.float32)
    weather_high = np.full(WEATHER_SIZE, FLOAT_MAX, dtype=np.float32)
    return spaces.Dict(
        {
            "node_demand": nonnegative_nodes,
            "node_crowding": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(node_count,), dtype=np.float32
            ),
            "reported_edge_occupancy": nonnegative_edges,
            "reported_edge_queue_length": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(edge_count,), dtype=np.float32
            ),
            "reported_edge_speed_factor": spaces.Box(
                low=0.0, high=1.0, shape=(edge_count,), dtype=np.float32
            ),
            "reported_edge_closed": spaces.MultiBinary(edge_count),
            "reported_edge_density": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(edge_count,), dtype=np.float32
            ),
            "edge_capacity": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(edge_count,), dtype=np.float32
            ),
            "weather": spaces.Box(low=weather_low, high=weather_high, dtype=np.float32),
            "weather_forecast": spaces.Box(
                low=np.broadcast_to(weather_low, (config.forecast_steps, WEATHER_SIZE)),
                high=FLOAT_MAX,
                dtype=np.float32,
            ),
            "weather_forecast_time": spaces.Box(
                low=0.0,
                high=FLOAT_MAX,
                shape=(config.forecast_steps,),
                dtype=np.float32,
            ),
            "weather_forecast_mask": spaces.MultiBinary(config.forecast_steps),
            "recent_incidents": spaces.Dict(
                {
                    "kind": spaces.MultiDiscrete(
                        np.full(
                            config.incident_capacity,
                            len(INCIDENT_KIND_NAMES),
                            dtype=np.int64,
                        )
                    ),
                    "target": spaces.MultiDiscrete(
                        np.full(
                            config.incident_capacity,
                            edge_count + 1,
                            dtype=np.int64,
                        )
                    ),
                    "age": spaces.Box(
                        low=0.0,
                        high=FLOAT_MAX,
                        shape=(config.incident_capacity,),
                        dtype=np.float32,
                    ),
                    "duration": spaces.Box(
                        low=0.0,
                        high=FLOAT_MAX,
                        shape=(config.incident_capacity,),
                        dtype=np.float32,
                    ),
                    "mask": spaces.MultiBinary(config.incident_capacity),
                }
            ),
            "remaining_time": spaces.Box(
                low=0.0,
                high=config.episode_duration_seconds,
                shape=(1,),
                dtype=np.float32,
            ),
            "action_masks": build_action_mask_space(
                topology, config.ability_count, config.group_count
            ),
        }
    )


def build_observation(
    sim: "MountainSim",
    config: ObservationConfig,
    action_masks: ActionMasks | None = None,
) -> Observation:
    """Return one isolated observation from the reported simulator state."""
    topology = sim.topology
    if topology is None or sim.weather_schedule is None:
        raise RuntimeError("reset the simulator before the observation")

    pop = sim.population
    pending = pop.location_kind == LocationKind.PENDING
    at_node = pop.location_kind == LocationKind.NODE
    demand_locations = pop.location_index[pending | at_node]
    node_demand = np.bincount(demand_locations, minlength=topology.node_count).astype(
        np.float32
    )
    node_crowding = np.bincount(
        pop.location_index[at_node], minlength=topology.node_count
    ).astype(np.float32)

    state = sim.state
    reported_occupancy = state.reported_occupancy.astype(np.float32, copy=True)
    reported_queue = state.reported_queue_length.astype(np.float32, copy=True)
    capacity = topology.edge_safe_capacity.astype(np.float32, copy=True)
    reported_density = np.divide(
        reported_occupancy + reported_queue,
        np.maximum(capacity, 1.0),
        dtype=np.float32,
    )
    forecast, forecast_time, forecast_mask = _weather_forecast(sim, config)
    if action_masks is None:
        action_masks = build_action_masks(
            topology,
            config.ability_count,
            config.group_count,
            edge_available=~state.reported_closed,
        )
    masks = {
        name: np.asarray(value, dtype=np.int8).copy()
        for name, value in action_masks.items()
    }

    observation = Observation(
        {
            "node_demand": node_demand,
            "node_crowding": node_crowding,
            "reported_edge_occupancy": reported_occupancy,
            "reported_edge_queue_length": reported_queue,
            "reported_edge_speed_factor": state.reported_speed_factor.astype(
                np.float32, copy=True
            ),
            "reported_edge_closed": state.reported_closed.astype(np.int8, copy=True),
            "reported_edge_density": reported_density,
            "edge_capacity": capacity,
            "weather": sim.weather.as_array().astype(np.float32),
            "weather_forecast": forecast,
            "weather_forecast_time": forecast_time,
            "weather_forecast_mask": forecast_mask,
            "recent_incidents": _recent_incidents(sim, config),
            "remaining_time": np.array(
                [max(config.episode_duration_seconds - sim.simulation_time, 0.0)],
                dtype=np.float32,
            ),
            "action_masks": masks,
        }
    )
    _require_finite(observation)
    observation_space = build_observation_space(topology, config)
    if not observation_space.contains(observation):
        raise ValueError("the observation is outside its configured space")
    return observation


def _weather_forecast(
    sim: "MountainSim", config: ObservationConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the next fixed number of scheduled weather changes."""
    schedule = sim.weather_schedule
    assert schedule is not None
    forecast = np.zeros((config.forecast_steps, WEATHER_SIZE), dtype=np.float32)
    times = np.zeros(config.forecast_steps, dtype=np.float32)
    mask = np.zeros(config.forecast_steps, dtype=np.int8)
    transitions = schedule.transitions[
        schedule.next_transition : schedule.next_transition + config.forecast_steps
    ]
    for index, transition in enumerate(transitions):
        forecast[index] = transition.weather.as_array().astype(np.float32)
        times[index] = max(transition.start_time_seconds - sim.simulation_time, 0.0)
        mask[index] = 1
    return forecast, times, mask


def _recent_incidents(sim: "MountainSim", config: ObservationConfig) -> IncidentArrays:
    """Encode recent visible failures and hazard events."""
    records: list[tuple[float, str, int, float]] = []
    if sim.failure_schedule is not None:
        records.extend(
            (
                event.start_time_seconds,
                event.kind.value,
                event.target,
                event.duration_seconds,
            )
            for event in sim.failure_schedule.events
            if event.controller_visible
            and event.start_time_seconds <= sim.simulation_time
        )
    records.extend(
        (
            event.start_time_seconds,
            event.event_type,
            event.edge_index,
            0.0,
        )
        for event in sim.hazard_events
        if event.start_time_seconds <= sim.simulation_time
    )
    records.sort(key=lambda value: (value[0], value[1], value[2]))
    records = records[-config.incident_capacity :]

    kind = np.zeros(config.incident_capacity, dtype=np.int64)
    target = np.zeros(config.incident_capacity, dtype=np.int64)
    age = np.zeros(config.incident_capacity, dtype=np.float32)
    duration = np.zeros(config.incident_capacity, dtype=np.float32)
    mask = np.zeros(config.incident_capacity, dtype=np.int8)
    for index, (start, event_kind, edge, event_duration) in enumerate(records):
        kind[index] = INCIDENT_KIND_INDEX[event_kind]
        target[index] = edge + 1
        age[index] = sim.simulation_time - start
        duration[index] = event_duration
        mask[index] = 1
    return {
        "kind": kind,
        "target": target,
        "age": age,
        "duration": duration,
        "mask": mask,
    }


def _require_finite(value: Any) -> None:
    """Reject a non-finite floating observation value."""
    if isinstance(value, dict):
        for child in value.values():
            _require_finite(child)
        return
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("the observation must contain only finite values")

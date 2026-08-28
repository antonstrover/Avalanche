"""Build the fixed Gymnasium observation."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from gymnasium import spaces

from avalanche.control.types import DecisionType, MonitorDecision, Observation
from avalanche.env.actions import (
    build_action_contract,
    build_control_permission_space,
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
INTERVENTION_DECISION_NAMES = (
    "padding",
    *(decision.value for decision in DecisionType),
)
INTERVENTION_DECISION_INDEX = {
    name: index for index, name in enumerate(INTERVENTION_DECISION_NAMES)
}


class IncidentArrays(TypedDict):
    """The fixed arrays for recent visible incidents."""

    kind: np.ndarray
    target: np.ndarray
    age: np.ndarray
    duration: np.ndarray
    mask: np.ndarray


class InterventionArrays(TypedDict):
    """The fixed arrays for recent monitor interventions."""

    decision: np.ndarray
    risk: np.ndarray
    age: np.ndarray
    edge_targets: np.ndarray
    node_targets: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class InterventionRecord:
    """Hold one completed monitor intervention and its proposal time."""

    simulation_time: float
    decision: MonitorDecision


@dataclass(frozen=True)
class ObservationConfig:
    """The fixed observation sizes for one environment."""

    episode_duration_seconds: float
    forecast_steps: int = 4
    incident_capacity: int = 16
    intervention_capacity: int = 16
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
        if self.intervention_capacity < 1:
            raise ValueError("the intervention capacity must be positive")
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
            "reported_edge_hazard": spaces.Box(
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
            "recent_interventions": spaces.Dict(
                {
                    "decision": spaces.MultiDiscrete(
                        np.full(
                            config.intervention_capacity,
                            len(INTERVENTION_DECISION_NAMES),
                            dtype=np.int64,
                        )
                    ),
                    "risk": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(config.intervention_capacity,),
                        dtype=np.float32,
                    ),
                    "age": spaces.Box(
                        low=0.0,
                        high=FLOAT_MAX,
                        shape=(config.intervention_capacity,),
                        dtype=np.float32,
                    ),
                    "edge_targets": spaces.MultiBinary(
                        (config.intervention_capacity, edge_count)
                    ),
                    "node_targets": spaces.MultiBinary(
                        (config.intervention_capacity, node_count)
                    ),
                    "mask": spaces.MultiBinary(config.intervention_capacity),
                }
            ),
            "remaining_time": spaces.Box(
                low=0.0,
                high=config.episode_duration_seconds,
                shape=(1,),
                dtype=np.float32,
            ),
            "control_permissions": build_control_permission_space(
                topology, config.ability_count, config.group_count
            ),
            "reported_edge_available": spaces.MultiBinary(edge_count),
            "reported_edge_weather_risk": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(edge_count,), dtype=np.float32
            ),
            "reported_edge_boarding_throughput": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(edge_count,), dtype=np.float32
            ),
            "reported_node_queued_no_route_count": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(node_count,), dtype=np.float32
            ),
            "reported_edge_onboard_blocked_count": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(edge_count,), dtype=np.float32
            ),
            "route_availability_missing": spaces.MultiBinary(edge_count),
            "route_speed_factor_missing": spaces.MultiBinary(edge_count),
            "route_density_ratio_missing": spaces.MultiBinary(edge_count),
            "route_weather_risk_missing": spaces.MultiBinary(edge_count),
            "route_queue_length_missing": spaces.MultiBinary(edge_count),
            "route_boarding_throughput_missing": spaces.MultiBinary(edge_count),
            "queued_no_route_count_missing": spaces.MultiBinary(node_count),
            "onboard_blocked_count_missing": spaces.MultiBinary(edge_count),
            "route_sensor_sample_time": spaces.Box(
                low=-FLOAT_MAX, high=FLOAT_MAX, shape=(1,), dtype=np.float32
            ),
            "route_sensor_report_time": spaces.Box(
                low=0.0, high=FLOAT_MAX, shape=(1,), dtype=np.float32
            ),
        }
    )


def build_observation(
    sim: MountainSim,
    config: ObservationConfig,
    interventions: Sequence[InterventionRecord] = (),
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
    packet = sim.route_sensor_packet
    if packet is None:
        raise RuntimeError("reset the route sensor before the observation")
    reported_occupancy = state.reported_occupancy.astype(np.float32, copy=True)
    capacity = topology.edge_safe_capacity.astype(np.float32, copy=True)
    reported_queue = packet.reported_queue_length.astype(np.float32, copy=True)
    reported_queue[packet.queue_length_missing] = capacity[packet.queue_length_missing]
    reported_speed = np.clip(
        packet.reported_speed_factor,
        sim.routing_config.minimum_reported_speed_factor,
        1.0,
    ).astype(np.float32)
    reported_speed[packet.speed_factor_missing] = np.float32(
        sim.routing_config.minimum_reported_speed_factor
    )
    reported_density = packet.reported_density_ratio.astype(np.float32, copy=True)
    reported_weather_risk = packet.reported_weather_risk.astype(np.float32, copy=True)
    risk_missing = packet.density_ratio_missing | packet.weather_risk_missing
    reported_hazard = np.clip(
        np.maximum(
            reported_density
            - np.float32(sim.reported_risk_config.density_reference_ratio),
            0.0,
        )
        + reported_weather_risk,
        sim.reported_risk_config.minimum,
        sim.reported_risk_config.maximum,
    ).astype(np.float32)
    reported_hazard[risk_missing] = np.float32(sim.reported_risk_config.missing_value)
    reported_throughput = packet.reported_boarding_throughput.astype(
        np.float32, copy=True
    )
    reported_throughput[packet.boarding_throughput_missing] = np.float32(
        sim.routing_config.minimum_boarding_throughput_per_second
    )
    reported_available = (
        packet.reported_availability & ~packet.availability_missing
    ).astype(np.int8)
    reported_queued_no_route = packet.reported_queued_no_route_count.astype(
        np.float32, copy=True
    )
    reported_queued_no_route[packet.queued_no_route_count_missing] = 0.0
    reported_onboard_blocked = packet.reported_onboard_blocked_count.astype(
        np.float32, copy=True
    )
    reported_onboard_blocked[packet.onboard_blocked_count_missing] = 0.0
    forecast, forecast_time, forecast_mask = _weather_forecast(sim, config)
    contract = build_action_contract(
        topology,
        config.ability_count,
        config.group_count,
        reported_edge_closed=~reported_available.astype(bool),
    )
    permissions = {
        name: np.asarray(value, dtype=np.int8).copy()
        for name, value in contract["control_permissions"].items()
    }

    observation = Observation(
        {
            "node_demand": node_demand,
            "node_crowding": node_crowding,
            "reported_edge_occupancy": reported_occupancy,
            "reported_edge_queue_length": reported_queue,
            "reported_edge_speed_factor": reported_speed,
            "reported_edge_closed": (~reported_available.astype(bool)).astype(np.int8),
            "reported_edge_density": reported_density,
            "reported_edge_hazard": reported_hazard.copy(),
            "edge_capacity": capacity,
            "weather": sim.weather.as_array().astype(np.float32),
            "weather_forecast": forecast,
            "weather_forecast_time": forecast_time,
            "weather_forecast_mask": forecast_mask,
            "recent_incidents": _recent_incidents(sim, config),
            "recent_interventions": _recent_interventions(sim, config, interventions),
            "remaining_time": np.array(
                [max(config.episode_duration_seconds - sim.simulation_time, 0.0)],
                dtype=np.float32,
            ),
            "control_permissions": permissions,
            "reported_edge_available": reported_available,
            "reported_edge_weather_risk": reported_weather_risk,
            "reported_edge_boarding_throughput": reported_throughput,
            "reported_node_queued_no_route_count": reported_queued_no_route,
            "reported_edge_onboard_blocked_count": reported_onboard_blocked,
            "route_availability_missing": packet.availability_missing.astype(
                np.int8, copy=True
            ),
            "route_speed_factor_missing": packet.speed_factor_missing.astype(
                np.int8, copy=True
            ),
            "route_density_ratio_missing": packet.density_ratio_missing.astype(
                np.int8, copy=True
            ),
            "route_weather_risk_missing": packet.weather_risk_missing.astype(
                np.int8, copy=True
            ),
            "route_queue_length_missing": packet.queue_length_missing.astype(
                np.int8, copy=True
            ),
            "route_boarding_throughput_missing": (
                packet.boarding_throughput_missing.astype(np.int8, copy=True)
            ),
            "queued_no_route_count_missing": (
                packet.queued_no_route_count_missing.astype(np.int8, copy=True)
            ),
            "onboard_blocked_count_missing": (
                packet.onboard_blocked_count_missing.astype(np.int8, copy=True)
            ),
            "route_sensor_sample_time": np.array(
                [packet.sample_time], dtype=np.float32
            ),
            "route_sensor_report_time": np.array(
                [packet.report_time], dtype=np.float32
            ),
        }
    )
    _require_finite(observation)
    observation_space = build_observation_space(topology, config)
    if not observation_space.contains(observation):
        raise ValueError("the observation is outside its configured space")
    return observation


def _weather_forecast(
    sim: MountainSim, config: ObservationConfig
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


def _recent_incidents(sim: MountainSim, config: ObservationConfig) -> IncidentArrays:
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


def _recent_interventions(
    sim: MountainSim,
    config: ObservationConfig,
    interventions: Sequence[InterventionRecord],
) -> InterventionArrays:
    """Encode recent completed monitor interventions."""
    topology = sim.topology
    assert topology is not None
    selected = [
        record
        for record in interventions
        if record.decision.decision is not DecisionType.ALLOW
    ]
    selected.sort(key=lambda record: record.simulation_time)
    selected = selected[-config.intervention_capacity :]

    decision = np.zeros(config.intervention_capacity, dtype=np.int64)
    risk = np.zeros(config.intervention_capacity, dtype=np.float32)
    age = np.zeros(config.intervention_capacity, dtype=np.float32)
    edge_targets = np.zeros(
        (config.intervention_capacity, topology.edge_count), dtype=np.int8
    )
    node_targets = np.zeros(
        (config.intervention_capacity, topology.node_count), dtype=np.int8
    )
    mask = np.zeros(config.intervention_capacity, dtype=np.int8)
    for index, record in enumerate(selected):
        if not np.isfinite(record.simulation_time):
            raise ValueError("the intervention time must be finite")
        if record.simulation_time > sim.simulation_time:
            raise ValueError("the intervention time must not be in the future")
        monitor_decision = record.decision
        decision[index] = INTERVENTION_DECISION_INDEX[monitor_decision.decision.value]
        risk[index] = monitor_decision.risk_score
        age[index] = sim.simulation_time - record.simulation_time
        mask[index] = 1
        for target in monitor_decision.related_infrastructure:
            if target.kind == "edge":
                if target.index >= topology.edge_count:
                    message = "the intervention edge target is outside the topology"
                    raise ValueError(message)
                edge_targets[index, target.index] = 1
            else:
                if target.index >= topology.node_count:
                    message = "the intervention node target is outside the topology"
                    raise ValueError(message)
                node_targets[index, target.index] = 1
    return {
        "decision": decision,
        "risk": risk,
        "age": age,
        "edge_targets": edge_targets,
        "node_targets": node_targets,
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

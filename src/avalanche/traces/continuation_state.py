"""Capture and restore simulator state for exact continuation."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any

import numpy as np

from avalanche.control.types import (
    OperationalSensorPacket,
    ReportedStranding,
    SensorCategory,
    SensorValue,
)
from avalanche.metrics import OnlineMetrics
from avalanche.scenarios.audits import AuditChannel, AuditMeasurement
from avalanche.scenarios.failures import FailureEvent, FailureKind, FailureTransitions
from avalanche.scenarios.operational_events import OperationalEvent, OperationalEventKind
from avalanche.scenarios.sensors import RouteSensorChannel, RouteSensorPacket
from avalanche.scenarios.weather import Weather
from avalanche.sim.engine import MountainSim
from avalanche.sim.hazards import HazardEvent
from avalanche.sim.movement import DynamicState, MovementTransitions
from avalanche.sim.population import SkierArrays

_DATACLASSES = {
    value.__name__: value
    for value in (
        AuditMeasurement,
        DynamicState,
        FailureEvent,
        FailureTransitions,
        HazardEvent,
        MovementTransitions,
        OperationalEvent,
        OperationalSensorPacket,
        ReportedStranding,
        RouteSensorPacket,
        SensorValue,
        SkierArrays,
        Weather,
    )
}
_ENUMS = {
    value.__name__: value
    for value in (FailureKind, OperationalEventKind, SensorCategory)
}


def capture_simulator_state(sim: MountainSim) -> dict[str, Any]:
    """Return every mutable value that can influence later execution."""
    if sim.weather_schedule is None:
        raise RuntimeError("reset the simulator before continuation work")
    if sim.audit_channel is None or sim.route_sensor_channel is None:
        raise RuntimeError("reset the simulator before continuation work")
    return {
        "simulation_time": sim.simulation_time,
        "movement_tick": sim.step,
        "tick_seconds": sim.tick_seconds,
        "time_epsilon_seconds": sim.time_epsilon_seconds,
        "population": freeze_state(sim.population),
        "dynamic_state": freeze_state(sim.state),
        "random_streams": {
            name: freeze_state(stream.bit_generator.state)
            for name, stream in sim.streams.items()
        },
        "weather": {
            "current": freeze_state(sim.weather_schedule.current),
            "next_transition": sim.weather_schedule.next_transition,
        },
        "hazard_events": freeze_state(sim.hazard_events),
        "active_failures": freeze_state(sim.active_failures),
        "failure_transitions": freeze_state(sim.failure_transitions),
        "active_operational_events": freeze_state(sim.active_operational_events),
        "audit": {
            "measurements": freeze_state(sim.audit_channel.measurements),
            "delivered": freeze_state(sim.delivered_audits),
        },
        "metrics": {
            name: freeze_state(value)
            for name, value in sim.metrics.__dict__.items()
            if name not in {"topology", "environment_context"}
        },
        "route_sensor": {
            "latest": freeze_state(sim.route_sensor_channel.latest),
            "current": freeze_state(sim.route_sensor_packet),
            "pending": freeze_state(sim.route_sensor_channel.pending),
            "pending_stranding": freeze_state(
                sim.route_sensor_channel.pending_stranding
            ),
            "last_sample_time": sim.route_sensor_channel.last_sample_time,
        },
        "last_movement_transitions": freeze_state(sim.last_movement_transitions),
        "stranding_interval_counts": tuple(
            {
                "location_kind": key[0],
                "topology_id": key[1],
                "count": value,
            }
            for key, value in sorted(sim._stranding_interval_counts.items())
        ),
    }


def restore_simulator_state(sim: MountainSim, state: dict[str, Any]) -> None:
    """Restore simulator state after one compatible reset."""
    if sim.weather_schedule is None:
        raise RuntimeError("reset the simulator before continuation work")
    if sim.audit_channel is None or sim.route_sensor_channel is None:
        raise RuntimeError("reset the simulator before continuation work")
    sim.simulation_time = float(state["simulation_time"])
    sim.step = int(state["movement_tick"])
    sim.tick_seconds = float(state["tick_seconds"])
    sim.time_epsilon_seconds = float(state["time_epsilon_seconds"])
    sim.population = thaw_state(state["population"])
    sim.state = thaw_state(state["dynamic_state"])
    for name, random_state in state["random_streams"].items():
        sim.streams[name].bit_generator.state = thaw_state(random_state)
    weather = state["weather"]
    sim.weather_schedule.current = thaw_state(weather["current"])
    sim.weather_schedule.next_transition = int(weather["next_transition"])
    sim.hazard_events = thaw_state(state["hazard_events"])
    sim.active_failures = thaw_state(state["active_failures"])
    sim.failure_transitions = thaw_state(state["failure_transitions"])
    sim.active_operational_events = thaw_state(state["active_operational_events"])
    audit = state["audit"]
    sim.audit_channel = AuditChannel(
        sim.audit_config,
        sim.streams["audit"],
        sim.control_interval_seconds,
        sim.streams["audit_missing"],
    )
    sim.audit_channel.measurements = thaw_state(audit["measurements"])
    sim.delivered_audits = thaw_state(audit["delivered"])
    for name, value in state["metrics"].items():
        setattr(sim.metrics, name, thaw_state(value))
    sensor = state["route_sensor"]
    sim.route_sensor_channel = RouteSensorChannel(
        sim.route_sensor_config,
        sim.control_interval_seconds,
        sim.streams["sensor"],
        sim.streams["blocked_sensor"],
        sim.streams["stranding_sensor"],
        sim.streams["operational_sensor"],
    )
    sim.route_sensor_channel.latest = thaw_state(sensor["latest"])
    sim.route_sensor_packet = thaw_state(sensor["current"])
    sim.route_sensor_channel.pending = thaw_state(sensor["pending"])
    sim.route_sensor_channel.pending_stranding = thaw_state(
        sensor["pending_stranding"]
    )
    sample_time = sensor["last_sample_time"]
    sim.route_sensor_channel.last_sample_time = (
        None if sample_time is None else float(sample_time)
    )
    sim.last_movement_transitions = thaw_state(state["last_movement_transitions"])
    sim._stranding_interval_counts = {
        (item["location_kind"], item["topology_id"]): int(item["count"])
        for item in state["stranding_interval_counts"]
    }


def freeze_state(value: Any) -> Any:
    """Convert one allowed runtime value into canonical state values."""
    if isinstance(value, Enum):
        name = type(value).__name__
        if name not in _ENUMS:
            raise TypeError(f"the enum type {name!r} is unsupported")
        return {
            "object_kind": "enum",
            "class_name": name,
            "value": freeze_state(value.value),
        }
    if value is None or isinstance(value, (bool, int, float, str, bytes, np.ndarray)):
        return value.copy() if isinstance(value, np.ndarray) else value
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in _DATACLASSES:
            raise TypeError(f"the data class {name!r} is unsupported")
        return {
            "object_kind": "data_class",
            "class_name": name,
            "fields": {
                item.name: freeze_state(getattr(value, item.name))
                for item in fields(value)
            },
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("a continuation mapping key must be text")
        return {key: freeze_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(freeze_state(item) for item in value)
    if isinstance(value, list):
        return [freeze_state(item) for item in value]
    raise TypeError(f"the continuation value {type(value).__name__!r} is unsupported")


def thaw_state(value: Any) -> Any:
    """Rebuild one allowed runtime value from canonical state values."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        kind = value.get("object_kind")
        if kind == "enum":
            enum_type = _ENUMS.get(value.get("class_name"))
            if enum_type is None or set(value) != {"object_kind", "class_name", "value"}:
                raise ValueError("the continuation enum state is invalid")
            return enum_type(thaw_state(value["value"]))
        if kind == "data_class":
            data_type = _DATACLASSES.get(value.get("class_name"))
            if data_type is None or set(value) != {
                "object_kind",
                "class_name",
                "fields",
            }:
                raise ValueError("the continuation data class state is invalid")
            field_state = value["fields"]
            if not isinstance(field_state, dict):
                raise ValueError("the continuation data class fields are invalid")
            return data_type(
                **{name: thaw_state(item) for name, item in field_state.items()}
            )
        if kind is not None:
            raise ValueError("the continuation object kind is invalid")
        return {key: thaw_state(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(thaw_state(item) for item in value)
    if isinstance(value, list):
        return [thaw_state(item) for item in value]
    return value


def metrics_type() -> type[OnlineMetrics]:
    """Expose the metric type for static state audits."""
    return OnlineMetrics

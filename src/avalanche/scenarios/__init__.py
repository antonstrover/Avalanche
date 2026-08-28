"""Build deterministic scenario schedules."""

from typing import Any

from avalanche.scenarios.audits import (
    AUDIT_SCHEMA_VERSION,
    AuditChannel,
    AuditMeasurement,
    audit_edge_count,
)
from avalanche.scenarios.sensors import (
    ROUTE_SENSOR_SCHEMA_VERSION,
    RouteSensorChannel,
    RouteSensorPacket,
    perfect_route_sensor_packet,
)

__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AuditChannel",
    "AuditMeasurement",
    "audit_edge_count",
    "ROUTE_SENSOR_SCHEMA_VERSION",
    "RouteSensorChannel",
    "RouteSensorPacket",
    "perfect_route_sensor_packet",
    "FailureEvent",
    "FailureKind",
    "FailureSchedule",
    "FailureTransitions",
    "apply_failures",
    "refresh_reported_telemetry",
    "resolve_failure_schedule",
    "Weather",
    "WeatherSchedule",
    "WeatherTransition",
    "apply_weather",
    "resolve_weather_schedule",
]


def __getattr__(name: str) -> Any:
    """Load movement-dependent scenario modules only when requested."""
    if name in {
        "FailureEvent",
        "FailureKind",
        "FailureSchedule",
        "FailureTransitions",
        "apply_failures",
        "refresh_reported_telemetry",
        "resolve_failure_schedule",
    }:
        from avalanche.scenarios import failures

        return getattr(failures, name)
    if name in {
        "Weather",
        "WeatherSchedule",
        "WeatherTransition",
        "apply_weather",
        "resolve_weather_schedule",
    }:
        from avalanche.scenarios import weather

        return getattr(weather, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

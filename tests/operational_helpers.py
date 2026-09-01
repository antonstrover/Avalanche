"""Build strict operational observations for controller and monitor tests."""

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from avalanche.control import (
    ControllerObservation,
    OperationalAudit,
    SensorValue,
    build_controller_observation,
    thaw_evidence,
)
from avalanche.control.types import ControllerVisibleEvent, operational_packet_identity
from avalanche.sim import MountainSim


@lru_cache
def _base_observation(path_text: str) -> ControllerObservation:
    """Return one cached observation with valid public static evidence."""
    sim = MountainSim(Path(path_text))
    sim.reset(0)
    return build_controller_observation(sim)


def controller_observation(
    mountain_path: Path,
    *,
    simulation_time: float = 0.0,
    sensor_values: dict[str, Any] | None = None,
    sensor_missing: dict[str, Any] | None = None,
    audits: tuple[OperationalAudit, ...] = (),
    events: tuple[ControllerVisibleEvent, ...] = (),
) -> ControllerObservation:
    """Return one strict controller observation with selected sensor values."""
    base = _base_observation(str(Path(mountain_path).resolve()))
    return replace_operational_observation(
        base,
        simulation_time=simulation_time,
        sensor_values=sensor_values,
        sensor_missing=sensor_missing,
        audits=audits,
        events=events,
    )


def replace_operational_observation(
    observation: ControllerObservation,
    *,
    simulation_time: float | None = None,
    sensor_values: dict[str, Any] | None = None,
    sensor_missing: dict[str, Any] | None = None,
    audits: tuple[OperationalAudit, ...] | None = None,
    events: tuple[ControllerVisibleEvent, ...] | None = None,
) -> ControllerObservation:
    """Build complete synthetic evidence and rebuild the packet identity."""
    evidence = observation.operational_evidence
    packet = evidence.packet
    report_time = (
        evidence.simulation_time if simulation_time is None else simulation_time
    )
    sample_time = report_time - packet.control_interval_seconds
    selected_values = sensor_values or {}
    selected_missing = sensor_missing or {}
    neutral_values = _neutral_sensor_values(observation)
    sensors: list[SensorValue] = []
    for sensor in packet.sensors:
        values = np.array(sensor.values, copy=True)
        values[sensor.missing] = neutral_values[sensor.name][sensor.missing]
        if sensor.name in selected_values:
            values = np.asarray(selected_values[sensor.name], dtype=sensor.values.dtype)
            if values.ndim == 0:
                values = np.full(sensor.values.shape, values, dtype=sensor.values.dtype)
        missing = np.zeros(sensor.missing.shape, dtype=np.bool_)
        if sensor.name in selected_missing:
            missing = np.asarray(selected_missing[sensor.name], dtype=np.bool_)
            if missing.ndim == 0:
                missing = np.full(sensor.missing.shape, missing, dtype=np.bool_)
        values = np.array(values, copy=True)
        if np.issubdtype(values.dtype, np.floating):
            values[missing] = np.nan
        else:
            values[missing] = 0
        sensors.append(
            replace(
                sensor,
                values=values,
                missing=missing,
                sample_time=sample_time,
                report_time=report_time,
            )
        )
    sensor_tuple = tuple(sensors)
    identity = operational_packet_identity(
        packet.policy_identity,
        sample_time,
        report_time,
        sensor_tuple,
    )
    packet = replace(packet, sensors=sensor_tuple, packet_identity=identity)
    evidence = replace(
        evidence,
        simulation_time=report_time,
        packet=packet,
        audits=evidence.audits if audits is None else audits,
        events=evidence.events if events is None else events,
    )
    return replace(observation, operational_evidence=evidence)


def _neutral_sensor_values(
    observation: ControllerObservation,
) -> dict[str, np.ndarray]:
    """Return complete calm sensor values for one synthetic observation."""
    evidence = observation.operational_evidence
    static = evidence.static
    nodes = static.node_count
    edges = static.edge_count
    failures = evidence.packet.failure_capacity
    return {
        "node_demand": np.zeros(nodes, dtype="<i8"),
        "node_crowding": np.zeros(nodes, dtype="<i8"),
        "edge_occupancy": np.zeros(edges, dtype="<i8"),
        "edge_density": np.zeros(edges, dtype="<f8"),
        "edge_speed_factor": np.ones(edges, dtype="<f8"),
        "edge_availability": np.ones(edges, dtype=np.bool_),
        "edge_weather_risk": np.zeros(edges, dtype="<f8"),
        "lift_queue_length": np.zeros(edges, dtype="<i8"),
        "lift_occupancy": np.zeros(edges, dtype="<i8"),
        "lift_boarding_throughput": (
            static.edge_lift_throughput.astype("<f8") / 3_600.0
        ),
        "weather": np.array([0.0, 10_000.0, 0.0, 5.0], dtype="<f8"),
        "visible_failure_kind": np.zeros(failures, dtype="<i2"),
        "visible_failure_target": np.zeros(failures, dtype="<i4"),
        "visible_failure_present": np.zeros(failures, dtype=np.bool_),
        "queued_no_route_count": np.zeros(nodes, dtype="<i8"),
        "onboard_blocked_count": np.zeros(edges, dtype="<i8"),
    }


def operational_event(
    kind: str,
    target: int,
    target_type: str,
    *,
    simulation_time: float = 60.0,
    severity: float = 0.6,
    remaining_seconds: float = 240.0,
) -> ControllerVisibleEvent:
    """Return one valid controller-visible operating event."""
    return ControllerVisibleEvent(
        schema_version=1,
        kind=kind,
        target=target,
        target_type=target_type,
        severity=severity,
        remaining_seconds=remaining_seconds,
        sample_time=simulation_time,
        report_time=simulation_time,
        provenance_id="controller_visible_operational_event",
    )


def operational_audit(
    observation: ControllerObservation,
    target_edge: int,
    reported_density: float,
    measured_density: float,
    *,
    missing: bool = False,
) -> OperationalAudit:
    """Return one valid audit that follows the observation's public policy."""
    evidence = observation.operational_evidence
    policy = thaw_evidence(evidence.static.audit_policy)
    interval = evidence.static.control_interval_seconds
    delivery = round(evidence.simulation_time / interval)
    delay = int(policy["delivery_intervals"])
    sample = delivery - delay
    if sample < 0:
        raise ValueError("an audit needs enough elapsed control intervals")
    return OperationalAudit(
        schema_version=2,
        target_edge=target_edge,
        sample_interval=sample,
        delivery_interval=delivery,
        sample_time=sample * interval,
        report_time=delivery * interval,
        reported_density=reported_density,
        measured_density=np.nan if missing else measured_density,
        missing=missing,
        provenance_id=str(policy["provenance_identifier"]),
        noise_policy_id=str(policy["noise_policy_identifier"]),
        delay_intervals=delay,
    )

"""Encode and restore complete simulator snapshots."""

import hashlib
import json
from dataclasses import fields
from typing import Any

import numpy as np

from avalanche.control import DecisionType
from avalanche.metrics import METRICS_VERSION, OnlineMetrics
from avalanche.scenarios.audits import AuditChannel, AuditMeasurement
from avalanche.scenarios.sensors import ROUTE_SENSOR_CHANNELS
from avalanche.scenarios.weather import Weather, WeatherSchedule
from avalanche.sim.engine import STREAM_NAMES, MountainSim
from avalanche.sim.hazards import HazardEvent
from avalanche.sim.movement import DynamicState, new_dynamic_state
from avalanche.sim.population import SkierArrays, display_progress, empty_population

SNAPSHOT_SCHEMA_VERSION = 3

_SNAPSHOT_KEYS = {
    "snapshot_schema_version",
    "run_id",
    "episode_id",
    "seed",
    "simulation_time",
    "step",
    "state_checksum",
    "topology_checksum",
    "context_checksum",
    "node_count",
    "edge_count",
    "skier_count",
    "tick_seconds",
    "arrays",
    "state_json",
}
_ARRAY_KEYS = {"name", "dtype", "shape", "data"}
_STATE_KEYS = {
    "population",
    "weather",
    "hazard_events",
    "active_failures",
    "active_operational_event_ids",
    "audit",
    "metrics",
    "random_streams",
}
_POPULATION_KEYS = {"arrived", "next_ticket"}
_WEATHER_KEYS = {"current", "next_transition"}
_AUDIT_KEYS = {"measurements", "delivered"}
_METRIC_KEYS = {
    "metrics_version",
    "group_count",
    "episode_duration_seconds",
    "newly_stranded_skiers",
    "cumulative_stranded_seconds",
    "harm_onset_at",
    "harm_onset_control_interval",
    "dangerous_density_seconds",
    "density_exposure_seconds",
    "reported_density_exposure_seconds",
    "capacity_violation_seconds",
    "reported_capacity_violation_seconds",
    "safe_evacuation_capacity_skiers_per_second",
    "lost_safe_evacuation_capacity_seconds",
    "queue_no_route_blocked_seconds",
    "onboard_blocked_seconds",
    "group_stranded_seconds",
    "decision_counts",
    "intervention_latency_seconds_sum",
    "intervention_latency_count",
    "monitor_latency_seconds_sum",
    "monitor_decision_count",
    "first_intervention_interval",
    "cumulative_stranded_seconds_before_first_intervention",
    "route_decision_count",
    "missing_sensor_route_decision_count",
    "missing_sensor_route_decision_counts",
}


class SnapshotSchemaError(ValueError):
    """Report invalid or unsupported snapshot data."""


def encode_snapshot(
    sim: MountainSim,
    *,
    run_id: str,
    episode_id: str,
    seed: int,
) -> dict[str, Any]:
    """Return one complete and versioned simulator snapshot."""
    _require_reset(sim)
    assert sim.topology is not None
    assert sim.weather_schedule is not None
    arrays = [
        _encode_array(f"population.{name}", values)
        for name, values in _snapshot_v3_population_arrays(sim.population)
    ]
    arrays.extend(
        _encode_array(f"state.{name}", values)
        for name, values in sim.state.checksum_fields()
    )
    state = {
        "population": {
            "arrived": sim.population.arrived,
            "next_ticket": sim.population.next_ticket,
        },
        "weather": {
            "current": sim.weather.as_array().tolist(),
            "next_transition": sim.weather_schedule.next_transition,
        },
        "hazard_events": [event.as_dict() for event in sim.hazard_events],
        "active_failures": [event.as_dict() for event in sim.active_failures],
        "active_operational_event_ids": [
            event.event_id for event in sim.active_operational_events
        ],
        "audit": _audit_state(sim),
        "metrics": _metric_state(sim.metrics),
        "random_streams": {
            name: stream.bit_generator.state for name, stream in sim.streams.items()
        },
    }
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "run_id": run_id,
        "episode_id": episode_id,
        "seed": seed,
        "simulation_time": sim.simulation_time,
        "step": sim.step,
        "state_checksum": sim.state_checksum(),
        "topology_checksum": _topology_checksum(sim),
        "context_checksum": _context_checksum(sim),
        "node_count": sim.topology.node_count,
        "edge_count": sim.topology.edge_count,
        "skier_count": len(sim.population),
        "tick_seconds": sim.tick_seconds,
        "arrays": arrays,
        "state_json": _canonical_json(state),
    }


def restore_snapshot(sim: MountainSim, row: dict[str, Any]) -> None:
    """Validate one snapshot and replace a reset simulator state."""
    _require_reset(sim)
    assert sim.topology is not None
    assert sim.weather_schedule is not None
    assert sim.failure_schedule is not None
    assert sim.operational_event_schedule is not None
    _require_keys(row, _SNAPSHOT_KEYS, "snapshot")
    version = _integer(row["snapshot_schema_version"], "snapshot schema version")
    if version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotSchemaError(
            f"the snapshot schema version {version} is unsupported"
        )
    raise SnapshotSchemaError(
        "snapshot version three is display-only and cannot restore formal state"
    )

    simulation_time = _finite_float(row["simulation_time"], "simulation time")
    step = _integer(row["step"], "step")
    seed = _integer(row["seed"], "seed")
    node_count = _integer(row["node_count"], "node count")
    edge_count = _integer(row["edge_count"], "edge count")
    skier_count = _integer(row["skier_count"], "skier count")
    tick_seconds = _finite_float(row["tick_seconds"], "tick seconds")
    expected_checksum = row["state_checksum"]
    if not isinstance(expected_checksum, str):
        raise SnapshotSchemaError("the snapshot checksum must be text")
    if not isinstance(row["run_id"], str) or not isinstance(row["episode_id"], str):
        raise SnapshotSchemaError("the snapshot identities must be text")
    if simulation_time < 0.0 or step < 0 or skier_count < 0 or tick_seconds <= 0.0:
        raise SnapshotSchemaError("the snapshot has an invalid scalar value")
    if seed < 0:
        raise SnapshotSchemaError("the snapshot seed must not be negative")
    if node_count != sim.topology.node_count or edge_count != sim.topology.edge_count:
        raise SnapshotSchemaError("the snapshot topology dimensions do not match")
    if row["topology_checksum"] != _topology_checksum(sim):
        raise SnapshotSchemaError("the snapshot topology does not match")
    if row["context_checksum"] != _context_checksum(sim):
        raise SnapshotSchemaError("the snapshot configuration does not match")

    decoded = _decode_arrays(sim, row["arrays"], skier_count)
    state = _decode_json(row["state_json"])
    _require_keys(state, _STATE_KEYS, "snapshot state")
    population_state = _mapping(state["population"], "population state")
    _require_keys(population_state, _POPULATION_KEYS, "population state")
    arrived = _integer(population_state["arrived"], "arrived count")
    next_ticket = _integer(population_state["next_ticket"], "next ticket")
    if not 0 <= arrived <= skier_count or next_ticket < 0:
        raise SnapshotSchemaError("the snapshot population counters are invalid")

    weather_state = _mapping(state["weather"], "weather state")
    _require_keys(weather_state, _WEATHER_KEYS, "weather state")
    current_weather = _weather(weather_state["current"])
    next_transition = _integer(
        weather_state["next_transition"], "weather transition index"
    )
    if not 0 <= next_transition <= len(sim.weather_schedule.transitions):
        raise SnapshotSchemaError("the weather transition index is invalid")

    new_streams = _random_streams(state["random_streams"])
    new_metrics = _metrics(state["metrics"])
    audit_state = _mapping(state["audit"], "audit state")
    _require_keys(audit_state, _AUDIT_KEYS, "audit state")
    measurements = _audit_measurements(
        audit_state["measurements"], "audit measurements"
    )
    delivered = _audit_measurements(audit_state["delivered"], "delivered audits")
    hazards = _hazard_events(state["hazard_events"])

    active_failures = sim.failure_schedule.active(simulation_time)
    if state["active_failures"] != [event.as_dict() for event in active_failures]:
        raise SnapshotSchemaError("the active failure state does not match")
    active_events = sim.operational_event_schedule.active(simulation_time)
    if state["active_operational_event_ids"] != [
        event.event_id for event in active_events
    ]:
        raise SnapshotSchemaError("the active operational state does not match")

    population_values = {
        item.name: decoded[f"population.{item.name}"]
        for item in fields(SkierArrays)
        if isinstance(getattr(empty_population(0), item.name), np.ndarray)
    }
    dynamic_values = {
        item.name: decoded[f"state.{item.name}"] for item in fields(DynamicState)
    }
    population = SkierArrays(
        **population_values,
        arrived=arrived,
        next_ticket=next_ticket,
    )
    dynamic_state = DynamicState(**dynamic_values)
    audit_channel = AuditChannel(sim.audit_config, new_streams["audit"])
    audit_channel.measurements = list(measurements)
    weather_schedule = WeatherSchedule(
        sim.weather_schedule.transitions,
        current_weather,
        next_transition,
    )

    previous = (
        sim.tick_seconds,
        sim.simulation_time,
        sim.step,
        sim.population,
        sim.state,
        sim.weather_schedule,
        sim.hazard_events,
        sim.active_failures,
        sim.active_operational_events,
        sim.streams,
        sim.audit_channel,
        sim.delivered_audits,
        sim.metrics,
    )
    replacement = (
        tick_seconds,
        simulation_time,
        step,
        population,
        dynamic_state,
        weather_schedule,
        list(hazards),
        active_failures,
        active_events,
        new_streams,
        audit_channel,
        delivered,
        new_metrics,
    )
    _replace_state(sim, replacement)
    if sim.state_checksum() != expected_checksum:
        _replace_state(sim, previous)
        raise SnapshotSchemaError("the restored state checksum does not match")


def _require_reset(sim: MountainSim) -> None:
    """Require the target simulator to have one resolved context."""
    values = (
        sim.topology,
        sim.routes,
        sim.weather_schedule,
        sim.failure_schedule,
        sim.audit_channel,
        sim.operational_event_schedule,
    )
    if any(value is None for value in values):
        raise SnapshotSchemaError("reset the simulator before snapshot work")


def _snapshot_v3_population_arrays(
    population: SkierArrays,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Return the version three display arrays."""
    arrays: list[tuple[str, np.ndarray]] = []
    for name, values in population.checksum_fields():
        if name == "required_travel_seconds":
            arrays.append(("progress", display_progress(population)))
        elif name not in {
            "remaining_travel_seconds",
            "queue_no_route_blocked_seconds",
            "onboard_blocked_seconds",
            "queue_source_node",
            "chosen_edge",
            "locally_rejected_edge",
            "first_stranded_at",
            "ever_stranded",
        }:
            arrays.append((name, values))
    return tuple(arrays)


def _replace_state(sim: MountainSim, values: tuple[Any, ...]) -> None:
    """Replace each mutable simulator state owner."""
    (
        sim.tick_seconds,
        sim.simulation_time,
        sim.step,
        sim.population,
        sim.state,
        sim.weather_schedule,
        sim.hazard_events,
        sim.active_failures,
        sim.active_operational_events,
        sim.streams,
        sim.audit_channel,
        sim.delivered_audits,
        sim.metrics,
    ) = values


def _encode_array(name: str, values: np.ndarray) -> dict[str, Any]:
    """Encode one array with an explicit portable representation."""
    dtype_name, dtype = _portable_dtype(values.dtype)
    portable = np.ascontiguousarray(values, dtype=dtype)
    return {
        "name": name,
        "dtype": dtype_name,
        "shape": list(portable.shape),
        "data": portable.tobytes(),
    }


def _decode_arrays(
    sim: MountainSim, value: Any, skier_count: int
) -> dict[str, np.ndarray]:
    """Validate and decode every required snapshot array."""
    if not isinstance(value, list):
        raise SnapshotSchemaError("the snapshot arrays must be a list")
    assert sim.topology is not None
    population_template = empty_population(0)
    state_template = new_dynamic_state(sim.topology)
    expected: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {}
    for name, array in population_template.checksum_fields():
        expected[f"population.{name}"] = (array.dtype, (skier_count,))
    for name, array in state_template.checksum_fields():
        expected[f"state.{name}"] = (array.dtype, array.shape)

    decoded: dict[str, np.ndarray] = {}
    for entry_value in value:
        entry = _mapping(entry_value, "snapshot array")
        _require_keys(entry, _ARRAY_KEYS, "snapshot array")
        name = entry["name"]
        if not isinstance(name, str):
            raise SnapshotSchemaError("a snapshot array name must be text")
        if name in decoded:
            raise SnapshotSchemaError(f"the snapshot array {name!r} is duplicated")
        if name not in expected:
            raise SnapshotSchemaError(f"the snapshot array {name!r} is unknown")
        native_dtype, expected_shape = expected[name]
        dtype_name, portable_dtype = _portable_dtype(native_dtype)
        if entry["dtype"] != dtype_name:
            raise SnapshotSchemaError(f"the snapshot array {name!r} has a bad type")
        shape = entry["shape"]
        if not isinstance(shape, list) or any(
            not isinstance(size, int) or isinstance(size, bool) for size in shape
        ):
            raise SnapshotSchemaError(f"the snapshot array {name!r} has a bad shape")
        if tuple(shape) != expected_shape:
            raise SnapshotSchemaError(f"the snapshot array {name!r} has a bad shape")
        data = entry["data"]
        if not isinstance(data, bytes):
            raise SnapshotSchemaError(f"the snapshot array {name!r} has bad data")
        size = int(np.prod(expected_shape, dtype=np.int64))
        if len(data) != size * portable_dtype.itemsize:
            raise SnapshotSchemaError(f"the snapshot array {name!r} has bad data")
        portable = np.frombuffer(data, dtype=portable_dtype).reshape(expected_shape)
        if native_dtype.kind == "b" and np.any(portable > 1):
            raise SnapshotSchemaError(f"the snapshot array {name!r} has bad flags")
        result = portable.astype(native_dtype, copy=True)
        if result.dtype.kind == "f" and not np.all(np.isfinite(result)):
            raise SnapshotSchemaError(f"the snapshot array {name!r} is not finite")
        decoded[name] = result
    missing = sorted(set(expected) - set(decoded))
    if missing:
        raise SnapshotSchemaError(f"the snapshot array {missing[0]!r} is missing")
    return decoded


def _portable_dtype(dtype: np.dtype[Any]) -> tuple[str, np.dtype[Any]]:
    """Return one stable snapshot type for a NumPy type."""
    value = np.dtype(dtype)
    types: dict[np.dtype[Any], tuple[str, np.dtype[Any]]] = {
        np.dtype(np.bool_): ("uint8", np.dtype("u1")),
        np.dtype(np.int8): ("int8", np.dtype("i1")),
        np.dtype(np.int32): ("int32-le", np.dtype("<i4")),
        np.dtype(np.int64): ("int64-le", np.dtype("<i8")),
        np.dtype(np.float32): ("float32-le", np.dtype("<f4")),
        np.dtype(np.float64): ("float64-le", np.dtype("<f8")),
    }
    try:
        return types[value]
    except KeyError:
        raise SnapshotSchemaError(f"the array type {value} is unsupported") from None


def _audit_state(sim: MountainSim) -> dict[str, Any]:
    """Return all pending and delivered audit state."""
    assert sim.audit_channel is not None
    return {
        "measurements": list(sim.audit_channel.complete_records()),
        "delivered": [item.privileged() for item in sim.delivered_audits],
    }


def _metric_state(metrics: OnlineMetrics) -> dict[str, Any]:
    """Return every mutable online metric accumulator."""
    return {
        "metrics_version": METRICS_VERSION,
        "group_count": metrics.group_count,
        "episode_duration_seconds": metrics.episode_duration_seconds,
        "newly_stranded_skiers": metrics.newly_stranded_skiers,
        "cumulative_stranded_seconds": metrics.cumulative_stranded_seconds,
        "harm_onset_at": metrics.harm_onset_at,
        "harm_onset_control_interval": metrics.harm_onset_control_interval,
        "dangerous_density_seconds": metrics.dangerous_density_seconds,
        "density_exposure_seconds": metrics.density_exposure_seconds,
        "reported_density_exposure_seconds": (
            metrics.reported_density_exposure_seconds
        ),
        "capacity_violation_seconds": metrics.capacity_violation_seconds,
        "reported_capacity_violation_seconds": (
            metrics.reported_capacity_violation_seconds
        ),
        "safe_evacuation_capacity_skiers_per_second": (
            metrics.safe_evacuation_capacity_skiers_per_second
        ),
        "lost_safe_evacuation_capacity_seconds": (
            metrics.lost_safe_evacuation_capacity_seconds
        ),
        "queue_no_route_blocked_seconds": metrics.queue_no_route_blocked_seconds,
        "onboard_blocked_seconds": metrics.onboard_blocked_seconds,
        "group_stranded_seconds": metrics.group_stranded_seconds.tolist(),
        "decision_counts": dict(metrics.decision_counts),
        "intervention_latency_seconds_sum": (metrics.intervention_latency_seconds_sum),
        "intervention_latency_count": metrics.intervention_latency_count,
        "monitor_latency_seconds_sum": metrics.monitor_latency_seconds_sum,
        "monitor_decision_count": metrics.monitor_decision_count,
        "first_intervention_interval": metrics.first_intervention_interval,
        "cumulative_stranded_seconds_before_first_intervention": (
            metrics.cumulative_stranded_seconds_before_first_intervention
        ),
        "route_decision_count": metrics.route_decision_count,
        "missing_sensor_route_decision_count": (
            metrics.missing_sensor_route_decision_count
        ),
        "missing_sensor_route_decision_counts": dict(
            metrics.missing_sensor_route_decision_counts
        ),
    }


def _metrics(value: Any) -> OnlineMetrics:
    """Validate and rebuild every online metric accumulator."""
    state = _mapping(value, "metric state")
    _require_keys(state, _METRIC_KEYS, "metric state")
    version = _nonnegative_integer(state["metrics_version"], "metrics version")
    if version != METRICS_VERSION:
        raise SnapshotSchemaError(
            f"the metrics schema version {version} is unsupported"
        )
    group_count = _integer(state["group_count"], "metric group count")
    duration = _finite_float(
        state["episode_duration_seconds"], "metric episode duration"
    )
    try:
        metrics = OnlineMetrics(group_count, duration)
    except ValueError as error:
        raise SnapshotSchemaError(str(error)) from error
    metrics.newly_stranded_skiers = _nonnegative_integer(
        state["newly_stranded_skiers"], "newly stranded metric"
    )
    metrics.cumulative_stranded_seconds = _nonnegative_float(
        state["cumulative_stranded_seconds"], "cumulative stranded metric"
    )
    onset = state["harm_onset_at"]
    metrics.harm_onset_at = (
        None if onset is None else _nonnegative_float(onset, "harm onset")
    )
    onset_interval = state["harm_onset_control_interval"]
    metrics.harm_onset_control_interval = (
        None
        if onset_interval is None
        else _nonnegative_integer(onset_interval, "harm onset interval")
    )
    metrics.dangerous_density_seconds = _nonnegative_float(
        state["dangerous_density_seconds"], "dangerous density metric"
    )
    metrics.density_exposure_seconds = _nonnegative_float(
        state["density_exposure_seconds"], "density exposure metric"
    )
    metrics.reported_density_exposure_seconds = _nonnegative_float(
        state["reported_density_exposure_seconds"],
        "reported density exposure metric",
    )
    metrics.capacity_violation_seconds = _nonnegative_float(
        state["capacity_violation_seconds"], "capacity violation metric"
    )
    metrics.reported_capacity_violation_seconds = _nonnegative_float(
        state["reported_capacity_violation_seconds"],
        "reported capacity violation metric",
    )
    metrics.safe_evacuation_capacity_skiers_per_second = _nonnegative_float(
        state["safe_evacuation_capacity_skiers_per_second"],
        "safe evacuation capacity metric",
    )
    metrics.lost_safe_evacuation_capacity_seconds = _nonnegative_float(
        state["lost_safe_evacuation_capacity_seconds"],
        "lost safe evacuation capacity metric",
    )
    metrics.queue_no_route_blocked_seconds = _nonnegative_float(
        state["queue_no_route_blocked_seconds"], "queue blocked metric"
    )
    metrics.onboard_blocked_seconds = _nonnegative_float(
        state["onboard_blocked_seconds"], "onboard blocked metric"
    )
    stranded = state["group_stranded_seconds"]
    if not isinstance(stranded, list) or len(stranded) != group_count:
        raise SnapshotSchemaError("the grouped stranded metric has a bad shape")
    metrics.group_stranded_seconds = np.array(
        [_nonnegative_float(item, "grouped stranded metric") for item in stranded],
        dtype=np.float64,
    )
    decision_counts = _mapping(state["decision_counts"], "decision counts")
    expected_decisions = {item.value for item in DecisionType}
    if set(decision_counts) != expected_decisions:
        raise SnapshotSchemaError("the decision counts have invalid fields")
    metrics.decision_counts = {
        name: _nonnegative_integer(count, "decision count")
        for name, count in decision_counts.items()
    }
    metrics.intervention_latency_seconds_sum = _nonnegative_float(
        state["intervention_latency_seconds_sum"], "intervention latency"
    )
    metrics.intervention_latency_count = _nonnegative_integer(
        state["intervention_latency_count"], "intervention latency count"
    )
    metrics.monitor_latency_seconds_sum = _nonnegative_float(
        state["monitor_latency_seconds_sum"], "monitor latency"
    )
    metrics.monitor_decision_count = _nonnegative_integer(
        state["monitor_decision_count"], "monitor decision count"
    )
    intervention = state["first_intervention_interval"]
    metrics.first_intervention_interval = (
        None
        if intervention is None
        else _nonnegative_integer(intervention, "first intervention")
    )
    harm = state["cumulative_stranded_seconds_before_first_intervention"]
    metrics.cumulative_stranded_seconds_before_first_intervention = (
        None
        if harm is None
        else _nonnegative_float(harm, "stranded seconds before first intervention")
    )
    metrics.route_decision_count = _nonnegative_integer(
        state["route_decision_count"], "route decision count"
    )
    metrics.missing_sensor_route_decision_count = _nonnegative_integer(
        state["missing_sensor_route_decision_count"],
        "missing-sensor route decision count",
    )
    route_counts = _mapping(
        state["missing_sensor_route_decision_counts"],
        "missing-sensor route decision counts",
    )
    if set(route_counts) != set(ROUTE_SENSOR_CHANNELS):
        raise SnapshotSchemaError("the missing route counts have invalid fields")
    metrics.missing_sensor_route_decision_counts = {
        name: _nonnegative_integer(route_counts[name], "missing route channel count")
        for name in ROUTE_SENSOR_CHANNELS
    }
    if metrics.missing_sensor_route_decision_count > metrics.route_decision_count:
        raise SnapshotSchemaError("missing route decisions exceed all route decisions")
    if any(
        count > metrics.route_decision_count
        for count in metrics.missing_sensor_route_decision_counts.values()
    ):
        raise SnapshotSchemaError("a missing route channel exceeds all route decisions")
    return metrics


def _random_streams(value: Any) -> dict[str, np.random.Generator]:
    """Validate and rebuild every independent random stream."""
    states = _mapping(value, "random streams")
    if set(states) != set(STREAM_NAMES):
        raise SnapshotSchemaError("the snapshot random streams do not match")
    streams: dict[str, np.random.Generator] = {}
    for name in STREAM_NAMES:
        state = _mapping(states[name], f"the {name} random stream")
        stream = np.random.default_rng()
        try:
            stream.bit_generator.state = state
        except (TypeError, ValueError) as error:
            raise SnapshotSchemaError(
                f"the {name} random stream state is invalid"
            ) from error
        streams[name] = stream
    return streams


def _audit_measurements(value: Any, label: str) -> tuple[AuditMeasurement, ...]:
    """Validate and rebuild one audit measurement sequence."""
    if not isinstance(value, list):
        raise SnapshotSchemaError(f"the {label} must be a list")
    try:
        return tuple(AuditMeasurement(**_mapping(item, label)) for item in value)
    except (TypeError, ValueError) as error:
        raise SnapshotSchemaError(f"the {label} are invalid") from error


def _hazard_events(value: Any) -> tuple[HazardEvent, ...]:
    """Validate and rebuild the recorded hazard events."""
    if not isinstance(value, list):
        raise SnapshotSchemaError("the hazard events must be a list")
    try:
        return tuple(HazardEvent(**_mapping(item, "hazard event")) for item in value)
    except (TypeError, ValueError) as error:
        raise SnapshotSchemaError("the hazard events are invalid") from error


def _weather(value: Any) -> Weather:
    """Validate and rebuild the current weather vector."""
    if not isinstance(value, list) or len(value) != 4:
        raise SnapshotSchemaError("the weather vector has a bad shape")
    return Weather(*(_finite_float(item, "weather value") for item in value))


def _topology_checksum(sim: MountainSim) -> str:
    """Return one identity for the complete static topology."""
    digest = hashlib.sha256()
    topology = sim.topology
    assert topology is not None
    identity = {"name": topology.name, "nodes": topology.node_ids}
    digest.update(_canonical_json(identity).encode())
    for item in fields(topology):
        value = getattr(topology, item.name)
        if isinstance(value, np.ndarray):
            encoded = _encode_array(item.name, value)
            metadata = {key: encoded[key] for key in ("name", "dtype", "shape")}
            digest.update(_canonical_json(metadata).encode())
            digest.update(encoded["data"])
    return digest.hexdigest()


def _context_checksum(sim: MountainSim) -> str:
    """Return one identity for the immutable continuation context."""
    assert sim.weather_schedule is not None
    assert sim.failure_schedule is not None
    assert sim.operational_event_schedule is not None
    context = {
        "tick_seconds": sim.tick_seconds,
        "time_epsilon_seconds": sim.time_epsilon_seconds,
        "weather_config": sim.weather_config.model_dump(mode="json"),
        "weather_transitions": [
            {
                "start_time_seconds": item.start_time_seconds,
                "weather": item.weather.as_array().tolist(),
            }
            for item in sim.weather_schedule.transitions
        ],
        "hazard_config": sim.hazard_config.model_dump(mode="json"),
        "failure_schedule": [item.as_dict() for item in sim.failure_schedule.events],
        "audit_config": sim.audit_config.model_dump(mode="json"),
        "operational_event_schedule": [
            item.complete() for item in sim.operational_event_schedule.events
        ],
        "environment_context": {
            "evacuation_target_edges": list(
                sim.environment_context.evacuation_target_edges
            ),
            "evacuation_target_abilities": [
                list(abilities)
                for abilities in sim.environment_context.evacuation_target_abilities
            ],
            "baseline_safe_evacuation_capacity_skiers_per_second": (
                sim.environment_context.baseline_safe_evacuation_capacity_skiers_per_second
            ),
        },
    }
    return hashlib.sha256(_canonical_json(context).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    """Return deterministic JSON without non-finite numbers."""
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_json(value: Any) -> dict[str, Any]:
    """Decode JSON and reject duplicate object fields."""
    if not isinstance(value, str):
        raise SnapshotSchemaError("the snapshot state must be JSON text")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, item in pairs:
            if name in result:
                raise SnapshotSchemaError(f"the JSON field {name!r} is duplicated")
            result[name] = item
        return result

    try:
        result = json.loads(value, object_pairs_hook=object_pairs)
    except SnapshotSchemaError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise SnapshotSchemaError("the snapshot state JSON is invalid") from error
    return _mapping(result, "snapshot state")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    """Require one string-keyed mapping."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SnapshotSchemaError(f"the {label} must be an object")
    return value


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    """Require one exact field set."""
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        raise SnapshotSchemaError(f"the {label} field {missing[0]!r} is missing")
    if extra:
        raise SnapshotSchemaError(f"the {label} field {extra[0]!r} is unknown")


def _integer(value: Any, label: str) -> int:
    """Require one integer scalar."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise SnapshotSchemaError(f"the {label} must be an integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    """Require one nonnegative integer scalar."""
    result = _integer(value, label)
    if result < 0:
        raise SnapshotSchemaError(f"the {label} must not be negative")
    return result


def _finite_float(value: Any, label: str) -> float:
    """Require one finite numeric scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnapshotSchemaError(f"the {label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise SnapshotSchemaError(f"the {label} must be finite")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    """Require one finite nonnegative scalar."""
    result = _finite_float(value, label)
    if result < 0.0:
        raise SnapshotSchemaError(f"the {label} must not be negative")
    return result

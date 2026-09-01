"""Build delayed operational packets for route choice."""

import hashlib
import json
from dataclasses import dataclass, replace

import numpy as np

from avalanche.config.models import SensorPolicyConfig
from avalanche.control.types import (
    MINIMUM_OPERATIONAL_SPEED_FACTOR,
    OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
    OPERATIONAL_SENSOR_SPECS,
    VISIBLE_FAILURE_CAPACITY,
    OperationalSensorPacket,
    ReportedStranding,
    SensorValue,
    operational_packet_identity,
)

ROUTE_SENSOR_SCHEMA_VERSION = 3
FAILURE_SENSOR_CAPACITY = VISIBLE_FAILURE_CAPACITY
ROUTE_SENSOR_CHANNELS = (
    "availability",
    "speed_factor",
    "density_ratio",
    "weather_risk",
    "queue_length",
    "boarding_throughput",
)
BLOCKED_SENSOR_CHANNELS = (
    "queued_no_route_count",
    "onboard_blocked_count",
)


def _immutable_array(values: np.ndarray, dtype: str) -> np.ndarray:
    """Copy one array onto immutable bytes."""
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class RouteSensorPacket:
    """Hold one immutable delayed route report."""

    schema_version: int
    sample_time: float
    report_time: float
    provenance: tuple[tuple[str, str], ...]
    policy_identity: str
    reported_availability: np.ndarray
    reported_speed_factor: np.ndarray
    reported_density_ratio: np.ndarray
    reported_weather_risk: np.ndarray
    reported_queue_length: np.ndarray
    reported_boarding_throughput: np.ndarray
    reported_queued_no_route_count: np.ndarray
    reported_onboard_blocked_count: np.ndarray
    availability_missing: np.ndarray
    speed_factor_missing: np.ndarray
    density_ratio_missing: np.ndarray
    weather_risk_missing: np.ndarray
    queue_length_missing: np.ndarray
    boarding_throughput_missing: np.ndarray
    queued_no_route_count_missing: np.ndarray
    onboard_blocked_count_missing: np.ndarray
    operational_packet: OperationalSensorPacket | None = None
    reported_stranding: tuple[ReportedStranding, ...] = ()

    def __post_init__(self) -> None:
        """Freeze every packet array on immutable bytes."""
        edge_numeric = (
            "reported_speed_factor",
            "reported_density_ratio",
            "reported_weather_risk",
            "reported_queue_length",
            "reported_boarding_throughput",
            "reported_onboard_blocked_count",
        )
        edge_boolean = (
            "reported_availability",
            "availability_missing",
            "speed_factor_missing",
            "density_ratio_missing",
            "weather_risk_missing",
            "queue_length_missing",
            "boarding_throughput_missing",
            "onboard_blocked_count_missing",
        )
        node_numeric = ("reported_queued_no_route_count",)
        node_boolean = ("queued_no_route_count_missing",)
        for name in (*edge_numeric, *node_numeric):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), "<f8"))
        for name in (*edge_boolean, *node_boolean):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), "|b1"))
        edge_shape = self.reported_availability.shape
        if len(edge_shape) != 1 or any(
            getattr(self, name).shape != edge_shape
            for name in (*edge_numeric, *edge_boolean)
        ):
            raise ValueError("each route sensor channel must have one edge shape")
        node_shape = self.reported_queued_no_route_count.shape
        if len(node_shape) != 1 or any(
            getattr(self, name).shape != node_shape
            for name in (*node_numeric, *node_boolean)
        ):
            raise ValueError("each queued blocked channel must have one node shape")

    @property
    def edge_count(self) -> int:
        """Return the edge count of this packet."""
        return int(self.reported_availability.size)

    @property
    def node_count(self) -> int:
        """Return the node count of this packet."""
        return int(self.reported_queued_no_route_count.size)


def route_sensor_policy_identity(policy: SensorPolicyConfig) -> str:
    """Return the stable identity of one route sensor policy."""
    payload = json.dumps(
        policy.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


class RouteSensorChannel:
    """Sample, delay, and retain operational route packets."""

    def __init__(
        self,
        policy: SensorPolicyConfig,
        control_interval_seconds: float,
        route_rng: np.random.Generator,
        blocked_rng: np.random.Generator,
        stranding_rng: np.random.Generator | None = None,
    ) -> None:
        if control_interval_seconds <= 0.0:
            raise ValueError("the control interval must be positive")
        self.policy = policy
        self.control_interval_seconds = float(control_interval_seconds)
        self.route_rng = route_rng
        self.blocked_rng = blocked_rng
        self.stranding_rng = route_rng if stranding_rng is None else stranding_rng
        self.policy_identity = route_sensor_policy_identity(policy)
        self.latest: RouteSensorPacket | None = None
        self.pending: list[RouteSensorPacket] = []
        self.pending_stranding: list[ReportedStranding] = []
        self.last_sample_time: float | None = None

    def bootstrap(
        self,
        *,
        availability: np.ndarray,
        speed_factor: np.ndarray,
        density_ratio: np.ndarray,
        weather_risk: np.ndarray,
        queue_length: np.ndarray,
        boarding_throughput: np.ndarray,
        queued_no_route_count: np.ndarray,
        onboard_blocked_count: np.ndarray,
        node_demand: np.ndarray | None = None,
        node_crowding: np.ndarray | None = None,
        edge_occupancy: np.ndarray | None = None,
        lift_occupancy: np.ndarray | None = None,
        weather: np.ndarray | None = None,
        visible_failure_kind: np.ndarray | None = None,
        visible_failure_target: np.ndarray | None = None,
        visible_failure_present: np.ndarray | None = None,
        stranding_locations: tuple[tuple[str, str, int], ...] = (),
    ) -> RouteSensorPacket:
        """Create the bootstrap report and the first delayed report."""
        values = {
            "availability": availability,
            "speed_factor": speed_factor,
            "density_ratio": density_ratio,
            "weather_risk": weather_risk,
            "queue_length": queue_length,
            "boarding_throughput": boarding_throughput,
            "queued_no_route_count": queued_no_route_count,
            "onboard_blocked_count": onboard_blocked_count,
            "node_demand": node_demand,
            "node_crowding": node_crowding,
            "edge_occupancy": edge_occupancy,
            "lift_occupancy": lift_occupancy,
            "weather": weather,
            "visible_failure_kind": visible_failure_kind,
            "visible_failure_target": visible_failure_target,
            "visible_failure_present": visible_failure_present,
        }
        bootstrap = self._sample(
            sample_time=-self.control_interval_seconds,
            report_time=0.0,
            **values,
        )
        if bootstrap.operational_packet is not None:
            bootstrap = replace(
                bootstrap,
                operational_packet=self._masked_operational_packet(
                    bootstrap.operational_packet
                ),
            )
        self.latest = bootstrap
        self.pending = [
            self._sample(
                sample_time=0.0,
                report_time=self.control_interval_seconds,
                **values,
            )
        ]
        self.pending_stranding = self._sample_stranding(
            stranding_locations,
            sample_time=0.0,
            report_time=(
                self.control_interval_seconds
                * self.policy.stranding_delay_control_intervals
            ),
        )
        self.last_sample_time = 0.0
        return self.latest

    def advance(
        self,
        simulation_time: float,
        *,
        availability: np.ndarray,
        speed_factor: np.ndarray,
        density_ratio: np.ndarray,
        weather_risk: np.ndarray,
        queue_length: np.ndarray,
        boarding_throughput: np.ndarray,
        queued_no_route_count: np.ndarray,
        onboard_blocked_count: np.ndarray,
        node_demand: np.ndarray | None = None,
        node_crowding: np.ndarray | None = None,
        edge_occupancy: np.ndarray | None = None,
        lift_occupancy: np.ndarray | None = None,
        weather: np.ndarray | None = None,
        visible_failure_kind: np.ndarray | None = None,
        visible_failure_target: np.ndarray | None = None,
        visible_failure_present: np.ndarray | None = None,
        stranding_locations: tuple[tuple[str, str, int], ...] = (),
    ) -> RouteSensorPacket:
        """Deliver due packets and sample one current packet."""
        self.deliver(simulation_time)
        if self.last_sample_time != simulation_time:
            self.pending.append(
                self._sample(
                    sample_time=simulation_time,
                    report_time=simulation_time + self.control_interval_seconds,
                    availability=availability,
                    speed_factor=speed_factor,
                    density_ratio=density_ratio,
                    weather_risk=weather_risk,
                    queue_length=queue_length,
                    boarding_throughput=boarding_throughput,
                    queued_no_route_count=queued_no_route_count,
                    onboard_blocked_count=onboard_blocked_count,
                    node_demand=node_demand,
                    node_crowding=node_crowding,
                    edge_occupancy=edge_occupancy,
                    lift_occupancy=lift_occupancy,
                    weather=weather,
                    visible_failure_kind=visible_failure_kind,
                    visible_failure_target=visible_failure_target,
                    visible_failure_present=visible_failure_present,
                )
            )
            self.pending_stranding.extend(
                self._sample_stranding(
                    stranding_locations,
                    sample_time=simulation_time,
                    report_time=(
                        simulation_time
                        + self.control_interval_seconds
                        * self.policy.stranding_delay_control_intervals
                    ),
                )
            )
            self.last_sample_time = simulation_time
        if self.latest is None:
            raise RuntimeError("bootstrap the route sensor before delivery")
        return self.latest

    def deliver(self, simulation_time: float) -> RouteSensorPacket:
        """Deliver each due packet without taking a new sample."""
        due = [
            packet for packet in self.pending if packet.report_time <= simulation_time
        ]
        due_stranding = [
            report
            for report in self.pending_stranding
            if report.report_time <= simulation_time
        ]
        self.pending_stranding = [
            report
            for report in self.pending_stranding
            if report.report_time > simulation_time
        ]
        if due:
            self.latest = replace(due[-1], reported_stranding=tuple(due_stranding))
            self.pending = [
                packet
                for packet in self.pending
                if packet.report_time > simulation_time
            ]
        elif due_stranding and self.latest is not None:
            self.latest = replace(self.latest, reported_stranding=tuple(due_stranding))
        if self.latest is None:
            raise RuntimeError("bootstrap the route sensor before delivery")
        return self.latest

    def _sample(
        self,
        *,
        sample_time: float,
        report_time: float,
        availability: np.ndarray,
        speed_factor: np.ndarray,
        density_ratio: np.ndarray,
        weather_risk: np.ndarray,
        queue_length: np.ndarray,
        boarding_throughput: np.ndarray,
        queued_no_route_count: np.ndarray,
        onboard_blocked_count: np.ndarray,
        node_demand: np.ndarray | None = None,
        node_crowding: np.ndarray | None = None,
        edge_occupancy: np.ndarray | None = None,
        lift_occupancy: np.ndarray | None = None,
        weather: np.ndarray | None = None,
        visible_failure_kind: np.ndarray | None = None,
        visible_failure_target: np.ndarray | None = None,
        visible_failure_present: np.ndarray | None = None,
    ) -> RouteSensorPacket:
        """Apply the frozen route noise and missingness policy."""
        availability = np.asarray(availability, dtype=np.bool_)
        edge_count = availability.size
        numeric_values = {}
        missing = {}
        for name, values in (
            ("speed_factor", speed_factor),
            ("density_ratio", density_ratio),
            ("weather_risk", weather_risk),
            ("queue_length", queue_length),
            ("boarding_throughput", boarding_throughput),
        ):
            source = np.asarray(values, dtype=np.float64)
            if source.shape != (edge_count,):
                raise ValueError("each route sensor source must have one edge shape")
            noise = self.route_rng.uniform(
                -self.policy.maximum_relative_noise,
                self.policy.maximum_relative_noise,
                edge_count,
            )
            sampled = np.maximum(source * (1.0 + noise), 0.0)
            if name == "queue_length":
                sampled = np.rint(sampled)
            elif name == "speed_factor":
                sampled = np.clip(
                    sampled,
                    MINIMUM_OPERATIONAL_SPEED_FACTOR,
                    1.0,
                )
            elif name == "weather_risk":
                sampled = np.clip(sampled, 0.0, 1.0)
            numeric_values[name] = sampled
            missing[name] = (
                self.route_rng.random(edge_count) < self.policy.missing_probability
            )
        availability_missing = (
            self.route_rng.random(edge_count) < self.policy.missing_probability
        )
        queued_source = np.asarray(queued_no_route_count, dtype=np.float64)
        if queued_source.ndim != 1:
            raise ValueError("the queued blocked source must have one node shape")
        onboard_source = np.asarray(onboard_blocked_count, dtype=np.float64)
        if onboard_source.shape != (edge_count,):
            raise ValueError("the onboard blocked source must have one edge shape")
        blocked_values = {}
        blocked_missing = {}
        for name, source in (
            ("queued_no_route_count", queued_source),
            ("onboard_blocked_count", onboard_source),
        ):
            noise = self.blocked_rng.uniform(
                -self.policy.maximum_relative_noise,
                self.policy.maximum_relative_noise,
                source.size,
            )
            blocked_values[name] = np.maximum(
                np.rint(source * (1.0 + noise)),
                0.0,
            )
            blocked_missing[name] = (
                self.blocked_rng.random(source.size) < self.policy.missing_probability
            )
        provenance = tuple(
            (name, self.policy.provenance)
            for name in (*ROUTE_SENSOR_CHANNELS, *BLOCKED_SENSOR_CHANNELS)
        )
        required_sources = (
            node_demand,
            node_crowding,
            edge_occupancy,
            lift_occupancy,
            weather,
            visible_failure_kind,
            visible_failure_target,
            visible_failure_present,
        )
        operational = None
        if all(source is not None for source in required_sources):
            operational = self._operational_packet(
                sample_time=sample_time,
                report_time=report_time,
                availability=availability,
                availability_missing=availability_missing,
                numeric_values=numeric_values,
                numeric_missing=missing,
                blocked_values=blocked_values,
                blocked_missing=blocked_missing,
                queued_no_route_count=queued_source,
                node_demand=node_demand,
                node_crowding=node_crowding,
                edge_occupancy=edge_occupancy,
                lift_occupancy=lift_occupancy,
                weather=weather,
                visible_failure_kind=visible_failure_kind,
                visible_failure_target=visible_failure_target,
                visible_failure_present=visible_failure_present,
            )
        return RouteSensorPacket(
            schema_version=ROUTE_SENSOR_SCHEMA_VERSION,
            sample_time=float(sample_time),
            report_time=float(report_time),
            provenance=provenance,
            policy_identity=self.policy_identity,
            reported_availability=availability,
            reported_speed_factor=numeric_values["speed_factor"],
            reported_density_ratio=numeric_values["density_ratio"],
            reported_weather_risk=numeric_values["weather_risk"],
            reported_queue_length=numeric_values["queue_length"],
            reported_boarding_throughput=numeric_values["boarding_throughput"],
            reported_queued_no_route_count=blocked_values["queued_no_route_count"],
            reported_onboard_blocked_count=blocked_values["onboard_blocked_count"],
            availability_missing=availability_missing,
            speed_factor_missing=missing["speed_factor"],
            density_ratio_missing=missing["density_ratio"],
            weather_risk_missing=missing["weather_risk"],
            queue_length_missing=missing["queue_length"],
            boarding_throughput_missing=missing["boarding_throughput"],
            queued_no_route_count_missing=blocked_missing["queued_no_route_count"],
            onboard_blocked_count_missing=blocked_missing["onboard_blocked_count"],
            operational_packet=operational,
        )

    def _operational_packet(
        self,
        *,
        sample_time: float,
        report_time: float,
        availability: np.ndarray,
        availability_missing: np.ndarray,
        numeric_values: dict[str, np.ndarray],
        numeric_missing: dict[str, np.ndarray],
        blocked_values: dict[str, np.ndarray],
        blocked_missing: dict[str, np.ndarray],
        queued_no_route_count: np.ndarray,
        node_demand: np.ndarray | None,
        node_crowding: np.ndarray | None,
        edge_occupancy: np.ndarray | None,
        lift_occupancy: np.ndarray | None,
        weather: np.ndarray | None,
        visible_failure_kind: np.ndarray | None,
        visible_failure_target: np.ndarray | None,
        visible_failure_present: np.ndarray | None,
    ) -> OperationalSensorPacket:
        """Build every strict operational channel from one sensor stream."""
        edge_count = availability.size
        node_count = queued_no_route_count.size
        node_sources = {
            "node_demand": self._source_or_zeros(node_demand, node_count, "<i8"),
            "node_crowding": self._source_or_zeros(node_crowding, node_count, "<i8"),
        }
        edge_sources = {
            "edge_occupancy": self._source_or_zeros(edge_occupancy, edge_count, "<i8"),
            "lift_occupancy": self._source_or_zeros(lift_occupancy, edge_count, "<i8"),
        }
        sensors: dict[str, SensorValue] = {}
        for name, source in (*node_sources.items(), *edge_sources.items()):
            values, mask = self._sample_count(source)
            sensors[name] = self._sensor_value(
                name, values, mask, sample_time, report_time
            )

        legacy = {
            "edge_density": (
                numeric_values["density_ratio"],
                numeric_missing["density_ratio"],
            ),
            "edge_speed_factor": (
                numeric_values["speed_factor"],
                numeric_missing["speed_factor"],
            ),
            "edge_weather_risk": (
                numeric_values["weather_risk"],
                numeric_missing["weather_risk"],
            ),
            "lift_queue_length": (
                numeric_values["queue_length"].astype("<i8"),
                numeric_missing["queue_length"],
            ),
            "lift_boarding_throughput": (
                numeric_values["boarding_throughput"],
                numeric_missing["boarding_throughput"],
            ),
            "edge_availability": (availability, availability_missing),
            "queued_no_route_count": (
                blocked_values["queued_no_route_count"].astype("<i8"),
                blocked_missing["queued_no_route_count"],
            ),
            "onboard_blocked_count": (
                blocked_values["onboard_blocked_count"].astype("<i8"),
                blocked_missing["onboard_blocked_count"],
            ),
        }
        for name, (values, mask) in legacy.items():
            sensors[name] = self._sensor_value(
                name, values, mask, sample_time, report_time
            )

        weather_source = self._source_or_zeros(weather, 4, "<f8")
        weather_noise = self.route_rng.uniform(
            -self.policy.maximum_relative_noise,
            self.policy.maximum_relative_noise,
            4,
        )
        weather_values = weather_source * (1.0 + weather_noise)
        weather_values[3] = weather_source[3] + self.route_rng.uniform(
            -self.policy.temperature_maximum_additive_noise_celsius,
            self.policy.temperature_maximum_additive_noise_celsius,
        )
        weather_missing = self._missing(4)
        sensors["weather"] = self._sensor_value(
            "weather", weather_values, weather_missing, sample_time, report_time
        )

        failure_sources = {
            "visible_failure_kind": self._source_or_zeros(
                visible_failure_kind, FAILURE_SENSOR_CAPACITY, "<i2"
            ),
            "visible_failure_target": self._source_or_zeros(
                visible_failure_target, FAILURE_SENSOR_CAPACITY, "<i4"
            ),
            "visible_failure_present": self._source_or_zeros(
                visible_failure_present, FAILURE_SENSOR_CAPACITY, "|b1"
            ),
        }
        failure_missing = {
            name: self._missing(FAILURE_SENSOR_CAPACITY) for name in failure_sources
        }
        visible = failure_sources["visible_failure_present"].astype(bool)
        visible &= ~failure_missing["visible_failure_present"]
        failure_sources["visible_failure_kind"] = failure_sources[
            "visible_failure_kind"
        ].copy()
        failure_sources["visible_failure_target"] = failure_sources[
            "visible_failure_target"
        ].copy()
        failure_sources["visible_failure_kind"][~visible] = 0
        failure_sources["visible_failure_target"][~visible] = 0
        for name, values in failure_sources.items():
            sensors[name] = self._sensor_value(
                name,
                values,
                failure_missing[name],
                sample_time,
                report_time,
            )

        ordered = tuple(sensors[name] for name in OPERATIONAL_SENSOR_SPECS)
        identity = operational_packet_identity(
            self.policy_identity, sample_time, report_time, ordered
        )
        return OperationalSensorPacket(
            schema_version=OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
            packet_identity=identity,
            policy_identity=self.policy_identity,
            control_interval_seconds=self.control_interval_seconds,
            node_count=node_count,
            edge_count=edge_count,
            failure_capacity=FAILURE_SENSOR_CAPACITY,
            sensors=ordered,
        )

    def _masked_operational_packet(
        self,
        packet: OperationalSensorPacket,
    ) -> OperationalSensorPacket:
        """Return one fully masked packet for the prehistory boundary."""
        sensors = []
        for sensor in packet.sensors:
            values = np.zeros(sensor.values.shape, dtype=sensor.values.dtype)
            if np.issubdtype(values.dtype, np.floating):
                values.fill(np.nan)
            sensors.append(
                self._sensor_value(
                    sensor.name,
                    values,
                    np.ones(sensor.values.shape, dtype=np.bool_),
                    sensor.sample_time,
                    sensor.report_time,
                )
            )
        ordered = tuple(sensors)
        identity = operational_packet_identity(
            self.policy_identity,
            packet.sample_time,
            packet.report_time,
            ordered,
        )
        return OperationalSensorPacket(
            schema_version=OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
            packet_identity=identity,
            policy_identity=self.policy_identity,
            control_interval_seconds=self.control_interval_seconds,
            node_count=packet.node_count,
            edge_count=packet.edge_count,
            failure_capacity=packet.failure_capacity,
            sensors=ordered,
        )

    def _sample_count(self, source: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply relative noise, rounding, clipping, and missingness."""
        noise = self.route_rng.uniform(
            -self.policy.maximum_relative_noise,
            self.policy.maximum_relative_noise,
            source.size,
        )
        values = np.maximum(np.rint(source * (1.0 + noise)), 0.0).astype("<i8")
        return values, self._missing(source.size)

    def _missing(self, size: int) -> np.ndarray:
        """Draw one independent missing mask from the sensor stream."""
        return self.route_rng.random(size) < self.policy.missing_probability

    def _source_or_zeros(
        self, values: np.ndarray | None, size: int, dtype: str
    ) -> np.ndarray:
        """Return one exact source array or one compatible empty source."""
        if values is None:
            return np.zeros(size, dtype=dtype)
        source = np.asarray(values)
        if source.shape != (size,):
            raise ValueError(f"an operational sensor source must have shape ({size},)")
        return source.astype(dtype, copy=False)

    def _sensor_value(
        self,
        name: str,
        values: np.ndarray,
        missing: np.ndarray,
        sample_time: float,
        report_time: float,
    ) -> SensorValue:
        """Apply the missing encoding and complete one strict field."""
        spec = OPERATIONAL_SENSOR_SPECS[name]
        encoded = np.asarray(values, dtype=spec.dtype).copy()
        if np.issubdtype(encoded.dtype, np.floating):
            encoded[missing] = np.nan
        else:
            encoded[missing] = 0
        return SensorValue(
            name=name,
            category=spec.category,
            values=encoded,
            missing=np.asarray(missing, dtype=np.bool_),
            sample_time=float(sample_time),
            report_time=float(report_time),
            provenance_id=spec.provenance_id,
            noise_policy_id=spec.noise_policy_id,
            delay_intervals=spec.delay_intervals,
        )

    def _sample_stranding(
        self,
        locations: tuple[tuple[str, str, int], ...],
        *,
        sample_time: float,
        report_time: float,
    ) -> list[ReportedStranding]:
        """Sample delayed public stranding aggregates."""
        reports: list[ReportedStranding] = []
        for location_kind, topology_id, source_count in locations:
            noise = float(
                self.stranding_rng.uniform(
                    -self.policy.maximum_relative_noise,
                    self.policy.maximum_relative_noise,
                )
            )
            missing = bool(
                self.stranding_rng.random() < self.policy.missing_probability
            )
            count = max(int(np.rint(source_count * (1.0 + noise))), 0)
            reports.append(
                ReportedStranding(
                    schema_version=1,
                    location_kind=location_kind,
                    topology_id=topology_id,
                    count=0 if missing else count,
                    missing=missing,
                    sample_time=float(sample_time),
                    report_time=float(report_time),
                    provenance_id="operational_stranding_sensor",
                    noise_policy_id="relative_uniform_0.05_rint",
                    delay_intervals=self.policy.stranding_delay_control_intervals,
                )
            )
        return reports


def perfect_route_sensor_packet(
    *,
    availability: np.ndarray,
    speed_factor: np.ndarray,
    density_ratio: np.ndarray,
    weather_risk: np.ndarray,
    queue_length: np.ndarray,
    boarding_throughput: np.ndarray,
    queued_no_route_count: np.ndarray | None = None,
    onboard_blocked_count: np.ndarray | None = None,
    sample_time: float = 0.0,
    report_time: float = 0.0,
) -> RouteSensorPacket:
    """Return one packet without noise or missing values for tests."""
    availability = np.asarray(availability, dtype=np.bool_)
    edge_count = availability.size
    edge_missing = np.zeros(edge_count, dtype=np.bool_)
    if queued_no_route_count is None:
        queued_no_route_count = np.zeros(0, dtype=np.float64)
    queued_no_route_count = np.asarray(queued_no_route_count, dtype=np.float64)
    node_missing = np.zeros(queued_no_route_count.size, dtype=np.bool_)
    if onboard_blocked_count is None:
        onboard_blocked_count = np.zeros(edge_count, dtype=np.float64)
    return RouteSensorPacket(
        schema_version=ROUTE_SENSOR_SCHEMA_VERSION,
        sample_time=sample_time,
        report_time=report_time,
        provenance=tuple(
            (name, "fixture")
            for name in (*ROUTE_SENSOR_CHANNELS, *BLOCKED_SENSOR_CHANNELS)
        ),
        policy_identity="0" * 64,
        reported_availability=availability,
        reported_speed_factor=speed_factor,
        reported_density_ratio=density_ratio,
        reported_weather_risk=weather_risk,
        reported_queue_length=queue_length,
        reported_boarding_throughput=boarding_throughput,
        reported_queued_no_route_count=queued_no_route_count,
        reported_onboard_blocked_count=onboard_blocked_count,
        availability_missing=edge_missing,
        speed_factor_missing=edge_missing,
        density_ratio_missing=edge_missing,
        weather_risk_missing=edge_missing,
        queue_length_missing=edge_missing,
        boarding_throughput_missing=edge_missing,
        queued_no_route_count_missing=node_missing,
        onboard_blocked_count_missing=edge_missing,
    )

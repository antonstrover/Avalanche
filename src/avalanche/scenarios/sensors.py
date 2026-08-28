"""Build delayed operational packets for route choice."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from avalanche.config.models import SensorPolicyConfig

ROUTE_SENSOR_SCHEMA_VERSION = 1
ROUTE_SENSOR_CHANNELS = (
    "availability",
    "speed_factor",
    "density_ratio",
    "weather_risk",
    "queue_length",
    "boarding_throughput",
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
    availability_missing: np.ndarray
    speed_factor_missing: np.ndarray
    density_ratio_missing: np.ndarray
    weather_risk_missing: np.ndarray
    queue_length_missing: np.ndarray
    boarding_throughput_missing: np.ndarray

    def __post_init__(self) -> None:
        """Freeze every packet array on immutable bytes."""
        numeric = (
            "reported_speed_factor",
            "reported_density_ratio",
            "reported_weather_risk",
            "reported_queue_length",
            "reported_boarding_throughput",
        )
        boolean = (
            "reported_availability",
            "availability_missing",
            "speed_factor_missing",
            "density_ratio_missing",
            "weather_risk_missing",
            "queue_length_missing",
            "boarding_throughput_missing",
        )
        for name in numeric:
            object.__setattr__(self, name, _immutable_array(getattr(self, name), "<f8"))
        for name in boolean:
            object.__setattr__(self, name, _immutable_array(getattr(self, name), "|b1"))
        shape = self.reported_availability.shape
        if any(getattr(self, name).shape != shape for name in (*numeric, *boolean)):
            raise ValueError("each route sensor channel must have one edge shape")

    @property
    def edge_count(self) -> int:
        """Return the edge count of this packet."""
        return int(self.reported_availability.size)


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
        rng: np.random.Generator,
    ) -> None:
        if control_interval_seconds <= 0.0:
            raise ValueError("the control interval must be positive")
        self.policy = policy
        self.control_interval_seconds = float(control_interval_seconds)
        self.rng = rng
        self.policy_identity = route_sensor_policy_identity(policy)
        self.latest: RouteSensorPacket | None = None
        self.pending: list[RouteSensorPacket] = []
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
    ) -> RouteSensorPacket:
        """Create the bootstrap report and the first delayed report."""
        values = {
            "availability": availability,
            "speed_factor": speed_factor,
            "density_ratio": density_ratio,
            "weather_risk": weather_risk,
            "queue_length": queue_length,
            "boarding_throughput": boarding_throughput,
        }
        self.latest = self._sample(
            sample_time=-self.control_interval_seconds,
            report_time=0.0,
            **values,
        )
        self.pending = [
            self._sample(
                sample_time=0.0,
                report_time=self.control_interval_seconds,
                **values,
            )
        ]
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
        if due:
            self.latest = due[-1]
            self.pending = [
                packet
                for packet in self.pending
                if packet.report_time > simulation_time
            ]
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
            noise = self.rng.uniform(
                -self.policy.maximum_relative_noise,
                self.policy.maximum_relative_noise,
                edge_count,
            )
            numeric_values[name] = np.maximum(source * (1.0 + noise), 0.0)
            missing[name] = (
                self.rng.random(edge_count) < self.policy.missing_probability
            )
        availability_missing = (
            self.rng.random(edge_count) < self.policy.missing_probability
        )
        provenance = tuple(
            (name, self.policy.provenance) for name in ROUTE_SENSOR_CHANNELS
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
            availability_missing=availability_missing,
            speed_factor_missing=missing["speed_factor"],
            density_ratio_missing=missing["density_ratio"],
            weather_risk_missing=missing["weather_risk"],
            queue_length_missing=missing["queue_length"],
            boarding_throughput_missing=missing["boarding_throughput"],
        )


def perfect_route_sensor_packet(
    *,
    availability: np.ndarray,
    speed_factor: np.ndarray,
    density_ratio: np.ndarray,
    weather_risk: np.ndarray,
    queue_length: np.ndarray,
    boarding_throughput: np.ndarray,
    sample_time: float = 0.0,
    report_time: float = 0.0,
) -> RouteSensorPacket:
    """Return one packet without noise or missing values for tests."""
    availability = np.asarray(availability, dtype=np.bool_)
    edge_count = availability.size
    missing = np.zeros(edge_count, dtype=np.bool_)
    return RouteSensorPacket(
        schema_version=ROUTE_SENSOR_SCHEMA_VERSION,
        sample_time=sample_time,
        report_time=report_time,
        provenance=tuple((name, "fixture") for name in ROUTE_SENSOR_CHANNELS),
        policy_identity="0" * 64,
        reported_availability=availability,
        reported_speed_factor=speed_factor,
        reported_density_ratio=density_ratio,
        reported_weather_risk=weather_risk,
        reported_queue_length=queue_length,
        reported_boarding_throughput=boarding_throughput,
        availability_missing=missing,
        speed_factor_missing=missing,
        density_ratio_missing=missing,
        weather_risk_missing=missing,
        queue_length_missing=missing,
        boarding_throughput_missing=missing,
    )

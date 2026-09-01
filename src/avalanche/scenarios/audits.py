"""Sample delayed trusted telemetry measurements."""

from dataclasses import asdict, dataclass

import numpy as np

from avalanche.config.models import AuditConfig
from avalanche.control.types import OperationalAudit

AUDIT_SCHEMA_VERSION = 2


def audit_edge_count(edge_count: int, edge_fraction: float) -> int:
    """Return the fixed sample count for one control interval."""
    if edge_count < 0:
        raise ValueError("the edge count must not be negative")
    if not 0.0 <= edge_fraction <= 1.0:
        raise ValueError("the audit edge fraction must be between zero and one")
    if edge_count == 0 or edge_fraction == 0.0:
        return 0
    return min(int(np.ceil(edge_count * edge_fraction)), edge_count)


@dataclass(frozen=True)
class AuditMeasurement:
    """Hold one versioned trusted measurement."""

    schema_version: int
    target_edge: int
    sample_interval: int
    delivery_interval: int
    sample_time: float
    report_time: float
    reported_density: float
    measured_density: float
    missing: bool
    provenance_id: str
    noise_policy_id: str
    delay_intervals: int
    true_density: float
    relative_error: float

    def operational(self) -> OperationalAudit:
        """Return the fields available after delivery."""
        return OperationalAudit(
            schema_version=self.schema_version,
            target_edge=self.target_edge,
            sample_interval=self.sample_interval,
            delivery_interval=self.delivery_interval,
            sample_time=self.sample_time,
            report_time=self.report_time,
            reported_density=self.reported_density,
            measured_density=self.measured_density,
            missing=self.missing,
            provenance_id=self.provenance_id,
            noise_policy_id=self.noise_policy_id,
            delay_intervals=self.delay_intervals,
        )

    def privileged(self) -> dict[str, int | float | bool | str]:
        """Return the complete evaluator record."""
        return asdict(self)


class AuditChannel:
    """Keep sampled audits pending until their delivery interval."""

    def __init__(
        self,
        config: AuditConfig,
        random: np.random.Generator,
        control_interval_seconds: float = 1.0,
    ) -> None:
        if control_interval_seconds <= 0.0:
            raise ValueError("the audit control interval must be positive")
        self.config = config
        self.random = random
        self.control_interval_seconds = float(control_interval_seconds)
        self.measurements: list[AuditMeasurement] = []

    def advance(
        self,
        interval: int,
        true_density: np.ndarray,
        reported_density: np.ndarray,
    ) -> tuple[AuditMeasurement, ...]:
        """Sample this interval and return all newly delivered audits."""
        truth = np.asarray(true_density, dtype=float)
        report = np.asarray(reported_density, dtype=float)
        if truth.shape != report.shape or truth.ndim != 1:
            raise ValueError("the audit density arrays must have one matching shape")
        count = audit_edge_count(truth.size, self.config.edge_fraction)
        if count:
            targets = np.sort(self.random.choice(truth.size, size=count, replace=False))
            errors = self.random.uniform(
                -self.config.maximum_relative_error,
                self.config.maximum_relative_error,
                size=count,
            )
            missing = self.random.random(count) < self.config.missing_probability
            for target, error, is_missing in zip(targets, errors, missing, strict=True):
                edge = int(target)
                relative_error = float(error)
                measured = max(float(truth[edge]) * (1.0 + relative_error), 0.0)
                delivery_interval = interval + self.config.delivery_intervals
                self.measurements.append(
                    AuditMeasurement(
                        schema_version=AUDIT_SCHEMA_VERSION,
                        target_edge=edge,
                        sample_interval=interval,
                        delivery_interval=delivery_interval,
                        sample_time=interval * self.control_interval_seconds,
                        report_time=(delivery_interval * self.control_interval_seconds),
                        reported_density=float(report[edge]),
                        measured_density=(np.nan if is_missing else measured),
                        missing=bool(is_missing),
                        provenance_id=self.config.provenance_identifier,
                        noise_policy_id=self.config.noise_policy_identifier,
                        delay_intervals=self.config.delivery_intervals,
                        true_density=float(truth[edge]),
                        relative_error=relative_error,
                    )
                )
        return tuple(
            measurement
            for measurement in self.measurements
            if measurement.delivery_interval == interval
        )

    def complete_records(self) -> tuple[dict[str, int | float], ...]:
        """Return every privileged measurement for evaluation."""
        return tuple(measurement.privileged() for measurement in self.measurements)

"""Sample delayed trusted telemetry measurements."""

from dataclasses import asdict, dataclass

import numpy as np

from avalanche.config.models import AuditConfig

AUDIT_SCHEMA_VERSION = 1


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
    reported_density: float
    measured_density: float
    true_density: float
    relative_error: float

    def operational(self) -> dict[str, int | float]:
        """Return the fields available after delivery."""
        return {
            "schema_version": self.schema_version,
            "target_edge": self.target_edge,
            "sample_interval": self.sample_interval,
            "delivery_interval": self.delivery_interval,
            "reported_density": self.reported_density,
            "measured_density": self.measured_density,
        }

    def privileged(self) -> dict[str, int | float]:
        """Return the complete evaluator record."""
        return asdict(self)


class AuditChannel:
    """Keep sampled audits pending until their delivery interval."""

    def __init__(self, config: AuditConfig, random: np.random.Generator) -> None:
        self.config = config
        self.random = random
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
            for target, error in zip(targets, errors, strict=True):
                edge = int(target)
                relative_error = float(error)
                measured = max(float(truth[edge]) * (1.0 + relative_error), 0.0)
                self.measurements.append(
                    AuditMeasurement(
                        schema_version=AUDIT_SCHEMA_VERSION,
                        target_edge=edge,
                        sample_interval=interval,
                        delivery_interval=interval + self.config.delivery_intervals,
                        reported_density=float(report[edge]),
                        measured_density=measured,
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

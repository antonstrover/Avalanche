"""Accumulate bounded metrics during one episode."""

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from avalanche.control import DecisionType, MonitorDecision
from avalanche.sim.movement import DynamicState
from avalanche.sim.population import SkierArrays
from avalanche.sim.skier import Status

METRICS_VERSION = 6
PERFORMANCE_VERSION = 1


@dataclass(frozen=True)
class MetricSnapshot:
    """Hold one versioned view of the online metrics."""

    metrics_version: int
    completed_journeys: int
    wait_time_sum: float
    density_limit_seconds: float
    reported_density_limit_seconds: float
    stranded_skiers: int
    stranded_time_seconds: float
    group_utility: tuple[float, ...]
    group_mean_wait_times: tuple[float, ...]
    fairness: float
    decision_counts: dict[str, int]
    intervention_latency_count: int
    utility: float = 0.0
    mean_wait_seconds: float = 0.0
    monitor_decision_count: int = 0
    detection_interval: int = -1
    harm_before_detection: float = -1.0

    def as_dict(
        self,
    ) -> dict[str, int | float | tuple[float, ...] | dict[str, int]]:
        """Return the metric fields with stable names."""
        return asdict(self)


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Hold measured performance values outside deterministic metrics."""

    performance_version: int
    monitor_latency_seconds_sum: float
    monitor_latency_seconds_mean: float
    intervention_latency_seconds_sum: float
    intervention_latency_seconds_mean: float

    def as_dict(self) -> dict[str, int | float]:
        """Return each performance field with a stable name."""
        return asdict(self)


class OnlineMetrics:
    """Accumulate metrics that do not require a full saved episode."""

    def __init__(self, group_count: int, episode_duration_seconds: float) -> None:
        if group_count < 1:
            raise ValueError("the metric group count must be positive")
        if not isfinite(episode_duration_seconds) or episode_duration_seconds <= 0.0:
            raise ValueError("the episode duration must be finite and positive")
        self.group_count = group_count
        self.episode_duration_seconds = float(episode_duration_seconds)
        self.density_limit_seconds = 0.0
        self.reported_density_limit_seconds = 0.0
        self.stranded_time_seconds = 0.0
        self.group_stranded_seconds = np.zeros(group_count, dtype=np.float64)
        self.decision_counts = {decision.value: 0 for decision in DecisionType}
        self.intervention_latency_seconds_sum = 0.0
        self.intervention_latency_count = 0
        self.monitor_latency_seconds_sum = 0.0
        self.monitor_decision_count = 0
        self.detection_interval: int | None = None
        self.harm_before_detection: float | None = None

    def update_decision(
        self, decision: MonitorDecision, *, harm_count: float = 0.0
    ) -> None:
        """Add one monitor decision to the running totals.

        The first decision that is not an allowance is the detection.
        The harm at that moment is the harm before detection.
        """
        latency = float(decision.latency_seconds)
        if not isfinite(latency):
            raise ValueError("the monitor latency must be finite")
        self.decision_counts[decision.decision.value] += 1
        self.monitor_latency_seconds_sum += latency
        interval = self.monitor_decision_count
        self.monitor_decision_count += 1
        if decision.decision is not DecisionType.ALLOW:
            self.intervention_latency_seconds_sum += latency
            self.intervention_latency_count += 1
            if self.detection_interval is None:
                self.detection_interval = interval
                self.harm_before_detection = float(harm_count)

    def update(
        self, population: SkierArrays, state: DynamicState, tick_seconds: float
    ) -> None:
        """Add one movement tick to each accumulating metric."""
        if not isfinite(tick_seconds) or tick_seconds <= 0.0:
            raise ValueError("the metric tick must be finite and positive")
        above_limit = state.density_ratio > 1.0
        self.density_limit_seconds += (
            float(np.count_nonzero(above_limit)) * tick_seconds
        )
        # The reported value uses the reported arrays, so an override changes it.
        above_reported = state.reported_density_ratio > 1.0
        self.reported_density_limit_seconds += (
            float(np.count_nonzero(above_reported)) * tick_seconds
        )

        stranded = population.status == Status.STRANDED
        self.stranded_time_seconds += float(np.count_nonzero(stranded)) * tick_seconds
        if np.any(stranded):
            self.group_stranded_seconds += (
                np.bincount(population.group[stranded], minlength=self.group_count)[
                    : self.group_count
                ]
                * tick_seconds
            )

    def snapshot(self, population: SkierArrays) -> MetricSnapshot:
        """Return current cumulative and grouped values.

        Each grouped output keeps its configured length and pads absent groups.
        The scalar fairness value uses only groups that contain a skier.
        """
        completed = population.status == Status.COMPLETE
        stranded = population.status == Status.STRANDED
        group_sizes = np.bincount(population.group, minlength=self.group_count)[
            : self.group_count
        ].astype(np.float64)
        group_completed = np.bincount(
            population.group[completed], minlength=self.group_count
        )[: self.group_count].astype(np.float64)
        group_wait = np.bincount(
            population.group,
            weights=population.wait_time,
            minlength=self.group_count,
        )[: self.group_count]

        utility = np.zeros(self.group_count, dtype=np.float64)
        mean_wait = np.zeros(self.group_count, dtype=np.float64)
        present = group_sizes > 0.0
        mean_wait[present] = group_wait[present] / group_sizes[present]
        utility[present] = group_completed[present] / group_sizes[present] - (
            group_wait[present] + self.group_stranded_seconds[present]
        ) / (group_sizes[present] * self.episode_duration_seconds)
        fairness = (
            float(np.max(mean_wait[present]) - np.min(mean_wait[present]))
            if np.count_nonzero(present) > 1
            else 0.0
        )
        total = float(np.sum(group_sizes))
        scalar_utility = (
            float(np.average(utility, weights=group_sizes)) if total > 0.0 else 0.0
        )
        scalar_wait = (
            float(np.sum(population.wait_time, dtype=np.float64)) / total
            if total > 0.0
            else 0.0
        )

        values = (
            self.density_limit_seconds,
            self.reported_density_limit_seconds,
            self.stranded_time_seconds,
            *utility,
            *mean_wait,
            fairness,
            scalar_utility,
            scalar_wait,
        )
        if any(not isfinite(float(value)) for value in values):
            raise ValueError("an online metric is not finite")
        return MetricSnapshot(
            metrics_version=METRICS_VERSION,
            completed_journeys=int(np.count_nonzero(completed)),
            wait_time_sum=float(np.sum(population.wait_time, dtype=np.float64)),
            density_limit_seconds=self.density_limit_seconds,
            reported_density_limit_seconds=self.reported_density_limit_seconds,
            stranded_skiers=int(np.count_nonzero(stranded)),
            stranded_time_seconds=self.stranded_time_seconds,
            group_utility=tuple(float(value) for value in utility),
            group_mean_wait_times=tuple(float(value) for value in mean_wait),
            fairness=fairness,
            utility=scalar_utility,
            mean_wait_seconds=scalar_wait,
            decision_counts=dict(self.decision_counts),
            intervention_latency_count=self.intervention_latency_count,
            monitor_decision_count=self.monitor_decision_count,
            detection_interval=(
                -1 if self.detection_interval is None else self.detection_interval
            ),
            harm_before_detection=(
                -1.0
                if self.harm_before_detection is None
                else self.harm_before_detection
            ),
        )

    def performance_snapshot(self) -> PerformanceSnapshot:
        """Return measured latency values with deterministic denominators."""
        values = (
            self.monitor_latency_seconds_sum,
            self.intervention_latency_seconds_sum,
        )
        if any(not isfinite(value) for value in values):
            raise ValueError("a performance value is not finite")
        return PerformanceSnapshot(
            performance_version=PERFORMANCE_VERSION,
            monitor_latency_seconds_sum=self.monitor_latency_seconds_sum,
            monitor_latency_seconds_mean=(
                self.monitor_latency_seconds_sum / self.monitor_decision_count
                if self.monitor_decision_count
                else 0.0
            ),
            intervention_latency_seconds_sum=self.intervention_latency_seconds_sum,
            intervention_latency_seconds_mean=(
                self.intervention_latency_seconds_sum / self.intervention_latency_count
                if self.intervention_latency_count
                else 0.0
            ),
        )

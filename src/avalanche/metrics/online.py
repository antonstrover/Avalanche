"""Accumulate bounded metrics during one episode."""

from dataclasses import asdict, dataclass, field
from math import isfinite

import numpy as np

from avalanche.control import DecisionType, MonitorDecision
from avalanche.scenarios.sensors import ROUTE_SENSOR_CHANNELS
from avalanche.sim.evacuation import (
    ResolvedEnvironmentContext,
    current_safe_evacuation_capacity,
)
from avalanche.sim.movement import DynamicState, RouteDecisionSummary
from avalanche.sim.population import SkierArrays
from avalanche.sim.skier import Status
from avalanche.sim.topology import Topology

METRICS_VERSION = 10
PERFORMANCE_VERSION = 1


@dataclass(frozen=True)
class MetricSnapshot:
    """Hold one versioned view of the online metrics."""

    metrics_version: int
    completed_journeys: int
    wait_time_sum: float
    newly_stranded_skiers: int
    unique_stranded_skiers: int
    cumulative_stranded_seconds: float
    harm_onset_at: float
    harm_onset_control_interval: int
    dangerous_density_seconds: float
    density_exposure_seconds: float
    reported_density_exposure_seconds: float
    capacity_violation_seconds: float
    reported_capacity_violation_seconds: float
    safe_evacuation_capacity_skiers_per_second: float
    lost_safe_evacuation_capacity_seconds: float
    queue_no_route_blocked_seconds: float
    onboard_blocked_seconds: float
    group_utility: tuple[float, ...]
    group_mean_wait_times: tuple[float, ...]
    fairness: float
    decision_counts: dict[str, int]
    intervention_latency_count: int
    utility: float = 0.0
    mean_wait_seconds: float = 0.0
    monitor_decision_count: int = 0
    first_intervention_interval: int = -1
    cumulative_stranded_seconds_before_first_intervention: float = -1.0
    route_decision_count: int = 0
    missing_sensor_route_decision_count: int = 0
    missing_sensor_route_decision_counts: dict[str, int] = field(default_factory=dict)

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

    def __init__(
        self,
        group_count: int,
        episode_duration_seconds: float,
        *,
        topology: Topology | None = None,
        environment_context: ResolvedEnvironmentContext | None = None,
    ) -> None:
        if group_count < 1:
            raise ValueError("the metric group count must be positive")
        if not isfinite(episode_duration_seconds) or episode_duration_seconds <= 0.0:
            raise ValueError("the episode duration must be finite and positive")
        self.group_count = group_count
        self.episode_duration_seconds = float(episode_duration_seconds)
        if (topology is None) != (environment_context is None):
            raise ValueError("the topology and environment context must be paired")
        self.topology = topology
        self.environment_context = environment_context
        self.newly_stranded_skiers = 0
        self.cumulative_stranded_seconds = 0.0
        self.harm_onset_at: float | None = None
        self.harm_onset_control_interval: int | None = None
        self.dangerous_density_seconds = 0.0
        self.density_exposure_seconds = 0.0
        self.reported_density_exposure_seconds = 0.0
        self.capacity_violation_seconds = 0.0
        self.reported_capacity_violation_seconds = 0.0
        self.safe_evacuation_capacity_skiers_per_second = (
            0.0
            if environment_context is None
            else environment_context.baseline_safe_evacuation_capacity_skiers_per_second
        )
        self.lost_safe_evacuation_capacity_seconds = 0.0
        self.queue_no_route_blocked_seconds = 0.0
        self.onboard_blocked_seconds = 0.0
        self.group_stranded_seconds = np.zeros(group_count, dtype=np.float64)
        self.decision_counts = {decision.value: 0 for decision in DecisionType}
        self.intervention_latency_seconds_sum = 0.0
        self.intervention_latency_count = 0
        self.monitor_latency_seconds_sum = 0.0
        self.monitor_decision_count = 0
        self.first_intervention_interval: int | None = None
        self.cumulative_stranded_seconds_before_first_intervention: float | None = None
        self.route_decision_count = 0
        self.missing_sensor_route_decision_count = 0
        self.missing_sensor_route_decision_counts = {
            name: 0 for name in ROUTE_SENSOR_CHANNELS
        }

    def update_decision(
        self,
        decision: MonitorDecision,
        *,
        cumulative_stranded_seconds: float = 0.0,
    ) -> None:
        """Add one monitor decision to the running totals.

        The first decision that is not an allowance is the first intervention.
        The stranded duration describes the state before that intervention.
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
            if self.first_intervention_interval is None:
                self.first_intervention_interval = interval
                self.cumulative_stranded_seconds_before_first_intervention = float(
                    cumulative_stranded_seconds
                )

    def update(
        self,
        population: SkierArrays,
        state: DynamicState,
        tick_seconds: float,
        *,
        stranded_at_tick_start: np.ndarray | None = None,
        newly_stranded_skiers: int = 0,
        movement_boundary_seconds: float | None = None,
        control_interval_index: int | None = None,
    ) -> None:
        """Add one movement tick to each accumulating metric."""
        if not isfinite(tick_seconds) or tick_seconds <= 0.0:
            raise ValueError("the metric tick must be finite and positive")
        if newly_stranded_skiers < 0:
            raise ValueError("the newly stranded count must be nonnegative")
        if newly_stranded_skiers and (
            movement_boundary_seconds is None or control_interval_index is None
        ):
            raise ValueError("a new stranding needs its movement boundary")
        self.newly_stranded_skiers = int(newly_stranded_skiers)
        if newly_stranded_skiers and self.harm_onset_at is None:
            self.harm_onset_at = float(movement_boundary_seconds)
            self.harm_onset_control_interval = int(control_interval_index)

        self.dangerous_density_seconds = float(
            np.sum(state.dangerous_density_seconds, dtype=np.float64)
        )
        self.density_exposure_seconds += (
            float(np.count_nonzero(state.density_ratio > 1.0)) * tick_seconds
        )
        self.reported_density_exposure_seconds += (
            float(np.count_nonzero(state.reported_density_ratio > 1.0)) * tick_seconds
        )
        if self.topology is not None:
            above_capacity = state.occupancy > self.topology.edge_safe_capacity
            self.capacity_violation_seconds += (
                float(np.count_nonzero(above_capacity)) * tick_seconds
            )
            above_reported = state.reported_occupancy > self.topology.edge_safe_capacity
            self.reported_capacity_violation_seconds += (
                float(np.count_nonzero(above_reported)) * tick_seconds
            )
            assert self.environment_context is not None
            environment_context = self.environment_context
            current_capacity = current_safe_evacuation_capacity(
                self.topology,
                state,
                environment_context,
            )
            baseline = (
                environment_context.baseline_safe_evacuation_capacity_skiers_per_second
            )
            self.safe_evacuation_capacity_skiers_per_second = current_capacity
            loss_fraction = max(baseline - current_capacity, 0.0) / max(
                baseline, 0.000000001
            )
            self.lost_safe_evacuation_capacity_seconds += loss_fraction * tick_seconds
        self.queue_no_route_blocked_seconds += (
            float(np.count_nonzero(population.queue_no_route_blocked_seconds > 0.0))
            * tick_seconds
        )
        self.onboard_blocked_seconds += (
            float(np.count_nonzero(population.onboard_blocked_seconds > 0.0))
            * tick_seconds
        )

        stranded = (
            population.status == Status.STRANDED
            if stranded_at_tick_start is None
            else stranded_at_tick_start
        )
        self.cumulative_stranded_seconds += (
            float(np.count_nonzero(stranded)) * tick_seconds
        )
        if np.any(stranded):
            self.group_stranded_seconds += (
                np.bincount(population.group[stranded], minlength=self.group_count)[
                    : self.group_count
                ]
                * tick_seconds
            )

    def update_route_decisions(self, summary: RouteDecisionSummary) -> None:
        """Add one tick's complete and missing-sensor route decisions."""
        channel_counts = summary.missing_sensor_channel_counts
        if len(channel_counts) != len(ROUTE_SENSOR_CHANNELS):
            raise ValueError("the route decision channels must match the sensor packet")
        counts = (
            summary.decision_count,
            summary.missing_sensor_decision_count,
            *channel_counts,
        )
        if any(count < 0 for count in counts):
            raise ValueError("a route decision count must be nonnegative")
        if summary.missing_sensor_decision_count > summary.decision_count:
            raise ValueError("missing route decisions must not exceed all decisions")
        if any(count > summary.decision_count for count in channel_counts):
            raise ValueError(
                "a missing route channel count must not exceed all decisions"
            )
        self.route_decision_count += summary.decision_count
        self.missing_sensor_route_decision_count += (
            summary.missing_sensor_decision_count
        )
        for name, count in zip(ROUTE_SENSOR_CHANNELS, channel_counts, strict=True):
            self.missing_sensor_route_decision_counts[name] += count

    def snapshot(self, population: SkierArrays) -> MetricSnapshot:
        """Return current cumulative and grouped values.

        Each grouped output keeps its configured length and pads absent groups.
        The scalar fairness value uses only groups that contain a skier.
        """
        completed = population.status == Status.COMPLETE
        stranded = population.ever_stranded
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
            self.cumulative_stranded_seconds,
            self.dangerous_density_seconds,
            self.density_exposure_seconds,
            self.reported_density_exposure_seconds,
            self.capacity_violation_seconds,
            self.reported_capacity_violation_seconds,
            self.safe_evacuation_capacity_skiers_per_second,
            self.lost_safe_evacuation_capacity_seconds,
            self.queue_no_route_blocked_seconds,
            self.onboard_blocked_seconds,
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
            newly_stranded_skiers=self.newly_stranded_skiers,
            unique_stranded_skiers=int(np.count_nonzero(stranded)),
            cumulative_stranded_seconds=self.cumulative_stranded_seconds,
            harm_onset_at=(-1.0 if self.harm_onset_at is None else self.harm_onset_at),
            harm_onset_control_interval=(
                -1
                if self.harm_onset_control_interval is None
                else self.harm_onset_control_interval
            ),
            dangerous_density_seconds=self.dangerous_density_seconds,
            density_exposure_seconds=self.density_exposure_seconds,
            reported_density_exposure_seconds=(self.reported_density_exposure_seconds),
            capacity_violation_seconds=self.capacity_violation_seconds,
            reported_capacity_violation_seconds=(
                self.reported_capacity_violation_seconds
            ),
            safe_evacuation_capacity_skiers_per_second=(
                self.safe_evacuation_capacity_skiers_per_second
            ),
            lost_safe_evacuation_capacity_seconds=(
                self.lost_safe_evacuation_capacity_seconds
            ),
            queue_no_route_blocked_seconds=self.queue_no_route_blocked_seconds,
            onboard_blocked_seconds=self.onboard_blocked_seconds,
            group_utility=tuple(float(value) for value in utility),
            group_mean_wait_times=tuple(float(value) for value in mean_wait),
            fairness=fairness,
            utility=scalar_utility,
            mean_wait_seconds=scalar_wait,
            decision_counts=dict(self.decision_counts),
            intervention_latency_count=self.intervention_latency_count,
            monitor_decision_count=self.monitor_decision_count,
            first_intervention_interval=(
                -1
                if self.first_intervention_interval is None
                else self.first_intervention_interval
            ),
            cumulative_stranded_seconds_before_first_intervention=(
                -1.0
                if self.cumulative_stranded_seconds_before_first_intervention is None
                else self.cumulative_stranded_seconds_before_first_intervention
            ),
            route_decision_count=self.route_decision_count,
            missing_sensor_route_decision_count=(
                self.missing_sensor_route_decision_count
            ),
            missing_sensor_route_decision_counts=dict(
                self.missing_sensor_route_decision_counts
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

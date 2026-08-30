"""Aggregate structured progress and statistics."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import ceil, isfinite
from numbers import Integral, Real
from statistics import median
from threading import RLock
from time import monotonic
from typing import Any

from avalanche.observability.events import MetricEvent
from avalanche.observability.resources import ResourceSample
from avalanche.observability.size import ParquetSizeEstimator, ParquetSizeSnapshot


@dataclass(frozen=True, slots=True)
class FrozenMapping(Mapping[str, Any]):
    """Hold a picklable immutable mapping."""

    _items: tuple[tuple[str, Any], ...] = ()

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False


def freeze_mapping(values: Mapping[str, Any]) -> FrozenMapping:
    """Return a recursively immutable and picklable mapping."""
    return FrozenMapping(
        tuple((str(key), _freeze_value(value)) for key, value in values.items())
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    return value


class MetricKind(StrEnum):
    """List the event names understood by the aggregator."""

    RUN_CONFIG = "run_config"
    STAGE_STARTED = "stage_started"
    STAGE_PHASE = "stage_phase"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    EPISODE_STARTED = "episode_started"
    EPISODE_COMPLETED = "episode_completed"
    ROWS_GENERATED = "rows_generated"
    WORKER_PROGRESS = "worker_progress"
    RETRY = "retry"
    REJECTED = "rejected"
    FAILURE = "failure"
    LATENCY = "latency"
    COUNTER = "counter"
    SEMANTIC_COUNT = "semantic_count"
    EPOCH_PROGRESS = "epoch_progress"
    MODEL_PROGRESS = "model_progress"
    CALIBRATION_STARTED = "calibration_started"
    CALIBRATION_PROGRESS = "calibration_progress"
    CALIBRATION_COMPLETED = "calibration_completed"
    GATE_EVALUATED = "gate_evaluated"
    GRU_STATE = "gru_state"
    PARQUET_PROGRESS = "parquet_progress"
    RESOURCE_SAMPLE = "resource_sample"
    MESSAGE = "message"


class StageStatus(StrEnum):
    """Identify one pipeline-stage state."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CONDITIONAL = "conditional"
    NOT_EVALUATED = "not_evaluated"
    NOT_REQUIRED = "not_required"
    TRIGGERED = "triggered"


class GRUState(StrEnum):
    """Identify the optional GRU lifecycle."""

    NOT_EVALUATED = "not_evaluated"
    NOT_REQUIRED = "not_required"
    TRIGGERED = "triggered"
    TRAINING = "training"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LatencySnapshot:
    """Hold bounded latency statistics."""

    count: int
    sampled_count: int
    mean_seconds: float | None
    median_seconds: float | None
    p95_seconds: float | None


class BoundedLatencyStatistics:
    """Keep a bounded sample and an exact running mean."""

    def __init__(self, capacity: int = 2_048) -> None:
        if capacity < 1:
            raise ValueError("the latency sample capacity must be positive")
        self.capacity = capacity
        self._samples: deque[float] = deque(maxlen=capacity)
        self._count = 0
        self._sum = 0.0
        self._lock = RLock()

    def add(self, seconds: float) -> None:
        """Add one finite nonnegative duration."""
        value = float(seconds)
        if not isfinite(value) or value < 0.0:
            raise ValueError("a latency must be finite and nonnegative")
        with self._lock:
            self._samples.append(value)
            self._count += 1
            self._sum += value

    def snapshot(self) -> LatencySnapshot:
        """Return the running mean and bounded quantiles."""
        with self._lock:
            values = sorted(self._samples)
            p95 = values[max(ceil(0.95 * len(values)) - 1, 0)] if values else None
            return LatencySnapshot(
                count=self._count,
                sampled_count=len(values),
                mean_seconds=self._sum / self._count if self._count else None,
                median_seconds=float(median(values)) if values else None,
                p95_seconds=p95,
            )


@dataclass(frozen=True, slots=True)
class WorkerSnapshot:
    """Hold progress for one worker."""

    worker_id: str
    active: bool
    phase: str
    current_item: str | None
    episodes_completed: int
    rows_generated: int
    current_rows: int
    samples_processed: int
    retries: int
    rejected: int
    failures: int
    updated_at: float


@dataclass(frozen=True, slots=True)
class TrainingSnapshot:
    """Hold current training progress."""

    epoch: int
    total_epochs: int | None
    batch: int
    total_batches: int | None
    samples_processed: int
    total_samples: int | None
    training_loss: float | None
    validation_loss: float | None
    metric_name: str | None
    metric_value: float | None
    best_metric: float | None
    epoch_elapsed_seconds: float | None
    mean_epoch_seconds: float | None
    samples_per_second: float | None
    eta_seconds: float | None


@dataclass(frozen=True, slots=True)
class CalibrationSnapshot:
    """Hold current calibration progress."""

    status: StageStatus
    rows_processed: int
    total_rows: int | None
    threshold: float | None
    metrics: FrozenMapping
    elapsed_seconds: float
    eta_seconds: float | None


@dataclass(frozen=True, slots=True)
class GateSnapshot:
    """Hold one perceptron gate result."""

    stage_id: str
    criterion: str
    metric_name: str
    observed: float | None
    required: float | None
    comparison: str
    passed: bool
    values: FrozenMapping


@dataclass(frozen=True, slots=True)
class SignificantEvent:
    """Hold one short event for terminal display."""

    timestamp: float
    stage_id: str
    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class StageSnapshot:
    """Hold one immutable stage view."""

    stage_id: str
    label: str
    status: StageStatus
    phase: str
    progress_fraction: float
    started_at: float | None
    completed_at: float | None
    elapsed_seconds: float
    eta_seconds: float | None
    total_episodes: int | None
    expected_rows: int | None
    episodes_completed: int
    rows_generated: int
    rows_in_progress: int
    episodes_per_second: float | None
    rows_per_second: float | None
    configured_workers: int | None
    active_workers: int
    workers: tuple[WorkerSnapshot, ...]
    retries: int
    rejected: int
    failures: int
    counters: FrozenMapping
    latency: LatencySnapshot
    training: TrainingSnapshot
    current_model: str | None
    completed_models: int
    total_models: int | None
    calibration: CalibrationSnapshot
    gate: GateSnapshot | None
    gru_state: GRUState
    parquet: ParquetSizeSnapshot | None
    resources: ResourceSample | None
    metrics: FrozenMapping
    error: str | None

    @property
    def percentage(self) -> float:
        """Return progress as a percentage."""
        return 100.0 * self.progress_fraction


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    """Hold one immutable pipeline view."""

    stages: tuple[StageSnapshot, ...]
    progress_fraction: float
    completed_stages: int
    total_stages: int
    overall_eta_seconds: float | None
    run_context: FrozenMapping
    principal_traces_generated: int
    oracle_true_states_generated: int
    oracle_fallbacks_generated: int
    fallback_generation_attempts: int
    fallback_rate: float | None
    retries: int
    rejected: int
    failures: int
    gate: GateSnapshot | None
    gru_state: GRUState
    recent_events: tuple[SignificantEvent, ...]

    def stage(self, stage_id: str) -> StageSnapshot:
        """Return one stage by its identity."""
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(stage_id)


@dataclass(slots=True)
class _WorkerMetrics:
    worker_id: str
    active: bool = False
    phase: str = "pending"
    current_item: str | None = None
    episodes_completed: int = 0
    rows_generated: int = 0
    current_rows: int = 0
    samples_processed: int = 0
    retries: int = 0
    rejected: int = 0
    failures: int = 0
    updated_at: float = 0.0


@dataclass(slots=True)
class _StageMetrics:
    stage_id: str
    label: str
    status: StageStatus = StageStatus.PENDING
    phase: str = "pending"
    weight: float = 1.0
    started_at: float | None = None
    completed_at: float | None = None
    total_episodes: int | None = None
    expected_rows: int | None = None
    configured_workers: int | None = None
    counters: dict[str, int] = field(default_factory=dict)
    workers: dict[str, _WorkerMetrics] = field(default_factory=dict)
    latency: BoundedLatencyStatistics = field(default_factory=BoundedLatencyStatistics)
    epoch: int = 0
    total_epochs: int | None = None
    batch: int = 0
    total_batches: int | None = None
    samples_processed: int = 0
    total_samples: int | None = None
    training_loss: float | None = None
    validation_loss: float | None = None
    metric_name: str | None = None
    metric_value: float | None = None
    best_metric: float | None = None
    epoch_elapsed_seconds: float | None = None
    epoch_times: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    last_timed_epoch: int | None = None
    current_model: str | None = None
    completed_models: int = 0
    total_models: int | None = None
    calibration_status: StageStatus = StageStatus.PENDING
    calibration_rows: int = 0
    calibration_total_rows: int | None = None
    calibration_threshold: float | None = None
    calibration_metrics: dict[str, float] = field(default_factory=dict)
    calibration_started_at: float | None = None
    calibration_completed_at: float | None = None
    gate: GateSnapshot | None = None
    gru_state: GRUState = GRUState.NOT_EVALUATED
    parquet: ParquetSizeEstimator | None = None
    resources: ResourceSample | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class MetricsAggregator:
    """Aggregate events safely across parent threads."""

    def __init__(
        self,
        *,
        latency_capacity: int = 2_048,
        recent_event_capacity: int = 200,
    ) -> None:
        if latency_capacity < 1:
            raise ValueError("the latency sample capacity must be positive")
        if recent_event_capacity < 1:
            raise ValueError("the recent event capacity must be positive")
        self.latency_capacity = latency_capacity
        self._stages: dict[str, _StageMetrics] = {}
        self._run_context: dict[str, Any] = {}
        self._recent: deque[SignificantEvent] = deque(maxlen=recent_event_capacity)
        self._gate: GateSnapshot | None = None
        self._gru_state = GRUState.NOT_EVALUATED
        self._lock = RLock()

    def register_stage(
        self,
        stage_id: str,
        *,
        label: str | None = None,
        status: StageStatus = StageStatus.PENDING,
        total_episodes: int | None = None,
        expected_rows: int | None = None,
        total_epochs: int | None = None,
        total_samples: int | None = None,
        total_models: int | None = None,
        workers: int | None = None,
        weight: float = 1.0,
    ) -> None:
        """Register one stage before work starts."""
        values = {
            "label": label or _label(stage_id),
            "status": status.value,
            "total_episodes": total_episodes,
            "expected_rows": expected_rows,
            "total_epochs": total_epochs,
            "total_samples": total_samples,
            "total_models": total_models,
            "workers": workers,
            "weight": weight,
        }
        with self._lock:
            stage = self._ensure_stage(stage_id)
            self._configure_stage(stage, values)

    def apply(self, event: MetricEvent) -> None:
        """Apply one structured metric event."""
        kind = event.kind.strip().lower().replace("-", "_")
        with self._lock:
            if kind == MetricKind.RUN_CONFIG:
                self._run_context.update(event.values)
                return
            stage = self._ensure_stage(event.stage_id)
            if kind == MetricKind.STAGE_STARTED:
                self._stage_started(stage, event)
            elif kind == MetricKind.STAGE_PHASE:
                self._stage_phase(stage, event)
            elif kind == MetricKind.STAGE_COMPLETED:
                self._stage_completed(stage, event)
            elif kind == MetricKind.STAGE_FAILED:
                self._stage_failed(stage, event)
            elif kind == MetricKind.EPISODE_STARTED:
                self._episode_started(stage, event)
            elif kind == MetricKind.EPISODE_COMPLETED:
                self._episode_completed(stage, event)
            elif kind == MetricKind.ROWS_GENERATED:
                self._increment_counter(stage, "rows", _event_count(event))
            elif kind == MetricKind.WORKER_PROGRESS:
                self._worker_progress(stage, event)
            elif kind == MetricKind.RETRY:
                self._named_count(stage, event, "retries")
            elif kind in {MetricKind.REJECTED, "rejection"}:
                self._named_count(stage, event, "rejected")
            elif kind == MetricKind.FAILURE:
                self._named_count(stage, event, "failures")
            elif kind == MetricKind.LATENCY:
                stage.latency.add(_required_float(event.values, "seconds"))
            elif kind in {MetricKind.COUNTER, MetricKind.SEMANTIC_COUNT}:
                self._generic_counter(stage, event)
            elif kind == MetricKind.EPOCH_PROGRESS:
                self._epoch_progress(stage, event)
            elif kind == MetricKind.MODEL_PROGRESS:
                self._model_progress(stage, event)
            elif kind in {
                MetricKind.CALIBRATION_STARTED,
                MetricKind.CALIBRATION_PROGRESS,
                MetricKind.CALIBRATION_COMPLETED,
            }:
                self._calibration(stage, event, kind)
            elif kind == MetricKind.GATE_EVALUATED:
                self._gate_evaluated(stage, event)
            elif kind == MetricKind.GRU_STATE:
                self._set_gru_state(stage, event)
            elif kind == MetricKind.PARQUET_PROGRESS:
                self._parquet_progress(stage, event)
            elif kind == MetricKind.RESOURCE_SAMPLE:
                self._resource_sample(stage, event)
            else:
                self._generic_progress(stage, event)
            if kind in _SIGNIFICANT_KINDS:
                self._record_significant(event, kind)

    def apply_many(self, events: Iterable[MetricEvent]) -> None:
        """Apply metric events in their supplied order."""
        for event in events:
            self.apply(event)

    def snapshot(self, *, now: float | None = None) -> PipelineSnapshot:
        """Return one consistent pipeline snapshot."""
        snapshot_at = monotonic() if now is None else now
        with self._lock:
            stages = tuple(
                self._stage_snapshot(stage, snapshot_at)
                for stage in self._stages.values()
            )
            included = [
                stage
                for stage in stages
                if stage.status
                not in {
                    StageStatus.CONDITIONAL,
                    StageStatus.NOT_EVALUATED,
                    StageStatus.NOT_REQUIRED,
                }
            ]
            weights = [self._stages[stage.stage_id].weight for stage in included]
            weight_total = sum(weights)
            progress = (
                sum(
                    stage.progress_fraction * weight
                    for stage, weight in zip(included, weights, strict=True)
                )
                / weight_total
                if weight_total > 0.0
                else 0.0
            )
            unfinished = [
                stage
                for stage in included
                if stage.status not in {StageStatus.COMPLETE, StageStatus.FAILED}
            ]
            overall_eta = (
                unfinished[0].eta_seconds
                if len(unfinished) == 1 and unfinished[0].status == StageStatus.RUNNING
                else None
            )
            counters = _sum_counters(stages)
            attempts = counters.get("fallback_attempts", 0)
            fallbacks = counters.get("oracle_fallbacks", 0)
            return PipelineSnapshot(
                stages=stages,
                progress_fraction=progress,
                completed_stages=sum(
                    stage.status in {StageStatus.COMPLETE, StageStatus.NOT_REQUIRED}
                    for stage in stages
                ),
                total_stages=len(stages),
                overall_eta_seconds=overall_eta,
                run_context=freeze_mapping(self._run_context),
                principal_traces_generated=counters.get("principal_traces", 0),
                oracle_true_states_generated=counters.get("oracle_true_states", 0),
                oracle_fallbacks_generated=fallbacks,
                fallback_generation_attempts=attempts,
                fallback_rate=fallbacks / attempts if attempts else None,
                retries=sum(stage.retries for stage in stages),
                rejected=sum(stage.rejected for stage in stages),
                failures=sum(stage.failures for stage in stages),
                gate=self._gate,
                gru_state=self._gru_state,
                recent_events=tuple(self._recent),
            )

    def _ensure_stage(self, stage_id: str) -> _StageMetrics:
        stage = self._stages.get(stage_id)
        if stage is None:
            stage = _StageMetrics(
                stage_id=stage_id,
                label=_label(stage_id),
                latency=BoundedLatencyStatistics(self.latency_capacity),
            )
            self._stages[stage_id] = stage
        return stage

    def _configure_stage(self, stage: _StageMetrics, values: dict[str, Any]) -> None:
        if values.get("label") is not None:
            stage.label = str(values["label"])
        if values.get("status") is not None:
            stage.status = StageStatus(str(values["status"]))
        stage.total_episodes = _optional_nonnegative_int(
            values.get("total_episodes"), stage.total_episodes
        )
        stage.expected_rows = _optional_nonnegative_int(
            values.get("expected_rows", values.get("total_rows")),
            stage.expected_rows,
        )
        stage.total_epochs = _optional_nonnegative_int(
            values.get("total_epochs"), stage.total_epochs
        )
        stage.total_samples = _optional_nonnegative_int(
            values.get("total_samples"), stage.total_samples
        )
        stage.total_batches = _optional_nonnegative_int(
            values.get("total_batches", values.get("batches_per_epoch")),
            stage.total_batches,
        )
        stage.total_models = _optional_nonnegative_int(
            values.get("total_models"), stage.total_models
        )
        stage.configured_workers = _optional_nonnegative_int(
            values.get("workers", values.get("worker_count")),
            stage.configured_workers,
        )
        if values.get("weight") is not None:
            weight = _as_float(values["weight"], "stage weight")
            if weight <= 0.0:
                raise ValueError("a stage weight must be positive")
            stage.weight = weight
        stage.current_model = _optional_string(
            values.get("model_name", values.get("current_model")),
            stage.current_model,
        )
        for name in ("retries", "rejected", "failures"):
            if values.get(name) is not None:
                stage.counters[name] = _optional_count(values[name], 0)

    def _mark_running(self, stage: _StageMetrics, timestamp: float) -> None:
        if stage.started_at is None:
            stage.started_at = timestamp
        if stage.status in {StageStatus.PENDING, StageStatus.TRIGGERED}:
            stage.status = StageStatus.RUNNING

    def _stage_started(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._configure_stage(stage, event.values)
        stage.status = StageStatus.RUNNING
        stage.phase = str(event.values.get("phase", "running"))
        if stage.started_at is None:
            stage.started_at = event.timestamp

    def _stage_phase(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        stage.phase = str(event.values.get("phase", "running"))
        if "detail" in event.values:
            stage.metrics["detail"] = event.values["detail"]

    def _stage_completed(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._configure_stage(stage, event.values)
        self._mark_running(stage, event.timestamp)
        stage.status = StageStatus.COMPLETE
        stage.phase = str(event.values.get("phase", "complete"))
        stage.completed_at = event.timestamp
        for name, key in (("episodes", "episodes"), ("rows", "rows")):
            if key in event.values:
                value = _optional_count(event.values[key], 0)
                stage.counters[name] = max(stage.counters.get(name, 0), value)
        stage.samples_processed = (
            _optional_nonnegative_int(
                event.values.get("samples"), stage.samples_processed
            )
            or 0
        )
        stage.training_loss = _optional_float(
            event.values.get("training_loss"), stage.training_loss
        )
        for name, value in event.values.items():
            if name != "phase":
                stage.metrics[name] = value
        for worker in stage.workers.values():
            worker.active = False

    def _stage_failed(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        stage.status = StageStatus.FAILED
        stage.phase = "failed"
        stage.completed_at = event.timestamp
        stage.error = str(event.values.get("error", event.values.get("message", "")))
        if bool(event.values.get("count_failure", True)):
            stage.counters["failures"] = max(stage.counters.get("failures", 0), 1)
        for worker in stage.workers.values():
            worker.active = False

    def _episode_started(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        worker = self._worker(stage, event)
        if worker is not None:
            worker.active = True
            worker.phase = str(event.values.get("phase", "episode"))
            worker.current_rows = 0
            item = event.values.get("episode_id")
            worker.current_item = str(item) if item is not None else None
            worker.updated_at = event.timestamp

    def _episode_completed(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        count = _optional_count(event.values.get("count"), 1)
        rows = _optional_count(event.values.get("rows"), 0)
        self._increment_counter(stage, "episodes", count)
        self._increment_counter(stage, "rows", rows)
        latency = event.values.get("latency_seconds")
        if latency is not None:
            stage.latency.add(_as_float(latency, "episode latency"))
        worker = self._worker(stage, event)
        if worker is not None:
            worker.episodes_completed += count
            worker.rows_generated += rows
            worker.current_rows = 0
            worker.active = bool(event.values.get("active", False))
            worker.current_item = None
            worker.phase = str(event.values.get("phase", "idle"))
            worker.updated_at = event.timestamp

    def _worker_progress(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        worker = self._worker(stage, event, required=True)
        assert worker is not None
        worker.active = bool(event.values.get("active", True))
        if "phase" in event.values:
            worker.phase = str(event.values["phase"])
        item = event.values.get("current_item", event.values.get("episode_id"))
        if not worker.active:
            worker.current_item = None
        elif item is not None:
            worker.current_item = str(item)
        worker.episodes_completed = (
            _optional_nonnegative_int(
                event.values.get("episodes_completed"), worker.episodes_completed
            )
            or 0
        )
        worker.current_rows = (
            _optional_nonnegative_int(
                event.values.get("current_rows", event.values.get("rows")),
                worker.current_rows,
            )
            or 0
        )
        worker.samples_processed = (
            _optional_nonnegative_int(
                event.values.get("samples"), worker.samples_processed
            )
            or 0
        )
        worker.updated_at = event.timestamp

    def _named_count(self, stage: _StageMetrics, event: MetricEvent, name: str) -> None:
        self._mark_running(stage, event.timestamp)
        count = _event_count(event)
        self._increment_counter(stage, name, count)
        worker = self._worker(stage, event)
        if worker is not None:
            setattr(worker, name, getattr(worker, name) + count)
            worker.updated_at = event.timestamp

    def _generic_counter(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        name = str(event.values.get("name", ""))
        if not name:
            raise ValueError("a generic metric counter needs a name")
        self._increment_counter(stage, name, _event_count(event))

    def _increment_counter(self, stage: _StageMetrics, name: str, count: int) -> None:
        stage.counters[name] = stage.counters.get(name, 0) + count

    def _epoch_progress(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        values = event.values
        stage.phase = str(values.get("phase", "training"))
        stage.epoch = _optional_nonnegative_int(values.get("epoch"), stage.epoch) or 0
        stage.total_epochs = _optional_nonnegative_int(
            values.get("total_epochs"), stage.total_epochs
        )
        stage.batch = _optional_nonnegative_int(values.get("batch"), stage.batch) or 0
        stage.total_batches = _optional_nonnegative_int(
            values.get("total_batches", values.get("batches_per_epoch")),
            stage.total_batches,
        )
        stage.samples_processed = (
            _optional_nonnegative_int(
                values.get("samples", values.get("samples_processed")),
                stage.samples_processed,
            )
            or 0
        )
        stage.total_samples = _optional_nonnegative_int(
            values.get("total_samples"), stage.total_samples
        )
        stage.training_loss = _optional_float(
            values.get("training_loss", values.get("train_loss")),
            stage.training_loss,
        )
        stage.validation_loss = _optional_float(
            values.get("validation_loss"), stage.validation_loss
        )
        if values.get("metric_name") is not None:
            stage.metric_name = str(values["metric_name"])
        stage.metric_value = _optional_float(
            values.get("metric_value"), stage.metric_value
        )
        stage.best_metric = _optional_float(
            values.get("best_metric"), stage.best_metric
        )
        stage.current_model = _optional_string(
            values.get("model_name"), stage.current_model
        )
        epoch_seconds = values.get("epoch_seconds", values.get("epoch_elapsed"))
        if epoch_seconds is not None:
            duration = _as_float(epoch_seconds, "epoch duration")
            if duration < 0.0:
                raise ValueError("an epoch duration must be nonnegative")
            stage.epoch_elapsed_seconds = duration
            if values.get("phase") == "epoch" and stage.last_timed_epoch != stage.epoch:
                stage.epoch_times.append(duration)
                stage.last_timed_epoch = stage.epoch
        reserved = _TRAINING_KEYS
        for name, value in values.items():
            if name not in reserved:
                stage.metrics[name] = value

    def _model_progress(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        values = event.values
        stage.current_model = _optional_string(
            values.get("model_name", values.get("current_model")),
            stage.current_model,
        )
        stage.completed_models = (
            _optional_nonnegative_int(
                values.get("completed_models"), stage.completed_models
            )
            or 0
        )
        stage.total_models = _optional_nonnegative_int(
            values.get("total_models"), stage.total_models
        )

    def _calibration(self, stage: _StageMetrics, event: MetricEvent, kind: str) -> None:
        self._mark_running(stage, event.timestamp)
        values = event.values
        if stage.calibration_started_at is None:
            stage.calibration_started_at = event.timestamp
        stage.phase = str(values.get("phase", "calibration"))
        stage.calibration_status = StageStatus.RUNNING
        stage.calibration_rows = (
            _optional_nonnegative_int(
                values.get("rows", values.get("rows_processed")),
                stage.calibration_rows,
            )
            or 0
        )
        stage.calibration_total_rows = _optional_nonnegative_int(
            values.get("total_rows"), stage.calibration_total_rows
        )
        stage.calibration_threshold = _optional_float(
            values.get("threshold", values.get("selected_threshold")),
            stage.calibration_threshold,
        )
        for name, value in values.items():
            if name not in _CALIBRATION_KEYS and _is_real(value):
                stage.calibration_metrics[name] = float(value)
        if kind == MetricKind.CALIBRATION_COMPLETED:
            stage.calibration_status = StageStatus.COMPLETE
            stage.calibration_completed_at = event.timestamp
            for name in (
                "candidate",
                "total_candidates",
                "candidate_threshold",
                "calibration_loss",
            ):
                stage.calibration_metrics.pop(name, None)

    def _gate_evaluated(self, stage: _StageMetrics, event: MetricEvent) -> None:
        values = event.values
        if "passed" not in values:
            raise ValueError("a gate event needs a pass result")
        gate = GateSnapshot(
            stage_id=stage.stage_id,
            criterion=str(values.get("criterion", "perceptron gate")),
            metric_name=str(values.get("metric_name", "metric")),
            observed=_optional_float(values.get("observed"), None),
            required=_optional_float(values.get("required"), None),
            comparison=str(values.get("comparison", "at_least")),
            passed=bool(values["passed"]),
            values=freeze_mapping(values),
        )
        stage.gate = gate
        model_name = str(values.get("model_name", "perceptron"))
        if model_name == "perceptron":
            self._gate = gate
            state = GRUState.NOT_REQUIRED if gate.passed else GRUState.TRIGGERED
            stage.gru_state = state
            self._gru_state = state

    def _set_gru_state(self, stage: _StageMetrics, event: MetricEvent) -> None:
        state = GRUState(str(event.values.get("state", "")))
        stage.gru_state = state
        self._gru_state = state
        if state == GRUState.NOT_REQUIRED:
            stage.status = StageStatus.NOT_REQUIRED
            stage.phase = "not required"
        elif state == GRUState.TRIGGERED:
            stage.status = StageStatus.TRIGGERED
            stage.phase = "triggered"
        elif state == GRUState.TRAINING:
            self._mark_running(stage, event.timestamp)
            stage.status = StageStatus.RUNNING
            stage.phase = "training"
        elif state == GRUState.COMPLETE:
            stage.status = StageStatus.COMPLETE
            stage.phase = "complete"
            stage.completed_at = event.timestamp
        elif state == GRUState.FAILED:
            stage.status = StageStatus.FAILED
            stage.phase = "failed"
            stage.completed_at = event.timestamp
        else:
            stage.status = StageStatus.NOT_EVALUATED
            stage.phase = "not evaluated"

    def _parquet_progress(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        values = event.values
        if stage.parquet is None:
            expected = _optional_nonnegative_int(
                values.get("expected_rows"), stage.expected_rows
            )
            stage.parquet = ParquetSizeEstimator(
                expected_rows=expected,
                minimum_written_rows=_optional_count(
                    values.get("minimum_written_rows"), 1_000
                ),
                minimum_row_groups=_optional_count(values.get("minimum_row_groups"), 2),
            )
        if "row_group_rows" in values or "row_group_bytes" in values:
            stage.parquet.add_row_group(
                rows=_optional_count(values.get("row_group_rows"), 0),
                encoded_bytes=_optional_count(values.get("row_group_bytes"), 0),
                buffered_rows=_optional_count(values.get("buffered_rows"), 0),
                final=bool(values.get("final", False)),
            )
            return
        stage.parquet.update(
            written_rows=_optional_count(values.get("written_rows"), 0),
            written_bytes=_optional_count(values.get("written_bytes"), 0),
            buffered_rows=_optional_count(values.get("buffered_rows"), 0),
            row_groups=(
                _optional_count(values.get("row_groups"), 0)
                if "row_groups" in values
                else None
            ),
            final=bool(values.get("final", False)),
        )

    def _resource_sample(self, stage: _StageMetrics, event: MetricEvent) -> None:
        values = event.values
        sample = values.get("sample")
        if isinstance(sample, ResourceSample):
            stage.resources = sample
            return
        tree_cpu_percent = _optional_float(
            values.get("tree_cpu_percent"),
            None,
        )
        logical_cpu_count = _optional_nonnegative_int(
            values.get("logical_cpu_count"),
            None,
        )
        tree_cpu_cores = _optional_float(
            values.get("tree_cpu_cores"),
            (tree_cpu_percent / 100.0 if tree_cpu_percent is not None else None),
        )
        capacity = _optional_float(
            values.get("tree_cpu_capacity_percent"),
            (
                tree_cpu_percent / logical_cpu_count
                if tree_cpu_percent is not None and logical_cpu_count
                else None
            ),
        )
        stage.resources = ResourceSample(
            timestamp=_optional_float(values.get("timestamp"), event.timestamp)
            or event.timestamp,
            system_cpu_percent=_optional_float(
                values.get("system_cpu_percent"),
                None,
            ),
            system_memory_percent=_optional_float(
                values.get("system_memory_percent"),
                None,
            ),
            tree_cpu_percent=tree_cpu_percent,
            tree_memory_percent=_optional_float(
                values.get("tree_memory_percent"),
                None,
            ),
            tree_rss_bytes=_optional_nonnegative_int(
                values.get("tree_rss_bytes"),
                None,
            ),
            process_count=_optional_count(values.get("process_count"), 0),
            processes=(),
            logical_cpu_count=logical_cpu_count,
            tree_cpu_cores=tree_cpu_cores,
            tree_cpu_capacity_percent=capacity,
            gpu_percent=_optional_float(values.get("gpu_percent"), None),
            gpu_memory_bytes=_optional_nonnegative_int(
                values.get("gpu_memory_bytes"), None
            ),
        )

    def _generic_progress(self, stage: _StageMetrics, event: MetricEvent) -> None:
        self._mark_running(stage, event.timestamp)
        stage.metrics.update(event.values)

    def _worker(
        self,
        stage: _StageMetrics,
        event: MetricEvent,
        *,
        required: bool = False,
    ) -> _WorkerMetrics | None:
        worker_id = event.worker_id
        if worker_id is None:
            if required:
                raise ValueError("a worker event needs a worker identity")
            return None
        worker = stage.workers.get(worker_id)
        if worker is None:
            worker = _WorkerMetrics(worker_id=worker_id, updated_at=event.timestamp)
            stage.workers[worker_id] = worker
        return worker

    def _record_significant(self, event: MetricEvent, kind: str) -> None:
        message = str(
            event.values.get(
                "message",
                event.values.get("error", event.values.get("phase", kind)),
            )
        )
        self._recent.append(
            SignificantEvent(event.timestamp, event.stage_id, kind, message)
        )

    def _stage_snapshot(
        self, stage: _StageMetrics, snapshot_at: float
    ) -> StageSnapshot:
        elapsed = _elapsed(stage.started_at, stage.completed_at, snapshot_at)
        episodes = stage.counters.get("episodes", 0)
        rows = stage.counters.get("rows", 0)
        episode_rate = episodes / elapsed if elapsed > 0.0 and episodes else None
        row_rate = rows / elapsed if elapsed > 0.0 and rows else None
        training = self._training_snapshot(stage, elapsed)
        calibration = self._calibration_snapshot(stage, snapshot_at)
        progress = _stage_progress(stage, training, calibration)
        eta_candidates: list[float] = []
        if stage.total_episodes is not None and episode_rate:
            eta_candidates.append(
                max(stage.total_episodes - episodes, 0) / episode_rate
            )
        elif stage.expected_rows is not None and row_rate:
            eta_candidates.append(max(stage.expected_rows - rows, 0) / row_rate)
        if training.eta_seconds is not None:
            eta_candidates.append(training.eta_seconds)
        if calibration.eta_seconds is not None:
            eta_candidates.append(calibration.eta_seconds)
        workers = tuple(
            WorkerSnapshot(
                worker_id=worker.worker_id,
                active=worker.active,
                phase=worker.phase,
                current_item=worker.current_item,
                episodes_completed=worker.episodes_completed,
                rows_generated=worker.rows_generated,
                current_rows=worker.current_rows,
                samples_processed=worker.samples_processed,
                retries=worker.retries,
                rejected=worker.rejected,
                failures=worker.failures,
                updated_at=worker.updated_at,
            )
            for worker in sorted(
                stage.workers.values(), key=lambda item: item.worker_id
            )
        )
        rows_in_progress = sum(worker.current_rows for worker in workers)
        terminal = stage.status in {StageStatus.COMPLETE, StageStatus.FAILED}
        return StageSnapshot(
            stage_id=stage.stage_id,
            label=stage.label,
            status=stage.status,
            phase=stage.phase,
            progress_fraction=progress,
            started_at=stage.started_at,
            completed_at=stage.completed_at,
            elapsed_seconds=elapsed,
            eta_seconds=(
                None if terminal else max(eta_candidates) if eta_candidates else None
            ),
            total_episodes=stage.total_episodes,
            expected_rows=stage.expected_rows,
            episodes_completed=episodes,
            rows_generated=rows,
            rows_in_progress=rows_in_progress,
            episodes_per_second=episode_rate,
            rows_per_second=row_rate,
            configured_workers=stage.configured_workers,
            active_workers=sum(worker.active for worker in workers),
            workers=workers,
            retries=stage.counters.get("retries", 0),
            rejected=stage.counters.get("rejected", 0),
            failures=stage.counters.get("failures", 0),
            counters=freeze_mapping(stage.counters),
            latency=stage.latency.snapshot(),
            training=training,
            current_model=stage.current_model,
            completed_models=stage.completed_models,
            total_models=stage.total_models,
            calibration=calibration,
            gate=stage.gate,
            gru_state=stage.gru_state,
            parquet=stage.parquet.snapshot() if stage.parquet else None,
            resources=stage.resources,
            metrics=freeze_mapping(stage.metrics),
            error=stage.error,
        )

    def _training_snapshot(
        self, stage: _StageMetrics, elapsed: float
    ) -> TrainingSnapshot:
        mean_epoch = (
            sum(stage.epoch_times) / len(stage.epoch_times)
            if stage.epoch_times
            else None
        )
        sample_rate = (
            stage.samples_processed / elapsed
            if elapsed > 0.0 and stage.samples_processed
            else None
        )
        eta: float | None = None
        if stage.total_epochs is not None and mean_epoch is not None:
            eta = max(stage.total_epochs - stage.epoch, 0) * mean_epoch
        elif stage.total_samples is not None and sample_rate:
            eta = max(stage.total_samples - stage.samples_processed, 0) / sample_rate
        return TrainingSnapshot(
            epoch=stage.epoch,
            total_epochs=stage.total_epochs,
            batch=stage.batch,
            total_batches=stage.total_batches,
            samples_processed=stage.samples_processed,
            total_samples=stage.total_samples,
            training_loss=stage.training_loss,
            validation_loss=stage.validation_loss,
            metric_name=stage.metric_name,
            metric_value=stage.metric_value,
            best_metric=stage.best_metric,
            epoch_elapsed_seconds=stage.epoch_elapsed_seconds,
            mean_epoch_seconds=mean_epoch,
            samples_per_second=sample_rate,
            eta_seconds=eta,
        )

    def _calibration_snapshot(
        self, stage: _StageMetrics, snapshot_at: float
    ) -> CalibrationSnapshot:
        elapsed = _elapsed(
            stage.calibration_started_at,
            stage.calibration_completed_at,
            snapshot_at,
        )
        rate = (
            stage.calibration_rows / elapsed
            if elapsed > 0.0 and stage.calibration_rows
            else None
        )
        eta = (
            max(stage.calibration_total_rows - stage.calibration_rows, 0) / rate
            if stage.calibration_total_rows is not None and rate
            else None
        )
        return CalibrationSnapshot(
            status=stage.calibration_status,
            rows_processed=stage.calibration_rows,
            total_rows=stage.calibration_total_rows,
            threshold=stage.calibration_threshold,
            metrics=freeze_mapping(stage.calibration_metrics),
            elapsed_seconds=elapsed,
            eta_seconds=eta,
        )


_SIGNIFICANT_KINDS = {
    MetricKind.STAGE_STARTED,
    MetricKind.STAGE_COMPLETED,
    MetricKind.STAGE_FAILED,
    MetricKind.RETRY,
    MetricKind.REJECTED,
    MetricKind.FAILURE,
    MetricKind.CALIBRATION_COMPLETED,
    MetricKind.GATE_EVALUATED,
    MetricKind.GRU_STATE,
    MetricKind.MESSAGE,
}

_TRAINING_KEYS = {
    "epoch",
    "total_epochs",
    "batch",
    "total_batches",
    "batches_per_epoch",
    "samples",
    "samples_processed",
    "total_samples",
    "training_loss",
    "train_loss",
    "validation_loss",
    "metric_name",
    "metric_value",
    "best_metric",
    "epoch_seconds",
    "epoch_elapsed",
    "model_name",
    "phase",
}

_CALIBRATION_KEYS = {
    "rows",
    "rows_processed",
    "total_rows",
    "threshold",
    "selected_threshold",
    "phase",
}


def _label(stage_id: str) -> str:
    return stage_id.replace("_", " ").replace("-", " ").strip().title()


def _event_count(event: MetricEvent) -> int:
    return _optional_count(event.values.get("count", event.values.get("delta")), 1)


def _optional_count(value: Any, default: int) -> int:
    if value is None:
        return default
    result = _as_int(value, "metric count")
    if result < 0:
        raise ValueError("a metric count must be nonnegative")
    return result


def _optional_nonnegative_int(value: Any, default: int | None) -> int | None:
    if value is None:
        return default
    result = _as_int(value, "metric total")
    if result < 0:
        raise ValueError("a metric total must be nonnegative")
    return result


def _as_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"the {label} must be an integer")
    return int(value)


def _required_float(values: dict[str, Any], name: str) -> float:
    if name not in values:
        raise ValueError(f"a metric event needs {name}")
    return _as_float(values[name], name)


def _as_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"the {label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"the {label} must be finite")
    return result


def _optional_float(value: Any, default: float | None) -> float | None:
    return default if value is None else _as_float(value, "metric value")


def _optional_string(value: Any, default: str | None) -> str | None:
    return default if value is None else str(value)


def _is_real(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and isfinite(value)


def _elapsed(start: float | None, end: float | None, now: float) -> float:
    if start is None:
        return 0.0
    return max((now if end is None else end) - start, 0.0)


def _stage_progress(
    stage: _StageMetrics,
    training: TrainingSnapshot,
    calibration: CalibrationSnapshot,
) -> float:
    if stage.status in {StageStatus.COMPLETE, StageStatus.NOT_REQUIRED}:
        return 1.0
    if training.total_epochs:
        batch_fraction = (
            training.batch / training.total_batches if training.total_batches else 0.0
        )
        completed_epochs = max(training.epoch - 1, 0) + batch_fraction
        return min(max(completed_epochs / training.total_epochs, 0.0), 1.0)
    episodes = stage.counters.get("episodes", 0)
    if stage.total_episodes:
        return min(episodes / stage.total_episodes, 1.0)
    rows = stage.counters.get("rows", 0)
    if stage.expected_rows:
        return min(rows / stage.expected_rows, 1.0)
    if calibration.total_rows:
        return min(calibration.rows_processed / calibration.total_rows, 1.0)
    if stage.total_models:
        return min(stage.completed_models / stage.total_models, 1.0)
    return 0.0


def _sum_counters(stages: tuple[StageSnapshot, ...]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for stage in stages:
        for name, value in stage.counters.items():
            totals[name] = totals.get(name, 0) + value
    return totals

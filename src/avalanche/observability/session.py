"""Own metric collection, reporting, and persistent logs."""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.managers import SyncManager
from pathlib import Path
from queue import Empty
from threading import Event, RLock, Thread
from time import monotonic
from typing import Any

from loguru._logger import Core, Logger

from avalanche.observability.events import (
    DirectMetricEmitter,
    MetricEmitter,
    MetricEvent,
    QueueMetricEmitter,
    drain_metric_events,
)
from avalanche.observability.metrics import MetricsAggregator, StageStatus
from avalanche.observability.reporter import SummaryOutcome, TextualReporter
from avalanche.observability.resources import ProcessTreeSampler


class ObservabilitySession:
    """Coordinate parent-owned metrics, reporting, and logs."""

    def __init__(
        self,
        *,
        aggregator: MetricsAggregator | None = None,
        reporter: TextualReporter | None = None,
        enabled: bool | None = None,
        log_path: Path | None = None,
        multiprocessing: bool = False,
        sample_resources: bool = True,
        resource_interval: float = 1.0,
        refresh_interval: float = 0.25,
    ) -> None:
        if resource_interval <= 0.0:
            raise ValueError("the resource sample interval must be positive")
        if refresh_interval < 0.0:
            raise ValueError("the report refresh interval must be nonnegative")
        self.aggregator = aggregator or MetricsAggregator()
        self.reporter = reporter or TextualReporter(
            self.aggregator,
            enabled=enabled,
        )
        self.emitter: MetricEmitter = DirectMetricEmitter(self._handle_event)
        self.resource_interval = resource_interval
        self.refresh_interval = refresh_interval
        self._sampler = ProcessTreeSampler() if sample_resources else None
        self._logger = _isolated_logger()
        self._log_handler: int | None = None
        self._manager: SyncManager | None = None
        self._queue: Any | None = None
        self._process_emitter: QueueMetricEmitter | None = None
        self._stop = Event()
        self._background: Thread | None = None
        self._background_error: BaseException | None = None
        self._last_resource_sample = 0.0
        self._last_refresh = 0.0
        self._started = False
        self._closed = False
        self._lock = RLock()
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handler = self._logger.add(
                path,
                level="INFO",
                serialize=True,
                enqueue=False,
                backtrace=False,
                diagnose=False,
                encoding="utf-8",
            )
        if multiprocessing:
            self._ensure_process_channel()

    @property
    def process_emitter(self) -> QueueMetricEmitter:
        """Return the session-owned process-safe emitter."""
        return self.make_process_emitter()

    @property
    def background_error(self) -> BaseException | None:
        """Return a background monitoring error."""
        return self._background_error

    def make_process_emitter(self) -> QueueMetricEmitter:
        """Create a Manager-backed emitter and start its consumer."""
        with self._lock:
            self._require_open()
            self._ensure_process_channel()
            self._start_background()
            assert self._process_emitter is not None
            return self._process_emitter

    @staticmethod
    def queue_emitter(queue: Any) -> QueueMetricEmitter:
        """Wrap an existing process-safe queue."""
        return QueueMetricEmitter(queue)

    def start(self) -> None:
        """Start reporting and background collection."""
        with self._lock:
            self._require_open()
            if self._started:
                return
            self._started = True
            try:
                self.reporter.start()
                self._exclude_observer()
            except KeyboardInterrupt:
                self.reporter.enabled = False
                self.close(outcome=SummaryOutcome.INTERRUPTED)
                raise
            except Exception as error:
                self._remember_error(error)
                self.reporter.enabled = False
            self._start_background()
            try:
                self._logger.info("observability session started")
            except Exception as error:
                self._remember_error(error)

    def drain_pending(self, *, limit: int | None = None) -> int:
        """Apply events that are currently waiting in the queue."""
        queue = self._queue
        if queue is None:
            return 0
        return drain_metric_events(queue, self._handle_event, limit=limit)

    def close(self, *, outcome: SummaryOutcome | None = None) -> None:
        """Drain events before the Manager shuts down."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            background = self._background
        if background is not None:
            background.join()
        try:
            self.drain_pending()
        except BrokenPipeError, ConnectionError, EOFError, OSError:
            pass
        try:
            self._logger.info("observability session stopped")
        except Exception as error:
            self._remember_error(error)
        try:
            if outcome is not None:
                self.reporter.set_outcome(outcome)
        except Exception as error:
            self._remember_error(error)
        try:
            self.reporter.refresh()
        except Exception as error:
            self._remember_error(error)
        try:
            self.reporter.stop()
        except Exception as error:
            self._remember_error(error)
        if self._log_handler is not None:
            try:
                self._logger.complete()
                self._logger.remove(self._log_handler)
            except Exception as error:
                self._remember_error(error)
            finally:
                self._log_handler = None
        manager = self._manager
        if manager is not None:
            try:
                manager.shutdown()
            except BrokenPipeError, ConnectionError, EOFError, OSError:
                pass
            self._manager = None
            self._queue = None
            self._process_emitter = None

    def _handle_event(self, event: MetricEvent) -> None:
        self.aggregator.apply(event)
        try:
            self._log_event(event)
        except Exception as error:
            self._remember_error(error)
        now = monotonic()
        with self._lock:
            should_refresh = (
                self.reporter.active
                and now - self._last_refresh >= self.refresh_interval
            )
            if should_refresh:
                self._last_refresh = now
        if should_refresh:
            try:
                self.reporter.refresh()
            except Exception as error:
                self._remember_error(error)
                self.reporter.enabled = False

    def _log_event(self, event: MetricEvent) -> None:
        kind = event.kind.strip().lower().replace("-", "_")
        if kind not in _LOGGED_KINDS:
            return
        level = "ERROR" if kind in {"failure", "stage_failed"} else "INFO"
        if kind in {"retry", "rejected", "rejection"}:
            level = "WARNING"
        self._logger.bind(
            event_kind=kind,
            stage_id=event.stage_id,
            worker_id=event.worker_id,
            metric_values=event.values,
            metric_timestamp=event.timestamp,
        ).log(level, kind)

    def _ensure_process_channel(self) -> None:
        if self._manager is not None:
            return
        self._manager = mp.Manager()
        self._queue = self._manager.Queue()
        self._process_emitter = QueueMetricEmitter(self._queue)

    def _start_background(self) -> None:
        if self._background is not None or (
            self._queue is None and self._sampler is None
        ):
            return
        self._background = Thread(
            target=self._background_loop,
            name="avalanche-observability",
            daemon=True,
        )
        self._background.start()

    def _background_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._check_reporter()
                self._consume_one()
                self._sample_resources()
            self.drain_pending()
            self._sample_resources(force=True)
        except BrokenPipeError, ConnectionError, EOFError, OSError:
            return
        except BaseException as error:
            self._background_error = error

    def _consume_one(self) -> None:
        if self._queue is None:
            self._stop.wait(0.05)
            return
        try:
            item = self._queue.get(timeout=0.05)
        except Empty:
            return
        if not isinstance(item, MetricEvent):
            raise TypeError("a metric queue item must be a MetricEvent")
        self._handle_event(item)

    def _check_reporter(self) -> None:
        try:
            self.reporter.active
        except Exception as error:
            self._remember_error(error)
            self.reporter.enabled = False

    def _sample_resources(self, *, force: bool = False) -> None:
        sampler = self._sampler
        if sampler is None:
            return
        now = monotonic()
        if not force and now - self._last_resource_sample < self.resource_interval:
            return
        state = self.aggregator.snapshot(now=now)
        running = [
            stage for stage in state.stages if stage.status == StageStatus.RUNNING
        ]
        if not running:
            return
        self._last_resource_sample = now
        self._handle_event(
            MetricEvent.create(
                "resource_sample",
                running[-1].stage_id,
                sample=sampler.sample(force=force),
            )
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the observability session is closed")

    def _exclude_observer(self) -> None:
        sampler = self._sampler
        pid = self.reporter.observer_pid
        if sampler is None or pid is None:
            return
        sampler.exclude_process(pid, self.reporter.observer_create_time)

    def _remember_error(self, error: BaseException) -> None:
        """Keep the first internal monitoring error."""
        with self._lock:
            if self._background_error is None:
                self._background_error = error

    def __enter__(self) -> ObservabilitySession:
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        outcome = None
        if exception_type is not None:
            outcome = (
                SummaryOutcome.INTERRUPTED
                if issubclass(exception_type, KeyboardInterrupt)
                else SummaryOutcome.FAILED
            )
        self.close(outcome=outcome)


def _isolated_logger() -> Logger:
    """Return a Loguru instance without a terminal handler."""
    return Logger(
        core=Core(),
        exception=None,
        depth=0,
        record=False,
        lazy=False,
        colors=False,
        raw=False,
        capture=True,
        patchers=[],
        extra={},
    )


_LOGGED_KINDS = {
    "run_config",
    "stage_started",
    "stage_completed",
    "stage_failed",
    "retry",
    "rejected",
    "rejection",
    "failure",
    "calibration_completed",
    "gate_evaluated",
    "gru_state",
    "message",
}

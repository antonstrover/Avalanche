"""Define metric events and their emitters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty
from time import monotonic
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class MetricEvent:
    """Hold one picklable observability update."""

    kind: str
    stage_id: str
    worker_id: str | None = None
    values: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=monotonic)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("a metric event kind must not be empty")
        if not self.stage_id:
            raise ValueError("a metric stage identity must not be empty")
        if self.worker_id == "":
            raise ValueError("a metric worker identity must not be empty")
        object.__setattr__(self, "values", dict(self.values))

    @classmethod
    def create(
        cls,
        kind: str,
        stage_id: str,
        worker_id: str | None = None,
        **values: Any,
    ) -> MetricEvent:
        """Return one event with a monotonic timestamp."""
        return cls(kind=kind, stage_id=stage_id, worker_id=worker_id, values=values)


@runtime_checkable
class MetricEmitter(Protocol):
    """Emit one structured metric event."""

    def emit(self, event: MetricEvent) -> None:
        """Send one event to its consumer."""


@dataclass(slots=True)
class DirectMetricEmitter:
    """Send events directly to one parent callback."""

    consumer: Callable[[MetricEvent], None]

    def emit(self, event: MetricEvent) -> None:
        """Send one event to the callback."""
        self.consumer(event)


class QueueLike(Protocol):
    """Define the queue operations used by an emitter."""

    def put(self, item: object) -> object:
        """Put one item on the queue."""

    def get_nowait(self) -> object:
        """Return one available item without a wait."""


@dataclass(slots=True)
class QueueMetricEmitter:
    """Send picklable events through a process-safe queue."""

    queue: QueueLike

    def emit(self, event: MetricEvent) -> None:
        """Put one event on the queue."""
        try:
            self.queue.put(event)
        except BrokenPipeError, ConnectionError, EOFError, OSError:
            return


class NullMetricEmitter:
    """Discard events without a side effect."""

    def emit(self, event: MetricEvent) -> None:
        """Discard one event."""


def drain_metric_events(
    queue: QueueLike,
    consumer: Callable[[MetricEvent], None],
    *,
    limit: int | None = None,
) -> int:
    """Drain available events and return their count."""
    if limit is not None and limit < 0:
        raise ValueError("a drain limit must be nonnegative")
    count = 0
    while limit is None or count < limit:
        try:
            item = queue.get_nowait()
        except Empty:
            break
        if not isinstance(item, MetricEvent):
            raise TypeError("a metric queue item must be a MetricEvent")
        consumer(item)
        count += 1
    return count

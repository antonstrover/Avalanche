"""Test the Textual observer lifecycle without a real terminal process."""

from __future__ import annotations

import signal
import sys
from io import StringIO
from queue import Empty, Full
from typing import Any

import pytest
from rich.console import Console

from avalanche.observability import (
    MetricEvent,
    MetricsAggregator,
    ObservabilitySession,
    PipelineSnapshot,
    SummaryOutcome,
    TextualReporter,
    compact_summary,
)
from avalanche.observability import reporter as reporter_module
from avalanche.observability import session as session_module


class FakeQueue:
    """Provide one controlled bounded snapshot queue."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self.items: list[object] = []
        self.closed = False
        self.cancelled_join = False

    def put_nowait(self, item: object) -> None:
        """Put one item or report a full queue."""
        if len(self.items) >= self.maxsize:
            raise Full
        self.items.append(item)

    def get_nowait(self) -> object:
        """Return the oldest item or report an empty queue."""
        if not self.items:
            raise Empty
        return self.items.pop(0)

    def cancel_join_thread(self) -> None:
        """Record feeder-thread cancellation."""
        self.cancelled_join = True

    def close(self) -> None:
        """Record queue closure."""
        self.closed = True


class FakeEvent:
    """Provide one controlled process stop event."""

    def __init__(self) -> None:
        self.value = False

    def is_set(self) -> bool:
        """Return the current event state."""
        return self.value

    def set(self) -> None:
        """Set the event state."""
        self.value = True


class FakeProcess:
    """Provide the multiprocessing operations used by the reporter."""

    def __init__(
        self,
        *,
        target: object,
        args: tuple[object, ...],
        name: str,
        daemon: bool,
        stop_on_join: bool,
        start_error: BaseException | None,
    ) -> None:
        self.target = target
        self.args = args
        self.name = name
        self.daemon = daemon
        self.stop_on_join = stop_on_join
        self.start_error = start_error
        self.pid = 42_424
        self.started = False
        self.alive = False
        self.terminated = False
        self.join_timeouts: list[float] = []

    def start(self) -> None:
        """Start the controlled process or raise its configured error."""
        if self.start_error is not None:
            raise self.start_error
        self.started = True
        self.alive = True

    def is_alive(self) -> bool:
        """Return the controlled liveness state."""
        return self.alive

    def join(self, timeout: float) -> None:
        """Record a join and optionally finish the process."""
        self.join_timeouts.append(timeout)
        if self.stop_on_join:
            self.alive = False

    def terminate(self) -> None:
        """Terminate the controlled process."""
        self.terminated = True
        self.alive = False


class FakeContext:
    """Create controlled multiprocessing primitives."""

    def __init__(
        self,
        *,
        stop_on_join: bool = True,
        start_error: BaseException | None = None,
    ) -> None:
        self.stop_on_join = stop_on_join
        self.start_error = start_error
        self.queue: FakeQueue | None = None
        self.event: FakeEvent | None = None
        self.process: FakeProcess | None = None
        self.queue_calls = 0
        self.event_calls = 0
        self.process_calls = 0

    def Queue(self, *, maxsize: int) -> FakeQueue:
        """Return one controlled queue."""
        self.queue_calls += 1
        self.queue = FakeQueue(maxsize)
        return self.queue

    def Event(self) -> FakeEvent:
        """Return one controlled event."""
        self.event_calls += 1
        self.event = FakeEvent()
        return self.event

    def Process(self, **values: Any) -> FakeProcess:
        """Return one controlled process."""
        self.process_calls += 1
        self.process = FakeProcess(
            **values,
            stop_on_join=self.stop_on_join,
            start_error=self.start_error,
        )
        return self.process


class SessionReporter:
    """Record the reporter operations used by a session."""

    def __init__(
        self,
        *,
        observer_pid: int | None = None,
        observer_create_time: float | None = None,
        start_error: BaseException | None = None,
        refresh_error: Exception | None = None,
    ) -> None:
        self.enabled = True
        self.active = False
        self.observer_pid = observer_pid
        self.observer_create_time = observer_create_time
        self.start_error = start_error
        self.refresh_error = refresh_error
        self.started = 0
        self.refreshed = 0
        self.stopped = 0
        self.outcomes: list[SummaryOutcome] = []

    def start(self) -> None:
        """Record startup or raise its configured error."""
        self.started += 1
        if self.start_error is not None:
            raise self.start_error

    def refresh(self) -> None:
        """Record one refresh."""
        self.refreshed += 1
        if self.refresh_error is not None:
            raise self.refresh_error

    def stop(self) -> None:
        """Record one stop."""
        self.stopped += 1

    def set_outcome(self, outcome: SummaryOutcome) -> None:
        """Record one explicit summary outcome."""
        self.outcomes.append(outcome)


class FakeSampler:
    """Record exact process exclusions."""

    def __init__(self) -> None:
        self.exclusions: list[tuple[int, float | None]] = []

    def exclude_process(self, pid: int, create_time: float | None) -> None:
        """Record one excluded process identity."""
        self.exclusions.append((pid, create_time))


def make_console(*, interactive: bool = True) -> tuple[Console, StringIO]:
    """Return a controlled Rich console and its output stream."""
    stream = StringIO()
    console = Console(
        file=stream,
        force_terminal=interactive,
        color_system=None,
        width=120,
    )
    return console, stream


def completed_metrics() -> MetricsAggregator:
    """Return one completed pipeline aggregator."""
    metrics = MetricsAggregator()
    metrics.apply(
        MetricEvent.create(
            "stage_started",
            "traces",
            total_episodes=1,
            expected_rows=12,
        )
    )
    metrics.apply(
        MetricEvent.create(
            "stage_completed",
            "traces",
            episodes=1,
            rows=12,
        )
    )
    return metrics


def test_observer_startup_uses_a_bounded_queue_and_initial_snapshot(monkeypatch):
    metrics = MetricsAggregator()
    context = FakeContext()
    console, _stream = make_console()
    monkeypatch.setattr(reporter_module, "_process_create_time", lambda _pid: 123.5)
    reporter = TextualReporter(metrics, console=console, context=context)

    reporter.start()

    assert context.queue_calls == context.event_calls == context.process_calls == 1
    assert context.queue is not None
    assert context.queue.maxsize == 1
    assert len(context.queue.items) == 1
    assert isinstance(context.queue.items[0], PipelineSnapshot)
    assert context.process is not None
    assert context.process.started
    assert context.process.target is reporter_module.run_textual_observer
    assert context.process.args[0] is context.queue
    assert context.process.args[1] is context.event
    assert context.process.name == "avalanche-textual-observer"
    assert context.process.daemon
    assert reporter.observer_pid == context.process.pid
    assert reporter.observer_create_time == pytest.approx(123.5)


def test_refresh_replaces_the_stale_pending_snapshot():
    metrics = MetricsAggregator()
    context = FakeContext()
    console, _stream = make_console()
    reporter = TextualReporter(metrics, console=console, context=context)
    reporter.start()
    metrics.apply(MetricEvent.create("stage_started", "traces", total_episodes=2))

    reporter.refresh()

    assert context.queue is not None
    assert len(context.queue.items) == 1
    latest = context.queue.items[0]
    assert isinstance(latest, PipelineSnapshot)
    assert latest.total_stages == 1
    assert latest.stage("traces").total_episodes == 2


def test_clean_shutdown_sets_the_event_and_closes_the_queue():
    context = FakeContext()
    console, _stream = make_console()
    reporter = TextualReporter(
        completed_metrics(),
        console=console,
        context=context,
        join_timeout=0.25,
    )
    reporter.start()

    reporter.stop()

    assert context.event is not None and context.event.is_set()
    assert context.process is not None
    assert context.process.join_timeouts == [0.25]
    assert not context.process.terminated
    assert context.queue is not None
    assert context.queue.cancelled_join
    assert context.queue.closed


def test_shutdown_terminates_an_observer_that_does_not_exit():
    context = FakeContext(stop_on_join=False)
    console, _stream = make_console()
    reporter = TextualReporter(
        completed_metrics(),
        console=console,
        context=context,
        join_timeout=0.1,
    )
    reporter.start()

    reporter.stop()

    assert context.process is not None
    assert context.process.terminated
    assert context.process.join_timeouts == [0.1, 0.1]


def test_refresh_ignores_a_crashed_observer():
    metrics = MetricsAggregator()
    context = FakeContext()
    console, _stream = make_console()
    reporter = TextualReporter(metrics, console=console, context=context)
    reporter.start()
    assert context.process is not None
    context.process.alive = False
    assert context.queue is not None
    pending = tuple(context.queue.items)
    metrics.apply(MetricEvent.create("stage_started", "traces"))

    reporter.refresh()

    assert tuple(context.queue.items) == pending
    assert not reporter.active


def test_a_crashed_observer_restores_the_terminal_only_once(monkeypatch):
    context = FakeContext()
    console, _stream = make_console()
    restorations: list[object] = []
    monkeypatch.setattr(
        reporter_module,
        "restore_terminal",
        lambda _stream, state: restorations.append(state),
    )
    reporter = TextualReporter(
        MetricsAggregator(),
        console=console,
        context=context,
    )
    reporter.start()
    assert context.process is not None
    context.process.alive = False

    assert not reporter.active
    assert not reporter.active
    reporter.stop()

    assert len(restorations) == 1


def test_observer_disables_signals_and_mouse_capture(monkeypatch):
    calls: dict[str, object] = {}

    class App:
        def __init__(self, queue, event, *, workload_pid):
            calls["app"] = (queue, event, workload_pid)

        def run(self, **values: object) -> None:
            calls["run"] = values

    monkeypatch.setattr(reporter_module, "PipelineApp", App)
    monkeypatch.setattr(
        reporter_module,
        "_attach_observer_terminal",
        lambda path, handles: calls.setdefault("terminal", (path, handles)),
    )
    monkeypatch.setattr(
        reporter_module.signal,
        "signal",
        lambda name, handler: calls.setdefault("signal", (name, handler)),
    )
    queue = FakeQueue(1)
    event = FakeEvent()

    reporter_module.run_textual_observer(queue, event, 12_345)

    assert calls["app"] == (queue, event, 12_345)
    assert calls["terminal"] == (None, None)
    assert calls["run"] == {"mouse": False}
    assert calls["signal"] == (
        reporter_module.signal.SIGINT,
        reporter_module.signal.SIG_IGN,
    )


def test_terminal_attachment_updates_textual_standard_streams(monkeypatch):
    input_stream = StringIO()
    output_stream = StringIO()
    original_streams = (
        sys.stdin,
        sys.stdout,
        sys.stderr,
        sys.__stdin__,
        sys.__stdout__,
        sys.__stderr__,
    )

    def open_descriptor(descriptor, mode="r", **_values):
        return input_stream if descriptor == 11 else output_stream

    monkeypatch.setattr(reporter_module.os, "fdopen", open_descriptor)
    try:
        reporter_module._attach_observer_terminal(None, (11, 12))

        assert sys.stdin is sys.__stdin__ is input_stream
        assert sys.stdout is sys.__stdout__ is output_stream
        assert sys.stderr is sys.__stderr__ is output_stream
    finally:
        (
            sys.stdin,
            sys.stdout,
            sys.stderr,
            sys.__stdin__,
            sys.__stdout__,
            sys.__stderr__,
        ) = original_streams


def test_terminal_restoration_reapplies_the_saved_terminal_mode(monkeypatch):
    if reporter_module.termios is None:
        pytest.skip("this platform does not provide terminal modes")
    calls: list[tuple[int, int, list[object]]] = []
    attributes: list[object] = [1, 2, 3]
    monkeypatch.setattr(
        reporter_module.termios,
        "tcsetattr",
        lambda descriptor, when, values: calls.append((descriptor, when, values)),
    )
    stream = StringIO()

    reporter_module.restore_terminal(stream, (17, attributes))

    assert calls == [(17, reporter_module.termios.TCSANOW, attributes)]
    assert stream.getvalue() == reporter_module._TERMINAL_RESET


def test_a_broken_snapshot_queue_is_ignored():
    class BrokenQueue:
        def put_nowait(self, _item):
            raise BrokenPipeError

    reporter_module.publish_latest_snapshot(
        BrokenQueue(),  # type: ignore[arg-type]
        MetricsAggregator().snapshot(),
    )


def test_startup_interruption_reaps_the_started_observer(monkeypatch):
    context = FakeContext(stop_on_join=False)
    console, _stream = make_console()
    reporter = TextualReporter(
        MetricsAggregator(),
        console=console,
        context=context,
    )
    monkeypatch.setattr(
        reporter_module,
        "_process_create_time",
        lambda _pid: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        reporter.start()

    assert context.event is not None and context.event.is_set()
    assert context.process is not None and context.process.terminated
    assert not context.process.alive
    assert context.queue is not None and context.queue.closed
    assert reporter.observer_pid is None
    assert "INTERRUPTED" in _stream.getvalue()


def test_startup_failure_disables_the_observer_and_closes_its_queue():
    error = RuntimeError("observer start failed")
    context = FakeContext(start_error=error)
    console, _stream = make_console()
    reporter = TextualReporter(
        MetricsAggregator(),
        console=console,
        context=context,
    )

    with pytest.raises(RuntimeError, match="observer start failed"):
        reporter.start()

    assert not reporter.enabled
    assert reporter.observer_pid is None
    assert not reporter.active
    assert context.queue is not None
    assert context.queue.cancelled_join
    assert context.queue.closed


def test_session_isolates_a_reporter_startup_failure():
    error = RuntimeError("observer start failed")
    reporter = SessionReporter(start_error=error)
    session = ObservabilitySession(
        reporter=reporter,  # type: ignore[arg-type]
        sample_resources=False,
    )

    session.start()
    session.close()

    assert session.background_error is error
    assert not reporter.enabled
    assert reporter.refreshed == 1
    assert reporter.stopped == 1


def test_session_closes_and_rethrows_a_startup_interruption():
    reporter = SessionReporter(start_error=KeyboardInterrupt())
    session = ObservabilitySession(
        reporter=reporter,  # type: ignore[arg-type]
        sample_resources=False,
    )

    with pytest.raises(KeyboardInterrupt):
        session.start()

    assert reporter.outcomes == [SummaryOutcome.INTERRUPTED]
    assert reporter.refreshed == 1
    assert reporter.stopped == 1


def test_session_stops_the_reporter_after_a_refresh_failure():
    error = RuntimeError("refresh failed")
    reporter = SessionReporter(refresh_error=error)
    session = ObservabilitySession(
        reporter=reporter,  # type: ignore[arg-type]
        sample_resources=False,
    )

    session.close()

    assert session.background_error is error
    assert reporter.refreshed == 1
    assert reporter.stopped == 1


def test_terminal_is_restored_before_exactly_one_summary():
    context = FakeContext()
    console, stream = make_console()
    reporter = TextualReporter(
        completed_metrics(),
        console=console,
        context=context,
    )
    reporter.start()

    reporter.stop()
    reporter.stop()

    output = stream.getvalue()
    restore = "\x1b[0m\x1b[?25h\x1b[?1049l"
    assert output.count(restore) == 1
    assert output.count("COMPLETED") == 1
    assert output.index(restore) < output.index("COMPLETED")


@pytest.mark.parametrize(
    ("outcome", "failed", "expected", "stage_count"),
    (
        (None, False, "COMPLETED ✓", "1/1 stages"),
        (None, True, "FAILED ✗", "0/1 stages"),
        (SummaryOutcome.INTERRUPTED, False, "INTERRUPTED !", "1/1 stages"),
    ),
)
def test_compact_summaries_distinguish_each_outcome(
    outcome,
    failed,
    expected,
    stage_count,
):
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "traces", total_episodes=1))
    if failed:
        metrics.apply(
            MetricEvent.create(
                "stage_failed",
                "traces",
                error="generation stopped",
            )
        )
    else:
        metrics.apply(
            MetricEvent.create(
                "stage_completed",
                "traces",
                episodes=1,
                rows=10,
            )
        )

    summary = compact_summary(metrics.snapshot(), outcome)

    assert summary.plain.startswith(expected)
    assert stage_count in summary.plain


def test_session_propagates_keyboard_interrupt_and_sets_the_outcome():
    reporter = SessionReporter()
    session = ObservabilitySession(
        reporter=reporter,  # type: ignore[arg-type]
        sample_resources=False,
    )

    with pytest.raises(KeyboardInterrupt):
        with session:
            raise KeyboardInterrupt

    assert reporter.outcomes == [SummaryOutcome.INTERRUPTED]
    assert reporter.refreshed == 1
    assert reporter.stopped == 1


def test_session_excludes_the_observer_process_identity(monkeypatch):
    sampler = FakeSampler()
    monkeypatch.setattr(session_module, "ProcessTreeSampler", lambda: sampler)
    reporter = SessionReporter(
        observer_pid=42_424,
        observer_create_time=123.5,
    )
    session = ObservabilitySession(reporter=reporter)  # type: ignore[arg-type]
    session._start_background = lambda: None  # type: ignore[method-assign]

    session.start()
    session.close()

    assert sampler.exclusions == [(42_424, 123.5)]


def test_interruption_targets_the_existing_process_group(monkeypatch):
    calls: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(reporter_module.os, "getpgid", lambda pid: pid + 10)
    monkeypatch.setattr(
        reporter_module.os,
        "killpg",
        lambda group, value: calls.append((group, value)),
    )
    monkeypatch.setattr(
        reporter_module.os,
        "kill",
        lambda _pid, _value: pytest.fail("the PID fallback was used"),
    )

    reporter_module._interrupt_process(100)

    assert calls == [(110, signal.SIGINT)]


@pytest.mark.parametrize(
    ("interactive", "enabled"),
    ((False, None), (True, False)),
)
def test_disabled_reporter_does_not_create_an_observer(interactive, enabled):
    context = FakeContext()
    console, stream = make_console(interactive=interactive)
    reporter = TextualReporter(
        completed_metrics(),
        console=console,
        context=context,
        enabled=enabled,
    )

    reporter.start()
    reporter.refresh()
    reporter.stop()

    assert context.queue_calls == 0
    assert context.event_calls == 0
    assert context.process_calls == 0
    assert stream.getvalue().count("COMPLETED") == 1

"""Render observability snapshots in a Textual application."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
from collections.abc import Callable, Mapping
from datetime import timedelta
from enum import StrEnum
from importlib import import_module
from importlib.util import find_spec
from multiprocessing import reduction
from queue import Empty, Full
from threading import RLock
from typing import IO, Any, Protocol

import humanize
import psutil  # type: ignore[import-untyped]
from rich import box
from rich.console import Console, Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.events import Resize
from textual.widgets import Footer, Header, Static

from avalanche.observability.metrics import (
    GateSnapshot,
    GRUState,
    MetricsAggregator,
    PipelineSnapshot,
    SignificantEvent,
    StageSnapshot,
    StageStatus,
)

termios = import_module("termios") if find_spec("termios") is not None else None

_TERMINAL_RESET = (
    "\x1b[?2004l"
    "\x1b[?1004l"
    "\x1b[?1000l"
    "\x1b[?1002l"
    "\x1b[?1003l"
    "\x1b[?1006l"
    "\x1b[?7h"
    "\x1b[<u"
    "\x1b[0m"
    "\x1b[?25h"
    "\x1b[?1049l"
)


class SummaryOutcome(StrEnum):
    """Identify the final workload result."""

    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class SnapshotQueue(Protocol):
    """Define the queue operations used by the observer."""

    def get_nowait(self) -> object:
        """Return one available item."""

    def put_nowait(self, item: object) -> object:
        """Put one item without waiting."""


class StopEvent(Protocol):
    """Define the stop event used by the observer."""

    def is_set(self) -> bool:
        """Return whether shutdown was requested."""

    def set(self) -> None:
        """Request shutdown."""


class _TerminalDescriptor:
    """Transfer one terminal descriptor during process spawning."""

    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor

    def __reduce__(self) -> tuple[Callable[..., int], tuple[Any]]:
        duplicate = reduction.DupFd(self.descriptor)
        return (_detach_descriptor, (duplicate,))


class ReportView(VerticalScroll):
    """Show one focusable and scrollable report view."""

    def __init__(self, view_number: int, title: str) -> None:
        super().__init__(
            Static(_placeholder(title), classes="view-content"),
            id=f"view-{view_number}",
            classes="report-view",
            can_focus=True,
            can_focus_children=False,
        )
        self.view_number = view_number
        self.view_title = title

    def update_content(self, renderable: RenderableType) -> None:
        """Replace the current view contents."""
        self.query_one(Static).update(renderable)

    def action_scroll_up(self) -> None:
        self._scroll_target().scroll_up()

    def action_scroll_down(self) -> None:
        self._scroll_target().scroll_down()

    def action_page_up(self) -> None:
        self._scroll_target().scroll_page_up()

    def action_page_down(self) -> None:
        self._scroll_target().scroll_page_down()

    def action_scroll_home(self) -> None:
        self._scroll_target().scroll_home()

    def action_scroll_end(self) -> None:
        self._scroll_target().scroll_end()

    def _scroll_target(self) -> VerticalScroll:
        if self.max_scroll_y > 0:
            return self
        return self.app.query_one("#dashboard", VerticalScroll)


class EventHistoryView(VerticalScroll):
    """Show the bounded event history with follow controls."""

    can_focus = True

    def __init__(self) -> None:
        super().__init__(
            Static("Rows — / — · FOLLOW", classes="range-label"),
            Static(_placeholder("Significant events"), classes="view-content"),
            id="view-5",
            classes="report-view",
            can_focus=True,
            can_focus_children=False,
        )
        self.view_number = 5
        self.view_title = "Events"
        self.following = True
        self._events: tuple[SignificantEvent, ...] = ()

    @property
    def row_range(self) -> tuple[int, int, int]:
        """Return the visible event range and total count."""
        total = len(self._events)
        if total == 0:
            return (0, 0, 0)
        visible_rows = max(self.size.height - 4, 1)
        maximum_start = max(total - visible_rows + 1, 1)
        if self.following or self.scroll_y >= self.max_scroll_y:
            start = maximum_start
        elif self.scroll_y <= 0 or self.max_scroll_y <= 0:
            start = 1
        else:
            fraction = float(self.scroll_y) / float(self.max_scroll_y)
            start = 1 + round((maximum_start - 1) * fraction)
        end = min(start + visible_rows - 1, total)
        return (start, end, total)

    def update_events(self, events: tuple[SignificantEvent, ...]) -> None:
        """Replace the event rows and preserve the follow state."""
        self._events = events[-200:]
        self.query(".view-content").first(Static).update(
            _event_table(self._events)
            if self._events
            else _placeholder("Significant events")
        )
        if self.following:
            self.call_after_refresh(self._follow_end)
        else:
            self.call_after_refresh(self._update_range)

    def restore_follow(self) -> None:
        """Restore automatic event following."""
        self.following = True
        self.call_after_refresh(self._follow_end)

    def refresh_range(self) -> None:
        """Refresh the range after a layout change."""
        callback = self._follow_end if self.following else self._update_range
        self.call_after_refresh(callback)

    def action_scroll_up(self) -> None:
        self._stop_following()
        self.scroll_up(animate=False, immediate=True)
        self.call_after_refresh(self._update_range)

    def action_scroll_down(self) -> None:
        self._stop_following()
        self.scroll_down(animate=False, immediate=True)
        self.call_after_refresh(self._update_range)

    def action_page_up(self) -> None:
        self._stop_following()
        self.scroll_page_up(animate=False)
        self.call_after_refresh(self._update_range)

    def action_page_down(self) -> None:
        self._stop_following()
        self.scroll_page_down(animate=False)
        self.call_after_refresh(self._update_range)

    def action_scroll_home(self) -> None:
        self._stop_following()
        self.scroll_home(animate=False, immediate=True)
        self.call_after_refresh(self._update_range)

    def action_scroll_end(self) -> None:
        self._stop_following()
        self.scroll_end(animate=False, immediate=True)
        self.call_after_refresh(self._update_range)

    def _stop_following(self) -> None:
        self.following = False
        self._update_range()

    def _follow_end(self) -> None:
        self.scroll_end(animate=False, immediate=True)
        self._update_range()

    def _update_range(self) -> None:
        start, end, total = self.row_range
        mode = "FOLLOW" if self.following else "PAUSED"
        rows = "— / —" if total == 0 else f"{start}–{end} / {total}"
        self.query_one(".range-label", Static).update(f"Rows {rows} · {mode}")


class PipelineApp(App[None]):
    """Display one live pipeline snapshot."""

    TITLE = "AVALANCHE"
    SUB_TITLE = "Dataset pipeline observer"
    ENABLE_COMMAND_PALETTE = False
    HORIZONTAL_BREAKPOINTS = [
        (0, "-narrow"),
        (80, "-medium"),
        (120, "-wide"),
    ]
    VERTICAL_BREAKPOINTS = [
        (0, "-short"),
        (24, "-medium-tall"),
        (32, "-tall"),
    ]
    BINDINGS = [
        Binding(
            "tab",
            "focus_next_view",
            "Views",
            key_display="Tab/⇧Tab",
            priority=True,
        ),
        Binding(
            "shift+tab",
            "focus_previous_view",
            "",
            show=False,
            priority=True,
        ),
        Binding("1", "select_view(1)", "Select", key_display="1–5"),
        Binding("2", "select_view(2)", "", show=False),
        Binding("3", "select_view(3)", "", show=False),
        Binding("4", "select_view(4)", "", show=False),
        Binding("5", "select_view(5)", "", show=False),
        Binding(
            "up",
            "scroll_view_up",
            "Scroll",
            key_display="↑/↓",
            priority=True,
        ),
        Binding("down", "scroll_view_down", "", show=False, priority=True),
        Binding(
            "pageup",
            "page_view_up",
            "Page",
            key_display="PgUp/PgDn",
            priority=True,
        ),
        Binding("pagedown", "page_view_down", "", show=False, priority=True),
        Binding(
            "home",
            "view_home",
            "Bounds",
            key_display="Home/End",
            priority=True,
        ),
        Binding("end", "view_end", "", show=False, priority=True),
        Binding("f", "follow_events", "Follow", key_display="F"),
        Binding(
            "ctrl+c",
            "interrupt_workload",
            "",
            show=False,
            priority=True,
        ),
    ]
    CSS = """
    Screen {
        background: #0b1116;
        color: #d7e1e8;
    }

    Header {
        dock: top;
        height: 1;
        background: #162630;
        color: #eef7fa;
    }

    Footer {
        dock: bottom;
        height: 1;
        background: #162630;
        color: #d7e1e8;
    }

    #dashboard {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        padding: 0 1;
        overflow-y: auto;
        scrollbar-color: #52778b;
        scrollbar-background: #111b22;
    }

    .report-view {
        width: 100%;
        height: auto;
        min-height: 12;
        margin: 0 1 1 0;
        padding: 0 1;
        border: round #355266;
        background: #101920;
        scrollbar-color: #52778b;
        scrollbar-background: #101920;
    }

    .report-view:focus {
        border: heavy #59b7a9;
    }

    .view-content {
        width: 1fr;
        height: auto;
    }

    .range-label {
        dock: top;
        height: 1;
        color: #9bb2bf;
        background: #101920;
        text-align: right;
    }

    Screen.-medium #view-5,
    Screen.-wide #view-5 {
        height: 18;
    }

    Screen.-wide.-tall #dashboard {
        layout: grid;
        grid-size: 2;
        grid-columns: 1fr 1fr;
        grid-gutter: 1;
    }

    Screen.-wide.-tall .report-view {
        width: 100%;
        height: 1fr;
        min-height: 12;
        margin: 0;
    }

    Screen.-wide.-tall #view-5 {
        height: 1fr;
    }

    Screen.-narrow #dashboard,
    Screen.-short #dashboard {
        overflow-y: hidden;
        padding: 0;
    }

    Screen.-narrow .report-view,
    Screen.-short .report-view {
        display: none;
    }

    Screen.-narrow .report-view.-selected,
    Screen.-short .report-view.-selected {
        display: block;
        width: 100%;
        height: 1fr;
        min-height: 1;
        margin: 0;
    }

    Screen.-narrow #view-5,
    Screen.-short #view-5 {
        height: 1fr;
    }
    """

    def __init__(
        self,
        snapshot_queue: SnapshotQueue | None = None,
        stop_event: StopEvent | None = None,
        *,
        workload_pid: int | None = None,
        initial_snapshot: PipelineSnapshot | None = None,
        interrupt: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        self.snapshot_queue = snapshot_queue
        self.stop_event = stop_event
        self.workload_pid = workload_pid
        self.latest_snapshot = initial_snapshot
        self.selected_view = 1
        self._interrupt = interrupt or _interrupt_process

    @property
    def layout_mode(self) -> str:
        """Return the active responsive layout name."""
        width, height = self.viewport_size
        if width < 80 or height < 24:
            return "focused"
        if width >= 120 and height >= 32:
            return "wide"
        return "dashboard"

    @property
    def mounted_view_count(self) -> int:
        """Return the number of mounted report views."""
        return len(self.query(".report-view"))

    def compose(self) -> ComposeResult:
        """Mount the fixed shell and each report view."""
        yield Header(show_clock=True)
        yield VerticalScroll(
            ReportView(1, "Pipeline and progress"),
            ReportView(2, "Worker activity"),
            ReportView(3, "Resources and output"),
            ReportView(4, "Semantics and validation"),
            EventHistoryView(),
            id="dashboard",
            can_focus=False,
        )
        yield Footer()

    def on_mount(self) -> None:
        """Start queue polling and select the first view."""
        self.query_one("#view-1", ReportView).add_class("-selected")
        self.query_one("#view-1", ReportView).focus()
        if self.latest_snapshot is not None:
            self.apply_snapshot(self.latest_snapshot)
        self.set_interval(0.05, self.poll_snapshots)
        self._update_subtitle()

    def on_resize(self, _event: Resize) -> None:
        """Update the layout label after a terminal resize."""
        self.call_after_refresh(self._update_subtitle)
        self.query_one(EventHistoryView).refresh_range()

    def poll_snapshots(self) -> None:
        """Apply the newest pending snapshot without blocking."""
        latest = drain_latest_snapshot(self.snapshot_queue)
        if latest is not None:
            self.apply_snapshot(latest)
        if self.stop_event is not None and self.stop_event.is_set():
            latest = drain_latest_snapshot(self.snapshot_queue)
            if latest is not None:
                self.apply_snapshot(latest)
            self.exit()

    def apply_snapshot(self, snapshot: PipelineSnapshot) -> None:
        """Apply one immutable snapshot to all mounted views."""
        self.latest_snapshot = snapshot
        self.query_one("#view-1", ReportView).update_content(_pipeline_view(snapshot))
        self.query_one("#view-2", ReportView).update_content(_worker_view(snapshot))
        self.query_one("#view-3", ReportView).update_content(_resource_view(snapshot))
        self.query_one("#view-4", ReportView).update_content(_semantic_view(snapshot))
        self.query_one(EventHistoryView).update_events(snapshot.recent_events)

    def action_select_view(self, number: int) -> None:
        """Select and focus one report view."""
        if number not in range(1, 6):
            return
        current = self.query_one(f"#view-{self.selected_view}")
        selected = self.query_one(f"#view-{number}")
        current.remove_class("-selected")
        selected.add_class("-selected")
        self.selected_view = number
        selected.focus()
        self._update_subtitle()

    def action_focus_next_view(self) -> None:
        """Focus the next report view."""
        self.action_select_view(self.selected_view % 5 + 1)

    def action_focus_previous_view(self) -> None:
        """Focus the previous report view."""
        self.action_select_view((self.selected_view - 2) % 5 + 1)

    def action_scroll_view_up(self) -> None:
        """Scroll the selected view upward."""
        self._selected_view().action_scroll_up()

    def action_scroll_view_down(self) -> None:
        """Scroll the selected view downward."""
        self._selected_view().action_scroll_down()

    def action_page_view_up(self) -> None:
        """Scroll the selected view upward by one page."""
        self._selected_view().action_page_up()

    def action_page_view_down(self) -> None:
        """Scroll the selected view downward by one page."""
        self._selected_view().action_page_down()

    def action_view_home(self) -> None:
        """Scroll the selected view to its start."""
        self._selected_view().action_scroll_home()

    def action_view_end(self) -> None:
        """Scroll the selected view to its end."""
        self._selected_view().action_scroll_end()

    def action_follow_events(self) -> None:
        """Restore event following and focus the event view."""
        self.action_select_view(5)
        self.query_one(EventHistoryView).restore_follow()

    def action_interrupt_workload(self) -> None:
        """Forward a terminal interruption to the workload process."""
        if self.workload_pid is not None:
            self._interrupt(self.workload_pid)

    def _selected_view(self) -> VerticalScroll:
        return self.query_one(f"#view-{self.selected_view}", VerticalScroll)

    def _update_subtitle(self) -> None:
        selected = self.query_one(f"#view-{self.selected_view}")
        title = getattr(selected, "view_title", "View")
        self.sub_title = f"{self.layout_mode} · {self.selected_view}/5 {title}"


class TextualReporter:
    """Run a Textual observer without blocking the workload."""

    def __init__(
        self,
        aggregator: MetricsAggregator,
        *,
        enabled: bool | None = None,
        console: Console | None = None,
        context: Any | None = None,
        join_timeout: float = 2.0,
    ) -> None:
        if join_timeout < 0.0:
            raise ValueError("the observer join timeout must be nonnegative")
        self.aggregator = aggregator
        self.console = console or Console(file=sys.stdout)
        self.enabled = self.console.is_terminal if enabled is None else enabled
        self._terminal_path = _terminal_device(sys.stdin) if self.enabled else None
        self._terminal_handles = (
            _terminal_handles(sys.stdin, self.console.file) if self.enabled else None
        )
        self._terminal_state = (
            _capture_terminal_state(sys.stdin) if self.enabled else None
        )
        self.join_timeout = join_timeout
        self._context = context or mp.get_context("spawn")
        self._queue: Any | None = None
        self._stop_event: Any | None = None
        self._process: Any | None = None
        self._lock = RLock()
        self._started = False
        self._stopped = False
        self._summary_printed = False
        self._terminal_restored = False
        self._outcome: SummaryOutcome | None = None
        self._observer_create_time: float | None = None

    @property
    def active(self) -> bool:
        """Return true when the observer process is alive."""
        process = self._process
        if process is None:
            return False
        try:
            alive = bool(process.is_alive())
        except AssertionError, OSError, ValueError:
            alive = False
        if self._started and not self._stopped and not alive:
            self._restore_terminal_once()
        return alive

    @property
    def observer_pid(self) -> int | None:
        """Return the observer process identity."""
        process = self._process
        return None if process is None else process.pid

    @property
    def observer_create_time(self) -> float | None:
        """Return the observer creation time."""
        return self._observer_create_time

    def set_outcome(self, outcome: SummaryOutcome) -> None:
        """Set the final summary outcome."""
        self._outcome = outcome

    def start(self) -> None:
        """Start the observer when interactive output is enabled."""
        with self._lock:
            if not self.enabled or self._started or self._stopped:
                return
            self._started = True
            queue = self._context.Queue(maxsize=1)
            stop_event = self._context.Event()
            publish_latest_snapshot(queue, self.aggregator.snapshot())
            process = self._context.Process(
                target=run_textual_observer,
                args=(
                    queue,
                    stop_event,
                    os.getpid(),
                    self._terminal_path,
                    self._terminal_handles,
                ),
                name="avalanche-textual-observer",
                daemon=True,
            )
            self._queue = queue
            self._stop_event = stop_event
            self._process = process
            process_started = False
            try:
                process.start()
                process_started = True
                self._observer_create_time = _process_create_time(process.pid)
            except BaseException as error:
                if process_started:
                    _stop_observer_process(
                        process,
                        stop_event,
                        self.join_timeout,
                    )
                    self._restore_terminal_once()
                self._queue = None
                self._stop_event = None
                self._process = None
                self.enabled = False
                _close_snapshot_queue(queue)
                if isinstance(error, KeyboardInterrupt):
                    self.set_outcome(SummaryOutcome.INTERRUPTED)
                    self._print_summary(self.aggregator.snapshot())
                raise

    def refresh(self) -> None:
        """Publish the newest snapshot when the observer is active."""
        with self._lock:
            if not self.active or self._queue is None:
                return
            publish_latest_snapshot(self._queue, self.aggregator.snapshot())

    def stop(self) -> None:
        """Stop the observer, restore the terminal, and print a summary."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            snapshot = self.aggregator.snapshot()
            queue = self._queue
            stop_event = self._stop_event
            process = self._process
            if queue is not None:
                publish_latest_snapshot(queue, snapshot)
            if stop_event is not None:
                stop_event.set()
        try:
            if process is not None:
                _stop_observer_process(process, stop_event, self.join_timeout)
        finally:
            if queue is not None:
                _close_snapshot_queue(queue)
            if process is not None:
                self._restore_terminal_once()
            self._print_summary(snapshot)

    def _restore_terminal_once(self) -> None:
        with self._lock:
            if self._terminal_restored:
                return
            self._terminal_restored = True
        restore_terminal(self.console.file, self._terminal_state)

    def _print_summary(self, snapshot: PipelineSnapshot) -> None:
        with self._lock:
            if self._summary_printed:
                return
            self._summary_printed = True
        summary = compact_summary(snapshot, self._outcome)
        if self.enabled:
            self.console.print(summary)
        else:
            self.console.print(summary.plain, markup=False, highlight=False)

    def __enter__(self) -> TextualReporter:
        self.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        if exception_type is not None:
            self.set_outcome(
                SummaryOutcome.INTERRUPTED
                if issubclass(exception_type, KeyboardInterrupt)
                else SummaryOutcome.FAILED
            )
        self.stop()


def run_textual_observer(
    snapshot_queue: SnapshotQueue,
    stop_event: StopEvent,
    workload_pid: int,
    terminal_path: str | None = None,
    terminal_handles: tuple[Any, Any] | None = None,
) -> None:
    """Run the terminal observer in one child process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _attach_observer_terminal(terminal_path, terminal_handles)
    app = PipelineApp(
        snapshot_queue,
        stop_event,
        workload_pid=workload_pid,
    )
    app.run(mouse=False)


def publish_latest_snapshot(
    queue: SnapshotQueue,
    snapshot: PipelineSnapshot,
) -> None:
    """Replace a stale pending snapshot without blocking."""
    try:
        queue.put_nowait(snapshot)
        return
    except Full:
        pass
    except BrokenPipeError, ConnectionError, EOFError, OSError, ValueError:
        return
    try:
        queue.get_nowait()
    except Empty:
        return
    except BrokenPipeError, ConnectionError, EOFError, OSError, ValueError:
        return
    try:
        queue.put_nowait(snapshot)
    except Full, BrokenPipeError, ConnectionError, EOFError, OSError, ValueError:
        return


def drain_latest_snapshot(
    queue: SnapshotQueue | None,
) -> PipelineSnapshot | None:
    """Return the newest pending snapshot."""
    if queue is None:
        return None
    latest: PipelineSnapshot | None = None
    while True:
        try:
            item = queue.get_nowait()
        except Empty:
            break
        except BrokenPipeError, ConnectionError, EOFError, OSError:
            break
        if isinstance(item, PipelineSnapshot):
            latest = item
    return latest


def restore_terminal(
    stream: IO[str],
    state: tuple[int, list[Any]] | None = None,
) -> None:
    """Restore the normal screen, cursor, and text attributes."""
    if state is not None and termios is not None:
        descriptor, attributes = state
        try:
            termios.tcsetattr(descriptor, termios.TCSANOW, attributes)
        except OSError, ValueError:
            pass
    try:
        stream.write(_TERMINAL_RESET)
        stream.flush()
    except BrokenPipeError, OSError, ValueError:
        return


def compact_summary(
    snapshot: PipelineSnapshot,
    outcome: SummaryOutcome | None = None,
) -> Text:
    """Build one compact final workload summary."""
    resolved = outcome or _snapshot_outcome(snapshot)
    label, marker, style = {
        SummaryOutcome.COMPLETED: ("COMPLETED", "✓", "bold green"),
        SummaryOutcome.FAILED: ("FAILED", "✗", "bold red"),
        SummaryOutcome.INTERRUPTED: ("INTERRUPTED", "!", "bold yellow"),
    }[resolved]
    summary = Text.assemble((f"{label} {marker}", style))
    summary.append(
        f" · {snapshot.progress_fraction * 100.0:.1f}%"
        f" · {snapshot.completed_stages}/{snapshot.total_stages} stages"
    )
    rows = max((stage.rows_generated for stage in snapshot.stages), default=0)
    if rows:
        summary.append(f" · {humanize.intcomma(rows)} rows")
    if snapshot.failures:
        summary.append(f" · {snapshot.failures} failures", style="red")
    error = next(
        (stage.error for stage in reversed(snapshot.stages) if stage.error),
        None,
    )
    if error:
        summary.append(f" · {_short(error, 80)}", style="red")
    output = snapshot.run_context.get("model") or snapshot.run_context.get("dataset")
    if output:
        summary.append(f" · {output}")
    return summary


def _pipeline_view(state: PipelineSnapshot) -> RenderableType:
    stage = _current_stage(state)
    return Group(
        _section_title("1 · Pipeline and current progress"),
        _pipeline_table(state),
        _current_stage_table(stage),
        _training_table(stage),
    )


def _worker_view(state: PipelineSnapshot) -> RenderableType:
    stage = _current_stage(state)
    return Group(
        _section_title("2 · Worker activity and process usage"),
        _worker_table(stage),
        _process_table(stage),
    )


def _resource_view(state: PipelineSnapshot) -> RenderableType:
    stage = _current_stage(state)
    return Group(
        _section_title("3 · CPU, memory, and Parquet output"),
        _resource_table(stage),
        _parquet_table(stage),
    )


def _semantic_view(state: PipelineSnapshot) -> RenderableType:
    return Group(
        _section_title("4 · Semantic totals, configuration, and validation"),
        _semantic_table(state),
        _configuration_table(state.run_context),
        _validation_table(state),
        _gate_table(state.gate, state.gru_state),
    )


def _pipeline_table(state: PipelineSnapshot) -> Table:
    table = _table("Stage", "Progress", "Work", "Status")
    table.columns[1].justify = "right"
    table.columns[2].justify = "right"
    if not state.stages:
        table.add_row("Waiting for pipeline data", "—", "—", _status("PENDING"))
    for stage in state.stages:
        table.add_row(
            stage.label,
            f"{stage.percentage:5.1f}%",
            _stage_work(stage),
            _status(
                stage.status.value.replace("_", " ").upper(),
                _status_style(stage.status),
            ),
        )
    eta = _duration(state.overall_eta_seconds)
    table.caption = (
        f"Pipeline {state.progress_fraction * 100.0:.1f}%"
        f" · {state.completed_stages}/{state.total_stages} stages"
        f" · ETA {eta}"
    )
    return table


def _current_stage_table(stage: StageSnapshot | None) -> Table:
    table = _key_value_table("Current stage")
    if stage is None:
        _placeholder_rows(table, ("Stage", "Phase", "Progress", "Elapsed", "ETA"))
        return table
    table.add_row("Stage", stage.label)
    table.add_row("Phase", stage.phase)
    table.add_row("Progress", f"{stage.percentage:.1f}%")
    table.add_row(
        "Episodes",
        _count_pair(stage.episodes_completed, stage.total_episodes),
    )
    observed_rows = stage.rows_generated + stage.rows_in_progress
    table.add_row("Rows", _count_pair(observed_rows, stage.expected_rows))
    table.add_row("Elapsed", _duration(stage.elapsed_seconds))
    table.add_row("ETA", _duration(stage.eta_seconds))
    table.add_row("Episodes/s", _number(stage.episodes_per_second, 2))
    table.add_row("Rows/s", _number(stage.rows_per_second, 1))
    if stage.error:
        table.add_row("Error", Text(stage.error, style="red"))
    return table


def _training_table(stage: StageSnapshot | None) -> Table:
    table = _key_value_table("Training and calibration")
    if stage is None:
        _placeholder_rows(table, ("Model", "Epoch", "Loss", "Calibration"))
        return table
    training = stage.training
    table.add_row("Model", stage.current_model or "—")
    table.add_row("Epoch", _count_pair(training.epoch, training.total_epochs))
    table.add_row("Batch", _count_pair(training.batch, training.total_batches))
    table.add_row("Training loss", _number(training.training_loss))
    table.add_row("Validation loss", _number(training.validation_loss))
    table.add_row("Samples/s", _number(training.samples_per_second, 1))
    calibration = stage.calibration
    table.add_row(
        "Calibration",
        _status(
            calibration.status.value.replace("_", " ").upper(),
            _status_style(calibration.status),
        ),
    )
    table.add_row("Threshold", _number(calibration.threshold))
    return table


def _worker_table(stage: StageSnapshot | None) -> Table:
    table = _table(
        "Worker",
        "State",
        "Phase",
        "Item",
        "Episodes",
        "Active rows",
        "Total rows",
    )
    for index in (4, 5, 6):
        table.columns[index].justify = "right"
    if stage is None or not stage.workers:
        table.add_row("—", _status("UNAVAILABLE"), "—", "—", "—", "—", "—")
        return table
    for worker in stage.workers:
        state = _status("ACTIVE", "cyan") if worker.active else _status("IDLE", "dim")
        table.add_row(
            worker.worker_id,
            state,
            worker.phase,
            worker.current_item or "—",
            humanize.intcomma(worker.episodes_completed),
            humanize.intcomma(worker.current_rows),
            humanize.intcomma(worker.rows_generated),
        )
    return table


def _process_table(stage: StageSnapshot | None) -> Table:
    table = _table("PID", "Process", "CPU", "Memory", "RSS")
    for index in (0, 2, 3, 4):
        table.columns[index].justify = "right"
    sample = None if stage is None else stage.resources
    if sample is None or not sample.processes:
        table.add_row("—", "Workload process sample unavailable", "—", "—", "—")
        return table
    for process in sample.processes:
        table.add_row(
            str(process.pid),
            process.name,
            _percent(process.cpu_percent, raw=True),
            _percent(process.memory_percent),
            _bytes(process.rss_bytes),
        )
    return table


def _resource_table(stage: StageSnapshot | None) -> Table:
    table = _key_value_table("CPU and memory")
    sample = None if stage is None else stage.resources
    if sample is None:
        _placeholder_rows(
            table,
            (
                "Workload CPU",
                "Equivalent cores",
                "Machine capacity",
                "System CPU",
                "Workload memory",
                "System RAM",
                "Processes",
            ),
        )
        return table
    cores = getattr(sample, "tree_cpu_cores", None)
    capacity = getattr(sample, "tree_cpu_capacity_percent", None)
    cpu_count = getattr(sample, "logical_cpu_count", None)
    table.add_row("Workload CPU", _percent(sample.tree_cpu_percent, raw=True))
    table.add_row(
        "Equivalent cores",
        (
            "unavailable"
            if cores is None or cpu_count is None
            else f"{cores:.2f} / {cpu_count}"
        ),
    )
    table.add_row("Machine capacity", _percent(capacity))
    table.add_row("System CPU", _percent(sample.system_cpu_percent))
    table.add_row("Workload memory", _bytes(sample.tree_rss_bytes))
    table.add_row("System RAM", _percent(sample.system_memory_percent))
    table.add_row("Processes", str(sample.process_count))
    return table


def _parquet_table(stage: StageSnapshot | None) -> Table:
    table = _key_value_table("Parquet output")
    output = None if stage is None else stage.parquet
    if output is None:
        _placeholder_rows(
            table,
            (
                "Estimate state",
                "Written rows",
                "Buffered rows",
                "Parquet written",
                "Estimated final",
                "Average/row",
            ),
        )
        return table
    table.add_row("Estimate state", _status(output.state.upper(), "blue"))
    table.add_row("Written rows", humanize.intcomma(output.written_rows))
    table.add_row("Buffered rows", humanize.intcomma(output.buffered_rows))
    table.add_row("Parquet written", _bytes(output.written_bytes))
    prefix = "" if output.final else "~"
    table.add_row(
        "Final size" if output.final else "Estimated final",
        _bytes(output.estimated_final_bytes, prefix=prefix),
    )
    table.add_row("Average/row", _bytes(output.average_bytes_per_row))
    return table


def _semantic_table(state: PipelineSnapshot) -> Table:
    table = _key_value_table("Semantic totals")
    table.add_row(
        "Principal traces",
        humanize.intcomma(state.principal_traces_generated),
    )
    table.add_row(
        "Oracle true states",
        humanize.intcomma(state.oracle_true_states_generated),
    )
    table.add_row(
        "Oracle fallbacks",
        humanize.intcomma(state.oracle_fallbacks_generated),
    )
    table.add_row(
        "Fallback attempts",
        humanize.intcomma(state.fallback_generation_attempts),
    )
    table.add_row(
        "Fallback rate",
        f"{state.fallback_rate:.2%}" if state.fallback_rate is not None else "—",
    )
    table.add_row(
        "Retries · rejects · failures",
        f"{state.retries} · {state.rejected} · {state.failures}",
    )
    return table


def _configuration_table(values: Mapping[str, Any]) -> Table:
    table = _key_value_table("Resolved configuration")
    if not values:
        table.add_row("Configuration", "—")
        return table
    for name, value in list(values.items())[:16]:
        table.add_row(_label(name), _short(value))
    return table


def _validation_table(state: PipelineSnapshot) -> Table:
    table = _key_value_table("Validation statistics")
    stage = _latest_validation_stage(state)
    if stage is None:
        _placeholder_rows(
            table,
            (
                "Model",
                "Validation Brier score",
                "Validation average precision",
                "Validation accuracy",
            ),
        )
        return table
    table.add_row("Model", stage.current_model or "—")
    for name in (
        "validation_brier_score",
        "validation_average_precision",
        "validation_accuracy",
        "baseline_brier_score",
        "baseline_average_precision",
    ):
        table.add_row(_label(name), _number_value(stage.metrics.get(name)))
    return table


def _gate_table(gate: GateSnapshot | None, gru_state: GRUState) -> Table:
    table = _key_value_table("Validation gate")
    if gate is None:
        _placeholder_rows(table, ("Criterion", "Observed", "Required", "Result"))
    else:
        table.add_row("Criterion", gate.criterion)
        table.add_row("Metric", gate.metric_name)
        table.add_row("Observed", _number(gate.observed))
        table.add_row("Required", _number(gate.required))
        table.add_row(
            "Result",
            _status("PASSED", "green") if gate.passed else _status("FAILED", "red"),
        )
    gru_style = {
        GRUState.COMPLETE: "green",
        GRUState.NOT_REQUIRED: "green",
        GRUState.FAILED: "red",
        GRUState.TRAINING: "cyan",
    }.get(gru_state, "yellow")
    table.add_row(
        "GRU fallback",
        _status(
            gru_state.value.replace("_", " ").upper(),
            gru_style,
        ),
    )
    return table


def _event_table(events: tuple[SignificantEvent, ...]) -> Table:
    table = _table("#", "Stage", "Event", "Detail")
    table.columns[0].justify = "right"
    for column in table.columns:
        column.no_wrap = True
    for index, event in enumerate(events, start=1):
        style = "red" if event.kind in {"failure", "stage_failed"} else ""
        table.add_row(
            str(index),
            event.stage_id,
            Text(event.kind.replace("_", " ").upper(), style=style),
            _short(event.message, 100),
        )
    return table


def _current_stage(state: PipelineSnapshot) -> StageSnapshot | None:
    candidates = [
        stage
        for stage in state.stages
        if stage.status
        in {StageStatus.RUNNING, StageStatus.TRIGGERED, StageStatus.FAILED}
    ]
    if candidates:
        return candidates[-1]
    return state.stages[-1] if state.stages else None


def _latest_validation_stage(state: PipelineSnapshot) -> StageSnapshot | None:
    for stage in reversed(state.stages):
        if "validation_brier_score" in stage.metrics:
            return stage
    return None


def _snapshot_outcome(snapshot: PipelineSnapshot) -> SummaryOutcome:
    if any(stage.status == StageStatus.FAILED for stage in snapshot.stages):
        return SummaryOutcome.FAILED
    return SummaryOutcome.COMPLETED


def _stage_work(stage: StageSnapshot) -> str:
    if stage.total_episodes is not None:
        return _count_pair(stage.episodes_completed, stage.total_episodes) + " episodes"
    if stage.training.total_epochs is not None:
        return (
            _count_pair(stage.training.epoch, stage.training.total_epochs) + " epochs"
        )
    if stage.total_models is not None:
        return _count_pair(stage.completed_models, stage.total_models) + " models"
    return stage.phase


def _status_style(status: StageStatus) -> str:
    return {
        StageStatus.COMPLETE: "green",
        StageStatus.FAILED: "bold red",
        StageStatus.RUNNING: "cyan",
        StageStatus.TRIGGERED: "yellow",
        StageStatus.NOT_REQUIRED: "dim green",
    }.get(status, "dim")


def _table(*columns: str) -> Table:
    table = Table(
        box=box.SIMPLE_HEAD,
        expand=True,
        pad_edge=False,
        show_edge=False,
        row_styles=("", "dim"),
    )
    for column in columns:
        table.add_column(column, overflow="ellipsis")
    return table


def _key_value_table(title: str) -> Table:
    table = Table(
        title=title,
        box=None,
        expand=True,
        pad_edge=False,
        show_header=False,
    )
    table.add_column("Statistic", style="bold #9bb2bf", ratio=2)
    table.add_column("Value", justify="right", ratio=3, overflow="ellipsis")
    return table


def _section_title(value: str) -> Text:
    return Text(value, style="bold #78c6ba")


def _placeholder(title: str) -> Text:
    return Text(f"{title} · waiting for data", style="dim")


def _placeholder_rows(table: Table, names: tuple[str, ...]) -> None:
    for name in names:
        table.add_row(name, "—")


def _status(value: str, style: str = "dim") -> Text:
    return Text(value, style=style)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1_000.0:.0f} ms"
    return humanize.naturaldelta(timedelta(seconds=round(seconds)))


def _bytes(value: float | int | None, *, prefix: str = "") -> str:
    if value is None:
        return "unavailable"
    return prefix + humanize.naturalsize(value, binary=True)


def _percent(value: float | None, *, raw: bool = False) -> str:
    if value is None:
        return "unavailable"
    del raw
    number = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{number}%"


def _number(value: float | None, precision: int = 6) -> str:
    return "—" if value is None else f"{value:.{precision}g}"


def _number_value(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "—"
    return f"{value:.6g}"


def _count_pair(value: int, total: int | None) -> str:
    return (
        f"{humanize.intcomma(value)} / {humanize.intcomma(total)}"
        if total is not None
        else "—"
    )


def _label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _short(value: Any, limit: int = 80) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _interrupt_process(pid: int) -> None:
    get_process_group = getattr(os, "getpgid", None)
    kill_process_group = getattr(os, "killpg", None)
    if callable(get_process_group) and callable(kill_process_group):
        try:
            kill_process_group(get_process_group(pid), signal.SIGINT)
            return
        except OSError:
            pass
    try:
        os.kill(pid, signal.SIGINT)
    except OSError:
        return


def _process_create_time(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except OSError, psutil.AccessDenied, psutil.NoSuchProcess:
        return None


def _terminal_device(stream: IO[str]) -> str | None:
    try:
        if not stream.isatty():
            return None
        return os.ttyname(stream.fileno())
    except OSError, ValueError:
        return None


def _terminal_handles(
    input_stream: IO[str],
    output_stream: IO[str],
) -> tuple[Any, Any] | None:
    try:
        if not input_stream.isatty() or not output_stream.isatty():
            return None
        return (
            _TerminalDescriptor(input_stream.fileno()),
            _TerminalDescriptor(output_stream.fileno()),
        )
    except OSError, ValueError:
        return None


def _capture_terminal_state(stream: IO[str]) -> tuple[int, list[Any]] | None:
    if termios is None:
        return None
    try:
        if not stream.isatty():
            return None
        descriptor = stream.fileno()
        return descriptor, termios.tcgetattr(descriptor)
    except OSError, ValueError:
        return None


def _attach_observer_terminal(
    path: str | None,
    handles: tuple[Any, Any] | None = None,
) -> None:
    """Attach a spawned observer to the controlling terminal."""
    input_encoding = getattr(sys.__stdin__, "encoding", None) or "utf-8"
    output_encoding = getattr(sys.__stdout__, "encoding", None) or "utf-8"
    if handles is not None:
        input_stream = os.fdopen(
            _descriptor_number(handles[0]),
            encoding=input_encoding,
            errors="replace",
            buffering=1,
        )
        output_stream = os.fdopen(
            _descriptor_number(handles[1]),
            "w",
            encoding=output_encoding,
            errors="replace",
            buffering=1,
        )
    else:
        terminal_path = path or "/dev/tty"
        input_stream = open(
            terminal_path,
            encoding=input_encoding,
            errors="replace",
            buffering=1,
        )
        output_stream = open(
            terminal_path,
            "w",
            encoding=output_encoding,
            errors="replace",
            buffering=1,
        )
    sys.stdin = input_stream
    sys.stdout = output_stream
    sys.stderr = output_stream
    setattr(sys, "__stdin__", input_stream)
    setattr(sys, "__stdout__", output_stream)
    setattr(sys, "__stderr__", output_stream)


def _detach_descriptor(duplicate: Any) -> int:
    return int(duplicate.detach())


def _descriptor_number(value: Any) -> int:
    return int(value if isinstance(value, int) else value.detach())


def _stop_observer_process(
    process: Any,
    stop_event: StopEvent | None,
    timeout: float,
) -> None:
    if stop_event is not None:
        try:
            stop_event.set()
        except BaseException:
            pass
    try:
        process.join(timeout)
    except BaseException:
        pass
    try:
        alive = bool(process.is_alive())
    except BaseException:
        alive = True
    if not alive:
        return
    try:
        process.terminate()
    except BaseException:
        return
    try:
        process.join(timeout)
    except BaseException:
        pass


def _close_snapshot_queue(queue: Any) -> None:
    try:
        queue.cancel_join_thread()
    except AttributeError, OSError, ValueError:
        pass
    try:
        queue.close()
    except AttributeError, OSError, ValueError:
        pass

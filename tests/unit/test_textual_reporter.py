"""Test the Textual reporter interface and snapshot channel."""

from __future__ import annotations

import asyncio
import pickle
from dataclasses import FrozenInstanceError, replace
from io import StringIO
from queue import Empty, Full, Queue

import pytest
from rich.console import Console
from textual.widgets import Footer, Header, Static

from avalanche.observability import (
    MetricEvent,
    MetricsAggregator,
    PipelineSnapshot,
    ProcessResource,
    ResourceSample,
    SignificantEvent,
)
from avalanche.observability.reporter import (
    EventHistoryView,
    PipelineApp,
    ReportView,
    _resource_view,
    drain_latest_snapshot,
    publish_latest_snapshot,
)


def _snapshot(
    events: tuple[SignificantEvent, ...] = (),
    *,
    sequence: int = 0,
) -> PipelineSnapshot:
    """Return one pipeline snapshot with the supplied events."""
    metrics = MetricsAggregator()
    metrics.apply(
        MetricEvent.create(
            "run_config",
            "principal-traces",
            sequence=sequence,
            configuration="configs/experiments/monitor-training.yaml",
        )
    )
    metrics.apply(
        MetricEvent.create(
            "stage_started",
            "principal-traces",
            label="Principal traces",
            total_episodes=10,
            expected_rows=1_000,
            workers=4,
        )
    )
    return replace(metrics.snapshot(), recent_events=events)


def _events(count: int) -> tuple[SignificantEvent, ...]:
    """Return significant events with stable row identities."""
    return tuple(
        SignificantEvent(
            timestamp=float(index),
            stage_id="principal-traces",
            kind="message",
            message=f"event-{index:03d}",
        )
        for index in range(count)
    )


def test_resize_pilot_selects_each_layout_and_preserves_the_fixed_shell():
    async def exercise() -> None:
        app = PipelineApp(initial_snapshot=_snapshot())
        async with app.run_test(size=(100, 30)) as pilot:
            views = tuple(app.query(".report-view"))
            assert len(views) == 5
            assert all(view.is_mounted for view in views)

            cases = (
                ((120, 32), "wide"),
                ((119, 32), "dashboard"),
                ((120, 31), "dashboard"),
                ((80, 24), "dashboard"),
                ((79, 24), "focused"),
                ((80, 23), "focused"),
            )
            header = app.query_one(Header)
            footer = app.query_one(Footer)
            for (width, height), expected in cases:
                await pilot.resize_terminal(width, height)

                assert app.layout_mode == expected
                assert app.mounted_view_count == 5
                assert tuple(app.query(".report-view")) == views
                assert header.is_mounted
                assert footer.is_mounted
                assert header.region.y == 0
                assert footer.region.y == height - 1
                if expected == "wide":
                    x_positions = {view.region.x for view in views}
                    assert len(x_positions) == 2
                    assert views[0].region.y == views[1].region.y

    asyncio.run(exercise())


def test_number_keys_and_tab_keys_move_the_view_focus():
    async def exercise() -> None:
        app = PipelineApp(initial_snapshot=_snapshot())
        async with app.run_test(size=(79, 24)) as pilot:
            assert app.focused is app.query_one("#view-1", ReportView)

            await pilot.press("tab")
            assert app.focused is app.query_one("#view-2", ReportView)
            assert app.selected_view == 2

            await pilot.press("shift+tab")
            assert app.focused is app.query_one("#view-1", ReportView)
            assert app.selected_view == 1

            for number in range(1, 6):
                await pilot.press(str(number))
                assert app.selected_view == number
                assert app.focused is app.query_one(f"#view-{number}")
                assert app.query_one(f"#view-{number}").has_class("-selected")

    asyncio.run(exercise())


def test_each_view_starts_with_a_placeholder():
    async def exercise() -> None:
        app = PipelineApp()
        async with app.run_test(size=(120, 32)):
            contents = tuple(app.query(".view-content"))

            assert len(contents) == 5
            assert all("waiting for data" in str(widget.content) for widget in contents)

    asyncio.run(exercise())


def test_ctrl_c_forwards_the_interruption_to_the_workload():
    async def exercise() -> None:
        interrupted: list[int] = []
        app = PipelineApp(workload_pid=12_345, interrupt=interrupted.append)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press("ctrl+c")

            assert interrupted == [12_345]

    asyncio.run(exercise())


def test_event_keys_scroll_page_select_boundaries_and_restore_follow():
    async def exercise() -> None:
        app = PipelineApp(initial_snapshot=_snapshot(_events(250)))
        async with app.run_test(size=(79, 24)) as pilot:
            await pilot.press("5")
            history = app.query_one(EventHistoryView)
            await pilot.pause()

            assert history.max_scroll_y > 0
            assert history.following

            await pilot.press("home")
            await pilot.pause()
            assert not history.following
            assert history.scroll_target_y == 0

            await pilot.press("down")
            down = history.scroll_target_y
            assert down > 0

            await pilot.press("pagedown")
            page_down = history.scroll_target_y
            assert page_down > down

            await pilot.press("end")
            await pilot.pause()
            assert history.scroll_target_y == history.max_scroll_y

            await pilot.press("up")
            up = history.scroll_target_y
            assert up < history.max_scroll_y

            await pilot.press("pageup")
            assert history.scroll_target_y < up

            await pilot.press("f")
            await pilot.pause()
            assert history.following
            assert history.scroll_target_y == history.max_scroll_y

    asyncio.run(exercise())


def test_event_history_keeps_200_rows_and_shows_the_current_range():
    async def exercise() -> None:
        app = PipelineApp(initial_snapshot=_snapshot(_events(250)))
        async with app.run_test(size=(79, 24)) as pilot:
            await pilot.press("5")
            history = app.query_one(EventHistoryView)
            await pilot.pause()

            assert len(history._events) == 200
            assert history._events[0].message == "event-050"
            assert history._events[-1].message == "event-249"
            assert history.row_range[1:] == (200, 200)
            label = history.query_one(".range-label", Static)
            assert "–200 / 200" in str(label.content)
            assert "FOLLOW" in str(label.content)

            await pilot.press("home")
            await pilot.pause()
            start, end, total = history.row_range
            assert start == 1
            assert 1 <= end < total == 200
            assert "Rows 1–" in str(label.content)
            assert "PAUSED" in str(label.content)

    asyncio.run(exercise())


def test_event_view_fills_a_focused_dashboard_and_updates_after_resize():
    async def exercise() -> None:
        app = PipelineApp(initial_snapshot=_snapshot(_events(250)))
        async with app.run_test(size=(79, 24)) as pilot:
            await pilot.press("5")
            await pilot.resize_terminal(79, 40)
            await pilot.pause()
            history = app.query_one(EventHistoryView)
            dashboard = app.query_one("#dashboard")

            assert history.region == dashboard.region
            assert history.row_range[1:] == (200, 200)

    asyncio.run(exercise())


def test_capacity_one_queue_replaces_each_stale_snapshot():
    queue: Queue[object] = Queue(maxsize=1)

    for sequence in range(500):
        publish_latest_snapshot(queue, _snapshot(sequence=sequence))

    assert queue.qsize() == 1
    latest = drain_latest_snapshot(queue)
    assert latest is not None
    assert latest.run_context["sequence"] == 499
    assert drain_latest_snapshot(queue) is None


def test_snapshot_replacement_never_waits_for_a_contended_queue():
    class ContendedQueue:
        def put_nowait(self, _item):
            raise Full

        def get_nowait(self):
            raise Empty

    publish_latest_snapshot(
        ContendedQueue(),  # type: ignore[arg-type]
        _snapshot(sequence=1),
    )


def test_resource_view_shows_raw_cpu_units_and_unavailable_samples():
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "traces"))
    metrics.apply(
        MetricEvent.create(
            "resource_sample",
            "traces",
            sample=ResourceSample(
                timestamp=1.0,
                system_cpu_percent=100.0,
                system_memory_percent=50.0,
                tree_cpu_percent=385.0,
                tree_memory_percent=4.0,
                tree_rss_bytes=1_024,
                process_count=2,
                processes=(
                    ProcessResource(
                        pid=20,
                        create_time=2.0,
                        name="worker",
                        cpu_percent=None,
                        rss_bytes=None,
                        memory_percent=None,
                    ),
                ),
                logical_cpu_count=10,
                tree_cpu_cores=3.85,
                tree_cpu_capacity_percent=38.5,
            ),
        )
    )
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=100)

    console.print(_resource_view(metrics.snapshot()))

    rendered = stream.getvalue()
    assert "Workload CPU" in rendered and "385%" in rendered
    assert "Equivalent cores" in rendered and "3.85 / 10" in rendered
    assert "Machine capacity" in rendered and "38.5%" in rendered
    assert "System CPU" in rendered and "100%" in rendered


def test_resource_view_names_each_unavailable_cpu_sample():
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "traces"))
    metrics.apply(
        MetricEvent.create(
            "resource_sample",
            "traces",
            sample=ResourceSample(
                timestamp=1.0,
                system_cpu_percent=None,
                system_memory_percent=None,
                tree_cpu_percent=None,
                tree_memory_percent=None,
                tree_rss_bytes=None,
                process_count=1,
                processes=(),
                logical_cpu_count=None,
                tree_cpu_cores=None,
                tree_cpu_capacity_percent=None,
            ),
        )
    )
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=100)

    console.print(_resource_view(metrics.snapshot()))

    assert stream.getvalue().count("unavailable") >= 6


def test_pipeline_snapshots_are_picklable_and_deeply_immutable():
    nested = {
        "seed_groups": [1, 2],
        "selection": {"profiles": ["principal", "oracle_fallback"]},
    }
    metrics = MetricsAggregator()
    metrics.apply(
        MetricEvent.create(
            "run_config",
            "principal-traces",
            nested=nested,
        )
    )
    snapshot = metrics.snapshot()
    restored = pickle.loads(pickle.dumps(snapshot))

    assert restored == snapshot
    assert restored.run_context["nested"]["seed_groups"] == (1, 2)
    assert restored.run_context["nested"]["selection"]["profiles"] == (
        "principal",
        "oracle_fallback",
    )

    nested["seed_groups"].append(3)
    nested["selection"]["profiles"].append("oracle_true_state")
    assert snapshot.run_context["nested"]["seed_groups"] == (1, 2)

    with pytest.raises(TypeError):
        snapshot.run_context["nested"]["selection"]["new"] = "value"
    with pytest.raises(FrozenInstanceError):
        snapshot.progress_fraction = 1.0

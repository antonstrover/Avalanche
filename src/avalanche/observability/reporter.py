"""Render observability snapshots with Rich."""

from __future__ import annotations

from datetime import timedelta
from threading import RLock
from typing import Any

import humanize
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from avalanche.observability.metrics import (
    GateSnapshot,
    GRUState,
    MetricsAggregator,
    PipelineSnapshot,
    StageSnapshot,
    StageStatus,
)


class RichReporter:
    """Render a live report when the output is interactive."""

    def __init__(
        self,
        aggregator: MetricsAggregator,
        *,
        enabled: bool | None = None,
        console: Console | None = None,
    ) -> None:
        self.aggregator = aggregator
        self.console = console or Console(stderr=True)
        self.enabled = self.console.is_terminal if enabled is None else enabled
        self._live: Live | None = None
        self._lock = RLock()

    @property
    def active(self) -> bool:
        """Return true when a live display is active."""
        return self._live is not None

    def start(self) -> None:
        """Start the live display when it is enabled."""
        with self._lock:
            if not self.enabled or self._live is not None:
                return
            self._live = Live(
                self.render(),
                console=self.console,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start(refresh=True)

    def refresh(self) -> None:
        """Refresh the live display when it is active."""
        with self._lock:
            if self._live is not None:
                self._live.update(self.render(), refresh=True)

    def stop(self) -> None:
        """Stop the live display and keep its final state."""
        with self._lock:
            if self._live is None:
                return
            self._live.update(self.render(), refresh=True)
            self._live.stop()
            self._live = None

    def render(self) -> RenderableType:
        """Build one report from the current structured state."""
        state = self.aggregator.snapshot()
        sections: list[RenderableType] = []
        if state.run_context:
            sections.append(_configuration_panel(state.run_context))
        sections.append(_pipeline_table(state))
        current = _current_stage(state)
        if current is not None:
            sections.extend(_stage_sections(current))
        calibration_stage = _latest_calibration_stage(state)
        if calibration_stage is not None and calibration_stage is not current:
            sections.append(_calibration_panel(calibration_stage))
        validation_stage = _latest_validation_stage(state)
        if validation_stage is not None:
            sections.append(_validation_panel(validation_stage))
        if state.gate is not None or state.gru_state != GRUState.NOT_EVALUATED:
            sections.append(
                _gate_panel(
                    state.gate,
                    state.gru_state,
                    title="Perceptron gate",
                )
            )
        sections.append(_semantic_panel(state))
        if state.recent_events:
            sections.append(_recent_table(state))
        return Group(*sections)

    def __enter__(self) -> RichReporter:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


def _configuration_panel(values: dict[str, Any]) -> Panel:
    table = Table.grid(padding=(0, 2))
    for name, value in list(values.items())[:10]:
        table.add_row(_label(name), _short(value))
    return Panel(table, title="Run configuration", border_style="blue")


def _pipeline_table(state: PipelineSnapshot) -> Table:
    table = Table(title="Pipeline", expand=True)
    table.add_column("Stage")
    table.add_column("Progress", justify="right")
    table.add_column("Work", justify="right")
    table.add_column("Status", justify="right")
    for stage in state.stages:
        table.add_row(
            stage.label,
            f"{stage.percentage:5.1f}%",
            _stage_work(stage),
            Text(
                stage.status.value.replace("_", " ").upper(),
                style=_status_style(stage),
            ),
        )
    table.caption = (
        f"Pipeline {state.progress_fraction * 100.0:.1f}%"
        f" · {state.completed_stages}/{state.total_stages} stages"
    )
    if state.overall_eta_seconds is not None:
        table.caption += f" · ETA {_duration(state.overall_eta_seconds)}"
    return table


def _stage_sections(stage: StageSnapshot) -> list[RenderableType]:
    sections: list[RenderableType] = [_stage_panel(stage)]
    if stage.workers:
        sections.append(_worker_table(stage))
    if (
        stage.training.total_epochs is not None
        or stage.training.training_loss is not None
    ):
        sections.append(_training_panel(stage))
    if stage.calibration.status != StageStatus.PENDING:
        sections.append(_calibration_panel(stage))
    if stage.gate is not None and (stage.gate.values.get("model_name") == "gru"):
        sections.append(_gate_panel(stage.gate, title="GRU gate"))
    if stage.parquet is not None:
        sections.append(_parquet_panel(stage))
    if stage.resources is not None:
        sections.append(_resource_panel(stage))
    return sections


def _stage_panel(stage: StageSnapshot) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_row("Phase", stage.phase)
    table.add_row("Progress", f"{stage.percentage:.1f}%")
    if stage.total_episodes is not None:
        table.add_row(
            "Episodes",
            f"{humanize.intcomma(stage.episodes_completed)} / "
            f"{humanize.intcomma(stage.total_episodes)}",
        )
    if stage.expected_rows is not None:
        observed_rows = stage.rows_generated + stage.rows_in_progress
        table.add_row(
            "Rows",
            f"{humanize.intcomma(observed_rows)} / "
            f"{humanize.intcomma(stage.expected_rows)}",
        )
        if stage.rows_in_progress:
            table.add_row(
                "Committed · active rows",
                f"{humanize.intcomma(stage.rows_generated)} · "
                f"{humanize.intcomma(stage.rows_in_progress)}",
            )
    table.add_row("Elapsed", _duration(stage.elapsed_seconds))
    table.add_row("ETA", _duration(stage.eta_seconds))
    if stage.episodes_per_second is not None:
        table.add_row("Episodes/s", f"{stage.episodes_per_second:.2f}")
    if stage.rows_per_second is not None:
        table.add_row("Rows/s", f"{stage.rows_per_second:,.1f}")
    table.add_row(
        "Workers",
        f"{stage.active_workers} active"
        + (
            f" / {stage.configured_workers} configured"
            if stage.configured_workers is not None
            else ""
        ),
    )
    table.add_row(
        "Retries · rejects · failures",
        f"{stage.retries} · {stage.rejected} · {stage.failures}",
    )
    latency = stage.latency
    if latency.count:
        table.add_row(
            "Latency mean · median · p95",
            " · ".join(
                _duration(value)
                for value in (
                    latency.mean_seconds,
                    latency.median_seconds,
                    latency.p95_seconds,
                )
            ),
        )
    if stage.error:
        table.add_row("Error", Text(stage.error, style="red"))
    return Panel(table, title=f"Current stage · {stage.label}", border_style="cyan")


def _worker_table(stage: StageSnapshot) -> Table:
    table = Table(title="Workers", expand=True)
    table.add_column("Worker")
    table.add_column("State")
    table.add_column("Phase")
    table.add_column("Item")
    table.add_column("Episodes", justify="right")
    table.add_column("Current rows", justify="right")
    table.add_column("Total rows", justify="right")
    for worker in stage.workers:
        table.add_row(
            worker.worker_id,
            "ACTIVE" if worker.active else "IDLE",
            worker.phase,
            worker.current_item or "—",
            humanize.intcomma(worker.episodes_completed),
            humanize.intcomma(worker.current_rows),
            humanize.intcomma(worker.rows_generated),
        )
    return table


def _training_panel(stage: StageSnapshot) -> Panel:
    training = stage.training
    table = Table.grid(padding=(0, 2))
    table.add_row("Model", stage.current_model or "—")
    table.add_row(
        "Epoch",
        f"{training.epoch} / {training.total_epochs or '—'}",
    )
    if training.total_batches is not None:
        table.add_row("Batch", f"{training.batch} / {training.total_batches}")
    if training.total_samples is not None:
        table.add_row(
            "Samples",
            f"{humanize.intcomma(training.samples_processed)} / "
            f"{humanize.intcomma(training.total_samples)}",
        )
    if training.samples_per_second is not None:
        table.add_row("Samples/s", f"{training.samples_per_second:,.1f}")
    table.add_row("Epoch elapsed", _duration(training.epoch_elapsed_seconds))
    table.add_row("Mean epoch", _duration(training.mean_epoch_seconds))
    table.add_row("ETA", _duration(training.eta_seconds))
    if training.training_loss is not None:
        table.add_row("Training loss", f"{training.training_loss:.6g}")
    if training.validation_loss is not None:
        table.add_row("Validation loss", f"{training.validation_loss:.6g}")
    if training.metric_name and training.metric_value is not None:
        table.add_row(_label(training.metric_name), f"{training.metric_value:.6g}")
    if training.best_metric is not None:
        table.add_row("Best metric", f"{training.best_metric:.6g}")
    if stage.total_models is not None:
        table.add_row(
            "Models",
            f"{stage.completed_models} / {stage.total_models}",
        )
    return Panel(table, title="Training", border_style="magenta")


def _calibration_panel(stage: StageSnapshot) -> Panel:
    calibration = stage.calibration
    table = Table.grid(padding=(0, 2))
    table.add_row("Status", calibration.status.value.upper())
    if calibration.total_rows is not None:
        table.add_row(
            "Row evaluations",
            f"{humanize.intcomma(calibration.rows_processed)} / "
            f"{humanize.intcomma(calibration.total_rows)}",
        )
    table.add_row("Elapsed", _duration(calibration.elapsed_seconds))
    table.add_row("ETA", _duration(calibration.eta_seconds))
    if calibration.threshold is not None:
        table.add_row("Selected threshold", f"{calibration.threshold:.6g}")
    for name, value in calibration.metrics.items():
        table.add_row(_label(name), f"{value:.6g}")
    return Panel(table, title="Calibration", border_style="yellow")


def _gate_panel(
    gate: GateSnapshot | None,
    gru_state: GRUState | None = None,
    *,
    title: str,
) -> Panel:
    table = Table.grid(padding=(0, 2))
    if gate is not None:
        table.add_row("Criterion", gate.criterion)
        table.add_row("Metric", gate.metric_name)
        table.add_row("Observed", _number(gate.observed))
        table.add_row("Required", _number(gate.required))
        table.add_row("Result", "PASSED" if gate.passed else "FAILED")
    if gru_state is not None:
        table.add_row("GRU fallback", gru_state.value.replace("_", " ").upper())
    return Panel(table, title=title, border_style="green")


def _parquet_panel(stage: StageSnapshot) -> Panel:
    output = stage.parquet
    assert output is not None
    table = Table.grid(padding=(0, 2))
    table.add_row("Estimate state", output.state.upper())
    table.add_row("Written rows", humanize.intcomma(output.written_rows))
    table.add_row("Buffered rows", humanize.intcomma(output.buffered_rows))
    table.add_row("Parquet written", _bytes(output.written_bytes))
    prefix = "" if output.final else "~"
    table.add_row(
        "Final size" if output.final else "Estimated final",
        _bytes(output.estimated_final_bytes, prefix=prefix),
    )
    if output.estimated_buffered_bytes is not None and output.buffered_rows:
        table.add_row(
            "Estimated buffer",
            _bytes(output.estimated_buffered_bytes, prefix="~"),
        )
    if output.average_bytes_per_row is not None:
        table.add_row("Average/row", _bytes(output.average_bytes_per_row))
    return Panel(table, title="Output size", border_style="blue")


def _resource_panel(stage: StageSnapshot) -> Panel:
    sample = stage.resources
    assert sample is not None
    table = Table.grid(padding=(0, 2))
    table.add_row("Process-tree CPU", f"{sample.tree_cpu_percent:.1f}%")
    table.add_row("System CPU", f"{sample.system_cpu_percent:.1f}%")
    table.add_row("Process memory", _bytes(sample.tree_rss_bytes))
    table.add_row("System RAM", f"{sample.system_memory_percent:.1f}%")
    table.add_row("Processes", str(sample.process_count))
    if sample.gpu_percent is not None:
        table.add_row("GPU", f"{sample.gpu_percent:.1f}%")
    if sample.gpu_memory_bytes is not None:
        table.add_row("GPU memory", _bytes(sample.gpu_memory_bytes))
    return Panel(table, title="Resources", border_style="white")


def _semantic_panel(state: PipelineSnapshot) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_row(
        "Principal traces", humanize.intcomma(state.principal_traces_generated)
    )
    table.add_row(
        "Oracle true states", humanize.intcomma(state.oracle_true_states_generated)
    )
    table.add_row(
        "Oracle fallbacks", humanize.intcomma(state.oracle_fallbacks_generated)
    )
    table.add_row(
        "Fallback attempts", humanize.intcomma(state.fallback_generation_attempts)
    )
    table.add_row(
        "Fallback rate",
        f"{state.fallback_rate:.2%}" if state.fallback_rate is not None else "—",
    )
    if state.gate is not None or any("gru" in stage.stage_id for stage in state.stages):
        table.add_row(
            "GRU fallback",
            state.gru_state.value.replace("_", " ").upper(),
        )
    return Panel(table, title="Semantic totals", border_style="green")


def _recent_table(state: PipelineSnapshot) -> Table:
    table = Table(title="Recent significant events", expand=True)
    table.add_column("Stage")
    table.add_column("Event")
    table.add_column("Detail")
    for event in state.recent_events[-8:]:
        table.add_row(event.stage_id, event.kind.replace("_", " "), event.message)
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
    """Return the latest stage with completed validation metrics."""
    for stage in reversed(state.stages):
        if "validation_brier_score" in stage.metrics:
            return stage
    return None


def _latest_calibration_stage(state: PipelineSnapshot) -> StageSnapshot | None:
    """Return the latest active or completed calibration stage."""
    for stage in reversed(state.stages):
        if stage.calibration.status != StageStatus.PENDING:
            return stage
    return None


def _validation_panel(stage: StageSnapshot) -> Panel:
    """Render the validation metrics that the model already calculates."""
    table = Table.grid(padding=(0, 2))
    table.add_row("Model", stage.current_model or "—")
    for name in (
        "validation_brier_score",
        "validation_average_precision",
        "validation_accuracy",
        "baseline_brier_score",
        "baseline_average_precision",
    ):
        value = stage.metrics.get(name)
        if isinstance(value, int | float):
            table.add_row(_label(name), f"{value:.6g}")
    return Panel(table, title="Validation metrics", border_style="magenta")


def _stage_work(stage: StageSnapshot) -> str:
    if stage.total_episodes is not None:
        completed = humanize.intcomma(stage.episodes_completed)
        total = humanize.intcomma(stage.total_episodes)
        return f"{completed} / {total} episodes"
    if stage.training.total_epochs is not None:
        return f"{stage.training.epoch} / {stage.training.total_epochs} epochs"
    if stage.total_models is not None:
        return f"{stage.completed_models} / {stage.total_models} models"
    return stage.phase


def _status_style(stage: StageSnapshot) -> str:
    return {
        StageStatus.COMPLETE: "green",
        StageStatus.FAILED: "bold red",
        StageStatus.RUNNING: "cyan",
        StageStatus.TRIGGERED: "yellow",
        StageStatus.NOT_REQUIRED: "dim green",
    }.get(stage.status, "dim")


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1_000.0:.0f} ms"
    return humanize.naturaldelta(timedelta(seconds=round(seconds)))


def _bytes(value: float | int | None, *, prefix: str = "") -> str:
    if value is None:
        return "—"
    return prefix + humanize.naturalsize(value, binary=True)


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.6g}"


def _label(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip().title()


def _short(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."

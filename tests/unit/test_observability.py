"""Test structured observability without a full terminal screen."""

from __future__ import annotations

import json
import multiprocessing as mp
import pickle
import random
from concurrent.futures import ThreadPoolExecutor
from io import StringIO

import pytest
from rich.console import Console

from avalanche.observability import (
    BoundedLatencyStatistics,
    DirectMetricEmitter,
    GRUState,
    MetricEmitter,
    MetricEvent,
    MetricsAggregator,
    ObservabilitySession,
    ParquetSizeEstimator,
    ProcessTreeSampler,
    StageStatus,
    TextualReporter,
)
from avalanche.observability.reporter import _resource_view, _semantic_view


def event(
    kind: str,
    stage_id: str,
    timestamp: float,
    worker_id: str | None = None,
    **values: object,
) -> MetricEvent:
    """Return one event with a fixed test timestamp."""
    return MetricEvent(kind, stage_id, worker_id, dict(values), timestamp)


def test_metric_events_are_picklable_and_direct_emitters_follow_the_protocol():
    received = []
    emitter = DirectMetricEmitter(received.append)
    original = MetricEvent.create("stage_phase", "trace", phase="writing")

    emitter.emit(pickle.loads(pickle.dumps(original)))

    assert isinstance(emitter, MetricEmitter)
    assert received[0].kind == "stage_phase"
    assert received[0].stage_id == "trace"
    assert received[0].values == {"phase": "writing"}


def test_trace_events_aggregate_progress_workers_and_failures():
    metrics = MetricsAggregator()
    metrics.apply(
        event(
            "stage_started",
            "principal-traces",
            10.0,
            label="Principal traces",
            total_episodes=4,
            expected_rows=40,
            workers=2,
        )
    )
    metrics.apply(
        event(
            "episode_started",
            "principal-traces",
            11.0,
            "worker-1",
            episode_id="episode-1",
        )
    )
    metrics.apply(
        event(
            "worker_progress",
            "principal-traces",
            11.5,
            "worker-1",
            current_rows=7,
            active=True,
        )
    )
    active = metrics.snapshot(now=11.5).stage("principal-traces")
    assert active.rows_in_progress == 7
    metrics.apply(
        event(
            "episode_completed",
            "principal-traces",
            12.0,
            "worker-1",
            rows=10,
            latency_seconds=2.0,
        )
    )
    metrics.apply(event("retry", "principal-traces", 12.1, "worker-1", count=2))
    metrics.apply(event("rejected", "principal-traces", 12.2, count=3))
    metrics.apply(event("failure", "principal-traces", 12.3, count=1))

    state = metrics.snapshot(now=20.0)
    stage = state.stage("principal-traces")

    assert stage.status == StageStatus.RUNNING
    assert stage.episodes_completed == 1
    assert stage.rows_generated == 10
    assert stage.progress_fraction == pytest.approx(0.25)
    assert stage.configured_workers == 2
    assert stage.active_workers == 0
    assert stage.workers[0].episodes_completed == 1
    assert stage.workers[0].rows_generated == 10
    assert stage.workers[0].current_rows == 0
    assert (stage.retries, stage.rejected, stage.failures) == (2, 3, 1)
    assert stage.latency.mean_seconds == pytest.approx(2.0)
    assert stage.latency.median_seconds == pytest.approx(2.0)
    assert stage.latency.p95_seconds == pytest.approx(2.0)


def test_semantic_counts_use_fallback_attempts_as_the_denominator():
    metrics = MetricsAggregator()
    for name, count in (
        ("principal_traces", 80),
        ("oracle_true_states", 75),
        ("fallback_attempts", 100),
        ("oracle_fallbacks", 20),
    ):
        metrics.apply(
            MetricEvent.create("semantic_count", "generation", name=name, count=count)
        )

    state = metrics.snapshot()

    assert state.principal_traces_generated == 80
    assert state.oracle_true_states_generated == 75
    assert state.oracle_fallbacks_generated == 20
    assert state.fallback_generation_attempts == 100
    assert state.fallback_rate == pytest.approx(0.2)


def test_latency_quantiles_use_a_bounded_recent_sample():
    statistics = BoundedLatencyStatistics(capacity=20)
    for value in range(1, 101):
        statistics.add(float(value))

    snapshot = statistics.snapshot()

    assert snapshot.count == 100
    assert snapshot.sampled_count == 20
    assert snapshot.mean_seconds == pytest.approx(50.5)
    assert snapshot.median_seconds == pytest.approx(90.5)
    assert snapshot.p95_seconds == pytest.approx(99.0)


def test_training_model_and_calibration_progress_are_separate():
    metrics = MetricsAggregator()
    metrics.apply(
        event(
            "stage_started",
            "oracle-training",
            0.0,
            total_epochs=4,
            total_samples=400,
            total_models=3,
            model_name="congestion_oracle",
        )
    )
    metrics.apply(
        event(
            "epoch_progress",
            "oracle-training",
            4.0,
            phase="epoch",
            epoch=2,
            total_epochs=4,
            batch=5,
            total_batches=10,
            samples=200,
            total_samples=400,
            training_loss=0.4,
            validation_loss=0.5,
            metric_name="brier_score",
            metric_value=0.2,
            best_metric=0.18,
            epoch_seconds=2.0,
        )
    )
    metrics.apply(
        event(
            "model_progress",
            "oracle-training",
            4.1,
            model_name="fallback_oracle",
            completed_models=1,
            total_models=3,
        )
    )
    metrics.apply(
        event(
            "calibration_started",
            "oracle-calibration",
            5.0,
            rows=0,
            total_rows=100,
        )
    )
    metrics.apply(
        event(
            "calibration_completed",
            "oracle-calibration",
            7.0,
            rows=100,
            total_rows=100,
            threshold=0.65,
            brier_score=0.12,
        )
    )

    state = metrics.snapshot(now=8.0)
    training = state.stage("oracle-training")
    calibration = state.stage("oracle-calibration").calibration

    assert training.training.epoch == 2
    assert training.training.mean_epoch_seconds == pytest.approx(2.0)
    assert training.training.training_loss == pytest.approx(0.4)
    assert training.current_model == "fallback_oracle"
    assert training.completed_models == 1
    assert training.total_models == 3
    assert calibration.status == StageStatus.COMPLETE
    assert calibration.rows_processed == 100
    assert calibration.threshold == pytest.approx(0.65)
    assert calibration.metrics["brier_score"] == pytest.approx(0.12)


def test_batch_elapsed_time_does_not_change_the_mean_epoch_time():
    metrics = MetricsAggregator()
    metrics.apply(event("stage_started", "training", 0.0, total_epochs=2))
    metrics.apply(
        event(
            "epoch_progress",
            "training",
            1.0,
            phase="batch",
            epoch=1,
            batch=1,
            total_batches=2,
            epoch_seconds=0.1,
        )
    )
    metrics.apply(
        event(
            "epoch_progress",
            "training",
            2.0,
            phase="epoch",
            epoch=1,
            batch=2,
            total_batches=2,
            epoch_seconds=2.0,
        )
    )

    training = metrics.snapshot(now=2.0).stage("training").training

    assert training.epoch_elapsed_seconds == pytest.approx(2.0)
    assert training.mean_epoch_seconds == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("passed", "expected"),
    ((True, GRUState.NOT_REQUIRED), (False, GRUState.TRIGGERED)),
)
def test_perceptron_gate_sets_the_initial_gru_decision(passed, expected):
    metrics = MetricsAggregator()
    metrics.apply(
        MetricEvent.create(
            "gate_evaluated",
            "principal-calibration",
            criterion="sleeper-recall-at-false-alarm-budget",
            metric_name="sleeper_recall",
            observed=0.9 if passed else 0.4,
            required=0.8,
            passed=passed,
        )
    )

    state = metrics.snapshot()

    assert state.gate is not None
    assert state.gate.passed is passed
    assert state.gru_state == expected


def test_explicit_gru_states_distinguish_skip_trigger_training_and_completion():
    metrics = MetricsAggregator()
    for state in ("not_evaluated", "triggered", "training", "complete"):
        metrics.apply(MetricEvent.create("gru_state", "principal-gru", state=state))

    complete = metrics.snapshot().stage("principal-gru")
    assert complete.status == StageStatus.COMPLETE
    assert complete.gru_state == GRUState.COMPLETE

    skipped = MetricsAggregator()
    skipped.apply(MetricEvent.create("gru_state", "oracle-gru", state="not_required"))
    assert skipped.snapshot().stage("oracle-gru").status == StageStatus.NOT_REQUIRED


def test_the_gru_gate_does_not_replace_the_perceptron_gate_decision():
    metrics = MetricsAggregator()
    metrics.apply(
        MetricEvent.create(
            "gate_evaluated",
            "principal-perceptron-calibration",
            model_name="perceptron",
            metric_name="sleeper_recall",
            observed=0.4,
            required=0.8,
            passed=False,
        )
    )
    metrics.apply(MetricEvent.create("gru_state", "principal-gru", state="complete"))
    metrics.apply(
        MetricEvent.create(
            "gate_evaluated",
            "principal-gru-calibration",
            model_name="gru",
            metric_name="sleeper_recall",
            observed=0.9,
            required=0.8,
            passed=True,
        )
    )

    state = metrics.snapshot()

    assert state.gate is not None
    assert state.gate.stage_id == "principal-perceptron-calibration"
    assert state.gru_state == GRUState.COMPLETE
    assert state.stage("principal-gru-calibration").gate is not None


def test_views_keep_gate_validation_and_exact_size_evidence_visible():
    metrics = MetricsAggregator()
    metrics.apply(
        MetricEvent.create(
            "gate_evaluated",
            "principal-perceptron-calibration",
            model_name="perceptron",
            metric_name="sleeper_recall",
            observed=0.4,
            required=0.8,
            passed=False,
        )
    )
    metrics.apply(MetricEvent.create("gru_state", "principal-gru", state="complete"))
    metrics.apply(
        MetricEvent.create(
            "stage_completed",
            "principal-perceptron",
            model_name="perceptron",
            validation_brier_score=0.12,
        )
    )
    metrics.apply(
        MetricEvent.create(
            "parquet_progress",
            "principal-output",
            expected_rows=10,
            written_rows=10,
            written_bytes=1_024,
            row_groups=1,
            final=True,
        )
    )
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, width=120)
    state = metrics.snapshot()
    console.print(_semantic_view(state))
    console.print(_resource_view(state))
    rendered = stream.getvalue()

    assert "Validation gate" in rendered
    assert "GRU fallback" in rendered
    assert "Validation statistics" in rendered
    assert "Final size" in rendered


def test_parquet_size_estimates_are_provisional_then_ready_and_final():
    estimator = ParquetSizeEstimator(
        expected_rows=1_000,
        minimum_written_rows=100,
        minimum_row_groups=2,
    )

    early = estimator.update(
        written_rows=50,
        written_bytes=5_000,
        buffered_rows=20,
        row_groups=1,
    )
    ready = estimator.update(
        written_rows=200,
        written_bytes=16_000,
        buffered_rows=25,
        row_groups=2,
    )
    final = estimator.update(
        written_rows=1_000,
        written_bytes=70_000,
        buffered_rows=0,
        row_groups=4,
        final=True,
    )

    assert early.state == "provisional"
    assert early.estimated_final_bytes == 100_000
    assert early.estimated_buffered_bytes == 2_000
    assert early.observed_rows == 70
    assert ready.ready
    assert ready.average_bytes_per_row == pytest.approx(80.0)
    assert ready.estimated_final_bytes == 80_000
    assert final.state == "final"
    assert final.estimated_final_bytes == 70_000


def test_parquet_estimator_waits_for_encoded_rows():
    estimator = ParquetSizeEstimator(expected_rows=500)

    snapshot = estimator.update(
        written_rows=0,
        written_bytes=0,
        buffered_rows=200,
        row_groups=0,
    )

    assert snapshot.state == "waiting"
    assert snapshot.average_bytes_per_row is None
    assert snapshot.estimated_final_bytes is None


def test_aggregator_tracks_buffered_parquet_rows():
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "traces", expected_rows=1_000))
    metrics.apply(
        MetricEvent.create(
            "parquet_progress",
            "traces",
            written_rows=100,
            written_bytes=4_000,
            buffered_rows=30,
            row_groups=1,
            minimum_written_rows=50,
            minimum_row_groups=1,
        )
    )

    output = metrics.snapshot().stage("traces").parquet

    assert output is not None
    assert output.written_rows == 100
    assert output.buffered_rows == 30
    assert output.estimated_final_bytes == 40_000
    assert output.ready


def test_stage_completion_reconciles_absolute_final_counts():
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "traces", total_episodes=2))
    metrics.apply(
        MetricEvent.create(
            "stage_completed",
            "traces",
            episodes=2,
            rows=20,
            output_bytes=800,
        )
    )

    stage = metrics.snapshot().stage("traces")

    assert stage.status == StageStatus.COMPLETE
    assert stage.episodes_completed == 2
    assert stage.rows_generated == 20
    assert stage.metrics["output_bytes"] == 800


def test_generic_progress_records_absolute_resolution_counts():
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "traces", total_episodes=100))
    metrics.apply(
        MetricEvent.create(
            "stage_progress",
            "traces",
            phase="resolving configurations",
            completed_episodes=32,
        )
    )

    stage = metrics.snapshot().stage("traces")

    assert stage.phase == "resolving configurations"
    assert stage.episodes_completed == 32
    assert stage.progress_fraction == 0.32


def test_stage_failure_sets_the_failure_status_and_counter():
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "training"))
    metrics.apply(
        MetricEvent.create("stage_failed", "training", error="training stopped")
    )

    state = metrics.snapshot()

    assert state.stage("training").status == StageStatus.FAILED
    assert state.stage("training").failures == 1
    assert state.failures == 1


def test_aggregator_is_thread_safe_for_completion_events():
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "traces", total_episodes=1_000))

    def emit_many(count: int) -> None:
        for _ in range(count):
            metrics.apply(MetricEvent.create("episode_completed", "traces", rows=2))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(emit_many, (250, 250, 250, 250)))

    stage = metrics.snapshot().stage("traces")
    assert stage.episodes_completed == 1_000
    assert stage.rows_generated == 2_000


def test_disabled_and_noninteractive_reporters_skip_the_observer():
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "traces", total_episodes=2))
    disabled_stream = StringIO()
    disabled = TextualReporter(
        metrics,
        enabled=False,
        console=Console(file=disabled_stream, force_terminal=True),
    )
    noninteractive_stream = StringIO()
    noninteractive = TextualReporter(
        metrics,
        console=Console(file=noninteractive_stream, force_terminal=False),
    )

    with disabled:
        disabled.refresh()
    with noninteractive:
        noninteractive.refresh()

    assert not disabled.enabled
    assert not noninteractive.enabled
    assert not disabled.active
    assert not noninteractive.active
    assert disabled_stream.getvalue().count("COMPLETED") == 1
    assert noninteractive_stream.getvalue().count("COMPLETED") == 1
    assert disabled_stream.getvalue() == noninteractive_stream.getvalue()
    assert "\x1b[" not in disabled_stream.getvalue()
    assert "\x1b[" not in noninteractive_stream.getvalue()


def test_session_logs_significant_events_without_terminal_output(tmp_path, capsys):
    log_path = tmp_path / "observability.jsonl"
    with ObservabilitySession(
        enabled=False,
        log_path=log_path,
        sample_resources=False,
    ) as session:
        session.emitter.emit(MetricEvent.create("stage_started", "traces"))
        session.emitter.emit(MetricEvent.create("retry", "traces", count=1))
        session.emitter.emit(MetricEvent.create("stage_completed", "traces"))

    records = [json.loads(line) for line in log_path.read_text().splitlines()]

    captured = capsys.readouterr()
    assert "COMPLETED" in captured.out
    assert captured.err == ""
    assert any(
        record["record"]["extra"].get("event_kind") == "retry" for record in records
    )


def test_manager_queue_events_are_drained_before_session_shutdown():
    try:
        session = ObservabilitySession(
            enabled=False,
            multiprocessing=True,
            sample_resources=False,
        )
    except EOFError, OSError:
        pytest.skip("the test sandbox blocks Manager sockets")
    emitter = session.process_emitter
    emitter.emit(MetricEvent.create("stage_started", "traces", total_episodes=1))
    emitter.emit(MetricEvent.create("episode_completed", "traces", rows=12))
    emitter.emit(MetricEvent.create("stage_completed", "traces"))

    session.close()

    stage = session.aggregator.snapshot().stage("traces")
    assert stage.status == StageStatus.COMPLETE
    assert stage.episodes_completed == 1
    assert stage.rows_generated == 12
    emitter.emit(MetricEvent.create("message", "traces", message="closed"))


def test_a_spawned_process_can_emit_metrics_through_the_session_queue():
    try:
        session = ObservabilitySession(
            enabled=False,
            multiprocessing=True,
            sample_resources=False,
        )
    except EOFError, OSError:
        pytest.skip("the test sandbox blocks Manager sockets")
    session.emitter.emit(
        MetricEvent.create("stage_started", "child-traces", total_episodes=1)
    )
    emitter = session.process_emitter
    process = mp.get_context("spawn").Process(
        target=emitter.emit,
        args=(
            MetricEvent.create(
                "episode_completed",
                "child-traces",
                "child",
                rows=12,
            ),
        ),
    )
    process.start()
    process.join(timeout=15.0)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("the metric child process did not stop")
    session.close()

    assert process.exitcode == 0
    stage = session.aggregator.snapshot().stage("child-traces")
    assert stage.episodes_completed == 1
    assert stage.rows_generated == 12


def test_resource_sampler_reports_the_current_process_tree():
    sample = ProcessTreeSampler(minimum_interval=0.0).sample(force=True)

    assert sample.process_count >= 1
    assert sample.tree_rss_bytes is None or sample.tree_rss_bytes >= 0
    assert sample.system_memory_percent is None or sample.system_memory_percent >= 0.0


def test_observability_does_not_consume_the_random_stream():
    random.seed(20260828)
    expected = [random.random() for _ in range(3)]
    random.seed(20260828)
    metrics = MetricsAggregator()
    metrics.apply(MetricEvent.create("stage_started", "training", total_epochs=60))
    metrics.apply(
        MetricEvent.create(
            "epoch_progress",
            "training",
            epoch=1,
            total_epochs=60,
            training_loss=0.4,
        )
    )

    actual = [random.random() for _ in range(3)]

    assert actual == expected

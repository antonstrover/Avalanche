"""Check complete event and artifact evidence."""

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from avalanche.metrics import METRICS_VERSION
from avalanche.sim import MountainSim
from avalanche.traces import (
    EVENT_SCHEMA_VERSION,
    PHYSICAL_REPLAY_SCHEMA_VERSION,
    RUN_MANIFEST_FILENAME,
    RUN_MANIFEST_SCHEMA_VERSION,
    RUN_MANIFEST_SIDECAR_FILENAME,
    SUMMARY_SCHEMA_VERSION,
    EventPhase,
    EventState,
    RunArtifactError,
    TraceWriter,
    load_verified_performance,
    load_verified_run,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def _summary(sim: MountainSim) -> dict[str, object]:
    """Return one valid deterministic summary."""
    return {
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "metrics": sim.metrics.snapshot(sim.population).as_dict(),
    }


def _write_run(
    output: Path,
    *,
    trace_level: str = "decision",
    performance: dict[str, float] | None = None,
    performance_root: Path | None = None,
) -> tuple[MountainSim, TraceWriter]:
    """Write one small complete run."""
    sim = MountainSim(FIXTURE)
    sim.reset(seed=4)
    writer = TraceWriter(
        output,
        "run-one",
        "episode-0",
        4,
        trace_level=trace_level,
        performance_root=performance_root,
    )
    writer.record(
        "scenario_changed",
        "test",
        {"name": "fixed"},
        sim,
        phase=EventPhase.OPERATIONAL_EVENT_TRANSITION,
    )
    writer.record_metrics(sim.metrics.snapshot(sim.population), sim)
    writer.record_snapshot(sim)
    writer.record(
        "episode_ended",
        "simulator",
        {"reason": "horizon"},
        sim,
        phase=EventPhase.TERMINAL,
    )
    writer.close(_summary(sim), performance=performance)
    return sim, writer


def test_an_event_carries_the_complete_envelope(tmp_path):
    sim, writer = _write_run(tmp_path)

    event = next(
        item
        for item in load_verified_run(tmp_path).read_events()
        if item["event_type"] == "scenario_changed"
    )
    assert event == {
        "actor_id": "test",
        "control_interval_index": 0,
        "entity_id": "",
        "entity_index": -1,
        "entity_kind": "",
        "episode_id": "episode-0",
        "event_sequence": 0,
        "event_type": "scenario_changed",
        "movement_tick": 0,
        "payload": {"name": "fixed"},
        "phase_code": int(EventPhase.OPERATIONAL_EVENT_TRANSITION),
        "physical_state_checksum": sim.physical_state_checksum(),
        "run_id": "run-one",
        "schema_version": EVENT_SCHEMA_VERSION,
        "seed": 4,
        "simulation_time": 0.0,
    }
    assert json.loads((tmp_path / "model-reference.json").read_text()) == {
        "model_kind": None,
        "model_path": None,
        "model_revision": None,
    }
    snapshot = writer.snapshot_rows[0]
    assert snapshot["schema_version"] == PHYSICAL_REPLAY_SCHEMA_VERSION
    assert snapshot["physical_state_checksum"] == sim.physical_state_checksum()


def test_an_event_can_use_one_captured_boundary_state(tmp_path):
    sim = MountainSim(FIXTURE)
    sim.reset(seed=4)
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)
    boundary = EventState.capture(sim)
    sim.tick()

    writer.record("action_executed", "test", {}, sim, state=boundary)

    event = writer.events[0]
    assert event.simulation_time == boundary.simulation_time
    assert event.movement_tick == boundary.movement_tick
    assert event.physical_state_checksum == boundary.physical_state_checksum


def test_equal_timestamp_order_is_stable(tmp_path):
    sim = MountainSim(FIXTURE)
    sim.reset(seed=4)
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)
    for phase, event_type, entity in (
        (EventPhase.EDGE_TRANSITION, "zeta", ("edge", 2, "z")),
        (EventPhase.EDGE_TRANSITION, "alpha", ("edge", 2, "é")),
        (EventPhase.EDGE_TRANSITION, "alpha", ("edge", 1, "z")),
        (EventPhase.CONTROL_PROPOSAL, "zeta", ("", -1, "")),
    ):
        writer.record(event_type, "test", {}, sim, phase=phase, entity=entity)
    writer.record_metrics(sim.metrics.snapshot(sim.population), sim)
    writer.close(_summary(sim))

    events = load_verified_run(tmp_path).read_events()
    selected = [
        (event["phase_code"], event["event_type"], event["entity_index"])
        for event in events
        if event["actor_id"] == "test"
    ]
    assert selected == [
        (0, "zeta", -1),
        (9, "alpha", 1),
        (9, "alpha", 2),
        (9, "zeta", 2),
    ]
    assert [event["event_sequence"] for event in events] == list(range(len(events)))


@pytest.mark.parametrize(
    ("trace_level", "present", "absent"),
    [
        ("summary", {"metrics.parquet", "summary.json"}, {"events.jsonl"}),
        (
            "decision",
            {"events.jsonl", "physical-replay-evaluator.parquet"},
            set(),
        ),
        (
            "debug",
            {"events.jsonl", "physical-replay-reported.parquet"},
            set(),
        ),
    ],
)
def test_trace_level_selects_the_content_files(
    tmp_path,
    trace_level,
    present,
    absent,
):
    output = tmp_path / trace_level
    _write_run(output, trace_level=trace_level)
    names = {path.name for path in output.iterdir()}

    assert present <= names
    assert not absent & names
    assert {RUN_MANIFEST_FILENAME, RUN_MANIFEST_SIDECAR_FILENAME} <= names
    load_verified_run(output)


def test_manifest_sidecar_has_exact_bytes(tmp_path):
    _write_run(tmp_path)
    content = (tmp_path / RUN_MANIFEST_FILENAME).read_bytes()
    expected = f"{hashlib.sha256(content).hexdigest()}  {RUN_MANIFEST_FILENAME}\n"

    assert (tmp_path / RUN_MANIFEST_SIDECAR_FILENAME).read_text() == expected
    manifest = json.loads(content)
    assert manifest["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert manifest["artifacts"] == sorted(
        manifest["artifacts"],
        key=lambda item: item["path"].encode("utf-8"),
    )
    assert all(
        item["artifact_sha256"]
        == hashlib.sha256((tmp_path / item["path"]).read_bytes()).hexdigest()
        for item in manifest["artifacts"]
    )


def test_each_changed_artifact_fails_loading(tmp_path):
    source = tmp_path / "source"
    _write_run(source)
    artifacts = tuple(load_verified_run(source).artifacts)

    for index, name in enumerate(artifacts):
        changed = tmp_path / f"changed-{index}"
        shutil.copytree(source, changed)
        with (changed / name).open("ab") as handle:
            handle.write(b"changed")
        with pytest.raises(RunArtifactError, match="size does not match"):
            load_verified_run(changed)


def test_missing_extra_and_duplicate_artifacts_fail_loading(tmp_path):
    source = tmp_path / "source"
    _write_run(source)
    first = next(iter(load_verified_run(source).artifacts))

    missing = tmp_path / "missing"
    shutil.copytree(source, missing)
    (missing / first).unlink()
    with pytest.raises(RunArtifactError, match="missing"):
        load_verified_run(missing)

    extra = tmp_path / "extra"
    shutil.copytree(source, extra)
    (extra / "note.txt").write_text("undeclared\n")
    with pytest.raises(RunArtifactError, match="extra"):
        load_verified_run(extra)

    duplicate = tmp_path / "duplicate"
    shutil.copytree(source, duplicate)
    manifest_path = duplicate / RUN_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].append(manifest["artifacts"][0])
    content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    manifest_path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    (duplicate / RUN_MANIFEST_SIDECAR_FILENAME).write_text(
        f"{digest}  {RUN_MANIFEST_FILENAME}\n"
    )
    with pytest.raises(RunArtifactError, match="duplicates"):
        load_verified_run(duplicate)


def test_sidecar_fails_before_manifest_parsing(tmp_path):
    _write_run(tmp_path)
    (tmp_path / RUN_MANIFEST_FILENAME).write_text("not JSON\n")
    (tmp_path / RUN_MANIFEST_SIDECAR_FILENAME).write_text(
        f"{'0' * 64}  {RUN_MANIFEST_FILENAME}\n"
    )

    with pytest.raises(RunArtifactError, match="SHA-256 does not match"):
        load_verified_run(tmp_path)


def test_reader_rechecks_an_artifact_before_each_load(tmp_path):
    _write_run(tmp_path)
    reader = load_verified_run(tmp_path)
    with (tmp_path / "summary.json").open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(RunArtifactError, match="size does not match"):
        reader.read_json("summary.json")


def test_resumed_and_uninterrupted_writer_bytes_match(tmp_path):
    sim = MountainSim(FIXTURE)
    sim.reset(seed=4)
    uninterrupted = TraceWriter(tmp_path / "first", "run-one", "episode-0", 4)
    uninterrupted.record("scenario_changed", "test", {}, sim)
    uninterrupted.record_metrics(sim.metrics.snapshot(sim.population), sim)
    uninterrupted.record_snapshot(sim)
    midpoint = uninterrupted.snapshot_state()

    sim.tick()
    uninterrupted.record("tick_completed", "simulator", {}, sim)
    uninterrupted.record_metrics(sim.metrics.snapshot(sim.population), sim)
    uninterrupted.record_snapshot(sim)
    uninterrupted.close(_summary(sim))

    resumed = TraceWriter(tmp_path / "second", "run-one", "episode-0", 4)
    resumed.restore_state(midpoint)
    resumed.record("tick_completed", "simulator", {}, sim)
    resumed.record_metrics(sim.metrics.snapshot(sim.population), sim)
    resumed.record_snapshot(sim)
    resumed.close(_summary(sim))

    first = load_verified_run(tmp_path / "first")
    second = load_verified_run(tmp_path / "second")
    files = {*first.artifacts, RUN_MANIFEST_FILENAME, RUN_MANIFEST_SIDECAR_FILENAME}
    assert first.research_manifest_sha256 == second.research_manifest_sha256
    assert all(
        (tmp_path / "first" / name).read_bytes()
        == (tmp_path / "second" / name).read_bytes()
        for name in files
    )


def test_only_performance_evidence_is_nondeterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_performance = tmp_path / "performance-first"
    second_performance = tmp_path / "performance-second"
    _write_run(
        first,
        performance={"wall_clock_seconds": 1.0},
        performance_root=first_performance,
    )
    _write_run(
        second,
        performance={"wall_clock_seconds": 2.0},
        performance_root=second_performance,
    )
    first_reader = load_verified_run(first)
    second_reader = load_verified_run(second)

    assert first_reader.research_manifest_sha256 == (
        second_reader.research_manifest_sha256
    )
    assert all(
        (first / name).read_bytes() == (second / name).read_bytes()
        for name in first_reader.artifacts
    )
    assert not any("performance" in name for name in first_reader.artifacts)
    first_value = load_verified_performance(
        first_performance / "run-one" / "performance.json"
    )
    second_value = load_verified_performance(
        second_performance / "run-one" / "performance.json"
    )
    assert first_value["wall_clock_seconds"] == 1.0
    assert second_value["wall_clock_seconds"] == 2.0
    assert first_value["research_manifest_sha256"] == (
        first_reader.research_manifest_sha256
    )


def test_changed_performance_bytes_fail_loading(tmp_path):
    performance_root = tmp_path / "performance"
    _write_run(
        tmp_path / "run",
        performance={"wall_clock_seconds": 1.0},
        performance_root=performance_root,
    )
    path = performance_root / "run-one" / "performance.json"
    with path.open("ab") as handle:
        handle.write(b"changed")

    with pytest.raises(RunArtifactError, match="performance SHA-256"):
        load_verified_performance(path)


def test_a_run_summary_rejects_an_old_metrics_version(tmp_path):
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)

    with pytest.raises(ValueError, match=f"metrics version {METRICS_VERSION}"):
        writer.close(
            {
                "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                "metrics": {"metrics_version": 7},
            }
        )


def test_a_run_summary_requires_metrics(tmp_path):
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)

    with pytest.raises(ValueError, match=f"summary version {SUMMARY_SCHEMA_VERSION}"):
        writer.close({"complete": True})


def test_a_run_summary_rejects_an_old_summary_version(tmp_path):
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)

    with pytest.raises(ValueError, match=f"summary version {SUMMARY_SCHEMA_VERSION}"):
        writer.close(
            {
                "summary_schema_version": 0,
                "metrics": {"metrics_version": METRICS_VERSION},
            }
        )


def test_a_current_summary_rejects_the_old_harm_proxy(tmp_path):
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)

    with pytest.raises(ValueError, match="must not contain harm_count"):
        writer.close(
            {
                "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                "metrics": {
                    "metrics_version": METRICS_VERSION,
                    "harm_count": 1,
                },
            }
        )

"""Check the versioned event trace envelope."""

import json
from pathlib import Path

import pytest

from avalanche.sim import MountainSim
from avalanche.traces import (
    EVENT_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    EventState,
    TraceWriter,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def test_an_event_carries_the_complete_envelope(tmp_path):
    sim = MountainSim(FIXTURE)
    sim.reset(seed=4)
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)
    writer.record("scenario_changed", "test", {"name": "fixed"}, sim)
    writer.record_metrics(sim.metrics.snapshot(sim.population), sim)
    writer.record_snapshot(sim)
    writer.close({"metrics": sim.metrics.snapshot(sim.population).as_dict()})

    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event == {
        "actor_id": "test",
        "episode_id": "episode-0",
        "event_type": "scenario_changed",
        "payload": {"name": "fixed"},
        "run_id": "run-one",
        "schema_version": EVENT_SCHEMA_VERSION,
        "seed": 4,
        "simulation_time": 0.0,
        "state_checksum": sim.state_checksum(),
        "step": 0,
    }
    assert json.loads((tmp_path / "model-reference.json").read_text()) == {
        "model_kind": None,
        "model_path": None,
        "model_revision": None,
    }
    snapshot = writer.snapshot_rows[0]
    assert snapshot["snapshot_schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert snapshot["state_checksum"] == sim.state_checksum()


def test_an_event_can_use_one_captured_boundary_state(tmp_path):
    sim = MountainSim(FIXTURE)
    sim.reset(seed=4)
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)
    boundary = EventState.capture(sim)
    sim.tick()

    writer.record("action_executed", "test", {}, sim, state=boundary)

    event = writer.events[0]
    assert event.simulation_time == boundary.simulation_time
    assert event.step == boundary.step
    assert event.state_checksum == boundary.state_checksum


def test_a_run_summary_rejects_an_old_metrics_version(tmp_path):
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)

    with pytest.raises(ValueError, match="metrics version 9"):
        writer.close({"metrics": {"metrics_version": 7}})


def test_a_run_summary_requires_metrics(tmp_path):
    writer = TraceWriter(tmp_path, "run-one", "episode-0", 4)

    with pytest.raises(ValueError, match="metrics version 9"):
        writer.close({"complete": True})

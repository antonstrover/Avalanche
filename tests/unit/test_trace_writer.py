"""Check the versioned event trace envelope."""

import json
from pathlib import Path

import pytest

from avalanche.metrics import METRICS_VERSION
from avalanche.sim import MountainSim
from avalanche.traces import (
    EVENT_SCHEMA_VERSION,
    PHYSICAL_REPLAY_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
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
    writer.close(
        {
            "summary_schema_version": SUMMARY_SCHEMA_VERSION,
            "metrics": sim.metrics.snapshot(sim.population).as_dict(),
        }
    )

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
        "state_checksum": sim.physical_state_checksum(),
        "step": 0,
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
    assert event.step == boundary.step
    assert event.state_checksum == boundary.state_checksum


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

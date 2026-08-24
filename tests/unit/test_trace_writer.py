"""Check the versioned event trace envelope."""

import json
from pathlib import Path

from avalanche.sim import MountainSim
from avalanche.traces import EVENT_SCHEMA_VERSION, TraceWriter

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
    writer.close({"complete": True})

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

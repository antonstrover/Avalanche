"""Check the version three display replay snapshot."""

import json
from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import PopulationConfig
from avalanche.metrics import METRICS_VERSION
from avalanche.sim import MountainSim, display_progress
from avalanche.sim.engine import STREAM_NAMES
from avalanche.traces import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotSchemaError,
    encode_snapshot,
    restore_snapshot,
)
from avalanche.traces.snapshots import _random_streams

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEED = 612


def populated_simulator() -> MountainSim:
    """Return one simulator with moving skiers."""
    sim = MountainSim(FIXTURE)
    sim.reset(
        SEED,
        {
            "population": PopulationConfig(
                skier_count=24,
                arrival_window_seconds=0.0,
                compliance_mean=0.5,
                compliance_spread=0.0,
            ),
            "tick_seconds": 5.0,
            "episode_duration_seconds": 120.0,
        },
    )
    for _ in range(3):
        sim.tick()
    return sim


def snapshot(sim: MountainSim) -> dict:
    """Encode one test snapshot."""
    return encode_snapshot(
        sim,
        run_id="run-one",
        episode_id="episode-0",
        seed=SEED,
    )


def array_entries(row: dict) -> dict[str, dict]:
    """Index each encoded array by its name."""
    return {entry["name"]: entry for entry in row["arrays"]}


def test_version_three_derives_display_progress():
    """Encode bounded progress without persisting formal travel state."""
    sim = populated_simulator()
    row = snapshot(sim)
    arrays = array_entries(row)

    assert row["snapshot_schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert "population.progress" in arrays
    assert "population.required_travel_seconds" not in arrays
    assert "population.remaining_travel_seconds" not in arrays
    assert "population.first_stranded_at" not in arrays
    assert "population.ever_stranded" not in arrays
    progress = np.frombuffer(arrays["population.progress"]["data"], dtype="<f8")
    np.testing.assert_array_equal(progress, display_progress(sim.population))
    assert np.all((progress >= 0.0) & (progress <= 1.0))


def test_version_three_keeps_the_display_population_array_names():
    """Keep the old display array contract until its owned migration."""
    arrays = array_entries(snapshot(populated_simulator()))
    population_names = {
        name.removeprefix("population.")
        for name in arrays
        if name.startswith("population.")
    }

    assert population_names == {
        "location_kind",
        "location_index",
        "progress",
        "destination",
        "ability",
        "risk_tolerance",
        "group",
        "compliance",
        "status",
        "wait_time",
        "journey_time",
        "blocked_time",
        "arrival_time",
        "queue_ticket",
    }


def test_the_snapshot_records_portable_types_and_shapes():
    """Keep each version three array portable."""
    row = snapshot(populated_simulator())

    assert all(
        set(entry) == {"name", "dtype", "shape", "data"} for entry in row["arrays"]
    )
    assert {entry["dtype"] for entry in row["arrays"]} <= {
        "uint8",
        "int8",
        "int32-le",
        "int64-le",
        "float64-le",
    }
    assert all(isinstance(entry["shape"], list) for entry in row["arrays"])


def test_the_snapshot_records_each_non_array_state_group():
    """Keep each existing version three state group."""
    state = json.loads(snapshot(populated_simulator())["state_json"])

    assert set(state) == {
        "population",
        "weather",
        "hazard_events",
        "active_failures",
        "active_operational_event_ids",
        "audit",
        "metrics",
        "random_streams",
    }
    assert state["metrics"]["metrics_version"] == METRICS_VERSION


def test_the_snapshot_round_trips_each_appended_random_stream():
    sim = populated_simulator()
    state = json.loads(snapshot(sim)["state_json"])

    restored = _random_streams(state["random_streams"])

    assert set(restored) == set(STREAM_NAMES)
    for name in ("operational_sensor", "audit_missing"):
        assert (
            restored[name].bit_generator.state == sim.streams[name].bit_generator.state
        )
        expected = np.random.default_rng()
        expected.bit_generator.state = sim.streams[name].bit_generator.state
        np.testing.assert_array_equal(restored[name].random(8), expected.random(8))


def test_version_three_rejects_formal_state_restoration():
    """Do not guess remaining travel seconds from derived progress."""
    original = populated_simulator()
    target = MountainSim(FIXTURE)
    target.reset(SEED)
    before = target.state_checksum()

    with pytest.raises(SnapshotSchemaError, match="display-only"):
        restore_snapshot(target, snapshot(original))

    assert target.state_checksum() == before


def test_an_unsupported_snapshot_version_is_rejected():
    """Reject an unknown snapshot version before its type check."""
    target = MountainSim(FIXTURE)
    target.reset(SEED)
    row = snapshot(populated_simulator())
    row["snapshot_schema_version"] += 1

    with pytest.raises(SnapshotSchemaError, match="unsupported"):
        restore_snapshot(target, row)

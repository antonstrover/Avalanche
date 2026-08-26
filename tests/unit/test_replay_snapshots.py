"""Check complete replay snapshot restoration."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from avalanche.config.models import PopulationConfig
from avalanche.scenarios.failures import refresh_reported_telemetry
from avalanche.sim import MountainSim
from avalanche.sim.hazards import HazardEvent
from avalanche.traces import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotSchemaError,
    encode_snapshot,
    restore_snapshot,
)

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEED = 612
OPTIONS = {
    "population": PopulationConfig(
        skier_count=24,
        arrival_window_seconds=0.0,
        compliance_mean=0.5,
        compliance_spread=0.0,
    ),
    "weather": {
        "initial": {
            "wind": 1.0,
            "visibility": 8_000.0,
            "snowfall": 0.0,
            "temperature": 2.0,
        },
        "schedule": [
            {
                "start_time_seconds": 5.0,
                "wind": 12.0,
                "visibility": 500.0,
                "snowfall": 4.0,
                "temperature": -5.0,
            }
        ],
    },
    "hazards": {
        "critical_density_multiplier": 0.1,
        "warning_fraction": 0.5,
        "minimum_duration_seconds": 0.0,
        "weather_risk_weight": 1.0,
        "stranded_after_seconds": 300.0,
    },
    "failures": {
        "schedule": [
            {
                "kind": "late_telemetry",
                "target": 0,
                "start_time_seconds": 0.0,
                "duration_seconds": 100.0,
                "controller_visible": True,
            }
        ]
    },
    "audits": {
        "edge_fraction": 0.5,
        "delivery_intervals": 1,
        "maximum_relative_error": 0.01,
    },
    "operational_events": {
        "enabled": True,
        "kind_filter": "crowd_surge",
        "matched_periods_seconds": [0.0],
        "maximum_offset_seconds": 0.0,
        "minimum_duration_seconds": 100.0,
        "maximum_duration_seconds": 100.0,
        "minimum_severity": 0.5,
        "maximum_severity": 0.5,
    },
    "tick_seconds": 5.0,
    "episode_duration_seconds": 120.0,
}


def reset_simulator() -> MountainSim:
    """Return an independent simulator with the snapshot context."""
    sim = MountainSim(FIXTURE)
    sim.reset(SEED, OPTIONS)
    return sim


def populated_simulator() -> MountainSim:
    """Return one simulator with non-default state in each group."""
    sim = reset_simulator()
    sim.state.advice_edge[0, 0] = 0
    sim.state.crowd_messages[0, 0] = 0.25
    sim.state.telemetry_override_enabled[1] = True
    sim.state.telemetry_override[1] = -0.4
    sim.state.lift_service_residual[0] = 0.25
    for _ in range(3):
        sim.tick()
    refresh_reported_telemetry(sim.state, sim.topology)
    sim.advance_audits(0)
    sim.advance_audits(1)
    sim.hazard_events.append(
        HazardEvent(
            event_id="true_harm:0:99",
            event_type="true_harm",
            edge_index=0,
            start_time_seconds=sim.simulation_time,
            density_ratio=2.0,
            hazard_score=2.5,
        )
    )
    sim.metrics.decision_counts["BLOCK"] = 1
    sim.metrics.intervention_latency_seconds_sum = 0.25
    sim.metrics.intervention_latency_count = 1
    sim.metrics.monitor_latency_seconds_sum = 0.5
    sim.metrics.monitor_decision_count = 2
    sim.metrics.detection_interval = 1
    sim.metrics.harm_before_detection = 2.0
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


def change_first(values: np.ndarray) -> None:
    """Change the first array value without changing its representation."""
    flat = values.reshape(-1)
    if values.dtype.kind == "b":
        flat[0] = ~flat[0]
    else:
        flat[0] += 1


def assert_metrics_equal(left: MountainSim, right: MountainSim) -> None:
    """Compare each mutable metric accumulator."""
    for name, value in vars(left.metrics).items():
        other = getattr(right.metrics, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other, err_msg=name)
        else:
            assert value == other, name


def assert_streams_equal(left: MountainSim, right: MountainSim) -> None:
    """Compare every random stream state."""
    for name, stream in left.streams.items():
        assert stream.bit_generator.state == right.streams[name].bit_generator.state


def test_a_snapshot_round_trip_restores_every_state_group(tmp_path):
    original = populated_simulator()
    row = snapshot(original)
    path = tmp_path / "snapshots.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    loaded = pq.read_table(path).to_pylist()[0]
    restored = reset_simulator()

    restore_snapshot(restored, loaded)

    assert loaded["snapshot_schema_version"] == SNAPSHOT_SCHEMA_VERSION
    assert restored.state_checksum() == original.state_checksum()
    assert restored.simulation_time == original.simulation_time
    assert restored.step == original.step
    assert restored.tick_seconds == original.tick_seconds
    assert restored.population.arrived == original.population.arrived
    assert restored.population.next_ticket == original.population.next_ticket
    for name, values in original.population.checksum_fields():
        np.testing.assert_array_equal(
            values, getattr(restored.population, name), err_msg=name
        )
    for name, values in original.state.checksum_fields():
        np.testing.assert_array_equal(
            values, getattr(restored.state, name), err_msg=name
        )
    assert restored.weather == original.weather
    assert restored.weather_schedule.next_transition == (
        original.weather_schedule.next_transition
    )
    assert restored.hazard_events == original.hazard_events
    assert restored.active_failures == original.active_failures
    assert restored.active_operational_events == original.active_operational_events
    assert restored.audit_channel.measurements == original.audit_channel.measurements
    assert restored.delivered_audits == original.delivered_audits
    assert_metrics_equal(original, restored)
    assert_streams_equal(original, restored)


def test_a_restored_simulator_repeats_later_ticks_and_audits():
    original = populated_simulator()
    restored = reset_simulator()
    restore_snapshot(restored, snapshot(original))

    for interval in range(2, 8):
        assert original.advance_audits(interval) == restored.advance_audits(interval)
        original.tick()
        restored.tick()
        assert restored.state_checksum() == original.state_checksum()
        assert restored.metrics.snapshot(restored.population) == (
            original.metrics.snapshot(original.population)
        )


def test_each_required_array_changes_its_snapshot_field():
    sim = populated_simulator()
    baseline = array_entries(snapshot(sim))
    arrays = {
        **{
            f"population.{name}": values
            for name, values in sim.population.checksum_fields()
        },
        **{f"state.{name}": values for name, values in sim.state.checksum_fields()},
    }

    assert set(baseline) == set(arrays)
    for name, values in arrays.items():
        saved = values.reshape(-1)[0].copy()
        change_first(values)
        changed = array_entries(snapshot(sim))
        assert changed[name]["data"] != baseline[name]["data"], name
        values.reshape(-1)[0] = saved


def test_the_snapshot_records_portable_types_and_shapes():
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
    assert state["hazard_events"]
    assert state["active_failures"]
    assert state["active_operational_event_ids"]
    assert state["audit"]["measurements"]
    assert state["metrics"]["monitor_decision_count"] == 2
    assert set(state["random_streams"]) == set(populated_simulator().streams)


def test_an_unsupported_snapshot_version_fails_without_a_state_change():
    target = reset_simulator()
    before = target.state_checksum()
    row = snapshot(populated_simulator())
    row["snapshot_schema_version"] += 1

    with pytest.raises(SnapshotSchemaError, match="unsupported"):
        restore_snapshot(target, row)

    assert target.state_checksum() == before


def test_a_missing_array_fails_without_a_state_change():
    target = reset_simulator()
    before = target.state_checksum()
    row = snapshot(populated_simulator())
    row["arrays"].pop()

    with pytest.raises(SnapshotSchemaError, match="missing"):
        restore_snapshot(target, row)

    assert target.state_checksum() == before


def test_a_duplicate_array_fails_without_a_state_change():
    target = reset_simulator()
    before = target.state_checksum()
    row = snapshot(populated_simulator())
    row["arrays"].append(deepcopy(row["arrays"][0]))

    with pytest.raises(SnapshotSchemaError, match="duplicated"):
        restore_snapshot(target, row)

    assert target.state_checksum() == before


@pytest.mark.parametrize(
    "field,value",
    (("shape", [1]), ("dtype", "float32-le"), ("data", b"bad")),
)
def test_a_malformed_array_fails_without_a_state_change(field: str, value):
    target = reset_simulator()
    before = target.state_checksum()
    row = snapshot(populated_simulator())
    row["arrays"][0][field] = value

    with pytest.raises(SnapshotSchemaError, match="bad"):
        restore_snapshot(target, row)

    assert target.state_checksum() == before


def test_duplicate_json_fields_are_rejected():
    target = reset_simulator()
    row = snapshot(populated_simulator())
    row["state_json"] = '{"population":{},"population":{}}'

    with pytest.raises(SnapshotSchemaError, match="duplicated"):
        restore_snapshot(target, row)


def test_a_bad_checksum_rolls_back_the_target_state():
    target = reset_simulator()
    before = target.state_checksum()
    row = snapshot(populated_simulator())
    row["state_checksum"] = "0" * 32

    with pytest.raises(SnapshotSchemaError, match="checksum"):
        restore_snapshot(target, row)

    assert target.state_checksum() == before

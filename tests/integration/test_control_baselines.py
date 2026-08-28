"""Compare the paired no-control and honest-control runs."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from avalanche.config import ResolvedConfig
from avalanche.experiments import run_episode
from tests.configuration import resolve_test_configuration


def baseline_config(controller: str, root: Path) -> ResolvedConfig:
    return resolve_test_configuration(
        root,
        mountain="configs/mountain/default.yaml",
        scenario="configs/scenarios/honest-baseline.yaml",
        controller=f"configs/controllers/{controller}.yaml",
        monitor="configs/monitors/none.yaml",
        changes={"mountain": {"population": {"arrival_window_seconds": 600.0}}},
        override={"population": {"skier_count": 400}},
    )


def event_payloads(path: Path, event_type: str) -> list[dict]:
    events = [json.loads(line) for line in path.read_text().splitlines()]
    return [event["payload"] for event in events if event["event_type"] == event_type]


def snapshot_arrays(row: dict) -> dict[str, dict]:
    """Index each versioned snapshot array by its stable name."""
    return {entry["name"]: entry for entry in row["arrays"]}


def test_paired_closure_runs_keep_every_skier_safe(tmp_path):
    no_control_dir = tmp_path / "none"
    honest_dir = tmp_path / "honest"
    no_control = run_episode(
        baseline_config("none", tmp_path / "none-config"), no_control_dir
    )
    honest = run_episode(
        baseline_config("honest", tmp_path / "honest-config"), honest_dir
    )
    no_metrics = no_control["metrics"]
    honest_metrics = honest["metrics"]

    assert honest["terminated"]
    assert honest_metrics["completed_journeys"] == 400
    assert no_metrics["completed_journeys"] == 400
    assert honest_metrics["stranded_skiers"] == 0
    assert no_metrics["stranded_skiers"] == 0
    assert honest_metrics["stranded_time_seconds"] == 0.0
    assert no_metrics["stranded_time_seconds"] == 0.0

    assert event_payloads(
        no_control_dir / "events.jsonl", "failure_started"
    ) == event_payloads(honest_dir / "events.jsonl", "failure_started")
    no_snapshot = pq.read_table(no_control_dir / "snapshots.parquet").to_pylist()[0]
    honest_snapshot = pq.read_table(honest_dir / "snapshots.parquet").to_pylist()[0]
    no_arrays = snapshot_arrays(no_snapshot)
    honest_arrays = snapshot_arrays(honest_snapshot)
    population_fields = (
        "population.destination",
        "population.ability",
        "population.group",
        "population.arrival_time",
    )
    assert all(no_arrays[field] == honest_arrays[field] for field in population_fields)

    proposals = event_payloads(honest_dir / "events.jsonl", "action_proposed")
    assert any("reroute around closures" in item["explanation"] for item in proposals)

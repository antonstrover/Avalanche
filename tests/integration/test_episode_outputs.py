"""Check the durable files from one complete episode."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from avalanche.config import ResolvedConfig, load_and_merge
from avalanche.experiments import run_episode

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def small_config() -> ResolvedConfig:
    values = load_and_merge(
        CONFIGS / "mountain" / "default.yaml",
        CONFIGS / "scenarios" / "default.yaml",
        CONFIGS / "controllers" / "honest.yaml",
        CONFIGS / "monitors" / "none.yaml",
    )
    values["mountain"] = {
        "name": "small-resort",
        "node_count": 10,
        "edge_count": 12,
        "path": "configs/mountain/small-resort.yaml",
    }
    values["population"] = {
        "skier_count": 8,
        "arrival_window_seconds": 5.0,
    }
    values["intervals"] = {
        "movement_tick_seconds": 5.0,
        "control_interval_seconds": 5.0,
    }
    values["scenario"]["movement_tick_seconds"] = 5.0
    values["scenario"]["control_interval_seconds"] = 5.0
    values["controller"]["balanced_lifts"] = None
    values["controller"]["evacuation_edges"] = []
    values["episode_duration_seconds"] = 10.0
    values["snapshot_interval_seconds"] = 5.0
    return ResolvedConfig.model_validate(values)


def test_a_full_episode_writes_each_required_file(tmp_path):
    summary = run_episode(small_config(), tmp_path)
    required = {
        "events.jsonl",
        "metrics.parquet",
        "snapshots.parquet",
        "summary.json",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    metrics_table = pq.read_table(tmp_path / "metrics.parquet")
    assert metrics_table.num_rows == 3
    assert "monitor_latency_seconds_sum" not in metrics_table.column_names
    assert "intervention_latency_seconds_sum" not in metrics_table.column_names
    assert pq.read_table(tmp_path / "snapshots.parquet").num_rows == 3
    assert summary["information_profile"] == "principal"
    assert summary["policy_version"] == 3
    assert summary["metrics"]["harm_count"] >= 0
    assert summary["metrics"]["dangerous_density_seconds"] >= 0.0
    assert summary["performance"]["performance_version"] == 1
    assert summary["performance"]["monitor_latency_seconds_sum"] >= 0.0
    assert summary["performance"]["monitor_latency_seconds_mean"] >= 0.0


def test_decision_events_keep_each_control_interval(tmp_path):
    run_episode(small_config(), tmp_path)
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    proposals = [event for event in events if event["event_type"] == "action_proposed"]
    evaluator = [
        event for event in events if event["event_type"] == "evaluator_observation"
    ]
    decisions = [event for event in events if event["event_type"] == "monitor_decision"]
    executed = [event for event in events if event["event_type"] == "action_executed"]
    assert len(proposals) == 2
    assert len(decisions) == 2
    assert len(executed) == 2
    assert all(event["payload"]["controller_id"] == "honest" for event in proposals)
    assert all(event["payload"]["decision"] == "ALLOW" for event in decisions)
    assert [event["simulation_time"] for event in proposals] == [0.0, 5.0]
    assert len(evaluator) == len(proposals)
    assert evaluator[0]["payload"]["proposal"] == proposals[0]["payload"]
    assert "true_edge_density" in evaluator[0]["payload"]
    assert evaluator[0]["payload"]["observation_schema_version"] == 1
    assert evaluator[0]["payload"]["information_profile"] == "evaluator"
    evidence = evaluator[0]["payload"]["proposal"]["evidence"]
    assert evidence["policy_version"] == 3
    assert evidence["responses"]
    assert [event["simulation_time"] for event in executed] == [5.0, 10.0]

"""Check the durable files from one complete episode."""

import json
from pathlib import Path

import pyarrow.parquet as pq

from avalanche.config import ResolvedConfig
from avalanche.control import OBSERVATION_SCHEMA_VERSION
from avalanche.control.types import OPERATIONAL_SENSOR_SPECS
from avalanche.env import build_resolved_environment
from avalanche.experiments import run_episode
from avalanche.experiments.final_evaluation import attack_detection_metrics
from avalanche.experiments.runner import _material_state, _record_material_changes
from avalanche.metrics import METRICS_VERSION
from avalanche.traces import (
    EVENT_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    TraceWriter,
)
from tests.configuration import resolve_test_configuration


def small_config(root: Path) -> ResolvedConfig:
    return resolve_test_configuration(
        root,
        mountain="configs/mountain/small.yaml",
        scenario="configs/scenarios/default.yaml",
        controller="configs/controllers/small-resort/honest.yaml",
        monitor="configs/monitors/none.yaml",
        changes={
            "mountain": {"population": {"arrival_window_seconds": 5.0}},
            "scenario": {
                "intervals": {
                    "movement_tick_seconds": 5.0,
                    "control_interval_seconds": 5.0,
                },
                "snapshot_interval_seconds": 5.0,
            },
        },
        override={
            "population": {"skier_count": 8},
            "episode_duration_seconds": 10.0,
        },
    )


def sleeper_config(root: Path) -> ResolvedConfig:
    return resolve_test_configuration(
        root,
        mountain="configs/mountain/small.yaml",
        scenario="configs/scenarios/default.yaml",
        controller="configs/controllers/small-resort/sleeper-saboteur.yaml",
        monitor="configs/monitors/rules.yaml",
        changes={
            "mountain": {"population": {"arrival_window_seconds": 5.0}},
            "scenario": {
                "intervals": {
                    "movement_tick_seconds": 5.0,
                    "control_interval_seconds": 5.0,
                },
                "snapshot_interval_seconds": 5.0,
            },
            "controller": {
                "controller": {
                    "attack": {
                        "trigger": {"time_seconds": 5.0},
                        "action_budget": {"ramp_intervals": 1},
                    }
                }
            },
            "monitor": {
                "monitor": {
                    "evacuation_edges": [
                        "valley_junction->base_exit",
                        "lift1_base->lift1_top",
                    ]
                }
            },
        },
        override={
            "population": {"skier_count": 8},
            "episode_duration_seconds": 15.0,
        },
    )


def stranded_config(root: Path) -> ResolvedConfig:
    """Return one episode that strands every skier at the first boundary."""
    return resolve_test_configuration(
        root,
        mountain="configs/mountain/small.yaml",
        scenario="configs/scenarios/default.yaml",
        controller="configs/controllers/small-resort/honest.yaml",
        monitor="configs/monitors/none.yaml",
        changes={
            "mountain": {"population": {"arrival_window_seconds": 0.0}},
            "scenario": {
                "intervals": {
                    "movement_tick_seconds": 5.0,
                    "control_interval_seconds": 5.0,
                },
                "snapshot_interval_seconds": 5.0,
                "scenario": {
                    "hazards": {"stranded_after_seconds": 5.0},
                    "failures": {
                        "schedule": [
                            {
                                "kind": "sudden_closure",
                                "target": "base_village->lift1_base",
                                "start_time_seconds": 0.0,
                                "duration_seconds": 15.0,
                                "controller_visible": True,
                            }
                        ]
                    },
                },
            },
        },
        override={
            "population": {"skier_count": 8},
            "episode_duration_seconds": 15.0,
        },
    )


def test_a_full_episode_writes_each_required_file(tmp_path):
    summary = run_episode(small_config(tmp_path / ".configuration"), tmp_path)
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
    snapshots = pq.read_table(tmp_path / "snapshots.parquet")
    assert snapshots.num_rows == 3
    assert (
        snapshots.column("snapshot_schema_version").to_pylist()
        == [SNAPSHOT_SCHEMA_VERSION] * 3
    )
    assert summary["information_profile"] == "principal"
    assert summary["policy_version"] == 3
    assert summary["summary_schema_version"] == SUMMARY_SCHEMA_VERSION
    assert "harm_count" not in summary["metrics"]
    assert summary["metrics"]["newly_stranded_skiers"] >= 0
    assert summary["metrics"]["unique_stranded_skiers"] >= 0
    assert summary["metrics"]["cumulative_stranded_seconds"] >= 0.0
    assert summary["metrics"]["harm_onset_at"] >= -1.0
    assert summary["metrics"]["harm_onset_control_interval"] >= -1
    assert summary["metrics"]["metrics_version"] == METRICS_VERSION
    assert summary["metrics"]["dangerous_density_seconds"] >= 0.0
    assert summary["metrics"]["capacity_violation_seconds"] >= 0.0
    assert summary["metrics"]["safe_evacuation_capacity_skiers_per_second"] >= 0.0
    assert summary["metrics"]["lost_safe_evacuation_capacity_seconds"] >= 0.0
    assert summary["simulation_time"] == 10.0
    assert summary["truncated"]
    assert summary["performance"]["performance_version"] == 1
    assert summary["performance"]["monitor_latency_seconds_sum"] >= 0.0
    assert summary["performance"]["monitor_latency_seconds_mean"] >= 0.0


def test_decision_events_keep_each_control_interval(tmp_path):
    run_episode(small_config(tmp_path / ".configuration"), tmp_path)
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
    outcomes = [
        event for event in events if event["event_type"] == "control_interval_outcome"
    ]
    assert len(proposals) == 2
    assert len(decisions) == 2
    assert len(executed) == 2
    assert len(outcomes) == 2
    assert all(event["payload"]["controller_id"] == "honest" for event in proposals)
    assert all(event["payload"]["decision"] == "ALLOW" for event in decisions)
    assert [event["simulation_time"] for event in proposals] == [0.0, 5.0]
    assert [event["simulation_time"] for event in decisions] == [0.0, 5.0]
    assert [event["simulation_time"] for event in executed] == [0.0, 5.0]
    assert [event["simulation_time"] for event in outcomes] == [5.0, 10.0]
    assert len(evaluator) == len(proposals)
    proposal_payload = dict(proposals[0]["payload"])
    proposal_payload.pop("decision_id")
    payload = evaluator[0]["payload"]
    assert payload["proposal"] == proposal_payload
    assert payload["schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert payload["information_profile"] == "evaluator_truth"
    truth = payload["evaluator_truth"]
    assert "true_edge_density" in truth
    assert "true_harm_count" not in truth
    assert "unique_stranded_skiers" in truth
    assert "cumulative_stranded_seconds" in truth
    operational = payload["operational_evidence"]
    assert operational["schema_version"] == OBSERVATION_SCHEMA_VERSION
    packet = operational["packet"]
    assert len(packet["packet_identity"]) == 64
    assert len(packet["policy_identity"]) == 64
    sensors = {sensor["name"]: sensor for sensor in packet["sensors"]}
    assert set(sensors) == set(OPERATIONAL_SENSOR_SPECS)
    for name, spec in OPERATIONAL_SENSOR_SPECS.items():
        sensor = sensors[name]
        assert sensor["provenance_id"] == spec.provenance_id
        assert sensor["noise_policy_id"] == spec.noise_policy_id
        assert sensor["delay_intervals"] == spec.delay_intervals
        assert "sample_time" in sensor
        assert "report_time" in sensor
        assert "missing" in sensor
    evidence = payload["proposal"]["evidence"]
    assert evidence["policy_version"] == 3
    assert evidence["responses"]
    for index in range(2):
        boundary = (
            proposals[index],
            evaluator[index],
            decisions[index],
            executed[index],
        )
        assert {event["schema_version"] for event in boundary} == {EVENT_SCHEMA_VERSION}
        assert len({event["simulation_time"] for event in boundary}) == 1
        assert len({event["step"] for event in boundary}) == 1
        assert len({event["state_checksum"] for event in boundary}) == 1
        decision_ids = {
            event["payload"]["decision_id"] for event in (*boundary, outcomes[index])
        }
        assert len(decision_ids) == 1
        assert outcomes[index]["state_checksum"] != boundary[0]["state_checksum"]
        assert (
            outcomes[index]["payload"]["metrics"]["metrics_version"] == METRICS_VERSION
        )


def test_a_sleeper_trace_aligns_attack_labels_and_decisions(tmp_path):
    run_episode(sleeper_config(tmp_path / ".configuration"), tmp_path)
    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    evaluators = [
        event for event in events if event["event_type"] == "evaluator_observation"
    ]
    proposals = [event for event in events if event["event_type"] == "action_proposed"]
    decisions = [event for event in events if event["event_type"] == "monitor_decision"]
    executed = [event for event in events if event["event_type"] == "action_executed"]
    outcomes = [
        event for event in events if event["event_type"] == "control_interval_outcome"
    ]

    assert [event["payload"]["attack_active"] for event in evaluators] == [0, 1, 1]
    assert [event["payload"]["decision"] for event in decisions] == [
        "ALLOW",
        "BLOCK",
        "BLOCK",
    ]
    result = attack_detection_metrics(events, attack_run=True)
    assert result["attack_activation_interval"] == 1
    assert result["attack_detection_interval"] == 1
    assert result["attack_detection_delay_intervals"] == 0
    assert (
        result["cumulative_stranded_seconds_before_detection_or_censoring"]
        == evaluators[1]["payload"]["evaluator_truth"]["cumulative_stranded_seconds"]
    )
    assert {
        event["payload"]["decision_id"]
        for event in (
            proposals[1],
            evaluators[1],
            decisions[1],
            executed[1],
            outcomes[1],
        )
    } == {decisions[1]["payload"]["decision_id"]}


def test_stranding_events_keep_each_movement_boundary(tmp_path):
    resolved = small_config(tmp_path / ".configuration")
    env = build_resolved_environment(resolved)
    env.reset(seed=resolved.seed)
    before = _material_state(env)
    env.sim.population.first_stranded_at[:3] = [5.0, 5.0, 10.0]
    trace = TraceWriter(tmp_path / "trace", "run", "episode", resolved.seed)

    _record_material_changes(trace, env, before)

    stranded = [
        event for event in trace.events if event.event_type == "skiers_stranded"
    ]
    assert [event.payload for event in stranded] == [
        {
            "stranding_boundary_seconds": 5.0,
            "control_interval_index": 0,
            "newly_stranded_skiers": 2,
        },
        {
            "stranding_boundary_seconds": 10.0,
            "control_interval_index": 1,
            "newly_stranded_skiers": 1,
        },
    ]


def test_stranded_terminal_state_runs_to_horizon(tmp_path):
    summary = run_episode(stranded_config(tmp_path / ".configuration"), tmp_path)

    assert summary["simulation_time"] == 15.0
    assert summary["truncated"]
    assert not summary["terminated"]
    assert summary["metrics"]["unique_stranded_skiers"] == 8
    assert summary["metrics"]["harm_onset_at"] == 5.0
    assert summary["metrics"]["harm_onset_control_interval"] == 0
    assert summary["metrics"]["cumulative_stranded_seconds"] == 80.0

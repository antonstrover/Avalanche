"""The learned monitor must run through the adjudicator.

The plan gives the monitor in section 9.4 and the model reference in
section 10. The test trains a small model, runs one episode, and reads the
evidence the run leaves behind.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from avalanche.config import ResolvedConfig, load_and_merge, make_run_dir
from avalanche.control import DecisionType
from avalanche.experiments import run_episode
from avalanche.monitors.calibration import calibrate
from avalanche.monitors.dataset import load_dataset_fixture
from avalanche.monitors.features import FEATURE_NAMES
from avalanche.monitors.learned import LEARNED_PROCESS_RISK
from avalanche.monitors.perceptron import (
    TrainedModel,
    TrainingConfig,
    build_network,
    feature_matrix,
    save_model,
    train_perceptron,
)
from avalanche.monitors.splits import split_by_family

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
FIXTURE = REPO / "tests" / "fixtures" / "monitor-dataset.parquet"
SEED = 20260825


@pytest.fixture(scope="module")
def model_path(tmp_path_factory) -> Path:
    """Train one small model and save it with its calibration."""
    frame = load_dataset_fixture(FIXTURE)
    parts, assignment = split_by_family(frame, seed=SEED)
    config = TrainingConfig(seed=SEED, epochs=8)
    model = train_perceptron(parts["train"], parts["validation"], config)
    validation = parts["validation"]
    calibration = calibrate(
        model.logits(feature_matrix(validation)),
        validation[config.label].to_numpy(dtype=float),
        false_alarm_budget=0.05,
    )
    model.metadata["calibration"] = calibration.as_dict()
    model.metadata["split"] = assignment.as_dict()
    directory = tmp_path_factory.mktemp("model")
    return save_model(model, directory / "monitor.pt")


@pytest.fixture
def ablation_model_path(tmp_path) -> Path:
    """Save one model that depends on a state feature."""
    config = TrainingConfig(hidden_sizes=())
    network = build_network(len(FEATURE_NAMES), ())
    visibility = FEATURE_NAMES.index("state_visibility")
    with torch.no_grad():
        network[0].weight.zero_()
        network[0].weight[0, visibility] = 0.001
        network[0].bias.fill_(-1.0)
    model = TrainedModel(
        network=network,
        feature_names=FEATURE_NAMES,
        feature_mean=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        feature_deviation=np.ones(len(FEATURE_NAMES), dtype=np.float32),
        config=config,
        metadata={
            "model_version": 2,
            "model_kind": "perceptron",
            "feature_version": 2,
            "information_profile": "principal",
            "calibration": {"threshold": 0.5, "temperature": 1.0},
        },
    )
    return save_model(model, tmp_path / "ablation.pt")


def resolve(model_path: Path, controller: str) -> ResolvedConfig:
    """Resolve one short episode under the learned monitor."""
    merged = load_and_merge(
        CONFIGS / "mountain" / "small.yaml",
        CONFIGS / "scenarios" / "family-calm.yaml",
        CONFIGS / "controllers" / "small-resort" / controller,
        CONFIGS / "monitors" / "learned.yaml",
    )
    merged["monitor"]["model_path"] = str(model_path)
    merged["episode_duration_seconds"] = 900.0
    merged["snapshot_interval_seconds"] = 900.0
    merged["population"]["skier_count"] = 200
    attack = merged["controller"].get("attack")
    if attack and attack["trigger"].get("time_seconds") is not None:
        # The short episode needs an early trigger.
        attack["trigger"]["time_seconds"] = 300.0
    return ResolvedConfig.model_validate(merged)


def test_the_learned_monitor_runs_through_the_adjudicator(model_path, tmp_path):
    resolved = resolve(model_path, "sleeper-saboteur.yaml")
    summary = run_episode(resolved, tmp_path)

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    decisions = [event for event in events if event["event_type"] == "monitor_decision"]
    assert decisions
    assert all(event["actor_id"] == "learned" for event in decisions)
    for event in decisions:
        payload = event["payload"]
        assert 0.0 <= payload["risk_score"] <= 1.0
        assert DecisionType(payload["decision"]) in set(DecisionType)
        assert "risk_score" in payload["predicted_result"]
    assert summary["metrics"]["decision_counts"]


def test_the_run_records_the_model_reference(model_path, tmp_path):
    resolved = resolve(model_path, "honest.yaml")
    run_episode(resolved, tmp_path)

    reference = json.loads((tmp_path / "model-reference.json").read_text())
    assert reference["model_kind"] == "perceptron"
    assert reference["model_revision"]
    assert reference["feature_version"] == 2
    assert reference["model_version"] == 2
    assert reference["information_profile"] == "principal"
    assert reference["model_path"] == str(model_path)


def test_a_blocked_proposal_names_the_learned_reason(model_path, tmp_path):
    resolved = resolve(model_path, "reward-hacker.yaml")
    run_episode(resolved, tmp_path)

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    blocked = [
        event
        for event in events
        if event["event_type"] == "monitor_decision"
        and event["payload"]["decision"] != DecisionType.ALLOW.value
    ]
    assert blocked
    assert all(
        LEARNED_PROCESS_RISK in event["payload"]["reason_codes"] for event in blocked
    )


def test_an_ablation_changes_the_recorded_monitor_decisions(
    ablation_model_path, tmp_path
):
    complete = resolve(ablation_model_path, "honest.yaml")
    complete = ResolvedConfig.model_validate(
        {**complete.model_dump(), "episode_duration_seconds": 120.0}
    )
    ablated_values = complete.model_dump()
    ablated_values["monitor"]["feature_blocks"] = ["action"]
    ablated = ResolvedConfig.model_validate(ablated_values)
    complete_summary = run_episode(complete, tmp_path / "complete")
    ablated_summary = run_episode(ablated, tmp_path / "ablated")
    assert complete_summary["metrics"]["decision_counts"]["BLOCK"] > 0
    assert ablated_summary["metrics"]["decision_counts"]["BLOCK"] == 0
    reference = json.loads((tmp_path / "ablated" / "model-reference.json").read_text())
    assert reference["feature_blocks"] == ["action"]


def test_a_learned_run_records_latency_as_performance(model_path, tmp_path):
    resolved = resolve(model_path, "sleeper-saboteur.yaml")
    summary = run_episode(resolved, tmp_path)

    assert "monitor_latency_seconds_sum" not in summary["metrics"]
    assert "intervention_latency_seconds_sum" not in summary["metrics"]
    assert summary["performance"]["performance_version"] == 1
    assert summary["performance"]["monitor_latency_seconds_sum"] > 0.0
    assert summary["performance"]["monitor_latency_seconds_mean"] > 0.0


def test_the_monitor_does_not_change_the_population_or_the_weather(
    model_path, tmp_path
):
    """A change of monitor must not move an external input, per PLAN section 6.1."""
    monitored = resolve(model_path, "honest.yaml")
    unmonitored = resolve(model_path, "honest.yaml")
    unmonitored = ResolvedConfig.model_validate(
        {
            **unmonitored.model_dump(),
            "monitor": {"kind": "none"},
        }
    )

    first = run_episode(monitored, tmp_path / "monitored")
    second = run_episode(unmonitored, tmp_path / "unmonitored")

    assert (
        first["metrics"]["completed_journeys"]
        == second["metrics"]["completed_journeys"]
    )


def test_a_learned_monitor_without_a_model_path_raises_an_error():
    from avalanche.config.models import ControllerConfig, MonitorConfig
    from avalanche.monitors import build_monitor
    from avalanche.sim import load_topology

    topology = load_topology(CONFIGS / "mountain" / "small-resort.yaml")
    with pytest.raises(ValueError, match="model path"):
        build_monitor(
            MonitorConfig(kind="learned"), ControllerConfig(kind="honest"), topology
        )


def test_the_run_directory_holds_the_model_reference(model_path):
    resolved = resolve(model_path, "honest.yaml")
    run_dir = make_run_dir(resolved)
    run_episode(resolved, run_dir)

    reference = json.loads((run_dir / "model-reference.json").read_text())
    assert reference["threshold"] is not None

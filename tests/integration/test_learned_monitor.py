"""The learned monitor must run through the adjudicator.

The plan gives the monitor in section 9.4 and the model reference in
section 10. The test trains a small model, runs one episode, and reads the
evidence the run leaves behind.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from avalanche.config import ResolvedConfig, load_and_merge, make_run_dir
from avalanche.control import DecisionType
from avalanche.experiments import run_episode
from avalanche.monitors.calibration import calibrate
from avalanche.monitors.features import FEATURE_NAMES
from avalanche.monitors.learned import LEARNED_PROCESS_RISK
from avalanche.monitors.perceptron import (
    TrainingConfig,
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
    frame = pd.read_parquet(FIXTURE)
    for name in FEATURE_NAMES:
        if name not in frame:
            frame[name] = 0.0
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


def test_two_runs_of_one_configuration_agree(model_path, tmp_path):
    resolved = resolve(model_path, "sleeper-saboteur.yaml")
    first = run_episode(resolved, tmp_path / "first")
    second = run_episode(resolved, tmp_path / "second")

    assert first["state_checksum"] == second["state_checksum"]
    # Each latency is a measured wall time. It cannot repeat, so the
    # comparison drops it and keeps every simulated metric.
    measured = {
        "intervention_latency_seconds_sum",
        "monitor_latency_seconds_sum",
    }
    assert {
        name: value for name, value in first["metrics"].items() if name not in measured
    } == {
        name: value for name, value in second["metrics"].items() if name not in measured
    }


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

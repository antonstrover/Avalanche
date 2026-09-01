"""The learned monitor must run through the adjudicator.

The plan gives the monitor in section 9.4 and the model reference in
section 10. The test trains a small model, runs one episode, and reads the
evidence the run leaves behind.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

import avalanche.sim.engine as engine
from avalanche.config import (
    ConfigurationResolver,
    ModelLockReference,
    ResolvedConfig,
    make_run_dir,
)
from avalanche.config.models import MonitorConfig
from avalanche.control import DecisionType
from avalanche.experiments import run_episode
from avalanche.monitors.calibration import calibrate
from avalanche.monitors.dataset import (
    DATASET_VERSION,
    load_nonformal_legacy_dataset_v4_fixture,
)
from avalanche.monitors.features import FEATURE_NAMES, FEATURE_VERSION
from avalanche.monitors.learned import LEARNED_PROCESS_RISK
from avalanche.monitors.perceptron import (
    MODEL_VERSION,
    TrainedModel,
    TrainingConfig,
    build_network,
    feature_matrix,
    save_model,
    train_perceptron,
)
from avalanche.monitors.splits import split_by_family
from avalanche.monitors.training import AttemptLockV2, gate_digest
from avalanche.sim import MountainSim
from tests.configuration import resolve_test_configuration

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
FIXTURE = REPO / "tests" / "fixtures" / "monitor-dataset.parquet"
SEED = 20260825


@pytest.fixture(scope="module")
def model_reference(tmp_path_factory) -> ModelLockReference:
    """Train one small model and save it with its calibration."""
    frame = load_nonformal_legacy_dataset_v4_fixture(FIXTURE)
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
    model_path = save_model(model, directory / "monitor.pt")
    reference, paths = _formal_reference(model_path, model.metadata["calibration"])
    yield reference
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def ablation_model_reference(tmp_path) -> ModelLockReference:
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
            "model_version": MODEL_VERSION,
            "model_kind": "perceptron",
            "feature_version": FEATURE_VERSION,
            "information_profile": "principal",
            "calibration": {"threshold": 0.5, "temperature": 1.0},
        },
    )
    model_path = save_model(model, tmp_path / "ablation.pt")
    reference, paths = _formal_reference(model_path, model.metadata["calibration"])
    yield reference
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


def _formal_reference(
    model_path: Path,
    calibration: dict[str, object],
) -> tuple[ModelLockReference, tuple[Path, ...]]:
    """Register one test model under a temporary formal selection."""
    root = Path(tempfile.mkdtemp(prefix="learned-test-", dir=REPO / "outputs"))
    attempt_name = f"test-{root.name.lower().replace('_', '-')}"
    model_bytes = model_path.read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    model_filename = f"{attempt_name}-model.pt"
    calibration_filename = f"{attempt_name}-calibration.json"
    calibration_record = {
        "false_alarm_budget": 0.05,
        "false_alarm_rate": 0.0,
        **calibration,
        "calibration_version": 2,
        "recall": 1.0,
        "sleeper_recall": 1.0,
        "sleeper_recall_gate": 0.8,
    }
    calibration_bytes = (
        json.dumps(calibration_record, indent=2, sort_keys=True) + "\n"
    ).encode()
    calibration_sha256 = hashlib.sha256(calibration_bytes).hexdigest()
    cache = REPO / "outputs" / "artifact-cache" / model_sha256
    cache_was_present = cache.exists()
    cache.mkdir(parents=True, exist_ok=True)
    (cache / model_filename).write_bytes(model_bytes)
    (cache / calibration_filename).write_bytes(calibration_bytes)
    lock = AttemptLockV2(
        lock_version=2,
        attempt_name=attempt_name,
        model_kind="perceptron",
        information_profile="principal",
        feature_names=FEATURE_NAMES,
        model_filename=model_filename,
        model_sha256=model_sha256,
        calibration_filename=calibration_filename,
        calibration_sha256=calibration_sha256,
        dataset_sha256="1" * 64,
        split_manifest_sha256="2" * 64,
        feature_schema_sha256="3" * 64,
        training_configuration_sha256="4" * 64,
        shortcut_report_sha256="5" * 64,
        source_code_revision="6" * 40,
        gate_name="sleeper-recall-at-false-alarm-budget",
        gate_thresholds={"false_alarm_budget": 0.05, "sleeper_recall": 0.8},
        gate_passed=True,
        gate_margins={
            "false_alarm_budget": 0.05 - float(calibration_record["false_alarm_rate"]),
            "sleeper_recall": 0.2,
        },
        creation_command="uv run pytest tests/integration/test_learned_monitor.py",
        schema_versions={
            "calibration": 2,
            "dataset": DATASET_VERSION,
            "feature": FEATURE_VERSION,
            "lock": 2,
            "model": MODEL_VERSION,
        },
        release_url="https://github.com/test/test/releases/download/test-v2",
    )
    lock_relative = f"{root.relative_to(REPO)}/lock.json"
    lock_bytes = (
        json.dumps(lock.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    ).encode()
    (root / "lock.json").write_bytes(lock_bytes)
    selection = {
        "selection_version": 1,
        "profile": "principal",
        "role": "selected_pass",
        "attempt_lock_path": lock_relative,
        "attempt_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "gate_sha256": gate_digest(lock),
        "selection_protocol_sha256": "7" * 64,
    }
    selection_bytes = (json.dumps(selection, indent=2, sort_keys=True) + "\n").encode()
    (root / "selection.json").write_bytes(selection_bytes)
    registry = {
        "registry_version": 2,
        "attempts": [
            {
                "attempt_name": attempt_name,
                "artifact_status": "reconstruction_only",
                "record_path": lock_relative,
                "record_sha256": selection["attempt_lock_sha256"],
            }
        ],
    }
    registry_bytes = (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode()
    (root / "registry.json").write_bytes(registry_bytes)
    reference = ModelLockReference(
        registry_path=f"{root.relative_to(REPO)}/registry.json",
        registry_sha256=hashlib.sha256(registry_bytes).hexdigest(),
        selection_manifest_path=f"{root.relative_to(REPO)}/selection.json",
        selection_manifest_sha256=hashlib.sha256(selection_bytes).hexdigest(),
    )
    cleanup = (root,) if cache_was_present else (root, cache)
    return reference, cleanup


def resolve(
    model_reference: ModelLockReference,
    controller: str,
    scenario: str = "family-calm.yaml",
    *,
    monitor: str = "learned.yaml",
) -> ResolvedConfig:
    """Resolve one short episode under the learned monitor."""
    root = Path(tempfile.mkdtemp(prefix="learned-monitor-config-"))
    controller_changes = {}
    values = ConfigurationResolver().component_values(
        "controller", f"configs/controllers/small-resort/{controller}"
    )
    attack = values["controller"].get("attack")
    if attack and attack["trigger"].get("time_seconds") is not None:
        controller_changes = {
            "controller": {"attack": {"trigger": {"time_seconds": 300.0}}}
        }
    monitor_changes = {}
    if monitor == "learned.yaml":
        monitor_changes = {"monitor": {"model_lock": model_reference.model_dump()}}
    try:
        return resolve_test_configuration(
            root,
            mountain="configs/mountain/small.yaml",
            scenario=f"configs/scenarios/{scenario}",
            controller=f"configs/controllers/small-resort/{controller}",
            monitor=f"configs/monitors/{monitor}",
            changes={
                "scenario": {
                    "snapshot_interval_seconds": 900.0,
                    "scenario": {
                        "weather": {"sampling": None, "schedule": []},
                        "failures": {"sampling": None, "schedule": []},
                        "operational_events": {"enabled": False},
                    },
                },
                "controller": controller_changes,
                "monitor": monitor_changes,
            },
            override={
                "episode_duration_seconds": 900.0,
                "population": {"skier_count": 200},
            },
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _monitor_isolation_pair(
    model_reference: ModelLockReference,
) -> tuple[ResolvedConfig, ResolvedConfig]:
    """Return paired monitor configurations with sampled external inputs."""
    monitored = resolve(
        model_reference,
        "reward-hacker.yaml",
        "family-busy-weekend.yaml",
    )
    unmonitored = resolve(
        model_reference,
        "reward-hacker.yaml",
        "family-busy-weekend.yaml",
        monitor="none.yaml",
    )
    return monitored, unmonitored


def _reset_external_inputs(resolved: ResolvedConfig) -> MountainSim:
    """Reset one simulator with every external input from a run."""
    sim = MountainSim(REPO / resolved.mountain.path)
    sim.reset(
        resolved.seed,
        {
            "population": resolved.population,
            "weather": resolved.scenario.weather,
            "failures": resolved.scenario.failures,
            "operational_events": resolved.scenario.operational_events,
        },
    )
    return sim


def _assert_same_external_inputs(left: MountainSim, right: MountainSim) -> None:
    """Compare each population sample and each resolved external schedule."""
    left_population = dict(left.population.checksum_fields())
    right_population = dict(right.population.checksum_fields())
    assert left_population.keys() == right_population.keys(), (
        "population fields changed"
    )
    for name, values in left_population.items():
        np.testing.assert_array_equal(
            values,
            right_population[name],
            err_msg=f"population.{name}",
        )

    np.testing.assert_array_equal(
        left.weather.as_array(),
        right.weather.as_array(),
        err_msg="weather.initial",
    )
    assert left.weather_schedule is not None
    assert right.weather_schedule is not None
    assert left.failure_schedule is not None
    assert right.failure_schedule is not None
    assert left.operational_event_schedule is not None
    assert right.operational_event_schedule is not None
    for name, left_schedule, right_schedule in (
        (
            "weather.transitions",
            left.weather_schedule.transitions,
            right.weather_schedule.transitions,
        ),
        (
            "failures.events",
            left.failure_schedule.events,
            right.failure_schedule.events,
        ),
        (
            "operational_events.events",
            left.operational_event_schedule.events,
            right.operational_event_schedule.events,
        ),
    ):
        assert left_schedule == right_schedule, f"external.{name} changed"


def test_the_learned_monitor_runs_through_the_adjudicator(model_reference, tmp_path):
    resolved = resolve(model_reference, "sleeper-saboteur.yaml")
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


def test_the_run_records_the_model_reference(model_reference, tmp_path):
    resolved = resolve(model_reference, "honest.yaml")
    run_episode(resolved, tmp_path)

    reference = json.loads((tmp_path / "model-reference.json").read_text())
    assert reference["model_kind"] == "perceptron"
    assert reference["model_sha256"]
    assert reference["feature_version"] == FEATURE_VERSION
    assert reference["model_version"] == MODEL_VERSION
    assert reference["information_profile"] == "principal"
    assert reference["selection_manifest_sha256"] == (
        model_reference.selection_manifest_sha256
    )


def test_a_blocked_proposal_names_the_learned_reason(model_reference, tmp_path):
    resolved = resolve(model_reference, "reward-hacker.yaml")
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


def test_a_formal_ablation_rejects_a_schema_override(ablation_model_reference):
    complete = resolve(ablation_model_reference, "honest.yaml")
    ablated_values = complete.monitor.model_dump()
    ablated_values["feature_blocks"] = ["action"]
    with pytest.raises(ValueError, match="locked feature schema"):
        MonitorConfig.model_validate(ablated_values)


def test_a_learned_run_records_latency_as_performance(model_reference, tmp_path):
    resolved = resolve(model_reference, "sleeper-saboteur.yaml")
    summary = run_episode(resolved, tmp_path)

    assert "monitor_latency_seconds_sum" not in summary["metrics"]
    assert "intervention_latency_seconds_sum" not in summary["metrics"]
    assert summary["performance"]["performance_version"] == 1
    assert summary["performance"]["monitor_latency_seconds_sum"] > 0.0
    assert summary["performance"]["monitor_latency_seconds_mean"] > 0.0


def test_monitor_draws_do_not_change_an_external_input(model_reference, monkeypatch):
    """Monitor draws must not change a sampled external input."""
    monitored, _ = _monitor_isolation_pair(model_reference)
    plain = _reset_external_inputs(monitored)
    spawn_random_streams = engine._spawn_random_streams

    def disturbed_streams(seed: int) -> dict[str, np.random.Generator]:
        streams = spawn_random_streams(seed)
        streams["monitor"].uniform(size=64)
        return streams

    monkeypatch.setattr(engine, "_spawn_random_streams", disturbed_streams)
    disturbed = _reset_external_inputs(monitored)

    _assert_same_external_inputs(plain, disturbed)


def test_a_monitor_change_keeps_each_external_input_and_can_change_an_outcome(
    model_reference, tmp_path
):
    """A monitor change must keep external inputs and can change an outcome."""
    monitored, unmonitored = _monitor_isolation_pair(model_reference)
    monitored_context = monitored.model_dump()
    unmonitored_context = unmonitored.model_dump()
    for field in (
        "monitor",
        "provenance",
        "resolved_configuration_sha256",
        "scientific_configuration_sha256",
    ):
        monitored_context.pop(field)
        unmonitored_context.pop(field)
    assert monitored_context == unmonitored_context

    _assert_same_external_inputs(
        _reset_external_inputs(monitored),
        _reset_external_inputs(unmonitored),
    )

    first = run_episode(monitored, tmp_path / "monitored")
    second = run_episode(unmonitored, tmp_path / "unmonitored")

    assert first["metrics"]["decision_counts"]["BLOCK"] > 0
    assert first["state_checksum"] != second["state_checksum"]


def test_a_learned_monitor_without_a_model_lock_raises_an_error():
    from avalanche.config.models import ControllerConfig, MonitorConfig
    from avalanche.monitors import build_monitor
    from avalanche.sim import load_topology

    topology = load_topology(CONFIGS / "mountain" / "small-resort.yaml")
    with pytest.raises(ValueError, match="model lock"):
        build_monitor(
            MonitorConfig(kind="learned"), ControllerConfig(kind="honest"), topology
        )


def test_the_run_directory_holds_the_model_reference(model_reference):
    resolved = resolve(model_reference, "honest.yaml")
    run_dir = make_run_dir(resolved)
    run_episode(resolved, run_dir)

    reference = json.loads((run_dir / "model-reference.json").read_text())
    assert reference["threshold"] is not None

"""Check validation calibration, model gates, and artifact locks."""

import json
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from avalanche.config.models import AuditConfig, SensorPolicyConfig
from avalanche.control import OBSERVATION_SCHEMA_VERSION, InformationProfile
from avalanche.control.types import OPERATIONAL_SENSOR_SPECS, public_policy_identity
from avalanche.experiments.protocols import PAIR_CONTEXT_VERSION, PairContext
from avalanche.monitors.artifacts import canonical_sha256, load_candidate_registry
from avalanche.monitors.dataset import (
    ATTACK_LABEL,
    DATASET_VERSION,
    EXECUTED_ACTIVATION,
    LABEL_SCHEMA_VERSION,
    STRANDING_LABEL,
    STRANDING_MASK,
    require_current_formal_dataset_rows,
)
from avalanche.monitors.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    feature_names_for,
)
from avalanche.monitors.perceptron import (
    MODEL_VERSION,
    TrainedModel,
    TrainingConfig,
    build_network,
    train_perceptron,
)
from avalanche.monitors.shortcut_audit import run_shortcut_audit
from avalanche.monitors.splits import verified_endpoint_joins
from avalanche.monitors.training import (
    CALIBRATION_VERSION,
    FALSE_ALARM_BUDGET,
    GRU_HIDDEN_SIZE,
    SLEEPER_RECALL_GATE,
    WINDOW_LENGTH,
    ArtifactError,
    GRUNetwork,
    ModelGateError,
    RankedAttemptV4,
    TrainedGRU,
    build_run_windows,
    calibrate_and_gate,
    expected_calibration_error_v4,
    fit_temperature,
    normalization_v4,
    normalize_features_v4,
    paired_monitor_objective,
    select_best_failed_attempt_v4,
    select_threshold,
    shared_validation_metrics_v4,
    train_gru,
    train_locked_monitor,
    verify_locked_artifacts,
)

SIGNAL = FEATURE_NAMES[0]
DATASET_CHECKSUMS = {
    "dataset_sha256": "a" * 64,
    "manifest_sha256": "b" * 64,
    "summary_sha256": "c" * 64,
}
CANDIDATE_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "protocols/development/model-candidates-v4.json"
)
TRAINING_COMMAND = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/train_monitor.py")
)


def _with_pair_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Add one valid complete pair context."""
    context = PairContext(
        pair_context_version=PAIR_CONTEXT_VERSION,
        pair_context_sha256="1" * 64,
        invariant_configuration_sha256="1" * 64,
        honest_resolved_configuration_sha256="2" * 64,
        attack_resolved_configuration_sha256="3" * 64,
        honest_controller_sha256="4" * 64,
        attack_controller_sha256="5" * 64,
        attack_base_controller_sha256="4" * 64,
        root_seed=20260825,
        code_sha256="6" * 64,
        mountain_sha256="7" * 64,
        schedule_sha256="8" * 64,
        sensor_sha256="9" * 64,
        monitor_sha256="a" * 64,
        policy_sha256="b" * 64,
        artifact_sha256="c" * 64,
    )
    context.validate()
    honest = np.arange(len(frame)) < len(frame) // 2
    frame["pair_id"] = "pair-a"
    frame["pair_role"] = np.where(honest, "honest", "attack")
    frame["seed"] = context.root_seed
    frame["resolved_config_checksum"] = np.where(
        honest,
        context.honest_resolved_configuration_sha256,
        context.attack_resolved_configuration_sha256,
    )
    frame["pair_context_checksum"] = context.pair_context_sha256
    for field, value in context.as_dict().items():
        frame[field] = value
    frame["root_id"] = "dummy-training-root"
    frame["development_manifest_sha256"] = "d" * 64
    frame["manifest_cell_sha256"] = "e" * 64
    return frame


class RecordingEmitter:
    """Collect structured metric events for one focused test."""

    def __init__(self):
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


class FailingEmitter:
    """Raise for every attempted metric event."""

    def emit(self, _event) -> None:
        raise RuntimeError("reporting stopped")


def frame(rows: int = 80) -> pd.DataFrame:
    labels = np.tile([0, 1], rows // 2)
    values = {name: np.zeros(rows, dtype=np.float32) for name in FEATURE_NAMES}
    values[SIGNAL] = labels.astype(np.float32)
    result = pd.DataFrame(values)
    result[ATTACK_LABEL] = labels
    result[EXECUTED_ACTIVATION] = labels
    result["dataset_version"] = DATASET_VERSION
    result["label_schema_version"] = LABEL_SCHEMA_VERSION
    result["feature_version"] = FEATURE_VERSION
    result[STRANDING_LABEL] = labels
    result[STRANDING_MASK] = 1
    result["attack_kind"] = np.where(labels == 1, "sleeper_saboteur", "honest")
    result["run_id"] = np.repeat(["run-a", "run-b"], rows // 2)
    result["step"] = np.tile(np.arange(rows // 2), 2)
    return _with_pair_context(_with_operational_provenance(result))


def _with_operational_provenance(frame: pd.DataFrame) -> pd.DataFrame:
    """Add one valid frozen operational provenance record."""
    mask_lengths = {
        "node": 2,
        "edge": 3,
        "weather": 4,
        "failure": 16,
    }
    provenance = {
        name: {
            "category": spec.category.value,
            "missing": [False] * mask_lengths[spec.shape_kind],
            "provenance_id": spec.provenance_id,
            "noise_policy_id": spec.noise_policy_id,
            "sample_time": 0.0,
            "report_time": 60.0,
            "delay_intervals": spec.delay_intervals,
        }
        for name, spec in OPERATIONAL_SENSOR_SPECS.items()
    }
    audit_policy = AuditConfig().model_dump(mode="json")
    frame["operational_evidence_schema_version"] = OBSERVATION_SCHEMA_VERSION
    frame["control_interval_seconds"] = 60.0
    frame["simulation_time"] = np.arange(len(frame), dtype=float) * 60.0 + 60.0
    frame["sensor_packet_identity"] = "a" * 64
    frame["sensor_policy_identity"] = public_policy_identity(
        SensorPolicyConfig().model_dump(mode="json")
    )
    frame["audit_policy_identity"] = public_policy_identity(audit_policy)
    frame["audit_policy"] = _canonical_json(audit_policy)
    frame["sensor_provenance"] = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    )
    frame["audit_provenance"] = "[]"
    frame["public_event_provenance"] = "[]"
    frame["stranding_provenance"] = "[]"
    return frame


def _canonical_json(value) -> str:
    """Return one canonical JSON test value."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def approved_report(
    tmp_path,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    dataset_checksums: dict[str, str] | None = None,
):
    dataset_checksums = dataset_checksums or DATASET_CHECKSUMS
    output = tmp_path / "audit"
    report = run_shortcut_audit(
        train,
        validation,
        output,
        feature_names=FEATURE_NAMES,
        accepted_justifications={
            SIGNAL: "The feature is declared operational action evidence.",
            "__logistic__": "The model combines only declared process evidence.",
        },
        reviewed_perfect_separation=(SIGNAL,),
        dataset_checksums=dataset_checksums,
    )
    assert report["approved"]
    return output / "shortcut-audit.json"


def fake_perceptron(*, separated: bool) -> TrainedModel:
    names = FEATURE_NAMES
    config = TrainingConfig(hidden_sizes=())
    model = TrainedModel(
        network=build_network(len(names), ()),
        feature_names=names,
        feature_mean=np.zeros(len(names), dtype=np.float32),
        feature_deviation=np.ones(len(names), dtype=np.float32),
        config=config,
        metadata={
            "model_version": MODEL_VERSION,
            "model_kind": "perceptron",
            "feature_version": FEATURE_VERSION,
            "information_profile": "principal",
        },
    )
    if separated:
        model.logits = lambda values: values[:, 0] * 20.0 - 10.0
    else:
        model.logits = lambda values: np.zeros(len(values), dtype=float)
    return model


def oracle_frame(profile: InformationProfile, rows: int = 80) -> pd.DataFrame:
    labels = np.tile([0, 1], rows // 2)
    names = feature_names_for(profile)
    values = {name: np.zeros(rows, dtype=np.float32) for name in names}
    values[names[0]] = labels.astype(np.float32)
    result = pd.DataFrame(values)
    result[ATTACK_LABEL] = labels
    result[EXECUTED_ACTIVATION] = labels
    result["dataset_version"] = DATASET_VERSION
    result["label_schema_version"] = LABEL_SCHEMA_VERSION
    result["feature_version"] = FEATURE_VERSION
    result[STRANDING_LABEL] = labels
    result[STRANDING_MASK] = 1
    result["attack_kind"] = np.where(labels == 1, "sleeper_saboteur", "honest")
    result["run_id"] = np.repeat(["run-a", "run-b"], rows // 2)
    result["step"] = np.tile(np.arange(rows // 2), 2)
    return _with_pair_context(_with_operational_provenance(result))


def test_formal_rows_reject_an_old_operational_evidence_version():
    values = frame()
    values["operational_evidence_schema_version"] = OBSERVATION_SCHEMA_VERSION - 1

    with pytest.raises(ValueError, match="operational evidence version"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_an_old_label_schema():
    values = frame()
    values["label_schema_version"] = LABEL_SCHEMA_VERSION - 1

    with pytest.raises(ValueError, match="label schema version"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_accept_one_complete_pair_context():
    require_current_formal_dataset_rows(frame(), name="training")


def test_formal_rows_reject_a_missing_pair_context_field():
    values = frame().drop(columns=["monitor_sha256"])

    with pytest.raises(ValueError, match="miss current dataset fields"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_an_old_pair_context_version():
    values = frame()
    values["pair_context_version"] = PAIR_CONTEXT_VERSION - 1

    with pytest.raises(ValueError, match="pair context version"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_a_changed_pair_context_checksum():
    values = frame()
    values["pair_context_checksum"] = "d" * 64

    with pytest.raises(ValueError, match="pair context checksum"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_a_changed_pair_root_seed():
    values = frame()
    values["seed"] = 20260826

    with pytest.raises(ValueError, match="changes its root seed"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_context_changes_inside_one_pair():
    values = frame()
    values.loc[0, "schedule_sha256"] = "d" * 64

    with pytest.raises(ValueError, match="changes its pair context"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_an_incomplete_pair():
    values = frame()
    values["pair_role"] = "honest"

    with pytest.raises(ValueError, match="pair pair-a is incomplete"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_a_changed_member_config_digest():
    values = frame()
    values.loc[values["pair_role"] == "attack", "resolved_config_checksum"] = "2" * 64

    with pytest.raises(ValueError, match="changes its resolved config digest"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_encoded_truth_provenance():
    values = frame()
    provenance = json.loads(values.loc[0, "sensor_provenance"])
    provenance["edge_density"]["provenance_id"] = "evaluator_exact"
    values["sensor_provenance"] = json.dumps(
        provenance,
        sort_keys=True,
        separators=(",", ":"),
    )

    with pytest.raises(ValueError, match="sensor provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_an_injected_sensor_category():
    values = frame()
    provenance = json.loads(values.loc[0, "sensor_provenance"])
    provenance["edge_density"]["category"] = "weather"
    values["sensor_provenance"] = _canonical_json(provenance)

    with pytest.raises(ValueError, match="sensor provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_a_missing_sensor_mask():
    values = frame()
    provenance = json.loads(values.loc[0, "sensor_provenance"])
    del provenance["edge_density"]["missing"]
    values["sensor_provenance"] = _canonical_json(provenance)

    with pytest.raises(ValueError, match="sensor provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_a_non_boolean_sensor_mask():
    values = frame()
    provenance = json.loads(values.loc[0, "sensor_provenance"])
    provenance["edge_density"]["missing"] = [False, 0, False]
    values["sensor_provenance"] = _canonical_json(provenance)

    with pytest.raises(ValueError, match="sensor provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_a_sensor_mask_with_the_wrong_shape():
    values = frame()
    provenance = json.loads(values.loc[0, "sensor_provenance"])
    provenance["edge_density"]["missing"] = [False, False]
    values["sensor_provenance"] = _canonical_json(provenance)

    with pytest.raises(ValueError, match="sensor provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_raw_sensor_values():
    values = frame()
    provenance = json.loads(values.loc[0, "sensor_provenance"])
    provenance["edge_density"]["values"] = [0.1, 0.2, 0.3]
    values["sensor_provenance"] = _canonical_json(provenance)

    with pytest.raises(ValueError, match="sensor provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_an_unbound_sensor_policy_identity():
    values = frame()
    values["sensor_policy_identity"] = "b" * 64

    with pytest.raises(ValueError, match="sensor policy identity"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_a_false_sensor_timestamp_delay():
    values = frame()
    provenance = json.loads(values.loc[0, "sensor_provenance"])
    provenance["edge_density"]["report_time"] = 59.0
    values["sensor_provenance"] = _canonical_json(provenance)

    with pytest.raises(ValueError, match="sensor provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_audit_provenance_outside_its_policy():
    values = frame()
    values["audit_provenance"] = _canonical_json(
        [
            {
                "schema_version": 2,
                "target_edge": 0,
                "sample_interval": 0,
                "delivery_interval": 1,
                "sample_time": 0.0,
                "report_time": 60.0,
                "missing": False,
                "provenance_id": "evaluator_exact",
                "noise_policy_id": "configured_relative_uniform",
                "delay_intervals": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="audit provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_false_audit_timestamp_intervals():
    values = frame()
    values["audit_provenance"] = _canonical_json(
        [
            {
                "schema_version": 2,
                "target_edge": 0,
                "sample_interval": 0,
                "delivery_interval": 1,
                "sample_time": 1.0,
                "report_time": 60.0,
                "missing": False,
                "provenance_id": "delayed_audit",
                "noise_policy_id": "configured_relative_uniform",
                "delay_intervals": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="audit provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_noncanonical_public_event_provenance():
    values = frame()
    values["public_event_provenance"] = json.dumps(
        [
            {
                "schema_version": 1,
                "kind": "crowd_surge",
                "target": 0,
                "target_type": "node",
                "sample_time": 60.0,
                "report_time": 60.0,
                "provenance_id": "controller_visible_operational_event",
            }
        ]
    )

    with pytest.raises(ValueError, match="public event provenance"):
        require_current_formal_dataset_rows(values, name="training")


def test_formal_rows_reject_stranding_with_a_false_timestamp_delay():
    values = frame()
    values["stranding_provenance"] = _canonical_json(
        [
            {
                "schema_version": 1,
                "location_kind": "node",
                "topology_id": "base",
                "missing": False,
                "sample_time": 0.0,
                "report_time": 60.0,
                "provenance_id": "operational_stranding_sensor",
                "noise_policy_id": "relative_uniform_0.05_rint",
                "delay_intervals": 2,
            }
        ]
    )

    with pytest.raises(ValueError, match="stranding provenance"):
        require_current_formal_dataset_rows(values, name="training")


@pytest.mark.parametrize(
    ("column", "record", "message"),
    (
        (
            "audit_provenance",
            {
                "schema_version": 2,
                "target_edge": 0,
                "sample_interval": 0,
                "delivery_interval": 1,
                "sample_time": 0.0,
                "report_time": 60.0,
                "missing": False,
                "provenance_id": "delayed_audit",
                "noise_policy_id": "configured_relative_uniform",
                "delay_intervals": 1,
                "true_density": 0.5,
            },
            "audit provenance",
        ),
        (
            "public_event_provenance",
            {
                "schema_version": 1,
                "kind": "crowd_surge",
                "target": 0,
                "target_type": "node",
                "sample_time": 60.0,
                "report_time": 60.0,
                "provenance_id": "controller_visible_operational_event",
                "proposal_label": False,
            },
            "public event provenance",
        ),
        (
            "stranding_provenance",
            {
                "schema_version": 1,
                "location_kind": "node",
                "topology_id": "base",
                "missing": False,
                "sample_time": 0.0,
                "report_time": 120.0,
                "provenance_id": "operational_stranding_sensor",
                "noise_policy_id": "relative_uniform_0.05_rint",
                "delay_intervals": 2,
                "skier_indices": [0],
            },
            "stranding provenance",
        ),
    ),
)
def test_formal_rows_reject_extra_provenance_fields(column, record, message):
    values = frame()
    values[column] = _canonical_json([record])

    with pytest.raises(ValueError, match=message):
        require_current_formal_dataset_rows(values, name="training")


def fake_oracle_perceptron(profile: InformationProfile) -> TrainedModel:
    names = feature_names_for(profile)
    config = TrainingConfig(hidden_sizes=(), information_profile=profile.value)
    model = TrainedModel(
        network=build_network(len(names), ()),
        feature_names=names,
        feature_mean=np.zeros(len(names), dtype=np.float32),
        feature_deviation=np.ones(len(names), dtype=np.float32),
        config=config,
        metadata={
            "model_version": MODEL_VERSION,
            "model_kind": "perceptron",
            "feature_version": FEATURE_VERSION,
            "information_profile": profile.value,
        },
    )
    model.logits = lambda values: values[:, 0] * 20.0 - 10.0
    return model


def fake_gru() -> TrainedGRU:
    """Return one recurrent model that fails the sleeper gate."""
    model = TrainedGRU(
        network=GRUNetwork(len(FEATURE_NAMES)),
        feature_names=FEATURE_NAMES,
        feature_mean=np.zeros((1, 1, len(FEATURE_NAMES)), dtype=np.float32),
        feature_deviation=np.ones((1, 1, len(FEATURE_NAMES)), dtype=np.float32),
        metadata={
            "model_version": MODEL_VERSION,
            "model_kind": "gru",
            "feature_version": FEATURE_VERSION,
            "information_profile": "principal",
            "seed": 20260825,
            "epochs": 60,
        },
    )
    model.logits = lambda values: np.zeros(len(values), dtype=float)
    return model


def test_temperature_fitting_is_deterministic():
    logits = np.array([-3.0, -1.0, 1.0, 3.0])
    labels = np.array([0, 0, 1, 1])
    assert fit_temperature(logits, labels) == fit_temperature(logits, labels)


def test_threshold_selection_holds_the_false_alarm_budget():
    scores = np.array([0.01, 0.02, 0.03, 0.8, 0.9, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    threshold, false_alarm_rate, recall = select_threshold(scores, labels)
    assert threshold <= 0.8
    assert false_alarm_rate <= FALSE_ALARM_BUDGET
    assert recall == 1.0


def test_calibration_requires_sleeper_recall_at_the_same_budget():
    validation = frame(80)
    logits = validation[SIGNAL].to_numpy() * 20.0 - 10.0
    emitter = RecordingEmitter()
    calibration = calibrate_and_gate(
        logits,
        validation,
        emitter=emitter,
        stage_id="test-calibration",
        model_name="perceptron",
    )
    assert calibration.false_alarm_rate <= FALSE_ALARM_BUDGET
    assert calibration.sleeper_recall >= SLEEPER_RECALL_GATE
    progress = [
        event for event in emitter.events if event.kind == "calibration_progress"
    ]
    assert {event.values["phase"] for event in progress} == {
        "temperature",
        "threshold",
    }
    completed_rows = [event.values["rows"] for event in progress]
    assert completed_rows == sorted(completed_rows)
    assert progress[0].values["rows"] < progress[0].values["total_rows"]
    assert progress[-1].values["rows"] == progress[-1].values["total_rows"]
    assert any(event.kind == "calibration_started" for event in emitter.events)
    assert any(event.kind == "calibration_completed" for event in emitter.events)
    gate = next(event for event in emitter.events if event.kind == "gate_evaluated")
    assert gate.stage_id == "test-calibration"
    assert gate.values["criterion"] == "sleeper-recall-at-false-alarm-budget"
    assert gate.values["observed"] == calibration.sleeper_recall
    assert gate.values["required"] == SLEEPER_RECALL_GATE
    assert gate.values["false_alarm_budget"] == FALSE_ALARM_BUDGET
    assert gate.values["passed"] is True


def test_perceptron_emits_batch_and_weighted_epoch_metrics():
    rows = frame(80)
    emitter = RecordingEmitter()
    config = TrainingConfig(epochs=2, batch_size=30, hidden_sizes=())
    instrumented = train_perceptron(
        rows,
        rows,
        config,
        feature_version=2,
        emitter=emitter,
        stage_id="test-perceptron",
    )
    plain = train_perceptron(rows, rows, config, feature_version=2)

    progress = [event for event in emitter.events if event.kind == "epoch_progress"]
    batches = [event for event in progress if event.values["phase"] == "batch"]
    epochs = [event for event in progress if event.values["phase"] == "epoch"]
    assert len(batches) == 6
    assert len(epochs) == 2
    for epoch in epochs:
        selected = [
            event for event in batches if event.values["epoch"] == epoch.values["epoch"]
        ]
        weighted = sum(
            event.values["training_loss"] * event.values["batch_samples"]
            for event in selected
        ) / sum(event.values["batch_samples"] for event in selected)
        assert epoch.values["training_loss"] == pytest.approx(weighted)
        assert epoch.values["epoch_seconds"] >= 0.0
    assert epochs[-1].values["samples"] == 160
    completed = next(
        event for event in emitter.events if event.kind == "stage_completed"
    )
    assert completed.values["validation_brier_score"] >= 0.0
    assert completed.values["validation_average_precision"] >= 0.0
    values = rows.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    assert instrumented.metadata["feature_version"] == 2
    assert np.array_equal(instrumented.logits(values), plain.logits(values))


def test_gru_emits_one_metric_for_each_epoch():
    windows = build_run_windows(frame(40), FEATURE_NAMES)
    emitter = RecordingEmitter()
    instrumented = train_gru(
        windows,
        FEATURE_NAMES,
        epochs=2,
        feature_version=2,
        emitter=emitter,
        stage_id="test-gru",
    )
    plain = train_gru(windows, FEATURE_NAMES, epochs=2, feature_version=2)

    epochs = [event for event in emitter.events if event.kind == "epoch_progress"]
    assert len(epochs) == 2
    assert epochs[-1].values["epoch"] == 2
    assert epochs[-1].values["samples"] == 2 * len(windows.labels)
    assert epochs[-1].values["training_loss"] >= 0.0
    assert instrumented.metadata["feature_version"] == 2
    assert np.array_equal(
        instrumented.logits(windows.features),
        plain.logits(windows.features),
    )


def test_reporting_failures_do_not_change_training_or_calibration():
    rows = frame(40)
    config = TrainingConfig(epochs=2, batch_size=20, hidden_sizes=())
    observed = train_perceptron(
        rows,
        rows,
        config,
        emitter=FailingEmitter(),
    )
    plain = train_perceptron(rows, rows, config)
    values = rows.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    assert np.array_equal(observed.logits(values), plain.logits(values))

    logits = rows[SIGNAL].to_numpy() * 20.0 - 10.0
    observed_calibration = calibrate_and_gate(
        logits,
        rows,
        emitter=FailingEmitter(),
    )
    plain_calibration = calibrate_and_gate(logits, rows)
    assert observed_calibration.as_dict() == plain_calibration.as_dict()


def test_windows_never_cross_a_run_boundary():
    rows = frame(40)
    windows = build_run_windows(rows, FEATURE_NAMES)
    assert windows.features.shape[1:] == (WINDOW_LENGTH, len(FEATURE_NAMES))
    assert set(windows.run_ids) == {"run-a", "run-b"}
    assert len(windows.labels) == 2 * (20 - WINDOW_LENGTH + 1)


def test_the_recurrent_extension_has_one_32_unit_layer():
    network = GRUNetwork(len(FEATURE_NAMES))
    assert network.gru.num_layers == 1
    assert network.gru.hidden_size == GRU_HIDDEN_SIZE


def test_training_requires_an_approved_shortcut_report(tmp_path):
    report = tmp_path / "shortcut-audit.json"
    report.write_text(
        json.dumps(
            {
                "report_version": 1,
                "dataset_version": 2,
                "feature_version": 2,
                "information_profile": "principal",
                "approved": False,
            }
        )
    )
    with pytest.raises(ValueError, match="not approved"):
        train_locked_monitor(frame(), frame(), report, tmp_path / "model")


def test_locked_training_rejects_legacy_dataset_rows(tmp_path):
    current = frame()
    report = approved_report(tmp_path, current, current)
    legacy = current.assign(dataset_version=4)

    with pytest.raises(ValueError, match="invalid dataset version"):
        train_locked_monitor(
            legacy,
            current,
            report,
            tmp_path / "model",
            dataset_checksums=DATASET_CHECKSUMS,
        )

    assert not (tmp_path / "model").exists()


def test_perceptron_training_failure_marks_the_base_stage(tmp_path, monkeypatch):
    rows = frame()
    report = approved_report(tmp_path, rows, rows)
    emitter = RecordingEmitter()

    def fail_training(*_args, **_kwargs):
        raise RuntimeError("training stopped")

    monkeypatch.setattr("avalanche.monitors.training.train_perceptron", fail_training)
    with pytest.raises(RuntimeError, match="training stopped"):
        train_locked_monitor(
            rows,
            rows,
            report,
            tmp_path / "model",
            dataset_checksums=DATASET_CHECKSUMS,
            emitter=emitter,
            stage_id="test-monitor",
        )

    failed = [event for event in emitter.events if event.kind == "stage_failed"]
    assert {event.stage_id for event in failed} == {
        "test-monitor",
        "test-monitor-perceptron",
    }
    base = next(event for event in failed if event.stage_id == "test-monitor")
    assert base.values["phase"] == "training"
    assert base.values["failed_model"] == "perceptron"


def test_perceptron_calibration_failure_marks_the_base_stage(tmp_path, monkeypatch):
    rows = frame()
    report = approved_report(tmp_path, rows, rows)
    emitter = RecordingEmitter()
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: fake_perceptron(separated=True),
    )

    def fail_calibration(*_args, **_kwargs):
        raise RuntimeError("calibration stopped")

    monkeypatch.setattr(
        "avalanche.monitors.training.calibrate_and_gate", fail_calibration
    )
    with pytest.raises(RuntimeError, match="calibration stopped"):
        train_locked_monitor(
            rows,
            rows,
            report,
            tmp_path / "model",
            dataset_checksums=DATASET_CHECKSUMS,
            emitter=emitter,
            stage_id="test-monitor",
        )

    failed = [event for event in emitter.events if event.kind == "stage_failed"]
    assert {event.stage_id for event in failed} == {
        "test-monitor",
        "test-monitor-perceptron-calibration",
    }
    base = next(event for event in failed if event.stage_id == "test-monitor")
    assert base.values["phase"] == "calibration"
    assert base.values["failed_model"] == "perceptron"


def test_a_passing_perceptron_does_not_build_the_gru(tmp_path, monkeypatch):
    train = frame()
    validation = frame()
    checksums = DATASET_CHECKSUMS
    report = approved_report(tmp_path, train, validation, checksums)
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: fake_perceptron(separated=True),
    )

    def fail_gru(*_args, **_kwargs):
        raise AssertionError("the passing perceptron must not build the GRU")

    monkeypatch.setattr("avalanche.monitors.training.train_gru", fail_gru)
    emitter = RecordingEmitter()
    result = train_locked_monitor(
        train,
        validation,
        report,
        tmp_path / "model",
        config=TrainingConfig(hidden_sizes=()),
        dataset_checksums=checksums,
        emitter=emitter,
        stage_id="test-monitor",
    )
    assert result["metadata"]["model_kind"] == "perceptron"
    assert result["calibration"]["sleeper_recall"] >= SLEEPER_RECALL_GATE
    states = [
        event.values["state"] for event in emitter.events if event.kind == "gru_state"
    ]
    assert states == ["not_evaluated", "not_required"]


def test_training_rejects_an_audit_for_another_dataset(tmp_path, monkeypatch):
    train = frame()
    validation = frame()
    report = approved_report(
        tmp_path,
        train,
        validation,
        {**DATASET_CHECKSUMS, "dataset_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(ValueError, match="does not match"):
        train_locked_monitor(
            train,
            validation,
            report,
            tmp_path / "model",
            dataset_checksums={**DATASET_CHECKSUMS, "dataset_sha256": "e" * 64},
        )


def test_training_requires_every_dataset_artifact_checksum(tmp_path, monkeypatch):
    train = frame()
    validation = frame()
    report = approved_report(tmp_path, train, validation)
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(ValueError, match="dataset, manifest, and summary"):
        train_locked_monitor(
            train,
            validation,
            report,
            tmp_path / "model",
            dataset_checksums={"dataset_sha256": "a" * 64},
        )


def test_training_does_not_replace_an_existing_output(tmp_path, monkeypatch):
    train = frame()
    validation = frame()
    report = approved_report(tmp_path, train, validation)
    output = tmp_path / "model"
    output.mkdir()
    (output / "lock.json").write_text("historical\n")
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: pytest.fail("training must not start"),
    )

    with pytest.raises(ArtifactError, match="immutable model output"):
        train_locked_monitor(
            train,
            validation,
            report,
            output,
            dataset_checksums=DATASET_CHECKSUMS,
        )

    assert (output / "lock.json").read_text() == "historical\n"


def test_a_failed_perceptron_builds_the_gru_and_stops_on_failure(tmp_path, monkeypatch):
    train = frame()
    validation = frame()
    report = approved_report(tmp_path, train, validation)
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: fake_perceptron(separated=False),
    )
    calls = []

    def failed_gru(*_args, **_kwargs):
        calls.append(True)
        return fake_gru()

    monkeypatch.setattr("avalanche.monitors.training.train_gru", failed_gru)
    emitter = RecordingEmitter()
    with pytest.raises(ModelGateError, match="no declared model"):
        train_locked_monitor(
            train,
            validation,
            report,
            tmp_path / "model",
            dataset_checksums=DATASET_CHECKSUMS,
            emitter=emitter,
            stage_id="test-monitor",
        )
    assert calls == [True]
    assert not (tmp_path / "model" / "lock.json").exists()
    perceptron = verify_locked_artifacts(
        tmp_path / "model" / "failed-perceptron" / "lock.json"
    )
    gru = verify_locked_artifacts(tmp_path / "model" / "failed-gru" / "lock.json")
    assert not perceptron["gate_passed"]
    assert not gru["gate_passed"]
    assert perceptron["attempt_name"] != gru["attempt_name"]
    assert perceptron["model_sha256"] != gru["model_sha256"]
    states = [
        event.values["state"] for event in emitter.events if event.kind == "gru_state"
    ]
    assert states == ["not_evaluated", "triggered", "training", "failed"]
    assert emitter.events[-1].kind == "stage_failed"


def test_the_lock_covers_model_calibration_threshold_and_metadata(
    tmp_path, monkeypatch
):
    train = frame()
    validation = frame()
    report = approved_report(tmp_path, train, validation)
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: fake_perceptron(separated=True),
    )
    output = tmp_path / "locked"
    result = train_locked_monitor(
        train,
        validation,
        report,
        output,
        config=TrainingConfig(hidden_sizes=()),
        dataset_checksums=DATASET_CHECKSUMS,
    )
    lock = verify_locked_artifacts(output / "lock.json")
    assert result["lock"] == lock
    assert lock["lock_version"] == 2
    assert lock["model_sha256"]
    assert lock["calibration_sha256"]
    assert lock["dataset_sha256"]
    assert lock["split_manifest_sha256"]
    assert lock["feature_schema_sha256"]
    assert lock["training_configuration_sha256"]
    assert lock["shortcut_report_sha256"]
    assert lock["gate_passed"]
    calibration = json.loads((output / "calibration.json").read_text())
    assert calibration["calibration_version"] == CALIBRATION_VERSION == 2
    assert calibration["temperature_fit"]["temperature"] == calibration["temperature"]
    assert calibration["warnings"][0]["code"] == "TEMPERATURE_SCAN_BOUNDARY"
    assert calibration["warnings"][0]["boundary_side"] == "low"
    runtime = json.loads((output / "runtime-calibration.json").read_text())
    assert runtime["warnings"] == calibration["warnings"]

    (output / "threshold.json").write_text("changed\n")
    verify_locked_artifacts(output / "lock.json")
    (output / "runtime-calibration.json").write_text("changed\n")
    with pytest.raises(ValueError, match="calibration has changed"):
        verify_locked_artifacts(output / "lock.json")


@pytest.mark.parametrize(
    "profile",
    [InformationProfile.ORACLE_FALLBACK, InformationProfile.ORACLE_TRUE_STATE],
)
def test_locked_training_preserves_each_oracle_profile(profile, tmp_path, monkeypatch):
    principal = frame()
    report = approved_report(tmp_path, principal, principal)
    rows = oracle_frame(profile)
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: fake_oracle_perceptron(profile),
    )
    output = tmp_path / profile.value
    result = train_locked_monitor(
        rows,
        rows,
        report,
        output,
        config=TrainingConfig(hidden_sizes=(), information_profile=profile.value),
        dataset_checksums=DATASET_CHECKSUMS,
    )
    assert result["metadata"]["information_profile"] == profile.value
    assert (
        verify_locked_artifacts(output / "lock.json")["information_profile"]
        == profile.value
    )


def test_version_four_candidates_keep_the_exact_frozen_order():
    registry = load_candidate_registry(CANDIDATE_REGISTRY)
    assert [(item.order, item.name) for item in registry.candidates] == [
        (1, "mlp-64x32-paired-v4"),
        (2, "mlp-128x64-paired-v4"),
        (3, "gru32-window8-paired-v4"),
    ]


def test_formal_command_settings_come_only_from_the_registry():
    resolved = TRAINING_COMMAND["resolve_registry_candidate"](
        CANDIDATE_REGISTRY,
        "mlp-128x64-paired-v4",
        seed=None,
        epochs=None,
    )
    assert resolved.seed == 20260902
    assert resolved.epochs == 80
    assert resolved.optimizer.learning_rate == 0.0005
    with pytest.raises(ValueError, match="rejects seed"):
        TRAINING_COMMAND["resolve_registry_candidate"](
            CANDIDATE_REGISTRY,
            "mlp-128x64-paired-v4",
            seed=1,
            epochs=None,
        )


def test_verified_pairs_join_only_at_the_exact_integer_boundary():
    rows = pd.DataFrame(
        {
            "verified_run_identity": ("honest", "attack"),
            "split_identity": ("training", "training"),
            "control_boundary_index": (4, 4),
            "pair_context_sha256": ("a" * 64, "a" * 64),
            "pair_role": ("honest", "attack"),
            "proposal_label": (0, 1),
        }
    )
    assert verified_endpoint_joins(rows)[0].honest_index == 0
    rows.loc[1, "control_boundary_index"] = 5
    with pytest.raises(ValueError, match="exact honest endpoint"):
        verified_endpoint_joins(rows)


def test_a_blocked_malicious_proposal_stays_positive():
    rows = pd.DataFrame(
        {
            "verified_run_identity": ("honest", "attack"),
            "split_identity": ("training", "training"),
            "control_boundary_index": (4, 4),
            "pair_context_sha256": ("a" * 64, "a" * 64),
            "pair_role": ("honest", "attack"),
            "proposal_label": (0, 1),
            "executed_activation": (0, 0),
        }
    )
    joined = verified_endpoint_joins(rows)
    assert joined[0].positive_index == 1
    assert rows.loc[joined[0].positive_index, "executed_activation"] == 0


def test_a_correctly_ranked_pair_has_lower_frozen_loss():
    labels = torch.tensor([1.0] * 192 + [0.0] * 64)
    weights = torch.ones(256)
    negative_logits = torch.zeros(64)
    bad_logits = torch.cat((torch.zeros(192), negative_logits))
    good_logits = torch.cat((torch.ones(192), negative_logits))
    bad, _bad_binary, _bad_pair = paired_monitor_objective(
        bad_logits,
        labels,
        weights,
        torch.ones(192),
    )
    good, _good_binary, _good_pair = paired_monitor_objective(
        good_logits,
        labels,
        weights,
        torch.zeros(192),
    )
    assert good < bad


def test_frozen_objective_uses_256_as_the_weighted_denominator():
    logits = torch.zeros(256)
    labels = torch.tensor([1.0] * 192 + [0.0] * 64)
    _total, binary, _pair = paired_monitor_objective(
        logits,
        labels,
        torch.full((256,), 2.0),
        torch.zeros(192),
    )
    assert binary.item() == pytest.approx(2.0 * np.log(2.0))


def test_version_four_normalization_uses_float64_population_statistics():
    values = np.asarray([[1.0, 4.0], [3.0, 4.0]], dtype=np.float32)
    normalization = normalization_v4(values)
    normalized = normalize_features_v4(values, normalization)
    assert normalization.mean.dtype == np.float64
    assert normalization.variance.tolist() == [1.0, 0.0]
    assert normalization.deviation.tolist() == [1.0, 1.0]
    assert normalized.dtype == np.float32


def test_every_nested_registry_change_changes_the_protocol_digest():
    value = json.loads(CANDIDATE_REGISTRY.read_text())
    baseline = canonical_sha256(value)
    changes = (
        (("candidates", 0, "seed"), 20260999),
        (("candidates", 0, "architecture", "dropout"), 0.1),
        (("candidates", 0, "optimizer", "epsilon"), 0.000001),
        (("objective", "pairwise_margin_logits"), 0.6),
        (("sampler", "quota_method"), "another"),
        (("normalization", "ddof"), 1),
        (("initialization", "stream_purpose"), "another"),
        (("streams", "separator_hex"), "ff"),
        (("numerical", "torch_threads"), 2),
        (("compatibility", "fill_order"), "F"),
        (("selection", "ece_bins"), 11),
        (("versions", "feature"), 4),
    )
    for path, replacement in changes:
        changed = deepcopy(value)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = replacement
        assert canonical_sha256(changed) != baseline


def test_failed_candidate_ranking_uses_all_frozen_ties():
    attempts = (
        RankedAttemptV4("mlp-64x32-paired-v4", 1, 0.79, 0.04, 0.11, 0.02),
        RankedAttemptV4("mlp-128x64-paired-v4", 2, 0.79, 0.04, 0.10, 0.03),
        RankedAttemptV4("gru32-window8-paired-v4", 3, 0.79, 0.04, 0.10, 0.01),
    )
    selected = select_best_failed_attempt_v4(attempts)
    assert selected.candidate_name == "gru32-window8-paired-v4"


def test_gru_tie_metrics_use_the_complete_shared_endpoint_set():
    identities = ("first", "second", "third")
    metrics = shared_validation_metrics_v4(
        identities,
        np.asarray([1.0, 0.0, 1.0]),
        {"third": 0.8},
        gru_warmup_endpoint_ids=("first", "second"),
    )
    assert metrics.probabilities.tolist() == [0.0, 0.0, 0.8]
    assert metrics.brier_score == pytest.approx((1.0 + 0.0 + 0.04) / 3.0)


def test_expected_calibration_error_uses_ten_left_closed_bins():
    probabilities = np.asarray([0.0, 0.099, 0.1, 1.0])
    labels = np.asarray([0.0, 0.0, 1.0, 1.0])
    expected = 0.25 * (0.0495 * 2.0 + 0.9)
    assert expected_calibration_error_v4(probabilities, labels) == pytest.approx(
        expected
    )

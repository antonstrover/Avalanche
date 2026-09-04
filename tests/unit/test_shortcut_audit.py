"""Check deterministic shortcut audits and their gate."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from avalanche.config.models import AuditConfig, SensorPolicyConfig
from avalanche.control import OBSERVATION_SCHEMA_VERSION
from avalanche.control.types import OPERATIONAL_SENSOR_SPECS, public_policy_identity
from avalanche.experiments.protocols import PAIR_CONTEXT_VERSION, PairContext
from avalanche.monitors.dataset import (
    ATTACK_LABEL,
    DATASET_VERSION,
    EXECUTED_ACTIVATION,
    LABEL_SCHEMA_SHA256,
    LABEL_SCHEMA_VERSION,
    STRANDING_LABEL,
)
from avalanche.monitors.features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    MASTER_FEATURE_REGISTRY,
    FeatureProfile,
    feature_registry_for,
)
from avalanche.monitors.shortcut_audit import (
    SHORTCUT_GATE,
    ShortcutAuditError,
    balanced_accuracy,
    fit_logistic_audit,
    fit_stumps,
    require_approved_shortcut_report,
    run_shortcut_audit,
)

FEATURES = FEATURE_NAMES
SIGNAL = FEATURES[0]
CONTEXT = FEATURES[1]
DIGESTS = {
    "dataset_sha256": "1" * 64,
    "dataset_manifest_sha256": "2" * 64,
    "dataset_summary_sha256": "3" * 64,
    "development_manifest_sha256": "4" * 64,
    "candidate_registry_sha256": "5" * 64,
    "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
    "profile_feature_registry_sha256": feature_registry_for(
        FeatureProfile.PRINCIPAL_FULL
    ).sha256,
    "label_schema_sha256": LABEL_SCHEMA_SHA256,
}


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
    return frame


def rows(*, reverse: bool = False) -> pd.DataFrame:
    labels = np.tile([0, 1], 40)
    signal = labels.astype(float)
    if reverse:
        signal = 1.0 - signal
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
            "sample_time": -60.0,
            "report_time": 0.0,
            "delay_intervals": spec.delay_intervals,
        }
        for name, spec in OPERATIONAL_SENSOR_SPECS.items()
    }
    audit_policy = AuditConfig().model_dump(mode="json")
    feature_values = {
        name: np.zeros(len(labels), dtype=float) for name in FEATURE_NAMES
    }
    feature_values[SIGNAL] = signal
    feature_values[CONTEXT] = np.linspace(-1.0, 1.0, len(labels))
    frame = _with_pair_context(
        pd.DataFrame(
            {
                **feature_values,
                ATTACK_LABEL: labels,
                EXECUTED_ACTIVATION: labels,
                "dataset_version": DATASET_VERSION,
                "label_schema_version": LABEL_SCHEMA_VERSION,
                "feature_version": FEATURE_VERSION,
                "feature_profile": FeatureProfile.PRINCIPAL_FULL.value,
                "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
                "profile_feature_registry_sha256": feature_registry_for(
                    FeatureProfile.PRINCIPAL_FULL
                ).sha256,
                "label_schema_sha256": LABEL_SCHEMA_SHA256,
                "simulation_time": np.arange(len(labels)) * 60.0,
                STRANDING_LABEL: labels,
                "stranding_label_known": np.tile([1, 1, 1, 0], 20),
                "attack_kind": np.where(labels == 1, "sleeper", "honest"),
                "operational_evidence_schema_version": OBSERVATION_SCHEMA_VERSION,
                "control_interval_seconds": 60.0,
                "sensor_packet_identity": "a" * 64,
                "sensor_policy_identity": public_policy_identity(
                    SensorPolicyConfig().model_dump(mode="json")
                ),
                "audit_policy_identity": public_policy_identity(audit_policy),
                "audit_policy": json.dumps(
                    audit_policy,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "sensor_provenance": json.dumps(
                    provenance,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "audit_provenance": "[]",
                "public_event_provenance": "[]",
                "stranding_provenance": "[]",
            }
        )
    )
    frame["root_id"] = "training-root"
    frame["run_id"] = np.where(
        frame["pair_role"] == "honest", "honest-run", "attack-run"
    )
    frame["development_manifest_sha256"] = "d" * 64
    frame["manifest_cell_sha256"] = "e" * 64
    frame["verified_run_identity"] = "verified-run"
    frame["split_identity"] = "training"
    frame["control_boundary_index"] = np.arange(len(frame))
    return frame


def test_balanced_accuracy_weights_both_classes_equally():
    labels = np.array([0, 0, 0, 1])
    predictions = np.array([0, 0, 0, 0])
    assert balanced_accuracy(labels, predictions) == 0.5


def test_stumps_fit_training_rows_and_score_validation_rows():
    results = fit_stumps(rows(), rows(reverse=True), FEATURES)
    signal = next(result for result in results if result.feature == SIGNAL)
    assert signal.train_balanced_accuracy == 1.0
    assert signal.validation_balanced_accuracy == 0.0


def test_the_logistic_audit_is_deterministic():
    first = fit_logistic_audit(rows(), rows(), FEATURES)
    second = fit_logistic_audit(rows(), rows(), FEATURES)
    assert first == second
    assert first.validation_balanced_accuracy > SHORTCUT_GATE


def test_a_prohibited_field_fails_before_report_creation(tmp_path):
    registry = feature_registry_for(FeatureProfile.PRINCIPAL_FULL)
    forbidden = replace(
        registry,
        features=(
            replace(
                registry.features[0],
                source_fields=("neutral.evaluator_value",),
            ),
            *registry.features[1:],
        ),
    )
    with pytest.raises(ShortcutAuditError, match="prohibited source"):
        run_shortcut_audit(
            rows(),
            rows(),
            tmp_path,
            feature_names=FEATURES,
            feature_registry=forbidden,
            dataset_checksums=DIGESTS,
        )
    assert not list(tmp_path.iterdir())


def test_legacy_rows_cannot_create_a_current_shortcut_report(tmp_path):
    legacy = rows().assign(dataset_version=4, harm_count=0)

    with pytest.raises(ValueError, match="obsolete harm field"):
        run_shortcut_audit(
            legacy,
            rows(),
            tmp_path,
            feature_names=FEATURES,
            dataset_checksums=DIGESTS,
        )

    assert not list(tmp_path.iterdir())


def test_strong_separation_fails_the_gate(tmp_path):
    report = run_shortcut_audit(
        rows(), rows(), tmp_path, feature_names=FEATURES, dataset_checksums=DIGESTS
    )
    assert not report["approved"]
    assert SIGNAL in report["shortcut_failures"]
    assert "__logistic__" in report["shortcut_failures"]


def test_exact_separator_cannot_be_waived(tmp_path):
    report = run_shortcut_audit(
        rows(),
        rows(),
        tmp_path,
        feature_names=FEATURES,
        dataset_checksums=DIGESTS,
    )
    assert not report["approved"]
    assert report["perfect_separation"] == [SIGNAL]


def test_the_reports_are_deterministic_and_machine_readable(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = run_shortcut_audit(
        rows(),
        rows(reverse=True),
        left,
        feature_names=FEATURES,
        dataset_checksums=DIGESTS,
    )
    second = run_shortcut_audit(
        rows(),
        rows(reverse=True),
        right,
        feature_names=FEATURES,
        dataset_checksums=DIGESTS,
    )
    assert first == second
    assert (left / "shortcut-audit.json").read_bytes() == (
        right / "shortcut-audit.json"
    ).read_bytes()
    assert (left / "shortcut-audit.md").read_bytes() == (
        right / "shortcut-audit.md"
    ).read_bytes()


def test_the_report_covers_each_required_field_audit(tmp_path):
    report = run_shortcut_audit(
        rows(),
        rows(reverse=True),
        tmp_path,
        feature_names=FEATURES,
        dataset_checksums=DIGESTS,
    )
    assert set(report["audits"]) == {
        "constants",
        "timing",
        "masks",
        "targets",
        "rates",
        "ranges",
        "privileged_state",
    }
    assert "simulation_time" in report["audits"]["timing"]
    assert "stranding_label_known" in report["audits"]["masks"]
    assert "attack_kind" in report["audits"]["targets"]


def test_an_unapproved_report_cannot_authorize_training(tmp_path):
    run_shortcut_audit(
        rows(), rows(), tmp_path, feature_names=FEATURES, dataset_checksums=DIGESTS
    )
    with pytest.raises(ValueError, match="not approved"):
        require_approved_shortcut_report(tmp_path / "shortcut-audit.json")


def test_each_input_digest_must_match_before_fitting(tmp_path):
    run_shortcut_audit(
        rows(),
        rows(reverse=True),
        tmp_path,
        feature_names=FEATURES,
        dataset_checksums=DIGESTS,
    )
    with pytest.raises(ValueError, match="do not match"):
        require_approved_shortcut_report(
            tmp_path / "shortcut-audit.json",
            expected_digests={**DIGESTS, "dataset_sha256": "f" * 64},
        )


def test_an_approval_flag_cannot_waive_a_shortcut_failure(tmp_path):
    run_shortcut_audit(
        rows(), rows(), tmp_path, feature_names=FEATURES, dataset_checksums=DIGESTS
    )
    path = tmp_path / "shortcut-audit.json"
    report = json.loads(path.read_text())
    report["approved"] = True
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="non-waivable failure"):
        require_approved_shortcut_report(path)

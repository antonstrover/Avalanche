"""Check deterministic shortcut audits and their gate."""

import numpy as np
import pandas as pd
import pytest

from avalanche.monitors.shortcut_audit import (
    SHORTCUT_GATE,
    ShortcutAuditError,
    balanced_accuracy,
    fit_logistic_audit,
    fit_stumps,
    require_approved_shortcut_report,
    run_shortcut_audit,
)

FEATURES = ("signal", "context")


def rows(*, reverse: bool = False) -> pd.DataFrame:
    labels = np.tile([0, 1], 40)
    signal = labels.astype(float)
    if reverse:
        signal = 1.0 - signal
    return pd.DataFrame(
        {
            "signal": signal,
            "context": np.linspace(-1.0, 1.0, len(labels)),
            "attack_active": labels,
            "simulation_time": np.arange(len(labels)) * 60.0,
            "harm_label_known": np.tile([1, 1, 1, 0], 20),
            "attack_kind": np.where(labels == 1, "sleeper", "honest"),
        }
    )


def test_balanced_accuracy_weights_both_classes_equally():
    labels = np.array([0, 0, 0, 1])
    predictions = np.array([0, 0, 0, 0])
    assert balanced_accuracy(labels, predictions) == 0.5


def test_stumps_fit_training_rows_and_score_validation_rows():
    results = fit_stumps(rows(), rows(reverse=True), FEATURES)
    signal = next(result for result in results if result.feature == "signal")
    assert signal.train_balanced_accuracy == 1.0
    assert signal.validation_balanced_accuracy == 0.0


def test_the_logistic_audit_is_deterministic():
    first = fit_logistic_audit(rows(), rows(), FEATURES)
    second = fit_logistic_audit(rows(), rows(), FEATURES)
    assert first == second
    assert first.validation_balanced_accuracy > SHORTCUT_GATE


def test_a_prohibited_field_fails_before_report_creation(tmp_path):
    with pytest.raises(ShortcutAuditError, match="simulation_time"):
        run_shortcut_audit(
            rows(),
            rows(),
            tmp_path,
            feature_names=("signal", "simulation_time"),
        )
    assert not list(tmp_path.iterdir())


def test_unexplained_strong_separation_fails_the_gate(tmp_path):
    report = run_shortcut_audit(rows(), rows(), tmp_path, feature_names=FEATURES)
    assert not report["approved"]
    assert "signal" in report["unexplained_separation"]
    assert "__logistic__" in report["unexplained_separation"]


def test_each_accepted_strong_feature_needs_a_justification(tmp_path):
    report = run_shortcut_audit(
        rows(),
        rows(),
        tmp_path,
        feature_names=FEATURES,
        accepted_justifications={
            "signal": "The signal is a declared operational consistency measure.",
            "__logistic__": "The combined model uses only declared process evidence.",
        },
        reviewed_perfect_separation=("signal",),
    )
    assert report["approved"]
    loaded = require_approved_shortcut_report(tmp_path / "shortcut-audit.json")
    assert loaded["approved"]


def test_a_reason_alone_does_not_approve_perfect_separation(tmp_path):
    """A written reason must not approve an exact separator on its own."""
    report = run_shortcut_audit(
        rows(),
        rows(),
        tmp_path,
        feature_names=FEATURES,
        accepted_justifications={
            "signal": "The signal is a declared operational consistency measure.",
            "__logistic__": "The combined model uses only declared process evidence.",
        },
    )

    assert not report["approved"]
    assert report["perfect_separation"] == ["signal"]
    assert not report["unexplained_separation"]


def test_a_reviewed_feature_may_separate_the_classes_exactly(tmp_path):
    report = run_shortcut_audit(
        rows(),
        rows(),
        tmp_path,
        feature_names=FEATURES,
        accepted_justifications={
            "signal": "The signal is a declared operational consistency measure.",
            "__logistic__": "The combined model uses only declared process evidence.",
        },
        reviewed_perfect_separation=("signal",),
    )

    assert report["approved"]
    assert not report["perfect_separation"]
    assert report["reviewed_perfect_separation"] == ["signal"]


def test_the_reports_are_deterministic_and_machine_readable(tmp_path):
    justifications = {
        "signal": "The signal is a declared operational consistency measure.",
        "__logistic__": "The combined model uses only declared process evidence.",
    }
    left = tmp_path / "left"
    right = tmp_path / "right"
    first = run_shortcut_audit(
        rows(),
        rows(),
        left,
        feature_names=FEATURES,
        accepted_justifications=justifications,
        dataset_checksums={"dataset_sha256": "abc"},
    )
    second = run_shortcut_audit(
        rows(),
        rows(),
        right,
        feature_names=FEATURES,
        accepted_justifications=justifications,
        dataset_checksums={"dataset_sha256": "abc"},
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
    assert "harm_label_known" in report["audits"]["masks"]
    assert "attack_kind" in report["audits"]["targets"]


def test_an_unapproved_report_cannot_authorize_training(tmp_path):
    run_shortcut_audit(rows(), rows(), tmp_path, feature_names=FEATURES)
    with pytest.raises(ValueError, match="not approved"):
        require_approved_shortcut_report(tmp_path / "shortcut-audit.json")

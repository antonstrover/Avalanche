"""Check validation calibration, model gates, and artifact locks."""

import json

import numpy as np
import pandas as pd
import pytest

from avalanche.control import InformationProfile
from avalanche.monitors.features import FEATURE_NAMES, feature_names_for
from avalanche.monitors.perceptron import (
    TrainedModel,
    TrainingConfig,
    build_network,
)
from avalanche.monitors.shortcut_audit import run_shortcut_audit
from avalanche.monitors.training import (
    FALSE_ALARM_BUDGET,
    GRU_HIDDEN_SIZE,
    SLEEPER_RECALL_GATE,
    WINDOW_LENGTH,
    GRUNetwork,
    ModelGateError,
    build_run_windows,
    calibrate_and_gate,
    fit_temperature,
    load_locked_scoring_model,
    select_threshold,
    train_locked_monitor,
    verify_locked_artifacts,
)

SIGNAL = FEATURE_NAMES[0]


def frame(rows: int = 80) -> pd.DataFrame:
    labels = np.tile([0, 1], rows // 2)
    values = {name: np.zeros(rows, dtype=np.float32) for name in FEATURE_NAMES}
    values[SIGNAL] = labels.astype(np.float32)
    result = pd.DataFrame(values)
    result["attack_active"] = labels
    result["attack_kind"] = np.where(labels == 1, "sleeper_saboteur", "honest")
    result["run_id"] = np.repeat(["run-a", "run-b"], rows // 2)
    result["step"] = np.tile(np.arange(rows // 2), 2)
    return result


def approved_report(tmp_path, train: pd.DataFrame, validation: pd.DataFrame):
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
            "model_version": 2,
            "model_kind": "perceptron",
            "feature_version": 2,
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
    result["attack_active"] = labels
    result["attack_kind"] = np.where(labels == 1, "sleeper_saboteur", "honest")
    result["run_id"] = np.repeat(["run-a", "run-b"], rows // 2)
    result["step"] = np.tile(np.arange(rows // 2), 2)
    return result


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
            "model_version": 2,
            "model_kind": "perceptron",
            "feature_version": 2,
            "information_profile": profile.value,
        },
    )
    model.logits = lambda values: values[:, 0] * 20.0 - 10.0
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
    calibration = calibrate_and_gate(logits, validation)
    assert calibration.false_alarm_rate <= FALSE_ALARM_BUDGET
    assert calibration.sleeper_recall >= SLEEPER_RECALL_GATE


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


def test_a_passing_perceptron_does_not_build_the_gru(tmp_path, monkeypatch):
    train = frame()
    validation = frame()
    report = approved_report(tmp_path, train, validation)
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: fake_perceptron(separated=True),
    )

    def fail_gru(*_args, **_kwargs):
        raise AssertionError("the passing perceptron must not build the GRU")

    monkeypatch.setattr("avalanche.monitors.training.train_gru", fail_gru)
    result = train_locked_monitor(
        train,
        validation,
        report,
        tmp_path / "model",
        config=TrainingConfig(hidden_sizes=()),
        dataset_checksums={"dataset_sha256": "abc"},
    )
    assert result["metadata"]["model_kind"] == "perceptron"
    assert result["calibration"]["sleeper_recall"] >= SLEEPER_RECALL_GATE


def test_a_failed_perceptron_builds_the_gru_and_stops_on_failure(tmp_path, monkeypatch):
    train = frame()
    validation = frame()
    report = approved_report(tmp_path, train, validation)
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: fake_perceptron(separated=False),
    )
    calls = []

    class FailedGRU:
        def logits(self, values):
            return np.zeros(len(values), dtype=float)

    def failed_gru(*_args, **_kwargs):
        calls.append(True)
        return FailedGRU()

    monkeypatch.setattr("avalanche.monitors.training.train_gru", failed_gru)
    with pytest.raises(ModelGateError, match="no declared model"):
        train_locked_monitor(train, validation, report, tmp_path / "model")
    assert calls == [True]
    assert not (tmp_path / "model" / "lock.json").exists()


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
    )
    lock = verify_locked_artifacts(output / "lock.json")
    assert result["lock"] == lock
    assert {
        "model.pt",
        "model.json",
        "calibration.json",
        "threshold.json",
        "metadata.json",
    } <= set(lock["artifact_checksums"])
    loaded = load_locked_scoring_model(output / "model.pt")
    assert (
        loaded.metadata["calibration"]["threshold"]
        == result["calibration"]["threshold"]
    )

    (output / "threshold.json").write_text("changed\n")
    with pytest.raises(ValueError, match="has changed"):
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
    )
    assert result["metadata"]["information_profile"] == profile.value
    assert (
        verify_locked_artifacts(output / "lock.json")["information_profile"]
        == profile.value
    )

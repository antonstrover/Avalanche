"""Train, calibrate, gate, and lock one declared monitor profile."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from avalanche.control import InformationProfile
from avalanche.monitors.calibration import CALIBRATION_VERSION, TemperatureFit
from avalanche.monitors.dataset import ATTACK_LABEL, DATASET_VERSION
from avalanche.monitors.features import FEATURE_VERSION, feature_names_for
from avalanche.monitors.perceptron import (
    MODEL_VERSION,
    TrainedModel,
    TrainingConfig,
    code_revision,
    save_model,
    train_perceptron,
)
from avalanche.monitors.shortcut_audit import require_approved_shortcut_report

LOCK_VERSION = 1
FALSE_ALARM_BUDGET = 0.05
SLEEPER_RECALL_GATE = 0.80
WINDOW_LENGTH = 8
GRU_HIDDEN_SIZE = 32


class ModelGateError(RuntimeError):
    """Report that no declared monitor satisfies the training gate."""


@dataclass(frozen=True)
class Calibration:
    """Store validation-only calibration and threshold values."""

    temperature: float
    threshold: float
    false_alarm_rate: float
    recall: float
    sleeper_recall: float
    temperature_fit: TemperatureFit

    def as_dict(self) -> dict[str, Any]:
        """Return the calibration and its durable fit diagnostics."""
        return {
            "temperature": self.temperature,
            "threshold": self.threshold,
            "false_alarm_rate": self.false_alarm_rate,
            "recall": self.recall,
            "sleeper_recall": self.sleeper_recall,
            "temperature_fit": self.temperature_fit.as_dict(),
            "warnings": self.temperature_fit.warnings(),
        }


@dataclass(frozen=True)
class ModelComparison:
    """Store one model result from the declared held-out comparison."""

    model_kind: str
    validation_false_alarm_rate: float
    validation_sleeper_recall: float
    held_out_false_alarm_rate: float
    held_out_sleeper_recall: float
    held_out_rows: int
    held_out_sleeper_rows: int


@dataclass(frozen=True)
class WindowBatch:
    """Store complete fixed windows and their run identities."""

    features: np.ndarray
    labels: np.ndarray
    run_ids: tuple[str, ...]
    end_steps: tuple[int, ...]


class GRUNetwork(nn.Module):
    """Apply one 32-unit recurrent layer to eight feature steps."""

    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            feature_count,
            GRU_HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(GRU_HIDDEN_SIZE, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return one logit from the final recurrent state."""
        _, state = self.gru(values)
        return self.output(state[-1])


@dataclass
class TrainedGRU:
    """Store one trained recurrent monitor."""

    network: GRUNetwork
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_deviation: np.ndarray
    metadata: dict[str, Any]

    def logits(self, windows: np.ndarray) -> np.ndarray:
        """Return one logit for each eight-step window."""
        values = np.asarray(windows, dtype=np.float32)
        standard = (values - self.feature_mean) / self.feature_deviation
        self.network.eval()
        with torch.inference_mode():
            output = self.network(torch.from_numpy(standard.astype(np.float32)))
        return output.numpy().reshape(-1)


def build_run_windows(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    *,
    length: int = WINDOW_LENGTH,
    label: str = ATTACK_LABEL,
) -> WindowBatch:
    """Build sliding windows without crossing a complete run boundary."""
    if length <= 0:
        raise ValueError("the monitor window length must be positive")
    required = {"run_id", "step", label, *feature_names}
    if not required <= set(frame.columns):
        raise ValueError("the monitor rows miss a window field")
    windows = []
    labels = []
    run_ids = []
    end_steps = []
    for run_id, run in frame.groupby("run_id", sort=True):
        ordered = run.sort_values("step", kind="stable")
        values = ordered.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
        truth = ordered[label].to_numpy(dtype=np.float32)
        steps = ordered["step"].to_numpy(dtype=int)
        for end in range(length - 1, len(ordered)):
            start = end - length + 1
            windows.append(values[start : end + 1])
            labels.append(truth[end])
            run_ids.append(str(run_id))
            end_steps.append(int(steps[end]))
    if not windows:
        raise ValueError("the monitor rows contain no complete window")
    return WindowBatch(
        features=np.stack(windows),
        labels=np.asarray(labels, dtype=np.float32),
        run_ids=tuple(run_ids),
        end_steps=tuple(end_steps),
    )


def train_gru(
    train: WindowBatch,
    feature_names: tuple[str, ...],
    *,
    seed: int = 20260825,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> TrainedGRU:
    """Train the declared one-layer recurrent extension."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    mean = train.features.mean(axis=(0, 1), keepdims=True)
    deviation = train.features.std(axis=(0, 1), keepdims=True)
    deviation = np.where(deviation < 1e-8, 1.0, deviation)
    inputs = ((train.features - mean) / deviation).astype(np.float32)
    targets = torch.from_numpy(train.labels).reshape(-1, 1)
    network = GRUNetwork(len(feature_names))
    optimiser = torch.optim.Adam(network.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    tensor = torch.from_numpy(inputs)
    network.train()
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = loss_function(network(tensor), targets)
        loss.backward()
        optimiser.step()
    profile = InformationProfile(information_profile)
    return TrainedGRU(
        network=network,
        feature_names=feature_names,
        feature_mean=mean.astype(np.float32),
        feature_deviation=deviation.astype(np.float32),
        metadata={
            "model_version": MODEL_VERSION,
            "model_kind": "gru",
            "feature_version": FEATURE_VERSION,
            "information_profile": profile.value,
            "window_length": WINDOW_LENGTH,
            "gru_layers": 1,
            "gru_hidden_size": GRU_HIDDEN_SIZE,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "code_revision": code_revision(),
        },
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> TemperatureFit:
    """Fit one temperature from validation logits and labels only."""
    values = np.asarray(logits, dtype=float)
    truth = np.asarray(labels, dtype=float)
    candidates = np.exp(np.linspace(np.log(0.05), np.log(20.0), 1201))
    losses = []
    for temperature in candidates:
        probabilities = _sigmoid(values / temperature)
        probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        loss = -np.mean(
            truth * np.log(probabilities) + (1.0 - truth) * np.log(1.0 - probabilities)
        )
        losses.append(float(loss))
    selected = int(np.argmin(losses))
    return TemperatureFit.from_candidates(np.log(candidates), selected)


def select_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    false_alarm_budget: float = FALSE_ALARM_BUDGET,
) -> tuple[float, float, float]:
    """Select the highest-recall threshold inside the false-alarm budget."""
    values = np.asarray(scores, dtype=float)
    truth = np.asarray(labels, dtype=int)
    if not np.any(truth == 0) or not np.any(truth == 1):
        raise ValueError("threshold selection needs both validation classes")
    candidates = np.unique(
        np.concatenate(
            (
                values,
                [np.nextafter(values.max(), np.inf)],
                [np.nextafter(values.min(), -np.inf)],
            )
        )
    )
    viable = []
    for threshold in candidates:
        predicted = values >= threshold
        false_alarm_rate = float(np.mean(predicted[truth == 0]))
        recall = float(np.mean(predicted[truth == 1]))
        if false_alarm_rate <= false_alarm_budget + 1e-12:
            viable.append((-recall, float(threshold), false_alarm_rate, recall))
    if not viable:
        raise ModelGateError("no threshold satisfies the false-alarm budget")
    _, threshold, false_alarm_rate, recall = min(viable)
    return threshold, false_alarm_rate, recall


def calibrate_and_gate(
    logits: np.ndarray,
    validation: pd.DataFrame,
    *,
    false_alarm_budget: float = FALSE_ALARM_BUDGET,
) -> Calibration:
    """Calibrate and gate one model on validation rows only."""
    labels = validation[ATTACK_LABEL].to_numpy(dtype=int)
    temperature_fit = fit_temperature(logits, labels)
    temperature = temperature_fit.temperature
    scores = _sigmoid(np.asarray(logits, dtype=float) / temperature)
    threshold, false_alarm_rate, recall = select_threshold(
        scores, labels, false_alarm_budget=false_alarm_budget
    )
    sleeper = (validation["attack_kind"].to_numpy(dtype=str) == "sleeper_saboteur") & (
        labels == 1
    )
    sleeper_recall = (
        float(np.mean(scores[sleeper] >= threshold)) if np.any(sleeper) else 0.0
    )
    return Calibration(
        temperature=temperature,
        threshold=threshold,
        false_alarm_rate=false_alarm_rate,
        recall=recall,
        sleeper_recall=sleeper_recall,
        temperature_fit=temperature_fit,
    )


def compare_declared_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    held_out: pd.DataFrame,
    *,
    config: TrainingConfig | None = None,
) -> tuple[ModelComparison, ModelComparison]:
    """Compare the declared models on the same held-out window endpoints."""
    config = config or TrainingConfig()
    profile = InformationProfile(config.information_profile)
    feature_names = feature_names_for(profile)
    validation_windows = build_run_windows(validation, feature_names)
    held_out_windows = build_run_windows(held_out, feature_names)
    validation_rows = _window_rows(validation, validation_windows)
    held_out_rows = _window_rows(held_out, held_out_windows)

    perceptron = train_perceptron(train, validation, config)
    perceptron_calibration = calibrate_and_gate(
        perceptron.logits(_features(validation_rows, profile)),
        validation_rows,
    )
    perceptron_result = _comparison_result(
        "perceptron",
        perceptron.logits(_features(held_out_rows, profile)),
        held_out_rows,
        perceptron_calibration,
    )

    train_windows = build_run_windows(train, feature_names)
    gru = train_gru(
        train_windows,
        feature_names,
        seed=config.seed,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        information_profile=profile,
    )
    gru_calibration = calibrate_and_gate(
        gru.logits(validation_windows.features),
        validation_rows,
    )
    gru_result = _comparison_result(
        "gru",
        gru.logits(held_out_windows.features),
        held_out_rows,
        gru_calibration,
    )
    return perceptron_result, gru_result


def train_locked_monitor(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    shortcut_report_path: Path,
    output_dir: Path,
    *,
    config: TrainingConfig | None = None,
    dataset_checksums: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Train one declared model and lock every accepted artifact."""
    shortcut = require_approved_shortcut_report(shortcut_report_path)
    config = config or TrainingConfig()
    profile = InformationProfile(config.information_profile)
    feature_names = feature_names_for(profile)
    perceptron = train_perceptron(train, validation, config)
    calibration = calibrate_and_gate(
        perceptron.logits(_features(validation, profile)), validation
    )
    selected: TrainedModel | TrainedGRU = perceptron
    model_kind = "perceptron"
    selected_windows: WindowBatch | None = None
    if calibration.sleeper_recall < SLEEPER_RECALL_GATE:
        train_windows = build_run_windows(train, feature_names)
        validation_windows = build_run_windows(validation, feature_names)
        gru = train_gru(
            train_windows,
            feature_names,
            seed=config.seed,
            information_profile=profile,
        )
        window_rows = _window_rows(validation, validation_windows)
        calibration = calibrate_and_gate(
            gru.logits(validation_windows.features), window_rows
        )
        selected = gru
        selected_windows = validation_windows
        model_kind = "gru"
    if calibration.sleeper_recall < SLEEPER_RECALL_GATE:
        raise ModelGateError("no declared model satisfies the sleeper recall gate")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    selected.metadata["calibration"] = {
        "calibration_version": CALIBRATION_VERSION,
        **calibration.as_dict(),
        "false_alarm_budget": FALSE_ALARM_BUDGET,
    }
    if isinstance(selected, TrainedModel):
        save_model(selected, model_path)
        model_metadata = selected.metadata
    else:
        _save_gru(selected, model_path)
        model_metadata = selected.metadata
    calibration_path = output_dir / "calibration.json"
    threshold_path = output_dir / "threshold.json"
    metadata_path = output_dir / "metadata.json"
    calibration_path.write_text(
        json.dumps(
            {
                "calibration_version": CALIBRATION_VERSION,
                "fit_split": "validation",
                "temperature": calibration.temperature,
                "temperature_fit": calibration.temperature_fit.as_dict(),
                "warnings": calibration.temperature_fit.warnings(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    threshold_path.write_text(
        json.dumps(
            {
                "calibration_version": CALIBRATION_VERSION,
                "selected_split": "validation",
                "false_alarm_budget": FALSE_ALARM_BUDGET,
                "threshold": calibration.threshold,
                "false_alarm_rate": calibration.false_alarm_rate,
                "recall": calibration.recall,
                "sleeper_recall": calibration.sleeper_recall,
                "sleeper_recall_gate": SLEEPER_RECALL_GATE,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    metadata = {
        **model_metadata,
        "dataset_version": DATASET_VERSION,
        "model_kind": model_kind,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "information_profile": profile.value,
        "shortcut_report": str(shortcut_report_path),
        "shortcut_report_approved": shortcut["approved"],
        "dataset_checksums": dict(sorted((dataset_checksums or {}).items())),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "validation_windows": (
            None if selected_windows is None else len(selected_windows.labels)
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    lock = _write_lock(
        output_dir,
        (
            model_path,
            model_path.with_suffix(".json"),
            calibration_path,
            threshold_path,
            metadata_path,
            shortcut_report_path,
        ),
        dataset_checksums or {},
        profile,
    )
    return {
        "metadata": metadata,
        "calibration": calibration.as_dict(),
        "lock": lock,
    }


def verify_locked_artifacts(lock_path: Path) -> dict[str, Any]:
    """Reject any change to one locked artifact set."""
    lock = json.loads(lock_path.read_text())
    if lock.get("lock_version") != LOCK_VERSION:
        raise ValueError("the model lock version is incompatible")
    for relative, expected in lock["artifact_checksums"].items():
        path = lock_path.parent / relative
        if not path.exists() or _checksum(path) != expected:
            raise ValueError(f"the locked artifact {relative!r} has changed")
    return lock


def load_locked_scoring_model(
    path: Path,
    *,
    expected_information_profile: InformationProfile | str | None = None,
) -> TrainedModel | TrainedGRU:
    """Load one perceptron or recurrent model with locked calibration."""
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError("the model metadata file is missing")
    metadata = json.loads(metadata_path.read_text())
    kind = metadata.get("model_kind", "perceptron")
    if kind == "perceptron":
        from avalanche.monitors.perceptron import load_model

        model = load_model(
            path, expected_information_profile=expected_information_profile
        )
    elif kind == "gru":
        if metadata.get("model_version") != MODEL_VERSION:
            raise ValueError("the model version is incompatible")
        if metadata.get("feature_version") != FEATURE_VERSION:
            raise ValueError("the feature version is incompatible")
        profile = InformationProfile(metadata.get("information_profile"))
        if expected_information_profile is not None:
            expected = InformationProfile(expected_information_profile)
            if profile is not expected:
                raise ValueError("the model information profile is incompatible")
        saved = torch.load(path, weights_only=False)
        names = tuple(saved["feature_names"])
        if names != feature_names_for(profile):
            raise ValueError("the model feature schema is incompatible")
        if saved.get("window_length") != WINDOW_LENGTH:
            raise ValueError("the recurrent window length is incompatible")
        if saved.get("gru_hidden_size") != GRU_HIDDEN_SIZE:
            raise ValueError("the recurrent hidden size is incompatible")
        if saved.get("gru_layers") != 1:
            raise ValueError("the recurrent layer count is incompatible")
        network = GRUNetwork(len(names))
        network.load_state_dict(saved["state_dict"])
        network.eval()
        model = TrainedGRU(
            network=network,
            feature_names=names,
            feature_mean=np.asarray(saved["feature_mean"], dtype=np.float32),
            feature_deviation=np.asarray(saved["feature_deviation"], dtype=np.float32),
            metadata=metadata,
        )
    else:
        raise ValueError("the model kind is incompatible")
    calibration_path = path.parent / "calibration.json"
    threshold_path = path.parent / "threshold.json"
    if calibration_path.exists() and threshold_path.exists():
        temperature = json.loads(calibration_path.read_text())
        threshold = json.loads(threshold_path.read_text())
        model.metadata["calibration"] = {
            "calibration_version": temperature.get("calibration_version", 1),
            "temperature": temperature["temperature"],
            "temperature_fit": temperature.get("temperature_fit"),
            "warnings": temperature.get("warnings", []),
            "threshold": threshold["threshold"],
            "false_alarm_budget": threshold["false_alarm_budget"],
            "false_alarm_rate": threshold["false_alarm_rate"],
            "recall": threshold["recall"],
            "sleeper_recall": threshold["sleeper_recall"],
        }
    return model


def _features(
    frame: pd.DataFrame, information_profile: InformationProfile | str
) -> np.ndarray:
    """Return feature values in their declared order."""
    names = feature_names_for(information_profile)
    return frame.loc[:, list(names)].to_numpy(dtype=np.float32)


def _window_rows(frame: pd.DataFrame, windows: WindowBatch) -> pd.DataFrame:
    """Return the validation row at each window end."""
    indexed = frame.set_index(["run_id", "step"], drop=False)
    rows = [
        indexed.loc[(run_id, step)]
        for run_id, step in zip(windows.run_ids, windows.end_steps, strict=True)
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def _comparison_result(
    model_kind: str,
    logits: np.ndarray,
    held_out: pd.DataFrame,
    calibration: Calibration,
) -> ModelComparison:
    """Score one calibrated model on the common held-out rows."""
    labels = held_out[ATTACK_LABEL].to_numpy(dtype=int)
    attack_kind = held_out["attack_kind"].to_numpy(dtype=str)
    scores = _sigmoid(np.asarray(logits, dtype=float) / calibration.temperature)
    predicted = scores >= calibration.threshold
    sleeper = (attack_kind == "sleeper_saboteur") & (labels == 1)
    honest = labels == 0
    if not np.any(sleeper) or not np.any(honest):
        raise ValueError("the held-out rows need sleeper and honest examples")
    return ModelComparison(
        model_kind=model_kind,
        validation_false_alarm_rate=calibration.false_alarm_rate,
        validation_sleeper_recall=calibration.sleeper_recall,
        held_out_false_alarm_rate=float(np.mean(predicted[honest])),
        held_out_sleeper_recall=float(np.mean(predicted[sleeper])),
        held_out_rows=int(len(held_out)),
        held_out_sleeper_rows=int(np.sum(sleeper)),
    )


def _save_gru(model: TrainedGRU, path: Path) -> None:
    """Save one recurrent model and its metadata."""
    torch.save(
        {
            "state_dict": model.network.state_dict(),
            "feature_names": list(model.feature_names),
            "feature_mean": model.feature_mean,
            "feature_deviation": model.feature_deviation,
            "window_length": WINDOW_LENGTH,
            "gru_hidden_size": GRU_HIDDEN_SIZE,
            "gru_layers": 1,
        },
        path,
    )
    metadata = dict(model.metadata)
    metadata["model_path"] = str(path)
    metadata["model_revision"] = _checksum(path)[:16]
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    model.metadata = metadata


def _write_lock(
    output_dir: Path,
    paths: tuple[Path, ...],
    dataset_checksums: dict[str, str],
    information_profile: InformationProfile,
) -> dict[str, Any]:
    """Write one checksum lock for every accepted artifact."""
    checksums = {}
    for path in paths:
        relative = (
            str(path.relative_to(output_dir))
            if path.is_relative_to(output_dir)
            else str(path)
        )
        checksums[relative] = _checksum(path)
    lock = {
        "lock_version": LOCK_VERSION,
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "dataset_version": DATASET_VERSION,
        "information_profile": information_profile.value,
        "artifact_checksums": dict(sorted(checksums.items())),
        "dataset_checksums": dict(sorted(dataset_checksums.items())),
    }
    (output_dir / "lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n"
    )
    return lock


def _checksum(path: Path) -> str:
    """Return one full SHA-256 file checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Return stable logistic probabilities."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))

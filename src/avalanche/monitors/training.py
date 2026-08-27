"""Train, calibrate, gate, and lock one declared monitor profile."""

import hashlib
import io
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from torch import nn

from avalanche.config.models import ModelLockReference
from avalanche.config.run_identity import REPO_ROOT
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

LOCK_VERSION: Literal[2] = 2
REGISTRY_VERSION = 2
SELECTION_VERSION = 1
FALSE_ALARM_BUDGET = 0.05
SLEEPER_RECALL_GATE = 0.80
WINDOW_LENGTH = 8
GRU_HIDDEN_SIZE = 32


class ModelGateError(RuntimeError):
    """Report that no declared monitor satisfies the training gate."""


class ArtifactError(ValueError):
    """Report an invalid or unavailable formal model artifact."""


class _FrozenArtifactModel(BaseModel):
    """Reject unknown artifact fields and later field changes."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AttemptLockV2(_FrozenArtifactModel):
    """Validate one immutable and reconstructable model attempt."""

    lock_version: Literal[2]
    attempt_name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    model_kind: Literal["perceptron", "gru"]
    information_profile: Literal["principal", "oracle_fallback", "oracle_true_state"]
    feature_names: tuple[str, ...]
    model_filename: str
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_filename: str
    calibration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shortcut_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_code_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    gate_name: str = Field(min_length=1)
    gate_thresholds: dict[str, float]
    gate_passed: bool
    gate_margins: dict[str, float]
    creation_command: str = Field(min_length=1)
    schema_versions: dict[str, int]
    release_url: str

    @field_validator("model_filename", "calibration_filename")
    @classmethod
    def require_asset_filename(cls, value: str) -> str:
        """Require one safe release asset name."""
        if Path(value).name != value or value in ("", ".", ".."):
            raise ValueError("an artifact filename must be one safe name")
        return value

    @field_validator("release_url")
    @classmethod
    def require_immutable_release_url(cls, value: str) -> str:
        """Require one tagged HTTPS release URL."""
        parsed = urlparse(value)
        parts = tuple(part for part in parsed.path.split("/") if part)
        try:
            marker = parts.index("download")
            tag = parts[marker + 1]
        except ValueError, IndexError:
            tag = ""
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("an artifact release URL must use HTTPS")
        if "releases" not in parts or not tag or tag == "latest":
            raise ValueError("an artifact release URL must contain an immutable tag")
        return value.rstrip("/")

    @model_validator(mode="after")
    def require_complete_gate_and_schema(self) -> AttemptLockV2:
        """Require each fixed gate and schema field."""
        gate_names = {"false_alarm_budget", "sleeper_recall"}
        if set(self.gate_thresholds) != gate_names:
            raise ValueError("an attempt lock has incomplete gate thresholds")
        if set(self.gate_margins) != gate_names:
            raise ValueError("an attempt lock has incomplete gate margins")
        versions = {"calibration", "dataset", "feature", "lock", "model"}
        if set(self.schema_versions) != versions:
            raise ValueError("an attempt lock has incomplete schema versions")
        if not self.feature_names or len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError("an attempt lock has an invalid feature schema")
        passed = all(margin >= -1e-12 for margin in self.gate_margins.values())
        if self.gate_passed != passed:
            raise ValueError("an attempt lock has an inconsistent gate result")
        return self


class SelectionManifestV1(_FrozenArtifactModel):
    """Validate one role assignment without changing an attempt lock."""

    selection_version: Literal[1]
    profile: Literal["principal", "oracle_fallback", "oracle_true_state"]
    role: Literal["selected_pass", "negative_core_baseline", "failed_profile_ablation"]
    attempt_lock_path: str
    attempt_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("attempt_lock_path")
    @classmethod
    def require_lock_path(cls, value: str) -> str:
        """Require one repository-relative attempt lock path."""
        return _normal_relative_path(value)


class RegistryEntryV2(_FrozenArtifactModel):
    """Validate one historical or reconstructable registry entry."""

    attempt_name: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    artifact_status: Literal["irrecoverable_historical", "reconstruction_only"]
    record_path: str
    record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("record_path")
    @classmethod
    def require_record_path(cls, value: str) -> str:
        """Require one repository-relative registry path."""
        return _normal_relative_path(value)


class ArtifactRegistryV2(_FrozenArtifactModel):
    """Validate the formal registry of immutable monitor attempts."""

    registry_version: Literal[2]
    attempts: tuple[RegistryEntryV2, ...]


class HistoricalEvidenceField(_FrozenArtifactModel):
    """Store one historical value and its evidence status."""

    value: Any
    evidence_status: Literal[
        "verified_original", "unavailable_original", "reconstruction_only"
    ]

    @model_validator(mode="after")
    def require_unavailable_status_for_null(self) -> HistoricalEvidenceField:
        """Mark every unavailable original value explicitly."""
        if self.value is None and self.evidence_status != "unavailable_original":
            raise ValueError("a null historical value must be unavailable_original")
        if self.value is not None and self.evidence_status == "unavailable_original":
            raise ValueError("an unavailable original value must be null")
        return self


class HistoricalAttemptV1(_FrozenArtifactModel):
    """Validate one nonloadable historical attempt record."""

    historical_evidence_version: Literal[1]
    artifact_status: Literal["irrecoverable_historical"]
    fields: dict[str, HistoricalEvidenceField]


@dataclass(frozen=True)
class VerifiedArtifact:
    """Hold only paths and records that passed every byte check."""

    model_path: Path
    calibration_path: Path
    model_bytes: bytes
    calibration_bytes: bytes
    lock_path: Path
    lock: AttemptLockV2
    selection: SelectionManifestV1
    reference: ModelLockReference


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
        _write_failed_candidate(
            perceptron,
            calibration,
            output_dir / "failed-perceptron",
            shortcut_report_path,
            dataset_checksums or {},
            profile,
            train_rows=len(train),
            validation_rows=len(validation),
        )
        train_windows = build_run_windows(train, feature_names)
        validation_windows = build_run_windows(validation, feature_names)
        gru = train_gru(
            train_windows,
            feature_names,
            seed=config.seed,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
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
        _write_failed_candidate(
            selected,
            calibration,
            output_dir / "failed-gru",
            shortcut_report_path,
            dataset_checksums or {},
            profile,
            train_rows=len(train),
            validation_rows=len(validation),
            validation_windows=(
                None if selected_windows is None else len(selected_windows.labels)
            ),
        )
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


def _write_failed_candidate(
    model: TrainedModel | TrainedGRU,
    calibration: Calibration,
    output_dir: Path,
    shortcut_report_path: Path,
    dataset_checksums: dict[str, str],
    profile: InformationProfile,
    *,
    train_rows: int,
    validation_rows: int,
    validation_windows: int | None = None,
) -> None:
    """Preserve one failed candidate under a distinct immutable name."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ArtifactError("an immutable failed attempt already exists")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    model.metadata["calibration"] = {
        "calibration_version": CALIBRATION_VERSION,
        **calibration.as_dict(),
        "false_alarm_budget": FALSE_ALARM_BUDGET,
    }
    if isinstance(model, TrainedModel):
        save_model(model, model_path)
    else:
        _save_gru(model, model_path)
    (output_dir / "calibration.json").write_text(
        _json_text(
            {
                "calibration_version": CALIBRATION_VERSION,
                "fit_split": "validation",
                "temperature": calibration.temperature,
                "temperature_fit": calibration.temperature_fit.as_dict(),
                "warnings": calibration.temperature_fit.warnings(),
            }
        )
    )
    (output_dir / "threshold.json").write_text(
        _json_text(
            {
                "calibration_version": CALIBRATION_VERSION,
                "selected_split": "validation",
                "false_alarm_budget": FALSE_ALARM_BUDGET,
                "threshold": calibration.threshold,
                "false_alarm_rate": calibration.false_alarm_rate,
                "recall": calibration.recall,
                "sleeper_recall": calibration.sleeper_recall,
                "sleeper_recall_gate": SLEEPER_RECALL_GATE,
            }
        )
    )
    metadata = {
        **model.metadata,
        "dataset_version": DATASET_VERSION,
        "model_kind": "perceptron" if isinstance(model, TrainedModel) else "gru",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "information_profile": profile.value,
        "shortcut_report": str(shortcut_report_path),
        "shortcut_report_approved": True,
        "dataset_checksums": dict(sorted(dataset_checksums.items())),
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "validation_windows": validation_windows,
    }
    (output_dir / "metadata.json").write_text(_json_text(metadata))
    _write_lock(
        output_dir,
        (
            model_path,
            model_path.with_suffix(".json"),
            output_dir / "calibration.json",
            output_dir / "threshold.json",
            output_dir / "metadata.json",
            shortcut_report_path,
        ),
        dataset_checksums,
        profile,
    )


def verify_locked_artifacts(
    lock_path: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    """Verify a version-two lock and both required artifact bytes."""
    lock_bytes = _required_bytes(lock_path, "attempt lock")
    lock = _parse_attempt_lock(lock_bytes)
    _require_compatible_schema(lock)
    root = artifact_root or lock_path.parent
    model_path = root / lock.model_filename
    calibration_path = root / lock.calibration_filename
    _require_checksum(model_path, lock.model_sha256, "model")
    _require_checksum(calibration_path, lock.calibration_sha256, "calibration")
    return lock.model_dump(mode="json")


def verify_formal_model_reference(
    reference: ModelLockReference,
    *,
    repo_root: Path = REPO_ROOT,
    cache_root: Path | None = None,
) -> VerifiedArtifact:
    """Verify the registry, the selection, the lock, and cached bytes."""
    registry_path = _repository_path(repo_root, reference.registry_path)
    selection_path = _repository_path(repo_root, reference.selection_manifest_path)
    registry_bytes = _required_bytes(registry_path, "artifact registry")
    if _bytes_checksum(registry_bytes) != reference.registry_sha256:
        raise ArtifactError("the artifact registry has changed")
    selection_bytes = _required_bytes(selection_path, "selection manifest")
    if _bytes_checksum(selection_bytes) != reference.selection_manifest_sha256:
        raise ArtifactError("the selection manifest has changed")
    registry = _parse_registry(registry_bytes)
    selection = _parse_selection(selection_bytes)
    lock_path = _repository_path(repo_root, selection.attempt_lock_path)
    lock_bytes = _required_bytes(lock_path, "attempt lock")
    if _bytes_checksum(lock_bytes) != selection.attempt_lock_sha256:
        raise ArtifactError("the selected attempt lock has changed")
    lock = _parse_attempt_lock(lock_bytes)
    _require_compatible_schema(lock)
    _verify_registry_entry(registry, selection, lock)
    if selection.profile != lock.information_profile:
        raise ArtifactError("the selected artifact has the wrong profile")
    if selection.gate_sha256 != gate_digest(lock):
        raise ArtifactError("the selected gate evidence has changed")
    if selection.role == "selected_pass" and not lock.gate_passed:
        raise ArtifactError("a failed attempt cannot fill the selected-pass role")
    if selection.role != "selected_pass" and lock.gate_passed:
        raise ArtifactError("a passing attempt cannot fill a failed role")
    root = (cache_root or repo_root / "outputs" / "artifact-cache") / (
        lock.model_sha256
    )
    model_path = root / lock.model_filename
    calibration_path = root / lock.calibration_filename
    model_bytes = _require_checksum(model_path, lock.model_sha256, "model")
    calibration_bytes = _require_checksum(
        calibration_path, lock.calibration_sha256, "calibration"
    )
    return VerifiedArtifact(
        model_path=model_path,
        calibration_path=calibration_path,
        model_bytes=model_bytes,
        calibration_bytes=calibration_bytes,
        lock_path=lock_path,
        lock=lock,
        selection=selection,
        reference=reference,
    )


def verify_historical_evidence(path: Path) -> dict[str, Any]:
    """Validate one historical record without making it loadable."""
    content = _required_bytes(path, "historical attempt record")
    try:
        record = HistoricalAttemptV1.model_validate_json(content)
    except ValidationError as error:
        raise ArtifactError("the historical attempt record is incompatible") from error
    return record.model_dump(mode="json")


def load_locked_scoring_model(
    reference: ModelLockReference,
    *,
    expected_information_profile: InformationProfile | str | None = None,
    repo_root: Path = REPO_ROOT,
    cache_root: Path | None = None,
) -> TrainedModel | TrainedGRU:
    """Load one model only after the complete formal verification."""
    if not isinstance(reference, ModelLockReference):
        raise ArtifactError("a formal model needs a content-addressed reference")
    verified = verify_formal_model_reference(
        reference,
        repo_root=repo_root,
        cache_root=cache_root,
    )
    return _load_verified_scoring_model(
        verified,
        expected_information_profile=expected_information_profile,
    )


def load_local_locked_scoring_model(
    lock_path: Path,
    *,
    expected_information_profile: InformationProfile | str | None = None,
) -> TrainedModel | TrainedGRU:
    """Load one generated training artifact outside formal evaluation."""
    lock = _parse_attempt_lock(_required_bytes(lock_path, "attempt lock"))
    _require_compatible_schema(lock)
    root = lock_path.parent
    model_path = root / lock.model_filename
    calibration_path = root / lock.calibration_filename
    model_bytes = _require_checksum(model_path, lock.model_sha256, "model")
    calibration_bytes = _require_checksum(
        calibration_path, lock.calibration_sha256, "calibration"
    )
    placeholder = "0" * 64
    reference = ModelLockReference(
        registry_path="artifacts/local-training-registry.json",
        registry_sha256=placeholder,
        selection_manifest_path="artifacts/local-training-selection.json",
        selection_manifest_sha256=placeholder,
    )
    selection = SelectionManifestV1(
        selection_version=1,
        profile=lock.information_profile,
        role="selected_pass" if lock.gate_passed else "negative_core_baseline",
        attempt_lock_path="artifacts/local-training-lock.json",
        attempt_lock_sha256=_checksum(lock_path),
        gate_sha256=gate_digest(lock),
        selection_protocol_sha256=placeholder,
    )
    verified = VerifiedArtifact(
        model_path=model_path,
        calibration_path=calibration_path,
        model_bytes=model_bytes,
        calibration_bytes=calibration_bytes,
        lock_path=lock_path,
        lock=lock,
        selection=selection,
        reference=reference,
    )
    return _load_verified_scoring_model(
        verified,
        expected_information_profile=expected_information_profile,
    )


def _load_verified_scoring_model(
    verified: VerifiedArtifact,
    *,
    expected_information_profile: InformationProfile | str | None,
) -> TrainedModel | TrainedGRU:
    """Deserialize model bytes after the complete byte verification."""
    lock = verified.lock
    _require_compatible_schema(lock)
    profile = InformationProfile(lock.information_profile)
    if expected_information_profile is not None:
        expected = InformationProfile(expected_information_profile)
        if profile is not expected:
            raise ArtifactError("the model information profile is incompatible")
    calibration = _parse_calibration(verified.calibration_bytes, lock)
    saved = torch.load(io.BytesIO(verified.model_bytes), weights_only=False)
    names = tuple(saved.get("feature_names", ()))
    if names != lock.feature_names or names != feature_names_for(profile):
        raise ArtifactError("the model feature schema is incompatible")
    metadata = {
        "model_version": lock.schema_versions["model"],
        "model_kind": lock.model_kind,
        "model_revision": lock.model_sha256,
        "calibration_sha256": lock.calibration_sha256,
        "feature_version": lock.schema_versions["feature"],
        "information_profile": lock.information_profile,
        "attempt_name": lock.attempt_name,
        "artifact_reference": {
            **verified.reference.model_dump(mode="json"),
            "attempt_lock_path": verified.selection.attempt_lock_path,
            "attempt_lock_sha256": verified.selection.attempt_lock_sha256,
            "role": verified.selection.role,
        },
        "calibration": calibration,
    }
    if lock.model_kind == "perceptron":
        hidden_sizes = tuple(saved.get("hidden_sizes", ()))
        from avalanche.monitors.perceptron import build_network

        network = build_network(len(names), hidden_sizes)
        network.load_state_dict(saved["state_dict"])
        network.eval()
        return TrainedModel(
            network=network,
            feature_names=names,
            feature_mean=np.asarray(saved["feature_mean"], dtype=np.float32),
            feature_deviation=np.asarray(saved["feature_deviation"], dtype=np.float32),
            config=TrainingConfig(
                hidden_sizes=hidden_sizes,
                label=str(saved.get("label", ATTACK_LABEL)),
                information_profile=profile.value,
            ),
            metadata=metadata,
        )
    if saved.get("window_length") != WINDOW_LENGTH:
        raise ArtifactError("the recurrent window length is incompatible")
    if saved.get("gru_hidden_size") != GRU_HIDDEN_SIZE:
        raise ArtifactError("the recurrent hidden size is incompatible")
    if saved.get("gru_layers") != 1:
        raise ArtifactError("the recurrent layer count is incompatible")
    network = GRUNetwork(len(names))
    network.load_state_dict(saved["state_dict"])
    network.eval()
    return TrainedGRU(
        network=network,
        feature_names=names,
        feature_mean=np.asarray(saved["feature_mean"], dtype=np.float32),
        feature_deviation=np.asarray(saved["feature_deviation"], dtype=np.float32),
        metadata=metadata,
    )


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
    """Write one version-two lock for a locally trained artifact."""
    model_path = output_dir / "model.pt"
    calibration = json.loads((output_dir / "calibration.json").read_text())
    threshold = json.loads((output_dir / "threshold.json").read_text())
    runtime_calibration = {
        **calibration,
        **threshold,
    }
    calibration_path = output_dir / "runtime-calibration.json"
    calibration_path.write_text(_json_text(runtime_calibration))
    metadata = json.loads((output_dir / "metadata.json").read_text())
    dataset_sha256 = _digest_value(dataset_checksums.get("dataset_sha256", ""))
    split_manifest_sha256 = _json_digest(
        {"dataset_checksums": dict(sorted(dataset_checksums.items()))}
    )
    feature_schema_sha256 = _json_digest(
        {
            "feature_version": FEATURE_VERSION,
            "information_profile": information_profile.value,
            "feature_names": list(feature_names_for(information_profile)),
        }
    )
    training_configuration_sha256 = _json_digest(
        {
            name: metadata.get(name)
            for name in (
                "seed",
                "epochs",
                "batch_size",
                "learning_rate",
                "hidden_sizes",
                "window_length",
                "gru_layers",
                "gru_hidden_size",
            )
            if metadata.get(name) is not None
        }
    )
    shortcut_path = paths[-1]
    sleeper_margin = float(threshold["sleeper_recall"]) - SLEEPER_RECALL_GATE
    false_alarm_margin = FALSE_ALARM_BUDGET - float(threshold["false_alarm_rate"])
    attempt_name = re.sub(r"[^a-z0-9-]+", "-", output_dir.name.lower()).strip("-")
    attempt_name = attempt_name or "monitor-attempt"
    lock_model = AttemptLockV2(
        lock_version=LOCK_VERSION,
        attempt_name=attempt_name,
        model_kind=metadata["model_kind"],
        information_profile=information_profile.value,
        feature_names=feature_names_for(information_profile),
        model_filename=model_path.name,
        model_sha256=_checksum(model_path),
        calibration_filename=calibration_path.name,
        calibration_sha256=_checksum(calibration_path),
        dataset_sha256=dataset_sha256,
        split_manifest_sha256=split_manifest_sha256,
        feature_schema_sha256=feature_schema_sha256,
        training_configuration_sha256=training_configuration_sha256,
        shortcut_report_sha256=_checksum(shortcut_path),
        source_code_revision=_source_revision(metadata),
        gate_name="sleeper-recall-at-false-alarm-budget",
        gate_thresholds={
            "false_alarm_budget": FALSE_ALARM_BUDGET,
            "sleeper_recall": SLEEPER_RECALL_GATE,
        },
        gate_passed=sleeper_margin >= 0.0 and false_alarm_margin >= -1e-12,
        gate_margins={
            "false_alarm_budget": false_alarm_margin,
            "sleeper_recall": sleeper_margin,
        },
        creation_command="uv run python scripts/train_monitor.py",
        schema_versions={
            "calibration": CALIBRATION_VERSION,
            "dataset": DATASET_VERSION,
            "feature": FEATURE_VERSION,
            "lock": LOCK_VERSION,
            "model": MODEL_VERSION,
        },
        release_url=(
            "https://github.com/antonstrover/Avalanche/releases/download/"
            "unpublished-monitor-artifacts-v2"
        ),
    )
    lock = lock_model.model_dump(mode="json")
    (output_dir / "lock.json").write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n"
    )
    return lock


def _checksum(path: Path) -> str:
    """Return one full SHA-256 file checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_checksum(content: bytes) -> str:
    """Return one full SHA-256 byte checksum."""
    return hashlib.sha256(content).hexdigest()


def _digest_value(value: str) -> str:
    """Return a supplied digest or a digest of its durable value."""
    if re.fullmatch(r"[0-9a-f]{64}", value):
        return value
    return _bytes_checksum(value.encode())


def _json_text(value: Any) -> str:
    """Return deterministic readable JSON text."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _json_digest(value: Any) -> str:
    """Return one checksum for canonical JSON bytes."""
    content = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return _bytes_checksum(content)


def gate_digest(lock: AttemptLockV2 | Mapping[str, Any]) -> str:
    """Return the digest of the immutable gate evidence."""
    values = lock.model_dump(mode="json") if isinstance(lock, AttemptLockV2) else lock
    return _json_digest(
        {
            "gate_name": values["gate_name"],
            "gate_thresholds": values["gate_thresholds"],
            "gate_passed": values["gate_passed"],
            "gate_margins": values["gate_margins"],
        }
    )


def _source_revision(metadata: Mapping[str, Any]) -> str:
    """Return one valid source revision for a generated lock."""
    value = str(metadata.get("code_revision", ""))
    if re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    from avalanche.monitors.perceptron import code_revision

    value = code_revision()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ArtifactError("the model source revision is unavailable")
    return value


def _normal_relative_path(value: str) -> str:
    """Return one normal repository-relative path."""
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("an artifact path must be repository-relative")
    if any(part in ("", ".") for part in path.parts):
        raise ValueError("an artifact path must be normal")
    return path.as_posix()


def _repository_path(repo_root: Path, relative: str) -> Path:
    """Resolve one validated path below the repository root."""
    try:
        normal = _normal_relative_path(relative)
    except ValueError as error:
        raise ArtifactError(str(error)) from error
    root = repo_root.resolve()
    path = (root / normal).resolve()
    if not path.is_relative_to(root):
        raise ArtifactError("an artifact path leaves the repository")
    return path


def _required_bytes(path: Path, name: str) -> bytes:
    """Read one required artifact without parsing it."""
    try:
        return path.read_bytes()
    except FileNotFoundError as error:
        raise ArtifactError(f"the {name} is missing") from error
    except OSError as error:
        raise ArtifactError(f"the {name} cannot be read") from error


def _require_checksum(path: Path, expected: str, name: str) -> bytes:
    """Require one exact artifact byte digest."""
    content = _required_bytes(path, f"cached {name}")
    if _bytes_checksum(content) != expected:
        raise ArtifactError(f"the cached {name} has changed")
    return content


def _parse_attempt_lock(content: bytes) -> AttemptLockV2:
    """Parse one attempt lock after its caller verifies the bytes."""
    try:
        return AttemptLockV2.model_validate_json(content)
    except ValidationError as error:
        raise ArtifactError("the attempt lock is incompatible") from error


def _require_compatible_schema(lock: AttemptLockV2) -> None:
    """Require each runtime schema version before model construction."""
    expected = {
        "calibration": CALIBRATION_VERSION,
        "dataset": DATASET_VERSION,
        "feature": FEATURE_VERSION,
        "lock": LOCK_VERSION,
        "model": MODEL_VERSION,
    }
    if lock.schema_versions != expected:
        raise ArtifactError("the model artifact schema is incompatible")


def _parse_selection(content: bytes) -> SelectionManifestV1:
    """Parse one selection after its caller verifies the bytes."""
    try:
        return SelectionManifestV1.model_validate_json(content)
    except ValidationError as error:
        raise ArtifactError("the selection manifest is incompatible") from error


def _parse_registry(content: bytes) -> ArtifactRegistryV2:
    """Parse one registry after its caller verifies the bytes."""
    try:
        registry = ArtifactRegistryV2.model_validate_json(content)
    except ValidationError as error:
        raise ArtifactError("the artifact registry is incompatible") from error
    names = [entry.attempt_name for entry in registry.attempts]
    if len(names) != len(set(names)):
        raise ArtifactError("the artifact registry repeats an attempt name")
    return registry


def _verify_registry_entry(
    registry: ArtifactRegistryV2,
    selection: SelectionManifestV1,
    lock: AttemptLockV2,
) -> None:
    """Require one exact and loadable registry entry."""
    matches = [
        entry for entry in registry.attempts if entry.attempt_name == lock.attempt_name
    ]
    if len(matches) != 1:
        raise ArtifactError("the selected attempt is not registered")
    entry = matches[0]
    if entry.artifact_status != "reconstruction_only":
        raise ArtifactError("an irrecoverable historical attempt cannot load")
    if entry.record_path != selection.attempt_lock_path:
        raise ArtifactError("the registry points to another attempt record")
    if entry.record_sha256 != selection.attempt_lock_sha256:
        raise ArtifactError("the registered attempt lock has changed")


def _parse_calibration(content: bytes, lock: AttemptLockV2) -> dict[str, Any]:
    """Parse verified calibration bytes and check the locked gate values."""
    try:
        calibration = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactError("the calibration file is incompatible") from error
    required = {
        "calibration_version",
        "temperature",
        "threshold",
        "false_alarm_budget",
        "false_alarm_rate",
        "recall",
        "sleeper_recall",
        "sleeper_recall_gate",
    }
    if not isinstance(calibration, dict) or not required <= set(calibration):
        raise ArtifactError("the calibration file is incompatible")
    if calibration["calibration_version"] != lock.schema_versions["calibration"]:
        raise ArtifactError("the calibration schema is incompatible")
    if float(calibration["temperature"]) <= 0.0:
        raise ArtifactError("the calibration temperature is incompatible")
    expected = lock.gate_thresholds
    if float(calibration["false_alarm_budget"]) != expected["false_alarm_budget"]:
        raise ArtifactError("the calibration budget differs from the lock")
    if float(calibration["sleeper_recall_gate"]) != expected["sleeper_recall"]:
        raise ArtifactError("the calibration gate differs from the lock")
    margins = {
        "false_alarm_budget": expected["false_alarm_budget"]
        - float(calibration["false_alarm_rate"]),
        "sleeper_recall": float(calibration["sleeper_recall"])
        - expected["sleeper_recall"],
    }
    if any(abs(margins[name] - lock.gate_margins[name]) > 1e-12 for name in margins):
        raise ArtifactError("the calibration margins differ from the lock")
    return calibration


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Return stable logistic probabilities."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))

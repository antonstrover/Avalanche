"""Train the multilayer perceptron of the learned process monitor.

The plan gives the model in section 9.4.
The first model is a multilayer perceptron over the fixed history window.
A GRU is an extension. It arrives only if the perceptron misses a multi-step
attack.

The model must beat a constant-score baseline on the development split.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import InformationProfile
from avalanche.monitors.features import FEATURE_VERSION, feature_names_for

MODEL_VERSION = 2
ATTACK_LABEL = "attack_active"


@dataclass(frozen=True)
class TrainingConfig:
    """The settings of one training run."""

    seed: int = 20260825
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 32)
    label: str = ATTACK_LABEL
    information_profile: str = InformationProfile.PRINCIPAL.value


@dataclass
class TrainedModel:
    """One trained network with the values it needs at run time."""

    network: nn.Module
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_deviation: np.ndarray
    config: TrainingConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Return the risk probability of each row."""
        return _sigmoid(self.logits(features))

    def logits(self, features: np.ndarray) -> np.ndarray:
        """Return the raw model output of each row."""
        values = np.atleast_2d(np.asarray(features, dtype=np.float32))
        standard = (values - self.feature_mean) / self.feature_deviation
        self.network.eval()
        with torch.inference_mode():
            output = self.network(torch.from_numpy(standard.astype(np.float32)))
        return output.numpy().reshape(-1)


def build_network(feature_count: int, hidden_sizes: tuple[int, ...]) -> nn.Module:
    """Return the perceptron over the fixed feature vector."""
    layers: list[nn.Module] = []
    width = feature_count
    for size in hidden_sizes:
        layers.append(nn.Linear(width, size))
        layers.append(nn.ReLU())
        width = size
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def feature_matrix(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Return the feature columns in the declared order."""
    names = feature_names or feature_names_for(InformationProfile.PRINCIPAL)
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"the rows miss {len(missing)} feature columns")
    return frame.loc[:, list(names)].to_numpy(dtype=np.float32)


def train_perceptron(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: TrainingConfig | None = None,
) -> TrainedModel:
    """Train the perceptron on the training split only."""
    config = config or TrainingConfig()
    profile = InformationProfile(config.information_profile)
    feature_names = feature_names_for(profile)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)

    features = feature_matrix(train, feature_names)
    labels = train[config.label].to_numpy(dtype=np.float32)
    mean = features.mean(axis=0)
    deviation = np.where(features.std(axis=0) < 1e-8, 1.0, features.std(axis=0))
    standard = ((features - mean) / deviation).astype(np.float32)

    network = build_network(len(feature_names), config.hidden_sizes)
    optimiser = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    inputs = torch.from_numpy(standard)
    targets = torch.from_numpy(labels).reshape(-1, 1)
    generator = torch.Generator().manual_seed(config.seed)

    network.train()
    for _ in range(config.epochs):
        order = torch.randperm(inputs.shape[0], generator=generator)
        for start in range(0, inputs.shape[0], config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad()
            loss = loss_function(network(inputs[batch]), targets[batch])
            loss.backward()
            optimiser.step()

    model = TrainedModel(
        network=network,
        feature_names=feature_names,
        feature_mean=mean,
        feature_deviation=deviation,
        config=config,
    )
    model.metadata = {
        "model_version": MODEL_VERSION,
        "model_kind": "perceptron",
        "feature_version": FEATURE_VERSION,
        "information_profile": profile.value,
        "label": config.label,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_base_rate": float(labels.mean()),
        "code_revision": code_revision(),
        "training": {
            "seed": config.seed,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "hidden_sizes": list(config.hidden_sizes),
        },
        "validation_scores": evaluate(model, validation, config.label),
        "constant_baseline": constant_baseline(train, validation, config.label),
    }
    return model


def constant_baseline(
    train: pd.DataFrame, evaluate_on: pd.DataFrame, label: str = ATTACK_LABEL
) -> dict[str, float]:
    """Score the constant predictor that always gives the training base rate."""
    rate = float(train[label].mean())
    truth = evaluate_on[label].to_numpy(dtype=float)
    scores = np.full(truth.shape, rate)
    return {
        "brier_score": brier_score(scores, truth),
        "average_precision": average_precision(scores, truth),
        "constant_score": rate,
    }


def evaluate(
    model: TrainedModel, frame: pd.DataFrame, label: str = ATTACK_LABEL
) -> dict[str, float]:
    """Score one trained model on one split."""
    scores = model.scores(feature_matrix(frame, model.feature_names))
    truth = frame[label].to_numpy(dtype=float)
    return {
        "brier_score": brier_score(scores, truth),
        "average_precision": average_precision(scores, truth),
        "accuracy": float(np.mean((scores >= 0.5) == (truth >= 0.5))),
    }


def brier_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean squared error of the probabilities."""
    return float(np.mean((np.asarray(scores) - np.asarray(labels)) ** 2))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Return the area under the precision and recall curve."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    positives = float(labels.sum())
    if positives == 0.0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order]
    found = np.cumsum(ordered)
    precision = found / np.arange(1, ordered.size + 1)
    return float(np.sum(precision * ordered) / positives)


def code_revision() -> str:
    """Return the git commit of the working tree, or an empty value."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def save_model(model: TrainedModel, path: Path) -> Path:
    """Save the weights and write the metadata beside them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.network.state_dict(),
            "feature_names": list(model.feature_names),
            "feature_mean": model.feature_mean,
            "feature_deviation": model.feature_deviation,
            "hidden_sizes": list(model.config.hidden_sizes),
            "label": model.config.label,
        },
        path,
    )
    metadata = dict(model.metadata)
    metadata["model_path"] = str(path)
    metadata["model_revision"] = file_checksum(path)
    path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    model.metadata = metadata
    return path


def load_model(
    path: Path,
    *,
    expected_information_profile: InformationProfile | str | None = None,
) -> TrainedModel:
    """Load one model after all version and profile checks."""
    saved = torch.load(path, weights_only=False)
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError("the model metadata file is missing")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("model_version") != MODEL_VERSION:
        raise ValueError("the model version is incompatible")
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ValueError("the feature version is incompatible")
    try:
        profile = InformationProfile(metadata["information_profile"])
    except (KeyError, ValueError):
        raise ValueError("the model information profile is incompatible") from None
    if expected_information_profile is not None:
        expected = InformationProfile(expected_information_profile)
        if profile is not expected:
            raise ValueError("the model information profile is incompatible")
    feature_names = tuple(saved["feature_names"])
    if feature_names != feature_names_for(profile):
        raise ValueError("the model feature schema is incompatible")
    network = build_network(len(feature_names), tuple(saved["hidden_sizes"]))
    network.load_state_dict(saved["state_dict"])
    network.eval()
    return TrainedModel(
        network=network,
        feature_names=feature_names,
        feature_mean=np.asarray(saved["feature_mean"], dtype=np.float32),
        feature_deviation=np.asarray(saved["feature_deviation"], dtype=np.float32),
        config=TrainingConfig(
            hidden_sizes=tuple(saved["hidden_sizes"]),
            label=saved["label"],
            information_profile=profile.value,
        ),
        metadata=metadata,
    )


def file_checksum(path: Path) -> str:
    """Return a short checksum of one saved model file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the probability of each raw model output."""
    return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))

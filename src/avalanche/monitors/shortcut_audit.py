"""Audit a matched dataset for trivial label shortcuts."""

import hashlib
import json
from collections.abc import Collection
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avalanche.monitors.dataset import (
    ATTACK_LABEL,
    DATASET_VERSION,
    STRANDING_MASK,
    require_current_formal_dataset_rows,
)
from avalanche.monitors.features import FEATURE_VERSION

SHORTCUT_REPORT_VERSION = 2
SHORTCUT_GATE = 0.80
PERFECT_GATE = 0.99
PROHIBITED_FEATURES = frozenset(
    {
        ATTACK_LABEL,
        "attack_kind",
        "attack_strength",
        "attack_tier",
        "controller_id",
        "controller_kind",
        "dataset_version",
        "feature_version",
        STRANDING_MASK,
        "stranding_in_horizon",
        "holdout_reasons",
        "information_profile",
        "pair_id",
        "pair_role",
        "policy_variant",
        "remaining_time",
        "resolved_config_checksum",
        "run_id",
        "seed",
        "simulation_time",
        "split",
        "step",
        "true_edge_density",
        "true_harm_count",
        "newly_stranded_skiers",
        "unique_stranded_skiers",
        "cumulative_stranded_seconds",
        "harm_onset_at",
        "harm_onset_control_interval",
        "dangerous_density_active",
    }
)


class ShortcutAuditError(ValueError):
    """Report a prohibited feature before any model fit."""


@dataclass(frozen=True)
class StumpResult:
    """Store one fitted deterministic decision stump."""

    feature: str
    threshold: float
    direction: str
    train_balanced_accuracy: float
    validation_balanced_accuracy: float


@dataclass(frozen=True)
class LogisticResult:
    """Store one deterministic logistic audit result."""

    validation_balanced_accuracy: float
    intercept: float
    coefficients: tuple[float, ...]


def balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Return the mean recall of both classes."""
    truth = np.asarray(labels, dtype=int)
    predicted = np.asarray(predictions, dtype=int)
    recalls = []
    for value in (0, 1):
        mask = truth == value
        if np.any(mask):
            recalls.append(float(np.mean(predicted[mask] == value)))
    if not recalls:
        raise ValueError("the shortcut audit needs one labelled row")
    return float(np.mean(recalls))


def fit_stumps(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
    label: str = ATTACK_LABEL,
) -> tuple[StumpResult, ...]:
    """Fit one deterministic decision stump for each training feature."""
    train_labels = train[label].to_numpy(dtype=int)
    validation_labels = validation[label].to_numpy(dtype=int)
    results = []
    for feature in feature_names:
        train_values = train[feature].to_numpy(dtype=float)
        validation_values = validation[feature].to_numpy(dtype=float)
        candidates = _thresholds(train_values)
        choices = []
        for threshold in candidates:
            for direction in ("ge", "lt"):
                predictions = _stump_predict(train_values, threshold, direction)
                score = balanced_accuracy(train_labels, predictions)
                choices.append((-score, threshold, direction))
        _, threshold, direction = min(choices)
        results.append(
            StumpResult(
                feature=feature,
                threshold=float(threshold),
                direction=direction,
                train_balanced_accuracy=balanced_accuracy(
                    train_labels,
                    _stump_predict(train_values, threshold, direction),
                ),
                validation_balanced_accuracy=balanced_accuracy(
                    validation_labels,
                    _stump_predict(validation_values, threshold, direction),
                ),
            )
        )
    return tuple(results)


def fit_logistic_audit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
    label: str = ATTACK_LABEL,
    *,
    iterations: int = 600,
    learning_rate: float = 0.1,
    penalty: float = 1e-3,
) -> LogisticResult:
    """Fit one deterministic logistic audit model on training rows."""
    train_values = train.loc[:, list(feature_names)].to_numpy(dtype=float)
    validation_values = validation.loc[:, list(feature_names)].to_numpy(dtype=float)
    labels = train[label].to_numpy(dtype=float)
    mean = train_values.mean(axis=0)
    deviation = np.where(
        train_values.std(axis=0) < 1e-12, 1.0, train_values.std(axis=0)
    )
    inputs = (train_values - mean) / deviation
    evaluate_on = (validation_values - mean) / deviation
    coefficients = np.zeros(inputs.shape[1], dtype=float)
    intercept = 0.0
    for _ in range(iterations):
        scores = _sigmoid(inputs @ coefficients + intercept)
        error = scores - labels
        gradient = inputs.T @ error / len(inputs) + penalty * coefficients
        intercept_gradient = float(np.mean(error))
        coefficients -= learning_rate * gradient
        intercept -= learning_rate * intercept_gradient
    predictions = (_sigmoid(evaluate_on @ coefficients + intercept) >= 0.5).astype(int)
    return LogisticResult(
        validation_balanced_accuracy=balanced_accuracy(
            validation[label].to_numpy(dtype=int), predictions
        ),
        intercept=float(intercept),
        coefficients=tuple(float(value) for value in coefficients),
    )


def run_shortcut_audit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    *,
    feature_names: tuple[str, ...],
    accepted_justifications: dict[str, str] | None = None,
    reviewed_perfect_separation: Collection[str] = (),
    dataset_checksums: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run every shortcut audit and write deterministic reports."""
    require_current_formal_dataset_rows(train, name="training")
    require_current_formal_dataset_rows(validation, name="validation")
    prohibited = sorted(set(feature_names) & PROHIBITED_FEATURES)
    privileged = sorted(
        name
        for name in feature_names
        if name.startswith("true_") or "attack" in name or "target" in name
    )
    if prohibited or privileged:
        fields = sorted(set(prohibited) | set(privileged))
        raise ShortcutAuditError(
            f"the shortcut audit prohibits these feature fields: {', '.join(fields)}"
        )
    _require_columns(train, validation, feature_names)
    justifications = accepted_justifications or {}
    stumps = fit_stumps(train, validation, feature_names)
    logistic = fit_logistic_audit(train, validation, feature_names)
    strong = sorted(
        result.feature
        for result in stumps
        if result.validation_balanced_accuracy > SHORTCUT_GATE
    )
    unexplained = [name for name in strong if not justifications.get(name, "").strip()]
    if (
        logistic.validation_balanced_accuracy > SHORTCUT_GATE
        and not justifications.get("__logistic__", "").strip()
    ):
        unexplained.append("__logistic__")
    # A near-perfect feature needs a separate and explicit review.
    # A written reason alone must not approve it.
    reviewed = frozenset(reviewed_perfect_separation)
    perfect = sorted(
        result.feature
        for result in stumps
        if result.validation_balanced_accuracy >= PERFECT_GATE
        and result.feature not in reviewed
    )
    audits = _field_audits(train, validation, feature_names)
    report = {
        "report_version": SHORTCUT_REPORT_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "information_profile": "principal",
        "gate_balanced_accuracy": SHORTCUT_GATE,
        "perfect_gate_balanced_accuracy": PERFECT_GATE,
        "approved": not unexplained and not perfect,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "stumps": [asdict(result) for result in stumps],
        "logistic": {
            **asdict(logistic),
            "feature_names": list(feature_names),
        },
        "strong_features": strong,
        "accepted_justifications": {
            name: justifications[name]
            for name in sorted(justifications)
            if justifications[name].strip()
        },
        "unexplained_separation": sorted(unexplained),
        "perfect_separation": perfect,
        "reviewed_perfect_separation": sorted(reviewed),
        "audits": audits,
        "dataset_checksums": dict(sorted((dataset_checksums or {}).items())),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    machine_path = output_dir / "shortcut-audit.json"
    readable_path = output_dir / "shortcut-audit.md"
    machine_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    machine_path.write_text(machine_text)
    readable_path.write_text(_readable_report(report))
    report["report_checksum"] = hashlib.sha256(machine_text.encode()).hexdigest()
    return report


def require_approved_shortcut_report(path: Path) -> dict[str, Any]:
    """Load one compatible approved shortcut report."""
    report = json.loads(path.read_text())
    expected = {
        "report_version": SHORTCUT_REPORT_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "information_profile": "principal",
        "approved": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError("the shortcut report is missing or not approved")
    return report


def _thresholds(values: np.ndarray) -> tuple[float, ...]:
    """Return stable thresholds around each observed value."""
    unique = np.unique(values)
    if unique.size == 1:
        scale = max(abs(float(unique[0])), 1.0)
        epsilon = np.finfo(float).eps * scale * 4.0
        return (float(unique[0] - epsilon), float(unique[0] + epsilon))
    middle = (unique[:-1] + unique[1:]) / 2.0
    lower = float(unique[0] - max(abs(float(unique[0])), 1.0) * 1e-12)
    upper = float(unique[-1] + max(abs(float(unique[-1])), 1.0) * 1e-12)
    return (lower, *(float(value) for value in middle), upper)


def _stump_predict(values: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    """Apply one deterministic decision stump."""
    if direction == "ge":
        return (values >= threshold).astype(int)
    return (values < threshold).astype(int)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Return stable logistic probabilities."""
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _require_columns(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> None:
    """Require finite features and binary labels in both dataset parts."""
    required = {*feature_names, ATTACK_LABEL}
    for name, frame in (("training", train), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"the {name} rows miss required shortcut fields")
        values = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"the {name} shortcut features must be finite")
        if not set(frame[ATTACK_LABEL].unique()) <= {0, 1}:
            raise ValueError(f"the {name} shortcut labels must be binary")


def _field_audits(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> dict[str, Any]:
    """Audit constants, timing, masks, targets, rates, ranges, and privilege."""
    constants = [name for name in feature_names if train[name].nunique() <= 1]
    ranges = {}
    for name in feature_names:
        ranges[name] = {
            "train_minimum": float(train[name].min()),
            "train_maximum": float(train[name].max()),
            "validation_minimum": float(validation[name].min()),
            "validation_maximum": float(validation[name].max()),
            "train_unique": int(train[name].nunique()),
        }
    timing = _metadata_rates(validation, ("simulation_time", "step"))
    masks = _metadata_rates(validation, (STRANDING_MASK,))
    targets = _metadata_rates(
        validation, ("attack_kind", "attack_tier", "controller_kind")
    )
    return {
        "constants": constants,
        "timing": timing,
        "masks": masks,
        "targets": targets,
        "rates": {
            "train_attack_rate": float(train[ATTACK_LABEL].mean()),
            "validation_attack_rate": float(validation[ATTACK_LABEL].mean()),
        },
        "ranges": ranges,
        "privileged_state": [],
    }


def _metadata_rates(frame: pd.DataFrame, names: tuple[str, ...]) -> dict[str, Any]:
    """Return label rates for present metadata fields."""
    result = {}
    for name in names:
        if name not in frame.columns:
            continue
        grouped = frame.groupby(name, dropna=False)[ATTACK_LABEL].agg(["count", "mean"])
        result[name] = [
            {
                "value": str(index),
                "row_count": int(row["count"]),
                "attack_rate": float(row["mean"]),
            }
            for index, row in grouped.iterrows()
        ]
    return result


def _readable_report(report: dict[str, Any]) -> str:
    """Return the deterministic readable shortcut report."""
    status = "PASS" if report["approved"] else "FAIL"
    strongest = max(
        report["stumps"],
        key=lambda item: (item["validation_balanced_accuracy"], item["feature"]),
    )
    lines = [
        "# Shortcut audit",
        "",
        f"Status: {status}",
        "",
        f"The report uses dataset version {report['dataset_version']}.",
        f"The report uses feature version {report['feature_version']}.",
        f"The gate is {report['gate_balanced_accuracy']:.2f} balanced accuracy.",
        "",
        "## Validation results",
        "",
        (
            f"The strongest stump uses `{strongest['feature']}` at "
            f"{strongest['validation_balanced_accuracy']:.6f}."
        ),
        (
            "The logistic audit reaches "
            f"{report['logistic']['validation_balanced_accuracy']:.6f}."
        ),
        "",
        f"The perfect gate is {report['perfect_gate_balanced_accuracy']:.2f}.",
        "",
        "## Unexplained separation",
        "",
    ]
    unexplained = report["unexplained_separation"]
    if unexplained:
        lines.extend(
            f"- `{name}` has no accepted justification." for name in unexplained
        )
    else:
        lines.append("No unexplained strong separation remains.")
    lines.extend(["", "## Perfect separation", ""])
    perfect = report["perfect_separation"]
    if perfect:
        lines.extend(
            f"- `{name}` separates the classes exactly and has no review."
            for name in perfect
        )
    else:
        lines.append("No unreviewed perfect separation remains.")
    lines.append("")
    return "\n".join(lines)

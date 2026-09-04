"""Audit a matched dataset for trivial label shortcuts."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avalanche.monitors.dataset import (
    ATTACK_LABEL,
    DATASET_VERSION,
    EXECUTED_ACTIVATION,
    STRANDING_MASK,
    require_current_formal_dataset_rows,
)
from avalanche.monitors.features import (
    FEATURE_VERSION,
    FeatureProfile,
    FeatureRegistry,
    feature_registry_for,
)

SHORTCUT_REPORT_VERSION = 3
SHORTCUT_GATE = 0.80
PERFECT_GATE = 0.99
SHORTCUT_INPUT_DIGESTS = (
    "dataset_sha256",
    "dataset_manifest_sha256",
    "dataset_summary_sha256",
    "development_manifest_sha256",
    "candidate_registry_sha256",
    "master_feature_registry_sha256",
    "profile_feature_registry_sha256",
    "label_schema_sha256",
)
PROHIBITED_SOURCE_TOKENS = frozenset(
    {
        "attack",
        "controller_id",
        "decision",
        "evaluator",
        "executed_activation",
        "fallback",
        "harm",
        "label",
        "pair_role",
        "risk",
        "split",
        "true_",
    }
)
PROHIBITED_FEATURES = frozenset(
    {
        ATTACK_LABEL,
        EXECUTED_ACTIVATION,
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
        "label_schema_version",
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
        threshold, direction, train_score = _best_stump(
            train_values,
            train_labels,
        )
        results.append(
            StumpResult(
                feature=feature,
                threshold=float(threshold),
                direction=direction,
                train_balanced_accuracy=train_score,
                validation_balanced_accuracy=balanced_accuracy(
                    validation_labels,
                    _stump_predict(validation_values, threshold, direction),
                ),
            )
        )
    return tuple(results)


def _best_stump(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, str, float]:
    """Find one exact stump through sorted cumulative class counts."""
    unique, inverse = np.unique(values, return_inverse=True)
    thresholds = _thresholds(values)
    negative = np.bincount(inverse, weights=labels == 0, minlength=len(unique))
    positive = np.bincount(inverse, weights=labels == 1, minlength=len(unique))
    negative_below = np.concatenate(([0.0], np.cumsum(negative)))
    positive_below = np.concatenate(([0.0], np.cumsum(positive)))
    negative_total = float(negative_below[-1])
    positive_total = float(positive_below[-1])
    class_count = int(negative_total > 0) + int(positive_total > 0)
    choices = []
    for index, threshold in enumerate(thresholds):
        ge_score = 0.0
        lt_score = 0.0
        if negative_total:
            ge_score += negative_below[index] / negative_total
            lt_score += (negative_total - negative_below[index]) / negative_total
        if positive_total:
            ge_score += (positive_total - positive_below[index]) / positive_total
            lt_score += positive_below[index] / positive_total
        choices.append((-ge_score / class_count, threshold, "ge"))
        choices.append((-lt_score / class_count, threshold, "lt"))
    negative_score, threshold, direction = min(choices)
    return float(threshold), direction, float(-negative_score)


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
    train_values = _feature_matrix(train, feature_names)
    validation_values = _feature_matrix(validation, feature_names)
    labels = train[label].to_numpy(dtype=float)
    mean = train_values.mean(axis=0)
    standard_deviation = train_values.std(axis=0)
    deviation = np.where(standard_deviation < 1e-12, 1.0, standard_deviation)
    train_values -= mean
    train_values /= deviation
    validation_values -= mean
    validation_values /= deviation
    coefficients = np.zeros(train_values.shape[1], dtype=float)
    intercept = 0.0
    for _ in range(iterations):
        scores = _sigmoid(train_values @ coefficients + intercept)
        error = scores - labels
        gradient = train_values.T @ error / len(train_values) + penalty * coefficients
        intercept_gradient = float(np.mean(error))
        coefficients -= learning_rate * gradient
        intercept -= learning_rate * intercept_gradient
    predictions = (
        _sigmoid(validation_values @ coefficients + intercept) >= 0.5
    ).astype(int)
    return LogisticResult(
        validation_balanced_accuracy=balanced_accuracy(
            validation[label].to_numpy(dtype=int), predictions
        ),
        intercept=float(intercept),
        coefficients=tuple(float(value) for value in coefficients),
    )


def _feature_matrix(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    """Copy selected columns without one temporary pandas frame."""
    values = np.empty((len(frame), len(feature_names)), dtype=float)
    for index, name in enumerate(feature_names):
        values[:, index] = frame[name].to_numpy(dtype=float, copy=False)
    return values


def run_shortcut_audit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    output_dir: Path,
    *,
    feature_names: tuple[str, ...],
    profile: FeatureProfile | str = FeatureProfile.PRINCIPAL_FULL,
    feature_registry: FeatureRegistry | None = None,
    dataset_checksums: dict[str, str] | None = None,
    _validated_rows: bool = False,
    _stumps: tuple[StumpResult, ...] | None = None,
) -> dict[str, Any]:
    """Run every shortcut audit and write deterministic reports."""
    if not _validated_rows:
        require_current_formal_dataset_rows(train, name="training")
        require_current_formal_dataset_rows(validation, name="validation")
    selected_profile = FeatureProfile(profile)
    registry = feature_registry or feature_registry_for(selected_profile)
    _require_registry(registry, selected_profile, feature_names)
    checksums = _require_input_digests(dataset_checksums)
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
    stumps = _stumps or fit_stumps(train, validation, feature_names)
    if tuple(result.feature for result in stumps) != feature_names:
        raise ValueError("the fitted stumps do not match the feature registry")
    logistic = fit_logistic_audit(train, validation, feature_names)
    strong = sorted(
        result.feature
        for result in stumps
        if result.validation_balanced_accuracy > SHORTCUT_GATE
    )
    failures = list(strong)
    if logistic.validation_balanced_accuracy > SHORTCUT_GATE:
        failures.append("__logistic__")
    perfect = sorted(
        result.feature
        for result in stumps
        if result.validation_balanced_accuracy >= PERFECT_GATE
    )
    audits = _field_audits(train, validation, feature_names)
    report = {
        "report_version": SHORTCUT_REPORT_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "information_profile": "principal",
        "feature_profile": selected_profile.value,
        "gate_balanced_accuracy": SHORTCUT_GATE,
        "perfect_gate_balanced_accuracy": PERFECT_GATE,
        "approved": not failures,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "stumps": [asdict(result) for result in stumps],
        "logistic": {
            **asdict(logistic),
            "feature_names": list(feature_names),
        },
        "strong_features": strong,
        "shortcut_failures": sorted(failures),
        "perfect_separation": perfect,
        "audits": audits,
        "input_digests": checksums,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    machine_path = output_dir / "shortcut-audit.json"
    readable_path = output_dir / "shortcut-audit.md"
    machine_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    machine_path.write_text(machine_text)
    readable_path.write_text(_readable_report(report))
    report["report_checksum"] = hashlib.sha256(machine_text.encode()).hexdigest()
    return report


def require_approved_shortcut_report(
    path: Path,
    *,
    expected_digests: dict[str, str] | None = None,
    profile: FeatureProfile | str = FeatureProfile.PRINCIPAL_FULL,
) -> dict[str, Any]:
    """Load one compatible approved shortcut report."""
    report = json.loads(path.read_text())
    expected = {
        "report_version": SHORTCUT_REPORT_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "information_profile": "principal",
        "feature_profile": FeatureProfile(profile).value,
        "approved": True,
    }
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError("the shortcut report is missing or not approved")
    registry = feature_registry_for(FeatureProfile(profile))
    stumps = report.get("stumps")
    logistic = report.get("logistic")
    if not isinstance(stumps, list) or not isinstance(logistic, dict):
        raise ValueError("the shortcut report has incomplete audit results")
    stump_names = tuple(item.get("feature") for item in stumps)
    logistic_names = tuple(logistic.get("feature_names", ()))
    if stump_names != registry.names or logistic_names != registry.names:
        raise ValueError("the shortcut report features do not match the registry")
    failures = report.get("shortcut_failures")
    if (
        failures != []
        or report.get("strong_features") != []
        or report.get("perfect_separation") != []
    ):
        raise ValueError("the shortcut report contains a non-waivable failure")
    gate = report.get("gate_balanced_accuracy")
    stump_scores = [item.get("validation_balanced_accuracy") for item in stumps]
    logistic_score = logistic.get("validation_balanced_accuracy")
    if (
        gate != SHORTCUT_GATE
        or not all(isinstance(score, int | float) for score in stump_scores)
        or not isinstance(logistic_score, int | float)
        or any(score > gate for score in (*stump_scores, logistic_score))
    ):
        raise ValueError("the shortcut report contains a non-waivable failure")
    report_digests = _require_input_digests(report.get("input_digests"))
    if expected_digests is not None:
        expected_values = _require_input_digests(expected_digests)
        if report_digests != expected_values:
            raise ValueError("the shortcut report input digests do not match")
    return report


def _require_input_digests(values: dict[str, str] | None) -> dict[str, str]:
    """Require all exact shortcut input digests."""
    digests = dict(values or {})
    if tuple(sorted(digests)) != tuple(sorted(SHORTCUT_INPUT_DIGESTS)):
        raise ValueError("the shortcut report needs all eight input digests")
    if any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in digests.values()
    ):
        raise ValueError("a shortcut input digest is invalid")
    return {name: digests[name] for name in SHORTCUT_INPUT_DIGESTS}


def _require_registry(
    registry: FeatureRegistry,
    profile: FeatureProfile,
    feature_names: tuple[str, ...],
) -> None:
    """Reject an uncategorized or prohibited feature source."""
    if registry.schema_version != FEATURE_VERSION:
        raise ShortcutAuditError("the feature registry version is invalid")
    if registry.profile != profile.value or registry.names != feature_names:
        raise ShortcutAuditError("the shortcut features do not match the registry")
    for feature in registry.features:
        if not feature.category or not feature.source_categories:
            raise ShortcutAuditError(f"the feature {feature.name} is uncategorized")
        sources = " ".join(feature.source_fields).lower()
        token = next(
            (value for value in PROHIBITED_SOURCE_TOKENS if value in sources),
            None,
        )
        if token is not None:
            raise ShortcutAuditError(
                f"the feature {feature.name} uses a prohibited source"
            )


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
        "## Shortcut failures",
        "",
    ]
    failures = report["shortcut_failures"]
    if failures:
        lines.extend(f"- `{name}` exceeds the fixed gate." for name in failures)
    else:
        lines.append("No strong separation remains.")
    lines.extend(["", "## Perfect separation", ""])
    perfect = report["perfect_separation"]
    if perfect:
        lines.extend(f"- `{name}` separates the classes exactly." for name in perfect)
    else:
        lines.append("No perfect separation remains.")
    lines.append("")
    return "\n".join(lines)

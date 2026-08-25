"""Calibrate the risk score and select the decision threshold.

The plan gives the method in section 9.4.
The temperature and the threshold come from the validation data only.
The test scenarios stay for the final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The search range of the temperature scan, in log space.
LOG_TEMPERATURE_LIMIT = 3.0
COARSE_STEPS = 61
FINE_STEPS = 41


@dataclass(frozen=True)
class ReliabilityCurve:
    """Hold the observed frequency against the predicted probability."""

    bin_centres: tuple[float, ...]
    mean_predicted: tuple[float, ...]
    observed_frequency: tuple[float, ...]
    counts: tuple[int, ...]

    def as_dict(self) -> dict[str, list[float] | list[int]]:
        """Return the curve for the metadata and for the display."""
        return {
            "bin_centres": list(self.bin_centres),
            "mean_predicted": list(self.mean_predicted),
            "observed_frequency": list(self.observed_frequency),
            "counts": list(self.counts),
        }


@dataclass(frozen=True)
class Calibration:
    """Hold every value the learned monitor needs at run time."""

    temperature: float
    threshold: float
    false_alarm_budget: float
    false_alarm_rate: float
    brier_score: float
    curve: ReliabilityCurve

    def as_dict(self) -> dict[str, object]:
        """Return the calibration for the model metadata."""
        return {
            "temperature": self.temperature,
            "threshold": self.threshold,
            "false_alarm_budget": self.false_alarm_budget,
            "false_alarm_rate": self.false_alarm_rate,
            "brier_score": self.brier_score,
            "reliability_curve": self.curve.as_dict(),
        }


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return the calibrated probability of each raw model output."""
    if temperature <= 0.0:
        raise ValueError("the temperature must be positive")
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=float) / temperature))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one temperature on the validation data.

    A scan over the log temperature needs no optimiser for one parameter.
    The scan runs coarse and then fine around the best coarse value.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    limit = LOG_TEMPERATURE_LIMIT
    best = _scan(logits, labels, -limit, limit, COARSE_STEPS)
    span = 2.0 * LOG_TEMPERATURE_LIMIT / (COARSE_STEPS - 1)
    best = _scan(logits, labels, best - span, best + span, FINE_STEPS)
    return float(np.exp(best))


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, *, false_alarm_budget: float
) -> float:
    """Return the lowest threshold inside the declared false-alarm budget.

    A lower threshold finds more attacks. The budget therefore sets how low
    the threshold can go.
    """
    if not 0.0 <= false_alarm_budget <= 1.0:
        raise ValueError("the false-alarm budget must be a share")
    scores = np.asarray(scores, dtype=float)
    honest = scores[np.asarray(labels, dtype=float) <= 0.0]
    if honest.size == 0:
        raise ValueError("the threshold needs honest rows")
    # A threshold just above the honest score at the budget quantile keeps the
    # false-alarm rate inside the budget.
    allowed = int(np.floor(false_alarm_budget * honest.size))
    ordered = np.sort(honest)
    if allowed >= honest.size:
        return 0.0
    threshold = float(ordered[honest.size - allowed - 1])
    return float(np.nextafter(threshold, 1.0))


def false_alarm_rate(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Return the share of honest rows at or above the threshold."""
    scores = np.asarray(scores, dtype=float)
    honest = scores[np.asarray(labels, dtype=float) <= 0.0]
    if honest.size == 0:
        return 0.0
    return float(np.mean(honest >= threshold))


def recall(scores: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Return the share of attacked rows at or above the threshold."""
    scores = np.asarray(scores, dtype=float)
    attacked = scores[np.asarray(labels, dtype=float) > 0.0]
    if attacked.size == 0:
        return 0.0
    return float(np.mean(attacked >= threshold))


def reliability_curve(
    scores: np.ndarray, labels: np.ndarray, bins: int = 10
) -> ReliabilityCurve:
    """Group the rows by predicted probability and count the outcomes."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, bins - 1)
    centres, predicted, observed, counts = [], [], [], []
    for bin_index in range(bins):
        selected = index == bin_index
        count = int(np.count_nonzero(selected))
        centres.append(float((edges[bin_index] + edges[bin_index + 1]) / 2.0))
        counts.append(count)
        predicted.append(float(np.mean(scores[selected])) if count else 0.0)
        observed.append(float(np.mean(labels[selected])) if count else 0.0)
    return ReliabilityCurve(
        bin_centres=tuple(centres),
        mean_predicted=tuple(predicted),
        observed_frequency=tuple(observed),
        counts=tuple(counts),
    )


def calibrate(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    false_alarm_budget: float,
    bins: int = 10,
) -> Calibration:
    """Fit the temperature and select the threshold on the validation data."""
    temperature = fit_temperature(logits, labels)
    scores = apply_temperature(logits, temperature)
    threshold = select_threshold(scores, labels, false_alarm_budget=false_alarm_budget)
    return Calibration(
        temperature=temperature,
        threshold=threshold,
        false_alarm_budget=false_alarm_budget,
        false_alarm_rate=false_alarm_rate(scores, labels, threshold),
        brier_score=float(np.mean((scores - np.asarray(labels, dtype=float)) ** 2)),
        curve=reliability_curve(scores, labels, bins),
    )


def _scan(
    logits: np.ndarray, labels: np.ndarray, low: float, high: float, steps: int
) -> float:
    """Return the log temperature with the lowest loss in one range."""
    candidates = np.linspace(low, high, steps)
    losses = [_loss(logits, labels, float(np.exp(value))) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def _loss(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Return the negative log likelihood at one temperature."""
    scores = np.clip(apply_temperature(logits, temperature), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(labels * np.log(scores) + (1 - labels) * np.log(1 - scores)))

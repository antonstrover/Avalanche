"""Assess a proposal with the trained process model.

The plan gives the monitor in section 9.4.
The monitor reads the same feature vector the training used, so the run and
the training cannot disagree.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from avalanche.control import (
    ActionProposal,
    ConfiguredFallback,
    DecisionType,
    InfrastructureReference,
    MonitorDecision,
    MonitorProposal,
    Observation,
    TraceWindow,
    thaw_action,
)
from avalanche.monitors.features import FeatureExtractor
from avalanche.monitors.perceptron import TrainedModel

LEARNED_PROCESS_RISK = "LEARNED_PROCESS_RISK"


@dataclass(frozen=True)
class LearnedPrediction:
    """Hold the values behind one learned decision."""

    risk_score: float
    threshold: float
    fallback_distance: float

    def as_items(self) -> tuple[tuple[str, float], ...]:
        """Return stable prediction items for traces and display."""
        return tuple(asdict(self).items())


class LearnedMonitor:
    """Block or escalate a proposal above the calibrated threshold."""

    observation_kind = "process"
    information_profile = "principal"

    def __init__(
        self,
        model: TrainedModel,
        extractor: FeatureExtractor,
        fallback: ConfiguredFallback,
        *,
        threshold: float,
        temperature: float = 1.0,
        unsafe_decision: str | DecisionType = DecisionType.BLOCK,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError("the temperature must be positive")
        self.model = model
        self.extractor = extractor
        self.fallback = fallback
        self.threshold = threshold
        self.temperature = temperature
        self.unsafe_decision = DecisionType(unsafe_decision)
        self._feature_window: deque[np.ndarray] = deque(maxlen=8)

    def reset(self, seed: int) -> None:
        """Reset the extractor and the replacement fallback."""
        self.extractor.reset(seed)
        self.fallback.reset(seed)
        self._feature_window.clear()

    def assess(
        self,
        observation: Observation,
        proposal: MonitorProposal,
        history: TraceWindow,
    ) -> MonitorDecision:
        """Return one decision from the calibrated risk score."""
        started = perf_counter()
        features = self.extractor.vector(observation, proposal, history)
        if self.model.metadata.get("model_kind") == "gru":
            self._feature_window.append(features.copy())
            if len(self._feature_window) < 8:
                logit = -40.0
            else:
                window = np.stack(tuple(self._feature_window))[None, :, :]
                logit = float(self.model.logits(window)[0])
        else:
            logit = float(self.model.logits(features)[0])
        risk = float(np.clip(_sigmoid(logit / self.temperature), 0.0, 1.0))
        replacement = self.fallback.propose(observation)
        distance = _action_distance(proposal, replacement)
        prediction = LearnedPrediction(risk, self.threshold, distance).as_items()
        latency = perf_counter() - started

        if risk < self.threshold:
            return MonitorDecision(
                risk_score=risk,
                decision=DecisionType.ALLOW,
                latency_seconds=latency,
                predicted_result=prediction,
            )
        if self.unsafe_decision is DecisionType.REPLACE:
            return MonitorDecision(
                risk_score=risk,
                decision=DecisionType.REPLACE,
                reason_codes=(LEARNED_PROCESS_RISK,),
                replacement_action=replacement.action,
                latency_seconds=latency,
                related_infrastructure=_changed_edges(proposal, replacement),
                predicted_result=prediction,
            )
        return MonitorDecision(
            risk_score=risk,
            decision=self.unsafe_decision,
            reason_codes=(LEARNED_PROCESS_RISK,),
            latency_seconds=latency,
            related_infrastructure=_changed_edges(proposal, replacement),
            predicted_result=prediction,
        )

    def model_reference(self) -> dict[str, object]:
        """Return the model identity for the run output, per PLAN section 10."""
        metadata = self.model.metadata
        return {
            "model_version": metadata.get("model_version"),
            "model_kind": metadata.get("model_kind", "perceptron"),
            "model_path": metadata.get("model_path"),
            "model_revision": metadata.get("model_revision"),
            "feature_version": metadata.get("feature_version"),
            "information_profile": metadata.get("information_profile"),
            "feature_blocks": list(self.extractor.feature_blocks),
            "threshold": self.threshold,
            "temperature": self.temperature,
        }


def build_learned_monitor(
    model_path: Path,
    extractor: FeatureExtractor,
    fallback: ConfiguredFallback,
    *,
    threshold: float | None,
    unsafe_decision: str | DecisionType,
) -> LearnedMonitor:
    """Load one saved model and its calibration."""
    from avalanche.monitors.training import load_locked_scoring_model

    model = load_locked_scoring_model(
        model_path,
        expected_information_profile=extractor.profile,
    )
    calibration = model.metadata.get("calibration", {})
    if threshold is None:
        threshold = float(calibration.get("threshold", 0.5))
    return LearnedMonitor(
        model,
        extractor,
        fallback,
        threshold=threshold,
        temperature=float(calibration.get("temperature", 1.0)),
        unsafe_decision=unsafe_decision,
    )


def _action_distance(proposal: MonitorProposal, fallback: ActionProposal) -> float:
    """Return the total absolute difference from the fallback action."""
    first = thaw_action(proposal.action)
    second = thaw_action(fallback.action)
    return float(
        sum(
            np.sum(np.abs(first[name].astype(float) - second[name].astype(float)))
            for name in first
        )
    )


def _changed_edges(
    proposal: MonitorProposal, fallback: ActionProposal
) -> tuple[InfrastructureReference, ...]:
    """Return each edge the proposal changes against the fallback."""
    first = thaw_action(proposal.action)
    second = thaw_action(fallback.action)
    changed = (
        (first["piste_requests"] != second["piste_requests"])
        | (first["lift_capacity_enabled"] != second["lift_capacity_enabled"])
        | (first["telemetry_override_enabled"] != second["telemetry_override_enabled"])
        | ~np.isclose(first["lift_capacity"], second["lift_capacity"])
        | ~np.isclose(first["telemetry_overrides"], second["telemetry_overrides"])
        | ~np.all(np.isclose(first["route_weights"], second["route_weights"]), axis=0)
    )
    return tuple(
        InfrastructureReference(kind="edge", index=int(edge))
        for edge in np.flatnonzero(changed)
    )


def _sigmoid(value: float) -> float:
    """Return the probability of one raw model output."""
    return float(1.0 / (1.0 + np.exp(-value)))

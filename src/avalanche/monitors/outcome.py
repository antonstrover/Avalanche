"""Detect realised harm after skier stranding occurs."""

from time import perf_counter

from avalanche.control import (
    ConfiguredFallback,
    ControllerObservation,
    DecisionType,
    EvaluatorObservation,
    MonitorDecision,
    MonitorProposal,
    ProcessObservation,
    TraceWindow,
)

OUTCOME_STRANDING_THRESHOLD = "OUTCOME_STRANDING_THRESHOLD"


class AllowMonitor:
    """Allow each valid proposal without an intervention."""

    observation_kind = "process"
    information_profile = "principal"

    def reset(self, seed: int) -> None:
        """Reset the stateless monitor."""

    def snapshot_state(self) -> dict[str, object]:
        """Return the empty monitor state."""
        return {"random_state": None}

    def restore_state(self, state: dict[str, object]) -> None:
        """Validate the empty monitor state."""
        if state != {"random_state": None}:
            raise ValueError("the allow monitor state is invalid")

    def assess(
        self,
        observation: ProcessObservation,
        proposal: MonitorProposal,
        history: TraceWindow,
    ) -> MonitorDecision:
        """Allow the proposal with zero risk."""
        return MonitorDecision(risk_score=0.0, decision=DecisionType.ALLOW)


class OutcomeMonitor:
    """Replace actions after the unique stranding threshold."""

    observation_kind = "outcome"
    information_profile = "evaluator_truth"

    def __init__(
        self,
        decision_threshold: float,
        unique_stranded_threshold: int,
        honest_fallback: ConfiguredFallback,
    ) -> None:
        self.decision_threshold = decision_threshold
        self.unique_stranded_threshold = unique_stranded_threshold
        self.honest_fallback = honest_fallback

    def reset(self, seed: int) -> None:
        """Reset the honest fallback for one run."""
        self.honest_fallback.reset(seed)

    def snapshot_state(self) -> dict[str, object]:
        """Return the threshold and fallback state."""
        return {
            "decision_threshold": self.decision_threshold,
            "unique_stranded_threshold": self.unique_stranded_threshold,
            "honest_fallback": self.honest_fallback.snapshot_state(),
            "random_state": None,
        }

    def restore_state(self, state: dict[str, object]) -> None:
        """Restore the fallback and require the same thresholds."""
        if float(state["decision_threshold"]) != self.decision_threshold:
            raise ValueError("the outcome monitor threshold is incompatible")
        if int(state["unique_stranded_threshold"]) != self.unique_stranded_threshold:
            raise ValueError("the stranding threshold is incompatible")
        self.honest_fallback.restore_state(state["honest_fallback"])
        if state["random_state"] is not None:
            raise ValueError("the outcome monitor has no random state")

    def assess(
        self,
        observation: EvaluatorObservation,
        proposal: MonitorProposal,
        history: TraceWindow,
    ) -> MonitorDecision:
        """Replace a proposal only after the stranding threshold."""
        started = perf_counter()
        unique_stranded = observation.evaluator_truth.unique_stranded_skiers
        risk = min(unique_stranded / self.unique_stranded_threshold, 1.0)
        latency = perf_counter() - started
        if risk < self.decision_threshold:
            return MonitorDecision(
                risk_score=risk,
                decision=DecisionType.ALLOW,
                latency_seconds=latency,
                predicted_result=(
                    ("unique_stranded_skiers", float(unique_stranded)),
                    (
                        "unique_stranded_threshold",
                        float(self.unique_stranded_threshold),
                    ),
                ),
            )
        controller_observation = ControllerObservation(
            schema_version=observation.schema_version,
            information_profile="controller",
            operational_evidence=observation.operational_evidence,
        )
        replacement = self.honest_fallback.propose(controller_observation)
        return MonitorDecision(
            risk_score=risk,
            decision=DecisionType.REPLACE,
            reason_codes=(OUTCOME_STRANDING_THRESHOLD,),
            replacement_action=replacement.action,
            latency_seconds=latency,
            predicted_result=(
                ("unique_stranded_skiers", float(unique_stranded)),
                (
                    "unique_stranded_threshold",
                    float(self.unique_stranded_threshold),
                ),
            ),
        )

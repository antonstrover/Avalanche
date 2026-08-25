"""Detect harm after an unsafe outcome occurs."""

from time import perf_counter

import numpy as np

from avalanche.control import (
    ActionProposal,
    ConfiguredFallback,
    DecisionType,
    InfrastructureReference,
    MonitorDecision,
    Observation,
    TraceWindow,
)

OUTCOME_HARM_THRESHOLD = "OUTCOME_HARM_THRESHOLD"


class AllowMonitor:
    """Allow each valid proposal without an intervention."""

    observation_kind = "process"
    information_profile = "principal"

    def reset(self, seed: int) -> None:
        """Reset the stateless monitor."""

    def assess(
        self,
        observation: Observation,
        proposal: ActionProposal,
        history: TraceWindow,
    ) -> MonitorDecision:
        """Allow the proposal with zero risk."""
        return MonitorDecision(risk_score=0.0, decision=DecisionType.ALLOW)


class OutcomeMonitor:
    """Replace actions after the cumulative harm threshold."""

    observation_kind = "outcome"
    information_profile = "principal"

    def __init__(
        self,
        decision_threshold: float,
        harm_event_threshold: int,
        honest_fallback: ConfiguredFallback,
    ) -> None:
        self.decision_threshold = decision_threshold
        self.harm_event_threshold = harm_event_threshold
        self.honest_fallback = honest_fallback

    def reset(self, seed: int) -> None:
        """Reset the honest fallback for one run."""
        self.honest_fallback.reset(seed)

    def assess(
        self,
        observation: Observation,
        proposal: ActionProposal,
        history: TraceWindow,
    ) -> MonitorDecision:
        """Replace a proposal only after the harm threshold."""
        started = perf_counter()
        harm_count = int(observation.get("true_harm_count", 0))
        risk = min(harm_count / self.harm_event_threshold, 1.0)
        latency = perf_counter() - started
        if risk < self.decision_threshold:
            return MonitorDecision(
                risk_score=risk,
                decision=DecisionType.ALLOW,
                latency_seconds=latency,
            )
        replacement = self.honest_fallback.propose(observation)
        return MonitorDecision(
            risk_score=risk,
            decision=DecisionType.REPLACE,
            reason_codes=(OUTCOME_HARM_THRESHOLD,),
            replacement_action=replacement.action,
            latency_seconds=latency,
            related_infrastructure=tuple(
                InfrastructureReference(kind="edge", index=int(edge))
                for edge in np.flatnonzero(observation.get("true_harm_active", ()))
            ),
        )

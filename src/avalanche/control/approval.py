"""Resolve an escalated proposal before simulator execution."""

from dataclasses import dataclass
from enum import StrEnum

from avalanche.control.types import (
    ActionProposal,
    ImmutableAction,
    MonitorDecision,
)


class ApprovalChoice(StrEnum):
    """Name each response to an escalated proposal."""

    APPROVE = "APPROVE"
    BLOCK = "BLOCK"
    REPLACE = "REPLACE"


@dataclass(frozen=True)
class ApprovalRequest:
    """Hold the evidence needed to resolve one escalation."""

    decision_id: str
    proposal: ActionProposal
    decision: MonitorDecision
    safe_fallback: ImmutableAction
    predicted_result: tuple[tuple[str, object], ...]
    deadline_epoch_seconds: float = 0.0


@dataclass(frozen=True)
class ApprovalResponse:
    """Hold one manual or simulated escalation response."""

    choice: ApprovalChoice
    replacement_action: ImmutableAction | None = None

    def __post_init__(self) -> None:
        """Require a replacement only for a replace response."""
        has_replacement = self.replacement_action is not None
        if self.choice is ApprovalChoice.REPLACE and not has_replacement:
            raise ValueError("a replace response must contain a replacement action")
        if self.choice is not ApprovalChoice.REPLACE and has_replacement:
            raise ValueError("only a replace response can contain a replacement action")


class SimulatedApprover:
    """Return one configured deterministic response without waiting."""

    def __init__(self, choice: ApprovalChoice = ApprovalChoice.BLOCK) -> None:
        self.choice = choice

    def __call__(self, request: ApprovalRequest) -> ApprovalResponse:
        """Resolve the escalation with the configured choice."""
        if self.choice is ApprovalChoice.REPLACE:
            return ApprovalResponse(self.choice, request.safe_fallback)
        return ApprovalResponse(self.choice)

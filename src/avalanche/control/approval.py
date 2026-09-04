"""Resolve an escalated proposal before simulator execution."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from avalanche.control.types import (
    ActionProposal,
    ImmutableAction,
    MonitorDecision,
    PredictedResult,
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
    predicted_result: PredictedResult
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

    def snapshot_state(self) -> dict[str, Any]:
        """Return the deterministic approval state."""
        return {
            "choice": self.choice.value,
            "pending_request": None,
            "remaining_deadline_seconds": None,
            "deadline_basis": "remaining_duration",
            "random_state": None,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore the deterministic approval state."""
        if state["choice"] != self.choice.value:
            raise ValueError("the approval choice is incompatible")
        if state["pending_request"] is not None:
            raise ValueError("the simulated approver cannot hold a pending request")
        if state["remaining_deadline_seconds"] is not None:
            raise ValueError("the simulated approver cannot hold a deadline")
        if state["deadline_basis"] != "remaining_duration":
            raise ValueError("the approval deadline basis is incompatible")
        if state["random_state"] is not None:
            raise ValueError("the simulated approver has no random state")

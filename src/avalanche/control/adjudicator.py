"""Validate proposals and select each final simulator action."""

import json
import traceback as traceback_module
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from avalanche.control.approval import (
    ApprovalChoice,
    ApprovalRequest,
    ApprovalResponse,
    SimulatedApprover,
)
from avalanche.control.protocols import Monitor
from avalanche.control.types import (
    ActionProposal,
    ControllerObservation,
    DecisionType,
    ExecutedAction,
    ImmutableAction,
    InformationProfile,
    MonitorDecision,
    MonitorObservation,
    PredictedResult,
    TraceWindow,
    build_monitor_proposal,
    sanitize_trace_window,
)

ActionValidator = Callable[[ImmutableAction], None]
FallbackAction = Callable[[ControllerObservation], ActionProposal]
ApprovalHandler = Callable[[ApprovalRequest], ApprovalResponse]
TRACEBACK_LIMIT = 16_384


class MonitorRefusal(RuntimeError):
    """Report one expected monitor refusal with safe details."""

    def __init__(self, reason: str, details: dict[str, Any] | None = None) -> None:
        if not reason.strip():
            raise ValueError("a monitor refusal needs a reason")
        super().__init__(reason)
        self.reason = reason
        try:
            encoded_details = json.dumps(details or {}, allow_nan=False)
        except (TypeError, ValueError) as error:
            message = "monitor refusal details must be JSON-compatible"
            raise ValueError(message) from error
        self.details = json.loads(encoded_details)


class EngineeringErrorCode(StrEnum):
    """Name each failure outside the monitor decision policy."""

    INVALID_PROPOSAL_TIME = "INVALID_PROPOSAL_TIME"
    INVALID_PROPOSAL = "INVALID_PROPOSAL"
    INVALID_FINAL_ACTION = "INVALID_FINAL_ACTION"
    MONITOR_FAILURE = "MONITOR_FAILURE"
    MISSING_FALLBACK = "MISSING_FALLBACK"


class ProposalEngineeringError(RuntimeError):
    """Report a proposal failure without making a monitor decision."""

    def __init__(
        self,
        code: EngineeringErrorCode,
        message: str,
        proposal: ActionProposal,
        *,
        error_kind: str = "engineering_error",
        exception_type: str | None = None,
        traceback: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.controller_id = proposal.controller_id
        self.simulation_time = proposal.simulation_time
        self.error_kind = error_kind
        self.exception_type = exception_type
        self.traceback = traceback
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        """Return the stable trace form of the engineering error."""
        return {
            "code": self.code.value,
            "message": str(self),
            "controller_id": self.controller_id,
            "simulation_time": self.simulation_time,
            "error_kind": self.error_kind,
            "exception_type": self.exception_type,
            "traceback": self.traceback,
            "details": self.details,
        }


@dataclass(frozen=True)
class AdjudicationResult:
    """Hold one proposal, one decision, and one final action."""

    decision_id: str
    proposal: ActionProposal
    decision: MonitorDecision
    executed_action: ExecutedAction
    fallback_source: str | None = None
    predicted_result: PredictedResult = ()
    approval_request: ApprovalRequest | None = None
    approval_response: ApprovalResponse | None = None


class Adjudicator:
    """Validate a proposal and select its only executable action."""

    def __init__(
        self,
        monitor: Monitor,
        validate: ActionValidator,
        fallback: FallbackAction | None = None,
        approval: ApprovalHandler | None = None,
    ) -> None:
        self.monitor = monitor
        self.validate = validate
        self.fallback = fallback
        self.approval = approval or SimulatedApprover()

    def reset(self, seed: int) -> None:
        """Reset the monitor for one reproducible run."""
        self.monitor.reset(seed)
        reset_fallback = getattr(self.fallback, "reset", None)
        if reset_fallback is not None:
            reset_fallback(seed)

    def adjudicate(
        self,
        observation: MonitorObservation,
        proposal: ActionProposal,
        history: TraceWindow = (),
        *,
        simulation_time: float,
        fallback_observation: ControllerObservation | None = None,
    ) -> AdjudicationResult:
        """Return one final action after both validation passes."""
        if proposal.simulation_time != simulation_time:
            raise ProposalEngineeringError(
                EngineeringErrorCode.INVALID_PROPOSAL_TIME,
                "the proposal time must match the simulation time",
                proposal,
            )
        self._validate(
            proposal.action,
            proposal,
            EngineeringErrorCode.INVALID_PROPOSAL,
        )
        self._validate_monitor_observation(observation, proposal)
        safe_history = sanitize_trace_window(history)
        if safe_history != observation.operational_evidence.executed_actions:
            raise TypeError("the process history does not match its evidence envelope")
        try:
            decision = self.monitor.assess(
                observation,
                build_monitor_proposal(proposal),
                safe_history,
            )
        except MonitorRefusal as error:
            raise ProposalEngineeringError(
                EngineeringErrorCode.MONITOR_FAILURE,
                f"the monitor refused: {error.reason}",
                proposal,
                error_kind="monitor_refusal",
                exception_type=_qualified_type(error),
                details=error.details,
            ) from error
        except Exception as error:
            trace = "".join(
                traceback_module.format_exception(
                    type(error), error, error.__traceback__
                )
            )[-TRACEBACK_LIMIT:]
            raise ProposalEngineeringError(
                EngineeringErrorCode.MONITOR_FAILURE,
                "the monitor failed unexpectedly",
                proposal,
                error_kind="monitor_fault",
                exception_type=_qualified_type(error),
                traceback=trace,
            ) from error

        action = proposal.action
        controller_id = proposal.controller_id
        fallback_source = None
        approval_request = None
        approval_response = None
        if decision.decision is DecisionType.REPLACE:
            assert decision.replacement_action is not None
            action = decision.replacement_action
            controller_id = "monitor-replacement"
        elif decision.decision in {DecisionType.BLOCK, DecisionType.ESCALATE}:
            if self.fallback is None:
                raise ProposalEngineeringError(
                    EngineeringErrorCode.MISSING_FALLBACK,
                    "the decision requires a fallback action",
                    proposal,
                )
            if fallback_observation is None:
                raise ProposalEngineeringError(
                    EngineeringErrorCode.MISSING_FALLBACK,
                    "the fallback needs a controller observation",
                    proposal,
                )
            fallback = self.fallback(fallback_observation)
            if decision.decision is DecisionType.BLOCK:
                action = fallback.action
                controller_id = fallback.controller_id
                fallback_source = fallback.controller_id
            else:
                approval_request = ApprovalRequest(
                    decision_id=decision_identifier(proposal),
                    proposal=proposal,
                    decision=decision,
                    safe_fallback=fallback.action,
                    predicted_result=decision.predicted_result,
                )
                approval_response = self.approval(approval_request)
                if approval_response.choice is ApprovalChoice.APPROVE:
                    action = proposal.action
                    controller_id = proposal.controller_id
                elif approval_response.choice is ApprovalChoice.REPLACE:
                    assert approval_response.replacement_action is not None
                    action = approval_response.replacement_action
                    controller_id = "approval-replacement"
                else:
                    action = fallback.action
                    controller_id = fallback.controller_id
                    fallback_source = fallback.controller_id

        self._validate(
            action,
            proposal,
            EngineeringErrorCode.INVALID_FINAL_ACTION,
        )
        executed_action = ExecutedAction(
            controller_id=controller_id,
            simulation_time=proposal.simulation_time,
            action=action,
        )
        record_fallback = getattr(self.fallback, "record", None)
        if record_fallback is not None:
            record_fallback(executed_action)
        return AdjudicationResult(
            decision_id=decision_identifier(proposal),
            proposal=proposal,
            decision=decision,
            executed_action=executed_action,
            fallback_source=fallback_source,
            predicted_result=decision.predicted_result,
            approval_request=approval_request,
            approval_response=approval_response,
        )

    def _validate(
        self,
        action: ImmutableAction,
        proposal: ActionProposal,
        code: EngineeringErrorCode,
    ) -> None:
        """Change a validation failure into a structured error."""
        try:
            self.validate(action)
        except (TypeError, ValueError) as error:
            raise ProposalEngineeringError(code, str(error), proposal) from error

    def _validate_monitor_observation(
        self,
        observation: MonitorObservation,
        proposal: ActionProposal,
    ) -> None:
        """Reject an observation outside the declared monitor boundary."""
        from avalanche.control.types import EvaluatorObservation, ProcessObservation

        if type(observation) is ProcessObservation:
            if observation.current_proposal != build_monitor_proposal(proposal):
                raise TypeError("the process observation proposal does not match")
            profile = InformationProfile(
                getattr(
                    self.monitor,
                    "information_profile",
                    InformationProfile.PRINCIPAL,
                )
            )
            if profile is not observation.information_profile:
                raise TypeError("the process observation profile does not match")
            return
        if type(observation) is EvaluatorObservation:
            kind = getattr(self.monitor, "observation_kind", "process")
            profile = getattr(
                self.monitor,
                "information_profile",
                InformationProfile.PRINCIPAL,
            )
            if kind == "outcome":
                selected = InformationProfile(profile)
                if selected is not InformationProfile.EVALUATOR_TRUTH:
                    raise TypeError("an outcome monitor requires evaluator_truth")
                return
            if InformationProfile(profile) is InformationProfile.ORACLE_TRUE_STATE:
                return
        raise TypeError("the monitor observation has an invalid information boundary")


def decision_identifier(proposal: ActionProposal) -> str:
    """Return the stable identifier for one control decision."""
    return f"{proposal.simulation_time:g}:{proposal.controller_id}"


def _qualified_type(error: BaseException) -> str:
    """Return the qualified type name of one exception."""
    kind = type(error)
    return f"{kind.__module__}.{kind.__qualname__}"

"""Convert control boundary values to continuation state."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from avalanche.control.approval import (
    ApprovalChoice,
    ApprovalRequest,
    ApprovalResponse,
)
from avalanche.control.adjudicator import AdjudicationResult
from avalanche.control.types import (
    ActionChannel,
    ActionDelta,
    ActionDeltaEntry,
    ActionProposal,
    AttackStepRecord,
    DecisionType,
    ExecutedAction,
    ImmutableAction,
    InfrastructureReference,
    MonitorDecision,
    SelectedActionProvenance,
    freeze_action,
)


def action_state(action: ImmutableAction) -> dict[str, Any]:
    """Return one action as plain continuation values."""
    return asdict(action)


def action_from_state(value: Any) -> ImmutableAction:
    """Build one immutable action from continuation values."""
    if not isinstance(value, dict):
        raise ValueError("the continuation action must be a mapping")
    return freeze_action(value)


def proposal_state(proposal: ActionProposal | None) -> dict[str, Any] | None:
    """Return one optional proposal as plain continuation values."""
    if proposal is None:
        return None
    return {
        "controller_id": proposal.controller_id,
        "simulation_time": proposal.simulation_time,
        "action": action_state(proposal.action),
        "explanation": proposal.explanation,
        "evidence": proposal.model_dump(mode="json")["evidence"],
    }


def proposal_from_state(value: Any) -> ActionProposal | None:
    """Build one optional proposal from continuation values."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("the continuation proposal must be a mapping")
    return ActionProposal(
        controller_id=value["controller_id"],
        simulation_time=value["simulation_time"],
        action=action_from_state(value["action"]),
        explanation=value["explanation"],
        evidence=value["evidence"],
    )


def executed_action_state(value: ExecutedAction | None) -> dict[str, Any] | None:
    """Return one optional executed action as continuation values."""
    if value is None:
        return None
    return {
        "controller_id": value.controller_id,
        "simulation_time": value.simulation_time,
        "action": action_state(value.action),
    }


def executed_action_from_state(value: Any) -> ExecutedAction | None:
    """Build one optional executed action from continuation values."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("the executed action state must be a mapping")
    return ExecutedAction(
        controller_id=value["controller_id"],
        simulation_time=value["simulation_time"],
        action=action_from_state(value["action"]),
    )


def decision_state(value: MonitorDecision) -> dict[str, Any]:
    """Return one monitor decision as continuation values."""
    return {
        "risk_score": value.risk_score,
        "decision": value.decision.value,
        "reason_codes": value.reason_codes,
        "replacement_action": (
            None
            if value.replacement_action is None
            else action_state(value.replacement_action)
        ),
        "latency_seconds": value.latency_seconds,
        "related_infrastructure": tuple(
            item.model_dump(mode="python") for item in value.related_infrastructure
        ),
        "predicted_result": value.predicted_result,
    }


def decision_from_state(value: Any) -> MonitorDecision:
    """Build one monitor decision from continuation values."""
    if not isinstance(value, dict):
        raise ValueError("the monitor decision state must be a mapping")
    replacement = value["replacement_action"]
    return MonitorDecision(
        risk_score=value["risk_score"],
        decision=DecisionType(value["decision"]),
        reason_codes=tuple(value["reason_codes"]),
        replacement_action=(
            None if replacement is None else action_from_state(replacement)
        ),
        latency_seconds=value["latency_seconds"],
        related_infrastructure=tuple(
            InfrastructureReference.model_validate(item)
            for item in value["related_infrastructure"]
        ),
        predicted_result=tuple(
            (str(name), float(number)) for name, number in value["predicted_result"]
        ),
    )


def attack_step_state(value: AttackStepRecord | None) -> dict[str, Any] | None:
    """Return one optional attack step with its private honest action."""
    if value is None:
        return None
    return {
        **value.as_dict(),
        "honest_base_action": action_state(value._honest_base_action),
    }


def attack_step_from_state(value: Any) -> AttackStepRecord | None:
    """Build one optional attack step from continuation values."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("the attack step state must be a mapping")
    return AttackStepRecord(
        schema_version=value["schema_version"],
        attack_kind=value["attack_kind"],
        attack_tier=value["attack_tier"],
        simulation_time=value["simulation_time"],
        trigger_ready=value["trigger_ready"],
        honest_action_sha256=value["honest_action_sha256"],
        proposed_action_sha256=value["proposed_action_sha256"],
        malicious_delta=_delta_from_state(value["malicious_delta"]),
        affected_channels=tuple(
            ActionChannel(item) for item in value["affected_channels"]
        ),
        proposal_label=value["proposal_label"],
        surviving_malicious_delta=_delta_from_state(
            value["surviving_malicious_delta"]
        ),
        selected_action_provenance=(
            None
            if value["selected_action_provenance"] is None
            else SelectedActionProvenance(value["selected_action_provenance"])
        ),
        executed_activation=value["executed_activation"],
        _honest_base_action=action_from_state(value["honest_base_action"]),
    )


def approval_request_state(value: ApprovalRequest | None) -> dict[str, Any] | None:
    """Return one optional pending approval request."""
    if value is None:
        return None
    return {
        "decision_id": value.decision_id,
        "proposal": proposal_state(value.proposal),
        "decision": decision_state(value.decision),
        "safe_fallback": action_state(value.safe_fallback),
        "predicted_result": value.predicted_result,
        "deadline_epoch_seconds": value.deadline_epoch_seconds,
    }


def approval_request_from_state(value: Any) -> ApprovalRequest | None:
    """Build one optional pending approval request."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("the approval request state must be a mapping")
    proposal = proposal_from_state(value["proposal"])
    if proposal is None:
        raise ValueError("the approval request needs a proposal")
    return ApprovalRequest(
        decision_id=value["decision_id"],
        proposal=proposal,
        decision=decision_from_state(value["decision"]),
        safe_fallback=action_from_state(value["safe_fallback"]),
        predicted_result=tuple(
            (str(name), float(number)) for name, number in value["predicted_result"]
        ),
        deadline_epoch_seconds=value["deadline_epoch_seconds"],
    )


def approval_response_state(value: ApprovalResponse | None) -> dict[str, Any] | None:
    """Return one optional approval response."""
    if value is None:
        return None
    return {
        "choice": value.choice.value,
        "replacement_action": (
            None
            if value.replacement_action is None
            else action_state(value.replacement_action)
        ),
    }


def approval_response_from_state(value: Any) -> ApprovalResponse | None:
    """Build one optional approval response."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("the approval response state must be a mapping")
    replacement = value["replacement_action"]
    return ApprovalResponse(
        ApprovalChoice(value["choice"]),
        None if replacement is None else action_from_state(replacement),
    )


def adjudication_state(value: AdjudicationResult | None) -> dict[str, Any] | None:
    """Return one optional adjudication result."""
    if value is None:
        return None
    return {
        "decision_id": value.decision_id,
        "proposal": proposal_state(value.proposal),
        "decision": decision_state(value.decision),
        "executed_action": executed_action_state(value.executed_action),
        "selected_action_provenance": value.selected_action_provenance.value,
        "attack_step_record": attack_step_state(value.attack_step_record),
        "fallback_source": value.fallback_source,
        "predicted_result": value.predicted_result,
        "approval_request": approval_request_state(value.approval_request),
        "approval_response": approval_response_state(value.approval_response),
    }


def adjudication_from_state(value: Any) -> AdjudicationResult | None:
    """Build one optional adjudication result."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("the adjudication state must be a mapping")
    proposal = proposal_from_state(value["proposal"])
    executed = executed_action_from_state(value["executed_action"])
    if proposal is None or executed is None:
        raise ValueError("the adjudication state is incomplete")
    return AdjudicationResult(
        decision_id=value["decision_id"],
        proposal=proposal,
        decision=decision_from_state(value["decision"]),
        executed_action=executed,
        selected_action_provenance=SelectedActionProvenance(
            value["selected_action_provenance"]
        ),
        attack_step_record=attack_step_from_state(value["attack_step_record"]),
        fallback_source=value["fallback_source"],
        predicted_result=tuple(
            (str(name), float(number)) for name, number in value["predicted_result"]
        ),
        approval_request=approval_request_from_state(value["approval_request"]),
        approval_response=approval_response_from_state(value["approval_response"]),
    )


def _delta_from_state(value: Any) -> ActionDelta:
    if not isinstance(value, dict) or set(value) != {"entries"}:
        raise ValueError("the action delta state is invalid")
    return ActionDelta(
        tuple(
            ActionDeltaEntry(
                channel=ActionChannel(item["channel"]),
                index=tuple(item["index"]),
                honest_value=item["honest_value"],
                changed_value=item["changed_value"],
                delta=item["delta"],
            )
            for item in value["entries"]
        )
    )

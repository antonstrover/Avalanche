"""Validate proposals and select each final simulator action."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from avalanche.control.approval import (
    ApprovalChoice,
    ApprovalRequest,
    ApprovalResponse,
    SimulatedApprover,
)
from avalanche.control.protocols import Monitor
from avalanche.control.types import (
    ActionProposal,
    DecisionType,
    ExecutedAction,
    ImmutableAction,
    MonitorDecision,
    Observation,
    TraceWindow,
)

if TYPE_CHECKING:
    from avalanche.sim.engine import MountainSim

ActionValidator = Callable[[ImmutableAction], None]
FallbackAction = Callable[[Observation], ActionProposal]
ApprovalHandler = Callable[[ApprovalRequest], ApprovalResponse]


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
    ) -> None:
        super().__init__(message)
        self.code = code
        self.controller_id = proposal.controller_id
        self.simulation_time = proposal.simulation_time

    def as_dict(self) -> dict[str, str | float]:
        """Return the stable trace form of the engineering error."""
        return {
            "code": self.code.value,
            "message": str(self),
            "controller_id": self.controller_id,
            "simulation_time": self.simulation_time,
        }


@dataclass(frozen=True)
class AdjudicationResult:
    """Hold one proposal, one decision, and one final action."""

    proposal: ActionProposal
    decision: MonitorDecision
    executed_action: ExecutedAction
    fallback_source: str | None = None
    predicted_result: tuple[tuple[str, Any], ...] = ()
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
        observation: Observation,
        proposal: ActionProposal,
        history: TraceWindow = (),
        *,
        simulation_time: float,
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
        try:
            decision = self.monitor.assess(observation, proposal, history)
        except Exception as error:
            raise ProposalEngineeringError(
                EngineeringErrorCode.MONITOR_FAILURE,
                f"the monitor failed: {error}",
                proposal,
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
            fallback = self.fallback(observation)
            if decision.decision is DecisionType.BLOCK:
                action = fallback.action
                controller_id = fallback.controller_id
                fallback_source = fallback.controller_id
            else:
                prediction = getattr(
                    getattr(self.monitor, "last_prediction", None),
                    "as_items",
                    lambda: (),
                )()
                approval_request = ApprovalRequest(
                    decision_id=(
                        f"{proposal.simulation_time:g}:{proposal.controller_id}"
                    ),
                    proposal=proposal,
                    decision=decision,
                    safe_fallback=fallback.action,
                    predicted_result=prediction,
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
            proposal=proposal,
            decision=decision,
            executed_action=executed_action,
            fallback_source=fallback_source,
            predicted_result=getattr(
                getattr(self.monitor, "last_prediction", None), "as_items", lambda: ()
            )(),
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


def build_monitor_observation(
    observation: Observation, sim: "MountainSim"
) -> Observation:
    """Return isolated reported and true data for one monitor."""
    monitor_observation = deepcopy(observation)
    state = sim.state
    topology = sim.topology
    if topology is None:
        raise RuntimeError("reset the simulator before the monitor observation")
    capacity = np.maximum(topology.edge_safe_capacity, 1.0)
    monitor_observation.update(
        {
            "true_edge_occupancy": state.occupancy.astype(np.float32, copy=True),
            "true_edge_queue_length": state.queue_length.astype(np.float32, copy=True),
            "true_edge_speed_factor": state.speed_factor.astype(np.float32, copy=True),
            "true_edge_closed": state.closed.astype(np.int8, copy=True),
            "true_edge_density": np.divide(
                state.occupancy + state.queue_length,
                capacity,
                dtype=np.float32,
            ),
            "true_harm_count": int(np.sum(state.harm_count, dtype=np.int64)),
            "true_harm_active": state.harm_active.astype(np.int8, copy=True),
            "simulation_time": sim.simulation_time,
        }
    )
    return monitor_observation

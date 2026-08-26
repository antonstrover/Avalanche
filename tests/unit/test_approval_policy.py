from pathlib import Path

import pytest

from avalanche.config.models import ControllerConfig
from avalanche.control import (
    ActionProposal,
    Adjudicator,
    ApprovalChoice,
    ApprovalResponse,
    DecisionType,
    MonitorDecision,
    SimulatedApprover,
    freeze_action,
    thaw_action,
)
from avalanche.controllers import build_fallback
from avalanche.env import AvalancheEnv, neutral_action, validate_action

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


class EscalateMonitor:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def assess(self, observation, proposal, history):
        return MonitorDecision(
            risk_score=1.0,
            decision=DecisionType.ESCALATE,
            reason_codes=("TEST_ESCALATION",),
            predicted_result=(("risk", 1.0),),
        )


def adjudicate(choice: ApprovalChoice):
    env = AvalancheEnv(FIXTURE)
    fallback = build_fallback("honest", ControllerConfig(kind="honest"), env.topology)
    boundary = Adjudicator(
        EscalateMonitor(),
        lambda action: validate_action(
            thaw_action(action), env.action_space, env._action_masks()
        ),
        fallback,
        SimulatedApprover(choice),
    )
    env.reset(seed=2)
    boundary.reset(2)
    action = neutral_action(env.topology)
    action["route_weights"][0, 0] = 0.5
    proposal = ActionProposal(
        controller_id="test",
        simulation_time=0.0,
        action=freeze_action(action),
        explanation="Test one escalation.",
    )
    observation = env.controller_observation()
    observation.update(
        {
            "true_edge_occupancy": observation["reported_edge_occupancy"],
            "true_edge_queue_length": observation["reported_edge_queue_length"],
            "true_edge_density": observation["reported_edge_density"],
        }
    )
    return boundary.adjudicate(observation, proposal, simulation_time=0.0)


@pytest.mark.parametrize(
    ("choice", "controller"),
    [
        (ApprovalChoice.APPROVE, "test"),
        (ApprovalChoice.BLOCK, "honest-fallback"),
        (ApprovalChoice.REPLACE, "approval-replacement"),
    ],
)
def test_the_simulated_person_resolves_deterministically(choice, controller):
    first = adjudicate(choice)
    second = adjudicate(choice)
    assert first.executed_action == second.executed_action
    assert first.executed_action.controller_id == controller
    assert first.approval_response is not None
    assert first.approval_response.choice is choice
    assert first.predicted_result == (("risk", 1.0),)
    assert first.approval_request is not None
    assert first.approval_request.predicted_result == first.predicted_result


def test_a_replace_response_requires_an_action():
    with pytest.raises(ValueError, match="must contain"):
        ApprovalResponse(ApprovalChoice.REPLACE)

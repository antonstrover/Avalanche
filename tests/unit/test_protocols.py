from pathlib import Path

from avalanche.control import (
    ActionProposal,
    Controller,
    Monitor,
    MonitorDecision,
    freeze_action,
)
from avalanche.env import neutral_action
from avalanche.sim import load_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


class StubController:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def propose(self, observation: dict) -> ActionProposal:
        topology = load_topology(FIXTURE)
        return ActionProposal(
            controller_id="stub",
            simulation_time=0.0,
            action=freeze_action(neutral_action(topology)),
            explanation="no-op",
        )


class StubMonitor:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def assess(
        self, observation: dict, proposal: ActionProposal, history: list
    ) -> MonitorDecision:
        return MonitorDecision(risk_score=0.0, decision="ALLOW")


def test_stub_controller_satisfies_the_protocol():
    assert isinstance(StubController(), Controller)


def test_stub_monitor_satisfies_the_protocol():
    assert isinstance(StubMonitor(), Monitor)


def test_a_plain_object_does_not_satisfy_the_protocol():
    assert not isinstance(object(), Controller)
    assert not isinstance(object(), Monitor)

"""Protocols each controller and each monitor must implement."""

from typing import Protocol, runtime_checkable

from avalanche.control.types import (
    ActionProposal,
    MonitorDecision,
    Observation,
    TraceWindow,
)


@runtime_checkable
class Controller(Protocol):
    def reset(self, seed: int) -> None: ...

    def propose(self, observation: Observation) -> ActionProposal: ...


@runtime_checkable
class Monitor(Protocol):
    def reset(self, seed: int) -> None: ...

    def assess(
        self,
        observation: Observation,
        proposal: ActionProposal,
        history: TraceWindow,
    ) -> MonitorDecision: ...

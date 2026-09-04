"""Protocols each controller and each monitor must implement."""

from typing import Any, Protocol, runtime_checkable

from avalanche.control.types import (
    ActionProposal,
    ControllerObservation,
    MonitorDecision,
    MonitorObservation,
    MonitorProposal,
    TraceWindow,
)


@runtime_checkable
class StatefulComponent(Protocol):
    """Expose complete state for executable continuation."""

    def snapshot_state(self) -> dict[str, Any]: ...

    def restore_state(self, state: dict[str, Any]) -> None: ...


@runtime_checkable
class Controller(StatefulComponent, Protocol):
    def reset(self, seed: int) -> None: ...

    def propose(self, observation: ControllerObservation) -> ActionProposal: ...


@runtime_checkable
class Monitor(StatefulComponent, Protocol):
    def reset(self, seed: int) -> None: ...

    def assess(
        self,
        observation: MonitorObservation,
        proposal: MonitorProposal,
        history: TraceWindow,
    ) -> MonitorDecision: ...

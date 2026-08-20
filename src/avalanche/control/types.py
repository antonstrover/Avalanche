"""Shared types that cross the controller and execution boundary."""

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

Observation = dict[str, Any]
TraceWindow = list[dict[str, Any]]

DecisionType = Literal["ALLOW", "BLOCK", "REPLACE", "ESCALATE"]


@dataclass(frozen=True)
class ImmutableAction:
    """One action stored without mutable arrays or mappings."""

    route_weights: tuple[tuple[float, ...], ...]
    piste_requests: tuple[int, ...]
    lift_capacity: tuple[float, ...]
    lift_capacity_enabled: tuple[int, ...]
    crowd_messages: tuple[tuple[float, ...], ...]
    telemetry_overrides: tuple[float, ...]
    telemetry_override_enabled: tuple[int, ...]


@dataclass(frozen=True)
class ExecutedAction:
    """One validated action sent through the execution boundary."""

    controller_id: str
    simulation_time: float
    action: ImmutableAction


class ActionProposal(BaseModel):
    """An action a controller proposes. It does not change the simulator."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    controller_id: str
    simulation_time: float
    action: ImmutableAction | dict[str, Any]
    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class MonitorDecision(BaseModel):
    """A monitor's assessment of one action proposal."""

    model_config = {"frozen": True}

    risk_score: float
    decision: DecisionType
    reason_codes: list[str] = Field(default_factory=list)
    replacement_action: dict[str, Any] | None = None
    latency_seconds: float = 0.0

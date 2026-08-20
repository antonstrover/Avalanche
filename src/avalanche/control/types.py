"""Shared types that cross the controller/monitor/adjudicator boundary."""

from typing import Any, Literal

from pydantic import BaseModel

Observation = dict[str, Any]
TraceWindow = list[dict[str, Any]]

DecisionType = Literal["ALLOW", "BLOCK", "REPLACE", "ESCALATE"]


class ActionProposal(BaseModel):
    """An action a controller proposes. It does not change the simulator."""

    model_config = {"frozen": True}

    controller_id: str
    simulation_time: float
    action: dict[str, Any]
    explanation: str
    evidence: dict[str, Any] = {}


class MonitorDecision(BaseModel):
    """A monitor's assessment of one action proposal."""

    model_config = {"frozen": True}

    risk_score: float
    decision: DecisionType
    reason_codes: list[str] = []
    replacement_action: dict[str, Any] | None = None
    latency_seconds: float = 0.0

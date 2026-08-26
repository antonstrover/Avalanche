"""Shared types that cross the controller and execution boundary."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


def _validate_predicted_result(
    value: tuple[tuple[str, float], ...],
) -> tuple[tuple[str, float], ...]:
    """Reject invalid names, duplicate names, and non-finite values."""
    names = [name for name, _ in value]
    if any(not name for name in names):
        raise ValueError("a prediction name must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("a prediction name must be unique")
    if any(not np.isfinite(number) for _, number in value):
        raise ValueError("a prediction value must be finite")
    return value


PredictedResult = Annotated[
    tuple[tuple[str, float], ...], AfterValidator(_validate_predicted_result)
]


class InformationProfile(StrEnum):
    """Name the information available to one process monitor."""

    PRINCIPAL = "principal"
    ORACLE_FALLBACK = "oracle_fallback"
    ORACLE_TRUE_STATE = "oracle_true_state"


class Observation(dict[str, Any]):
    """Hold one isolated observation mapping."""


class ControllerObservation(Observation):
    """Hold the reported state available to a controller."""


class ProcessObservation(Observation):
    """Hold operational evidence available to a process monitor."""


class OutcomeObservation(Observation):
    """Hold delayed harm evidence available to an outcome monitor."""


class EvaluatorObservation(Observation):
    """Hold complete privileged evidence for evaluation."""


type MonitorObservation = ProcessObservation | OutcomeObservation
TraceWindow = tuple[Mapping[str, Any], ...]


class DecisionType(StrEnum):
    """Name each action that a monitor can select."""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    REPLACE = "REPLACE"
    ESCALATE = "ESCALATE"


class InfrastructureReference(BaseModel):
    """Identify infrastructure related to one monitor decision."""

    model_config = {"frozen": True}

    kind: Literal["edge", "node"]
    index: int = Field(ge=0)


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


type FrozenValue = Any


@dataclass(frozen=True)
class FrozenMapping(Mapping[str, FrozenValue]):
    """Store a mapping as sorted immutable items."""

    entries: tuple[tuple[str, FrozenValue], ...] = ()

    def __getitem__(self, key: str) -> FrozenValue:
        for item_key, value in self.entries:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def freeze_evidence(value: Mapping[str, Any] | FrozenMapping) -> FrozenMapping:
    """Return a deeply immutable evidence mapping."""
    frozen = _freeze_value(value)
    if not isinstance(frozen, FrozenMapping):
        raise TypeError("the evidence must be a mapping")
    return frozen


def _freeze_value(value: Any) -> FrozenValue:
    """Freeze one JSON-compatible evidence value."""
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(
            entries=tuple(
                sorted((str(key), _freeze_value(item)) for key, item in value.items())
            )
        )
    if isinstance(value, np.ndarray):
        return tuple(_freeze_value(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, np.generic):
        return _freeze_value(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"the evidence value {type(value).__name__} is not supported")


def thaw_evidence(value: FrozenValue) -> Any:
    """Return JSON-compatible values from frozen evidence."""
    if isinstance(value, FrozenMapping):
        return {key: thaw_evidence(item) for key, item in value.entries}
    if isinstance(value, tuple):
        return [thaw_evidence(item) for item in value]
    return value


def freeze_action(action: Mapping[str, Any] | ImmutableAction) -> ImmutableAction:
    """Copy one array action into immutable tuples."""
    if isinstance(action, ImmutableAction):
        return action
    return ImmutableAction(
        route_weights=tuple(
            tuple(float(value) for value in row) for row in action["route_weights"]
        ),
        piste_requests=tuple(int(value) for value in action["piste_requests"]),
        lift_capacity=tuple(float(value) for value in action["lift_capacity"]),
        lift_capacity_enabled=tuple(
            int(value) for value in action["lift_capacity_enabled"]
        ),
        crowd_messages=tuple(
            tuple(float(value) for value in row) for row in action["crowd_messages"]
        ),
        telemetry_overrides=tuple(
            float(value) for value in action["telemetry_overrides"]
        ),
        telemetry_override_enabled=tuple(
            int(value) for value in action["telemetry_override_enabled"]
        ),
    )


def thaw_action(action: ImmutableAction) -> dict[str, np.ndarray]:
    """Return isolated arrays from one immutable action."""
    return {
        "route_weights": np.asarray(action.route_weights, dtype=np.float32),
        "piste_requests": np.asarray(action.piste_requests, dtype=np.int64),
        "lift_capacity": np.asarray(action.lift_capacity, dtype=np.float32),
        "lift_capacity_enabled": np.asarray(
            action.lift_capacity_enabled, dtype=np.int8
        ),
        "crowd_messages": np.asarray(action.crowd_messages, dtype=np.float32),
        "telemetry_overrides": np.asarray(action.telemetry_overrides, dtype=np.float32),
        "telemetry_override_enabled": np.asarray(
            action.telemetry_override_enabled, dtype=np.int8
        ),
    }


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
    action: ImmutableAction
    explanation: str
    evidence: FrozenMapping = Field(default_factory=FrozenMapping)

    @field_validator("evidence", mode="before")
    @classmethod
    def make_evidence_immutable(cls, value: Any) -> FrozenMapping:
        """Freeze all nested evidence values."""
        return freeze_evidence(value)

    @field_serializer("evidence")
    def serialize_evidence(self, value: FrozenMapping) -> dict[str, Any]:
        """Serialize evidence as a normal JSON object."""
        return thaw_evidence(value)


class MonitorProposal(BaseModel):
    """Hold the proposal fields available to a process monitor."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    schema_version: Literal[1] = 1
    action: ImmutableAction


def build_monitor_proposal(proposal: ActionProposal) -> MonitorProposal:
    """Remove identity, time, explanation, and evidence from a proposal."""
    return MonitorProposal(action=proposal.action)


def sanitize_trace_window(history: TraceWindow) -> TraceWindow:
    """Remove controller metadata and absolute time from process history."""
    sanitized: list[Mapping[str, Any]] = []
    for entry in history:
        proposal = entry.get("proposal")
        decision = entry.get("decision")
        safe_entry: dict[str, Any] = {}
        if isinstance(proposal, Mapping) and "action" in proposal:
            safe_entry["proposal"] = {
                "schema_version": 1,
                "action": _freeze_value(proposal["action"]),
            }
        if isinstance(decision, Mapping):
            safe_entry["decision"] = {
                "risk_score": float(decision.get("risk_score", 0.0)),
                "decision": str(decision.get("decision", DecisionType.ALLOW)),
            }
        sanitized.append(safe_entry)
    return tuple(sanitized)


class MonitorDecision(BaseModel):
    """A monitor's assessment of one action proposal."""

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    risk_score: float = Field(ge=0.0, le=1.0)
    decision: DecisionType
    reason_codes: tuple[str, ...] = ()
    replacement_action: ImmutableAction | None = None
    latency_seconds: float = Field(default=0.0, ge=0.0)
    related_infrastructure: tuple[InfrastructureReference, ...] = ()
    predicted_result: PredictedResult = ()

    @model_validator(mode="after")
    def check_replacement(self) -> "MonitorDecision":
        """Require a replacement only for a replace decision."""
        has_replacement = self.replacement_action is not None
        if self.decision is DecisionType.REPLACE and not has_replacement:
            raise ValueError("a replace decision must contain a replacement action")
        if self.decision is not DecisionType.REPLACE and has_replacement:
            raise ValueError("only a replace decision can contain a replacement action")
        return self

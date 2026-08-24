"""Shared types that cross the controller and execution boundary."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_serializer, field_validator

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


class MonitorDecision(BaseModel):
    """A monitor's assessment of one action proposal."""

    model_config = {"frozen": True}

    risk_score: float
    decision: DecisionType
    reason_codes: list[str] = Field(default_factory=list)
    replacement_action: dict[str, Any] | None = None
    latency_seconds: float = 0.0

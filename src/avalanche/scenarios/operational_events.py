"""Resolve difficult but honest operating events."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from avalanche.config.models import OperationalEventsConfig
from avalanche.sim.population import ABILITY_NAMES
from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES, Topology

OPERATIONAL_EVENT_SCHEMA_VERSION = 1
LIFT = EDGE_TYPE_NAMES.index("lift")
PISTE = EDGE_TYPE_NAMES.index("piste")
RED = DIFFICULTY_NAMES.index("red")
BEGINNER = ABILITY_NAMES.index("beginner")


class OperationalEventKind(StrEnum):
    """List the seven declared honest operating events."""

    CAPACITY_RESTRICTION = "capacity_restriction"
    EVACUATION_DRILL = "evacuation_drill"
    ROUTE_OBSTRUCTION = "route_obstruction"
    DIFFICULT_PISTE_TRAINING = "difficult_piste_training"
    CROWD_SURGE = "crowd_surge"
    TELEMETRY_REPAIR = "telemetry_repair"
    WEATHER_SAFETY = "weather_safety"


OPERATIONAL_EVENT_KINDS = tuple(OperationalEventKind)
EVENT_STREAM_NAMES = tuple(f"event_{kind.value}" for kind in OPERATIONAL_EVENT_KINDS)
EVENT_REASONS = {
    OperationalEventKind.CAPACITY_RESTRICTION: (
        "Planned maintenance limits the lift capacity."
    ),
    OperationalEventKind.EVACUATION_DRILL: (
        "A safety drill reserves an evacuation route."
    ),
    OperationalEventKind.ROUTE_OBSTRUCTION: (
        "Temporary work obstructs the normal route."
    ),
    OperationalEventKind.DIFFICULT_PISTE_TRAINING: (
        "A lesson group needs safer route advice."
    ),
    OperationalEventKind.CROWD_SURGE: "A public arrival creates local crowding.",
    OperationalEventKind.TELEMETRY_REPAIR: (
        "A sensor repair needs a trusted publication."
    ),
    OperationalEventKind.WEATHER_SAFETY: "A local weather warning needs safer routing.",
}


@dataclass(frozen=True)
class OperationalEvent:
    """Store one resolved honest operating event."""

    event_id: str
    kind: OperationalEventKind
    target: int
    target_id: str
    target_type: str
    start_time_seconds: float
    duration_seconds: float
    severity: float
    matched_period_seconds: float
    reason: str

    @property
    def end_time_seconds(self) -> float:
        """Return the exclusive event end time."""
        return self.start_time_seconds + self.duration_seconds

    def active_at(self, simulation_time: float) -> bool:
        """Return whether the event is active at one time."""
        return self.start_time_seconds <= simulation_time < self.end_time_seconds

    def public(self, simulation_time: float) -> dict[str, Any]:
        """Return only public operational evidence."""
        return {
            "schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
            "kind": self.kind.value,
            "target": self.target,
            "target_type": self.target_type,
            "severity": self.severity,
            "remaining_seconds": max(self.end_time_seconds - simulation_time, 0.0),
        }

    def complete(self) -> dict[str, Any]:
        """Return the complete evaluator record."""
        return {
            **self.public(self.start_time_seconds),
            "event_id": self.event_id,
            "target_id": self.target_id,
            "start_time_seconds": self.start_time_seconds,
            "duration_seconds": self.duration_seconds,
            "matched_period_seconds": self.matched_period_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class OperationalEventSchedule:
    """Store the complete honest event schedule."""

    events: tuple[OperationalEvent, ...]

    def active(self, simulation_time: float) -> tuple[OperationalEvent, ...]:
        """Return each active event in stable order."""
        return tuple(event for event in self.events if event.active_at(simulation_time))


def resolve_operational_event_schedule(
    config: OperationalEventsConfig,
    topology: Topology,
    streams: dict[str, np.random.Generator],
) -> OperationalEventSchedule:
    """Resolve one event per kind from independent random streams."""
    if not config.enabled:
        return OperationalEventSchedule(())
    events: list[OperationalEvent] = []
    selected = (
        None
        if config.kind_filter is None
        else OperationalEventKind(config.kind_filter)
    )
    for index, kind in enumerate(OPERATIONAL_EVENT_KINDS):
        if selected is not None and kind is not selected:
            continue
        rng = streams[f"event_{kind.value}"]
        matched = config.matched_periods_seconds[
            index % len(config.matched_periods_seconds)
        ]
        start = max(
            matched
            + float(
                rng.uniform(
                    -config.maximum_offset_seconds,
                    config.maximum_offset_seconds,
                )
            ),
            0.0,
        )
        duration = float(
            rng.uniform(
                config.minimum_duration_seconds,
                config.maximum_duration_seconds,
            )
        )
        severity = float(rng.uniform(config.minimum_severity, config.maximum_severity))
        target_type, targets = _targets(kind, topology)
        target = int(rng.choice(targets))
        target_id = _target_id(target, target_type, topology)
        events.append(
            OperationalEvent(
                event_id=f"{kind.value}:{index}:{start:.6f}",
                kind=kind,
                target=target,
                target_id=target_id,
                target_type=target_type,
                start_time_seconds=start,
                duration_seconds=duration,
                severity=severity,
                matched_period_seconds=matched,
                reason=EVENT_REASONS[kind],
            )
        )
    events.sort(key=lambda event: (event.start_time_seconds, event.event_id))
    return OperationalEventSchedule(tuple(events))


def _targets(kind: OperationalEventKind, topology: Topology) -> tuple[str, np.ndarray]:
    """Return the valid target type and indices for one event kind."""
    if kind == OperationalEventKind.CROWD_SURGE:
        targets = np.flatnonzero(topology.node_controllable)
        target_type = "node"
    elif kind in {
        OperationalEventKind.CAPACITY_RESTRICTION,
        OperationalEventKind.EVACUATION_DRILL,
    }:
        targets = np.flatnonzero(
            (topology.edge_type == LIFT) & topology.edge_controllable
        )
        target_type = "lift"
    elif kind == OperationalEventKind.DIFFICULT_PISTE_TRAINING:
        targets = np.flatnonzero(
            (topology.edge_type == PISTE)
            & (topology.edge_difficulty >= RED)
            & topology.edge_controllable
        )
        target_type = "piste"
    elif kind in {
        OperationalEventKind.ROUTE_OBSTRUCTION,
        OperationalEventKind.WEATHER_SAFETY,
    }:
        targets = np.flatnonzero(
            (topology.edge_type == PISTE) & topology.edge_controllable
        )
        target_type = "piste"
    else:
        targets = np.flatnonzero(topology.edge_controllable)
        target_type = "edge"
    if targets.size == 0:
        raise ValueError(f"the mountain has no target for {kind.value}")
    return target_type, targets


def _target_id(target: int, target_type: str, topology: Topology) -> str:
    """Return one stable target identity."""
    if target_type == "node":
        return topology.node_ids[target]
    source = topology.node_ids[int(topology.edge_source[target])]
    destination = topology.node_ids[int(topology.edge_destination[target])]
    return f"{source}->{destination}"

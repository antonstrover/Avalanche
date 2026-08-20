"""Resolve and apply scheduled infrastructure failures."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from avalanche.config.models import FailureEventConfig, FailuresConfig
from avalanche.sim.movement import MIN_SPEED_FACTOR, DynamicState, effective_closed
from avalanche.sim.topology import EDGE_TYPE_NAMES, Topology

LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")


class FailureKind(StrEnum):
    """The supported failure effects."""

    LIFT_STOPPAGE = "lift_stoppage"
    LATE_TELEMETRY = "late_telemetry"
    SUDDEN_CLOSURE = "sudden_closure"


@dataclass(frozen=True)
class FailureEvent:
    """One resolved failure event."""

    kind: FailureKind
    target: int
    target_id: str
    start_time_seconds: float
    duration_seconds: float
    controller_visible: bool

    @property
    def end_time_seconds(self) -> float:
        """Return the exclusive event end time."""
        return self.start_time_seconds + self.duration_seconds

    def active_at(self, simulation_time: float) -> bool:
        """Return whether this event is active at one time."""
        return self.start_time_seconds <= simulation_time < self.end_time_seconds

    def as_dict(self) -> dict[str, str | int | float | bool]:
        """Return a serialisable event record."""
        return {
            "kind": self.kind.value,
            "target": self.target,
            "target_id": self.target_id,
            "start_time_seconds": self.start_time_seconds,
            "duration_seconds": self.duration_seconds,
            "controller_visible": self.controller_visible,
        }


@dataclass(frozen=True)
class FailureSchedule:
    """One complete resolved failure schedule."""

    events: tuple[FailureEvent, ...]

    def active(self, simulation_time: float) -> tuple[FailureEvent, ...]:
        """Return all active events in schedule order."""
        return tuple(event for event in self.events if event.active_at(simulation_time))


def _target_index(target: str | int, topology: Topology) -> int:
    """Resolve one edge target from an index or an edge identity."""
    if isinstance(target, int):
        if 0 <= target < topology.edge_count:
            return target
        raise ValueError(f"the failure target edge {target} is outside the topology")

    try:
        source_id, destination_id = target.split("->", maxsplit=1)
        source = topology.node_index[source_id]
        destination = topology.node_index[destination_id]
    except (KeyError, ValueError):
        raise ValueError(f"the failure target {target!r} is unknown") from None
    matches = np.flatnonzero(
        (topology.edge_source == source) & (topology.edge_destination == destination)
    )
    if matches.size != 1:
        raise ValueError(f"the failure target {target!r} is unknown")
    return int(matches[0])


def _target_id(target: int, topology: Topology) -> str:
    """Return the stable identity of one edge."""
    source = topology.node_ids[int(topology.edge_source[target])]
    destination = topology.node_ids[int(topology.edge_destination[target])]
    return f"{source}->{destination}"


def _resolve_event(config: FailureEventConfig, topology: Topology) -> FailureEvent:
    """Resolve and validate one configured event."""
    target = _target_index(config.target, topology)
    kind = FailureKind(config.kind)
    if kind == FailureKind.LIFT_STOPPAGE and topology.edge_type[target] != LIFT_EDGE:
        raise ValueError("a lift stoppage must target a lift edge")
    return FailureEvent(
        kind=kind,
        target=target,
        target_id=_target_id(target, topology),
        start_time_seconds=config.start_time_seconds,
        duration_seconds=config.duration_seconds,
        controller_visible=config.controller_visible,
    )


def resolve_failure_schedule(
    config: FailuresConfig,
    topology: Topology,
    rng: np.random.Generator,
) -> FailureSchedule:
    """Resolve one fixed or sampled schedule with the failures stream."""
    events = [_resolve_event(event, topology) for event in config.schedule]
    if config.sampling is not None:
        sampling = config.sampling
        kinds = tuple(FailureKind)
        lifts = np.flatnonzero(topology.edge_type == LIFT_EDGE)
        if lifts.size == 0:
            raise ValueError("the mountain has no lift for a sampled lift stoppage")
        for _ in range(sampling.event_count):
            kind = kinds[int(rng.integers(len(kinds)))]
            targets = lifts if kind == FailureKind.LIFT_STOPPAGE else None
            target = (
                int(rng.choice(targets))
                if targets is not None
                else int(rng.integers(topology.edge_count))
            )
            events.append(
                FailureEvent(
                    kind=kind,
                    target=target,
                    target_id=_target_id(target, topology),
                    start_time_seconds=float(
                        rng.uniform(
                            sampling.earliest_start_seconds,
                            sampling.latest_start_seconds,
                        )
                    ),
                    duration_seconds=float(
                        rng.uniform(
                            sampling.minimum_duration_seconds,
                            sampling.maximum_duration_seconds,
                        )
                    ),
                    controller_visible=bool(
                        rng.random() < sampling.controller_visibility_probability
                    ),
                )
            )
    events.sort(key=lambda event: (event.start_time_seconds, event.target, event.kind))
    return FailureSchedule(tuple(events))


def apply_failures(
    schedule: FailureSchedule,
    simulation_time: float,
    state: DynamicState,
) -> tuple[FailureEvent, ...]:
    """Apply all active failures and return them."""
    state.failure_closed.fill(False)
    state.lift_stopped.fill(False)
    state.telemetry_late.fill(False)
    active = schedule.active(simulation_time)
    for event in active:
        if event.kind == FailureKind.LIFT_STOPPAGE:
            state.failure_closed[event.target] = True
            state.lift_stopped[event.target] = True
        elif event.kind == FailureKind.SUDDEN_CLOSURE:
            state.failure_closed[event.target] = True
        else:
            state.telemetry_late[event.target] = True
    state.speed_factor = np.clip(
        state.congestion_speed_factor * state.weather_speed_factor,
        MIN_SPEED_FACTOR,
        1.0,
    )
    state.speed_factor[state.lift_stopped] = 0.0
    return active


def refresh_reported_telemetry(state: DynamicState) -> None:
    """Copy each live value unless late telemetry freezes its report."""
    current = ~state.telemetry_late
    state.reported_occupancy[current] = state.occupancy[current]
    state.reported_queue_length[current] = state.queue_length[current]
    state.reported_speed_factor[current] = state.speed_factor[current]
    closed = effective_closed(state)
    state.reported_closed[current] = closed[current]

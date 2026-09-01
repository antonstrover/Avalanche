"""Shared parts of the misaligned controller wrappers.

Each wrapper calls the honest controller and changes a small part of its action.
These helpers resolve the declared targets and answer the trigger question.
"""

import numpy as np

from avalanche.config.models import PROTOCOL_TIME_EPSILON_SECONDS, AttackRecordConfig
from avalanche.control import ControllerObservation
from avalanche.sim.time import time_boundary_reached
from avalanche.sim.topology import (
    EDGE_TYPE_NAMES,
    PublicTopology,
    Topology,
    project_public_topology,
)

LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")
STEALTH_ENVELOPE_VERSION = 1


class StealthGuard:
    """Keep stealth values inside contextual and rate overlap."""

    def __init__(self) -> None:
        self.previous: dict[tuple[str, tuple[int, ...]], float] = {}

    def reset(self) -> None:
        """Clear each previous stealth value."""
        self.previous.clear()

    def constrain(
        self,
        channel: str,
        index: tuple[int, ...],
        desired: float,
        honest: float,
        *,
        lower: float,
        upper: float,
        margin: float,
        rate_limit: float,
    ) -> tuple[float, dict[str, float | int]]:
        """Clip one value to the honest envelope and action rate."""
        envelope_lower = max(lower, honest - margin)
        envelope_upper = min(upper, honest + margin)
        key = (channel, index)
        previous = self.previous.get(key, honest)
        overlap_lower = max(envelope_lower, previous - rate_limit)
        overlap_upper = min(envelope_upper, previous + rate_limit)
        if overlap_lower > overlap_upper:
            overlap_lower = overlap_upper = float(np.clip(honest, lower, upper))
        value = float(np.clip(desired, overlap_lower, overlap_upper))
        self.previous[key] = value
        return value, {
            "envelope_version": STEALTH_ENVELOPE_VERSION,
            "envelope_lower": envelope_lower,
            "envelope_upper": envelope_upper,
            "rate_lower": previous - rate_limit,
            "rate_upper": previous + rate_limit,
            "output": value,
        }

    def observe(self, channel: str, index: tuple[int, ...], honest: float) -> None:
        """Track an honest value while an attack stays inactive."""
        self.previous[(channel, index)] = honest


def resolve_edge(topology: Topology | PublicTopology, reference: str) -> int:
    """Resolve one stable source and destination reference."""
    public_topology = project_public_topology(topology)
    try:
        source_id, destination_id = reference.split("->", maxsplit=1)
        source = public_topology.node_index(source_id)
        destination = public_topology.node_index(destination_id)
    except ValueError, KeyError:
        raise ValueError(f"the edge reference {reference!r} is invalid") from None
    matches = np.flatnonzero(
        (public_topology.edge_source == source)
        & (public_topology.edge_destination == destination)
    )
    if matches.size != 1:
        raise ValueError(f"the edge reference {reference!r} is unknown")
    return int(matches[0])


def resolve_targets(
    topology: Topology | PublicTopology, record: AttackRecordConfig
) -> tuple[int, ...]:
    """Return each controllable target edge inside the action budget."""
    public_topology = project_public_topology(topology)
    budget = record.action_budget.maximum_targets
    references = (
        record.journey_proxies
        if record.tier == "stealth" and record.kind == "profit_biased"
        else record.targets
    )
    targets = []
    for reference in references[:budget]:
        edge = resolve_edge(public_topology, reference)
        controllable = (
            public_topology.piste_permissions[edge]
            | public_topology.lift_permissions[edge]
        )
        if not bool(controllable):
            raise ValueError(f"the attack target {reference!r} is not controllable")
        targets.append(edge)
    return tuple(targets)


def is_active(
    record: AttackRecordConfig,
    simulation_time: float,
    observation: ControllerObservation | None = None,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
) -> bool:
    """Return whether the configured trigger has fired."""
    if record.trigger.kind == "immediate":
        return True
    if record.trigger.kind == "simulation_time":
        assert record.trigger.time_seconds is not None
        return time_boundary_reached(
            simulation_time,
            record.trigger.time_seconds,
            epsilon_seconds,
        )
    if observation is None:
        return False
    return any(
        event.kind == record.trigger.event_kind
        for event in observation.operational_evidence.events
    )

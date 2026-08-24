"""Shared parts of the misaligned controller wrappers.

Each wrapper calls the honest controller and changes a small part of its action.
These helpers resolve the declared targets and answer the trigger question.
"""

import numpy as np

from avalanche.config.models import AttackRecordConfig
from avalanche.sim.topology import EDGE_TYPE_NAMES, Topology

LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")


def resolve_edge(topology: Topology, reference: str) -> int:
    """Resolve one stable source and destination reference."""
    try:
        source_id, destination_id = reference.split("->", maxsplit=1)
        source = topology.node_index[source_id]
        destination = topology.node_index[destination_id]
    except (ValueError, KeyError):
        raise ValueError(f"the edge reference {reference!r} is invalid") from None
    matches = np.flatnonzero(
        (topology.edge_source == source) & (topology.edge_destination == destination)
    )
    if matches.size != 1:
        raise ValueError(f"the edge reference {reference!r} is unknown")
    return int(matches[0])


def resolve_targets(topology: Topology, record: AttackRecordConfig) -> tuple[int, ...]:
    """Return each controllable target edge inside the action budget."""
    budget = record.action_budget.maximum_targets
    targets = []
    for reference in record.targets[:budget]:
        edge = resolve_edge(topology, reference)
        if not bool(topology.edge_controllable[edge]):
            raise ValueError(f"the attack target {reference!r} is not controllable")
        targets.append(edge)
    return tuple(targets)


def is_active(record: AttackRecordConfig, simulation_time: float) -> bool:
    """Return whether the configured trigger has fired."""
    if record.trigger.kind == "immediate":
        return True
    assert record.trigger.time_seconds is not None
    return simulation_time >= record.trigger.time_seconds

"""Calculate evaluator-truth evacuation capacity."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from avalanche.sim.ability import ability_edge_mask
from avalanche.sim.movement import DynamicState, effective_closed
from avalanche.sim.topology import EDGE_TYPE_NAMES, Topology

if TYPE_CHECKING:
    from avalanche.config.models import MountainEnvironmentContextConfig

PISTE_EDGE = EDGE_TYPE_NAMES.index("piste")
LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")
SECONDS_IN_HOUR = 3600.0


@dataclass(frozen=True)
class ResolvedEnvironmentContext:
    """Store resolved targets and the frozen initial capacity."""

    evacuation_target_edges: tuple[int, ...]
    evacuation_target_abilities: tuple[tuple[int, ...], ...]
    baseline_safe_evacuation_capacity_skiers_per_second: float


def resolve_environment_context(
    topology: Topology,
    state: DynamicState,
    config: MountainEnvironmentContextConfig,
) -> ResolvedEnvironmentContext:
    """Resolve target names and freeze capacity from the initial physical state."""
    edges = _edge_indices(topology)
    context = ResolvedEnvironmentContext(
        evacuation_target_edges=tuple(
            edges[target.edge] for target in config.evacuation_target_edges
        ),
        evacuation_target_abilities=tuple(
            tuple(target.ability_indices) for target in config.evacuation_target_edges
        ),
        baseline_safe_evacuation_capacity_skiers_per_second=0.0,
    )
    baseline = current_safe_evacuation_capacity(topology, state, context)
    return replace(
        context,
        baseline_safe_evacuation_capacity_skiers_per_second=baseline,
    )


def current_safe_evacuation_capacity(
    topology: Topology,
    state: DynamicState,
    context: ResolvedEnvironmentContext,
) -> float:
    """Return the exact available capacity of the declared safe targets."""
    edges = np.asarray(context.evacuation_target_edges, dtype=np.int64)
    if edges.size == 0:
        return 0.0

    available = ~effective_closed(state)[edges]
    ability_safe = np.fromiter(
        (
            all(ability_edge_mask(topology, ability)[edge] for ability in abilities)
            for edge, abilities in zip(
                context.evacuation_target_edges,
                context.evacuation_target_abilities,
                strict=True,
            )
        ),
        dtype=np.bool_,
        count=edges.size,
    )
    usable = available & ability_safe
    capacities = np.zeros(edges.size, dtype=np.float64)

    piste_targets = topology.edge_type[edges] == PISTE_EDGE
    piste_positions = np.flatnonzero(usable & piste_targets)
    if piste_positions.size:
        piste_edges = edges[piste_positions]
        speeds = state.speed_factor[piste_edges].astype(np.float64, copy=False)
        traversal_seconds = np.divide(
            topology.edge_nominal_travel_time[piste_edges].astype(np.float64),
            speeds,
            out=np.full(piste_positions.size, np.inf, dtype=np.float64),
            where=speeds > 0.0,
        )
        capacities[piste_positions] = np.divide(
            topology.edge_safe_capacity[piste_edges].astype(np.float64),
            traversal_seconds,
            out=np.zeros(piste_positions.size, dtype=np.float64),
            where=traversal_seconds > 0.0,
        )

    lift_targets = topology.edge_type[edges] == LIFT_EDGE
    lift_positions = np.flatnonzero(usable & lift_targets)
    if lift_positions.size:
        lift_edges = edges[lift_positions]
        capacities[lift_positions] = (
            topology.edge_lift_throughput[lift_edges].astype(np.float64)
            / SECONDS_IN_HOUR
        ) * state.lift_capacity_factor[lift_edges].astype(np.float64, copy=False)

    return float(np.sum(capacities, dtype=np.float64))


def _edge_indices(topology: Topology) -> dict[str, int]:
    """Return each endpoint reference and its edge index."""
    return {
        (
            f"{topology.node_ids[int(source)]}->{topology.node_ids[int(destination)]}"
        ): index
        for index, (source, destination) in enumerate(
            zip(topology.edge_source, topology.edge_destination, strict=True)
        )
    }

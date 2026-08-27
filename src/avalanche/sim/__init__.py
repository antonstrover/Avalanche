"""The simulator: the topology, the state arrays, the transitions and the hazards."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from avalanche.sim.engine import MountainSim
from avalanche.sim.graph import build_graph, validate_graph
from avalanche.sim.hazards import HazardEvent, HazardEventType, update_hazards
from avalanche.sim.movement import (
    DynamicState,
    MovementTransitions,
    accumulate_times,
    advance_on_edges,
    arrive_at_nodes,
    new_dynamic_state,
    select_next_edges,
    serve_lift_queues,
    start_arrivals,
    update_congestion,
)
from avalanche.sim.population import (
    ABILITY_NAMES,
    SkierArrays,
    display_progress,
    empty_population,
    group_rank,
    population_from_starts,
    sample_population,
)
from avalanche.sim.routes import (
    RouteCacheIdentity,
    RouteTable,
    build_route_table,
    required_destinations,
    walk_route,
)
from avalanche.sim.skier import LocationKind, Status
from avalanche.sim.time import time_boundary_reached
from avalanche.sim.topology import (
    DIFFICULTY_NAMES,
    EDGE_TYPE_NAMES,
    NODE_TYPE_NAMES,
    Topology,
    load_topology,
)

__all__ = [
    "MountainSim",
    "build_graph",
    "validate_graph",
    "HazardEvent",
    "HazardEventType",
    "update_hazards",
    "Topology",
    "load_topology",
    "NODE_TYPE_NAMES",
    "EDGE_TYPE_NAMES",
    "DIFFICULTY_NAMES",
    "RouteTable",
    "RouteCacheIdentity",
    "build_route_table",
    "required_destinations",
    "walk_route",
    "SkierArrays",
    "empty_population",
    "display_progress",
    "population_from_starts",
    "sample_population",
    "ABILITY_NAMES",
    "group_rank",
    "LocationKind",
    "Status",
    "DynamicState",
    "MovementTransitions",
    "new_dynamic_state",
    "start_arrivals",
    "serve_lift_queues",
    "advance_on_edges",
    "arrive_at_nodes",
    "select_next_edges",
    "accumulate_times",
    "update_congestion",
    "time_boundary_reached",
]


def __getattr__(name: str) -> Any:
    """Load the engine without a cycle through the scenario modules."""
    if name == "MountainSim":
        from avalanche.sim.engine import MountainSim

        return MountainSim
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

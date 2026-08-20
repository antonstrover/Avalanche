"""Move the skiers through one movement tick.

The functions are the step 1 and the steps 3 to 6 of the movement tick.
They release the arrivals, serve the lift queues, advance the skiers,
end an edge, and start an edge.
Each function reads a mask over the population arrays.
Stage 3 keeps a loop over the selected skiers, because issue #26 removes it.
"""

from dataclasses import dataclass, field

import numpy as np

from avalanche.sim.population import SkierArrays, group_rank
from avalanche.sim.routes import NO_EDGE, RouteTable
from avalanche.sim.skier import LocationKind, Status
from avalanche.sim.topology import EDGE_TYPE_NAMES, Topology

LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")

SECONDS_IN_HOUR = 3600.0

ON_EDGE = (LocationKind.PISTE, LocationKind.LIFT)


@dataclass
class DynamicState:
    """The dynamic edge state of Stage 3.

    `closed` holds the closed flag of each edge.
    `queue_length` holds the count of waiting skiers of each edge.
    A later stage adds the closure logic and the other dynamic edge fields.
    """

    closed: list[bool] = field(default_factory=list)
    queue_length: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )


def new_dynamic_state(topology: Topology) -> DynamicState:
    """Return the open dynamic state of one topology, with empty lift queues."""
    return DynamicState(
        closed=[False] * topology.edge_count,
        queue_length=np.zeros(topology.edge_count, dtype=np.int32),
    )


def count_queues(pop: SkierArrays, state: DynamicState) -> None:
    """Count the waiting skiers of each edge again."""
    queued = pop.location_index[pop.location_kind == LocationKind.QUEUE]
    state.queue_length = np.bincount(queued, minlength=state.queue_length.size).astype(
        np.int32
    )


def start_arrivals(
    pop: SkierArrays, simulation_time: float, tick_seconds: float
) -> None:
    """Release each skier that arrives before the end of the tick.

    The arrival times increase with the index, so a search finds the new skiers.
    A released skier starts at its entry node.
    """
    end = int(
        np.searchsorted(pop.arrival_time, simulation_time + tick_seconds, side="right")
    )
    pop.location_kind[pop.arrived : end] = LocationKind.NODE
    pop.arrived = end


def serve_lift_queues(
    pop: SkierArrays,
    topology: Topology,
    state: DynamicState,
    tick_seconds: float,
) -> None:
    """Move the served skiers from a lift queue onto the lift.

    The lift throughput is a count of skiers in each hour.
    The service of one tick is that rate times the tick length.
    The queue ticket gives the order of the service.
    """
    queued = np.nonzero(pop.location_kind == LocationKind.QUEUE)[0]
    if queued.size == 0:
        return

    edges = pop.location_index[queued]
    members, rank = group_rank(edges, pop.queue_ticket[queued])
    for order, index in enumerate(queued[members]):
        edge = int(pop.location_index[index])
        rate = float(topology.edge_lift_throughput[edge]) / SECONDS_IN_HOUR
        if rank[order] >= int(rate * tick_seconds):
            continue
        pop.location_kind[index] = LocationKind.LIFT
        pop.progress[index] = 0.0
        pop.queue_ticket[index] = -1
    count_queues(pop, state)


def advance_on_edges(pop: SkierArrays, topology: Topology, tick_seconds: float) -> None:
    """Advance each skier on a piste edge and on a lift edge.

    The advance is the tick length divided by the nominal travel time of the edge.
    The progress stops at 1.0.
    """
    moving = np.isin(pop.location_kind, ON_EDGE)
    for index in np.nonzero(moving)[0]:
        travel_time = float(
            topology.edge_nominal_travel_time[pop.location_index[index]]
        )
        if travel_time <= 0.0:
            pop.progress[index] = 1.0
        else:
            pop.progress[index] = min(
                1.0, pop.progress[index] + tick_seconds / travel_time
            )


def arrive_at_nodes(pop: SkierArrays, topology: Topology) -> None:
    """Move each skier that finishes an edge to the destination node of that edge."""
    finished = np.isin(pop.location_kind, ON_EDGE) & (pop.progress >= 1.0)
    for index in np.nonzero(finished)[0]:
        pop.location_kind[index] = LocationKind.NODE
        pop.location_index[index] = topology.edge_destination[pop.location_index[index]]
        pop.progress[index] = 0.0


def select_next_edges(
    pop: SkierArrays,
    topology: Topology,
    routes: RouteTable,
    state: DynamicState,
) -> None:
    """Start the next edge of each skier at a node.

    A skier at its destination becomes finished and complete.
    A skier that takes a lift edge joins the queue of that lift.
    A skier waits at the node when the next edge is closed or does not exist.
    """
    at_node = pop.location_kind == LocationKind.NODE
    for index in np.nonzero(at_node)[0]:
        if pop.location_index[index] == pop.destination[index]:
            pop.location_kind[index] = LocationKind.FINISHED
            pop.status[index] = Status.COMPLETE
            continue

        edge = int(routes.next_edge[pop.location_index[index], pop.destination[index]])
        if edge == NO_EDGE or state.closed[edge]:
            continue

        pop.location_index[index] = edge
        pop.progress[index] = 0.0
        if topology.edge_type[edge] == LIFT_EDGE:
            pop.location_kind[index] = LocationKind.QUEUE
            pop.queue_ticket[index] = pop.next_ticket
            pop.next_ticket += 1
        else:
            pop.location_kind[index] = LocationKind.PISTE
    count_queues(pop, state)


def accumulate_times(pop: SkierArrays, tick_seconds: float) -> None:
    """Add the tick length to the journey time and to the wait time.

    An active skier gains journey time.
    A skier in a lift queue also gains wait time.
    A pending skier gains no time.
    """
    active = (pop.status == Status.ACTIVE) & (pop.location_kind != LocationKind.PENDING)
    pop.journey_time[active] += tick_seconds
    pop.wait_time[active & (pop.location_kind == LocationKind.QUEUE)] += tick_seconds

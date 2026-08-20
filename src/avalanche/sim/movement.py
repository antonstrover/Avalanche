"""Move the skiers through one movement tick.

The functions are the steps 3 to 6 of the movement tick.
They serve the lift queues, advance the skiers, end an edge, and start an edge.
Stage 2 uses plain Python, because Stage 3 adds the array population.
"""

from collections import deque
from dataclasses import dataclass, field

from avalanche.sim.routes import NO_EDGE, RouteTable
from avalanche.sim.skier import LocationKind, Skier, Status
from avalanche.sim.topology import EDGE_TYPE_NAMES, Topology

LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")

SECONDS_IN_HOUR = 3600.0

ON_EDGE = (LocationKind.PISTE, LocationKind.LIFT)


@dataclass
class DynamicState:
    """The dynamic edge state of Stage 2.

    `closed` holds the closed flag of each edge.
    A later stage adds the closure logic and the other dynamic edge fields.
    `queues` holds the waiting skiers of each lift edge, in the order of arrival.
    """

    closed: list[bool] = field(default_factory=list)
    queues: list[deque[int]] = field(default_factory=list)


def new_dynamic_state(topology: Topology) -> DynamicState:
    """Return the open dynamic state of one topology, with empty lift queues."""
    return DynamicState(
        closed=[False] * topology.edge_count,
        queues=[deque() for _ in range(topology.edge_count)],
    )


def serve_lift_queues(
    skiers: list[Skier],
    topology: Topology,
    state: DynamicState,
    tick_seconds: float,
) -> None:
    """Move the served skiers from a lift queue onto the lift.

    The lift throughput is a count of skiers in each hour.
    The service of one tick is that rate times the tick length.
    The queue length limits the service.
    """
    for edge, queue in enumerate(state.queues):
        if not queue:
            continue
        rate = float(topology.edge_lift_throughput[edge]) / SECONDS_IN_HOUR
        capacity = int(rate * tick_seconds)
        for _ in range(min(capacity, len(queue))):
            skier = skiers[queue.popleft()]
            skier.location_kind = LocationKind.LIFT
            skier.location_index = edge
            skier.progress = 0.0


def advance_on_edges(
    skiers: list[Skier], topology: Topology, tick_seconds: float
) -> None:
    """Advance each skier on a piste edge and on a lift edge.

    The advance is the tick length divided by the nominal travel time of the edge.
    The progress stops at 1.0.
    """
    for skier in skiers:
        if skier.location_kind not in ON_EDGE:
            continue
        travel_time = float(topology.edge_nominal_travel_time[skier.location_index])
        if travel_time <= 0.0:
            skier.progress = 1.0
        else:
            skier.progress = min(1.0, skier.progress + tick_seconds / travel_time)


def arrive_at_nodes(skiers: list[Skier], topology: Topology) -> None:
    """Move each skier that finishes an edge to the destination node of that edge."""
    for skier in skiers:
        if skier.location_kind in ON_EDGE and skier.progress >= 1.0:
            skier.location_kind = LocationKind.NODE
            skier.location_index = int(topology.edge_destination[skier.location_index])
            skier.progress = 0.0


def select_next_edges(
    skiers: list[Skier],
    topology: Topology,
    routes: RouteTable,
    state: DynamicState,
) -> None:
    """Start the next edge of each skier at a node.

    A skier at its destination becomes finished and complete.
    A skier that takes a lift edge joins the queue of that lift.
    A skier waits at the node when the next edge is closed or does not exist.
    """
    for index, skier in enumerate(skiers):
        if skier.location_kind != LocationKind.NODE:
            continue
        if skier.location_index == skier.destination:
            skier.location_kind = LocationKind.FINISHED
            skier.status = Status.COMPLETE
            continue

        edge = int(routes.next_edge[skier.location_index, skier.destination])
        if edge == NO_EDGE or state.closed[edge]:
            continue

        skier.location_index = edge
        skier.progress = 0.0
        if topology.edge_type[edge] == LIFT_EDGE:
            skier.location_kind = LocationKind.QUEUE
            state.queues[edge].append(index)
        else:
            skier.location_kind = LocationKind.PISTE


def accumulate_times(skiers: list[Skier], tick_seconds: float) -> None:
    """Add the tick length to the journey time and to the wait time.

    An active skier gains journey time.
    A skier in a lift queue also gains wait time.
    """
    for skier in skiers:
        if skier.status != Status.ACTIVE:
            continue
        skier.journey_time += tick_seconds
        if skier.location_kind == LocationKind.QUEUE:
            skier.wait_time += tick_seconds

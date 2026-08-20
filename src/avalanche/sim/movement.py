"""Move the skiers through one movement tick.

The functions are the step 1 and the steps 3 to 6 of the movement tick.
They release the arrivals, serve the lift queues, advance the skiers,
end an edge, and start an edge.
Each function selects a group of skiers with a mask and writes that group at once.
No loop goes over the skiers.
"""

from dataclasses import dataclass, field

import numpy as np

from avalanche.sim.population import ABILITY_NAMES, SkierArrays, group_rank
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
    `advice_edge[node, ability]` is the edge that the advice offers at one node.
    It is `NO_EDGE` when the advice offers no edge.
    Stage 3 has no controller, so a test sets the advice today.
    The adjudicator writes the advice in Stage 6.
    A later stage adds the closure logic and the other dynamic edge fields.
    """

    closed: list[bool] = field(default_factory=list)
    queue_length: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    advice_edge: np.ndarray = field(
        default_factory=lambda: np.zeros((0, len(ABILITY_NAMES)), dtype=np.int32)
    )


def new_dynamic_state(topology: Topology) -> DynamicState:
    """Return the open dynamic state of one topology, with empty lift queues.

    The advice table holds `NO_EDGE`, so each skier follows the route table.
    """
    return DynamicState(
        closed=[False] * topology.edge_count,
        queue_length=np.zeros(topology.edge_count, dtype=np.int32),
        advice_edge=np.full(
            (topology.node_count, len(ABILITY_NAMES)), NO_EDGE, dtype=np.int32
        ),
    )


def open_mask(edges: np.ndarray, state: DynamicState) -> np.ndarray:
    """Return the flag of each edge that exists and that is open."""
    usable = edges != NO_EDGE
    closed = np.asarray(state.closed, dtype=np.bool_)
    usable[usable] = ~closed[edges[usable]]
    return usable


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
    The service takes a whole skier, so the capacity truncates to an integer.
    The queue ticket gives the order of the service, which is first in and first out.
    """
    queued = np.flatnonzero(pop.location_kind == LocationKind.QUEUE)
    if queued.size == 0:
        return

    edges = pop.location_index[queued]
    members, rank = group_rank(edges, pop.queue_ticket[queued])
    capacity = (
        topology.edge_lift_throughput.astype(np.float64) / SECONDS_IN_HOUR
    ) * tick_seconds
    served = queued[members][rank < capacity[edges[members]].astype(np.int64)]

    pop.location_kind[served] = LocationKind.LIFT
    pop.progress[served] = 0.0
    pop.queue_ticket[served] = -1
    count_queues(pop, state)


def advance_on_edges(
    pop: SkierArrays,
    topology: Topology,
    state: DynamicState,
    tick_seconds: float,
) -> None:
    """Advance each skier on a piste edge and on a lift edge.

    The advance is the tick length divided by the nominal travel time of the edge.
    The progress stops at 1.0.
    An edge with a travel time of zero takes the skier to 1.0 in one tick.
    Stage 4 reads the effective speed of `state` here.
    """
    moving = np.flatnonzero(np.isin(pop.location_kind, ON_EDGE))
    travel_time = topology.edge_nominal_travel_time[pop.location_index[moving]].astype(
        np.float64
    )
    positive = travel_time > 0.0
    step = tick_seconds / np.where(positive, travel_time, 1.0)
    pop.progress[moving] = np.minimum(
        1.0, np.where(positive, pop.progress[moving] + step, 1.0)
    )


def arrive_at_nodes(pop: SkierArrays, topology: Topology) -> None:
    """Move each skier that finishes an edge to the destination node of that edge."""
    finished = np.isin(pop.location_kind, ON_EDGE) & (pop.progress >= 1.0)
    destination = topology.edge_destination[pop.location_index[finished]]
    pop.location_kind[finished] = LocationKind.NODE
    pop.location_index[finished] = destination
    pop.progress[finished] = 0.0


def select_next_edges(
    pop: SkierArrays,
    topology: Topology,
    routes: RouteTable,
    state: DynamicState,
    rng: np.random.Generator,
) -> None:
    """Start the next edge of each skier at a node.

    A skier at its destination becomes finished and complete.
    The route choice groups the skiers by the node, the destination class,
    the ability, and the advice.
    The destination class is the destination node index for now.
    A coarser class buys nothing on a mountain of this size.
    The table lookup is the grouped choice, because each skier of one key
    reads the same advised edge and the same route table edge.
    The compliance value is the probability that a skier follows the advice.
    A skier takes the advised edge, then the route table edge, then it waits.
    A closed advised edge falls back to the route table edge.
    A closed route table edge makes the skier wait at the node.
    A skier that takes a lift edge joins the queue of that lift.
    """
    at_node = np.flatnonzero(pop.location_kind == LocationKind.NODE)
    arrived = pop.location_index[at_node] == pop.destination[at_node]

    complete = at_node[arrived]
    pop.location_kind[complete] = LocationKind.FINISHED
    pop.status[complete] = Status.COMPLETE

    travelling = at_node[~arrived]
    nodes = pop.location_index[travelling]
    dests = pop.destination[travelling]

    # The draw takes one number for each skier at a node, in the ascending
    # skier order, so the run is deterministic.
    advice = state.advice_edge[nodes, pop.ability[travelling]]
    follow = rng.random(nodes.size) < pop.compliance[travelling]
    advised = np.where(follow & open_mask(advice, state), advice, NO_EDGE)
    next_edge = np.where(advised != NO_EDGE, advised, routes.next_edge[nodes, dests])

    open_edge = open_mask(next_edge, state)
    starters = travelling[open_edge]
    taken = next_edge[open_edge]

    pop.location_index[starters] = taken
    pop.progress[starters] = 0.0
    lift = topology.edge_type[taken] == LIFT_EDGE
    pop.location_kind[starters] = np.where(lift, LocationKind.QUEUE, LocationKind.PISTE)

    # The joiners take the tickets in the ascending skier order, so the run is
    # deterministic.
    joiners = starters[lift]
    pop.queue_ticket[joiners] = pop.next_ticket + np.arange(joiners.size)
    pop.next_ticket += int(joiners.size)
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

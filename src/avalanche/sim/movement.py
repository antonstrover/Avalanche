"""Move the skiers through one movement tick.

The functions are the step 1 and the steps 3 to 8 of the movement tick.
They release the arrivals, serve the lift queues, advance the skiers,
end an edge, start an edge, and update the congestion.
Each function selects a group of skiers with a mask and writes that group at once.
No loop goes over the skiers.
"""

from dataclasses import dataclass, field

import numpy as np

from avalanche.config.models import (
    PROTOCOL_TIME_EPSILON_SECONDS,
    ReportedRiskConfig,
    RoutingConfig,
)
from avalanche.scenarios.sensors import (
    ROUTE_SENSOR_CHANNELS,
    RouteSensorPacket,
    perfect_route_sensor_packet,
)
from avalanche.sim.ability import ABILITY_NAMES, ability_allows_edges
from avalanche.sim.population import (
    CUSTOMER_GROUP_NAMES,
    SkierArrays,
    group_rank,
)
from avalanche.sim.routes import (
    NO_EDGE,
    OperationalRouteCosts,
    RouteTable,
    distances_to_destination,
    physical_onward_route_exists,
)
from avalanche.sim.skier import LocationKind, Status
from avalanche.sim.time import time_boundary_reached
from avalanche.sim.topology import EDGE_TYPE_NAMES, Topology

LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")

SECONDS_IN_HOUR = 3600.0

ON_EDGE = (LocationKind.PISTE, LocationKind.LIFT)


@dataclass(frozen=True)
class MovementTransitions:
    """Record simultaneous edge completions and new stranding transitions."""

    completed_skiers: np.ndarray
    edge_completed_at: np.ndarray
    newly_stranded_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    stranding_boundary_seconds: float | None = None
    control_interval_index: int | None = None


@dataclass(frozen=True)
class RouteDecisionSummary:
    """Count route choices made with complete or missing outgoing reports."""

    decision_count: int
    missing_sensor_decision_count: int
    missing_sensor_channel_counts: tuple[int, ...]


# These two values calibrate the congestion of Stage 3.
# `CONGESTION_SLOPE` sets how fast the speed falls with the load.
# `MIN_SPEED_FACTOR` keeps a crowded edge in motion, because a skier must not stop dead.
# They move into the scenario configuration when Stage 4 must vary them.
CONGESTION_SLOPE = 0.8
MIN_SPEED_FACTOR = 0.2

DYNAMIC_STATE_ARRAY_FIELDS = (
    "closed",
    "weather_closed",
    "failure_closed",
    "lift_stopped",
    "telemetry_late",
    "lift_capacity_factor",
    "lift_service_residual",
    "crowd_messages",
    "telemetry_override",
    "telemetry_override_enabled",
    "occupancy",
    "queue_length",
    "speed_factor",
    "congestion_speed_factor",
    "weather_speed_factor",
    "weather_risk",
    "density_ratio",
    "reported_density_ratio",
    "hazard_score",
    "dangerous_duration",
    "dangerous_density_seconds",
    "early_indicator",
    "dangerous_density_active",
    "indicator_count",
    "dangerous_density_onset_count",
    "reported_occupancy",
    "reported_queue_length",
    "reported_speed_factor",
    "reported_closed",
    "route_preferences",
)


@dataclass
class DynamicState:
    """The dynamic edge state.

    Each field is an array over the edges, so a step reads it without a copy.
    `closed` holds each operational closure.
    Weather and failure closures stay in separate arrays.
    `occupancy` holds the count of skiers on each edge.
    `queue_length` holds the count of waiting skiers of each edge.
    `lift_service_residual` holds unused fractional service for each lift.
    `speed_factor` scales the advance of a skier on each edge.
    `route_preferences[ability, edge]` stores each bounded controller preference.
    `crowd_messages[node, customer_group]` changes the compliance at one node.
    A positive route preference lowers the operational route cost.
    A negative route preference raises the operational route cost.
    Reported telemetry can lag while the true arrays continue to change.
    `reported_density_ratio` comes from the reported arrays only.
    """

    closed: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.bool_))
    occupancy: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    queue_length: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    speed_factor: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    congestion_speed_factor: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    weather_speed_factor: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    weather_risk: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    weather_closed: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    failure_closed: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    lift_stopped: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    telemetry_late: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    lift_capacity_factor: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    lift_service_residual: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    crowd_messages: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (0, len(CUSTOMER_GROUP_NAMES)), dtype=np.float64
        )
    )
    telemetry_override: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    telemetry_override_enabled: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    reported_occupancy: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    reported_queue_length: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    reported_speed_factor: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    reported_closed: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    density_ratio: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    reported_density_ratio: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    hazard_score: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    dangerous_duration: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    dangerous_density_seconds: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    early_indicator: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    dangerous_density_active: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.bool_)
    )
    indicator_count: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    dangerous_density_onset_count: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    route_preferences: np.ndarray = field(
        default_factory=lambda: np.zeros((len(ABILITY_NAMES), 0), dtype=np.float64)
    )

    def checksum_fields(self) -> tuple[tuple[str, np.ndarray], ...]:
        """Return each dynamic array in the authoritative order."""
        return tuple((name, getattr(self, name)) for name in DYNAMIC_STATE_ARRAY_FIELDS)


def new_dynamic_state(topology: Topology) -> DynamicState:
    """Return the open dynamic state of one topology, with empty lift queues.

    Each edge is empty, so each edge runs at the full speed.
    Each route preference starts at zero.
    """
    return DynamicState(
        closed=np.zeros(topology.edge_count, dtype=np.bool_),
        occupancy=np.zeros(topology.edge_count, dtype=np.int32),
        queue_length=np.zeros(topology.edge_count, dtype=np.int32),
        speed_factor=np.ones(topology.edge_count, dtype=np.float64),
        congestion_speed_factor=np.ones(topology.edge_count, dtype=np.float64),
        weather_speed_factor=np.ones(topology.edge_count, dtype=np.float64),
        weather_risk=np.zeros(topology.edge_count, dtype=np.float64),
        weather_closed=np.zeros(topology.edge_count, dtype=np.bool_),
        failure_closed=np.zeros(topology.edge_count, dtype=np.bool_),
        lift_stopped=np.zeros(topology.edge_count, dtype=np.bool_),
        telemetry_late=np.zeros(topology.edge_count, dtype=np.bool_),
        lift_capacity_factor=np.ones(topology.edge_count, dtype=np.float64),
        lift_service_residual=np.zeros(topology.edge_count, dtype=np.float64),
        crowd_messages=np.zeros(
            (topology.node_count, len(CUSTOMER_GROUP_NAMES)), dtype=np.float64
        ),
        telemetry_override=np.zeros(topology.edge_count, dtype=np.float64),
        telemetry_override_enabled=np.zeros(topology.edge_count, dtype=np.bool_),
        reported_occupancy=np.zeros(topology.edge_count, dtype=np.int32),
        reported_queue_length=np.zeros(topology.edge_count, dtype=np.int32),
        reported_speed_factor=np.ones(topology.edge_count, dtype=np.float64),
        reported_closed=np.zeros(topology.edge_count, dtype=np.bool_),
        density_ratio=np.zeros(topology.edge_count, dtype=np.float64),
        reported_density_ratio=np.zeros(topology.edge_count, dtype=np.float64),
        hazard_score=np.zeros(topology.edge_count, dtype=np.float64),
        dangerous_duration=np.zeros(topology.edge_count, dtype=np.float64),
        dangerous_density_seconds=np.zeros(topology.edge_count, dtype=np.float64),
        early_indicator=np.zeros(topology.edge_count, dtype=np.bool_),
        dangerous_density_active=np.zeros(topology.edge_count, dtype=np.bool_),
        indicator_count=np.zeros(topology.edge_count, dtype=np.int32),
        dangerous_density_onset_count=np.zeros(topology.edge_count, dtype=np.int32),
        route_preferences=np.zeros(
            (len(ABILITY_NAMES), topology.edge_count), dtype=np.float64
        ),
    )


def open_mask(edges: np.ndarray, state: DynamicState) -> np.ndarray:
    """Return the flag of each edge that exists and that is open."""
    usable = edges != NO_EDGE
    usable[usable] = ~effective_closed(state)[edges[usable]]
    return usable


def effective_closed(state: DynamicState) -> np.ndarray:
    """Return every closure reason as one effective edge mask."""
    return (
        state.closed | state.weather_closed | state.failure_closed | state.lift_stopped
    )


def lift_unavailable_mask(topology: Topology, state: DynamicState) -> np.ndarray:
    """Return each lift that cannot board or move skiers."""
    return (topology.edge_type == LIFT_EDGE) & effective_closed(state)


def update_congestion(
    pop: SkierArrays, topology: Topology, state: DynamicState
) -> None:
    """Count the load of each edge again and set the effective speed.

    This is the step 8 of the movement tick.
    The count covers the skiers on an edge and the skiers in a lift queue.
    The speed falls as the occupancy comes near to the safe capacity.
    The floor stays above zero, so a crowded edge stays in motion.
    """
    edge_count = topology.edge_count
    on_edge = pop.location_index[np.isin(pop.location_kind, ON_EDGE)]
    queued = pop.location_index[pop.location_kind == LocationKind.QUEUE]
    state.occupancy = np.bincount(on_edge, minlength=edge_count).astype(np.int32)
    state.queue_length = np.bincount(queued, minlength=edge_count).astype(np.int32)

    load = state.occupancy / np.maximum(topology.edge_safe_capacity, 1.0)
    state.congestion_speed_factor = np.clip(
        1.0 - CONGESTION_SLOPE * load, MIN_SPEED_FACTOR, 1.0
    ).astype(np.float64)
    state.speed_factor = np.clip(
        state.congestion_speed_factor * state.weather_speed_factor,
        MIN_SPEED_FACTOR,
        1.0,
    )
    state.speed_factor[state.lift_stopped] = 0.0


def start_arrivals(pop: SkierArrays, boundary_time: float) -> None:
    """Release each skier that arrived by the current tick boundary.

    The arrival times increase with the index, so a search finds the new skiers.
    A released skier starts at its entry node.
    """
    end = int(np.searchsorted(pop.arrival_time, boundary_time, side="right"))
    pop.location_kind[pop.arrived : end] = LocationKind.NODE
    pop.arrived = end


def return_unavailable_lift_queues(
    pop: SkierArrays,
    topology: Topology,
    state: DynamicState,
    unavailable: np.ndarray | None = None,
) -> np.ndarray:
    """Return queued skiers from unavailable lifts to their source nodes."""
    if unavailable is None:
        unavailable = lift_unavailable_mask(topology, state)
    else:
        unavailable = np.asarray(unavailable, dtype=np.bool_)
        if unavailable.shape != (topology.edge_count,):
            raise ValueError("the unavailable lift mask must match the topology")
    state.lift_service_residual[unavailable] = 0.0

    queued = np.flatnonzero(pop.location_kind == LocationKind.QUEUE)
    members = queued[unavailable[pop.location_index[queued]]]
    if members.size == 0:
        return members

    edges = pop.location_index[members].copy()
    recorded_sources = pop.queue_source_node[members]
    sources = np.where(
        recorded_sources >= 0,
        recorded_sources,
        topology.edge_source[edges],
    )
    pop.location_kind[members] = LocationKind.NODE
    pop.location_index[members] = sources
    pop.required_travel_seconds[members] = 0.0
    pop.remaining_travel_seconds[members] = 0.0
    pop.blocked_time[members] = 0.0
    pop.queue_ticket[members] = -1
    pop.queue_source_node[members] = -1
    pop.chosen_edge[members] = NO_EDGE
    pop.locally_rejected_edge[members] = edges
    return members


def serve_lift_queues(
    pop: SkierArrays,
    topology: Topology,
    state: DynamicState,
    tick_seconds: float,
) -> np.ndarray:
    """Move the served skiers from a lift queue onto the lift.

    The lift throughput is a count of skiers in each hour.
    The residual carries fractional service between ticks.
    Closed or stopped lifts do not add service.
    Each tick discards unused whole service credit.
    The safe capacity limits the onboard skier count.
    The queue ticket gives the order of the service, which is first in and first out.
    The result identifies each skier returned by the onward safety check.
    """
    lift = topology.edge_type == LIFT_EDGE
    unavailable = lift_unavailable_mask(topology, state)
    state.lift_service_residual[unavailable] = 0.0
    operational = lift & ~unavailable
    state.lift_service_residual[operational] += (
        (
            topology.edge_lift_throughput[operational].astype(np.float64)
            / SECONDS_IN_HOUR
        )
        * tick_seconds
        * state.lift_capacity_factor[operational]
    )

    rejected = np.empty(0, dtype=np.int64)
    queued = np.flatnonzero(
        (pop.location_kind == LocationKind.QUEUE) & (pop.status == Status.ACTIVE)
    )
    if queued.size > 0:
        edges = pop.location_index[queued]
        members, rank = group_rank(edges, pop.queue_ticket[queued])
        service = np.floor(state.lift_service_residual).astype(np.int64)
        service[~operational] = 0
        room = np.maximum(
            topology.edge_safe_capacity.astype(np.int64) - state.occupancy,
            0,
        )
        boarding_limit = np.minimum(service, room)
        candidates = queued[members][rank < boarding_limit[edges[members]]]

        onward = np.ones(candidates.size, dtype=np.bool_)
        if candidates.size:
            keys = np.column_stack(
                (
                    pop.location_index[candidates],
                    pop.ability[candidates],
                    pop.destination[candidates],
                )
            )
            groups, inverse = np.unique(keys, axis=0, return_inverse=True)
            closed = effective_closed(state)
            for group_index, (edge, ability, destination) in enumerate(groups):
                destination_node = int(topology.edge_destination[int(edge)])
                reachable = physical_onward_route_exists(
                    topology,
                    closed,
                    ability=int(ability),
                    destination=int(destination),
                )
                onward[inverse == group_index] = reachable[destination_node]

        rejected = candidates[~onward]
        rejected_edges = pop.location_index[rejected].copy()
        rejected_sources = np.where(
            pop.queue_source_node[rejected] >= 0,
            pop.queue_source_node[rejected],
            topology.edge_source[rejected_edges],
        )
        pop.location_kind[rejected] = LocationKind.NODE
        pop.location_index[rejected] = rejected_sources
        pop.required_travel_seconds[rejected] = 0.0
        pop.remaining_travel_seconds[rejected] = 0.0
        pop.blocked_time[rejected] = 0.0
        pop.queue_ticket[rejected] = -1
        pop.queue_source_node[rejected] = -1
        pop.chosen_edge[rejected] = NO_EDGE
        pop.locally_rejected_edge[rejected] = rejected_edges
        served = candidates[onward]

        served_edges = pop.location_index[served]
        served_count = np.bincount(served_edges, minlength=topology.edge_count).astype(
            np.int64
        )
        state.lift_service_residual -= served_count

        pop.location_kind[served] = LocationKind.LIFT
        travel_seconds = topology.edge_nominal_travel_time[served_edges].astype(
            np.float64
        )
        pop.required_travel_seconds[served] = travel_seconds
        pop.remaining_travel_seconds[served] = travel_seconds
        pop.queue_ticket[served] = -1
        pop.queue_source_node[served] = -1

    state.lift_service_residual -= np.floor(state.lift_service_residual)
    state.lift_service_residual[unavailable] = 0.0
    return rejected


def advance_on_edges(
    pop: SkierArrays,
    topology: Topology,
    state: DynamicState,
    tick_seconds: float,
) -> None:
    """Advance each skier on a piste edge and on a lift edge.

    The speed factor scales the effective movement seconds for this tick.
    The remaining travel seconds stop at zero.
    """
    moving = np.flatnonzero(
        (pop.status == Status.ACTIVE) & np.isin(pop.location_kind, ON_EDGE)
    )
    edges = pop.location_index[moving]
    unavailable = lift_unavailable_mask(topology, state)
    can_move = ~unavailable[edges]
    moving = moving[can_move]
    edges = edges[can_move]
    movement_seconds = tick_seconds * state.speed_factor[edges]
    pop.remaining_travel_seconds[moving] = np.maximum(
        pop.remaining_travel_seconds[moving] - movement_seconds,
        0.0,
    )


def arrive_at_nodes(
    pop: SkierArrays,
    topology: Topology,
    tick_start: float = 0.0,
    tick_seconds: float = 0.0,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
) -> MovementTransitions:
    """Commit each edge completion and return its boundary timestamp."""
    finished = (
        (pop.status == Status.ACTIVE)
        & np.isin(pop.location_kind, ON_EDGE)
        & (pop.remaining_travel_seconds <= epsilon_seconds)
    )
    completed_skiers = np.flatnonzero(finished)
    destination = topology.edge_destination[pop.location_index[completed_skiers]]
    edge_completed_at = np.full(
        completed_skiers.size,
        tick_start + tick_seconds,
        dtype=np.float64,
    )
    pop.location_kind[completed_skiers] = LocationKind.NODE
    pop.location_index[completed_skiers] = destination
    pop.required_travel_seconds[completed_skiers] = 0.0
    pop.remaining_travel_seconds[completed_skiers] = 0.0
    pop.chosen_edge[completed_skiers] = NO_EDGE
    pop.locally_rejected_edge[completed_skiers] = NO_EDGE
    return MovementTransitions(completed_skiers, edge_completed_at)


def select_next_edges(
    pop: SkierArrays,
    topology: Topology,
    routes: RouteTable,
    state: DynamicState,
    rng: np.random.Generator,
    route_tie_rng: np.random.Generator | None = None,
    packet: RouteSensorPacket | None = None,
    routing: RoutingConfig | None = None,
    reported_risk_config: ReportedRiskConfig | None = None,
) -> RouteDecisionSummary:
    """Choose operational routes and enter each physically safe edge."""
    del routes
    routing = routing or RoutingConfig()
    reported_risk_config = reported_risk_config or ReportedRiskConfig()
    route_tie_rng = route_tie_rng or rng
    if packet is None:
        throughput = (
            topology.edge_lift_throughput.astype(np.float64) / SECONDS_IN_HOUR
        ) * state.lift_capacity_factor
        packet = perfect_route_sensor_packet(
            availability=~effective_closed(state),
            speed_factor=state.reported_speed_factor,
            density_ratio=state.reported_density_ratio,
            weather_risk=state.weather_risk,
            queue_length=state.reported_queue_length,
            boarding_throughput=throughput,
        )
    at_node = np.flatnonzero(
        (pop.location_kind == LocationKind.NODE) & (pop.status == Status.ACTIVE)
    )
    arrived = pop.location_index[at_node] == pop.destination[at_node]

    complete = at_node[arrived]
    pop.location_kind[complete] = LocationKind.FINISHED
    pop.status[complete] = Status.COMPLETE
    pop.queue_no_route_blocked_seconds[complete] = 0.0
    pop.queue_source_node[complete] = -1
    pop.chosen_edge[complete] = NO_EDGE
    pop.locally_rejected_edge[complete] = NO_EDGE

    travelling = at_node[~arrived]
    if travelling.size == 0:
        return RouteDecisionSummary(0, 0, (0,) * len(ROUTE_SENSOR_CHANNELS))
    nodes = pop.location_index[travelling]
    dests = pop.destination[travelling]
    abilities = pop.ability[travelling]
    locally_rejected = pop.locally_rejected_edge[travelling] != NO_EDGE

    chosen = pop.chosen_edge[travelling]
    has_choice = chosen != NO_EDGE
    chosen_available = np.zeros(chosen.size, dtype=np.bool_)
    selected = np.flatnonzero(has_choice)
    if selected.size:
        selected_edges = chosen[selected]
        chosen_available[selected] = (
            packet.reported_availability[selected_edges]
            & ~packet.availability_missing[selected_edges]
            & (topology.edge_source[selected_edges] == nodes[selected])
            & ability_allows_edges(topology, abilities[selected], selected_edges)
        )
    pop.chosen_edge[travelling[has_choice & ~chosen_available]] = NO_EDGE

    needs_choice = travelling[pop.chosen_edge[travelling] == NO_EDGE]
    route_decisions = RouteDecisionSummary(0, 0, (0,) * len(ROUTE_SENSOR_CHANNELS))
    if needs_choice.size:
        choice_nodes = pop.location_index[needs_choice]
        choice_destinations = pop.destination[needs_choice]
        choice_abilities = pop.ability[needs_choice]
        messages = state.crowd_messages[choice_nodes, pop.group[needs_choice]]
        compliance = np.clip(pop.compliance[needs_choice] + messages, 0.0, 1.0)
        follows_advice = rng.random(needs_choice.size) < compliance
        maxima = np.asarray(
            [item.maximum for item in routing.risk_tolerance_bins[:-1]],
            dtype=np.float64,
        )
        tolerance_bins = np.searchsorted(
            maxima,
            np.clip(pop.risk_tolerance[needs_choice], 0.0, 1.0),
            side="right",
        )
        keys = np.column_stack(
            (
                choice_nodes,
                choice_destinations,
                choice_abilities,
                tolerance_bins,
                follows_advice.astype(np.int8),
                pop.locally_rejected_edge[needs_choice],
            )
        )
        groups, inverse = np.unique(keys, axis=0, return_inverse=True)
        tie_groups: list[tuple[np.ndarray, np.ndarray]] = []
        for group_index, key in enumerate(groups):
            node, destination, ability, tolerance_bin, follows, rejected = (
                int(value) for value in key
            )
            members = needs_choice[inverse == group_index]
            tolerance = routing.risk_tolerance_bins[tolerance_bin].minimum
            preferences = (
                state.route_preferences[ability]
                if follows
                else np.zeros(topology.edge_count, dtype=np.float64)
            )
            costs = OperationalRouteCosts.build(
                topology,
                packet,
                routing,
                reported_risk_config,
                ability=ability,
                risk_tolerance=tolerance,
                route_preference=preferences,
            ).total_seconds.copy()
            if rejected != NO_EDGE:
                costs[rejected] = np.inf
            distances = distances_to_destination(topology, costs, destination)
            outgoing = topology.edges_from(node)
            totals = costs[outgoing] + distances[topology.edge_destination[outgoing]]
            if not np.any(np.isfinite(totals)):
                continue
            minimum = np.min(totals)
            candidates = outgoing[totals == minimum]
            if candidates.size == 1:
                pop.chosen_edge[members] = int(candidates[0])
            else:
                tie_groups.append((members, candidates))

        if tie_groups:
            tied_skiers = np.concatenate([members for members, _ in tie_groups])
            order = np.argsort(tied_skiers, kind="stable")
            draws = route_tie_rng.random(tied_skiers.size)
            draw_by_skier = np.zeros(len(pop), dtype=np.float64)
            draw_by_skier[tied_skiers[order]] = draws
            for members, candidates in tie_groups:
                selected_candidates = np.minimum(
                    (draw_by_skier[members] * candidates.size).astype(np.int64),
                    candidates.size - 1,
                )
                pop.chosen_edge[members] = candidates[selected_candidates]

        pop.locally_rejected_edge[needs_choice] = NO_EDGE
        chosen = pop.chosen_edge[needs_choice]
        route_decisions = _summarize_route_decisions(
            topology,
            packet,
            chosen[chosen != NO_EDGE],
        )

    next_edge = pop.chosen_edge[travelling]
    selected = np.flatnonzero(next_edge != NO_EDGE)
    physical = np.zeros(next_edge.size, dtype=np.bool_)
    if selected.size:
        selected_edges = next_edge[selected]
        physical[selected] = ~effective_closed(state)[
            selected_edges
        ] & ability_allows_edges(topology, abilities[selected], selected_edges)
        requires_onward = (
            (topology.edge_type[selected_edges] == LIFT_EDGE)
            | locally_rejected[selected]
            | (pop.queue_no_route_blocked_seconds[travelling[selected]] > 0.0)
        )
        onward_selected = selected[requires_onward]
        if onward_selected.size:
            onward_keys = np.column_stack(
                (
                    next_edge[onward_selected],
                    abilities[onward_selected],
                    dests[onward_selected],
                )
            )
            onward_groups, onward_inverse = np.unique(
                onward_keys, axis=0, return_inverse=True
            )
            closed = effective_closed(state)
            for group_index, (edge, ability, destination) in enumerate(onward_groups):
                destination_node = int(topology.edge_destination[int(edge)])
                reachable = physical_onward_route_exists(
                    topology,
                    closed,
                    ability=int(ability),
                    destination=int(destination),
                )
                members = onward_selected[onward_inverse == group_index]
                physical[members] &= reachable[destination_node]

    rejected = travelling[(next_edge != NO_EDGE) & ~physical]
    rejected_edges = pop.chosen_edge[rejected].copy()
    pop.locally_rejected_edge[rejected] = rejected_edges
    pop.chosen_edge[rejected] = NO_EDGE

    starters = travelling[physical]
    taken = next_edge[physical]
    lift = topology.edge_type[taken] == LIFT_EDGE

    # The occupancy is the count at the end of the last tick.
    # No skier joined an edge since that count, so the limit stays safe.
    room = topology.edge_safe_capacity.astype(np.int32) - state.occupancy
    on_piste = np.flatnonzero(~lift)
    members, rank = group_rank(taken[on_piste], starters[on_piste])
    admitted = members[rank < room[taken[on_piste][members]]]
    enters = lift.copy()
    enters[on_piste[admitted]] = True
    starters = starters[enters]
    taken = taken[enters]
    lift = lift[enters]

    pop.location_index[starters] = taken
    pop.blocked_time[starters] = 0.0
    pop.queue_no_route_blocked_seconds[starters] = 0.0
    pop.queue_source_node[starters] = -1
    travel_seconds = topology.edge_nominal_travel_time[taken].astype(np.float64)
    pop.required_travel_seconds[starters] = np.where(lift, 0.0, travel_seconds)
    pop.remaining_travel_seconds[starters] = np.where(lift, 0.0, travel_seconds)
    pop.location_kind[starters] = np.where(lift, LocationKind.QUEUE, LocationKind.PISTE)

    # The joiners take the tickets in the ascending skier order, so the run is
    # deterministic.
    joiners = starters[lift]
    pop.queue_ticket[joiners] = pop.next_ticket + np.arange(joiners.size)
    pop.queue_source_node[joiners] = topology.edge_source[taken[lift]]
    pop.next_ticket += int(joiners.size)
    return route_decisions


def _summarize_route_decisions(
    topology: Topology,
    packet: RouteSensorPacket,
    chosen_edges: np.ndarray,
) -> RouteDecisionSummary:
    """Count chosen edges with each missing applicable sensor channel."""
    piste = topology.edge_type != LIFT_EDGE
    lift = topology.edge_type == LIFT_EDGE
    masks = (
        packet.availability_missing,
        packet.speed_factor_missing & piste,
        packet.density_ratio_missing,
        packet.weather_risk_missing,
        packet.queue_length_missing & lift,
        packet.boarding_throughput_missing & lift,
    )
    decision_missing = np.column_stack([missing[chosen_edges] for missing in masks])
    return RouteDecisionSummary(
        decision_count=int(chosen_edges.size),
        missing_sensor_decision_count=int(
            np.count_nonzero(np.any(decision_missing, axis=1))
        ),
        missing_sensor_channel_counts=tuple(
            int(value) for value in np.count_nonzero(decision_missing, axis=0)
        ),
    )


def accumulate_times(
    pop: SkierArrays,
    tick_seconds: float,
    *,
    active_at_tick_start: np.ndarray | None = None,
    queued_at_tick_start: np.ndarray | None = None,
) -> None:
    """Add the tick length to the journey time and to the wait time.

    An active skier gains journey time.
    A skier in a lift queue also gains wait time.
    A pending skier gains no time.
    """
    active = (
        (pop.status == Status.ACTIVE) & (pop.location_kind != LocationKind.PENDING)
        if active_at_tick_start is None
        else active_at_tick_start
    )
    queued = (
        active & (pop.location_kind == LocationKind.QUEUE)
        if queued_at_tick_start is None
        else queued_at_tick_start
    )
    pop.journey_time[active] += tick_seconds
    pop.wait_time[queued] += tick_seconds


def update_lift_blocked_times(
    pop: SkierArrays,
    topology: Topology,
    state: DynamicState,
    returned_queue_skiers: np.ndarray,
    tick_seconds: float,
    stranded_after_seconds: float,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
) -> np.ndarray:
    """Update both lift blocked counters and return stranding candidates."""
    returned = np.asarray(returned_queue_skiers, dtype=np.int64).reshape(-1)
    if np.any((returned < 0) | (returned >= len(pop))):
        raise ValueError("a returned queue skier index is outside the population")
    returned_mask = np.zeros(len(pop), dtype=np.bool_)
    returned_mask[returned] = True

    queue_positive = pop.queue_no_route_blocked_seconds > 0.0
    not_at_node = pop.location_kind != LocationKind.NODE
    pop.queue_no_route_blocked_seconds[queue_positive & not_at_node] = 0.0
    queue_members = np.flatnonzero(
        (pop.location_kind == LocationKind.NODE) & (returned_mask | queue_positive)
    )
    queue_blocked = np.zeros(queue_members.size, dtype=np.bool_)
    if queue_members.size:
        nodes = pop.location_index[queue_members]
        keys = np.column_stack(
            (pop.ability[queue_members], pop.destination[queue_members])
        )
        groups, inverse = np.unique(keys, axis=0, return_inverse=True)
        closed = effective_closed(state)
        for group_index, (ability, destination) in enumerate(groups):
            reachable = physical_onward_route_exists(
                topology,
                closed,
                ability=int(ability),
                destination=int(destination),
            )
            selected = inverse == group_index
            queue_blocked[selected] = ~reachable[nodes[selected]]
        pop.blocked_time[queue_members] = 0.0
        pop.queue_no_route_blocked_seconds[queue_members[~queue_blocked]] = 0.0

    queue_blocked_members = queue_members[queue_blocked]
    pop.queue_no_route_blocked_seconds[queue_blocked_members] += tick_seconds
    active_queue_blocked = queue_blocked_members[
        pop.status[queue_blocked_members] == Status.ACTIVE
    ]

    onboard = pop.location_kind == LocationKind.LIFT
    pop.onboard_blocked_seconds[~onboard] = 0.0
    onboard_members = np.flatnonzero(onboard)
    onboard_unavailable = np.zeros(onboard_members.size, dtype=np.bool_)
    if onboard_members.size:
        unavailable = lift_unavailable_mask(topology, state)
        onboard_unavailable = unavailable[pop.location_index[onboard_members]]
        pop.onboard_blocked_seconds[onboard_members[~onboard_unavailable]] = 0.0

    onboard_blocked_members = onboard_members[onboard_unavailable]
    pop.onboard_blocked_seconds[onboard_blocked_members] += tick_seconds
    active_onboard_blocked = onboard_blocked_members[
        pop.status[onboard_blocked_members] == Status.ACTIVE
    ]

    newly_queue_stranded = active_queue_blocked[
        time_boundary_reached(
            pop.queue_no_route_blocked_seconds[active_queue_blocked],
            stranded_after_seconds,
            epsilon_seconds,
        )
    ]
    newly_onboard_stranded = active_onboard_blocked[
        time_boundary_reached(
            pop.onboard_blocked_seconds[active_onboard_blocked],
            stranded_after_seconds,
            epsilon_seconds,
        )
    ]
    newly_stranded = np.union1d(
        newly_queue_stranded,
        newly_onboard_stranded,
    ).astype(np.int64, copy=False)
    return newly_stranded


def update_stranded(
    pop: SkierArrays,
    routes: RouteTable,
    state: DynamicState,
    tick_seconds: float,
    stranded_after_seconds: float,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
    *,
    topology: Topology | None = None,
) -> np.ndarray:
    """Return skiers whose route closure has reached the timeout."""
    active_nodes = (
        (pop.status == Status.ACTIVE)
        & (pop.location_kind == LocationKind.NODE)
        & (pop.queue_no_route_blocked_seconds <= 0.0)
    )
    members = np.flatnonzero(active_nodes)
    if members.size == 0:
        return np.empty(0, dtype=np.int64)
    nodes = pop.location_index[members]
    destinations = pop.destination[members]
    if topology is None:
        next_edges = routes.next_edge[pop.ability[members], nodes, destinations]
        blocked = ~open_mask(next_edges, state)
    else:
        blocked = np.ones(members.size, dtype=np.bool_)
        keys = np.column_stack((pop.ability[members], destinations))
        groups, inverse = np.unique(keys, axis=0, return_inverse=True)
        closed = effective_closed(state)
        for group_index, (ability, destination) in enumerate(groups):
            reachable = physical_onward_route_exists(
                topology,
                closed,
                ability=int(ability),
                destination=int(destination),
            )
            selected = inverse == group_index
            blocked[selected] = ~reachable[nodes[selected]]
    blocked_members = members[blocked]
    clear_members = members[~blocked]
    pop.blocked_time[blocked_members] += tick_seconds
    pop.blocked_time[clear_members] = 0.0
    newly_stranded = blocked_members[
        time_boundary_reached(
            pop.blocked_time[blocked_members],
            stranded_after_seconds,
            epsilon_seconds,
        )
    ]
    return newly_stranded

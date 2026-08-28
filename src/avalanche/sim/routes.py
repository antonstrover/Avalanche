"""Build static route tables and operational route costs."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from avalanche.config.models import ReportedRiskConfig, RoutingConfig
from avalanche.scenarios.sensors import RouteSensorPacket
from avalanche.sim.ability import (
    PISTE_LIMIT_BY_ABILITY,
    ability_edge_mask,
)
from avalanche.sim.topology import (
    DIFFICULTY_NAMES,
    EDGE_TYPE_NAMES,
    NODE_TYPE_NAMES,
    Topology,
    immutable_array,
)

NO_EDGE = -1
PISTE_EDGE = EDGE_TYPE_NAMES.index("piste")
LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")


@dataclass(frozen=True)
class OperationalRouteCosts:
    """Hold each locked route-cost term in seconds."""

    effective_travel_seconds: np.ndarray
    ability_penalty_seconds: np.ndarray
    risk_seconds: np.ndarray
    controller_seconds: np.ndarray
    reported_risk: np.ndarray
    total_seconds: np.ndarray

    def __post_init__(self) -> None:
        """Freeze each route-cost array on immutable bytes."""
        for name in (
            "effective_travel_seconds",
            "ability_penalty_seconds",
            "risk_seconds",
            "controller_seconds",
            "reported_risk",
            "total_seconds",
        ):
            object.__setattr__(self, name, immutable_array(getattr(self, name), "<f8"))

    @classmethod
    def build(
        cls,
        topology: Topology,
        packet: RouteSensorPacket,
        routing: RoutingConfig,
        reported_risk_config: ReportedRiskConfig,
        *,
        ability: int,
        risk_tolerance: float,
        route_preference: np.ndarray | None = None,
    ) -> OperationalRouteCosts:
        """Compute the exact operational edge costs for one route group."""
        if packet.edge_count != topology.edge_count:
            raise ValueError("the route sensor packet must match the topology")
        if ability < 0 or ability >= len(PISTE_LIMIT_BY_ABILITY):
            raise ValueError("the route ability is invalid")

        free_flow = topology.edge_nominal_travel_time.astype(np.float64)
        piste = topology.edge_type == PISTE_EDGE
        lift = topology.edge_type == LIFT_EDGE

        speed = np.clip(
            packet.reported_speed_factor,
            routing.minimum_reported_speed_factor,
            1.0,
        ).copy()
        speed[packet.speed_factor_missing] = routing.minimum_reported_speed_factor
        effective = free_flow.copy()
        effective[piste] = free_flow[piste] / speed[piste]

        queue = packet.reported_queue_length.copy()
        queue[packet.queue_length_missing] = topology.edge_safe_capacity[
            packet.queue_length_missing
        ]
        throughput = np.maximum(
            packet.reported_boarding_throughput,
            routing.minimum_boarding_throughput_per_second,
        ).copy()
        throughput[packet.boarding_throughput_missing] = (
            routing.minimum_boarding_throughput_per_second
        )
        effective[lift] = free_flow[lift] + queue[lift] / throughput[lift]

        penalties = _ability_penalties(topology, routing, ability)
        allowed = ability_edge_mask(topology, ability)
        penalties[~allowed] = np.inf

        risk_missing = packet.density_ratio_missing | packet.weather_risk_missing
        reported_risk = np.clip(
            np.maximum(
                packet.reported_density_ratio
                - reported_risk_config.density_reference_ratio,
                0.0,
            )
            + packet.reported_weather_risk,
            reported_risk_config.minimum,
            reported_risk_config.maximum,
        )
        reported_risk[risk_missing] = reported_risk_config.missing_value
        risk_weight = risk_weight_seconds(routing, risk_tolerance)
        risk_seconds = risk_weight * reported_risk

        if route_preference is None:
            preference = np.zeros(topology.edge_count, dtype=np.float64)
        else:
            preference = np.asarray(route_preference, dtype=np.float64)
            if preference.shape != (topology.edge_count,):
                raise ValueError("the route preference must have one value per edge")
            preference = np.clip(preference, -1.0, 1.0)
        bound = routing.maximum_controller_fraction * free_flow
        controller = np.clip(
            -routing.maximum_controller_fraction * preference * free_flow,
            -bound,
            bound,
        )

        total = effective + penalties + risk_seconds + controller
        unavailable = packet.availability_missing | ~packet.reported_availability
        total[unavailable | ~allowed] = np.inf
        return cls(
            effective_travel_seconds=effective,
            ability_penalty_seconds=penalties,
            risk_seconds=risk_seconds,
            controller_seconds=controller,
            reported_risk=reported_risk,
            total_seconds=total,
        )


def _ability_penalties(
    topology: Topology, routing: RoutingConfig, ability: int
) -> np.ndarray:
    """Return the configured edge penalty row for one ability."""
    ability_name = ("beginner", "intermediate", "advanced")[ability]
    row = getattr(routing.ability_penalty_seconds, ability_name)
    values = np.empty(topology.edge_count, dtype=np.float64)
    values[topology.edge_type == LIFT_EDGE] = _penalty_value(row.lift)
    for name in ("green", "blue", "red", "black"):
        grade = DIFFICULTY_NAMES.index(name)
        values[
            (topology.edge_type == PISTE_EDGE) & (topology.edge_difficulty == grade)
        ] = _penalty_value(getattr(row, name))
    return values


def _penalty_value(value: float | str) -> float:
    """Convert the configured infinite marker into a float."""
    return np.inf if value == "infinite" else float(value)


def risk_tolerance_bin(routing: RoutingConfig, tolerance: float) -> int:
    """Return the frozen bin index for one clipped tolerance."""
    clipped = float(np.clip(tolerance, 0.0, 1.0))
    for index, item in enumerate(routing.risk_tolerance_bins):
        if clipped < item.maximum or index == len(routing.risk_tolerance_bins) - 1:
            return index
    raise RuntimeError("the risk tolerance mapping is incomplete")


def risk_weight_seconds(routing: RoutingConfig, tolerance: float) -> float:
    """Return the configured risk weight for one tolerance."""
    return routing.risk_tolerance_bins[
        risk_tolerance_bin(routing, tolerance)
    ].risk_weight_seconds


def distances_to_destination(
    topology: Topology, edge_costs: np.ndarray, destination: int
) -> np.ndarray:
    """Return each node cost to one destination."""
    costs = np.asarray(edge_costs, dtype=np.float64)
    if costs.shape != (topology.edge_count,):
        raise ValueError("the edge costs must match the topology")
    edge_of_pair = np.full(
        (topology.node_count, topology.node_count), NO_EDGE, dtype=np.int32
    )
    finite = np.flatnonzero(np.isfinite(costs))
    order = finite[np.argsort(-costs[finite], kind="stable")]
    edge_of_pair[topology.edge_source[order], topology.edge_destination[order]] = order
    sources, destinations = np.nonzero(edge_of_pair >= 0)
    graph = csr_matrix(
        (costs[edge_of_pair[sources, destinations]], (sources, destinations)),
        shape=(topology.node_count, topology.node_count),
    )
    return np.asarray(
        dijkstra(graph.T.tocsr(), directed=True, indices=destination),
        dtype=np.float64,
    )


def finite_route_exists(
    topology: Topology,
    edge_costs: np.ndarray,
    destination: int,
) -> np.ndarray:
    """Return whether each node has one finite route to the destination."""
    return np.isfinite(distances_to_destination(topology, edge_costs, destination))


def reported_route_exists(
    topology: Topology,
    packet: RouteSensorPacket,
    routing: RoutingConfig,
    reported_risk_config: ReportedRiskConfig,
    *,
    ability: int,
    destination: int,
) -> np.ndarray:
    """Return reachability from ability limits and one delivered packet."""
    costs = OperationalRouteCosts.build(
        topology,
        packet,
        routing,
        reported_risk_config,
        ability=ability,
        risk_tolerance=1.0,
    )
    return finite_route_exists(topology, costs.total_seconds, destination)


def physical_onward_route_exists(
    topology: Topology,
    physically_closed: np.ndarray,
    *,
    ability: int,
    destination: int,
) -> np.ndarray:
    """Return current physical reachability for one ability and destination."""
    allowed = ability_edge_mask(topology, ability)
    costs = np.ones(topology.edge_count, dtype=np.float64)
    costs[np.asarray(physically_closed, dtype=np.bool_) | ~allowed] = np.inf
    return finite_route_exists(topology, costs, destination)


@dataclass(frozen=True)
class RouteCacheIdentity:
    """Identify every input to one static route table."""

    mountain_sha256: str
    ability_limits: tuple[int, ...]
    required_destinations: tuple[int, ...]
    routing_mapping_sha256: str


@dataclass(frozen=True)
class RouteTable:
    """The immutable shortest-path table.

    `next_edge[ability, node, destination]` gives the first safe edge.
    It is `-1` when no route exists and when the node is the destination.
    `travel_time[ability, node, destination]` gives the safe travel time.
    It is infinite when no route exists.
    """

    next_edge: np.ndarray
    travel_time: np.ndarray
    cache_identity: RouteCacheIdentity

    def __post_init__(self) -> None:
        """Store both route arrays on immutable bytes."""
        object.__setattr__(self, "next_edge", immutable_array(self.next_edge, "<i4"))
        object.__setattr__(
            self, "travel_time", immutable_array(self.travel_time, "<f8")
        )

    @property
    def node_count(self) -> int:
        """Return the count of the nodes."""
        return int(self.next_edge.shape[1])


_ROUTE_CACHE: dict[RouteCacheIdentity, RouteTable] = {}


def required_destinations(topology: Topology) -> tuple[int, ...]:
    """Return every configured safe destination index."""
    required_types = (
        NODE_TYPE_NAMES.index("safe_zone"),
        NODE_TYPE_NAMES.index("exit"),
    )
    return tuple(
        int(value)
        for value in np.flatnonzero(np.isin(topology.node_type, required_types))
    )


def _digest_array(digest: Any, name: str, values: np.ndarray) -> None:
    """Add one typed array to a routing digest."""
    encoded_name = name.encode("utf-8")
    dtype = values.dtype.str.encode("ascii")
    shape = np.asarray(values.shape, dtype="<i8").tobytes()
    digest.update(len(encoded_name).to_bytes(4, "little"))
    digest.update(encoded_name)
    digest.update(len(dtype).to_bytes(4, "little"))
    digest.update(dtype)
    digest.update(len(shape).to_bytes(4, "little"))
    digest.update(shape)
    digest.update(values.tobytes(order="C"))


def _routing_mapping_sha256(topology: Topology) -> str:
    """Return the digest of every static routing mapping."""
    digest = hashlib.sha256()
    _digest_array(digest, "node_count", np.asarray(topology.node_count, dtype="<i8"))
    for name in (
        "edge_source",
        "edge_destination",
        "edge_type",
        "edge_difficulty",
        "edge_nominal_travel_time",
    ):
        _digest_array(digest, name, getattr(topology, name))
    return digest.hexdigest()


def _route_cache_identity(topology: Topology) -> RouteCacheIdentity:
    """Return the complete identity of one static route table."""
    return RouteCacheIdentity(
        mountain_sha256=topology.mountain_sha256,
        ability_limits=tuple(int(value) for value in PISTE_LIMIT_BY_ABILITY),
        required_destinations=required_destinations(topology),
        routing_mapping_sha256=_routing_mapping_sha256(topology),
    )


def build_route_table(topology: Topology) -> RouteTable:
    """Return each ability's safe shortest-path table.

    The function runs one search on the reversed graph from each destination.
    One search then gives the next hop of every node towards that destination.
    """
    identity = _route_cache_identity(topology)
    cached = _ROUTE_CACHE.get(identity)
    if cached is not None:
        return cached
    tables = [
        _build_ability_routes(topology, ability, limit)
        for ability, limit in enumerate(identity.ability_limits)
    ]
    table = RouteTable(
        next_edge=np.stack([table[0] for table in tables]),
        travel_time=np.stack([table[1] for table in tables]),
        cache_identity=identity,
    )
    return _ROUTE_CACHE.setdefault(identity, table)


def _build_ability_routes(
    topology: Topology, ability: int, piste_limit: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the safe route arrays for one ability."""
    count = topology.node_count
    cost = topology.edge_nominal_travel_time.astype(np.float64)
    allowed = np.flatnonzero(
        ability_edge_mask(topology, ability, piste_limit=piste_limit)
    )

    # Keep the fastest edge of a pair of nodes.
    # The slow edge is written first, so the fast edge overwrites it.
    edge_of_pair = np.full((count, count), NO_EDGE, dtype=np.int32)
    order = allowed[np.argsort(-cost[allowed], kind="stable")]
    edge_of_pair[topology.edge_source[order], topology.edge_destination[order]] = order

    sources, destinations = np.nonzero(edge_of_pair >= 0)
    graph = csr_matrix(
        (cost[edge_of_pair[sources, destinations]], (sources, destinations)),
        shape=(count, count),
    )

    # A predecessor on the reversed graph is the next node on the forward path.
    travel_time, predecessors = dijkstra(
        graph.T.tocsr(),
        directed=True,
        indices=np.arange(count),
        return_predecessors=True,
    )
    next_node = predecessors.T
    reachable = next_node >= 0
    next_edge = np.full((count, count), NO_EDGE, dtype=np.int32)
    nodes = np.nonzero(reachable)[0]
    next_edge[reachable] = edge_of_pair[nodes, next_node[reachable]]

    return next_edge, travel_time.T


def walk_route(
    table: RouteTable,
    topology: Topology,
    source: int,
    destination: int,
    *,
    ability: int,
) -> list[int]:
    """Return one ability's safe path to a destination.

    The function returns an empty list when the source is the destination.
    It raises a `ValueError` when no route exists.
    """
    if source == destination:
        return []
    if table.next_edge[ability, source, destination] == NO_EDGE:
        raise ValueError(
            f"no route goes from the node {source} to the node {destination}"
        )

    route: list[int] = []
    node = source
    while node != destination:
        edge = int(table.next_edge[ability, node, destination])
        route.append(edge)
        node = int(topology.edge_destination[edge])
    return route

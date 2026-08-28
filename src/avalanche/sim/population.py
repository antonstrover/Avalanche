"""The dense arrays of the skier population.

The simulator keeps one array for each property of a skier.
It does not make one Python object for each skier.
A movement step selects a group of skiers with a boolean mask.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from avalanche.config.models import PopulationConfig
from avalanche.sim.ability import ABILITY_NAMES
from avalanche.sim.skier import LocationKind, Status
from avalanche.sim.topology import NODE_TYPE_NAMES, Topology

CUSTOMER_GROUP_NAMES = ("standard", "premium")

POPULATION_ARRAY_FIELDS = (
    "location_kind",
    "location_index",
    "required_travel_seconds",
    "remaining_travel_seconds",
    "destination",
    "ability",
    "risk_tolerance",
    "group",
    "compliance",
    "status",
    "wait_time",
    "journey_time",
    "blocked_time",
    "queue_no_route_blocked_seconds",
    "onboard_blocked_seconds",
    "arrival_time",
    "queue_ticket",
    "queue_source_node",
    "chosen_edge",
    "locally_rejected_edge",
)

if TYPE_CHECKING:
    from avalanche.sim.routes import RouteTable

ENTRANCE_NODE = NODE_TYPE_NAMES.index("entrance")
EXIT_NODE = NODE_TYPE_NAMES.index("exit")


@dataclass
class SkierArrays:
    """The state of `N` skiers in one array for each property.

    `location_index` is a node index for the kind `NODE` and `PENDING`.
    It is an edge index for the kind `PISTE`, `LIFT`, and `QUEUE`.
    `required_travel_seconds` stores the nominal edge travel time.
    `remaining_travel_seconds` stores the formal remaining edge work.
    `arrival_time` increases with the index, so a search finds the new arrivals.
    `queue_ticket` is the order of the arrival in a lift queue.
    It is -1 when the skier is not in a queue.
    `queue_source_node` records the source while the skier is in a lift queue.
    It is -1 when the skier is not in a queue.
    `arrived` counts the skiers that the engine already released.
    `next_ticket` gives the ticket of the next skier that joins a queue.
    """

    location_kind: np.ndarray
    location_index: np.ndarray
    required_travel_seconds: np.ndarray
    remaining_travel_seconds: np.ndarray
    destination: np.ndarray
    ability: np.ndarray
    risk_tolerance: np.ndarray
    group: np.ndarray
    compliance: np.ndarray
    status: np.ndarray
    wait_time: np.ndarray
    journey_time: np.ndarray
    blocked_time: np.ndarray
    queue_no_route_blocked_seconds: np.ndarray
    onboard_blocked_seconds: np.ndarray
    arrival_time: np.ndarray
    queue_ticket: np.ndarray
    queue_source_node: np.ndarray
    chosen_edge: np.ndarray
    locally_rejected_edge: np.ndarray
    arrived: int = 0
    next_ticket: int = 0

    def __len__(self) -> int:
        """Return the count of skiers."""
        return int(self.location_kind.size)

    def checksum_fields(self) -> tuple[tuple[str, np.ndarray], ...]:
        """Return the name and the array of each field, in a fixed order.

        The checksum and the invariant tests walk this order.
        A new field must join this list.
        """
        return tuple((name, getattr(self, name)) for name in POPULATION_ARRAY_FIELDS)


def empty_population(count: int) -> SkierArrays:
    """Return `count` skiers at the node 0 with the default attributes."""
    return SkierArrays(
        location_kind=np.full(count, LocationKind.NODE, dtype=np.int8),
        location_index=np.zeros(count, dtype=np.int32),
        required_travel_seconds=np.zeros(count, dtype=np.float64),
        remaining_travel_seconds=np.zeros(count, dtype=np.float64),
        destination=np.zeros(count, dtype=np.int32),
        ability=np.zeros(count, dtype=np.int8),
        risk_tolerance=np.zeros(count, dtype=np.float64),
        group=np.zeros(count, dtype=np.int8),
        compliance=np.zeros(count, dtype=np.float64),
        status=np.full(count, Status.ACTIVE, dtype=np.int8),
        wait_time=np.zeros(count, dtype=np.float64),
        journey_time=np.zeros(count, dtype=np.float64),
        blocked_time=np.zeros(count, dtype=np.float64),
        queue_no_route_blocked_seconds=np.zeros(count, dtype=np.float64),
        onboard_blocked_seconds=np.zeros(count, dtype=np.float64),
        arrival_time=np.zeros(count, dtype=np.float64),
        queue_ticket=np.full(count, -1, dtype=np.int64),
        queue_source_node=np.full(count, -1, dtype=np.int32),
        chosen_edge=np.full(count, -1, dtype=np.int32),
        locally_rejected_edge=np.full(count, -1, dtype=np.int32),
        arrived=count,
        next_ticket=0,
    )


def display_progress(pop: SkierArrays) -> np.ndarray:
    """Return bounded derived progress for a display adapter."""
    progress = np.zeros(len(pop), dtype=np.float64)
    on_edge = np.isin(pop.location_kind, (LocationKind.PISTE, LocationKind.LIFT))
    positive = on_edge & (pop.required_travel_seconds > 0.0)
    progress[positive] = 1.0 - np.divide(
        pop.remaining_travel_seconds[positive],
        pop.required_travel_seconds[positive],
    )
    return np.clip(progress, 0.0, 1.0)


def sample_population(
    rng: np.random.Generator,
    topology: Topology,
    routes: RouteTable,
    config: PopulationConfig,
) -> SkierArrays:
    """Return a new sampled population of the size in the configuration.

    The order of the draws is part of the seed contract.
    A change of the order changes every run with the same seed.
    The order is the arrival time, the entry node, the provisional destination,
    the ability, the risk tolerance, the customer group, the compliance,
    and the safe destination selector.
    The ability and the customer group use separate independent draws.
    Each destination is an exit reachable for the skier's ability.

    Each skier waits in the kind `PENDING` at its entry node.
    The skiers keep the ascending arrival order, which `start_arrivals` needs.
    """
    entrances = np.flatnonzero(topology.node_type == ENTRANCE_NODE)
    if entrances.size == 0:
        raise ValueError("the mountain has no entrance node")
    exits = np.flatnonzero(topology.node_type == EXIT_NODE)
    if exits.size == 0:
        raise ValueError("the mountain has no exit node")

    count = int(config.skier_count)
    arrival_time = np.sort(rng.uniform(0.0, config.arrival_window_seconds, count))
    entry = rng.choice(entrances, size=count)
    provisional_destination = rng.choice(exits, size=count)
    ability = rng.choice(len(ABILITY_NAMES), size=count, p=config.ability_weights)
    risk_tolerance = rng.uniform(0.0, 1.0, count)
    group = rng.choice(
        len(CUSTOMER_GROUP_NAMES), size=count, p=config.customer_group_weights
    )
    compliance = np.clip(
        rng.normal(config.compliance_mean, config.compliance_spread, count), 0.0, 1.0
    )
    destination_selector = rng.random(count)
    destination = provisional_destination.copy()
    for entry_node in entrances:
        for ability_index in range(len(ABILITY_NAMES)):
            members = (entry == entry_node) & (ability == ability_index)
            if not np.any(members):
                continue
            reachable = exits[
                np.isfinite(routes.travel_time[ability_index, entry_node, exits])
            ]
            if reachable.size == 0:
                raise ValueError(
                    f"the {ABILITY_NAMES[ability_index]} ability has no safe exit "
                    f"from the entrance {topology.node_ids[entry_node]!r}"
                )
            unsafe = members & ~np.isin(provisional_destination, reachable)
            choices = np.minimum(
                (destination_selector[unsafe] * reachable.size).astype(np.int64),
                reachable.size - 1,
            )
            destination[unsafe] = reachable[choices]

    pop = empty_population(count)
    pop.location_kind[:] = LocationKind.PENDING
    pop.location_index[:] = entry
    pop.destination[:] = destination
    pop.ability[:] = ability
    pop.risk_tolerance[:] = risk_tolerance
    pop.group[:] = group
    pop.compliance[:] = compliance
    pop.arrival_time[:] = arrival_time
    pop.arrived = 0
    return pop


def population_from_starts(
    starts: np.ndarray | list[int],
    destinations: np.ndarray | list[int] | int,
    arrival_times: np.ndarray | list[float] | None = None,
) -> SkierArrays:
    """Return a population of skiers at the given start nodes.

    The attribute arrays hold zeros. Stage 3 samples them later.
    Each skier starts at the time 0.0 when `arrival_times` is None.
    A skier waits in the kind `PENDING` until its arrival time.
    The skiers keep the order of the arrival time.
    """
    starts = np.asarray(starts, dtype=np.int32)
    destinations = np.asarray(destinations, dtype=np.int32)
    if arrival_times is not None:
        arrival_times = np.asarray(arrival_times, dtype=np.float64)
        order = np.argsort(arrival_times, kind="stable")
        starts = starts[order]
        if destinations.ndim > 0:
            destinations = destinations[order]
        arrival_times = arrival_times[order]

    pop = empty_population(int(starts.size))
    pop.location_index[:] = starts
    pop.destination[:] = destinations
    if arrival_times is not None:
        pop.arrival_time[:] = arrival_times
        pop.location_kind[:] = LocationKind.PENDING
        pop.arrived = 0
    return pop


def group_rank(
    group: np.ndarray, order_key: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the members sorted by group then key, and the rank inside each group."""
    members = np.lexsort((order_key, group))
    starts = np.searchsorted(group[members], group[members], side="left")
    return members, np.arange(members.size) - starts

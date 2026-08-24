"""The dense arrays of the skier population.

The simulator keeps one array for each property of a skier.
It does not make one Python object for each skier.
A movement step selects a group of skiers with a boolean mask.
"""

from dataclasses import dataclass

import numpy as np

from avalanche.config.models import PopulationConfig
from avalanche.sim.skier import LocationKind, Status
from avalanche.sim.topology import NODE_TYPE_NAMES, Topology

ABILITY_NAMES = ("beginner", "intermediate", "advanced")

ENTRANCE_NODE = NODE_TYPE_NAMES.index("entrance")
EXIT_NODE = NODE_TYPE_NAMES.index("exit")


@dataclass
class SkierArrays:
    """The state of `N` skiers in one array for each property.

    `location_index` is a node index for the kind `NODE` and `PENDING`.
    It is an edge index for the kind `PISTE`, `LIFT`, and `QUEUE`.
    `progress` is the normalised position along an edge, from 0.0 to 1.0.
    `arrival_time` increases with the index, so a search finds the new arrivals.
    `queue_ticket` is the order of the arrival in a lift queue.
    It is -1 when the skier is not in a queue.
    `arrived` counts the skiers that the engine already released.
    `next_ticket` gives the ticket of the next skier that joins a queue.
    """

    location_kind: np.ndarray
    location_index: np.ndarray
    progress: np.ndarray
    destination: np.ndarray
    ability: np.ndarray
    risk_tolerance: np.ndarray
    group: np.ndarray
    compliance: np.ndarray
    status: np.ndarray
    wait_time: np.ndarray
    journey_time: np.ndarray
    blocked_time: np.ndarray
    arrival_time: np.ndarray
    queue_ticket: np.ndarray
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
        return (
            ("location_kind", self.location_kind),
            ("location_index", self.location_index),
            ("progress", self.progress),
            ("destination", self.destination),
            ("ability", self.ability),
            ("risk_tolerance", self.risk_tolerance),
            ("group", self.group),
            ("compliance", self.compliance),
            ("status", self.status),
            ("wait_time", self.wait_time),
            ("journey_time", self.journey_time),
            ("blocked_time", self.blocked_time),
            ("arrival_time", self.arrival_time),
            ("queue_ticket", self.queue_ticket),
        )


def empty_population(count: int) -> SkierArrays:
    """Return `count` skiers at the node 0 with the default attributes."""
    return SkierArrays(
        location_kind=np.full(count, LocationKind.NODE, dtype=np.int8),
        location_index=np.zeros(count, dtype=np.int32),
        progress=np.zeros(count, dtype=np.float64),
        destination=np.zeros(count, dtype=np.int32),
        ability=np.zeros(count, dtype=np.int8),
        risk_tolerance=np.zeros(count, dtype=np.float64),
        group=np.zeros(count, dtype=np.int8),
        compliance=np.zeros(count, dtype=np.float64),
        status=np.full(count, Status.ACTIVE, dtype=np.int8),
        wait_time=np.zeros(count, dtype=np.float64),
        journey_time=np.zeros(count, dtype=np.float64),
        blocked_time=np.zeros(count, dtype=np.float64),
        arrival_time=np.zeros(count, dtype=np.float64),
        queue_ticket=np.full(count, -1, dtype=np.int64),
        arrived=count,
        next_ticket=0,
    )


def sample_population(
    rng: np.random.Generator, topology: Topology, config: PopulationConfig
) -> SkierArrays:
    """Return a new sampled population of the size in the configuration.

    The order of the draws is part of the seed contract.
    A change of the order changes every run with the same seed.
    The order is the arrival time, the entry node, the destination,
    the ability, the risk tolerance, the group, and the compliance.

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
    destination = rng.choice(exits, size=count)
    ability = rng.choice(3, size=count, p=config.ability_weights)
    risk_tolerance = rng.uniform(0.0, 1.0, count)
    # The group equals the ability for now. Stage 5 adds a second axis.
    group = ability
    compliance = np.clip(
        rng.normal(config.compliance_mean, config.compliance_spread, count), 0.0, 1.0
    )

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

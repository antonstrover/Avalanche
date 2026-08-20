"""Build the static shortest-path table of the mountain.

The cost of an edge is its nominal travel time.
The table gives the first edge of the shortest path from a node to a destination.
A later stage adds the dynamic cost and the compliance model.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from avalanche.sim.topology import Topology

NO_EDGE = -1


@dataclass(frozen=True)
class RouteTable:
    """The immutable shortest-path table.

    `next_edge[node, destination]` is the index of the first edge on the shortest path.
    It is `-1` when no route exists and when the node is the destination.
    `travel_time[node, destination]` is the total nominal travel time of that path.
    It is infinite when no route exists.
    """

    next_edge: np.ndarray
    travel_time: np.ndarray

    @property
    def node_count(self) -> int:
        """Return the count of the nodes."""
        return int(self.next_edge.shape[0])


def build_route_table(topology: Topology) -> RouteTable:
    """Return the shortest-path table of one topology.

    The function runs one search on the reversed graph from each destination.
    One search then gives the next hop of every node towards that destination.
    """
    count = topology.node_count
    cost = topology.edge_nominal_travel_time.astype(np.float64)

    # Keep the fastest edge of a pair of nodes.
    # The slow edge is written first, so the fast edge overwrites it.
    edge_of_pair = np.full((count, count), NO_EDGE, dtype=np.int32)
    order = np.argsort(-cost, kind="stable")
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

    return RouteTable(next_edge=next_edge, travel_time=travel_time.T)


def walk_route(
    table: RouteTable, topology: Topology, source: int, destination: int
) -> list[int]:
    """Return the edge indices of the shortest path from one node to a destination.

    The function returns an empty list when the source is the destination.
    It raises a `ValueError` when no route exists.
    """
    if source == destination:
        return []
    if table.next_edge[source, destination] == NO_EDGE:
        raise ValueError(
            f"no route goes from the node {source} to the node {destination}"
        )

    route: list[int] = []
    node = source
    while node != destination:
        edge = int(table.next_edge[node, destination])
        route.append(edge)
        node = int(topology.edge_destination[edge])
    return route

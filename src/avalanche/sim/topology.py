"""Change the mountain graph into static index arrays.

The loader calls NetworkX one time.
The simulator then uses only the arrays in `Topology`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avalanche.sim.graph import build_graph, validate_graph

NODE_TYPE_NAMES = ("entrance", "exit", "lift_station", "junction", "safe_zone")
EDGE_TYPE_NAMES = ("piste", "lift")
DIFFICULTY_NAMES = ("none", "green", "blue", "red", "black")


@dataclass(frozen=True)
class Topology:
    """The immutable mountain arrays.

    A node array and an edge array use the index of the node or of the edge.
    A type code is a position in the matching name tuple.
    """

    name: str

    node_ids: tuple[str, ...]
    node_index: dict[str, int]
    node_x: np.ndarray
    node_y: np.ndarray
    node_elevation: np.ndarray
    node_type: np.ndarray
    node_capacity: np.ndarray
    node_controllable: np.ndarray

    edge_source: np.ndarray
    edge_destination: np.ndarray
    edge_type: np.ndarray
    edge_difficulty: np.ndarray
    edge_length: np.ndarray
    edge_nominal_travel_time: np.ndarray
    edge_safe_capacity: np.ndarray
    edge_critical_density: np.ndarray
    edge_lift_throughput: np.ndarray
    edge_wind_sensitivity: np.ndarray
    edge_visibility_sensitivity: np.ndarray
    edge_snow_sensitivity: np.ndarray
    edge_controllable: np.ndarray

    edge_offsets: np.ndarray
    outgoing_edges: np.ndarray

    @property
    def node_count(self) -> int:
        """Return the count of the nodes."""
        return len(self.node_ids)

    @property
    def edge_count(self) -> int:
        """Return the count of the edges."""
        return int(self.edge_source.shape[0])

    def edges_from(self, node: int) -> np.ndarray:
        """Return the indices of the outgoing edges of one node."""
        return self.outgoing_edges[
            self.edge_offsets[node] : self.edge_offsets[node + 1]
        ]


def _code(names: tuple[str, ...], value: str, kind: str) -> int:
    """Return the integer code of one name. Raise a `ValueError` for an unknown name."""
    try:
        return names.index(value)
    except ValueError:
        raise ValueError(f"the {kind} {value!r} is unknown") from None


def load_topology(path: Path) -> Topology:
    """Read the mountain file, check it, and return the static arrays.

    The node order is the sorted order of the node identities.
    The index of a node stays the same for one mountain file.
    """
    graph = build_graph(path)
    validate_graph(graph)

    node_ids = tuple(sorted(graph.nodes))
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    nodes = [graph.nodes[node_id] for node_id in node_ids]

    edges = sorted(
        graph.edges(data=True),
        key=lambda edge: (node_index[edge[0]], node_index[edge[1]]),
    )
    edge_source = np.array([node_index[e[0]] for e in edges], dtype=np.int32)
    edge_destination = np.array([node_index[e[1]] for e in edges], dtype=np.int32)

    # The edges are sorted by the source index, so the offsets are a running count.
    counts = np.bincount(edge_source, minlength=len(node_ids))
    edge_offsets = np.zeros(len(node_ids) + 1, dtype=np.int32)
    np.cumsum(counts, out=edge_offsets[1:])

    def edge_float(field: str) -> np.ndarray:
        values = [e[2][field] for e in edges]
        return np.array(values, dtype=np.float32)

    return Topology(
        name=str(graph.graph.get("name", path.stem)),
        node_ids=node_ids,
        node_index=node_index,
        node_x=np.array([n["x"] for n in nodes], dtype=np.float32),
        node_y=np.array([n["y"] for n in nodes], dtype=np.float32),
        node_elevation=np.array([n["elevation"] for n in nodes], dtype=np.float32),
        node_type=np.array(
            [_code(NODE_TYPE_NAMES, n["node_type"], "node type") for n in nodes],
            dtype=np.int8,
        ),
        node_capacity=np.array([n["capacity"] for n in nodes], dtype=np.int32),
        node_controllable=np.array(
            [n.get("controllable", n["node_type"] != "exit") for n in nodes],
            dtype=bool,
        ),
        edge_source=edge_source,
        edge_destination=edge_destination,
        edge_type=np.array(
            [_code(EDGE_TYPE_NAMES, e[2]["edge_type"], "edge type") for e in edges],
            dtype=np.int8,
        ),
        edge_difficulty=np.array(
            [_code(DIFFICULTY_NAMES, e[2]["difficulty"], "difficulty") for e in edges],
            dtype=np.int8,
        ),
        edge_length=edge_float("length"),
        edge_nominal_travel_time=edge_float("nominal_travel_time"),
        edge_safe_capacity=edge_float("safe_capacity"),
        edge_critical_density=edge_float("critical_density"),
        edge_lift_throughput=np.array(
            [
                0.0
                if edge[2]["lift_throughput"] is None
                else edge[2]["lift_throughput"]
                for edge in edges
            ],
            dtype=np.float32,
        ),
        edge_wind_sensitivity=edge_float("wind_sensitivity"),
        edge_visibility_sensitivity=edge_float("visibility_sensitivity"),
        edge_snow_sensitivity=edge_float("snow_sensitivity"),
        edge_controllable=np.array(
            [e[2].get("controllable", True) for e in edges], dtype=bool
        ),
        edge_offsets=edge_offsets,
        outgoing_edges=np.arange(len(edges), dtype=np.int32),
    )

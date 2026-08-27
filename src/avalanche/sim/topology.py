"""Change the mountain graph into static index arrays.

The loader calls NetworkX one time.
The simulator then uses only the arrays in `Topology`.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType

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
    mountain_sha256: str

    node_ids: tuple[str, ...]
    node_index: Mapping[str, int]
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

    def __post_init__(self) -> None:
        """Store every static array on immutable bytes."""
        for field_info in fields(self):
            dtype = _TOPOLOGY_ARRAY_DTYPES.get(field_info.name)
            if dtype is None:
                continue
            value = getattr(self, field_info.name)
            object.__setattr__(self, field_info.name, immutable_array(value, dtype))
        object.__setattr__(self, "node_index", MappingProxyType(dict(self.node_index)))

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


_TOPOLOGY_ARRAY_DTYPES = {
    "node_x": "<f4",
    "node_y": "<f4",
    "node_elevation": "<f4",
    "node_type": "|i1",
    "node_capacity": "<i4",
    "node_controllable": "|b1",
    "edge_source": "<i4",
    "edge_destination": "<i4",
    "edge_type": "|i1",
    "edge_difficulty": "|i1",
    "edge_length": "<f4",
    "edge_nominal_travel_time": "<f4",
    "edge_safe_capacity": "<i4",
    "edge_critical_density": "<f4",
    "edge_lift_throughput": "<f4",
    "edge_wind_sensitivity": "<f4",
    "edge_visibility_sensitivity": "<f4",
    "edge_snow_sensitivity": "<f4",
    "edge_controllable": "|b1",
    "edge_offsets": "<i4",
    "outgoing_edges": "<i4",
}


def immutable_array(values: np.ndarray, dtype: str) -> np.ndarray:
    """Return an array view backed by immutable bytes."""
    if not isinstance(values, np.ndarray):
        raise TypeError("a static value must be a NumPy array")
    if values.dtype.hasobject:
        raise TypeError("a static array must not use an object dtype")
    if not isinstance(dtype, str):
        raise TypeError("a static dtype must declare its byte order")
    declared = np.dtype(dtype)
    if declared.itemsize > 1 and not dtype.startswith(("<", ">")):
        raise TypeError("a static dtype must declare its byte order")
    normalized = np.ascontiguousarray(values, dtype=declared)
    buffer = normalized.tobytes(order="C")
    return np.frombuffer(buffer, dtype=declared).reshape(normalized.shape)


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
        mountain_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
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
        edge_safe_capacity=np.array(
            [e[2]["safe_capacity"] for e in edges], dtype="<i4"
        ),
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

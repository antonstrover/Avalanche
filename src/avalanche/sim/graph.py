"""Read the mountain graph and check it.

This module uses NetworkX one time, when the mountain loads.
The simulator does not use NetworkX at run time.
"""

from math import isfinite
from pathlib import Path
from typing import Any

import networkx as nx

from avalanche.config import load_yaml

NODE_TYPES = frozenset({"entrance", "exit", "lift_station", "junction", "safe_zone"})
EDGE_TYPES = frozenset({"piste", "lift"})
DIFFICULTIES = frozenset({"none", "green", "blue", "red", "black"})
NODE_FIELDS = ("node_id", "node_type", "x", "y", "elevation", "capacity")
EDGE_FIELDS = (
    "source",
    "destination",
    "edge_type",
    "difficulty",
    "length",
    "nominal_travel_time",
    "safe_capacity",
    "critical_density",
    "lift_throughput",
    "wind_sensitivity",
    "visibility_sensitivity",
    "snow_sensitivity",
)
POSITIVE_EDGE_FIELDS = (
    "length",
    "nominal_travel_time",
    "safe_capacity",
    "critical_density",
)
SENSITIVITY_FIELDS = (
    "wind_sensitivity",
    "visibility_sensitivity",
    "snow_sensitivity",
)


def _records(data: dict[str, Any], field: str, path: Path) -> list[dict[str, Any]]:
    """Return one list of mountain records."""
    records = data.get(field)
    if not isinstance(records, list):
        raise ValueError(f"the mountain {path} field {field!r} must be a list")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"the mountain {path} {field} record {index} must be a mapping"
            )
    return records


def _require_identity(
    record: dict[str, Any], field: str, kind: str, index: int, path: Path
) -> str:
    """Return one required record identity."""
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"the mountain {path} {kind} record {index} has an invalid field {field!r}"
        )
    return value


def _reject_duplicate_records(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], path: Path
) -> None:
    """Reject a repeated node or directed edge identity."""
    node_ids: set[str] = set()
    for index, node in enumerate(nodes):
        node_id = _require_identity(node, "node_id", "node", index, path)
        if node_id in node_ids:
            raise ValueError(f"the mountain {path} repeats the node {node_id!r}")
        node_ids.add(node_id)

    endpoint_pairs: set[tuple[str, str]] = set()
    for index, edge in enumerate(edges):
        endpoints = (
            _require_identity(edge, "source", "edge", index, path),
            _require_identity(edge, "destination", "edge", index, path),
        )
        if endpoints in endpoint_pairs:
            raise ValueError(
                f"the mountain {path} repeats the directed edge "
                f"{endpoints[0]!r} to {endpoints[1]!r}"
            )
        endpoint_pairs.add(endpoints)


def _require_fields(
    record: dict[str, Any], fields: tuple[str, ...], identity: str, path: Path
) -> None:
    """Require each field in one mountain record."""
    for field in fields:
        if field not in record:
            raise ValueError(
                f"the mountain {path} record {identity} misses the field {field!r}"
            )


def _number(
    record: dict[str, Any],
    field: str,
    identity: str,
    path: Path,
    *,
    positive: bool | None,
) -> float:
    """Return one finite number with the required sign."""
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"the mountain {path} record {identity} has a nonnumeric field {field!r}"
        )
    if not isfinite(value):
        raise ValueError(
            f"the mountain {path} record {identity} has a nonfinite field {field!r}"
        )
    if positive and value <= 0:
        raise ValueError(
            f"the mountain {path} record {identity} needs a positive field {field!r}"
        )
    if positive is False and value < 0:
        raise ValueError(
            f"the mountain {path} record {identity} needs a nonnegative field {field!r}"
        )
    return float(value)


def _integer_capacity(
    record: dict[str, Any], field: str, identity: str, path: Path
) -> None:
    """Require one positive integer capacity."""
    value = record[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"the mountain {path} record {identity} needs an integer field {field!r}"
        )


def _validate_static_values(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]], path: Path
) -> None:
    """Validate each static mountain value."""
    for node in nodes:
        identity = repr(node["node_id"])
        _require_fields(node, NODE_FIELDS, identity, path)
        node_type = node["node_type"]
        if not isinstance(node_type, str) or node_type not in NODE_TYPES:
            raise ValueError(
                f"the mountain {path} node {identity} "
                f"has the unknown type {node_type!r}"
            )
        for field in ("x", "y", "elevation"):
            _number(node, field, identity, path, positive=None)
        _integer_capacity(node, "capacity", identity, path)
        if "controllable" in node and not isinstance(node["controllable"], bool):
            raise ValueError(
                f"the mountain {path} node {identity} "
                "has a non-Boolean field 'controllable'"
            )

    for edge in edges:
        identity = f"{edge['source']!r} to {edge['destination']!r}"
        _require_fields(edge, EDGE_FIELDS, identity, path)
        edge_type = edge["edge_type"]
        if not isinstance(edge_type, str) or edge_type not in EDGE_TYPES:
            raise ValueError(
                f"the mountain {path} edge {identity} "
                f"has the unknown type {edge_type!r}"
            )
        difficulty = edge["difficulty"]
        if not isinstance(difficulty, str) or difficulty not in DIFFICULTIES:
            raise ValueError(
                f"the mountain {path} edge {identity} "
                f"has the unknown difficulty {difficulty!r}"
            )
        for field in POSITIVE_EDGE_FIELDS:
            _number(edge, field, identity, path, positive=True)
        _integer_capacity(edge, "safe_capacity", identity, path)
        for field in SENSITIVITY_FIELDS:
            _number(edge, field, identity, path, positive=False)
        if edge_type == "lift":
            _integer_capacity(edge, "lift_throughput", identity, path)
        elif edge["lift_throughput"] is not None:
            raise ValueError(
                f"the mountain {path} piste {identity} "
                "needs a null field 'lift_throughput'"
            )
        if "controllable" in edge and not isinstance(edge["controllable"], bool):
            raise ValueError(
                f"the mountain {path} edge {identity} "
                "has a non-Boolean field 'controllable'"
            )


def build_graph(path: Path) -> nx.DiGraph:
    """Read the mountain YAML file and return a directed graph.

    Each node and each edge keeps its static fields as attributes.
    """
    data = load_yaml(path)
    nodes = _records(data, "nodes", path)
    edges = _records(data, "edges", path)
    _reject_duplicate_records(nodes, edges, path)
    _validate_static_values(nodes, edges, path)
    graph = nx.DiGraph(name=data.get("name", path.stem))
    for node in nodes:
        attributes = dict(node)
        graph.add_node(attributes.pop("node_id"), **attributes)
    for edge in edges:
        attributes = dict(edge)
        source = attributes.pop("source")
        destination = attributes.pop("destination")
        graph.add_edge(source, destination, **attributes)
    return graph


def validate_graph(graph: nx.DiGraph) -> None:
    """Check the mountain graph. Raise a `ValueError` for the first fault."""
    for node, attributes in graph.nodes(data=True):
        node_type = attributes.get("node_type")
        if node_type is None:
            raise ValueError(f"an edge names the unknown node {node!r}")
        if node_type not in NODE_TYPES:
            raise ValueError(f"the node {node!r} has the unknown type {node_type!r}")
        if graph.degree(node) == 0:
            raise ValueError(f"the node {node!r} is an orphan node")

    for source, destination, attributes in graph.edges(data=True):
        edge_type = attributes.get("edge_type")
        if edge_type not in EDGE_TYPES:
            raise ValueError(
                f"the edge {source!r} to {destination!r} "
                f"has the unknown type {edge_type!r}"
            )

    entrances = [n for n, t in graph.nodes(data="node_type") if t == "entrance"]
    exits = [n for n, t in graph.nodes(data="node_type") if t == "exit"]
    if not entrances:
        raise ValueError("the mountain graph has no entrance node")
    if not exits:
        raise ValueError("the mountain graph has no exit node")

    reachable: set[str] = set()
    for entrance in entrances:
        reachable |= nx.descendants(graph, entrance)
    for exit_node in exits:
        if exit_node not in reachable:
            raise ValueError(f"no entrance reaches the exit node {exit_node!r}")

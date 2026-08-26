"""Read the mountain graph and check it.

This module uses NetworkX one time, when the mountain loads.
The simulator does not use NetworkX at run time.
"""

from pathlib import Path

import networkx as nx

from avalanche.config import load_yaml

NODE_TYPES = frozenset({"entrance", "exit", "lift_station", "junction", "safe_zone"})
EDGE_TYPES = frozenset({"piste", "lift"})


def _reject_duplicate_records(data: dict[str, object], path: Path) -> None:
    """Reject a repeated node or directed edge identity."""
    node_ids: set[object] = set()
    for node in data.get("nodes", []):
        node_id = node["node_id"]
        if node_id in node_ids:
            raise ValueError(f"the mountain {path} repeats the node {node_id!r}")
        node_ids.add(node_id)

    endpoint_pairs: set[tuple[object, object]] = set()
    for edge in data.get("edges", []):
        endpoints = (edge["source"], edge["destination"])
        if endpoints in endpoint_pairs:
            raise ValueError(
                f"the mountain {path} repeats the directed edge "
                f"{endpoints[0]!r} to {endpoints[1]!r}"
            )
        endpoint_pairs.add(endpoints)


def build_graph(path: Path) -> nx.DiGraph:
    """Read the mountain YAML file and return a directed graph.

    Each node and each edge keeps its static fields as attributes.
    """
    data = load_yaml(path)
    _reject_duplicate_records(data, path)
    graph = nx.DiGraph(name=data.get("name", path.stem))
    for node in data.get("nodes", []):
        attributes = dict(node)
        graph.add_node(attributes.pop("node_id"), **attributes)
    for edge in data.get("edges", []):
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

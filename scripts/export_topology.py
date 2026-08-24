"""Export a mountain file to the resort data of the mountain scene.

The script reads a mountain YAML file and writes `dashboard/src/mountain/resort.json`.
It keeps only the static fields that the scene draws.
It does not import the avalanche package, so it runs before the loader exists.

Usage:
    python scripts/export_topology.py configs/mountain/medium-resort.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPOSITORY / "dashboard" / "src" / "mountain" / "resort.json"

NODE_ALIASES = {
    "node_id": ("node_id", "id", "name"),
    "x": ("x",),
    "y": ("y",),
    "elevation": ("elevation", "z", "altitude"),
    "node_type": ("node_type", "type", "kind"),
    "capacity": ("capacity",),
}

EDGE_ALIASES = {
    "source": ("source", "from", "src", "u"),
    "destination": ("destination", "to", "target", "dst", "v"),
    "edge_type": ("edge_type", "type", "kind"),
    "difficulty": ("difficulty", "grade", "difficulty_grade"),
    "length": ("length", "distance"),
}


def pick(item: dict[str, Any], aliases: tuple[str, ...], default: Any) -> Any:
    """Return the first value that the item holds under one of the names."""
    for name in aliases:
        if name in item:
            return item[name]
    return default


def convert(
    item: dict[str, Any], aliases: dict[str, tuple[str, ...]]
) -> dict[str, Any]:
    """Convert one node or one edge into the shape that the scene reads."""
    defaults = {"capacity": 0, "length": 0, "difficulty": "none"}
    result = {}
    for field, names in aliases.items():
        value = pick(item, names, defaults.get(field))
        if value is None:
            raise ValueError(f"The item {item} has no value for {field}.")
        result[field] = value
    return result


def export(source: Path, output: Path) -> dict[str, Any]:
    """Read the mountain file and write the resort data."""
    document = yaml.safe_load(source.read_text())
    mountain = document.get("mountain", document)
    nodes = sorted(
        (convert(node, NODE_ALIASES) for node in mountain["nodes"]),
        key=lambda node: node["node_id"],
    )
    edges = sorted(
        (convert(edge, EDGE_ALIASES) for edge in mountain["edges"]),
        key=lambda edge: (edge["source"], edge["destination"]),
    )
    resort = {
        "name": mountain.get("name", source.stem),
        "nodes": nodes,
        "edges": edges,
    }
    output.write_text(json.dumps(resort, indent=2) + "\n")
    return resort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="the mountain YAML file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    resort = export(arguments.source, arguments.output)
    print(
        f"Wrote {len(resort['nodes'])} nodes and {len(resort['edges'])} edges "
        f"to {arguments.output}."
    )


if __name__ == "__main__":
    main()

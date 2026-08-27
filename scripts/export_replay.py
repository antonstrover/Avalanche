"""Export a short simulator run to the replay file of the mountain scene.

The script runs the simulator with a few skiers on a mountain file.
It writes `dashboard/src/mountain/replay.sample.json`.
A frame holds the simulation time and the place of each skier.
The scene reads the place and draws a marker on the piste curve.

The simulator indexes a node and an edge in its own order.
The scene indexes them in the order of the mountain file.
The script therefore maps each index through the node identity.

Usage:
    python scripts/export_replay.py configs/mountain/medium-resort.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from avalanche.config import ConfigurationResolver
from avalanche.sim import (
    NODE_TYPE_NAMES,
    LocationKind,
    MountainSim,
    Status,
    population_from_starts,
)
from avalanche.sim.population import display_progress

REPOSITORY = Path(__file__).resolve().parent.parent
SCENE = REPOSITORY / "dashboard" / "src" / "mountain"
DEFAULT_OUTPUT = SCENE / "replay.sample.json"
DEFAULT_RESORT = SCENE / "resort.json"

SKIER_COUNT = 6
ARRIVAL_TICKS = 12
FRAME_TICKS = 4
TICK_LIMIT = 4000
KIND_NAMES = {
    LocationKind.NODE: "node",
    LocationKind.PISTE: "piste",
    LocationKind.LIFT: "lift",
    LocationKind.QUEUE: "queue",
    LocationKind.FINISHED: "finished",
}


def scene_indices(
    resort_path: Path,
) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """Return the node index and the edge index that the scene uses."""
    resort = json.loads(resort_path.read_text())
    nodes = {node["node_id"]: index for index, node in enumerate(resort["nodes"])}
    edges = {
        (edge["source"], edge["destination"]): index
        for index, edge in enumerate(resort["edges"])
    }
    return nodes, edges


def place(
    index: int,
    simulation: MountainSim,
    nodes: dict[str, int],
    edges: dict[tuple[str, str], int],
) -> list[Any]:
    """Return the place of one skier in the indices of the scene."""
    topology = simulation.topology
    assert topology is not None
    pop = simulation.population
    location_kind = LocationKind(pop.location_kind[index])
    location_index = int(pop.location_index[index])
    kind = KIND_NAMES[location_kind]
    if location_kind == LocationKind.FINISHED:
        return [kind, -1, 0.0]
    if location_kind == LocationKind.NODE:
        node_id = topology.node_ids[location_index]
        return [kind, nodes[node_id], 0.0]
    source = topology.node_ids[topology.edge_source[location_index]]
    destination = topology.node_ids[topology.edge_destination[location_index]]
    progress = display_progress(pop)
    return [kind, edges[(source, destination)], round(float(progress[index]), 4)]


def export(source: Path, resort_path: Path, output: Path) -> dict[str, Any]:
    """Run the simulator and write the replay file."""
    nodes, edges = scene_indices(resort_path)
    simulation = MountainSim(source)
    simulation.reset(seed=7)
    topology = simulation.topology
    assert topology is not None

    entrance_type = NODE_TYPE_NAMES.index("entrance")
    exit_type = NODE_TYPE_NAMES.index("exit")
    start = int(np.flatnonzero(topology.node_type == entrance_type)[0])
    exit_node = int(np.flatnonzero(topology.node_type == exit_type)[0])
    # The skiers arrive one after another, so the markers spread along the pistes.
    arrivals = [
        order * ARRIVAL_TICKS * simulation.tick_seconds for order in range(SKIER_COUNT)
    ]
    simulation.population = population_from_starts(
        starts=[start] * SKIER_COUNT,
        destinations=exit_node,
        arrival_times=arrivals,
    )
    pop = simulation.population

    frames = []
    for _ in range(TICK_LIMIT):
        simulation.tick()
        if simulation.step % FRAME_TICKS == 1:
            frames.append(
                {
                    "time": simulation.simulation_time,
                    "skiers": [
                        place(index, simulation, nodes, edges)
                        for index in range(pop.arrived)
                    ],
                }
            )
        if pop.arrived == SKIER_COUNT and bool(np.all(pop.status == Status.COMPLETE)):
            break

    replay = {
        "mountain": topology.name,
        "tick_seconds": simulation.tick_seconds,
        "skier_count": SKIER_COUNT,
        "frames": frames,
    }
    output.write_text(json.dumps(replay) + "\n")
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="the mountain component path")
    parser.add_argument("--resort", type=Path, default=DEFAULT_RESORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    resolved = ConfigurationResolver().resolve(
        arguments.source,
        "configs/scenarios/default.yaml",
        "configs/controllers/none.yaml",
        "configs/monitors/none.yaml",
    )
    replay = export(
        REPOSITORY / resolved.mountain.path,
        arguments.resort,
        arguments.output,
    )
    print(
        f"Wrote {len(replay['frames'])} frames for {replay['skier_count']} skiers "
        f"to {arguments.output}."
    )


if __name__ == "__main__":
    main()

"""Export simulator evidence to the mountain scene replay file.

Pass a mountain component to generate a short demonstration.
Pass a formal run directory to export its verified evaluator replay.
It writes `dashboard/src/mountain/replay.sample.json`.
A frame holds the simulation time and the place of each skier.
The scene reads the place and draws a marker on the piste curve.

The simulator indexes a node and an edge in its own order.
The scene indexes them in the order of the mountain file.
The script therefore maps each index through the node identity.

Usage:
    python scripts/export_replay.py configs/mountain/medium-resort.yaml
    python scripts/export_replay.py outputs/<run_id>
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
from avalanche.sim.topology import load_topology
from avalanche.traces import (
    EVALUATOR_REPLAY_FILENAME,
    load_physical_replay_snapshot,
    load_verified_run,
)

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


def export_verified_run(
    run_dir: Path,
    resort_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Export evaluator frames from one completely verified run."""
    reader = load_verified_run(run_dir)
    configuration = reader.read_configuration()
    mountain_path = Path(configuration["mountain"]["path"])
    if not mountain_path.is_absolute():
        mountain_path = REPOSITORY / mountain_path
    topology = load_topology(mountain_path)
    nodes, edges = scene_indices(resort_path)
    table = reader.read_parquet(EVALUATOR_REPLAY_FILENAME)
    frames = []
    skier_count = 0
    tick_seconds = float(configuration["intervals"]["movement_tick_seconds"])
    for encoded in table.to_pylist():
        row = load_physical_replay_snapshot(encoded)
        population = row["state"]["population"]
        location_kind = np.asarray(population["location_kind"])
        location_index = np.asarray(population["location_index"])
        required = np.asarray(population["required_travel_seconds"])
        remaining = np.asarray(population["remaining_travel_seconds"])
        skier_count = int(location_kind.size)
        frames.append(
            {
                "time": float(row["simulation_time"]),
                "skiers": [
                    _replay_place(
                        int(location_kind[index]),
                        int(location_index[index]),
                        float(required[index]),
                        float(remaining[index]),
                        topology,
                        nodes,
                        edges,
                    )
                    for index in range(skier_count)
                    if int(location_kind[index]) != int(LocationKind.PENDING)
                ],
            }
        )
    replay = {
        "mountain": topology.name,
        "tick_seconds": tick_seconds,
        "skier_count": skier_count,
        "research_manifest_sha256": reader.research_manifest_sha256,
        "frames": frames,
    }
    output.write_text(json.dumps(replay) + "\n")
    return replay


def _replay_place(
    kind_value: int,
    location_index: int,
    required_seconds: float,
    remaining_seconds: float,
    topology: Any,
    nodes: dict[str, int],
    edges: dict[tuple[str, str], int],
) -> list[Any]:
    """Map one verified replay location into scene indices."""
    kind = LocationKind(kind_value)
    name = KIND_NAMES[kind]
    if kind == LocationKind.FINISHED:
        return [name, -1, 0.0]
    if kind == LocationKind.NODE:
        return [name, nodes[topology.node_ids[location_index]], 0.0]
    source = topology.node_ids[int(topology.edge_source[location_index])]
    destination = topology.node_ids[int(topology.edge_destination[location_index])]
    progress = (
        0.0
        if required_seconds <= 0.0
        else np.clip(1.0 - remaining_seconds / required_seconds, 0.0, 1.0)
    )
    return [name, edges[(source, destination)], round(float(progress), 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="a mountain component or formal run directory")
    parser.add_argument("--resort", type=Path, default=DEFAULT_RESORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    source = Path(arguments.source)
    if source.is_dir():
        replay = export_verified_run(source, arguments.resort, arguments.output)
    else:
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

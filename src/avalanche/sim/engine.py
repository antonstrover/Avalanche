"""Run the mountain simulator.

The engine owns the reset and the movement tick loop.
The tick keeps the recorded order of the steps, because the order changes a run.
Stage 2 has no population sampling, no weather, and no hazards.
"""

import hashlib
from collections import deque
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np

from avalanche.sim.movement import (
    DynamicState,
    accumulate_times,
    advance_on_edges,
    arrive_at_nodes,
    new_dynamic_state,
    select_next_edges,
    serve_lift_queues,
)
from avalanche.sim.routes import RouteTable, build_route_table
from avalanche.sim.skier import LocationKind, Skier
from avalanche.sim.topology import Topology, load_topology

STREAM_NAMES = ("population", "weather", "failures", "controller", "monitor")
DEFAULT_TICK_SECONDS = 5.0


class MountainSim:
    """The mountain simulator of one run.

    Call `reset` one time before the first tick.
    Call `tick` for each movement tick.
    """

    def __init__(self, mountain_path: Path) -> None:
        """Store the mountain file. The reset loads it."""
        self.mountain_path = Path(mountain_path)
        self.tick_seconds = DEFAULT_TICK_SECONDS
        self.simulation_time = 0.0
        self.step = 0
        self.topology: Topology | None = None
        self.routes: RouteTable | None = None
        self.state = DynamicState()
        self.skiers: list[Skier] = []
        self.streams: dict[str, np.random.Generator] = {}

    def reset(
        self, seed: int, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode and return the first observation and the metadata.

        `options` can give the `tick_seconds` value of the run.
        """
        options = options or {}

        # 1. Make the independent random streams from the run seed.
        # A change of one part must not change the numbers of another part.
        self.streams = dict(
            zip(
                STREAM_NAMES,
                np.random.default_rng(seed).spawn(len(STREAM_NAMES)),
                strict=True,
            )
        )

        # 2. Load the immutable topology and the route table.
        self.topology = load_topology(self.mountain_path)
        self.routes = build_route_table(self.topology)

        # 3. Sample the skier attributes and the arrival times. Stage 3 adds this.
        self.skiers = []

        # 4. Start the weather and the scheduled failures. Stage 4 adds this.

        # 5. Clear the dynamic state, the trace buffers, and the metrics.
        self.tick_seconds = float(options.get("tick_seconds", DEFAULT_TICK_SECONDS))
        self.state = new_dynamic_state(self.topology)
        self.simulation_time = 0.0
        self.step = 0

        # 6. Build the first observation.
        # 7. Return the observation and the run metadata.
        return self.observation(), self.metadata(seed)

    def add_skier(self, skier: Skier) -> int:
        """Add one skier to the population and return its index.

        Stage 3 replaces this method with the sampled arrivals.
        """
        self.skiers.append(skier)
        return len(self.skiers) - 1

    def tick(self) -> None:
        """Run one movement tick in the recorded order of the steps.

        The steps read the state at the start of the tick.
        The engine commits the new state at the end of the tick.
        The order of the iteration then does not bias the result.
        """
        assert self.topology is not None, "call the reset before the first tick"
        assert self.routes is not None, "call the reset before the first tick"

        skiers = [copy(skier) for skier in self.skiers]
        state = DynamicState(
            closed=list(self.state.closed),
            queues=[deque(queue) for queue in self.state.queues],
        )

        # 1. Start the scheduled arrivals. Stage 3 adds this.
        # 2. Update the weather and the scheduled failures. Stage 4 adds this.
        # 3. Give lift service to the skiers in a queue.
        serve_lift_queues(skiers, self.topology, state, self.tick_seconds)
        # 4. Move the skiers on a piste and on a lift.
        advance_on_edges(skiers, self.topology, self.tick_seconds)
        # 5. Move the skiers that finish an edge to the destination node.
        arrive_at_nodes(skiers, self.topology)
        # 6. Select the next edge for each skier at a node.
        #    This step also applies the closures of the step 7.
        select_next_edges(skiers, self.topology, self.routes, state)
        # 7. Apply the ability limits and the capacity limits. Stage 3 adds these.
        # 8. Calculate the density, the speeds, and the hazards. Stage 4 adds this.
        # 9. Update the true outcomes and the online metrics. Stage 5 adds the metrics.
        accumulate_times(skiers, self.tick_seconds)
        # 10. Write the material events to the trace buffer. Stage 5 adds this.

        # Commit the new state together.
        self.skiers = skiers
        self.state = state
        self.simulation_time += self.tick_seconds
        self.step += 1

    def observation(self) -> dict[str, Any]:
        """Return the observation of the current state.

        Stage 4 replaces this dictionary with the Gymnasium observation.
        """
        assert self.topology is not None, "call the reset before the observation"
        occupancy = [0] * self.topology.edge_count
        for skier in self.skiers:
            if skier.location_kind in (LocationKind.PISTE, LocationKind.LIFT):
                occupancy[skier.location_index] += 1
        return {
            "simulation_time": self.simulation_time,
            "step": self.step,
            "skier_count": len(self.skiers),
            "edge_occupancy": occupancy,
            "edge_queue_length": [len(queue) for queue in self.state.queues],
            "edge_closed": list(self.state.closed),
        }

    def metadata(self, seed: int) -> dict[str, Any]:
        """Return the metadata of the run."""
        assert self.topology is not None, "call the reset before the metadata"
        return {
            "mountain": self.topology.name,
            "mountain_path": str(self.mountain_path),
            "node_count": self.topology.node_count,
            "edge_count": self.topology.edge_count,
            "seed": seed,
            "streams": list(STREAM_NAMES),
            "tick_seconds": self.tick_seconds,
        }

    def state_checksum(self) -> str:
        """Return the digest of the dynamic state.

        The digest covers the simulation time, then each skier, then each edge.
        Each part goes into the digest in the order of its index.
        A float uses a fixed representation, so the digest stays stable.
        """
        digest = hashlib.blake2b(digest_size=16)

        def add(*values: object) -> None:
            digest.update(("|".join(map(str, values)) + "\n").encode())

        add(repr(self.simulation_time))
        for skier in self.skiers:
            add(
                int(skier.location_kind),
                skier.location_index,
                repr(skier.progress),
                int(skier.status),
                skier.destination,
                repr(skier.wait_time),
                repr(skier.journey_time),
            )
        for edge, queue in enumerate(self.state.queues):
            add(int(self.state.closed[edge]), *queue)
        return digest.hexdigest()

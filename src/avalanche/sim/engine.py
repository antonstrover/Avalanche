"""Run the mountain simulator.

The engine owns the reset and the movement tick loop.
The tick keeps the recorded order of the steps, because the order changes a run.
Stage 3 has no weather and no hazards.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from avalanche.config.models import PopulationConfig
from avalanche.sim.movement import (
    DynamicState,
    accumulate_times,
    advance_on_edges,
    arrive_at_nodes,
    new_dynamic_state,
    select_next_edges,
    serve_lift_queues,
    start_arrivals,
)
from avalanche.sim.population import SkierArrays, empty_population, sample_population
from avalanche.sim.routes import RouteTable, build_route_table
from avalanche.sim.skier import LocationKind
from avalanche.sim.topology import Topology, load_topology

STREAM_NAMES = (
    "population",
    "choice",
    "weather",
    "failures",
    "controller",
    "monitor",
)
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
        self.population: SkierArrays = empty_population(0)
        self.streams: dict[str, np.random.Generator] = {}

    def reset(
        self, seed: int, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start a new episode and return the first observation and the metadata.

        `options` can give the `tick_seconds` value of the run.
        `options` can give a `population` configuration, as a model or as a dict.
        The reset keeps an empty population when `options` gives no population.
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

        # 3. Sample the skier attributes and the arrival times.
        # The population uses only its own stream, so a controller cannot change it.
        population = options.get("population")
        if population is None:
            self.population = empty_population(0)
        else:
            if not isinstance(population, PopulationConfig):
                population = PopulationConfig.model_validate(population)
            self.population = sample_population(
                self.streams["population"], self.topology, population
            )

        # 4. Start the weather and the scheduled failures. Stage 4 adds this.

        # 5. Clear the dynamic state, the trace buffers, and the metrics.
        self.tick_seconds = float(options.get("tick_seconds", DEFAULT_TICK_SECONDS))
        self.state = new_dynamic_state(self.topology)
        self.simulation_time = 0.0
        self.step = 0

        # 6. Build the first observation.
        # 7. Return the observation and the run metadata.
        return self.observation(), self.metadata(seed)

    def tick(self) -> None:
        """Run one movement tick in the recorded order of the steps.

        Each step writes into the arrays of the population.
        A masked assignment computes the whole right side before it writes.
        The order of the iteration then does not bias the result.
        """
        assert self.topology is not None, "call the reset before the first tick"
        assert self.routes is not None, "call the reset before the first tick"

        pop = self.population
        # 1. Start the scheduled arrivals.
        start_arrivals(pop, self.simulation_time, self.tick_seconds)
        # 2. Update the weather and the scheduled failures. Stage 4 adds this.
        # 3. Give lift service to the skiers in a queue.
        serve_lift_queues(pop, self.topology, self.state, self.tick_seconds)
        # 4. Move the skiers on a piste and on a lift.
        advance_on_edges(pop, self.topology, self.tick_seconds)
        # 5. Move the skiers that finish an edge to the destination node.
        arrive_at_nodes(pop, self.topology)
        # 6. Select the next edge for each skier at a node.
        #    This step also applies the closures of the step 7.
        select_next_edges(pop, self.topology, self.routes, self.state)
        # 7. Apply the ability limits and the capacity limits. Stage 3 adds these.
        # 8. Calculate the density, the speeds, and the hazards. Stage 4 adds this.
        # 9. Update the true outcomes and the online metrics. Stage 5 adds the metrics.
        accumulate_times(pop, self.tick_seconds)
        # 10. Write the material events to the trace buffer. Stage 5 adds this.

        self.simulation_time += self.tick_seconds
        self.step += 1

    def observation(self) -> dict[str, Any]:
        """Return the observation of the current state.

        Stage 4 replaces this dictionary with the Gymnasium observation.
        """
        assert self.topology is not None, "call the reset before the observation"
        pop = self.population
        edge_count = self.topology.edge_count
        on_edge = np.isin(pop.location_kind, (LocationKind.PISTE, LocationKind.LIFT))
        queued = pop.location_index[pop.location_kind == LocationKind.QUEUE]
        return {
            "simulation_time": self.simulation_time,
            "step": self.step,
            "skier_count": len(pop),
            "edge_occupancy": np.bincount(
                pop.location_index[on_edge], minlength=edge_count
            ).tolist(),
            "edge_queue_length": np.bincount(queued, minlength=edge_count).tolist(),
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

        The digest covers the simulation time, the population, and the edge state.
        The name of each field goes into the digest before the values of that field.
        A rename or a new order therefore changes the digest.
        The digest is stable on one platform.
        """
        digest = hashlib.blake2b(digest_size=16)
        state_fields = (
            ("closed", np.asarray(self.state.closed, dtype=np.bool_)),
            ("queue_length", self.state.queue_length),
        )
        digest.update(np.float64(self.simulation_time).tobytes())
        for name, array in (*self.population.checksum_fields(), *state_fields):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

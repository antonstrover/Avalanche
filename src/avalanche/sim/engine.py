"""Run the mountain simulator.

The engine owns the reset and the movement tick loop.
The tick keeps the recorded order of the steps, because the order changes a run.
The engine applies deterministic weather and hazard conditions.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from avalanche.config.models import (
    FailuresConfig,
    HazardConfig,
    PopulationConfig,
    WeatherConfig,
)
from avalanche.scenarios.failures import (
    FailureEvent,
    FailureSchedule,
    apply_failures,
    refresh_reported_telemetry,
    resolve_failure_schedule,
)
from avalanche.scenarios.weather import (
    Weather,
    WeatherSchedule,
    apply_weather,
    resolve_weather_schedule,
)
from avalanche.sim.hazards import HazardEvent, update_hazards
from avalanche.sim.movement import (
    DynamicState,
    accumulate_times,
    advance_on_edges,
    arrive_at_nodes,
    effective_closed,
    new_dynamic_state,
    select_next_edges,
    serve_lift_queues,
    start_arrivals,
    update_congestion,
)
from avalanche.sim.population import SkierArrays, empty_population, sample_population
from avalanche.sim.routes import RouteTable, build_route_table
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
        self.weather_config = WeatherConfig()
        self.weather_schedule: WeatherSchedule | None = None
        self.hazard_config = HazardConfig()
        self.hazard_events: list[HazardEvent] = []
        self.failures_config = FailuresConfig()
        self.failure_schedule: FailureSchedule | None = None
        self.active_failures: tuple[FailureEvent, ...] = ()

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

        # 4. Resolve the weather with only its independent random stream.
        weather = options.get("weather", WeatherConfig())
        if not isinstance(weather, WeatherConfig):
            weather = WeatherConfig.model_validate(weather)
        self.weather_config = weather
        self.weather_schedule = resolve_weather_schedule(
            weather, self.streams["weather"]
        )
        failures = options.get("failures", FailuresConfig())
        if not isinstance(failures, FailuresConfig):
            failures = FailuresConfig.model_validate(failures)
        self.failures_config = failures
        self.failure_schedule = resolve_failure_schedule(
            failures, self.topology, self.streams["failures"]
        )

        hazards = options.get("hazards", HazardConfig())
        if not isinstance(hazards, HazardConfig):
            hazards = HazardConfig.model_validate(hazards)
        self.hazard_config = hazards

        # 5. Clear the dynamic state, the trace buffers, and the metrics.
        self.tick_seconds = float(options.get("tick_seconds", DEFAULT_TICK_SECONDS))
        self.state = new_dynamic_state(self.topology)
        self.simulation_time = 0.0
        self.step = 0
        self.hazard_events = []
        self._update_weather()
        self._update_failures()
        refresh_reported_telemetry(self.state)

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
        # 2. Update the weather and the scheduled failures.
        self._update_weather()
        self._update_failures()
        # 3. Give lift service to the skiers in a queue.
        serve_lift_queues(pop, self.topology, self.state, self.tick_seconds)
        # 4. Move the skiers on a piste and on a lift.
        advance_on_edges(pop, self.topology, self.state, self.tick_seconds)
        # 5. Move the skiers that finish an edge to the destination node.
        arrive_at_nodes(pop, self.topology)
        # 6. Select the next edge for each skier at a node.
        # 7. Apply the closures and the capacity limits.
        #    The step 6 applies both limits, because it chooses the edge.
        select_next_edges(
            pop, self.topology, self.routes, self.state, self.streams["choice"]
        )
        # 8. Calculate the occupancy, the speeds, and the hazards.
        update_congestion(pop, self.topology, self.state)
        self.hazard_events.extend(
            update_hazards(
                self.topology,
                self.state,
                self.hazard_config,
                self.tick_seconds,
                self.simulation_time + self.tick_seconds,
            )
        )
        refresh_reported_telemetry(self.state)
        # 9. Update the true outcomes and the online metrics. Stage 5 adds the metrics.
        accumulate_times(pop, self.tick_seconds)
        # 10. Write the material events to the trace buffer. Stage 5 adds this.

        self.simulation_time += self.tick_seconds
        self.step += 1

    def _update_weather(self) -> None:
        """Apply the current scheduled weather to the simulator state."""
        assert self.topology is not None
        assert self.weather_schedule is not None
        self.weather_schedule.update(self.simulation_time)
        apply_weather(self.weather, self.weather_config, self.topology, self.state)

    def _update_failures(self) -> None:
        """Apply the active scheduled failures to the simulator state."""
        assert self.failure_schedule is not None
        self.active_failures = apply_failures(
            self.failure_schedule, self.simulation_time, self.state
        )

    @property
    def weather(self) -> Weather:
        """Return the current weather state."""
        assert self.weather_schedule is not None, "call the reset before the weather"
        return self.weather_schedule.current

    def observation(self) -> dict[str, Any]:
        """Return the observation of the current state.

        Stage 4 replaces this dictionary with the Gymnasium observation.
        """
        assert self.topology is not None, "call the reset before the observation"
        pop = self.population
        return {
            "simulation_time": self.simulation_time,
            "step": self.step,
            "skier_count": len(pop),
            "edge_occupancy": self.state.occupancy.tolist(),
            "edge_queue_length": self.state.queue_length.tolist(),
            "edge_speed_factor": self.state.speed_factor.tolist(),
            "edge_closed": effective_closed(self.state).tolist(),
            "edge_failure_closed": self.state.failure_closed.tolist(),
            "reported_edge_occupancy": self.state.reported_occupancy.tolist(),
            "reported_edge_queue_length": self.state.reported_queue_length.tolist(),
            "reported_edge_speed_factor": self.state.reported_speed_factor.tolist(),
            "reported_edge_closed": self.state.reported_closed.tolist(),
            "edge_weather_risk": self.state.weather_risk.tolist(),
            "edge_density_ratio": self.state.density_ratio.tolist(),
            "edge_hazard_score": self.state.hazard_score.tolist(),
            "edge_dangerous_duration": self.state.dangerous_duration.tolist(),
            "edge_dangerous_density_seconds": (
                self.state.dangerous_density_seconds.tolist()
            ),
            "edge_hazard_indicator": self.state.early_indicator.tolist(),
            "edge_harm": self.state.harm_active.tolist(),
            "hazard_events": [event.as_dict() for event in self.hazard_events],
            "weather": self.weather.as_array().tolist(),
            "active_failures": [
                event.as_dict()
                for event in self.active_failures
                if event.controller_visible
            ],
        }

    def metadata(self, seed: int) -> dict[str, Any]:
        """Return the metadata of the run."""
        assert self.topology is not None, "call the reset before the metadata"
        assert self.weather_schedule is not None, "call the reset before the metadata"
        assert self.failure_schedule is not None, "call the reset before the metadata"
        return {
            "mountain": self.topology.name,
            "mountain_path": str(self.mountain_path),
            "node_count": self.topology.node_count,
            "edge_count": self.topology.edge_count,
            "seed": seed,
            "streams": list(STREAM_NAMES),
            "tick_seconds": self.tick_seconds,
            "weather_schedule": [
                {
                    "start_time_seconds": transition.start_time_seconds,
                    "weather": transition.weather.as_array().tolist(),
                }
                for transition in self.weather_schedule.transitions
            ],
            "hazards": self.hazard_config.model_dump(mode="json"),
            "failure_schedule": [
                event.as_dict() for event in self.failure_schedule.events
            ],
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
            ("closed", self.state.closed),
            ("weather_closed", self.state.weather_closed),
            ("failure_closed", self.state.failure_closed),
            ("lift_stopped", self.state.lift_stopped),
            ("telemetry_late", self.state.telemetry_late),
            ("occupancy", self.state.occupancy),
            ("queue_length", self.state.queue_length),
            ("speed_factor", self.state.speed_factor),
            ("weather_risk", self.state.weather_risk),
            ("density_ratio", self.state.density_ratio),
            ("hazard_score", self.state.hazard_score),
            ("dangerous_duration", self.state.dangerous_duration),
            (
                "dangerous_density_seconds",
                self.state.dangerous_density_seconds,
            ),
            ("early_indicator", self.state.early_indicator),
            ("harm_active", self.state.harm_active),
            ("indicator_count", self.state.indicator_count),
            ("harm_count", self.state.harm_count),
            ("reported_occupancy", self.state.reported_occupancy),
            ("reported_queue_length", self.state.reported_queue_length),
            ("reported_speed_factor", self.state.reported_speed_factor),
            ("reported_closed", self.state.reported_closed),
        )
        digest.update(np.float64(self.simulation_time).tobytes())
        if self.weather_schedule is not None:
            digest.update(b"weather")
            digest.update(self.weather.as_array().tobytes())
        for name, array in (*self.population.checksum_fields(), *state_fields):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(array).tobytes())
        return digest.hexdigest()

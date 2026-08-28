"""Run the mountain simulator.

The engine owns the reset and the movement tick loop.
The tick keeps the recorded order of the steps, because the order changes a run.
The engine applies deterministic weather and hazard conditions.
"""

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from avalanche.config.models import (
    PROTOCOL_TIME_EPSILON_SECONDS,
    AuditConfig,
    FailuresConfig,
    HazardConfig,
    NumericsConfig,
    OperationalEventsConfig,
    PopulationConfig,
    ReportedRiskConfig,
    RoutingConfig,
    SensorPolicyConfig,
    WeatherConfig,
)
from avalanche.metrics import OnlineMetrics
from avalanche.scenarios.audits import AuditChannel, AuditMeasurement
from avalanche.scenarios.failures import (
    FailureEvent,
    FailureKind,
    FailureSchedule,
    apply_failures,
    refresh_reported_telemetry,
    resolve_failure_schedule,
)
from avalanche.scenarios.operational_events import (
    EVENT_STREAM_NAMES,
    OperationalEvent,
    OperationalEventSchedule,
    resolve_operational_event_schedule,
)
from avalanche.scenarios.sensors import RouteSensorChannel, RouteSensorPacket
from avalanche.scenarios.weather import (
    Weather,
    WeatherSchedule,
    apply_weather,
    resolve_weather_schedule,
)
from avalanche.sim.hazards import HazardEvent, update_hazards
from avalanche.sim.movement import (
    DynamicState,
    MovementTransitions,
    accumulate_times,
    advance_on_edges,
    arrive_at_nodes,
    effective_closed,
    new_dynamic_state,
    select_next_edges,
    serve_lift_queues,
    start_arrivals,
    update_congestion,
    update_stranded,
)
from avalanche.sim.population import (
    CUSTOMER_GROUP_NAMES,
    SkierArrays,
    empty_population,
    sample_population,
)
from avalanche.sim.routes import RouteTable, build_route_table
from avalanche.sim.skier import LocationKind, Status
from avalanche.sim.topology import Topology, load_topology

STREAM_NAMES = (
    "population",
    "choice",
    "weather",
    "failures",
    "controller",
    "monitor",
    "audit",
    "policy",
    *EVENT_STREAM_NAMES,
    "sensor",
    "route_tie",
)
DEFAULT_TICK_SECONDS = 5.0
DEFAULT_EPISODE_SECONDS = 3_600.0


def _spawn_random_streams(seed: int) -> dict[str, np.random.Generator]:
    """Create each independent random stream for one run."""
    return dict(
        zip(
            STREAM_NAMES,
            np.random.default_rng(seed).spawn(len(STREAM_NAMES)),
            strict=True,
        )
    )


class MountainSim:
    """The mountain simulator of one run.

    Call `reset` one time before the first tick.
    Call `tick` for each movement tick.
    """

    def __init__(self, mountain_path: Path) -> None:
        """Store the mountain file. The reset loads it."""
        self.mountain_path = Path(mountain_path)
        self.tick_seconds = DEFAULT_TICK_SECONDS
        self.time_epsilon_seconds = NumericsConfig(
            time_epsilon_seconds=PROTOCOL_TIME_EPSILON_SECONDS
        ).time_epsilon_seconds
        self.simulation_time = 0.0
        self.step = 0
        self.topology: Topology | None = None
        self.routes: RouteTable | None = None
        self.routing_config = RoutingConfig()
        self.reported_risk_config = ReportedRiskConfig()
        self.route_sensor_config = SensorPolicyConfig()
        self.control_interval_seconds = 60.0
        self.route_sensor_channel: RouteSensorChannel | None = None
        self.route_sensor_packet: RouteSensorPacket | None = None
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
        self.audit_config = AuditConfig()
        self.audit_channel: AuditChannel | None = None
        self.delivered_audits: tuple[AuditMeasurement, ...] = ()
        self.operational_event_schedule: OperationalEventSchedule | None = None
        self.active_operational_events: tuple[OperationalEvent, ...] = ()
        self.metrics = OnlineMetrics(len(CUSTOMER_GROUP_NAMES), DEFAULT_EPISODE_SECONDS)
        self.last_movement_transitions = MovementTransitions(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        )

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
        self.streams = _spawn_random_streams(seed)

        # 2. Load the immutable topology and the route table.
        self.topology = load_topology(self.mountain_path)
        self.routes = build_route_table(self.topology)
        routing = options.get("routing", RoutingConfig())
        if not isinstance(routing, RoutingConfig):
            routing = RoutingConfig.model_validate(routing)
        self.routing_config = routing
        route_sensor = options.get("route_sensor", SensorPolicyConfig())
        if not isinstance(route_sensor, SensorPolicyConfig):
            route_sensor = SensorPolicyConfig.model_validate(route_sensor)
        self.route_sensor_config = route_sensor
        reported_risk = options.get("reported_risk", ReportedRiskConfig())
        if not isinstance(reported_risk, ReportedRiskConfig):
            reported_risk = ReportedRiskConfig.model_validate(reported_risk)
        self.reported_risk_config = reported_risk
        self.control_interval_seconds = float(
            options.get("control_interval_seconds", 60.0)
        )

        # 3. Sample the skier attributes and the arrival times.
        # The population uses only its own stream, so a controller cannot change it.
        population = options.get("population")
        if population is None:
            self.population = empty_population(0)
        else:
            if not isinstance(population, PopulationConfig):
                population = PopulationConfig.model_validate(population)
            self.population = sample_population(
                self.streams["population"], self.topology, self.routes, population
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
        audits = options.get("audits", AuditConfig())
        if not isinstance(audits, AuditConfig):
            audits = AuditConfig.model_validate(audits)
        self.audit_config = audits
        self.audit_channel = AuditChannel(audits, self.streams["audit"])
        self.delivered_audits = ()
        operational_events = options.get("operational_events", {})
        self.operational_event_schedule = resolve_operational_event_schedule(
            OperationalEventsConfig.model_validate(operational_events),
            self.topology,
            self.streams,
        )
        self.active_operational_events = self.operational_event_schedule.active(0.0)

        hazards = options.get("hazards", HazardConfig())
        if not isinstance(hazards, HazardConfig):
            hazards = HazardConfig.model_validate(hazards)
        self.hazard_config = hazards

        numerics = options.get(
            "numerics",
            NumericsConfig(time_epsilon_seconds=PROTOCOL_TIME_EPSILON_SECONDS),
        )
        if not isinstance(numerics, NumericsConfig):
            numerics = NumericsConfig.model_validate(numerics)
        self.time_epsilon_seconds = numerics.time_epsilon_seconds

        # 5. Clear the dynamic state, the trace buffers, and the metrics.
        self.tick_seconds = float(options.get("tick_seconds", DEFAULT_TICK_SECONDS))
        self.state = new_dynamic_state(self.topology)
        self.simulation_time = 0.0
        self.step = 0
        self.hazard_events = []
        episode_seconds = float(
            options.get("episode_duration_seconds", DEFAULT_EPISODE_SECONDS)
        )
        self.metrics = OnlineMetrics(len(CUSTOMER_GROUP_NAMES), episode_seconds)
        self.last_movement_transitions = MovementTransitions(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        )
        self._update_weather()
        self._update_failures()
        self._update_operational_events()
        refresh_reported_telemetry(self.state, self.topology)
        self.route_sensor_channel = RouteSensorChannel(
            self.route_sensor_config,
            self.control_interval_seconds,
            self.streams["sensor"],
        )
        self.route_sensor_packet = self.route_sensor_channel.bootstrap(
            **self._route_sensor_sources()
        )

        # 6. Build the first observation.
        # 7. Return the observation and the run metadata.
        return self.observation(), self.metadata(seed)

    def tick(self) -> MovementTransitions:
        """Run one movement tick in the recorded order of the steps.

        Each step writes into the arrays of the population.
        A masked assignment computes the whole right side before it writes.
        The order of the iteration then does not bias the result.
        """
        assert self.topology is not None, "call the reset before the first tick"
        assert self.routes is not None, "call the reset before the first tick"

        pop = self.population
        # 1. Start the scheduled arrivals.
        start_arrivals(pop, self.simulation_time)
        active_at_tick_start = (pop.status == Status.ACTIVE) & (
            pop.location_kind != LocationKind.PENDING
        )
        stranded_at_tick_start = pop.status == Status.STRANDED
        # 2. Update the weather and the scheduled failures.
        self._update_weather()
        self._update_failures()
        # 3. Give lift service to the skiers in a queue.
        serve_lift_queues(pop, self.topology, self.state, self.tick_seconds)
        queued_at_tick_start = active_at_tick_start & (
            pop.location_kind == LocationKind.QUEUE
        )
        # 4. Move the skiers on a piste and on a lift.
        advance_on_edges(pop, self.topology, self.state, self.tick_seconds)
        # 5. Move the skiers that finish an edge to the destination node.
        transitions = arrive_at_nodes(
            pop,
            self.topology,
            self.simulation_time,
            self.tick_seconds,
            self.time_epsilon_seconds,
        )
        self.last_movement_transitions = transitions
        choice_time = self.simulation_time + self.tick_seconds
        if self._is_control_boundary(choice_time):
            assert self.route_sensor_channel is not None
            self.route_sensor_packet = self.route_sensor_channel.deliver(choice_time)
        # 6. Select the next edge for each skier at a node.
        # 7. Apply the closures and the capacity limits.
        #    The step 6 applies both limits, because it chooses the edge.
        route_decisions = select_next_edges(
            pop,
            self.topology,
            self.routes,
            self.state,
            self.streams["choice"],
            self.streams["route_tie"],
            self.route_sensor_packet,
            self.routing_config,
            self.reported_risk_config,
        )
        self.metrics.update_route_decisions(route_decisions)
        update_stranded(
            pop,
            self.routes,
            self.state,
            self.tick_seconds,
            self.hazard_config.stranded_after_seconds,
            self.time_epsilon_seconds,
            topology=self.topology,
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
                self.time_epsilon_seconds,
            )
        )
        refresh_reported_telemetry(self.state, self.topology)
        # 9. Update the true outcomes and the online metrics. Stage 5 adds the metrics.
        accumulate_times(
            pop,
            self.tick_seconds,
            active_at_tick_start=active_at_tick_start,
            queued_at_tick_start=queued_at_tick_start,
        )
        self.metrics.update(
            pop,
            self.state,
            self.tick_seconds,
            stranded_at_tick_start=stranded_at_tick_start,
        )
        # 10. Write the material events to the trace buffer. Stage 5 adds this.

        self.simulation_time += self.tick_seconds
        self.step += 1
        if self._is_control_boundary(self.simulation_time):
            assert self.route_sensor_channel is not None
            self.route_sensor_packet = self.route_sensor_channel.advance(
                self.simulation_time,
                **self._route_sensor_sources(),
            )
        self._update_operational_events()
        return transitions

    def _is_control_boundary(self, simulation_time: float) -> bool:
        """Return whether one time is an exact control boundary."""
        ratio = simulation_time / self.control_interval_seconds
        return abs(ratio - round(ratio)) <= self.time_epsilon_seconds

    def _route_sensor_sources(self) -> dict[str, np.ndarray]:
        """Return the current reported sources for route sampling."""
        assert self.topology is not None
        closed = self.state.closed | self.state.weather_closed
        for event in self.active_failures:
            if not event.controller_visible:
                continue
            if event.kind in (FailureKind.LIFT_STOPPAGE, FailureKind.SUDDEN_CLOSURE):
                closed[event.target] = True
        throughput = (
            self.topology.edge_lift_throughput.astype(np.float64) / 3600.0
        ) * self.state.lift_capacity_factor
        return {
            "availability": ~closed,
            "speed_factor": self.state.reported_speed_factor,
            "density_ratio": self.state.reported_density_ratio,
            "weather_risk": self.state.weather_risk,
            "queue_length": self.state.reported_queue_length,
            "boarding_throughput": throughput,
        }

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

    def _update_operational_events(self) -> None:
        """Expose each active honest operating event."""
        assert self.operational_event_schedule is not None
        self.active_operational_events = self.operational_event_schedule.active(
            self.simulation_time
        )

    def advance_audits(self, interval: int) -> tuple[AuditMeasurement, ...]:
        """Sample audits and deliver measurements due this interval."""
        if self.topology is None or self.audit_channel is None:
            raise RuntimeError("reset the simulator before an audit sample")
        capacity = np.maximum(self.topology.edge_safe_capacity, 1.0)
        true_density = np.divide(
            self.state.occupancy + self.state.queue_length,
            capacity,
            dtype=np.float64,
        )
        reported_density = np.divide(
            self.state.reported_occupancy + self.state.reported_queue_length,
            capacity,
            dtype=np.float64,
        )
        self.delivered_audits = self.audit_channel.advance(
            interval, true_density, reported_density
        )
        return self.delivered_audits

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
        packet = self.route_sensor_packet
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
            "reported_edge_density_ratio": self.state.reported_density_ratio.tolist(),
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
            "operational_events": [
                event.public(self.simulation_time)
                for event in self.active_operational_events
            ],
            "route_sensor": (
                None
                if packet is None
                else {
                    "schema_version": packet.schema_version,
                    "sample_time": packet.sample_time,
                    "report_time": packet.report_time,
                    "provenance": dict(packet.provenance),
                    "policy_identity": packet.policy_identity,
                    "reported_availability": packet.reported_availability.tolist(),
                    "reported_speed_factor": packet.reported_speed_factor.tolist(),
                    "reported_density_ratio": packet.reported_density_ratio.tolist(),
                    "reported_weather_risk": packet.reported_weather_risk.tolist(),
                    "reported_queue_length": packet.reported_queue_length.tolist(),
                    "reported_boarding_throughput": (
                        packet.reported_boarding_throughput.tolist()
                    ),
                    "availability_missing": packet.availability_missing.tolist(),
                    "speed_factor_missing": packet.speed_factor_missing.tolist(),
                    "density_ratio_missing": packet.density_ratio_missing.tolist(),
                    "weather_risk_missing": packet.weather_risk_missing.tolist(),
                    "queue_length_missing": packet.queue_length_missing.tolist(),
                    "boarding_throughput_missing": (
                        packet.boarding_throughput_missing.tolist()
                    ),
                }
            ),
        }

    def metadata(self, seed: int) -> dict[str, Any]:
        """Return the metadata of the run."""
        assert self.topology is not None, "call the reset before the metadata"
        assert self.weather_schedule is not None, "call the reset before the metadata"
        assert self.failure_schedule is not None, "call the reset before the metadata"
        assert self.operational_event_schedule is not None
        return {
            "mountain": self.topology.name,
            "mountain_path": str(self.mountain_path),
            "node_count": self.topology.node_count,
            "edge_count": self.topology.edge_count,
            "seed": seed,
            "streams": list(STREAM_NAMES),
            "tick_seconds": self.tick_seconds,
            "time_epsilon_seconds": self.time_epsilon_seconds,
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
            "audits": self.audit_config.model_dump(mode="json"),
            "operational_event_schedule": [
                event.complete() for event in self.operational_event_schedule.events
            ],
        }

    def state_checksum(self) -> str:
        """Return the digest of the dynamic state.

        The digest covers each input that can change a later movement tick.
        It excludes immutable configuration and derived event views.
        It excludes metrics and history because they cannot change movement.
        The digest is stable on one platform.
        """
        digest = hashlib.blake2b(digest_size=16)
        _digest_array(digest, "simulation_time", np.array(self.simulation_time, "<f8"))
        _digest_array(digest, "step", np.array(self.step, "<i8"))
        _digest_array(digest, "tick_seconds", np.array(self.tick_seconds, "<f8"))
        _digest_array(
            digest,
            "time_epsilon_seconds",
            np.array(self.time_epsilon_seconds, "<f8"),
        )
        _digest_array(
            digest,
            "population.arrived",
            np.array(self.population.arrived, "<i8"),
        )
        _digest_array(
            digest,
            "population.next_ticket",
            np.array(self.population.next_ticket, "<i8"),
        )
        if self.weather_schedule is not None:
            _digest_array(digest, "weather", self.weather.as_array())
            _digest_array(
                digest,
                "weather.next_transition",
                np.array(self.weather_schedule.next_transition, "<i8"),
            )
        for name, array in self.population.checksum_fields():
            _digest_array(digest, f"population.{name}", array)
        for name, array in self.state.checksum_fields():
            _digest_array(digest, f"state.{name}", array)
        for stream_name in ("choice", "sensor", "route_tie"):
            stream = self.streams.get(stream_name)
            if stream is None:
                continue
            _digest_bytes(
                digest,
                f"random.{stream_name}",
                json.dumps(
                    stream.bit_generator.state,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            )
        if self.route_sensor_packet is not None:
            _digest_route_packet(
                digest, "route_sensor.latest", self.route_sensor_packet
            )
        if self.route_sensor_channel is not None:
            for index, packet in enumerate(self.route_sensor_channel.pending):
                _digest_route_packet(digest, f"route_sensor.pending.{index}", packet)
        return digest.hexdigest()


def _digest_route_packet(digest: Any, prefix: str, packet: RouteSensorPacket) -> None:
    """Add one complete route packet to a state digest."""
    _digest_array(digest, f"{prefix}.sample_time", np.array(packet.sample_time, "<f8"))
    _digest_array(digest, f"{prefix}.report_time", np.array(packet.report_time, "<f8"))
    _digest_bytes(digest, f"{prefix}.policy_identity", packet.policy_identity.encode())
    for name in (
        "reported_availability",
        "reported_speed_factor",
        "reported_density_ratio",
        "reported_weather_risk",
        "reported_queue_length",
        "reported_boarding_throughput",
        "availability_missing",
        "speed_factor_missing",
        "density_ratio_missing",
        "weather_risk_missing",
        "queue_length_missing",
        "boarding_throughput_missing",
    ):
        _digest_array(digest, f"{prefix}.{name}", getattr(packet, name))


def _digest_array(digest: Any, name: str, values: np.ndarray) -> None:
    """Add one named array with a stable type and shape."""
    array = np.asarray(values)
    dtype = array.dtype.newbyteorder("<")
    portable = np.ascontiguousarray(array, dtype=dtype)
    _digest_bytes(digest, name, portable.tobytes())
    _digest_bytes(digest, f"{name}.dtype", dtype.str.encode())
    shape = struct.pack("<Q", portable.ndim) + b"".join(
        struct.pack("<Q", size) for size in portable.shape
    )
    _digest_bytes(digest, f"{name}.shape", shape)


def _digest_bytes(digest: Any, name: str, values: bytes) -> None:
    """Add one named byte value with unambiguous lengths."""
    encoded_name = name.encode()
    digest.update(struct.pack("<Q", len(encoded_name)))
    digest.update(encoded_name)
    digest.update(struct.pack("<Q", len(values)))
    digest.update(values)

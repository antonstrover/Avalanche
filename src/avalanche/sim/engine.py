"""Run the mountain simulator.

The engine owns the reset and the movement tick loop.
The tick keeps the recorded order of the steps, because the order changes a run.
The engine applies deterministic weather and hazard conditions.
"""

from pathlib import Path
from typing import Any

import numpy as np

from avalanche.config.models import (
    PROTOCOL_TIME_EPSILON_SECONDS,
    AuditConfig,
    EnvironmentContextConfig,
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
    FailureTransitions,
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
from avalanche.scenarios.sensors import (
    FAILURE_SENSOR_CAPACITY,
    RouteSensorChannel,
    RouteSensorPacket,
)
from avalanche.scenarios.weather import (
    Weather,
    WeatherSchedule,
    apply_weather,
    resolve_weather_schedule,
)
from avalanche.sim.evacuation import (
    ResolvedEnvironmentContext,
    resolve_environment_context,
)
from avalanche.sim.hazards import HazardEvent, update_hazards
from avalanche.sim.movement import (
    DynamicState,
    MovementTransitions,
    accumulate_times,
    advance_on_edges,
    arrive_at_nodes,
    effective_closed,
    lift_unavailable_mask,
    new_dynamic_state,
    return_unavailable_lift_queues,
    select_next_edges,
    serve_lift_queues,
    start_arrivals,
    update_congestion,
    update_lift_blocked_times,
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
    "blocked_sensor",
    "stranding_sensor",
    "operational_sensor",
    "audit_missing",
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
        self.failure_transitions = FailureTransitions((), (), ())
        self.audit_config = AuditConfig()
        self.audit_channel: AuditChannel | None = None
        self.delivered_audits: tuple[AuditMeasurement, ...] = ()
        self.operational_event_schedule: OperationalEventSchedule | None = None
        self.active_operational_events: tuple[OperationalEvent, ...] = ()
        self.environment_context = ResolvedEnvironmentContext((), (), 0.0)
        self.metrics = OnlineMetrics(len(CUSTOMER_GROUP_NAMES), DEFAULT_EPISODE_SECONDS)
        self.last_movement_transitions = MovementTransitions(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        )
        self._stranding_interval_counts: dict[tuple[str, str], int] = {}

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
        self.audit_channel = AuditChannel(
            audits,
            self.streams["audit"],
            self.control_interval_seconds,
            self.streams["audit_missing"],
        )
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
        self.active_failures = ()
        self.failure_transitions = FailureTransitions((), (), ())
        episode_seconds = float(
            options.get("episode_duration_seconds", DEFAULT_EPISODE_SECONDS)
        )
        self.last_movement_transitions = MovementTransitions(
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        )
        self._stranding_interval_counts = {}
        self._update_weather()
        self._update_failures()
        self._update_operational_events()
        environment_context = options.get("environment_context")
        if environment_context is None:
            self.environment_context = ResolvedEnvironmentContext((), (), 0.0)
        else:
            if not isinstance(environment_context, EnvironmentContextConfig):
                environment_context = EnvironmentContextConfig.model_validate(
                    environment_context
                )
            self.environment_context = resolve_environment_context(
                self.topology,
                self.state,
                environment_context.for_mountain(self.topology.name),
            )
        self.metrics = OnlineMetrics(
            len(CUSTOMER_GROUP_NAMES),
            episode_seconds,
            topology=self.topology,
            environment_context=self.environment_context,
        )
        refresh_reported_telemetry(self.state, self.topology)
        self.route_sensor_channel = RouteSensorChannel(
            self.route_sensor_config,
            self.control_interval_seconds,
            self.streams["sensor"],
            self.streams["blocked_sensor"],
            self.streams["stranding_sensor"],
            self.streams["operational_sensor"],
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
        unavailable_lifts = lift_unavailable_mask(self.topology, self.state)
        returned_queue_skiers = return_unavailable_lift_queues(
            pop,
            self.topology,
            self.state,
            unavailable_lifts,
        )
        # 3. Give lift service to the skiers in a queue.
        onward_rejected_skiers = serve_lift_queues(
            pop, self.topology, self.state, self.tick_seconds
        )
        returned_queue_skiers = np.union1d(
            returned_queue_skiers,
            onward_rejected_skiers,
        ).astype(np.int64, copy=False)
        lift_stranding_candidates = update_lift_blocked_times(
            pop,
            self.topology,
            self.state,
            returned_queue_skiers,
            self.tick_seconds,
            self.hazard_config.stranded_after_seconds,
            self.time_epsilon_seconds,
        )
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
        choice_time = self.simulation_time + self.tick_seconds
        control_interval_index = int(
            self.simulation_time / self.control_interval_seconds
        )
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
        node_stranding_candidates = update_stranded(
            pop,
            self.routes,
            self.state,
            self.tick_seconds,
            self.hazard_config.stranded_after_seconds,
            self.time_epsilon_seconds,
            topology=self.topology,
        )
        stranding_candidates = np.union1d(
            lift_stranding_candidates,
            node_stranding_candidates,
        ).astype(np.int64, copy=False)
        newly_stranded = stranding_candidates[
            (pop.status[stranding_candidates] == Status.ACTIVE)
            & ~pop.ever_stranded[stranding_candidates]
        ]
        pop.status[newly_stranded] = Status.STRANDED
        pop.first_stranded_at[newly_stranded] = choice_time
        pop.ever_stranded[newly_stranded] = True
        transitions = MovementTransitions(
            completed_skiers=transitions.completed_skiers,
            edge_completed_at=transitions.edge_completed_at,
            newly_stranded_indices=newly_stranded,
            stranding_boundary_seconds=(choice_time if newly_stranded.size else None),
            control_interval_index=(
                control_interval_index if newly_stranded.size else None
            ),
        )
        self.last_movement_transitions = transitions
        self._record_stranding(newly_stranded)
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
            newly_stranded_skiers=int(newly_stranded.size),
            movement_boundary_seconds=choice_time,
            control_interval_index=control_interval_index,
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
            self._stranding_interval_counts = {}
        self._update_operational_events()
        return transitions

    def _is_control_boundary(self, simulation_time: float) -> bool:
        """Return whether one time is an exact control boundary."""
        ratio = simulation_time / self.control_interval_seconds
        return abs(ratio - round(ratio)) <= self.time_epsilon_seconds

    def _route_sensor_sources(self) -> dict[str, Any]:
        """Return the current reported sources for route sampling."""
        assert self.topology is not None
        visible_failure_closed = np.zeros(self.topology.edge_count, dtype=np.bool_)
        for event in self.active_failures:
            if event.controller_visible and event.kind != FailureKind.LATE_TELEMETRY:
                visible_failure_closed[event.target] = True
        closed = self.state.closed | self.state.weather_closed | visible_failure_closed
        for event in self.active_failures:
            if not event.controller_visible:
                continue
            if event.kind in (FailureKind.LIFT_STOPPAGE, FailureKind.SUDDEN_CLOSURE):
                closed[event.target] = True
        throughput = (
            self.topology.edge_lift_throughput.astype(np.float64) / 3600.0
        ) * self.state.lift_capacity_factor
        queued = (self.population.queue_no_route_blocked_seconds > 0.0) & (
            self.population.location_kind == LocationKind.NODE
        )
        queued_no_route_count = np.bincount(
            self.population.location_index[queued],
            minlength=self.topology.node_count,
        ).astype(np.float64)
        onboard = (self.population.onboard_blocked_seconds > 0.0) & (
            self.population.location_kind == LocationKind.LIFT
        )
        onboard_blocked_count = np.bincount(
            self.population.location_index[onboard],
            minlength=self.topology.edge_count,
        ).astype(np.float64)
        speed = self.state.reported_speed_factor.copy()
        for event in self.active_failures:
            if event.controller_visible or event.kind != FailureKind.LIFT_STOPPAGE:
                continue
            speed[event.target] = np.clip(
                self.state.congestion_speed_factor[event.target]
                * self.state.weather_speed_factor[event.target],
                self.routing_config.minimum_reported_speed_factor,
                1.0,
            )
        pending = self.population.location_kind == LocationKind.PENDING
        at_node = self.population.location_kind == LocationKind.NODE
        demand_locations = self.population.location_index[pending | at_node]
        node_demand = np.bincount(
            demand_locations,
            minlength=self.topology.node_count,
        ).astype("<i8")
        node_crowding = np.bincount(
            self.population.location_index[at_node],
            minlength=self.topology.node_count,
        ).astype("<i8")
        lift_code = 1
        lift_occupancy = np.where(
            self.topology.edge_type == lift_code,
            self.state.reported_occupancy,
            0,
        ).astype("<i8")
        failure_kind = np.zeros(FAILURE_SENSOR_CAPACITY, dtype="<i2")
        failure_target = np.zeros(FAILURE_SENSOR_CAPACITY, dtype="<i4")
        failure_present = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.bool_)
        visible = tuple(
            event for event in self.active_failures if event.controller_visible
        )[:FAILURE_SENSOR_CAPACITY]
        failure_names = tuple(FailureKind)
        for index, event in enumerate(visible):
            failure_kind[index] = failure_names.index(event.kind) + 1
            failure_target[index] = event.target
            failure_present[index] = True
        return {
            "availability": ~closed,
            "speed_factor": speed,
            "density_ratio": self.state.reported_density_ratio,
            "weather_risk": self.state.weather_risk,
            "queue_length": self.state.reported_queue_length,
            "boarding_throughput": throughput,
            "queued_no_route_count": queued_no_route_count,
            "onboard_blocked_count": onboard_blocked_count,
            "node_demand": node_demand,
            "node_crowding": node_crowding,
            "edge_occupancy": self.state.reported_occupancy.astype("<i8"),
            "lift_occupancy": lift_occupancy,
            "weather": self.weather.as_array().astype("<f8"),
            "visible_failure_kind": failure_kind,
            "visible_failure_target": failure_target,
            "visible_failure_present": failure_present,
            "stranding_locations": tuple(
                (kind, topology_id, count)
                for (kind, topology_id), count in sorted(
                    self._stranding_interval_counts.items()
                )
            ),
        }

    def _record_stranding(self, skier_indices: np.ndarray) -> None:
        """Group new stranding by its public topology location."""
        if skier_indices.size == 0:
            return
        assert self.topology is not None
        for skier in skier_indices:
            location_kind = LocationKind(int(self.population.location_kind[skier]))
            location_index = int(self.population.location_index[skier])
            if location_kind == LocationKind.NODE:
                kind = location_kind.name.lower()
                topology_id = self.topology.node_ids[location_index]
            elif location_kind in {
                LocationKind.PISTE,
                LocationKind.LIFT,
                LocationKind.QUEUE,
            }:
                kind = location_kind.name.lower()
                source = self.topology.node_ids[
                    int(self.topology.edge_source[location_index])
                ]
                destination = self.topology.node_ids[
                    int(self.topology.edge_destination[location_index])
                ]
                topology_id = f"{source}->{destination}"
            else:
                continue
            key = (kind, topology_id)
            self._stranding_interval_counts[key] = (
                self._stranding_interval_counts.get(key, 0) + 1
            )

    def _update_weather(self) -> None:
        """Apply the current scheduled weather to the simulator state."""
        assert self.topology is not None
        assert self.weather_schedule is not None
        self.weather_schedule.update(self.simulation_time)
        apply_weather(self.weather, self.weather_config, self.topology, self.state)

    def _update_failures(self) -> None:
        """Apply the active scheduled failures to the simulator state."""
        assert self.failure_schedule is not None
        self.failure_transitions = apply_failures(
            self.failure_schedule,
            self.simulation_time,
            self.state,
            self.active_failures,
        )
        self.active_failures = self.failure_transitions.active

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
            "edge_density_warning": self.state.early_indicator.tolist(),
            "edge_dangerous_density_active": (
                self.state.dangerous_density_active.tolist()
            ),
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
                    "reported_queued_no_route_count": (
                        packet.reported_queued_no_route_count.tolist()
                    ),
                    "reported_onboard_blocked_count": (
                        packet.reported_onboard_blocked_count.tolist()
                    ),
                    "availability_missing": packet.availability_missing.tolist(),
                    "speed_factor_missing": packet.speed_factor_missing.tolist(),
                    "density_ratio_missing": packet.density_ratio_missing.tolist(),
                    "weather_risk_missing": packet.weather_risk_missing.tolist(),
                    "queue_length_missing": packet.queue_length_missing.tolist(),
                    "boarding_throughput_missing": (
                        packet.boarding_throughput_missing.tolist()
                    ),
                    "queued_no_route_count_missing": (
                        packet.queued_no_route_count_missing.tolist()
                    ),
                    "onboard_blocked_count_missing": (
                        packet.onboard_blocked_count_missing.tolist()
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

    def physical_replay_state(self, view_kind: str = "evaluator") -> dict[str, Any]:
        """Return the replay-visible physical state for one display view."""
        if view_kind not in {"reported", "evaluator"}:
            raise ValueError("the physical replay view is invalid")
        if self.topology is None or self.route_sensor_packet is None:
            raise RuntimeError("reset the simulator before replay work")
        if view_kind == "reported":
            packet = self.route_sensor_packet.operational_packet
            if packet is None:
                raise RuntimeError("the reported replay needs an operational packet")
            sensors = {sensor.name: sensor for sensor in packet.sensors}
            state = {
                "node": {
                    name: sensors[name].values
                    for name in ("node_demand", "node_crowding")
                },
                "edge": {
                    name: sensors[name].values
                    for name in (
                        "edge_occupancy",
                        "edge_density",
                        "edge_speed_factor",
                        "edge_availability",
                        "edge_weather_risk",
                        "lift_queue_length",
                        "lift_occupancy",
                        "lift_boarding_throughput",
                    )
                },
                "masks": {name: sensor.missing for name, sensor in sensors.items()},
                "weather": sensors["weather"].values,
                "failures": {
                    name: sensors[name].values
                    for name in (
                        "visible_failure_kind",
                        "visible_failure_target",
                        "visible_failure_present",
                    )
                },
                "precursors": {
                    "reported_density_ratio": sensors["edge_density"].values,
                    "reported_hazard_score": (
                        sensors["edge_density"].values
                        + self.hazard_config.weather_risk_weight
                        * sensors["edge_weather_risk"].values
                    ),
                },
                "reported_stranding": tuple(
                    {
                        "location_kind": item.location_kind,
                        "topology_id": item.topology_id,
                        "count": item.count,
                        "missing": item.missing,
                    }
                    for item in self.route_sensor_packet.reported_stranding
                ),
            }
        else:
            at_node = self.population.location_kind == LocationKind.NODE
            node_crowding = np.bincount(
                self.population.location_index[at_node],
                minlength=self.topology.node_count,
            ).astype("<i8")
            state = {
                "node": {"node_crowding": node_crowding},
                "edge": {
                    "occupancy": self.state.occupancy,
                    "queue_length": self.state.queue_length,
                    "density_ratio": self.state.density_ratio,
                    "speed_factor": self.state.speed_factor,
                    "availability": ~effective_closed(self.state),
                    "weather_risk": self.state.weather_risk,
                },
                "weather": self.weather.as_array(),
                "failures": tuple(item.as_dict() for item in self.active_failures),
                "precursors": {
                    "hazard_score": self.state.hazard_score,
                    "early_indicator": self.state.early_indicator,
                    "dangerous_density_active": self.state.dangerous_density_active,
                },
                "population": {
                    "location_kind": self.population.location_kind,
                    "location_index": self.population.location_index,
                    "required_travel_seconds": (
                        self.population.required_travel_seconds
                    ),
                    "remaining_travel_seconds": (
                        self.population.remaining_travel_seconds
                    ),
                    "status": self.population.status,
                },
            }
        return {
            "view_kind": view_kind,
            "simulation_time": self.simulation_time,
            "movement_tick": self.step,
            "topology_artifact_reference": {
                "name": self.topology.name,
                "sha256": self.topology.mountain_sha256,
            },
            "state": state,
        }

    def physical_state_checksum(self, view_kind: str = "evaluator") -> str:
        """Return the SHA-256 identity of one replay display view."""
        from avalanche.traces.checksums import named_checksum

        return named_checksum(
            self.physical_replay_state(view_kind),
            allow_nonfinite=True,
        )

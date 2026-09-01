"""Check each observation information boundary."""

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from avalanche.config.models import (
    AttackBudgetConfig,
    AttackRecordConfig,
    AttackSuccessConfig,
    AttackTriggerConfig,
    ControllerConfig,
    MonitorConfig,
    SensorPolicyConfig,
)
from avalanche.control import (
    ActionProposal,
    Adjudicator,
    ControllerObservation,
    DecisionType,
    EvaluatorObservation,
    InformationProfile,
    MonitorDecision,
    OutcomeObservation,
    ProcessObservation,
    build_controller_observation,
    build_evaluator_observation,
    build_monitor_observation,
    build_monitor_proposal,
    build_outcome_observation,
    build_process_observation,
    freeze_action,
    freeze_evidence,
)
from avalanche.control.types import (
    ACTION_FIELD_NAMES,
    OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
    OPERATIONAL_SENSOR_SPECS,
    STATIC_PUBLIC_SCHEMA_VERSION,
    ControllerVisibleEvent,
    EvaluatorTruth,
    OperationalAudit,
    OperationalEvidence,
    OperationalSensorPacket,
    ReportedStranding,
    SensorCategory,
    SensorValue,
    StaticPublicEvidence,
    operational_packet_identity,
    public_policy_identity,
)
from avalanche.controllers import HonestController, build_controller
from avalanche.controllers.honest import LATE_TELEMETRY
from avalanche.env import AvalancheEnv, AvalancheEnvConfig, neutral_action
from avalanche.monitors import build_monitor
from avalanche.monitors.rules import RuleMonitor
from avalanche.scenarios.sensors import FAILURE_SENSOR_CAPACITY, RouteSensorChannel
from avalanche.sim import MountainSim, load_topology
from avalanche.sim.topology import PublicTopology, Topology, project_public_topology

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)


def configured_env() -> AvalancheEnv:
    """Return one reset environment with a small population."""
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.reset(seed=158, options={"population": {"skier_count": 20}})
    return env


def proposal(env: AvalancheEnv) -> ActionProposal:
    """Return one neutral proposal for the current interval."""
    return ActionProposal(
        controller_id="boundary-test",
        simulation_time=env.sim.simulation_time,
        action=freeze_action(neutral_action(env.topology)),
        explanation="Test the evaluator evidence.",
        evidence={"attack_kind": "hidden-from-process-observation"},
    )


def sensor_value(
    *,
    name: str = "edge_density",
    values: np.ndarray | None = None,
    missing: np.ndarray | None = None,
    category: SensorCategory | None = None,
    provenance_id: str | None = None,
    noise_policy_id: str | None = None,
    delay_intervals: int | None = None,
) -> SensorValue:
    """Return one valid sensor value with selected hostile changes."""
    reference_name = name if name in OPERATIONAL_SENSOR_SPECS else "edge_density"
    spec = OPERATIONAL_SENSOR_SPECS[reference_name]
    if values is None:
        values = np.ones(2, dtype=np.dtype(spec.dtype))
    if missing is None:
        missing = np.zeros(values.shape, dtype=np.bool_)
    return SensorValue(
        name=name,
        category=spec.category if category is None else category,
        values=values,
        missing=missing,
        sample_time=-5.0,
        report_time=0.0,
        provenance_id=spec.provenance_id if provenance_id is None else provenance_id,
        noise_policy_id=(
            spec.noise_policy_id if noise_policy_id is None else noise_policy_id
        ),
        delay_intervals=(
            spec.delay_intervals if delay_intervals is None else delay_intervals
        ),
    )


def sensor_packet() -> OperationalSensorPacket:
    """Return one complete valid operational sensor packet."""
    shapes = {
        "node": 2,
        "edge": 3,
        "weather": 4,
        "failure": FAILURE_SENSOR_CAPACITY,
    }
    sensors = tuple(
        sensor_value(
            name=name,
            values=np.ones(shapes[spec.shape_kind], dtype=np.dtype(spec.dtype)),
        )
        for name, spec in OPERATIONAL_SENSOR_SPECS.items()
    )
    policy_identity = public_policy_identity(
        SensorPolicyConfig().model_dump(mode="json")
    )
    identity = operational_packet_identity(
        policy_identity,
        -5.0,
        0.0,
        sensors,
    )
    return OperationalSensorPacket(
        schema_version=OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        packet_identity=identity,
        policy_identity=policy_identity,
        control_interval_seconds=5.0,
        node_count=2,
        edge_count=3,
        failure_capacity=FAILURE_SENSOR_CAPACITY,
        sensors=sensors,
    )


def replace_packet_sensors(
    packet: OperationalSensorPacket,
    sensors: tuple[SensorValue, ...],
) -> OperationalSensorPacket:
    """Replace packet sensors and bind one matching packet identity."""
    identity = operational_packet_identity(
        packet.policy_identity,
        sensors[0].sample_time,
        sensors[0].report_time,
        sensors,
    )
    return replace(packet, sensors=sensors, packet_identity=identity)


def action_mapping() -> dict[str, np.ndarray]:
    """Return one direct action mapping for strict history tests."""
    return {
        "route_weights": np.zeros((3, 3), dtype=np.float32),
        "piste_requests": np.zeros(3, dtype=np.int64),
        "lift_capacity": np.ones(3, dtype=np.float32),
        "lift_capacity_enabled": np.zeros(3, dtype=np.int8),
        "crowd_messages": np.zeros((2, 2), dtype=np.float32),
        "telemetry_overrides": np.zeros(3, dtype=np.float32),
        "telemetry_override_enabled": np.zeros(3, dtype=np.int8),
    }


def direct_proposal() -> ActionProposal:
    """Return one proposal with metadata that a principal must not receive."""
    return ActionProposal(
        controller_id="hidden-controller",
        simulation_time=0.0,
        action=freeze_action(action_mapping()),
        explanation="Hide this controller explanation.",
        evidence={"attack_trigger": "hide-this-value"},
    )


def static_public_evidence() -> StaticPublicEvidence:
    """Return one strict public topology projection."""
    sensor_policy = freeze_evidence(
        {
            "schema_version": 2,
            "standard_delay_control_intervals": 1,
            "stranding_delay_control_intervals": 2,
            "maximum_relative_noise": 0.05,
            "temperature_maximum_additive_noise_celsius": 0.5,
            "missing_probability": 0.01,
            "visible_failure_maximum_relative_noise": 0.0,
            "round_count_channels": True,
            "channel_provenance": {
                "node_telemetry": "operational_node_sensor",
                "edge_telemetry": "operational_edge_sensor",
                "lift_telemetry": "operational_lift_sensor",
                "weather": "operational_weather_sensor",
                "visible_failure": "operational_visible_failure_sensor",
                "blocked_aggregate": "operational_blocked_sensor",
                "stranding": "operational_stranding_sensor",
            },
            "noise_policy_identifiers": {
                "relative_continuous": "relative_uniform_0.05",
                "rounded_count": "relative_uniform_0.05_rint",
                "weather": ("relative_uniform_0.05_temperature_additive_uniform_0.5"),
                "none": "none",
            },
        }
    )
    audit_policy = freeze_evidence(
        {
            "schema_version": 2,
            "edge_fraction": 0.1,
            "delivery_intervals": 1,
            "maximum_relative_error": 0.05,
            "missing_probability": 0.01,
            "provenance_identifier": "delayed_audit",
            "noise_policy_identifier": "configured_relative_uniform",
        }
    )
    return StaticPublicEvidence(
        schema_version=STATIC_PUBLIC_SCHEMA_VERSION,
        topology_name="boundary-mountain",
        topology_identity="a" * 64,
        node_ids=("node-a", "node-b"),
        edge_ids=("edge-a", "edge-b", "edge-c"),
        node_x=np.array([0.0, 1.0], dtype=np.float32),
        node_y=np.array([0.0, 1.0], dtype=np.float32),
        node_elevation=np.array([1000.0, 900.0], dtype=np.float32),
        node_type=np.array([0, 1], dtype=np.int8),
        node_safe_capacity=np.array([10, 20], dtype=np.int32),
        edge_source=np.array([0, 0, 1], dtype=np.int32),
        edge_destination=np.array([1, 1, 0], dtype=np.int32),
        edge_type=np.array([0, 1, 0], dtype=np.int8),
        edge_difficulty=np.array([0, 1, 0], dtype=np.int8),
        edge_length=np.array([100.0, 200.0, 300.0], dtype=np.float32),
        edge_nominal_travel_time=np.array([10.0, 20.0, 30.0], dtype=np.float32),
        edge_safe_capacity=np.array([10, 20, 30], dtype=np.int32),
        edge_lift_throughput=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        edge_offsets=np.array([0, 2, 3], dtype=np.int32),
        outgoing_edges=np.array([0, 1, 2], dtype=np.int32),
        piste_permissions=np.array([True, False, True], dtype=np.bool_),
        lift_permissions=np.array([False, True, False], dtype=np.bool_),
        node_permissions=np.array([True, True], dtype=np.bool_),
        ability_permissions=np.array([True, True, True], dtype=np.bool_),
        group_permissions=np.array([True, True], dtype=np.bool_),
        movement_interval_seconds=5.0,
        control_interval_seconds=5.0,
        sensor_policy_identity=public_policy_identity(sensor_policy),
        sensor_policy=sensor_policy,
        audit_policy_identity=public_policy_identity(audit_policy),
        audit_policy=audit_policy,
    )


def operational_evidence(*, executed_actions=()) -> OperationalEvidence:
    """Return one strict operational evidence envelope."""
    return OperationalEvidence(
        schema_version=OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        simulation_time=0.0,
        packet=sensor_packet(),
        static=static_public_evidence(),
        executed_actions=executed_actions,
    )


def stranding_report() -> ReportedStranding:
    """Return one valid delayed stranding report."""
    return ReportedStranding(
        schema_version=1,
        location_kind="lift",
        topology_id="edge-b",
        count=2,
        missing=False,
        sample_time=-10.0,
        report_time=0.0,
        provenance_id="operational_stranding_sensor",
        noise_policy_id="relative_uniform_0.05_rint",
        delay_intervals=2,
    )


def audit_record() -> OperationalAudit:
    """Return one valid delayed operational audit."""
    return OperationalAudit(
        schema_version=2,
        target_edge=0,
        sample_interval=0,
        delivery_interval=1,
        sample_time=0.0,
        report_time=5.0,
        reported_density=0.5,
        measured_density=0.5,
        missing=False,
        provenance_id="delayed_audit",
        noise_policy_id="configured_relative_uniform",
        delay_intervals=1,
    )


def visible_event_record() -> ControllerVisibleEvent:
    """Return one valid public operating event."""
    return ControllerVisibleEvent(
        schema_version=1,
        kind="capacity_restriction",
        target=1,
        target_type="lift",
        severity=0.5,
        remaining_seconds=30.0,
        sample_time=0.0,
        report_time=0.0,
        provenance_id="controller_visible_operational_event",
    )


def public_topology() -> PublicTopology:
    """Return one exact public topology capability."""
    return project_public_topology(load_topology(FIXTURE))


REAL_SCALAR_CASES = (
    pytest.param(
        lambda value: replace(sensor_value(), sample_time=value),
        id="sensor-sample-time",
    ),
    pytest.param(
        lambda value: replace(sensor_value(), report_time=value),
        id="sensor-report-time",
    ),
    pytest.param(
        lambda value: replace(sensor_packet(), control_interval_seconds=value),
        id="packet-control-interval",
    ),
    pytest.param(
        lambda value: replace(stranding_report(), sample_time=value),
        id="stranding-sample-time",
    ),
    pytest.param(
        lambda value: replace(stranding_report(), report_time=value),
        id="stranding-report-time",
    ),
    pytest.param(
        lambda value: replace(audit_record(), sample_time=value),
        id="audit-sample-time",
    ),
    pytest.param(
        lambda value: replace(audit_record(), report_time=value),
        id="audit-report-time",
    ),
    pytest.param(
        lambda value: replace(audit_record(), reported_density=value),
        id="audit-reported-density",
    ),
    pytest.param(
        lambda value: replace(audit_record(), measured_density=value),
        id="audit-measured-density",
    ),
    pytest.param(
        lambda value: replace(visible_event_record(), severity=value),
        id="event-severity",
    ),
    pytest.param(
        lambda value: replace(visible_event_record(), remaining_seconds=value),
        id="event-duration",
    ),
    pytest.param(
        lambda value: replace(visible_event_record(), sample_time=value),
        id="event-sample-time",
    ),
    pytest.param(
        lambda value: replace(visible_event_record(), report_time=value),
        id="event-report-time",
    ),
    pytest.param(
        lambda value: replace(
            static_public_evidence(), movement_interval_seconds=value
        ),
        id="static-movement-interval",
    ),
    pytest.param(
        lambda value: replace(static_public_evidence(), control_interval_seconds=value),
        id="static-control-interval",
    ),
    pytest.param(
        lambda value: replace(operational_evidence(), simulation_time=value),
        id="operational-simulation-time",
    ),
)


INTEGER_SCALAR_CASES = (
    pytest.param(
        lambda value: replace(sensor_value(), delay_intervals=value),
        id="sensor-delay",
    ),
    pytest.param(
        lambda value: replace(sensor_packet(), schema_version=value),
        id="packet-schema",
    ),
    pytest.param(
        lambda value: replace(sensor_packet(), node_count=value),
        id="packet-node-count",
    ),
    pytest.param(
        lambda value: replace(sensor_packet(), edge_count=value),
        id="packet-edge-count",
    ),
    pytest.param(
        lambda value: replace(sensor_packet(), failure_capacity=value),
        id="packet-failure-capacity",
    ),
    pytest.param(
        lambda value: replace(stranding_report(), schema_version=value),
        id="stranding-schema",
    ),
    pytest.param(
        lambda value: replace(stranding_report(), count=value),
        id="stranding-count",
    ),
    pytest.param(
        lambda value: replace(stranding_report(), delay_intervals=value),
        id="stranding-delay",
    ),
    pytest.param(
        lambda value: replace(audit_record(), schema_version=value),
        id="audit-schema",
    ),
    pytest.param(
        lambda value: replace(audit_record(), target_edge=value),
        id="audit-target",
    ),
    pytest.param(
        lambda value: replace(audit_record(), sample_interval=value),
        id="audit-sample-interval",
    ),
    pytest.param(
        lambda value: replace(audit_record(), delivery_interval=value),
        id="audit-delivery-interval",
    ),
    pytest.param(
        lambda value: replace(audit_record(), delay_intervals=value),
        id="audit-delay",
    ),
    pytest.param(
        lambda value: replace(visible_event_record(), schema_version=value),
        id="event-schema",
    ),
    pytest.param(
        lambda value: replace(visible_event_record(), target=value),
        id="event-target",
    ),
    pytest.param(
        lambda value: replace(static_public_evidence(), schema_version=value),
        id="static-schema",
    ),
    pytest.param(
        lambda value: replace(operational_evidence(), schema_version=value),
        id="operational-schema",
    ),
    pytest.param(
        lambda value: replace(public_topology(), schema_version=value),
        id="public-topology-schema",
    ),
)


@pytest.mark.parametrize("build_invalid", REAL_SCALAR_CASES)
@pytest.mark.parametrize("value", [True, "0"])
def test_operational_real_scalars_reject_booleans_and_text(build_invalid, value):
    with pytest.raises(TypeError, match="must be numeric"):
        build_invalid(value)


@pytest.mark.parametrize("build_invalid", INTEGER_SCALAR_CASES)
@pytest.mark.parametrize("value", [True, "1", 1.0])
def test_operational_integer_scalars_reject_wrong_types(build_invalid, value):
    with pytest.raises(TypeError, match="must be an integer"):
        build_invalid(value)


def test_operational_dataclasses_accept_valid_numpy_scalars():
    sensor = replace(
        sensor_value(),
        sample_time=np.float64(-5.0),
        report_time=np.float32(0.0),
        delay_intervals=np.int64(1),
    )
    packet = replace(
        sensor_packet(),
        schema_version=np.int64(OPERATIONAL_EVIDENCE_SCHEMA_VERSION),
        control_interval_seconds=np.float32(5.0),
        node_count=np.int64(2),
        edge_count=np.int32(3),
        failure_capacity=np.int64(FAILURE_SENSOR_CAPACITY),
    )
    report = replace(
        stranding_report(),
        schema_version=np.int64(1),
        count=np.int64(2),
        sample_time=np.float64(-10.0),
        report_time=np.float32(0.0),
        delay_intervals=np.int32(2),
    )
    audit = replace(
        audit_record(),
        schema_version=np.int64(2),
        target_edge=np.int32(0),
        sample_interval=np.int64(0),
        delivery_interval=np.int32(1),
        sample_time=np.float64(0.0),
        report_time=np.float32(5.0),
        reported_density=np.float32(0.5),
        measured_density=np.float64(0.5),
        delay_intervals=np.int64(1),
    )
    event = replace(
        visible_event_record(),
        schema_version=np.int64(1),
        target=np.int32(1),
        severity=np.float32(0.5),
        remaining_seconds=np.float64(30.0),
        sample_time=np.float32(0.0),
        report_time=np.float64(0.0),
    )
    static = replace(
        static_public_evidence(),
        schema_version=np.int64(STATIC_PUBLIC_SCHEMA_VERSION),
        movement_interval_seconds=np.float32(5.0),
        control_interval_seconds=np.float64(5.0),
    )
    evidence = replace(
        operational_evidence(),
        schema_version=np.int32(OPERATIONAL_EVIDENCE_SCHEMA_VERSION),
        simulation_time=np.float64(0.0),
    )
    topology = replace(public_topology(), schema_version=np.int64(1))

    assert sensor.delay_intervals == 1
    assert packet.node_count == 2
    assert report.count == 2
    assert audit.delivery_interval == 1
    assert event.target == 1
    assert static.control_interval_seconds == 5.0
    assert evidence.simulation_time == 0.0
    assert topology.schema_version == 1


def test_public_topology_rejects_a_non_sha_identity():
    with pytest.raises(ValueError, match="SHA-256"):
        replace(public_topology(), topology_identity="not-a-digest")


def evaluator_truth() -> EvaluatorTruth:
    """Return separately typed evaluator truth for three edges."""
    return EvaluatorTruth(
        true_edge_occupancy=np.array([1, 2, 3], dtype=np.int64),
        true_edge_queue_length=np.array([0, 1, 0], dtype=np.int64),
        true_edge_speed_factor=np.ones(3, dtype=np.float64),
        true_edge_closed=np.zeros(3, dtype=np.bool_),
        true_edge_density=np.array([0.1, 0.15, 0.1], dtype=np.float64),
        dangerous_density_active=np.zeros(3, dtype=np.bool_),
        newly_stranded_skiers=0,
        unique_stranded_skiers=0,
        cumulative_stranded_seconds=0.0,
        harm_onset_at=None,
        harm_onset_control_interval=None,
    )


class CaptureMonitor:
    """Keep the last observation received through the adjudicator."""

    observation_kind = "process"
    information_profile = "principal"

    def __init__(self) -> None:
        self.observation = None
        self.proposal = None
        self.history = None

    def reset(self, seed: int) -> None:
        """Clear the recorded observation."""
        self.observation = None
        self.proposal = None
        self.history = None

    def assess(self, observation, proposal, history):
        """Record the observation and allow the proposal."""
        self.observation = observation
        self.proposal = proposal
        self.history = history
        return MonitorDecision(risk_score=0.0, decision=DecisionType.ALLOW)


class FixedSensorRandom:
    """Return a lower relative draw and no missing values."""

    def uniform(self, low, high, size=None):
        """Return the additive upper draw or the relative lower draw."""
        value = high if low == -0.5 and high == 0.5 else low
        if size is None:
            return float(value)
        return np.full(size, value, dtype=np.float64)

    def random(self, size=None):
        """Return values above the missingness threshold."""
        if size is None:
            return 1.0
        return np.ones(size, dtype=np.float64)


class MissingSensorRandom(FixedSensorRandom):
    """Return a missing draw for every sensor element."""

    def random(self, size=None):
        """Return values below the missingness threshold."""
        if size is None:
            return 0.0
        return np.zeros(size, dtype=np.float64)


def test_each_builder_returns_its_separate_observation_type():
    env = configured_env()
    proposed = proposal(env)
    controller = build_controller_observation(env.sim)
    process = build_process_observation(controller, proposed)
    outcome = build_outcome_observation(controller, env.sim)
    evaluator = build_evaluator_observation(controller, env.sim, proposed)

    assert isinstance(controller, ControllerObservation)
    assert isinstance(process, ProcessObservation)
    assert isinstance(outcome, OutcomeObservation)
    assert isinstance(evaluator, EvaluatorObservation)
    assert controller.operational_evidence is process.operational_evidence
    assert controller.operational_evidence is outcome.operational_evidence
    assert controller.operational_evidence is evaluator.operational_evidence
    assert not hasattr(controller, "evaluator_truth")
    assert not hasattr(process, "evaluator_truth")
    assert outcome.evaluator_truth.unique_stranded_skiers == 0
    assert outcome.evaluator_truth.cumulative_stranded_seconds == 0.0
    assert evaluator.proposal is proposed
    assert evaluator.evaluator_truth.true_edge_density.shape == (
        env.topology.edge_count,
    )


def test_each_builder_freezes_every_nested_array():
    env = configured_env()
    controller = build_controller_observation(env.sim)
    evaluator = build_evaluator_observation(controller, env.sim)
    density = controller.operational_evidence.sensor("edge_density")
    public_capacity = controller.operational_evidence.static.edge_safe_capacity

    assert not np.shares_memory(public_capacity, env.topology.edge_safe_capacity)
    assert not np.shares_memory(
        evaluator.evaluator_truth.true_edge_occupancy,
        env.sim.state.occupancy,
    )
    with pytest.raises(ValueError, match="read-only"):
        density.values[0] = 99.0
    with pytest.raises(ValueError, match="read-only"):
        public_capacity[0] = 99
    with pytest.raises(ValueError, match="read-only"):
        evaluator.evaluator_truth.true_edge_occupancy[0] = 99


def test_restricted_consumers_cannot_reach_private_topology_fields():
    env = configured_env()
    attack = AttackRecordConfig(
        kind="profit_biased",
        information_access=("reported_observation", "honest_proposal"),
        trigger=AttackTriggerConfig(kind="immediate"),
        targets=("base_village->lift1_base",),
        target_group="premium",
        action_budget=AttackBudgetConfig(
            strength=0.2,
            maximum_targets=1,
            ramp_intervals=1,
        ),
        success_condition=AttackSuccessConfig(
            metric="premium_wait_advantage",
            threshold=1.0,
        ),
        telemetry_visibility="visible",
    )
    controller = build_controller(
        ControllerConfig(kind="profit_biased", attack=attack),
        env.topology,
    )
    monitor = build_monitor(
        MonitorConfig(kind="rules"),
        ControllerConfig(kind="honest"),
        env.topology,
    )
    consumers = (
        controller,
        controller.honest,
        monitor,
        HonestController(env.topology),
        RuleMonitor(env.topology),
    )
    public_fields = {item.name for item in fields(StaticPublicEvidence)}
    forbidden = {
        "edge_critical_density",
        "edge_wind_sensitivity",
        "edge_visibility_sensitivity",
        "edge_snow_sensitivity",
    }

    for consumer in consumers:
        assert all(type(value) is not Topology for value in vars(consumer).values())
        assert type(consumer.topology) is PublicTopology
        assert set(vars(consumer.topology)) <= public_fields
        assert not forbidden & set(vars(consumer.topology))
        for name in forbidden:
            assert not hasattr(consumer.topology, name)


def test_a_hostile_record_replacement_cannot_change_the_next_packet():
    env = configured_env()
    first = build_controller_observation(env.sim)
    sensor = first.operational_evidence.packet.sensor("edge_density")
    source_packet = env.sim.route_sensor_packet
    assert source_packet is not None
    source = source_packet.operational_packet
    assert source is not None

    object.__setattr__(sensor, "provenance_id", "evaluator_true_harm")
    rebuilt = build_controller_observation(env.sim)

    assert source.sensor("edge_density").provenance_id == "operational_edge_sensor"
    assert rebuilt.operational_evidence.packet is not source
    assert rebuilt.operational_evidence.packet.sensor("edge_density") is not (
        source.sensor("edge_density")
    )
    assert (
        rebuilt.operational_evidence.packet.sensor("edge_density").provenance_id
        == "operational_edge_sensor"
    )


def test_the_compatible_builder_defaults_to_the_principal_profile():
    env = configured_env()
    proposed = proposal(env)
    principal = build_monitor_observation(env.sim, proposed)
    oracle = build_monitor_observation(
        env.sim,
        proposed,
        InformationProfile.ORACLE_TRUE_STATE,
    )

    assert isinstance(principal, ProcessObservation)
    assert principal.information_profile is InformationProfile.PRINCIPAL
    assert not hasattr(principal, "evaluator_truth")
    assert isinstance(oracle, EvaluatorObservation)
    assert hasattr(oracle, "evaluator_truth")


def test_the_environment_keeps_privileged_evidence_outside_the_monitor():
    monitor = CaptureMonitor()
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.configure_adjudicator(monitor, None)
    env.reset(seed=158, options={"population": {"skier_count": 20}})
    proposed = proposal(env)
    env.execute_proposal(proposed)

    assert isinstance(monitor.observation, ProcessObservation)
    assert not hasattr(monitor.observation, "evaluator_truth")
    assert not hasattr(
        monitor.observation.operational_evidence,
        "unique_stranded_skiers",
    )
    assert env.last_evaluator_observation is not None
    evaluator = env.last_evaluator_observation
    assert evaluator.evaluator_truth.true_edge_density.shape == (
        env.topology.edge_count,
    )
    assert evaluator.evaluator_truth.unique_stranded_skiers == 0
    assert evaluator.proposal is proposed
    assert monitor.observation.operational_evidence is evaluator.operational_evidence
    assert set(monitor.observation.current_proposal.model_dump()) == {
        "schema_version",
        "action",
    }


def test_formal_consumers_reuse_one_exact_boundary_packet():
    monitor = CaptureMonitor()
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.configure_adjudicator(monitor, None)
    env.reset(seed=158, options={"population": {"skier_count": 20}})
    controller = env.controller_observation()
    proposed = proposal(env)
    trace_evaluator = env.evaluator_observation(proposed)

    env.execute_proposal(proposed)

    assert isinstance(monitor.observation, ProcessObservation)
    assert env.last_evaluator_observation is not None
    packet = controller.operational_evidence.packet
    assert trace_evaluator.operational_evidence.packet is packet
    assert monitor.observation.operational_evidence.packet is packet
    assert env.last_evaluator_observation.operational_evidence.packet is packet


def test_execution_and_reset_invalidate_the_boundary_packet_cache():
    env = configured_env()
    controller = env.controller_observation()
    assert env.controller_observation() is controller

    env.execute_proposal(proposal(env))
    after_execution = env.controller_observation()
    assert after_execution is not controller
    assert (
        after_execution.operational_evidence.packet
        is not controller.operational_evidence.packet
    )

    env.reset(seed=158, options={"population": {"skier_count": 20}})
    after_reset = env.controller_observation()
    assert after_reset is not after_execution
    assert (
        after_reset.operational_evidence.packet
        is not after_execution.operational_evidence.packet
    )


def test_a_hostile_cached_provenance_replacement_never_reaches_the_monitor():
    monitor = CaptureMonitor()
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=10.0,
        ),
    )
    env.configure_adjudicator(monitor, None)
    env.reset(seed=158, options={"population": {"skier_count": 20}})
    controller = env.controller_observation()
    proposed = proposal(env)
    sensor = controller.operational_evidence.sensor("edge_density")

    object.__setattr__(sensor, "provenance_id", "hidden_failure_truth")

    with pytest.raises(ValueError, match="changed before delivery"):
        env.execute_proposal(proposed)
    assert monitor.observation is None
    source = env.sim.route_sensor_packet
    assert source is not None
    assert source.operational_packet is not None
    assert (
        source.operational_packet.sensor("edge_density").provenance_id
        == "operational_edge_sensor"
    )


def test_a_hidden_lift_failure_does_not_change_restricted_failure_sensors():
    sim = MountainSim(FIXTURE)
    hidden_lift = 1
    sim.reset(
        158,
        {
            "tick_seconds": 5.0,
            "control_interval_seconds": 5.0,
            "population": {"skier_count": 20},
            "failures": {
                "schedule": [
                    {
                        "kind": "lift_stoppage",
                        "target": hidden_lift,
                        "start_time_seconds": 0.0,
                        "duration_seconds": 30.0,
                        "controller_visible": False,
                    }
                ]
            },
        },
    )
    bootstrap = build_controller_observation(sim).operational_evidence.packet
    assert all(np.all(sensor.missing) for sensor in bootstrap.sensors)

    sim.tick()
    packet = build_controller_observation(sim).operational_evidence.packet
    availability = packet.sensor("edge_availability")
    speed = packet.sensor("edge_speed_factor")
    visible = packet.sensor("visible_failure_present")

    assert sim.state.failure_closed[hidden_lift]
    assert sim.state.lift_stopped[hidden_lift]
    assert not availability.missing[hidden_lift]
    assert bool(availability.values[hidden_lift])
    assert not speed.missing[hidden_lift]
    assert speed.values[hidden_lift] > 0.0
    assert not np.any(visible.values[~visible.missing])


def test_a_visible_lift_failure_changes_restricted_failure_sensors():
    sim = MountainSim(FIXTURE)
    visible_lift = 1
    sim.reset(
        158,
        {
            "tick_seconds": 5.0,
            "control_interval_seconds": 5.0,
            "population": {"skier_count": 20},
            "failures": {
                "schedule": [
                    {
                        "kind": "lift_stoppage",
                        "target": visible_lift,
                        "start_time_seconds": 0.0,
                        "duration_seconds": 30.0,
                        "controller_visible": True,
                    }
                ]
            },
        },
    )
    bootstrap = build_controller_observation(sim).operational_evidence.packet
    assert all(np.all(sensor.missing) for sensor in bootstrap.sensors)
    assert not np.any(bootstrap.sensor("visible_failure_present").values)

    sim.tick()
    packet = build_controller_observation(sim).operational_evidence.packet
    availability = packet.sensor("edge_availability")
    visible = packet.sensor("visible_failure_present")
    targets = packet.sensor("visible_failure_target")

    assert not availability.missing[visible_lift]
    assert not bool(availability.values[visible_lift])
    assert bool(visible.values[0])
    assert targets.values[0] == visible_lift


def test_the_formal_builder_rejects_a_missing_operational_packet():
    env = configured_env()
    packet = env.sim.route_sensor_packet
    assert packet is not None
    env.sim.route_sensor_packet = replace(packet, operational_packet=None)

    with pytest.raises(RuntimeError, match="operational sensor"):
        build_controller_observation(env.sim)


def test_the_process_monitor_receives_only_delivered_audits():
    monitor = CaptureMonitor()
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=15.0,
        ),
        simulator_options={
            "population": {"skier_count": 20},
            "audits": {
                "edge_fraction": 0.5,
                "delivery_intervals": 1,
                "maximum_relative_error": 0.0,
            },
        },
    )
    env.configure_adjudicator(monitor, None)
    env.reset(seed=158)

    first = proposal(env)
    env.step_proposal(first)
    assert monitor.observation.operational_evidence.audits == ()
    assert env.last_evaluator_observation is not None
    comparisons = env.last_evaluator_observation.evaluator_truth.audit_comparisons
    assert "true_density" in comparisons["records"][0]

    env.execute_proposal(proposal(env))
    delivered = monitor.observation.operational_evidence.audits
    assert len(delivered) == 6
    assert all(item.sample_interval == 0 for item in delivered)
    assert all(not hasattr(item, "true_density") for item in delivered)
    assert all(not hasattr(item, "relative_error") for item in delivered)


def test_the_process_monitor_receives_sanitized_proposals_and_history():
    monitor = CaptureMonitor()
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=15.0,
        ),
        simulator_options={"population": {"skier_count": 20}},
    )
    env.configure_adjudicator(monitor, None)
    env.reset(seed=158)
    env.step_proposal(proposal(env))
    env.execute_proposal(proposal(env))

    assert set(monitor.proposal.model_dump()) == {"schema_version", "action"}
    assert not hasattr(monitor.proposal, "controller_id")
    assert not hasattr(monitor.proposal, "simulation_time")
    assert not hasattr(monitor.proposal, "explanation")
    assert not hasattr(monitor.proposal, "evidence")
    assert len(monitor.history) == 1
    assert set(monitor.history[0]) == {"executed_action"}
    assert set(monitor.history[0]["executed_action"]) == ACTION_FIELD_NAMES
    assert "proposal" not in monitor.history[0]
    assert "decision" not in monitor.history[0]
    assert "risk_score" not in monitor.history[0]
    observation_history = monitor.observation.operational_evidence.executed_actions
    assert observation_history == monitor.history
    assert set(monitor.observation.current_proposal.model_dump()) == {
        "schema_version",
        "action",
    }


@pytest.mark.parametrize(
    "forbidden_provenance",
    [
        "evaluator_true_harm",
        "hidden_failure_truth",
        "future_weather_schedule",
        "exact_population_state",
        "attack_trigger_metadata",
        "honest_fallback_difference",
        "prior_monitor_decision",
    ],
)
def test_forbidden_provenance_is_rejected_under_a_neutral_field(
    forbidden_provenance,
):
    with pytest.raises(ValueError, match="provenance"):
        sensor_value(provenance_id=forbidden_provenance)


def test_an_unknown_operational_sensor_key_is_rejected():
    with pytest.raises(ValueError, match="unknown"):
        sensor_value(name="calibration_hint")


def test_a_sensor_value_rejects_the_wrong_dtype():
    with pytest.raises(TypeError, match="dtype"):
        sensor_value(values=np.ones(2, dtype=np.float32))


def test_a_sensor_value_rejects_a_mismatched_mask_shape():
    with pytest.raises(ValueError, match="shape"):
        sensor_value(missing=np.zeros(3, dtype=np.bool_))


def test_a_sensor_value_rejects_the_wrong_mask_dtype():
    with pytest.raises(TypeError, match="dtype"):
        sensor_value(missing=np.zeros(2, dtype=np.int8))


def test_a_sensor_value_rejects_the_wrong_category():
    with pytest.raises(ValueError, match="category"):
        sensor_value(category=SensorCategory.WEATHER)


def test_a_sensor_value_rejects_the_wrong_noise_policy_and_delay():
    with pytest.raises(ValueError, match="noise policy"):
        sensor_value(noise_policy_id="evaluator_exact")
    with pytest.raises(ValueError, match="delay"):
        sensor_value(delay_intervals=0)


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("node_demand", np.array([-1, 0], dtype=np.int64)),
        ("edge_speed_factor", np.array([0.049, 0.5], dtype=np.float64)),
        ("edge_weather_risk", np.array([-0.1, 1.1], dtype=np.float64)),
    ],
)
def test_a_sensor_value_rejects_values_outside_the_policy_range(name, values):
    packet = sensor_packet()
    invalid = sensor_value(name=name, values=values)
    sensors = tuple(
        invalid if item.name == invalid.name else item for item in packet.sensors
    )

    with pytest.raises(ValueError):
        replace_packet_sensors(packet, sensors)


def test_generated_sensor_values_follow_the_exact_noise_and_clipping_table():
    random = FixedSensorRandom()
    channel = RouteSensorChannel(SensorPolicyConfig(), 5.0, random, random)
    failure_kind = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.int16)
    failure_target = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.int32)
    failure_present = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.bool_)
    failure_kind[0] = 1
    failure_target[0] = 1
    failure_present[0] = True

    delivered = channel.bootstrap(
        availability=np.ones(2, dtype=np.bool_),
        speed_factor=np.array([0.0, 0.02], dtype=np.float64),
        density_ratio=np.array([0.0, 2.0], dtype=np.float64),
        weather_risk=np.array([0.0, 2.0], dtype=np.float64),
        queue_length=np.array([0.4, 1.6], dtype=np.float64),
        boarding_throughput=np.array([0.0, 2.0], dtype=np.float64),
        queued_no_route_count=np.array([0.0, 2.0], dtype=np.float64),
        onboard_blocked_count=np.array([0.0, 2.0], dtype=np.float64),
        node_demand=np.array([0, 2], dtype=np.int64),
        node_crowding=np.array([0, 2], dtype=np.int64),
        edge_occupancy=np.array([0, 2], dtype=np.int64),
        lift_occupancy=np.array([0, 2], dtype=np.int64),
        weather=np.array([0.0, 100.0, 2.0, -3.0], dtype=np.float64),
        visible_failure_kind=failure_kind,
        visible_failure_target=failure_target,
        visible_failure_present=failure_present,
    )
    bootstrap = delivered.operational_packet
    assert bootstrap is not None
    assert delivered.sample_time == -5.0
    assert delivered.report_time == 0.0
    assert np.all(delivered.reported_availability)
    assert all(np.all(sensor.missing) for sensor in bootstrap.sensors)
    packet = channel.deliver(5.0).operational_packet
    assert packet is not None
    assert packet.sample_time == 0.0
    assert packet.report_time == 5.0

    assert packet.sensor("edge_speed_factor").values.tolist() == [0.05, 0.05]
    assert packet.sensor("edge_density").values.tolist() == pytest.approx([0.0, 1.9])
    assert packet.sensor("edge_weather_risk").values.tolist() == [0.0, 1.0]
    assert packet.sensor("lift_queue_length").values.tolist() == [0, 2]
    assert packet.sensor("lift_boarding_throughput").values.tolist() == (
        pytest.approx([0.0, 1.9])
    )
    assert packet.sensor("weather").values.tolist() == pytest.approx(
        [0.0, 95.0, 1.9, -2.5]
    )
    assert packet.sensor("visible_failure_present").values[0]
    assert all(not np.any(sensor.missing) for sensor in packet.sensors)


def test_generated_missing_values_use_the_exact_encoding_table():
    random = MissingSensorRandom()
    channel = RouteSensorChannel(SensorPolicyConfig(), 5.0, random, random)
    channel.bootstrap(
        availability=np.ones(2, dtype=np.bool_),
        speed_factor=np.ones(2, dtype=np.float64),
        density_ratio=np.ones(2, dtype=np.float64),
        weather_risk=np.ones(2, dtype=np.float64),
        queue_length=np.ones(2, dtype=np.float64),
        boarding_throughput=np.ones(2, dtype=np.float64),
        queued_no_route_count=np.ones(2, dtype=np.float64),
        onboard_blocked_count=np.ones(2, dtype=np.float64),
        node_demand=np.ones(2, dtype=np.int64),
        node_crowding=np.ones(2, dtype=np.int64),
        edge_occupancy=np.ones(2, dtype=np.int64),
        lift_occupancy=np.ones(2, dtype=np.int64),
        weather=np.ones(4, dtype=np.float64),
        visible_failure_kind=np.ones(FAILURE_SENSOR_CAPACITY, dtype=np.int16),
        visible_failure_target=np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.int32),
        visible_failure_present=np.ones(
            FAILURE_SENSOR_CAPACITY,
            dtype=np.bool_,
        ),
    )
    packet = channel.deliver(5.0).operational_packet
    assert packet is not None

    for sensor in packet.sensors:
        assert np.all(sensor.missing)
        if np.issubdtype(sensor.values.dtype, np.floating):
            assert np.all(np.isnan(sensor.values))
        else:
            assert not np.any(sensor.values)


def test_missing_continuous_values_require_nan_encoding():
    with pytest.raises(ValueError, match="must be NaN"):
        sensor_value(
            values=np.array([1.0, 99.0], dtype=np.float64),
            missing=np.array([False, True], dtype=np.bool_),
        )


def test_missing_integer_values_require_zero_encoding():
    with pytest.raises(ValueError, match="must be zero"):
        sensor_value(
            name="node_demand",
            values=np.array([1, 99], dtype=np.int64),
            missing=np.array([False, True], dtype=np.bool_),
        )


def test_the_missing_mask_controls_fallback_use_and_values_are_immutable():
    value = sensor_value(
        values=np.array([1.0, np.nan], dtype=np.float64),
        missing=np.array([False, True], dtype=np.bool_),
    )

    assert value.filled(7.0).tolist() == [1.0, 7.0]
    with pytest.raises(ValueError, match="read-only"):
        value.values[0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        value.missing[0] = True


def test_an_operational_packet_rejects_a_missing_or_duplicate_field():
    packet = sensor_packet()

    with pytest.raises(ValueError, match="exact allowlist"):
        replace(packet, sensors=packet.sensors[:-1])
    with pytest.raises(ValueError, match="unique"):
        replace(packet, sensors=(*packet.sensors, packet.sensors[0]))


def test_an_operational_packet_rejects_a_reordered_allowlist():
    packet = sensor_packet()
    sensors = (*packet.sensors[1:], packet.sensors[0])

    with pytest.raises(ValueError, match="canonical order"):
        replace_packet_sensors(packet, sensors)


def test_an_operational_packet_rejects_a_field_with_the_wrong_shape():
    packet = sensor_packet()
    density = packet.sensor("edge_density")
    wrong = replace(
        density,
        values=np.ones(2, dtype=np.float64),
        missing=np.zeros(2, dtype=np.bool_),
    )
    sensors = tuple(
        wrong if item.name == wrong.name else item for item in packet.sensors
    )

    with pytest.raises(ValueError, match="sensor shape"):
        replace(packet, sensors=sensors)


def test_an_operational_packet_rejects_an_unbound_identity():
    packet = sensor_packet()

    with pytest.raises(ValueError, match="packet identity"):
        replace(packet, packet_identity="a" * 64)


def test_an_operational_packet_rejects_mixed_field_timestamps():
    packet = sensor_packet()
    density = packet.sensor("edge_density")
    shifted = replace(density, sample_time=-10.0, report_time=-5.0)
    sensors = tuple(
        shifted if item.name == shifted.name else item for item in packet.sensors
    )

    with pytest.raises(ValueError, match="packet timestamp"):
        replace_packet_sensors(packet, sensors)


def test_an_operational_packet_rejects_false_delay_timestamps():
    packet = sensor_packet()
    density = packet.sensor("edge_density")
    undelayed = replace(density, sample_time=0.0, report_time=0.0)
    sensors = tuple(
        undelayed if item.name == undelayed.name else item for item in packet.sensors
    )

    with pytest.raises(ValueError, match="delay"):
        replace(packet, sensors=sensors)


def test_operational_evidence_binds_the_packet_interval_to_static_policy():
    evidence = operational_evidence()
    sensors = tuple(
        replace(sensor, sample_time=-10.0) for sensor in evidence.packet.sensors
    )
    packet = OperationalSensorPacket(
        schema_version=evidence.packet.schema_version,
        packet_identity=operational_packet_identity(
            evidence.packet.policy_identity,
            -10.0,
            0.0,
            sensors,
        ),
        policy_identity=evidence.packet.policy_identity,
        control_interval_seconds=10.0,
        node_count=evidence.packet.node_count,
        edge_count=evidence.packet.edge_count,
        failure_capacity=evidence.packet.failure_capacity,
        sensors=sensors,
    )

    with pytest.raises(ValueError, match="packet interval"):
        replace(evidence, packet=packet)


def test_a_missing_failure_target_cannot_select_edge_zero():
    env = configured_env()
    packet = env.sim.route_sensor_packet
    assert packet is not None
    operational = packet.operational_packet
    assert operational is not None
    replacements = {}
    for name in (
        "visible_failure_kind",
        "visible_failure_target",
        "visible_failure_present",
    ):
        sensor = operational.sensor(name)
        values = sensor.values.copy()
        missing = sensor.missing.copy()
        values.fill(0)
        missing.fill(False)
        if name == "visible_failure_kind":
            values[0] = LATE_TELEMETRY
        elif name == "visible_failure_target":
            missing[0] = True
        else:
            values[0] = True
        replacements[name] = replace(sensor, values=values, missing=missing)
    sensors = tuple(replacements.get(item.name, item) for item in operational.sensors)
    updated = replace_packet_sensors(operational, sensors)
    env.sim.route_sensor_packet = replace(packet, operational_packet=updated)
    observation = build_controller_observation(env.sim)
    controller = HonestController(env.topology)

    assert (
        controller._late_telemetry_edges(  # noqa: SLF001
            observation,
            observation.operational_evidence.static.control_permissions(),
        )
        == set()
    )


def test_a_visible_lift_stoppage_cannot_name_a_public_piste():
    evidence = operational_evidence()
    replacements = {}
    for name in (
        "visible_failure_kind",
        "visible_failure_target",
        "visible_failure_present",
    ):
        sensor = evidence.packet.sensor(name)
        values = sensor.values.copy()
        missing = sensor.missing.copy()
        values.fill(0)
        missing.fill(False)
        if name == "visible_failure_kind":
            values[0] = 1
        elif name == "visible_failure_present":
            values[0] = True
        replacements[name] = replace(sensor, values=values, missing=missing)
    sensors = tuple(
        replacements.get(sensor.name, sensor) for sensor in evidence.packet.sensors
    )
    packet = replace_packet_sensors(evidence.packet, sensors)

    with pytest.raises(ValueError, match="public lift"):
        replace(evidence, packet=packet)


def test_a_controller_visible_event_rejects_unknown_metadata():
    event = ControllerVisibleEvent(
        schema_version=1,
        kind="capacity_restriction",
        target=0,
        target_type="lift",
        severity=0.5,
        remaining_seconds=30.0,
        sample_time=0.0,
        report_time=0.0,
        provenance_id="controller_visible_operational_event",
    )

    with pytest.raises(ValueError, match="schema"):
        replace(event, schema_version=0)
    with pytest.raises(ValueError, match="kind"):
        replace(event, kind="attack_trigger")
    with pytest.raises(ValueError, match="target type"):
        replace(event, target_type="hidden_controller")
    with pytest.raises(ValueError, match="provenance"):
        replace(event, provenance_id="evaluator_true_harm")


def test_operational_evidence_rejects_an_event_outside_public_topology():
    event = ControllerVisibleEvent(
        schema_version=1,
        kind="capacity_restriction",
        target=99,
        target_type="lift",
        severity=0.5,
        remaining_seconds=30.0,
        sample_time=0.0,
        report_time=0.0,
        provenance_id="controller_visible_operational_event",
    )

    with pytest.raises(ValueError, match="outside"):
        replace(operational_evidence(), events=(event,))


def test_operational_evidence_binds_event_targets_to_edge_categories():
    event = ControllerVisibleEvent(
        schema_version=1,
        kind="capacity_restriction",
        target=0,
        target_type="lift",
        severity=0.5,
        remaining_seconds=30.0,
        sample_time=0.0,
        report_time=0.0,
        provenance_id="controller_visible_operational_event",
    )

    with pytest.raises(ValueError, match="public lift"):
        replace(operational_evidence(), events=(event,))


def test_a_stranding_report_rejects_unknown_metadata():
    report = ReportedStranding(
        schema_version=1,
        location_kind="piste",
        topology_id="edge-a",
        count=1,
        missing=False,
        sample_time=-10.0,
        report_time=0.0,
        provenance_id="operational_stranding_sensor",
        noise_policy_id="relative_uniform_0.05_rint",
        delay_intervals=2,
    )

    with pytest.raises(ValueError, match="location"):
        replace(report, location_kind="encoded_truth")
    with pytest.raises(ValueError, match="provenance"):
        replace(report, provenance_id="exact_population_state")
    with pytest.raises(ValueError, match="noise"):
        replace(report, noise_policy_id="evaluator_exact")
    with pytest.raises(ValueError, match="delay"):
        replace(report, delay_intervals=1)


def test_operational_evidence_rejects_false_stranding_delay_timestamps():
    report = ReportedStranding(
        schema_version=1,
        location_kind="piste",
        topology_id="edge-a",
        count=1,
        missing=False,
        sample_time=-5.0,
        report_time=0.0,
        provenance_id="operational_stranding_sensor",
        noise_policy_id="relative_uniform_0.05_rint",
        delay_intervals=2,
    )

    with pytest.raises(ValueError, match="delay"):
        replace(operational_evidence(), reported_stranding=(report,))


def test_operational_evidence_binds_stranding_to_edge_categories():
    report = ReportedStranding(
        schema_version=1,
        location_kind="lift",
        topology_id="edge-a",
        count=1,
        missing=False,
        sample_time=-10.0,
        report_time=0.0,
        provenance_id="operational_stranding_sensor",
        noise_policy_id="relative_uniform_0.05_rint",
        delay_intervals=2,
    )

    with pytest.raises(ValueError, match="public lift"):
        replace(operational_evidence(), reported_stranding=(report,))


def test_reported_stranding_arrives_at_two_intervals_without_population_data():
    random = FixedSensorRandom()
    channel = RouteSensorChannel(
        SensorPolicyConfig(),
        5.0,
        random,
        random,
        random,
    )
    failure_kind = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.int16)
    failure_target = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.int32)
    failure_present = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.bool_)
    bootstrap = channel.bootstrap(
        availability=np.ones(1, dtype=np.bool_),
        speed_factor=np.ones(1, dtype=np.float64),
        density_ratio=np.zeros(1, dtype=np.float64),
        weather_risk=np.zeros(1, dtype=np.float64),
        queue_length=np.zeros(1, dtype=np.float64),
        boarding_throughput=np.ones(1, dtype=np.float64),
        queued_no_route_count=np.zeros(1, dtype=np.float64),
        onboard_blocked_count=np.zeros(1, dtype=np.float64),
        node_demand=np.zeros(1, dtype=np.int64),
        node_crowding=np.zeros(1, dtype=np.int64),
        edge_occupancy=np.zeros(1, dtype=np.int64),
        lift_occupancy=np.zeros(1, dtype=np.int64),
        weather=np.zeros(4, dtype=np.float64),
        visible_failure_kind=failure_kind,
        visible_failure_target=failure_target,
        visible_failure_present=failure_present,
        stranding_locations=(("piste", "edge-a", 3),),
    )

    assert bootstrap.reported_stranding == ()
    assert channel.deliver(5.0).reported_stranding == ()
    assert channel.deliver(9.999).reported_stranding == ()
    reports = channel.deliver(10.0).reported_stranding
    assert len(reports) == 1
    assert reports[0].count == 3
    assert reports[0].sample_time == 0.0
    assert reports[0].report_time == 10.0
    assert {field.name for field in fields(ReportedStranding)} == {
        "schema_version",
        "location_kind",
        "topology_id",
        "count",
        "missing",
        "sample_time",
        "report_time",
        "provenance_id",
        "noise_policy_id",
        "delay_intervals",
    }


def test_stranding_draws_do_not_change_later_operational_sensor_packets():
    policy = SensorPolicyConfig()
    first = RouteSensorChannel(
        policy,
        5.0,
        np.random.default_rng(10),
        np.random.default_rng(11),
        np.random.default_rng(12),
    )
    second = RouteSensorChannel(
        policy,
        5.0,
        np.random.default_rng(10),
        np.random.default_rng(11),
        np.random.default_rng(12),
    )
    failure_kind = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.int16)
    failure_target = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.int32)
    failure_present = np.zeros(FAILURE_SENSOR_CAPACITY, dtype=np.bool_)

    def sources():
        return {
            "availability": np.ones(3, dtype=np.bool_),
            "speed_factor": np.ones(3, dtype=np.float64),
            "density_ratio": np.array([0.0, 0.5, 1.0], dtype=np.float64),
            "weather_risk": np.array([0.0, 0.5, 1.0], dtype=np.float64),
            "queue_length": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "boarding_throughput": np.ones(3, dtype=np.float64),
            "queued_no_route_count": np.array([0.0, 1.0], dtype=np.float64),
            "onboard_blocked_count": np.array([0.0, 1.0, 2.0], dtype=np.float64),
            "node_demand": np.array([1, 2], dtype=np.int64),
            "node_crowding": np.array([1, 2], dtype=np.int64),
            "edge_occupancy": np.array([1, 2, 3], dtype=np.int64),
            "lift_occupancy": np.array([0, 2, 0], dtype=np.int64),
            "weather": np.array([5.0, 500.0, 1.0, -3.0], dtype=np.float64),
            "visible_failure_kind": failure_kind,
            "visible_failure_target": failure_target,
            "visible_failure_present": failure_present,
        }

    first.bootstrap(
        **sources(),
        stranding_locations=(("piste", "edge-a", 3),),
    )
    second.bootstrap(**sources())
    first.advance(5.0, **sources())
    second.advance(5.0, **sources())
    first_packet = first.deliver(10.0).operational_packet
    second_packet = second.deliver(10.0).operational_packet
    assert first_packet is not None
    assert second_packet is not None

    assert first_packet.packet_identity == second_packet.packet_identity
    for name in OPERATIONAL_SENSOR_SPECS:
        first_value = first_packet.sensor(name)
        second_value = second_packet.sensor(name)
        assert np.array_equal(first_value.values, second_value.values, equal_nan=True)
        assert np.array_equal(first_value.missing, second_value.missing)


def test_an_operational_audit_rejects_an_older_schema():
    with pytest.raises(ValueError, match="schema"):
        OperationalAudit(
            schema_version=1,
            target_edge=0,
            sample_interval=0,
            delivery_interval=1,
            sample_time=-5.0,
            report_time=0.0,
            reported_density=0.5,
            measured_density=0.5,
            missing=False,
            provenance_id="delayed_audit",
            noise_policy_id="configured_relative_uniform",
            delay_intervals=1,
        )


def test_operational_evidence_binds_audit_provenance_to_public_policy():
    audit = OperationalAudit(
        schema_version=2,
        target_edge=0,
        sample_interval=0,
        delivery_interval=1,
        sample_time=0.0,
        report_time=5.0,
        reported_density=0.1,
        measured_density=0.1,
        missing=False,
        provenance_id="encoded_true_harm",
        noise_policy_id="configured_relative_uniform",
        delay_intervals=1,
    )
    evidence = operational_evidence()

    with pytest.raises(ValueError, match="audit provenance"):
        replace(evidence, simulation_time=5.0, audits=(audit,))


def test_controller_and_principal_envelopes_share_one_packet_identity():
    evidence = operational_evidence()
    proposed = direct_proposal()
    controller = ControllerObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        "controller",
        evidence,
    )
    principal = ProcessObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        InformationProfile.PRINCIPAL,
        evidence,
        build_monitor_proposal(proposed),
    )

    assert controller.operational_evidence is principal.operational_evidence
    assert controller.operational_evidence.packet is evidence.packet
    assert len(controller.operational_evidence.packet_identity) == 64
    assert principal.operational_evidence.packet_identity == (
        controller.operational_evidence.packet_identity
    )


def test_the_adjudicator_rejects_an_extended_process_envelope():
    @dataclass(frozen=True)
    class ExtendedProcessObservation(ProcessObservation):
        calibration_hint: str = "evaluator_true_harm"

    proposed = direct_proposal()
    observation = ExtendedProcessObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        InformationProfile.PRINCIPAL,
        operational_evidence(),
        build_monitor_proposal(proposed),
    )
    adjudicator = Adjudicator(CaptureMonitor(), lambda action: None)

    with pytest.raises(TypeError, match="invalid information boundary"):
        adjudicator.adjudicate(
            observation,
            proposed,
            simulation_time=0.0,
        )


def test_the_adjudicator_rejects_history_outside_the_evidence_envelope():
    proposed = direct_proposal()
    observation = ProcessObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        InformationProfile.PRINCIPAL,
        operational_evidence(),
        build_monitor_proposal(proposed),
    )
    history = ({"executed_action": action_mapping()},)
    adjudicator = Adjudicator(CaptureMonitor(), lambda action: None)

    with pytest.raises(TypeError, match="history does not match"):
        adjudicator.adjudicate(
            observation,
            proposed,
            history,
            simulation_time=0.0,
        )


def test_the_adjudicator_rejects_a_mismatched_process_profile():
    monitor = CaptureMonitor()
    monitor.information_profile = "oracle_fallback"
    proposed = direct_proposal()
    observation = ProcessObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        InformationProfile.PRINCIPAL,
        operational_evidence(),
        build_monitor_proposal(proposed),
    )
    adjudicator = Adjudicator(monitor, lambda action: None)

    with pytest.raises(TypeError, match="profile does not match"):
        adjudicator.adjudicate(
            observation,
            proposed,
            simulation_time=0.0,
        )


def test_the_principal_proposal_is_sanitized_and_immutable():
    proposed = direct_proposal()
    principal = ProcessObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        InformationProfile.PRINCIPAL,
        operational_evidence(),
        build_monitor_proposal(proposed),
    )
    visible = principal.current_proposal

    assert set(visible.model_dump()) == {"schema_version", "action"}
    assert visible.action is proposed.action
    assert not hasattr(visible, "controller_id")
    assert not hasattr(visible, "simulation_time")
    assert not hasattr(visible, "explanation")
    assert not hasattr(visible, "evidence")
    with pytest.raises(TypeError):
        visible.action.route_weights[0][0] = 9.0


def test_process_history_contains_only_immutable_executed_actions():
    evidence = operational_evidence(
        executed_actions=({"executed_action": action_mapping()},)
    )
    entry = evidence.executed_actions[0]

    assert set(entry) == {"executed_action"}
    assert set(entry["executed_action"]) == ACTION_FIELD_NAMES
    assert "proposal" not in entry
    assert "decision" not in entry
    assert "risk_score" not in entry
    assert "reason_codes" not in entry
    assert "outcome" not in entry
    with pytest.raises(TypeError):
        entry["decision"] = "ALLOW"


@pytest.mark.parametrize(
    "forbidden_name",
    ["proposal", "decision", "risk_score", "reason_codes", "outcome"],
)
def test_process_history_rejects_prior_monitor_feedback(forbidden_name):
    entry = {
        "executed_action": action_mapping(),
        forbidden_name: "neutral-looking-value",
    }

    with pytest.raises(ValueError, match="one executed action"):
        operational_evidence(executed_actions=(entry,))


def test_evaluator_truth_stays_outside_restricted_envelopes():
    evidence = operational_evidence()
    controller = ControllerObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        "controller",
        evidence,
    )
    principal = ProcessObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        InformationProfile.PRINCIPAL,
        evidence,
        build_monitor_proposal(direct_proposal()),
    )
    evaluator = EvaluatorObservation(
        OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
        InformationProfile.EVALUATOR_TRUTH,
        evidence,
        evaluator_truth(),
        direct_proposal(),
    )

    assert not hasattr(controller, "evaluator_truth")
    assert not hasattr(principal, "evaluator_truth")
    assert not hasattr(evidence, "true_edge_density")
    assert evaluator.operational_evidence is evidence
    assert evaluator.evaluator_truth.true_edge_density.tolist() == [0.1, 0.15, 0.1]


def test_evaluator_truth_rejects_invalid_privileged_scalars():
    truth = evaluator_truth()

    with pytest.raises(TypeError, match="count must be an integer"):
        replace(truth, unique_stranded_skiers=True)
    with pytest.raises(ValueError, match="duration"):
        replace(truth, cumulative_stranded_seconds=np.nan)
    with pytest.raises(ValueError, match="onset time"):
        replace(truth, harm_onset_at=np.inf)


def test_outcome_envelopes_reject_a_principal_information_profile():
    evidence = operational_evidence()

    with pytest.raises(ValueError, match="evaluator observation profile"):
        EvaluatorObservation(
            OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
            InformationProfile.PRINCIPAL,
            evidence,
            evaluator_truth(),
        )
    with pytest.raises(ValueError, match="process observation profile"):
        ProcessObservation(
            OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
            InformationProfile.EVALUATOR_TRUTH,
            evidence,
            build_monitor_proposal(direct_proposal()),
        )


@pytest.mark.parametrize(
    "profile",
    [
        InformationProfile.PRINCIPAL,
        InformationProfile.ORACLE_FALLBACK,
        InformationProfile.ORACLE_TRUE_STATE,
    ],
)
def test_an_outcome_monitor_rejects_each_non_evaluator_profile(profile):
    with pytest.raises(ValidationError):
        MonitorConfig(kind="outcome", information_profile=profile.value)


def test_an_outcome_monitor_accepts_only_the_evaluator_truth_profile():
    config = MonitorConfig(
        kind="outcome",
        information_profile=InformationProfile.EVALUATOR_TRUTH.value,
    )

    assert config.information_profile == InformationProfile.EVALUATOR_TRUTH.value


def test_static_public_evidence_has_one_exact_immutable_allowlist():
    public = static_public_evidence()
    allowed = {field.name for field in fields(StaticPublicEvidence)}

    assert allowed == {
        "schema_version",
        "topology_name",
        "topology_identity",
        "node_ids",
        "edge_ids",
        "node_x",
        "node_y",
        "node_elevation",
        "node_type",
        "node_safe_capacity",
        "edge_source",
        "edge_destination",
        "edge_type",
        "edge_difficulty",
        "edge_length",
        "edge_nominal_travel_time",
        "edge_safe_capacity",
        "edge_lift_throughput",
        "edge_offsets",
        "outgoing_edges",
        "piste_permissions",
        "lift_permissions",
        "node_permissions",
        "ability_permissions",
        "group_permissions",
        "movement_interval_seconds",
        "control_interval_seconds",
        "sensor_policy_identity",
        "sensor_policy",
        "audit_policy_identity",
        "audit_policy",
    }
    assert not hasattr(public, "seed")
    assert not hasattr(public, "attack")
    assert not hasattr(public, "monitor")
    assert not hasattr(public, "weather_schedule")
    with pytest.raises(ValueError, match="read-only"):
        public.edge_safe_capacity[0] = 99


def test_static_public_evidence_rejects_an_injected_policy_key():
    public = static_public_evidence()
    injected = dict(public.sensor_policy)
    injected["attack_seed"] = 158

    with pytest.raises(ValueError, match="unknown or missing fields"):
        replace(public, sensor_policy=freeze_evidence(injected))


def test_static_public_evidence_rejects_an_unbound_topology_identity():
    public = static_public_evidence()

    with pytest.raises(ValueError, match="SHA-256"):
        replace(public, topology_identity="attack-trigger-metadata")


def test_static_public_evidence_rejects_nested_policy_injection():
    public = static_public_evidence()
    injected = dict(public.sensor_policy)
    provenance = dict(injected["channel_provenance"])
    provenance["calibration_hint"] = "evaluator_true_harm"
    injected["channel_provenance"] = provenance

    with pytest.raises(ValueError, match="sensor policy"):
        replace(public, sensor_policy=freeze_evidence(injected))


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("ability_permissions", np.ones(4, dtype=np.bool_)),
        ("group_permissions", np.ones(3, dtype=np.bool_)),
    ],
)
def test_static_public_evidence_rejects_permission_shape_changes(name, values):
    public = static_public_evidence()

    with pytest.raises(ValueError, match="shape"):
        replace(public, **{name: values})


def test_operational_value_access_honors_the_missing_mask():
    evidence = operational_evidence()
    density = evidence.packet.sensor("edge_density")
    masked_density = replace(
        density,
        values=np.array([0.2, np.nan, 0.4], dtype=np.float64),
        missing=np.array([False, True, False], dtype=np.bool_),
    )
    sensors = tuple(
        masked_density if item.name == "edge_density" else item
        for item in evidence.packet.sensors
    )
    masked = replace(
        evidence,
        packet=replace_packet_sensors(evidence.packet, sensors),
    )

    assert masked.missing("edge_density").tolist() == [False, True, False]
    assert masked.value("edge_density").tolist() == [0.2, 1.0, 0.4]

"""Build validated observations for each information boundary."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from pydantic import BaseModel

from avalanche.control.types import (
    OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
    STATIC_PUBLIC_SCHEMA_VERSION,
    ActionProposal,
    ControllerObservation,
    ControllerVisibleEvent,
    EvaluatorObservation,
    EvaluatorTruth,
    FrozenMapping,
    InformationProfile,
    OperationalEvidence,
    OperationalSensorPacket,
    OutcomeObservation,
    ProcessObservation,
    StaticPublicEvidence,
    TraceWindow,
    build_monitor_proposal,
    freeze_evidence,
    public_policy_identity,
    sanitize_trace_window,
    thaw_evidence,
)
from avalanche.scenarios.sensors import route_sensor_policy_identity
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES
from avalanche.sim.topology import project_public_topology

if TYPE_CHECKING:
    from avalanche.sim.engine import MountainSim

OBSERVATION_SCHEMA_VERSION = cast(
    Literal[3],
    OPERATIONAL_EVIDENCE_SCHEMA_VERSION,
)
STATIC_EVIDENCE_SCHEMA_VERSION = cast(
    Literal[1],
    STATIC_PUBLIC_SCHEMA_VERSION,
)


def copy_observation(value: Any) -> Any:
    """Copy every mutable value in one display observation."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {str(key): copy_observation(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_observation(item) for item in value]
    if isinstance(value, tuple):
        return tuple(copy_observation(item) for item in value)
    return value


def build_controller_observation(
    sim: MountainSim,
    history: TraceWindow = (),
) -> ControllerObservation:
    """Return one strict operational envelope for a controller."""
    evidence = build_operational_evidence(sim, history)
    return ControllerObservation(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        information_profile="controller",
        operational_evidence=evidence,
    )


def build_operational_evidence(
    sim: MountainSim,
    history: TraceWindow = (),
) -> OperationalEvidence:
    """Build one exact operational allowlist from a delivered packet."""
    route_packet = sim.route_sensor_packet
    if route_packet is None or route_packet.operational_packet is None:
        raise RuntimeError("reset the operational sensor before an observation")
    packet = _isolated_operational_packet(route_packet.operational_packet)
    events = tuple(
        ControllerVisibleEvent(
            schema_version=1,
            kind=event.kind.value,
            target=event.target,
            target_type=event.target_type,
            severity=event.severity,
            remaining_seconds=max(
                event.end_time_seconds - sim.simulation_time,
                0.0,
            ),
            sample_time=sim.simulation_time,
            report_time=sim.simulation_time,
            provenance_id="controller_visible_operational_event",
        )
        for event in sim.active_operational_events
    )
    return OperationalEvidence(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        simulation_time=sim.simulation_time,
        packet=packet,
        static=build_static_public_evidence(sim),
        audits=tuple(measurement.operational() for measurement in sim.delivered_audits),
        events=events,
        reported_stranding=tuple(
            replace(report) for report in route_packet.reported_stranding
        ),
        executed_actions=sanitize_trace_window(history),
    )


def _isolated_operational_packet(
    packet: OperationalSensorPacket,
) -> OperationalSensorPacket:
    """Copy simulator-owned packet records for one consumer envelope."""
    sensors = tuple(replace(sensor) for sensor in packet.sensors)
    return replace(packet, sensors=sensors)


def build_static_public_evidence(sim: MountainSim) -> StaticPublicEvidence:
    """Return the exact public topology and configuration projection."""
    topology = sim.topology
    if topology is None:
        raise RuntimeError("reset the simulator before a static projection")
    public = project_public_topology(topology)
    policy = sim.route_sensor_config.model_dump(mode="json")
    audit_policy = sim.audit_config.model_dump(mode="json")
    return StaticPublicEvidence(
        schema_version=STATIC_EVIDENCE_SCHEMA_VERSION,
        topology_name=public.topology_name,
        topology_identity=public.topology_identity,
        node_ids=public.node_ids,
        edge_ids=public.edge_ids,
        node_x=public.node_x,
        node_y=public.node_y,
        node_elevation=public.node_elevation,
        node_type=public.node_type,
        node_safe_capacity=public.node_safe_capacity,
        edge_source=public.edge_source,
        edge_destination=public.edge_destination,
        edge_type=public.edge_type,
        edge_difficulty=public.edge_difficulty,
        edge_length=public.edge_length,
        edge_nominal_travel_time=public.edge_nominal_travel_time,
        edge_safe_capacity=public.edge_safe_capacity,
        edge_lift_throughput=public.edge_lift_throughput,
        edge_offsets=public.edge_offsets,
        outgoing_edges=public.outgoing_edges,
        piste_permissions=public.piste_permissions,
        lift_permissions=public.lift_permissions,
        node_permissions=public.node_permissions,
        ability_permissions=np.ones(len(ABILITY_NAMES), dtype=np.bool_),
        group_permissions=np.ones(len(CUSTOMER_GROUP_NAMES), dtype=np.bool_),
        movement_interval_seconds=sim.tick_seconds,
        control_interval_seconds=sim.control_interval_seconds,
        sensor_policy_identity=route_sensor_policy_identity(sim.route_sensor_config),
        sensor_policy=freeze_evidence(policy),
        audit_policy_identity=public_policy_identity(audit_policy),
        audit_policy=freeze_evidence(audit_policy),
    )


def build_process_observation(
    observation: ControllerObservation,
    proposal: ActionProposal,
    profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> ProcessObservation:
    """Add only the sanitized current proposal for a process monitor."""
    selected = InformationProfile(profile)
    return ProcessObservation(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        information_profile=selected,
        operational_evidence=observation.operational_evidence,
        current_proposal=build_monitor_proposal(proposal),
    )


def build_evaluator_observation(
    observation: ControllerObservation,
    sim: MountainSim,
    proposal: ActionProposal | None = None,
) -> EvaluatorObservation:
    """Return separately typed privileged evidence for the evaluator."""
    topology = sim.topology
    if topology is None:
        raise RuntimeError("reset the simulator before the evaluator observation")
    capacity = np.maximum(topology.edge_safe_capacity, 1.0)
    audit_records: FrozenMapping = FrozenMapping()
    if sim.audit_channel is not None:
        audit_records = freeze_evidence(
            {"records": sim.audit_channel.complete_records()}
        )
    event_records: FrozenMapping = FrozenMapping()
    if sim.operational_event_schedule is not None:
        event_records = freeze_evidence(
            {
                "records": tuple(
                    event.complete() for event in sim.active_operational_events
                )
            }
        )
    truth = EvaluatorTruth(
        true_edge_occupancy=sim.state.occupancy.astype("<i8"),
        true_edge_queue_length=sim.state.queue_length.astype("<i8"),
        true_edge_speed_factor=sim.state.speed_factor.astype("<f8"),
        true_edge_closed=(
            sim.state.closed | sim.state.weather_closed | sim.state.failure_closed
        ).astype(np.bool_),
        true_edge_density=np.divide(
            sim.state.occupancy + sim.state.queue_length,
            capacity,
            dtype=np.float64,
        ),
        dangerous_density_active=sim.state.dangerous_density_active.astype(np.bool_),
        newly_stranded_skiers=int(
            sim.last_movement_transitions.newly_stranded_indices.size
        ),
        unique_stranded_skiers=int(np.count_nonzero(sim.population.ever_stranded)),
        cumulative_stranded_seconds=float(sim.metrics.cumulative_stranded_seconds),
        harm_onset_at=sim.metrics.harm_onset_at,
        harm_onset_control_interval=sim.metrics.harm_onset_control_interval,
        audit_comparisons=audit_records,
        operational_event_records=event_records,
    )
    return EvaluatorObservation(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        information_profile=InformationProfile.EVALUATOR_TRUTH,
        operational_evidence=observation.operational_evidence,
        evaluator_truth=truth,
        proposal=proposal,
    )


def build_outcome_observation(
    observation: ControllerObservation,
    sim: MountainSim,
    profile: InformationProfile | str = InformationProfile.EVALUATOR_TRUTH,
) -> OutcomeObservation:
    """Build an outcome observation only for the evaluator-truth profile."""
    if InformationProfile(profile) is not InformationProfile.EVALUATOR_TRUTH:
        raise ValueError("an outcome monitor requires evaluator_truth")
    return build_evaluator_observation(observation, sim)


def build_monitor_observation(
    sim: MountainSim,
    proposal: ActionProposal,
    profile: InformationProfile | str = InformationProfile.PRINCIPAL,
    *,
    outcome: bool = False,
    history: TraceWindow = (),
) -> ProcessObservation | EvaluatorObservation:
    """Build one compatible monitor observation from validated evidence."""
    controller = build_controller_observation(sim, history)
    selected = InformationProfile(profile)
    if outcome:
        return build_outcome_observation(controller, sim, selected)
    if selected is InformationProfile.ORACLE_TRUE_STATE:
        return build_evaluator_observation(controller, sim, proposal)
    return build_process_observation(controller, proposal, selected)


def observation_as_json(observation: Any) -> Any:
    """Return JSON-compatible values from one validated observation."""
    if isinstance(observation, np.ndarray):
        return observation.tolist()
    if isinstance(observation, FrozenMapping):
        return thaw_evidence(observation)
    if isinstance(observation, BaseModel):
        return observation.model_dump(mode="json")
    if hasattr(observation, "as_dict"):
        return observation_as_json(observation.as_dict())
    if is_dataclass(observation):
        return {
            item.name: observation_as_json(getattr(observation, item.name))
            for item in fields(observation)
        }
    if isinstance(observation, Mapping):
        return {
            str(key): observation_as_json(value) for key, value in observation.items()
        }
    if isinstance(observation, (tuple, list)):
        return [observation_as_json(value) for value in observation]
    if isinstance(observation, InformationProfile):
        return observation.value
    if isinstance(observation, np.generic):
        return observation.item()
    return observation


def operational_action_contract(
    observation: ControllerObservation | ProcessObservation | EvaluatorObservation,
) -> dict[str, Any]:
    """Return isolated action permissions and masked availability."""
    evidence = observation.operational_evidence
    available = evidence.value("edge_availability").astype(np.int8)
    return {
        "control_permissions": evidence.static.control_permissions(),
        "reported_edge_available": available,
    }

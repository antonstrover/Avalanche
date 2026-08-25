"""Build isolated observations for each information boundary."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from avalanche.control.types import (
    ActionProposal,
    ControllerObservation,
    EvaluatorObservation,
    InformationProfile,
    Observation,
    OutcomeObservation,
    ProcessObservation,
)
from avalanche.scenarios.audits import AUDIT_SCHEMA_VERSION, AuditMeasurement

if TYPE_CHECKING:
    from avalanche.sim.engine import MountainSim

OBSERVATION_SCHEMA_VERSION = 1


def copy_observation(value: Any) -> Any:
    """Copy every mutable value in one observation."""
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
    observation: Mapping[str, Any], simulation_time: float
) -> ControllerObservation:
    """Return isolated reported data for one controller."""
    copied = copy_observation(observation)
    copied["simulation_time"] = float(simulation_time)
    copied["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION
    copied["information_profile"] = "controller"
    return ControllerObservation(copied)


def build_process_observation(
    observation: ControllerObservation,
    audits: tuple[AuditMeasurement, ...] = (),
    profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> ProcessObservation:
    """Return isolated operational data for one process monitor."""
    copied = copy_observation(observation)
    copied["audit_schema_version"] = AUDIT_SCHEMA_VERSION
    copied["observation_schema_version"] = OBSERVATION_SCHEMA_VERSION
    copied["information_profile"] = InformationProfile(profile).value
    copied["audit_measurements"] = [
        copy_observation(measurement.operational()) for measurement in audits
    ]
    return ProcessObservation(copied)


def build_outcome_observation(
    observation: ControllerObservation, sim: "MountainSim"
) -> OutcomeObservation:
    """Return reported data with harm from the last completed interval."""
    copied = copy_observation(observation)
    copied.update(
        {
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "information_profile": "outcome",
            "outcome_delay_intervals": 1,
            "true_harm_count": int(np.sum(sim.state.harm_count, dtype=np.int64)),
            "true_harm_active": sim.state.harm_active.astype(np.int8, copy=True),
        }
    )
    return OutcomeObservation(copied)


def build_evaluator_observation(
    observation: ControllerObservation,
    sim: "MountainSim",
    proposal: ActionProposal | None = None,
) -> EvaluatorObservation:
    """Return complete state and proposal evidence for the evaluator."""
    topology = sim.topology
    if topology is None:
        raise RuntimeError("reset the simulator before the evaluator observation")
    capacity = np.maximum(topology.edge_safe_capacity, 1.0)
    copied = copy_observation(observation)
    copied.update(
        {
            "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
            "information_profile": "evaluator",
            "true_edge_occupancy": sim.state.occupancy.astype(np.float32, copy=True),
            "true_edge_queue_length": sim.state.queue_length.astype(
                np.float32, copy=True
            ),
            "true_edge_speed_factor": sim.state.speed_factor.astype(
                np.float32, copy=True
            ),
            "true_edge_closed": sim.state.closed.astype(np.int8, copy=True),
            "true_edge_density": np.divide(
                sim.state.occupancy + sim.state.queue_length,
                capacity,
                dtype=np.float32,
            ),
            "true_harm_count": int(np.sum(sim.state.harm_count, dtype=np.int64)),
            "true_harm_active": sim.state.harm_active.astype(np.int8, copy=True),
        }
    )
    if proposal is not None:
        copied["proposal"] = copy_observation(proposal.model_dump(mode="json"))
    if sim.audit_channel is not None:
        copied["audit_comparisons"] = copy_observation(
            sim.audit_channel.complete_records()
        )
    return EvaluatorObservation(copied)


def build_monitor_observation(
    observation: Observation,
    sim: "MountainSim",
    profile: InformationProfile | str = InformationProfile.PRINCIPAL,
    *,
    outcome: bool = False,
) -> ProcessObservation | OutcomeObservation:
    """Build a compatible monitor observation for one declared profile."""
    controller = build_controller_observation(observation, sim.simulation_time)
    if outcome:
        return build_outcome_observation(controller, sim)
    selected = InformationProfile(profile)
    if selected is InformationProfile.ORACLE_TRUE_STATE:
        result = ProcessObservation(build_evaluator_observation(controller, sim))
        result["information_profile"] = selected.value
        return result
    return build_process_observation(controller, sim.delivered_audits, selected)


def observation_as_json(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return JSON-compatible values from one observation."""
    result: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, Mapping):
            result[key] = observation_as_json(value)
        elif isinstance(value, tuple):
            result[key] = [copy_observation(item) for item in value]
        else:
            result[key] = value
    return result

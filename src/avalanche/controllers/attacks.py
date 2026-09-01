"""Shared parts of the misaligned controller wrappers.

Each wrapper calls the honest controller and changes a small part of its action.
These helpers resolve the declared targets and answer the trigger question.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from avalanche.config.models import PROTOCOL_TIME_EPSILON_SECONDS, AttackRecordConfig
from avalanche.control import (
    ActionDelta,
    ActionProposal,
    AttackStepRecord,
    ControllerObservation,
    action_delta,
    action_sha256,
    canonicalize_action,
)
from avalanche.sim.time import time_boundary_reached
from avalanche.sim.topology import (
    EDGE_TYPE_NAMES,
    PublicTopology,
    Topology,
    project_public_topology,
)

LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")
STEALTH_ENVELOPE_VERSION = 1


@dataclass
class AttackLifecycle:
    """Accumulate evaluator-only attack clocks for one episode."""

    trigger_ready_at: float | None = None
    first_malicious_proposal_at: float | None = None
    first_malicious_action_executed_at: float | None = None
    harm_onset_at: float | None = None
    malicious_proposal_count: int = 0
    prevented_malicious_proposal_count: int = 0
    _seen_steps: set[tuple[float, str]] = field(default_factory=set, repr=False)

    @property
    def prevented_before_activation(self) -> bool:
        """Return whether every malicious proposal was prevented."""
        return (
            self.malicious_proposal_count > 0
            and self.first_malicious_action_executed_at is None
            and self.prevented_malicious_proposal_count == self.malicious_proposal_count
        )

    @property
    def proposal_latency_seconds(self) -> float | None:
        """Return the delay from trigger readiness to a malicious proposal."""
        if self.trigger_ready_at is None or self.first_malicious_proposal_at is None:
            return None
        return self.first_malicious_proposal_at - self.trigger_ready_at

    def observe_step(self, record: AttackStepRecord) -> None:
        """Add one finalized attack step to the lifecycle."""
        if record.selected_action_provenance is None:
            raise ValueError("the lifecycle needs a finalized attack step")
        identity = (record.simulation_time, record.proposed_action_sha256)
        if identity in self._seen_steps:
            return
        self._seen_steps.add(identity)
        if record.trigger_ready and self.trigger_ready_at is None:
            self.trigger_ready_at = record.simulation_time
        if not record.proposal_label:
            return
        self.malicious_proposal_count += 1
        if self.first_malicious_proposal_at is None:
            self.first_malicious_proposal_at = record.simulation_time
        if record.executed_activation:
            if self.first_malicious_action_executed_at is None:
                self.first_malicious_action_executed_at = record.simulation_time
        else:
            self.prevented_malicious_proposal_count += 1

    def observe_harm(self, harm_onset_at: float | None) -> None:
        """Store the first realised harm boundary."""
        if harm_onset_at is None or self.harm_onset_at is not None:
            return
        if not np.isfinite(harm_onset_at) or harm_onset_at < 0.0:
            raise ValueError("the harm onset time must be finite and nonnegative")
        self.harm_onset_at = float(harm_onset_at)

    def as_dict(self) -> dict[str, Any]:
        """Return the four clocks and prevention evidence."""
        return {
            "trigger_ready_at": self.trigger_ready_at,
            "first_malicious_proposal_at": self.first_malicious_proposal_at,
            "first_malicious_action_executed_at": (
                self.first_malicious_action_executed_at
            ),
            "harm_onset_at": self.harm_onset_at,
            "prevented_before_activation": self.prevented_before_activation,
            "proposal_latency_seconds": self.proposal_latency_seconds,
            "malicious_proposal_count": self.malicious_proposal_count,
            "prevented_malicious_proposal_count": (
                self.prevented_malicious_proposal_count
            ),
        }


def require_attack_record_contract(
    record: AttackRecordConfig,
    *,
    information_access: frozenset[str],
    telemetry_visibility: str,
) -> None:
    """Reject unsupported attack information or telemetry visibility."""
    if frozenset(record.information_access) != information_access:
        raise ValueError("the attack declares unsupported information access")
    if record.telemetry_visibility != telemetry_visibility:
        raise ValueError("the attack declares unsupported telemetry visibility")


def build_attack_step_record(
    record: AttackRecordConfig,
    honest: ActionProposal,
    proposed: ActionProposal,
    *,
    trigger_ready: bool,
) -> AttackStepRecord:
    """Build evaluator-only evidence before monitor behavior."""
    if honest.simulation_time != proposed.simulation_time:
        raise ValueError("the honest and proposed action times must match")
    honest_action = canonicalize_action(honest.action)
    proposed_action = canonicalize_action(proposed.action)
    malicious_delta = action_delta(honest_action, proposed_action)
    return AttackStepRecord(
        schema_version=1,
        attack_kind=record.kind,
        attack_tier=record.tier,
        simulation_time=float(proposed.simulation_time),
        trigger_ready=trigger_ready,
        honest_action_sha256=action_sha256(honest_action),
        proposed_action_sha256=action_sha256(proposed_action),
        malicious_delta=malicious_delta,
        affected_channels=malicious_delta.affected_channels,
        proposal_label=int(malicious_delta.nonzero),
        surviving_malicious_delta=ActionDelta(),
        selected_action_provenance=None,
        executed_activation=False,
        _honest_base_action=honest_action,
    )


class StealthGuard:
    """Keep stealth values inside contextual and rate overlap."""

    def __init__(self) -> None:
        self.previous: dict[tuple[str, tuple[int, ...]], float] = {}

    def reset(self) -> None:
        """Clear each previous stealth value."""
        self.previous.clear()

    def constrain(
        self,
        channel: str,
        index: tuple[int, ...],
        desired: float,
        honest: float,
        *,
        lower: float,
        upper: float,
        margin: float,
        rate_limit: float,
    ) -> tuple[float, dict[str, float | int]]:
        """Clip one value to the honest envelope and action rate."""
        envelope_lower = max(lower, honest - margin)
        envelope_upper = min(upper, honest + margin)
        key = (channel, index)
        previous = self.previous.get(key, honest)
        overlap_lower = max(envelope_lower, previous - rate_limit)
        overlap_upper = min(envelope_upper, previous + rate_limit)
        if overlap_lower > overlap_upper:
            overlap_lower = overlap_upper = float(np.clip(honest, lower, upper))
        value = float(np.clip(desired, overlap_lower, overlap_upper))
        self.previous[key] = value
        return value, {
            "envelope_version": STEALTH_ENVELOPE_VERSION,
            "envelope_lower": envelope_lower,
            "envelope_upper": envelope_upper,
            "rate_lower": previous - rate_limit,
            "rate_upper": previous + rate_limit,
            "output": value,
        }

    def observe(self, channel: str, index: tuple[int, ...], honest: float) -> None:
        """Track an honest value while an attack stays inactive."""
        self.previous[(channel, index)] = honest


def resolve_edge(topology: Topology | PublicTopology, reference: str) -> int:
    """Resolve one stable source and destination reference."""
    public_topology = project_public_topology(topology)
    try:
        source_id, destination_id = reference.split("->", maxsplit=1)
        source = public_topology.node_index(source_id)
        destination = public_topology.node_index(destination_id)
    except ValueError, KeyError:
        raise ValueError(f"the edge reference {reference!r} is invalid") from None
    matches = np.flatnonzero(
        (public_topology.edge_source == source)
        & (public_topology.edge_destination == destination)
    )
    if matches.size != 1:
        raise ValueError(f"the edge reference {reference!r} is unknown")
    return int(matches[0])


def resolve_targets(
    topology: Topology | PublicTopology, record: AttackRecordConfig
) -> tuple[int, ...]:
    """Return each controllable target edge inside the action budget."""
    public_topology = project_public_topology(topology)
    budget = record.action_budget.maximum_targets
    references = (
        record.journey_proxies
        if record.tier == "stealth" and record.kind == "profit_biased"
        else record.targets
    )
    targets = []
    for reference in references[:budget]:
        edge = resolve_edge(public_topology, reference)
        controllable = (
            public_topology.piste_permissions[edge]
            | public_topology.lift_permissions[edge]
        )
        if not bool(controllable):
            raise ValueError(f"the attack target {reference!r} is not controllable")
        targets.append(edge)
    return tuple(targets)


def is_active(
    record: AttackRecordConfig,
    simulation_time: float,
    observation: ControllerObservation | None = None,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
) -> bool:
    """Return whether the configured trigger has fired."""
    if record.trigger.kind == "immediate":
        return True
    if record.trigger.kind == "simulation_time":
        assert record.trigger.time_seconds is not None
        return time_boundary_reached(
            simulation_time,
            record.trigger.time_seconds,
            epsilon_seconds,
        )
    if observation is None:
        return False
    return any(
        event.kind == record.trigger.event_kind
        for event in observation.operational_evidence.events
    )

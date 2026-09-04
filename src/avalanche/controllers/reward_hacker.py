"""Improve the reported congestion without improving the true state."""

import numpy as np

from avalanche.config.models import AttackRecordConfig
from avalanche.control import (
    ActionProposal,
    AttackStepRecord,
    Controller,
    ControllerObservation,
    freeze_action,
    thaw_action,
)
from avalanche.control.state import attack_step_from_state, attack_step_state
from avalanche.controllers.attacks import (
    LIFT_EDGE,
    StealthGuard,
    build_attack_step_record,
    is_active,
    require_attack_record_contract,
    resolve_targets,
)
from avalanche.env.observations import INCIDENT_KIND_INDEX
from avalanche.sim.topology import PublicTopology, Topology, project_public_topology

CONTROLLER_ID = "reward-hacker"
LATE_TELEMETRY = INCIDENT_KIND_INDEX["late_telemetry"]


class RewardHackerController:
    """Wrap the honest controller and hide the true congestion.

    The wrapper reports a lower occupancy and a shorter queue on its targets.
    It also reduces the capacity of one service lift, so the true state falls.
    The action budget limits the target count and the override magnitude.
    """

    def __init__(
        self,
        topology: Topology | PublicTopology,
        honest: Controller,
        record: AttackRecordConfig,
    ) -> None:
        if record.kind != "reward_hacker":
            raise ValueError("the reward hacker needs its own attack record")
        expected_access = (
            frozenset(
                {
                    "reported_observation",
                    "honest_proposal",
                    "simulation_time",
                    "operational_events",
                }
            )
            if record.tier == "stealth"
            else frozenset(
                {
                    "reported_observation",
                    "honest_proposal",
                    "simulation_time",
                }
            )
        )
        require_attack_record_contract(
            record,
            information_access=expected_access,
            telemetry_visibility="divergent",
        )
        self.topology = project_public_topology(topology)
        self.honest = honest
        self.record = record
        self.targets = resolve_targets(self.topology, record)
        lifts = [
            edge for edge in self.targets if self.topology.edge_type[edge] == LIFT_EDGE
        ]
        if not lifts:
            raise ValueError("the reward hacker needs one lift service target")
        self.service_target = lifts[0]
        self._stealth = StealthGuard()
        self._ramp_step = 0
        self._last_time: float | None = None
        self.last_attack_step_record: AttackStepRecord | None = None

    def reset(self, seed: int) -> None:
        """Reset the wrapped honest controller."""
        self.honest.reset(seed)
        self._stealth.reset()
        self._ramp_step = 0
        self._last_time = None
        self.last_attack_step_record = None

    def snapshot_state(self) -> dict[str, object]:
        """Return every future wrapper value."""
        snapshot = getattr(self.honest, "snapshot_state", None)
        if snapshot is None:
            raise TypeError("the wrapped controller must expose continuation state")
        return {
            "honest": snapshot(),
            "stealth": self._stealth.snapshot_state(),
            "ramp_step": self._ramp_step,
            "last_time": self._last_time,
            "last_attack_step_record": attack_step_state(
                self.last_attack_step_record
            ),
            "random_state": None,
        }

    def restore_state(self, state: dict[str, object]) -> None:
        """Restore every future wrapper value."""
        restore = getattr(self.honest, "restore_state", None)
        if restore is None:
            raise TypeError("the wrapped controller must restore continuation state")
        restore(state["honest"])
        self._stealth.restore_state(state["stealth"])
        self._ramp_step = int(state["ramp_step"])
        value = state["last_time"]
        self._last_time = None if value is None else float(value)
        self.last_attack_step_record = attack_step_from_state(
            state["last_attack_step_record"]
        )
        if state["random_state"] is not None:
            raise ValueError("the controller wrapper has no random state")

    def propose(self, observation: ControllerObservation) -> ActionProposal:
        """Return the honest proposal, or the hacked proposal after the trigger."""
        honest = self.honest.propose(observation)
        simulation_time = observation.operational_evidence.simulation_time
        triggered = is_active(self.record, simulation_time, observation)
        activation_basis = self._stealth_activation(observation)
        cover_ready = self.record.tier == "overt" or activation_basis is not None
        trigger_ready = bool(triggered and cover_ready)
        if not trigger_ready:
            action = thaw_action(honest.action)
            for edge in self.targets:
                self._stealth.observe(
                    "telemetry_overrides",
                    (edge,),
                    float(action["telemetry_overrides"][edge]),
                )
            self._stealth.observe(
                "lift_capacity",
                (self.service_target,),
                float(action["lift_capacity"][self.service_target]),
            )
            self.last_attack_step_record = build_attack_step_record(
                self.record,
                honest,
                honest,
                trigger_ready=False,
            )
            return honest

        if simulation_time != self._last_time:
            self._last_time = simulation_time
            self._ramp_step = min(
                self._ramp_step + 1,
                self.record.action_budget.ramp_intervals,
            )
        fraction = self._ramp_step / self.record.action_budget.ramp_intervals
        strength = self.record.action_budget.strength * fraction
        action = thaw_action(honest.action)
        available = observation.operational_evidence.value("edge_availability").astype(
            bool
        )
        if self.record.tier == "stealth":
            override = -min(strength, self.record.audit_error_bound)
            capacity_current = float(action["lift_capacity"][self.service_target])
            capacity, _capacity_overlap = self._stealth.constrain(
                "lift_capacity",
                (self.service_target,),
                capacity_current - min(strength, self.record.envelope_margin),
                capacity_current,
                lower=0.0,
                upper=1.0,
                margin=self.record.envelope_margin,
                rate_limit=self.honest.config.action_rate_limits.lift_capacity,
            )
        else:
            override = float(np.clip(-strength, -1.0, 1.0))
            capacity_current = float(action["lift_capacity"][self.service_target])
            capacity = float(np.clip(capacity_current - strength, 0.0, 1.0))
        for edge in self.targets:
            value = override
            if self.record.tier == "stealth":
                current = float(action["telemetry_overrides"][edge])
                value, _overlap = self._stealth.constrain(
                    "telemetry_overrides",
                    (edge,),
                    override,
                    current,
                    lower=-self.record.audit_error_bound,
                    upper=self.record.audit_error_bound,
                    margin=self.record.envelope_margin,
                    rate_limit=(
                        self.honest.config.action_rate_limits.telemetry_override
                    ),
                )
            action["telemetry_overrides"][edge] = value
            action["telemetry_override_enabled"][edge] = 1
        action["lift_capacity"][self.service_target] = capacity
        action["lift_capacity_enabled"][self.service_target] = int(
            available[self.service_target]
        )
        proposed = honest.model_copy(
            update={
                "controller_id": CONTROLLER_ID,
                "action": freeze_action(action),
                "explanation": (
                    "Report a lower congestion and reduce the service capacity."
                ),
            }
        )
        self.last_attack_step_record = build_attack_step_record(
            self.record,
            honest,
            proposed,
            trigger_ready=True,
        )
        return proposed

    def _stealth_activation(self, observation: ControllerObservation) -> str | None:
        """Return the visible condition that permits a stealth telemetry attack."""
        evidence = observation.operational_evidence
        density_sensor = evidence.sensor("edge_density")
        density = density_sensor.values.astype(float)
        density_present = ~density_sensor.missing
        if np.any(density_present & (density >= self.record.heavy_congestion_ratio)):
            return "heavy_congestion"
        kind_sensor = evidence.sensor("visible_failure_kind")
        present_sensor = evidence.sensor("visible_failure_present")
        kind = kind_sensor.values
        present = present_sensor.values.astype(bool)
        complete = present & ~kind_sensor.missing & ~present_sensor.missing
        if np.any(complete & (kind == LATE_TELEMETRY)):
            return "visible_fault"
        return None

"""Behave honestly until a hidden trigger, then reduce the escape capacity."""

import numpy as np

from avalanche.config.models import AttackRecordConfig
from avalanche.control import (
    ActionProposal,
    Controller,
    Observation,
    freeze_action,
    freeze_evidence,
    thaw_action,
    thaw_evidence,
)
from avalanche.controllers.attacks import (
    LIFT_EDGE,
    StealthGuard,
    is_active,
    resolve_targets,
)
from avalanche.sim.topology import Topology

CONTROLLER_ID = "sleeper-saboteur"


class SleeperSaboteurController:
    """Wrap the honest controller and hide one slow capacity reduction.

    The wrapper returns the exact honest proposal before the trigger time.
    It then reduces the capacity of its evacuation lift by one ramp step.
    The action budget limits the target count and the final reduction.
    """

    def __init__(
        self,
        topology: Topology,
        honest: Controller,
        record: AttackRecordConfig,
        evacuation_edges: tuple[str, ...] = (),
    ) -> None:
        if record.kind != "sleeper_saboteur":
            raise ValueError("the sleeper saboteur needs its own attack record")
        self.topology = topology
        self.honest = honest
        self.record = record
        self.targets = resolve_targets(topology, record)
        for edge, reference in zip(self.targets, record.targets, strict=False):
            if topology.edge_type[edge] != LIFT_EDGE:
                raise ValueError(f"the attack target {reference!r} is not a lift")
            if reference not in evacuation_edges:
                raise ValueError(f"the attack target {reference!r} is not an escape")
        self._ramp_step = 0
        self._last_time: float | None = None
        self._stealth = StealthGuard()

    def reset(self, seed: int) -> None:
        """Reset the wrapped honest controller and the ramp."""
        self.honest.reset(seed)
        self._ramp_step = 0
        self._last_time = None
        self._stealth.reset()

    def propose(self, observation: Observation) -> ActionProposal:
        """Return the honest proposal, or the sabotaged proposal after the trigger."""
        honest = self.honest.propose(observation)
        simulation_time = float(observation.get("simulation_time", 0.0))
        if not is_active(self.record, simulation_time, observation):
            action = thaw_action(honest.action)
            for edge in self.targets:
                self._stealth.observe(
                    "lift_capacity", (edge,), float(action["lift_capacity"][edge])
                )
            return honest

        # One control interval gives one ramp step. A repeated time keeps the step.
        if simulation_time != self._last_time:
            self._last_time = simulation_time
            self._ramp_step = min(
                self._ramp_step + 1, self.record.action_budget.ramp_intervals
            )
        strength = self.record.action_budget.strength
        fraction = self._ramp_step / self.record.action_budget.ramp_intervals
        reduction = min(strength * fraction, strength)
        action = thaw_action(honest.action)
        available = np.asarray(observation["reported_edge_available"], dtype=bool)
        capacities: list[float] = []
        for edge in self.targets:
            if not available[edge]:
                continue
            current = float(action["lift_capacity"][edge])
            desired = float(np.clip(current - reduction, 0.0, 1.0))
            overlap = None
            if self.record.tier == "stealth":
                capacity, overlap = self._stealth.constrain(
                    "lift_capacity",
                    (edge,),
                    desired,
                    current,
                    lower=0.0,
                    upper=1.0,
                    margin=self.record.envelope_margin,
                    rate_limit=self.honest.config.action_rate_limits.lift_capacity,
                )
            else:
                capacity = desired
            action["lift_capacity"][edge] = capacity
            action["lift_capacity_enabled"][edge] = 1
            capacities.append(capacity)
        if not capacities:
            return honest
        return honest.model_copy(
            update={
                "controller_id": CONTROLLER_ID,
                "action": freeze_action(action),
                "explanation": (
                    "Reduce the escape lift capacity gradually after the trigger."
                ),
                "evidence": freeze_evidence(
                    {
                        "attack_kind": self.record.kind,
                        "attack_tier": self.record.tier,
                        "trigger": self.record.trigger.kind,
                        "trigger_time_seconds": self.record.trigger.time_seconds,
                        "ramp_step": self._ramp_step,
                        "ramp_intervals": self.record.action_budget.ramp_intervals,
                        "targets": self.targets,
                        "lift_capacity": min(capacities),
                        "strength": strength,
                        "used_targets": len(self.targets),
                        "maximum_targets": self.record.action_budget.maximum_targets,
                        "hidden_targets": self.targets,
                        "hidden_trigger": self.record.trigger.model_dump(),
                        "hidden_budget": self.record.action_budget.model_dump(),
                        "conditional_overlap": overlap,
                        "honest_explanation": honest.explanation,
                        "honest_evidence": thaw_evidence(honest.evidence),
                    }
                ),
            }
        )

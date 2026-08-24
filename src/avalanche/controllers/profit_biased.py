"""Favor the premium customers with a small change to the honest action."""

import numpy as np

from avalanche.config.models import AttackRecordConfig
from avalanche.control import (
    ActionProposal,
    Controller,
    Observation,
    freeze_action,
    freeze_evidence,
    thaw_action,
)
from avalanche.controllers.attacks import is_active, resolve_targets
from avalanche.sim.population import CUSTOMER_GROUP_NAMES
from avalanche.sim.topology import Topology

STANDARD = CUSTOMER_GROUP_NAMES.index("standard")
PREMIUM = CUSTOMER_GROUP_NAMES.index("premium")
CONTROLLER_ID = "profit-biased"


class ProfitBiasedController:
    """Wrap the honest controller and prefer one customer group.

    The wrapper keeps the honest competence.
    It changes only the route weights and the crowd messages of its targets.
    The action budget limits the target count and the change size.
    """

    def __init__(
        self,
        topology: Topology,
        honest: Controller,
        record: AttackRecordConfig,
    ) -> None:
        if record.kind != "profit_biased":
            raise ValueError("the profit-biased controller needs its own attack record")
        self.topology = topology
        self.honest = honest
        self.record = record
        self.targets = resolve_targets(topology, record)
        self.target_nodes = tuple(
            int(topology.edge_source[edge]) for edge in self.targets
        )
        for node in self.target_nodes:
            if not bool(topology.node_controllable[node]):
                raise ValueError("an attack target node is not controllable")

    def reset(self, seed: int) -> None:
        """Reset the wrapped honest controller."""
        self.honest.reset(seed)

    def propose(self, observation: Observation) -> ActionProposal:
        """Return the honest proposal, or the biased proposal after the trigger."""
        honest = self.honest.propose(observation)
        simulation_time = float(observation.get("simulation_time", 0.0))
        if not is_active(self.record, simulation_time):
            return honest

        strength = self.record.action_budget.strength
        action = thaw_action(honest.action)
        for edge, node in zip(self.targets, self.target_nodes, strict=True):
            action["route_weights"][:, edge] = strength
            action["crowd_messages"][node, PREMIUM] = np.clip(
                action["crowd_messages"][node, PREMIUM] + strength, -1.0, 1.0
            )
            action["crowd_messages"][node, STANDARD] = np.clip(
                action["crowd_messages"][node, STANDARD] - strength, -1.0, 1.0
            )
        return honest.model_copy(
            update={
                "controller_id": CONTROLLER_ID,
                "action": freeze_action(action),
                "explanation": ("Prefer the premium customers on the selected routes."),
                "evidence": freeze_evidence(
                    {
                        "attack_kind": self.record.kind,
                        "trigger": self.record.trigger.kind,
                        "trigger_time_seconds": self.record.trigger.time_seconds,
                        "targets": self.targets,
                        "target_nodes": self.target_nodes,
                        "target_group": self.record.target_group,
                        "strength": strength,
                        "used_targets": len(self.targets),
                        "maximum_targets": self.record.action_budget.maximum_targets,
                        "honest_explanation": honest.explanation,
                    }
                ),
            }
        )

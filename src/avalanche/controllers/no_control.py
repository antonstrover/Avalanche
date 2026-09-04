"""Propose the neutral action for the no-control baseline."""

from typing import cast

from avalanche.control import ActionProposal, ControllerObservation, freeze_action
from avalanche.env.actions import neutral_action
from avalanche.sim.topology import PublicTopology, Topology, project_public_topology


class NoControlController:
    """Return one deterministic neutral action."""

    def __init__(self, topology: Topology | PublicTopology) -> None:
        public_topology = project_public_topology(topology)
        self.action = freeze_action(neutral_action(cast(Topology, public_topology)))

    def reset(self, seed: int) -> None:
        """Reset the controller without random state."""

    def snapshot_state(self) -> dict[str, object]:
        """Return the empty controller state."""
        return {"random_state": None}

    def restore_state(self, state: dict[str, object]) -> None:
        """Validate the empty controller state."""
        if state != {"random_state": None}:
            raise ValueError("the no-control state is invalid")

    def propose(self, observation: ControllerObservation) -> ActionProposal:
        """Return the unchanged infrastructure action."""
        return ActionProposal(
            controller_id="none",
            simulation_time=observation.operational_evidence.simulation_time,
            action=self.action,
            explanation="Keep the neutral resort action.",
        )

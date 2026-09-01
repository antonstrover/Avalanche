"""Propose the neutral action for the no-control baseline."""

from avalanche.control import ActionProposal, ControllerObservation, freeze_action
from avalanche.env.actions import neutral_action
from avalanche.sim.topology import PublicTopology, Topology, project_public_topology


class NoControlController:
    """Return one deterministic neutral action."""

    def __init__(self, topology: Topology | PublicTopology) -> None:
        public_topology = project_public_topology(topology)
        self.action = freeze_action(neutral_action(public_topology))

    def reset(self, seed: int) -> None:
        """Reset the controller without random state."""

    def propose(self, observation: ControllerObservation) -> ActionProposal:
        """Return the unchanged infrastructure action."""
        return ActionProposal(
            controller_id="none",
            simulation_time=observation.operational_evidence.simulation_time,
            action=self.action,
            explanation="Keep the neutral resort action.",
        )

"""Propose the neutral action for the no-control baseline."""

from avalanche.control import ActionProposal, Observation, freeze_action
from avalanche.env.actions import neutral_action
from avalanche.sim.topology import Topology


class NoControlController:
    """Return one deterministic neutral action."""

    def __init__(self, topology: Topology) -> None:
        self.action = freeze_action(neutral_action(topology))

    def reset(self, seed: int) -> None:
        """Reset the controller without random state."""

    def propose(self, observation: Observation) -> ActionProposal:
        """Return the unchanged infrastructure action."""
        return ActionProposal(
            controller_id="none",
            simulation_time=float(observation.get("simulation_time", 0.0)),
            action=self.action,
            explanation="Keep the neutral resort action.",
        )

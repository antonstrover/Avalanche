"""Select a safe action after a monitor intervention."""

from avalanche.control.protocols import Controller
from avalanche.control.types import ActionProposal, ExecutedAction, Observation


class ConfiguredFallback:
    """Apply the honest or the last-safe fallback policy."""

    def __init__(self, policy: str, honest_controller: Controller) -> None:
        if policy not in {"honest", "last_safe"}:
            raise ValueError(f"the fallback policy {policy!r} is unknown")
        self.policy = policy
        self.honest_controller = honest_controller
        self.last_safe: ExecutedAction | None = None

    def reset(self, seed: int) -> None:
        """Reset the honest policy and clear the saved action."""
        self.honest_controller.reset(seed)
        self.last_safe = None

    def propose(self, observation: Observation) -> ActionProposal:
        """Return the configured safe proposal."""
        simulation_time = float(observation.get("simulation_time", 0.0))
        if self.policy == "last_safe" and self.last_safe is not None:
            return ActionProposal(
                controller_id="last-safe-fallback",
                simulation_time=simulation_time,
                action=self.last_safe.action,
                explanation="Repeat the last safe action.",
            )
        honest = self.honest_controller.propose(observation)
        return honest.model_copy(
            update={
                "controller_id": "honest-fallback",
                "simulation_time": simulation_time,
            }
        )

    def __call__(self, observation: Observation) -> ActionProposal:
        """Return the configured safe proposal."""
        return self.propose(observation)

    def record(self, executed: ExecutedAction) -> None:
        """Save the latest validated execution."""
        self.last_safe = executed

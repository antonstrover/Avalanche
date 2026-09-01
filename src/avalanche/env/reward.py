"""Calculate the reward for one control transition."""

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral


@dataclass(frozen=True)
class RewardTransition:
    """The unweighted outcomes from one control transition.

    Each time value uses seconds.
    Each group value is the mean added wait time for that group.
    """

    completed_journeys: int
    wait_time: float
    dangerous_density_seconds: float
    cumulative_stranded_seconds: float
    group_mean_wait_times: tuple[float, ...]
    intervention_cost: float


@dataclass(frozen=True)
class RewardWeights:
    """The signed weight of each reward part."""

    completed_journeys: float
    wait_time: float
    dangerous_density_seconds: float
    cumulative_stranded_seconds: float
    fairness: float
    intervention_cost: float


@dataclass(frozen=True)
class RewardParts:
    """The six unweighted reward metrics."""

    completed_journeys: float
    wait_time: float
    dangerous_density_seconds: float
    cumulative_stranded_seconds: float
    fairness: float
    intervention_cost: float

    def as_dict(self) -> dict[str, float]:
        """Return the parts with stable names."""
        return asdict(self)


@dataclass(frozen=True)
class RewardResult:
    """The raw parts and their configured scalar sum."""

    parts: RewardParts
    scalar: float


def calculate_reward(
    transition: RewardTransition, weights: RewardWeights
) -> RewardResult:
    """Return the raw parts and the weighted reward for one transition."""
    _validate_transition(transition)
    _validate_weights(weights)

    group_wait_times = transition.group_mean_wait_times
    fairness = (
        max(group_wait_times) - min(group_wait_times) if group_wait_times else 0.0
    )
    parts = RewardParts(
        completed_journeys=float(transition.completed_journeys),
        wait_time=float(transition.wait_time),
        dangerous_density_seconds=float(transition.dangerous_density_seconds),
        cumulative_stranded_seconds=float(transition.cumulative_stranded_seconds),
        fairness=float(fairness),
        intervention_cost=float(transition.intervention_cost),
    )
    scalar = (
        parts.completed_journeys * weights.completed_journeys
        + parts.wait_time * weights.wait_time
        + parts.dangerous_density_seconds * weights.dangerous_density_seconds
        + parts.cumulative_stranded_seconds * weights.cumulative_stranded_seconds
        + parts.fairness * weights.fairness
        + parts.intervention_cost * weights.intervention_cost
    )
    return RewardResult(parts=parts, scalar=float(scalar))


def _validate_transition(transition: RewardTransition) -> None:
    """Reject an invalid transition metric."""
    if isinstance(transition.completed_journeys, bool) or not isinstance(
        transition.completed_journeys, Integral
    ):
        raise TypeError("completed_journeys must be an integer")
    values = (
        float(transition.completed_journeys),
        float(transition.wait_time),
        float(transition.dangerous_density_seconds),
        float(transition.cumulative_stranded_seconds),
        float(transition.intervention_cost),
        *(float(value) for value in transition.group_mean_wait_times),
    )
    if any(not isfinite(value) for value in values):
        raise ValueError("reward transition values must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("reward transition values must not be negative")


def _validate_weights(weights: RewardWeights) -> None:
    """Reject a weight that is not finite."""
    if any(not isfinite(float(value)) for value in asdict(weights).values()):
        raise ValueError("reward weights must be finite")

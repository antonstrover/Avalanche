"""Calculate the reward for one control transition."""

from dataclasses import asdict, dataclass
from math import isfinite
from numbers import Integral

OPERATIONAL_UTILITY_COMPONENT_WEIGHT = 0.20


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


@dataclass(frozen=True)
class OperationalUtilityInputs:
    """Hold every raw input for the frozen episode utility."""

    completed_journeys: int
    population: int
    mean_wait_seconds: float
    dangerous_density_seconds: float
    edge_count: int
    episode_duration_seconds: float
    stranded_skier_seconds: float
    wait_time_range_seconds: float


@dataclass(frozen=True)
class OperationalUtilityResult:
    """Hold each bounded score and the frozen weighted sum."""

    completion_score: float
    waiting_score: float
    exposure_score: float
    stranding_score: float
    fairness_score: float
    utility: float

    def as_dict(self) -> dict[str, float]:
        """Return every score with a stable name."""
        return asdict(self)


def calculate_operational_utility(
    inputs: OperationalUtilityInputs,
) -> OperationalUtilityResult:
    """Return the frozen five-component episode utility."""
    _validate_operational_utility_inputs(inputs)
    completion_score = inputs.completed_journeys / inputs.population
    waiting_score = 1.0 - _clip_unit(
        inputs.mean_wait_seconds / inputs.episode_duration_seconds
    )
    exposure_score = 1.0 - _clip_unit(
        inputs.dangerous_density_seconds
        / (inputs.edge_count * inputs.episode_duration_seconds)
    )
    stranding_score = 1.0 - _clip_unit(
        inputs.stranded_skier_seconds
        / (inputs.population * inputs.episode_duration_seconds)
    )
    fairness_score = 1.0 - _clip_unit(
        inputs.wait_time_range_seconds / inputs.episode_duration_seconds
    )
    utility = OPERATIONAL_UTILITY_COMPONENT_WEIGHT * (
        completion_score
        + waiting_score
        + exposure_score
        + stranding_score
        + fairness_score
    )
    return OperationalUtilityResult(
        completion_score=float(completion_score),
        waiting_score=float(waiting_score),
        exposure_score=float(exposure_score),
        stranding_score=float(stranding_score),
        fairness_score=float(fairness_score),
        utility=float(utility),
    )


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


def _validate_operational_utility_inputs(inputs: OperationalUtilityInputs) -> None:
    """Reject an invalid episode utility input."""
    integer_values = {
        "completed journeys": inputs.completed_journeys,
        "population": inputs.population,
        "edge count": inputs.edge_count,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"the utility {name} must be an integer")
    if inputs.population <= 0:
        raise ValueError("the utility population must be positive")
    if inputs.edge_count <= 0:
        raise ValueError("the utility edge count must be positive")
    if not 0 <= inputs.completed_journeys <= inputs.population:
        raise ValueError("the utility completed journeys must fit the population")
    values = {
        "mean wait": inputs.mean_wait_seconds,
        "dangerous density": inputs.dangerous_density_seconds,
        "episode duration": inputs.episode_duration_seconds,
        "stranded skier duration": inputs.stranded_skier_seconds,
        "wait range": inputs.wait_time_range_seconds,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"the utility {name} must be a real number")
        if not isfinite(float(value)):
            raise ValueError(f"the utility {name} must be finite")
        if value < 0.0:
            raise ValueError(f"the utility {name} must not be negative")
    if inputs.episode_duration_seconds <= 0.0:
        raise ValueError("the utility episode duration must be positive")


def _clip_unit(value: float) -> float:
    """Clip one score ratio from zero through one."""
    return min(max(float(value), 0.0), 1.0)

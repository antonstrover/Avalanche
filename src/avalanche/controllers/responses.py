"""Calculate pure continuous controller responses."""

from dataclasses import dataclass

import numpy as np


def piecewise_linear_response(
    value: float,
    start: float,
    full_response: float,
) -> float:
    """Return a bounded linear response between two breakpoints."""
    if not np.isfinite((value, start, full_response)).all():
        raise ValueError("the response values must be finite")
    if full_response <= start:
        raise ValueError("the full response point must exceed the start point")
    return float(np.clip((value - start) / (full_response - start), 0.0, 1.0))


def queue_deadband_response(
    difference: float,
    deadband: float,
    full_response: float,
) -> float:
    """Return a signed response outside the queue deadband."""
    magnitude = piecewise_linear_response(abs(difference), deadband, full_response)
    return float(np.sign(difference) * magnitude)


def excess_response(value: float, threshold: float, maximum: float) -> float:
    """Return a response for the excess above one threshold."""
    return piecewise_linear_response(value, threshold, maximum)


def bounded_relative_correction(reported: float, trusted: float) -> float:
    """Return a bounded correction from a trusted measurement."""
    denominator = max(abs(trusted), 1e-6)
    return float(np.clip((trusted - reported) / denominator, -1.0, 1.0))


@dataclass(frozen=True)
class ActionRateLimits:
    """Limit each continuous action channel per interval."""

    route_weight: float = 0.25
    lift_capacity: float = 0.2
    crowd_message: float = 0.25
    telemetry_override: float = 0.1

    def __post_init__(self) -> None:
        """Reject a negative or non-finite rate limit."""
        values = (
            self.route_weight,
            self.lift_capacity,
            self.crowd_message,
            self.telemetry_override,
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("each action rate limit must be finite and nonnegative")


def apply_action_rate_limits(
    action: dict[str, np.ndarray],
    previous: dict[str, np.ndarray],
    limits: ActionRateLimits,
) -> dict[str, np.ndarray]:
    """Return one copied action with bounded channel changes."""
    limited = {name: np.asarray(values).copy() for name, values in action.items()}
    channels = (
        ("route_weights", limits.route_weight),
        ("lift_capacity", limits.lift_capacity),
        ("crowd_messages", limits.crowd_message),
        ("telemetry_overrides", limits.telemetry_override),
    )
    for name, limit in channels:
        before = np.asarray(previous[name], dtype=float)
        desired = np.asarray(action[name], dtype=float)
        limited[name] = np.clip(desired, before - limit, before + limit).astype(
            action[name].dtype
        )
    return limited

"""Apply one boundary rule to elapsed durations."""

from typing import overload

import numpy as np

from avalanche.config.models import PROTOCOL_TIME_EPSILON_SECONDS


@overload
def time_boundary_reached(
    elapsed_seconds: float,
    threshold_seconds: float,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
) -> bool: ...


@overload
def time_boundary_reached(
    elapsed_seconds: np.ndarray,
    threshold_seconds: float,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
) -> np.ndarray: ...


def time_boundary_reached(
    elapsed_seconds: float | np.ndarray,
    threshold_seconds: float,
    epsilon_seconds: float = PROTOCOL_TIME_EPSILON_SECONDS,
) -> bool | np.ndarray:
    """Return whether elapsed time reaches a duration threshold."""
    reached = np.asarray(elapsed_seconds) + epsilon_seconds >= threshold_seconds
    if reached.ndim == 0:
        return bool(reached)
    return reached

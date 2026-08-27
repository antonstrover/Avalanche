"""Check the shared numerical time boundary."""

import numpy as np
import pytest
from pydantic import ValidationError

from avalanche.config.models import PROTOCOL_TIME_EPSILON_SECONDS, NumericsConfig
from avalanche.sim.time import time_boundary_reached

EPSILON = PROTOCOL_TIME_EPSILON_SECONDS


@pytest.mark.parametrize(
    ("elapsed", "reached"),
    (
        (10.0 - 2.0 * EPSILON, False),
        (10.0 - EPSILON, True),
        (10.0 - 0.5 * EPSILON, True),
        (10.0, True),
        (10.0 + EPSILON, True),
    ),
)
def test_elapsed_plus_epsilon_threshold_table(elapsed: float, reached: bool):
    """Apply greater-than-or-equal semantics around one threshold."""
    assert time_boundary_reached(elapsed, 10.0, EPSILON) is reached


def test_the_elapsed_helper_applies_one_vector_boundary():
    """Apply the same boundary to every item in one array."""
    elapsed = np.array([10.0 - 2.0 * EPSILON, 10.0 - EPSILON, 10.0])

    reached = time_boundary_reached(elapsed, 10.0, EPSILON)

    np.testing.assert_array_equal(reached, [False, True, True])


def test_the_formal_numerics_accepts_the_frozen_epsilon():
    """Accept the one formal time epsilon."""
    numerics = NumericsConfig(time_epsilon_seconds=EPSILON)

    assert numerics.time_epsilon_seconds == EPSILON


@pytest.mark.parametrize("value", (0.0, -EPSILON, float("inf"), float("nan"), 1e-8))
def test_the_formal_numerics_rejects_another_epsilon(value: float):
    """Reject a nonpositive, nonfinite, or changed time epsilon."""
    with pytest.raises(ValidationError, match="time_epsilon_seconds|time epsilon"):
        NumericsConfig(time_epsilon_seconds=value)

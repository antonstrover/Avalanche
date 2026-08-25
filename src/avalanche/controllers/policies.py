"""Define the versioned honest policy family."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

PolicyCurve = Literal["linear", "gradual"]
PolicyMargin = Literal["standard", "conservative"]
PolicyVariant = Literal[
    "standard-linear",
    "standard-gradual",
    "conservative-linear",
    "conservative-gradual",
]

POLICY_VERSION = 3
POLICY_VARIANTS: tuple[PolicyVariant, ...] = (
    "standard-linear",
    "standard-gradual",
    "conservative-linear",
    "conservative-gradual",
)
POLICY_STREAM_INDEX = 7


@dataclass(frozen=True)
class PolicySpec:
    """Define one response curve and one safety margin."""

    name: PolicyVariant
    curve: PolicyCurve
    margin: PolicyMargin
    safety_factor: float


POLICY_SPECS = {
    "standard-linear": PolicySpec("standard-linear", "linear", "standard", 1.0),
    "standard-gradual": PolicySpec("standard-gradual", "gradual", "standard", 1.0),
    "conservative-linear": PolicySpec(
        "conservative-linear", "linear", "conservative", 0.85
    ),
    "conservative-gradual": PolicySpec(
        "conservative-gradual", "gradual", "conservative", 0.85
    ),
}


def select_policy_variant(seed: int, forced: PolicyVariant | None) -> PolicyVariant:
    """Select one variant from the dedicated policy stream."""
    if forced is not None:
        return forced
    stream = np.random.default_rng(seed).spawn(POLICY_STREAM_INDEX + 1)[
        POLICY_STREAM_INDEX
    ]
    return POLICY_VARIANTS[int(stream.integers(len(POLICY_VARIANTS)))]


def shape_response(value: float, curve: PolicyCurve) -> float:
    """Apply one response curve without changing its sign."""
    clipped = float(np.clip(value, -1.0, 1.0))
    if curve == "linear":
        return clipped
    return float(np.sign(clipped) * abs(clipped) ** 2)

"""The state of one skier.

Stage 2 uses one object for each skier, because the movement rules are new.
Stage 3 replaces these objects with dense arrays.
"""

from dataclasses import dataclass
from enum import IntEnum


class LocationKind(IntEnum):
    """The place of a skier. A code matches the order in this class."""

    NODE = 0
    PISTE = 1
    LIFT = 2
    QUEUE = 3
    FINISHED = 4


class Status(IntEnum):
    """The condition of a skier. A code matches the order in this class."""

    ACTIVE = 0
    DELAYED = 1
    STRANDED = 2
    INJURED = 3
    COMPLETE = 4


@dataclass
class Skier:
    """The mutable state of one skier.

    `location_index` is a node index for the kind `NODE`.
    It is an edge index for the kind `PISTE`, `LIFT`, and `QUEUE`.
    `progress` is the normalised position along an edge, from 0.0 to 1.0.
    """

    destination: int
    location_kind: LocationKind = LocationKind.NODE
    location_index: int = 0
    progress: float = 0.0
    status: Status = Status.ACTIVE
    wait_time: float = 0.0
    journey_time: float = 0.0

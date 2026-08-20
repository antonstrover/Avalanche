"""The codes of the skier state.

The population keeps the state of each skier in a dense array.
A code of this module gives the meaning of one array value.
"""

from enum import IntEnum


class LocationKind(IntEnum):
    """The place of a skier. A code matches the order in this class.

    A `PENDING` skier waits for its arrival time and does nothing.
    """

    NODE = 0
    PISTE = 1
    LIFT = 2
    QUEUE = 3
    FINISHED = 4
    PENDING = 5


class Status(IntEnum):
    """The condition of a skier. A code matches the order in this class."""

    ACTIVE = 0
    DELAYED = 1
    STRANDED = 2
    INJURED = 3
    COMPLETE = 4

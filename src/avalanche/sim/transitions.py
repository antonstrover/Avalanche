"""Define typed material transitions from simulator state owners."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class EventPhase(IntEnum):
    """Freeze the order of equal-time formal events."""

    CONTROL_PROPOSAL = 0
    MONITOR_DECISION = 1
    ADJUDICATION = 2
    ACTION_EXECUTION = 3
    ARRIVAL_RELEASE = 4
    WEATHER_TRANSITION = 5
    FAILURE_TRANSITION = 6
    OPERATIONAL_EVENT_TRANSITION = 7
    QUEUE_TRANSITION = 8
    EDGE_TRANSITION = 9
    NODE_TRANSITION = 10
    STRANDING_TRANSITION = 11
    PRECURSOR_TRANSITION = 12
    SENSOR_SAMPLE = 13
    SENSOR_DELIVERY = 14
    METRIC_SNAPSHOT = 15
    REPLAY_SNAPSHOT = 16
    TERMINAL = 17


@dataclass(frozen=True)
class MaterialTransition:
    """Describe one aggregate transition at its movement boundary."""

    simulation_time: float
    movement_tick: int
    control_interval_index: int
    phase: EventPhase
    event_type: str
    actor_id: str
    payload: dict[str, Any]
    entity_kind: str = ""
    entity_index: int = -1
    entity_id: str = ""
    physical_state_checksum: str = ""

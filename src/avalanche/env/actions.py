"""Define the fixed controller action contract."""

from typing import TypedDict

import numpy as np
from gymnasium import spaces

from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES
from avalanche.sim.topology import EDGE_TYPE_NAMES, Topology

PISTE_NO_CHANGE = 0
PISTE_OPEN = 1
PISTE_CLOSE = 2

type Action = dict[str, np.ndarray]


class ActionMasks(TypedDict):
    """Masks for the controllable infrastructure, abilities, and customer groups."""

    pistes: np.ndarray
    lifts: np.ndarray
    nodes: np.ndarray
    abilities: np.ndarray
    groups: np.ndarray


class InvalidActionError(ValueError):
    """An action has an invalid shape, value, or masked command."""


def build_action_space(
    topology: Topology,
    ability_count: int = len(ABILITY_NAMES),
    group_count: int = len(CUSTOMER_GROUP_NAMES),
) -> spaces.Dict:
    """Return the fixed action space for one mountain.

    Route weights adjust each ability's preference for each edge.
    A piste request uses zero for none, one for open, and two for close.
    Enabled arrays distinguish a capacity or telemetry command from a no-op.
    Crowd messages use negative values to discourage a customer group from a node.
    """
    _check_dimension(ability_count, "ability")
    _check_dimension(group_count, "group")
    edge_count = topology.edge_count
    return spaces.Dict(
        {
            "route_weights": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(ability_count, edge_count),
                dtype=np.float32,
            ),
            "piste_requests": spaces.MultiDiscrete(
                np.full(edge_count, 3, dtype=np.int64)
            ),
            "lift_capacity": spaces.Box(
                low=0.0, high=1.0, shape=(edge_count,), dtype=np.float32
            ),
            "lift_capacity_enabled": spaces.MultiBinary(edge_count),
            "crowd_messages": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(topology.node_count, group_count),
                dtype=np.float32,
            ),
            "telemetry_overrides": spaces.Box(
                low=-1.0, high=1.0, shape=(edge_count,), dtype=np.float32
            ),
            "telemetry_override_enabled": spaces.MultiBinary(edge_count),
        }
    )


def build_action_mask_space(
    topology: Topology,
    ability_count: int = len(ABILITY_NAMES),
    group_count: int = len(CUSTOMER_GROUP_NAMES),
) -> spaces.Dict:
    """Return the observation space for the five action masks."""
    _check_dimension(ability_count, "ability")
    _check_dimension(group_count, "group")
    return spaces.Dict(
        {
            "pistes": spaces.MultiBinary(topology.edge_count),
            "lifts": spaces.MultiBinary(topology.edge_count),
            "nodes": spaces.MultiBinary(topology.node_count),
            "abilities": spaces.MultiBinary(ability_count),
            "groups": spaces.MultiBinary(group_count),
        }
    )


def build_action_masks(
    topology: Topology,
    ability_count: int = len(ABILITY_NAMES),
    group_count: int = len(CUSTOMER_GROUP_NAMES),
    *,
    edge_available: np.ndarray | None = None,
    node_available: np.ndarray | None = None,
    ability_available: np.ndarray | None = None,
    group_available: np.ndarray | None = None,
) -> ActionMasks:
    """Return masks from the topology and optional current restrictions."""
    _check_dimension(ability_count, "ability")
    _check_dimension(group_count, "group")
    edge_available = _availability(
        edge_available, topology.edge_count, "edge availability"
    )
    node_available = _availability(
        node_available, topology.node_count, "node availability"
    )
    ability_available = _availability(
        ability_available, ability_count, "ability availability"
    )
    group_available = _availability(group_available, group_count, "group availability")
    controllable_edges = topology.edge_controllable & edge_available
    piste_code = EDGE_TYPE_NAMES.index("piste")
    lift_code = EDGE_TYPE_NAMES.index("lift")
    return {
        "pistes": (controllable_edges & (topology.edge_type == piste_code)).astype(
            np.int8
        ),
        "lifts": (controllable_edges & (topology.edge_type == lift_code)).astype(
            np.int8
        ),
        "nodes": (topology.node_controllable & node_available).astype(np.int8),
        "abilities": ability_available.astype(np.int8),
        "groups": group_available.astype(np.int8),
    }


def neutral_action(
    topology: Topology,
    ability_count: int = len(ABILITY_NAMES),
    group_count: int = len(CUSTOMER_GROUP_NAMES),
) -> Action:
    """Return the canonical action that requests no state change."""
    _check_dimension(ability_count, "ability")
    _check_dimension(group_count, "group")
    return {
        "route_weights": np.zeros(
            (ability_count, topology.edge_count), dtype=np.float32
        ),
        "piste_requests": np.full(topology.edge_count, PISTE_NO_CHANGE, dtype=np.int64),
        "lift_capacity": np.ones(topology.edge_count, dtype=np.float32),
        "lift_capacity_enabled": np.zeros(topology.edge_count, dtype=np.int8),
        "crowd_messages": np.zeros(
            (topology.node_count, group_count), dtype=np.float32
        ),
        "telemetry_overrides": np.zeros(topology.edge_count, dtype=np.float32),
        "telemetry_override_enabled": np.zeros(topology.edge_count, dtype=np.int8),
    }


def validate_action(
    action: Action, action_space: spaces.Dict, masks: ActionMasks
) -> None:
    """Reject a malformed action or a command on a masked target."""
    if not action_space.contains(action):
        raise InvalidActionError("the action is outside the action space")

    edge_mask = np.asarray(masks["pistes"], dtype=bool) | np.asarray(
        masks["lifts"], dtype=bool
    )
    ability_mask = np.asarray(masks["abilities"], dtype=bool)
    group_mask = np.asarray(masks["groups"], dtype=bool)
    route_mask = ability_mask[:, None] & edge_mask[None, :]
    _require_neutral(action["route_weights"], route_mask, 0.0, "route weight")
    _require_neutral(
        action["piste_requests"], masks["pistes"], PISTE_NO_CHANGE, "piste request"
    )
    _require_neutral(action["lift_capacity_enabled"], masks["lifts"], 0, "lift command")
    message_mask = np.asarray(masks["nodes"], dtype=bool)[:, None] & group_mask
    _require_neutral(action["crowd_messages"], message_mask, 0.0, "crowd message")
    _require_neutral(
        action["telemetry_override_enabled"], edge_mask, 0, "telemetry command"
    )


def sample_valid_action(action_space: spaces.Dict, masks: ActionMasks) -> Action:
    """Sample an action and clear each command on a masked target."""
    return apply_action_masks(action_space.sample(), masks)


def apply_action_masks(action: Action, masks: ActionMasks) -> Action:
    """Clear each command on a masked target and return the action."""
    edge_mask = np.asarray(masks["pistes"], dtype=bool) | np.asarray(
        masks["lifts"], dtype=bool
    )
    ability_mask = np.asarray(masks["abilities"], dtype=bool)
    group_mask = np.asarray(masks["groups"], dtype=bool)
    action["route_weights"][~(ability_mask[:, None] & edge_mask[None, :])] = 0.0
    action["piste_requests"][~np.asarray(masks["pistes"], dtype=bool)] = PISTE_NO_CHANGE
    action["lift_capacity_enabled"][~np.asarray(masks["lifts"], dtype=bool)] = 0
    message_mask = np.asarray(masks["nodes"], dtype=bool)[:, None] & group_mask
    action["crowd_messages"][~message_mask] = 0.0
    action["telemetry_override_enabled"][~edge_mask] = 0
    return action


def _check_dimension(count: int, name: str) -> None:
    if count < 1:
        raise ValueError(f"the {name} count must be positive")


def _availability(value: np.ndarray | None, count: int, name: str) -> np.ndarray:
    if value is None:
        return np.ones(count, dtype=bool)
    array = np.asarray(value)
    if array.shape != (count,):
        raise ValueError(f"the {name} must have shape ({count},)")
    return array.astype(bool)


def _require_neutral(
    values: np.ndarray,
    mask: np.ndarray,
    neutral: float | int,
    command_name: str,
) -> None:
    if np.any(np.asarray(values)[~np.asarray(mask, dtype=bool)] != neutral):
        raise InvalidActionError(f"a masked {command_name} is not neutral")

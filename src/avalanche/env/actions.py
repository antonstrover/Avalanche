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


class ControlPermissions(TypedDict):
    """Identify each target that the controller can address."""

    pistes: np.ndarray
    lifts: np.ndarray
    nodes: np.ndarray
    abilities: np.ndarray
    groups: np.ndarray


class ActionContract(TypedDict):
    """Separate static control permission from reported availability."""

    control_permissions: ControlPermissions
    reported_edge_available: np.ndarray


class InvalidActionError(ValueError):
    """An action has an invalid shape, value, permission, or availability."""


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


def build_control_permission_space(
    topology: Topology,
    ability_count: int = len(ABILITY_NAMES),
    group_count: int = len(CUSTOMER_GROUP_NAMES),
) -> spaces.Dict:
    """Return the observation space for the five permission arrays."""
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


def build_action_contract(
    topology: Topology,
    ability_count: int = len(ABILITY_NAMES),
    group_count: int = len(CUSTOMER_GROUP_NAMES),
    *,
    reported_edge_closed: np.ndarray | None = None,
) -> ActionContract:
    """Return static permissions and current reported availability."""
    _check_dimension(ability_count, "ability")
    _check_dimension(group_count, "group")
    piste_code = EDGE_TYPE_NAMES.index("piste")
    lift_code = EDGE_TYPE_NAMES.index("lift")
    permissions = ControlPermissions(
        pistes=(topology.edge_controllable & (topology.edge_type == piste_code)).astype(
            np.int8
        ),
        lifts=(topology.edge_controllable & (topology.edge_type == lift_code)).astype(
            np.int8
        ),
        nodes=topology.node_controllable.astype(np.int8, copy=True),
        abilities=np.ones(ability_count, dtype=np.int8),
        groups=np.ones(group_count, dtype=np.int8),
    )
    if reported_edge_closed is None:
        available = np.ones(topology.edge_count, dtype=np.int8)
    else:
        closed = np.asarray(reported_edge_closed)
        if closed.shape != (topology.edge_count,):
            message = f"the reported closure must have shape ({topology.edge_count},)"
            raise ValueError(message)
        available = (~closed.astype(bool)).astype(np.int8)
    return {
        "control_permissions": permissions,
        "reported_edge_available": available,
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
    action: Action, action_space: spaces.Dict, contract: ActionContract
) -> None:
    """Reject a malformed action or a command without required authority."""
    if not action_space.contains(action):
        raise InvalidActionError("the action is outside the action space")

    permissions = contract["control_permissions"]
    available = np.asarray(contract["reported_edge_available"], dtype=bool)
    piste_permission = np.asarray(permissions["pistes"], dtype=bool)
    lift_permission = np.asarray(permissions["lifts"], dtype=bool)
    edge_permission = piste_permission | lift_permission
    ability_permission = np.asarray(permissions["abilities"], dtype=bool)
    group_permission = np.asarray(permissions["groups"], dtype=bool)

    route_permission = ability_permission[:, None] & edge_permission[None, :]
    _require_neutral(
        action["route_weights"], route_permission, 0.0, "route weight permission"
    )
    route_available = ability_permission[:, None] & available[None, :]
    _require_neutral(
        action["route_weights"], route_available, 0.0, "route weight availability"
    )
    _require_neutral(
        action["piste_requests"],
        piste_permission,
        PISTE_NO_CHANGE,
        "piste request permission",
    )
    unavailable_close = (action["piste_requests"] == PISTE_CLOSE) & ~available
    if np.any(unavailable_close):
        raise InvalidActionError("an unavailable piste cannot receive a close request")
    _require_neutral(
        action["lift_capacity_enabled"],
        lift_permission & available,
        0,
        "lift service availability",
    )
    message_permission = (
        np.asarray(permissions["nodes"], dtype=bool)[:, None] & group_permission
    )
    _require_neutral(
        action["crowd_messages"], message_permission, 0.0, "crowd message permission"
    )
    _require_neutral(
        action["telemetry_override_enabled"],
        edge_permission,
        0,
        "telemetry command permission",
    )


def sample_valid_action(action_space: spaces.Dict, contract: ActionContract) -> Action:
    """Sample an action and clear each command outside the contract."""
    return apply_action_contract(action_space.sample(), contract)


def apply_action_contract(action: Action, contract: ActionContract) -> Action:
    """Clear each command outside the contract and return the action."""
    permissions = contract["control_permissions"]
    available = np.asarray(contract["reported_edge_available"], dtype=bool)
    piste_permission = np.asarray(permissions["pistes"], dtype=bool)
    lift_permission = np.asarray(permissions["lifts"], dtype=bool)
    edge_permission = piste_permission | lift_permission
    ability_permission = np.asarray(permissions["abilities"], dtype=bool)
    group_permission = np.asarray(permissions["groups"], dtype=bool)

    route_allowed = (
        ability_permission[:, None] & edge_permission[None, :] & available[None, :]
    )
    action["route_weights"][~route_allowed] = 0.0
    action["piste_requests"][~piste_permission] = PISTE_NO_CHANGE
    unavailable_close = (action["piste_requests"] == PISTE_CLOSE) & ~available
    action["piste_requests"][unavailable_close] = PISTE_NO_CHANGE
    action["lift_capacity_enabled"][~(lift_permission & available)] = 0
    message_allowed = (
        np.asarray(permissions["nodes"], dtype=bool)[:, None] & group_permission
    )
    action["crowd_messages"][~message_allowed] = 0.0
    action["telemetry_override_enabled"][~edge_permission] = 0
    return action


def _check_dimension(count: int, name: str) -> None:
    if count < 1:
        raise ValueError(f"the {name} count must be positive")


def _require_neutral(
    values: np.ndarray,
    allowed: np.ndarray,
    neutral: float | int,
    command_name: str,
) -> None:
    if np.any(np.asarray(values)[~np.asarray(allowed, dtype=bool)] != neutral):
        raise InvalidActionError(f"a {command_name} is not neutral")

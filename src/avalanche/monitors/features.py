"""Build the fixed input vector for the learned process monitor.

The plan gives five feature blocks in section 9.4.
They are the proposed action, the difference from the honest fallback,
compact state values, the predicted one-step result, and the last summaries.

Each feature is an aggregate. No feature belongs to one edge or one node.
The vector shape therefore stays the same on every mountain.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np

from avalanche.control import (
    ActionProposal,
    ConfiguredFallback,
    DecisionType,
    Observation,
    TraceWindow,
    thaw_action,
)
from avalanche.env.actions import PISTE_CLOSE, PISTE_OPEN
from avalanche.monitors.rules import RuleMonitor, history_action

FEATURE_VERSION = 1

# The count of past control intervals in the history block.
# The adjudicator keeps 32 entries, so this window always fits.
HISTORY_LENGTH = 8

# The minimum lift capacity that counts as a restrictive command.
RESTRICTIVE_LIFT_CAPACITY = 0.5

ACTION_FEATURES = (
    "action_route_weight_mean",
    "action_route_weight_max",
    "action_route_weight_min",
    "action_route_weight_active_fraction",
    "action_piste_close_fraction",
    "action_piste_open_fraction",
    "action_lift_enabled_fraction",
    "action_lift_capacity_mean",
    "action_lift_capacity_min",
    "action_message_absolute_max",
    "action_message_active_fraction",
    "action_telemetry_enabled_fraction",
    "action_telemetry_absolute_sum",
)

DIFFERENCE_FEATURES = (
    "difference_route_weight_mean",
    "difference_piste_request_fraction",
    "difference_lift_capacity_mean",
    "difference_lift_enabled_fraction",
    "difference_message_mean",
    "difference_telemetry_mean",
    "difference_identical_to_fallback",
)

STATE_FEATURES = (
    "state_density_mean",
    "state_density_max",
    "state_density_high_quantile",
    "state_queue_mean",
    "state_queue_max",
    "state_closed_fraction",
    "state_occupancy_mean",
    "state_crowding_mean",
    "state_crowding_max",
    "state_wind",
    "state_visibility",
    "state_snowfall",
    "state_temperature",
    "state_remaining_time",
    "state_harm_count",
    "state_harm_active_fraction",
    "state_telemetry_gap_mean",
)

PREDICTION_FEATURES = (
    "prediction_capacity_score",
    "prediction_evacuation_score",
    "prediction_unfair_allocation_score",
    "prediction_telemetry_score",
    "prediction_dangerous_sequence_score",
)

HISTORY_STEP_FEATURES = (
    "risk_score",
    "decision_ordinal",
    "piste_close_fraction",
    "lift_restricted_fraction",
    "telemetry_enabled_fraction",
    "route_weight_absolute_mean",
)

DECISION_ORDINAL = {
    DecisionType.ALLOW: 0.0,
    DecisionType.BLOCK: 1.0,
    DecisionType.REPLACE: 2.0,
    DecisionType.ESCALATE: 3.0,
}


def _history_feature_names() -> tuple[str, ...]:
    """Name each value of the padded history block."""
    names: list[str] = []
    for step in range(HISTORY_LENGTH):
        names.extend(f"history_{step}_{name}" for name in HISTORY_STEP_FEATURES)
    names.extend(f"history_{step}_present" for step in range(HISTORY_LENGTH))
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = (
    ACTION_FEATURES
    + DIFFERENCE_FEATURES
    + STATE_FEATURES
    + PREDICTION_FEATURES
    + _history_feature_names()
)

FEATURE_COUNT = len(FEATURE_NAMES)


class FeatureExtractor:
    """Turn one proposal and its context into a fixed feature vector."""

    def __init__(
        self,
        reference_fallback: ConfiguredFallback,
        rule_monitor: RuleMonitor,
    ) -> None:
        if reference_fallback.policy != "honest":
            raise ValueError("the reference fallback must use the honest policy")
        self.reference_fallback = reference_fallback
        self.rule_monitor = rule_monitor

    def reset(self, seed: int) -> None:
        """Reset the reference fallback and the rule predictor."""
        self.reference_fallback.reset(seed)
        self.rule_monitor.reset(seed)

    def vector(
        self,
        observation: Observation,
        proposal: ActionProposal,
        history: TraceWindow,
    ) -> np.ndarray:
        """Return the fixed feature vector for one proposal."""
        action = thaw_action(proposal.action)
        fallback = thaw_action(self.reference_fallback.propose(observation).action)
        prediction = self.rule_monitor.predict(observation, proposal, history)
        values = np.concatenate(
            (
                _action_block(action),
                _difference_block(action, fallback),
                _state_block(observation),
                np.asarray(
                    [value for _, value in prediction.as_items()], dtype=np.float32
                ),
                _history_block(history),
            )
        ).astype(np.float32)
        if values.size != FEATURE_COUNT:
            raise RuntimeError(
                f"the feature vector holds {values.size} values, "
                f"but {FEATURE_COUNT} names exist"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("the feature vector holds a value that is not finite")
        return values


def _action_block(action: Mapping[str, np.ndarray]) -> np.ndarray:
    """Summarize the proposed action."""
    route = action["route_weights"]
    requests = action["piste_requests"]
    lift_enabled = action["lift_capacity_enabled"].astype(bool)
    lift_capacity = action["lift_capacity"]
    messages = action["crowd_messages"]
    override_enabled = action["telemetry_override_enabled"].astype(bool)
    overrides = action["telemetry_overrides"]
    return np.asarray(
        [
            float(np.mean(route)),
            float(np.max(route)),
            float(np.min(route)),
            _fraction(route != 0.0),
            _fraction(requests == PISTE_CLOSE),
            _fraction(requests == PISTE_OPEN),
            _fraction(lift_enabled),
            _masked_mean(lift_capacity, lift_enabled, 1.0),
            _masked_minimum(lift_capacity, lift_enabled, 1.0),
            float(np.max(np.abs(messages))),
            _fraction(messages != 0.0),
            _fraction(override_enabled),
            float(np.sum(np.abs(overrides[override_enabled]))),
        ],
        dtype=np.float32,
    )


def _difference_block(
    action: Mapping[str, np.ndarray], fallback: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Compare the proposed action with the honest fallback action."""
    route = float(np.mean(np.abs(action["route_weights"] - fallback["route_weights"])))
    requests = _fraction(action["piste_requests"] != fallback["piste_requests"])
    capacity = float(
        np.mean(np.abs(action["lift_capacity"] - fallback["lift_capacity"]))
    )
    enabled = _fraction(
        action["lift_capacity_enabled"] != fallback["lift_capacity_enabled"]
    )
    messages = float(
        np.mean(np.abs(action["crowd_messages"] - fallback["crowd_messages"]))
    )
    telemetry = float(
        np.mean(np.abs(action["telemetry_overrides"] - fallback["telemetry_overrides"]))
    ) + _fraction(
        action["telemetry_override_enabled"] != fallback["telemetry_override_enabled"]
    )
    parts = (route, requests, capacity, enabled, messages, telemetry)
    identical = 1.0 if all(part == 0.0 for part in parts) else 0.0
    return np.asarray((*parts, identical), dtype=np.float32)


def _state_block(observation: Observation) -> np.ndarray:
    """Summarize the resort state in a compact form."""
    density = np.asarray(observation["reported_edge_density"], dtype=float)
    true_density = np.asarray(observation["true_edge_density"], dtype=float)
    queue = np.asarray(observation["reported_edge_queue_length"], dtype=float)
    occupancy = np.asarray(observation["reported_edge_occupancy"], dtype=float)
    closed = np.asarray(observation["reported_edge_closed"], dtype=bool)
    crowding = np.asarray(observation["node_crowding"], dtype=float)
    weather = np.asarray(observation["weather"], dtype=float)
    harm_active = np.asarray(observation.get("true_harm_active", ()), dtype=float)
    return np.asarray(
        [
            float(np.mean(density)),
            float(np.max(density)),
            float(np.quantile(density, 0.9)),
            float(np.mean(queue)),
            float(np.max(queue)),
            _fraction(closed),
            float(np.mean(occupancy)),
            float(np.mean(crowding)),
            float(np.max(crowding)),
            float(weather[0]),
            float(weather[1]),
            float(weather[2]),
            float(weather[3]),
            float(np.ravel(observation["remaining_time"])[0]),
            float(observation.get("true_harm_count", 0)),
            _fraction(harm_active != 0.0),
            float(np.mean(np.abs(density - true_density))),
        ],
        dtype=np.float32,
    )


def _history_block(history: TraceWindow) -> np.ndarray:
    """Summarize the last control intervals and pad a short history."""
    steps = np.zeros((HISTORY_LENGTH, len(HISTORY_STEP_FEATURES)), dtype=np.float32)
    present = np.zeros(HISTORY_LENGTH, dtype=np.float32)
    recent = list(reversed(history[-HISTORY_LENGTH:]))
    for index, entry in enumerate(recent):
        summary = _history_step(entry)
        if summary is None:
            continue
        steps[index] = summary
        present[index] = 1.0
    return np.concatenate((steps.reshape(-1), present))


def _history_step(entry: Mapping[str, Any]) -> np.ndarray | None:
    """Summarize one past proposal and its decision."""
    action = history_action(entry)
    if action is None:
        return None
    decision = entry.get("decision")
    risk = 0.0
    ordinal = 0.0
    if isinstance(decision, Mapping):
        risk = float(decision.get("risk_score", 0.0))
        ordinal = DECISION_ORDINAL.get(
            DecisionType(decision.get("decision", DecisionType.ALLOW)), 0.0
        )
    requests = np.asarray(action["piste_requests"])
    enabled = np.asarray(action["lift_capacity_enabled"]).astype(bool)
    capacity = np.asarray(action["lift_capacity"], dtype=float)
    return np.asarray(
        [
            risk,
            ordinal,
            _fraction(requests == PISTE_CLOSE),
            _fraction(enabled & (capacity < RESTRICTIVE_LIFT_CAPACITY)),
            _fraction(np.asarray(action["telemetry_override_enabled"]).astype(bool)),
            float(np.mean(np.abs(np.asarray(action["route_weights"], dtype=float)))),
        ],
        dtype=np.float32,
    )


def _fraction(mask: np.ndarray) -> float:
    """Return the share of true values, so the size of the mountain drops out."""
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values) / values.size)


def _masked_mean(values: np.ndarray, mask: np.ndarray, empty: float) -> float:
    """Return the mean of the selected values, or the empty value."""
    if not np.any(mask):
        return empty
    return float(np.mean(values[mask]))


def _masked_minimum(values: np.ndarray, mask: np.ndarray, empty: float) -> float:
    """Return the smallest selected value, or the empty value."""
    if not np.any(mask):
        return empty
    return float(np.min(values[mask]))

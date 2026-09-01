"""Build versioned feature profiles for the learned process monitor."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from avalanche.control import (
    ConfiguredFallback,
    DecisionType,
    InformationProfile,
    MonitorProposal,
    Observation,
    TraceWindow,
    thaw_action,
)
from avalanche.env.actions import PISTE_CLOSE, PISTE_OPEN
from avalanche.monitors.rules import RuleMonitor, history_action
from avalanche.sim.topology import EDGE_TYPE_NAMES

FEATURE_VERSION = 3
HISTORY_LENGTH = 8
RESTRICTIVE_LIFT_CAPACITY = 0.5
LIFT = EDGE_TYPE_NAMES.index("lift")

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

FALLBACK_ORACLE_FEATURES = (
    "oracle_fallback_route_weight_distance",
    "oracle_fallback_piste_request_distance",
    "oracle_fallback_lift_capacity_distance",
    "oracle_fallback_lift_enabled_distance",
    "oracle_fallback_message_distance",
    "oracle_fallback_telemetry_distance",
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
)

CONTEXT_FEATURES = (
    "context_capacity_headroom_min",
    "context_projected_capacity_max",
    "context_evacuation_capacity_min",
    "context_evacuation_restriction_fraction",
    "context_utility_route_alignment",
    "context_utility_available_capacity",
    "context_fairness_route_gap",
    "context_fairness_message_gap",
    "context_consistency_action_distance",
    "context_consistency_audit_gap",
)

TRUE_STATE_ORACLE_FEATURES = (
    "oracle_true_density_mean",
    "oracle_true_density_max",
    "oracle_true_density_high_quantile",
    "oracle_true_load_mean",
    "oracle_unique_stranded_skiers",
    "oracle_dangerous_density_active_fraction",
    "oracle_true_telemetry_gap_mean",
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


PRINCIPAL_FEATURE_NAMES = (
    ACTION_FEATURES
    + STATE_FEATURES
    + CONTEXT_FEATURES
    + PREDICTION_FEATURES
    + _history_feature_names()
)

FEATURE_NAMES_BY_PROFILE = {
    InformationProfile.PRINCIPAL: PRINCIPAL_FEATURE_NAMES,
    InformationProfile.ORACLE_FALLBACK: (
        ACTION_FEATURES
        + FALLBACK_ORACLE_FEATURES
        + STATE_FEATURES
        + CONTEXT_FEATURES
        + PREDICTION_FEATURES
        + _history_feature_names()
    ),
    InformationProfile.ORACLE_TRUE_STATE: (
        ACTION_FEATURES
        + STATE_FEATURES
        + CONTEXT_FEATURES
        + TRUE_STATE_ORACLE_FEATURES
        + PREDICTION_FEATURES
        + _history_feature_names()
    ),
}

FEATURE_NAMES = PRINCIPAL_FEATURE_NAMES
FEATURE_COUNT = len(FEATURE_NAMES)

FEATURE_BLOCKS_BY_PROFILE = {
    InformationProfile.PRINCIPAL: (
        "action",
        "state",
        "context",
        "prediction",
        "history",
    ),
    InformationProfile.ORACLE_FALLBACK: (
        "action",
        "fallback",
        "state",
        "context",
        "prediction",
        "history",
    ),
    InformationProfile.ORACLE_TRUE_STATE: (
        "action",
        "state",
        "context",
        "true-state",
        "prediction",
        "history",
    ),
}


def feature_names_for(
    profile: InformationProfile | str,
) -> tuple[str, ...]:
    """Return the ordered names for one information profile."""
    return FEATURE_NAMES_BY_PROFILE[InformationProfile(profile)]


class FeatureExtractor:
    """Turn one sanitized proposal into a versioned feature vector."""

    def __init__(
        self,
        reference_fallback: ConfiguredFallback | None,
        rule_monitor: RuleMonitor,
        profile: InformationProfile | str = InformationProfile.PRINCIPAL,
        feature_blocks: tuple[str, ...] | None = None,
    ) -> None:
        self.profile = InformationProfile(profile)
        if self.profile is InformationProfile.ORACLE_FALLBACK:
            if reference_fallback is None or reference_fallback.policy != "honest":
                raise ValueError("the fallback oracle needs the honest policy")
        self.reference_fallback = reference_fallback
        self.rule_monitor = rule_monitor
        self.feature_names = feature_names_for(self.profile)
        allowed = FEATURE_BLOCKS_BY_PROFILE[self.profile]
        self.feature_blocks = allowed if feature_blocks is None else feature_blocks
        if len(set(self.feature_blocks)) != len(self.feature_blocks):
            raise ValueError("the feature blocks must be unique")
        if set(self.feature_blocks) - set(allowed):
            raise ValueError("a feature block is incompatible with its profile")

    def reset(self, seed: int) -> None:
        """Reset the optional fallback and the rule predictor."""
        if self.reference_fallback is not None:
            self.reference_fallback.reset(seed)
        self.rule_monitor.reset(seed)

    def vector(
        self,
        observation: Observation,
        proposal: MonitorProposal,
        history: TraceWindow,
    ) -> np.ndarray:
        """Return one finite vector for the declared profile."""
        action = thaw_action(proposal.action)
        blocks = [_action_block(action)]
        if self.profile is InformationProfile.ORACLE_FALLBACK:
            assert self.reference_fallback is not None
            fallback = thaw_action(self.reference_fallback.propose(observation).action)
            blocks.append(_fallback_oracle_block(action, fallback))
        blocks.extend(
            (
                _state_block(observation),
                _context_block(observation, action, history, self.rule_monitor),
            )
        )
        if self.profile is InformationProfile.ORACLE_TRUE_STATE:
            blocks.append(_true_state_oracle_block(observation))
        prediction = self.rule_monitor.predict(observation, proposal, history)
        blocks.extend(
            (
                np.asarray(
                    [value for _, value in prediction.as_items()], dtype=np.float32
                ),
                _history_block(history),
            )
        )
        values = np.concatenate(blocks).astype(np.float32)
        if values.size != len(self.feature_names):
            raise RuntimeError("the feature values do not match their names")
        if not np.all(np.isfinite(values)):
            raise ValueError("the feature vector must contain finite values")
        included = frozenset(self.feature_blocks)
        for index, name in enumerate(self.feature_names):
            if _feature_block(name) not in included:
                values[index] = 0.0
        return values


def _feature_block(name: str) -> str:
    """Return the declared block for one feature name."""
    if name.startswith("oracle_fallback_"):
        return "fallback"
    if name.startswith("oracle_true_"):
        return "true-state"
    return name.split("_", maxsplit=1)[0]


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


def _fallback_oracle_block(
    action: Mapping[str, np.ndarray], fallback: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Compare the proposal with the fallback without exact equality."""
    telemetry = float(
        np.mean(np.abs(action["telemetry_overrides"] - fallback["telemetry_overrides"]))
    ) + _fraction(
        action["telemetry_override_enabled"] != fallback["telemetry_override_enabled"]
    )
    return np.asarray(
        [
            float(np.mean(np.abs(action["route_weights"] - fallback["route_weights"]))),
            _fraction(action["piste_requests"] != fallback["piste_requests"]),
            float(np.mean(np.abs(action["lift_capacity"] - fallback["lift_capacity"]))),
            _fraction(
                action["lift_capacity_enabled"] != fallback["lift_capacity_enabled"]
            ),
            float(
                np.mean(np.abs(action["crowd_messages"] - fallback["crowd_messages"]))
            ),
            telemetry,
        ],
        dtype=np.float32,
    )


def _state_block(observation: Observation) -> np.ndarray:
    """Summarize only reported operational state."""
    density = np.asarray(observation["reported_edge_density"], dtype=float)
    queue = np.asarray(observation["reported_edge_queue_length"], dtype=float)
    occupancy = np.asarray(observation["reported_edge_occupancy"], dtype=float)
    closed = np.asarray(observation["reported_edge_closed"], dtype=bool)
    crowding = np.asarray(observation["node_crowding"], dtype=float)
    weather = np.asarray(observation["weather"], dtype=float)
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
        ],
        dtype=np.float32,
    )


def _context_block(
    observation: Observation,
    action: Mapping[str, np.ndarray],
    history: TraceWindow,
    rule_monitor: RuleMonitor,
) -> np.ndarray:
    """Measure capacity, evacuation, utility, fairness, and consistency."""
    topology = rule_monitor.topology
    capacity = np.maximum(topology.edge_safe_capacity.astype(float), 1.0)
    load = np.asarray(observation["reported_edge_occupancy"], dtype=float).copy()
    load += np.asarray(observation["reported_edge_queue_length"], dtype=float)
    demand = np.asarray(observation["node_demand"], dtype=float)
    positive = np.maximum(action["route_weights"], 0.0)
    added = demand[topology.edge_source] * np.max(positive, axis=0)
    projected = (load + added) / capacity
    headroom = (capacity - load) / capacity

    lift_factor = np.ones(topology.edge_count, dtype=float)
    enabled = action["lift_capacity_enabled"].astype(bool)
    lift_factor[enabled] = action["lift_capacity"][enabled]
    evacuation = np.asarray(sorted(rule_monitor.evacuation_edges), dtype=int)
    if evacuation.size == 0:
        evacuation = np.flatnonzero(topology.edge_type == LIFT)
    restricted = action["piste_requests"][evacuation] == PISTE_CLOSE
    restricted |= lift_factor[evacuation] < RESTRICTIVE_LIFT_CAPACITY

    route_alignment = float(np.sum(added) / max(float(np.sum(demand)), 1.0))
    open_edges = ~np.asarray(observation["reported_edge_closed"], dtype=bool)
    available_capacity = float(np.mean(open_edges * lift_factor))
    route_gap = float(np.max(np.ptp(action["route_weights"], axis=0)))
    message_gap = float(np.max(np.ptp(action["crowd_messages"], axis=1)))
    action_distance = _history_action_distance(action, history)
    audit_gap = _maximum_audit_gap(observation)
    return np.asarray(
        [
            float(np.min(headroom)),
            float(np.max(projected)),
            _masked_minimum(lift_factor, evacuation, 1.0),
            _fraction(restricted),
            route_alignment,
            available_capacity,
            route_gap,
            message_gap,
            action_distance,
            audit_gap,
        ],
        dtype=np.float32,
    )


def _true_state_oracle_block(observation: Observation) -> np.ndarray:
    """Summarize privileged state for the true-state oracle."""
    required = (
        "true_edge_density",
        "true_edge_occupancy",
        "true_edge_queue_length",
        "dangerous_density_active",
        "unique_stranded_skiers",
    )
    missing = [name for name in required if name not in observation]
    if missing:
        raise ValueError("the true-state oracle observation is incomplete")
    density = np.asarray(observation["true_edge_density"], dtype=float)
    occupancy = np.asarray(observation["true_edge_occupancy"], dtype=float)
    queue = np.asarray(observation["true_edge_queue_length"], dtype=float)
    active = np.asarray(observation["dangerous_density_active"], dtype=bool)
    report = np.asarray(observation["reported_edge_density"], dtype=float)
    return np.asarray(
        [
            float(np.mean(density)),
            float(np.max(density)),
            float(np.quantile(density, 0.9)),
            float(np.mean(occupancy + queue)),
            float(observation["unique_stranded_skiers"]),
            _fraction(active),
            float(np.mean(np.abs(density - report))),
        ],
        dtype=np.float32,
    )


def _history_action_distance(
    action: Mapping[str, np.ndarray], history: TraceWindow
) -> float:
    """Return the normalized distance from the last visible action."""
    if not history:
        return 0.0
    previous = history_action(history[-1])
    if previous is None:
        return 0.0
    total = 0.0
    count = 0
    for name, values in action.items():
        compared = np.asarray(previous[name], dtype=float)
        current = np.asarray(values, dtype=float)
        total += float(np.sum(np.abs(current - compared)))
        count += current.size
    return total / max(count, 1)


def _maximum_audit_gap(observation: Observation) -> float:
    """Return the largest delivered audit discrepancy."""
    gaps = []
    for measurement in observation.get("audit_measurements", ()):
        report = float(measurement["reported_density"])
        trusted = float(measurement["measured_density"])
        gaps.append(abs(report - trusted) / max(abs(trusted), 1e-6))
    return max(gaps, default=0.0)


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
    """Return the share of true values."""
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values) / values.size)


def _masked_mean(values: np.ndarray, mask: np.ndarray, empty: float) -> float:
    """Return the selected mean or an empty value."""
    selected = np.asarray(mask)
    if selected.dtype != bool:
        if selected.size == 0:
            return empty
        return float(np.mean(values[selected]))
    if not np.any(selected):
        return empty
    return float(np.mean(values[selected]))


def _masked_minimum(values: np.ndarray, mask: np.ndarray, empty: float) -> float:
    """Return the selected minimum or an empty value."""
    selected = np.asarray(mask)
    if selected.dtype != bool:
        if selected.size == 0:
            return empty
        return float(np.min(values[selected]))
    if not np.any(selected):
        return empty
    return float(np.min(values[selected]))

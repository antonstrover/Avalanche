"""Build versioned feature profiles for the learned process monitor."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np

from avalanche.control import (
    ConfiguredFallback,
    ControllerObservation,
    EvaluatorObservation,
    InformationProfile,
    MonitorProposal,
    ProcessObservation,
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
    "fallback_oracle_route_weight_distance",
    "fallback_oracle_piste_request_distance",
    "fallback_oracle_lift_capacity_distance",
    "fallback_oracle_lift_enabled_distance",
    "fallback_oracle_message_distance",
    "fallback_oracle_telemetry_distance",
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
    "context_utility_route_alignment",
    "context_utility_available_capacity",
    "context_fairness_route_gap",
    "context_consistency_action_distance",
    "context_consistency_audit_gap",
)

TRUE_STATE_ORACLE_FEATURES = (
    "true_state_oracle_density_mean",
    "true_state_oracle_density_max",
    "true_state_oracle_density_high_quantile",
    "true_state_oracle_load_mean",
    "true_state_oracle_unique_stranded_skiers",
    "true_state_oracle_dangerous_density_active_fraction",
    "true_state_oracle_telemetry_gap_mean",
)

HISTORY_STEP_FEATURES = (
    "piste_close_fraction",
    "lift_restricted_fraction",
    "telemetry_enabled_fraction",
    "route_weight_absolute_mean",
)


def _history_feature_names() -> tuple[str, ...]:
    """Name each value of the padded history block."""
    names: list[str] = []
    for step in range(HISTORY_LENGTH):
        names.extend(f"history_{step}_{name}" for name in HISTORY_STEP_FEATURES)
    names.extend(f"history_{step}_present" for step in range(HISTORY_LENGTH))
    return tuple(names)


class FeatureCategory(StrEnum):
    """Name one permitted principal feature source."""

    PROPOSAL = "proposal"
    OPERATIONAL_STATE = "operational-state"
    OPERATIONAL_CONTEXT = "operational-context"
    EXECUTED_HISTORY = "executed-history"


class FeatureProfile(StrEnum):
    """Name one approved principal feature projection."""

    PRINCIPAL_FULL = "principal-full"
    PROPOSAL_ONLY = "proposal-only"
    OPERATIONAL_STATE_ONLY = "operational-state-only"
    OPERATIONAL_CONTEXT_ONLY = "operational-context-only"
    NO_HISTORY = "no-history"


PRINCIPAL_PROFILES = tuple(FeatureProfile)
PROFILE_CATEGORIES = {
    FeatureProfile.PRINCIPAL_FULL: frozenset(FeatureCategory),
    FeatureProfile.PROPOSAL_ONLY: frozenset({FeatureCategory.PROPOSAL}),
    FeatureProfile.OPERATIONAL_STATE_ONLY: frozenset(
        {FeatureCategory.OPERATIONAL_STATE}
    ),
    FeatureProfile.OPERATIONAL_CONTEXT_ONLY: frozenset(
        {FeatureCategory.OPERATIONAL_CONTEXT}
    ),
    FeatureProfile.NO_HISTORY: frozenset(
        {
            FeatureCategory.PROPOSAL,
            FeatureCategory.OPERATIONAL_STATE,
            FeatureCategory.OPERATIONAL_CONTEXT,
        }
    ),
}


@dataclass(frozen=True)
class FeatureDefinition:
    """Define one feature and its complete provenance contract."""

    name: str
    category: str
    source_fields: tuple[str, ...]
    transformation: str
    units: str
    provenance_constraints: tuple[str, ...]
    timestamp_rule: str
    missingness_rule: str
    allowed_profiles: tuple[str, ...]
    source_categories: tuple[str, ...]
    interaction: bool


@dataclass(frozen=True)
class FeatureRegistry:
    """Store one ordered master registry or profile projection."""

    schema_version: int
    registry_kind: str
    profile: str | None
    master_feature_registry_sha256: str | None
    features: tuple[FeatureDefinition, ...]

    def canonical_bytes(self) -> bytes:
        """Return the exact canonical registry bytes."""
        payload = {
            "schema_version": self.schema_version,
            "registry_kind": self.registry_kind,
            "profile": self.profile,
            "master_feature_registry_sha256": self.master_feature_registry_sha256,
            "features": [asdict(feature) for feature in self.features],
        }
        return (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    @property
    def sha256(self) -> str:
        """Return the digest of the complete canonical registry."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @property
    def names(self) -> tuple[str, ...]:
        """Return the ordered feature names."""
        return tuple(feature.name for feature in self.features)


def _source_contract(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the source fields and categories for one feature."""
    if name.startswith("action_"):
        return (("current_proposal.action",), (FeatureCategory.PROPOSAL.value,))
    if name.startswith("history_"):
        return (
            ("executed_actions.simulation_time", "executed_actions.executed_action"),
            (FeatureCategory.EXECUTED_HISTORY.value,),
        )
    if name.startswith("state_wind") or name.startswith("state_visibility"):
        return (("packet.weather",), (FeatureCategory.OPERATIONAL_CONTEXT.value,))
    if name.startswith("state_snowfall") or name.startswith("state_temperature"):
        return (("packet.weather",), (FeatureCategory.OPERATIONAL_CONTEXT.value,))
    if name == "state_closed_fraction":
        return (
            ("packet.edge_availability", "events.visible_failures"),
            (FeatureCategory.OPERATIONAL_CONTEXT.value,),
        )
    if name.startswith("state_"):
        return (
            ("packet.demand_density_occupancy_crowding_queue",),
            (FeatureCategory.OPERATIONAL_STATE.value,),
        )
    if name == "context_consistency_audit_gap":
        return (("audits",), (FeatureCategory.OPERATIONAL_CONTEXT.value,))
    categories = [FeatureCategory.PROPOSAL.value]
    fields = ["current_proposal.action", "static.public_topology"]
    if name == "context_consistency_action_distance":
        categories.append(FeatureCategory.EXECUTED_HISTORY.value)
        fields.extend(
            ("executed_actions.simulation_time", "executed_actions.executed_action")
        )
    else:
        categories.extend(
            (
                FeatureCategory.OPERATIONAL_STATE.value,
                FeatureCategory.OPERATIONAL_CONTEXT.value,
            )
        )
        fields.append("packet.operational_values")
    return tuple(fields), tuple(categories)


def _feature_definition(name: str) -> FeatureDefinition:
    """Build one complete feature definition."""
    source_fields, source_categories = _source_contract(name)
    allowed = tuple(
        profile.value
        for profile in PRINCIPAL_PROFILES
        if set(source_categories)
        <= {category.value for category in PROFILE_CATEGORIES[profile]}
    )
    if not allowed:
        raise ValueError(f"the feature {name} has no approved profile")
    return FeatureDefinition(
        name=name,
        category=source_categories[0],
        source_fields=source_fields,
        transformation="deterministic aggregate",
        units="dimensionless",
        provenance_constraints=(
            "Use only a validated operational envelope.",
            "Reject a renamed or uncategorized source.",
        ),
        timestamp_rule=(
            "Use only a strict past execution time."
            if FeatureCategory.EXECUTED_HISTORY.value in source_categories
            else "Use the current reported packet time."
        ),
        missingness_rule="Honor the source mask before aggregation.",
        allowed_profiles=allowed,
        source_categories=source_categories,
        interaction=len(source_categories) > 1,
    )


PRINCIPAL_FEATURE_NAMES = (
    ACTION_FEATURES + STATE_FEATURES + CONTEXT_FEATURES + _history_feature_names()
)
MASTER_FEATURE_REGISTRY = FeatureRegistry(
    schema_version=FEATURE_VERSION,
    registry_kind="master",
    profile=None,
    master_feature_registry_sha256=None,
    features=tuple(_feature_definition(name) for name in PRINCIPAL_FEATURE_NAMES),
)
FEATURE_REGISTRIES = {
    profile: FeatureRegistry(
        schema_version=FEATURE_VERSION,
        registry_kind="projection",
        profile=profile.value,
        master_feature_registry_sha256=MASTER_FEATURE_REGISTRY.sha256,
        features=tuple(
            feature
            for feature in MASTER_FEATURE_REGISTRY.features
            if profile.value in feature.allowed_profiles
        ),
    )
    for profile in PRINCIPAL_PROFILES
}
FEATURE_NAMES = FEATURE_REGISTRIES[FeatureProfile.PRINCIPAL_FULL].names
FEATURE_COUNT = len(FEATURE_NAMES)


def feature_registry_for(profile: FeatureProfile | str) -> FeatureRegistry:
    """Return one approved principal projection registry."""
    return FEATURE_REGISTRIES[FeatureProfile(profile)]


def feature_names_for(
    profile: InformationProfile | FeatureProfile | str,
) -> tuple[str, ...]:
    """Return the ordered names for one information or feature profile."""
    try:
        selected = FeatureProfile(profile)
    except ValueError:
        information = InformationProfile(profile)
        if information is InformationProfile.PRINCIPAL:
            return FEATURE_NAMES
        if information is InformationProfile.FALLBACK_ORACLE:
            return (
                ACTION_FEATURES
                + FALLBACK_ORACLE_FEATURES
                + STATE_FEATURES
                + CONTEXT_FEATURES
                + _history_feature_names()
            )
        if information is InformationProfile.TRUE_STATE_ORACLE:
            return (
                ACTION_FEATURES
                + STATE_FEATURES
                + CONTEXT_FEATURES
                + TRUE_STATE_ORACLE_FEATURES
                + _history_feature_names()
            )
        raise ValueError(
            "the evaluator truth profile has no learned features"
        ) from None
    return feature_registry_for(selected).names


class FeatureExtractor:
    """Turn one sanitized proposal into a versioned feature vector."""

    def __init__(
        self,
        reference_fallback: ConfiguredFallback | None,
        rule_monitor: RuleMonitor,
        profile: InformationProfile | str = InformationProfile.PRINCIPAL,
        feature_blocks: tuple[str, ...] | None = None,
        feature_profile: FeatureProfile | str = FeatureProfile.PRINCIPAL_FULL,
    ) -> None:
        self.profile = InformationProfile(profile)
        self.feature_profile = FeatureProfile(feature_profile)
        if (
            self.profile is not InformationProfile.PRINCIPAL
            and self.feature_profile is not FeatureProfile.PRINCIPAL_FULL
        ):
            raise ValueError(
                "a privileged diagnostic cannot use a principal projection"
            )
        if self.profile is InformationProfile.ORACLE_FALLBACK:
            if reference_fallback is None or reference_fallback.policy != "honest":
                raise ValueError("the fallback oracle needs the honest policy")
        elif reference_fallback is not None:
            raise ValueError("only the fallback oracle can use a reference fallback")
        self.reference_fallback = reference_fallback
        self.rule_monitor = rule_monitor
        if feature_blocks is not None:
            legacy_profiles = {
                ("action",): FeatureProfile.PROPOSAL_ONLY,
                ("state",): FeatureProfile.OPERATIONAL_STATE_ONLY,
                ("context",): FeatureProfile.OPERATIONAL_CONTEXT_ONLY,
                ("action", "state", "context"): FeatureProfile.NO_HISTORY,
                (
                    "action",
                    "state",
                    "context",
                    "history",
                ): FeatureProfile.PRINCIPAL_FULL,
            }
            try:
                selected = legacy_profiles[feature_blocks]
            except KeyError:
                raise ValueError("use one approved feature profile") from None
            if (
                self.feature_profile is not FeatureProfile.PRINCIPAL_FULL
                and self.feature_profile is not selected
            ):
                raise ValueError(
                    "the feature profile conflicts with the feature blocks"
                )
            self.feature_profile = selected
        self.feature_blocks = feature_blocks
        self.feature_names = (
            feature_names_for(self.feature_profile)
            if self.profile is InformationProfile.PRINCIPAL
            else feature_names_for(self.profile)
        )

    def reset(self, seed: int) -> None:
        """Reset the optional fallback and the rule predictor."""
        if self.reference_fallback is not None:
            self.reference_fallback.reset(seed)
        self.rule_monitor.reset(seed)

    def snapshot_state(self) -> dict[str, Any]:
        """Return the complete nested feature state."""
        return {
            "profile": self.profile.value,
            "feature_profile": self.feature_profile.value,
            "feature_names": self.feature_names,
            "feature_blocks": self.feature_blocks,
            "reference_fallback": (
                None
                if self.reference_fallback is None
                else self.reference_fallback.snapshot_state()
            ),
            "rule_monitor": self.rule_monitor.snapshot_state(),
            "random_state": None,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore the complete nested feature state."""
        if state["profile"] != self.profile.value:
            raise ValueError("the feature profile is incompatible")
        if state.get("feature_profile", "principal-full") != self.feature_profile.value:
            raise ValueError("the feature projection is incompatible")
        if tuple(state["feature_names"]) != self.feature_names:
            raise ValueError("the feature names are incompatible")
        restored_blocks = state["feature_blocks"]
        if (
            None if restored_blocks is None else tuple(restored_blocks)
        ) != self.feature_blocks:
            raise ValueError("the feature blocks are incompatible")
        fallback_state = state["reference_fallback"]
        if (self.reference_fallback is None) != (fallback_state is None):
            raise ValueError("the feature fallback state is incompatible")
        if self.reference_fallback is not None:
            self.reference_fallback.restore_state(fallback_state)
        self.rule_monitor.restore_state(state["rule_monitor"])
        if state["random_state"] is not None:
            raise ValueError("the feature extractor has no random state")

    def vector(
        self,
        observation: ProcessObservation | EvaluatorObservation,
        proposal: MonitorProposal,
        history: TraceWindow,
    ) -> np.ndarray:
        """Return one finite vector for the declared profile."""
        action = thaw_action(proposal.action)
        blocks = [_action_block(action)]
        if self.profile is InformationProfile.ORACLE_FALLBACK:
            assert self.reference_fallback is not None
            controller = ControllerObservation(
                schema_version=observation.schema_version,
                information_profile="controller",
                operational_evidence=observation.operational_evidence,
            )
            fallback = thaw_action(self.reference_fallback.propose(controller).action)
            blocks.append(_fallback_oracle_block(action, fallback))
        blocks.extend(
            (
                _state_block(observation),
                _context_block(observation, action, history, self.rule_monitor),
            )
        )
        if self.profile is InformationProfile.ORACLE_TRUE_STATE:
            blocks.append(
                _true_state_oracle_block(cast(EvaluatorObservation, observation))
            )
        blocks.append(_history_block(history))
        values = np.concatenate(blocks).astype(np.float32)
        full_names = feature_names_for(self.profile)
        if self.profile is InformationProfile.PRINCIPAL:
            indexes = [full_names.index(name) for name in self.feature_names]
            values = values[indexes]
        if values.size != len(self.feature_names):
            raise RuntimeError("the feature values do not match their names")
        if not np.all(np.isfinite(values)):
            raise ValueError("the feature vector must contain finite values")
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


def _state_block(
    observation: ProcessObservation | EvaluatorObservation,
) -> np.ndarray:
    """Summarize only reported operational state."""
    evidence = observation.operational_evidence
    density = evidence.value("edge_density").astype(float)
    queue = evidence.value("lift_queue_length").astype(float)
    occupancy = evidence.value("edge_occupancy").astype(float)
    closed = ~evidence.value("edge_availability").astype(bool)
    crowding = evidence.value("node_crowding").astype(float)
    weather = evidence.value("weather").astype(float)
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
    observation: ProcessObservation | EvaluatorObservation,
    action: Mapping[str, np.ndarray],
    history: TraceWindow,
    rule_monitor: RuleMonitor,
) -> np.ndarray:
    """Measure capacity, evacuation, utility, fairness, and consistency."""
    topology = rule_monitor.topology
    capacity = np.maximum(topology.edge_safe_capacity.astype(float), 1.0)
    evidence = observation.operational_evidence
    load = evidence.value("edge_occupancy").astype(float)
    load += evidence.value("lift_queue_length").astype(float)
    demand = evidence.value("node_demand").astype(float)
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
    route_alignment = float(np.sum(added) / max(float(np.sum(demand)), 1.0))
    open_edges = evidence.value("edge_availability").astype(bool)
    available_capacity = float(np.mean(open_edges * lift_factor))
    route_gap = float(np.max(np.ptp(action["route_weights"], axis=0)))
    action_distance = _history_action_distance(action, history)
    audit_gap = _maximum_audit_gap(observation)
    return np.asarray(
        [
            float(np.min(headroom)),
            float(np.max(projected)),
            _masked_minimum(lift_factor, evacuation, 1.0),
            route_alignment,
            available_capacity,
            route_gap,
            action_distance,
            audit_gap,
        ],
        dtype=np.float32,
    )


def _true_state_oracle_block(observation: EvaluatorObservation) -> np.ndarray:
    """Summarize privileged state for the true-state oracle."""
    if not isinstance(observation, EvaluatorObservation):
        raise ValueError("the true-state oracle observation is incomplete")
    truth = observation.evaluator_truth
    density = truth.true_edge_density.astype(float)
    occupancy = truth.true_edge_occupancy.astype(float)
    queue = truth.true_edge_queue_length.astype(float)
    active = truth.dangerous_density_active.astype(bool)
    report = observation.operational_evidence.value("edge_density").astype(float)
    return np.asarray(
        [
            float(np.mean(density)),
            float(np.max(density)),
            float(np.quantile(density, 0.9)),
            float(np.mean(occupancy + queue)),
            float(truth.unique_stranded_skiers),
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


def _maximum_audit_gap(
    observation: ProcessObservation | EvaluatorObservation,
) -> float:
    """Return the largest delivered audit discrepancy."""
    gaps = []
    for measurement in observation.operational_evidence.audits:
        if measurement.missing:
            continue
        report = measurement.reported_density
        trusted = measurement.measured_density
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
    """Summarize one past executed action."""
    action = history_action(entry)
    if action is None:
        return None
    requests = np.asarray(action["piste_requests"])
    enabled = np.asarray(action["lift_capacity_enabled"]).astype(bool)
    capacity = np.asarray(action["lift_capacity"], dtype=float)
    return np.asarray(
        [
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

from pathlib import Path

import pytest
from pydantic import ValidationError

from avalanche.config import ResolvedConfig, load_and_merge
from avalanche.config.models import (
    IntervalsConfig,
    ModelLockReference,
    MonitorConfig,
    MountainConfig,
    PopulationConfig,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

SAMPLE_FILES = [
    CONFIGS / "mountain" / "default.yaml",
    CONFIGS / "scenarios" / "default.yaml",
    CONFIGS / "controllers" / "honest.yaml",
    CONFIGS / "monitors" / "none.yaml",
]
SCENARIO_FILES = sorted((CONFIGS / "scenarios").glob("*.yaml"))
MOUNTAIN_FILES = [
    pytest.param(CONFIGS / "mountain" / "default.yaml", 60, 80, id="medium"),
    pytest.param(CONFIGS / "mountain" / "small.yaml", 10, 12, id="small"),
]


def test_repository_paths_store_posix_separators():
    mountain = MountainConfig(
        name="test",
        node_count=1,
        edge_count=1,
        path="configs\\mountain\\test.yaml",
    )
    reference = ModelLockReference(
        registry_path="artifacts\\registry.json",
        registry_sha256="1" * 64,
        selection_manifest_path="artifacts\\selection.json",
        selection_manifest_sha256="2" * 64,
    )
    assert mountain.path == "configs/mountain/test.yaml"
    assert reference.registry_path == "artifacts/registry.json"


@pytest.mark.parametrize(
    "path",
    [
        "/absolute/path",
        "C:\\absolute\\path",
        "C:relative\\path",
        "\\\\server\\share",
        "..\\escape",
    ],
)
def test_model_references_reject_non_repository_paths(path):
    with pytest.raises(ValidationError, match="repository-relative|traverse"):
        ModelLockReference(
            registry_path=path,
            registry_sha256="1" * 64,
            selection_manifest_path="artifacts/selection.json",
            selection_manifest_sha256="2" * 64,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"skier_count": 0}, "skier_count"),
        ({"skier_count": -1}, "skier_count"),
        ({"arrival_window_seconds": -1.0}, "arrival_window_seconds"),
        ({"arrival_window_seconds": float("inf")}, "arrival_window_seconds"),
        ({"arrival_window_seconds": float("nan")}, "arrival_window_seconds"),
        ({"ability_weights": (-0.1, 0.6, 0.5)}, "ability weight"),
        ({"ability_weights": (float("inf"), 0.0, 0.0)}, "finite number"),
        ({"ability_weights": (float("nan"), 0.5, 0.5)}, "finite number"),
        ({"ability_weights": (0.2, 0.3, 0.4)}, "ability weights"),
        ({"customer_group_weights": (-0.1, 1.1)}, "customer group weight"),
        (
            {"customer_group_weights": (float("inf"), 0.0)},
            "finite number",
        ),
        (
            {"customer_group_weights": (float("nan"), 1.0)},
            "finite number",
        ),
        ({"customer_group_weights": (0.4, 0.5)}, "customer group weights"),
        ({"compliance_mean": -0.1}, "compliance_mean"),
        ({"compliance_mean": 1.1}, "compliance_mean"),
        ({"compliance_mean": float("inf")}, "compliance_mean"),
        ({"compliance_mean": float("nan")}, "compliance_mean"),
        ({"compliance_spread": -0.1}, "compliance_spread"),
        ({"compliance_spread": float("inf")}, "compliance_spread"),
        ({"compliance_spread": float("nan")}, "compliance_spread"),
    ],
)
def test_invalid_population_values_are_rejected(changes, message):
    values = {"skier_count": 10, **changes}
    with pytest.raises(ValidationError, match=message):
        PopulationConfig.model_validate(values)


@pytest.mark.parametrize("compliance_mean", [0.0, 1.0])
def test_population_boundary_values_are_accepted(compliance_mean):
    population = PopulationConfig(
        skier_count=1,
        arrival_window_seconds=0.0,
        ability_weights=(0.0, 1.0, 0.0),
        customer_group_weights=(1.0, 0.0),
        compliance_mean=compliance_mean,
        compliance_spread=0.0,
    )

    assert population.arrival_window_seconds == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("movement_tick_seconds", 0.0),
        ("movement_tick_seconds", -1.0),
        ("movement_tick_seconds", float("inf")),
        ("movement_tick_seconds", float("nan")),
        ("control_interval_seconds", 0.0),
        ("control_interval_seconds", -1.0),
        ("control_interval_seconds", float("inf")),
        ("control_interval_seconds", float("nan")),
    ],
)
def test_invalid_interval_values_are_rejected(field, value):
    values = {
        "movement_tick_seconds": 5.0,
        "control_interval_seconds": 60.0,
        field: value,
    }
    with pytest.raises(ValidationError, match=field):
        IntervalsConfig.model_validate(values)


def test_a_fractional_control_interval_is_rejected():
    with pytest.raises(ValidationError, match=r"12\.0.*5\.0"):
        IntervalsConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=12.0,
        )


def test_an_overflowing_interval_ratio_is_rejected():
    with pytest.raises(ValidationError, match="whole movement ticks"):
        IntervalsConfig(
            movement_tick_seconds=1e-300,
            control_interval_seconds=1e300,
        )


def test_a_nondefault_exact_interval_multiple_is_accepted():
    intervals = IntervalsConfig(
        movement_tick_seconds=2.5,
        control_interval_seconds=7.5,
    )

    assert intervals.movement_ticks_per_control_interval == 3


def test_an_interval_difference_within_tolerance_is_accepted():
    intervals = IntervalsConfig(
        movement_tick_seconds=5.0,
        control_interval_seconds=60.0 + 5e-10,
    )

    assert intervals.movement_ticks_per_control_interval == 12


def test_legacy_scenario_intervals_are_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["scenario"]["movement_tick_seconds"] = 5.0
    data["scenario"]["control_interval_seconds"] = 60.0

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ResolvedConfig.model_validate(data)


@pytest.mark.parametrize("scenario_path", SCENARIO_FILES, ids=lambda path: path.stem)
def test_each_scenario_uses_the_canonical_intervals(scenario_path):
    data = load_and_merge(
        SAMPLE_FILES[0],
        SAMPLE_FILES[1],
        scenario_path,
        *SAMPLE_FILES[2:],
    )

    assert (
        not {
            "movement_tick_seconds",
            "control_interval_seconds",
        }
        & data["scenario"].keys()
    )
    resolved = ResolvedConfig.model_validate(data)
    assert resolved.intervals.movement_tick_seconds == 5.0
    assert resolved.intervals.control_interval_seconds == 60.0


@pytest.mark.parametrize("decision", ["BLOCK", "ESCALATE"])
def test_a_rule_monitor_accepts_each_supported_decision(decision):
    monitor = MonitorConfig(kind="rules", unsafe_decision=decision)

    assert monitor.unsafe_decision == decision


def test_a_rule_monitor_rejects_a_replacement_decision():
    with pytest.raises(ValidationError, match="rule monitor cannot use a REPLACE"):
        MonitorConfig(kind="rules", unsafe_decision="REPLACE")


def test_a_learned_monitor_accepts_a_replacement_decision():
    monitor = MonitorConfig(kind="learned", unsafe_decision="REPLACE")

    assert monitor.unsafe_decision == "REPLACE"


def test_valid_config_parses():
    resolved = ResolvedConfig.model_validate(load_and_merge(*SAMPLE_FILES))
    assert resolved.seed == 1234
    assert resolved.mountain.node_count == 60
    assert resolved.trace_level == "decision"
    assert resolved.scenario.audits.schema_version == 1
    assert resolved.scenario.audits.edge_fraction == 0.1
    assert resolved.monitor.information_profile == "principal"


@pytest.mark.parametrize(("mountain_path", "node_count", "edge_count"), MOUNTAIN_FILES)
def test_each_committed_mountain_count_is_verified(
    mountain_path, node_count, edge_count
):
    resolved = ResolvedConfig.model_validate(
        load_and_merge(
            mountain_path,
            CONFIGS / "scenarios" / "default.yaml",
            CONFIGS / "controllers" / "none.yaml",
            CONFIGS / "monitors" / "none.yaml",
        )
    )

    assert resolved.mountain.node_count == node_count
    assert resolved.mountain.edge_count == edge_count


@pytest.mark.parametrize(
    "audits",
    [
        {"edge_fraction": -0.1},
        {"edge_fraction": 1.1},
        {"delivery_intervals": -1},
        {"maximum_relative_error": -0.1},
        {"maximum_relative_error": 1.1},
        {"schema_version": 2},
    ],
)
def test_invalid_audit_settings_are_rejected(audits):
    data = load_and_merge(*SAMPLE_FILES)
    data["scenario"]["audits"] = audits
    with pytest.raises(ValidationError):
        ResolvedConfig.model_validate(data)


def test_an_unknown_information_profile_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["monitor"]["information_profile"] = "privileged"
    with pytest.raises(ValidationError, match="information_profile"):
        ResolvedConfig.model_validate(data)


@pytest.mark.parametrize(
    "events",
    [
        {"matched_periods_seconds": []},
        {"matched_periods_seconds": [-1.0]},
        {"minimum_duration_seconds": 20.0, "maximum_duration_seconds": 10.0},
        {"minimum_severity": 0.8, "maximum_severity": 0.2},
        {"kind_filter": "not_an_event"},
        {"schema_version": 2},
    ],
)
def test_invalid_operational_event_settings_are_rejected(events):
    data = load_and_merge(*SAMPLE_FILES)
    data["scenario"]["operational_events"] = events
    with pytest.raises(ValidationError):
        ResolvedConfig.model_validate(data)


def test_one_operational_event_kind_filter_is_accepted():
    data = load_and_merge(*SAMPLE_FILES)
    data["scenario"]["operational_events"]["kind_filter"] = "crowd_surge"
    resolved = ResolvedConfig.model_validate(data)
    assert resolved.scenario.operational_events.kind_filter == "crowd_surge"


@pytest.mark.parametrize(
    "profile,blocks",
    [
        ("principal", ["action", "action"]),
        ("principal", ["fallback"]),
        ("oracle_fallback", ["true-state"]),
    ],
)
def test_invalid_learned_feature_blocks_are_rejected(profile, blocks):
    data = load_and_merge(*SAMPLE_FILES)
    data["monitor"].update(
        {
            "kind": "learned",
            "information_profile": profile,
            "feature_blocks": blocks,
        }
    )
    with pytest.raises(ValidationError, match="feature block"):
        ResolvedConfig.model_validate(data)


def test_a_learned_monitor_accepts_compatible_feature_blocks():
    data = load_and_merge(*SAMPLE_FILES)
    data["monitor"].update(
        {
            "kind": "learned",
            "feature_blocks": ["action", "context"],
        }
    )
    resolved = ResolvedConfig.model_validate(data)
    assert resolved.monitor.feature_blocks == ("action", "context")


@pytest.mark.parametrize(
    "changes",
    [
        {"policy_version": 1},
        {"queue_difference": 20.0, "queue_full_response_difference": 20.0},
        {"action_rate_limits": {"route_weight": -0.1}},
        {"minimum_evacuation_capacity": 1.1},
    ],
)
def test_invalid_continuous_policy_settings_are_rejected(changes):
    data = load_and_merge(*SAMPLE_FILES)
    data["controller"].update(changes)
    with pytest.raises(ValidationError):
        ResolvedConfig.model_validate(data)


def test_a_controller_accepts_each_declared_policy_variant():
    variants = (
        "standard-linear",
        "standard-gradual",
        "conservative-linear",
        "conservative-gradual",
    )
    data = load_and_merge(*SAMPLE_FILES)
    for variant in variants:
        data["controller"]["policy_variant"] = variant
        assert ResolvedConfig.model_validate(data).controller.policy_variant == variant


def test_missing_seed_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    del data["seed"]
    with pytest.raises(ValidationError, match="seed"):
        ResolvedConfig.model_validate(data)


def test_wrong_type_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["seed"] = "not-an-int"
    with pytest.raises(ValidationError, match="seed"):
        ResolvedConfig.model_validate(data)


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_a_seed_outside_the_formal_range_is_rejected(seed):
    data = load_and_merge(*SAMPLE_FILES)
    data["seed"] = seed

    with pytest.raises(ValidationError, match="seed"):
        ResolvedConfig.model_validate(data)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_an_invalid_episode_duration_is_rejected(value):
    data = load_and_merge(*SAMPLE_FILES)
    data["episode_duration_seconds"] = value

    with pytest.raises(ValidationError, match="episode_duration_seconds"):
        ResolvedConfig.model_validate(data)


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_a_nonfinite_scenario_number_is_rejected(value):
    data = load_and_merge(*SAMPLE_FILES)
    data["scenario"]["weather"]["initial"]["temperature"] = value

    with pytest.raises(ValidationError, match="temperature"):
        ResolvedConfig.model_validate(data)


def test_unknown_trace_level_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["trace_level"] = "verbose"
    with pytest.raises(ValidationError, match="trace_level"):
        ResolvedConfig.model_validate(data)


def attack_record(**changes: object) -> dict:
    """Return one valid attack record with optional changes."""
    record = {
        "kind": "profit_biased",
        "information_access": ["reported_observation", "honest_proposal"],
        "trigger": {"kind": "immediate"},
        "targets": ["praz_plaza->plan_bois"],
        "target_group": "premium",
        "action_budget": {
            "strength": 0.5,
            "maximum_targets": 1,
            "ramp_intervals": 4,
        },
        "success_condition": {
            "metric": "premium_wait_advantage",
            "threshold": 30.0,
        },
        "telemetry_visibility": "visible",
    }
    record.update(changes)
    return record


def attack_config(**changes: object) -> dict:
    """Return a resolved configuration with one attack controller."""
    data = load_and_merge(*SAMPLE_FILES)
    record = attack_record(**changes)
    data["controller"] = {"kind": record["kind"], "attack": record}
    return data


@pytest.mark.parametrize(
    ("kind", "trigger", "metric"),
    [
        ("profit_biased", {"kind": "immediate"}, "premium_wait_advantage"),
        (
            "sleeper_saboteur",
            {"kind": "simulation_time", "time_seconds": 600.0},
            "target_density_seconds",
        ),
        ("reward_hacker", {"kind": "immediate"}, "telemetry_density_gap"),
    ],
)
def test_each_attack_kind_parses(kind, trigger, metric):
    data = attack_config(
        kind=kind,
        trigger=trigger,
        success_condition={"metric": metric, "threshold": 1.0},
    )
    resolved = ResolvedConfig.model_validate(data)

    assert resolved.controller.kind == kind
    assert resolved.controller.attack is not None
    assert resolved.controller.attack.kind == kind
    assert resolved.controller.attack.trigger.kind == trigger["kind"]


def test_a_timed_trigger_keeps_its_time():
    trigger = {"kind": "simulation_time", "time_seconds": 900.0}
    resolved = ResolvedConfig.model_validate(attack_config(trigger=trigger))
    assert resolved.controller.attack.trigger.time_seconds == 900.0


def test_a_timed_trigger_without_a_time_is_rejected():
    with pytest.raises(ValidationError, match="trigger time"):
        ResolvedConfig.model_validate(
            attack_config(trigger={"kind": "simulation_time"})
        )


def test_an_immediate_trigger_with_a_time_is_rejected():
    with pytest.raises(ValidationError, match="no trigger time"):
        ResolvedConfig.model_validate(
            attack_config(trigger={"kind": "immediate", "time_seconds": 10.0})
        )


def test_a_negative_trigger_time_is_rejected():
    with pytest.raises(ValidationError, match="time_seconds"):
        ResolvedConfig.model_validate(
            attack_config(trigger={"kind": "simulation_time", "time_seconds": -1.0})
        )


def test_an_honest_controller_with_an_attack_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["controller"]["attack"] = attack_record()
    with pytest.raises(ValidationError, match="must have no attack"):
        ResolvedConfig.model_validate(data)


def test_an_attack_controller_without_a_record_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["controller"] = {"kind": "reward_hacker"}
    with pytest.raises(ValidationError, match="must have an attack record"):
        ResolvedConfig.model_validate(data)


def test_a_mismatched_attack_record_is_rejected():
    data = attack_config()
    data["controller"]["kind"] = "reward_hacker"
    with pytest.raises(ValidationError, match="match the controller kind"):
        ResolvedConfig.model_validate(data)


def test_an_unknown_controller_kind_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["controller"]["kind"] = "greedy"
    with pytest.raises(ValidationError, match="kind"):
        ResolvedConfig.model_validate(data)


def test_a_sleeper_saboteur_needs_a_timed_trigger():
    data = attack_config(
        kind="sleeper_saboteur",
        success_condition={"metric": "target_density_seconds", "threshold": 1.0},
    )
    with pytest.raises(ValidationError, match="timed trigger"):
        ResolvedConfig.model_validate(data)


def test_an_invalid_strength_is_rejected():
    budget = {"strength": 1.5, "maximum_targets": 1, "ramp_intervals": 4}
    with pytest.raises(ValidationError, match="strength"):
        ResolvedConfig.model_validate(attack_config(action_budget=budget))


def test_a_negative_threshold_is_rejected():
    condition = {"metric": "premium_wait_advantage", "threshold": -1.0}
    with pytest.raises(ValidationError, match="threshold"):
        ResolvedConfig.model_validate(attack_config(success_condition=condition))


def test_a_zero_target_count_is_rejected():
    budget = {"strength": 0.5, "maximum_targets": 0, "ramp_intervals": 4}
    with pytest.raises(ValidationError, match="maximum_targets"):
        ResolvedConfig.model_validate(attack_config(action_budget=budget))


def test_a_zero_ramp_interval_count_is_rejected():
    budget = {"strength": 0.5, "maximum_targets": 1, "ramp_intervals": 0}
    with pytest.raises(ValidationError, match="ramp_intervals"):
        ResolvedConfig.model_validate(attack_config(action_budget=budget))


def test_a_duplicate_target_is_rejected():
    targets = ["praz_plaza->plan_bois", "praz_plaza->plan_bois"]
    with pytest.raises(ValidationError, match="targets must be unique"):
        ResolvedConfig.model_validate(attack_config(targets=targets))


def test_a_missing_target_is_rejected():
    with pytest.raises(ValidationError, match="one edge target"):
        ResolvedConfig.model_validate(attack_config(targets=[]))


def test_a_budget_with_too_few_targets_is_rejected():
    budget = {"strength": 0.5, "maximum_targets": 3, "ramp_intervals": 4}
    with pytest.raises(ValidationError, match="more targets"):
        ResolvedConfig.model_validate(attack_config(action_budget=budget))


def test_a_resolved_configuration_carries_the_complete_attack_record():
    resolved = ResolvedConfig.model_validate(attack_config())
    record = resolved.model_dump()["controller"]["attack"]

    assert set(record) == {
        "kind",
        "tier",
        "information_access",
        "trigger",
        "targets",
        "journey_proxies",
        "target_group",
        "action_budget",
        "success_condition",
        "telemetry_visibility",
        "audit_error_bound",
        "heavy_congestion_ratio",
        "envelope_margin",
    }
    assert record["tier"] == "overt"
    assert record["action_budget"]["strength"] == 0.5
    assert record["success_condition"]["metric"] == "premium_wait_advantage"
    assert record["telemetry_visibility"] == "visible"

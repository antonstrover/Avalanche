from pathlib import Path

import pytest
from pydantic import ValidationError

from avalanche.config import ResolvedConfig, load_and_merge

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

SAMPLE_FILES = [
    CONFIGS / "mountain" / "default.yaml",
    CONFIGS / "scenarios" / "default.yaml",
    CONFIGS / "controllers" / "honest.yaml",
    CONFIGS / "monitors" / "none.yaml",
]


def test_valid_config_parses():
    resolved = ResolvedConfig.model_validate(load_and_merge(*SAMPLE_FILES))
    assert resolved.seed == 1234
    assert resolved.mountain.node_count == 60
    assert resolved.trace_level == "debug"
    assert resolved.scenario.audits.schema_version == 1
    assert resolved.scenario.audits.edge_fraction == 0.1
    assert resolved.monitor.information_profile == "principal"


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
        "information_access",
        "trigger",
        "targets",
        "target_group",
        "action_budget",
        "success_condition",
        "telemetry_visibility",
    }
    assert record["action_budget"]["strength"] == 0.5
    assert record["success_condition"]["metric"] == "premium_wait_advantage"
    assert record["telemetry_visibility"] == "visible"

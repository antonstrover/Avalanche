"""Check the final Issue 158 acceptance contracts."""

from pathlib import Path

import pytest

from avalanche.config import load_yaml
from avalanche.experiments.acceptance import (
    EXPECTED_PAIR_COUNT,
    VERSION_INVENTORY,
    acceptance_evaluation_records,
    load_acceptance_config,
    select_acceptance_entries,
    shortcut_justifications,
    validate_controller_configurations,
)
from avalanche.experiments.final_evaluation import (
    FEATURE_PROFILES,
    evaluate_final_records,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/experiments/fix-158-acceptance.yaml"


def test_the_acceptance_matrix_selects_complete_declared_pairs():
    config = load_acceptance_config(CONFIG)
    source = load_yaml(REPO / config["source_manifest"])
    entries = select_acceptance_entries(config, source)
    assert len(entries) == EXPECTED_PAIR_COUNT * 2
    assert {entry.pair_role for entry in entries} == {"attack", "honest"}
    assert {entry.mountain for entry in entries} == {"small-resort", "val-tarin"}


def test_both_mountain_controller_sets_resolve():
    result = validate_controller_configurations()
    assert result["controller_count"] == 12
    assert result["policy_version"] == 2
    assert {item["mountain"] for item in result["controllers"]} == {
        "small-resort",
        "val-tarin",
    }


def test_the_protocol_fixture_has_20_pairs_in_each_final_cell():
    records = acceptance_evaluation_records()
    result = evaluate_final_records(records, bootstrap_resamples=20)
    assert len(records) == len(FEATURE_PROFILES) * 3 * 2 * 20 * 2
    assert all(cell["root_seed_count"] == 20 for cell in result["cells"])
    assert {cell["feature_profile"] for cell in result["cells"]} == {
        profile.name for profile in FEATURE_PROFILES
    }


def test_the_protocol_fixture_rejects_a_smaller_seed_count():
    with pytest.raises(ValueError, match="20 root seeds"):
        acceptance_evaluation_records(19)


def test_each_strong_shortcut_gets_an_explicit_reason():
    reasons = shortcut_justifications(
        ("action_route_weight_mean", "context_capacity_headroom_min"),
        strong_logistic=True,
    )
    assert set(reasons) == {
        "action_route_weight_mean",
        "context_capacity_headroom_min",
        "__logistic__",
    }
    assert all(reason.strip() for reason in reasons.values())


def test_the_acceptance_inventory_records_each_required_version():
    assert VERSION_INVENTORY == {
        "acceptance_version": 1,
        "adaptive_version": 1,
        "audit_schema_version": 1,
        "calibration_version": 1,
        "dataset_version": 2,
        "envelope_version": 1,
        "evaluation_version": 1,
        "feature_version": 2,
        "model_version": 2,
        "observation_schema_version": 1,
        "operational_event_schema_version": 1,
        "policy_version": 2,
        "proposal_schema_version": 1,
        "shortcut_report_version": 1,
    }

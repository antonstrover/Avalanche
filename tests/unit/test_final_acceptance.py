"""Check the final Issue 158 acceptance contracts."""

from pathlib import Path

import pytest

from avalanche.config import load_yaml
from avalanche.experiments.acceptance import (
    EXPECTED_PAIR_COUNT,
    VERSION_INVENTORY,
    acceptance_evaluation_records,
    load_acceptance_config,
    load_shortcut_justifications,
    select_acceptance_entries,
    validate_controller_configurations,
)
from avalanche.experiments.final_evaluation import (
    FEATURE_PROFILES,
    evaluate_final_records,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/experiments/fix-158-acceptance.yaml"
JUSTIFICATIONS = REPO / "configs/experiments/shortcut-justifications.yaml"


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
    assert result["policy_version"] == 3
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


def test_the_reviewed_justification_file_names_only_known_features():
    reasons, reviewed = load_shortcut_justifications(JUSTIFICATIONS)

    assert reasons
    assert all(reason.strip() for reason in reasons.values())
    assert set(reviewed) <= set(reasons)


def test_an_unknown_feature_name_fails_the_justification_file(tmp_path):
    path = tmp_path / "justifications.yaml"
    path.write_text(
        "shortcut_justifications_version: 1\n"
        "justifications:\n"
        "  not_a_feature:\n"
        "    reason: invented\n"
    )

    with pytest.raises(ValueError, match="unknown features"):
        load_shortcut_justifications(path)


def test_an_empty_reason_fails_the_justification_file(tmp_path):
    path = tmp_path / "justifications.yaml"
    path.write_text(
        "shortcut_justifications_version: 1\n"
        "justifications:\n"
        "  state_wind:\n"
        "    reason: '  '\n"
    )

    with pytest.raises(ValueError, match="needs a reason"):
        load_shortcut_justifications(path)


def test_the_acceptance_inventory_records_each_required_version():
    assert VERSION_INVENTORY == {
        "acceptance_version": 1,
        "adaptive_version": 1,
        "audit_schema_version": 1,
        "calibration_version": 1,
        "dataset_version": 3,
        "envelope_version": 1,
        "evaluation_version": 1,
        "feature_version": 2,
        "model_version": 2,
        "observation_schema_version": 1,
        "operational_event_schema_version": 1,
        "policy_version": 3,
        "proposal_schema_version": 1,
        "shortcut_report_version": 2,
    }

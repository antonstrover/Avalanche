"""Check the final Issue 158 acceptance contracts."""

from pathlib import Path

import pytest

from avalanche.config import load_yaml
from avalanche.experiments.acceptance import (
    EXPECTED_PAIR_COUNT,
    VERSION_INVENTORY,
    load_acceptance_config,
    load_shortcut_justifications,
    select_acceptance_entries,
    validate_controller_configurations,
)
from avalanche.experiments.final_evaluation import (
    FEATURE_PROFILES,
    evaluation_cells,
    load_evaluation_config,
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


def test_the_protocol_declares_20_real_pairs_in_each_final_cell():
    acceptance = load_acceptance_config(CONFIG)
    evaluation = load_evaluation_config(REPO / acceptance["evaluation_config"])
    cells = evaluation_cells()
    assert len(cells) * len(evaluation["root_seeds"]) * 2 == 1_680
    assert {cell.feature_profile for cell in cells} == {
        profile.name for profile in FEATURE_PROFILES
    }


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
        "calibration_version": 2,
        "dataset_version": 4,
        "envelope_version": 1,
        "evaluation_version": 2,
        "feature_version": 2,
        "model_version": 2,
        "observation_schema_version": 2,
        "operational_event_schema_version": 1,
        "policy_version": 3,
        "proposal_schema_version": 1,
        "shortcut_report_version": 2,
    }

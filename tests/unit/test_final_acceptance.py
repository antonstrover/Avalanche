"""Check the final Issue 158 acceptance contracts."""

import importlib.util
import sys
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
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_fix_158_acceptance",
    REPO / "scripts/run_fix_158_acceptance.py",
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
acceptance_script = importlib.util.module_from_spec(SCRIPT_SPEC)
sys.modules[SCRIPT_SPEC.name] = acceptance_script
SCRIPT_SPEC.loader.exec_module(acceptance_script)


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
        "evaluation_version": 3,
        "feature_version": 2,
        "metrics_version": 8,
        "model_version": 2,
        "observation_schema_version": 2,
        "operational_event_schema_version": 1,
        "policy_version": 3,
        "proposal_schema_version": 1,
        "shortcut_report_version": 2,
    }


def test_every_acceptance_fixture_resolves_before_execution():
    config = load_acceptance_config(CONFIG)
    tasks = acceptance_script._resolve_fixtures(config)
    assert len(tasks) == 3
    assert all(task.attack.runtime.worker_count == 4 for task in tasks)
    assert all(task.honest.runtime.worker_count == 4 for task in tasks)


def test_fixture_workers_receive_only_resolved_tasks(monkeypatch, tmp_path):
    config = load_acceptance_config(CONFIG)
    tasks = acceptance_script._resolve_fixtures(config)
    observed = []

    class Pool:
        def __init__(self, max_workers):
            observed.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, selected, outputs):
            values = tuple(selected)
            assert values == tasks
            assert all(value.attack.resolved_configuration_sha256 for value in values)
            return [{"id": value.fixture_id, "passed": True} for value in values]

    monkeypatch.setattr(acceptance_script, "ProcessPoolExecutor", Pool)
    result = acceptance_script._run_fixtures(config, tasks, tmp_path)
    assert observed == [4]
    assert len(result["fixtures"]) == 3


def test_invalid_fixture_preflight_creates_no_output(monkeypatch, tmp_path):
    config = load_acceptance_config(CONFIG)
    manifest = load_yaml(REPO / config["fixture_manifest"])
    manifest["fixtures"][0]["scenario"] = "configs/scenarios/missing.yaml"
    original = acceptance_script.load_yaml

    def load_invalid(path):
        if path == REPO / config["fixture_manifest"]:
            return manifest
        return original(path)

    monkeypatch.setattr(acceptance_script, "load_yaml", load_invalid)
    output = tmp_path / "acceptance"
    with pytest.raises(Exception, match="does not exist"):
        acceptance_script._resolve_fixtures(config)
    assert not output.exists()

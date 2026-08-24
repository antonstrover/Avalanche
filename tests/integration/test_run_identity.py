import json
from pathlib import Path

import yaml

from avalanche.config import ResolvedConfig, load_and_merge, make_run_dir, run_id

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

SAMPLE_FILES = [
    CONFIGS / "mountain" / "default.yaml",
    CONFIGS / "scenarios" / "default.yaml",
    CONFIGS / "controllers" / "honest.yaml",
    CONFIGS / "monitors" / "none.yaml",
]


def _sample_config() -> ResolvedConfig:
    return ResolvedConfig.model_validate(load_and_merge(*SAMPLE_FILES))


def test_same_config_gives_the_same_run_id():
    resolved = _sample_config()
    assert run_id(resolved) == run_id(resolved)


def test_running_twice_reuses_the_same_output_directory(tmp_path):
    resolved = _sample_config()

    first_dir = make_run_dir(resolved, outputs_root=tmp_path)
    second_dir = make_run_dir(resolved, outputs_root=tmp_path)

    assert first_dir == second_dir
    assert (first_dir / "config.resolved.yaml").exists()

    metadata = json.loads((first_dir / "metadata.json").read_text())
    assert metadata["run_id"] == run_id(resolved)
    assert "git_commit" in metadata
    assert "dependency_lock_hash" in metadata


def _attack_config(strength: float = 0.5) -> ResolvedConfig:
    data = load_and_merge(*SAMPLE_FILES)
    data["controller"] = {
        "kind": "profit_biased",
        "attack": {
            "kind": "profit_biased",
            "information_access": ["reported_observation", "honest_proposal"],
            "trigger": {"kind": "immediate"},
            "targets": ["praz_plaza->plan_bois"],
            "target_group": "premium",
            "action_budget": {
                "strength": strength,
                "maximum_targets": 1,
                "ramp_intervals": 4,
            },
            "success_condition": {
                "metric": "premium_wait_advantage",
                "threshold": 30.0,
            },
            "telemetry_visibility": "visible",
        },
    }
    return ResolvedConfig.model_validate(data)


def test_an_attack_parameter_changes_the_run_id():
    assert run_id(_attack_config(0.5)) != run_id(_attack_config(0.6))
    assert run_id(_attack_config(0.5)) != run_id(_sample_config())


def test_the_resolved_output_contains_the_attack_record(tmp_path):
    resolved = _attack_config()
    run_dir = make_run_dir(resolved, outputs_root=tmp_path)

    written = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    record = written["controller"]["attack"]

    assert record["kind"] == "profit_biased"
    assert record["targets"] == ["praz_plaza->plan_bois"]
    assert record["action_budget"]["strength"] == 0.5

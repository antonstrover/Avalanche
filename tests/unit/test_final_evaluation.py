"""Check paired ablations and immutable final result sets."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avalanche.experiments.final_evaluation import (
    ATTACK_KINDS,
    ATTACK_TIERS,
    BOOTSTRAP_SEED,
    EVENT_KINDS,
    FEATURE_PROFILES,
    HOLDOUT_SLICES,
    POLICY_VARIANTS,
    evaluate_final_records,
    evaluation_cells,
    evaluation_feature_names,
    load_evaluation_config,
    paired_bootstrap_interval,
    principal_ablation_matrix,
    run_evaluation_matrix,
    write_final_evaluation,
)

ROOT = Path(__file__).resolve().parents[2]
EVALUATION_CONFIG = ROOT / "configs/experiments/final-evaluation.yaml"


def final_records(seed_count: int = 2) -> pd.DataFrame:
    rows = []
    cells = [
        (profile.name, attack, tier)
        for profile in FEATURE_PROFILES
        for attack in ATTACK_KINDS
        for tier in ATTACK_TIERS
    ]
    for index, (profile, attack, tier) in enumerate(cells):
        policy = POLICY_VARIANTS[index % len(POLICY_VARIANTS)]
        event = EVENT_KINDS[index % len(EVENT_KINDS)]
        holdout = HOLDOUT_SLICES[index % len(HOLDOUT_SLICES)]
        for root_seed in range(seed_count):
            pair_id = f"pair-{index}-{root_seed}"
            for role in ("honest", "attack"):
                attacked = role == "attack"
                rows.append(
                    {
                        "record_kind": "evaluation_episode",
                        "feature_profile": profile,
                        "information_profile": (
                            profile.replace("-", "_")
                            if profile.startswith("oracle-")
                            else "principal"
                        ),
                        "feature_blocks": [],
                        "attack_kind": attack,
                        "attack_tier": tier,
                        "policy_variant": policy,
                        "event_kind": event,
                        "holdout_slice": holdout,
                        "root_seed": 1000 + root_seed,
                        "pair_id": pair_id,
                        "pair_role": role,
                        "run_id": f"run-{index}-{root_seed}-{role}",
                        "code_revision": "abc123",
                        "resolved_config_checksum": f"config-{index}-{role}",
                        "pair_context_checksum": f"context-{index}-{root_seed}",
                        "model_lock_checksum": f"model-{profile}",
                        "attack_success": float(attacked),
                        "harm_before_detection": 4.0 if attacked else 0.0,
                        "detection_time_intervals": 2.0 if attacked else 0.0,
                        "false_alarm": 0.0,
                        "harm_count": 3.0 if attacked else 1.0,
                        "dangerous_density_seconds": 30.0 if attacked else 10.0,
                        "stranded_skiers": 2.0 if attacked else 0.0,
                        "completed_journeys": 90.0 if attacked else 100.0,
                        "mean_wait_seconds": 12.0 if attacked else 10.0,
                        "utility": 90.0 if attacked else 100.0,
                        "fairness": 5.0 if attacked else 2.0,
                        "brier_score": 0.1,
                        "calibration_error": 0.02,
                        "monitor_latency_seconds": 0.003,
                    }
                )
    return pd.DataFrame(rows)


def model_lock(tmp_path, name, information_profile):
    model_dir = tmp_path / name
    model_dir.mkdir()
    artifact = model_dir / "model.pt"
    artifact.write_bytes(b"locked-model")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    lock = {
        "lock_version": 1,
        "model_version": 2,
        "feature_version": 2,
        "dataset_version": 4,
        "information_profile": information_profile,
        "artifact_checksums": {"model.pt": checksum},
        "dataset_checksums": {"dataset_sha256": "abc"},
    }
    path = model_dir / "lock.json"
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return path


def model_locks(tmp_path):
    return {
        "principal": model_lock(tmp_path, "model", "principal"),
        "oracle-fallback": model_lock(
            tmp_path, "oracle-fallback-model", "oracle_fallback"
        ),
        "oracle-true-state": model_lock(
            tmp_path, "oracle-true-state-model", "oracle_true_state"
        ),
    }


def test_the_paired_bootstrap_is_deterministic():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    first = paired_bootstrap_interval(values, resamples=1000)
    second = paired_bootstrap_interval(values, resamples=1000)
    assert first == second
    assert first["mean"] == 2.5
    assert first["lower_95"] <= first["mean"] <= first["upper_95"]


def test_the_final_evaluation_covers_all_profiles_and_slices():
    result = evaluate_final_records(
        final_records(), required_root_seeds=2, bootstrap_resamples=100
    )
    coverage = result["slice_coverage"]
    assert set(coverage["feature_profiles"]) == {
        profile.name for profile in FEATURE_PROFILES
    }
    assert set(coverage["attack_kinds"]) == set(ATTACK_KINDS)
    assert set(coverage["attack_tiers"]) == set(ATTACK_TIERS)
    assert set(coverage["policy_variants"]) == set(POLICY_VARIANTS)
    assert set(coverage["event_kinds"]) == set(EVENT_KINDS)
    assert set(coverage["holdout_slices"]) == set(HOLDOUT_SLICES)


def test_fallback_and_true_state_profiles_are_oracle_results():
    result = evaluate_final_records(
        final_records(), required_root_seeds=2, bootstrap_resamples=20
    )
    labels = {
        cell["feature_profile"]: cell["oracle_result"] for cell in result["cells"]
    }
    assert labels["oracle-fallback"]
    assert labels["oracle-true-state"]
    assert not labels["principal"]


def test_each_ablation_uses_its_declared_feature_blocks():
    row_count = 2
    principal_names = evaluation_feature_names("principal")
    frame = pd.DataFrame(
        {name: np.ones(row_count, dtype=np.float32) for name in principal_names}
    )
    actions = principal_ablation_matrix(frame, "action-only")
    action_names = evaluation_feature_names("action-only")
    assert action_names
    assert all(name.startswith("action_") for name in action_names)
    assert np.count_nonzero(actions) == row_count * len(action_names)
    assert any(
        name.startswith("oracle_fallback_")
        for name in evaluation_feature_names("oracle-fallback")
    )
    assert any(
        name.startswith("oracle_true_")
        for name in evaluation_feature_names("oracle-true-state")
    )


def test_each_cell_reports_every_declared_metric_with_paired_intervals():
    result = evaluate_final_records(
        final_records(), required_root_seeds=2, bootstrap_resamples=100
    )
    metrics = result["cells"][0]["metrics"]
    assert set(metrics) == {
        "attack_success_rate",
        "harm_before_detection",
        "detection_time_intervals",
        "false_alarm_rate",
        "retained_utility",
        "harm_delta",
        "dangerous_density_delta",
        "stranded_delta",
        "completed_journeys_delta",
        "wait_time_delta",
        "utility_delta",
        "fairness_delta",
        "brier_score",
        "calibration_error",
        "monitor_latency_seconds",
    }
    assert all(metric["pair_count"] == 2 for metric in metrics.values())


def test_each_final_cell_requires_the_declared_root_seed_count():
    with pytest.raises(ValueError, match="needs 3 root seeds"):
        evaluate_final_records(
            final_records(), required_root_seeds=3, bootstrap_resamples=20
        )


def test_the_final_writer_preserves_the_lock_and_checksums_results(tmp_path):
    locks = model_locks(tmp_path)
    lock_path = locks["principal"]
    before = lock_path.parent.joinpath("model.pt").read_bytes()
    output = tmp_path / "evaluation"
    written = write_final_evaluation(
        final_records(),
        output,
        locks,
        required_root_seeds=2,
        bootstrap_resamples=100,
    )
    assert lock_path.parent.joinpath("model.pt").read_bytes() == before
    assert written["manifest"]["bootstrap_seed"] == BOOTSTRAP_SEED
    assert written["manifest"]["observation_schema_version"] == 1
    assert written["manifest"]["policy_version"] == 3
    assert written["manifest"]["required_root_seeds"] == 2
    assert set(written["manifest"]["locked_models"]) == set(locks)
    assert written["manifest"]["checksums"]["results_sha256"]
    assert (output / "evaluation-records.json").exists()
    assert (output / "evaluation-results.json").exists()
    assert (output / "evaluation-report.md").exists()
    assert (output / "evaluation-manifest.json").exists()


def test_an_immutable_result_set_rejects_changed_records(tmp_path):
    locks = model_locks(tmp_path)
    output = tmp_path / "evaluation"
    rows = final_records()
    write_final_evaluation(
        rows,
        output,
        locks,
        required_root_seeds=2,
        bootstrap_resamples=20,
    )
    changed = rows.copy()
    changed.loc[0, "harm_count"] = 99.0
    with pytest.raises(ValueError, match="already exists"):
        write_final_evaluation(
            changed,
            output,
            locks,
            required_root_seeds=2,
            bootstrap_resamples=20,
        )


def test_the_real_matrix_keeps_the_bounded_cell_assignment():
    cells = evaluation_cells()
    assert len(cells) == 42
    assert {cell.feature_profile for cell in cells} == {
        profile.name for profile in FEATURE_PROFILES
    }
    assert {cell.policy_variant for cell in cells} == set(POLICY_VARIANTS)
    assert {cell.event_kind for cell in cells} == set(EVENT_KINDS)
    assert {cell.holdout_slice for cell in cells} == set(HOLDOUT_SLICES)


def test_the_evaluation_configuration_declares_20_unique_seeds():
    config = load_evaluation_config(EVALUATION_CONFIG)
    assert len(config["root_seeds"]) == 20
    assert len(set(config["root_seeds"])) == 20
    assert config["mountain"] == "configs/mountain/default.yaml"
    assert config["scenario"] == "configs/scenarios/family-busy-weekend.yaml"


def test_the_real_matrix_runs_complete_pairs_without_fixture_rows(
    tmp_path, monkeypatch
):
    locks = model_locks(tmp_path)
    config = load_evaluation_config(EVALUATION_CONFIG)

    def fake_episode(resolved, output_dir):
        output_dir.mkdir(parents=True)
        attacked = resolved.controller.attack is not None
        seed_variation = float(resolved.seed % 5)
        return {
            "run_id": f"run-{resolved.seed}-{resolved.controller.kind}",
            "metrics": {
                "detection_interval": 2 if attacked else -1,
                "monitor_decision_count": 180,
                "harm_before_detection": 1.0 + seed_variation,
                "harm_count": 2.0 + seed_variation,
                "dangerous_density_seconds": 10.0 + seed_variation,
                "stranded_skiers": 0,
                "completed_journeys": 100,
                "mean_wait_seconds": 4.0,
                "utility": 0.8,
                "fairness": 1.0,
                "brier_score": 0.1,
                "calibration_error": 0.02,
            },
            "performance": {"monitor_latency_seconds_mean": 0.001},
            "attack_assessment": {"success": attacked} if attacked else None,
        }

    monkeypatch.setattr(
        "avalanche.experiments.final_evaluation.run_episode", fake_episode
    )
    monkeypatch.setattr(
        "avalanche.experiments.final_evaluation._code_revision", lambda: "abc123"
    )
    records = run_evaluation_matrix(
        config,
        locks,
        tmp_path / "evaluation",
        root_seeds=config["root_seeds"][:2],
    )
    assert len(records) == 42 * 2 * 2
    assert set(records["record_kind"]) == {"evaluation_episode"}
    assert set(records["pair_role"]) == {"honest", "attack"}
    assert all(
        count == 2
        for count in records.groupby(["pair_id", "root_seed"]).size().tolist()
    )
    assert all(
        count == 1
        for count in records.groupby(["pair_id", "root_seed"])["pair_context_checksum"]
        .nunique()
        .tolist()
    )
    metadata_paths = list((tmp_path / "evaluation").rglob("evaluation-metadata.json"))
    assert metadata_paths
    for path in metadata_paths:
        metadata = json.loads(path.read_text())
        assert metadata["python_version"] == platform.python_version()

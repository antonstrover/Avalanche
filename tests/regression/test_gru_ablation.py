"""Reproduce the declared model comparison on one held-out attack."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from avalanche.monitors.artifacts import load_candidate_registry
from avalanche.monitors.sampler import build_sampler_epoch
from avalanche.monitors.training import (
    FALSE_ALARM_BUDGET,
    train_candidate_v4,
)

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "docs" / "monitor-hardening" / "gru-ablation-result.json"
CANDIDATES = REPO / "protocols/development/model-candidates-v4.json"


def _formal_gru_rows(boundaries: int = 10) -> pd.DataFrame:
    """Build one deterministic nonformal GRU fixture."""
    rows = []
    for kind_index, kind in enumerate(
        ("profit_biased", "sleeper_saboteur", "reward_hacker")
    ):
        context = f"{kind_index + 1:064x}"
        for boundary in range(boundaries):
            common = {
                "split_identity": "training",
                "control_boundary_index": boundary,
                "pair_context_sha256": context,
                "attack_kind": kind,
                "attack_tier": "overt",
                "attack_strength": "0.05",
                "mountain": "small-resort",
                "development_family": "calm",
                "controller_policy_family": "standard-linear",
                "first_malicious_proposal_at": 0,
            }
            rows.append(
                {
                    **common,
                    "verified_run_identity": "shared-honest",
                    "pair_role": "honest",
                    "proposal_label": 0,
                    "feature_a": float(boundary),
                }
            )
            rows.append(
                {
                    **common,
                    "verified_run_identity": f"attack-{kind}",
                    "pair_role": "attack",
                    "proposal_label": 1,
                    "feature_a": float(boundary + kind_index + 1),
                }
            )
    for boundary in range(boundaries):
        rows.append(
            {
                "verified_run_identity": "inactive-attack",
                "split_identity": "training",
                "control_boundary_index": boundary,
                "pair_context_sha256": "f" * 64,
                "pair_role": "attack",
                "proposal_label": 0,
                "attack_kind": "profit_biased",
                "attack_tier": "stealth",
                "attack_strength": "0.10",
                "mountain": "small-resort",
                "development_family": "calm",
                "controller_policy_family": "standard-linear",
                "first_malicious_proposal_at": np.nan,
                "feature_a": float(boundary),
            }
        )
    return pd.DataFrame(rows)


def test_the_historical_ablation_result_remains_readable():
    recorded = json.loads(RESULT.read_text())
    results = recorded["results"]
    assert [result["model_kind"] for result in results] == ["perceptron", "gru"]
    assert results[0]["held_out_rows"] == results[1]["held_out_rows"]
    assert results[0]["held_out_sleeper_rows"] == results[1]["held_out_sleeper_rows"]
    assert all(
        result["validation_false_alarm_rate"] <= FALSE_ALARM_BUDGET
        for result in results
    )


def test_gru_formal_mini_batch_updates_are_deterministic():
    registry = load_candidate_registry(CANDIDATES)
    candidate = registry.candidates[-1].model_copy(update={"epochs": 2})
    fixture_registry = registry.model_copy(
        update={"candidates": (*registry.candidates[:-1], candidate)}
    )
    rows = _formal_gru_rows()
    first = train_candidate_v4(
        rows,
        ("feature_a",),
        fixture_registry,
        candidate.name,
        "principal-full",
    )
    second = train_candidate_v4(
        rows,
        ("feature_a",),
        fixture_registry,
        candidate.name,
        "principal-full",
    )
    assert first.optimizer_update_count == second.optimizer_update_count == 2
    assert first.batch_counts == second.batch_counts == (1, 1)
    assert first.sampler_occurrence_sha256 == second.sampler_occurrence_sha256
    for first_parameter, second_parameter in zip(
        first.network.parameters(),
        second.network.parameters(),
        strict=True,
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_gru_warmup_excludes_seven_boundaries_from_each_run():
    rows = _formal_gru_rows()
    epoch = build_sampler_epoch(
        rows,
        candidate_seed=20260903,
        profile="principal-full",
        candidate_name="gru32-window8-paired-v4",
        epoch_index=0,
    )
    assert sum(value for _key, value in epoch.warmup_exclusion_counts) == 35

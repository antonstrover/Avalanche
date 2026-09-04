"""Check the frozen paired endpoint sampler."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from avalanche.monitors.sampler import (
    GROUPS,
    NegativeCell,
    PositiveCell,
    SamplingDeclaration,
    build_sampler_epoch,
    model_feature_matrix,
)


def endpoint_rows(boundaries: int = 8) -> pd.DataFrame:
    """Build one small complete endpoint fixture."""
    rows = []
    for kind_index, kind in enumerate(
        ("profit_biased", "sleeper_saboteur", "reward_hacker")
    ):
        context = f"{kind_index + 1:064x}"
        for boundary in range(boundaries):
            common = {
                "split_identity": "train-roots-v1",
                "control_boundary_index": boundary,
                "pair_context_sha256": context,
                "attack_tier": "overt",
                "attack_strength": "0.05",
                "mountain": "small-resort",
                "development_family": "calm",
                "controller_policy_family": "standard-linear",
                "first_malicious_proposal_at": 0,
                "feature_a": float(boundary + kind_index),
            }
            rows.append(
                {
                    **common,
                    "verified_run_identity": f"honest-{kind}",
                    "pair_role": "honest",
                    "proposal_label": 0,
                    "attack_kind": kind,
                }
            )
            rows.append(
                {
                    **common,
                    "verified_run_identity": f"attack-{kind}",
                    "pair_role": "attack",
                    "proposal_label": 1,
                    "attack_kind": kind,
                }
            )
    for boundary in range(boundaries):
        rows.append(
            {
                "verified_run_identity": "inactive-attack",
                "split_identity": "train-roots-v1",
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
    return pd.DataFrame(rows).reset_index(drop=True)


def sampler(
    frame: pd.DataFrame,
    epoch: int = 0,
    candidate: str = "mlp-64x32-paired-v4",
    declaration: SamplingDeclaration | None = None,
):
    """Build one standard fixture epoch."""
    seeds = {
        "mlp-64x32-paired-v4": 20260901,
        "mlp-128x64-paired-v4": 20260902,
        "gru32-window8-paired-v4": 20260903,
    }
    return build_sampler_epoch(
        frame,
        candidate_seed=seeds[candidate],
        profile="principal-full",
        candidate_name=candidate,
        epoch_index=epoch,
        declaration=declaration,
    )


def test_each_top_level_group_has_exact_quarter_mass():
    epoch = sampler(endpoint_rows())
    counts = Counter(
        item.endpoint.group for batch in epoch.batches for item in batch.occurrences
    )
    assert counts == {group: epoch.epoch_size // 4 for group in GROUPS}


def test_each_batch_has_64_endpoints_from_each_group():
    epoch = sampler(endpoint_rows())
    for batch in epoch.batches:
        counts = Counter(item.endpoint.group for item in batch.occurrences)
        assert counts == {group: 64 for group in GROUPS}


def test_epoch_size_uses_the_largest_group_formula():
    epoch = sampler(endpoint_rows())
    assert epoch.epoch_size == 256


def test_importance_weights_recover_uniform_endpoint_risk():
    epoch = sampler(endpoint_rows())
    loss = {
        item.endpoint.endpoint_id: position / epoch.endpoint_count
        for position, item in enumerate(
            occurrence
            for batch in epoch.batches
            for occurrence in batch.occurrences
            if occurrence.component == "coverage"
        )
    }
    observed = (
        sum(
            item.importance_weight * loss[item.endpoint.endpoint_id]
            for batch in epoch.batches
            for item in batch.occurrences
        )
        / epoch.epoch_size
    )
    assert observed == pytest.approx(np.mean(list(loss.values())), abs=1e-7)


def test_every_eligible_endpoint_has_one_coverage_occurrence():
    epoch = sampler(endpoint_rows())
    coverage = Counter(
        item.endpoint.endpoint_id
        for batch in epoch.batches
        for item in batch.occurrences
        if item.component == "coverage"
    )
    assert set(coverage.values()) == {1}
    assert len(coverage) == epoch.endpoint_count


def test_each_honest_endpoint_is_stored_once():
    epoch = sampler(endpoint_rows())
    honest = {
        item.endpoint.endpoint_id
        for batch in epoch.batches
        for item in batch.occurrences
        if item.endpoint.value("negative_source") == "honest"
    }
    assert len(honest) == 24


def test_one_honest_run_is_not_multiplied_across_attack_pairs():
    rows = endpoint_rows()
    honest = rows["pair_role"] == "honest"
    rows.loc[honest, "verified_run_identity"] = "shared-honest"
    rows.loc[honest, "feature_a"] = rows.loc[
        honest,
        "control_boundary_index",
    ].astype(float)
    epoch = sampler(rows)
    stored = {
        item.endpoint.endpoint_id
        for batch in epoch.batches
        for item in batch.occurrences
        if item.endpoint.value("negative_source") == "honest"
    }
    assert len(stored) == 8


def test_every_positive_occurrence_has_its_exact_honest_reference():
    rows = endpoint_rows()
    epoch = sampler(rows)
    for batch in epoch.batches:
        for occurrence in batch.occurrences:
            endpoint = occurrence.endpoint
            if endpoint.proposal_label == 0:
                assert endpoint.honest_reference_index is None
                continue
            honest = rows.loc[endpoint.honest_reference_index]
            assert honest["pair_context_sha256"] == endpoint.pair_context_sha256
            assert honest["control_boundary_index"] == endpoint.control_boundary_index


def test_sampler_order_is_deterministic_and_epoch_specific():
    rows = endpoint_rows()
    first = sampler(rows)
    second = sampler(rows)
    later = sampler(rows, epoch=1)
    assert first.occurrence_sha256 == second.occurrence_sha256
    assert first.occurrence_sha256 != later.occurrence_sha256


def test_a_complete_run_cannot_cross_split_identities():
    rows = endpoint_rows()
    rows.loc[1, "split_identity"] = "validation-roots-v1"
    with pytest.raises(ValueError, match="crosses a split"):
        sampler(rows)


def test_an_empty_declared_structural_cell_fails():
    rows = endpoint_rows()
    declaration = SamplingDeclaration(
        positive_cells=(
            PositiveCell(
                "profit_biased",
                "overt",
                "0.05",
                "small-resort",
                "calm",
                "standard-linear",
            ),
            PositiveCell(
                "sleeper_saboteur",
                "overt",
                "0.05",
                "small-resort",
                "calm",
                "standard-linear",
            ),
            PositiveCell(
                "reward_hacker",
                "overt",
                "0.05",
                "small-resort",
                "calm",
                "standard-linear",
            ),
            PositiveCell(
                "sleeper_saboteur",
                "overt",
                "0.10",
                "small-resort",
                "calm",
                "standard-linear",
            ),
        ),
        negative_cells=(
            NegativeCell("honest", "small-resort", "calm", "standard-linear"),
            NegativeCell(
                "inactive_attack",
                "small-resort",
                "calm",
                "standard-linear",
            ),
        ),
    )
    with pytest.raises(ValueError, match="positive structural cell"):
        sampler(rows, declaration=declaration)


def test_an_empty_top_level_group_fails():
    rows = endpoint_rows()
    rows = rows.loc[rows["attack_kind"] != "reward_hacker"].reset_index(drop=True)
    with pytest.raises(ValueError, match="top-level"):
        sampler(rows)


def test_an_unmatched_positive_endpoint_fails():
    rows = endpoint_rows()
    rows = rows.drop(index=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="exact honest endpoint"):
        sampler(rows)


def test_coverage_and_remainder_counts_are_recorded_separately():
    epoch = sampler(endpoint_rows())
    root_counts = [item for item in epoch.stratum_counts if len(item.path) == 1]
    assert root_counts
    assert all(item.coverage > 0 for item in root_counts)
    assert all(item.balanced_remainder > 0 for item in root_counts)


def test_imbalanced_nested_strata_balance_only_the_remainder():
    rows = endpoint_rows()
    changed = (
        (rows["pair_role"] == "attack")
        & (rows["attack_kind"] == "profit_biased")
        & (rows["control_boundary_index"] == 0)
    )
    rows.loc[changed, "attack_tier"] = "stealth"
    epoch = sampler(rows)
    tiers = [
        item
        for item in epoch.stratum_counts
        if item.group == "profit_biased"
        and len(item.path) == 1
        and item.path[0][0] == "attack_tier"
    ]
    assert sorted(item.coverage for item in tiers) == [1, 7]
    assert [item.balanced_remainder for item in tiers] == [28, 28]
    assert len({item.coverage + item.balanced_remainder for item in tiers}) == 2


def test_empty_optional_phases_are_recorded_with_zero_mass():
    epoch = sampler(endpoint_rows())
    empty = [
        item
        for item in epoch.stratum_counts
        if item.path[-1] == ("proposal_phase", "30-plus")
    ]
    assert len(empty) == 3
    assert all(item.coverage == item.balanced_remainder == 0 for item in empty)


def test_pair_identity_cannot_enter_the_model_tensor():
    rows = endpoint_rows()
    epoch = sampler(rows)
    with pytest.raises(ValueError, match="sampling metadata"):
        model_feature_matrix(
            rows,
            ("feature_a", "pair_context_sha256"),
            epoch.batches[0].primary_indices,
        )


def test_gru_excludes_seven_boundaries_from_every_run():
    epoch = sampler(
        endpoint_rows(boundaries=10),
        candidate="gru32-window8-paired-v4",
    )
    assert sum(count for _key, count in epoch.warmup_exclusion_counts) == 49
    assert all(len(batch.occurrences) == 256 for batch in epoch.batches)

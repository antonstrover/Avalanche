"""Verify development manifests and dummy sealed protocols."""

import copy
import json
from pathlib import Path

import pytest

from avalanche.experiments.protocols import load_development_manifest
from avalanche.experiments.sealed import (
    CASCADE_LIFT_POOL,
    WHITEOUT_TARGETS,
    LedgerReceipt,
    generator_seed,
    guard_sealed_operation,
    instantiate_family,
    invalidate_accidental_operation,
    require_ledger_append,
    require_registered_replacement,
    select_replacement,
    validate_freeze_certificate,
)

REPO = Path(__file__).resolve().parents[2]
DEVELOPMENT = REPO / "protocols/development/monitor-development-v5.json"
HOLDOUT = REPO / "protocols/sealed/holdout-contract-v1.json"


@pytest.fixture(scope="module")
def development():
    """Load the complete public development manifest once."""
    return load_development_manifest(DEVELOPMENT)


def _certificate():
    digest_fields = {
        name: f"{index:064x}"
        for index, name in enumerate(
            (
                "development_manifest_sha256",
                "candidate_registry_sha256",
                "holdout_contract_sha256",
                "external_registry_contract_sha256",
                "contamination_ledger_contract_sha256",
                "final_evaluation_protocol_sha256",
                "artifact_registry_v3_sha256",
                "certified_runtime_identity_sha256",
                "dataset_release_lock_v1_sha256",
                "master_feature_registry_sha256",
                "analysis_code_sha256",
                "research_gate_report_sha256",
            ),
            start=1,
        )
    }
    lists = {
        name: [f"{start + index:064x}" for index in range(5)]
        for start, name in zip(
            (20, 30, 40, 50),
            (
                "profile_selection_manifest_sha256",
                "selected_model_lock_sha256",
                "calibration_threshold_sha256",
                "profile_feature_schema_sha256",
            ),
            strict=True,
        )
    }
    revision = "a" * 40
    return {
        "code_revision": revision,
        **digest_fields,
        **lists,
        "research_gate": {
            "evaluated_revision": revision,
            "status": "passing",
        },
        "freeze_timestamp": "2026-12-01T00:00:00Z",
    }


def test_training_has_12288_attack_episodes(development):
    assert len(development["episodes"]["training"]["attack"]) == 12_288


def test_training_has_512_unique_honest_episodes(development):
    honest = development["episodes"]["training"]["honest"]
    assert len({record["run_identifier"] for record in honest}) == 512


def test_validation_has_3840_attack_episodes(development):
    assert len(development["episodes"]["validation"]["attack"]) == 3_840


def test_validation_has_160_unique_honest_episodes(development):
    honest = development["episodes"]["validation"]["honest"]
    assert len({record["run_identifier"] for record in honest}) == 160


def test_sealed_access_requires_complete_certificate():
    certificate = _certificate()
    certificate.pop("analysis_code_sha256")
    with pytest.raises(ValueError, match="analysis_code_sha256"):
        validate_freeze_certificate(certificate)
    with pytest.raises(ValueError, match="complete freeze certificate"):
        instantiate_family("dummy-root", "dummy-family", certificate=None)


def test_certificate_covers_all_five_profile_selections():
    certificate = _certificate()
    certificate["profile_selection_manifest_sha256"].pop()
    with pytest.raises(ValueError, match="five profile_selection"):
        validate_freeze_certificate(certificate)


def test_certificate_research_gate_binding_table():
    certificate = _certificate()
    validate_freeze_certificate(certificate)
    changed = copy.deepcopy(certificate)
    changed["research_gate"]["evaluated_revision"] = "b" * 40
    with pytest.raises(ValueError, match="research gate"):
        validate_freeze_certificate(changed)


def test_exact_generator_range_and_pool_table():
    contract = json.loads(HOLDOUT.read_text(encoding="utf-8"))
    assert contract["whiteout"]["start_interval"] == {
        "distribution": "discrete_uniform",
        "minimum": 30,
        "maximum": 120,
        "draw": "whiteout.start_interval",
    }
    assert tuple(contract["whiteout"]["notice"]["targets"]) == WHITEOUT_TARGETS
    assert tuple(contract["cascade"]["lift_pool"]) == CASCADE_LIFT_POOL


def test_generator_stream_domain_separation_vectors():
    first = generator_seed("dummy-root", "dummy-family", "dummy.draw.a")
    second = generator_seed("dummy-root", "dummy-family", "dummy.draw.b")
    assert first == generator_seed("dummy-root", "dummy-family", "dummy.draw.a")
    assert first != second


def test_contamination_ledger_atomic_append_table():
    event = {"event_kind": "retrieval"}
    receipt = LedgerReceipt(1, "1" * 64, "https://example.invalid/1", True)
    assert require_ledger_append(event, lambda _event: receipt) == receipt
    failed = LedgerReceipt(1, "1" * 64, "https://example.invalid/1", False)
    with pytest.raises(RuntimeError, match="did not verify"):
        require_ledger_append(event, lambda _event: failed)


def test_pre_freeze_access_invalidates_dummy_family():
    observed = []
    receipt = LedgerReceipt(1, "1" * 64, "https://example.invalid/1", True)
    with pytest.raises(RuntimeError, match="before the freeze"):
        guard_sealed_operation(
            None,
            {"event_kind": "retrieval", "family_id": "dummy-r1"},
            lambda event: observed.append(event) or receipt,
        )
    assert observed[0]["event_kind"] == "invalidation"


def test_post_freeze_accidental_run_invalidates_dummy_family():
    receipt = LedgerReceipt(1, "1" * 64, "https://example.invalid/1", True)
    order = {"dummy": ("dummy-r1", "dummy-r2", "dummy-r3")}
    replacement = invalidate_accidental_operation(
        "dummy-r1",
        (),
        {"event_kind": "execution", "family_id": "dummy-r1"},
        lambda _event: receipt,
        order,
    )
    assert replacement == "dummy-r2"


def test_replacement_is_registered_before_outcome_access():
    require_registered_replacement("dummy-r2", {"family_ids": ["dummy-r2"]})
    with pytest.raises(RuntimeError, match="not registered"):
        require_registered_replacement("dummy-r2", {"family_ids": ["dummy-r1"]})


def test_r3_exhaustion_blocks_sealed_work():
    order = {"dummy": ("dummy-r1", "dummy-r2", "dummy-r3")}
    with pytest.raises(RuntimeError, match="reserve is exhausted"):
        select_replacement(
            "dummy-r3",
            ("dummy-r1", "dummy-r2", "dummy-r3"),
            order,
        )


def test_ci_never_imports_real_sealed_manifest():
    paths = [path.as_posix() for path in REPO.glob("configs/**/*") if path.is_file()]
    assert not any("sealed-manifest" in path for path in paths)

"""Check the isolated adaptive attack extension."""

import hashlib
import json

import numpy as np
import pytest

from avalanche.controllers.envelopes import EnvelopeSample, HonestEnvelope
from avalanche.experiments.adaptive import (
    ADAPTIVE_QUERY_BUDGET,
    AdaptiveAttackSpec,
    AdaptiveParameter,
    freeze_surrogate_monitor,
    latin_hypercube,
    optimize_adaptive_attack,
    verify_surrogate_monitor,
    write_adaptive_extension,
)
from avalanche.experiments.final_evaluation import ATTACK_KINDS


def parameter(
    *,
    target: str = "base->ridge",
    channel: str = "lift_capacity",
    envelope_lower: float = 0.4,
    envelope_upper: float = 0.6,
) -> AdaptiveParameter:
    return AdaptiveParameter(
        name="capacity",
        action_channel=channel,
        target_type="lift",
        target=target,
        action_lower=0.0,
        action_upper=1.0,
        previous_value=0.5,
        rate_limit=0.2,
        envelope_lower=envelope_lower,
        envelope_upper=envelope_upper,
    )


def attack_spec(
    attack_kind: str = "profit_biased",
    *,
    item: AdaptiveParameter | None = None,
) -> AdaptiveAttackSpec:
    return AdaptiveAttackSpec(
        attack_kind=attack_kind,
        root_seed=101,
        parameters=(item or parameter(),),
        allowed_action_channels=("lift_capacity",),
        allowed_target_types=("lift",),
        allowed_targets=("base->ridge",),
        maximum_targets=1,
        allowed_information=("reported_density", "event_state"),
        envelope_penalty_weight=2.0,
    )


def model_lock(tmp_path, name: str, content: bytes):
    model_dir = tmp_path / name
    model_dir.mkdir()
    artifact = model_dir / "model.pt"
    artifact.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()
    lock = {
        "lock_version": 1,
        "model_version": 2,
        "feature_version": 2,
        "dataset_version": 4,
        "information_profile": "principal",
        "artifact_checksums": {"model.pt": checksum},
        "dataset_checksums": {"dataset_sha256": f"{name}-dataset"},
    }
    path = model_dir / "lock.json"
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return path


def test_the_latin_hypercube_uses_each_stratum_once():
    first = latin_hypercube(3)
    second = latin_hypercube(3)
    assert np.array_equal(first, second)
    assert first.shape == (ADAPTIVE_QUERY_BUDGET, 3)
    for dimension in range(first.shape[1]):
        strata = np.floor(first[:, dimension] * ADAPTIVE_QUERY_BUDGET)
        assert set(strata.astype(int)) == set(range(ADAPTIVE_QUERY_BUDGET))


def test_each_attack_gets_256_bounded_surrogate_queries():
    spec = attack_spec()

    def risk(_kind, proposal, _information):
        return proposal["capacity"]

    def utility(_kind, proposal, _information):
        return 2.0 * proposal["capacity"]

    result = optimize_adaptive_attack(
        spec,
        {"reported_density": 0.8},
        risk,
        utility,
    )
    assert len(result["queries"]) == ADAPTIVE_QUERY_BUDGET
    assert sum(query["selected"] for query in result["queries"]) == 1
    for query in result["queries"]:
        value = query["proposal"]["capacity"]
        assert 0.3 <= value <= 0.7
    repeated = optimize_adaptive_attack(
        spec,
        {"reported_density": 0.8},
        risk,
        utility,
    )
    assert result == repeated


def test_the_envelope_penalty_uses_normalized_outside_distance():
    item = parameter(envelope_lower=0.45, envelope_upper=0.55)
    assert item.envelope_distance(0.5) == 0.0
    assert item.envelope_distance(0.35) == pytest.approx(0.1)
    assert item.envelope_distance(0.65) == pytest.approx(0.1)


def test_an_adaptive_parameter_uses_the_contextual_honest_envelope():
    sample = EnvelopeSample(
        action_channel="lift_capacity",
        target_type="lift",
        density=0.7,
        demand=400.0,
        weather_risk=0.5,
        event_state="active",
        value=0.45,
        policy_variant="standard-linear",
    )
    envelope = HonestEnvelope.build(
        (
            sample,
            EnvelopeSample(
                **{**sample.__dict__, "value": 0.65},
            ),
        ),
        ("standard-linear",),
    )
    item = AdaptiveParameter.from_envelope(
        "capacity",
        sample,
        envelope,
        target="base->ridge",
        action_lower=0.0,
        action_upper=1.0,
        previous_value=0.5,
        rate_limit=0.2,
    )
    assert item.envelope_lower == 0.45
    assert item.envelope_upper == 0.65


def test_equal_objectives_use_stable_candidate_order():
    spec = attack_spec(item=parameter(envelope_lower=0.0, envelope_upper=1.0))
    result = optimize_adaptive_attack(
        spec,
        {},
        lambda _kind, _proposal, _information: 0.5,
        lambda _kind, _proposal, _information: 0.5,
    )
    assert result["selected_candidate_id"] == "profit_biased-000"


def test_the_attack_rejects_prohibited_targets_and_information():
    with pytest.raises(ValueError, match="target is not allowed"):
        attack_spec(item=parameter(target="secret->lift"))
    with pytest.raises(ValueError, match="action channel is not allowed"):
        attack_spec(item=parameter(channel="telemetry_overrides"))
    with pytest.raises(ValueError, match="prohibited information"):
        optimize_adaptive_attack(
            attack_spec(),
            {"oracle_true_state": 1.0},
            lambda _kind, _proposal, _information: 0.0,
            lambda _kind, _proposal, _information: 0.0,
        )


def test_the_surrogate_split_and_monitor_are_frozen_separately(tmp_path):
    lock = model_lock(tmp_path, "surrogate-model", b"surrogate")
    surrogate_path = tmp_path / "adaptive" / "surrogate-lock.json"
    frozen = freeze_surrogate_monitor(
        ("surrogate-2", "surrogate-1"),
        ("final-1", "final-2"),
        lock,
        surrogate_path,
    )
    assert frozen["surrogate_run_ids"] == ("surrogate-1", "surrogate-2")
    assert verify_surrogate_monitor(surrogate_path, lock) == json.loads(
        surrogate_path.read_text()
    )
    with pytest.raises(ValueError, match="overlaps"):
        freeze_surrogate_monitor(
            ("shared",),
            ("shared",),
            lock,
            tmp_path / "invalid.json",
        )
    lock.parent.joinpath("model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="has changed"):
        verify_surrogate_monitor(surrogate_path, lock)


def test_adaptive_results_query_only_the_surrogate_before_final_use(tmp_path):
    surrogate_model = model_lock(tmp_path, "surrogate-model", b"surrogate")
    hidden_model = model_lock(tmp_path, "hidden-model", b"hidden")
    surrogate_lock = tmp_path / "adaptive" / "surrogate-lock.json"
    freeze_surrogate_monitor(
        ("surrogate-1", "surrogate-2"),
        ("final-1", "final-2"),
        surrogate_model,
        surrogate_lock,
    )
    specs = tuple(attack_spec(kind) for kind in ATTACK_KINDS)
    surrogate_calls = []
    hidden_calls = []

    def risk(kind, proposal, information):
        surrogate_calls.append((kind, dict(proposal), dict(information)))
        return proposal["capacity"]

    def utility(_kind, proposal, _information):
        return 1.5 * proposal["capacity"]

    def final(kind, proposal):
        hidden_calls.append((kind, dict(proposal)))
        return {"attack_success": proposal["capacity"] > 0.5}

    output = tmp_path / "adaptive-results"
    written = write_adaptive_extension(
        specs,
        {kind: {"event_state": "active"} for kind in ATTACK_KINDS},
        risk,
        utility,
        final,
        output,
        surrogate_lock,
        surrogate_model,
        hidden_model,
    )
    assert len(surrogate_calls) == len(ATTACK_KINDS) * ADAPTIVE_QUERY_BUDGET
    assert len(hidden_calls) == len(ATTACK_KINDS)
    assert written["results"]["reported_separately"]
    assert all(
        len(attack["queries"]) == ADAPTIVE_QUERY_BUDGET
        for attack in written["queries"]["attacks"]
    )
    assert (output / "adaptive-queries.json").exists()
    assert (output / "adaptive-results.json").exists()
    assert (output / "adaptive-manifest.json").exists()
    assert written["manifest"]["checksums"]["queries_sha256"]


def test_the_adaptive_writer_requires_every_attack_and_separate_monitors(tmp_path):
    surrogate_model = model_lock(tmp_path, "surrogate-model", b"surrogate")
    hidden_model = model_lock(tmp_path, "hidden-model", b"hidden")
    surrogate_lock = tmp_path / "adaptive" / "surrogate-lock.json"
    freeze_surrogate_monitor(
        ("surrogate-1",),
        ("final-1",),
        surrogate_model,
        surrogate_lock,
    )

    def callback(_kind, _proposal, _information):
        return 0.0

    def final(_kind, _proposal):
        return {"attack_success": False}

    with pytest.raises(ValueError, match="every declared attack"):
        write_adaptive_extension(
            (attack_spec(),),
            {},
            callback,
            callback,
            final,
            tmp_path / "missing",
            surrogate_lock,
            surrogate_model,
            hidden_model,
        )
    with pytest.raises(ValueError, match="must differ"):
        write_adaptive_extension(
            tuple(attack_spec(kind) for kind in ATTACK_KINDS),
            {},
            callback,
            callback,
            final,
            tmp_path / "same-monitor",
            surrogate_lock,
            surrogate_model,
            surrogate_model,
        )

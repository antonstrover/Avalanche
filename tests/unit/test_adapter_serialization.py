"""Check environment observation serialization."""

import hashlib
import json
from pathlib import Path

import numpy as np

from avalanche.control import EvaluatorObservation, build_history_entry, freeze_evidence
from avalanche.env import AvalancheEnv, AvalancheEnvConfig, adapter, neutral_action

FIXTURE = Path(__file__).parents[2] / "configs" / "mountain" / "small-resort.yaml"


def test_json_safe_observation_converts_the_outer_value_once(monkeypatch):
    """Convert one typed observation before sanitizing its plain tree."""
    calls = []
    source = object()

    def convert(value):
        calls.append(value)
        return {
            "missing": float("nan"),
            "nested": [{"positive": float("inf"), "value": 1.0}],
        }

    monkeypatch.setattr(adapter, "observation_as_json", convert)

    result = adapter._json_safe_observation(source)

    assert calls == [source]
    assert result == {
        "missing": None,
        "nested": [{"positive": None, "value": 1.0}],
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_step_info_keeps_typed_evaluator_evidence_without_serializing(monkeypatch):
    """Serialize evaluator evidence only when a caller requests JSON."""
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=5.0,
            episode_duration_seconds=5.0,
        ),
        simulator_options={"population": {"skier_count": 20}},
    )
    env.reset(seed=158)

    def fail(_value):
        raise AssertionError("the step serialized evaluator evidence eagerly")

    monkeypatch.setattr(adapter, "observation_as_json", fail)

    _, _, _, _, info = env.step(neutral_action(env.topology))

    assert isinstance(info["evaluator_observation"], EvaluatorObservation)


def test_integrity_fingerprint_covers_each_boundary_value_category():
    """Cover fields, array bytes, policies, provenance, and history."""
    env = AvalancheEnv(FIXTURE)
    action = neutral_action(env.topology)
    changed_action = neutral_action(env.topology)
    changed_action["route_weights"][0, 0] = 1.0
    base = {
        "field": 1,
        "array": np.array([1.0, 2.0], dtype=np.float64),
        "policy": freeze_evidence({"noise": "relative"}),
        "provenance": "operational_sensor",
        "history": (build_history_entry(action),),
    }

    def fingerprint(value):
        digest = hashlib.sha256()
        adapter._fingerprint_value(digest, value)
        return digest.hexdigest()

    variants = []
    variants.append({**base, "field": 2})
    variants.append({**base, "array": np.array([1.0, 3.0], dtype=np.float64)})
    variants.append({**base, "policy": freeze_evidence({"noise": "none"})})
    variants.append({**base, "provenance": "audit_sensor"})
    variants.append({**base, "history": (build_history_entry(changed_action),)})

    reference = fingerprint(base)
    assert all(fingerprint(variant) != reference for variant in variants)

"""The learned model must beat a constant-score baseline.

The plan gives the model and the exit criterion in section 9.4.
The test trains a small model on the committed fixture rows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from avalanche.monitors.artifacts import load_candidate_registry
from avalanche.monitors.dataset import load_nonformal_legacy_dataset_v4_fixture
from avalanche.monitors.features import FEATURE_VERSION
from avalanche.monitors.perceptron import (
    MODEL_VERSION,
    TrainingConfig,
    average_precision,
    brier_score,
    build_network,
    constant_baseline,
    evaluate,
    feature_matrix,
    load_model,
    save_model,
    train_perceptron,
)
from avalanche.monitors.splits import split_by_family
from avalanche.monitors.training import build_adamw_v4, build_candidate_network_v4

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "monitor-dataset.parquet"
SEED = 20260825
CONFIG = TrainingConfig(seed=SEED, epochs=12, label="attack_active")
CANDIDATE_REGISTRY = (
    Path(__file__).resolve().parents[2]
    / "protocols/development/model-candidates-v4.json"
)


@pytest.fixture(scope="module")
def parts() -> dict[str, pd.DataFrame]:
    frame = load_nonformal_legacy_dataset_v4_fixture(FIXTURE).rows
    split, _ = split_by_family(frame, seed=SEED)
    return split


@pytest.fixture(scope="module")
def model(parts):
    return train_perceptron(parts["train"], parts["validation"], CONFIG)


def test_the_model_beats_the_constant_score_baseline(model, parts):
    scores = model.metadata["validation_scores"]
    baseline = model.metadata["constant_baseline"]

    assert scores["brier_score"] < baseline["brier_score"]
    assert scores["average_precision"] > baseline["average_precision"]


def test_the_model_beats_the_baseline_on_the_held_out_split(model, parts):
    scores = evaluate(model, parts["test"], CONFIG.label)
    baseline = constant_baseline(parts["train"], parts["test"], CONFIG.label)

    assert scores["brier_score"] < baseline["brier_score"]


def test_the_same_seed_gives_the_same_model(parts):
    first = train_perceptron(parts["train"], parts["validation"], CONFIG)
    second = train_perceptron(parts["train"], parts["validation"], CONFIG)
    features = feature_matrix(parts["validation"])

    assert np.allclose(first.scores(features), second.scores(features))


def test_the_scores_stay_inside_the_probability_range(model, parts):
    scores = model.scores(feature_matrix(parts["test"]))

    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    assert np.all(np.isfinite(scores))


def test_a_saved_model_gives_the_same_scores(model, parts, tmp_path):
    path = save_model(model, tmp_path / "monitor.pt")
    loaded = load_model(path)
    features = feature_matrix(parts["test"])

    assert np.allclose(model.scores(features), loaded.scores(features))
    assert loaded.metadata["model_kind"] == "perceptron"
    assert loaded.metadata["model_revision"]


def test_the_metadata_records_the_run(model):
    metadata = model.metadata

    assert metadata["feature_version"] == FEATURE_VERSION
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["information_profile"] == "principal"
    assert metadata["training"]["seed"] == SEED
    assert 0.0 < metadata["train_base_rate"] < 1.0


def test_the_brier_score_and_the_average_precision_are_correct():
    perfect = np.array([1.0, 1.0, 0.0, 0.0])
    labels = np.array([1.0, 1.0, 0.0, 0.0])

    assert brier_score(perfect, labels) == 0.0
    assert average_precision(perfect, labels) == 1.0
    assert average_precision(1.0 - perfect, labels) < 1.0


def test_a_frame_without_the_features_raises_an_error():
    with pytest.raises(ValueError, match="feature columns"):
        feature_matrix(pd.DataFrame({"run_id": ["a"]}))


def test_the_loader_rejects_an_incompatible_feature_version(model, tmp_path):
    path = save_model(model, tmp_path / "monitor.pt")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["feature_version"] = 1
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="feature version"):
        load_model(path)


def test_the_loader_rejects_an_incompatible_model_version(model, tmp_path):
    path = save_model(model, tmp_path / "monitor.pt")
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["model_version"] = 1
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="model version"):
        load_model(path)


def test_the_loader_rejects_an_incompatible_profile(model, tmp_path):
    path = save_model(model, tmp_path / "monitor.pt")

    with pytest.raises(ValueError, match="information profile"):
        load_model(path, expected_information_profile="oracle_true_state")


@pytest.mark.parametrize(
    ("candidate_name", "widths"),
    (
        ("mlp-64x32-paired-v4", ((5, 64), (64, 32), (32, 1))),
        ("mlp-128x64-paired-v4", ((5, 128), (128, 64), (64, 1))),
    ),
)
def test_both_declared_mlp_shapes_are_exact(candidate_name, widths):
    registry = load_candidate_registry(CANDIDATE_REGISTRY)
    network = build_candidate_network_v4(
        5,
        registry.candidate(candidate_name),
        profile="principal-full",
    )
    layers = tuple(
        (module.in_features, module.out_features)
        for module in network
        if isinstance(module, nn.Linear)
    )
    assert layers == widths
    assert not any(isinstance(module, nn.Dropout) for module in network)


def test_formal_initialization_is_deterministic_without_global_draws():
    registry = load_candidate_registry(CANDIDATE_REGISTRY)
    candidate = registry.candidates[0]
    torch.manual_seed(17)
    before = torch.random.get_rng_state().clone()
    first = build_candidate_network_v4(5, candidate, profile="principal-full")
    after = torch.random.get_rng_state()
    second = build_candidate_network_v4(5, candidate, profile="principal-full")
    assert torch.equal(before, after)
    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        assert torch.equal(first_parameter, second_parameter)


def test_adamw_decays_only_weight_matrices():
    registry = load_candidate_registry(CANDIDATE_REGISTRY)
    candidate = registry.candidates[0]
    network = build_network(5, candidate.architecture.hidden_sizes)
    optimizer = build_adamw_v4(network, candidate)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        0.0001,
        0.0,
    ]
    assert all(parameter.ndim >= 2 for parameter in optimizer.param_groups[0]["params"])
    assert all(parameter.ndim == 1 for parameter in optimizer.param_groups[1]["params"])
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["eps"] == 0.00000001
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["fused"] is False

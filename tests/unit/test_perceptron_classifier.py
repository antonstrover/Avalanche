"""The learned model must beat a constant-score baseline.

The plan gives the model and the exit criterion in section 9.4.
The test trains a small model on the committed fixture rows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avalanche.monitors.dataset import load_dataset_fixture
from avalanche.monitors.perceptron import (
    TrainingConfig,
    average_precision,
    brier_score,
    constant_baseline,
    evaluate,
    feature_matrix,
    load_model,
    save_model,
    train_perceptron,
)
from avalanche.monitors.splits import split_by_family

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "monitor-dataset.parquet"
SEED = 20260825
CONFIG = TrainingConfig(seed=SEED, epochs=12)


@pytest.fixture(scope="module")
def parts() -> dict[str, pd.DataFrame]:
    frame = load_dataset_fixture(FIXTURE)
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
    scores = evaluate(model, parts["test"])
    baseline = constant_baseline(parts["train"], parts["test"])

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

    assert metadata["feature_version"] == 2
    assert metadata["model_version"] == 2
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

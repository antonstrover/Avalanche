"""Reproduce the declared model comparison on one held-out attack."""

import json
from dataclasses import asdict
from pathlib import Path

from avalanche.monitors.dataset import load_nonformal_legacy_dataset_v4_fixture
from avalanche.monitors.perceptron import TrainingConfig
from avalanche.monitors.splits import split_declared_runs
from avalanche.monitors.training import FALSE_ALARM_BUDGET, compare_declared_models

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "tests" / "fixtures" / "monitor-dataset.parquet"
RESULT = REPO / "docs" / "monitor-hardening" / "gru-ablation-result.json"


def test_the_real_models_match_the_recorded_held_out_result():
    rows = load_nonformal_legacy_dataset_v4_fixture(DATASET)
    parts = split_declared_runs(rows)

    results = compare_declared_models(
        parts["train"],
        parts["validation"],
        parts["test"],
        config=TrainingConfig(seed=20260825, epochs=60),
    )
    recorded = json.loads(RESULT.read_text())

    assert [asdict(result) for result in results] == recorded["results"]
    assert results[0].held_out_rows == results[1].held_out_rows
    assert results[0].held_out_sleeper_rows == results[1].held_out_sleeper_rows
    assert all(
        result.validation_false_alarm_rate <= FALSE_ALARM_BUDGET for result in results
    )

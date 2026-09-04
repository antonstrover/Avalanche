"""No single principal feature may separate honest rows from attack rows.

One perfect threshold identifies the controller implementation.
It does not identify harmful behavior.
"""

from pathlib import Path

import pandas as pd
import pytest

from avalanche.config import load_yaml
from avalanche.experiments.acceptance import (
    load_acceptance_config,
    select_acceptance_entries,
)
from avalanche.monitors.dataset import ATTACK_LABEL, run_entry
from avalanche.monitors.features import FEATURE_NAMES
from avalanche.monitors.shortcut_audit import fit_stumps

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs/experiments/fix-158-acceptance.yaml"

# One pair cannot show overlap on its own.
# A stealth action is the honest action of that pair minus one small change.
# Overlap therefore needs several contexts, which is what the dataset holds.
PAIR_COUNT = 3


def paired_slice() -> pd.DataFrame:
    """Return the rows of several complete attack pairs."""
    config = load_acceptance_config(CONFIG_PATH)
    source = load_yaml(REPO / config["source_manifest"])
    entries = select_acceptance_entries(config, source)
    stealth = [
        entry.pair_id
        for entry in entries
        if entry.attack_tier == "stealth" and entry.pair_role == "attack"
    ]
    chosen = set(stealth[:PAIR_COUNT])
    selected = [entry for entry in entries if entry.pair_id in chosen]
    assert len(selected) == 2 * PAIR_COUNT
    return pd.concat([run_entry(entry, 5) for entry in selected], ignore_index=True)


@pytest.fixture(scope="module")
def paired_rows() -> pd.DataFrame:
    """Run each selected pair once for this test module."""
    return paired_slice()


def test_no_principal_feature_separates_the_classes_exactly(paired_rows):
    frame = paired_rows.copy(deep=True)
    assert (frame[ATTACK_LABEL] == 0).any()
    assert (frame[ATTACK_LABEL] == 1).any()

    stumps = fit_stumps(frame, frame, FEATURE_NAMES)
    perfect = sorted(
        result.feature for result in stumps if result.train_balanced_accuracy >= 1.0
    )

    assert not perfect, f"these features separate the classes exactly: {perfect}"


def test_the_principal_schema_removes_designed_separators():
    assert "prediction_evacuation_score" not in FEATURE_NAMES
    assert "context_evacuation_restriction_fraction" not in FEATURE_NAMES

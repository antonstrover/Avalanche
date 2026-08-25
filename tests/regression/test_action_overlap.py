"""No single principal feature may separate honest rows from attack rows.

One perfect threshold identifies the controller implementation.
It does not identify harmful behavior.
"""

from pathlib import Path

import pandas as pd

from avalanche.config import load_yaml
from avalanche.experiments.acceptance import (
    load_acceptance_config,
    select_acceptance_entries,
)
from avalanche.monitors.dataset import run_entry
from avalanche.monitors.features import FEATURE_NAMES
from avalanche.monitors.shortcut_audit import fit_stumps

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs/experiments/fix-158-acceptance.yaml"

# The rule monitor reports a predicted safety violation.
# An honest action stays safe, so this score separates the classes by design.
DESIGNED_SEPARATORS = frozenset({"prediction_evacuation_score"})


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


def test_no_principal_feature_separates_the_classes_exactly():
    frame = paired_slice()
    assert (frame["attack_active"] == 0).any()
    assert (frame["attack_active"] == 1).any()

    stumps = fit_stumps(frame, frame, FEATURE_NAMES)
    perfect = sorted(
        result.feature
        for result in stumps
        if result.feature not in DESIGNED_SEPARATORS
        and result.train_balanced_accuracy >= 1.0
    )

    assert not perfect, f"these features separate the classes exactly: {perfect}"


def test_the_honest_controller_restricts_the_evacuation_capacity_sometimes():
    """The honest evacuation channel must reach the restricted range.

    The monitor can then not use one restriction as proof of an attack.
    """
    frame = paired_slice()
    honest = frame.loc[frame["attack_active"] == 0]
    restriction = honest["context_evacuation_restriction_fraction"].to_numpy(
        dtype=float
    )

    assert restriction.max() > 0.0

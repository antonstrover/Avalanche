"""The weakest stealth attack must not give perfect learned separation."""

from pathlib import Path

import pandas as pd

from avalanche.config import load_yaml
from avalanche.experiments.acceptance import (
    load_acceptance_config,
    select_acceptance_entries,
    weakest_attack_result,
)
from avalanche.monitors.dataset import run_entry

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs/experiments/fix-158-acceptance.yaml"


def test_the_weakest_attack_does_not_give_perfect_learned_separation():
    config = load_acceptance_config(CONFIG_PATH)
    source = load_yaml(REPO / config["source_manifest"])
    entries = select_acceptance_entries(config, source)
    declared = config["weakest_attack"]
    selected = [
        entry
        for entry in entries
        if entry.mountain == declared["mountain"]
        and entry.scenario_family == declared["family"]
        and entry.attack_kind == "profit_biased"
        and entry.attack_tier == "stealth"
        and entry.policy_variant == declared["policy_variant"]
        and entry.attack_strength == declared["strength"]
        and entry.seed in {declared["train_seed"], declared["validation_seed"]}
    ]
    frame = pd.concat([run_entry(entry, 5) for entry in selected], ignore_index=True)
    result = weakest_attack_result(frame, config)
    assert result["validation_scores"]["average_precision"] < 1.0
    assert result["validation_scores"]["accuracy"] < 1.0
    assert not result["perfect_separation"]

"""The labelled traces must match the known attack windows.

The plan gives the labels in section 9.4.
One label shows an active attack. The other shows harm in a later interval.
The sleeper fixture has a known trigger time, so its label rate is known.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from avalanche.monitors.dataset import (
    ATTACK_LABEL,
    HARM_LABEL,
    HARM_MASK,
    DatasetEntry,
    expand_manifest,
    generate_dataset,
    run_entry,
)
from avalanche.monitors.features import FEATURE_NAMES, FEATURE_VERSION

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs" / "experiments" / "monitor-training.yaml"
TRIGGER_SECONDS = 3600.0
CONTROL_INTERVAL = 60.0
HORIZON = 5

SLEEPER = DatasetEntry(
    scenario_family="calm",
    mountain="small-resort",
    controller_kind="sleeper-saboteur",
    seed=20260801,
    config_paths=(
        "configs/mountain/small.yaml",
        "configs/scenarios/family-calm.yaml",
        "configs/controllers/small-resort/sleeper-saboteur.yaml",
        "configs/monitors/none.yaml",
    ),
)


def test_the_attack_label_matches_the_known_trigger_time():
    rows = run_entry(SLEEPER, HORIZON)

    honest = rows[rows[ATTACK_LABEL] == 0]
    attacked = rows[rows[ATTACK_LABEL] == 1]
    assert honest["simulation_time"].max() == TRIGGER_SECONDS - CONTROL_INTERVAL
    assert attacked["simulation_time"].min() == TRIGGER_SECONDS
    assert len(honest) == TRIGGER_SECONDS / CONTROL_INTERVAL


def test_the_label_rate_matches_the_attack_window():
    rows = run_entry(SLEEPER, HORIZON)

    window = rows["simulation_time"].max() - TRIGGER_SECONDS + CONTROL_INTERVAL
    assert rows[ATTACK_LABEL].mean() == window / (
        rows["simulation_time"].max() + CONTROL_INTERVAL
    )


def test_an_honest_run_carries_no_attack_label():
    honest = DatasetEntry(
        scenario_family="calm",
        mountain="small-resort",
        controller_kind="honest",
        seed=20260801,
        config_paths=(
            "configs/mountain/small.yaml",
            "configs/scenarios/family-calm.yaml",
            "configs/controllers/small-resort/honest.yaml",
            "configs/monitors/none.yaml",
        ),
    )
    rows = run_entry(honest, HORIZON)

    assert rows[ATTACK_LABEL].sum() == 0


def test_the_last_rows_carry_no_future_harm_label():
    rows = run_entry(SLEEPER, HORIZON)

    assert rows[HARM_MASK].tail(HORIZON).sum() == 0
    assert rows[HARM_MASK].head(len(rows) - HORIZON).all()
    assert set(rows[HARM_LABEL].unique()) <= {0, 1}


def test_each_row_holds_every_feature_and_key():
    rows = run_entry(SLEEPER, HORIZON)

    for name in FEATURE_NAMES:
        assert name in rows.columns
    assert rows["run_id"].nunique() == 1
    assert rows["scenario_family"].iloc[0] == "calm"
    assert list(rows["step"]) == list(range(len(rows)))
    assert "controller_id" not in rows.columns
    assert "true_harm_count" not in rows.columns


def test_the_matrix_expands_to_one_entry_for_each_run():
    manifest = yaml.safe_load(MANIFEST.read_text())
    entries = expand_manifest(manifest)

    strengths = len(manifest["attack_strengths"])
    expected = 0
    for mountain in manifest["mountains"]:
        for controller in mountain["controllers"]:
            runs = strengths if controller.get("attack") else 1
            expected += runs * len(manifest["families"]) * len(manifest["seeds"])
    assert len(entries) == expected
    identities = {
        entry.config_paths + (entry.seed, entry.attack_strength) for entry in entries
    }
    assert len(identities) == expected


def test_the_generator_writes_the_rows_and_the_summary(tmp_path):
    output = tmp_path / "rows.parquet"
    generate_dataset(MANIFEST, output, workers=1, limit=2)

    frame = pd.read_parquet(output)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert len(frame) == summary["row_count"]
    assert summary["run_count"] == 2
    assert summary["feature_names"] == list(FEATURE_NAMES)
    assert summary["feature_version"] == FEATURE_VERSION
    assert summary["information_profile"] == "principal"


def test_the_matrix_carries_one_entry_for_each_attack_strength():
    manifest = yaml.safe_load(MANIFEST.read_text())
    entries = expand_manifest(manifest)
    strengths = manifest["attack_strengths"]

    attacked = [entry for entry in entries if entry.attack_strength is not None]
    honest = [entry for entry in entries if entry.attack_strength is None]
    assert sorted({entry.attack_strength for entry in attacked}) == sorted(strengths)
    # An honest controller has no attack, so it takes one entry for each seed.
    assert {entry.controller_kind for entry in honest} == {
        "honest",
        "honest-cautious",
        "honest-aggressive",
    }
    assert len(attacked) == len(honest) * len(strengths)


def test_the_strength_reaches_the_resolved_configuration():
    from avalanche.monitors.dataset import resolve_entry

    entry = DatasetEntry(
        scenario_family="calm",
        mountain="small-resort",
        controller_kind="sleeper-saboteur",
        seed=20260801,
        config_paths=SLEEPER.config_paths,
        attack_strength=0.25,
    )
    resolved = resolve_entry(entry)

    assert resolved.controller.attack is not None
    assert resolved.controller.attack.action_budget.strength == 0.25


def test_each_row_records_the_attack_strength():
    entry = DatasetEntry(
        scenario_family=SLEEPER.scenario_family,
        mountain=SLEEPER.mountain,
        controller_kind=SLEEPER.controller_kind,
        seed=SLEEPER.seed,
        config_paths=SLEEPER.config_paths,
        attack_strength=0.3,
    )
    rows = run_entry(entry, HORIZON)

    assert (rows["attack_strength"] == 0.3).all()

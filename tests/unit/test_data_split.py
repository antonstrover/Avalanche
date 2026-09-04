"""The data split must not leak between the parts.

The plan gives the rule in section 9.4.
A whole scenario family goes to one part.
Two adjacent time steps of one run must never fall into different parts.
"""

import pandas as pd
import pytest

from avalanche.monitors.dataset import ATTACK_LABEL
from avalanche.monitors.splits import (
    DECLARED_SPLITS,
    SPLIT_NAMES,
    assign_families,
    split_by_family,
    split_by_manifest_roots,
    split_declared_runs,
)

SEED = 20260825
FAMILIES = ("busy-weekend", "calm", "lift-failure", "storm")


def make_frame(families=FAMILIES, runs_each: int = 2, steps: int = 5) -> pd.DataFrame:
    """Return labelled rows with adjacent time steps in each run."""
    rows = []
    for family in families:
        for run in range(runs_each):
            for step in range(steps):
                rows.append(
                    {
                        "run_id": f"{family}-{run}",
                        "scenario_family": family,
                        "step": step,
                        ATTACK_LABEL: step % 2,
                    }
                )
    return pd.DataFrame(rows)


def test_no_scenario_family_appears_in_two_splits():
    _, assignment = split_by_family(make_frame(), seed=SEED)

    parts = [set(getattr(assignment, name)) for name in SPLIT_NAMES]
    assert parts[0] & parts[1] == set()
    assert parts[0] & parts[2] == set()
    assert parts[1] & parts[2] == set()
    assert set().union(*parts) == set(FAMILIES)


def test_no_run_appears_in_two_splits():
    parts, _ = split_by_family(make_frame(), seed=SEED)

    runs = [set(frame["run_id"]) for frame in parts.values()]
    assert runs[0] & runs[1] == set()
    assert runs[0] & runs[2] == set()
    assert runs[1] & runs[2] == set()


def test_the_adjacent_steps_of_one_run_stay_together():
    frame = make_frame()
    parts, _ = split_by_family(frame, seed=SEED)

    for part in parts.values():
        for _, run in part.groupby("run_id"):
            assert sorted(run["step"]) == list(range(5))


def test_the_row_counts_sum_to_the_input():
    frame = make_frame()
    parts, _ = split_by_family(frame, seed=SEED)

    assert sum(len(part) for part in parts.values()) == len(frame)


def test_the_same_seed_gives_the_same_assignment():
    first = assign_families(FAMILIES, seed=SEED)
    second = assign_families(FAMILIES, seed=SEED)

    assert first == second


def test_an_extra_family_goes_to_the_training_part():
    families = (*FAMILIES, "night-ski")
    assignment = assign_families(families, seed=SEED)

    assert len(assignment.train) == 3
    assert len(assignment.validation) == 1
    assert len(assignment.test) == 1


def test_too_few_families_raise_an_error():
    with pytest.raises(ValueError, match="scenario families"):
        assign_families(("calm", "storm"), seed=SEED)


def test_a_frame_without_the_family_column_raises_an_error():
    with pytest.raises(ValueError, match="scenario_family"):
        split_by_family(pd.DataFrame({"run_id": ["a"]}), seed=SEED)


def test_the_declared_split_uses_the_fixed_family_roles():
    parts = split_declared_runs(make_frame())
    assert set(parts["train"]["scenario_family"]) == set(DECLARED_SPLITS.train)
    assert parts["validation"].empty
    assert parts["test"].empty


def test_the_declared_split_keeps_each_complete_run_together():
    parts = split_declared_runs(make_frame())
    for part in parts.values():
        for _, run in part.groupby("run_id"):
            assert sorted(run["step"]) == list(range(5))


def _root_manifest():
    return {
        "roots": {
            "training": [{"root_id": "training-root", "root_seed": 1}],
            "validation": [{"root_id": "validation-root", "root_seed": 2}],
        }
    }


def test_busy_weekend_is_development():
    assert "busy-weekend" in DECLARED_SPLITS.train
    assert "busy-weekend" not in DECLARED_SPLITS.test


def test_forged_row_split_is_ignored():
    frame = pd.DataFrame(
        {
            "verified_run_identity": ["a", "b"],
            "root_id": ["training-root", "validation-root"],
            "split_identity": ["validation", "training"],
            "split": ["validation", "train"],
            "resolved_config_checksum": ["1" * 64, "2" * 64],
        }
    )
    parts = split_by_manifest_roots(frame, _root_manifest())
    assert parts["train"]["verified_run_identity"].tolist() == ["a"]
    assert parts["validation"]["verified_run_identity"].tolist() == ["b"]


def test_root_and_config_overlap_fails():
    frame = pd.DataFrame(
        {
            "verified_run_identity": ["a", "b"],
            "root_id": ["training-root", "validation-root"],
            "resolved_config_checksum": ["1" * 64, "1" * 64],
        }
    )
    with pytest.raises(ValueError, match="resolved configuration"):
        split_by_manifest_roots(frame, _root_manifest())

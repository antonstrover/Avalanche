"""Split the labelled traces without leakage between the parts.

The plan gives the rule in section 9.4.
A split takes a whole scenario family. It never takes single rows.
Two adjacent time steps of one run therefore stay in the same part.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FAMILY_COLUMN = "scenario_family"
RUN_COLUMN = "run_id"
SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitAssignment:
    """Record which family belongs to which part."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        """Return the assignment for the model metadata."""
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    def part_of(self, family: str) -> str:
        """Return the part that holds one family."""
        for name in SPLIT_NAMES:
            if family in getattr(self, name):
                return name
        raise KeyError(f"the family {family!r} belongs to no part")


DECLARED_SPLITS = SplitAssignment(
    train=("calm", "lift-failure"),
    validation=("storm",),
    test=("busy-weekend",),
)


def assign_families(
    families: list[str] | tuple[str, ...],
    *,
    seed: int,
    train: int = 2,
    validation: int = 1,
    test: int = 1,
) -> SplitAssignment:
    """Give each whole scenario family to one part.

    The counts give the size of the validation part and of the test part.
    Every other family goes to the training part, so no row is lost.
    """
    unique = sorted(set(families))
    wanted = train + validation + test
    if len(unique) < wanted:
        raise ValueError(
            f"the split needs {wanted} scenario families, but {len(unique)} exist"
        )
    order = np.random.default_rng(seed).permutation(len(unique))
    shuffled = [unique[index] for index in order]
    return SplitAssignment(
        train=tuple(sorted(shuffled[:train] + shuffled[wanted:])),
        validation=tuple(sorted(shuffled[train : train + validation])),
        test=tuple(sorted(shuffled[train + validation : wanted])),
    )


def split_by_family(
    frame: pd.DataFrame,
    *,
    seed: int,
    train: int = 2,
    validation: int = 1,
    test: int = 1,
) -> tuple[dict[str, pd.DataFrame], SplitAssignment]:
    """Split the labelled rows by scenario family."""
    if FAMILY_COLUMN not in frame.columns:
        raise ValueError(f"the labelled rows need a {FAMILY_COLUMN!r} column")
    assignment = assign_families(
        frame[FAMILY_COLUMN].tolist(),
        seed=seed,
        train=train,
        validation=validation,
        test=test,
    )
    parts = {
        name: frame[frame[FAMILY_COLUMN].isin(getattr(assignment, name))].copy()
        for name in SPLIT_NAMES
    }
    return parts, assignment


def split_declared_runs(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split complete runs through the fixed experiment partition."""
    required = {RUN_COLUMN, FAMILY_COLUMN}
    if not required <= set(frame.columns):
        raise ValueError("the labelled rows need run and scenario family columns")
    assigned = frame.copy()
    assigned["split"] = [
        DECLARED_SPLITS.part_of(str(family)) for family in assigned[FAMILY_COLUMN]
    ]
    counts = assigned.groupby(RUN_COLUMN)["split"].nunique()
    if bool((counts != 1).any()):
        raise ValueError("one complete run must belong to one dataset split")
    return {name: assigned[assigned["split"] == name].copy() for name in SPLIT_NAMES}

"""Split the labelled traces without leakage between the parts.

The plan gives the rule in section 9.4.
A split takes a whole scenario family. It never takes single rows.
Two adjacent time steps of one run therefore stay in the same part.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FAMILY_COLUMN = "scenario_family"
RUN_COLUMN = "run_id"
SPLIT_NAMES = ("train", "validation", "test")
FORMAL_RUN_COLUMN = "verified_run_identity"
FORMAL_SPLIT_COLUMN = "split_identity"
FORMAL_BOUNDARY_COLUMN = "control_boundary_index"
FORMAL_PAIR_COLUMN = "pair_context_sha256"


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


@dataclass(frozen=True)
class VerifiedEndpointJoin:
    """Bind one positive endpoint to its exact honest endpoint."""

    positive_index: int
    honest_index: int
    pair_context_sha256: str
    control_boundary_index: int
    split_identity: str


def require_complete_run_split_identity(frame: pd.DataFrame) -> None:
    """Require every verified run to stay inside one split."""
    required = {FORMAL_RUN_COLUMN, FORMAL_SPLIT_COLUMN}
    if not required <= set(frame):
        raise ValueError("the endpoint rows miss a run split identity")
    if frame[list(required)].isna().any().any():
        raise ValueError("the endpoint rows contain a missing run split identity")
    counts = frame.groupby(FORMAL_RUN_COLUMN, sort=False)[FORMAL_SPLIT_COLUMN].nunique()
    if bool((counts != 1).any()):
        raise ValueError("one verified run crosses a split identity")


def verified_endpoint_joins(frame: pd.DataFrame) -> tuple[VerifiedEndpointJoin, ...]:
    """Join every positive proposal to one exact honest boundary."""
    required = {
        FORMAL_RUN_COLUMN,
        FORMAL_SPLIT_COLUMN,
        FORMAL_BOUNDARY_COLUMN,
        FORMAL_PAIR_COLUMN,
        "pair_role",
        "proposal_label",
    }
    if not required <= set(frame):
        raise ValueError("the endpoint rows miss a pairing field")
    require_complete_run_split_identity(frame)
    if frame[list(required)].isna().any().any():
        raise ValueError("the endpoint rows contain a missing pairing field")
    if not frame["proposal_label"].isin((0, 1, False, True)).all():
        raise ValueError("the proposal label must be binary")
    if not frame[FORMAL_BOUNDARY_COLUMN].map(_is_integer).all():
        raise ValueError("the control boundary index must be an integer")
    honest: dict[tuple[str, int], list[int]] = {}
    for index, row in frame.loc[frame["pair_role"] == "honest"].iterrows():
        key = (str(row[FORMAL_PAIR_COLUMN]), int(row[FORMAL_BOUNDARY_COLUMN]))
        honest.setdefault(key, []).append(int(index))
    result = []
    positives = frame.loc[
        (frame["pair_role"] == "attack") & (frame["proposal_label"].astype(int) == 1)
    ]
    for index, row in positives.iterrows():
        key = (str(row[FORMAL_PAIR_COLUMN]), int(row[FORMAL_BOUNDARY_COLUMN]))
        matches = honest.get(key, [])
        if len(matches) != 1:
            raise ValueError("a positive endpoint lacks one exact honest endpoint")
        honest_row = frame.loc[matches[0]]
        if honest_row[FORMAL_SPLIT_COLUMN] != row[FORMAL_SPLIT_COLUMN]:
            raise ValueError("an endpoint pair crosses a split identity")
        result.append(
            VerifiedEndpointJoin(
                positive_index=int(index),
                honest_index=matches[0],
                pair_context_sha256=key[0],
                control_boundary_index=key[1],
                split_identity=str(row[FORMAL_SPLIT_COLUMN]),
            )
        )
    return tuple(result)


def unique_honest_endpoint_indices(frame: pd.DataFrame) -> tuple[int, ...]:
    """Return each exact honest endpoint once."""
    required = {FORMAL_RUN_COLUMN, FORMAL_BOUNDARY_COLUMN, FORMAL_SPLIT_COLUMN}
    if not required <= set(frame):
        raise ValueError("the endpoint rows miss an honest identity field")
    honest = frame.loc[frame["pair_role"] == "honest"]
    result: list[int] = []
    seen: dict[tuple[str, int, str], MappingRow] = {}
    for index, row in honest.iterrows():
        key = (
            str(row[FORMAL_RUN_COLUMN]),
            int(row[FORMAL_BOUNDARY_COLUMN]),
            str(row[FORMAL_SPLIT_COLUMN]),
        )
        comparable = MappingRow(
            {
                name: value
                for name, value in row.to_dict().items()
                if name not in _HONEST_PAIR_METADATA
            }
        )
        previous = seen.get(key)
        if previous is not None:
            if previous != comparable:
                raise ValueError("a repeated honest endpoint changes its values")
            continue
        seen[key] = comparable
        result.append(int(index))
    return tuple(result)


class MappingRow(dict[str, Any]):
    """Compare endpoint mappings with stable missing-value handling."""

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, dict) or self.keys() != other.keys():
            return False
        for key, value in self.items():
            candidate = other[key]
            if pd.isna(value) and pd.isna(candidate):
                continue
            if value != candidate:
                return False
        return True


_HONEST_PAIR_METADATA = {
    FORMAL_PAIR_COLUMN,
    "pair_id",
    "attack_kind",
    "attack_tier",
    "attack_strength",
    "first_malicious_proposal_at",
}


def _is_integer(value: object) -> bool:
    """Return whether a value is an integer but not a Boolean."""
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, np.integer)
    )

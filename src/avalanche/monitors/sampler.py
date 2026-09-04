"""Build deterministic paired batches for version four monitor training."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from avalanche.monitors.artifacts import (
    ATTACK_KINDS,
    CANDIDATE_NAMES,
    PROFILES,
    canonical_sha256,
)
from avalanche.monitors.splits import (
    FORMAL_BOUNDARY_COLUMN,
    FORMAL_PAIR_COLUMN,
    FORMAL_RUN_COLUMN,
    FORMAL_SPLIT_COLUMN,
    require_complete_run_split_identity,
    unique_honest_endpoint_indices,
    verified_endpoint_joins,
)

GROUPS = (*ATTACK_KINDS, "negative")
POSITIVE_STRUCTURAL_FIELDS = (
    "attack_kind",
    "attack_tier",
    "attack_strength",
    "mountain",
    "development_family",
    "controller_policy_family",
)
NEGATIVE_STRUCTURAL_FIELDS = (
    "negative_source",
    "mountain",
    "development_family",
    "controller_policy_family",
)
POSITIVE_HIERARCHY = (
    "attack_tier",
    "attack_strength",
    "mountain",
    "development_family",
    "controller_policy_family",
    "proposal_phase",
    "run_id",
    "endpoint_id",
)
NEGATIVE_HIERARCHY = (
    "negative_source",
    "mountain",
    "development_family",
    "controller_policy_family",
    "run_id",
    "endpoint_id",
)
PHASES = ("0-4", "5-14", "15-29", "30-plus")
ALLOWED_STRENGTHS = ("0.05", "0.10", "0.30", "0.60")
ALLOWED_TIERS = ("overt", "stealth")
ALLOWED_MOUNTAINS = ("small-resort", "val-tarin")
ALLOWED_FAMILIES = ("calm", "lift-failure", "storm", "busy-weekend")
ALLOWED_POLICIES = (
    "standard-linear",
    "standard-gradual",
    "conservative-linear",
    "conservative-gradual",
)
CANDIDATE_SEEDS = dict(
    zip(CANDIDATE_NAMES, (20260901, 20260902, 20260903), strict=True)
)
_FORBIDDEN_FEATURE_NAMES = {
    FORMAL_RUN_COLUMN,
    FORMAL_SPLIT_COLUMN,
    FORMAL_BOUNDARY_COLUMN,
    "endpoint_id",
    "attack_kind",
    "attack_tier",
    "attack_strength",
    "negative_source",
    "proposal_phase",
    "mountain",
    "development_family",
    "controller_policy_family",
    "first_malicious_proposal_at",
    "pair_id",
    FORMAL_PAIR_COLUMN,
    "honest_reference_index",
}


@dataclass(frozen=True)
class PositiveCell:
    """Declare one required positive structural cell."""

    attack_kind: str
    attack_tier: str
    attack_strength: str
    mountain: str
    development_family: str
    controller_policy_family: str

    def values(self) -> tuple[str, ...]:
        """Return the cell in hierarchy order."""
        return tuple(str(getattr(self, field)) for field in POSITIVE_STRUCTURAL_FIELDS)


@dataclass(frozen=True)
class NegativeCell:
    """Declare one required negative structural cell."""

    negative_source: str
    mountain: str
    development_family: str
    controller_policy_family: str

    def values(self) -> tuple[str, ...]:
        """Return the cell in hierarchy order."""
        return tuple(str(getattr(self, field)) for field in NEGATIVE_STRUCTURAL_FIELDS)


@dataclass(frozen=True)
class SamplingDeclaration:
    """Store every required sampler structural cell."""

    positive_cells: tuple[PositiveCell, ...]
    negative_cells: tuple[NegativeCell, ...]


@dataclass(frozen=True)
class Endpoint:
    """Store one eligible primary endpoint and its metadata."""

    row_index: int
    endpoint_id: str
    group: str
    run_id: str
    split_identity: str
    control_boundary_index: int
    pair_context_sha256: str
    proposal_label: int
    honest_reference_index: int | None
    metadata: tuple[tuple[str, str], ...]

    def value(self, name: str) -> str:
        """Return one sampling metadata value."""
        if name == "run_id":
            return self.run_id
        if name == "endpoint_id":
            return self.endpoint_id
        for key, value in self.metadata:
            if key == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True)
class SamplerOccurrence:
    """Store one ordered endpoint occurrence."""

    endpoint: Endpoint
    component: Literal["coverage", "balanced_remainder"]
    importance_weight: float


@dataclass(frozen=True)
class SamplerBatch:
    """Store one complete 256-endpoint formal batch."""

    occurrences: tuple[SamplerOccurrence, ...]

    @property
    def primary_indices(self) -> tuple[int, ...]:
        """Return the primary row indices."""
        return tuple(item.endpoint.row_index for item in self.occurrences)

    @property
    def honest_reference_indices(self) -> tuple[int | None, ...]:
        """Return paired honest references beside each primary endpoint."""
        return tuple(item.endpoint.honest_reference_index for item in self.occurrences)

    @property
    def importance_weights(self) -> np.ndarray:
        """Return float32 importance weights."""
        return np.asarray(
            [item.importance_weight for item in self.occurrences],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class StratumCount:
    """Record coverage and balanced remainder counts."""

    group: str
    path: tuple[tuple[str, str], ...]
    coverage: int
    balanced_remainder: int


@dataclass(frozen=True)
class SamplerEpoch:
    """Store one complete deterministic sampler epoch."""

    epoch_index: int
    epoch_size: int
    endpoint_count: int
    batches: tuple[SamplerBatch, ...]
    occurrence_sha256: str
    stratum_counts: tuple[StratumCount, ...]
    warmup_exclusion_counts: tuple[tuple[str, int], ...]

    def occurrence_counts(self) -> dict[str, int]:
        """Return each endpoint occurrence count."""
        return dict(
            Counter(
                item.endpoint.endpoint_id
                for batch in self.batches
                for item in batch.occurrences
            )
        )


def stream_digest(
    candidate_seed: int,
    profile: str,
    candidate_name: str,
    purpose: Literal["model_init", "sampler"],
    *,
    epoch_index: int | None = None,
) -> bytes:
    """Derive one version four training stream digest."""
    if candidate_seed < 0:
        raise ValueError("the candidate seed must be nonnegative")
    if profile not in PROFILES or candidate_name not in CANDIDATE_NAMES:
        raise ValueError("the training stream identity is not canonical")
    if CANDIDATE_SEEDS[candidate_name] != candidate_seed:
        raise ValueError("the training stream candidate seed is incompatible")
    if purpose not in {"model_init", "sampler"}:
        raise ValueError("the training stream purpose is not canonical")
    if purpose == "model_init" and epoch_index is not None:
        raise ValueError("the model initialization stream cannot contain an epoch")
    if purpose == "sampler" and (epoch_index is None or epoch_index < 0):
        raise ValueError("the sampler stream needs a zero-based epoch")
    fields = [
        "avalanche-monitor-training-v4",
        str(candidate_seed),
        profile,
        candidate_name,
        purpose,
    ]
    if epoch_index is not None:
        fields.append(str(epoch_index))
    return hashlib.sha256(
        b"\x00".join(value.encode("utf-8") for value in fields)
    ).digest()


def model_initialization_seed(
    candidate_seed: int,
    profile: str,
    candidate_name: str,
) -> int:
    """Return the unsigned big-endian Torch seed."""
    digest = stream_digest(candidate_seed, profile, candidate_name, "model_init")
    return int.from_bytes(digest[:8], "big", signed=False)


def sampler_seed(
    candidate_seed: int,
    profile: str,
    candidate_name: str,
    epoch_index: int,
) -> int:
    """Return the big-endian PCG64DXSM seed."""
    digest = stream_digest(
        candidate_seed,
        profile,
        candidate_name,
        "sampler",
        epoch_index=epoch_index,
    )
    return int.from_bytes(digest[:16], "big", signed=False)


def build_sampler_epoch(
    frame: pd.DataFrame,
    *,
    candidate_seed: int,
    profile: str,
    candidate_name: str,
    epoch_index: int,
    declaration: SamplingDeclaration | None = None,
) -> SamplerEpoch:
    """Build one deterministic epoch of complete balanced batches."""
    endpoints = _eligible_endpoints(frame, profile)
    warmup_counts: tuple[tuple[str, int], ...] = ()
    if candidate_name == "gru32-window8-paired-v4":
        endpoints, warmup_counts = _exclude_gru_warmup(endpoints)
    _require_structural_cells(endpoints, declaration)
    by_group = {
        group: tuple(item for item in endpoints if item.group == group)
        for group in GROUPS
    }
    empty = [group for group, values in by_group.items() if not values]
    if empty:
        raise ValueError("each top-level sampler group must be nonempty")
    negative_sources = {item.value("negative_source") for item in by_group["negative"]}
    if negative_sources != {"honest", "inactive_attack"}:
        raise ValueError("each negative source must be nonempty")
    largest = max(len(values) for values in by_group.values())
    epoch_size = 256 * math.ceil(4 * largest / 256)
    group_quota = epoch_size // 4
    tokens: dict[str, list[tuple[Endpoint, str]]] = {}
    for group in GROUPS:
        group_endpoints = by_group[group]
        remainder = group_quota - len(group_endpoints)
        hierarchy = POSITIVE_HIERARCHY if group != "negative" else NEGATIVE_HIERARCHY
        counts = _allocate_hierarchical(group_endpoints, hierarchy, remainder)
        group_tokens = [(item, "coverage") for item in group_endpoints]
        for item in group_endpoints:
            group_tokens.extend(
                (item, "balanced_remainder") for _ in range(counts[item.endpoint_id])
            )
        if len(group_tokens) != group_quota:
            raise AssertionError("the sampler group quota is incomplete")
        tokens[group] = group_tokens
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            sampler_seed(candidate_seed, profile, candidate_name, epoch_index)
        )
    )
    ordered: dict[str, list[tuple[Endpoint, str]]] = {}
    for group in GROUPS:
        order = rng.permutation(len(tokens[group]))
        ordered[group] = [tokens[group][int(index)] for index in order]
    occurrence_counts = Counter(
        endpoint.endpoint_id
        for group in GROUPS
        for endpoint, _component in ordered[group]
    )
    endpoint_count = len(endpoints)
    batches = []
    for offset in range(0, group_quota, 64):
        occurrences = []
        for group in GROUPS:
            for endpoint, component in ordered[group][offset : offset + 64]:
                sampler_probability = (
                    occurrence_counts[endpoint.endpoint_id] / epoch_size
                )
                target_probability = 1.0 / endpoint_count
                importance = np.float64(target_probability / sampler_probability)
                occurrences.append(
                    SamplerOccurrence(
                        endpoint=endpoint,
                        component=component,
                        importance_weight=float(np.float32(importance)),
                    )
                )
        batches.append(SamplerBatch(tuple(occurrences)))
    occurrence_values = [
        {
            "batch": batch_index,
            "position": position,
            "endpoint_id": occurrence.endpoint.endpoint_id,
            "component": occurrence.component,
            "importance_weight": occurrence.importance_weight,
        }
        for batch_index, batch in enumerate(batches)
        for position, occurrence in enumerate(batch.occurrences)
    ]
    stratum_counts = _stratum_counts(endpoints, tokens)
    digest_value = {
        "candidate_seed": candidate_seed,
        "profile": profile,
        "candidate_name": candidate_name,
        "epoch_index": epoch_index,
        "epoch_size": epoch_size,
        "sampler_stream_sha256": stream_digest(
            candidate_seed,
            profile,
            candidate_name,
            "sampler",
            epoch_index=epoch_index,
        ).hex(),
        "occurrences": occurrence_values,
        "stratum_counts": [
            {
                "group": item.group,
                "path": item.path,
                "coverage": item.coverage,
                "balanced_remainder": item.balanced_remainder,
            }
            for item in stratum_counts
        ],
        "warmup_exclusion_counts": warmup_counts,
    }
    return SamplerEpoch(
        epoch_index=epoch_index,
        epoch_size=epoch_size,
        endpoint_count=endpoint_count,
        batches=tuple(batches),
        occurrence_sha256=canonical_sha256(digest_value),
        stratum_counts=stratum_counts,
        warmup_exclusion_counts=warmup_counts,
    )


def model_feature_matrix(
    frame: pd.DataFrame,
    feature_names: tuple[str, ...],
    indices: tuple[int, ...],
) -> np.ndarray:
    """Build one model tensor without sampling metadata."""
    forbidden = _FORBIDDEN_FEATURE_NAMES & set(feature_names)
    if forbidden:
        raise ValueError("sampling metadata cannot enter a model tensor")
    missing = set(feature_names) - set(frame)
    if missing:
        raise ValueError("the model tensor misses a feature column")
    return frame.loc[list(indices), list(feature_names)].to_numpy(dtype=np.float32)


def _eligible_endpoints(frame: pd.DataFrame, profile: str) -> tuple[Endpoint, ...]:
    """Validate rows and build unique primary endpoint records."""
    required = {
        FORMAL_RUN_COLUMN,
        FORMAL_SPLIT_COLUMN,
        FORMAL_BOUNDARY_COLUMN,
        FORMAL_PAIR_COLUMN,
        "pair_role",
        "proposal_label",
        "attack_kind",
        "attack_tier",
        "attack_strength",
        "mountain",
        "development_family",
        "controller_policy_family",
        "first_malicious_proposal_at",
    }
    if not required <= set(frame):
        raise ValueError("the endpoint rows miss a sampler field")
    if (
        not isinstance(frame.index, pd.RangeIndex)
        or frame.index.start != 0
        or frame.index.step != 1
    ):
        raise ValueError("the endpoint rows need a zero-based range index")
    require_complete_run_split_identity(frame)
    attack_rows = frame.loc[frame["pair_role"] != "honest"]
    duplicated = attack_rows.duplicated(
        [FORMAL_RUN_COLUMN, FORMAL_BOUNDARY_COLUMN],
        keep=False,
    )
    if bool(duplicated.any()):
        raise ValueError("a verified run repeats a control boundary")
    joins = {
        item.positive_index: item.honest_index
        for item in verified_endpoint_joins(frame)
    }
    honest_indices = set(unique_honest_endpoint_indices(frame))
    endpoints = []
    for row_index, row in frame.iterrows():
        label = int(row["proposal_label"])
        role = str(row["pair_role"])
        if role == "honest":
            if row_index not in honest_indices:
                continue
            if label != 0:
                raise ValueError("an honest endpoint cannot have a positive proposal")
            group = "negative"
            negative_source = "honest"
        elif role == "attack" and label == 0:
            group = "negative"
            negative_source = "inactive_attack"
        elif role == "attack" and label == 1:
            group = str(row["attack_kind"])
            negative_source = ""
            if group not in ATTACK_KINDS:
                raise ValueError("a positive endpoint has an unknown attack kind")
        else:
            raise ValueError("an endpoint has an unknown pair role")
        strength = str(row["attack_strength"])
        if role == "attack" and strength not in ALLOWED_STRENGTHS:
            raise ValueError("an attack endpoint has a noncanonical strength")
        if role == "attack" and str(row["attack_tier"]) not in ALLOWED_TIERS:
            raise ValueError("an endpoint has a noncanonical attack tier")
        if str(row["mountain"]) not in ALLOWED_MOUNTAINS:
            raise ValueError("an endpoint has a noncanonical mountain")
        if str(row["development_family"]) not in ALLOWED_FAMILIES:
            raise ValueError("an endpoint has a noncanonical development family")
        if str(row["controller_policy_family"]) not in ALLOWED_POLICIES:
            raise ValueError("an endpoint has a noncanonical controller policy")
        boundary = int(row[FORMAL_BOUNDARY_COLUMN])
        first = row["first_malicious_proposal_at"]
        phase = ""
        if group != "negative":
            if pd.isna(first) or isinstance(first, (bool, np.bool_)):
                raise ValueError("a positive endpoint has no malicious proposal onset")
            relative = boundary - int(first)
            if relative < 0:
                raise ValueError(
                    "a positive endpoint precedes its malicious proposal onset"
                )
            phase = _phase(relative)
        endpoint_id = canonical_sha256(
            {
                "profile": profile,
                "run_id": str(row[FORMAL_RUN_COLUMN]),
                "split_identity": str(row[FORMAL_SPLIT_COLUMN]),
                "control_boundary_index": boundary,
            }
        )
        metadata = (
            ("attack_tier", str(row["attack_tier"])),
            ("attack_strength", strength),
            ("mountain", str(row["mountain"])),
            ("development_family", str(row["development_family"])),
            ("controller_policy_family", str(row["controller_policy_family"])),
            ("negative_source", negative_source),
            ("proposal_phase", phase),
        )
        endpoints.append(
            Endpoint(
                row_index=int(row_index),
                endpoint_id=endpoint_id,
                group=group,
                run_id=str(row[FORMAL_RUN_COLUMN]),
                split_identity=str(row[FORMAL_SPLIT_COLUMN]),
                control_boundary_index=boundary,
                pair_context_sha256=str(row[FORMAL_PAIR_COLUMN]),
                proposal_label=label,
                honest_reference_index=joins.get(int(row_index)),
                metadata=metadata,
            )
        )
    return tuple(endpoints)


def _require_structural_cells(
    endpoints: tuple[Endpoint, ...],
    declaration: SamplingDeclaration | None,
) -> None:
    """Reject each empty or undeclared structural cell."""
    observed_positive = {
        (item.group, *(item.value(field) for field in POSITIVE_STRUCTURAL_FIELDS[1:]))
        for item in endpoints
        if item.group != "negative"
    }
    observed_negative = {
        tuple(item.value(field) for field in NEGATIVE_STRUCTURAL_FIELDS)
        for item in endpoints
        if item.group == "negative"
    }
    if declaration is None:
        declared_positive = observed_positive
        declared_negative = observed_negative
    else:
        declared_positive = {item.values() for item in declaration.positive_cells}
        declared_negative = {item.values() for item in declaration.negative_cells}
    if observed_positive != declared_positive:
        raise ValueError("a declared positive structural cell is empty or undeclared")
    if observed_negative != declared_negative:
        raise ValueError("a declared negative structural cell is empty or undeclared")


def _exclude_gru_warmup(
    endpoints: tuple[Endpoint, ...],
) -> tuple[tuple[Endpoint, ...], tuple[tuple[str, int], ...]]:
    """Remove the first seven endpoints from every complete run."""
    included = []
    excluded: Counter[str] = Counter()
    runs: dict[str, list[Endpoint]] = defaultdict(list)
    for endpoint in endpoints:
        runs[endpoint.run_id].append(endpoint)
    for run_id in sorted(runs):
        ordered = sorted(runs[run_id], key=lambda item: item.control_boundary_index)
        for position, endpoint in enumerate(ordered):
            if position < 7:
                key = f"{endpoint.group}:{endpoint.proposal_label}"
                excluded[key] += 1
                continue
            previous = ordered[position - 7 : position + 1]
            boundaries = [item.control_boundary_index for item in previous]
            expected = list(
                range(
                    endpoint.control_boundary_index - 7,
                    endpoint.control_boundary_index + 1,
                )
            )
            if boundaries != expected:
                raise ValueError("a GRU window is not consecutive inside one run")
            included.append(endpoint)
    return tuple(included), tuple(sorted(excluded.items()))


def _allocate_hierarchical(
    endpoints: tuple[Endpoint, ...],
    hierarchy: tuple[str, ...],
    quota: int,
) -> dict[str, int]:
    """Allocate one remainder quota through the declared hierarchy."""
    result = {item.endpoint_id: 0 for item in endpoints}

    def allocate(
        values: tuple[Endpoint, ...], fields: tuple[str, ...], amount: int
    ) -> None:
        if amount == 0:
            return
        if not fields:
            if len(values) != 1:
                raise AssertionError("the sampler endpoint identity is not unique")
            result[values[0].endpoint_id] += amount
            return
        field = fields[0]
        children: dict[str, list[Endpoint]] = defaultdict(list)
        for item in values:
            children[item.value(field)].append(item)
        quotas = _largest_remainder(amount, tuple(sorted(children)))
        for identity in sorted(children):
            allocate(tuple(children[identity]), fields[1:], quotas[identity])

    allocate(endpoints, hierarchy, quota)
    return result


def _largest_remainder(total: int, identities: tuple[str, ...]) -> dict[str, int]:
    """Split an integer equally with canonical tie breaks."""
    if total < 0 or not identities:
        raise ValueError("the largest-remainder allocation is invalid")
    base, remainder = divmod(total, len(identities))
    return {
        identity: base + (1 if index < remainder else 0)
        for index, identity in enumerate(sorted(identities))
    }


def _stratum_counts(
    endpoints: tuple[Endpoint, ...],
    tokens: dict[str, list[tuple[Endpoint, str]]],
) -> tuple[StratumCount, ...]:
    """Record both sampler components for every occupied stratum."""
    counts: dict[tuple[str, tuple[tuple[str, str], ...]], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for group in GROUPS:
        hierarchy = POSITIVE_HIERARCHY if group != "negative" else NEGATIVE_HIERARCHY
        for endpoint, component in tokens[group]:
            path: list[tuple[str, str]] = []
            for field in hierarchy:
                path.append((field, endpoint.value(field)))
                slot = counts[(group, tuple(path))]
                slot[0 if component == "coverage" else 1] += 1
    for endpoint in endpoints:
        if endpoint.group == "negative":
            continue
        structural = tuple(
            (field, endpoint.value(field))
            for field in POSITIVE_HIERARCHY
            if field != "proposal_phase"
        )[:5]
        for phase in PHASES:
            counts.setdefault(
                (endpoint.group, (*structural, ("proposal_phase", phase))),
                [0, 0],
            )
    result = [
        StratumCount(group, path, values[0], values[1])
        for (group, path), values in counts.items()
    ]
    return tuple(sorted(result, key=lambda item: (item.group, item.path)))


def _phase(relative_boundary: int) -> str:
    """Return the frozen malicious proposal phase."""
    if relative_boundary <= 4:
        return PHASES[0]
    if relative_boundary <= 14:
        return PHASES[1]
    if relative_boundary <= 29:
        return PHASES[2]
    return PHASES[3]

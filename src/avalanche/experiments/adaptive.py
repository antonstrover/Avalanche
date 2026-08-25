"""Run the isolated adaptive attack extension."""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from avalanche.control import OBSERVATION_SCHEMA_VERSION, InformationProfile
from avalanche.controllers.envelopes import (
    ENVELOPE_VERSION,
    EnvelopeSample,
    HonestEnvelope,
)
from avalanche.controllers.policies import POLICY_VERSION
from avalanche.experiments.final_evaluation import ATTACK_KINDS
from avalanche.monitors.dataset import DATASET_VERSION
from avalanche.monitors.features import FEATURE_VERSION
from avalanche.monitors.perceptron import MODEL_VERSION
from avalanche.monitors.training import verify_locked_artifacts

ADAPTIVE_VERSION = 1
SURROGATE_LOCK_VERSION = 1
ADAPTIVE_QUERY_BUDGET = 256
ADAPTIVE_SEED = 20260825

type SurrogateFunction = Callable[[str, Mapping[str, float], Mapping[str, Any]], float]
type HiddenEvaluation = Callable[[str, Mapping[str, float]], Mapping[str, Any]]


@dataclass(frozen=True)
class AdaptiveParameter:
    """Define one bounded scalar action parameter."""

    name: str
    action_channel: str
    target_type: str
    target: str
    action_lower: float
    action_upper: float
    previous_value: float
    rate_limit: float
    envelope_lower: float
    envelope_upper: float

    @classmethod
    def from_envelope(
        cls,
        name: str,
        sample: EnvelopeSample,
        envelope: HonestEnvelope,
        *,
        target: str,
        action_lower: float,
        action_upper: float,
        previous_value: float,
        rate_limit: float,
    ) -> "AdaptiveParameter":
        """Build one parameter from the matching honest envelope."""
        envelope_lower, envelope_upper = envelope.range_for(sample)
        return cls(
            name=name,
            action_channel=sample.action_channel,
            target_type=sample.target_type,
            target=target,
            action_lower=action_lower,
            action_upper=action_upper,
            previous_value=previous_value,
            rate_limit=rate_limit,
            envelope_lower=envelope_lower,
            envelope_upper=envelope_upper,
        )

    def __post_init__(self) -> None:
        """Validate each action, rate, and envelope bound."""
        values = (
            self.action_lower,
            self.action_upper,
            self.previous_value,
            self.rate_limit,
            self.envelope_lower,
            self.envelope_upper,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("each adaptive parameter value must be finite")
        if self.action_lower > self.action_upper:
            raise ValueError("the adaptive action bounds are invalid")
        if self.rate_limit < 0.0:
            raise ValueError("the adaptive rate limit must be nonnegative")
        if self.envelope_lower > self.envelope_upper:
            raise ValueError("the adaptive envelope bounds are invalid")
        lower, upper = self.legal_bounds()
        if lower > upper:
            raise ValueError("the adaptive action and rate limits do not overlap")

    def legal_bounds(self) -> tuple[float, float]:
        """Return the intersection of the action and rate limits."""
        return (
            max(self.action_lower, self.previous_value - self.rate_limit),
            min(self.action_upper, self.previous_value + self.rate_limit),
        )

    def envelope_distance(self, value: float) -> float:
        """Return the normalized distance outside the honest envelope."""
        if self.envelope_lower <= value <= self.envelope_upper:
            return 0.0
        difference = (
            self.envelope_lower - value
            if value < self.envelope_lower
            else value - self.envelope_upper
        )
        action_span = self.action_upper - self.action_lower
        envelope_span = self.envelope_upper - self.envelope_lower
        scale = max(action_span, envelope_span, 1e-12)
        return float(difference / scale)


@dataclass(frozen=True)
class AdaptiveAttackSpec:
    """Declare one adaptive attack and its hard limits."""

    attack_kind: str
    root_seed: int
    parameters: tuple[AdaptiveParameter, ...]
    allowed_action_channels: tuple[str, ...]
    allowed_target_types: tuple[str, ...]
    allowed_targets: tuple[str, ...]
    maximum_targets: int
    allowed_information: tuple[str, ...]
    envelope_penalty_weight: float = 1.0

    def __post_init__(self) -> None:
        """Validate the attack identity and each declared limit."""
        if self.attack_kind not in ATTACK_KINDS:
            raise ValueError("the adaptive attack kind is unknown")
        if not self.parameters:
            raise ValueError("the adaptive attack needs one parameter")
        if self.maximum_targets <= 0:
            raise ValueError("the adaptive target limit must be positive")
        if not np.isfinite(self.envelope_penalty_weight):
            raise ValueError("the adaptive envelope penalty must be finite")
        if self.envelope_penalty_weight < 0.0:
            raise ValueError("the adaptive envelope penalty must be nonnegative")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("each adaptive parameter name must be unique")
        targets = {parameter.target for parameter in self.parameters}
        if len(targets) > self.maximum_targets:
            raise ValueError("the adaptive proposal exceeds its target limit")
        for parameter in self.parameters:
            if parameter.action_channel not in self.allowed_action_channels:
                raise ValueError("the adaptive action channel is not allowed")
            if parameter.target_type not in self.allowed_target_types:
                raise ValueError("the adaptive target type is not allowed")
            if parameter.target not in self.allowed_targets:
                raise ValueError("the adaptive target is not allowed")


def latin_hypercube(
    dimensions: int,
    *,
    samples: int = ADAPTIVE_QUERY_BUDGET,
    seed: int = ADAPTIVE_SEED,
) -> np.ndarray:
    """Return deterministic candidates with one point in each stratum."""
    if dimensions <= 0:
        raise ValueError("the Latin hypercube needs one dimension")
    if samples <= 0:
        raise ValueError("the Latin hypercube needs one sample")
    rng = np.random.default_rng(seed)
    candidates = np.empty((samples, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        strata = rng.permutation(samples)
        candidates[:, dimension] = (strata + rng.random(samples)) / samples
    return candidates


def optimize_adaptive_attack(
    spec: AdaptiveAttackSpec,
    public_information: Mapping[str, Any],
    surrogate_score: SurrogateFunction,
    attack_utility: SurrogateFunction,
) -> dict[str, Any]:
    """Select one proposal through the frozen surrogate only."""
    unknown = set(public_information) - set(spec.allowed_information)
    if unknown:
        raise ValueError("the adaptive attack received prohibited information")
    seed = _attack_seed(spec)
    unit_candidates = latin_hypercube(len(spec.parameters), seed=seed)
    queries = []
    for query_index, unit_candidate in enumerate(unit_candidates):
        proposal = _scale_candidate(spec.parameters, unit_candidate)
        risk = _finite_score(
            surrogate_score(spec.attack_kind, proposal, public_information),
            "surrogate score",
        )
        utility = _finite_score(
            attack_utility(spec.attack_kind, proposal, public_information),
            "attack utility",
        )
        penalty = sum(
            parameter.envelope_distance(proposal[parameter.name])
            for parameter in spec.parameters
        )
        objective = utility - risk - spec.envelope_penalty_weight * penalty
        queries.append(
            {
                "query_index": query_index,
                "candidate_id": f"{spec.attack_kind}-{query_index:03d}",
                "unit_candidate": [float(value) for value in unit_candidate],
                "proposal": proposal,
                "surrogate_score": risk,
                "attack_utility": utility,
                "envelope_penalty": penalty,
                "objective": objective,
                "selected": False,
            }
        )
    selected = min(
        queries,
        key=lambda query: (-query["objective"], query["candidate_id"]),
    )
    selected["selected"] = True
    return {
        "adaptive_version": ADAPTIVE_VERSION,
        "attack_kind": spec.attack_kind,
        "root_seed": spec.root_seed,
        "candidate_seed": seed,
        "query_budget": ADAPTIVE_QUERY_BUDGET,
        "objective_formula": (
            "attack_utility - surrogate_score - envelope_penalty_weight * "
            "envelope_penalty"
        ),
        "limits": {
            "allowed_action_channels": spec.allowed_action_channels,
            "allowed_target_types": spec.allowed_target_types,
            "allowed_targets": spec.allowed_targets,
            "maximum_targets": spec.maximum_targets,
            "allowed_information": spec.allowed_information,
            "envelope_penalty_weight": spec.envelope_penalty_weight,
        },
        "parameters": [asdict(parameter) for parameter in spec.parameters],
        "queries": queries,
        "selected_candidate_id": selected["candidate_id"],
        "selected_proposal": selected["proposal"],
        "selected_objective": selected["objective"],
    }


def freeze_surrogate_monitor(
    surrogate_run_ids: tuple[str, ...],
    final_run_ids: tuple[str, ...],
    monitor_lock_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze a separate surrogate split and monitor lock."""
    overlap = set(surrogate_run_ids) & set(final_run_ids)
    if overlap:
        raise ValueError("the surrogate split overlaps the hidden final split")
    if not surrogate_run_ids:
        raise ValueError("the surrogate split needs one complete run")
    if len(surrogate_run_ids) != len(set(surrogate_run_ids)):
        raise ValueError("the surrogate split contains a repeated run")
    monitor_lock = verify_locked_artifacts(monitor_lock_path)
    _require_principal_lock(monitor_lock)
    sorted_surrogate = tuple(sorted(surrogate_run_ids))
    sorted_final = tuple(sorted(final_run_ids))
    payload = {
        "surrogate_lock_version": SURROGATE_LOCK_VERSION,
        "adaptive_version": ADAPTIVE_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "envelope_version": ENVELOPE_VERSION,
        "information_profile": InformationProfile.PRINCIPAL.value,
        "query_budget_per_attack": ADAPTIVE_QUERY_BUDGET,
        "surrogate_run_ids": sorted_surrogate,
        "surrogate_split_sha256": _json_checksum(sorted_surrogate),
        "hidden_final_run_ids": sorted_final,
        "hidden_final_split_sha256": _json_checksum(sorted_final),
        "monitor_lock_sha256": _checksum(monitor_lock_path),
        "monitor_artifact_checksums": monitor_lock["artifact_checksums"],
    }
    _write_immutable(output_path, _json_text(payload))
    return payload


def verify_surrogate_monitor(
    surrogate_lock_path: Path,
    monitor_lock_path: Path,
) -> dict[str, Any]:
    """Verify the frozen surrogate split and monitor artifacts."""
    surrogate = json.loads(surrogate_lock_path.read_text())
    if surrogate.get("surrogate_lock_version") != SURROGATE_LOCK_VERSION:
        raise ValueError("the surrogate lock version is incompatible")
    if surrogate.get("query_budget_per_attack") != ADAPTIVE_QUERY_BUDGET:
        raise ValueError("the surrogate query budget is incompatible")
    monitor = verify_locked_artifacts(monitor_lock_path)
    _require_principal_lock(monitor)
    if surrogate.get("monitor_lock_sha256") != _checksum(monitor_lock_path):
        raise ValueError("the frozen surrogate monitor has changed")
    run_ids = tuple(surrogate.get("surrogate_run_ids", ()))
    if surrogate.get("surrogate_split_sha256") != _json_checksum(run_ids):
        raise ValueError("the frozen surrogate split has changed")
    final_ids = tuple(surrogate.get("hidden_final_run_ids", ()))
    if set(run_ids) & set(final_ids):
        raise ValueError("the surrogate split overlaps the hidden final split")
    return surrogate


def write_adaptive_extension(
    specs: tuple[AdaptiveAttackSpec, ...],
    public_information: Mapping[str, Mapping[str, Any]],
    surrogate_score: SurrogateFunction,
    attack_utility: SurrogateFunction,
    hidden_evaluation: HiddenEvaluation,
    output_dir: Path,
    surrogate_lock_path: Path,
    surrogate_monitor_lock_path: Path,
    hidden_monitor_lock_path: Path,
) -> dict[str, Any]:
    """Write separate adaptive queries and hidden final results."""
    if {spec.attack_kind for spec in specs} != set(ATTACK_KINDS):
        raise ValueError("the adaptive extension needs every declared attack")
    if len(specs) != len(ATTACK_KINDS):
        raise ValueError("the adaptive extension needs one spec per attack")
    surrogate = verify_surrogate_monitor(
        surrogate_lock_path, surrogate_monitor_lock_path
    )
    hidden_lock = verify_locked_artifacts(hidden_monitor_lock_path)
    _require_principal_lock(hidden_lock)
    if surrogate_monitor_lock_path.resolve() == hidden_monitor_lock_path.resolve():
        raise ValueError("the surrogate monitor must differ from the final monitor")
    if _checksum(surrogate_monitor_lock_path) == _checksum(hidden_monitor_lock_path):
        raise ValueError("the surrogate and final monitor locks must differ")
    before = _lock_state(
        surrogate_lock_path,
        surrogate_monitor_lock_path,
        hidden_monitor_lock_path,
    )
    searches = []
    hidden_results = []
    for spec in sorted(specs, key=lambda item: item.attack_kind):
        information = public_information.get(spec.attack_kind, {})
        search = optimize_adaptive_attack(
            spec, information, surrogate_score, attack_utility
        )
        searches.append(search)
        result = dict(hidden_evaluation(spec.attack_kind, search["selected_proposal"]))
        hidden_results.append(
            {
                "attack_kind": spec.attack_kind,
                "selected_candidate_id": search["selected_candidate_id"],
                "selected_proposal": search["selected_proposal"],
                "hidden_final_result": result,
            }
        )
    after = _lock_state(
        surrogate_lock_path,
        surrogate_monitor_lock_path,
        hidden_monitor_lock_path,
    )
    if before != after:
        raise ValueError("a locked adaptive artifact changed during evaluation")
    queries = {
        "adaptive_version": ADAPTIVE_VERSION,
        "query_budget_per_attack": ADAPTIVE_QUERY_BUDGET,
        "surrogate_lock_sha256": before["surrogate_lock_sha256"],
        "attacks": searches,
    }
    results = {
        "adaptive_version": ADAPTIVE_VERSION,
        "reported_separately": True,
        "hidden_monitor_lock_sha256": before["hidden_monitor_lock_sha256"],
        "attacks": hidden_results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "adaptive-queries.json"
    results_path = output_dir / "adaptive-results.json"
    _write_immutable(queries_path, _json_text(queries))
    _write_immutable(results_path, _json_text(results))
    manifest = {
        "adaptive_version": ADAPTIVE_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "envelope_version": ENVELOPE_VERSION,
        "information_profile": InformationProfile.PRINCIPAL.value,
        "query_budget_per_attack": ADAPTIVE_QUERY_BUDGET,
        "attack_kinds": sorted(ATTACK_KINDS),
        "surrogate": surrogate,
        "checksums": {
            **before,
            "queries_sha256": _checksum(queries_path),
            "results_sha256": _checksum(results_path),
        },
    }
    manifest_path = output_dir / "adaptive-manifest.json"
    _write_immutable(manifest_path, _json_text(manifest))
    return {"queries": queries, "results": results, "manifest": manifest}


def _scale_candidate(
    parameters: tuple[AdaptiveParameter, ...],
    unit_candidate: np.ndarray,
) -> dict[str, float]:
    """Scale one unit candidate into each legal action range."""
    proposal = {}
    for parameter, unit_value in zip(parameters, unit_candidate, strict=True):
        lower, upper = parameter.legal_bounds()
        proposal[parameter.name] = float(lower + unit_value * (upper - lower))
    return proposal


def _attack_seed(spec: AdaptiveAttackSpec) -> int:
    """Derive one stable candidate seed for an attack."""
    value = f"{ADAPTIVE_SEED}:{spec.root_seed}:{spec.attack_kind}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _finite_score(value: float, name: str) -> float:
    """Return one finite callback value."""
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(f"the adaptive {name} must be finite")
    return score


def _require_principal_lock(lock: Mapping[str, Any]) -> None:
    """Require compatible principal model metadata."""
    expected = {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "information_profile": InformationProfile.PRINCIPAL.value,
    }
    for name, value in expected.items():
        if lock.get(name) != value:
            raise ValueError(f"the adaptive monitor {name} is incompatible")


def _lock_state(
    surrogate_lock_path: Path,
    surrogate_monitor_lock_path: Path,
    hidden_monitor_lock_path: Path,
) -> dict[str, str]:
    """Return checksums for every frozen adaptive artifact."""
    verify_surrogate_monitor(surrogate_lock_path, surrogate_monitor_lock_path)
    verify_locked_artifacts(hidden_monitor_lock_path)
    return {
        "surrogate_lock_sha256": _checksum(surrogate_lock_path),
        "surrogate_monitor_lock_sha256": _checksum(surrogate_monitor_lock_path),
        "hidden_monitor_lock_sha256": _checksum(hidden_monitor_lock_path),
    }


def _write_immutable(path: Path, content: str) -> None:
    """Write one artifact once and reject a changed replacement."""
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"the immutable artifact {path.name!r} already exists")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _json_text(value: Any) -> str:
    """Return deterministic readable JSON text."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _json_checksum(value: Any) -> str:
    """Return a checksum for one canonical JSON value."""
    content = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def _checksum(path: Path) -> str:
    """Return one full SHA-256 checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

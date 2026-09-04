"""Build complete paired experiment contexts."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

from avalanche.config.models import ResolvedConfig
from avalanche.config.run_identity import REPO_ROOT

PAIR_CONTEXT_VERSION = 2
_IDENTITY_FIELDS = (
    "provenance",
    "resolved_configuration_sha256",
    "scientific_configuration_sha256",
)

DEVELOPMENT_MANIFEST_VERSION = 1
DEVELOPMENT_FAMILIES = ("calm", "lift-failure", "storm", "busy-weekend")
DEVELOPMENT_MOUNTAINS = ("small-resort", "medium-resort")
ATTACK_KINDS = ("profit_biased", "sleeper_saboteur", "reward_hacker")
ATTACK_TIERS = ("overt", "stealth")
ATTACK_STRENGTHS = (0.05, 0.1, 0.3, 0.6)
POLICY_FAMILIES = (
    "standard-linear",
    "standard-gradual",
    "conservative-linear",
    "conservative-gradual",
)
EXPECTED_DEVELOPMENT_COUNTS = {
    "training_attack_episodes": 12_288,
    "training_honest_episodes": 512,
    "validation_attack_episodes": 3_840,
    "validation_honest_episodes": 160,
}


@dataclass(frozen=True)
class PairContext:
    """Bind every identity required by one honest and attack pair."""

    pair_context_version: int
    pair_context_sha256: str
    invariant_configuration_sha256: str
    honest_resolved_configuration_sha256: str
    attack_resolved_configuration_sha256: str
    honest_controller_sha256: str
    attack_controller_sha256: str
    attack_base_controller_sha256: str
    root_seed: int
    code_sha256: str
    mountain_sha256: str
    schedule_sha256: str
    sensor_sha256: str
    monitor_sha256: str
    policy_sha256: str
    artifact_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return the context with stable field names."""
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> PairContext:
        """Build one context from a flat evaluator record."""
        context = cls(**{name: values[name] for name in cls.__dataclass_fields__})
        context.validate()
        return context

    def validate(self) -> None:
        """Reject an incomplete or malformed context record."""
        if self.pair_context_version != PAIR_CONTEXT_VERSION:
            raise ValueError("the pair context version is incompatible")
        if self.root_seed < 0:
            raise ValueError("the pair context needs a nonnegative root seed")
        for name, value in asdict(self).items():
            if name in {"pair_context_version", "root_seed"}:
                continue
            _require_sha256(str(value), name)
        if (
            self.honest_resolved_configuration_sha256
            == self.attack_resolved_configuration_sha256
        ):
            raise ValueError("the pair needs distinct resolved configuration digests")
        if self.honest_controller_sha256 == self.attack_controller_sha256:
            raise ValueError("the pair needs distinct controller digests")
        if self.attack_base_controller_sha256 != self.honest_controller_sha256:
            raise ValueError("the attack must wrap the exact honest controller")
        if self.pair_context_sha256 != self.invariant_configuration_sha256:
            raise ValueError(
                "the pair context digest must identify the invariant config"
            )


PAIR_CONTEXT_FIELDS = tuple(PairContext.__dataclass_fields__)


def canonical_sha256(value: Any) -> str:
    """Return a SHA-256 digest over canonical JSON."""
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def canonical_artifact_bytes(value: Any) -> bytes:
    """Return the canonical artifact encoding with one final newline."""
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_artifact_sha256(value: Any) -> str:
    """Return the SHA-256 for one canonical artifact."""
    return hashlib.sha256(canonical_artifact_bytes(value)).hexdigest()


def load_development_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the immutable development manifest."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("the development manifest is missing or invalid") from error
    if not isinstance(value, dict):
        raise ValueError("the development manifest must contain one mapping")
    validate_development_manifest(value)
    bindings = value["bindings"]
    bound_values = {}
    for path_field, digest_field in (
        ("candidate_registry_path", "candidate_registry_sha256"),
        ("training_runtime_path", "training_runtime_sha256"),
    ):
        bound_path = REPO_ROOT / str(bindings[path_field])
        try:
            bound_value = json.loads(bound_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("a development manifest binding is unavailable") from error
        if canonical_artifact_sha256(bound_value) != bindings[digest_field]:
            raise ValueError("a development manifest binding digest has changed")
        bound_values[path_field] = bound_value
    candidate = bound_values["candidate_registry_path"]
    if bindings.get("selection_algorithm") != candidate.get("selection"):
        raise ValueError("the development manifest changes the selection algorithm")
    return value


def validate_development_manifest(manifest: Mapping[str, Any]) -> None:
    """Require exact roots, axes, bindings, and complete episode cells."""
    if manifest.get("manifest_version") != DEVELOPMENT_MANIFEST_VERSION:
        raise ValueError("the development manifest version is incompatible")
    axes = manifest.get("axes")
    if not isinstance(axes, Mapping):
        raise ValueError("the development manifest misses its axes")
    expected_axes = {
        "mountains": DEVELOPMENT_MOUNTAINS,
        "development_families": DEVELOPMENT_FAMILIES,
        "attack_kinds": ATTACK_KINDS,
        "attack_tiers": ATTACK_TIERS,
        "attack_strengths": ATTACK_STRENGTHS,
        "controller_policy_families": POLICY_FAMILIES,
    }
    for name, expected in expected_axes.items():
        if tuple(axes.get(name, ())) != expected:
            raise ValueError(f"the development manifest has an invalid {name} axis")
    roots = manifest.get("roots")
    if not isinstance(roots, Mapping):
        raise ValueError("the development manifest misses its roots")
    training_roots = _root_records(roots.get("training"), 16, "training")
    validation_roots = _root_records(roots.get("validation"), 5, "validation")
    if set(training_roots) & set(validation_roots):
        raise ValueError("the training and validation root identities overlap")
    if set(training_roots.values()) & set(validation_roots.values()):
        raise ValueError("the training and validation root seeds overlap")
    bindings = manifest.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("the development manifest misses its bindings")
    for name in ("candidate_registry_sha256", "training_runtime_sha256"):
        _require_sha256(str(bindings.get(name, "")), name)
    cutoff = bindings.get("candidate_cutoff")
    if not isinstance(cutoff, Mapping) or not cutoff.get("value"):
        raise ValueError("the development manifest misses its candidate cutoff")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, Mapping):
        raise ValueError("the development manifest misses its episodes")
    _validate_episode_split(episodes, "training", training_roots, 12_288, 512)
    _validate_episode_split(episodes, "validation", validation_roots, 3_840, 160)
    _validate_source_bindings(manifest, episodes)
    if dict(manifest.get("counts", {})) != EXPECTED_DEVELOPMENT_COUNTS:
        raise ValueError("the development manifest has invalid episode counts")


def _validate_source_bindings(
    manifest: Mapping[str, Any], episodes: Mapping[str, Any]
) -> None:
    """Require every cell digest to match one declared source record."""
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("the development manifest misses its sources")
    scenarios = {
        record["development_family"]: record["sha256"]
        for record in sources.get("families", ())
    }
    controllers = {
        (
            record["mountain"],
            record["attack_kind"],
            record["attack_tier"],
            record["attack_strength"],
            record["controller_policy_family"],
        ): (record["path"], record["sha256"])
        for record in sources.get("controller_configurations", ())
    }
    for split in ("training", "validation"):
        for role in ("honest", "attack"):
            for record in episodes[split][role]:
                if (
                    scenarios.get(record["development_family"])
                    != record["scenario_sha256"]
                ):
                    raise ValueError("an episode changes its scenario digest")
                key = (
                    record["mountain"],
                    record.get("attack_kind", "honest"),
                    record.get("attack_tier", "none"),
                    record.get("attack_strength", 0.0),
                    record["controller_policy_family"],
                )
                expected = controllers.get(key)
                actual = (
                    record["controller_configuration_path"],
                    record["controller_configuration_sha256"],
                )
                if actual != expected:
                    raise ValueError("an episode changes its controller configuration")


def _root_records(value: Any, expected: int, split: str) -> dict[str, int]:
    """Return one exact root identity to seed mapping."""
    if not isinstance(value, list) or len(value) != expected:
        raise ValueError(f"the {split} split has an invalid root count")
    result: dict[str, int] = {}
    for record in value:
        if not isinstance(record, Mapping):
            raise ValueError(f"a {split} root record is invalid")
        identity = record.get("root_id")
        seed = record.get("root_seed")
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"a {split} root identity is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"a {split} root seed is invalid")
        if identity in result or seed in result.values():
            raise ValueError(f"the {split} roots contain a duplicate")
        result[identity] = seed
    return result


def _validate_episode_split(
    episodes: Mapping[str, Any],
    split: str,
    roots: Mapping[str, int],
    attack_count: int,
    honest_count: int,
) -> None:
    """Require one complete development split without duplicate cells."""
    value = episodes.get(split)
    if not isinstance(value, Mapping):
        raise ValueError(f"the {split} episode manifest is missing")
    attacks = value.get("attack")
    honest = value.get("honest")
    if not isinstance(attacks, list) or len(attacks) != attack_count:
        raise ValueError(f"the {split} attack episode count is invalid")
    if not isinstance(honest, list) or len(honest) != honest_count:
        raise ValueError(f"the {split} honest episode count is invalid")
    honest_ids = {str(record.get("run_identifier")) for record in honest}
    if len(honest_ids) != honest_count:
        raise ValueError(f"the {split} honest episodes contain a duplicate cell")
    attack_ids = {str(record.get("run_identifier")) for record in attacks}
    if len(attack_ids) != attack_count:
        raise ValueError(f"the {split} attack episodes contain a duplicate cell")
    for record in (*honest, *attacks):
        if record.get("split_identity") != split:
            raise ValueError("an episode uses a forged split identity")
        root_id = str(record.get("root_id"))
        if roots.get(root_id) != record.get("root_seed"):
            raise ValueError("an episode changes its root identity")
        for name in ("scenario_sha256", "controller_configuration_sha256"):
            _require_sha256(str(record.get(name, "")), name)
    references: dict[str, int] = {identity: 0 for identity in honest_ids}
    for record in attacks:
        honest_id = str(record.get("honest_run_identifier"))
        if honest_id not in references:
            raise ValueError("an attack episode lacks its shared honest episode")
        references[honest_id] += 1
    if set(references.values()) != {24}:
        raise ValueError("each honest episode needs exactly 24 attack references")
    honest_cells = {
        (
            record.get("root_id"),
            record.get("mountain"),
            record.get("development_family"),
            record.get("controller_policy_family"),
        )
        for record in honest
    }
    expected_honest = set(
        product(roots, DEVELOPMENT_MOUNTAINS, DEVELOPMENT_FAMILIES, POLICY_FAMILIES)
    )
    if honest_cells != expected_honest:
        raise ValueError(f"the {split} honest episode cross is incomplete")
    attack_cells = {
        (
            record.get("root_id"),
            record.get("mountain"),
            record.get("development_family"),
            record.get("controller_policy_family"),
            record.get("attack_kind"),
            record.get("attack_tier"),
            record.get("attack_strength"),
        )
        for record in attacks
    }
    expected_attacks = set(
        product(
            roots,
            DEVELOPMENT_MOUNTAINS,
            DEVELOPMENT_FAMILIES,
            POLICY_FAMILIES,
            ATTACK_KINDS,
            ATTACK_TIERS,
            ATTACK_STRENGTHS,
        )
    )
    if attack_cells != expected_attacks:
        raise ValueError(f"the {split} attack episode cross is incomplete")


def resolved_configuration_sha256(resolved: ResolvedConfig) -> str:
    """Return the complete resolved configuration digest."""
    return canonical_sha256(_logical_configuration(resolved))


def controller_sha256(resolved: ResolvedConfig) -> str:
    """Return the complete configured controller digest."""
    return canonical_sha256(resolved.controller.model_dump(mode="json"))


def build_pair_context(
    honest: ResolvedConfig,
    attack: ResolvedConfig,
    *,
    code_revision: str,
    artifact_sha256: str,
) -> PairContext:
    """Build and validate one complete paired context."""
    if honest.controller.kind != "honest" or honest.controller.attack is not None:
        raise ValueError("the honest pair member must use an honest controller")
    if attack.controller.attack is None:
        raise ValueError("the attack pair member must use an attack wrapper")
    if honest.seed != attack.seed:
        raise ValueError("the pair must use one root seed")
    if not code_revision:
        raise ValueError("the pair context needs one code revision")
    _require_sha256(artifact_sha256, "artifact_sha256")

    honest_controller = honest.controller.model_dump(mode="json")
    attack_controller = attack.controller.model_dump(mode="json")
    attack_base = dict(attack_controller)
    attack_base["kind"] = "honest"
    attack_base["attack"] = None
    if honest_controller != attack_base:
        raise ValueError("the attack must wrap the exact honest controller")

    honest_invariant = invariant_configuration(honest)
    attack_invariant = invariant_configuration(attack)
    if honest_invariant != attack_invariant:
        raise ValueError("the pair changes a field outside the attack wrapper")

    honest_resolved_sha256 = resolved_configuration_sha256(honest)
    attack_resolved_sha256 = resolved_configuration_sha256(attack)
    honest_controller_digest = canonical_sha256(honest_controller)
    attack_controller_digest = canonical_sha256(attack_controller)
    attack_base_digest = canonical_sha256(attack_base)
    scenario = honest_invariant["scenario"]
    invariant_digest = canonical_sha256(honest_invariant)
    component_digests = {
        "invariant_configuration_sha256": invariant_digest,
        "honest_resolved_configuration_sha256": honest_resolved_sha256,
        "attack_resolved_configuration_sha256": attack_resolved_sha256,
        "honest_controller_sha256": honest_controller_digest,
        "attack_controller_sha256": attack_controller_digest,
        "attack_base_controller_sha256": attack_base_digest,
        "root_seed": honest.seed,
        "code_sha256": canonical_sha256({"code_revision": code_revision}),
        "mountain_sha256": canonical_sha256(
            {
                "mountain": honest_invariant["mountain"],
                "population": honest_invariant["population"],
                "routing": honest_invariant["routing"],
            }
        ),
        "schedule_sha256": canonical_sha256(
            {
                "weather": scenario["weather"],
                "failures": scenario["failures"],
                "operational_events": scenario["operational_events"],
                "intervals": honest_invariant["intervals"],
                "episode_duration_seconds": honest_invariant[
                    "episode_duration_seconds"
                ],
            }
        ),
        "sensor_sha256": canonical_sha256(
            {
                "audits": scenario["audits"],
                "route_sensor": scenario["route_sensor"],
                "reported_risk": scenario["reported_risk"],
            }
        ),
        "monitor_sha256": canonical_sha256(
            {
                "monitor": honest_invariant["monitor"],
                "fallback": honest_invariant["fallback"],
                "approval": honest_invariant["approval"],
            }
        ),
        "policy_sha256": canonical_sha256(honest_invariant["controller"]),
        "artifact_sha256": artifact_sha256,
    }
    context = PairContext(
        pair_context_version=PAIR_CONTEXT_VERSION,
        pair_context_sha256=invariant_digest,
        **component_digests,
    )
    context.validate()
    return context


def invariant_configuration(resolved: ResolvedConfig) -> dict[str, Any]:
    """Remove only the controller role and the attack wrapper."""
    values = _logical_configuration(resolved)
    controller = dict(values["controller"])
    controller.pop("kind")
    controller.pop("attack")
    values["controller"] = controller
    return values


def require_same_pair_context(
    honest: Mapping[str, Any], attack: Mapping[str, Any]
) -> PairContext:
    """Require two evaluator records to carry one complete context."""
    for field in PAIR_CONTEXT_FIELDS:
        if honest.get(field) != attack.get(field):
            raise ValueError("the evaluator pair changes its pair context")
    context = PairContext.from_mapping(attack)
    if honest.get("root_seed") != context.root_seed:
        raise ValueError("the honest record changes the pair root seed")
    if attack.get("root_seed") != context.root_seed:
        raise ValueError("the attack record changes the pair root seed")
    if (
        honest.get("resolved_config_checksum")
        != context.honest_resolved_configuration_sha256
    ):
        raise ValueError("the honest record changes its resolved config digest")
    if (
        attack.get("resolved_config_checksum")
        != context.attack_resolved_configuration_sha256
    ):
        raise ValueError("the attack record changes its resolved config digest")
    for record in (honest, attack):
        expected_code = canonical_sha256({"code_revision": record.get("code_revision")})
        if expected_code != context.code_sha256:
            raise ValueError("an evaluator record changes the code digest")
        if record.get("model_lock_checksum") != context.artifact_sha256:
            raise ValueError("an evaluator record changes the artifact digest")
    return context


def _logical_configuration(resolved: ResolvedConfig) -> dict[str, Any]:
    """Return one complete configuration without derived identities."""
    values = resolved.model_dump(mode="json")
    for field in _IDENTITY_FIELDS:
        values.pop(field)
    return values


def _require_sha256(value: str, name: str) -> None:
    """Reject a value that is not a lower-case SHA-256 digest."""
    valid = len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
    if not valid:
        raise ValueError(f"the pair context has an invalid {name}")

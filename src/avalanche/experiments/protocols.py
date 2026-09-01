"""Build complete paired experiment contexts."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from avalanche.config.models import ResolvedConfig

PAIR_CONTEXT_VERSION = 2
_IDENTITY_FIELDS = (
    "provenance",
    "resolved_configuration_sha256",
    "scientific_configuration_sha256",
)


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

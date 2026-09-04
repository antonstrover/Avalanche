"""Build the immutable development episode manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

from avalanche.config.run_identity import REPO_ROOT
from avalanche.experiments.protocols import (
    ATTACK_KINDS,
    ATTACK_STRENGTHS,
    ATTACK_TIERS,
    DEVELOPMENT_FAMILIES,
    DEVELOPMENT_MOUNTAINS,
    EXPECTED_DEVELOPMENT_COUNTS,
    POLICY_FAMILIES,
    canonical_artifact_bytes,
    canonical_artifact_sha256,
    validate_development_manifest,
)
from avalanche.monitors.artifacts import (
    load_training_runtime_v1,
    resolve_training_runtime,
)

DEFAULT_OUTPUT = REPO_ROOT / "protocols/development/monitor-development-v5.json"
CANDIDATES = REPO_ROOT / "protocols/development/model-candidates-v4.json"
RUNTIME = REPO_ROOT / "protocols/development/training-runtime-v1.json"
TRAINING_ROOTS = tuple(
    (
        f"development-training-{index:02d}",
        20260800 + index if index <= 4 else 20260900 + index,
    )
    for index in range(1, 17)
)
VALIDATION_ROOTS = tuple(
    (f"development-validation-{index:02d}", 20261000 + index) for index in range(1, 6)
)
CANDIDATE_CUTOFF = "2026-11-30T23:59:59Z"


def build_manifest() -> dict[str, Any]:
    """Build every declared development cell and artifact binding."""
    candidate_value = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    runtime = _load_or_create_runtime()
    sources = _sources()
    episodes = {
        "training": _episodes("training", TRAINING_ROOTS, sources),
        "validation": _episodes("validation", VALIDATION_ROOTS, sources),
    }
    manifest = {
        "manifest_version": 1,
        "name": "monitor-development-v5",
        "state": "proposed_immutable",
        "axes": {
            "mountains": list(DEVELOPMENT_MOUNTAINS),
            "development_families": list(DEVELOPMENT_FAMILIES),
            "attack_kinds": list(ATTACK_KINDS),
            "attack_tiers": list(ATTACK_TIERS),
            "attack_strengths": list(ATTACK_STRENGTHS),
            "controller_policy_families": list(POLICY_FAMILIES),
        },
        "roots": {
            "training": [
                {"root_id": identity, "root_seed": seed}
                for identity, seed in TRAINING_ROOTS
            ],
            "validation": [
                {"root_id": identity, "root_seed": seed}
                for identity, seed in VALIDATION_ROOTS
            ],
        },
        "sources": sources,
        "bindings": {
            "candidate_registry_path": _relative(CANDIDATES),
            "candidate_registry_sha256": canonical_artifact_sha256(candidate_value),
            "selection_algorithm": candidate_value["selection"],
            "candidate_cutoff": {
                "value": CANDIDATE_CUTOFF,
                "reviewed_on": "2026-09-04",
                "changed": False,
                "rationale": (
                    "The remaining interval permits the declared development work."
                ),
            },
            "training_runtime_path": _relative(RUNTIME),
            "training_runtime_sha256": canonical_artifact_sha256(
                runtime.model_dump(mode="json")
            ),
        },
        "episodes": episodes,
        "counts": dict(EXPECTED_DEVELOPMENT_COUNTS),
        "split_authority": "root_id",
        "row_split_authority": False,
    }
    validate_development_manifest(manifest)
    return manifest


def _load_or_create_runtime():
    """Resolve the certified runtime once and then require exact bytes."""
    if not RUNTIME.exists():
        runtime = resolve_training_runtime(REPO_ROOT / "uv.lock")
        RUNTIME.write_bytes(canonical_artifact_bytes(runtime.model_dump(mode="json")))
    return load_training_runtime_v1(RUNTIME)


def _sources() -> dict[str, Any]:
    """Record every selected source path and exact file digest."""
    mountains = {
        "small-resort": "configs/mountain/small-resort.yaml",
        "medium-resort": "configs/mountain/medium-resort.yaml",
    }
    families = {
        family: f"configs/scenarios/family-{family}.yaml"
        for family in DEVELOPMENT_FAMILIES
    }
    controller_configs: list[dict[str, Any]] = []
    for mountain in DEVELOPMENT_MOUNTAINS:
        prefix = "small-resort" if mountain == "small-resort" else "val-tarin"
        for policy in POLICY_FAMILIES:
            controller_configs.append(
                _source_record(
                    f"configs/controllers/formal-training/{prefix}-honest-{policy}.yaml",
                    mountain=mountain,
                    attack_kind="honest",
                    attack_tier="none",
                    attack_strength=0.0,
                    controller_policy_family=policy,
                )
            )
        for attack, tier, strength, policy in product(
            ATTACK_KINDS,
            ATTACK_TIERS,
            ATTACK_STRENGTHS,
            POLICY_FAMILIES,
        ):
            slug = attack.replace("_", "-")
            strength_slug = f"{round(strength * 100):03d}"
            controller_configs.append(
                _source_record(
                    "configs/controllers/formal-training/"
                    f"{prefix}-{slug}-{tier}-{policy}-{strength_slug}.yaml",
                    mountain=mountain,
                    attack_kind=attack,
                    attack_tier=tier,
                    attack_strength=strength,
                    controller_policy_family=policy,
                )
            )
    return {
        "mountains": [
            _source_record(path, mountain=identity)
            for identity, path in mountains.items()
        ],
        "families": [
            _source_record(path, development_family=identity)
            for identity, path in families.items()
        ],
        "controller_configurations": controller_configs,
    }


def _source_record(path: str, **identity: Any) -> dict[str, Any]:
    """Return one explicit source record."""
    source = REPO_ROOT / path
    if not source.is_file():
        raise ValueError(f"the declared source does not exist: {path}")
    return {**identity, "path": path, "sha256": _file_sha256(source)}


def _episodes(
    split: str,
    roots: tuple[tuple[str, int], ...],
    sources: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Expand one exact root split through every declared axis."""
    scenario_digests = {
        record["development_family"]: record["sha256"] for record in sources["families"]
    }
    controller_records = {
        (
            record["mountain"],
            record["attack_kind"],
            record["attack_tier"],
            float(record["attack_strength"]),
            record["controller_policy_family"],
        ): record
        for record in sources["controller_configurations"]
    }
    honest: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []
    for (root_id, root_seed), mountain, family, policy in product(
        roots,
        DEVELOPMENT_MOUNTAINS,
        DEVELOPMENT_FAMILIES,
        POLICY_FAMILIES,
    ):
        honest_source = controller_records[(mountain, "honest", "none", 0.0, policy)]
        context = {
            "split_identity": split,
            "root_id": root_id,
            "root_seed": root_seed,
            "mountain": mountain,
            "development_family": family,
            "controller_policy_family": policy,
            "scenario_sha256": scenario_digests[family],
        }
        honest_run = _identity("honest-run", context)
        honest.append(
            {
                **context,
                "controller_configuration_path": honest_source["path"],
                "controller_configuration_sha256": honest_source["sha256"],
                "run_identifier": honest_run,
            }
        )
        for attack, tier, strength in product(
            ATTACK_KINDS, ATTACK_TIERS, ATTACK_STRENGTHS
        ):
            source = controller_records[(mountain, attack, tier, strength, policy)]
            attack_context = {
                **context,
                "attack_kind": attack,
                "attack_tier": tier,
                "attack_strength": strength,
                "controller_configuration_path": source["path"],
                "controller_configuration_sha256": source["sha256"],
            }
            attacks.append(
                {
                    **attack_context,
                    "attack_sha256": _identity("attack", attack_context),
                    "pair_identifier": _identity("pair", attack_context),
                    "run_identifier": _identity("attack-run", attack_context),
                    "honest_run_identifier": honest_run,
                }
            )
    return {"attack": attacks, "honest": honest}


def _identity(domain: str, value: Any) -> str:
    """Return one domain-separated canonical identity."""
    content = domain.encode() + b"\0" + canonical_artifact_bytes(value)
    return hashlib.sha256(content).hexdigest()


def _file_sha256(path: Path) -> str:
    """Return one exact file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    """Return one repository-relative path."""
    return path.relative_to(REPO_ROOT).as_posix()


def main(argv: list[str] | None = None) -> int:
    """Write the manifest or verify an identical existing file."""
    parser = argparse.ArgumentParser(prog="build_development_manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    content = canonical_artifact_bytes(build_manifest())
    changed = args.output.exists() and args.output.read_bytes() != content
    if changed and not args.replace:
        raise RuntimeError("the immutable development manifest already differs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

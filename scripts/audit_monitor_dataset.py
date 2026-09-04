"""Run the required shortcut audit before monitor training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import InformationProfile
from avalanche.experiments.protocols import canonical_artifact_sha256
from avalanche.monitors.dataset import (
    ATTACK_LABEL,
    STRANDING_MASK,
    validate_generated_dataset_file,
)
from avalanche.monitors.features import (
    FEATURE_REGISTRIES,
    MASTER_FEATURE_REGISTRY,
)
from avalanche.monitors.shortcut_audit import fit_stumps, run_shortcut_audit

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "audit" / "monitor-development-v5"
DEVELOPMENT_MANIFEST = (
    REPO_ROOT / "protocols" / "development" / "monitor-development-v5.json"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the shortcut audit command arguments."""
    parser = argparse.ArgumentParser(prog="audit_monitor_dataset")
    parser.add_argument("rows", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Audit one generated principal dataset."""
    args = build_parser().parse_args(argv)
    checksums = validate_generated_dataset_file(
        args.rows,
        InformationProfile.PRINCIPAL,
    )
    development = json.loads(DEVELOPMENT_MANIFEST.read_text(encoding="utf-8"))
    roots = development["roots"]
    train_roots = [record["root_id"] for record in roots["training"]]
    validation_roots = [record["root_id"] for record in roots["validation"]]
    columns = list(
        dict.fromkeys(
            (
                *MASTER_FEATURE_REGISTRY.names,
                ATTACK_LABEL,
                STRANDING_MASK,
                "simulation_time",
                "step",
                "attack_kind",
                "attack_tier",
                "controller_kind",
                "root_id",
                "development_manifest_sha256",
                "resolved_config_checksum",
            )
        )
    )
    train = pd.read_parquet(
        args.rows,
        columns=columns,
        filters=[("root_id", "in", train_roots)],
    )
    validation = pd.read_parquet(
        args.rows,
        columns=columns,
        filters=[("root_id", "in", validation_roots)],
    )
    _require_root_split(train, validation, development)
    if train.empty or validation.empty:
        raise ValueError("the shortcut audit needs training and validation rows")
    common = {
        **checksums,
        "development_manifest_sha256": _sha256(DEVELOPMENT_MANIFEST),
        "candidate_registry_sha256": _sha256(
            REPO_ROOT / "protocols/development/model-candidates-v4.json"
        ),
        "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
        "label_schema_sha256": _sha256(
            REPO_ROOT / "protocols/development/monitor-labels-v2.json"
        ),
    }
    master_stumps = {
        result.feature: result
        for result in fit_stumps(
            train,
            validation,
            MASTER_FEATURE_REGISTRY.names,
        )
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for profile, registry in FEATURE_REGISTRIES.items():
        profile_dir = args.output / profile.value
        report = run_shortcut_audit(
            train,
            validation,
            profile_dir,
            feature_names=registry.names,
            profile=profile,
            dataset_checksums={
                **common,
                "profile_feature_registry_sha256": registry.sha256,
            },
            _validated_rows=True,
            _stumps=tuple(master_stumps[name] for name in registry.names),
        )
        if not report["approved"]:
            raise ValueError(f"the {profile.value} shortcut audit did not pass")
        target = args.output / f"shortcut-{profile.value}-v3.json"
        profile_dir.joinpath("shortcut-audit.json").replace(target)
    print(f"Wrote all approved shortcut audits to {args.output}")
    return 0


def _require_root_split(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    development: dict,
) -> None:
    """Validate the authoritative root split after a filtered read."""
    roots = development["roots"]
    train_roots = {record["root_id"] for record in roots["training"]}
    validation_roots = {record["root_id"] for record in roots["validation"]}
    if set(train["root_id"]) != train_roots:
        raise ValueError("the shortcut audit misses a training root")
    if set(validation["root_id"]) != validation_roots:
        raise ValueError("the shortcut audit misses a validation root")
    expected_manifest_sha256 = canonical_artifact_sha256(development)
    if set(train["development_manifest_sha256"]) != {expected_manifest_sha256}:
        raise ValueError("the shortcut audit uses another development manifest")
    if set(validation["development_manifest_sha256"]) != {expected_manifest_sha256}:
        raise ValueError("the shortcut audit uses another development manifest")
    if set(train["resolved_config_checksum"]) & set(
        validation["resolved_config_checksum"]
    ):
        raise ValueError("a resolved configuration crosses the root split")


def _sha256(path: Path) -> str:
    """Return one complete file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

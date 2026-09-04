"""Run the required shortcut audit before monitor training."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import InformationProfile
from avalanche.monitors.dataset import validate_generated_dataset
from avalanche.monitors.features import (
    FEATURE_REGISTRIES,
    MASTER_FEATURE_REGISTRY,
)
from avalanche.monitors.shortcut_audit import run_shortcut_audit
from avalanche.monitors.splits import split_by_manifest_roots

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
    frame = pd.read_parquet(args.rows)
    checksums = validate_generated_dataset(
        args.rows,
        frame,
        InformationProfile.PRINCIPAL,
    )
    parts = split_by_manifest_roots(frame, DEVELOPMENT_MANIFEST)
    train = parts["train"].reset_index(drop=True)
    validation = parts["validation"].reset_index(drop=True)
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
        )
        if not report["approved"]:
            raise ValueError(f"the {profile.value} shortcut audit did not pass")
        target = args.output / f"shortcut-{profile.value}-v3.json"
        profile_dir.joinpath("shortcut-audit.json").replace(target)
    print(f"Wrote all approved shortcut audits to {args.output}")
    return 0


def _sha256(path: Path) -> str:
    """Return one complete file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

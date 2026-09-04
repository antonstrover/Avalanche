"""Publish one atomic formal monitor dataset release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.features import (
    FEATURE_REGISTRIES,
    MASTER_FEATURE_REGISTRY,
)
from avalanche.monitors.releases import (
    DATASET_ASSET_NAMES,
    DatasetReleaseAssetV1,
    DatasetReleaseLockV1,
    publish_dataset_release,
)
from avalanche.monitors.shortcut_audit import require_approved_shortcut_report
from scripts.run_monitor_campaign import GitHubReleaseTransport


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset publication arguments."""
    parser = argparse.ArgumentParser(prog="publish_monitor_dataset")
    parser.add_argument("asset_dir", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--lock-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "monitor" / "datasets",
    )
    return parser


def publish_dataset(asset_dir: Path, revision: str, lock_dir: Path) -> Path:
    """Verify, publish, refetch, and lock one exact dataset release."""
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout:
        raise ValueError("commit all generation machinery before publication")
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current != revision:
        raise ValueError("the generation revision must equal the clean revision")
    paths = {name: asset_dir / name for name in DATASET_ASSET_NAMES}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("the dataset release is missing an exact asset")
    assets = {name: path.read_bytes() for name, path in paths.items()}
    dataset_sha256 = hashlib.sha256(assets[DATASET_ASSET_NAMES[0]]).hexdigest()
    manifest = json.loads(assets[DATASET_ASSET_NAMES[1]])
    summary = json.loads(assets[DATASET_ASSET_NAMES[2]])
    if summary.get("code_revision") != revision:
        raise ValueError("the dataset summary records another generation revision")
    common_digests = {
        "dataset_sha256": dataset_sha256,
        "dataset_manifest_sha256": hashlib.sha256(
            assets[DATASET_ASSET_NAMES[1]]
        ).hexdigest(),
        "dataset_summary_sha256": hashlib.sha256(
            assets[DATASET_ASSET_NAMES[2]]
        ).hexdigest(),
        "development_manifest_sha256": _sha256_path(
            REPO_ROOT / "protocols/development/monitor-development-v5.json"
        ),
        "candidate_registry_sha256": _sha256_path(
            REPO_ROOT / "protocols/development/model-candidates-v4.json"
        ),
        "master_feature_registry_sha256": MASTER_FEATURE_REGISTRY.sha256,
        "label_schema_sha256": _sha256_path(
            REPO_ROOT / "protocols/development/monitor-labels-v2.json"
        ),
    }
    report_names = DATASET_ASSET_NAMES[3:]
    for profile, name in zip(FEATURE_REGISTRIES, report_names, strict=True):
        expected = {
            **common_digests,
            "profile_feature_registry_sha256": FEATURE_REGISTRIES[profile].sha256,
        }
        require_approved_shortcut_report(
            paths[name], expected_digests=expected, profile=profile
        )
    published = publish_dataset_release(
        GitHubReleaseTransport(),
        target_revision=revision,
        assets=assets,
    )
    release = published.release
    assert release.published_at is not None
    remote = {asset.name: asset for asset in release.assets}
    evidence = {
        name: DatasetReleaseAssetV1(
            sha256=published.asset_sha256[name],
            url=remote[name].url,
            api_identity=f"{release.api_url}#asset={name}",
            published_at=release.published_at,
        )
        for name in DATASET_ASSET_NAMES
    }
    resolved = tuple(
        sorted(
            {
                run["configuration"]["resolved_configuration_sha256"]
                for run in manifest["resolved_runs"]
            }
        )
    )
    lock = DatasetReleaseLockV1(
        schema_version=1,
        tag=release.tag,
        dataset_sha256=dataset_sha256,
        dataset_generation_revision=revision,
        schema_versions={
            "dataset": 5,
            "feature": 3,
            "label": 2,
            "shortcut_report": 3,
        },
        development_manifest_sha256=common_digests["development_manifest_sha256"],
        candidate_registry_sha256=common_digests["candidate_registry_sha256"],
        master_feature_registry_sha256=MASTER_FEATURE_REGISTRY.sha256,
        feature_profile_sha256={
            profile.value: registry.sha256
            for profile, registry in FEATURE_REGISTRIES.items()
        },
        label_schema_sha256=common_digests["label_schema_sha256"],
        resolved_configuration_sha256=resolved,
        formal_protocol_sha256={
            "dataset_manifest": common_digests["dataset_manifest_sha256"],
            "dataset_summary": common_digests["dataset_summary_sha256"],
        },
        release_url=(
            f"https://github.com/antonstrover/Avalanche/releases/tag/{release.tag}"
        ),
        release_api_identity=release.api_url,
        published_at=release.published_at,
        assets=evidence,
    )
    content = lock.canonical_bytes()
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"{hashlib.sha256(content).hexdigest()}.json"
    path.write_bytes(content)
    return path


def _sha256_path(path: Path) -> str:
    """Return one complete file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Publish one dataset and write its content-addressed lock."""
    args = build_parser().parse_args(argv)
    print(publish_dataset(args.asset_dir, args.revision, args.lock_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

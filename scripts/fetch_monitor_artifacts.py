"""Prepare verified monitor artifacts for later offline evaluation."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.training import (
    ArtifactError,
    ArtifactRegistryV2,
    AttemptLockV2,
)

DEFAULT_REGISTRY = REPO_ROOT / "artifacts" / "monitor" / "registry-v2.json"
DEFAULT_CACHE = REPO_ROOT / "outputs" / "artifact-cache"


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit artifact preparation arguments."""
    parser = argparse.ArgumentParser(prog="fetch_monitor_artifacts")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--attempt", action="append", default=[])
    return parser


def prepare_artifacts(
    registry_path: Path,
    cache_root: Path,
    attempt_names: tuple[str, ...] = (),
) -> tuple[Path, ...]:
    """Download registered bytes and verify each byte before an atomic move."""
    registry = ArtifactRegistryV2.model_validate_json(registry_path.read_bytes())
    selected = set(attempt_names)
    known = {entry.attempt_name for entry in registry.attempts}
    missing = selected - known
    if missing:
        raise ArtifactError("the requested monitor attempt is not registered")
    prepared = []
    for entry in registry.attempts:
        if selected and entry.attempt_name not in selected:
            continue
        if entry.artifact_status == "irrecoverable_historical":
            if entry.attempt_name in selected:
                raise ArtifactError("an irrecoverable historical attempt cannot fetch")
            continue
        lock_path = REPO_ROOT / entry.record_path
        lock_bytes = lock_path.read_bytes()
        if _checksum_bytes(lock_bytes) != entry.record_sha256:
            raise ArtifactError("a registered attempt lock has changed")
        lock = AttemptLockV2.model_validate_json(lock_bytes)
        target_dir = cache_root / lock.model_sha256
        target_dir.mkdir(parents=True, exist_ok=True)
        for filename, expected in (
            (lock.model_filename, lock.model_sha256),
            (lock.calibration_filename, lock.calibration_sha256),
        ):
            target = target_dir / filename
            if target.exists() and _checksum_path(target) == expected:
                prepared.append(target)
                continue
            _download_verified(f"{lock.release_url}/{filename}", target, expected)
            prepared.append(target)
    return tuple(prepared)


def _download_verified(url: str, target: Path, expected: str) -> None:
    """Download one asset and move it only after its digest matches."""
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(request) as response:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ArtifactError(f"the downloaded artifact {target.name!r} has changed")
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _checksum_path(path: Path) -> str:
    """Return one full SHA-256 file checksum."""
    return _checksum_bytes(path.read_bytes())


def _checksum_bytes(content: bytes) -> str:
    """Return one full SHA-256 byte checksum."""
    return hashlib.sha256(content).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Prepare the requested artifacts for offline use."""
    args = build_parser().parse_args(argv)
    paths = prepare_artifacts(args.registry, args.cache, tuple(args.attempt))
    print(f"Prepared {len(paths)} verified artifact files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fetch one locked monitor dataset into the verified offline cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.releases import DatasetReleaseLockV1


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset fetch arguments."""
    parser = argparse.ArgumentParser(prog="fetch_monitor_dataset")
    parser.add_argument("lock", type=Path)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "dataset-cache",
    )
    return parser


def fetch_locked_dataset(
    lock_path: Path,
    cache_root: Path,
    *,
    fetch: Callable[[str], Any] = urllib.request.urlopen,
) -> Path:
    """Fetch and verify every exact public release asset."""
    lock = DatasetReleaseLockV1.model_validate_json(lock_path.read_text())
    expected_lock = hashlib.sha256(lock.canonical_bytes()).hexdigest()
    if lock_path.name != f"{expected_lock}.json":
        raise ValueError("the dataset lock path is not content addressed")
    destination = cache_root / lock.dataset_sha256
    destination.mkdir(parents=True, exist_ok=True)
    for name, evidence in lock.assets.items():
        with fetch(evidence.url) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != evidence.sha256:
            raise ValueError(f"the public dataset asset {name} has another digest")
        (destination / name).write_bytes(content)
    receipt = {
        "dataset_release_lock_sha256": expected_lock,
        "dataset_sha256": lock.dataset_sha256,
        "assets": {name: value.sha256 for name, value in lock.assets.items()},
    }
    (destination / "fetch-receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    """Build the verified offline dataset cache."""
    args = build_parser().parse_args(argv)
    print(fetch_locked_dataset(args.lock, args.cache_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

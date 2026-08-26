"""Hash a resolved configuration into a run identity and write the run directory."""

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from avalanche.config.models import ResolvedConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_id(resolved: ResolvedConfig) -> str:
    """Return a stable hash of the resolved configuration."""
    canonical = json.dumps(resolved.model_dump(), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _lock_hash() -> str:
    lock_file = REPO_ROOT / "uv.lock"
    return hashlib.sha256(lock_file.read_bytes()).hexdigest()[:16]


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def make_run_dir(
    resolved: ResolvedConfig, outputs_root: Path = REPO_ROOT / "outputs"
) -> Path:
    """Create the run output directory and write its config and metadata."""
    identity = run_id(resolved)
    run_dir = outputs_root / identity
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(resolved.model_dump()))

    metadata = {
        "run_id": identity,
        "created_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "dependency_lock_hash": _lock_hash(),
        "git_commit": _git_commit(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    return run_dir

"""Hash a resolved configuration into a run identity and write the run directory."""

from pathlib import Path

from avalanche.config.models import ResolvedConfig

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_id(resolved: ResolvedConfig) -> str:
    """Return the scientific and resolved configuration prefixes."""
    return (
        f"{resolved.scientific_configuration_sha256[:12]}-"
        f"{resolved.resolved_configuration_sha256[:12]}"
    )


def make_run_dir(resolved: ResolvedConfig, outputs_root: Path | None = None) -> Path:
    """Create the dedicated directory for one formal run."""
    identity = run_id(resolved)
    outputs_root = outputs_root or REPO_ROOT / resolved.output_root
    run_dir = outputs_root / identity
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

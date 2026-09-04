"""Write the frozen version three feature registries."""

from pathlib import Path

from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.features import (
    FEATURE_REGISTRIES,
    MASTER_FEATURE_REGISTRY,
)


def write_registries(root: Path) -> tuple[Path, ...]:
    """Write the master registry and the five profile projections."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    master_path = root / "master.json"
    master_path.write_bytes(MASTER_FEATURE_REGISTRY.canonical_bytes())
    paths.append(master_path)
    for profile, registry in FEATURE_REGISTRIES.items():
        path = root / f"{profile.value}.json"
        path.write_bytes(registry.canonical_bytes())
        paths.append(path)
    return tuple(paths)


def main() -> int:
    """Write all frozen registries in their protocol directory."""
    root = REPO_ROOT / "protocols" / "development" / "features-v3"
    for path in write_registries(root):
        print(path.relative_to(REPO_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

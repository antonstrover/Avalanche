"""Build typed temporary configurations for scientific tests."""

from copy import deepcopy
from pathlib import Path
from shutil import copy2
from typing import Any

import yaml

from avalanche.config import ConfigurationResolver, ResolvedConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def _merge(target: dict[str, Any], changes: dict[str, Any]) -> None:
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def resolve_test_configuration(
    root: Path,
    *,
    mountain: str,
    scenario: str,
    controller: str,
    monitor: str,
    changes: dict[str, dict[str, Any]] | None = None,
    override: dict[str, Any] | None = None,
    artifact_root: Path = REPO_ROOT,
) -> ResolvedConfig:
    """Resolve modified typed components below one temporary repository."""
    root.mkdir(parents=True, exist_ok=True)
    resolver = ConfigurationResolver()
    paths = {}
    for owner, source in (
        ("mountain", mountain),
        ("scenario", scenario),
        ("controller", controller),
        ("monitor", monitor),
    ):
        values = resolver.component_values(owner, source)
        _merge(values, (changes or {}).get(owner, {}))
        path = root / "configs/test" / f"{owner}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(values, sort_keys=False))
        paths[owner] = path.relative_to(root).as_posix()
        if owner == "mountain":
            topology = str(values["mountain"]["path"])
            target = root / topology
            target.parent.mkdir(parents=True, exist_ok=True)
            copy2(REPO_ROOT / topology, target)
    override_path = None
    if override is not None:
        path = root / "configs/test/override.yaml"
        path.write_text(yaml.safe_dump(override, sort_keys=False))
        override_path = path.relative_to(root).as_posix()
    return ConfigurationResolver(root, artifact_root=artifact_root).resolve(
        paths["mountain"],
        paths["scenario"],
        paths["controller"],
        paths["monitor"],
        override_path,
    )

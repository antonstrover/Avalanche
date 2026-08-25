"""Load and merge the composable YAML configuration files."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file and return its top-level mapping."""
    with open(path) as handle:
        data = yaml.safe_load(handle)
    return data or {}


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge mappings in order. A later mapping overrides an earlier one."""
    merged: dict[str, Any] = {}
    for config in configs:
        _merge_into(merged, config)
    return merged


def _merge_into(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge one mapping into the result.

    The result takes a copy of each value. A merge must never change a
    mapping the caller gave it, because a caller can reuse one loaded
    mapping for many merges.
    """
    for key, value in source.items():
        existing = target.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            _merge_into(existing, value)
        else:
            target[key] = deepcopy(value)


def load_and_merge(*paths: Path) -> dict[str, Any]:
    """Read each YAML file in order and merge the results into one mapping."""
    return merge_configs(*(load_yaml(path) for path in paths))

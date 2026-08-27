"""Load YAML and retain a legacy read-only display merger."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class ConfigLoadError(Exception):
    """Report one configuration file loading failure."""

    def __init__(self, path: Path, message: str) -> None:
        self.path = path
        super().__init__(f"{message}: {path}")


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file and return its top-level mapping."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as error:
        raise ConfigLoadError(path, "the configuration file does not exist") from error
    except PermissionError as error:
        raise ConfigLoadError(path, "the configuration file is not readable") from error
    except UnicodeError as error:
        raise ConfigLoadError(
            path, "the configuration file is not valid UTF-8"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigLoadError(
            path, "the configuration file contains invalid YAML"
        ) from error
    except OSError as error:
        raise ConfigLoadError(path, "the configuration file cannot be read") from error
    if not isinstance(data, dict):
        raise ConfigLoadError(path, "the configuration root must be a mapping")
    includes = data.pop("include", ())
    if isinstance(includes, str):
        includes = (includes,)
    if includes:
        if not isinstance(includes, (list, tuple)) or not all(
            isinstance(value, str) for value in includes
        ):
            raise ConfigLoadError(path, "the include field must contain text paths")
        included = [load_yaml(path.parent / value) for value in includes]
        return merge_configs(*included, data)
    return data


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Merge historical display mappings without resolving a formal run."""
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
    """Read and merge historical display data without formal validation."""
    return merge_configs(*(load_yaml(path) for path in paths))

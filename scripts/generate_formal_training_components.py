"""Generate explicit controller components for the training matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from avalanche.config import ConfigurationResolver
from avalanche.config.run_identity import REPO_ROOT

SOURCE = REPO_ROOT / "configs/experiments/monitor-training.yaml"
OUTPUT = REPO_ROOT / "configs/controllers/formal-training"
MANIFEST = REPO_ROOT / "configs/experiments/monitor-training-components.yaml"


def _token(value: float) -> str:
    """Return one stable strength token."""
    return f"{round(value * 100):03d}"


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write one stable generated YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def generate() -> None:
    """Generate every concrete training controller selection."""
    source = yaml.safe_load(SOURCE.read_text())
    variants = tuple(str(value) for value in source["policy_variants"])
    strengths = tuple(float(value) for value in source["attack_strengths"])
    selection: dict[str, Any] = {
        "component_version": 1,
        "honest": {},
        "attacks": {},
    }
    resolver = ConfigurationResolver()
    for mountain in source["mountains"]:
        mountain_id = str(mountain["id"])
        honest_key = f"{mountain_id}-honest"
        honest_values = resolver.component_values(
            "controller", str(mountain["honest_config"])
        )
        _write_yaml(OUTPUT / "base" / f"{honest_key}.yaml", honest_values)
        honest_components = {}
        for variant in variants:
            name = f"{honest_key}-{variant}.yaml"
            _write_yaml(
                OUTPUT / name,
                {
                    "include": f"base/{honest_key}.yaml",
                    "controller": {"policy_variant": variant},
                },
            )
            honest_components[variant] = f"configs/controllers/formal-training/{name}"
        selection["honest"][mountain_id] = honest_components
        attacks = {}
        for declared in mountain["controllers"]:
            controller_id = str(declared["id"])
            base_key = f"{mountain_id}-{controller_id}"
            controller_values = resolver.component_values(
                "controller", str(declared["config"])
            )
            _write_yaml(
                OUTPUT / "base" / f"{base_key}.yaml",
                controller_values,
            )
            components = []
            for variant in variants:
                for strength in strengths:
                    name = f"{base_key}-{variant}-{_token(strength)}.yaml"
                    _write_yaml(
                        OUTPUT / name,
                        {
                            "include": f"base/{base_key}.yaml",
                            "controller": {
                                "policy_variant": variant,
                                "attack": {"action_budget": {"strength": strength}},
                            },
                        },
                    )
                    components.append(
                        {
                            "policy_variant": variant,
                            "attack_strength": strength,
                            "config": f"configs/controllers/formal-training/{name}",
                        }
                    )
            attacks[controller_id] = components
        selection["attacks"][mountain_id] = attacks
    _write_yaml(MANIFEST, selection)


if __name__ == "__main__":
    generate()

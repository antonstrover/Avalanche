from pathlib import Path

import pytest

from avalanche.config import (
    ConfigurationResolutionError,
    ConfigurationResolver,
    load_yaml,
)
from avalanche.monitors.dataset import expand_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = tuple(
    path.relative_to(REPO_ROOT).as_posix()
    for path in sorted((REPO_ROOT / "configs/scenarios").glob("*.yaml"))
)
MEDIUM_CONTROLLERS = tuple(
    path.relative_to(REPO_ROOT).as_posix()
    for pattern in ("*.yaml", "stealth/*.yaml")
    for path in sorted((REPO_ROOT / "configs/controllers").glob(pattern))
    if path.name != "none.yaml"
)
SMALL_CONTROLLERS = tuple(
    path.relative_to(REPO_ROOT).as_posix()
    for pattern in ("small-resort/*.yaml", "stealth/small-resort/*.yaml")
    for path in sorted((REPO_ROOT / "configs/controllers").glob(pattern))
)


@pytest.mark.parametrize(
    ("mountain", "controllers", "excluded_scenario", "monitors"),
    [
        (
            "configs/mountain/default.yaml",
            MEDIUM_CONTROLLERS,
            "configs/scenarios/failure-examples.yaml",
            ("none", "outcome", "rules"),
        ),
        (
            "configs/mountain/small.yaml",
            SMALL_CONTROLLERS,
            "configs/scenarios/honest-baseline.yaml",
            ("none", "outcome"),
        ),
    ],
)
def test_every_declared_compatible_composition_resolves(
    mountain, controllers, excluded_scenario, monitors
):
    resolver = ConfigurationResolver()
    controllers = (*controllers, "configs/controllers/none.yaml")
    scenarios = (value for value in SCENARIOS if value != excluded_scenario)
    for scenario in scenarios:
        for controller in controllers:
            for monitor in monitors:
                resolved = resolver.resolve(
                    mountain,
                    scenario,
                    controller,
                    f"configs/monitors/{monitor}.yaml",
                )
                assert resolved.resolved_configuration_sha256 != "0" * 64


@pytest.mark.parametrize(
    ("mountain", "scenario", "controller", "monitor"),
    [
        (
            "configs/mountain/default.yaml",
            "configs/scenarios/failure-examples.yaml",
            "configs/controllers/none.yaml",
            "configs/monitors/none.yaml",
        ),
        (
            "configs/mountain/small.yaml",
            "configs/scenarios/default.yaml",
            "configs/controllers/honest.yaml",
            "configs/monitors/none.yaml",
        ),
        (
            "configs/mountain/small.yaml",
            "configs/scenarios/default.yaml",
            "configs/controllers/none.yaml",
            "configs/monitors/rules.yaml",
        ),
        (
            "configs/mountain/default.yaml",
            "configs/scenarios/default.yaml",
            "configs/controllers/none.yaml",
            "configs/monitors/learned.yaml",
        ),
    ],
)
def test_every_declared_incompatible_composition_is_rejected(
    mountain, scenario, controller, monitor
):
    with pytest.raises(ConfigurationResolutionError):
        ConfigurationResolver().resolve(mountain, scenario, controller, monitor)


def test_every_declared_training_composition_resolves():
    manifest = load_yaml(REPO_ROOT / "configs/experiments/monitor-training.yaml")
    selections = {entry.config_paths for entry in expand_manifest(manifest)}
    resolver = ConfigurationResolver()

    for mountain, scenario, controller, monitor in sorted(selections):
        resolved = resolver.resolve(mountain, scenario, controller, monitor)
        assert resolved.resolved_configuration_sha256 != "0" * 64

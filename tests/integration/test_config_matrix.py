from argparse import Namespace
from itertools import product
from pathlib import Path

import pytest

from avalanche.cli import simulate
from avalanche.config import (
    ConfigurationResolutionError,
    ConfigurationResolver,
    load_yaml,
)
from avalanche.monitors.dataset import expand_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPATIBILITY = load_yaml(REPO_ROOT / "tests/fixtures/configuration-compatibility.yaml")


def _selection_matrix():
    mountains = COMPATIBILITY["mountains"]
    scenarios = COMPATIBILITY["scenarios"]
    controllers = COMPATIBILITY["controllers"]
    monitors = COMPATIBILITY["monitors"]
    for mountain_id, scenario_id, controller, monitor in product(
        mountains, scenarios, controllers, monitors
    ):
        compatible = all(
            mountain_id in allowed
            for allowed in (
                scenarios[scenario_id],
                controllers[controller],
                monitors[monitor],
            )
        )
        paths = (
            mountains[mountain_id],
            f"configs/scenarios/{scenario_id}.yaml",
            controller,
            monitor,
        )
        yield compatible, paths


def test_every_declared_live_composition_matches_the_compatibility_fixture(
    monkeypatch,
):
    output_calls = []
    monkeypatch.setattr("avalanche.cli.make_run_dir", output_calls.append)
    resolver = ConfigurationResolver()
    for compatible, paths in _selection_matrix():
        if compatible:
            resolved = resolver.resolve(*paths)
            assert resolved.resolved_configuration_sha256 != "0" * 64
            continue
        with pytest.raises(ConfigurationResolutionError):
            resolver.resolve(*paths)
        arguments = Namespace(
            mountain=paths[0],
            scenario=paths[1],
            controller=paths[2],
            monitor=paths[3],
            override=None,
            preflight=False,
        )
        assert simulate(arguments) == 1
    assert output_calls == []


def test_the_learned_monitor_remains_dependency_blocked():
    assert COMPATIBILITY["blocked_monitors"] == {
        "configs/monitors/learned.yaml": "gap-015"
    }
    with pytest.raises(ConfigurationResolutionError, match="verified selection"):
        ConfigurationResolver().resolve(
            "configs/mountain/default.yaml",
            "configs/scenarios/default.yaml",
            "configs/controllers/honest.yaml",
            "configs/monitors/learned.yaml",
        )


def test_every_declared_training_composition_resolves():
    manifest = load_yaml(REPO_ROOT / "configs/experiments/monitor-training.yaml")
    selections = {
        (*entry.config_paths, entry.override_path)
        for entry in expand_manifest(manifest)
    }
    resolver = ConfigurationResolver()

    for mountain, scenario, controller, monitor, override in sorted(selections):
        resolved = resolver.resolve(mountain, scenario, controller, monitor, override)
        assert resolved.resolved_configuration_sha256 != "0" * 64


def test_every_declared_attack_fixture_resolves():
    fixtures = load_yaml(REPO_ROOT / "configs/experiments/attack-fixtures.yaml")[
        "fixtures"
    ]
    resolver = ConfigurationResolver()
    for fixture in fixtures:
        for controller in (fixture["controller"], fixture["paired_controller"]):
            resolved = resolver.resolve(
                fixture["mountain"],
                fixture["scenario"],
                controller,
                fixture["monitor"],
                fixture["override"],
            )
            assert resolved.seed == fixture["seed"]

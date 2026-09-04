from argparse import Namespace
from itertools import product
from pathlib import Path
from time import perf_counter

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
    capsys,
):
    output_calls = []
    monkeypatch.setattr("avalanche.cli.make_run_dir", output_calls.append)
    resolver = ConfigurationResolver()
    cli_checked = False
    started = perf_counter()
    for compatible, paths in _selection_matrix():
        if compatible:
            resolved = resolver.resolve(*paths)
            assert resolved.resolved_configuration_sha256 != "0" * 64
            continue
        if not cli_checked:
            arguments = Namespace(
                mountain=paths[0],
                scenario=paths[1],
                controller=paths[2],
                monitor=paths[3],
                override=None,
                preflight=False,
            )
            assert simulate(arguments) == 1
            cli_checked = True
            continue
        with pytest.raises(ConfigurationResolutionError):
            resolver.resolve(*paths)
    elapsed = perf_counter() - started
    with capsys.disabled():
        print(f"\nThe live configuration matrix finished in {elapsed:.3f} seconds.")

    assert cli_checked
    assert elapsed < 30.0
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


def test_every_declared_training_composition_resolves(capsys):
    started = perf_counter()
    manifest = load_yaml(REPO_ROOT / "configs/experiments/monitor-training.yaml")
    selections = {
        (*entry.config_paths, entry.override_path)
        for entry in expand_manifest(manifest)
    }
    resolver = ConfigurationResolver()

    for mountain, scenario, controller, monitor, override in sorted(selections):
        resolved = resolver.resolve(mountain, scenario, controller, monitor, override)
        assert resolved.resolved_configuration_sha256 != "0" * 64
    elapsed = perf_counter() - started
    with capsys.disabled():
        print(f"\nThe training configuration matrix finished in {elapsed:.3f} seconds.")

    assert elapsed / len(manifest["seeds"]) < 3.0


def test_every_declared_attack_fixture_resolves():
    fixtures = load_yaml(REPO_ROOT / "configs/experiments/attack-fixtures.yaml")[
        "fixtures"
    ]
    resolver = ConfigurationResolver()
    for fixture in fixtures:
        for run in fixture["runs"]:
            for controller in (fixture["controller"], fixture["paired_controller"]):
                resolved = resolver.resolve(
                    fixture["mountain"],
                    fixture["scenario"],
                    controller,
                    fixture["monitor"],
                    run["override"],
                )
                assert resolved.seed == run["seed"]


def test_cached_included_composition_matches_a_fresh_resolution():
    fixture = load_yaml(REPO_ROOT / "configs/experiments/attack-fixtures.yaml")[
        "fixtures"
    ][0]
    selection = (
        fixture["mountain"],
        fixture["scenario"],
        fixture["controller"],
        fixture["monitor"],
        fixture["runs"][0]["override"],
    )
    resolver = ConfigurationResolver()

    first = resolver.resolve(*selection)
    repeated = resolver.resolve(*selection)
    fresh = ConfigurationResolver().resolve(*selection)

    assert repeated == first
    assert repeated == fresh

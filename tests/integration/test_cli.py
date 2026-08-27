import json
import subprocess
import sys
from pathlib import Path

import pytest

from avalanche.cli import main

REPO_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_ARGS = [
    "--mountain",
    "configs/mountain/default.yaml",
    "--scenario",
    "configs/scenarios/default.yaml",
    "--controller",
    "configs/controllers/honest.yaml",
    "--monitor",
    "configs/monitors/none.yaml",
]


def _run(args: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    return exit_info.value.code


def test_validate_config_accepts_four_named_components():
    assert _run(["validate-config", *SAMPLE_ARGS]) == 0


def test_validate_config_reports_complete_stable_evidence(capsys):
    assert _run(["validate-config", *SAMPLE_ARGS]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "OK"
    assert len(report["resolved_configuration_sha256"]) == 64
    assert len(report["scientific_configuration_sha256"]) == 64
    assert report["provenance"]


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_cli_preflight_is_working_directory_independent(command):
    arguments = [command, *SAMPLE_ARGS]
    if command == "simulate":
        arguments.append("--preflight")
    outputs = []
    for directory in (REPO_ROOT, REPO_ROOT / "dashboard"):
        result = subprocess.run(
            [sys.executable, "-m", "avalanche.cli", *arguments],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_a_missing_component_is_rejected(command):
    assert _run([command, *SAMPLE_ARGS[:-2]]) == 2


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_a_duplicate_component_is_rejected(command):
    assert _run([command, *SAMPLE_ARGS, "--monitor", SAMPLE_ARGS[-1]]) == 2


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_an_absolute_component_path_is_rejected(command, capsys):
    args = SAMPLE_ARGS.copy()
    args[1] = str(Path("configs/mountain/default.yaml").resolve())
    assert _run([command, *args]) == 1
    assert "repository-relative" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_a_traversing_component_path_is_rejected(command, capsys):
    args = SAMPLE_ARGS.copy()
    args[1] = "configs/mountain/../mountain/default.yaml"
    assert _run([command, *args]) == 1
    assert "must not traverse" in capsys.readouterr().err


def test_a_missing_component_file_is_reported(capsys):
    args = SAMPLE_ARGS.copy()
    args[3] = "configs/scenarios/missing.yaml"
    assert _run(["validate-config", *args]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_simulate_validates_before_it_creates_output(monkeypatch):
    calls = []
    monkeypatch.setattr("avalanche.cli.make_run_dir", lambda value: calls.append(value))
    args = SAMPLE_ARGS.copy()
    args[1] = "configs/mountain/small.yaml"
    assert _run(["simulate", *args]) == 1
    assert calls == []


def test_simulate_passes_one_validated_configuration(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls = []
    monkeypatch.setattr("avalanche.cli.make_run_dir", lambda value: run_dir)
    monkeypatch.setattr(
        "avalanche.cli.run_episode",
        lambda resolved, output: calls.append((resolved, output)),
    )
    assert _run(["simulate", *SAMPLE_ARGS]) == 0
    assert calls[0][0].resolved_configuration_sha256 != "0" * 64
    assert calls[0][1] == run_dir


def test_simulate_preflight_creates_no_output(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("avalanche.cli.make_run_dir", lambda value: calls.append(value))
    assert _run(["simulate", *SAMPLE_ARGS, "--preflight"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "OK"
    assert calls == []

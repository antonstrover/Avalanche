from pathlib import Path

import pytest

from avalanche.cli import main

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

SAMPLE_ARGS = [
    str(CONFIGS / "mountain" / "default.yaml"),
    str(CONFIGS / "scenarios" / "default.yaml"),
    str(CONFIGS / "controllers" / "honest.yaml"),
    str(CONFIGS / "monitors" / "none.yaml"),
]


def _run(args: list[str]) -> int:
    with pytest.raises(SystemExit) as exit_info:
        main(args)
    return exit_info.value.code


def test_validate_config_accepts_a_valid_config():
    assert _run(["validate-config", *SAMPLE_ARGS]) == 0


def test_validate_config_rejects_a_malformed_config(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("seed: not-an-int\n")
    assert _run(["validate-config", *SAMPLE_ARGS, str(bad_file)]) == 1


def test_simulate_makes_a_run_directory_with_a_placeholder_episode(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "avalanche.cli.make_run_dir",
        lambda resolved: (tmp_path / "run").resolve(),
    )
    (tmp_path / "run").mkdir()

    assert _run(["simulate", *SAMPLE_ARGS]) == 0
    assert (tmp_path / "run" / "episode.placeholder.txt").exists()


def test_sweep_and_analyse_report_not_yet_implemented():
    assert _run(["sweep"]) == 1
    assert _run(["analyse"]) == 1

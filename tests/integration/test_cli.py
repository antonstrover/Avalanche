from pathlib import Path

import pytest

from avalanche.cli import main
from avalanche.config import load_yaml

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

SAMPLE_ARGS = [
    str(CONFIGS / "mountain" / "default.yaml"),
    str(CONFIGS / "scenarios" / "default.yaml"),
    str(CONFIGS / "controllers" / "honest.yaml"),
    str(CONFIGS / "monitors" / "none.yaml"),
]
MOUNTAIN_CONFIGS = [
    pytest.param(CONFIGS / "mountain" / "default.yaml", 60, 80, id="medium"),
    pytest.param(CONFIGS / "mountain" / "small.yaml", 10, 12, id="small"),
]
SHARED_ARGS = [
    str(CONFIGS / "scenarios" / "default.yaml"),
    str(CONFIGS / "controllers" / "none.yaml"),
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


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_config_commands_report_a_missing_file(command, tmp_path, capsys):
    bad_file = tmp_path / "missing.yaml"

    assert _run([command, *SAMPLE_ARGS, str(bad_file)]) == 1

    captured = capsys.readouterr()
    assert str(bad_file) in captured.err
    assert "does not exist" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_config_commands_report_an_unreadable_file(
    command, tmp_path, capsys, monkeypatch
):
    bad_file = tmp_path / "unreadable.yaml"
    bad_file.write_text("seed: 1\n")
    original_open = Path.open

    def open_except_bad(path, *args, **kwargs):
        if path == bad_file:
            raise PermissionError("test permission failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_except_bad)
    assert _run([command, *SAMPLE_ARGS, str(bad_file)]) == 1

    captured = capsys.readouterr()
    assert str(bad_file) in captured.err
    assert "not readable" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_config_commands_report_invalid_utf8(command, tmp_path, capsys):
    bad_file = tmp_path / "invalid-utf8.yaml"
    bad_file.write_bytes(b"seed: \xff\n")

    assert _run([command, *SAMPLE_ARGS, str(bad_file)]) == 1

    captured = capsys.readouterr()
    assert str(bad_file) in captured.err
    assert "not valid UTF-8" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_config_commands_report_invalid_yaml(command, tmp_path, capsys):
    bad_file = tmp_path / "invalid.yaml"
    bad_file.write_text("seed: [\n")

    assert _run([command, *SAMPLE_ARGS, str(bad_file)]) == 1

    captured = capsys.readouterr()
    assert str(bad_file) in captured.err
    assert "invalid YAML" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
@pytest.mark.parametrize("contents", ["value\n", "- item\n", '"text"\n'])
def test_config_commands_reject_a_nonmapping_root(command, contents, tmp_path, capsys):
    bad_file = tmp_path / "invalid-root.yaml"
    bad_file.write_text(contents)

    assert _run([command, *SAMPLE_ARGS, str(bad_file)]) == 1

    captured = capsys.readouterr()
    assert str(bad_file) in captured.err
    assert "root must be a mapping" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("command", ["validate-config", "simulate"])
def test_config_commands_reject_rule_replacement(command, tmp_path, capsys):
    bad_file = tmp_path / "rule-replace.yaml"
    bad_file.write_text("monitor:\n  kind: rules\n  unsafe_decision: REPLACE\n")

    assert _run([command, *SAMPLE_ARGS, str(bad_file)]) == 1

    captured = capsys.readouterr()
    assert "rule monitor cannot use a REPLACE" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("mountain_path", "node_count", "edge_count"), MOUNTAIN_CONFIGS
)
@pytest.mark.parametrize("field", ["node_count", "edge_count"])
def test_validate_config_rejects_a_wrong_mountain_count(
    mountain_path,
    node_count,
    edge_count,
    field,
    tmp_path,
    capsys,
):
    loaded_value = {"node_count": node_count, "edge_count": edge_count}[field]
    declared = loaded_value + 1
    override = tmp_path / "wrong-count.yaml"
    override.write_text(f"mountain:\n  {field}: {declared}\n")

    assert (
        _run(["validate-config", str(mountain_path), *SHARED_ARGS, str(override)]) == 1
    )

    captured = capsys.readouterr()
    mountain = str(load_yaml(mountain_path)["mountain"]["path"])
    assert mountain in captured.err
    assert field in captured.err
    assert str(declared) in captured.err
    assert str(loaded_value) in captured.err


@pytest.mark.parametrize(("field", "loaded"), [("node_count", 60), ("edge_count", 80)])
def test_a_wrong_mountain_count_prevents_run_directory_creation(
    field, loaded, tmp_path, monkeypatch
):
    override = tmp_path / "wrong-count.yaml"
    override.write_text(f"mountain:\n  {field}: {loaded + 1}\n")
    calls = []
    monkeypatch.setattr(
        "avalanche.cli.make_run_dir", lambda resolved: calls.append(resolved)
    )

    assert _run(["simulate", *SAMPLE_ARGS, str(override)]) == 1
    assert calls == []


def test_simulate_runs_an_episode_in_the_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "avalanche.cli.make_run_dir",
        lambda resolved: (tmp_path / "run").resolve(),
    )
    (tmp_path / "run").mkdir()
    calls = []
    monkeypatch.setattr(
        "avalanche.cli.run_episode",
        lambda resolved, run_dir: calls.append((resolved, run_dir)),
    )

    assert _run(["simulate", *SAMPLE_ARGS]) == 0
    assert calls[0][1] == (tmp_path / "run").resolve()


def test_sweep_and_analyse_report_not_yet_implemented():
    assert _run(["sweep"]) == 1
    assert _run(["analyse"]) == 1

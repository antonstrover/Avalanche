from pathlib import Path

import pytest

from avalanche.config import ConfigLoadError, load_yaml, merge_configs
from avalanche.config.loader import load_and_merge

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

SAMPLE_FILES = [
    CONFIGS / "mountain" / "default.yaml",
    CONFIGS / "scenarios" / "default.yaml",
    CONFIGS / "controllers" / "honest.yaml",
    CONFIGS / "monitors" / "none.yaml",
]


def test_load_yaml_reads_a_mapping():
    data = load_yaml(SAMPLE_FILES[0])
    assert data["mountain"]["name"] == "val-tarin"


def test_load_yaml_reports_a_missing_file(tmp_path):
    path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigLoadError, match=str(path)) as error_info:
        load_yaml(path)

    assert isinstance(error_info.value.__cause__, FileNotFoundError)


def test_load_yaml_reports_an_unreadable_file(tmp_path, monkeypatch):
    path = tmp_path / "unreadable.yaml"
    path.write_text("seed: 1\n")

    def reject_open(_path, *args, **kwargs):
        raise PermissionError("test permission failure")

    monkeypatch.setattr(Path, "open", reject_open)
    with pytest.raises(ConfigLoadError, match=str(path)) as error_info:
        load_yaml(path)

    assert isinstance(error_info.value.__cause__, PermissionError)


def test_load_yaml_reports_invalid_utf8(tmp_path):
    path = tmp_path / "invalid-utf8.yaml"
    path.write_bytes(b"seed: \xff\n")

    with pytest.raises(ConfigLoadError, match=str(path)) as error_info:
        load_yaml(path)

    assert isinstance(error_info.value.__cause__, UnicodeError)


def test_load_yaml_reports_invalid_yaml(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("seed: [\n")

    with pytest.raises(ConfigLoadError, match=str(path)) as error_info:
        load_yaml(path)

    assert error_info.value.__cause__ is not None


@pytest.mark.parametrize("contents", ["", "null\n", "value\n", "- item\n"])
def test_load_yaml_rejects_a_nonmapping_root(tmp_path, contents):
    path = tmp_path / "invalid-root.yaml"
    path.write_text(contents)

    with pytest.raises(ConfigLoadError, match="root must be a mapping") as error_info:
        load_yaml(path)

    assert str(path) in str(error_info.value)


def test_merge_configs_combines_and_overrides():
    base = {"a": 1, "nested": {"x": 1, "y": 1}}
    override = {"a": 2, "nested": {"y": 2}}
    merged = merge_configs(base, override)
    assert merged == {"a": 2, "nested": {"x": 1, "y": 2}}


def test_load_and_merge_resolves_the_sample_configs():
    resolved = load_and_merge(*SAMPLE_FILES)
    assert resolved["mountain"]["name"] == "val-tarin"
    assert resolved["population"]["skier_count"] == 5000
    assert resolved["intervals"]["movement_tick_seconds"] == 5
    assert resolved["controller"]["kind"] == "honest"
    assert resolved["monitor"]["kind"] == "none"
    assert resolved["seed"] == 1234


def test_a_merge_does_not_change_the_mapping_it_reads():
    """A caller can reuse one loaded mapping for many merges."""
    defaults = {"scenario": {"failures": {"schedule": []}}}
    with_schedule = {
        "scenario": {"failures": {"schedule": [{"kind": "lift_stoppage"}]}}
    }
    with_sampling = {"scenario": {"failures": {"sampling": {"event_count": 2}}}}

    merge_configs(defaults, with_schedule)
    second = merge_configs(defaults, with_sampling)

    assert defaults == {"scenario": {"failures": {"schedule": []}}}
    assert second["scenario"]["failures"] == {
        "schedule": [],
        "sampling": {"event_count": 2},
    }

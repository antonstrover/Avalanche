from pathlib import Path

from avalanche.config import load_yaml, merge_configs
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

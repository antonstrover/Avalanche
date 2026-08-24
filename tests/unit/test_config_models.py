from pathlib import Path

import pytest
from pydantic import ValidationError

from avalanche.config import ResolvedConfig, load_and_merge

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

SAMPLE_FILES = [
    CONFIGS / "mountain" / "default.yaml",
    CONFIGS / "scenarios" / "default.yaml",
    CONFIGS / "controllers" / "honest.yaml",
    CONFIGS / "monitors" / "none.yaml",
]


def test_valid_config_parses():
    resolved = ResolvedConfig.model_validate(load_and_merge(*SAMPLE_FILES))
    assert resolved.seed == 1234
    assert resolved.mountain.node_count == 60
    assert resolved.trace_level == "debug"


def test_missing_seed_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    del data["seed"]
    with pytest.raises(ValidationError, match="seed"):
        ResolvedConfig.model_validate(data)


def test_wrong_type_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["seed"] = "not-an-int"
    with pytest.raises(ValidationError, match="seed"):
        ResolvedConfig.model_validate(data)


def test_unknown_trace_level_is_rejected():
    data = load_and_merge(*SAMPLE_FILES)
    data["trace_level"] = "verbose"
    with pytest.raises(ValidationError, match="trace_level"):
        ResolvedConfig.model_validate(data)

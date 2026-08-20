import json
from pathlib import Path

from avalanche.config import ResolvedConfig, load_and_merge, make_run_dir, run_id

CONFIGS = Path(__file__).resolve().parents[2] / "configs"

SAMPLE_FILES = [
    CONFIGS / "mountain" / "default.yaml",
    CONFIGS / "scenarios" / "default.yaml",
    CONFIGS / "controllers" / "honest.yaml",
    CONFIGS / "monitors" / "none.yaml",
]


def _sample_config() -> ResolvedConfig:
    return ResolvedConfig.model_validate(load_and_merge(*SAMPLE_FILES))


def test_same_config_gives_the_same_run_id():
    resolved = _sample_config()
    assert run_id(resolved) == run_id(resolved)


def test_running_twice_reuses_the_same_output_directory(tmp_path):
    resolved = _sample_config()

    first_dir = make_run_dir(resolved, outputs_root=tmp_path)
    second_dir = make_run_dir(resolved, outputs_root=tmp_path)

    assert first_dir == second_dir
    assert (first_dir / "config.resolved.yaml").exists()

    metadata = json.loads((first_dir / "metadata.json").read_text())
    assert metadata["run_id"] == run_id(resolved)
    assert "git_commit" in metadata
    assert "dependency_lock_hash" in metadata

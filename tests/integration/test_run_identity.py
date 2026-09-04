from pathlib import Path

from avalanche.config import ConfigurationResolver, make_run_dir, run_id

SAMPLE = {
    "mountain": "configs/mountain/default.yaml",
    "scenario": "configs/scenarios/default.yaml",
    "controller": "configs/controllers/honest.yaml",
    "monitor": "configs/monitors/none.yaml",
}


def _sample_config():
    return ConfigurationResolver().resolve(**SAMPLE)


def test_same_config_gives_the_same_run_id():
    first = _sample_config()
    second = _sample_config()
    assert run_id(first) == run_id(second)
    assert run_id(first) == (
        f"{first.scientific_configuration_sha256[:12]}-"
        f"{first.resolved_configuration_sha256[:12]}"
    )


def test_running_twice_reuses_the_same_output_directory(tmp_path):
    resolved = _sample_config()
    first_dir = make_run_dir(resolved, outputs_root=tmp_path)
    second_dir = make_run_dir(resolved, outputs_root=tmp_path)
    assert first_dir == second_dir
    assert not tuple(first_dir.iterdir())


def test_a_scientific_parameter_changes_both_digests():
    honest = _sample_config()
    attack = ConfigurationResolver().resolve(
        mountain="configs/mountain/default.yaml",
        scenario="configs/scenarios/default.yaml",
        controller="configs/controllers/profit-biased.yaml",
        monitor="configs/monitors/none.yaml",
    )
    assert honest.scientific_configuration_sha256 != (
        attack.scientific_configuration_sha256
    )
    assert honest.resolved_configuration_sha256 != attack.resolved_configuration_sha256


def test_output_root_does_not_change_the_scientific_run_prefix():
    base = _sample_config()
    changed = ConfigurationResolver().resolve(
        **SAMPLE, override="configs/overrides/alternate-output.yaml"
    )
    assert base.scientific_configuration_sha256 == (
        changed.scientific_configuration_sha256
    )
    assert base.resolved_configuration_sha256 != changed.resolved_configuration_sha256


def test_worker_count_does_not_change_the_scientific_run_prefix():
    base = _sample_config()
    changed = ConfigurationResolver().resolve(
        **SAMPLE, override="configs/overrides/parallel.yaml"
    )
    assert base.scientific_configuration_sha256 == (
        changed.scientific_configuration_sha256
    )
    assert base.resolved_configuration_sha256 != changed.resolved_configuration_sha256


def test_paths_in_the_identity_are_repository_relative():
    resolved = _sample_config()
    assert not Path(resolved.mountain.path).is_absolute()
    assert all(
        record.source_path is None or not Path(record.source_path).is_absolute()
        for record in resolved.provenance
    )

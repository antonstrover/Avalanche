from pathlib import Path

import pytest
import yaml

from avalanche.config import (
    ConfigLoadError,
    ConfigurationResolutionError,
    ConfigurationResolver,
    load_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE = {
    "mountain": "configs/mountain/default.yaml",
    "scenario": "configs/scenarios/default.yaml",
    "controller": "configs/controllers/honest.yaml",
    "monitor": "configs/monitors/none.yaml",
}


def _leaf_pointers(value, parts=()):
    if isinstance(value, dict):
        if not value:
            return {"/" + "/".join(parts)}
        return {
            pointer
            for key, nested in value.items()
            for pointer in _leaf_pointers(nested, (*parts, str(key)))
        }
    if isinstance(value, list):
        if not value:
            return {"/" + "/".join(parts)}
        return {
            pointer
            for index, nested in enumerate(value)
            for pointer in _leaf_pointers(nested, (*parts, str(index)))
        }
    return {"/" + "/".join(parts)}


def test_load_yaml_reads_a_mapping():
    data = load_yaml(REPO_ROOT / SAMPLE["mountain"])
    assert data["mountain"]["name"] == "val-tarin"


def test_load_yaml_reports_a_missing_file(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(ConfigLoadError, match=str(path)):
        load_yaml(path)


def test_the_resolver_records_each_provenance_kind():
    resolved = ConfigurationResolver().resolve(**SAMPLE)
    kinds = {record.kind for record in resolved.provenance}
    assert kinds == {"explicit", "schema_default", "derived"}
    explicit = next(
        record for record in resolved.provenance if record.kind == "explicit"
    )
    assert explicit.source_path is not None
    assert explicit.line and explicit.column
    assert explicit.source_sha256 and len(explicit.source_sha256) == 64


def test_every_logical_leaf_has_one_provenance_record():
    resolved = ConfigurationResolver().resolve(**SAMPLE)
    logical = resolved.model_dump(
        mode="json",
        exclude={
            "provenance",
            "resolved_configuration_sha256",
            "scientific_configuration_sha256",
        },
    )
    expected = _leaf_pointers(logical) | {
        "/resolved_configuration_sha256",
        "/scientific_configuration_sha256",
    }
    pointers = [record.pointer for record in resolved.provenance]

    assert set(pointers) == expected
    assert len(pointers) == len(set(pointers))


def test_the_resolver_is_immutable():
    resolved = ConfigurationResolver().resolve(**SAMPLE)
    with pytest.raises(Exception, match="frozen"):
        resolved.seed = 9
    with pytest.raises(Exception, match="frozen"):
        resolved.population.skier_count = 9


def test_an_override_changes_only_approved_paths():
    resolved = ConfigurationResolver().resolve(
        **SAMPLE, override="configs/overrides/quick.yaml"
    )
    assert resolved.seed == 7
    assert resolved.population.skier_count == 200
    assert (
        next(
            record for record in resolved.provenance if record.pointer == "/seed"
        ).owner
        == "override"
    )


def _copy_sample(root: Path) -> None:
    for source in (*SAMPLE.values(), "configs/mountain/medium-resort.yaml"):
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO_ROOT / source).read_bytes())


def test_a_forbidden_override_path_is_rejected(tmp_path):
    _copy_sample(tmp_path)
    override = tmp_path / "configs/forbidden.yaml"
    override.write_text("controller:\n  kind: none\n")
    with pytest.raises(ConfigurationResolutionError, match="not owned"):
        ConfigurationResolver(tmp_path).resolve(
            **SAMPLE, override="configs/forbidden.yaml"
        )


def test_a_windows_absolute_component_path_is_rejected():
    with pytest.raises(ConfigurationResolutionError, match="repository-relative"):
        ConfigurationResolver().resolve(
            **{**SAMPLE, "mountain": "C:\\configs\\mountain.yaml"}
        )


def test_a_component_symlink_cannot_leave_the_repository(tmp_path):
    root = tmp_path / "repository"
    _copy_sample(root)
    outside = tmp_path / "outside.yaml"
    outside.write_text((REPO_ROOT / SAMPLE["controller"]).read_text())
    link = root / "configs/controllers/escaped.yaml"
    link.symlink_to(outside)

    with pytest.raises(ConfigurationResolutionError, match="leaves the repository"):
        ConfigurationResolver(root).resolve(
            **{**SAMPLE, "controller": "configs/controllers/escaped.yaml"}
        )


def test_a_value_in_two_ownership_domains_is_rejected(tmp_path):
    _copy_sample(tmp_path)
    mountain = load_yaml(tmp_path / SAMPLE["mountain"])
    mountain["seed"] = 8
    (tmp_path / SAMPLE["mountain"]).write_text(
        yaml.safe_dump(mountain, sort_keys=False)
    )

    with pytest.raises(ConfigurationResolutionError, match="not owned"):
        ConfigurationResolver(tmp_path).resolve(**SAMPLE)


def test_output_root_changes_only_the_resolved_digest(tmp_path):
    _copy_sample(tmp_path)
    override = tmp_path / "configs/output.yaml"
    override.write_text("output_root: alternate-outputs\n")
    resolver = ConfigurationResolver(tmp_path)
    base = resolver.resolve(**SAMPLE)
    changed = resolver.resolve(**SAMPLE, override="configs/output.yaml")
    assert base.resolved_configuration_sha256 != changed.resolved_configuration_sha256
    assert (
        base.scientific_configuration_sha256 == changed.scientific_configuration_sha256
    )


def test_working_directory_does_not_change_identity(monkeypatch, tmp_path):
    resolver = ConfigurationResolver()
    first = resolver.resolve(**SAMPLE)
    monkeypatch.chdir(tmp_path)
    second = resolver.resolve(**SAMPLE)
    assert first.resolved_configuration_sha256 == second.resolved_configuration_sha256
    assert first.provenance == second.provenance


def test_an_including_component_replaces_one_owned_value(tmp_path):
    _copy_sample(tmp_path)
    scenario = tmp_path / "configs/scenarios/included.yaml"
    scenario.write_text("include: default.yaml\nseed: 99\n")

    resolved = ConfigurationResolver(tmp_path).resolve(
        **{**SAMPLE, "scenario": "configs/scenarios/included.yaml"}
    )

    assert resolved.seed == 99
    seed = next(record for record in resolved.provenance if record.pointer == "/seed")
    interval = next(
        record
        for record in resolved.provenance
        if record.pointer == "/intervals/control_interval_seconds"
    )
    assert seed.source_path == "configs/scenarios/included.yaml"
    assert interval.source_path == "configs/scenarios/default.yaml"


def test_the_resolver_rejects_a_reward_hacker_without_a_lift_target(tmp_path):
    _copy_sample(tmp_path)
    values = load_yaml(REPO_ROOT / "configs/controllers/reward-hacker.yaml")
    values["controller"]["attack"]["targets"] = ["praz_plaza->melezes_base"]
    values["controller"]["attack"]["action_budget"]["maximum_targets"] = 1
    controller = tmp_path / "configs/controllers/invalid.yaml"
    controller.write_text(yaml.safe_dump(values, sort_keys=False))
    with pytest.raises(ConfigurationResolutionError, match="lift service target"):
        ConfigurationResolver(tmp_path).resolve(
            **{**SAMPLE, "controller": "configs/controllers/invalid.yaml"}
        )


def test_the_resolver_rejects_a_sleeper_target_outside_escape_lifts(tmp_path):
    _copy_sample(tmp_path)
    values = load_yaml(REPO_ROOT / "configs/controllers/sleeper-saboteur.yaml")
    values["controller"]["attack"]["targets"] = ["praz_plaza->melezes_base"]
    controller = tmp_path / "configs/controllers/invalid.yaml"
    controller.write_text(yaml.safe_dump(values, sort_keys=False))
    with pytest.raises(ConfigurationResolutionError) as error:
        ConfigurationResolver(tmp_path).resolve(
            **{**SAMPLE, "controller": "configs/controllers/invalid.yaml"}
        )

    assert "is not a lift" in str(error.value)
    assert "is not an escape" in str(error.value)


def test_the_resolver_collects_reference_and_schedule_errors(tmp_path):
    _copy_sample(tmp_path)
    scenario = load_yaml(tmp_path / SAMPLE["scenario"])
    scenario["scenario"]["failures"] = {
        "schedule": [
            {
                "kind": "lift_stoppage",
                "target": "missing->edge",
                "start_time_seconds": 28700,
                "duration_seconds": 200,
            }
        ]
    }
    (tmp_path / SAMPLE["scenario"]).write_text(
        yaml.safe_dump(scenario, sort_keys=False)
    )

    with pytest.raises(ConfigurationResolutionError) as error:
        ConfigurationResolver(tmp_path).resolve(**SAMPLE)

    assert "unknown edge" in str(error.value)
    assert "ends after the episode" in str(error.value)


def test_schedule_errors_survive_a_topology_load_error(tmp_path):
    _copy_sample(tmp_path)
    mountain = load_yaml(tmp_path / SAMPLE["mountain"])
    mountain["mountain"]["path"] = "configs/mountain/missing.yaml"
    (tmp_path / SAMPLE["mountain"]).write_text(
        yaml.safe_dump(mountain, sort_keys=False)
    )
    scenario = load_yaml(tmp_path / SAMPLE["scenario"])
    scenario["snapshot_interval_seconds"] = 30000
    (tmp_path / SAMPLE["scenario"]).write_text(
        yaml.safe_dump(scenario, sort_keys=False)
    )

    with pytest.raises(ConfigurationResolutionError) as error:
        ConfigurationResolver(tmp_path).resolve(**SAMPLE)

    assert "missing.yaml" in str(error.value)
    assert "snapshot_interval_seconds" in str(error.value)

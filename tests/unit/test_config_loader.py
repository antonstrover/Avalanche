import hashlib
from pathlib import Path

import pytest
import yaml

from avalanche.config import (
    ConfigLoadError,
    ConfigurationResolutionError,
    ConfigurationResolver,
    composition,
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


@pytest.mark.parametrize(
    ("pointer", "values", "expected"),
    [
        ("/seed", {"seed": 8}, 8),
        ("/episode_duration_seconds", {"episode_duration_seconds": 900}, 900.0),
        ("/population/skier_count", {"population": {"skier_count": 200}}, 200),
        ("/trace_level", {"trace_level": "summary"}, "summary"),
        ("/output_root", {"output_root": "alternate-outputs"}, "alternate-outputs"),
        ("/runtime/worker_count", {"runtime": {"worker_count": 2}}, 2),
    ],
)
def test_each_formal_override_path_is_explicit(tmp_path, pointer, values, expected):
    _copy_sample(tmp_path)
    override = tmp_path / "configs/override.yaml"
    override.write_text(yaml.safe_dump(values, sort_keys=False))
    resolved = ConfigurationResolver(tmp_path).resolve(
        **SAMPLE, override="configs/override.yaml"
    )
    target = resolved.model_dump(mode="json")
    for part in pointer.strip("/").split("/"):
        target = target[part]
    provenance = next(item for item in resolved.provenance if item.pointer == pointer)
    assert target == expected
    assert provenance.kind == "explicit"
    assert provenance.owner == "override"


@pytest.mark.parametrize(
    "values",
    [
        {"mountain": {}},
        {"scenario": {}},
        {"controller": {}},
        {"monitor": {}},
        {"fallback": {}},
        {"approval": {}},
        {"population": {"ability_weights": [0.3, 0.5, 0.2]}},
        {"runtime": {"queue_size": 2}},
    ],
)
def test_each_forbidden_override_domain_is_rejected(tmp_path, values):
    _copy_sample(tmp_path)
    override = tmp_path / "configs/override.yaml"
    override.write_text(yaml.safe_dump(values, sort_keys=False))
    with pytest.raises(ConfigurationResolutionError):
        ConfigurationResolver(tmp_path).resolve(
            **SAMPLE, override="configs/override.yaml"
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


def test_stored_separator_forms_have_one_identity(tmp_path):
    _copy_sample(tmp_path)
    resolver = ConfigurationResolver(tmp_path)
    first = resolver.resolve(**SAMPLE)
    mountain = load_yaml(tmp_path / SAMPLE["mountain"])
    mountain["mountain"]["path"] = "configs\\mountain\\medium-resort.yaml"
    (tmp_path / SAMPLE["mountain"]).write_text(
        yaml.safe_dump(mountain, sort_keys=False)
    )
    second = resolver.resolve(**SAMPLE)
    assert second.mountain.path == "configs/mountain/medium-resort.yaml"
    assert first.resolved_configuration_sha256 == second.resolved_configuration_sha256


def test_live_resolution_loads_the_topology_once(monkeypatch):
    from avalanche.sim import topology

    original = topology.load_topology
    calls = []

    def load_once(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(topology, "load_topology", load_once)
    ConfigurationResolver().resolve_live(**SAMPLE, seed=9)
    assert len(calls) == 1


def test_unchanged_sources_reuse_each_instance_cache(monkeypatch):
    from avalanche.sim import topology

    compose = composition.yaml.compose
    load_topology = topology.load_topology
    safe_routes = composition._safe_routes
    calls = {"parse": 0, "topology": 0, "safe_routes": 0}

    def tracked_compose(*args, **kwargs):
        calls["parse"] += 1
        return compose(*args, **kwargs)

    def tracked_topology(path):
        calls["topology"] += 1
        return load_topology(path)

    def tracked_safe_routes(loaded):
        calls["safe_routes"] += 1
        return safe_routes(loaded)

    monkeypatch.setattr(composition.yaml, "compose", tracked_compose)
    monkeypatch.setattr(topology, "load_topology", tracked_topology)
    monkeypatch.setattr(composition, "_safe_routes", tracked_safe_routes)
    resolver = ConfigurationResolver()

    first = resolver.resolve(**SAMPLE)
    second = resolver.resolve(**SAMPLE)

    assert first == second
    assert calls == {"parse": 4, "topology": 1, "safe_routes": 1}

    another = ConfigurationResolver()
    assert another.resolve(**SAMPLE) == first
    assert calls == {"parse": 8, "topology": 2, "safe_routes": 2}


def test_cached_sources_are_copied_before_include_merging(tmp_path):
    _copy_sample(tmp_path)
    scenario = tmp_path / "configs/scenarios/included.yaml"
    scenario.write_text("include: default.yaml\nseed: 99\n")
    selection = {**SAMPLE, "scenario": "configs/scenarios/included.yaml"}
    resolver = ConfigurationResolver(tmp_path)
    expected = resolver.resolve(**selection)

    displayed = resolver.component_values("scenario", selection["scenario"])
    displayed["intervals"]["control_interval_seconds"] = 300
    repeated = resolver.resolve(**selection)

    assert repeated == expected
    assert repeated.intervals.control_interval_seconds == 60.0


def test_an_include_byte_change_reapplies_the_include(tmp_path):
    _copy_sample(tmp_path)
    scenario = tmp_path / "configs/scenarios/included.yaml"
    scenario.write_text("include: default.yaml\nseed: 99\n")
    selection = {**SAMPLE, "scenario": "configs/scenarios/included.yaml"}
    resolver = ConfigurationResolver(tmp_path)
    first = resolver.resolve(**selection)
    included = tmp_path / SAMPLE["scenario"]
    included.write_text(
        included.read_text().replace("trace_level: decision", "trace_level: summary")
    )

    second = resolver.resolve(**selection)
    first_trace = next(
        record for record in first.provenance if record.pointer == "/trace_level"
    )
    second_trace = next(
        record for record in second.provenance if record.pointer == "/trace_level"
    )

    assert first.trace_level == "decision"
    assert second.trace_level == "summary"
    assert first.resolved_configuration_sha256 != second.resolved_configuration_sha256
    assert first.scientific_configuration_sha256 != (
        second.scientific_configuration_sha256
    )
    assert first_trace.source_path == second_trace.source_path
    assert first_trace.source_sha256 != second_trace.source_sha256


def test_a_source_byte_change_preserves_exact_logical_identities(tmp_path):
    _copy_sample(tmp_path)
    resolver = ConfigurationResolver(tmp_path)
    first = resolver.resolve(**SAMPLE)
    controller = tmp_path / SAMPLE["controller"]
    controller.write_text(controller.read_text() + "\n# Keep the same values.\n")

    second = resolver.resolve(**SAMPLE)
    first_kind = next(
        record for record in first.provenance if record.pointer == "/controller/kind"
    )
    second_kind = next(
        record for record in second.provenance if record.pointer == "/controller/kind"
    )

    assert first.resolved_configuration_sha256 == second.resolved_configuration_sha256
    assert first.scientific_configuration_sha256 == (
        second.scientific_configuration_sha256
    )
    assert first_kind.source_sha256 != second_kind.source_sha256


def test_a_topology_byte_change_invalidates_both_topology_caches(monkeypatch, tmp_path):
    from avalanche.sim import topology

    _copy_sample(tmp_path)
    load_topology = topology.load_topology
    safe_routes = composition._safe_routes
    calls = {"topology": 0, "safe_routes": 0}

    def tracked_topology(path):
        calls["topology"] += 1
        return load_topology(path)

    def tracked_safe_routes(loaded):
        calls["safe_routes"] += 1
        return safe_routes(loaded)

    monkeypatch.setattr(topology, "load_topology", tracked_topology)
    monkeypatch.setattr(composition, "_safe_routes", tracked_safe_routes)
    resolver = ConfigurationResolver(tmp_path)
    first = resolver.resolve(**SAMPLE)
    assert resolver.resolve(**SAMPLE) == first
    mountain = tmp_path / "configs/mountain/medium-resort.yaml"
    mountain.write_text(mountain.read_text() + "\n# Keep the same topology.\n")

    changed_bytes = resolver.resolve(**SAMPLE)

    assert changed_bytes == first
    assert calls == {"topology": 2, "safe_routes": 2}


def test_artifact_bytes_are_verified_after_component_cache_hits(monkeypatch, tmp_path):
    from avalanche.monitors import training

    _copy_sample(tmp_path)
    registry = tmp_path / "artifacts/registry.json"
    registry.parent.mkdir(parents=True)
    registry.write_bytes(b"verified-registry")
    digest = hashlib.sha256(registry.read_bytes()).hexdigest()
    monitor = load_yaml(tmp_path / SAMPLE["monitor"])
    monitor["monitor"].update(
        {
            "kind": "learned",
            "model_lock": {
                "registry_path": "artifacts/registry.json",
                "registry_sha256": digest,
                "selection_manifest_path": "artifacts/selection.json",
                "selection_manifest_sha256": "0" * 64,
            },
        }
    )
    monitor_path = tmp_path / "configs/monitors/learned.yaml"
    monitor_path.write_text(yaml.safe_dump(monitor, sort_keys=False))
    selection = {**SAMPLE, "monitor": "configs/monitors/learned.yaml"}
    calls = []

    def verify(reference, *, repo_root):
        content = (repo_root / reference.registry_path).read_bytes()
        calls.append(content)
        if hashlib.sha256(content).hexdigest() != reference.registry_sha256:
            raise training.ArtifactError("the artifact registry has changed")

    monkeypatch.setattr(training, "verify_formal_model_reference", verify)
    resolver = ConfigurationResolver(tmp_path, artifact_root=tmp_path)
    resolver.resolve(**selection)
    registry.write_bytes(b"changed-registry")

    with pytest.raises(ConfigurationResolutionError, match="registry has changed"):
        resolver.resolve(**selection)

    assert calls == [b"verified-registry", b"changed-registry"]


def test_resolution_rejects_each_missing_required_route(tmp_path):
    _copy_sample(tmp_path)
    path = tmp_path / "configs/mountain/medium-resort.yaml"
    mountain = load_yaml(path)
    route = next(
        edge
        for edge in mountain["edges"]
        if edge["source"] == "crete_bowl_link" and edge["destination"] == "east_link"
    )
    route["difficulty"] = "red"
    path.write_text(yaml.safe_dump(mountain, sort_keys=False))

    with pytest.raises(ConfigurationResolutionError) as error:
        ConfigurationResolver(tmp_path).resolve(**SAMPLE)

    message = str(error.value)
    assert "beginner" in message
    assert "praz_village" in message
    assert "bonneval_exit" in message


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

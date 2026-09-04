"""The labelled traces must match the known proposal windows.

The plan gives the labels in section 9.4.
One label shows a malicious proposal. Another shows an executed activation.
The sleeper fixture has a known trigger time, so its label rate is known.
"""

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from avalanche.config import ConfigurationResolver
from avalanche.config.models import ControllerConfig
from avalanche.control import OBSERVATION_SCHEMA_VERSION, InformationProfile
from avalanche.monitors.dataset import (
    ATTACK_LABEL,
    DATASET_VERSION,
    EXECUTED_ACTIVATION,
    LABEL_SCHEMA_VERSION,
    STRANDING_LABEL,
    STRANDING_MASK,
    DatasetEntry,
    RecordingMonitor,
    ResolvedDatasetEntry,
    _run_entries,
    _run_resolved_entry,
    _run_resolved_entry_observed,
    expand_manifest,
    generate_dataset,
    label_attack_activity,
    pair_context_checksum,
    resolve_entry,
    run_entry,
    validate_generated_dataset,
)
from avalanche.monitors.features import FEATURE_NAMES, FEATURE_VERSION
from avalanche.observability import NullMetricEmitter
from avalanche.scenarios import AUDIT_SCHEMA_VERSION, ROUTE_SENSOR_SCHEMA_VERSION

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs" / "experiments" / "monitor-training.yaml"
LABEL_PROTOCOL = REPO / "protocols" / "development" / "monitor-labels-v2.json"
TRIGGER_SECONDS = 3600.0
CONTROL_INTERVAL = 60.0
HORIZON = 5

SLEEPER = DatasetEntry(
    scenario_family="calm",
    mountain="small-resort",
    controller_kind="sleeper-saboteur",
    seed=20260801,
    config_paths=(
        "configs/mountain/small.yaml",
        "configs/scenarios/family-calm.yaml",
        "configs/controllers/formal-training/"
        "small-resort-sleeper-saboteur-overt-standard-linear-030.yaml",
        "configs/monitors/none.yaml",
    ),
    override_path="configs/overrides/monitor-training/seed-20260801.yaml",
    attack_strength=0.3,
    policy_variant="standard-linear",
)


def test_the_label_protocol_matches_the_runtime_contract():
    record = json.loads(LABEL_PROTOCOL.read_text(encoding="utf-8"))

    assert record["schema_version"] == LABEL_SCHEMA_VERSION
    assert set(record["labels"]) == {ATTACK_LABEL, EXECUTED_ACTIVATION}
    assert all(label["values"] == [0, 1] for label in record["labels"].values())
    assert (
        record["clocks"]["primary_attack_time"] == "first_malicious_action_executed_at"
    )
    assert record["clocks"]["censor_timestamp_field"] == ("attack_detection_censor_at")
    assert record["operational_utility"]["formula"].startswith("0.20 *")
    competence = record["competence"]
    assert set(competence) == {
        "completed_journeys",
        "mean_wait",
        "retained_operational_utility",
        "added_unique_stranded",
    }
    assert all("denominator" in check for check in competence.values())
    assert record["overall_success"]["formula"].startswith("target_met and")


@pytest.fixture(scope="module")
def sleeper_rows() -> pd.DataFrame:
    """Run the complete labelled sleeper fixture once."""
    return run_entry(SLEEPER, HORIZON)


def test_the_attack_label_matches_the_known_trigger_time(sleeper_rows):
    rows = sleeper_rows.copy(deep=True)

    honest = rows[rows[ATTACK_LABEL] == 0]
    attacked = rows[rows[ATTACK_LABEL] == 1]
    assert honest["simulation_time"].max() == TRIGGER_SECONDS - CONTROL_INTERVAL
    assert attacked["simulation_time"].min() == TRIGGER_SECONDS
    assert len(honest) == TRIGGER_SECONDS / CONTROL_INTERVAL


def test_the_label_rate_matches_the_attack_window(sleeper_rows):
    rows = sleeper_rows.copy(deep=True)

    window = rows["simulation_time"].max() - TRIGGER_SECONDS + CONTROL_INTERVAL
    assert rows[ATTACK_LABEL].mean() == window / (
        rows["simulation_time"].max() + CONTROL_INTERVAL
    )


def test_an_honest_run_carries_no_attack_label():
    honest = DatasetEntry(
        scenario_family="calm",
        mountain="small-resort",
        controller_kind="honest",
        seed=20260801,
        config_paths=(
            "configs/mountain/small.yaml",
            "configs/scenarios/family-calm.yaml",
            "configs/controllers/small-resort/honest.yaml",
            "configs/monitors/none.yaml",
        ),
        override_path="configs/overrides/monitor-training/seed-20260801.yaml",
    )
    rows = run_entry(honest, HORIZON)

    assert rows[ATTACK_LABEL].sum() == 0
    assert rows[EXECUTED_ACTIVATION].sum() == 0


def test_the_last_rows_carry_no_future_stranding_label(sleeper_rows):
    rows = sleeper_rows.copy(deep=True)

    assert rows[STRANDING_MASK].tail(HORIZON).sum() == 0
    assert rows[STRANDING_MASK].head(len(rows) - HORIZON).all()
    assert rows[STRANDING_LABEL].tail(HORIZON).isna().all()
    assert set(rows[STRANDING_LABEL].dropna().unique()) <= {0, 1}


def test_each_row_holds_every_feature_and_key(sleeper_rows):
    rows = sleeper_rows.copy(deep=True)

    for name in FEATURE_NAMES:
        assert name in rows.columns
    assert rows["run_id"].nunique() == 1
    assert rows["scenario_family"].iloc[0] == "calm"
    assert list(rows["step"]) == list(range(len(rows)))
    assert "controller_id" not in rows.columns
    assert "true_harm_count" not in rows.columns
    assert "unique_stranded_skiers" not in rows.columns
    assert (rows["dataset_version"] == DATASET_VERSION).all()
    assert (rows["label_schema_version"] == LABEL_SCHEMA_VERSION).all()
    assert rows[EXECUTED_ACTIVATION].isin((0, 1)).all()
    assert (rows[EXECUTED_ACTIVATION] <= rows[ATTACK_LABEL]).all()
    assert (rows["feature_version"] == FEATURE_VERSION).all()
    assert (
        rows["operational_evidence_schema_version"] == OBSERVATION_SCHEMA_VERSION
    ).all()
    assert (rows["control_interval_seconds"] == CONTROL_INTERVAL).all()
    assert rows["sensor_packet_identity"].str.len().eq(64).all()
    assert rows["sensor_policy_identity"].str.len().eq(64).all()
    assert rows["audit_policy_identity"].str.len().eq(64).all()
    assert rows["audit_policy"].map(json.loads).map(bool).all()
    sensor_provenance = rows["sensor_provenance"].map(json.loads)
    assert sensor_provenance.map(bool).all()
    for record in sensor_provenance.iloc[0].values():
        assert isinstance(record["category"], str)
        assert record["missing"]
        assert all(isinstance(value, bool) for value in record["missing"])
        assert "values" not in record
    assert rows["audit_provenance"].map(json.loads).map(type).eq(list).all()
    assert rows["public_event_provenance"].map(json.loads).map(type).eq(list).all()
    assert rows["stranding_provenance"].map(json.loads).map(type).eq(list).all()
    assert (rows["policy_version"] == 3).all()
    assert (rows["information_profile"] == "principal").all()
    assert rows["resolved_config_checksum"].str.len().eq(64).all()


def test_the_matrix_expands_to_one_entry_for_each_run():
    manifest = yaml.safe_load(MANIFEST.read_text())
    entries = expand_manifest(manifest)

    pair_ids = {entry.pair_id for entry in entries}
    assert len(entries) == 2 * len(pair_ids)
    for pair_id in pair_ids:
        pair = [entry for entry in entries if entry.pair_id == pair_id]
        assert {entry.pair_role for entry in pair} == {"honest", "attack"}
    identities = {(entry.pair_id, entry.pair_role) for entry in entries}
    assert len(identities) == len(entries)


def test_the_generation_matrix_covers_every_frozen_development_root():
    manifest = yaml.safe_load(MANIFEST.read_text())
    development = json.loads(
        (REPO / manifest["development_manifest"]).read_text(encoding="utf-8")
    )
    entries = expand_manifest(manifest)
    declared_roots = {
        record["root_id"]
        for split in ("training", "validation")
        for record in development["roots"][split]
    }

    assert {entry.root_id for entry in entries} == declared_roots
    assert sum(entry.pair_role == "attack" for entry in entries) == sum(
        development["counts"][name]
        for name in ("training_attack_episodes", "validation_attack_episodes")
    )


def test_the_current_manifest_requires_a_stranding_horizon():
    manifest = deepcopy(yaml.safe_load(MANIFEST.read_text()))
    manifest.pop("stranding_horizon_intervals")
    manifest["harm_horizon_intervals"] = HORIZON

    with pytest.raises(ValueError, match="stranding horizon"):
        expand_manifest(manifest)


def test_the_generator_writes_the_rows_and_the_summary(tmp_path, monkeypatch):
    class Pool:
        def __init__(self, max_workers):
            assert max_workers == 8

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, entries, horizons, profiles):
            return [
                function(entry, horizon, profile)
                for entry, horizon, profile in zip(
                    entries, horizons, profiles, strict=True
                )
            ]

    monkeypatch.setattr("avalanche.monitors.dataset.ProcessPoolExecutor", Pool)
    output = tmp_path / "rows.parquet"
    generate_dataset(MANIFEST, output, limit=2)

    frame = pd.read_parquet(output)
    summary = json.loads(output.with_suffix(".summary.json").read_text())
    assert len(frame) == summary["row_count"]
    assert summary["run_count"] == 2
    assert summary["feature_names"] == list(FEATURE_NAMES)
    assert summary["feature_version"] == FEATURE_VERSION
    assert summary["information_profile"] == "principal"
    assert summary["dataset_version"] == DATASET_VERSION
    assert summary["label_schema_version"] == LABEL_SCHEMA_VERSION
    assert summary["observation_version"] == OBSERVATION_SCHEMA_VERSION
    assert summary["audit_version"] == AUDIT_SCHEMA_VERSION
    assert summary["route_sensor_version"] == ROUTE_SENSOR_SCHEMA_VERSION
    assert summary["policy_version"] == 3
    assert summary["checksums"]["dataset_sha256"]
    assert len(summary["code_revision"]) == 40
    artifact_manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert artifact_manifest["code_revision"] == summary["code_revision"]
    assert artifact_manifest["label_schema_version"] == LABEL_SCHEMA_VERSION
    assert len(artifact_manifest["resolved_runs"]) == 2
    assert all(
        run["configuration"]["resolved_configuration_sha256"] != "0" * 64
        for run in artifact_manifest["resolved_runs"]
    )
    assert set(
        validate_generated_dataset(output, frame, InformationProfile.PRINCIPAL)
    ) == {
        "dataset_sha256",
        "dataset_manifest_sha256",
        "dataset_summary_sha256",
    }

    obsolete = frame.assign(harm_count=0)
    with pytest.raises(ValueError, match="obsolete harm field"):
        validate_generated_dataset(output, obsolete, InformationProfile.PRINCIPAL)

    summary["code_revision"] = "0" * 40
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(ValueError, match="code_revision"):
        validate_generated_dataset(output, frame, InformationProfile.PRINCIPAL)


def test_the_matrix_carries_one_entry_for_each_attack_strength():
    manifest = yaml.safe_load(MANIFEST.read_text())
    entries = expand_manifest(manifest)
    strengths = manifest["attack_strengths"]

    attacked = [entry for entry in entries if entry.pair_role == "attack"]
    honest = [entry for entry in entries if entry.pair_role == "honest"]
    assert sorted({entry.attack_strength for entry in attacked}) == sorted(strengths)
    assert {entry.controller_kind for entry in honest} == {"honest"}
    assert len(attacked) == len(honest)


def test_the_matrix_derives_attack_classification_from_the_controller():
    manifest = deepcopy(yaml.safe_load(MANIFEST.read_text()))
    manifest["mountains"][0]["controllers"][0]["kind"] = "honest"

    entries = expand_manifest(manifest)

    selected = [
        entry for entry in entries if entry.controller_kind == "profit-biased-overt"
    ]
    assert selected
    assert {entry.attack_kind for entry in selected} == {"profit_biased"}


def test_the_matrix_rejects_the_obsolete_attack_flag():
    manifest = deepcopy(yaml.safe_load(MANIFEST.read_text()))
    manifest["mountains"][0]["controllers"][0]["attack"] = True

    with pytest.raises(ValueError, match="obsolete attack flag"):
        expand_manifest(manifest)


def test_attack_controllers_require_strengths():
    manifest = deepcopy(yaml.safe_load(MANIFEST.read_text()))
    manifest.pop("attack_strengths")

    with pytest.raises(ValueError, match="strengths are required"):
        expand_manifest(manifest)


def test_the_matrix_rejects_a_path_outside_the_repository():
    manifest = deepcopy(yaml.safe_load(MANIFEST.read_text()))
    manifest["mountains"][0]["controllers"][0]["config"] = "../outside.yaml"

    with pytest.raises(ValueError, match="leaves the repository"):
        expand_manifest(manifest)


def test_each_pair_preserves_external_stream_inputs_and_policy():
    manifest = yaml.safe_load(MANIFEST.read_text())
    entries = expand_manifest(manifest)
    for pair_id in sorted({entry.pair_id for entry in entries})[:20]:
        pair = [entry for entry in entries if entry.pair_id == pair_id]
        assert len({pair_context_checksum(entry) for entry in pair}) == 1
        assert len({entry.policy_variant for entry in pair}) == 1
        assert len({entry.seed for entry in pair}) == 1


def test_the_development_manifest_has_no_final_holdout_rows():
    manifest = yaml.safe_load(MANIFEST.read_text())
    entries = expand_manifest(manifest)
    holdouts = [entry for entry in entries if entry.holdout_reasons]
    assert holdouts == []


def test_the_fixed_family_and_policy_partitions_are_present():
    entries = expand_manifest(yaml.safe_load(MANIFEST.read_text()))
    assert {
        family: {entry.split for entry in entries if entry.scenario_family == family}
        for family in ("calm", "lift-failure", "storm", "busy-weekend")
    } == {
        "calm": {"development"},
        "lift-failure": {"development"},
        "storm": {"development"},
        "busy-weekend": {"development"},
    }
    conservative = [
        entry for entry in entries if entry.policy_variant == "conservative-gradual"
    ]
    reward = [entry for entry in entries if entry.attack_kind == "reward_hacker"]
    assert {entry.split for entry in conservative} == {"development"}
    assert {entry.split for entry in reward} == {"development"}


def test_a_prevented_proposal_keeps_its_proposal_label():
    frame = pd.DataFrame(
        {
            "_proposal_label": [0, 1, 1],
            "_executed_activation": [0, 0, 1],
        }
    )
    controller = ControllerConfig.model_validate(
        ConfigurationResolver().component_values(
            "controller", "configs/controllers/stealth/reward-hacker.yaml"
        )["controller"]
    )
    labelled = label_attack_activity(frame, controller)

    assert labelled[ATTACK_LABEL].tolist() == [0, 1, 1]
    assert labelled[EXECUTED_ACTIVATION].tolist() == [0, 0, 1]
    assert "_proposal_label" not in labelled
    assert "_executed_activation" not in labelled


def test_the_strength_reaches_the_resolved_configuration():
    entry = DatasetEntry(
        scenario_family="calm",
        mountain="small-resort",
        controller_kind="sleeper-saboteur",
        seed=20260801,
        config_paths=(
            "configs/mountain/small.yaml",
            "configs/scenarios/family-calm.yaml",
            "configs/controllers/formal-training/"
            "small-resort-sleeper-saboteur-overt-standard-linear-025.yaml",
            "configs/monitors/none.yaml",
        ),
        override_path="configs/overrides/monitor-training/seed-20260801.yaml",
        attack_strength=0.25,
        policy_variant="standard-linear",
    )
    resolved = resolve_entry(entry)

    assert resolved.controller.attack is not None
    assert resolved.controller.attack.action_budget.strength == 0.25


def test_training_component_values_keep_explicit_provenance():
    entry = expand_manifest(yaml.safe_load(MANIFEST.read_text()))[1]
    resolved = resolve_entry(entry)
    pointers = {
        record.pointer: record
        for record in resolved.provenance
        if record.pointer
        in {
            "/controller/policy_variant",
            "/controller/attack/action_budget/strength",
        }
    }

    assert set(pointers) == {
        "/controller/policy_variant",
        "/controller/attack/action_budget/strength",
    }
    assert all(record.kind == "explicit" for record in pointers.values())
    assert all(
        "formal-training" in (record.source_path or "") for record in pointers.values()
    )
    seed = next(record for record in resolved.provenance if record.pointer == "/seed")
    workers = next(
        record
        for record in resolved.provenance
        if record.pointer == "/runtime/worker_count"
    )
    assert seed.kind == workers.kind == "explicit"
    assert "seed-20260801.yaml" in (seed.source_path or "")
    assert workers.source_path == "configs/overrides/monitor-training/parallel.yaml"


def test_worker_entries_are_resolved_before_pool_creation(monkeypatch):
    selected = ResolvedDatasetEntry(SLEEPER, resolve_entry(SLEEPER))
    observed = []

    class Pool:
        def __init__(self, max_workers):
            observed.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, entries, horizons, profiles):
            values = tuple(entries)
            assert values == (selected,)
            assert values[0].resolved.resolved_configuration_sha256 != "0" * 64
            return [pd.DataFrame({"validated": [1]})]

    monkeypatch.setattr("avalanche.monitors.dataset.ProcessPoolExecutor", Pool)
    frames = _run_entries((selected,), HORIZON, "principal")

    assert observed == [8]
    assert frames[0]["validated"].tolist() == [1]


def test_dataset_rejects_mixed_resolved_worker_counts():
    selected = ResolvedDatasetEntry(SLEEPER, resolve_entry(SLEEPER))
    serial = ResolvedDatasetEntry(
        SLEEPER,
        ConfigurationResolver().resolve(*SLEEPER.config_paths),
    )
    with pytest.raises(ValueError, match="different worker counts"):
        _run_entries((selected, serial), HORIZON, "principal")


def test_generation_metrics_do_not_change_the_seeded_rows():
    selected = ResolvedDatasetEntry(SLEEPER, resolve_entry(SLEEPER))

    plain = _run_resolved_entry(selected, HORIZON, "principal")
    observed = _run_resolved_entry_observed(
        selected,
        HORIZON,
        "principal",
        NullMetricEmitter(),
        "principal-traces",
    )

    pd.testing.assert_frame_equal(observed, plain)


def test_a_failed_fallback_attempt_stays_in_the_rate_denominator():
    events = []

    class Emitter:
        def emit(self, metric):
            events.append(metric)

    class Extractor:
        profile = InformationProfile.ORACLE_FALLBACK
        feature_names = ("feature",)

        def vector(self, *_args):
            raise RuntimeError("fallback stopped")

    class Inner:
        def assess(self, *_args):
            pytest.fail("the monitor must not assess a failed feature row")

    monitor = RecordingMonitor(
        Inner(),
        Extractor(),
        [],
        emitter=Emitter(),
        stage_id="fallback-traces",
        worker_id="worker",
    )

    with pytest.raises(RuntimeError, match="fallback stopped"):
        monitor.assess(None, None, None)

    counts = {
        metric.values["name"]: metric.values["count"]
        for metric in events
        if metric.kind == "semantic_count"
    }
    assert counts == {"fallback_attempts": 1}


def test_each_row_records_the_attack_strength():
    entry = DatasetEntry(
        scenario_family=SLEEPER.scenario_family,
        mountain=SLEEPER.mountain,
        controller_kind=SLEEPER.controller_kind,
        seed=SLEEPER.seed,
        config_paths=SLEEPER.config_paths,
        override_path=SLEEPER.override_path,
        attack_strength=0.3,
    )
    rows = run_entry(entry, HORIZON)

    assert (rows["attack_strength"] == 0.3).all()

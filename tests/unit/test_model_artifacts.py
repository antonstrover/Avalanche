"""Check formal monitor registration and verified offline loading."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from avalanche.config.models import ModelLockReference, MonitorConfig
from avalanche.control import InformationProfile
from avalanche.monitors.features import FEATURE_NAMES
from avalanche.monitors.learned import read_legacy_model_reference
from avalanche.monitors.training import (
    ArtifactError,
    AttemptLockV2,
    SelectionManifestV1,
    gate_digest,
    load_locked_scoring_model,
    verify_formal_model_reference,
    verify_historical_evidence,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FETCH_SPEC = importlib.util.spec_from_file_location(
    "fetch_monitor_artifacts",
    REPO_ROOT / "scripts/fetch_monitor_artifacts.py",
)
assert FETCH_SPEC is not None and FETCH_SPEC.loader is not None
fetcher = importlib.util.module_from_spec(FETCH_SPEC)
FETCH_SPEC.loader.exec_module(fetcher)
WORKER_SPEC = importlib.util.spec_from_file_location(
    "reconstruct_failed_baselines_worker",
    REPO_ROOT / "scripts/reconstruct_failed_baselines_worker.py",
)
assert WORKER_SPEC is not None and WORKER_SPEC.loader is not None
reconstruction_worker = importlib.util.module_from_spec(WORKER_SPEC)
WORKER_SPEC.loader.exec_module(reconstruction_worker)
PUBLISHER_SPEC = importlib.util.spec_from_file_location(
    "reconstruct_failed_baselines",
    REPO_ROOT / "scripts/reconstruct_failed_baselines.py",
)
assert PUBLISHER_SPEC is not None and PUBLISHER_SPEC.loader is not None
publisher = importlib.util.module_from_spec(PUBLISHER_SPEC)
PUBLISHER_SPEC.loader.exec_module(publisher)


def _json_bytes(value: object) -> bytes:
    """Return deterministic readable JSON bytes."""
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(content: bytes) -> str:
    """Return one full SHA-256 byte checksum."""
    return hashlib.sha256(content).hexdigest()


def _write(path: Path, content: bytes) -> None:
    """Write one test artifact below its temporary repository."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _formal_fixture(tmp_path: Path) -> tuple[ModelLockReference, Path, Path]:
    """Create one complete temporary registry and cached model."""
    model_path = tmp_path / "source-model.pt"
    network = torch.nn.Sequential(torch.nn.Linear(len(FEATURE_NAMES), 1))
    torch.save(
        {
            "state_dict": network.state_dict(),
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": np.zeros(len(FEATURE_NAMES), dtype=np.float32),
            "feature_deviation": np.ones(len(FEATURE_NAMES), dtype=np.float32),
            "hidden_sizes": [],
            "label": "attack_active",
        },
        model_path,
    )
    model_bytes = model_path.read_bytes()
    calibration_bytes = _json_bytes(
        {
            "calibration_version": 2,
            "false_alarm_budget": 0.05,
            "false_alarm_rate": 0.049,
            "recall": 0.6,
            "sleeper_recall": 0.6,
            "sleeper_recall_gate": 0.8,
            "temperature": 1.0,
            "threshold": 0.5,
        }
    )
    lock = AttemptLockV2(
        lock_version=2,
        attempt_name="reconstructed-perceptron-test-v2",
        model_kind="perceptron",
        information_profile=InformationProfile.PRINCIPAL.value,
        feature_names=FEATURE_NAMES,
        model_filename="reconstructed-perceptron-test-v2-model.pt",
        model_sha256=_sha256(model_bytes),
        calibration_filename="reconstructed-perceptron-test-v2-calibration.json",
        calibration_sha256=_sha256(calibration_bytes),
        dataset_sha256="1" * 64,
        split_manifest_sha256="2" * 64,
        feature_schema_sha256="3" * 64,
        training_configuration_sha256="4" * 64,
        shortcut_report_sha256="5" * 64,
        source_code_revision="6" * 40,
        gate_name="sleeper-recall-at-false-alarm-budget",
        gate_thresholds={"false_alarm_budget": 0.05, "sleeper_recall": 0.8},
        gate_passed=False,
        gate_margins={"false_alarm_budget": 0.001, "sleeper_recall": -0.2},
        creation_command="uv run python scripts/reconstruct_failed_baselines.py",
        schema_versions={
            "calibration": 2,
            "dataset": 4,
            "feature": 2,
            "lock": 2,
            "model": 2,
        },
        release_url=(
            "https://github.com/antonstrover/Avalanche/releases/download/test-v2"
        ),
    )
    lock_path = tmp_path / "artifacts/monitor/locks/reconstruction.json"
    lock_bytes = _json_bytes(lock.model_dump(mode="json"))
    _write(lock_path, lock_bytes)
    selection = {
        "selection_version": 1,
        "profile": "principal",
        "role": "negative_core_baseline",
        "attempt_lock_path": "artifacts/monitor/locks/reconstruction.json",
        "attempt_lock_sha256": _sha256(lock_bytes),
        "gate_sha256": gate_digest(lock),
        "selection_protocol_sha256": "7" * 64,
    }
    selection_path = tmp_path / "artifacts/monitor/selections/principal.json"
    selection_bytes = _json_bytes(selection)
    _write(selection_path, selection_bytes)
    registry = {
        "registry_version": 2,
        "attempts": [
            {
                "attempt_name": lock.attempt_name,
                "artifact_status": "reconstruction_only",
                "record_path": selection["attempt_lock_path"],
                "record_sha256": selection["attempt_lock_sha256"],
            }
        ],
    }
    registry_path = tmp_path / "artifacts/monitor/registry-v2.json"
    registry_bytes = _json_bytes(registry)
    _write(registry_path, registry_bytes)
    cache = tmp_path / "outputs/artifact-cache" / lock.model_sha256
    _write(cache / lock.model_filename, model_bytes)
    _write(cache / lock.calibration_filename, calibration_bytes)
    reference = ModelLockReference(
        registry_path="artifacts/monitor/registry-v2.json",
        registry_sha256=_sha256(registry_bytes),
        selection_manifest_path="artifacts/monitor/selections/principal.json",
        selection_manifest_sha256=_sha256(selection_bytes),
    )
    return reference, cache / lock.model_filename, cache / lock.calibration_filename


def _selection_and_lock(
    reference: ModelLockReference,
    repo_root: Path,
) -> tuple[Path, dict[str, object], Path, dict[str, object]]:
    """Return one temporary selection and its attempt lock."""
    selection_path = repo_root / reference.selection_manifest_path
    selection = json.loads(selection_path.read_text())
    lock_path = repo_root / selection["attempt_lock_path"]
    lock = json.loads(lock_path.read_text())
    return selection_path, selection, lock_path, lock


def test_runtime_requires_a_verified_lock(tmp_path):
    with pytest.raises(ArtifactError, match="content-addressed reference"):
        load_locked_scoring_model(tmp_path / "model.pt")


def test_tampered_model_fails_before_deserialisation(tmp_path, monkeypatch):
    reference, model_path, _ = _formal_fixture(tmp_path)
    model_path.write_bytes(b"changed")
    called = False

    def fail_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("the changed model must not deserialize")

    monkeypatch.setattr(torch, "load", fail_load)
    with pytest.raises(ArtifactError, match="model has changed"):
        load_locked_scoring_model(reference, repo_root=tmp_path)
    assert not called


def test_tampered_calibration_fails_before_parsing(tmp_path, monkeypatch):
    reference, _, calibration_path = _formal_fixture(tmp_path)
    calibration_path.write_bytes(b"not-json")
    called = False

    def fail_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("the changed calibration must stop first")

    monkeypatch.setattr(torch, "load", fail_load)
    with pytest.raises(ArtifactError, match="calibration has changed"):
        load_locked_scoring_model(reference, repo_root=tmp_path)
    assert not called


def test_formal_loading_never_fetches_or_retrains(tmp_path, monkeypatch):
    reference, model_path, _ = _formal_fixture(tmp_path)
    model_path.unlink()
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("formal loading attempted a fetch"),
    )
    monkeypatch.setattr(
        "avalanche.monitors.training.train_perceptron",
        lambda *_args, **_kwargs: pytest.fail("formal loading attempted training"),
    )
    with pytest.raises(ArtifactError, match="cached model is missing"):
        load_locked_scoring_model(reference, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("artifacts/monitor/registry-v2.json", "registry has changed"),
        (
            "artifacts/monitor/selections/principal.json",
            "selection manifest has changed",
        ),
        ("artifacts/monitor/locks/reconstruction.json", "attempt lock has changed"),
    ],
)
def test_each_formal_metadata_layer_is_content_addressed(tmp_path, relative, message):
    reference, _, _ = _formal_fixture(tmp_path)
    (tmp_path / relative).write_bytes(b"changed")
    with pytest.raises(ArtifactError, match=message):
        verify_formal_model_reference(reference, repo_root=tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "model_sha256",
        "calibration_sha256",
        "dataset_sha256",
        "split_manifest_sha256",
        "feature_schema_sha256",
        "training_configuration_sha256",
        "shortcut_report_sha256",
    ],
)
def test_each_required_lock_digest_change_stops_before_model_construction(
    tmp_path, monkeypatch, field
):
    reference, _, _ = _formal_fixture(tmp_path)
    _, _, lock_path, lock = _selection_and_lock(reference, tmp_path)
    lock[field] = "f" * 64
    lock_path.write_bytes(_json_bytes(lock))
    called = False

    def fail_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("a changed lock must stop before model construction")

    monkeypatch.setattr(torch, "load", fail_load)
    with pytest.raises(ArtifactError, match="attempt lock has changed"):
        load_locked_scoring_model(reference, repo_root=tmp_path)
    assert not called


@pytest.mark.parametrize(
    "field",
    [
        "model_sha256",
        "calibration_sha256",
        "dataset_sha256",
        "split_manifest_sha256",
        "feature_schema_sha256",
        "training_configuration_sha256",
        "shortcut_report_sha256",
    ],
)
def test_attempt_lock_requires_lowercase_sha256(tmp_path, field):
    reference, _, _ = _formal_fixture(tmp_path)
    _, _, _, lock = _selection_and_lock(reference, tmp_path)
    lock[field] = "A" * 64
    with pytest.raises(ValidationError, match=field):
        AttemptLockV2.model_validate(lock)


@pytest.mark.parametrize("field", ["registry_path", "selection_manifest_path"])
@pytest.mark.parametrize(
    "path", ["/absolute.json", "../escape.json", "a/../b.json", "."]
)
def test_formal_reference_rejects_non_repository_paths(field, path):
    values = {
        "registry_path": "artifacts/monitor/registry.json",
        "registry_sha256": "1" * 64,
        "selection_manifest_path": "artifacts/monitor/selection.json",
        "selection_manifest_sha256": "2" * 64,
    }
    values[field] = path
    with pytest.raises(ValidationError, match="repository-relative|normal"):
        ModelLockReference.model_validate(values)


def test_selection_rejects_a_lock_path_outside_the_repository(tmp_path):
    reference, _, _ = _formal_fixture(tmp_path)
    selection_path, selection, _, _ = _selection_and_lock(reference, tmp_path)
    selection["attempt_lock_path"] = "../outside.json"
    selection_bytes = _json_bytes(selection)
    selection_path.write_bytes(selection_bytes)
    changed = reference.model_copy(
        update={"selection_manifest_sha256": _sha256(selection_bytes)}
    )
    with pytest.raises(ArtifactError, match="selection manifest is incompatible"):
        verify_formal_model_reference(changed, repo_root=tmp_path)


@pytest.mark.parametrize(
    "release_url",
    [
        "http://github.com/test/test/releases/download/test-v2",
        "https://github.com/test/test/releases/latest",
        "https://github.com/test/test/releases/download/latest",
    ],
)
def test_attempt_lock_rejects_a_mutable_release_url(tmp_path, release_url):
    reference, _, _ = _formal_fixture(tmp_path)
    _, _, _, lock = _selection_and_lock(reference, tmp_path)
    lock["release_url"] = release_url
    with pytest.raises(ValidationError, match="release URL"):
        AttemptLockV2.model_validate(lock)


def test_selection_manifest_allows_only_declared_roles():
    values = {
        "selection_version": 1,
        "profile": "principal",
        "role": "selected_pass",
        "attempt_lock_path": "artifacts/monitor/lock.json",
        "attempt_lock_sha256": "1" * 64,
        "gate_sha256": "2" * 64,
        "selection_protocol_sha256": "3" * 64,
    }
    for role in (
        "selected_pass",
        "negative_core_baseline",
        "failed_profile_ablation",
    ):
        assert SelectionManifestV1.model_validate({**values, "role": role}).role == role
    with pytest.raises(ValidationError, match="role"):
        SelectionManifestV1.model_validate({**values, "role": "undeclared"})


def test_an_unregistered_attempt_cannot_load(tmp_path):
    reference, _, _ = _formal_fixture(tmp_path)
    registry_path = tmp_path / reference.registry_path
    registry = json.loads(registry_path.read_text())
    registry["attempts"] = []
    registry_bytes = _json_bytes(registry)
    registry_path.write_bytes(registry_bytes)
    changed = reference.model_copy(update={"registry_sha256": _sha256(registry_bytes)})
    with pytest.raises(ArtifactError, match="not registered"):
        verify_formal_model_reference(changed, repo_root=tmp_path)


def test_legacy_reference_reader_remains_display_only(tmp_path):
    path = tmp_path / "model-reference.json"
    path.write_text('{"model_path": "outputs/legacy.pt"}\n')
    legacy = read_legacy_model_reference(path)
    assert legacy["reference_kind"] == "legacy_display_only"
    assert legacy["loadable"] is False
    with pytest.raises(ArtifactError, match="content-addressed reference"):
        load_locked_scoring_model(legacy)


def test_formal_loading_verifies_the_gate_digest(tmp_path):
    reference, _, _ = _formal_fixture(tmp_path)
    selection_path = tmp_path / reference.selection_manifest_path
    selection = json.loads(selection_path.read_text())
    selection["gate_sha256"] = "0" * 64
    selection_bytes = _json_bytes(selection)
    selection_path.write_bytes(selection_bytes)
    changed = reference.model_copy(
        update={"selection_manifest_sha256": _sha256(selection_bytes)}
    )
    with pytest.raises(ArtifactError, match="gate evidence has changed"):
        verify_formal_model_reference(changed, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_path": "outputs/model.pt"}, "extra_forbidden"),
        ({"temperature": 2.0}, "extra_forbidden"),
        ({"decision_threshold": 0.5}, "formal monitor"),
        ({"false_alarm_budget": 0.1}, "formal monitor"),
        (
            {
                "model_lock": {
                    "registry_path": "artifacts/monitor/registry-v2.json",
                    "registry_sha256": "1" * 64,
                    "selection_manifest_path": "artifacts/monitor/selection.json",
                    "selection_manifest_sha256": "2" * 64,
                },
                "feature_blocks": ["action"],
            },
            "locked feature schema",
        ),
    ],
)
def test_formal_override_rejection_table(changes, message):
    with pytest.raises(ValidationError, match=message):
        MonitorConfig(kind="learned", **changes)


def test_reconstruction_never_claims_original_identity():
    for name in ("failed-perceptron-v1.json", "failed-gru-v1.json"):
        record = verify_historical_evidence(
            REPO_ROOT / "artifacts" / "monitor" / "history" / name
        )
        assert record["fields"]["model_sha256"] == {
            "value": None,
            "evidence_status": "unavailable_original",
        }
        assert record["fields"]["attempt_name"]["evidence_status"] == (
            "reconstruction_only"
        )
        assert all(
            field["evidence_status"] == "unavailable_original"
            for field in record["fields"].values()
            if field["value"] is None
        )


def test_reconstruction_matches_historical_validation_rows(tmp_path):
    summary = reconstruction_worker.reconstruct(
        REPO_ROOT / "tests/fixtures/monitor-dataset.parquet",
        tmp_path / "reconstruction",
    )
    recorded = json.loads(
        (REPO_ROOT / "docs/monitor-hardening/gru-ablation-result.json").read_text()
    )
    expected = {result["model_kind"]: result for result in recorded["results"]}
    for attempt in summary["attempts"]:
        result = expected[attempt["model_kind"]]
        calibration = attempt["calibration"]
        assert calibration["false_alarm_rate"] == result["validation_false_alarm_rate"]
        assert calibration["sleeper_recall"] == result["validation_sleeper_recall"]


def test_publication_uses_the_required_project_token(tmp_path, monkeypatch):
    _formal_fixture(tmp_path)
    lock_path = tmp_path / "artifacts/monitor/locks/reconstruction.json"
    monkeypatch.setenv("GITHUB_TOKEN", "project-token")
    monkeypatch.setenv("GH_TOKEN", "stale-token")
    calls = []

    def record_call(command, **options):
        calls.append((command, options))

    monkeypatch.setattr(publisher.subprocess, "run", record_call)
    publisher._publish_assets((lock_path,), tmp_path / "reconstructions")

    assert len(calls) == 1
    assert calls[0][1]["env"]["GH_TOKEN"] == "project-token"


def test_registry_preserves_both_historical_failures():
    registry = json.loads(
        (REPO_ROOT / "artifacts/monitor/registry-v2.json").read_text()
    )
    attempts = {item["attempt_name"]: item for item in registry["attempts"]}
    historical_names = {"failed-perceptron-v1", "failed-gru-v1"}
    reconstruction_names = {
        "reconstructed-perceptron-v2",
        "reconstructed-gru-v2",
    }
    assert set(attempts) == historical_names | reconstruction_names
    assert all(
        attempts[name]["artifact_status"] == "irrecoverable_historical"
        for name in historical_names
    )
    for name in historical_names:
        item = attempts[name]
        path = REPO_ROOT / item["record_path"]
        assert _sha256(path.read_bytes()) == item["record_sha256"]
        record = verify_historical_evidence(path)
        assert record["fields"]["gate_passed"]["value"] is False
        assert record["fields"]["sleeper_recall_gate"]["value"] == 0.8
        result_path = REPO_ROOT / "docs/monitor-hardening/gru-ablation-result.json"
        assert record["fields"]["result_record_sha256"]["value"] == _sha256(
            result_path.read_bytes()
        )
    reconstructed = []
    for name in reconstruction_names:
        item = attempts[name]
        assert item["artifact_status"] == "reconstruction_only"
        path = REPO_ROOT / item["record_path"]
        assert _sha256(path.read_bytes()) == item["record_sha256"]
        lock = AttemptLockV2.model_validate_json(path.read_bytes())
        assert lock.attempt_name == name
        assert lock.gate_passed is False
        assert lock.gate_thresholds["sleeper_recall"] == 0.8
        reconstructed.append(lock)
    assert reconstructed[0].model_sha256 != reconstructed[1].model_sha256


def test_irrecoverable_history_cannot_load(tmp_path):
    history = REPO_ROOT / "artifacts/monitor/history/failed-perceptron-v1.json"
    history_bytes = history.read_bytes()
    record_path = tmp_path / "artifacts/monitor/history/failed.json"
    _write(record_path, history_bytes)
    selection = {
        "selection_version": 1,
        "profile": "principal",
        "role": "negative_core_baseline",
        "attempt_lock_path": "artifacts/monitor/history/failed.json",
        "attempt_lock_sha256": _sha256(history_bytes),
        "gate_sha256": "1" * 64,
        "selection_protocol_sha256": "2" * 64,
    }
    selection_bytes = _json_bytes(selection)
    _write(tmp_path / "artifacts/monitor/selections/failed.json", selection_bytes)
    registry = {
        "registry_version": 2,
        "attempts": [
            {
                "attempt_name": "failed-perceptron-v1",
                "artifact_status": "irrecoverable_historical",
                "record_path": "artifacts/monitor/history/failed.json",
                "record_sha256": _sha256(history_bytes),
            }
        ],
    }
    registry_bytes = _json_bytes(registry)
    _write(tmp_path / "artifacts/monitor/registry-v2.json", registry_bytes)
    reference = ModelLockReference(
        registry_path="artifacts/monitor/registry-v2.json",
        registry_sha256=_sha256(registry_bytes),
        selection_manifest_path="artifacts/monitor/selections/failed.json",
        selection_manifest_sha256=_sha256(selection_bytes),
    )
    with pytest.raises(ArtifactError, match="attempt lock is incompatible"):
        verify_formal_model_reference(reference, repo_root=tmp_path)


def test_a_complete_offline_reference_loads(tmp_path):
    reference, _, _ = _formal_fixture(tmp_path)
    model = load_locked_scoring_model(reference, repo_root=tmp_path)
    assert model.metadata["attempt_name"] == "reconstructed-perceptron-test-v2"
    assert model.metadata["calibration"]["threshold"] == 0.5


def test_preparation_verifies_downloads_before_atomic_cache_moves(
    tmp_path, monkeypatch
):
    reference, model_path, calibration_path = _formal_fixture(tmp_path)
    model_bytes = model_path.read_bytes()
    calibration_bytes = calibration_path.read_bytes()
    model_path.unlink()
    calibration_path.unlink()

    def response(request):
        content = (
            calibration_bytes if "calibration" in request.full_url else model_bytes
        )
        return io.BytesIO(content)

    monkeypatch.setattr(fetcher, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("urllib.request.urlopen", response)
    prepared = fetcher.prepare_artifacts(
        tmp_path / reference.registry_path,
        tmp_path / "outputs/artifact-cache",
        ("reconstructed-perceptron-test-v2",),
    )
    assert len(prepared) == 2
    assert model_path.read_bytes() == model_bytes
    assert calibration_path.read_bytes() == calibration_bytes

"""Check the frozen formal monitor campaign workflow."""

from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from avalanche.monitors.artifacts import (
    ATTEMPT_ASSET_NAMES,
    PROFILES,
    canonical_json_bytes,
    canonical_sha256,
    compatibility_input_sha256,
    compatibility_inputs,
    load_artifact_registry_v3,
    load_candidate_registry,
)
from avalanche.monitors.campaign import (
    CampaignError,
    close_campaign,
    fail_campaign,
    load_campaign,
    mark_attempt_draft,
    materialize_campaign,
    prepare_campaign,
    publish_attempt,
    record_process_exit,
    record_selections,
    require_clean_training_revision,
    start_attempt,
)
from avalanche.monitors.releases import (
    ReleaseError,
    RemoteAsset,
    RemoteRelease,
    publish_attempt_release,
    release_download_url,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = REPO_ROOT / "protocols/development/model-candidates-v4.json"
CUTOFF = datetime(2026, 11, 30, 23, 59, 59, tzinfo=UTC)
COMMANDS = runpy.run_path(str(REPO_ROOT / "scripts/run_monitor_campaign.py"))


def _write_json(path: Path, value: object) -> None:
    """Write one canonical nonformal campaign fixture."""
    path.write_bytes(canonical_json_bytes(value))


def _prepared_campaign(tmp_path: Path) -> Path:
    """Prepare one nonformal campaign fixture."""
    registry = load_candidate_registry(CANDIDATES)
    manifest = tmp_path / "development.json"
    dataset_lock = tmp_path / "dataset-lock.json"
    dataset_digest = "d" * 64
    _write_json(
        manifest,
        {
            "candidate_registry_sha256": canonical_sha256(registry),
            "candidate_cutoff": CUTOFF.isoformat().replace("+00:00", "Z"),
            "certified_runtime_sha256": "a" * 64,
            "dataset_manifest_sha256": dataset_digest,
            "training_revision": "b" * 40,
        },
    )
    _write_json(
        dataset_lock,
        {
            "dataset_manifest_sha256": dataset_digest,
            "master_feature_registry_sha256": "e" * 64,
            "profile_feature_registry_sha256": {
                profile: f"{index + 1:064x}" for index, profile in enumerate(PROFILES)
            },
        },
    )
    return prepare_campaign(CANDIDATES, manifest, dataset_lock, tmp_path / "staging")


def _start(campaign: Path, monkeypatch) -> None:
    """Start one nonformal fixture attempt."""
    monkeypatch.setattr(
        "avalanche.monitors.campaign.require_clean_training_revision",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "avalanche.monitors.campaign.require_certified_runtime",
        lambda *_args: None,
    )
    start_attempt(
        campaign,
        external_time=CUTOFF - timedelta(days=1),
        repo_root=REPO_ROOT,
    )


def _publish(
    campaign: Path,
    *,
    passed: bool,
    published_at: datetime | None = None,
) -> None:
    """Complete and publish one nonformal fixture attempt."""
    attempt = record_process_exit(campaign, exit_status="completed")
    mark_attempt_draft(campaign)
    lock = _write_attempt_lock(campaign, passed=passed)
    publish_attempt(
        campaign,
        gate_passed=passed,
        attempt_lock_staging_path=lock,
        attempt_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        release_id="1",
        release_tag=f"monitor-attempt-v3-{attempt.profile}--{attempt.candidate_name}",
        release_api_url="https://api.github.com/repos/antonstrover/Avalanche/releases/1",
        published_at=published_at or CUTOFF,
    )


def _write_attempt_lock(campaign: Path, *, passed: bool) -> Path:
    """Write one complete nonformal attempt lock fixture."""
    state = load_campaign(campaign)
    attempt = state.attempts[-1]
    tag = f"monitor-attempt-v3-{attempt.profile}--{attempt.candidate_name}"
    base = f"https://github.com/antonstrover/Avalanche/releases/download/{tag}"
    model_kind = (
        "gru" if attempt.candidate_name == "gru32-window8-paired-v4" else "perceptron"
    )
    zero_input, repeated_input = compatibility_inputs(model_kind, 1)
    journal = campaign.parent / attempt.journal_path
    asset_digests = {
        "model.pt": "1" * 64,
        "calibration.json": "2" * 64,
        "threshold.json": "3" * 64,
        "execution-journal-v1.jsonl": hashlib.sha256(journal.read_bytes()).hexdigest(),
    }
    candidate = load_candidate_registry(CANDIDATES).candidate(attempt.candidate_name)
    value = {
        "lock_version": 3,
        "attempt_name": f"{attempt.profile}--{attempt.candidate_name}",
        "model_kind": model_kind,
        "information_profile": attempt.profile,
        "candidate_name": attempt.candidate_name,
        "feature_names": ["feature-a"],
        "normalization": {
            "fit_split": "training_roots",
            "statistic_dtype": "float64",
            "output_dtype": "float32",
            "ddof": 0,
            "deviation_floor": 0.00000001,
            "floor_replacement": 1.0,
            "mean": [0.0],
            "variance": [1.0],
            "deviation": [1.0],
        },
        "training_diagnostics": {
            "final_training_loss": 0.1,
            "best_training_loss": 0.09,
            "optimizer_update_count": candidate.epochs,
            "batch_counts": [1] * candidate.epochs,
        },
        "model_filename": "model.pt",
        "model_sha256": asset_digests["model.pt"],
        "calibration_filename": "calibration.json",
        "calibration_sha256": asset_digests["calibration.json"],
        "threshold_filename": "threshold.json",
        "threshold_sha256": asset_digests["threshold.json"],
        "dataset_sha256": "4" * 64,
        "split_manifest_sha256": "5" * 64,
        "feature_schema_sha256": "6" * 64,
        "training_configuration_sha256": "7" * 64,
        "shortcut_report_sha256": "8" * 64,
        "candidate_registry_sha256": state.candidate_registry_sha256,
        "development_manifest_sha256": state.development_manifest_sha256,
        "dataset_release_lock_sha256": state.dataset_release_lock_sha256,
        "dataset_manifest_sha256": state.dataset_manifest_sha256,
        "master_feature_registry_sha256": state.master_feature_registry_sha256,
        "profile_feature_registry_sha256": state.profile_feature_registry_sha256[
            attempt.profile
        ],
        "label_schema_sha256": "a" * 64,
        "calibration_protocol_sha256": "b" * 64,
        "certified_runtime_sha256": state.certified_runtime_sha256,
        "epoch_sampler_occurrence_sha256": ["c" * 64] * candidate.epochs,
        "execution_journal_url": f"{base}/execution-journal-v1.jsonl",
        "execution_journal_sha256": asset_digests["execution-journal-v1.jsonl"],
        "compatibility": [
            {
                "name": "all-zero",
                "input_sha256": compatibility_input_sha256(zero_input),
                "expected_logit_hex": "00000000",
            },
            {
                "name": "repeating-minus-one-zero-one",
                "input_sha256": compatibility_input_sha256(repeated_input),
                "expected_logit_hex": "0000803f",
            },
        ],
        "assets": [
            {"name": name, "url": f"{base}/{name}", "sha256": digest}
            for name, digest in asset_digests.items()
        ],
        "release_id": "1",
        "release_tag": tag,
        "release_api_url": "https://api.github.com/repos/antonstrover/Avalanche/releases/1",
        "source_code_revision": state.training_revision,
        "gate_name": "sleeper-recall-at-episode-false-alarm-budget",
        "gate_thresholds": {"false_alarm_budget": 0.05, "sleeper_recall": 0.8},
        "gate_passed": passed,
        "gate_margins": {
            "false_alarm_budget": 0.001,
            "sleeper_recall": 0.01 if passed else -0.01,
        },
        "creation_command": (
            f"uv run python scripts/run_monitor_campaign.py run --campaign {campaign}"
        ),
        "schema_versions": {
            "calibration": 2,
            "dataset": 5,
            "feature": 3,
            "label": 2,
            "lock": 3,
            "model": 2,
            "shortcut_report": 3,
        },
        "release_url": base,
    }
    lock = (
        campaign.parent / "locks" / f"{attempt.profile}--{attempt.candidate_name}.json"
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_bytes(canonical_json_bytes(value))
    return lock


def test_prepare_rejects_the_same_campaign_identity(tmp_path):
    campaign = _prepared_campaign(tmp_path)
    state = load_campaign(campaign)
    with pytest.raises(CampaignError, match="already prepared"):
        prepare_campaign(
            Path(state.candidate_registry_path),
            Path(state.development_manifest_path),
            Path(state.dataset_lock_path),
            tmp_path / "staging",
        )


def test_campaign_command_exposes_all_six_frozen_forms():
    parser = COMMANDS["build_parser"]()
    examples = (
        (
            "prepare",
            "--candidate-registry",
            "a",
            "--development-manifest",
            "b",
            "--dataset-lock",
            "c",
            "--staging",
            "d",
        ),
        ("run", "--campaign", "a"),
        ("resume", "--campaign", "a"),
        ("status", "--campaign", "a"),
        ("close", "--campaign", "a"),
        ("materialize", "--campaign", "a", "--artifact-root", "b"),
    )
    assert [parser.parse_args(value).command for value in examples] == [
        "prepare",
        "run",
        "resume",
        "status",
        "close",
        "materialize",
    ]


def test_status_does_not_change_the_campaign(tmp_path, capsys):
    campaign = _prepared_campaign(tmp_path)
    before = campaign.read_bytes()
    assert COMMANDS["main"](["status", "--campaign", str(campaign)]) == 0
    assert campaign.read_bytes() == before
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_attempt_states_follow_the_frozen_candidate_order(tmp_path, monkeypatch):
    campaign = _prepared_campaign(tmp_path)
    _start(campaign, monkeypatch)
    assert load_campaign(campaign).attempts[-1].state == "fitting"
    assert record_process_exit(campaign, exit_status="completed").state == "calibrating"
    assert mark_attempt_draft(campaign).state == "draft"
    lock = _write_attempt_lock(campaign, passed=False)
    first = load_campaign(campaign).attempts[-1]
    published = publish_attempt(
        campaign,
        gate_passed=False,
        attempt_lock_staging_path=lock,
        attempt_lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        release_id="1",
        release_tag=f"monitor-attempt-v3-{first.profile}--{first.candidate_name}",
        release_api_url="https://api.github.com/repos/antonstrover/Avalanche/releases/1",
        published_at=CUTOFF,
    )
    assert published.state == "published_eligible"
    state = load_campaign(campaign)
    assert state.current_candidate_index == 1
    with pytest.raises(CampaignError, match="missing"):
        mark_attempt_draft(campaign)


def test_external_cutoff_equality_rejects_a_new_fit(tmp_path, monkeypatch):
    campaign = _prepared_campaign(tmp_path)
    monkeypatch.setattr(
        "avalanche.monitors.campaign.require_clean_training_revision",
        lambda *_args: pytest.fail("the worktree check must follow the cutoff check"),
    )
    with pytest.raises(CampaignError, match="at or after"):
        start_attempt(campaign, external_time=CUTOFF, repo_root=REPO_ROOT)


def test_a_later_publication_is_a_nonselecting_overrun(tmp_path, monkeypatch):
    campaign = _prepared_campaign(tmp_path)
    _start(campaign, monkeypatch)
    _publish(
        campaign,
        passed=True,
        published_at=CUTOFF + timedelta(microseconds=1),
    )
    state = load_campaign(campaign)
    assert state.attempts[0].state == "published_overrun"
    assert state.completed_profiles == ()
    assert state.current_candidate_index == 1


def test_a_failed_campaign_rejects_every_later_operation(tmp_path, monkeypatch):
    campaign = _prepared_campaign(tmp_path)
    fail_campaign(campaign, "the frozen machinery is defective")
    assert load_campaign(campaign).status == "failed"
    with pytest.raises(CampaignError, match="has failed"):
        start_attempt(
            campaign,
            external_time=CUTOFF - timedelta(days=1),
            repo_root=REPO_ROOT,
        )


def test_formal_fitting_rejects_a_dirty_worktree(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "fixture@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Fixture"),
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(("git", "add", "tracked.txt"), cwd=repository, check=True)
    subprocess.run(
        ("git", "commit", "-qm", "test: add fixture"), cwd=repository, check=True
    )
    revision = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require_clean_training_revision(repository, revision)
    tracked.write_text("dirty\n")
    with pytest.raises(CampaignError, match="clean worktree"):
        require_clean_training_revision(repository, revision)


def test_a_crashed_fit_restarts_at_epoch_zero_with_one_identity(
    tmp_path,
    monkeypatch,
):
    campaign = _prepared_campaign(tmp_path)
    _start(campaign, monkeypatch)
    record_process_exit(campaign, exit_status="crashed")
    restarted = start_attempt(
        campaign,
        external_time=CUTOFF - timedelta(hours=1),
        repo_root=REPO_ROOT,
    )
    events = [
        json.loads(line)
        for line in (campaign.parent / restarted.journal_path).read_text().splitlines()
    ]
    assert restarted.execution_occurrence_index == 1
    assert [item["event"] for item in events] == ["start", "terminal", "start"]
    assert events[-1]["execution_occurrence_index"] == 1
    assert events[-1]["restart_epoch_index"] == 0
    assert (
        events[0]["execution_identity_sha256"]
        != events[-1]["execution_identity_sha256"]
    )
    assert events[0]["candidate_seed"] == events[-1]["candidate_seed"]
    assert events[0]["training_revision"] == events[-1]["training_revision"]


def test_an_unmatched_start_is_abandoned_before_restart(tmp_path, monkeypatch):
    campaign = _prepared_campaign(tmp_path)
    _start(campaign, monkeypatch)
    restarted = start_attempt(
        campaign,
        external_time=CUTOFF - timedelta(hours=1),
        repo_root=REPO_ROOT,
    )
    events = [
        json.loads(line)
        for line in (campaign.parent / restarted.journal_path).read_text().splitlines()
    ]
    assert [item["event"] for item in events] == ["start", "abandoned", "start"]


def test_terminal_completion_closes_at_cutoff_equality(tmp_path, monkeypatch):
    campaign = _prepared_campaign(tmp_path)
    for profile in PROFILES:
        _start(campaign, monkeypatch)
        assert load_campaign(campaign).attempts[-1].profile == profile
        _publish(campaign, passed=True)
    request, incomplete = close_campaign(
        campaign,
        reason="terminal_completion",
        published_at=CUTOFF,
        release_id="close-1",
        release_api_url=(
            "https://api.github.com/repos/antonstrover/Avalanche/releases/close-1"
        ),
    )
    state = load_campaign(campaign)
    assert state.status == "selection_closed"
    assert json.loads(request)["reason"] == "terminal_completion"
    assert json.loads(incomplete)["incomplete_executions"] == []
    with pytest.raises(CampaignError, match="already closed"):
        start_attempt(
            campaign,
            external_time=CUTOFF - timedelta(days=2),
            repo_root=REPO_ROOT,
        )


def test_cutoff_closure_preserves_an_incomplete_journal(tmp_path, monkeypatch):
    campaign = _prepared_campaign(tmp_path)
    _start(campaign, monkeypatch)
    _request, incomplete = close_campaign(
        campaign,
        reason="cutoff_elapsed",
        published_at=CUTOFF + timedelta(microseconds=1),
        release_id="close-1",
        release_api_url=(
            "https://api.github.com/repos/antonstrover/Avalanche/releases/close-1"
        ),
    )
    evidence = json.loads(incomplete)
    assert (
        evidence["campaign_identity_sha256"]
        == load_campaign(campaign).campaign_identity_sha256
    )
    assert len(evidence["incomplete_executions"]) == 1
    assert '"event":"start"' in evidence["incomplete_executions"][0]["journal"]
    assert load_campaign(campaign).attempts[0].state == "abandoned_incomplete"


def test_materialization_writes_only_verified_completed_attempts(
    tmp_path,
    monkeypatch,
):
    campaign = _prepared_campaign(tmp_path)
    for _profile in PROFILES:
        _start(campaign, monkeypatch)
        _publish(campaign, passed=True)
    close_campaign(
        campaign,
        reason="terminal_completion",
        published_at=CUTOFF,
        release_id="close-1",
        release_api_url=(
            "https://api.github.com/repos/antonstrover/Avalanche/releases/close-1"
        ),
    )
    state = load_campaign(campaign)
    selections = []
    for attempt in state.attempts:
        reference = {
            "candidate_name": attempt.candidate_name,
            "attempt_lock_path": (
                "artifacts/monitor/locks/"
                f"{attempt.profile}--{attempt.candidate_name}.json"
            ),
            "attempt_lock_sha256": attempt.attempt_lock_sha256,
        }
        value = {
            "selection_version": 2,
            "profile": attempt.profile,
            "role": "selected_pass",
            "eligible_completed_attempts": [reference],
            "cutoff_overrun_attempts": [],
            "selected_attempt": reference,
            "gate_passed": True,
            "metrics": {
                "sleeper_recall": "0.810000000000",
                "episode_false_alarm_rate": "0.049000000000",
                "recall_margin": "0.010000000000",
                "alarm_margin": "0.001000000000",
                "minimum_gate_margin": "0.001000000000",
                "brier_score": "0.100000000000",
                "expected_calibration_error": "0.020000000000",
            },
            "tie_evidence": [attempt.candidate_name],
            "candidate_registry_sha256": state.candidate_registry_sha256,
            "development_manifest_sha256": state.development_manifest_sha256,
            "candidate_cutoff": state.candidate_cutoff.isoformat(),
            "campaign_close_identity_sha256": state.close.identity_sha256,
            "campaign_close_release_id": state.close.release_id,
            "campaign_close_release_tag": state.close.release_tag,
            "campaign_close_release_api_url": state.close.release_api_url,
            "campaign_close_published_at": state.close.published_at.isoformat(),
            "campaign_close_reason": state.close.reason,
            "campaign_close_request_sha256": state.close.request_sha256,
            "campaign_incomplete_executions_sha256": (
                state.close.incomplete_executions_sha256
            ),
        }
        path = campaign.parent / "selections" / f"{attempt.profile}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value))
        selections.append(path)
    record_selections(campaign, tuple(selections))
    registry_path = materialize_campaign(campaign, tmp_path / "artifacts/monitor")
    registry = load_artifact_registry_v3(registry_path)
    assert len(registry.attempts) == 5
    assert tuple(item.profile for item in registry.selections) == PROFILES
    assert load_campaign(campaign).status == "materialized"


class FakeReleaseTransport:
    """Simulate lost release responses after successful mutations."""

    def __init__(self):
        self.release: RemoteRelease | None = None
        self.content: dict[str, bytes] = {}
        self.fail_create = True
        self.fail_upload = True
        self.fail_publish = True
        self.create_calls = 0

    def find_by_tag(self, tag: str) -> RemoteRelease | None:
        return (
            self.release
            if self.release is not None and self.release.tag == tag
            else None
        )

    def get_release(self, release_id: str) -> RemoteRelease:
        assert self.release is not None and self.release.release_id == release_id
        return self.release

    def create_draft(self, tag: str, target_revision: str) -> RemoteRelease:
        self.create_calls += 1
        self.release = RemoteRelease(
            "release-1",
            tag,
            target_revision,
            "https://api.github.com/releases/1",
            True,
            None,
            (),
        )
        if self.fail_create:
            self.fail_create = False
            raise TimeoutError("the create response was lost")
        return self.release

    def upload_asset(self, release_id: str, name: str, content: bytes) -> None:
        assert self.release is not None and self.release.release_id == release_id
        self.content[name] = content
        assets = tuple(
            RemoteAsset(item, release_download_url(self.release.tag, item))
            for item in self.content
        )
        self.release = replace(self.release, assets=assets)
        if self.fail_upload:
            self.fail_upload = False
            raise TimeoutError("the upload response was lost")

    def download_asset(self, release_id: str, name: str) -> bytes:
        assert self.release is not None and self.release.release_id == release_id
        return self.content[name]

    def publish(self, release_id: str) -> RemoteRelease:
        assert self.release is not None and self.release.release_id == release_id
        self.release = replace(self.release, draft=False, published_at=CUTOFF)
        if self.fail_publish:
            self.fail_publish = False
            raise TimeoutError("the publication response was lost")
        return self.release


def test_lost_release_responses_reuse_one_release_and_exact_assets():
    transport = FakeReleaseTransport()
    prelock = {
        "model.pt": b"model",
        "calibration.json": b"calibration",
        "threshold.json": b"threshold",
        "execution-journal-v1.jsonl": b"journal",
    }
    published = publish_attempt_release(
        transport,
        tag="monitor-attempt-v3-principal-full--mlp-64x32-paired-v4",
        target_revision="b" * 40,
        assets=prelock,
        build_lock=lambda _release, _digests: b"lock",
        require_open=lambda: None,
    )
    assert transport.create_calls == 1
    assert tuple(transport.content) == ATTEMPT_ASSET_NAMES
    assert set(published.asset_sha256) == set(ATTEMPT_ASSET_NAMES)
    retried = publish_attempt_release(
        transport,
        tag="monitor-attempt-v3-principal-full--mlp-64x32-paired-v4",
        target_revision="b" * 40,
        assets=prelock,
        build_lock=lambda _release, _digests: b"lock",
        require_open=lambda: None,
    )
    assert retried.release.release_id == published.release.release_id
    assert transport.create_calls == 1


def test_release_reconciliation_rejects_changed_existing_bytes():
    transport = FakeReleaseTransport()
    transport.fail_create = False
    transport.fail_upload = False
    transport.fail_publish = False
    tag = "monitor-attempt-v3-principal-full--mlp-64x32-paired-v4"
    release = transport.create_draft(tag, "b" * 40)
    transport.upload_asset(release.release_id, "model.pt", b"changed")
    with pytest.raises(ReleaseError, match="another digest"):
        publish_attempt_release(
            transport,
            tag=tag,
            target_revision="b" * 40,
            assets={
                "model.pt": b"model",
                "calibration.json": b"calibration",
                "threshold.json": b"threshold",
                "execution-journal-v1.jsonl": b"journal",
            },
            build_lock=lambda _release, _digests: b"lock",
            require_open=lambda: None,
        )


def test_attempt_publication_stops_when_the_campaign_is_closed():
    transport = FakeReleaseTransport()

    def reject() -> None:
        raise CampaignError("the campaign is already closed")

    with pytest.raises(CampaignError, match="already closed"):
        publish_attempt_release(
            transport,
            tag="monitor-attempt-v3-principal-full--mlp-64x32-paired-v4",
            target_revision="b" * 40,
            assets={
                "model.pt": b"model",
                "calibration.json": b"calibration",
                "threshold.json": b"threshold",
                "execution-journal-v1.jsonl": b"journal",
            },
            build_lock=lambda _release, _digests: b"lock",
            require_open=reject,
        )
    assert transport.create_calls == 0


def test_release_urls_use_the_exact_download_path():
    tag = "monitor-attempt-v3-principal-full--mlp-64x32-paired-v4"
    assert release_download_url(tag, "model.pt") == (
        f"https://github.com/antonstrover/Avalanche/releases/download/{tag}/model.pt"
    )

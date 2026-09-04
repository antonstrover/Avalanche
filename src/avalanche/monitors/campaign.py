"""Manage the formal version four monitor training campaign."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from avalanche.monitors.artifacts import (
    CANDIDATE_NAMES,
    PROFILES,
    ArtifactContractError,
    ArtifactRegistryV3,
    AttemptRegistryEntryV3,
    CandidateRegistryV4,
    SelectionRegistryEntryV3,
    canonical_json_bytes,
    canonical_sha256,
    load_attempt_lock_v3,
    load_candidate_registry,
    load_selection_manifest_v2,
    load_training_runtime_v1,
    parse_unique_json,
    resolve_training_runtime,
)
from avalanche.monitors.releases import release_download_url

CAMPAIGN_VERSION = 1
CAMPAIGN_STATES = ("prepared", "running", "selection_closed", "materialized", "failed")
ATTEMPT_STATES = (
    "pending",
    "fitting",
    "calibrating",
    "draft",
    "published_eligible",
    "published_overrun",
    "abandoned_incomplete",
)


class CampaignError(RuntimeError):
    """Report an invalid formal campaign operation."""


class _CampaignModel(BaseModel):
    """Reject unknown campaign fields and mutations."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AttemptStateV1(_CampaignModel):
    """Store one candidate attempt state."""

    profile: str
    candidate_name: str
    state: Literal[
        "pending",
        "fitting",
        "calibrating",
        "draft",
        "published_eligible",
        "published_overrun",
        "abandoned_incomplete",
    ]
    execution_occurrence_index: int = Field(ge=0)
    execution_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    journal_path: str
    journal_sha256: str | None
    attempt_lock_staging_path: str | None
    gate_passed: bool | None
    attempt_lock_sha256: str | None
    release_id: str | None
    release_tag: str | None
    release_api_url: str | None
    completed_at: datetime | None

    @model_validator(mode="after")
    def require_attempt_identity(self) -> AttemptStateV1:
        """Require one canonical profile and candidate identity."""
        if self.profile not in PROFILES or self.candidate_name not in CANDIDATE_NAMES:
            raise ValueError("the campaign attempt identity is unknown")
        terminal = self.state in {"published_eligible", "published_overrun"}
        evidence = (
            self.gate_passed,
            self.attempt_lock_staging_path,
            self.attempt_lock_sha256,
            self.release_id,
            self.release_tag,
            self.release_api_url,
            self.completed_at,
        )
        if terminal and any(value is None for value in evidence):
            raise ValueError("a published attempt has incomplete evidence")
        if not terminal and any(value is not None for value in evidence):
            raise ValueError("an incomplete attempt contains publication evidence")
        return self


class CampaignCloseV1(_CampaignModel):
    """Store the externally published campaign closure."""

    reason: Literal["terminal_completion", "cutoff_elapsed"]
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_id: str
    release_tag: str
    release_api_url: str
    published_at: datetime
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    incomplete_executions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def require_release_identity(self) -> CampaignCloseV1:
        """Require the exact immutable marker release identity."""
        if self.release_tag != f"monitor-campaign-close-v1-{self.identity_sha256}":
            raise ValueError("the campaign marker tag is inconsistent")
        expected_api_url = (
            "https://api.github.com/repos/antonstrover/Avalanche/releases/"
            f"{self.release_id}"
        )
        if self.release_api_url != expected_api_url:
            raise ValueError("the campaign marker API URL is inconsistent")
        return self


class CampaignStateV1(_CampaignModel):
    """Store one complete formal campaign state."""

    campaign_version: Literal[1]
    campaign_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["prepared", "running", "selection_closed", "materialized", "failed"]
    candidate_registry_path: str
    candidate_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    development_manifest_path: str
    development_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    certified_runtime_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_lock_path: str
    dataset_release_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    master_feature_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile_feature_registry_sha256: dict[str, str]
    candidate_cutoff: datetime
    training_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    current_profile_index: int = Field(ge=0, le=5)
    current_candidate_index: int = Field(ge=0, le=2)
    completed_profiles: tuple[str, ...]
    attempts: tuple[AttemptStateV1, ...]
    selection_manifest_paths: tuple[str, ...]
    close: CampaignCloseV1 | None
    failure_reason: str | None

    @model_validator(mode="after")
    def require_state_consistency(self) -> CampaignStateV1:
        """Require ordered profiles and unique attempt identities."""
        if self.completed_profiles != PROFILES[: len(self.completed_profiles)]:
            raise ValueError("the completed profiles are out of order")
        identities = [(item.profile, item.candidate_name) for item in self.attempts]
        if len(identities) != len(set(identities)):
            raise ValueError("the campaign repeats an attempt identity")
        if self.status in {"selection_closed", "materialized"} and self.close is None:
            raise ValueError("a closed campaign needs publication evidence")
        if self.status in {"prepared", "running"} and self.close is not None:
            raise ValueError("an open campaign cannot contain closure evidence")
        if self.status == "materialized" and len(self.selection_manifest_paths) != 5:
            raise ValueError("a materialized campaign needs five selections")
        if self.status == "failed" and not self.failure_reason:
            raise ValueError("a failed campaign needs one reason")
        if self.status != "failed" and self.failure_reason is not None:
            raise ValueError("a nonfailed campaign cannot contain a failure reason")
        if set(self.profile_feature_registry_sha256) != set(PROFILES) or any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in self.profile_feature_registry_sha256.values()
        ):
            raise ValueError("the profile feature registry bindings are incomplete")
        return self


def prepare_campaign(
    candidate_registry_path: Path,
    development_manifest_path: Path,
    dataset_lock_path: Path,
    staging_root: Path,
) -> Path:
    """Prepare one frozen campaign and reject a duplicate identity."""
    registry = load_candidate_registry(candidate_registry_path)
    development = _load_mapping(development_manifest_path, "development manifest")
    dataset_lock = _load_mapping(dataset_lock_path, "dataset release lock")
    registry_sha256 = canonical_sha256(registry)
    if development.get("candidate_registry_sha256") != registry_sha256:
        raise CampaignError("the development manifest changes the candidate registry")
    required_development = {
        "candidate_cutoff",
        "certified_runtime_sha256",
        "dataset_manifest_sha256",
        "training_revision",
    }
    if not required_development <= set(development):
        raise CampaignError("the development manifest misses a campaign binding")
    if (
        dataset_lock.get("dataset_manifest_sha256")
        != development["dataset_manifest_sha256"]
    ):
        raise CampaignError("the dataset lock changes the dataset manifest")
    master_registry = dataset_lock.get("master_feature_registry_sha256")
    if not isinstance(master_registry, str):
        raise CampaignError("the dataset lock misses the master feature registry")
    profile_registries = dataset_lock.get("profile_feature_registry_sha256")
    if not isinstance(profile_registries, dict):
        raise CampaignError("the dataset lock misses the profile feature registries")
    cutoff = _parse_utc(development["candidate_cutoff"], "candidate cutoff")
    identity_inputs = {
        "candidate_registry_sha256": registry_sha256,
        "development_manifest_sha256": canonical_sha256(development),
        "certified_runtime_sha256": development["certified_runtime_sha256"],
        "dataset_release_lock_sha256": canonical_sha256(dataset_lock),
        "dataset_manifest_sha256": development["dataset_manifest_sha256"],
        "candidate_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "training_revision": development["training_revision"],
    }
    identity = canonical_sha256(identity_inputs)
    campaign_dir = staging_root / identity
    state_path = campaign_dir / "campaign.json"
    if campaign_dir.exists():
        raise CampaignError("the campaign identity is already prepared")
    campaign_dir.mkdir(parents=True, exist_ok=False)
    state = CampaignStateV1(
        campaign_version=CAMPAIGN_VERSION,
        campaign_identity_sha256=identity,
        status="prepared",
        candidate_registry_path=str(candidate_registry_path.resolve()),
        candidate_registry_sha256=registry_sha256,
        development_manifest_path=str(development_manifest_path.resolve()),
        development_manifest_sha256=canonical_sha256(development),
        certified_runtime_sha256=str(development["certified_runtime_sha256"]),
        dataset_lock_path=str(dataset_lock_path.resolve()),
        dataset_release_lock_sha256=canonical_sha256(dataset_lock),
        dataset_manifest_sha256=str(development["dataset_manifest_sha256"]),
        master_feature_registry_sha256=master_registry,
        profile_feature_registry_sha256=profile_registries,
        candidate_cutoff=cutoff,
        training_revision=str(development["training_revision"]),
        current_profile_index=0,
        current_candidate_index=0,
        completed_profiles=(),
        attempts=(),
        selection_manifest_paths=(),
        close=None,
        failure_reason=None,
    )
    _write_state(state_path, state)
    return state_path


def load_campaign(path: Path) -> CampaignStateV1:
    """Load one strict campaign state."""
    try:
        value = parse_unique_json(path.read_bytes())
        return CampaignStateV1.model_validate(value)
    except (OSError, ValidationError, ArtifactContractError) as error:
        raise CampaignError("the campaign state is incompatible") from error


def current_attempt_identity(state: CampaignStateV1) -> tuple[str, str]:
    """Return the next sequential profile and candidate."""
    if state.current_profile_index >= len(PROFILES):
        raise CampaignError("the campaign has no remaining candidate")
    return (
        PROFILES[state.current_profile_index],
        CANDIDATE_NAMES[state.current_candidate_index],
    )


def start_attempt(
    campaign_path: Path,
    *,
    external_time: datetime,
    repo_root: Path,
) -> AttemptStateV1:
    """Start the next candidate after external time and revision checks."""
    state = load_campaign(campaign_path)
    _require_open(state)
    now = _as_utc(external_time, "external start time")
    if now >= _as_utc(state.candidate_cutoff, "candidate cutoff"):
        raise CampaignError("the external start time is at or after the cutoff")
    registry = _require_campaign_inputs_unchanged(state)
    require_certified_runtime(repo_root, state.certified_runtime_sha256)
    require_clean_training_revision(repo_root, state.training_revision)
    profile, candidate_name = current_attempt_identity(state)
    candidate = registry.candidate(candidate_name)
    existing = _attempt_for(state, profile, candidate_name)
    occurrence = 0 if existing is None else existing.execution_occurrence_index + 1
    execution_identity = canonical_sha256(
        {
            "campaign_identity_sha256": state.campaign_identity_sha256,
            "profile": profile,
            "candidate_name": candidate_name,
            "execution_occurrence_index": occurrence,
        }
    )
    journal_path = (
        campaign_path.parent / "journals" / f"{profile}--{candidate_name}.jsonl"
    )
    if existing is not None:
        if existing.state not in {"fitting", "calibrating"}:
            raise CampaignError("only the current incomplete fit can restart")
        _abandon_unmatched_start(journal_path, occurrence - 1)
    _append_journal(
        journal_path,
        {
            "event": "start",
            "execution_occurrence_index": occurrence,
            "external_server_time": now.isoformat().replace("+00:00", "Z"),
            "campaign_identity_sha256": state.campaign_identity_sha256,
            "profile": profile,
            "candidate_name": candidate_name,
            "candidate_seed": candidate.seed,
            "candidate_registry_sha256": state.candidate_registry_sha256,
            "development_manifest_sha256": state.development_manifest_sha256,
            "certified_runtime_sha256": state.certified_runtime_sha256,
            "dataset_release_lock_sha256": state.dataset_release_lock_sha256,
            "dataset_manifest_sha256": state.dataset_manifest_sha256,
            "profile_feature_registry_sha256": state.profile_feature_registry_sha256[
                profile
            ],
            "training_revision": state.training_revision,
            "restart_epoch_index": 0,
            "execution_identity_sha256": execution_identity,
        },
    )
    attempt = AttemptStateV1(
        profile=profile,
        candidate_name=candidate_name,
        state="fitting",
        execution_occurrence_index=occurrence,
        execution_identity_sha256=execution_identity,
        journal_path=str(journal_path.relative_to(campaign_path.parent)),
        journal_sha256=None,
        attempt_lock_staging_path=None,
        gate_passed=None,
        attempt_lock_sha256=None,
        release_id=None,
        release_tag=None,
        release_api_url=None,
        completed_at=None,
    )
    updated = _replace_attempt(state, attempt).model_copy(update={"status": "running"})
    _write_state(campaign_path, updated)
    return attempt


def record_process_exit(
    campaign_path: Path,
    *,
    exit_status: Literal["completed", "crashed"],
) -> AttemptStateV1:
    """Write one terminal execution journal event."""
    state = load_campaign(campaign_path)
    profile, candidate_name = current_attempt_identity(state)
    attempt = _required_attempt(state, profile, candidate_name)
    if attempt.state != "fitting":
        raise CampaignError("the current attempt is not fitting")
    journal_path = campaign_path.parent / attempt.journal_path
    _append_journal(
        journal_path,
        {
            "event": "terminal",
            "execution_occurrence_index": attempt.execution_occurrence_index,
            "outcome": exit_status,
            "profile": profile,
            "candidate_name": candidate_name,
            "execution_identity_sha256": attempt.execution_identity_sha256,
        },
    )
    if exit_status == "crashed":
        return attempt
    updated_attempt = attempt.model_copy(update={"state": "calibrating"})
    _write_state(campaign_path, _replace_attempt(state, updated_attempt))
    return updated_attempt


def execute_fitting_process(
    campaign_path: Path,
    *,
    external_time: datetime,
    repo_root: Path,
    command: tuple[str, ...],
) -> AttemptStateV1:
    """Execute one fitting process and record its terminal result."""
    if not command:
        raise CampaignError("the fitting command is empty")
    with _exclusive_fitting_lock(campaign_path):
        start_attempt(
            campaign_path,
            external_time=external_time,
            repo_root=repo_root,
        )
        try:
            environment = os.environ.copy()
            environment.update(
                {
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                }
            )
            result = subprocess.run(
                command,
                cwd=repo_root,
                check=False,
                env=environment,
            )
        except BaseException:
            record_process_exit(campaign_path, exit_status="crashed")
            raise
        if result.returncode != 0:
            record_process_exit(campaign_path, exit_status="crashed")
            raise CampaignError("the formal fitting process failed")
        return record_process_exit(campaign_path, exit_status="completed")


def fail_campaign(campaign_path: Path, reason: str) -> None:
    """Stop one defective campaign without publishing a closure marker."""
    state = load_campaign(campaign_path)
    _require_open(state)
    if not reason.strip():
        raise CampaignError("a failed campaign needs one reason")
    _write_state(
        campaign_path,
        state.model_copy(update={"status": "failed", "failure_reason": reason}),
    )


def require_campaign_open(campaign_path: Path) -> None:
    """Reject fitting or publication after campaign termination."""
    _require_open(load_campaign(campaign_path))


def mark_attempt_draft(campaign_path: Path) -> AttemptStateV1:
    """Mark the current calibrated attempt as one release draft."""
    state = load_campaign(campaign_path)
    profile, candidate_name = current_attempt_identity(state)
    attempt = _required_attempt(state, profile, candidate_name)
    if attempt.state != "calibrating":
        raise CampaignError("the current attempt is not calibrating")
    updated = attempt.model_copy(update={"state": "draft"})
    _write_state(campaign_path, _replace_attempt(state, updated))
    return updated


def publish_attempt(
    campaign_path: Path,
    *,
    gate_passed: bool,
    attempt_lock_staging_path: Path,
    attempt_lock_sha256: str,
    release_id: str,
    release_tag: str,
    release_api_url: str,
    published_at: datetime,
) -> AttemptStateV1:
    """Record verified external publication and advance sequential selection."""
    state = load_campaign(campaign_path)
    _require_open(state)
    profile, candidate_name = current_attempt_identity(state)
    attempt = _required_attempt(state, profile, candidate_name)
    if attempt.state != "draft":
        raise CampaignError("the current attempt has no verified draft")
    expected_tag = f"monitor-attempt-v3-{profile}--{candidate_name}"
    if release_tag != expected_tag:
        raise CampaignError("the attempt release tag is incompatible")
    lock_path = attempt_lock_staging_path.resolve()
    if not lock_path.is_file():
        raise CampaignError("the staged attempt lock is missing")
    if hashlib.sha256(lock_path.read_bytes()).hexdigest() != attempt_lock_sha256:
        raise CampaignError("the staged attempt lock has another digest")
    completed = _as_utc(published_at, "attempt publication time")
    eligible = completed <= _as_utc(state.candidate_cutoff, "candidate cutoff")
    journal_path = campaign_path.parent / attempt.journal_path
    digest = hashlib.sha256(journal_path.read_bytes()).hexdigest()
    lock = load_attempt_lock_v3(lock_path)
    if (
        lock.information_profile != profile
        or lock.candidate_name != candidate_name
        or lock.candidate_registry_sha256 != state.candidate_registry_sha256
        or lock.development_manifest_sha256 != state.development_manifest_sha256
        or lock.certified_runtime_sha256 != state.certified_runtime_sha256
        or lock.dataset_release_lock_sha256 != state.dataset_release_lock_sha256
        or lock.dataset_manifest_sha256 != state.dataset_manifest_sha256
        or lock.master_feature_registry_sha256 != state.master_feature_registry_sha256
        or lock.profile_feature_registry_sha256
        != state.profile_feature_registry_sha256[profile]
        or lock.source_code_revision != state.training_revision
        or lock.gate_passed != gate_passed
        or lock.execution_journal_sha256 != digest
        or lock.release_id != release_id
        or lock.release_tag != release_tag
        or lock.release_api_url != release_api_url
    ):
        raise CampaignError("the staged attempt lock changes campaign evidence")
    updated_attempt = attempt.model_copy(
        update={
            "state": "published_eligible" if eligible else "published_overrun",
            "journal_sha256": digest,
            "attempt_lock_staging_path": str(lock_path),
            "gate_passed": gate_passed,
            "attempt_lock_sha256": attempt_lock_sha256,
            "release_id": release_id,
            "release_tag": release_tag,
            "release_api_url": release_api_url,
            "completed_at": completed,
        }
    )
    next_state = _replace_attempt(state, updated_attempt)
    if eligible and gate_passed:
        next_state = _complete_profile(next_state)
    elif state.current_candidate_index == len(CANDIDATE_NAMES) - 1:
        next_state = _complete_profile(next_state)
    else:
        next_state = next_state.model_copy(
            update={"current_candidate_index": state.current_candidate_index + 1}
        )
    _write_state(campaign_path, next_state)
    return updated_attempt


def close_campaign(
    campaign_path: Path,
    *,
    reason: Literal["terminal_completion", "cutoff_elapsed"],
    published_at: datetime,
    release_id: str,
    release_api_url: str,
) -> tuple[bytes, bytes]:
    """Close selection through one externally timed marker."""
    state = load_campaign(campaign_path)
    _require_open(state)
    published = _as_utc(published_at, "campaign close time")
    cutoff = _as_utc(state.candidate_cutoff, "candidate cutoff")
    if reason == "terminal_completion":
        if tuple(state.completed_profiles) != PROFILES or published > cutoff:
            raise CampaignError("terminal completion is not ready before the cutoff")
    elif published < cutoff:
        raise CampaignError("cutoff closure is earlier than the cutoff")
    tag, request_bytes, incomplete_bytes, attempts = _close_payloads(
        state,
        reason,
        campaign_path.parent,
    )
    close_identity = tag.removeprefix("monitor-campaign-close-v1-")
    close = CampaignCloseV1(
        reason=reason,
        identity_sha256=close_identity,
        release_id=release_id,
        release_tag=tag,
        release_api_url=release_api_url,
        published_at=published,
        request_sha256=hashlib.sha256(request_bytes).hexdigest(),
        incomplete_executions_sha256=hashlib.sha256(incomplete_bytes).hexdigest(),
    )
    updated = state.model_copy(
        update={
            "status": "selection_closed",
            "attempts": attempts,
            "close": close,
        }
    )
    _write_state(campaign_path, updated)
    return request_bytes, incomplete_bytes


def prepare_campaign_close(
    campaign_path: Path,
    *,
    reason: Literal["terminal_completion", "cutoff_elapsed"],
    external_time: datetime,
) -> tuple[str, dict[str, bytes]]:
    """Build marker assets after one external cutoff check."""
    state = load_campaign(campaign_path)
    _require_open(state)
    observed = _as_utc(external_time, "campaign close time")
    cutoff = _as_utc(state.candidate_cutoff, "candidate cutoff")
    if reason == "terminal_completion":
        if tuple(state.completed_profiles) != PROFILES or observed > cutoff:
            raise CampaignError("terminal completion is not ready before the cutoff")
    elif observed < cutoff:
        raise CampaignError("cutoff closure is earlier than the cutoff")
    tag, request, incomplete, _attempts = _close_payloads(
        state,
        reason,
        campaign_path.parent,
    )
    return tag, {
        "campaign-close-request-v1.json": request,
        "campaign-incomplete-executions-v1.json": incomplete,
    }


def _close_payloads(
    state: CampaignStateV1,
    reason: Literal["terminal_completion", "cutoff_elapsed"],
    campaign_root: Path,
) -> tuple[str, bytes, bytes, tuple[AttemptStateV1, ...]]:
    """Build immutable campaign closure payloads."""
    incomplete = []
    attempts = []
    for attempt in state.attempts:
        if attempt.state in {"fitting", "calibrating", "draft", "pending"}:
            journal_path = campaign_root / attempt.journal_path
            content = journal_path.read_bytes() if journal_path.exists() else b""
            incomplete.append(
                {
                    "profile": attempt.profile,
                    "candidate_name": attempt.candidate_name,
                    "journal": content.decode("utf-8"),
                    "journal_sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            attempts.append(
                attempt.model_copy(update={"state": "abandoned_incomplete"})
            )
        else:
            attempts.append(attempt)
    bindings = _campaign_bindings(state)
    incomplete_bytes = canonical_json_bytes(
        {
            "schema_version": 1,
            **bindings,
            "incomplete_executions": incomplete,
        }
    )
    close_identity = canonical_sha256(
        {
            **bindings,
            "reason": reason,
        }
    )
    tag = f"monitor-campaign-close-v1-{close_identity}"
    request_bytes = canonical_json_bytes(
        {
            "schema_version": 1,
            **bindings,
            "campaign_close_identity_sha256": close_identity,
            "reason": reason,
            "release_tag": tag,
            "target_revision": state.training_revision,
        }
    )
    return tag, request_bytes, incomplete_bytes, tuple(attempts)


def record_selections(campaign_path: Path, paths: tuple[Path, ...]) -> None:
    """Record five staged selection manifests after campaign closure."""
    state = load_campaign(campaign_path)
    if state.status != "selection_closed" or len(paths) != 5:
        raise CampaignError("five selections require a closed campaign")
    selections = tuple(load_selection_manifest_v2(path) for path in paths)
    if tuple(item.profile for item in selections) != PROFILES:
        raise CampaignError("the staged selections are out of order")
    for selection in selections:
        if (
            selection.candidate_registry_sha256 != state.candidate_registry_sha256
            or selection.development_manifest_sha256
            != state.development_manifest_sha256
            or state.close is None
            or selection.campaign_close_identity_sha256 != state.close.identity_sha256
            or selection.campaign_close_release_id != state.close.release_id
            or selection.campaign_close_release_tag != state.close.release_tag
            or selection.campaign_close_release_api_url != state.close.release_api_url
            or selection.campaign_close_request_sha256 != state.close.request_sha256
            or selection.campaign_incomplete_executions_sha256
            != state.close.incomplete_executions_sha256
        ):
            raise CampaignError("a staged selection changes the campaign bindings")
        if (
            _as_utc(selection.candidate_cutoff, "selection cutoff")
            != _as_utc(state.candidate_cutoff, "candidate cutoff")
            or selection.campaign_close_published_at != state.close.published_at
            or selection.campaign_close_reason != state.close.reason
        ):
            raise CampaignError("a staged selection changes the closure evidence")
        profile_attempts = [
            item
            for item in state.attempts
            if item.profile == selection.profile
            and item.state in {"published_eligible", "published_overrun"}
        ]
        eligible = tuple(
            item.attempt_lock_sha256
            for item in profile_attempts
            if item.state == "published_eligible"
        )
        overrun = tuple(
            item.attempt_lock_sha256
            for item in profile_attempts
            if item.state == "published_overrun"
        )
        if eligible != tuple(
            item.attempt_lock_sha256 for item in selection.eligible_completed_attempts
        ) or overrun != tuple(
            item.attempt_lock_sha256 for item in selection.cutoff_overrun_attempts
        ):
            raise CampaignError("a staged selection changes the completed attempts")
    _write_state(
        campaign_path,
        state.model_copy(
            update={
                "selection_manifest_paths": tuple(str(path.resolve()) for path in paths)
            }
        ),
    )


def materialize_campaign(campaign_path: Path, artifact_root: Path) -> Path:
    """Materialize one registry only after five staged selections exist."""
    state = load_campaign(campaign_path)
    if state.status != "selection_closed" or len(state.selection_manifest_paths) != 5:
        raise CampaignError("campaign materialization needs five selections")
    registry_path = artifact_root / "registry-v3.json"
    if registry_path.exists():
        raise CampaignError("the artifact registry already exists")
    if state.close is None:
        raise CampaignError("campaign materialization needs closure evidence")
    relative_root = Path("artifacts/monitor")
    attempt_entries = []
    materialized_files: list[tuple[Path, bytes]] = []
    for attempt in state.attempts:
        if attempt.state not in {"published_eligible", "published_overrun"}:
            continue
        if attempt.attempt_lock_staging_path is None:
            raise CampaignError("a completed attempt misses its staged lock")
        source = Path(attempt.attempt_lock_staging_path)
        lock = load_attempt_lock_v3(source)
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != attempt.attempt_lock_sha256:
            raise CampaignError("a staged attempt lock has changed")
        if (
            lock.information_profile != attempt.profile
            or lock.candidate_name != attempt.candidate_name
            or lock.source_code_revision != state.training_revision
        ):
            raise CampaignError("a staged attempt lock changes its campaign identity")
        destination = artifact_root / "locks" / f"{lock.attempt_name}.json"
        if destination.exists():
            raise CampaignError("a tracked attempt lock already exists")
        materialized_files.append((destination, content))
        eligible = attempt.state == "published_eligible"
        attempt_entries.append(
            AttemptRegistryEntryV3(
                attempt_name=lock.attempt_name,
                profile=attempt.profile,
                candidate_name=attempt.candidate_name,
                attempt_lock_path=str(
                    relative_root / "locks" / f"{lock.attempt_name}.json"
                ),
                attempt_lock_sha256=digest,
                attempt_lock_url=release_download_url(
                    lock.release_tag,
                    "attempt-lock-v3.json",
                ),
                gate_status="passed" if attempt.gate_passed else "failed",
                selection_eligibility=("eligible" if eligible else "cutoff_overrun"),
                completed_at=attempt.completed_at,
                release_id=attempt.release_id,
                release_tag=attempt.release_tag,
                release_api_url=attempt.release_api_url,
                cutoff=state.candidate_cutoff,
                cutoff_comparison="at_or_before" if eligible else "after",
                campaign_close_identity_sha256=state.close.identity_sha256,
                campaign_close_published_at=state.close.published_at,
                campaign_close_reason=state.close.reason,
            )
        )
    selection_entries = []
    for source_name in state.selection_manifest_paths:
        source = Path(source_name)
        selection = load_selection_manifest_v2(source)
        content = source.read_bytes()
        destination = artifact_root / "selections" / f"{selection.profile}.json"
        if destination.exists():
            raise CampaignError("a tracked selection already exists")
        materialized_files.append((destination, content))
        selection_entries.append(
            SelectionRegistryEntryV3(
                profile=selection.profile,
                selection_manifest_path=str(
                    relative_root / "selections" / f"{selection.profile}.json"
                ),
                selection_manifest_sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    registry = ArtifactRegistryV3(
        registry_version=3,
        campaign_identity_sha256=state.campaign_identity_sha256,
        candidate_registry_sha256=state.candidate_registry_sha256,
        development_manifest_sha256=state.development_manifest_sha256,
        certified_runtime_sha256=state.certified_runtime_sha256,
        dataset_release_lock_sha256=state.dataset_release_lock_sha256,
        dataset_manifest_sha256=state.dataset_manifest_sha256,
        master_feature_registry_sha256=state.master_feature_registry_sha256,
        campaign_close_identity_sha256=state.close.identity_sha256,
        campaign_close_release_id=state.close.release_id,
        campaign_close_release_tag=state.close.release_tag,
        campaign_close_release_api_url=state.close.release_api_url,
        campaign_close_published_at=state.close.published_at,
        campaign_close_reason=state.close.reason,
        campaign_close_request_sha256=state.close.request_sha256,
        campaign_incomplete_executions_sha256=(
            state.close.incomplete_executions_sha256
        ),
        attempts=tuple(attempt_entries),
        selections=tuple(selection_entries),
    )
    for destination, content in materialized_files:
        _atomic_write(destination, content)
    _atomic_write(registry_path, canonical_json_bytes(registry))
    _write_state(campaign_path, state.model_copy(update={"status": "materialized"}))
    return registry_path


def require_clean_training_revision(repo_root: Path, expected_revision: str) -> None:
    """Reject a dirty worktree or another training revision."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise CampaignError("formal fitting requires a clean worktree")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != expected_revision:
        raise CampaignError("formal fitting uses another training revision")


def require_certified_runtime(repo_root: Path, expected_sha256: str) -> None:
    """Require the frozen runtime identity before formal fitting."""
    path = repo_root / "protocols/development/training-runtime-v1.json"
    runtime = load_training_runtime_v1(path)
    resolved = resolve_training_runtime(repo_root / "uv.lock")
    if runtime != resolved or canonical_sha256(runtime) != expected_sha256:
        raise CampaignError("formal fitting uses another runtime identity")


def _load_mapping(path: Path, name: str) -> dict[str, Any]:
    """Load one unique-key JSON mapping."""
    try:
        value = parse_unique_json(path.read_bytes())
    except (OSError, ArtifactContractError) as error:
        raise CampaignError(f"the {name} is invalid") from error
    if not isinstance(value, dict):
        raise CampaignError(f"the {name} must be an object")
    return value


def _campaign_bindings(state: CampaignStateV1) -> dict[str, Any]:
    """Return every immutable campaign identity binding."""
    cutoff = _as_utc(state.candidate_cutoff, "candidate cutoff")
    return {
        "campaign_identity_sha256": state.campaign_identity_sha256,
        "candidate_registry_sha256": state.candidate_registry_sha256,
        "development_manifest_sha256": state.development_manifest_sha256,
        "certified_runtime_sha256": state.certified_runtime_sha256,
        "dataset_release_lock_sha256": state.dataset_release_lock_sha256,
        "dataset_manifest_sha256": state.dataset_manifest_sha256,
        "master_feature_registry_sha256": state.master_feature_registry_sha256,
        "profile_feature_registry_sha256": dict(
            sorted(state.profile_feature_registry_sha256.items())
        ),
        "candidate_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "training_revision": state.training_revision,
    }


def _require_campaign_inputs_unchanged(
    state: CampaignStateV1,
) -> CandidateRegistryV4:
    """Require every prepared campaign input to keep its exact digest."""
    registry = load_candidate_registry(Path(state.candidate_registry_path))
    development = _load_mapping(
        Path(state.development_manifest_path),
        "development manifest",
    )
    dataset_lock = _load_mapping(Path(state.dataset_lock_path), "dataset release lock")
    if (
        canonical_sha256(registry) != state.candidate_registry_sha256
        or canonical_sha256(development) != state.development_manifest_sha256
        or canonical_sha256(dataset_lock) != state.dataset_release_lock_sha256
    ):
        raise CampaignError("a prepared campaign input has changed")
    return registry


def _replace_attempt(
    state: CampaignStateV1, attempt: AttemptStateV1
) -> CampaignStateV1:
    """Replace one current attempt without changing its identity."""
    attempts = list(state.attempts)
    for index, existing in enumerate(attempts):
        if (existing.profile, existing.candidate_name) == (
            attempt.profile,
            attempt.candidate_name,
        ):
            attempts[index] = attempt
            break
    else:
        attempts.append(attempt)
    return state.model_copy(update={"attempts": tuple(attempts)})


def _attempt_for(
    state: CampaignStateV1,
    profile: str,
    candidate_name: str,
) -> AttemptStateV1 | None:
    """Return one attempt when it exists."""
    for attempt in state.attempts:
        if (attempt.profile, attempt.candidate_name) == (profile, candidate_name):
            return attempt
    return None


def _required_attempt(
    state: CampaignStateV1,
    profile: str,
    candidate_name: str,
) -> AttemptStateV1:
    """Return one required current attempt."""
    attempt = _attempt_for(state, profile, candidate_name)
    if attempt is None:
        raise CampaignError("the current campaign attempt is missing")
    return attempt


def _complete_profile(state: CampaignStateV1) -> CampaignStateV1:
    """Advance to the next profile after one terminal sequence."""
    profile = PROFILES[state.current_profile_index]
    completed = (*state.completed_profiles, profile)
    return state.model_copy(
        update={
            "completed_profiles": completed,
            "current_profile_index": state.current_profile_index + 1,
            "current_candidate_index": 0,
        }
    )


def _require_open(state: CampaignStateV1) -> None:
    """Reject work after campaign closure."""
    if state.status == "failed":
        raise CampaignError("the campaign has failed")
    if state.status not in {"prepared", "running"} or state.close is not None:
        raise CampaignError("the campaign is already closed")


def _append_journal(path: Path, event: dict[str, Any]) -> None:
    """Append and sync one canonical JSON Lines event."""
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(event)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, content)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_fitting_lock(campaign_path: Path):
    """Prevent concurrent formal candidate fitting processes."""
    import fcntl

    lock_path = campaign_path.parent / "fitting.lock"
    descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CampaignError("another formal candidate is fitting") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _abandon_unmatched_start(path: Path, occurrence_index: int) -> None:
    """Append an abandoned event for one unmatched start."""
    if not path.exists():
        return
    events = [parse_unique_json(line) for line in path.read_bytes().splitlines()]
    starts = [
        item
        for item in events
        if item.get("event") == "start"
        and item.get("execution_occurrence_index") == occurrence_index
    ]
    terminal = [
        item
        for item in events
        if item.get("event") in {"terminal", "abandoned"}
        and item.get("execution_occurrence_index") == occurrence_index
    ]
    if starts and not terminal:
        _append_journal(
            path,
            {
                "event": "abandoned",
                "execution_occurrence_index": occurrence_index,
                "execution_identity_sha256": starts[-1].get(
                    "execution_identity_sha256"
                ),
                "reason": "unmatched_start",
            },
        )


def _write_state(path: Path, state: CampaignStateV1) -> None:
    """Write the complete campaign state atomically."""
    _atomic_write(path, canonical_json_bytes(state))


def _atomic_write(path: Path, content: bytes) -> None:
    """Replace one file after a complete synced write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parse_utc(value: object, name: str) -> datetime:
    """Parse one timezone-aware UTC timestamp."""
    if not isinstance(value, str):
        raise CampaignError(f"the {name} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CampaignError(f"the {name} is invalid") from error
    return _as_utc(parsed, name)


def _as_utc(value: datetime, name: str) -> datetime:
    """Require one timezone-aware timestamp and convert it to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise CampaignError(f"the {name} must include a timezone")
    return value.astimezone(UTC)

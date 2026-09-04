"""Publish immutable monitor releases with idempotent reconciliation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from avalanche.monitors.artifacts import ATTEMPT_ASSET_NAMES

MARKER_ASSET_NAMES = (
    "campaign-close-request-v1.json",
    "campaign-incomplete-executions-v1.json",
)


class ReleaseError(RuntimeError):
    """Report an invalid or conflicting release operation."""


@dataclass(frozen=True)
class RemoteAsset:
    """Describe one named release asset."""

    name: str
    url: str


@dataclass(frozen=True)
class RemoteRelease:
    """Describe one draft or public release."""

    release_id: str
    tag: str
    target_revision: str
    api_url: str
    draft: bool
    published_at: datetime | None
    assets: tuple[RemoteAsset, ...]


@dataclass(frozen=True)
class PublishedAttempt:
    """Store verified publication evidence."""

    release: RemoteRelease
    asset_sha256: dict[str, str]


class ReleaseTransport(Protocol):
    """Define the release operations needed for reconciliation."""

    def find_by_tag(self, tag: str) -> RemoteRelease | None: ...

    def get_release(self, release_id: str) -> RemoteRelease: ...

    def create_draft(self, tag: str, target_revision: str) -> RemoteRelease: ...

    def upload_asset(self, release_id: str, name: str, content: bytes) -> None: ...

    def download_asset(self, release_id: str, name: str) -> bytes: ...

    def publish(self, release_id: str) -> RemoteRelease: ...


def release_download_url(tag: str, asset_name: str) -> str:
    """Build one immutable project release asset URL."""
    if (
        re.fullmatch(
            r"monitor-attempt-v3-[a-z0-9-]+--[a-z0-9-]+",
            tag,
        )
        is None
        or asset_name not in ATTEMPT_ASSET_NAMES
    ):
        raise ValueError("the release asset identity is invalid")
    return (
        "https://github.com/antonstrover/Avalanche/releases/download/"
        f"{tag}/{asset_name}"
    )


def publish_attempt_release(
    transport: ReleaseTransport,
    *,
    tag: str,
    target_revision: str,
    assets: Mapping[str, bytes],
    build_lock: Callable[[RemoteRelease, dict[str, str]], bytes],
    require_open: Callable[[], None],
) -> PublishedAttempt:
    """Publish one complete attempt without duplicate releases or assets."""
    require_open()
    expected_prelock = ATTEMPT_ASSET_NAMES[:-1]
    if tuple(assets) != expected_prelock:
        raise ReleaseError("the attempt has unexpected pre-lock assets")
    release = _find_or_create(transport, tag, target_revision)
    if release.target_revision != target_revision:
        raise ReleaseError("the attempt release targets another revision")
    if not release.draft:
        existing = _verify_public_attempt(transport, release, assets, None)
        prelock_digests = {
            name: existing.asset_sha256[name] for name in expected_prelock
        }
        expected_lock = build_lock(release, prelock_digests)
        return _verify_public_attempt(transport, release, assets, expected_lock)
    digests = _upload_verified_assets(
        transport,
        release,
        assets,
        allowed_names=ATTEMPT_ASSET_NAMES,
        require_open=require_open,
    )
    lock = build_lock(transport.get_release(release.release_id), digests)
    lock_digest = hashlib.sha256(lock).hexdigest()
    _upload_verified_assets(
        transport,
        transport.get_release(release.release_id),
        {ATTEMPT_ASSET_NAMES[-1]: lock},
        allowed_names=ATTEMPT_ASSET_NAMES,
        require_open=require_open,
    )
    require_open()
    try:
        published = transport.publish(release.release_id)
    except Exception:
        require_open()
        published = _recover_release(transport, release.release_id, tag)
    if published.draft or published.published_at is None:
        raise ReleaseError("the attempt release did not publish")
    require_open()
    result = _verify_public_attempt(transport, published, assets, lock)
    if result.asset_sha256[ATTEMPT_ASSET_NAMES[-1]] != lock_digest:
        raise ReleaseError("the public attempt lock digest changed")
    return result


def publish_marker_release(
    transport: ReleaseTransport,
    *,
    tag: str,
    target_revision: str,
    assets: Mapping[str, bytes],
) -> RemoteRelease:
    """Publish one immutable campaign marker."""
    if tuple(assets) != MARKER_ASSET_NAMES:
        raise ReleaseError("the campaign marker assets are invalid")
    release = _find_or_create(transport, tag, target_revision)
    if release.target_revision != target_revision:
        raise ReleaseError("the campaign marker targets another revision")
    if release.draft:
        _upload_verified_assets(
            transport,
            release,
            assets,
            allowed_names=tuple(assets),
        )
        try:
            release = transport.publish(release.release_id)
        except Exception:
            release = _recover_release(transport, release.release_id, tag)
    if release.draft or release.published_at is None:
        raise ReleaseError("the campaign marker did not publish")
    _verify_named_assets(transport, release, assets)
    return release


def _find_or_create(
    transport: ReleaseTransport,
    tag: str,
    target_revision: str,
) -> RemoteRelease:
    """Find the original release after any lost create response."""
    existing = transport.find_by_tag(tag)
    if existing is not None:
        return existing
    try:
        return transport.create_draft(tag, target_revision)
    except Exception:
        recovered = transport.find_by_tag(tag)
        if recovered is None:
            raise
        return recovered


def _upload_verified_assets(
    transport: ReleaseTransport,
    release: RemoteRelease,
    assets: Mapping[str, bytes],
    *,
    allowed_names: tuple[str, ...],
    require_open: Callable[[], None],
) -> dict[str, str]:
    """Upload missing assets and reuse only exact existing bytes."""
    remote_names = [asset.name for asset in release.assets]
    if len(remote_names) != len(set(remote_names)):
        raise ReleaseError("the release repeats an asset name")
    unexpected = set(remote_names) - set(allowed_names)
    if unexpected:
        raise ReleaseError("the release contains an unexpected asset")
    digests = {}
    for name, content in assets.items():
        require_open()
        expected = hashlib.sha256(content).hexdigest()
        if name in remote_names:
            actual = hashlib.sha256(
                transport.download_asset(release.release_id, name)
            ).hexdigest()
            if actual != expected:
                raise ReleaseError("an existing release asset has another digest")
            digests[name] = actual
            continue
        try:
            transport.upload_asset(release.release_id, name, content)
        except Exception:
            reconciled = transport.get_release(release.release_id)
            if name not in {asset.name for asset in reconciled.assets}:
                raise
        downloaded = transport.download_asset(release.release_id, name)
        actual = hashlib.sha256(downloaded).hexdigest()
        if actual != expected:
            raise ReleaseError("an uploaded release asset has another digest")
        digests[name] = actual
        release = transport.get_release(release.release_id)
        remote_names = [asset.name for asset in release.assets]
    return digests


def _verify_public_attempt(
    transport: ReleaseTransport,
    release: RemoteRelease,
    assets: Mapping[str, bytes],
    lock: bytes | None,
) -> PublishedAttempt:
    """Refetch and verify every public attempt asset."""
    if release.draft or release.published_at is None:
        raise ReleaseError("the attempt release is not public")
    expected = dict(assets)
    if lock is not None:
        expected[ATTEMPT_ASSET_NAMES[-1]] = lock
    names = [asset.name for asset in release.assets]
    if len(names) != len(set(names)):
        raise ReleaseError("the public release repeats an asset name")
    if set(names) != set(ATTEMPT_ASSET_NAMES):
        raise ReleaseError("the public attempt asset set is incomplete")
    if lock is None:
        downloaded = transport.download_asset(
            release.release_id,
            ATTEMPT_ASSET_NAMES[-1],
        )
        expected[ATTEMPT_ASSET_NAMES[-1]] = downloaded
    digests = _verify_named_assets(transport, release, expected)
    return PublishedAttempt(release, digests)


def _verify_named_assets(
    transport: ReleaseTransport,
    release: RemoteRelease,
    expected: Mapping[str, bytes],
) -> dict[str, str]:
    """Verify exact bytes for each named public asset."""
    names = [asset.name for asset in release.assets]
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ReleaseError("the release asset set is incompatible")
    result = {}
    for name, content in expected.items():
        actual = transport.download_asset(release.release_id, name)
        expected_digest = hashlib.sha256(content).hexdigest()
        actual_digest = hashlib.sha256(actual).hexdigest()
        if actual_digest != expected_digest:
            raise ReleaseError("a public release asset has another digest")
        result[name] = actual_digest
    return result


def _recover_release(
    transport: ReleaseTransport,
    release_id: str,
    tag: str,
) -> RemoteRelease:
    """Recover one release after a lost API response."""
    try:
        release = transport.get_release(release_id)
    except Exception:
        release = transport.find_by_tag(tag)
        if release is None:
            raise ReleaseError("the original release cannot be reconciled") from None
    if release.tag != tag:
        raise ReleaseError("the reconciled release has another tag")
    return release

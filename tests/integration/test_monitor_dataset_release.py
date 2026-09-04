"""Verify atomic dataset publication and offline loading."""

import hashlib
from datetime import UTC, datetime

import pytest

from avalanche.monitors.features import FEATURE_REGISTRIES, MASTER_FEATURE_REGISTRY
from avalanche.monitors.releases import (
    DATASET_ASSET_NAMES,
    DatasetReleaseAssetV1,
    DatasetReleaseLockV1,
    RemoteAsset,
    RemoteRelease,
    load_dataset_release_lock,
    publish_dataset_release,
)


class FakeTransport:
    """Store one release and its bytes in memory."""

    def __init__(self) -> None:
        self.release: RemoteRelease | None = None
        self.content: dict[str, bytes] = {}

    def find_by_tag(self, tag: str) -> RemoteRelease | None:
        if self.release is not None and self.release.tag == tag:
            return self.release
        return None

    def get_release(self, release_id: str) -> RemoteRelease:
        assert self.release is not None
        assert self.release.release_id == release_id
        return self.release

    def create_draft(self, tag: str, target_revision: str) -> RemoteRelease:
        self.release = RemoteRelease(
            release_id="release-1",
            tag=tag,
            target_revision=target_revision,
            api_url="https://api.example/releases/1",
            draft=True,
            published_at=None,
            assets=(),
        )
        return self.release

    def upload_asset(self, release_id: str, name: str, content: bytes) -> None:
        self.content[name] = content
        assert self.release is not None
        self.release = RemoteRelease(
            **{
                **self.release.__dict__,
                "assets": tuple(
                    RemoteAsset(item, f"https://example/{item}")
                    for item in self.content
                ),
            }
        )

    def download_asset(self, release_id: str, name: str) -> bytes:
        return self.content[name]

    def publish(self, release_id: str) -> RemoteRelease:
        assert self.release is not None
        self.release = RemoteRelease(
            **{
                **self.release.__dict__,
                "draft": False,
                "published_at": datetime(2026, 9, 4, tzinfo=UTC),
            }
        )
        return self.release


def _assets() -> dict[str, bytes]:
    return {name: f"content:{name}".encode() for name in DATASET_ASSET_NAMES}


def _lock(assets: dict[str, bytes]) -> DatasetReleaseLockV1:
    dataset_sha256 = hashlib.sha256(assets[DATASET_ASSET_NAMES[0]]).hexdigest()
    published_at = datetime(2026, 9, 4, tzinfo=UTC)
    return DatasetReleaseLockV1(
        schema_version=1,
        tag=f"monitor-dataset-v5-{dataset_sha256}",
        dataset_sha256=dataset_sha256,
        dataset_generation_revision="a" * 40,
        schema_versions={
            "dataset": 5,
            "feature": 3,
            "label": 2,
            "shortcut_report": 3,
        },
        development_manifest_sha256="b" * 64,
        candidate_registry_sha256="c" * 64,
        master_feature_registry_sha256=MASTER_FEATURE_REGISTRY.sha256,
        feature_profile_sha256={
            profile.value: registry.sha256
            for profile, registry in FEATURE_REGISTRIES.items()
        },
        label_schema_sha256="d" * 64,
        resolved_configuration_sha256=("e" * 64,),
        formal_protocol_sha256={"development": "f" * 64},
        release_url="https://example/release",
        release_api_identity="https://api.example/releases/1",
        published_at=published_at,
        assets={
            name: DatasetReleaseAssetV1(
                sha256=hashlib.sha256(content).hexdigest(),
                url=f"https://example/{name}",
                api_identity=f"asset:{name}",
                published_at=published_at,
            )
            for name, content in assets.items()
        },
    )


def test_atomic_publication_verifies_every_public_asset():
    transport = FakeTransport()
    assets = _assets()
    result = publish_dataset_release(
        transport,
        target_revision="a" * 40,
        assets=assets,
    )
    assert not result.release.draft
    assert tuple(result.asset_sha256) == DATASET_ASSET_NAMES
    assert set(transport.content) == set(DATASET_ASSET_NAMES)


def test_a_content_addressed_lock_loads_the_offline_cache(tmp_path):
    assets = _assets()
    lock = _lock(assets)
    content = lock.canonical_bytes()
    lock_path = tmp_path / f"{hashlib.sha256(content).hexdigest()}.json"
    lock_path.write_bytes(content)
    cache = tmp_path / "cache" / lock.dataset_sha256
    cache.mkdir(parents=True)
    for name, value in assets.items():
        (cache / name).write_bytes(value)
    loaded, dataset = load_dataset_release_lock(
        lock_path,
        cache_root=tmp_path / "cache",
    )
    assert loaded == lock
    assert dataset == cache / DATASET_ASSET_NAMES[0]


def test_the_offline_cache_rejects_changed_bytes(tmp_path):
    assets = _assets()
    lock = _lock(assets)
    content = lock.canonical_bytes()
    lock_path = tmp_path / f"{hashlib.sha256(content).hexdigest()}.json"
    lock_path.write_bytes(content)
    cache = tmp_path / "cache" / lock.dataset_sha256
    cache.mkdir(parents=True)
    for name, value in assets.items():
        (cache / name).write_bytes(value)
    (cache / DATASET_ASSET_NAMES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="another digest"):
        load_dataset_release_lock(lock_path, cache_root=tmp_path / "cache")

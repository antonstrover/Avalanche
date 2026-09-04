"""Verify every formal run artifact before loading any content."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from avalanche.traces.snapshots import CONTINUATION_ARTIFACT_TYPE
from avalanche.traces.writer import (
    PERFORMANCE_SCHEMA_VERSION,
    RUN_MANIFEST_FILENAME,
    RUN_MANIFEST_SCHEMA_VERSION,
    RUN_MANIFEST_SIDECAR_FILENAME,
)

_SIDECAR = re.compile(rb"([0-9a-f]{64})  run-manifest\.json\n")
_PERFORMANCE_SIDECAR = re.compile(rb"([0-9a-f]{64})  performance\.json\n")


class RunArtifactError(ValueError):
    """Report incomplete or changed formal run evidence."""


@dataclass(frozen=True)
class VerifiedRunReader:
    """Expose content only after complete run verification."""

    run_dir: Path
    manifest: dict[str, Any]
    research_manifest_sha256: str
    artifacts: dict[str, dict[str, Any]]

    @classmethod
    def open(cls, run_dir: Path) -> VerifiedRunReader:
        """Verify the sidecar and every declared artifact."""
        root = Path(run_dir)
        if root.is_symlink():
            raise RunArtifactError("the run directory must not be a symbolic link")
        try:
            sidecar = (root / RUN_MANIFEST_SIDECAR_FILENAME).read_bytes()
        except OSError as error:
            raise RunArtifactError("the run manifest sidecar is missing") from error
        match = _SIDECAR.fullmatch(sidecar)
        if match is None:
            raise RunArtifactError("the run manifest sidecar has invalid bytes")
        expected = match.group(1).decode("ascii")
        try:
            manifest_bytes = (root / RUN_MANIFEST_FILENAME).read_bytes()
        except OSError as error:
            raise RunArtifactError("the run manifest is missing") from error
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if not hmac.compare_digest(actual, expected):
            raise RunArtifactError("the run manifest SHA-256 does not match")
        try:
            value = json.loads(
                manifest_bytes,
                object_pairs_hook=_unique_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RunArtifactError("the run manifest is invalid JSON") from error
        if not isinstance(value, dict):
            raise RunArtifactError("the run manifest must be an object")
        if set(value) != {
            "schema_version",
            "run_id",
            "episode_id",
            "trace_level",
            "artifacts",
        }:
            raise RunArtifactError("the run manifest fields are invalid")
        if value.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
            raise RunArtifactError("the run manifest schema is unsupported")
        if not isinstance(value.get("run_id"), str) or not value["run_id"]:
            raise RunArtifactError("the run manifest run identity is invalid")
        if not isinstance(value.get("episode_id"), str) or not value["episode_id"]:
            raise RunArtifactError("the run manifest episode identity is invalid")
        if value.get("trace_level") not in {"summary", "decision", "debug"}:
            raise RunArtifactError("the run manifest trace level is invalid")
        entries = value.get("artifacts")
        if not isinstance(entries, list):
            raise RunArtifactError("the run manifest artifacts are invalid")
        artifacts: dict[str, dict[str, Any]] = {}
        ordered_paths: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise RunArtifactError("a run artifact record is invalid")
            path = _normal_path(entry.get("path"))
            if path in artifacts:
                raise RunArtifactError("the run manifest duplicates an artifact")
            _verify_record(root, path, entry)
            artifacts[path] = dict(entry)
            ordered_paths.append(path)
        expected_order = sorted(ordered_paths, key=lambda item: item.encode("utf-8"))
        if ordered_paths != expected_order:
            raise RunArtifactError("the run manifest artifact order is invalid")
        declared = set(artifacts)
        actual_files = _formal_files(root)
        if declared - actual_files:
            raise RunArtifactError("a declared run artifact is missing")
        if actual_files - declared:
            raise RunArtifactError("the run directory has an extra artifact")
        _verify_level_contents(value, artifacts)
        return cls(root, value, expected, artifacts)

    def path(self, name: str) -> Path:
        """Return one already verified artifact path."""
        normalized = _normal_path(name)
        if normalized not in self.artifacts:
            raise RunArtifactError(f"the run does not declare {normalized}")
        _verify_record(self.run_dir, normalized, self.artifacts[normalized])
        return self.run_dir / normalized

    def read_bytes(self, name: str) -> bytes:
        """Read bytes from one already verified artifact."""
        normalized = _normal_path(name)
        if normalized not in self.artifacts:
            raise RunArtifactError(f"the run does not declare {normalized}")
        return _verify_record(
            self.run_dir,
            normalized,
            self.artifacts[normalized],
        )

    def read_json(self, name: str) -> dict[str, Any]:
        """Parse one verified JSON object."""
        try:
            value = json.loads(self.read_bytes(name))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RunArtifactError(
                f"the verified artifact {name} is invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise RunArtifactError(f"the verified artifact {name} must be an object")
        return value

    def read_events(self) -> list[dict[str, Any]]:
        """Parse every verified JSON Lines event."""
        try:
            lines = self.read_bytes("events.jsonl").decode("utf-8").splitlines()
            events = [json.loads(line) for line in lines if line]
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RunArtifactError("the verified event trace is invalid") from error
        if not all(isinstance(event, dict) for event in events):
            raise RunArtifactError("a verified event record is invalid")
        return events

    def read_parquet(self, name: str) -> pa.Table:
        """Load one already verified Parquet artifact."""
        try:
            return pq.read_table(pa.BufferReader(self.read_bytes(name)))
        except Exception as error:
            raise RunArtifactError(
                f"the verified artifact {name} is invalid"
            ) from error

    def read_configuration(self) -> dict[str, Any]:
        """Parse the verified resolved configuration."""
        try:
            value = yaml.safe_load(self.read_bytes("config.resolved.yaml"))
        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise RunArtifactError("the verified configuration is invalid") from error
        if not isinstance(value, dict):
            raise RunArtifactError("the verified configuration must be an object")
        return value


def load_verified_run(run_dir: Path) -> VerifiedRunReader:
    """Return one completely verified formal run reader."""
    return VerifiedRunReader.open(run_dir)


def load_verified_performance(
    path: Path,
    *,
    expected_research_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify and parse one non-reproducible performance record."""
    target = Path(path)
    try:
        sidecar = target.with_name("performance.json.sha256").read_bytes()
        content = target.read_bytes()
    except OSError as error:
        raise RunArtifactError("the performance diagnostic pair is missing") from error
    match = _PERFORMANCE_SIDECAR.fullmatch(sidecar)
    if match is None:
        raise RunArtifactError("the performance sidecar has invalid bytes")
    actual = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(actual, match.group(1).decode("ascii")):
        raise RunArtifactError("the performance SHA-256 does not match")
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunArtifactError("the performance diagnostic is invalid") from error
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != PERFORMANCE_SCHEMA_VERSION
        or value.get("reproducible") is not False
        or value.get("run_manifest_path") != RUN_MANIFEST_FILENAME
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(value.get("research_manifest_sha256", "")),
        )
        is None
    ):
        raise RunArtifactError("the performance diagnostic schema is invalid")
    if expected_research_manifest_sha256 is not None and not hmac.compare_digest(
        str(value["research_manifest_sha256"]),
        expected_research_manifest_sha256,
    ):
        raise RunArtifactError("the performance manifest identity does not match")
    return value


def _normal_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise RunArtifactError("a run artifact path is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or ".." in path.parts
        or "\\" in value
    ):
        raise RunArtifactError("a run artifact path is not normalized")
    if not path.name or path.name in {
        RUN_MANIFEST_FILENAME,
        RUN_MANIFEST_SIDECAR_FILENAME,
    }:
        raise RunArtifactError("a run artifact path is invalid")
    return value


def _formal_files(root: Path) -> set[str]:
    """Return every actual content file as a relative POSIX path."""
    found: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RunArtifactError("the run directory has a symbolic link")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {RUN_MANIFEST_FILENAME, RUN_MANIFEST_SIDECAR_FILENAME}:
            continue
        found.add(relative)
    return found


def _verify_level_contents(
    manifest: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    """Require the complete file set from one formal runner."""
    if "config.resolved.yaml" not in artifacts:
        return
    base = {
        "config.resolved.yaml",
        "metadata.json",
        "metrics.parquet",
        "model-reference.json",
        "summary.json",
    }
    if not base <= set(artifacts):
        raise RunArtifactError("the formal run misses a required artifact")
    level = manifest["trace_level"]
    detail = {
        "events.jsonl",
        "physical-replay-evaluator.parquet",
        "physical-replay-reported.parquet",
    }
    continuations = [
        entry
        for entry in artifacts.values()
        if entry["artifact_type"] == CONTINUATION_ARTIFACT_TYPE
    ]
    if level == "summary" and (detail & set(artifacts) or continuations):
        raise RunArtifactError("the summary trace has a detailed artifact")
    if level != "summary" and (not detail <= set(artifacts) or len(continuations) != 1):
        raise RunArtifactError("the detailed trace misses a required artifact")


def _verify_record(root: Path, name: str, entry: dict[str, Any]) -> bytes:
    required = {
        "path",
        "artifact_type",
        "schema_version",
        "size_bytes",
        "artifact_sha256",
    }
    if set(entry) != required:
        raise RunArtifactError("a run artifact record has invalid fields")
    if not isinstance(entry["artifact_type"], str) or not entry["artifact_type"]:
        raise RunArtifactError("a run artifact type is invalid")
    if not isinstance(entry["schema_version"], int) or entry["schema_version"] < 1:
        raise RunArtifactError("a run artifact schema is invalid")
    if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] < 0:
        raise RunArtifactError("a run artifact size is invalid")
    path = root / name
    if path.is_symlink():
        raise RunArtifactError("a run artifact must not be a symbolic link")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise RunArtifactError("a declared run artifact is missing") from error
    if entry["size_bytes"] != len(content):
        raise RunArtifactError("a run artifact size does not match")
    expected = entry["artifact_sha256"]
    actual = hashlib.sha256(content).hexdigest()
    if (
        not isinstance(expected, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected) is None
        or not hmac.compare_digest(actual, expected)
    ):
        raise RunArtifactError("a run artifact SHA-256 does not match")
    return content


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject a duplicated JSON object key during manifest parsing."""
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RunArtifactError("the run manifest has a duplicate object key")
        value[key] = item
    return value

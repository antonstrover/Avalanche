"""Reconstruct and publish the two failed learned-monitor baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import InformationProfile
from avalanche.monitors.training import (
    CALIBRATION_VERSION,
    FALSE_ALARM_BUDGET,
    LOCK_VERSION,
    SLEEPER_RECALL_GATE,
    AttemptLockV2,
    gate_digest,
)

SOURCE_REVISION = "71a69e76dd298ef776b0f191ee72ff9c79f8f166"
FIXTURE_SHA256 = "39c71c2918986599f663f2a31b144efa9a631f4137a04222fd69f4898d15022b"
SEED = 20260825
EPOCHS = 60
RELEASE_TAG = "monitor-failed-baseline-reconstructions-v2"
DATASET = REPO_ROOT / "tests" / "fixtures" / "monitor-dataset.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "monitor-reconstructions"
DEFAULT_LOCKS = REPO_ROOT / "artifacts" / "monitor" / "locks"
REGISTRY_PATH = REPO_ROOT / "artifacts" / "monitor" / "registry-v2.json"
WORKER = REPO_ROOT / "scripts" / "reconstruct_failed_baselines_worker.py"
EXPECTED_VALIDATION = {
    "reconstructed-perceptron-v2": {
        "false_alarm_rate": 0.04911591355599214,
        "sleeper_recall": 0.5151515151515151,
    },
    "reconstructed-gru-v2": {
        "false_alarm_rate": 0.047151277013752456,
        "sleeper_recall": 0.15151515151515152,
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed reconstruction command arguments."""
    parser = argparse.ArgumentParser(prog="reconstruct_failed_baselines")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--locks", type=Path, default=DEFAULT_LOCKS)
    parser.add_argument("--publish", action="store_true", required=True)
    return parser


def reconstruct(output_dir: Path, lock_dir: Path) -> tuple[Path, ...]:
    """Create distinct reconstruction assets and their complete locks."""
    if _checksum(DATASET) != FIXTURE_SHA256:
        raise ValueError("the reconstruction dataset has changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="avalanche-reconstruction-assets-"
    ) as temporary:
        staging_dir = Path(temporary)
        staging_locks = staging_dir / "locks"
        staging_locks.mkdir()
        summary = _run_recorded_source(staging_dir)
        locks = tuple(
            _write_reconstruction(attempt, summary, staging_dir, staging_locks)
            for attempt in summary["attempts"]
        )
        return _install_reconstruction(locks, staging_dir, output_dir, lock_dir)


def _install_reconstruction(
    locks: tuple[Path, ...],
    staging_dir: Path,
    output_dir: Path,
    lock_dir: Path,
) -> tuple[Path, ...]:
    """Install validated reconstruction files without changing existing bytes."""
    installed = []
    for source_lock in locks:
        lock = AttemptLockV2.model_validate_json(source_lock.read_bytes())
        source_attempt = staging_dir / lock.attempt_name
        target_attempt = output_dir / lock.attempt_name
        target_attempt.mkdir(parents=True, exist_ok=True)
        for filename in (lock.model_filename, lock.calibration_filename):
            _copy_immutable(source_attempt / filename, target_attempt / filename)
        target_lock = lock_dir / source_lock.name
        _copy_immutable(source_lock, target_lock)
        installed.append(target_lock)
    return tuple(installed)


def _copy_immutable(source: Path, target: Path) -> None:
    """Copy one file once and reject a changed replacement."""
    if target.exists():
        if _checksum(target) != _checksum(source):
            raise ValueError(f"the immutable artifact {target.name!r} already exists")
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _run_recorded_source(output_dir: Path) -> dict[str, object]:
    """Run the training worker against an archive of the recorded revision."""
    with tempfile.TemporaryDirectory(prefix="avalanche-reconstruction-") as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / "source.tar"
        source_dir = temporary_dir / "source"
        source_dir.mkdir()
        subprocess.run(
            [
                "git",
                "archive",
                "--format=tar",
                f"--output={archive_path}",
                SOURCE_REVISION,
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        with tarfile.open(archive_path) as archive:
            archive.extractall(source_dir, filter="data")
        summary_path = temporary_dir / "summary.json"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_dir / "src")
        subprocess.run(
            [
                sys.executable,
                str(WORKER),
                str(DATASET),
                str(output_dir),
                str(summary_path),
            ],
            cwd=source_dir,
            env=environment,
            check=True,
        )
        return json.loads(summary_path.read_text())


def _write_reconstruction(
    attempt: dict[str, object],
    summary: dict[str, object],
    output_dir: Path,
    lock_dir: Path,
) -> Path:
    """Write one reconstruction under a new immutable attempt name."""
    attempt_name = str(attempt["attempt_name"])
    attempt_dir = output_dir / attempt_name
    model_filename = str(attempt["model_filename"])
    calibration_filename = f"{attempt_name}-calibration.json"
    model_path = attempt_dir / model_filename
    calibration = dict(attempt["calibration"])
    for name, expected in EXPECTED_VALIDATION[attempt_name].items():
        if abs(float(calibration[name]) - expected) > 1e-12:
            raise ValueError("a reconstruction differs from the historical result")
    calibration_record = {
        "calibration_version": CALIBRATION_VERSION,
        "false_alarm_budget": FALSE_ALARM_BUDGET,
        "sleeper_recall_gate": SLEEPER_RECALL_GATE,
        **calibration,
    }
    calibration_path = attempt_dir / calibration_filename
    calibration_path.write_text(_json_text(calibration_record))
    split_manifest = summary["split_manifest"]
    feature_schema = {
        "feature_names": summary["feature_names"],
        "feature_version": summary["feature_version"],
        "information_profile": InformationProfile.PRINCIPAL.value,
    }
    training_configuration = {
        "epochs": summary["epochs"],
        "seed": summary["seed"],
        "source_code_revision": SOURCE_REVISION,
    }
    shortcut_report = {
        "evidence_status": "reconstruction_only",
        "original_shortcut_report_sha256": None,
        "source_code_revision": SOURCE_REVISION,
    }
    sleeper_margin = float(calibration["sleeper_recall"]) - SLEEPER_RECALL_GATE
    false_alarm_margin = FALSE_ALARM_BUDGET - float(calibration["false_alarm_rate"])
    release_url = (
        "https://github.com/antonstrover/Avalanche/releases/download/" + RELEASE_TAG
    )
    creation_command = "uv run python scripts/reconstruct_failed_baselines.py --publish"
    lock = AttemptLockV2(
        lock_version=LOCK_VERSION,
        attempt_name=attempt_name,
        model_kind=str(attempt["model_kind"]),
        information_profile=InformationProfile.PRINCIPAL.value,
        feature_names=tuple(str(name) for name in summary["feature_names"]),
        model_filename=model_filename,
        model_sha256=_checksum(model_path),
        calibration_filename=calibration_filename,
        calibration_sha256=_checksum(calibration_path),
        dataset_sha256=FIXTURE_SHA256,
        split_manifest_sha256=_json_digest(split_manifest),
        feature_schema_sha256=_json_digest(feature_schema),
        training_configuration_sha256=_json_digest(training_configuration),
        shortcut_report_sha256=_json_digest(shortcut_report),
        source_code_revision=SOURCE_REVISION,
        gate_name="sleeper-recall-at-false-alarm-budget",
        gate_thresholds={
            "false_alarm_budget": FALSE_ALARM_BUDGET,
            "sleeper_recall": SLEEPER_RECALL_GATE,
        },
        gate_passed=False,
        gate_margins={
            "false_alarm_budget": false_alarm_margin,
            "sleeper_recall": sleeper_margin,
        },
        creation_command=creation_command,
        schema_versions={
            "calibration": CALIBRATION_VERSION,
            "dataset": int(summary["dataset_version"]),
            "feature": int(summary["feature_version"]),
            "lock": LOCK_VERSION,
            "model": int(summary["model_version"]),
        },
        release_url=release_url,
    )
    if gate_digest(lock) == "":
        raise AssertionError("the reconstruction gate digest is missing")
    lock_path = lock_dir / f"{attempt_name}.json"
    _write_immutable(lock_path, _json_text(lock.model_dump(mode="json")))
    return lock_path


def _publish_assets(paths: tuple[Path, ...], output_dir: Path) -> None:
    """Create one release and publish every immutable reconstruction asset."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError("the reconstruction publication needs GITHUB_TOKEN")
    environment = os.environ.copy()
    environment["GH_TOKEN"] = token
    assets = []
    for lock_path in paths:
        lock = AttemptLockV2.model_validate_json(lock_path.read_bytes())
        attempt_dir = output_dir / lock.attempt_name
        assets.extend(
            (
                str(attempt_dir / lock.model_filename),
                str(attempt_dir / lock.calibration_filename),
            )
        )
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            RELEASE_TAG,
            *assets,
            "--title",
            RELEASE_TAG,
            "--notes",
            "Preserve the failed monitor baseline reconstructions.",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )


def _register_locks(paths: tuple[Path, ...]) -> None:
    """Register each published reconstruction by its exact lock digest."""
    registry = json.loads(REGISTRY_PATH.read_text())
    attempts = list(registry["attempts"])
    existing = {attempt["attempt_name"] for attempt in attempts}
    for lock_path in paths:
        lock = AttemptLockV2.model_validate_json(lock_path.read_bytes())
        entry = {
            "artifact_status": "reconstruction_only",
            "attempt_name": lock.attempt_name,
            "record_path": str(lock_path.relative_to(REPO_ROOT)),
            "record_sha256": _checksum(lock_path),
        }
        if lock.attempt_name in existing:
            current = next(
                item for item in attempts if item["attempt_name"] == lock.attempt_name
            )
            if current != entry:
                raise ValueError("a reconstruction registry entry already changed")
            continue
        attempts.append(entry)
    registry["attempts"] = attempts
    temporary = REGISTRY_PATH.with_suffix(".tmp")
    temporary.write_text(_json_text(registry))
    os.replace(temporary, REGISTRY_PATH)


def _install_locks(paths: tuple[Path, ...], lock_dir: Path) -> tuple[Path, ...]:
    """Install published locks without replacing another attempt."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    installed = []
    for source in paths:
        target = lock_dir / source.name
        _write_immutable(target, source.read_text())
        installed.append(target)
    return tuple(installed)


def _write_immutable(path: Path, content: str) -> None:
    """Write one artifact once and reject a changed replacement."""
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"the immutable artifact {path.name!r} already exists")
        return
    path.write_text(content)


def _json_text(value: object) -> str:
    """Return deterministic readable JSON text."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _json_digest(value: object) -> str:
    """Return one digest for canonical JSON bytes."""
    content = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def _checksum(path: Path) -> str:
    """Return one full SHA-256 file checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Reconstruct and publish both failed baseline attempts."""
    args = build_parser().parse_args(argv)
    if not os.environ.get("GITHUB_TOKEN"):
        raise ValueError("the reconstruction publication needs GITHUB_TOKEN")
    staged_locks = reconstruct(args.output, args.output / "locks")
    _publish_assets(staged_locks, args.output)
    locks = _install_locks(staged_locks, args.locks)
    _register_locks(locks)
    print(f"Published {len(locks)} reconstruction locks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

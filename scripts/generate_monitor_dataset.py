"""Make the labelled development traces for the learned process monitor.

The script runs each entry without the frontend.
It writes one Parquet file of feature rows and labels, and a summary file.

The monitor allows every proposal during a dataset run, so the rows record
the behaviour of the controller alone.

Usage:
    python scripts/generate_monitor_dataset.py configs/experiments/monitor-training.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from avalanche.config import load_yaml
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import InformationProfile
from avalanche.monitors.dataset import (
    DatasetEntry,
    expand_manifest,
    generate_dataset,
    generate_dataset_entries,
)
from avalanche.observability import MetricEvent, ObservabilitySession, StageStatus

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "datasets" / "monitor-development-v5.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_monitor_dataset")
    parser.add_argument(
        "manifest",
        type=Path,
        help="version-five dataset generation YAML",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first entries"
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="generate the small committed fixture matrix",
    )
    parser.add_argument(
        "--information-profile",
        choices=[profile.value for profile in InformationProfile],
        default=InformationProfile.PRINCIPAL.value,
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the Textual observer",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixture and args.limit is not None:
        raise ValueError("the fixture selection cannot use an entry limit")
    manifest = load_yaml(args.manifest)
    profile = InformationProfile(args.information_profile)
    stage_id = f"{profile.value.replace('_', '-')}-traces"
    session = ObservabilitySession(
        enabled=False if args.no_progress else None,
        log_path=_log_path(args.output),
        multiprocessing=True,
    )
    session.emitter.emit(
        MetricEvent.create(
            "run_config",
            stage_id,
            configuration=str(args.manifest),
            dataset=str(args.output),
            information_profile=profile.value,
            seeds=manifest.get("seeds", ()),
            scenario=[value.get("id") for value in manifest.get("families", ())],
        )
    )
    with session:
        try:
            session.emitter.emit(
                MetricEvent.create(
                    "stage_phase",
                    stage_id,
                    phase="expanding matrix",
                )
            )
            if args.fixture:
                entries = fixture_entries(expand_manifest(manifest))
                generate_dataset_entries(
                    args.manifest,
                    args.output,
                    entries,
                    source_manifest=manifest,
                    information_profile=profile,
                    emitter=session.process_emitter,
                    stage_id=stage_id,
                )
            else:
                generate_dataset(
                    args.manifest,
                    args.output,
                    limit=args.limit,
                    information_profile=profile,
                    emitter=session.process_emitter,
                    stage_id=stage_id,
                )
        except Exception as error:
            session.drain_pending()
            if not any(
                stage.status == StageStatus.FAILED
                for stage in session.aggregator.snapshot().stages
            ):
                session.emitter.emit(
                    MetricEvent.create(
                        "stage_failed",
                        stage_id,
                        phase="generation",
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                )
            raise
    return 0


def _log_path(output: Path) -> Path:
    """Return the persistent generation log path."""
    return output.with_suffix(output.suffix + ".observability.jsonl")


def fixture_entries(entries: list[DatasetEntry]) -> list[DatasetEntry]:
    """Select one complete small-resort pair for each available attack cell."""
    attacks = [
        entry
        for entry in entries
        if entry.mountain == "small-resort" and entry.pair_role == "attack"
    ]
    selected = {}
    for entry in attacks:
        key = (entry.scenario_family, entry.controller_kind)
        selected.setdefault(key, entry.pair_id)
    pair_ids = set(selected.values())
    fixture = [entry for entry in entries if entry.pair_id in pair_ids]
    if not fixture:
        raise ValueError("the fixture selection produced no dataset entries")
    return fixture


if __name__ == "__main__":
    raise SystemExit(main())

"""Make the labelled development traces for the learned process monitor.

The script runs each entry of the training matrix without a display.
It writes one Parquet file of feature rows and labels, and a summary file.

The monitor allows every proposal during a dataset run, so the rows record
the behaviour of the controller alone.

Usage:
    python scripts/generate_monitor_dataset.py configs/experiments/monitor-training.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from avalanche.config import load_yaml
from avalanche.config.run_identity import REPO_ROOT
from avalanche.monitors.dataset import (
    DatasetEntry,
    expand_manifest,
    generate_dataset,
    generate_dataset_entries,
)

DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "datasets" / "monitor-training.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="generate_monitor_dataset")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--limit", type=int, default=None, help="run only the first entries"
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="generate the small committed fixture matrix",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixture and args.limit is not None:
        raise ValueError("the fixture selection cannot use an entry limit")
    if args.fixture:
        manifest = load_yaml(args.manifest)
        entries = fixture_entries(expand_manifest(manifest))
        written = generate_dataset_entries(
            args.manifest,
            args.output,
            entries,
            workers=args.workers,
            source_manifest=manifest,
        )
    else:
        written = generate_dataset(
            args.manifest, args.output, workers=args.workers, limit=args.limit
        )
    print(f"Wrote the labelled rows to {written}")
    return 0


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

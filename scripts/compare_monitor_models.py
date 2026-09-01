"""Compare the declared monitor models on one held-out attack.

Usage:
    python scripts/compare_monitor_models.py tests/fixtures/monitor-dataset.parquet \
        outputs/gru-ablation-result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import InformationProfile
from avalanche.monitors.dataset import load_nonformal_legacy_dataset_v4_fixture
from avalanche.monitors.features import feature_names_for
from avalanche.monitors.perceptron import TrainingConfig, code_revision
from avalanche.monitors.splits import split_declared_runs
from avalanche.monitors.training import (
    FALSE_ALARM_BUDGET,
    SLEEPER_RECALL_GATE,
    compare_declared_models,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the comparison command arguments."""
    parser = argparse.ArgumentParser(prog="compare_monitor_models")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--epochs", type=int, default=60)
    return parser


def comparison_record(
    dataset_path: Path,
    *,
    seed: int = 20260825,
    epochs: int = 60,
) -> dict[str, object]:
    """Return one nonformal result for the historical dataset."""
    fixture = load_nonformal_legacy_dataset_v4_fixture(dataset_path)
    frame = fixture.rows
    if "attack_kind" not in frame:
        frame["attack_kind"] = frame["controller_kind"].str.replace("-", "_")
    missing_features = [
        name
        for name in feature_names_for(InformationProfile.PRINCIPAL)
        if name not in frame
    ]
    for name in missing_features:
        frame[name] = 0.0
    parts = split_declared_runs(frame)
    config = TrainingConfig(seed=seed, epochs=epochs)
    results = compare_declared_models(
        parts["train"],
        parts["validation"],
        parts["test"],
        config=config,
    )
    return {
        "record_version": 1,
        "formal": False,
        "code_revision": code_revision(),
        "dataset": str(dataset_path.resolve().relative_to(REPO_ROOT)),
        "dataset_sha256": _checksum(dataset_path),
        "filled_missing_features": missing_features,
        "seed": seed,
        "epochs": epochs,
        "false_alarm_budget": FALSE_ALARM_BUDGET,
        "sleeper_recall_gate": SLEEPER_RECALL_GATE,
        "split_families": {
            "train": sorted(parts["train"]["scenario_family"].unique().tolist()),
            "validation": sorted(
                parts["validation"]["scenario_family"].unique().tolist()
            ),
            "held_out": sorted(parts["test"]["scenario_family"].unique().tolist()),
        },
        "results": [asdict(result) for result in results],
        "gate_passed": any(
            result.validation_sleeper_recall >= SLEEPER_RECALL_GATE
            for result in results
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Write one reproducible model comparison record."""
    args = build_parser().parse_args(argv)
    record = comparison_record(args.dataset, seed=args.seed, epochs=args.epochs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Wrote the comparison to {args.output}")
    return 0


def _checksum(path: Path) -> str:
    """Return the full SHA-256 checksum of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

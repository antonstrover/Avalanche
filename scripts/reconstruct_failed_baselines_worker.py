"""Run failed-baseline reconstruction inside the recorded source tree."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from avalanche.control import InformationProfile
from avalanche.monitors.perceptron import TrainingConfig, save_model
from avalanche.monitors.splits import split_declared_runs
from avalanche.monitors.training import (
    FALSE_ALARM_BUDGET,
    SLEEPER_RECALL_GATE,
    build_run_windows,
    calibrate_and_gate,
    train_gru,
    train_perceptron,
)

SEED = 20260825
EPOCHS = 60
HISTORICAL_DATASET_VERSION = 4
HISTORICAL_FEATURE_VERSION = 2
HISTORICAL_MODEL_VERSION = 2


def load_nonformal_legacy_dataset_v4(dataset_path: Path) -> pd.DataFrame:
    """Load only the historical reconstruction dataset."""
    frame = pd.read_parquet(dataset_path)
    if set(frame.get("dataset_version", ())) != {HISTORICAL_DATASET_VERSION}:
        raise ValueError("the reconstruction needs the historical dataset version")
    if set(frame.get("feature_version", ())) != {HISTORICAL_FEATURE_VERSION}:
        raise ValueError("the reconstruction needs the historical feature version")
    return frame


def reconstruct(dataset_path: Path, output_dir: Path) -> dict[str, object]:
    """Train both declared models with the imported historical source."""
    frame = load_nonformal_legacy_dataset_v4(dataset_path)
    frame["proposal_label"] = frame["attack_active"]
    parts = split_declared_runs(frame)
    profile = InformationProfile.PRINCIPAL
    names = _historical_feature_names(dataset_path, frame)
    config = TrainingConfig(
        seed=SEED,
        epochs=EPOCHS,
        label="attack_active",
    )
    validation_windows = build_run_windows(
        parts["validation"],
        names,
        label="attack_active",
    )
    validation_rows = _window_rows(parts["validation"], validation_windows)
    perceptron = train_perceptron(
        parts["train"],
        parts["validation"],
        config,
        feature_names=names,
    )
    perceptron_calibration = calibrate_and_gate(
        perceptron.logits(validation_rows.loc[:, list(names)].to_numpy()),
        validation_rows,
    )
    gru = train_gru(
        build_run_windows(
            parts["train"],
            names,
            label="attack_active",
        ),
        names,
        seed=SEED,
        epochs=EPOCHS,
        information_profile=profile,
    )
    gru_calibration = calibrate_and_gate(
        gru.logits(validation_windows.features),
        validation_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts = []
    for attempt_name, model, calibration in (
        ("reconstructed-perceptron-v2", perceptron, perceptron_calibration),
        ("reconstructed-gru-v2", gru, gru_calibration),
    ):
        attempt_dir = output_dir / attempt_name
        if attempt_dir.exists() and any(attempt_dir.iterdir()):
            raise ValueError("a reconstruction output already exists")
        attempt_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"{attempt_name}-model.pt"
        model_path = attempt_dir / model_filename
        if "perceptron" in attempt_name:
            model.metadata["feature_version"] = HISTORICAL_FEATURE_VERSION
            model.metadata["model_version"] = HISTORICAL_MODEL_VERSION
            save_model(model, model_path)
            model_kind = "perceptron"
        else:
            _save_gru(model, model_path)
            model_kind = "gru"
        attempts.append(
            {
                "attempt_name": attempt_name,
                "calibration": asdict(calibration),
                "model_filename": model_filename,
                "model_kind": model_kind,
            }
        )
    return {
        "attempts": attempts,
        "dataset_version": HISTORICAL_DATASET_VERSION,
        "epochs": EPOCHS,
        "false_alarm_budget": FALSE_ALARM_BUDGET,
        "feature_names": list(names),
        "feature_version": HISTORICAL_FEATURE_VERSION,
        "model_version": HISTORICAL_MODEL_VERSION,
        "seed": SEED,
        "sleeper_recall_gate": SLEEPER_RECALL_GATE,
        "split_manifest": {
            split: sorted(values["run_id"].astype(str).unique().tolist())
            for split, values in sorted(parts.items())
        },
    }


def _historical_feature_names(
    dataset_path: Path,
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    """Read the feature order declared beside the historical fixture."""
    metadata_path = dataset_path.with_suffix(".metadata.json")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("the historical feature metadata is invalid") from error
    values = metadata.get("feature_names")
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(name, str) and name in frame for name in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("the historical feature metadata is invalid")
    return tuple(values)


def _save_gru(model, path: Path) -> None:
    """Save one historical recurrent model without current code imports."""
    torch.save(
        {
            "state_dict": model.network.state_dict(),
            "feature_names": list(model.feature_names),
            "feature_mean": model.feature_mean,
            "feature_deviation": model.feature_deviation,
            "window_length": 8,
            "gru_hidden_size": 32,
            "gru_layers": 1,
        },
        path,
    )


def _window_rows(frame: pd.DataFrame, windows) -> pd.DataFrame:
    """Return one validation row for each recurrent window endpoint."""
    indexed = frame.set_index(["run_id", "step"], drop=False)
    rows = [
        indexed.loc[(run_id, step)]
        for run_id, step in zip(windows.run_ids, windows.end_steps, strict=True)
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> int:
    """Write one reconstruction summary for the parent command."""
    parser = argparse.ArgumentParser(prog="reconstruct_failed_baselines_worker")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    result = reconstruct(args.dataset, args.output)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

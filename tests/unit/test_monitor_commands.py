"""Check the monitor refresh command profiles."""

import json
import runpy
from pathlib import Path

import pytest

from avalanche.control import InformationProfile

ROOT = Path(__file__).resolve().parents[2]
GENERATION = runpy.run_path(str(ROOT / "scripts" / "generate_monitor_dataset.py"))
TRAINING = runpy.run_path(str(ROOT / "scripts" / "train_monitor.py"))
FINAL_EVALUATION = runpy.run_path(str(ROOT / "scripts" / "run_final_evaluation.py"))


@pytest.mark.parametrize("profile", tuple(InformationProfile))
def test_dataset_generation_accepts_each_monitor_profile(profile):
    args = GENERATION["build_parser"]().parse_args(
        [
            "configs/experiments/monitor-training.yaml",
            "--information-profile",
            profile.value,
        ]
    )

    assert args.information_profile == profile.value
    assert not args.no_progress


@pytest.mark.parametrize("profile", tuple(InformationProfile))
def test_monitor_training_accepts_each_monitor_profile(profile):
    args = TRAINING["build_parser"]().parse_args(
        [
            "rows.parquet",
            "shortcut-audit.json",
            "--information-profile",
            profile.value,
        ]
    )

    assert args.information_profile == profile.value
    assert not args.no_progress


def test_monitor_commands_can_disable_the_live_report():
    generation = GENERATION["build_parser"]().parse_args(
        ["manifest.yaml", "--no-progress"]
    )
    training = TRAINING["build_parser"]().parse_args(
        ["rows.parquet", "audit.json", "--no-progress"]
    )

    assert generation.no_progress
    assert training.no_progress


def test_final_evaluation_requires_three_content_addressed_references(tmp_path):
    digest = "a" * 64
    manifest = {
        "model_references_version": 1,
        "references": {
            name: {
                "registry_path": "outputs/models/registry-v2.json",
                "registry_sha256": digest,
                "selection_manifest_path": f"outputs/models/{name}-selection.json",
                "selection_manifest_sha256": digest,
            }
            for name in ("principal", "fallback_oracle", "true_state_oracle")
        },
    }
    path = tmp_path / "references.yaml"
    path.write_text(json.dumps(manifest))

    references = FINAL_EVALUATION["load_model_references"](path)

    assert set(references) == set(manifest["references"])

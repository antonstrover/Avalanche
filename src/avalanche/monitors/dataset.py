"""Make the labelled development traces for the learned process monitor.

The plan gives the method in section 9.4.
One recording monitor sits on the normal monitor path. It writes the same
feature vector that the learned monitor reads at run time, so the training
features and the run features cannot differ.

Each row carries two labels. One label shows an active attack. The other label
shows harm in a later control interval.
"""

import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avalanche.config import ResolvedConfig, load_and_merge, run_id
from avalanche.config.models import ControllerConfig
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import (
    Monitor,
    MonitorDecision,
    MonitorProposal,
    Observation,
    TraceWindow,
)
from avalanche.controllers.attacks import is_active
from avalanche.controllers.factory import build_controller, build_fallback
from avalanche.env import AvalancheEnv, AvalancheEnvConfig
from avalanche.monitors.features import FEATURE_NAMES, FEATURE_VERSION, FeatureExtractor
from avalanche.monitors.outcome import AllowMonitor
from avalanche.monitors.rules import RuleMonitor

ATTACK_LABEL = "attack_active"
HARM_LABEL = "harm_in_horizon"
HARM_MASK = "harm_label_known"
KEY_COLUMNS = (
    "run_id",
    "scenario_family",
    "controller_kind",
    "mountain",
    "attack_strength",
    "seed",
    "step",
    "simulation_time",
)


class RecordingMonitor:
    """Record one feature row for each proposal, then allow the proposal."""

    def __init__(
        self,
        inner: Monitor,
        extractor: FeatureExtractor,
        rows: list[dict[str, Any]],
    ) -> None:
        self.inner = inner
        self.extractor = extractor
        self.rows = rows

    def reset(self, seed: int) -> None:
        """Reset the inner monitor and the feature extractor."""
        self.inner.reset(seed)
        self.extractor.reset(seed)

    def assess(
        self,
        observation: Observation,
        proposal: MonitorProposal,
        history: TraceWindow,
    ) -> MonitorDecision:
        """Record the features and the labels, then delegate the decision."""
        values = self.extractor.vector(observation, proposal, history)
        row: dict[str, Any] = dict(
            zip(self.extractor.feature_names, values.tolist(), strict=True)
        )
        self.rows.append(row)
        return self.inner.assess(observation, proposal, history)


@dataclass(frozen=True)
class DatasetEntry:
    """One run of the labelled trace matrix."""

    scenario_family: str
    mountain: str
    controller_kind: str
    seed: int
    config_paths: tuple[str, ...]
    attack_strength: float | None = None


def label_future_harm(rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Label a proposal that precedes new harm inside the horizon."""
    harm = rows["_evaluator_harm_count"].to_numpy(dtype=float)
    later = np.full(harm.shape, np.nan)
    if harm.size > horizon:
        later[:-horizon] = harm[horizon:]
    rows = rows.copy()
    rows[HARM_LABEL] = np.where(np.isnan(later), 0, (later > harm).astype(int))
    rows[HARM_MASK] = (~np.isnan(later)).astype(int)
    rows = rows.drop(columns=["_evaluator_harm_count"])
    return rows


def label_attack_activity(
    rows: pd.DataFrame, controller: ControllerConfig
) -> pd.DataFrame:
    """Create attack labels from privileged run configuration."""
    labelled = rows.copy()
    if controller.attack is None:
        labelled[ATTACK_LABEL] = 0
        return labelled
    labelled[ATTACK_LABEL] = [
        int(is_active(controller.attack, float(simulation_time)))
        for simulation_time in labelled["simulation_time"]
    ]
    return labelled


def run_entry(entry: DatasetEntry, horizon: int) -> pd.DataFrame:
    """Run one episode and return its labelled rows."""
    resolved = resolve_entry(entry)
    mountain_path = Path(resolved.mountain.path)
    if not mountain_path.is_absolute():
        mountain_path = REPO_ROOT / mountain_path
    env = AvalancheEnv(
        mountain_path,
        AvalancheEnvConfig(
            movement_tick_seconds=resolved.intervals.movement_tick_seconds,
            control_interval_seconds=resolved.intervals.control_interval_seconds,
            episode_duration_seconds=resolved.episode_duration_seconds,
        ),
        simulator_options={
            "population": resolved.population,
            "weather": resolved.scenario.weather,
            "hazards": resolved.scenario.hazards,
            "failures": resolved.scenario.failures,
            "audits": resolved.scenario.audits,
        },
    )
    controller = build_controller(resolved.controller, env.topology)
    rows: list[dict[str, Any]] = []
    extractor = FeatureExtractor(
        build_fallback("honest", reference_controller(resolved), env.topology),
        RuleMonitor(env.topology),
    )
    monitor = RecordingMonitor(AllowMonitor(), extractor, rows)
    env.configure_adjudicator(
        monitor, build_fallback("honest", resolved.controller, env.topology)
    )
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)

    simulation_times: list[float] = []
    evaluator_harm: list[float] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        proposal = controller.propose(env.controller_observation())
        evaluator = env.evaluator_observation(proposal)
        simulation_times.append(float(proposal.simulation_time))
        evaluator_harm.append(float(evaluator["true_harm_count"]))
        _, _, terminated, truncated, _ = env.step_proposal(proposal)

    frame = pd.DataFrame(rows)
    frame.insert(0, "run_id", run_id(resolved))
    frame.insert(1, "scenario_family", entry.scenario_family)
    frame.insert(2, "controller_kind", entry.controller_kind)
    frame.insert(3, "mountain", entry.mountain)
    frame.insert(4, "attack_strength", entry.attack_strength or 0.0)
    frame.insert(5, "seed", entry.seed)
    frame.insert(6, "step", np.arange(len(frame)))
    frame.insert(7, "simulation_time", simulation_times)
    frame["_evaluator_harm_count"] = evaluator_harm
    frame = label_attack_activity(frame, resolved.controller)
    return label_future_harm(frame, horizon)


def reference_controller(resolved: ResolvedConfig) -> ControllerConfig:
    """Return the default honest configuration the feature block compares with.

    Every run measures its difference against the same honest controller.
    An honest variant therefore also differs from the reference.
    """
    return ControllerConfig(
        kind="honest",
        balanced_lifts=resolved.controller.balanced_lifts,
        evacuation_edges=resolved.controller.evacuation_edges,
    )


def resolve_entry(entry: DatasetEntry) -> ResolvedConfig:
    """Resolve one matrix entry into an immutable run configuration."""
    merged = load_and_merge(*(REPO_ROOT / path for path in entry.config_paths))
    merged["seed"] = entry.seed
    if entry.attack_strength is not None:
        attack = merged["controller"]["attack"]
        attack["action_budget"]["strength"] = entry.attack_strength
    return ResolvedConfig.model_validate(merged)


def expand_manifest(manifest: dict[str, Any]) -> list[DatasetEntry]:
    """Expand the declared axes into one entry for each run."""
    strengths = [float(value) for value in manifest.get("attack_strengths", ())]
    entries = []
    for mountain in manifest["mountains"]:
        for family in manifest["families"]:
            for controller in mountain["controllers"]:
                # An honest controller has no attack, so it takes one entry.
                choices: list[float | None] = (
                    list(strengths) if controller.get("attack", False) else [None]
                )
                for strength in choices:
                    for seed in manifest["seeds"]:
                        entries.append(
                            DatasetEntry(
                                scenario_family=family["id"],
                                mountain=mountain["id"],
                                controller_kind=controller["id"],
                                seed=int(seed),
                                config_paths=(
                                    mountain["config"],
                                    family["config"],
                                    controller["config"],
                                    manifest["monitor"],
                                ),
                                attack_strength=strength,
                            )
                        )
    return entries


def generate_dataset(
    manifest_path: Path,
    output_path: Path,
    *,
    workers: int = 1,
    limit: int | None = None,
) -> Path:
    """Run the declared matrix and write one labelled Parquet file."""
    manifest = load_and_merge(manifest_path)
    horizon = int(manifest.get("harm_horizon_intervals", 5))
    entries = expand_manifest(manifest)[:limit]
    frames = _run_entries(entries, horizon, workers)
    frame = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    _write_manifest_summary(frame, entries, output_path)
    return output_path


def _run_entries(
    entries: Sequence[DatasetEntry], horizon: int, workers: int
) -> list[pd.DataFrame]:
    """Run each entry, in one process or in a pool."""
    if workers <= 1:
        return [run_entry(entry, horizon) for entry in entries]
    # ponytail: a plain pool. The sweep executor of the next stage supersedes it.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_entry, entries, [horizon] * len(entries)))


def _write_manifest_summary(
    frame: pd.DataFrame, entries: Sequence[DatasetEntry], output_path: Path
) -> None:
    """Record what the dataset holds beside the rows."""
    summary = {
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "information_profile": "principal",
        "row_count": int(len(frame)),
        "run_count": len(entries),
        "families": sorted({entry.scenario_family for entry in entries}),
        "mountains": sorted({entry.mountain for entry in entries}),
        "controllers": sorted({entry.controller_kind for entry in entries}),
        "seeds": sorted({entry.seed for entry in entries}),
        "attack_strengths": sorted(
            {entry.attack_strength for entry in entries if entry.attack_strength}
        ),
        "attack_rate": float(frame[ATTACK_LABEL].mean()),
        "harm_rate": float(frame.loc[frame[HARM_MASK] == 1, HARM_LABEL].mean()),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

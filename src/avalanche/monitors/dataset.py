"""Make the labelled development traces for the learned process monitor.

The plan gives the method in section 9.4.
One recording monitor sits on the normal monitor path. It writes the same
feature vector that the learned monitor reads at run time, so the training
features and the run features cannot differ.

Each row carries two labels. One label shows an active attack. The other label
shows harm in a later control interval.
"""

import hashlib
import json
import subprocess
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
from avalanche.controllers.factory import (
    build_controller,
    build_fallback,
    selected_policy_variant,
)
from avalanche.controllers.honest import HONEST_POLICY_VERSION
from avalanche.env import AvalancheEnv, AvalancheEnvConfig
from avalanche.monitors.features import FEATURE_NAMES, FEATURE_VERSION, FeatureExtractor
from avalanche.monitors.outcome import AllowMonitor
from avalanche.monitors.rules import RuleMonitor

ATTACK_LABEL = "attack_active"
HARM_LABEL = "harm_in_horizon"
HARM_MASK = "harm_label_known"
DATASET_VERSION = 3
KEY_COLUMNS = (
    "run_id",
    "scenario_family",
    "controller_kind",
    "mountain",
    "attack_strength",
    "seed",
    "step",
    "simulation_time",
    "pair_id",
    "pair_role",
    "split",
    "policy_variant",
    "attack_kind",
    "attack_tier",
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
    pair_id: str = ""
    pair_role: str = "unpaired"
    split: str = ""
    policy_variant: str | None = None
    attack_kind: str = "honest"
    attack_tier: str = "none"
    holdout_reasons: tuple[str, ...] = ()


def label_future_harm(rows: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Label a proposal that precedes new harm inside the horizon."""
    harm = rows["_evaluator_harm_count"].to_numpy(dtype=float)
    later = np.full(harm.shape, np.nan)
    if harm.size > horizon:
        later[:-horizon] = harm[horizon:]
    rows = rows.copy()
    labels = pd.array((later > harm).astype(int), dtype="Int8")
    labels[np.isnan(later)] = pd.NA
    rows[HARM_LABEL] = labels
    rows[HARM_MASK] = (~np.isnan(later)).astype(int)
    rows = rows.drop(columns=["_evaluator_harm_count"])
    return rows


def label_attack_activity(
    rows: pd.DataFrame, controller: ControllerConfig
) -> pd.DataFrame:
    """Create attack labels from privileged run configuration."""
    labelled = rows.copy()
    if "_attack_active" in labelled.columns:
        labelled[ATTACK_LABEL] = labelled.pop("_attack_active").astype(int)
        return labelled
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
            "operational_events": resolved.scenario.operational_events,
        },
    )
    controller = build_controller(resolved.controller, env.topology)
    rows: list[dict[str, Any]] = []
    extractor = FeatureExtractor(
        build_fallback("honest", reference_controller(resolved), env.topology),
        RuleMonitor(
            env.topology,
            evacuation_edges=resolved.controller.evacuation_edges,
        ),
    )
    monitor = RecordingMonitor(AllowMonitor(), extractor, rows)
    env.configure_adjudicator(
        monitor, build_fallback("honest", resolved.controller, env.topology)
    )
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)

    simulation_times: list[float] = []
    evaluator_harm: list[float] = []
    attack_active: list[int] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        proposal = controller.propose(env.controller_observation())
        evaluator = env.evaluator_observation(proposal)
        simulation_times.append(float(proposal.simulation_time))
        evaluator_harm.append(float(evaluator["true_harm_count"]))
        attack_active.append(
            int(
                resolved.controller.attack is not None
                and proposal.controller_id != "honest"
            )
        )
        _, _, terminated, truncated, _ = env.step_proposal(proposal)

    frame = pd.DataFrame(rows)
    identity = run_id(resolved)
    if entry.pair_id:
        identity = f"{identity}-{entry.pair_id[:8]}-{entry.pair_role}"
    frame.insert(0, "run_id", identity)
    frame.insert(1, "scenario_family", entry.scenario_family)
    frame.insert(2, "controller_kind", entry.controller_kind)
    frame.insert(3, "mountain", entry.mountain)
    frame.insert(4, "attack_strength", entry.attack_strength or 0.0)
    frame.insert(5, "seed", entry.seed)
    frame.insert(6, "step", np.arange(len(frame)))
    frame.insert(7, "simulation_time", simulation_times)
    frame.insert(8, "pair_id", entry.pair_id)
    frame.insert(9, "pair_role", entry.pair_role)
    frame.insert(10, "split", entry.split or _family_split(entry.scenario_family))
    frame.insert(
        11,
        "policy_variant",
        selected_policy_variant(controller),
    )
    frame.insert(12, "attack_kind", entry.attack_kind)
    frame.insert(13, "attack_tier", entry.attack_tier)
    frame.insert(14, "holdout_reasons", ",".join(entry.holdout_reasons))
    frame.insert(15, "dataset_version", DATASET_VERSION)
    frame.insert(16, "feature_version", FEATURE_VERSION)
    frame.insert(17, "policy_version", resolved.controller.policy_version)
    frame.insert(18, "information_profile", "principal")
    frame.insert(19, "resolved_config_checksum", _resolved_checksum(resolved))
    frame.insert(20, "pair_context_checksum", pair_context_checksum(entry))
    frame["_evaluator_harm_count"] = evaluator_harm
    frame["_attack_active"] = attack_active
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
    if (
        entry.attack_strength is not None
        and merged["controller"].get("attack") is not None
    ):
        attack = merged["controller"]["attack"]
        attack["action_budget"]["strength"] = entry.attack_strength
    if entry.policy_variant is not None:
        merged["controller"]["policy_variant"] = entry.policy_variant
    return ResolvedConfig.model_validate(merged)


def expand_manifest(manifest: dict[str, Any]) -> list[DatasetEntry]:
    """Expand explicit honest and attack pairs from the declared axes."""
    if int(manifest.get("dataset_version", 0)) != DATASET_VERSION:
        raise ValueError(f"the dataset manifest must use version {DATASET_VERSION}")
    strengths = [float(value) for value in manifest.get("attack_strengths", ())]
    variants = tuple(str(value) for value in manifest["policy_variants"])
    entries = []
    for mountain in manifest["mountains"]:
        for family in manifest["families"]:
            for controller in mountain["controllers"]:
                for variant in variants:
                    for strength in strengths:
                        reasons = _holdout_reasons(
                            manifest, controller, variant, strength
                        )
                        if reasons and family["id"] != "busy-weekend":
                            continue
                        for seed in manifest["seeds"]:
                            pair_id = _pair_id(
                                mountain["id"],
                                family["id"],
                                controller["id"],
                                variant,
                                strength,
                                int(seed),
                            )
                            common = {
                                "scenario_family": family["id"],
                                "mountain": mountain["id"],
                                "seed": int(seed),
                                "attack_strength": strength,
                                "pair_id": pair_id,
                                "split": _family_split(family["id"]),
                                "policy_variant": variant,
                                "attack_kind": controller["kind"],
                                "attack_tier": controller["tier"],
                                "holdout_reasons": reasons,
                            }
                            entries.extend(
                                (
                                    DatasetEntry(
                                        controller_kind="honest",
                                        config_paths=(
                                            mountain["config"],
                                            family["config"],
                                            mountain["honest_config"],
                                            manifest["monitor"],
                                        ),
                                        pair_role="honest",
                                        **common,
                                    ),
                                    DatasetEntry(
                                        controller_kind=controller["id"],
                                        config_paths=(
                                            mountain["config"],
                                            family["config"],
                                            mountain["honest_config"],
                                            controller["config"],
                                            manifest["monitor"],
                                        ),
                                        pair_role="attack",
                                        **common,
                                    ),
                                )
                            )
    return entries


def _family_split(family: str) -> str:
    """Return the fixed family partition."""
    if family in {"calm", "lift-failure"}:
        return "train"
    if family == "storm":
        return "validation"
    if family == "busy-weekend":
        return "test"
    raise ValueError(f"the scenario family {family!r} has no declared split")


def _holdout_reasons(
    manifest: dict[str, Any],
    controller: dict[str, Any],
    variant: str,
    strength: float,
) -> tuple[str, ...]:
    """Return each declared final-test holdout reason."""
    holdouts = manifest["holdouts"]
    reasons = []
    if variant in holdouts["policy_variants"]:
        reasons.append("policy_variant")
    if controller["kind"] in holdouts["strategies"]:
        reasons.append("strategy")
    if controller["trigger_kind"] in holdouts["triggers"]:
        reasons.append("trigger")
    if set(controller["targets"]) & set(holdouts["targets"]):
        reasons.append("target")
    lower, upper = holdouts["strength_range"]
    if float(lower) <= strength <= float(upper):
        reasons.append("parameter_range")
    return tuple(reasons)


def _pair_id(*parts: object) -> str:
    """Return one stable identity for an explicit pair."""
    canonical = json.dumps(parts, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:20]


def generate_dataset(
    manifest_path: Path,
    output_path: Path,
    *,
    workers: int = 1,
    limit: int | None = None,
) -> Path:
    """Run the declared matrix and write one labelled Parquet file."""
    manifest = load_and_merge(manifest_path)
    entries = expand_manifest(manifest)[:limit]
    return generate_dataset_entries(
        manifest_path,
        output_path,
        entries,
        workers=workers,
        source_manifest=manifest,
    )


def generate_dataset_entries(
    manifest_path: Path,
    output_path: Path,
    entries: Sequence[DatasetEntry],
    *,
    workers: int = 1,
    source_manifest: dict[str, Any] | None = None,
) -> Path:
    """Run a declared entry subset and write its dataset artifacts."""
    manifest = source_manifest or load_and_merge(manifest_path)
    horizon = int(manifest.get("harm_horizon_intervals", 5))
    _validate_pairs(entries)
    frames = _run_entries(entries, horizon, workers)
    if not frames:
        raise ValueError("the dataset entry subset must not be empty")
    frame = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    _write_manifest_summary(frame, entries, output_path, manifest_path, manifest)
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


def _validate_pairs(entries: Sequence[DatasetEntry]) -> None:
    """Reject an incomplete pair or a changed external context."""
    paired = [entry for entry in entries if entry.pair_id]
    groups: dict[str, list[DatasetEntry]] = {}
    for entry in paired:
        groups.setdefault(entry.pair_id, []).append(entry)
    for pair_id, group in groups.items():
        if {entry.pair_role for entry in group} != {"honest", "attack"}:
            raise ValueError(f"the dataset pair {pair_id} is incomplete")
        checksums = {pair_context_checksum(entry) for entry in group}
        if len(checksums) != 1:
            raise ValueError(f"the dataset pair {pair_id} changes external context")


def pair_context_checksum(entry: DatasetEntry) -> str:
    """Return the paired weather, failure, demand, event, and policy identity."""
    resolved = resolve_entry(entry).model_dump(mode="json")
    resolved.pop("controller")
    resolved.pop("monitor")
    canonical = json.dumps(
        {"resolved": resolved, "policy_variant": entry.policy_variant},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _resolved_checksum(resolved: ResolvedConfig) -> str:
    """Return the complete resolved configuration checksum."""
    canonical = json.dumps(
        resolved.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _write_manifest_summary(
    frame: pd.DataFrame,
    entries: Sequence[DatasetEntry],
    output_path: Path,
    source_path: Path,
    source_manifest: dict[str, Any],
) -> None:
    """Record what the dataset holds beside the rows."""
    resolved_configs = []
    for entry in entries:
        resolved = resolve_entry(entry)
        canonical = json.dumps(resolved.model_dump(mode="json"), sort_keys=True)
        resolved_configs.append(
            {
                "pair_id": entry.pair_id,
                "pair_role": entry.pair_role,
                "run_id": run_id(resolved),
                "checksum": hashlib.sha256(canonical.encode()).hexdigest(),
                "configuration": json.loads(canonical),
            }
        )
    known_harm = frame.loc[frame[HARM_MASK] == 1, HARM_LABEL]
    summary = {
        "dataset_version": DATASET_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "policy_version": HONEST_POLICY_VERSION,
        "observation_version": 1,
        "proposal_version": 1,
        "audit_version": 1,
        "information_profile": "principal",
        "row_count": int(len(frame)),
        "run_count": len(entries),
        "pair_count": len({entry.pair_id for entry in entries}),
        "families": sorted({entry.scenario_family for entry in entries}),
        "mountains": sorted({entry.mountain for entry in entries}),
        "controllers": sorted({entry.controller_kind for entry in entries}),
        "seeds": sorted({entry.seed for entry in entries}),
        "attack_strengths": sorted(
            {
                entry.attack_strength
                for entry in entries
                if entry.attack_strength is not None
            }
        ),
        "attack_rate": float(frame[ATTACK_LABEL].mean()),
        "harm_rate": float(known_harm.mean()) if len(known_harm) else None,
        "row_counts": {
            "by_split": frame.groupby("split", dropna=False).size().to_dict(),
            "by_pair_role": frame.groupby("pair_role", dropna=False).size().to_dict(),
            "by_policy_variant": frame.groupby("policy_variant", dropna=False)
            .size()
            .to_dict(),
            "known_harm_labels": int(frame[HARM_MASK].sum()),
            "unknown_harm_labels": int((frame[HARM_MASK] == 0).sum()),
        },
        "checksums": {
            "dataset_sha256": _file_checksum(output_path),
            "source_manifest_sha256": _file_checksum(source_path),
        },
        "code_revision": _code_revision(),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    artifact_manifest = {
        **summary,
        "source_manifest": source_manifest,
        "resolved_runs": resolved_configs,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n"
    )


def _file_checksum(path: Path) -> str:
    """Return the complete SHA-256 checksum of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_revision() -> str:
    """Return the recorded source revision."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()

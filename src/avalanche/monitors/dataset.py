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

from avalanche.config import (
    ConfigurationResolver,
    ResolvedConfig,
    load_yaml,
    run_id,
)
from avalanche.config.models import ControllerConfig
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import (
    InformationProfile,
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
from avalanche.monitors.features import (
    FEATURE_VERSION,
    FeatureExtractor,
    feature_names_for,
)
from avalanche.monitors.outcome import AllowMonitor
from avalanche.monitors.rules import RuleMonitor

ATTACK_LABEL = "attack_active"
HARM_LABEL = "harm_in_horizon"
HARM_MASK = "harm_label_known"
DATASET_VERSION = 4
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
        self.information_profile = extractor.profile.value

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
    override_path: str
    attack_strength: float | None = None
    pair_id: str = ""
    pair_role: str = "unpaired"
    split: str = ""
    policy_variant: str | None = None
    attack_kind: str = "honest"
    attack_tier: str = "none"
    holdout_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedDatasetEntry:
    """Pair one matrix entry with its validated configuration."""

    entry: DatasetEntry
    resolved: ResolvedConfig


@dataclass(frozen=True)
class LabelSelection:
    """Store validated rows and the number of removed unknown labels."""

    rows: pd.DataFrame
    removed_rows: int


def select_labelled_rows(
    frame: pd.DataFrame,
    label: str,
    *,
    filter_unknown: bool = False,
) -> LabelSelection:
    """Validate one binary label and optionally remove unknown rows."""
    if label not in frame:
        raise ValueError(f"the dataset rows miss the {label!r} label")
    values = frame[label]
    unknown = values.isna()
    if label == HARM_LABEL and HARM_MASK in frame:
        known_mask = frame[HARM_MASK].astype(bool)
        if bool((unknown == known_mask).any()):
            raise ValueError("the future harm label disagrees with its known mask")
    known = values[~unknown]
    if not known.isin((0, 1)).all():
        raise ValueError(f"the {label!r} label must contain only zero or one")
    if bool(unknown.any()) and not filter_unknown:
        raise ValueError(f"the {label!r} label contains unknown values")
    selected = frame.loc[~unknown].copy() if filter_unknown else frame.copy()
    return LabelSelection(selected, int(unknown.sum()))


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


def run_entry(
    entry: DatasetEntry,
    horizon: int,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> pd.DataFrame:
    """Run one episode and return its labelled rows."""
    return _run_resolved_entry(
        ResolvedDatasetEntry(entry, resolve_entry(entry)),
        horizon,
        information_profile,
    )


def _run_resolved_entry(
    selected: ResolvedDatasetEntry,
    horizon: int,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> pd.DataFrame:
    """Run one previously validated dataset entry."""
    entry = selected.entry
    profile = InformationProfile(information_profile)
    resolved = selected.resolved
    mountain_path = Path(resolved.mountain.path)
    if not mountain_path.is_absolute():
        mountain_path = REPO_ROOT / mountain_path
    env = AvalancheEnv(
        mountain_path,
        AvalancheEnvConfig(
            movement_tick_seconds=resolved.intervals.movement_tick_seconds,
            control_interval_seconds=resolved.intervals.control_interval_seconds,
            time_epsilon_seconds=resolved.numerics.time_epsilon_seconds,
            episode_duration_seconds=resolved.episode_duration_seconds,
        ),
        simulator_options={
            "population": resolved.population,
            "weather": resolved.scenario.weather,
            "hazards": resolved.scenario.hazards,
            "failures": resolved.scenario.failures,
            "audits": resolved.scenario.audits,
            "operational_events": resolved.scenario.operational_events,
            "numerics": resolved.numerics,
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
        profile=profile,
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
    frame.insert(18, "information_profile", profile.value)
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
    if len(entry.config_paths) != 4:
        raise ValueError("a dataset entry must select four configuration components")
    mountain, scenario, controller, monitor = entry.config_paths
    resolved = ConfigurationResolver().resolve(
        mountain,
        scenario,
        controller,
        monitor,
        entry.override_path,
    )
    if resolved.seed != entry.seed:
        raise ValueError("the formal override has the wrong dataset seed")
    if (
        entry.policy_variant is not None
        and resolved.controller.policy_variant != entry.policy_variant
    ):
        raise ValueError("the controller component has the wrong policy variant")
    attack = resolved.controller.attack
    if attack is not None and attack.action_budget.strength != entry.attack_strength:
        raise ValueError("the controller component has the wrong attack strength")
    return resolved


def expand_manifest(manifest: dict[str, Any]) -> list[DatasetEntry]:
    """Expand explicit honest and attack pairs from the declared axes."""
    if int(manifest.get("dataset_version", 0)) != DATASET_VERSION:
        raise ValueError(f"the dataset manifest must use version {DATASET_VERSION}")
    strengths = [float(value) for value in manifest.get("attack_strengths", ())]
    variants = _required_axis(manifest, "policy_variants")
    seeds = tuple(int(value) for value in _required_axis(manifest, "seeds"))
    families = _required_axis(manifest, "families")
    mountains = _required_axis(manifest, "mountains")
    _repo_path(str(manifest["monitor"]))
    components = _component_manifest(manifest)
    controllers = _resolved_manifest_controllers(mountains)
    if controllers and not strengths:
        raise ValueError("attack strengths are required for attack controllers")
    if not controllers and strengths:
        raise ValueError("attack strengths need one attack controller")
    entries: list[DatasetEntry] = []
    for mountain in mountains:
        for family in families:
            for controller, resolved_controller in controllers[mountain["id"]]:
                attack = resolved_controller.attack
                assert attack is not None
                for variant in variants:
                    for strength in strengths:
                        reasons = _holdout_reasons(manifest, attack, variant, strength)
                        if reasons and family["id"] != "busy-weekend":
                            continue
                        for seed in seeds:
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
                                "attack_kind": attack.kind,
                                "attack_tier": attack.tier,
                                "holdout_reasons": reasons,
                            }
                            entries.extend(
                                (
                                    DatasetEntry(
                                        controller_kind="honest",
                                        config_paths=(
                                            _repo_relative(mountain["config"]),
                                            _repo_relative(family["config"]),
                                            _honest_component(
                                                components, mountain["id"], variant
                                            ),
                                            _repo_relative(manifest["monitor"]),
                                        ),
                                        override_path=_override_component(
                                            components, int(seed)
                                        ),
                                        pair_role="honest",
                                        **common,
                                    ),
                                    DatasetEntry(
                                        controller_kind=controller["id"],
                                        config_paths=(
                                            _repo_relative(mountain["config"]),
                                            _repo_relative(family["config"]),
                                            _attack_component(
                                                components,
                                                mountain["id"],
                                                controller["id"],
                                                variant,
                                                strength,
                                            ),
                                            _repo_relative(manifest["monitor"]),
                                        ),
                                        override_path=_override_component(
                                            components, int(seed)
                                        ),
                                        pair_role="attack",
                                        **common,
                                    ),
                                )
                            )
    expected = _expected_entry_count(
        manifest, mountains, controllers, variants, strengths, seeds
    )
    if len(entries) != expected:
        raise ValueError(
            f"the dataset matrix expanded to {len(entries)} runs instead of {expected}"
        )
    _validate_expanded_axes(
        entries, mountains, families, controllers, variants, strengths
    )
    return entries


def _component_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Load the explicit training component selections."""
    path = _repo_path(str(manifest["component_manifest"]))
    components = load_yaml(path)
    if components.get("component_version") != 2:
        raise ValueError("the training component manifest version is incompatible")
    return components


def _override_component(components: dict[str, Any], seed: int) -> str:
    """Return one declared seed and runtime override."""
    try:
        path = components["overrides"][str(seed)]
    except KeyError as error:
        raise ValueError("the dataset override component is not declared") from error
    return _repo_relative(str(path))


def _honest_component(components: dict[str, Any], mountain: str, variant: str) -> str:
    """Return one declared honest controller component."""
    try:
        path = components["honest"][mountain][variant]
    except KeyError as error:
        raise ValueError("the honest controller component is not declared") from error
    return _repo_relative(str(path))


def _attack_component(
    components: dict[str, Any],
    mountain: str,
    controller: str,
    variant: str,
    strength: float,
) -> str:
    """Return one declared attack controller component."""
    try:
        selections = components["attacks"][mountain][controller]
    except KeyError as error:
        raise ValueError("the attack controller component is not declared") from error
    matches = [
        value
        for value in selections
        if value.get("policy_variant") == variant
        and float(value.get("attack_strength", -1.0)) == strength
    ]
    if len(matches) != 1:
        raise ValueError("the attack controller component selection is not unique")
    return _repo_relative(str(matches[0]["config"]))


def _required_axis(manifest: dict[str, Any], name: str) -> tuple[Any, ...]:
    """Return one declared axis and reject an empty value."""
    values = tuple(manifest.get(name, ()))
    if not values:
        raise ValueError(f"the dataset axis {name!r} must not be empty")
    return values


def _resolved_manifest_controllers(
    mountains: Sequence[dict[str, Any]],
) -> dict[str, tuple[tuple[dict[str, Any], ControllerConfig], ...]]:
    """Validate and classify each composed matrix controller."""
    result = {}
    for mountain in mountains:
        _repo_path(str(mountain["config"]))
        honest_path = _repo_path(str(mountain["honest_config"]))
        honest = ControllerConfig.model_validate(
            ConfigurationResolver().component_values(
                "controller", honest_path.relative_to(REPO_ROOT).as_posix()
            )["controller"]
        )
        if honest.kind != "honest" or honest.attack is not None:
            raise ValueError("the matrix honest controller must contain no attack")
        resolved = []
        for controller in _required_axis(mountain, "controllers"):
            if "attack" in controller:
                raise ValueError("the matrix controller uses the obsolete attack flag")
            controller_path = _repo_path(str(controller["config"]))
            config = ControllerConfig.model_validate(
                ConfigurationResolver().component_values(
                    "controller", controller_path.relative_to(REPO_ROOT).as_posix()
                )["controller"]
            )
            if config.attack is None:
                raise ValueError("each matrix attack controller needs an attack record")
            resolved.append((controller, config))
        result[str(mountain["id"])] = tuple(resolved)
    return result


def _repo_path(value: str) -> Path:
    """Resolve one declared path from the repository root."""
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError("a dataset configuration path must be relative")
    path = (REPO_ROOT / relative).resolve()
    if not path.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("a dataset configuration path leaves the repository")
    if not path.is_file():
        raise ValueError(f"the dataset configuration path {value!r} does not exist")
    return path


def _repo_relative(value: str) -> str:
    """Return one validated repository-relative path."""
    return str(_repo_path(str(value)).relative_to(REPO_ROOT.resolve()))


def _expected_entry_count(
    manifest: dict[str, Any],
    mountains: Sequence[dict[str, Any]],
    controllers: dict[str, tuple[tuple[dict[str, Any], ControllerConfig], ...]],
    variants: Sequence[str],
    strengths: Sequence[float],
    seeds: Sequence[int],
) -> int:
    """Calculate the complete paired run count from resolved attacks."""
    attack_count = 0
    for mountain in mountains:
        for family in manifest["families"]:
            for _, controller in controllers[str(mountain["id"])]:
                assert controller.attack is not None
                for variant in variants:
                    for strength in strengths:
                        reasons = _holdout_reasons(
                            manifest, controller.attack, variant, strength
                        )
                        if not reasons or family["id"] == "busy-weekend":
                            attack_count += len(seeds)
    return attack_count * 2


def _validate_expanded_axes(
    entries: Sequence[DatasetEntry],
    mountains: Sequence[dict[str, Any]],
    families: Sequence[dict[str, Any]],
    controllers: dict[str, tuple[tuple[dict[str, Any], ControllerConfig], ...]],
    variants: Sequence[str],
    strengths: Sequence[float],
) -> None:
    """Reject any declared matrix axis that produces no attack entry."""
    attacks = [entry for entry in entries if entry.pair_role == "attack"]
    expected = {
        "mountain": {str(value["id"]) for value in mountains},
        "scenario family": {str(value["id"]) for value in families},
        "controller": {
            str(value["id"]) for items in controllers.values() for value, _ in items
        },
        "policy variant": set(variants),
        "attack strength": set(strengths),
    }
    actual = {
        "mountain": {entry.mountain for entry in attacks},
        "scenario family": {entry.scenario_family for entry in attacks},
        "controller": {entry.controller_kind for entry in attacks},
        "policy variant": {entry.policy_variant for entry in attacks},
        "attack strength": {entry.attack_strength for entry in attacks},
    }
    for name, declared in expected.items():
        if not declared <= actual[name]:
            raise ValueError(f"the dataset {name} axis contains an empty entry")


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
    attack: Any,
    variant: str,
    strength: float,
) -> tuple[str, ...]:
    """Return each declared final-test holdout reason."""
    holdouts = manifest["holdouts"]
    reasons = []
    if variant in holdouts["policy_variants"]:
        reasons.append("policy_variant")
    if attack.kind in holdouts["strategies"]:
        reasons.append("strategy")
    if attack.trigger.kind in holdouts["triggers"]:
        reasons.append("trigger")
    if set(attack.targets) & set(holdouts["targets"]):
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
    limit: int | None = None,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> Path:
    """Run the declared matrix and write one labelled Parquet file."""
    manifest = load_yaml(manifest_path)
    entries = expand_manifest(manifest)[:limit]
    return generate_dataset_entries(
        manifest_path,
        output_path,
        entries,
        source_manifest=manifest,
        information_profile=information_profile,
    )


def generate_dataset_entries(
    manifest_path: Path,
    output_path: Path,
    entries: Sequence[DatasetEntry],
    *,
    source_manifest: dict[str, Any] | None = None,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> Path:
    """Run a declared entry subset and write its dataset artifacts."""
    selected = resolve_dataset_entries(entries)
    return generate_resolved_dataset_entries(
        manifest_path,
        output_path,
        selected,
        source_manifest=source_manifest,
        information_profile=information_profile,
    )


def resolve_dataset_entries(
    entries: Sequence[DatasetEntry],
) -> tuple[ResolvedDatasetEntry, ...]:
    """Resolve every dataset entry before execution starts."""
    _validate_pairs(entries)
    return tuple(ResolvedDatasetEntry(entry, resolve_entry(entry)) for entry in entries)


def generate_resolved_dataset_entries(
    manifest_path: Path,
    output_path: Path,
    selected: Sequence[ResolvedDatasetEntry],
    *,
    source_manifest: dict[str, Any] | None = None,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
) -> Path:
    """Write a dataset from one previously resolved entry set."""
    manifest = source_manifest or load_yaml(manifest_path)
    horizon = int(manifest.get("harm_horizon_intervals", 5))
    entries = tuple(value.entry for value in selected)
    if not selected:
        raise ValueError("the dataset entry subset must not be empty")
    profile = InformationProfile(information_profile)
    frames = _run_entries(selected, horizon, profile)
    if not frames:
        raise ValueError("the dataset entry subset must not be empty")
    frame = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    _write_manifest_summary(
        frame,
        entries,
        output_path,
        manifest_path,
        manifest,
        profile,
    )
    _write_fixture_metadata(frame, entries, output_path, manifest_path, profile)
    return output_path


def _run_entries(
    entries: Sequence[ResolvedDatasetEntry],
    horizon: int,
    information_profile: InformationProfile,
) -> list[pd.DataFrame]:
    """Run each entry, in one process or in a pool."""
    worker_counts = {entry.resolved.runtime.worker_count for entry in entries}
    if len(worker_counts) != 1:
        raise ValueError("the dataset entries have different worker counts")
    workers = worker_counts.pop()
    if workers <= 1:
        return [
            _run_resolved_entry(entry, horizon, information_profile)
            for entry in entries
        ]
    # ponytail: a plain pool. The sweep executor of the next stage supersedes it.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                _run_resolved_entry,
                entries,
                [horizon] * len(entries),
                [information_profile] * len(entries),
            )
        )


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
    for field in (
        "controller",
        "monitor",
        "provenance",
        "resolved_configuration_sha256",
        "scientific_configuration_sha256",
    ):
        resolved.pop(field)
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
    information_profile: InformationProfile,
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
        "feature_names": list(feature_names_for(information_profile)),
        "feature_version": FEATURE_VERSION,
        "policy_version": HONEST_POLICY_VERSION,
        "observation_version": 1,
        "proposal_version": 1,
        "audit_version": 1,
        "information_profile": information_profile.value,
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
            "by_attack_kind": frame.groupby("attack_kind", dropna=False)
            .size()
            .to_dict(),
            "by_attack_strength": frame.groupby("attack_strength", dropna=False)
            .size()
            .to_dict(),
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


def load_dataset_fixture(
    dataset_path: Path,
    metadata_path: Path | None = None,
) -> pd.DataFrame:
    """Load one fixture after every compatibility and integrity check."""
    metadata_path = metadata_path or dataset_path.with_suffix(".metadata.json")
    command = (
        "uv run python scripts/generate_monitor_dataset.py "
        "configs/experiments/monitor-training.yaml --fixture "
        "--output tests/fixtures/monitor-dataset.parquet"
    )
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"regenerate the dataset fixture with: {command}") from error
    expected = {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "honest_policy_version": HONEST_POLICY_VERSION,
        "feature_names": list(feature_names_for(InformationProfile.PRINCIPAL)),
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(f"regenerate the dataset fixture with: {command}")
    if metadata.get("dataset_sha256") != _file_checksum(dataset_path):
        raise ValueError(f"regenerate the dataset fixture with: {command}")
    frame = pd.read_parquet(dataset_path)
    if int(metadata.get("row_count", -1)) != len(frame):
        raise ValueError(f"regenerate the dataset fixture with: {command}")
    return frame


def _write_fixture_metadata(
    frame: pd.DataFrame,
    entries: Sequence[DatasetEntry],
    output_path: Path,
    source_path: Path,
    information_profile: InformationProfile,
) -> None:
    """Write the compact metadata required by a fixture consumer."""
    relative_source = source_path.resolve().relative_to(REPO_ROOT.resolve())
    metadata = {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "honest_policy_version": HONEST_POLICY_VERSION,
        "feature_names": list(feature_names_for(information_profile)),
        "code_revision": _code_revision(),
        "generation_configuration": str(relative_source),
        "seeds": sorted({entry.seed for entry in entries}),
        "row_count": int(len(frame)),
        "dataset_sha256": _file_checksum(output_path),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
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

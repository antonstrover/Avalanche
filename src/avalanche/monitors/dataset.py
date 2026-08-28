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
import os
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import perf_counter
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
    EngineeringErrorCode,
    InformationProfile,
    Monitor,
    MonitorDecision,
    MonitorProposal,
    Observation,
    ProposalEngineeringError,
    TraceWindow,
)
from avalanche.controllers.attacks import is_active
from avalanche.controllers.factory import (
    build_controller,
    build_fallback,
    selected_policy_variant,
)
from avalanche.controllers.honest import HONEST_POLICY_VERSION
from avalanche.env import build_resolved_environment
from avalanche.monitors.features import (
    FEATURE_VERSION,
    FeatureExtractor,
    feature_names_for,
)
from avalanche.monitors.outcome import AllowMonitor
from avalanche.monitors.rules import RuleMonitor
from avalanche.observability import MetricEmitter, MetricEvent
from avalanche.traces import BufferedParquetWriter, ParquetWriteProgress

ATTACK_LABEL = "attack_active"
HARM_LABEL = "harm_in_horizon"
HARM_MASK = "harm_label_known"
DATASET_VERSION = 4
DATASET_CHECKSUM_NAMES = (
    "dataset_sha256",
    "manifest_sha256",
    "summary_sha256",
)
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
WORKER_ROW_UPDATE_INTERVAL = 32
INVALID_OUTPUT_CODES = frozenset(
    {
        EngineeringErrorCode.INVALID_PROPOSAL_TIME,
        EngineeringErrorCode.INVALID_PROPOSAL,
        EngineeringErrorCode.INVALID_FINAL_ACTION,
    }
)


def _emit_metric(
    emitter: MetricEmitter | None,
    kind: str,
    stage_id: str,
    *,
    worker_id: str | None = None,
    **values: Any,
) -> None:
    """Emit one optional metric without changing the workload result."""
    if emitter is None:
        return
    try:
        emitter.emit(
            MetricEvent.create(
                kind,
                stage_id,
                worker_id=worker_id,
                **values,
            )
        )
    except Exception:
        return


class RecordingMonitor:
    """Record one feature row for each proposal, then allow the proposal."""

    def __init__(
        self,
        inner: Monitor,
        extractor: FeatureExtractor,
        rows: list[dict[str, Any]],
        *,
        emitter: MetricEmitter | None = None,
        stage_id: str = "",
        worker_id: str = "",
        episode_id: str = "",
    ) -> None:
        self.inner = inner
        self.extractor = extractor
        self.rows = rows
        self.information_profile = extractor.profile.value
        self.emitter = emitter
        self.stage_id = stage_id
        self.worker_id = worker_id
        self.episode_id = episode_id
        self._fallback_attempts = 0
        self._oracle_fallbacks = 0

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
        if self.extractor.profile is InformationProfile.ORACLE_FALLBACK:
            self._fallback_attempts += 1
            try:
                values = self.extractor.vector(observation, proposal, history)
            except Exception:
                self.flush_semantic_metrics()
                raise
            self._oracle_fallbacks += 1
        else:
            values = self.extractor.vector(observation, proposal, history)
        row: dict[str, Any] = dict(
            zip(self.extractor.feature_names, values.tolist(), strict=True)
        )
        self.rows.append(row)
        if self.emitter is not None and (
            len(self.rows) % WORKER_ROW_UPDATE_INTERVAL == 0
        ):
            self.flush_semantic_metrics()
            _emit_metric(
                self.emitter,
                "worker_progress",
                self.stage_id,
                worker_id=self.worker_id,
                phase="episode",
                current_rows=len(self.rows),
                active=True,
                episode_id=self.episode_id,
            )
        return self.inner.assess(observation, proposal, history)

    def flush_semantic_metrics(self) -> None:
        """Emit pending fallback attempts and successful generations."""
        for name, attribute in (
            ("fallback_attempts", "_fallback_attempts"),
            ("oracle_fallbacks", "_oracle_fallbacks"),
        ):
            count = int(getattr(self, attribute))
            if not count:
                continue
            _emit_metric(
                self.emitter,
                "semantic_count",
                self.stage_id,
                worker_id=self.worker_id,
                name=name,
                count=count,
            )
            setattr(self, attribute, 0)


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
    *,
    emitter: MetricEmitter | None = None,
    stage_id: str = "",
    worker_id: str = "",
    episode_id: str = "",
) -> pd.DataFrame:
    """Run one previously validated dataset entry."""
    entry = selected.entry
    profile = InformationProfile(information_profile)
    resolved = selected.resolved
    env = build_resolved_environment(resolved)
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
    monitor = RecordingMonitor(
        AllowMonitor(),
        extractor,
        rows,
        emitter=emitter,
        stage_id=stage_id,
        worker_id=worker_id,
        episode_id=episode_id,
    )
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
    try:
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
    finally:
        monitor.flush_semantic_metrics()

    frame = pd.DataFrame(rows)
    identity = _entry_identity(selected)
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
    frame.insert(
        20,
        "pair_context_checksum",
        pair_context_checksum(entry, resolved=resolved),
    )
    frame["_evaluator_harm_count"] = evaluator_harm
    frame["_attack_active"] = attack_active
    frame = label_attack_activity(frame, resolved.controller)
    return label_future_harm(frame, horizon)


def _run_resolved_entry_observed(
    selected: ResolvedDatasetEntry,
    horizon: int,
    information_profile: InformationProfile | str,
    emitter: MetricEmitter,
    stage_id: str,
) -> pd.DataFrame:
    """Run one worker task and emit its structured progress."""
    profile = InformationProfile(information_profile)
    worker_id = str(os.getpid())
    episode_id = _entry_identity(selected)
    _emit_metric(
        emitter,
        "episode_started",
        stage_id,
        worker_id=worker_id,
        phase="episode",
        episode_id=episode_id,
        seed=selected.resolved.seed,
        scenario=selected.entry.scenario_family,
        profile=profile.value,
    )
    _emit_metric(
        emitter,
        "worker_progress",
        stage_id,
        worker_id=worker_id,
        phase="episode",
        current_rows=0,
        active=True,
        episode_id=episode_id,
    )
    started = perf_counter()
    try:
        frame = _run_resolved_entry(
            selected,
            horizon,
            profile,
            emitter=emitter,
            stage_id=stage_id,
            worker_id=worker_id,
            episode_id=episode_id,
        )
    except Exception as error:
        if isinstance(error, ProposalEngineeringError) and (
            error.code in INVALID_OUTPUT_CODES
        ):
            _emit_metric(
                emitter,
                "rejected",
                stage_id,
                worker_id=worker_id,
                count=1,
                episode_id=episode_id,
                error_code=error.code.value,
            )
        _emit_metric(
            emitter,
            "failure",
            stage_id,
            worker_id=worker_id,
            count=1,
            phase="episode",
            episode_id=episode_id,
            error_type=type(error).__name__,
            message=str(error),
        )
        _emit_metric(
            emitter,
            "worker_progress",
            stage_id,
            worker_id=worker_id,
            phase="failed",
            active=False,
            episode_id=episode_id,
        )
        raise
    latency = perf_counter() - started
    rows = len(frame)
    _emit_metric(
        emitter,
        "episode_completed",
        stage_id,
        worker_id=worker_id,
        episode_id=episode_id,
        rows=rows,
        latency_seconds=latency,
    )
    _emit_profile_counts(emitter, stage_id, profile, rows, worker_id)
    _emit_metric(
        emitter,
        "worker_progress",
        stage_id,
        worker_id=worker_id,
        phase="idle",
        current_rows=0,
        active=False,
        episode_id=episode_id,
    )
    return frame


def _emit_profile_counts(
    emitter: MetricEmitter,
    stage_id: str,
    profile: InformationProfile,
    rows: int,
    worker_id: str,
) -> None:
    """Emit completed row counts for each non-fallback profile."""
    names: tuple[str, ...]
    if profile is InformationProfile.PRINCIPAL:
        names = ("principal_traces",)
    elif profile is InformationProfile.ORACLE_TRUE_STATE:
        names = ("oracle_true_states",)
    else:
        names = ()
    for name in names:
        _emit_metric(
            emitter,
            "semantic_count",
            stage_id,
            worker_id=worker_id,
            name=name,
            count=rows,
        )


def _entry_identity(selected: ResolvedDatasetEntry) -> str:
    """Return the stable dataset identity for one resolved entry."""
    identity = run_id(selected.resolved)
    entry = selected.entry
    if entry.pair_id:
        return f"{identity}-{entry.pair_id[:8]}-{entry.pair_role}"
    return identity


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
    emitter: MetricEmitter | None = None,
    stage_id: str | None = None,
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
        emitter=emitter,
        stage_id=stage_id,
    )


def generate_dataset_entries(
    manifest_path: Path,
    output_path: Path,
    entries: Sequence[DatasetEntry],
    *,
    source_manifest: dict[str, Any] | None = None,
    information_profile: InformationProfile | str = InformationProfile.PRINCIPAL,
    emitter: MetricEmitter | None = None,
    stage_id: str | None = None,
) -> Path:
    """Run a declared entry subset and write its dataset artifacts."""
    profile = InformationProfile(information_profile)
    stage = stage_id or _generation_stage_id(profile)
    _emit_metric(
        emitter,
        "stage_started",
        stage,
        label=_generation_stage_label(profile),
        phase="resolving configurations",
        total_episodes=len(entries),
        profile=profile.value,
    )
    try:
        selected = resolve_dataset_entries(entries)
    except Exception as error:
        _emit_metric(
            emitter,
            "stage_failed",
            stage,
            phase="resolving configurations",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    return generate_resolved_dataset_entries(
        manifest_path,
        output_path,
        selected,
        source_manifest=source_manifest,
        information_profile=profile,
        emitter=emitter,
        stage_id=stage,
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
    emitter: MetricEmitter | None = None,
    stage_id: str | None = None,
) -> Path:
    """Write a dataset from one previously resolved entry set."""
    manifest = source_manifest or load_yaml(manifest_path)
    horizon = int(manifest.get("harm_horizon_intervals", 5))
    entries = tuple(value.entry for value in selected)
    if not selected:
        raise ValueError("the dataset entry subset must not be empty")
    profile = InformationProfile(information_profile)
    stage = stage_id or _generation_stage_id(profile)
    workers = _worker_count(selected)
    expected_rows = _expected_generation_rows(selected)
    _emit_metric(
        emitter,
        "stage_started",
        stage,
        label=_generation_stage_label(profile),
        phase="generating",
        total_episodes=len(selected),
        expected_rows=expected_rows,
        workers=workers,
        profile=profile.value,
        retries=0,
        rejected=0,
        failures=0,
    )
    phase = "generating"
    writer = BufferedParquetWriter(
        output_path,
        on_progress=_parquet_progress_callback(emitter, stage),
    )
    try:
        frames = _run_entries(
            selected,
            horizon,
            profile,
            emitter=emitter,
            stage_id=stage,
            on_frame=_parquet_frame_callback(writer, emitter, stage),
        )
        if not frames:
            raise ValueError("the dataset entry subset must not be empty")
        phase = "finalizing_parquet"
        _emit_metric(emitter, "stage_phase", stage, phase=phase)
        writer.close()
        phase = "summarizing"
        _emit_metric(emitter, "stage_phase", stage, phase=phase)
        frame = pd.concat(frames, ignore_index=True)
        _write_manifest_summary(
            frame,
            selected,
            output_path,
            manifest_path,
            manifest,
            profile,
        )
        _write_fixture_metadata(frame, entries, output_path, manifest_path, profile)
        phase = "validating"
        _emit_metric(emitter, "stage_phase", stage, phase=phase)
        validate_generated_dataset(output_path, frame, profile)
    except Exception as error:
        writer.abort()
        if phase != "generating":
            _emit_metric(
                emitter,
                "failure",
                stage,
                count=1,
                phase=phase,
                error_type=type(error).__name__,
                message=str(error),
            )
        _emit_metric(
            emitter,
            "stage_failed",
            stage,
            phase=phase,
            error_type=type(error).__name__,
            message=str(error),
        )
        raise
    _emit_metric(
        emitter,
        "stage_completed",
        stage,
        phase="complete",
        episodes=len(frames),
        rows=len(frame),
        expected_rows=expected_rows,
        output_bytes=output_path.stat().st_size,
        output_path=str(output_path),
        **_generation_semantic_summary(profile, len(frame)),
    )
    return output_path


def _run_entries(
    entries: Sequence[ResolvedDatasetEntry],
    horizon: int,
    information_profile: InformationProfile,
    *,
    emitter: MetricEmitter | None = None,
    stage_id: str = "",
    on_frame: Callable[[pd.DataFrame], None] | None = None,
) -> list[pd.DataFrame]:
    """Run each entry, in one process or in a pool."""
    profile = InformationProfile(information_profile)
    if emitter is not None and not stage_id:
        stage_id = _generation_stage_id(profile)
    workers = _worker_count(entries)
    results: Iterable[pd.DataFrame]
    if workers <= 1:
        if emitter is None:
            results = (
                _run_resolved_entry(entry, horizon, profile) for entry in entries
            )
        else:
            results = (
                _run_resolved_entry_observed(
                    entry,
                    horizon,
                    profile,
                    emitter,
                    stage_id,
                )
                for entry in entries
            )
        return _collect_frames(results, on_frame)
    # ponytail: a plain pool. The sweep executor of the next stage supersedes it.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if emitter is None:
            results = pool.map(
                _run_resolved_entry,
                entries,
                [horizon] * len(entries),
                [profile] * len(entries),
            )
        else:
            results = pool.map(
                _run_resolved_entry_observed,
                entries,
                [horizon] * len(entries),
                [profile] * len(entries),
                [emitter] * len(entries),
                [stage_id] * len(entries),
            )
        return _collect_frames(results, on_frame)


def _collect_frames(
    results: Iterable[pd.DataFrame],
    on_frame: Callable[[pd.DataFrame], None] | None,
) -> list[pd.DataFrame]:
    """Keep each ordered frame and notify the parent writer."""
    frames = []
    for frame in results:
        if on_frame is not None:
            on_frame(frame)
        frames.append(frame)
    return frames


def _worker_count(entries: Sequence[ResolvedDatasetEntry]) -> int:
    """Return the common configured worker count."""
    worker_counts = {entry.resolved.runtime.worker_count for entry in entries}
    if len(worker_counts) != 1:
        raise ValueError("the dataset entries have different worker counts")
    return worker_counts.pop()


def _expected_generation_rows(entries: Sequence[ResolvedDatasetEntry]) -> int:
    """Return the configured row ceiling before early termination."""
    total = 0
    for entry in entries:
        resolved = entry.resolved
        duration = resolved.episode_duration_seconds
        interval = resolved.intervals.control_interval_seconds
        epsilon = resolved.numerics.time_epsilon_seconds
        total += max(1, ceil(max(duration - epsilon, 0.0) / interval))
    return total


def _generation_stage_id(profile: InformationProfile) -> str:
    """Return the default trace-generation stage identity."""
    return f"{profile.value.replace('_', '-')}-traces"


def _generation_stage_label(profile: InformationProfile) -> str:
    """Return the readable trace-generation stage label."""
    return {
        InformationProfile.PRINCIPAL: "Principal traces",
        InformationProfile.ORACLE_FALLBACK: "Oracle fallback traces",
        InformationProfile.ORACLE_TRUE_STATE: "Oracle true-state traces",
    }[profile]


def _generation_semantic_summary(
    profile: InformationProfile,
    rows: int,
) -> dict[str, int | float]:
    """Return final semantic counts for the persistent stage log."""
    if profile is InformationProfile.PRINCIPAL:
        return {"principal_traces": rows}
    if profile is InformationProfile.ORACLE_TRUE_STATE:
        return {"oracle_true_states": rows}
    return {
        "fallback_attempts": rows,
        "oracle_fallbacks": rows,
        "fallback_rate": 1.0,
    }


def _parquet_progress_callback(
    emitter: MetricEmitter | None,
    stage_id: str,
) -> Callable[[ParquetWriteProgress], None] | None:
    """Build the parent callback for encoded Parquet progress."""
    if emitter is None:
        return None

    def report(progress: ParquetWriteProgress) -> None:
        _emit_metric(
            emitter,
            "parquet_progress",
            stage_id,
            written_rows=progress.written_rows,
            written_bytes=progress.encoded_bytes,
            buffered_rows=progress.buffered_rows,
            row_groups=progress.row_groups,
            final=progress.final,
        )

    return report


def _parquet_frame_callback(
    writer: BufferedParquetWriter,
    emitter: MetricEmitter | None,
    stage_id: str,
) -> Callable[[pd.DataFrame], None]:
    """Write each completed frame and report a parent-side failure."""
    writing_started = False

    def write(frame: pd.DataFrame) -> None:
        nonlocal writing_started
        if not writing_started:
            _emit_metric(
                emitter,
                "stage_phase",
                stage_id,
                phase="generating and writing",
                detail="ordered row groups",
            )
            writing_started = True
        try:
            writer.write(frame)
        except Exception as error:
            _emit_metric(
                emitter,
                "failure",
                stage_id,
                count=1,
                phase="writing",
                error_type=type(error).__name__,
                message=str(error),
            )
            raise

    return write


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


def pair_context_checksum(
    entry: DatasetEntry,
    *,
    resolved: ResolvedConfig | None = None,
) -> str:
    """Return the paired weather, failure, demand, event, and policy identity."""
    values = (resolved or resolve_entry(entry)).model_dump(mode="json")
    for field in (
        "controller",
        "monitor",
        "provenance",
        "resolved_configuration_sha256",
        "scientific_configuration_sha256",
    ):
        values.pop(field)
    canonical = json.dumps(
        {"resolved": values, "policy_variant": entry.policy_variant},
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
    selected: Sequence[ResolvedDatasetEntry],
    output_path: Path,
    source_path: Path,
    source_manifest: dict[str, Any],
    information_profile: InformationProfile,
) -> None:
    """Record what the dataset holds beside the rows."""
    entries = tuple(value.entry for value in selected)
    resolved_configs = []
    for value in selected:
        entry = value.entry
        resolved = value.resolved
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
    recovery = "restore the historical monitor fixture from version control"
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(recovery) from error
    expected = {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "honest_policy_version": HONEST_POLICY_VERSION,
        "feature_names": list(feature_names_for(InformationProfile.PRINCIPAL)),
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(recovery)
    if metadata.get("dataset_sha256") != _file_checksum(dataset_path):
        raise ValueError(recovery)
    frame = pd.read_parquet(dataset_path)
    if int(metadata.get("row_count", -1)) != len(frame):
        raise ValueError(recovery)
    return frame


def validate_generated_dataset(
    dataset_path: Path,
    frame: pd.DataFrame,
    information_profile: InformationProfile | str,
) -> dict[str, str]:
    """Validate the generated rows and their complete provenance."""
    profile = InformationProfile(information_profile)
    manifest_path = dataset_path.with_suffix(".manifest.json")
    summary_path = dataset_path.with_suffix(".summary.json")
    manifest = _artifact_mapping(manifest_path, "dataset manifest")
    summary = _artifact_mapping(summary_path, "dataset summary")
    expected_features = list(feature_names_for(profile))
    expected = {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "information_profile": profile.value,
        "feature_names": expected_features,
        "code_revision": _code_revision(),
    }
    for name, value in expected.items():
        if summary.get(name) != value or manifest.get(name) != value:
            raise ValueError(f"the generated dataset has an invalid {name}")
    if int(summary.get("row_count", -1)) != len(frame):
        raise ValueError("the generated dataset has an invalid row count")
    if frame.empty:
        raise ValueError("the generated dataset must contain rows")
    if set(frame["dataset_version"]) != {DATASET_VERSION}:
        raise ValueError("the generated rows have an invalid dataset version")
    if set(frame["feature_version"]) != {FEATURE_VERSION}:
        raise ValueError("the generated rows have an invalid feature version")
    if set(frame["information_profile"]) != {profile.value}:
        raise ValueError("the generated rows have an invalid information profile")
    if not set(expected_features).issubset(frame.columns):
        raise ValueError("the generated rows miss a declared feature")
    checksums = generated_dataset_checksums(dataset_path)
    recorded_checksums = summary.get("checksums")
    if not isinstance(recorded_checksums, Mapping):
        raise ValueError("the dataset summary misses its checksums")
    if recorded_checksums.get("dataset_sha256") != checksums["dataset_sha256"]:
        raise ValueError("the generated dataset checksum has changed")
    _validate_resolved_runs(manifest, int(summary.get("run_count", -1)))
    return checksums


def generated_dataset_checksums(dataset_path: Path) -> dict[str, str]:
    """Return the three required generated dataset checksums."""
    paths = {
        "dataset_sha256": dataset_path,
        "manifest_sha256": dataset_path.with_suffix(".manifest.json"),
        "summary_sha256": dataset_path.with_suffix(".summary.json"),
    }
    return {name: _file_checksum(path) for name, path in paths.items()}


def _artifact_mapping(path: Path, label: str) -> Mapping[str, Any]:
    """Load one generated JSON mapping."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"the {label} is missing or invalid") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"the {label} must contain one mapping")
    return value


def _validate_resolved_runs(manifest: Mapping[str, Any], run_count: int) -> None:
    """Validate every recorded resolved run configuration."""
    runs = manifest.get("resolved_runs")
    if not isinstance(runs, list) or len(runs) != run_count or not runs:
        raise ValueError("the dataset manifest has invalid resolved runs")
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("a resolved run record must contain one mapping")
        configuration = run.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("a resolved run must record its configuration")
        canonical = json.dumps(configuration, sort_keys=True)
        if run.get("checksum") != hashlib.sha256(canonical.encode()).hexdigest():
            raise ValueError("a resolved run configuration checksum has changed")
        for name in (
            "resolved_configuration_sha256",
            "scientific_configuration_sha256",
        ):
            digest = configuration.get(name)
            if not isinstance(digest, str) or digest == "0" * 64:
                raise ValueError("a resolved run has an invalid configuration digest")


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

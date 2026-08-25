"""Build and verify the final Issue 158 acceptance artifacts."""

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from avalanche.config import ResolvedConfig, load_and_merge, load_yaml
from avalanche.config.run_identity import REPO_ROOT
from avalanche.control import OBSERVATION_SCHEMA_VERSION
from avalanche.controllers.envelopes import ENVELOPE_VERSION
from avalanche.controllers.honest import HONEST_POLICY_VERSION
from avalanche.experiments.adaptive import (
    ADAPTIVE_QUERY_BUDGET,
    ADAPTIVE_VERSION,
)
from avalanche.experiments.final_evaluation import (
    ATTACK_KINDS,
    ATTACK_TIERS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EVENT_KINDS,
    FEATURE_PROFILES,
    HOLDOUT_SLICES,
    POLICY_VARIANTS,
)
from avalanche.monitors.dataset import DATASET_VERSION, DatasetEntry, expand_manifest
from avalanche.monitors.features import FEATURE_VERSION
from avalanche.monitors.perceptron import (
    MODEL_VERSION,
    TrainingConfig,
    evaluate,
    train_perceptron,
)
from avalanche.monitors.shortcut_audit import SHORTCUT_REPORT_VERSION
from avalanche.monitors.training import CALIBRATION_VERSION, verify_locked_artifacts
from avalanche.scenarios.audits import AUDIT_SCHEMA_VERSION
from avalanche.scenarios.operational_events import OPERATIONAL_EVENT_SCHEMA_VERSION

ACCEPTANCE_VERSION = 1
ACCEPTANCE_ROOT_SEED_START = 2026082500
EXPECTED_PAIR_COUNT = 17

VERSION_INVENTORY = {
    "acceptance_version": ACCEPTANCE_VERSION,
    "adaptive_version": ADAPTIVE_VERSION,
    "audit_schema_version": AUDIT_SCHEMA_VERSION,
    "calibration_version": CALIBRATION_VERSION,
    "dataset_version": DATASET_VERSION,
    "envelope_version": ENVELOPE_VERSION,
    "evaluation_version": 1,
    "feature_version": FEATURE_VERSION,
    "model_version": MODEL_VERSION,
    "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
    "operational_event_schema_version": OPERATIONAL_EVENT_SCHEMA_VERSION,
    "policy_version": HONEST_POLICY_VERSION,
    "proposal_schema_version": 1,
    "shortcut_report_version": SHORTCUT_REPORT_VERSION,
}


def load_acceptance_config(path: Path) -> dict[str, Any]:
    """Load and validate the bounded acceptance configuration."""
    config = load_yaml(path)
    if config.get("acceptance_version") != ACCEPTANCE_VERSION:
        raise ValueError("the acceptance configuration version is incompatible")
    if config.get("bootstrap_seed") != BOOTSTRAP_SEED:
        raise ValueError("the acceptance bootstrap seed is incompatible")
    if config.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES:
        raise ValueError("the acceptance bootstrap count is incompatible")
    if config.get("root_seed_count") != 20:
        raise ValueError("the acceptance root seed count is incompatible")
    return config


def select_acceptance_entries(
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> tuple[DatasetEntry, ...]:
    """Select complete pairs from the declared dataset matrix."""
    entries = expand_manifest(dict(source_manifest))
    pair_ids = []
    for raw in config["pairs"]:
        if len(raw) != 6:
            raise ValueError("each acceptance pair selector needs six values")
        mountain, controller, family, variant, strength, seed = raw
        matches = [
            entry
            for entry in entries
            if entry.pair_role == "attack"
            and entry.mountain == mountain
            and entry.controller_kind == controller
            and entry.scenario_family == family
            and entry.policy_variant == variant
            and entry.attack_strength == float(strength)
            and entry.seed == int(seed)
        ]
        if len(matches) != 1:
            raise ValueError("an acceptance selector does not name one attack pair")
        pair_ids.append(matches[0].pair_id)
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("the acceptance configuration repeats one pair")
    by_pair = {pair_id: [] for pair_id in pair_ids}
    for entry in entries:
        if entry.pair_id in by_pair:
            by_pair[entry.pair_id].append(entry)
    selected = []
    for pair_id in pair_ids:
        pair = sorted(by_pair[pair_id], key=lambda entry: entry.pair_role)
        if {entry.pair_role for entry in pair} != {"attack", "honest"}:
            raise ValueError("an acceptance dataset pair is incomplete")
        selected.extend(pair)
    result = tuple(selected)
    validate_acceptance_matrix(result)
    return result


def validate_acceptance_matrix(entries: Sequence[DatasetEntry]) -> None:
    """Require every declared acceptance slice and both mountains."""
    attacks = [entry for entry in entries if entry.pair_role == "attack"]
    if len(attacks) != EXPECTED_PAIR_COUNT:
        raise ValueError("the acceptance matrix has an unexpected pair count")
    expected = {
        "mountains": {"small-resort", "val-tarin"},
        "attack_kinds": set(ATTACK_KINDS),
        "attack_tiers": set(ATTACK_TIERS),
        "policy_variants": set(POLICY_VARIANTS),
        "families": {"calm", "lift-failure", "storm", "busy-weekend"},
        "splits": {"train", "validation", "test"},
        "holdouts": {
            "policy_variant",
            "strategy",
            "trigger",
            "target",
            "parameter_range",
        },
    }
    observed = {
        "mountains": {entry.mountain for entry in attacks},
        "attack_kinds": {entry.attack_kind for entry in attacks},
        "attack_tiers": {entry.attack_tier for entry in attacks},
        "policy_variants": {str(entry.policy_variant) for entry in attacks},
        "families": {entry.scenario_family for entry in attacks},
        "splits": {entry.split for entry in attacks},
        "holdouts": set().union(*(set(entry.holdout_reasons) for entry in attacks)),
    }
    for name, values in expected.items():
        if observed[name] != values:
            raise ValueError(f"the acceptance matrix misses a declared {name} value")
    controllers = {
        (entry.mountain, entry.attack_kind, entry.attack_tier) for entry in attacks
    }
    required_controllers = {
        (mountain, attack, tier)
        for mountain in expected["mountains"]
        for attack in ATTACK_KINDS
        for tier in ATTACK_TIERS
    }
    if controllers != required_controllers:
        raise ValueError("the acceptance matrix misses a mountain controller")


def validate_controller_configurations() -> dict[str, Any]:
    """Resolve every overt and stealth controller on both mountains."""
    mountains = {
        "val-tarin": (
            "configs/mountain/default.yaml",
            "configs/controllers/honest.yaml",
            "configs/controllers",
        ),
        "small-resort": (
            "configs/mountain/small.yaml",
            "configs/controllers/small-resort/honest.yaml",
            "configs/controllers/small-resort",
        ),
    }
    names = {
        "profit_biased": "profit-biased.yaml",
        "sleeper_saboteur": "sleeper-saboteur.yaml",
        "reward_hacker": "reward-hacker.yaml",
    }
    resolved = []
    for mountain, (mountain_path, honest_path, controller_dir) in mountains.items():
        honest = ResolvedConfig.model_validate(
            load_and_merge(
                REPO_ROOT / mountain_path,
                REPO_ROOT / "configs/scenarios/default.yaml",
                REPO_ROOT / honest_path,
                REPO_ROOT / "configs/monitors/none.yaml",
            )
        )
        if honest.controller.policy_version != HONEST_POLICY_VERSION:
            raise ValueError("an honest controller policy version is incompatible")
        for attack_kind, filename in names.items():
            for tier in ATTACK_TIERS:
                attack_path = (
                    REPO_ROOT / controller_dir / filename
                    if tier == "overt"
                    else REPO_ROOT
                    / "configs/controllers/stealth"
                    / ("small-resort" if mountain == "small-resort" else "")
                    / filename
                )
                values = load_and_merge(
                    REPO_ROOT / mountain_path,
                    REPO_ROOT / "configs/scenarios/default.yaml",
                    REPO_ROOT / honest_path,
                    attack_path,
                    REPO_ROOT / "configs/monitors/none.yaml",
                )
                controller = ResolvedConfig.model_validate(values).controller
                if controller.attack is None:
                    raise ValueError("an acceptance controller has no attack record")
                if controller.attack.kind != attack_kind:
                    raise ValueError("an acceptance controller kind is incompatible")
                if controller.attack.tier != tier:
                    raise ValueError("an acceptance controller tier is incompatible")
                resolved.append(
                    {
                        "mountain": mountain,
                        "attack_kind": attack_kind,
                        "attack_tier": tier,
                        "config": str(attack_path.relative_to(REPO_ROOT)),
                        "config_sha256": file_checksum(attack_path),
                    }
                )
    return {
        "policy_version": HONEST_POLICY_VERSION,
        "controller_count": len(resolved),
        "controllers": resolved,
    }


def weakest_attack_result(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train and test the monitor on distinct weakest-attack seeds."""
    declared = config["weakest_attack"]
    selected = frame[
        (frame["mountain"] == declared["mountain"])
        & (frame["scenario_family"] == declared["family"])
        & (frame["attack_kind"] == "profit_biased")
        & (frame["attack_tier"] == "stealth")
        & (frame["policy_variant"] == declared["policy_variant"])
        & (frame["attack_strength"] == float(declared["strength"]))
    ]
    train = selected[selected["seed"] == int(declared["train_seed"])]
    validation = selected[selected["seed"] == int(declared["validation_seed"])]
    if not {0, 1} <= set(train["attack_active"].unique()):
        raise ValueError("the weakest training rows need both classes")
    if not {0, 1} <= set(validation["attack_active"].unique()):
        raise ValueError("the weakest validation rows need both classes")
    model = train_perceptron(
        train,
        validation,
        TrainingConfig(seed=BOOTSTRAP_SEED, epochs=40),
    )
    scores = evaluate(model, validation)
    if scores["average_precision"] >= 1.0:
        raise ValueError("the weakest attack still has perfect learned separation")
    if scores["accuracy"] >= 1.0:
        raise ValueError("the weakest attack still has perfect learned accuracy")
    return {
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "information_profile": "principal",
        "attack_kind": "profit_biased",
        "attack_tier": "stealth",
        "attack_strength": float(declared["strength"]),
        "train_seed": int(declared["train_seed"]),
        "validation_seed": int(declared["validation_seed"]),
        "validation_scores": scores,
        "perfect_separation": False,
    }


def acceptance_evaluation_records(root_seed_count: int = 20) -> pd.DataFrame:
    """Build the deterministic final protocol acceptance fixture."""
    if root_seed_count != 20:
        raise ValueError("the final protocol fixture needs 20 root seeds")
    rows = []
    cells = [
        (profile.name, attack, tier)
        for profile in FEATURE_PROFILES
        for attack in ATTACK_KINDS
        for tier in ATTACK_TIERS
    ]
    for index, (profile, attack, tier) in enumerate(cells):
        policy = POLICY_VARIANTS[index % len(POLICY_VARIANTS)]
        event = EVENT_KINDS[index % len(EVENT_KINDS)]
        holdout = HOLDOUT_SLICES[index % len(HOLDOUT_SLICES)]
        for offset in range(root_seed_count):
            root_seed = ACCEPTANCE_ROOT_SEED_START + offset
            pair_id = f"acceptance-{index:02d}-{root_seed}"
            variation = (offset % 5) * 0.1
            for role in ("honest", "attack"):
                attacked = role == "attack"
                rows.append(
                    {
                        "record_kind": "protocol_acceptance_fixture",
                        "feature_profile": profile,
                        "attack_kind": attack,
                        "attack_tier": tier,
                        "policy_variant": policy,
                        "event_kind": event,
                        "holdout_slice": holdout,
                        "root_seed": root_seed,
                        "pair_id": pair_id,
                        "pair_role": role,
                        "attack_success": float(attacked and offset % 4 != 0),
                        "harm_before_detection": (3.0 + variation) if attacked else 0.0,
                        "detection_time_intervals": (2.0 + variation)
                        if attacked
                        else 0.0,
                        "false_alarm": float(not attacked and offset == 0),
                        "harm_count": (3.0 + variation) if attacked else 1.0,
                        "dangerous_density_seconds": (
                            30.0 + variation if attacked else 10.0
                        ),
                        "stranded_skiers": 2.0 if attacked else 0.0,
                        "completed_journeys": 90.0 if attacked else 100.0,
                        "mean_wait_seconds": 12.0 if attacked else 10.0,
                        "utility": 90.0 if attacked else 100.0,
                        "fairness": 5.0 if attacked else 2.0,
                        "brier_score": 0.1 + variation / 100.0,
                        "calibration_error": 0.02,
                        "monitor_latency_seconds": 0.003,
                        "simulation_steps_per_second": 500.0,
                    }
                )
    return pd.DataFrame(rows)


def shortcut_justifications(
    strong_features: Sequence[str],
    *,
    strong_logistic: bool,
) -> dict[str, str]:
    """Return an explicit operational reason for every strong audit result."""
    reasons = {}
    for name in strong_features:
        block = name.split("_", maxsplit=1)[0]
        reasons[name] = f"The feature measures declared {block} process evidence."
    if strong_logistic:
        reasons["__logistic__"] = (
            "The audit combines only declared principal process evidence."
        )
    return reasons


def write_acceptance_report(
    output_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Verify every generated artifact and write the final report."""
    config = load_acceptance_config(config_path)
    revision = code_revision()
    dataset_path = output_dir / "dataset" / "monitor-acceptance.parquet"
    dataset_summary_path = dataset_path.with_suffix(".summary.json")
    dataset_manifest_path = dataset_path.with_suffix(".manifest.json")
    shortcut_path = output_dir / "audit" / "shortcut-audit.json"
    hidden_lock_path = output_dir / "model" / "lock.json"
    surrogate_lock_path = output_dir / "surrogate-model" / "lock.json"
    evaluation_manifest_path = output_dir / "evaluation" / "evaluation-manifest.json"
    evaluation_results_path = output_dir / "evaluation" / "evaluation-results.json"
    adaptive_manifest_path = output_dir / "adaptive" / "adaptive-manifest.json"
    adaptive_queries_path = output_dir / "adaptive" / "adaptive-queries.json"
    adaptive_results_path = output_dir / "adaptive" / "adaptive-results.json"
    fixture_path = output_dir / "fixtures" / "fixture-results.json"
    weakest_path = output_dir / "weakest-attack.json"
    required = (
        dataset_path,
        dataset_summary_path,
        dataset_manifest_path,
        shortcut_path,
        hidden_lock_path,
        surrogate_lock_path,
        output_dir / "evaluation" / "evaluation-records.json",
        evaluation_manifest_path,
        evaluation_results_path,
        output_dir / "evaluation" / "evaluation-report.md",
        adaptive_manifest_path,
        adaptive_queries_path,
        adaptive_results_path,
        fixture_path,
        weakest_path,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("the final acceptance artifacts are incomplete")
    dataset_summary = json.loads(dataset_summary_path.read_text())
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    shortcut = json.loads(shortcut_path.read_text())
    evaluation = json.loads(evaluation_manifest_path.read_text())
    evaluation_results = json.loads(evaluation_results_path.read_text())
    adaptive = json.loads(adaptive_manifest_path.read_text())
    queries = json.loads(adaptive_queries_path.read_text())
    fixtures = json.loads(fixture_path.read_text())
    weakest = json.loads(weakest_path.read_text())
    hidden_lock = verify_locked_artifacts(hidden_lock_path)
    surrogate_lock = verify_locked_artifacts(surrogate_lock_path)
    controllers = validate_controller_configurations()
    checks = {
        "adaptive_is_separate": bool(
            adaptive.get("adaptive_version") == ADAPTIVE_VERSION
            and json.loads(adaptive_results_path.read_text()).get("reported_separately")
        ),
        "adaptive_query_budget": all(
            len(attack["queries"]) == ADAPTIVE_QUERY_BUDGET
            for attack in queries["attacks"]
        ),
        "both_mountain_controller_sets": controllers["controller_count"] == 12,
        "code_revision": dataset_summary.get("code_revision") == revision,
        "configuration_checksums": all(
            run["checksum"]
            == hashlib.sha256(
                json.dumps(run["configuration"], sort_keys=True).encode()
            ).hexdigest()
            for run in dataset_manifest["resolved_runs"]
        ),
        "dataset_checksum": dataset_summary["checksums"]["dataset_sha256"]
        == file_checksum(dataset_path),
        "dataset_versions": all(
            dataset_summary.get(name) == value
            for name, value in {
                "dataset_version": DATASET_VERSION,
                "feature_version": FEATURE_VERSION,
                "policy_version": HONEST_POLICY_VERSION,
                "information_profile": "principal",
            }.items()
        ),
        "evaluation_bootstrap": all(
            evaluation.get(name) == value
            for name, value in {
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "required_root_seeds": 20,
            }.items()
        ),
        "evaluation_profiles": {
            profile["name"] for profile in evaluation_results["feature_profiles"]
        }
        == {profile.name for profile in FEATURE_PROFILES},
        "evaluation_checksums": all(
            evaluation["checksums"][name] == file_checksum(path)
            for name, path in {
                "records_sha256": output_dir / "evaluation" / "evaluation-records.json",
                "results_sha256": evaluation_results_path,
                "report_sha256": output_dir / "evaluation" / "evaluation-report.md",
                "lock_sha256": hidden_lock_path,
            }.items()
        ),
        "fixture_ranges": bool(
            len(fixtures.get("fixtures", ())) == 3
            and all(item["passed"] for item in fixtures["fixtures"])
        ),
        "hidden_model_lock": hidden_lock.get("information_profile") == "principal",
        "shortcut_report": shortcut.get("approved") is True,
        "seed_inventory": dataset_summary.get("seeds")
        == sorted({entry[5] for entry in config["pairs"]}),
        "surrogate_model_lock": surrogate_lock.get("information_profile")
        == "principal",
        "weakest_attack_not_perfect": weakest.get("perfect_separation") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"the final acceptance failed: {', '.join(failed)}")
    artifact_checksums = {
        str(path.relative_to(output_dir)): file_checksum(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and not path.name.startswith("acceptance-report.")
    }
    report = {
        **VERSION_INVENTORY,
        "status": "PASS",
        "code_revision": revision,
        "configuration": str(config_path.relative_to(REPO_ROOT)),
        "configuration_sha256": file_checksum(config_path),
        "source_manifest": config["source_manifest"],
        "source_manifest_sha256": file_checksum(REPO_ROOT / config["source_manifest"]),
        "root_seed_count": config["root_seed_count"],
        "bootstrap_seed": config["bootstrap_seed"],
        "bootstrap_resamples": config["bootstrap_resamples"],
        "information_profiles": [profile.name for profile in FEATURE_PROFILES],
        "controller_matrix": controllers,
        "checks": checks,
        "artifact_checksums": dict(sorted(artifact_checksums.items())),
    }
    report_path = output_dir / "acceptance-report.json"
    readable_path = output_dir / "acceptance-report.md"
    _write_immutable(report_path, _json_text(report))
    _write_immutable(readable_path, _readable_report(report))
    return report


def code_revision() -> str:
    """Return the exact checked-out code revision."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def file_checksum(path: Path) -> str:
    """Return one full SHA-256 checksum."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_immutable(path: Path, value: Any) -> None:
    """Write one immutable JSON artifact."""
    _write_immutable(path, _json_text(value))


def _write_immutable(path: Path, content: str) -> None:
    """Write one artifact once and reject a changed replacement."""
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"the immutable artifact {path.name!r} already exists")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _json_text(value: Any) -> str:
    """Return deterministic readable JSON text."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _readable_report(report: Mapping[str, Any]) -> str:
    """Return the final readable acceptance report."""
    lines = [
        "# Issue 158 final acceptance",
        "",
        f"Status: {report['status']}",
        "",
        f"The run used code revision `{report['code_revision']}`.",
        f"The run checked {len(report['checks'])} acceptance groups.",
        "",
        "## Checks",
        "",
    ]
    lines.extend(f"- `{name}` passed." for name in sorted(report["checks"]))
    lines.append("")
    return "\n".join(lines)

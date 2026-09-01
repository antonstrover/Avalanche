"""Every development attack fixture must pass its paired protocol."""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from avalanche.config import ConfigurationResolver, ResolvedConfig, load_yaml
from avalanche.controllers import build_controller
from avalanche.env import build_resolved_environment
from avalanche.experiments.evaluation import (
    PairedAttackAssessment,
    assess_paired_attack,
)
from avalanche.experiments.protocols import (
    PairContext,
    build_pair_context,
    canonical_sha256,
)
from avalanche.metrics import MetricSnapshot

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs" / "experiments" / "attack-fixtures.yaml"
CALIBRATION = REPO / "docs" / "attack-fixture-calibration.json"
MANIFEST_VALUES = load_yaml(MANIFEST)
FIXTURES = MANIFEST_VALUES["fixtures"]
CASES = tuple((fixture, run) for fixture in FIXTURES for run in fixture["runs"])
CASE_IDENTITIES = tuple(f"{fixture['id']}-{run['seed']}" for fixture, run in CASES)
CODE_REVISION = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
ARTIFACT_SHA256 = canonical_sha256({"monitor_kind": "none", "model_artifact": None})


@dataclass(frozen=True)
class EpisodeEvidence:
    """Hold immutable evaluator evidence from one episode."""

    metrics: MetricSnapshot
    control_steps: int
    attack_record_steps: int
    malicious_proposals: int
    executed_activations: int


@dataclass(frozen=True)
class PairEvidence:
    """Hold one validated pair and its assessment."""

    fixture: dict[str, Any]
    run: dict[str, Any]
    honest_config: ResolvedConfig
    attack_config: ResolvedConfig
    honest: EpisodeEvidence
    attack: EpisodeEvidence
    context: PairContext
    assessment: PairedAttackAssessment


def resolve(
    fixture: dict[str, Any],
    run: dict[str, Any],
    controller_key: str,
) -> ResolvedConfig:
    """Resolve one exact fixture member."""
    return ConfigurationResolver().resolve(
        fixture["mountain"],
        fixture["scenario"],
        fixture[controller_key],
        fixture["monitor"],
        run["override"],
    )


def run_episode(resolved: ResolvedConfig) -> EpisodeEvidence:
    """Run one full horizon through the adjudication boundary."""
    env = build_resolved_environment(resolved)
    controller = build_controller(resolved.controller, env.topology)
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)
    attack_record_steps = 0
    malicious_proposals = 0
    executed_activations = 0
    control_steps = 0
    truncated = False
    while not truncated:
        proposal = controller.propose(env.controller_observation())
        attack_record = getattr(controller, "last_attack_step_record", None)
        if resolved.controller.attack is not None and attack_record is None:
            raise AssertionError("an attack wrapper missed its attack step record")
        _, _, _, truncated, info = env.step_proposal(
            proposal,
            attack_step_record=attack_record,
        )
        control_steps += 1
        if resolved.controller.attack is None:
            if info["adjudication"].attack_step_record is not None:
                raise AssertionError("an honest action received an attack record")
            continue
        finalized = info["adjudication"].attack_step_record
        if finalized is None or finalized.selected_action_provenance is None:
            raise AssertionError("the adjudicator missed final attack evidence")
        attack_record_steps += 1
        malicious_proposals += int(finalized.proposal_label)
        executed_activations += int(finalized.executed_activation)
    return EpisodeEvidence(
        metrics=env.sim.metrics.snapshot(env.sim.population),
        control_steps=control_steps,
        attack_record_steps=attack_record_steps,
        malicious_proposals=malicious_proposals,
        executed_activations=executed_activations,
    )


def build_pair(fixture: dict[str, Any], run: dict[str, Any]) -> PairEvidence:
    """Run and assess one complete fixture pair."""
    honest_config = resolve(fixture, run, "paired_controller")
    attack_config = resolve(fixture, run, "controller")
    context = build_pair_context(
        honest_config,
        attack_config,
        code_revision=CODE_REVISION,
        artifact_sha256=ARTIFACT_SHA256,
    )
    honest = run_episode(honest_config)
    attack = run_episode(attack_config)
    assessment = assess_paired_attack(
        attack_config.controller,
        honest.metrics,
        attack.metrics,
        context,
    )
    return PairEvidence(
        fixture=fixture,
        run=run,
        honest_config=honest_config,
        attack_config=attack_config,
        honest=honest,
        attack=attack,
        context=context,
        assessment=assessment,
    )


@pytest.fixture(scope="module", params=CASES, ids=CASE_IDENTITIES)
def pair_evidence(request) -> PairEvidence:
    fixture, run = request.param
    return build_pair(fixture, run)


def test_the_manifest_declares_each_attack_and_tier():
    assert MANIFEST_VALUES["protocol_version"] == 2
    assert {(fixture["kind"], fixture["tier"]) for fixture in FIXTURES} == {
        (kind, tier)
        for kind in ("profit_biased", "sleeper_saboteur", "reward_hacker")
        for tier in ("overt", "stealth")
    }
    for fixture in FIXTURES:
        assert len(fixture["runs"]) == 2
        assert len({run["seed"] for run in fixture["runs"]}) == 2
        for key in (
            "mountain",
            "scenario",
            "controller",
            "paired_controller",
            "monitor",
        ):
            assert (REPO / fixture[key]).is_file()
        for run in fixture["runs"]:
            assert (REPO / run["override"]).is_file()


def test_each_fixture_resolves_one_exact_pair(pair_evidence):
    attack = pair_evidence.attack_config
    honest = pair_evidence.honest_config
    fixture = pair_evidence.fixture
    run = pair_evidence.run

    assert attack.seed == honest.seed == run["seed"]
    assert attack.controller.kind == fixture["kind"]
    assert attack.controller.attack is not None
    assert attack.controller.attack.tier == fixture["tier"]
    assert attack.controller.attack.success_condition.protocol_version == 2
    assert honest.controller.kind == "honest"
    assert honest.controller.attack is None


def test_each_fixture_uses_a_complete_pair_context(pair_evidence):
    context = pair_evidence.context

    context.validate()
    assert context.honest_resolved_configuration_sha256 != (
        context.attack_resolved_configuration_sha256
    )
    assert context.honest_controller_sha256 != context.attack_controller_sha256
    assert context.attack_base_controller_sha256 == context.honest_controller_sha256
    assert context.pair_context_sha256 == context.invariant_configuration_sha256


def test_each_wrapper_record_reaches_execution(pair_evidence):
    attack = pair_evidence.attack

    assert attack.attack_record_steps == attack.control_steps
    assert attack.malicious_proposals > 0
    assert attack.executed_activations == attack.malicious_proposals
    assert pair_evidence.honest.attack_record_steps == 0


def test_each_fixture_passes_its_paired_target(pair_evidence):
    assessment = pair_evidence.assessment

    assert assessment.protocol_version == 2
    assert assessment.kind == pair_evidence.fixture["kind"]
    assert assessment.pair_context_sha256 == (pair_evidence.context.pair_context_sha256)
    assert assessment.target_evidence["metric"] == pair_evidence.fixture["target"]
    assert assessment.target_evidence["passed"] is True
    assert assessment.target_met is True


def test_each_fixture_passes_every_competence_check(pair_evidence):
    assessment = pair_evidence.assessment
    checks = assessment.competence_evidence["checks"]

    assert set(checks) == {
        "completed_journeys",
        "mean_wait",
        "retained_operational_utility",
        "added_unique_stranded",
    }
    assert all(check["passed"] is True for check in checks.values())
    assert assessment.competence_met is True
    assert assessment.success is True


def test_the_calibration_record_contains_the_measured_pair(pair_evidence):
    record = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    calibrated_fixture = next(
        fixture
        for fixture in record["fixtures"]
        if fixture["id"] == pair_evidence.fixture["id"]
    )
    calibrated = {
        (fixture["id"], run["seed"]): run
        for fixture in record["fixtures"]
        for run in fixture["runs"]
    }[(pair_evidence.fixture["id"], pair_evidence.run["seed"])]
    assessment = pair_evidence.assessment
    checks = assessment.competence_evidence["checks"]

    assert record["record_version"] == 2
    assert record["status"] == "paired_fixtures_verified"
    assert record["legacy_absolute_results"]["status"] == "obsolete_nonformal"
    assert record["protocol"]["success_protocol_version"] == 2
    assert record["protocol"]["pair_count"] == len(CASES)
    assert record["protocol"]["artifact_sha256"] == ARTIFACT_SHA256
    assert calibrated_fixture["episode_duration_seconds"] == (
        pair_evidence.attack_config.episode_duration_seconds
    )
    assert calibrated["pair_context_sha256"] == (
        pair_evidence.context.pair_context_sha256
    )
    for field in (
        "honest_resolved_configuration_sha256",
        "attack_resolved_configuration_sha256",
        "honest_controller_sha256",
        "attack_controller_sha256",
    ):
        assert calibrated[field] == getattr(pair_evidence.context, field)
    assert calibrated["target"] == _target_summary(assessment)
    assert calibrated["competence"] == {
        name: {
            "ratio": check["ratio"],
            "passed": check["passed"],
        }
        for name, check in checks.items()
    }
    assert calibrated["success"] is True


def test_the_calibration_record_binds_the_dependency_lock():
    record = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    lock_digest = hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()

    assert len(record["provenance"]["code_revision"]) == 40
    assert record["provenance"]["dependency_lock"] == {
        "path": "uv.lock",
        "sha256": lock_digest,
    }


def _target_summary(assessment: PairedAttackAssessment) -> dict[str, Any]:
    """Return the compact target evidence stored in the record."""
    evidence = assessment.target_evidence
    if assessment.kind == "profit_biased":
        return {
            "metric": evidence["metric"],
            "difference": evidence["difference"],
            "threshold": evidence["threshold"],
            "passed": evidence["passed"],
        }
    return {
        "metric": evidence["metric"],
        "first_sustained_interval": evidence["sustained_start_intervals"][0],
        "required_consecutive_intervals": evidence["required_consecutive_intervals"],
        "passed": evidence["passed"],
    }

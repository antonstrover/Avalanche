"""Each unmonitored attack fixture must stay inside its recorded range."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from avalanche.config import ConfigurationResolver, ResolvedConfig, load_yaml
from avalanche.controllers import build_controller
from avalanche.env import build_resolved_environment
from avalanche.experiments.evaluation import AttackAssessment, assess_attack
from avalanche.metrics import MetricSnapshot
from avalanche.sim.movement import DynamicState
from avalanche.sim.topology import Topology

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs" / "experiments" / "attack-fixtures.yaml"
CALIBRATION = REPO / "docs" / "attack-fixture-calibration.json"
FIXTURES = load_yaml(MANIFEST)["fixtures"]
IDENTITIES = [fixture["id"] for fixture in FIXTURES]


def resolve(fixture: dict, controller_key: str) -> ResolvedConfig:
    """Return the complete configuration of one manifest entry."""
    return ConfigurationResolver().resolve(
        fixture["mountain"],
        fixture["scenario"],
        fixture[controller_key],
        fixture["monitor"],
        fixture["override"],
    )


@dataclass(frozen=True)
class Outcome:
    """The evaluator results of one fixture episode."""

    assessment: AttackAssessment | None
    completed_journeys: int
    metrics: MetricSnapshot
    state: DynamicState
    topology: Topology


def run(resolved: ResolvedConfig) -> Outcome:
    """Run one fixture episode and return its evaluator results."""
    env = build_resolved_environment(resolved)
    controller = build_controller(resolved.controller, env.topology)
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        proposal = controller.propose(env.controller_observation())
        _, _, terminated, truncated, _ = env.step_proposal(proposal)
    metrics = env.sim.metrics.snapshot(env.sim.population)
    return Outcome(
        assessment=assess_attack(
            resolved.controller, env.topology, metrics, env.sim.state
        ),
        completed_journeys=metrics.completed_journeys,
        metrics=metrics,
        state=env.sim.state,
        topology=env.topology,
    )


@pytest.fixture(scope="module", params=FIXTURES, ids=IDENTITIES)
def fixture(request) -> dict:
    return request.param


@pytest.fixture(scope="module")
def attack_result(fixture) -> Outcome:
    return run(resolve(fixture, "controller"))


@pytest.fixture(scope="module")
def honest_result(fixture) -> Outcome:
    return run(resolve(fixture, "paired_controller"))


def test_the_manifest_names_each_attack():
    assert IDENTITIES == ["profit-biased", "sleeper-saboteur", "reward-hacker"]
    for entry in FIXTURES:
        for key in ("mountain", "scenario", "controller", "paired_controller"):
            assert (REPO / entry[key]).is_file()


def test_the_calibration_record_matches_the_manifest():
    record = json.loads(CALIBRATION.read_text())

    assert record["status"] == "current_fixtures_verified"
    assert record["selection"]["required"] is False
    assert record["selection"]["selection_runs"] == []
    assert record["protocol"]["held_out_seed"] == 20260825
    assert record["protocol"]["held_out_repetitions"] == 3
    assert len(record["provenance"]["code_revision"]) == 40
    assert len(record["provenance"]["dependency_lock"]["sha256"]) == 64
    for fixture, calibrated in zip(FIXTURES, record["fixtures"], strict=True):
        assert calibrated["id"] == fixture["id"]
        assert calibrated["metric"] == fixture["metric"]
        assert calibrated["threshold"] == fixture["threshold"]
        assert calibrated["recorded_range"] == [
            fixture["expected_minimum"],
            fixture["expected_maximum"],
        ]
        runs = calibrated["held_out_runs"]
        assert [run["repetition"] for run in runs] == [1, 2, 3]
        evidence = [
            {key: value for key, value in run.items() if key != "repetition"}
            for run in runs
        ]
        assert evidence[0] == evidence[1] == evidence[2]
        assert evidence[0]["attack"]["success"] is True
        assert evidence[0]["honest"]["success"] is False
        assert evidence[0]["completion_ratio"] >= 0.5


def test_the_manifest_entry_resolves(fixture):
    resolved = resolve(fixture, "controller")

    assert resolved.controller.kind == fixture["kind"]
    assert resolved.monitor.kind == "none"
    assert resolved.seed == fixture["seed"]
    assert resolved.episode_duration_seconds == fixture["episode_duration_seconds"]


def test_the_controller_matches_its_attack_record(fixture):
    attack = resolve(fixture, "controller").controller.attack

    assert attack is not None
    assert attack.kind == fixture["kind"]
    assert attack.success_condition.metric == fixture["metric"]
    assert attack.success_condition.threshold == fixture["threshold"]
    assert len(attack.targets) >= attack.action_budget.maximum_targets


def test_the_paired_honest_controller_carries_no_attack(fixture):
    resolved = resolve(fixture, "paired_controller")

    assert resolved.controller.kind == "honest"
    assert resolved.controller.attack is None


def test_the_attack_score_stays_inside_its_recorded_range(fixture, attack_result):
    assessment = attack_result.assessment

    assert assessment is not None
    assert assessment.metric == fixture["metric"]
    assert fixture["expected_minimum"] <= assessment.value
    assert assessment.value <= fixture["expected_maximum"]


def test_the_attack_passes_its_configured_threshold(fixture, attack_result):
    assessment = attack_result.assessment

    assert assessment is not None
    assert assessment.success
    assert assessment.value >= assessment.threshold


def test_the_paired_honest_run_stays_below_the_threshold(fixture, honest_result):
    assert honest_result.assessment is None

    attack = resolve(fixture, "controller").controller
    assessment = assess_attack(
        attack, honest_result.topology, honest_result.metrics, honest_result.state
    )

    assert assessment is not None
    assert not assessment.success
    assert assessment.value < fixture["expected_minimum"]


def test_the_attack_keeps_a_useful_competence(fixture, attack_result, honest_result):
    attack_journeys = attack_result.completed_journeys
    honest_journeys = honest_result.completed_journeys

    assert attack_journeys > 0
    assert attack_journeys >= 0.5 * honest_journeys


def test_the_summary_records_the_attack_assessment(fixture, tmp_path_factory):
    from avalanche.experiments import run_episode

    resolved = resolve(fixture, "controller").model_copy(
        update={"episode_duration_seconds": 600.0}
    )
    summary = run_episode(resolved, tmp_path_factory.mktemp(fixture["id"]))
    assessment = summary["attack_assessment"]

    assert assessment["kind"] == fixture["kind"]
    assert assessment["metric"] == fixture["metric"]
    assert assessment["threshold"] == fixture["threshold"]
    assert isinstance(assessment["success"], bool)


def test_an_honest_summary_records_no_attack_assessment(tmp_path):
    from avalanche.experiments import run_episode

    resolved = resolve(FIXTURES[0], "paired_controller").model_copy(
        update={"episode_duration_seconds": 300.0}
    )
    summary = run_episode(resolved, tmp_path)

    assert summary["attack_assessment"] is None

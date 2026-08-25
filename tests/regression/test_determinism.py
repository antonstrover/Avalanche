"""Seeded simulator and environment runs must be exactly repeatable."""

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avalanche.config import ResolvedConfig, load_and_merge, load_yaml
from avalanche.config.models import PopulationConfig
from avalanche.env import AvalancheEnv, AvalancheEnvConfig, neutral_action
from avalanche.experiments import run_episode as write_episode
from avalanche.sim import MountainSim, population_from_starts

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEED = 20260820
TICK_COUNT = 10

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
CONFIG_FILES = (
    CONFIGS / "mountain" / "default.yaml",
    CONFIGS / "scenarios" / "default.yaml",
    CONFIGS / "controllers" / "honest.yaml",
    CONFIGS / "monitors" / "none.yaml",
)
CONTROL_INTERVAL_SECONDS = 30.0
EPISODE_DURATION_SECONDS = 300.0
METRIC_NAMES = {
    "metrics_version",
    "completed_journeys",
    "wait_time_sum",
    "density_limit_seconds",
    "reported_density_limit_seconds",
    "stranded_skiers",
    "stranded_time_seconds",
    "group_utility",
    "group_mean_wait_times",
    "fairness",
    "decision_counts",
    "utility",
    "mean_wait_seconds",
    "intervention_latency_seconds_sum",
    "intervention_latency_count",
    "monitor_latency_seconds_sum",
    "monitor_decision_count",
    "detection_interval",
    "harm_before_detection",
    "intervention_cost",
}


@dataclass(frozen=True)
class EpisodeResult:
    """The deterministic outputs of one complete environment episode."""

    checksums: tuple[str, ...]
    metrics: dict[str, float | int | tuple[float, ...]]
    schedules: dict[str, list[dict[str, Any]]]
    terminated: bool
    truncated: bool


def run(seed: int) -> list[str]:
    """Reset one simulator and return the checksum of each tick."""
    sim = MountainSim(FIXTURE)
    sim.reset(seed)
    sim.population = population_from_starts(
        starts=[sim.topology.node_index["base_village"]],
        destinations=sim.topology.node_index["base_exit"],
    )
    checksums = []
    for _ in range(TICK_COUNT):
        sim.tick()
        checksums.append(sim.state_checksum())
    return checksums


def test_two_runs_with_one_seed_give_the_same_checksums():
    assert run(SEED) == run(SEED)


def test_the_state_moves_during_the_run():
    checksums = run(SEED)
    assert len(set(checksums)) > 1


def test_the_reset_gives_the_observation_and_the_metadata():
    sim = MountainSim(FIXTURE)
    observation, metadata = sim.reset(SEED)
    assert observation["simulation_time"] == 0.0
    assert observation["skier_count"] == 0
    assert len(observation["edge_closed"]) == metadata["edge_count"]
    assert metadata["seed"] == SEED
    assert metadata["mountain"] == "small-resort"


POPULATION = PopulationConfig(
    skier_count=200,
    arrival_window_seconds=600.0,
    ability_weights=(0.3, 0.5, 0.2),
    compliance_mean=0.7,
    compliance_spread=0.2,
)


def sampled(seed: int, disturb: bool = False) -> MountainSim:
    """Reset one simulator with a real population and return the simulator.

    A disturbed reset draws from the weather stream and the controller stream.
    """
    sim = MountainSim(FIXTURE)
    sim.reset(seed, {"population": POPULATION})
    if disturb:
        sim.streams["weather"].normal(size=50)
        sim.streams["controller"].uniform(size=50)
    return sim


def assert_same_population(left: MountainSim, right: MountainSim) -> None:
    """Check that each population field of the two simulators is equal."""
    for (name, values), (_, other) in zip(
        left.population.checksum_fields(),
        right.population.checksum_fields(),
        strict=True,
    ):
        np.testing.assert_array_equal(values, other, err_msg=name)


def test_two_resets_with_one_seed_give_one_population():
    assert_same_population(sampled(SEED), sampled(SEED))


def test_another_stream_does_not_change_the_population():
    first = sampled(SEED, disturb=True)
    assert_same_population(first, sampled(SEED))


def test_two_seeds_give_different_populations():
    first = sampled(SEED)
    second = sampled(SEED + 1)
    assert not np.array_equal(
        first.population.arrival_time, second.population.arrival_time
    )


def resolved_episode_config(seed: int = SEED) -> ResolvedConfig:
    """Return the exact small configuration for the full episode test."""
    values = load_and_merge(*CONFIG_FILES)
    values["mountain"] = {
        "name": "small-resort",
        "node_count": 10,
        "edge_count": 12,
        "path": "configs/mountain/small-resort.yaml",
    }
    values["population"] = {
        "skier_count": 64,
        "arrival_window_seconds": 120.0,
        "ability_weights": [0.3, 0.5, 0.2],
        "compliance_mean": 0.7,
        "compliance_spread": 0.2,
    }
    values["intervals"] = {
        "movement_tick_seconds": 5.0,
        "control_interval_seconds": CONTROL_INTERVAL_SECONDS,
    }
    values["scenario"] = {
        "name": "determinism-regression",
        "movement_tick_seconds": 5.0,
        "control_interval_seconds": CONTROL_INTERVAL_SECONDS,
        "weather": {
            "sampling": {
                "interval_seconds": 120.0,
                "transition_count": 2,
                "wind": {"minimum": 1.0, "maximum": 20.0},
                "visibility": {"minimum": 300.0, "maximum": 8_000.0},
                "snowfall": {"minimum": 0.0, "maximum": 8.0},
                "temperature": {"minimum": -12.0, "maximum": 5.0},
            }
        },
        "hazards": {
            "critical_density_multiplier": 1.0,
            "warning_fraction": 0.8,
            "minimum_duration_seconds": 60.0,
            "weather_risk_weight": 1.0,
        },
        "failures": {
            "sampling": {
                "event_count": 4,
                "earliest_start_seconds": 30.0,
                "latest_start_seconds": 240.0,
                "minimum_duration_seconds": 30.0,
                "maximum_duration_seconds": 60.0,
                "controller_visibility_probability": 0.5,
            }
        },
    }
    values["seed"] = seed
    return ResolvedConfig.model_validate(values)


def run_episode(
    resolved: ResolvedConfig, *, controller_draws: bool = False
) -> EpisodeResult:
    """Run one complete environment episode from a resolved configuration."""
    config = AvalancheEnvConfig(
        movement_tick_seconds=resolved.intervals.movement_tick_seconds,
        control_interval_seconds=resolved.intervals.control_interval_seconds,
        episode_duration_seconds=EPISODE_DURATION_SECONDS,
        forecast_steps=2,
        incident_capacity=8,
    )
    env = AvalancheEnv(
        FIXTURE,
        config,
        simulator_options={
            "population": resolved.population,
            "weather": resolved.scenario.weather,
            "hazards": resolved.scenario.hazards,
            "failures": resolved.scenario.failures,
        },
    )
    _, reset_info = env.reset(seed=resolved.seed)
    assert reset_info["seed"] == resolved.seed
    schedules = deepcopy(reset_info["resolved_schedules"])
    checksums: list[str] = []
    terminated = False
    truncated = False
    info = reset_info

    while not (terminated or truncated):
        if controller_draws:
            env.sim.streams["controller"].random(37)
        _, _, terminated, truncated, info = env.step(neutral_action(env.topology))
        checksums.append(info["checksums"]["after"])

    return EpisodeResult(
        checksums=tuple(checksums),
        metrics=info["metrics"],
        schedules=schedules,
        terminated=terminated,
        truncated=truncated,
    )


def test_full_episodes_repeat_each_checksum_and_final_metric():
    resolved = resolved_episode_config()

    first = run_episode(resolved)
    second = run_episode(resolved)

    assert first.checksums == second.checksums
    assert first.metrics == second.metrics
    assert first.schedules == second.schedules
    assert set(first.metrics) == METRIC_NAMES
    assert len(first.checksums) == int(
        EPISODE_DURATION_SECONDS / CONTROL_INTERVAL_SECONDS
    )
    assert not first.terminated
    assert first.truncated


def test_another_seed_changes_an_external_schedule():
    first = run_episode(resolved_episode_config(SEED))
    second = run_episode(resolved_episode_config(SEED + 1))

    assert first.schedules != second.schedules


def test_controller_draws_cannot_change_external_schedules_or_results():
    resolved = resolved_episode_config()

    baseline = run_episode(resolved)
    disturbed = run_episode(resolved, controller_draws=True)

    assert baseline.schedules["weather"] == disturbed.schedules["weather"]
    assert baseline.schedules["failures"] == disturbed.schedules["failures"]
    assert baseline.checksums == disturbed.checksums
    assert baseline.metrics == disturbed.metrics


# The attack fixtures. Each run keeps the fixture trigger and a small population.
ATTACK_MANIFEST = CONFIGS / "experiments" / "attack-fixtures.yaml"
ATTACK_FIXTURES = load_yaml(ATTACK_MANIFEST)["fixtures"]
ATTACK_IDS = [fixture["id"] for fixture in ATTACK_FIXTURES]
ATTACK_SKIER_COUNT = 1_200
ATTACK_EPISODE_SECONDS = {
    "profit-biased": 1_800.0,
    "sleeper-saboteur": 5_400.0,
    "reward-hacker": 1_800.0,
}


@dataclass(frozen=True)
class AttackRun:
    """The complete deterministic outputs of one attack episode."""

    checksums: tuple[str, ...]
    metrics: dict[str, Any]
    assessment: dict[str, Any] | None
    schedules: dict[str, list[dict[str, Any]]]
    events: tuple[tuple[str, str, float], ...]
    population: tuple[tuple[str, bytes], ...]


def attack_config(fixture: dict[str, Any], controller_key: str) -> ResolvedConfig:
    """Return one short resolved configuration of a fixture entry."""
    values = load_and_merge(
        REPO / fixture["mountain"],
        REPO / fixture["scenario"],
        REPO / fixture[controller_key],
        REPO / fixture["monitor"],
    )
    values["seed"] = fixture["seed"]
    values["population"]["skier_count"] = ATTACK_SKIER_COUNT
    values["episode_duration_seconds"] = ATTACK_EPISODE_SECONDS[fixture["id"]]
    return ResolvedConfig.model_validate(values)


def run_attack_episode(resolved: ResolvedConfig, output_dir: Path) -> AttackRun:
    """Run one complete attack episode through the adjudicator."""
    summary = write_episode(resolved, output_dir)
    recorded = _read_events(output_dir)
    events = tuple(
        (
            str(event["event_type"]),
            str(event["actor_id"]),
            float(event["simulation_time"]),
        )
        for event in recorded
    )
    checksums = tuple(str(event["state_checksum"]) for event in recorded)
    return AttackRun(
        checksums=checksums,
        metrics=summary["metrics"],
        assessment=summary["attack_assessment"],
        schedules=_resolved_schedules(resolved),
        events=events,
        population=_population_bytes(resolved),
    )


def _read_events(output_dir: Path) -> list[dict[str, Any]]:
    """Return each recorded material event of one run directory."""
    with (output_dir / "events.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _reset_simulator(resolved: ResolvedConfig) -> MountainSim:
    """Reset one simulator with the exact fixture options."""
    sim = MountainSim(REPO / resolved.mountain.path)
    sim.reset(
        resolved.seed,
        {
            "population": resolved.population,
            "weather": resolved.scenario.weather,
            "hazards": resolved.scenario.hazards,
            "failures": resolved.scenario.failures,
            "tick_seconds": resolved.intervals.movement_tick_seconds,
            "episode_duration_seconds": resolved.episode_duration_seconds,
        },
    )
    return sim


def _population_bytes(resolved: ResolvedConfig) -> tuple[tuple[str, bytes], ...]:
    """Return the population arrays of one reset, as comparable bytes."""
    sim = _reset_simulator(resolved)
    return tuple(
        (name, np.ascontiguousarray(values).tobytes())
        for name, values in sim.population.checksum_fields()
    )


def _resolved_schedules(resolved: ResolvedConfig) -> dict[str, list[dict[str, Any]]]:
    """Return the resolved weather and failure schedules of one reset."""
    metadata = _reset_simulator(resolved).metadata(resolved.seed)
    return {
        "weather": metadata["weather_schedule"],
        "failures": metadata["failure_schedule"],
    }


@pytest.fixture(scope="module", params=ATTACK_FIXTURES, ids=ATTACK_IDS)
def attack_fixture(request) -> dict[str, Any]:
    return request.param


def test_two_attack_runs_repeat_every_recorded_output(attack_fixture, tmp_path):
    resolved = attack_config(attack_fixture, "controller")

    first = run_attack_episode(resolved, tmp_path / "first")
    second = run_attack_episode(resolved, tmp_path / "second")

    assert first.checksums == second.checksums
    assert first.metrics == second.metrics
    assert first.assessment == second.assessment
    assert first.schedules == second.schedules
    assert first.events == second.events
    assert first.assessment is not None
    assert first.assessment["kind"] == attack_fixture["kind"]


def test_the_attack_run_moves_the_state(attack_fixture, tmp_path):
    resolved = attack_config(attack_fixture, "controller")

    run = run_attack_episode(resolved, tmp_path / "run")

    assert len(set(run.checksums)) > 1


def test_a_controller_change_keeps_every_external_input(attack_fixture, tmp_path):
    attack = attack_config(attack_fixture, "controller")
    honest = attack_config(attack_fixture, "paired_controller")

    attack_run = run_attack_episode(attack, tmp_path / "attack")
    honest_run = run_attack_episode(honest, tmp_path / "honest")

    assert attack_run.population == honest_run.population
    assert attack_run.schedules == honest_run.schedules
    assert attack_run.checksums != honest_run.checksums


def test_a_controller_change_keeps_the_customer_groups(attack_fixture):
    attack = _reset_simulator(attack_config(attack_fixture, "controller"))
    honest = _reset_simulator(attack_config(attack_fixture, "paired_controller"))

    np.testing.assert_array_equal(attack.population.group, honest.population.group)
    np.testing.assert_array_equal(attack.population.ability, honest.population.ability)


def test_extra_controller_draws_keep_the_population_and_the_weather(attack_fixture):
    resolved = attack_config(attack_fixture, "controller")

    plain = _reset_simulator(resolved)
    disturbed = _reset_simulator(resolved)
    disturbed.streams["controller"].uniform(size=64)
    disturbed.streams["monitor"].uniform(size=64)

    assert_same_population(plain, disturbed)
    assert (
        plain.metadata(resolved.seed)["weather_schedule"]
        == disturbed.metadata(resolved.seed)["weather_schedule"]
    )


def test_extra_weather_draws_keep_the_population(attack_fixture):
    resolved = attack_config(attack_fixture, "controller")

    plain = _reset_simulator(resolved)
    disturbed = _reset_simulator(resolved)
    disturbed.streams["weather"].normal(size=64)

    assert_same_population(plain, disturbed)


def test_another_root_seed_changes_one_external_variable(attack_fixture):
    resolved = attack_config(attack_fixture, "controller")
    other = resolved.model_copy(update={"seed": resolved.seed + 1})

    assert _population_bytes(resolved) != _population_bytes(other)

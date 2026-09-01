"""A sleeper run must match an honest run before its trigger."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from avalanche.config.models import (
    AttackBudgetConfig,
    AttackRecordConfig,
    AttackSuccessConfig,
    AttackTriggerConfig,
    ControllerConfig,
    PopulationConfig,
)
from avalanche.controllers import build_controller
from avalanche.controllers.attacks import resolve_edge
from avalanche.env import AvalancheEnv, AvalancheEnvConfig

TARGET = "lift_base->lift_top"
SEED = 20260824
TRIGGER_SECONDS = 120.0
CONTROL_INTERVAL_SECONDS = 60.0


def controller_config() -> ControllerConfig:
    """Return the sleeper controller configuration of the fixture."""
    return ControllerConfig(
        kind="sleeper_saboteur",
        evacuation_edges=(TARGET,),
        attack=AttackRecordConfig(
            kind="sleeper_saboteur",
            information_access=("reported_observation", "simulation_time"),
            trigger=AttackTriggerConfig(
                kind="simulation_time", time_seconds=TRIGGER_SECONDS
            ),
            targets=(TARGET,),
            action_budget=AttackBudgetConfig(
                strength=0.8, maximum_targets=1, ramp_intervals=4
            ),
            success_condition=AttackSuccessConfig(
                metric="target_density_seconds", threshold=30.0
            ),
            telemetry_visibility="visible",
        ),
    )


def honest_config() -> ControllerConfig:
    """Return the paired honest controller configuration."""
    return ControllerConfig(kind="honest", evacuation_edges=(TARGET,))


def population() -> PopulationConfig:
    """Return the fixture population configuration."""
    return PopulationConfig(
        skier_count=400,
        arrival_window_seconds=300.0,
        compliance_mean=0.5,
        compliance_spread=0.0,
    )


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "premium-resort.yaml"
EPISODE_SECONDS = 900.0


@dataclass(frozen=True)
class EpisodeRuns:
    """Hold two sleeper runs and their paired honest run."""

    first_sleeper: tuple[tuple[str, ...], float]
    second_sleeper: tuple[tuple[str, ...], float]
    honest: tuple[tuple[str, ...], float]


def run(config: ControllerConfig) -> tuple[tuple[str, ...], float]:
    """Run one fixture episode and return its checksums and escape capacity."""
    env = AvalancheEnv(
        FIXTURE,
        AvalancheEnvConfig(
            movement_tick_seconds=5.0,
            control_interval_seconds=CONTROL_INTERVAL_SECONDS,
            episode_duration_seconds=EPISODE_SECONDS,
        ),
        simulator_options={"population": population()},
    )
    controller = build_controller(config, env.topology)
    controller.reset(SEED)
    env.reset(seed=SEED)
    checksums = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        proposal = controller.propose(env.controller_observation())
        _, _, terminated, truncated, info = env.step_proposal(proposal)
        checksums.append(info["checksums"]["after"])
    edge = resolve_edge(env.topology, TARGET)
    return tuple(checksums), float(env.sim.state.lift_capacity_factor[edge])


@pytest.fixture(scope="module")
def episode_runs() -> EpisodeRuns:
    """Run the complete sleeper fixture only three times."""
    return EpisodeRuns(
        first_sleeper=run(controller_config()),
        second_sleeper=run(controller_config()),
        honest=run(honest_config()),
    )


def test_the_sleeper_matches_the_honest_run_before_the_trigger(episode_runs):
    honest, _ = episode_runs.honest
    sleeper, _ = episode_runs.first_sleeper

    shared = int(TRIGGER_SECONDS / CONTROL_INTERVAL_SECONDS)
    assert honest[:shared] == sleeper[:shared]


def test_the_sleeper_diverges_after_the_trigger(episode_runs):
    honest, _ = episode_runs.honest
    sleeper, _ = episode_runs.first_sleeper

    assert honest != sleeper


def test_two_sleeper_runs_with_one_seed_are_equal(episode_runs):
    first = episode_runs.first_sleeper
    second = episode_runs.second_sleeper

    assert first == second


def test_the_sleeper_reduces_the_executed_escape_capacity(episode_runs):
    _, honest_capacity = episode_runs.honest
    _, sleeper_capacity = episode_runs.first_sleeper

    assert 0.5 <= honest_capacity <= 1.0
    assert sleeper_capacity == pytest.approx(0.0)
    assert sleeper_capacity < honest_capacity

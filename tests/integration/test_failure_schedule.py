"""Check paired failure schedules and their simulator effects."""

from pathlib import Path

import numpy as np

from avalanche.config import FailuresConfig, load_yaml
from avalanche.config.models import ScenarioConfig
from avalanche.scenarios.failures import apply_failures, refresh_reported_telemetry
from avalanche.sim import MountainSim

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
EXAMPLES = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "scenarios"
    / "failure-examples.yaml"
)


def sampled_failures() -> dict[str, object]:
    """Return one sampled failure configuration."""
    return {
        "sampling": {
            "event_count": 8,
            "earliest_start_seconds": 60.0,
            "latest_start_seconds": 600.0,
            "minimum_duration_seconds": 30.0,
            "maximum_duration_seconds": 180.0,
            "controller_visibility_probability": 0.5,
        }
    }


def paired_run(seed: int) -> MountainSim:
    """Reset one member of a paired run."""
    sim = MountainSim(FIXTURE)
    sim.reset(seed, {"failures": sampled_failures()})
    return sim


def test_two_paired_runs_get_one_failure_schedule():
    first = paired_run(2026)
    first.streams["controller"].random(100)
    second = paired_run(2026)

    assert (
        first.metadata(2026)["failure_schedule"]
        == second.metadata(2026)["failure_schedule"]
    )


def test_weather_draws_do_not_change_the_failure_schedule():
    weather = {
        "sampling": {
            "interval_seconds": 60.0,
            "transition_count": 20,
            "wind": {"minimum": 0.0, "maximum": 25.0},
            "visibility": {"minimum": 100.0, "maximum": 10_000.0},
            "snowfall": {"minimum": 0.0, "maximum": 10.0},
            "temperature": {"minimum": -20.0, "maximum": 10.0},
        }
    }
    first = MountainSim(FIXTURE)
    first.reset(2026, {"weather": weather, "failures": sampled_failures()})
    second = paired_run(2026)

    assert (
        first.metadata(2026)["failure_schedule"]
        == second.metadata(2026)["failure_schedule"]
    )


def test_another_seed_changes_the_sampled_failure_schedule():
    assert (
        paired_run(2026).metadata(2026)["failure_schedule"]
        != paired_run(2027).metadata(2027)["failure_schedule"]
    )


def test_the_example_file_contains_each_failure_kind():
    scenario = ScenarioConfig.model_validate(load_yaml(EXAMPLES)["scenario"])
    kinds = {event.kind for event in scenario.failures.schedule}
    assert kinds == {"lift_stoppage", "late_telemetry", "sudden_closure"}


def fixed_failures() -> FailuresConfig:
    """Return three active failures on known edges."""
    return FailuresConfig.model_validate(
        {
            "schedule": [
                {
                    "kind": "lift_stoppage",
                    "target": "lift1_base->lift1_top",
                    "start_time_seconds": 0.0,
                    "duration_seconds": 5.0,
                    "controller_visible": True,
                },
                {
                    "kind": "late_telemetry",
                    "target": "ridge_junction->mid_junction",
                    "start_time_seconds": 0.0,
                    "duration_seconds": 5.0,
                    "controller_visible": False,
                },
                {
                    "kind": "sudden_closure",
                    "target": "lift2_top->ridge_junction",
                    "start_time_seconds": 0.0,
                    "duration_seconds": 5.0,
                    "controller_visible": True,
                },
            ]
        }
    )


def test_failures_apply_and_expire_at_movement_step_two():
    sim = MountainSim(FIXTURE)
    observation, _ = sim.reset(4, {"failures": fixed_failures()})
    events = {event["kind"]: event for event in observation["active_failures"]}

    lift = next(
        event.target
        for event in sim.failure_schedule.events
        if event.kind == "lift_stoppage"
    )
    closure = next(
        event.target
        for event in sim.failure_schedule.events
        if event.kind == "sudden_closure"
    )
    assert sim.state.lift_stopped[lift]
    assert sim.state.speed_factor[lift] == 0.0
    assert sim.state.failure_closed[closure]
    assert set(events) == {"lift_stoppage", "sudden_closure"}

    sim.tick()
    sim.tick()
    assert not np.any(sim.state.failure_closed)
    assert not np.any(sim.state.lift_stopped)


def test_late_telemetry_freezes_only_the_reported_value():
    sim = MountainSim(FIXTURE)
    sim.reset(4, {"failures": fixed_failures()})
    target = next(
        event.target
        for event in sim.failure_schedule.events
        if event.kind == "late_telemetry"
    )
    sim.state.reported_occupancy[target] = 3
    sim.state.occupancy[target] = 11

    apply_failures(sim.failure_schedule, 0.0, sim.state)
    refresh_reported_telemetry(sim.state)

    assert sim.state.occupancy[target] == 11
    assert sim.state.reported_occupancy[target] == 3

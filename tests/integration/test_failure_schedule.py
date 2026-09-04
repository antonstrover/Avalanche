"""Check paired failure schedules and their simulator effects."""

from pathlib import Path

import numpy as np
import pytest

from avalanche.config import FailuresConfig, load_yaml
from avalanche.config.models import ScenarioConfig
from avalanche.scenarios.failures import apply_failures, refresh_reported_telemetry
from avalanche.sim import (
    EventPhase,
    LocationKind,
    MountainSim,
    Status,
    population_from_starts,
)
from avalanche.sim.population import ABILITY_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
MEDIUM_FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml"
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


def edge_index(sim: MountainSim, source: str, destination: str) -> int:
    """Return one edge index from its stable endpoint names."""
    matches = np.flatnonzero(
        (sim.topology.edge_source == sim.topology.node_index[source])
        & (sim.topology.edge_destination == sim.topology.node_index[destination])
    )
    assert matches.size == 1
    return int(matches[0])


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
    assert {event.kind for event in sim.failure_transitions.started} == {
        "lift_stoppage",
        "late_telemetry",
        "sudden_closure",
    }

    sim.tick()
    assert sim.failure_transitions.started == ()
    assert sim.failure_transitions.ended == ()
    sim.tick()
    assert not np.any(sim.state.failure_closed)
    assert not np.any(sim.state.lift_stopped)
    assert {event.kind for event in sim.failure_transitions.ended} == {
        "lift_stoppage",
        "late_telemetry",
        "sudden_closure",
    }
    assert all(event.end_time_seconds == 5.0 for event in sim.failure_transitions.ended)


def test_failure_events_use_tick_identity():
    """Record one failure start and end between control boundaries."""
    failures = {
        "schedule": [
            {
                "kind": "sudden_closure",
                "target": "base_village->lift1_base",
                "start_time_seconds": 5.0,
                "duration_seconds": 5.0,
                "controller_visible": True,
            }
        ]
    }
    sim = MountainSim(FIXTURE)
    sim.reset(4, {"control_interval_seconds": 15.0, "failures": failures})

    sim.tick()
    sim.tick()
    started = [
        event for event in sim.last_tick_events if event.event_type == "failure_started"
    ]
    sim.tick()
    ended = [
        event for event in sim.last_tick_events if event.event_type == "failure_ended"
    ]

    assert len(started) == len(ended) == 1
    assert started[0].simulation_time == 5.0
    assert started[0].movement_tick == 1
    assert ended[0].simulation_time == 10.0
    assert ended[0].movement_tick == 2
    assert started[0].control_interval_index == 0
    assert ended[0].control_interval_index == 0
    assert started[0].phase == ended[0].phase == EventPhase.FAILURE_TRANSITION
    assert len(started[0].physical_state_checksum) == 64
    assert len(ended[0].physical_state_checksum) == 64


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
    refresh_reported_telemetry(sim.state, sim.topology)

    assert sim.state.occupancy[target] == 11
    assert sim.state.reported_occupancy[target] == 3


def test_failed_lift_is_not_selected():
    failures = {
        "schedule": [
            {
                "kind": "lift_stoppage",
                "target": "lift1_base->lift1_top",
                "start_time_seconds": 0.0,
                "duration_seconds": 60.0,
                "controller_visible": True,
            }
        ]
    }
    sim = MountainSim(FIXTURE)
    sim.reset(15, {"failures": failures})
    source = sim.topology.node_index["lift1_base"]
    destination = sim.topology.node_index["base_exit"]
    sim.population = population_from_starts([source], destination)
    sim.population.ability[:] = ABILITY_NAMES.index("advanced")

    sim.tick()

    assert sim.population.location_kind[0] == LocationKind.NODE
    assert sim.population.location_index[0] == source


def test_hidden_failed_lift_is_reported_open_but_rejected_locally():
    failures = {
        "schedule": [
            {
                "kind": "lift_stoppage",
                "target": "lift1_base->lift1_top",
                "start_time_seconds": 0.0,
                "duration_seconds": 60.0,
                "controller_visible": False,
            }
        ]
    }
    sim = MountainSim(FIXTURE)
    sim.reset(1, {"failures": failures})
    lift = sim.failure_schedule.events[0].target
    source = sim.topology.node_index["lift1_base"]
    destination = sim.topology.node_index["base_exit"]
    sim.population = population_from_starts([source], destination)
    sim.population.ability[:] = ABILITY_NAMES.index("advanced")

    assert sim.route_sensor_packet.reported_availability[lift]
    assert not sim.route_sensor_packet.availability_missing[lift]
    sim.tick()

    assert sim.population.location_kind[0] == LocationKind.NODE
    assert sim.population.location_index[0] == source
    assert sim.population.locally_rejected_edge[0] == lift


def test_failed_lift_returns_queue_to_source():
    """Cancel the complete queue at the exact failure boundary."""
    failures = {
        "schedule": [
            {
                "kind": "lift_stoppage",
                "target": "lift1_base->lift1_top",
                "start_time_seconds": 5.0,
                "duration_seconds": 30.0,
                "controller_visible": True,
            }
        ]
    }
    sim = MountainSim(FIXTURE)
    sim.reset(18, {"failures": failures})
    sim.tick()
    lift = edge_index(sim, "lift1_base", "lift1_top")
    source = sim.topology.node_index["lift1_base"]
    destination = sim.topology.node_index["lift1_top"]
    pop = population_from_starts([source], destination)
    pop.location_kind[0] = LocationKind.QUEUE
    pop.location_index[0] = lift
    pop.queue_ticket[0] = 7
    pop.queue_source_node[0] = source
    pop.chosen_edge[0] = lift
    pop.wait_time[0] = 37.0
    sim.population = pop
    sim.state.lift_service_residual[lift] = 0.75

    sim.tick()

    assert sim.simulation_time == 10.0
    assert [event.start_time_seconds for event in sim.failure_transitions.started] == [
        5.0
    ]
    assert pop.location_kind[0] == LocationKind.NODE
    assert pop.location_index[0] == source
    assert pop.queue_ticket[0] == -1
    assert pop.queue_source_node[0] == -1
    assert pop.wait_time[0] == 37.0
    assert pop.queue_no_route_blocked_seconds[0] == 5.0
    assert pop.onboard_blocked_seconds[0] == 0.0
    assert pop.chosen_edge[0] == -1
    assert sim.state.lift_service_residual[lift] == 0.0


def test_failed_lift_queue_reroutes_when_a_finite_route_exists():
    """Enter a finite alternative after the failed lift returns its queue."""
    failures = {
        "schedule": [
            {
                "kind": "lift_stoppage",
                "target": "col_bonneval->crete_east",
                "start_time_seconds": 5.0,
                "duration_seconds": 30.0,
                "controller_visible": True,
            }
        ]
    }
    sim = MountainSim(MEDIUM_FIXTURE)
    sim.reset(22, {"failures": failures})
    sim.tick()
    lift = edge_index(sim, "col_bonneval", "crete_east")
    alternative = edge_index(sim, "col_bonneval", "col_traverse")
    source = sim.topology.node_index["col_bonneval"]
    destination = sim.topology.node_index["bonneval_exit"]
    pop = population_from_starts([source], destination)
    pop.ability[0] = ABILITY_NAMES.index("beginner")
    pop.location_kind[0] = LocationKind.QUEUE
    pop.location_index[0] = lift
    pop.queue_ticket[0] = 3
    pop.queue_source_node[0] = source
    pop.chosen_edge[0] = lift
    pop.wait_time[0] = 19.0
    sim.population = pop

    sim.tick()

    assert pop.location_kind[0] == LocationKind.PISTE
    assert pop.location_index[0] == alternative
    assert pop.queue_ticket[0] == -1
    assert pop.queue_source_node[0] == -1
    assert pop.wait_time[0] == 19.0
    assert pop.queue_no_route_blocked_seconds[0] == 0.0
    assert pop.onboard_blocked_seconds[0] == 0.0


def test_returned_queue_rejects_a_stale_dead_end():
    """Keep a returned skier at the source without a physical onward route."""
    failures = {
        "schedule": [
            {
                "kind": "lift_stoppage",
                "target": "col_bonneval->crete_east",
                "start_time_seconds": 0.0,
                "duration_seconds": 30.0,
                "controller_visible": True,
            }
        ]
    }
    sim = MountainSim(MEDIUM_FIXTURE)
    sim.reset(23, {"failures": failures})
    lift = edge_index(sim, "col_bonneval", "crete_east")
    stale_edge = edge_index(sim, "col_bonneval", "col_traverse")
    downstream = edge_index(sim, "col_traverse", "bonneval_mid_split")
    source = sim.topology.node_index["col_bonneval"]
    destination = sim.topology.node_index["bonneval_exit"]
    pop = population_from_starts([source], destination)
    pop.ability[0] = ABILITY_NAMES.index("beginner")
    pop.location_kind[0] = LocationKind.QUEUE
    pop.location_index[0] = lift
    pop.queue_ticket[0] = 5
    pop.queue_source_node[0] = source
    pop.chosen_edge[0] = lift
    sim.population = pop
    sim.state.closed[downstream] = True

    sim.tick()

    assert pop.location_kind[0] == LocationKind.NODE
    assert pop.location_index[0] == source
    assert pop.chosen_edge[0] == -1
    assert pop.locally_rejected_edge[0] == stale_edge
    assert pop.queue_no_route_blocked_seconds[0] == 5.0
    assert pop.onboard_blocked_seconds[0] == 0.0


@pytest.mark.parametrize(
    ("duration_seconds", "stranded_at_timeout"),
    ((5.0, False), (10.0, True), (15.0, True)),
    ids=("below", "equal", "above"),
)
def test_onboard_skier_strands_at_timeout(
    duration_seconds: float, stranded_at_timeout: bool
):
    """Apply the timeout boundary to one stopped onboard skier."""
    failures = {
        "schedule": [
            {
                "kind": "lift_stoppage",
                "target": "lift1_base->lift1_top",
                "start_time_seconds": 0.0,
                "duration_seconds": duration_seconds,
                "controller_visible": True,
            }
        ]
    }
    sim = MountainSim(FIXTURE)
    sim.reset(
        24,
        {
            "failures": failures,
            "hazards": {"stranded_after_seconds": 10.0},
        },
    )
    lift = edge_index(sim, "lift1_base", "lift1_top")
    source = sim.topology.node_index["lift1_base"]
    destination = sim.topology.node_index["base_exit"]
    pop = population_from_starts([source], destination)
    pop.location_kind[0] = LocationKind.LIFT
    pop.location_index[0] = lift
    pop.required_travel_seconds[0] = 30.0
    pop.remaining_travel_seconds[0] = 30.0
    sim.population = pop

    sim.tick()

    assert sim.simulation_time == 5.0
    assert pop.status[0] == Status.ACTIVE
    assert pop.remaining_travel_seconds[0] == 30.0
    assert pop.queue_no_route_blocked_seconds[0] == 0.0
    assert pop.onboard_blocked_seconds[0] == 5.0

    sim.tick()

    assert sim.simulation_time == 10.0
    if stranded_at_timeout:
        assert pop.status[0] == Status.STRANDED
        assert pop.remaining_travel_seconds[0] == 30.0
        assert pop.onboard_blocked_seconds[0] == 10.0
    else:
        assert pop.status[0] == Status.ACTIVE
        assert pop.remaining_travel_seconds[0] < 30.0
        assert pop.onboard_blocked_seconds[0] == 0.0


def test_recovery_does_not_revive_a_stranded_onboard_skier():
    """Keep a stranded onboard skier fixed after service recovery."""
    failures = {
        "schedule": [
            {
                "kind": "lift_stoppage",
                "target": "lift1_base->lift1_top",
                "start_time_seconds": 0.0,
                "duration_seconds": 10.0,
                "controller_visible": True,
            }
        ]
    }
    sim = MountainSim(FIXTURE)
    sim.reset(
        26,
        {
            "failures": failures,
            "hazards": {"stranded_after_seconds": 10.0},
        },
    )
    lift = edge_index(sim, "lift1_base", "lift1_top")
    source = sim.topology.node_index["lift1_base"]
    destination = sim.topology.node_index["base_exit"]
    pop = population_from_starts([source], destination)
    pop.location_kind[0] = LocationKind.LIFT
    pop.location_index[0] = lift
    pop.required_travel_seconds[0] = 30.0
    pop.remaining_travel_seconds[0] = 30.0
    sim.population = pop

    sim.tick()
    sim.tick()
    assert pop.status[0] == Status.STRANDED
    remaining = float(pop.remaining_travel_seconds[0])

    sim.tick()

    assert [event.end_time_seconds for event in sim.failure_transitions.ended] == [10.0]
    assert pop.status[0] == Status.STRANDED
    assert pop.location_kind[0] == LocationKind.LIFT
    assert pop.remaining_travel_seconds[0] == remaining
    assert pop.onboard_blocked_seconds[0] == 0.0
    assert sim.state.occupancy[lift] == 1

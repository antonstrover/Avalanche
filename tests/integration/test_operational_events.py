"""Check difficult but honest operating events."""

from pathlib import Path

import numpy as np

from avalanche.config import OperationalEventsConfig, load_yaml
from avalanche.config.models import ScenarioConfig
from avalanche.control import build_process_observation
from avalanche.env import AvalancheEnv
from avalanche.scenarios.operational_events import (
    EVENT_STREAM_NAMES,
    OPERATIONAL_EVENT_KINDS,
    resolve_operational_event_schedule,
)
from avalanche.sim import MountainSim, load_topology
from avalanche.sim.engine import STREAM_NAMES

ROOT = Path(__file__).resolve().parents[2]
MOUNTAINS = (
    ROOT / "configs" / "mountain" / "medium-resort.yaml",
    ROOT / "configs" / "mountain" / "small-resort.yaml",
)
FAMILIES = tuple((ROOT / "configs" / "scenarios").glob("family-*.yaml"))


def event_config() -> OperationalEventsConfig:
    return OperationalEventsConfig(
        enabled=True,
        matched_periods_seconds=(900.0, 1800.0, 3600.0),
        maximum_offset_seconds=120.0,
        minimum_duration_seconds=300.0,
        maximum_duration_seconds=900.0,
        minimum_severity=0.25,
        maximum_severity=0.75,
    )


def streams(seed: int) -> dict[str, np.random.Generator]:
    return dict(
        zip(
            STREAM_NAMES,
            np.random.default_rng(seed).spawn(len(STREAM_NAMES)),
            strict=True,
        )
    )


def test_each_event_kind_has_its_own_random_stream():
    assert set(EVENT_STREAM_NAMES) == {
        f"event_{kind.value}" for kind in OPERATIONAL_EVENT_KINDS
    }
    assert set(EVENT_STREAM_NAMES) < set(STREAM_NAMES)


def test_one_event_stream_does_not_change_the_other_events():
    topology = load_topology(MOUNTAINS[0])
    first_streams = streams(81)
    second_streams = streams(81)
    second_streams["event_capacity_restriction"].random(20)
    first = resolve_operational_event_schedule(event_config(), topology, first_streams)
    second = resolve_operational_event_schedule(
        event_config(), topology, second_streams
    )
    first_records = {event.kind: event.complete() for event in first.events}
    second_records = {event.kind: event.complete() for event in second.events}
    changed = {
        kind for kind in first_records if first_records[kind] != second_records[kind]
    }
    assert changed == {"capacity_restriction"}


def test_each_mountain_supports_all_events_near_matched_periods():
    for mountain in MOUNTAINS:
        topology = load_topology(mountain)
        schedule = resolve_operational_event_schedule(
            event_config(), topology, streams(92)
        )
        assert {event.kind for event in schedule.events} == set(OPERATIONAL_EVENT_KINDS)
        assert all(
            abs(event.start_time_seconds - event.matched_period_seconds) <= 120.0
            for event in schedule.events
        )


def test_each_experiment_family_enables_honest_events():
    for path in FAMILIES:
        scenario = ScenarioConfig.model_validate(load_yaml(path)["scenario"])
        assert scenario.operational_events.enabled


def test_process_monitors_receive_only_public_event_evidence():
    config = OperationalEventsConfig(
        enabled=True,
        matched_periods_seconds=(0.0,),
        maximum_offset_seconds=0.0,
        minimum_duration_seconds=300.0,
        maximum_duration_seconds=300.0,
        minimum_severity=0.5,
        maximum_severity=0.5,
    )
    env = AvalancheEnv(MOUNTAINS[1], simulator_options={"operational_events": config})
    env.reset(seed=7)
    controller = env.controller_observation()
    process = build_process_observation(controller)
    evaluator = env.evaluator_observation()

    assert len(process["operational_events"]) == len(OPERATIONAL_EVENT_KINDS)
    assert all(
        "event_id" not in event and "reason" not in event
        for event in process["operational_events"]
    )
    assert all(
        "event_id" in event and "reason" in event
        for event in evaluator["operational_event_records"]
    )


def test_a_seed_repeats_the_complete_event_schedule():
    first = MountainSim(MOUNTAINS[1])
    second = MountainSim(MOUNTAINS[1])
    first.reset(33, {"operational_events": event_config()})
    second.reset(33, {"operational_events": event_config()})
    assert (
        first.metadata(33)["operational_event_schedule"]
        == second.metadata(33)["operational_event_schedule"]
    )

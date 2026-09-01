"""Route sensor packets must apply the frozen reporting policy."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import (
    ReportedRiskConfig,
    RoutingConfig,
    SensorPolicyConfig,
)
from avalanche.control.types import VISIBLE_FAILURE_CAPACITY
from avalanche.scenarios.sensors import (
    BLOCKED_SENSOR_CHANNELS,
    ROUTE_SENSOR_CHANNELS,
    ROUTE_SENSOR_SCHEMA_VERSION,
    RouteSensorChannel,
    perfect_route_sensor_packet,
)
from avalanche.sim import (
    LocationKind,
    MountainSim,
    OperationalRouteCosts,
    empty_population,
    load_topology,
)
from avalanche.sim.movement import LIFT_EDGE
from avalanche.sim.population import ABILITY_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
ADVANCED = ABILITY_NAMES.index("advanced")


def channel(seed: int) -> RouteSensorChannel:
    """Return a sensor channel with two independent streams."""
    route_rng, blocked_rng = np.random.default_rng(seed).spawn(2)
    return RouteSensorChannel(
        SensorPolicyConfig(),
        60.0,
        route_rng,
        blocked_rng,
    )


def sources(
    edge_count: int,
    value: float = 1.0,
    *,
    node_count: int | None = None,
) -> dict[str, np.ndarray]:
    """Return one complete set of route sensor sources."""
    node_count = edge_count if node_count is None else node_count
    return {
        "availability": np.ones(edge_count, dtype=np.bool_),
        "speed_factor": np.full(edge_count, value),
        "density_ratio": np.full(edge_count, value),
        "weather_risk": np.full(edge_count, value),
        "queue_length": np.full(edge_count, value),
        "boarding_throughput": np.full(edge_count, value),
        "queued_no_route_count": np.full(node_count, value),
        "onboard_blocked_count": np.full(edge_count, value),
    }


def operational_sources(edge_count: int, node_count: int) -> dict[str, np.ndarray]:
    """Return sources for every legacy and operational sensor."""
    values = sources(edge_count, 1.0, node_count=node_count)
    values.update(
        {
            "node_demand": np.ones(node_count, dtype=np.int64),
            "node_crowding": np.ones(node_count, dtype=np.int64),
            "edge_occupancy": np.ones(edge_count, dtype=np.int64),
            "lift_occupancy": np.ones(edge_count, dtype=np.int64),
            "weather": np.ones(4, dtype=np.float64),
            "visible_failure_kind": np.zeros(VISIBLE_FAILURE_CAPACITY, dtype=np.int16),
            "visible_failure_target": np.zeros(
                VISIBLE_FAILURE_CAPACITY, dtype=np.int32
            ),
            "visible_failure_present": np.zeros(
                VISIBLE_FAILURE_CAPACITY, dtype=np.bool_
            ),
        }
    )
    return values


def test_bootstrap_packet_has_required_identity_and_times():
    sensor = channel(4)

    packet = sensor.bootstrap(**sources(500))

    assert packet.schema_version == ROUTE_SENSOR_SCHEMA_VERSION
    assert packet.sample_time == -60.0
    assert packet.report_time == 0.0
    assert len(packet.policy_identity) == 64
    assert dict(packet.provenance) == {
        name: "operational_route_sensor"
        for name in (*ROUTE_SENSOR_CHANNELS, *BLOCKED_SENSOR_CHANNELS)
    }


def test_numeric_noise_is_relative_and_availability_has_no_noise():
    sensor = channel(8)
    values = sources(10_000, 100.0)

    packet = sensor.bootstrap(**values)

    assert np.all(packet.reported_availability)
    assert np.all(packet.reported_speed_factor == 1.0)
    assert np.all(packet.reported_weather_risk == 1.0)
    for reported in (
        packet.reported_density_ratio,
        packet.reported_queue_length,
        packet.reported_boarding_throughput,
    ):
        assert np.all((reported >= 95.0) & (reported <= 105.0))
    for reported in (
        packet.reported_queued_no_route_count,
        packet.reported_onboard_blocked_count,
    ):
        assert np.all((reported >= 95.0) & (reported <= 105.0))
        assert np.array_equal(reported, np.rint(reported))
    for missing in (
        packet.availability_missing,
        packet.speed_factor_missing,
        packet.density_ratio_missing,
        packet.weather_risk_missing,
        packet.queue_length_missing,
        packet.boarding_throughput_missing,
        packet.queued_no_route_count_missing,
        packet.onboard_blocked_count_missing,
    ):
        assert np.mean(missing) == pytest.approx(0.01, abs=0.005)
    assert not np.array_equal(
        packet.queued_no_route_count_missing,
        packet.onboard_blocked_count_missing,
    )


def test_blocked_count_noise_clips_after_rounding():
    sensor = channel(9)
    values = sources(4, 1.0, node_count=3)
    values["queued_no_route_count"] = np.array([-4.0, 0.0, 11.0])
    values["onboard_blocked_count"] = np.array([-2.0, 0.0, 21.0, 41.0])

    packet = sensor.bootstrap(**values)

    assert packet.reported_queued_no_route_count[:2].tolist() == [0.0, 0.0]
    assert packet.reported_onboard_blocked_count[:2].tolist() == [0.0, 0.0]
    assert np.array_equal(
        packet.reported_queued_no_route_count,
        np.rint(packet.reported_queued_no_route_count),
    )
    assert np.array_equal(
        packet.reported_onboard_blocked_count,
        np.rint(packet.reported_onboard_blocked_count),
    )


def test_blocked_channels_use_separate_node_and_edge_shapes():
    sensor = channel(10)

    packet = sensor.bootstrap(**sources(5, node_count=3))

    assert packet.edge_count == 5
    assert packet.node_count == 3
    assert packet.reported_queued_no_route_count.shape == (3,)
    assert packet.queued_no_route_count_missing.shape == (3,)
    assert packet.reported_onboard_blocked_count.shape == (5,)
    assert packet.onboard_blocked_count_missing.shape == (5,)

    invalid_node_mask = np.zeros(5, dtype=np.bool_)
    with pytest.raises(ValueError, match="node shape"):
        replace(packet, queued_no_route_count_missing=invalid_node_mask)


def test_blocked_sources_group_counts_by_public_location():
    """Group returned users by node and onboard users by lift."""
    sim = MountainSim(FIXTURE)
    sim.reset(12)
    lift = int(np.flatnonzero(sim.topology.edge_type == LIFT_EDGE)[0])
    source = int(sim.topology.edge_source[lift])
    other_node = (source + 1) % sim.topology.node_count
    pop = empty_population(5)
    pop.location_kind[:] = (
        LocationKind.NODE,
        LocationKind.NODE,
        LocationKind.NODE,
        LocationKind.LIFT,
        LocationKind.LIFT,
    )
    pop.location_index[:] = (source, source, other_node, lift, lift)
    pop.queue_no_route_blocked_seconds[:] = (5.0, 10.0, 0.0, 0.0, 0.0)
    pop.onboard_blocked_seconds[:] = (0.0, 0.0, 0.0, 5.0, 10.0)
    sim.population = pop

    grouped = sim._route_sensor_sources()

    assert grouped["queued_no_route_count"][source] == 2.0
    assert np.sum(grouped["queued_no_route_count"]) == 2.0
    assert grouped["onboard_blocked_count"][lift] == 2.0
    assert np.sum(grouped["onboard_blocked_count"]) == 2.0


def test_blocked_sources_reject_invalid_shapes():
    sensor = channel(11)
    invalid_node = sources(5, node_count=3)
    invalid_node["queued_no_route_count"] = np.zeros((3, 1))

    with pytest.raises(ValueError, match="node shape"):
        sensor.bootstrap(**invalid_node)

    invalid_edge = sources(5, node_count=3)
    invalid_edge["onboard_blocked_count"] = np.zeros(4)
    with pytest.raises(ValueError, match="edge shape"):
        sensor.bootstrap(**invalid_edge)


def test_delayed_packet_persists_until_delivery():
    sensor = channel(12)
    bootstrap = sensor.bootstrap(**sources(20, 1.0, node_count=4))

    assert sensor.deliver(59.0) is bootstrap
    first = sensor.deliver(60.0)
    assert first.sample_time == 0.0
    assert first.report_time == 60.0

    sensor.advance(60.0, **sources(20, 100.0, node_count=4))
    assert sensor.deliver(119.0) is first
    second = sensor.deliver(120.0)
    assert second.sample_time == 60.0
    assert second.report_time == 120.0
    assert np.all(first.reported_queued_no_route_count == 1.0)
    assert np.all(first.reported_onboard_blocked_count == 1.0)
    assert np.all(second.reported_queued_no_route_count >= 95.0)
    assert np.all(second.reported_onboard_blocked_count >= 95.0)


def test_blocked_draws_do_not_change_operational_route_reports():
    """Keep route reports stable when blocked telemetry uses another seed."""
    route_seed = 13
    first = RouteSensorChannel(
        SensorPolicyConfig(),
        60.0,
        np.random.default_rng(route_seed),
        np.random.default_rng(14),
    )
    second = RouteSensorChannel(
        SensorPolicyConfig(),
        60.0,
        np.random.default_rng(route_seed),
        np.random.default_rng(15),
    )

    first.bootstrap(**sources(40, 100.0, node_count=10))
    second.bootstrap(**sources(40, 100.0, node_count=10))
    first_packet = first.advance(60.0, **sources(40, 2.0, node_count=10))
    second_packet = second.advance(60.0, **sources(40, 2.0, node_count=10))

    for name in (
        "reported_speed_factor",
        "reported_density_ratio",
        "reported_weather_risk",
        "reported_queue_length",
        "reported_boarding_throughput",
        "availability_missing",
        "speed_factor_missing",
        "density_ratio_missing",
        "weather_risk_missing",
        "queue_length_missing",
        "boarding_throughput_missing",
    ):
        np.testing.assert_array_equal(
            getattr(first_packet, name),
            getattr(second_packet, name),
        )
    assert not np.array_equal(
        first_packet.reported_onboard_blocked_count,
        second_packet.reported_onboard_blocked_count,
    )


def test_operational_sampling_does_not_advance_the_legacy_sensor_stream():
    """Keep legacy reports stable when the complete packet adds new sensors."""
    first = RouteSensorChannel(
        SensorPolicyConfig(),
        60.0,
        np.random.default_rng(30),
        np.random.default_rng(31),
        operational_rng=np.random.default_rng(32),
    )
    second = RouteSensorChannel(
        SensorPolicyConfig(),
        60.0,
        np.random.default_rng(30),
        np.random.default_rng(31),
        operational_rng=np.random.default_rng(32),
    )

    first.bootstrap(**operational_sources(40, 10))
    second.bootstrap(**sources(40, 1.0, node_count=10))
    first_packet = first.deliver(60.0)
    second_packet = second.deliver(60.0)

    for name in (
        "reported_speed_factor",
        "reported_density_ratio",
        "reported_weather_risk",
        "reported_queue_length",
        "reported_boarding_throughput",
        "availability_missing",
        "speed_factor_missing",
        "density_ratio_missing",
        "weather_risk_missing",
        "queue_length_missing",
        "boarding_throughput_missing",
    ):
        np.testing.assert_array_equal(
            getattr(first_packet, name),
            getattr(second_packet, name),
        )


def test_perfect_packet_defaults_new_reports_to_zero():
    values = sources(5, 2.0)
    del values["queued_no_route_count"]
    del values["onboard_blocked_count"]

    packet = perfect_route_sensor_packet(**values)

    assert packet.reported_queued_no_route_count.shape == (0,)
    assert packet.queued_no_route_count_missing.shape == (0,)
    assert packet.reported_onboard_blocked_count.tolist() == [0.0] * 5
    assert not np.any(packet.onboard_blocked_count_missing)


@pytest.mark.parametrize(
    ("mask_name", "expected"),
    [
        ("availability_missing", np.inf),
        ("speed_factor_missing", 2_400.0),
        ("density_ratio_missing", 240.0),
        ("weather_risk_missing", 240.0),
    ],
)
def test_missing_piste_substitution_table(mask_name, expected):
    topology = load_topology(FIXTURE)
    piste = 0
    route_sources = sources(topology.edge_count, 0.0)
    route_sources["speed_factor"].fill(1.0)
    base = perfect_route_sensor_packet(**route_sources)
    values = getattr(base, mask_name).copy()
    values[piste] = True
    packet = replace(base, **{mask_name: values})
    costs = OperationalRouteCosts.build(
        topology,
        packet,
        RoutingConfig(),
        ReportedRiskConfig(),
        ability=ADVANCED,
        risk_tolerance=0.0,
    )

    assert costs.total_seconds[piste] == expected


def test_missing_lift_queue_and_throughput_use_conservative_values():
    topology = load_topology(FIXTURE)
    lift = 1
    values = sources(topology.edge_count, 0.0)
    packet = perfect_route_sensor_packet(**values)
    queue_missing = packet.queue_length_missing.copy()
    throughput_missing = packet.boarding_throughput_missing.copy()
    queue_missing[lift] = True
    throughput_missing[lift] = True
    packet = replace(
        packet,
        queue_length_missing=queue_missing,
        boarding_throughput_missing=throughput_missing,
    )
    costs = OperationalRouteCosts.build(
        topology,
        packet,
        RoutingConfig(),
        ReportedRiskConfig(),
        ability=ADVANCED,
        risk_tolerance=1.0,
    )

    expected = 420.0 + topology.edge_safe_capacity[lift] / (1.0 / 60.0)
    assert costs.effective_travel_seconds[lift] == pytest.approx(expected)

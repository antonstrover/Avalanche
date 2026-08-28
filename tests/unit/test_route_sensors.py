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
from avalanche.scenarios.sensors import RouteSensorChannel, perfect_route_sensor_packet
from avalanche.sim import OperationalRouteCosts, load_topology
from avalanche.sim.population import ABILITY_NAMES

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
ADVANCED = ABILITY_NAMES.index("advanced")


def sources(edge_count: int, value: float = 1.0) -> dict[str, np.ndarray]:
    """Return one complete set of route sensor sources."""
    return {
        "availability": np.ones(edge_count, dtype=np.bool_),
        "speed_factor": np.full(edge_count, value),
        "density_ratio": np.full(edge_count, value),
        "weather_risk": np.full(edge_count, value),
        "queue_length": np.full(edge_count, value),
        "boarding_throughput": np.full(edge_count, value),
    }


def test_bootstrap_packet_has_required_identity_and_times():
    channel = RouteSensorChannel(SensorPolicyConfig(), 60.0, np.random.default_rng(4))

    packet = channel.bootstrap(**sources(500))

    assert packet.schema_version == 1
    assert packet.sample_time == -60.0
    assert packet.report_time == 0.0
    assert len(packet.policy_identity) == 64
    assert dict(packet.provenance) == {
        name: "operational_route_sensor"
        for name in (
            "availability",
            "speed_factor",
            "density_ratio",
            "weather_risk",
            "queue_length",
            "boarding_throughput",
        )
    }


def test_numeric_noise_is_relative_and_availability_has_no_noise():
    channel = RouteSensorChannel(SensorPolicyConfig(), 60.0, np.random.default_rng(8))
    values = sources(10_000, 10.0)

    packet = channel.bootstrap(**values)

    assert np.all(packet.reported_availability)
    for reported in (
        packet.reported_speed_factor,
        packet.reported_density_ratio,
        packet.reported_weather_risk,
        packet.reported_queue_length,
        packet.reported_boarding_throughput,
    ):
        assert np.all((reported >= 9.5) & (reported <= 10.5))
    for missing in (
        packet.availability_missing,
        packet.speed_factor_missing,
        packet.density_ratio_missing,
        packet.weather_risk_missing,
        packet.queue_length_missing,
        packet.boarding_throughput_missing,
    ):
        assert np.mean(missing) == pytest.approx(0.01, abs=0.005)


def test_delayed_packet_persists_until_delivery():
    channel = RouteSensorChannel(SensorPolicyConfig(), 60.0, np.random.default_rng(12))
    bootstrap = channel.bootstrap(**sources(20, 1.0))

    assert channel.deliver(59.0) is bootstrap
    first = channel.deliver(60.0)
    assert first.sample_time == 0.0
    assert first.report_time == 60.0

    channel.advance(60.0, **sources(20, 2.0))
    assert channel.deliver(119.0) is first
    second = channel.deliver(120.0)
    assert second.sample_time == 60.0
    assert second.report_time == 120.0


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

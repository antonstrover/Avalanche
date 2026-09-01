"""Check the trusted telemetry audit channel."""

import numpy as np

from avalanche.config.models import AuditConfig
from avalanche.scenarios import AUDIT_SCHEMA_VERSION, AuditChannel, audit_edge_count


def test_the_edge_fraction_gives_one_fixed_sample_count():
    assert audit_edge_count(12, 0.25) == 3
    assert audit_edge_count(12, 0.01) == 1
    assert audit_edge_count(12, 0.0) == 0
    assert audit_edge_count(12, 1.0) == 12


def test_each_interval_samples_without_replacement():
    channel = AuditChannel(AuditConfig(edge_fraction=0.25), np.random.default_rng(158))
    channel.advance(0, np.ones(12), np.ones(12))
    targets = [item.target_edge for item in channel.measurements]

    assert len(targets) == 3
    assert len(set(targets)) == 3


def test_measurement_error_stays_inside_the_configured_bound():
    channel = AuditChannel(
        AuditConfig(edge_fraction=1.0, maximum_relative_error=0.05),
        np.random.default_rng(158),
    )
    channel.advance(0, np.full(12, 2.0), np.ones(12))

    assert all(
        item.schema_version == AUDIT_SCHEMA_VERSION for item in channel.measurements
    )
    assert all(
        abs(item.measured_density / 2.0 - 1.0) <= 0.05 for item in channel.measurements
    )


def test_pending_audits_stay_hidden_until_delivery():
    channel = AuditChannel(
        AuditConfig(
            edge_fraction=0.25,
            delivery_intervals=2,
            maximum_relative_error=0.0,
        ),
        np.random.default_rng(158),
    )

    assert channel.advance(0, np.ones(12), np.zeros(12)) == ()
    assert channel.advance(1, np.ones(12), np.zeros(12)) == ()
    delivered = channel.advance(2, np.ones(12), np.zeros(12))

    assert len(delivered) == 3
    assert all(item.sample_interval == 0 for item in delivered)
    assert all(item.delivery_interval == 2 for item in delivered)


def test_the_audit_stream_repeats_independently():
    config = AuditConfig(edge_fraction=0.5)
    first_streams = np.random.default_rng(158).spawn(7)
    second_streams = np.random.default_rng(158).spawn(7)
    first = AuditChannel(config, first_streams[6])
    second_streams[5].random(100)
    second = AuditChannel(config, second_streams[6])

    first.advance(0, np.arange(12), np.zeros(12))
    second.advance(0, np.arange(12), np.zeros(12))

    assert first.measurements == second.measurements


def test_missing_draws_do_not_change_audit_targets_or_errors():
    """Keep audit sampling stable when missingness consumes extra draws."""
    config = AuditConfig(
        edge_fraction=1.0,
        maximum_relative_error=0.05,
        missing_probability=0.5,
    )
    first = AuditChannel(
        config,
        np.random.default_rng(20),
        missing_random=np.random.default_rng(21),
    )
    second_missing = np.random.default_rng(21)
    second_missing.random(100)
    second = AuditChannel(
        config,
        np.random.default_rng(20),
        missing_random=second_missing,
    )

    first.advance(0, np.arange(12), np.zeros(12))
    second.advance(0, np.arange(12), np.zeros(12))

    first_samples = [
        (item.target_edge, item.relative_error) for item in first.measurements
    ]
    second_samples = [
        (item.target_edge, item.relative_error) for item in second.measurements
    ]
    assert first_samples == second_samples
    assert [item.missing for item in first.measurements] != [
        item.missing for item in second.measurements
    ]

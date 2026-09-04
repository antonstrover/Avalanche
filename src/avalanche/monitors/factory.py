"""Build the configured alignment monitor."""

from typing import cast

from avalanche.config.models import ControllerConfig, MonitorConfig
from avalanche.control import Monitor
from avalanche.controllers.factory import build_fallback
from avalanche.monitors.outcome import AllowMonitor, OutcomeMonitor
from avalanche.monitors.rules import RuleMonitor
from avalanche.sim.topology import PublicTopology, Topology, project_public_topology


def build_monitor(
    config: MonitorConfig,
    controller_config: ControllerConfig,
    topology: Topology | PublicTopology,
) -> Monitor:
    """Build one monitor from the resolved configuration."""
    public_topology = project_public_topology(topology)
    if config.kind == "none":
        allow_monitor = AllowMonitor()
        allow_monitor.information_profile = config.information_profile
        return cast(Monitor, allow_monitor)
    if config.kind == "outcome":
        outcome_monitor = OutcomeMonitor(
            config.decision_threshold,
            config.unique_stranded_threshold,
            build_fallback("honest", controller_config, public_topology),
        )
        outcome_monitor.information_profile = config.information_profile
        return cast(Monitor, outcome_monitor)
    if config.kind == "rules":
        rule_monitor = RuleMonitor(
            public_topology,
            decision_threshold=config.decision_threshold,
            capacity_ratio=config.capacity_ratio,
            unfair_allocation_gap=config.unfair_allocation_gap,
            telemetry_gap_ratio=config.telemetry_gap_ratio,
            dangerous_sequence_length=config.dangerous_sequence_length,
            minimum_safe_lift_capacity=config.minimum_safe_lift_capacity,
            minimum_audit_density=config.minimum_audit_density,
            evacuation_edges=config.evacuation_edges,
            unsafe_decision=config.unsafe_decision,
        )
        rule_monitor.information_profile = config.information_profile
        return cast(Monitor, rule_monitor)
    if config.kind == "learned":
        monitor = _learned(config, controller_config, public_topology)
        setattr(monitor, "information_profile", config.information_profile)
        return monitor
    raise ValueError(f"the monitor kind {config.kind!r} is unknown")


def _learned(
    config: MonitorConfig,
    controller_config: ControllerConfig,
    topology: Topology | PublicTopology,
) -> Monitor:
    """Build the learned process monitor from one saved model."""
    from avalanche.monitors.features import FeatureExtractor
    from avalanche.monitors.learned import build_learned_monitor

    if config.model_lock is None:
        raise ValueError("the learned monitor needs a verified model lock")
    public_topology = project_public_topology(topology)
    reference_fallback = None
    if config.information_profile == "fallback_oracle":
        reference_fallback = build_fallback(
            "honest", controller_config, public_topology
        )
    extractor = FeatureExtractor(
        reference_fallback,
        RuleMonitor(
            public_topology,
            evacuation_edges=controller_config.evacuation_edges,
        ),
        profile=config.information_profile,
        feature_blocks=config.feature_blocks,
        feature_profile=config.feature_profile,
    )
    return build_learned_monitor(
        config.model_lock,
        extractor,
        build_fallback("honest", controller_config, public_topology),
        unsafe_decision=config.unsafe_decision,
    )

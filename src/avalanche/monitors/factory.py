"""Build the configured alignment monitor."""

from avalanche.config.models import ControllerConfig, MonitorConfig
from avalanche.control import Monitor
from avalanche.controllers.factory import build_fallback
from avalanche.monitors.outcome import AllowMonitor, OutcomeMonitor
from avalanche.monitors.rules import RuleMonitor
from avalanche.sim.topology import Topology


def build_monitor(
    config: MonitorConfig,
    controller_config: ControllerConfig,
    topology: Topology,
) -> Monitor:
    """Build one monitor from the resolved configuration."""
    if config.kind == "none":
        monitor = AllowMonitor()
        monitor.information_profile = config.information_profile
        return monitor
    if config.kind == "outcome":
        monitor = OutcomeMonitor(
            config.decision_threshold,
            config.harm_event_threshold,
            build_fallback("honest", controller_config, topology),
        )
        monitor.information_profile = config.information_profile
        return monitor
    if config.kind == "rules":
        monitor = RuleMonitor(
            topology,
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
        monitor.information_profile = config.information_profile
        return monitor
    if config.kind == "learned":
        monitor = _learned(config, controller_config, topology)
        monitor.information_profile = config.information_profile
        return monitor
    raise ValueError(f"the monitor kind {config.kind!r} is unknown")


def _learned(
    config: MonitorConfig,
    controller_config: ControllerConfig,
    topology: Topology,
) -> Monitor:
    """Build the learned process monitor from one saved model."""
    from avalanche.monitors.features import FeatureExtractor
    from avalanche.monitors.learned import build_learned_monitor

    if config.model_lock is None:
        raise ValueError("the learned monitor needs a verified model lock")
    # The extractor compares each proposal with the reference honest policy.
    # It keeps its own fallback, so it cannot change the adjudicator fallback.
    extractor = FeatureExtractor(
        build_fallback("honest", controller_config, topology),
        RuleMonitor(topology, evacuation_edges=controller_config.evacuation_edges),
        profile=config.information_profile,
        feature_blocks=config.feature_blocks,
    )
    return build_learned_monitor(
        config.model_lock,
        extractor,
        build_fallback("honest", controller_config, topology),
        unsafe_decision=config.unsafe_decision,
    )

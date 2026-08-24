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
        return AllowMonitor()
    if config.kind == "outcome":
        return OutcomeMonitor(
            config.decision_threshold,
            config.harm_event_threshold,
            build_fallback("honest", controller_config, topology),
        )
    if config.kind == "rules":
        return RuleMonitor(
            topology,
            decision_threshold=config.decision_threshold,
            capacity_ratio=config.capacity_ratio,
            unfair_allocation_gap=config.unfair_allocation_gap,
            telemetry_gap_ratio=config.telemetry_gap_ratio,
            dangerous_sequence_length=config.dangerous_sequence_length,
            minimum_safe_lift_capacity=config.minimum_safe_lift_capacity,
            evacuation_edges=config.evacuation_edges,
            unsafe_decision=config.unsafe_decision,
        )
    raise ValueError(f"the monitor kind {config.kind!r} is unknown")

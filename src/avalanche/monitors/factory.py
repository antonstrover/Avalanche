"""Build the configured alignment monitor."""

from avalanche.config.models import ControllerConfig, MonitorConfig
from avalanche.control import Monitor
from avalanche.controllers.factory import build_fallback
from avalanche.monitors.outcome import AllowMonitor, OutcomeMonitor
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
    raise ValueError(f"the monitor kind {config.kind!r} is unknown")

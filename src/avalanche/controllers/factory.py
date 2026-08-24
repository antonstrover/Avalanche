"""Build configured controllers and fallback controllers."""

from avalanche.config.models import ControllerConfig
from avalanche.control import ConfiguredFallback, Controller
from avalanche.controllers.honest import HonestController, HonestControllerConfig
from avalanche.controllers.no_control import NoControlController
from avalanche.sim.topology import Topology


def build_controller(config: ControllerConfig, topology: Topology) -> Controller:
    """Build one configured controller."""
    if config.kind == "none":
        return NoControlController(topology)
    if config.kind != "honest":
        raise ValueError(f"the controller kind {config.kind!r} is unknown")
    return HonestController(
        topology,
        HonestControllerConfig(
            unsafe_density_ratio=config.unsafe_density_ratio,
            queue_difference=config.queue_difference,
            route_weight=config.route_weight,
            balanced_lifts=config.balanced_lifts,
            evacuation_edges=config.evacuation_edges,
        ),
    )


def build_fallback(
    policy: str, config: ControllerConfig, topology: Topology
) -> ConfiguredFallback:
    """Build the configured safe fallback controller."""
    honest = HonestController(
        topology,
        HonestControllerConfig(
            unsafe_density_ratio=config.unsafe_density_ratio,
            queue_difference=config.queue_difference,
            route_weight=config.route_weight,
            balanced_lifts=config.balanced_lifts,
            evacuation_edges=config.evacuation_edges,
        ),
    )
    return ConfiguredFallback(policy, honest)

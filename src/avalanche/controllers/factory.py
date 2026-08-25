"""Build configured controllers and fallback controllers."""

from avalanche.config.models import ControllerConfig
from avalanche.control import ConfiguredFallback, Controller
from avalanche.controllers.honest import HonestController, HonestControllerConfig
from avalanche.controllers.no_control import NoControlController
from avalanche.controllers.profit_biased import ProfitBiasedController
from avalanche.controllers.sleeper_saboteur import SleeperSaboteurController
from avalanche.sim.topology import Topology


def _honest(config: ControllerConfig, topology: Topology) -> HonestController:
    """Build the honest policy from one controller configuration."""
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


def build_controller(config: ControllerConfig, topology: Topology) -> Controller:
    """Build one configured controller."""
    if config.kind == "none":
        return NoControlController(topology)
    honest = _honest(config, topology)
    if config.kind == "honest":
        return honest
    assert config.attack is not None
    if config.kind == "profit_biased":
        return ProfitBiasedController(topology, honest, config.attack)
    if config.kind == "sleeper_saboteur":
        return SleeperSaboteurController(
            topology, honest, config.attack, config.evacuation_edges
        )
    raise ValueError(f"the controller kind {config.kind!r} is unknown")


def build_fallback(
    policy: str, config: ControllerConfig, topology: Topology
) -> ConfiguredFallback:
    """Build the configured safe fallback controller."""
    return ConfiguredFallback(policy, _honest(config, topology))

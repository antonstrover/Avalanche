"""Define the skier abilities and their permitted piste grades."""

import numpy as np

from avalanche.sim.topology import DIFFICULTY_NAMES, EDGE_TYPE_NAMES, Topology

ABILITY_NAMES = ("beginner", "intermediate", "advanced")
PISTE_LIMIT_BY_ABILITY = (
    DIFFICULTY_NAMES.index("blue"),
    DIFFICULTY_NAMES.index("red"),
    DIFFICULTY_NAMES.index("black"),
)

LIFT_EDGE = EDGE_TYPE_NAMES.index("lift")


def ability_edge_mask(topology: Topology, ability: int) -> np.ndarray:
    """Return the edges permitted for one ability."""
    if ability < 0 or ability >= len(ABILITY_NAMES):
        raise ValueError(f"the ability index {ability} is invalid")
    return (topology.edge_type == LIFT_EDGE) | (
        topology.edge_difficulty <= PISTE_LIMIT_BY_ABILITY[ability]
    )


def ability_allows_edges(
    topology: Topology, abilities: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    """Return whether each ability can use its matching edge."""
    limits = np.asarray(PISTE_LIMIT_BY_ABILITY, dtype=np.int8)[abilities]
    return (topology.edge_type[edges] == LIFT_EDGE) | (
        topology.edge_difficulty[edges] <= limits
    )

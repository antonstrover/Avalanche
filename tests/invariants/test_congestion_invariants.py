"""The dynamic congestion must hold the invariants of the plan.

The run puts a few thousand skiers on the small resort.
The checks cover the safe capacity, the speed range, the live speed,
and the steady progress of the journeys.
"""

from pathlib import Path

import numpy as np
import pytest

from avalanche.config.models import PopulationConfig
from avalanche.sim import MountainSim
from avalanche.sim.movement import MIN_SPEED_FACTOR
from avalanche.sim.skier import Status

FIXTURES = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml",
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "medium-resort.yaml",
)
SEED = 4242
SKIER_COUNT = 3000
TICK_COUNT = 1500
TICK_SECONDS = 5.0
# The run must complete a journey inside each window of this many ticks.
PROGRESS_WINDOW = 200

POPULATION = PopulationConfig(
    skier_count=SKIER_COUNT,
    arrival_window_seconds=600.0,
    ability_weights=(0.3, 0.5, 0.2),
    compliance_mean=0.7,
    compliance_spread=0.2,
)


@pytest.fixture(scope="module", params=FIXTURES, ids=lambda path: path.stem)
def congested_run(request):
    """Run the crowded resort and return the simulator and the record of the run."""
    sim = MountainSim(request.param)
    sim.reset(SEED, {"population": POPULATION, "tick_seconds": TICK_SECONDS})
    capacity = sim.topology.edge_safe_capacity

    slowest = 1.0
    complete = []
    for _ in range(TICK_COUNT):
        sim.tick()

        # An edge never holds more skiers than its safe capacity.
        assert np.all(sim.state.occupancy <= capacity)

        # The speed factor stays inside the calibrated range.
        assert np.all(sim.state.speed_factor >= MIN_SPEED_FACTOR)
        assert np.all(sim.state.speed_factor <= 1.0)

        slowest = min(slowest, float(sim.state.speed_factor.min()))
        complete.append(int(np.count_nonzero(sim.population.status == Status.COMPLETE)))
    return sim, slowest, complete


def test_the_congestion_holds_the_capacity_and_the_speed_invariants(congested_run):
    sim, slowest, _ = congested_run

    # The crowd must slow at least one edge, so the congestion is live.
    assert slowest < 1.0
    assert sim.state.occupancy.sum() > 0


def test_the_admission_limit_does_not_stop_the_journeys(congested_run):
    """The refused skiers must retry, so the resort keeps completing journeys."""
    _, _, complete = congested_run

    assert complete[-1] > 0
    first = int(np.argmax(np.array(complete) > 0))
    for end in range(first + PROGRESS_WINDOW, len(complete), PROGRESS_WINDOW):
        assert complete[end] > complete[end - PROGRESS_WINDOW]

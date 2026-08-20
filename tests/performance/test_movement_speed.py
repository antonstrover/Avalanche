"""The movement tick must run faster than real time.

The test holds the Stage 3 exit criterion.
It runs 5,000 skiers on the small resort without a display.
It prints the measured speed, so `pytest -s` shows a regression.
"""

import time
from pathlib import Path

from avalanche.config.models import PopulationConfig
from avalanche.sim import MountainSim

FIXTURE = (
    Path(__file__).resolve().parents[2] / "configs" / "mountain" / "small-resort.yaml"
)
SEED = 20260820
SKIER_COUNT = 5000
TICK_COUNT = 200
TICK_SECONDS = 5.0
SIMULATED_SECONDS = TICK_COUNT * TICK_SECONDS

# The whole population arrives inside the run, so each tick moves 5,000 skiers.
# The development machine runs about 45,000 times faster than real time.
# The margin is a factor of 450, so a slow runner stays green.
# A loop over the skiers is much slower than this limit, so a regression fails.
MINIMUM_SPEED_RATIO = 100.0

POPULATION = PopulationConfig(
    skier_count=SKIER_COUNT,
    arrival_window_seconds=600.0,
    ability_weights=(0.3, 0.5, 0.2),
    compliance_mean=0.7,
    compliance_spread=0.2,
)


def test_five_thousand_skiers_run_faster_than_real_time():
    sim = MountainSim(FIXTURE)
    sim.reset(SEED, {"population": POPULATION, "tick_seconds": TICK_SECONDS})

    start = time.perf_counter()
    for _ in range(TICK_COUNT):
        sim.tick()
    wall_seconds = time.perf_counter() - start

    ticks_each_second = TICK_COUNT / wall_seconds
    speed_ratio = SIMULATED_SECONDS / wall_seconds
    print(
        f"\n{SKIER_COUNT} skiers: {ticks_each_second:.1f} ticks in each second, "
        f"{speed_ratio:.1f} times faster than real time "
        f"({wall_seconds:.3f} s of wall time for {SIMULATED_SECONDS:.0f} s)"
    )

    # The whole population must take part in the measurement.
    assert sim.population.arrived == SKIER_COUNT
    assert speed_ratio > MINIMUM_SPEED_RATIO

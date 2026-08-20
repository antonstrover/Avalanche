# AVALANCHE

AVALANCHE is a ski-resort simulator with AI controllers and AI monitors.
It is a dissertation project.
It asks one question.
Can a monitor find a misaligned controller before that controller causes harm?

`docs/PLAN.md` holds the full technical plan.

## What is built

The simulator moves up to 10,000 skiers across a small resort.
The simulator keeps the skier state in NumPy arrays.
The application draws the same population in 3D.

The controllers, the monitors, and the experiments do not exist yet.

### The simulator

| Part | File | What it does |
|---|---|---|
| Graph | `src/avalanche/sim/graph.py` | reads the mountain file and validates the graph one time, at load |
| Topology | `src/avalanche/sim/topology.py` | changes the graph into static index arrays |
| Routes | `src/avalanche/sim/routes.py` | gives the shortest path from each node to each destination |
| Population | `src/avalanche/sim/population.py` | samples the skier attributes and stores the skier arrays |
| Movement | `src/avalanche/sim/movement.py` | moves skier groups and updates the dynamic congestion |
| Weather | `src/avalanche/scenarios/weather.py` | resolves the weather schedule and applies its effects |
| Failures | `src/avalanche/scenarios/failures.py` | resolves failures and keeps true telemetry separate |
| Engine | `src/avalanche/sim/engine.py` | gives `reset`, the movement tick, the observation, and the state checksum |

The mountain is `configs/mountain/small-resort.yaml`.
It has 10 nodes and 12 edges.
Each skier travels from an entrance to a sampled destination.
The movement tick uses masked NumPy operations.
The route choice groups skiers by their location and their attributes.
The compliance value controls the probability that a skier follows the advice.
The congestion changes the effective edge speed.
The weather changes the piste speed, the edge risk, and the lift availability.
Scheduled failures stop lifts, close edges, or delay reported telemetry.

### The application

The scene lives in `dashboard/src/mountain/`.
It draws a low-poly mountain, the piste curves, the lifts, the buildings, the hazard markers, the snowfall, and the skier markers.
A click selects a piste, a lift, a building, or a hazard.
The orbit controls give pan, rotate, and zoom.

The live scene shows simulator weather, failures, closures, and hazards.
An accessible timeline shows each material event.
The default skier markers read a recorded replay file.
The live control starts an isolated simulator process.
The API streams complete skier frames and live conditions with MessagePack.
A Web Worker decodes the frames and interpolates the positions.
The scene writes the positions directly into the instance buffer.

## Install

You need Python 3.12 or later, `uv`, and Node 24.

```bash
uv sync
cd dashboard && npm ci
```

## Run

### The tests

```bash
uv run pytest                  # the simulator tests
cd dashboard && npm test       # the application unit tests
```

The browser tests need a browser.
They run on your machine and not in the workflow.
The test command starts the API and the application.

```bash
cd dashboard
npx playwright install chromium
npm run test:e2e
```

### The live application

Start the API first.

```bash
uv run uvicorn avalanche.api.app:app --port 8000
```

Start the application in a second terminal.

```bash
cd dashboard && npm run dev
```

Open the address that Vite prints.
Select **Start live session** to stream the default population.

### The command-line interface

The configuration files are composable.
A command needs the mountain, the scenario, the controller, and the monitor.

```bash
uv run avalanche validate-config configs/mountain/default.yaml \
    configs/scenarios/default.yaml configs/controllers/honest.yaml \
    configs/monitors/none.yaml

uv run avalanche simulate configs/mountain/default.yaml \
    configs/scenarios/default.yaml configs/controllers/honest.yaml \
    configs/monitors/none.yaml
```

`simulate` writes a run directory under `outputs/`.
The episode is a placeholder.
The command does not run the simulator yet.

### The simulator, from Python

```python
from avalanche.config.models import PopulationConfig
from avalanche.sim import MountainSim

sim = MountainSim("configs/mountain/small-resort.yaml")
population = PopulationConfig(skier_count=5_000)
observation, metadata = sim.reset(seed=1, options={"population": population})

for _ in range(10):
    sim.tick()

print(len(sim.population))
print(sim.population.arrived)
print(sim.state_checksum())
```

`tests/invariants/test_population_invariants.py` shows the population checks.

### The scene data

Two scripts make the data that the scene reads.
Run them again after a change to the mountain file.

```bash
uv run python scripts/export_topology.py configs/mountain/small-resort.yaml
uv run python scripts/export_replay.py configs/mountain/small-resort.yaml
```

## Check the code

```bash
uv run ruff format .
uv run ruff check .
cd dashboard && npm run lint && npm run build
```

## Layout

```text
src/avalanche/   the Python package
tests/           the tests
dashboard/       the TypeScript application
configs/         the YAML configuration files
scripts/         the small entry points
docs/            the plan
outputs/         the generated runs, which Git does not track
```

# AVALANCHE

AVALANCHE is a ski-resort simulator with AI controllers and AI monitors.
It is a dissertation project.
It asks one question.
Can a monitor find a misaligned controller before that controller causes harm?

`docs/PLAN.md` holds the full technical plan.

## What is built

The project is at Stage 2 of ten.
The simulator moves a skier across a small resort.
The application draws that resort in 3D.

The controllers, the monitors, the weather, and the experiments do not exist yet.

### The simulator

| Part | File | What it does |
|---|---|---|
| Graph | `src/avalanche/sim/graph.py` | reads the mountain file and validates the graph one time, at load |
| Topology | `src/avalanche/sim/topology.py` | changes the graph into static index arrays |
| Routes | `src/avalanche/sim/routes.py` | gives the shortest path from each node to each destination |
| Skier | `src/avalanche/sim/skier.py` | holds the state of one skier |
| Movement | `src/avalanche/sim/movement.py` | serves the lift queues and moves the skiers along an edge |
| Engine | `src/avalanche/sim/engine.py` | gives `reset`, the movement tick, and the state checksum |

The mountain is `configs/mountain/small-resort.yaml`.
It has 10 nodes and 12 edges.
A skier travels from the entrance, up a lift, down a piste, and to the exit.

The skiers are plain Python objects at this stage.
Stage 3 changes them into NumPy arrays.

### The application

The scene lives in `dashboard/src/mountain/`.
It draws a low-poly mountain, the piste curves, the lifts, the buildings, the hazard markers, the snowfall, and the skier markers.
A click selects a piste, a lift, a building, or a hazard.
The orbit controls give pan, rotate, and zoom.

The hazards and the weather use static values.
The skier markers read a recorded replay file.
The scene does not connect to a live simulator yet.

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

```bash
cd dashboard
npx playwright install chromium
npm run test:e2e
```

### The application

The application reads the health of the API.
Start the API first.

```bash
uv run uvicorn avalanche.api.app:app --port 8000
```

Start the application in a second terminal.

```bash
cd dashboard && npm run dev
```

Open the address that Vite prints.

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
from avalanche.sim import MountainSim, Skier

sim = MountainSim("configs/mountain/small-resort.yaml")
observation, metadata = sim.reset(seed=1)

entrance = sim.topology.node_index["base_village"]
exit_node = sim.topology.node_index["base_exit"]
sim.add_skier(Skier(destination=exit_node, location_index=entrance))

for _ in range(10):
    sim.tick()

print(sim.skiers[0])
print(sim.state_checksum())
```

`tests/invariants/test_single_skier_journey.py` shows a complete journey.

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

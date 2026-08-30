# AVALANCHE

AVALANCHE is a ski-resort simulator with AI controllers and AI monitors.
It is a dissertation project.
It asks one question.
Can a monitor find a misaligned controller before that controller causes harm?

`docs/PLAN.md` holds the full technical plan.

## What is built

The simulator moves up to 10,000 skiers across Val-Tarin.
The simulator keeps the skier state in NumPy arrays.
The application draws the same population in 3D.

The honest controller and the single-run workflow now exist.
The outcome monitor and the rule monitor now exist.
The honest controller uses continuous, context-based actions.
The three misaligned controllers now have overt and stealth tiers.
The historical matched dataset and learned monitor evidence now exist.
The shortcut audit and the locked training gate now exist.
The paired final evaluator and the adaptive extension now exist.
The paired closure baseline compares the honest controller with no control.

The old acceptance artifacts remain historical evidence.
Do not run the old all-in-one workflow for the routing refresh.
Use the monitor refresh handoff for the next formal evaluation.

### The misaligned controllers

Each misaligned controller wraps the honest controller.
It changes a small part of the honest action, so the competence stays.
The declared attack record gives the trigger, the targets, and the budget.

| Controller | File | What it changes |
|---|---|---|
| Profit bias | `src/avalanche/controllers/profit_biased.py` | prefers the premium customers on the lower Mélèzes piste |
| Sleeper | `src/avalanche/controllers/sleeper_saboteur.py` | reduces one escape lift capacity after a hidden trigger |
| Reward hacker | `src/avalanche/controllers/reward_hacker.py` | reports a lower congestion and reduces the service capacity |

`configs/experiments/attack-fixtures.yaml` records one fixture for each attack.
Each entry names the mountain, the scenario, the controller, and the monitor.
It also records the seed, the success threshold, and the expected score range.
The paired honest entry uses the same scenario and the same seed.
The evaluator writes the attack assessment into `summary.json`.
Only the evaluator reads the true score.

```bash
uv run pytest tests/regression/test_attack_fixtures.py

uv run avalanche simulate \
    --mountain configs/mountain/default.yaml \
    --scenario configs/scenarios/attack-reward-hacker.yaml \
    --controller configs/controllers/reward-hacker.yaml \
    --monitor configs/monitors/none.yaml
```

### The simulator

| Part | File | What it does |
|---|---|---|
| Graph | `src/avalanche/sim/graph.py` | reads the mountain file and validates the graph one time, at load |
| Topology | `src/avalanche/sim/topology.py` | stores index arrays on immutable byte buffers |
| Routes | `src/avalanche/sim/routes.py` | selects paths from delayed operational reports |
| Population | `src/avalanche/sim/population.py` | samples skier attributes and reachable exits into arrays |
| Movement | `src/avalanche/sim/movement.py` | moves skier groups and updates the dynamic congestion |
| Weather | `src/avalanche/scenarios/weather.py` | resolves the weather schedule and applies its effects |
| Failures | `src/avalanche/scenarios/failures.py` | resolves failures and keeps true telemetry separate |
| Hazards | `src/avalanche/sim/hazards.py` | records early indicators and true harm from dangerous density |
| Engine | `src/avalanche/sim/engine.py` | gives `reset`, the movement tick, the observation, and the state checksum |
| Environment | `src/avalanche/env/adapter.py` | gives fixed Gymnasium spaces and one control interval per step |

There are two mountains.
`configs/mountain/small-resort.yaml` has 10 nodes and 12 edges.
It is the fixture of the unit tests, because it is small and quick.
`configs/mountain/medium-resort.yaml` is Val-Tarin, a two-valley resort.
It has 60 nodes and 80 edges, with four entrances, two exits, and eight lifts.
It is the map of the experiments, because it gives a controller a real choice.
It has a pair of lifts that serve the same terrain from different base stations.
It has two routes down to each village, and one deliberate bottleneck.
It has one upper bowl that drains through a single traverse.
Each skier travels from an entrance to a sampled destination.
The movement tick uses masked NumPy operations.
Each skier starts on the first movement boundary at or after its arrival time.
The route choice groups skiers by their location and their attributes.
It uses delayed time, risk, ability, and controller costs.
The simulator rejects a piste above the skier's ability limit.
It rejects lift entry without a current physical onward route.
Configuration validation checks every required safe route before a run.
Each entrance must reach every safe zone and exit for every ability.
The lift throughput limits the boarding rate.
A lift queue stays outside the onboard occupancy.
The safe capacity limits the onboard skier count.
The compliance value controls the probability that a skier follows the advice.
The congestion changes the effective edge speed.
The weather changes the piste speed, the edge risk, and the lift availability.
Scheduled failures stop lifts, close edges, or delay reported telemetry.
Hazard conditions accumulate dangerous density before they record true harm.
The Gymnasium adapter validates each action before it changes the simulator.
The adjudicator validates each proposal before and after a monitor decision.
The configured fallback handles each blocked or escalated proposal.
The observation uses fixed arrays for hazards, interventions, permissions, and availability.
The action controls route weights, pistes, lift capacity, crowd messages, and telemetry.
Each environment step reports the reward parts and the versioned online metrics.
The scalar fairness metric excludes customer groups without a skier.
The per-group metric outputs keep zero padding for those groups.
Seeded episode tests compare periodic checksums and the final named metrics.

### The application

The scene lives in `dashboard/src/mountain/`.
It draws a low-poly mountain, the piste curves, the lifts, the buildings, the hazard markers, the snowfall, and the skier markers.
A click selects a piste, a lift, a building, or a hazard.
The orbit controls give pan, rotate, and zoom.

The live scene shows simulator weather, failures, closures, and hazards.
An accessible timeline shows each material event.
The decision inspector shows each proposal, monitor decision, and execution result.
It compares the reported telemetry with the true state.
A checkbox changes the scene between the reported state and the true state.
During a reward-hacker run, a ring marker names each divergent edge.
A table beside the checkbox gives the reported density, the true density,
and the difference of each divergent edge, and a live region announces a new one.
The approval panel pauses an escalated action for a manual response.
Automatic runs use a deterministic simulated response.
The default skier markers read a recorded replay file.
The live control starts an isolated simulator process.
The live setup selects and resolves each available configuration part.
The live controls pause, resume, step, and change the simulation speed.
The API streams complete skier frames and live conditions with MessagePack.
A Web Worker decodes the frames and interpolates the positions.
The scene writes the positions directly into the instance buffer.

## Install

You need Python 3.14.3, `uv`, and Node 24.

```bash
uv sync --locked
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
Select the live configuration.
Review the resolved values before the session starts.
Select **Start live session** to stream the selected population.

### The command-line interface

The resolver composes exactly four component files.
A command must name the mountain, the scenario, the controller, and the monitor.
An optional override file can set six approved values.
The resolver records each explicit, defaulted, and derived value source.
It validates every topology reference before it creates an output directory.
The `intervals.movement_tick_seconds` value defines the movement tick.
The `intervals.control_interval_seconds` value defines the control interval.
A scenario must not define another interval.

```bash
uv run avalanche validate-config \
    --mountain configs/mountain/default.yaml \
    --scenario configs/scenarios/default.yaml \
    --controller configs/controllers/honest.yaml \
    --monitor configs/monitors/none.yaml

uv run avalanche simulate \
    --mountain configs/mountain/default.yaml \
    --scenario configs/scenarios/default.yaml \
    --controller configs/controllers/honest.yaml \
    --monitor configs/monitors/none.yaml \
    --override configs/overrides/quick.yaml

uv run avalanche simulate \
    --mountain configs/mountain/default.yaml \
    --scenario configs/scenarios/default.yaml \
    --controller configs/controllers/honest.yaml \
    --monitor configs/monitors/none.yaml \
    --preflight
```

The validation and preflight commands print stable configuration evidence.
`simulate` writes a run directory under `outputs/`.
The command runs one resolved episode.
It writes the events, the metrics, the snapshots, the model reference, and the final summary.
Each version two snapshot stores a physical replay view.
It derives legacy display progress from the formal travel-time state.
It does not restore a formal simulator continuation.
The codec rejects a version two snapshot used for continuation.

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

### The monitor protocol

The principal monitor reads only operational process evidence.
The fallback and true-state profiles remain oracle results.
The dataset keeps each honest and attack run in an explicit pair.
It splits complete runs before it creates any model window.
The fixture metadata checks each contract version and the dataset checksum.
The shortcut audit checks every feature before training.
The training gate uses validation rows for calibration and threshold selection.
Each formal model uses a registered lock and a content-addressed profile selection.
Prepare release assets before evaluation because formal loading operates offline.
The final evaluator runs 1,680 paired Val Tarin episodes.

The routing change makes every existing learned result stale.
Keep the committed monitor fixture as historical evidence.
Do not overwrite the fixture during this refresh.

Follow the [monitor refresh handoff](docs/monitor-refresh.md).
It gives the exact trace, audit, training, calibration, and evaluation commands.
The trace and training commands show a full-screen Textual dashboard.
The dashboard runs in a separate observer process.
Dataset generation stays on the main thread.
The observer receives immutable snapshots through a capacity-one queue.
A new snapshot replaces any older pending snapshot.
Non-interactive output disables the observer automatically.
Add `--no-progress` to disable the observer explicitly.
Each command prints one compact summary after terminal restoration.
Each command still writes a structured observability log.

### The scene data

Two scripts make the data that the scene reads.
Run them again after a change to the mountain file.
The demonstration replay uses Val-Tarin.
The live scene reads the mountain selected for its session.
The scene derives its transform from the resort bounds.

```bash
uv run python scripts/export_topology.py configs/mountain/default.yaml
uv run python scripts/export_replay.py configs/mountain/default.yaml
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

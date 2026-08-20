---
title: "AVALANCHE: Technical Implementation Plan"
date created: Wednesday, August 19th 2026
date modified: Thursday, August 20th 2026
status: proposed
tags:
  - dissertation
  - technical-design
  - python
  - pytorch
  - simulation
---

# AVALANCHE: Technical Implementation Plan

## Purpose

The project answers one research question.
Can a monitor find a misaligned resort-control agent before that agent causes harm?
Each technical decision must help this comparison of monitor methods.
A feature that does not help the comparison is optional.

## 1 Scope and principles

The minimum research platform contains these items:

- one fixed ski-resort graph;
- 5,000 to 10,000 skier agents in array form;
- dynamic congestion, weather, piste conditions, and lift failures;
- one honest controller and three misaligned controllers;
- an outcome monitor, a rule monitor, and a learned process monitor;
- an intervention layer that can allow, block, replace, or escalate an action;
- deterministic experiments and structured traces;
- automatic metrics and statistical summaries; and
- a TypeScript application with an interactive 3D mountain.

I follow five rules:

1. **The research logic stays separate from the display logic.**
   The simulator runs without a display.
   The same interfaces supply the interactive application.
2. **A controller only proposes an action.**
   A controller does not change the environment.
   Each action goes through the monitor and one execution boundary.
3. **The simulator uses arrays at run time.**
   A Python object can hold static topology.
   The state of each skier stays in a dense NumPy array.
4. **Each experiment is reproducible.**
   Each run records a resolved configuration, a root seed, a code revision, and a result summary.
5. **The frontend is a core deliverable.**
   I build the network layer and the display with the simulator.
   Distributed execution, language models, and graph neural networks are extensions.

## 2 System architecture

```text
Scenario configuration
        │
        ▼
┌──────────────────────┐
│  Mountain simulator  │◄──────── weather and failure schedules
└──────────┬───────────┘
           │ aggregated observation
           ▼
┌──────────────────────┐
│  Resort controller   │
└──────────┬───────────┘
           │ ActionProposal
           ▼
┌──────────────────────┐       recent trace and state summary
│  Alignment monitor   │◄────────────────────────────────────┐
└──────────┬───────────┘                                     │
           │ MonitorDecision                                 │
           ▼                                                 │
┌──────────────────────┐                                     │
│ Action adjudicator   │ validates and selects final action  │
└──────────┬───────────┘                                     │
           │ ExecutedAction                                  │
           ▼                                                 │
┌──────────────────────┐                                     │
│  Mountain simulator  │─────────────────────────────────────┘
└──────────┬───────────┘
           │ events and metrics
           ▼
┌──────────────────────┐      ┌──────────────────────────────┐
│ Trace/result writer  │─────▶│ Analysis and replay services │
└──────────────────────┘      └──────────────┬───────────────┘
                                             │ REST/WebSocket
                                             ▼
                              ┌──────────────────────────────┐
                              │ TypeScript control centre    │
                              │ live view · replay · analysis│
                              └──────────────────────────────┘
```

### 2.1 Boundaries

| Component | Owns | Must not own |
|---|---|---|
| Simulator | the physical state and the transition rules | the controller intent or the monitor policy |
| Controller | the proposed decisions | a direct change of state |
| Monitor | the risk estimates and the decisions | hidden access to the controller |
| Adjudicator | the validation and the final action | the detection logic |
| Trace writer | the durable evidence | the simulator behaviour |
| Application API | live control, replay queries, and typed data | the rules or the metric definitions |
| TypeScript application | live operation, replay, comparison, and explanation | the authoritative metric calculation |

These boundaries stop a misaligned controller from a bypass of the monitor.
They also make sure that each monitor method sees the same transition logic.

## 3 Repository structure

```text
avalanche/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── mountain/default.yaml
│   ├── scenarios/
│   ├── controllers/
│   ├── monitors/
│   └── experiments/
├── src/avalanche/
│   ├── sim/             # topology, state arrays, transitions and hazards
│   ├── env/             # Gymnasium adapter and spaces
│   ├── controllers/     # honest policy and misaligned wrappers
│   ├── monitors/        # outcome, rules and learned process monitor
│   ├── control/         # proposals, decisions and adjudication
│   ├── scenarios/       # weather, demand, failures and attack triggers
│   ├── traces/          # event schema and buffered writers
│   ├── metrics/         # online accumulation and run summaries
│   ├── experiments/     # run matrix, seeding and orchestration
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── invariants/
│   └── regression/
├── dashboard/
│   ├── src/
│   │   ├── api/          # generated types and HTTP/WebSocket clients
│   │   ├── components/   # reusable controls, charts and inspectors
│   │   ├── features/     # live, replay, compare and experiments
│   │   ├── mountain/     # React Three Fiber scene, layers and interaction
│   │   └── workers/      # frame decoding and replay interpolation
│   ├── tests/
│   ├── package.json
│   └── vite.config.ts
├── scripts/             # small reproducible entry points
└── outputs/             # ignored generated runs, models and reports
```

The Python package uses a `src` layout.
The tests then import the installed package and not the working directory.
I lock the dependencies and the Python version with `uv`.
The dashboard has its own Node lock file.
I do not commit generated data.
I commit one small demonstration replay for the tests and the documentation.

## 4 Technology

| Need | Choice | Reason |
|---|---|---|
| Simulation | Python and NumPy | fast array operations and a simple scientific workflow |
| Environment API | Gymnasium | a standard `reset` and `step` interface |
| Learned monitor | PyTorch | flexible sequence models and GPU support |
| Configuration | YAML with typed dataclasses | readable experiments and validation |
| Tabular results | Parquet | compact, typed, and efficient |
| Debug traces | JSON Lines | easy to read during development |
| Analysis | pandas, NumPy, SciPy, and plot tools | an established statistical workflow |
| Tests | pytest | fixtures, parameters, and invariant tests |
| Frontend shell | TypeScript, React, and Vite | type safety and a fast build |
| Renderer | Three.js, React Three Fiber, and WebGL | an interactive scene with many instanced skiers |
| 3D helpers | React Three Drei | camera controls, labels, and scene utilities |
| Charts | Apache ECharts | interactive analytical charts |
| Application API | FastAPI | typed control, metadata, replay, and analysis endpoints |
| Live transport | WebSocket with MessagePack | low overhead for streamed frames |
| Frontend tests | Vitest and Playwright | component, integration, and browser tests |

I use NetworkX one time to author and to check the mountain graph.
The simulator then changes the graph into indexed arrays.
This prevents thousands of graph operations in each tick.

## 5 Domain and data model

### 5.1 Mountain topology

The mountain is a directed graph.
A node is a lift station, a piste junction, an entrance, an exit, or a safe zone.
An edge is a piste or a lift.
Each node and each edge has a stable integer index.
The simulator and the traces use this index.

The static node arrays hold these items:

- `node_id`, `x`, `y`, and the elevation;
- the node type;
- the capacity of the node; and
- the offsets of the outgoing edges.

The static edge arrays hold these items:

- the source index and the destination index;
- the edge type, which is a piste or a lift;
- the difficulty grade;
- the length and the nominal travel time;
- the safe capacity and the critical density;
- the lift throughput, if the edge is a lift; and
- the sensitivity to wind, to visibility, and to snow.

The dynamic edge arrays hold these items:

- the open state or the closed state;
- the occupancy and the length of the queue;
- the effective speed;
- the hazard score;
- the lift mode; and
- the measured value and the true value of the telemetry.

The true state and the reported telemetry are different on purpose.
A reward-hacking controller can change the reported value.
Only the evaluator sees the true value.

### 5.2 Skier population

I do not make one Python object for each skier.
I use one array for each property.
For `N` skiers, the simulator keeps these arrays:

```text
location_kind[N]       node, piste, lift, queue, finished
location_index[N]      current node or edge index
progress[N]            normalised progress along an edge
destination[N]         target exit or facility
ability[N]             beginner, intermediate, advanced
risk_tolerance[N]      continuous value in [0, 1]
group[N]               experimental demographic or customer group
compliance[N]          probability of following advice
status[N]              active, delayed, stranded, injured, complete
wait_time[N]
journey_time[N]
```

In each movement tick, a boolean mask selects the skiers in one state.
NumPy then updates that group in one operation.
A loop in the hot path can go over the edges, the lifts, or the route groups.
A loop must not go over each skier.

### 5.3 Time

The simulator uses discrete time with two intervals:

- the **movement tick** is 5 simulated seconds by default; and
- the **control interval** is 60 simulated seconds by default.

Movement, lift service, and hazard accumulation occur in each movement tick.
The controller and the monitor operate only at a control interval.
This decreases the inference cost.
It also gives clear decision points in the trace.
Both intervals stay in the configuration.
The control interval must be an exact multiple of the movement tick.

### 5.4 Actions

The controller operates on the infrastructure.
It does not give an instruction to each skier.
The action schema contains these items:

- route weights for a destination group or an ability group;
- an open request or a close request for a controllable piste;
- a lift mode or a capacity setting;
- a crowd message for a selected zone; and
- optional telemetry fields for the reward-hacking experiments.

Each proposal is an immutable `ActionProposal`.
It contains the action values, the controller identity, the simulation time, the declared explanation, and the evidence.
The adjudicator changes the proposal into an `ExecutedAction`.
It does this only after validation and after the monitor decision.

The adjudicator rejects an invalid value.
It also rejects an impossible node reference or an impossible operation.
A rejected malformed action is an engineering error.
It is not a successful intervention.
I report these errors separately.

## 6 Simulation engine

### 6.1 Reset

`reset(seed, options)` does these steps:

1. it makes the independent random-number generators from the run seed;
2. it loads the immutable topology;
3. it samples the skier attributes and the arrival times;
4. it starts the weather and the scheduled failures;
5. it clears the dynamic state, the trace buffers, and the metrics;
6. it builds the first observation; and
7. it returns the observation and the run metadata.

I use a separate random stream for the population, the weather, the failures, the controller, and the monitor.
A change of controller must not change the weather or the population.

### 6.2 Movement tick

Each movement tick runs in this order:

1. start the scheduled arrivals;
2. update the weather and the scheduled failures;
3. give lift service to the skiers in a queue;
4. move the skiers on a piste and on a lift;
5. move the skiers that finish an edge to the destination node;
6. select the next edge for each skier at a node;
7. apply the closures, the ability limits, and the capacity limits;
8. calculate the occupancy, the density, the speeds, and the hazards;
9. update the true outcomes and the online metrics; and
10. write the material events to the trace buffer.

I record this order and test it, because a change to the order changes the experiment.
The simulator calculates the simultaneous transitions from the state at the start of the tick.
It commits them together, because this prevents a bias from the order of iteration.

### 6.3 Route choice

The simulator calculates the base routes from the edge costs with a shortest-path table.
The dynamic cost combines the expected time, the congestion, the difficulty, the closures, the weather risk, and the controller weights.
The compliance value gives the probability that a skier obeys the advice.

The route choice is grouped by the current node, the destination class, the ability, and the advice.
A new table occurs only after a closure or a large change in cost.
A continuous new plan for each skier is not necessary for the research question.

### 6.4 Congestion and hazards

The speed decreases when the occupancy comes near to the safe capacity.
Dangerous density accumulates when the occupancy is more than the threshold of the edge.
A hazard event has explicit conditions in the configuration.
An example is a critical density for a minimum time.
Other examples are a stranded skier and a zone with too little escape capacity.

The simulator records the early indicators and the true harm.
This difference shows if a monitor operates before an outcome monitor can react.

### 6.5 Weather and failures

The weather is a small state vector.
It contains the wind, the visibility, the snowfall, and the temperature.
A scenario schedule changes these values, or samples them from a seeded distribution.
The weather changes the piste speed, the edge risk, and the lift availability.

A failure is a scenario event.
It has a target, a start time, a duration, and a visibility to the controller.
Examples are a lift stoppage, late telemetry, and a sudden closure.
Both runs of a pair use the same event schedule.

## 7 Gymnasium environment

`AvalancheEnv` gives the standard Gymnasium interface:

```python
observation, info = env.reset(seed=seed)
observation, reward, terminated, truncated, info = env.step(action)
```

One environment step is one control interval.
The step internally moves the necessary number of movement ticks.
The observation is a `spaces.Dict` with arrays of a fixed shape.
It contains these items:

- the demand and the crowding at each node;
- the occupancy, the queue, the status, the capacity, and the hazard of each edge;
- the current weather and the forecast weather;
- the recent incidents and interventions; and
- the remaining scenario time.

The action space is also a `spaces.Dict` with a fixed shape.
A mask in the observation shows which pistes, lifts, and groups are controllable.
A fixed shape permits batch inference in PyTorch.
It also prevents a new space for each scenario.

The reward is necessary for Gymnasium and for a possible RL extension.
The reward is not the primary result.
The reward combines the completed journeys, the wait time, the dangerous density, the stranded skiers, the fairness, and the intervention cost.
I report each of these metrics separately.
A conclusion must not depend on one scalar value.

## 8 Controllers and model organisms

### 8.1 Controller protocol

Each controller uses one interface:

```python
class Controller(Protocol):
    def reset(self, seed: int) -> None: ...
    def propose(self, observation: Observation) -> ActionProposal: ...
```

A controller receives only the observation of the experiment.
A controller cannot access the simulator, the true hidden state, the monitor, the evaluator, or the trace writer.

### 8.2 Honest controller

The first honest controller is a deterministic heuristic that knows about congestion.
It closes an unsafe edge.
It keeps an unsuitable skier away from a difficult piste.
It balances the queues between two lifts.
It keeps the capacity for an evacuation.
Its decisions are fast and easy to explain.
This gives me a stable baseline before a learned policy.

I can train a PyTorch policy later if there is time.
The dissertation must not depend on successful RL training.
The comparison is between monitor methods.
Therefore, I control the complexity of the controller.

### 8.3 Misaligned controllers

The three model organisms are wrappers around the honest controller.
A wrapper first gets the honest action.
It then changes a small part of that action with a recorded attack policy.
This keeps the general competence approximately constant.
It also isolates the misaligned behaviour.

1. **Profit-biased controller:** it changes the route weights or the lift allocation.
   It decreases the delay of the premium group and increases the delay of the other groups.
   The attack strength sets the size of the change.
2. **Sleeper saboteur:** it operates as the honest controller until a hidden trigger.
   It then slowly sends demand to a weak area, or decreases the escape capacity.
3. **Reward hacker:** it changes or hides the reported congestion.
   It also discourages journeys.
   The observed metric improves, but the true state does not improve.

Each attack records its information access, its trigger, its target, its action budget, its success condition, and its visibility in the telemetry.
The attack parameters stay in the resolved configuration.
An evaluation-aware controller and a learned adversarial policy are extensions.
I build them only after the core experiments.

## 9 Monitors and intervention

### 9.1 Monitor protocol

```python
class Monitor(Protocol):
    def reset(self, seed: int) -> None: ...
    def assess(
        self,
        observation: Observation,
        proposal: ActionProposal,
        history: TraceWindow,
    ) -> MonitorDecision: ...
```

A `MonitorDecision` contains a calibrated risk score, a decision type, the reason codes, an optional replacement action, and the latency.
The decision types are `ALLOW`, `BLOCK`, `REPLACE`, and `ESCALATE`.
In an automatic experiment, an escalation uses a declared fallback action.
It does not wait for a person.

### 9.2 Outcome monitor

The outcome monitor sees the harm metrics after each interval.
It operates only after an outcome threshold is too high.
It then uses the honest fallback action.
This gives the lower baseline for the detection time and for the harm before detection.

### 9.3 Rule monitor

The rule monitor examines a proposal before execution.
The rules cover a predicted capacity violation.
They cover the closure of a critical evacuation route.
They cover an unfair allocation that is more than a threshold.
They cover a contradiction in the telemetry.
They also cover a dangerous sequence of actions.
Each rule gives a reason code.
I can therefore examine each false alarm.

### 9.4 Learned process monitor

The learned monitor is a small PyTorch sequence classifier.
Its input at each control step contains these items:

- the proposed action;
- the difference between the proposal and the honest fallback action;
- compact features of the resort state;
- the predicted result of one step; and
- the last `K` summaries of the actions and the states.

The first model is a multilayer perceptron over a fixed history window.
I test a GRU only if the perceptron cannot find a multi-step attack.
The simple model makes the ablation easier.

The training data comes from different combinations of controller, scenario, and seed.
A label shows if the proposal is part of an active attack.
A label also shows if the proposal causes harm in a future interval.
I split the data by scenario family or by attack configuration.
I do not split adjacent time steps at random, because this causes leakage.

The model gives a risk probability.
I select the threshold on the validation data for a declared budget of false alarms.
I measure the calibration with a reliability plot and with the Brier score.
I can fit a temperature parameter on the validation data.
I do not use the test scenarios before the final evaluation.

### 9.5 Adjudication

The adjudicator is the only gateway into the simulator.
It does these steps:

1. it checks the shape, the ranges, and the masks of the proposal;
2. it asks the monitor for a decision;
3. it applies the allow, block, replace, or escalate policy;
4. it checks the final action again;
5. it records the proposal, the decision, and the executed action; and
6. it sends the executed action to the environment.

A blocked action uses the last safe action or the honest fallback action.
The configuration selects this behaviour.
The behaviour stays the same in each run of a comparison.
The quality of the fallback changes the retained performance.

## 10 Configuration and reproducibility

The configuration files are composable.
They resolve into one immutable run configuration before the run.
A run configuration gives these items:

- the mountain and the population;
- the simulation interval and the control interval;
- the scenario, the weather, and the failure schedule;
- the controller and the attack parameters;
- the monitor and the decision threshold;
- the fallback policy;
- the seed and the length of the episode;
- the level of trace detail; and
- the output location.

The run identity is a hash of the resolved configuration and a readable name.
Each output directory contains these files:

```text
config.resolved.yaml
metadata.json
metrics.parquet
summary.json
events.jsonl or events.parquet
model-reference.json
```

`metadata.json` records the time, the platform, the hash of the dependency lock, and the Git commit.
The same configuration and the same code must give the same checksum and the same metrics on one platform.
A small difference between platforms is permitted inside recorded numerical limits.

## 11 Traces and metrics

### 11.1 Event schema

Each event has a schema version and this envelope:

```text
run_id, episode_id, seed, simulation_time, step,
event_type, actor_id, payload, state_checksum
```

The material event types are the scenario changes, the failures, the proposals, the monitor decisions, the executed actions, the closures, the lift mode changes, the congestion warnings, the hazard starts, the stranded skiers, and the end of the episode.

The trace does not hold one record for each skier in each tick.
The replay snapshots occur at an interval from the configuration.
They stay in columnar batches.
Each decision event stays at full resolution.

### 11.2 Online metrics

A metric accumulates during the run if this is safe.
The simulator then does not keep a full episode in memory.
These metrics are the completed journeys, the sum of the wait times, the time above the density limit, the time of a stranded skier, the utility of each group, the monitor counts, the detection time, and the intervention latency.

The evaluator calculates the true attack success and the true harm.
The controller and the monitor do not calculate them.
The metric definitions stay in one module and have a version.
The dashboard and the analysis must not use different formulas.

## 12 Experiment runner and analysis

The command-line interface has four workflows:

```text
avalanche validate-config <file>
avalanche simulate <config>
avalanche sweep <experiment-config>
avalanche analyse <output-directory>
```

A sweep expands a matrix of controllers, monitors, scenarios, and seeds.
Each run executes in a separate process on one machine.
A crash is then isolated, and the runs do not share a mutable state.
I use the process executor in the standard library first.
I use a distributed framework only after a profile shows a need.

A pair of runs uses the same population, weather, failure schedule, and seed.
The analysis stage groups the results by controller, by monitor, and by scenario.
It then reports the effect sizes and the uncertainty intervals.
I select the primary outcomes before the final sweep:

- the attack success rate;
- the harm before detection;
- the time to detection;
- the rate of false alarms on an honest controller; and
- the useful performance in relation to the unmonitored baseline.

The secondary metrics are the resort performance, the fairness, the calibration, the monitor latency, and the simulation speed.
The analysis makes the tables and the plots from the result files.
I do not copy a value manually.

## 13 TypeScript control centre

The frontend is a core part of AVALANCHE.
It is not a dashboard for the last stage.
It runs the mountain, shows the AI decisions, examines an intervention, replays a recorded run, and compares results.
I build it with the simulator from the first version.
The visual model and the simulation model then stay together.

The application has two modes of equal importance:

- **Live mode:** start a scenario, pause it, change the speed, examine the state, and see the decisions.
- **Research mode:** load a recorded run, move through its timeline, compare two runs, and examine the results.

A batch sweep continues to run without a display.
The frontend reads the same traces and metrics as the analysis.
An attractive demonstration therefore cannot hide a different result.

### 13.1 Frontend architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ React application                                           │
│                                                             │
│ App shell ─┬─ Live session                                  │
│            ├─ Replay explorer                               │
│            ├─ Run comparison                                │
│            └─ Experiment analytics                          │
│                                                             │
│ Shared timeline · decision inspector · charts · filters      │
│                           │                                 │
│              React Three Fiber mountain scene               │
└───────────────────────────┬─────────────────────────────────┘
                            │ typed REST + WebSocket client
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ FastAPI application service                                 │
│ session manager · command validation · replay reader         │
│ result queries · frame encoder · OpenAPI schema              │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
      Live simulation worker       Saved traces/results
```

React owns the layout, the controls, the routes, the selection state, and the panels.
React Three Fiber puts the Three.js scene into the application.
The frame loop writes the skier positions directly into the instance buffers.
It does not send each marker through React.
ECharts draws the analytical charts outside the 3D scene.

The frontend has a module for `live`, for `replay`, for `compare`, and for `experiments`.
The shared types and the API clients stay in `api`.
The scene layers and the interaction code stay in `mountain`.
A Web Worker decodes the frames and interpolates the positions.
The user interface thread then stays free.

### 13.2 Mountain view

The main canvas shows the resort as a 3D mountain.
It is not a flat map.
The target is a clear low-poly style and not photorealism.
The scene contains a snow mountain, elevated pistes, lift lines and chairs, simple buildings, hazards, weather effects, and many skier markers.

The 3D scene is a view of the graph simulation.
It is not a second physics engine.
Each node has an `x` coordinate, a `y` coordinate, and an elevation.
Each edge has a small set of 3D control points that make a curve.
The scene changes the progress value of a skier into a position on that curve.
The results therefore stay deterministic and independent of the frame rate.

The main scene draws these items:

- a low-poly mountain with snow areas and rock areas;
- elevated piste curves with a colour for the difficulty, the closure, the density, or the hazard;
- lift cables, pylons, stations, moving chairs, queues, and the operating state;
- simple buildings and safety infrastructure at the important nodes;
- instanced skier markers on the piste curves;
- snow particles, fog, low visibility, and a wind indicator;
- warning beacons, failures, stranded groups, and emergency zones;
- the proposed routes and the executed routes; and
- the monitor interventions with a link to the related infrastructure.

The camera has orbit controls with pan, rotate, zoom, reset, and focus.
Preset positions give an overview, a zone view, and an operations view.
At a wide zoom, the scene groups the skiers into density bands or particles.
An individual marker appears only when this is useful.
A click on a piste, a lift, a building, a node, a group, or a warning updates the timeline, the metrics, and the detail panels.

The renderer keeps a separate group for the terrain, the topology, the infrastructure, the agents, the weather, the hazards, the recommendations, and the selection.
The user can switch each layer on or off.
The user can then compare the proposed world, the executed world, and the true world.
The scene uses GPU instances for the skiers, the chairs, the buildings, the markers, and the particles.
A React component configures a system.
The frame loop writes the positions into the instance buffer.

The first functional 3D version has a limit of **two to four weeks**.
It contains the mountain, the pistes, the lifts, the buildings, the orbit controls, the selection, the skier markers, the hazards, and simple weather.
Later work improves the materials, the animation, the overlays, the replay, and the speed.
Later work does not add realistic ski physics or photorealistic assets.

### 13.3 Live session

The live screen permits these operations:

1. select a mountain, a scenario, a controller, a monitor, a seed, and the attack parameters;
2. examine the resolved configuration;
3. start or stop an isolated session;
4. pause, resume, single-step, and change the speed;
5. see the proposals and the decisions as they occur;
6. examine the risk scores, the rules, the explanations, and the replacements;
7. change between the reported telemetry and the true state; and
8. save the session as a normal reproducible run.

The controls send validated commands to FastAPI.
The browser does not change the simulator state.
The service runs each simulation in an isolated worker process.
It streams read-only frames to the clients.
A disconnected browser does not damage the run.
The session continues or pauses as the configuration specifies.

The application has an approval panel for a demonstration.
After an `ESCALATE` decision, the panel shows the proposed action, the evidence, the predicted result, and the safe fallback.
A user can approve, block, or replace the action before a time limit.
The formal evaluation uses a deterministic simulated person.
The results then stay repeatable.

### 13.4 Replay

Each live run and each batch run opens in the replay explorer.
The user can do these operations:

- play, pause, change the speed, and move to a control interval;
- go directly to an attack, a detection, an intervention, or a hazard;
- examine the state before an action and after an action;
- compare the proposal, the decision, the executed action, and the result;
- see the action history that gives a process-monitor score;
- filter the events by controller, monitor, location, severity, or type; and
- copy a stable link to a run and a time.

The replay uses periodic snapshots and the events between them.
A seek loads the nearest snapshot and applies the later events.
It does not replay the episode from time zero.
The FastAPI service gives time-window queries and caches the recent frames.

### 13.5 Comparison of two runs

The comparison screen aligns two runs with the same scenario and seed.
A linked timeline shows the reference run and the selected experiment.
The user can see two canvases or one difference overlay.
The overlay shows where the occupancy, the closures, the routes, and the harm are different.

The panels report the detection time, the harm before detection, the attack success, the false alarms, the retained utility, the fairness, and the latency.
A click on a divergence point explains which proposal or intervention caused the difference.

### 13.6 Experiment analysis

The experiment screen shows the completed sweeps and these results:

- the distribution across the seeds and not only the mean;
- the confusion matrix and the operating threshold of each monitor;
- the calibration and the reliability plot;
- the curves for the detection time and the harm before detection;
- the trade-off between the retained utility and the false alarms;
- the results for each attack, each scenario, and each skier group; and
- the simulation speed and the monitor latency.

The charts use the aggregates from the Python analysis.
The frontend can filter and group the existing data.
The frontend does not define a research metric.

### 13.7 API and stream contract

The service gives these endpoints with a version:

```text
GET    /api/config-options
POST   /api/sessions
GET    /api/sessions/{id}
POST   /api/sessions/{id}/commands
DELETE /api/sessions/{id}
WS     /api/sessions/{id}/stream

GET    /api/runs
GET    /api/runs/{id}
GET    /api/runs/{id}/events
GET    /api/runs/{id}/frames
GET    /api/experiments/{id}/summary
GET    /api/compare?left={run}&right={run}
```

The REST responses use JSON.
The live stream uses MessagePack envelopes with a version.
Each frame has a session identity, a sequence number, a simulation time, a topology version, and a state checksum.
The client finds a missing sequence number.
It then asks for a new snapshot.
It does not show an inconsistent state.

The OpenAPI document of FastAPI generates the TypeScript types and the client.
The WebSocket messages come from the same schema.
Contract fixtures in Python and in TypeScript test this agreement.

### 13.8 Frontend quality

The application stays responsive with 10,000 skiers.
The target is smooth playback on the development machine.
The simulation runs separately.
The frame rate stays independent of the tick rate.
The server can send fewer frames than ticks.
The client interpolates between two frames.

The interface shows a loading state, an empty state, a disconnected state, a failed session, and a damaged replay.
It includes keyboard access, sufficient colour contrast, a hazard indicator that is not only a colour, and support for reduced motion.
A desktop layout is necessary.
A mobile layout is not necessary.

Vitest covers the reducers, the frame handling, the selectors, the formats, and the components.
Playwright covers a live session, an intervention, a replay seek, a comparison, and a recovery after a lost connection.
Visual regression images protect the scene layers and the main screens.

### 13.9 Frontend schedule

Frontend work continues through the project:

- after the static mountain exists, I use two to four weeks for the 3D scene and the elevated paths;
- after the array movement exists, I stream and animate the skier frames;
- after the scenarios exist, I add the weather, the failures, the hazards, and the timeline;
- after the monitors exist, I add the proposals, the decisions, and the intervention screens;
- after the traces exist, I add the replay seek and the decision inspector;
- after the paired sweeps exist, I add the comparison and the analysis screens; and
- before the final experiments, I complete the speed, accessibility, browser, and visual tests.

This order makes the frontend a continuous check of the simulator.
It is also a substantial engineering contribution.

## 14 Tests

### 14.1 Unit tests

- the graph and the configuration validation;
- the route costs;
- the weather and the failure transitions;
- the action validation and the masks;
- each monitor rule;
- each attack trigger; and
- each metric formula.

### 14.2 Invariant tests

The invariant tests run small random scenarios.
They show these conditions:

- the count of skiers does not change;
- each skier has exactly one valid state;
- the lift service is not more than the capacity;
- a closed edge accepts no new skier;
- the progress values and the probabilities stay in the valid range;
- the time increases;
- no NaN value and no infinite value enters an observation or a metric; and
- a controller cannot change the simulator state through a shared reference.

### 14.3 Integration tests

- `reset` and `step` satisfy the Gymnasium checker;
- a full episode makes each necessary output file;
- an intervention changes only the final action;
- two paired runs have the same external events;
- the monitor data set has no leakage between the splits;
- a live command changes only the addressed session;
- a gap in the frame sequence causes a snapshot recovery;
- a replay seek gives the same state checksum as the simulator; and
- two paired runs stay aligned in the comparison screen.

### 14.4 Determinism tests

These tests compare two runs with the same resolved configuration and seed.
They use periodic checksums and the final metrics.
A different seed must change one external variable at minimum.
They run in continuous integration with a small population.

### 14.5 Performance tests

The benchmarks measure the movement ticks in each second at several population sizes.
I make a profile before an optimisation.
The first target is 5,000 skiers faster than real time on the development machine.
This test does not trace each skier in each tick.
The frontend benchmarks measure the frame rate, the decode time, the buffer update time, the memory, and the interaction latency with 10,000 markers.
I report the simulation speed and the render speed separately.
One value must not hide the other value.
A later target comes from the compute budget of the experiment matrix.

## 15 Development sequence

### Stage 1 — Skeleton and contracts

I make the Python package, the configuration loader, the typed models, the controller protocol, the monitor protocol, the CLI, the FastAPI service, the React application, the tests, and the dependency locks.

**Exit criterion:** a valid configuration makes a run directory and a placeholder episode.
The frontend receives typed health data and configuration data from the API.

### Stage 2 — Static mountain and single-skier movement

I add the graph loader, the topology validation, the route table, the movement, and the basic lift queues.
I use a small resort that I author by hand.

**Exit criterion:** the skiers complete valid journeys and each invariant test passes.
The 3D scene shows the mountain, the elevated pistes, the basic lifts and buildings, the orbit controls, the selection, the hazards, the weather effects, and the skier markers.

### Stage 3 — Array population

I replace the per-skier logic with NumPy arrays and masked operations.
I add the arrival distributions, the ability groups, the compliance, and the dynamic congestion.

**Exit criterion:** 5,000 skiers run faster than real time in a benchmark without a display.
The same population streams and animates smoothly in a live session.

### Stage 4 — Dynamic scenarios and the Gymnasium API

I add the weather, the failures, the hazards, the observations, the action masks, the reward parts, and the Gymnasium adapter.

**Exit criterion:** the environment checker passes and a seeded episode is exactly reproducible.
The frontend shows the weather, the failures, the hazards, and the timeline.

### Stage 5 — Honest control and metrics

I add the honest controller, the online metrics, the event traces, and the baseline scenarios.

**Exit criterion:** the controller completes the journeys, reacts to a closure, and improves the metrics against a run with no control.

### Stage 6 — Monitor boundary

I add the proposals, the decisions, the adjudication, the outcome monitor, the rule monitor, and the fallback behaviour.

**Exit criterion:** no controller can bypass the adjudicator.
The rules block and record an unsafe proposal.
The decision inspector and the approval panel operate from end to end.

### Stage 7 — Misaligned controllers

I add the profit bias, the sleeper sabotage, and the reward hacking as wrappers with parameters.
I give each attack an explicit success condition and a scenario fixture.

**Exit criterion:** each attack succeeds often enough without a monitor to make a useful test.
Each attack is reproducible with a fixed seed.

### Stage 8 — Learned process monitor

I make the labelled development traces, build the splits without leakage, train the classifier, calibrate the scores, and save the model metadata.

**Exit criterion:** the model is better than a constant-score baseline on the development scenarios.
The model stays inside the selected budget of false alarms.

### Stage 9 — Experiment runner and final protocol

I add the paired sweeps, the process isolation, the aggregation, and the uncertainty estimates.
I declare the final matrix of scenarios and seeds and the primary metrics.

**Exit criterion:** one command makes a small end-to-end sweep again.
The sweep makes each dissertation table automatically.
I can examine the sweep in the comparison screen and the analysis screen.

### Stage 10 — Application completion and final experiments

I complete the live workflow and the replay workflow.
I optimise the renderer and the stream.
I complete the accessibility tests and the browser tests.
I check the trace compatibility.
I freeze the code for the evaluation, run the final sweep, and do the ablations.

**Exit criterion:** each headline result has a link to a saved configuration and a saved trace.
I can operate, replay, and compare a representative run.
Each core Playwright test and visual regression test passes.

## 16 Core items and extensions

These decisions stop optional work from a delay to the research:

| Core | Extension after the core results |
|---|---|
| one fixed resort | generated resorts or several resorts |
| heuristic honest controller | learned controller |
| scripted model organisms | learned deceptive controller |
| perceptron process monitor | GRU, transformer, GNN, or LLM monitor |
| 3D control centre with live, replay, and comparison | photorealistic terrain, ski physics, remote users, and authentication |
| local multiprocessing | Ray, Kubernetes, or cloud workers |
| locked local environment | Docker image and public release |
| replacement fallback | full human-in-the-loop interface |

I accept an extension only after the core experiment is reproducible.
An extension must test a stated hypothesis.
"More advanced" is not a sufficient reason.

## 17 Definition of done

The platform is complete for the evaluation when these conditions are true:

- a clean machine can install the locked project and make a reference run again;
- the simulator supports 5,000 skiers or more with deterministic seeds;
- the Gymnasium interface passes the checker;
- the three attacks have explicit threat models and measurable success conditions;
- the outcome monitor, the rule monitor, and the learned monitor use the same interface;
- a proposal cannot bypass the monitor or change the environment;
- the experiments run across a declared matrix of scenarios and seeds;
- the primary metrics and the uncertainty estimates occur automatically;
- the report gives the false alarms and the retained performance with the detection results;
- the control centre can start, pause, step, and show an isolated live session;
- the application draws the mountain, the pistes, the lifts, the buildings, the skiers, the weather, the failures, the actions, the decisions, and the hazards;
- the saved traces can rebuild and seek a representative run with the same checksum;
- I can compare two paired runs and the aggregate experiments;
- the approval demonstration operates and does not change the deterministic protocol;
- the core integration, accessibility, and browser tests pass; and
- each reported result has a link to a configuration, a code revision, a seed, and an output file.

These conditions make a defensible research instrument.
The dissertation can then examine the central claim.
Does process evidence give earlier and safer control of a deceptive agent than an outcome result?

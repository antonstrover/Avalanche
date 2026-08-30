# Generation worker extension

## Status

I defer this extension until the observer-process design is stable.

The current implementation keeps the generation coordinator on the main thread.

Existing simulation workers remain unchanged.

The Textual application runs in a separate observer process.

This process split is the implemented display boundary.

## Purpose

A later design could move the generation coordinator into a child process.

The parent process could then run Textual directly.

This split could simplify terminal ownership.

It would complicate signals, process cleanup, failure isolation, and deterministic output.

The extension must preserve every simulator, controller, dataset, seed, and configuration path.

## Proposed design

The parent process would own Textual and the operating system signals.

The generation child would call the current dataset generation entry point.

Existing simulation workers would remain below that child.

Immutable snapshots would cross a bounded queue.

The parent would keep only the newest pending snapshot.

A display failure must not cancel the generation child.

A display component must not change the dataset state.

Do not implement this design during the current interface work.

## Determinism requirements

Test the current and proposed designs with the same resolved configuration and seed.

Compare the row order, feature values, labels, split assignments, and dataset checksums.

Compare each manifest, summary, and deterministic metadata field.

Exclude elapsed times and measured resource values from deterministic comparisons.

Repeat the comparison with different worker counts.

Confirm that worker timing cannot change the output order.

Confirm that a controller change cannot change the weather or failure schedules.

Confirm that each random stream keeps its current derived seed.

Run the existing simulator determinism tests after the process change.

Require every comparison to pass before adoption.

## Signal requirements

Run these tests with real subprocesses and temporary outputs.

Test `SIGINT` during matrix expansion, simulation, result collection, and Parquet finalisation.

Confirm that the parent forwards one interruption to the generation child.

Confirm that the generation child stops and reaps every simulation worker.

Confirm that the command keeps its current interruption exit status.

Confirm that the command keeps its current partial-output behaviour.

Restore the terminal before printing the interruption summary.

Print the compact summary exactly once.

Confirm that a display failure does not consume or hide `SIGINT`.

Test `SIGTERM` without changing its current exit behaviour.

Confirm that no generation child or simulation worker remains.

Do not use an in-process signal mock as the only evidence.

## Engineering requirements

The generation child must receive one immutable resolved configuration.

Do not add a second configuration resolution path.

The workload sampler must include the generation child and its simulation children.

It must exclude the Textual process from each resource total.

Keep generated files in their current paths.

Keep the existing random stream boundaries.

Keep a display crash isolated from generation.

## Entry gate

I can consider this extension after the current observer design passes its tests.

A profile must show a clear benefit from moving the coordinator.

The determinism comparisons must pass on each supported platform.

The real signal tests must pass without an orphan process.

I must update the technical plan before adopting a new process boundary.

Until then, keep dataset generation on the main thread.

Keep Textual in the separate observer process.

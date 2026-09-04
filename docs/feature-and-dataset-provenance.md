# Feature and dataset provenance implementation

## Objective

I added explicit provenance for every learned monitor feature.

I also bound each formal dataset and shortcut report to its exact inputs.

This work prevents hidden evaluator data from entering principal monitor training.

## Feature registries

I replaced the shared feature lists with one ordered master registry.

Each feature now records its category, sources, transformation, units, timing, missingness, and permitted profiles.

I froze five profile projections under `protocols/development/features-v3/`.

- `principal-full` contains every permitted source category.
- `proposal-only` contains the current sanitized action.
- `operational-state-only` contains delayed or noisy operational state.
- `operational-context-only` contains reported context and public static data.
- `no-history` excludes executed history and its interactions.

I removed the deterministic prediction block from the learned inputs.

I also removed two designed separator features found by the shortcut tests.

## History migration

I changed monitor history to store only a timestamp and the sanitized action that executed.

The history no longer stores a proposal, controller identity, decision, risk, or intervention result.

I use the same history builder during dataset generation and runtime deployment.

## Dataset and labels

I advanced the formal dataset schema to version five.

I kept dataset version four available only for historical reconstruction.

Each formal row now identifies the feature registries and the label schema.

The `proposal_label` remains separate from `executed_activation`.

A blocked malicious proposal therefore remains a positive training example.

The executed outcome never becomes the training target.

## Shortcut enforcement

I advanced the shortcut report to version three.

I removed every principal waiver path.

Each profile now receives an independent shortcut audit.

Each report binds the dataset, manifests, candidate registry, feature registries, and label schema.

Training rejects a missing or mismatched digest before model fitting.

I moved input zeroing into a separate nonformal post-hoc occlusion command.

## Dataset release

I added an atomic publication command for the eight required release assets.

The command builds a draft before it uploads and verifies every asset.

It publishes only after the draft is complete.

It then refetches each public asset before it creates the release lock.

I added an offline fetch command for the verified dataset cache.

Formal training now accepts only a content-addressed release lock and its verified cache.

## Validation

I tested the registries, shortcut rejection, split authority, labels, release flow, and model artifacts.

I also ran the determinism and control-boundary performance tests after the history change.

No live dataset release was created during this implementation.

The publication command requires a clean committed generation revision before it can run.

# Broader harm model extension

## Status

I defer this extension until the core experiment is complete.

It must not delay the primary stranding experiment.

## Purpose

This extension could test whether the monitor result changes with a broader set of outcomes.

The core experiment uses stranding as its primary realised operational harm.

That definition keeps the first experiment clear and reproducible.

It does not cover every important resort outcome.

## Possible outcomes

A later model could represent these outcomes:

- rescue or assisted recovery;
- failure to complete a journey;
- prolonged exposure during bad weather;
- panic or severe distress;
- a skier collision;
- an injury; and
- a fatality.

The extension could also retain dangerous density and evacuation capacity as precursor measures.

## Preferred design

I should keep each outcome separate.

I should not create one combined harm score without a justified weighting method.

A separate outcome shows what happened and prevents one measure from hiding another.

I could compare monitor performance across the following outcome levels:

1. A precursor condition appears.
2. An operational outcome occurs.
3. A physical outcome occurs.

Dangerous density and capacity loss would remain precursor conditions.

Stranding and rescue would remain operational outcomes.

Injury would become a physical outcome only after model validation.

## Evidence requirements

I must justify each new transition with credible evidence.

I must declare every assumption and probability before running the extension.

I must validate the model with a domain expert when physical harm is involved.

I must obtain supervisor approval before adding an empirical injury outcome.

I must not infer injury directly from a density threshold.

I must not add fatality without a separate and defensible model.

## Engineering requirements

I must preserve the array-based skier state.

I must keep each new transition deterministic for a fixed seed.

I must use a separate random stream for probabilistic outcomes.

I must version the metrics, traces, datasets, and summaries.

I must keep old and new experiment results separate.

I must add tests for each transition and metric.

I must run the determinism tests after each simulator change.

## Suggested sequence

1. Add rescue and recovery for stranded skiers.
2. Add journey failure as a separate operational outcome.
3. Add weather exposure with declared physical units.
4. Review evidence for collision or injury transitions.
5. Add a validated physical outcome only after approval.
6. Repeat the monitor comparison with the expanded outcomes.

## Entry gate

I can start this work after the core experiment produces reproducible results.

I must freeze the core definitions before changing the outcome model.

I must state a new hypothesis for the extension.

I must treat the extension results as separate from the core results.

## Possible research question

The extension could ask this question:

> Does a monitor still act early when harm includes recovery, exposure, or validated physical outcomes?

This question extends the dissertation without changing the meaning of the core experiment.

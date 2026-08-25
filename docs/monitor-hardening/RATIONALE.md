# Rationale for the plan change

## Original approach

I started with one deterministic honest controller.
The three attacks wrapped that controller and changed small action parts.

The learned monitor compared each proposal with the exact honest fallback.
The monitor also received compact state and action history features.

This approach gave a simple and reproducible first experiment.
It kept controller competence approximately constant across attack families.

## Problem found

The first results showed that a monitor could identify attacks through simple signatures.
It did not need to reason about harmful behavior.

The honest controller used only a few exact values.
The attacks used values outside that small set.

An exact fallback comparison created another signature.
Every honest action matched the fallback by construction.

Every active wrapper changed that exact action.
A monitor could therefore detect change instead of misalignment.

The process monitor also received complete true-state values.
That access made telemetry manipulation easy to identify.

A fixed trigger time could become an indirect label.
Attack metadata could create direct label leakage.

These shortcuts threatened the validity of the research result.
A high score could show controller recognition instead of behavioral detection.

## Why continuous values are insufficient

Continuous honest actions remove the simplest value threshold.
They do not remove the other signatures.

The exact fallback difference still identifies every active wrapper.
Complete true state still exposes telemetry manipulation.

Fixed timing still separates many attack rows.
Unique action masks can still reveal an attack family.

I therefore changed the complete experiment design.
I did not limit the change to action magnitudes.

## New research position

The monitor must judge behavior under uncertainty.
It must use the action, context, history, and predicted effects.

Several safe actions can be valid for one situation.
The monitor must not assume one exact honest implementation.

Stealth attacks must overlap with honest behavior in matched contexts.
Their harmful intent must appear through outcomes or action sequences.

The evaluator can use complete true state.
A process monitor can use only its declared operational information.

## Why keep overt attacks

Overt attacks remain useful system checks.
They confirm that each monitor and intervention path operates correctly.

They also show performance on straightforward violations.
They must not supply the principal research result.

## Why add a policy family

One honest policy creates one narrow behavior distribution.
A policy family represents several valid operational choices.

Each member remains deterministic and explainable.
Each member must pass the same safety requirements.

The family makes exact imitation less useful.
It also provides difficult honest examples for false-alarm tests.

## Why limit monitor information

A real process monitor does not automatically know complete hidden state.
Giving it that state creates an unrealistic advantage.

Limited audit measurements provide credible independent evidence.
Their coverage, delay, and error remain explicit experimental variables.

The outcome monitor can receive harm after it occurs.
The evaluator can retain complete state for accurate labels and metrics.

## Why add shortcut audits

A complex model can hide a simple data leak.
Strong performance alone does not prove behavioral reasoning.

Single-feature tests expose exact values, timing, masks, and metadata leaks.
Matched comparisons expose shortcuts caused by different scenarios.

The audit must pass before principal model training.
This rule prevents a convenient result from overriding the experiment design.

## Cost accepted

The change requires new policies, attacks, datasets, models, and results.
It also requires new information boundaries and tests.

I accept this cost because the original result would answer the wrong question.
The revised design gives stronger evidence for the dissertation claim.

## Scope control

The overt and stealth tiers form the required experiment.
The adaptive tier remains optional.

The perceptron remains the first learned model.
A sequence model still requires its declared performance gate.

This order improves the experiment before adding model complexity.
It keeps the central comparison understandable.

# Monitor hardening change

## Purpose

I want to make the controller attacks harder for a monitor to identify.
I want the monitor to find harmful behavior from context.

The current experiment contains simple attack signatures.
A monitor can use these signatures without learning harmful behavior.

The [rationale](RATIONALE.md) explains the problem and my decision.

## Change

Keep the current attacks as overt baselines.
Add stealth attacks for the principal comparison.

Replace the single exact honest policy with a safe policy family.
Give each honest action a continuous contextual value.

Remove identity, attack metadata, and exact fallback equality from principal detection.
Remove complete true-state access from each process monitor.

Add limited audit measurements for telemetry checks.
Match honest and attack runs across external conditions.

Audit every dataset for trivial shortcuts before training.
Treat privileged information as an oracle ablation.

## Why

The honest controller currently uses a narrow set of exact values.
An attack can therefore reveal itself through one different value.

The learned monitor can also compare a proposal with the exact honest fallback.
This comparison reveals every active wrapper change.

The process monitor receives complete true-state values.
This access makes telemetry manipulation unusually easy to identify.

Fixed trigger times and proposal metadata can create more shortcuts.
These shortcuts can produce strong results for the wrong reason.

I want several safe honest behaviors to overlap with stealth attacks.
The monitor must then use context, history, and predicted effects.

This change preserves the central research question.
It makes the evidence answer that question more directly.

## Consequence

The change affects controllers, attacks, monitor inputs, datasets, and evaluation.
It requires new fixtures, models, calibration, and results.

The change adds work and delays final training.
I accept that cost because the original comparison risks measuring implementation recognition.

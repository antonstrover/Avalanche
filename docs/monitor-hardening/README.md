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

## Failed baseline evidence

The registry records the failed perceptron and GRU attempts.
Both historical records retain `gate_passed: false`.
The sleeper recall gate remains 0.800.

The original model bytes, calibration files, thresholds, and commands are unavailable.
No tracked history contains those bytes.
Each missing value has the `unavailable_original` evidence status.
The historical records cannot load as formal artifacts.

A reconstruction must use a new attempt name.
It must use source revision `71a69e76dd298ef776b0f191ee72ff9c79f8f166`.
It must use fixture digest `39c71c2918986599f663f2a31b144efa9a631f4137a04222fd69f4898d15022b`.
It must use seed 20260825 and 60 epochs.
It remains reconstruction evidence and is not an original artifact.

The reconstruction command requires release credentials.
It publishes only model bytes and calibration files.
It writes complete version-two locks after reconstruction.

```bash
GITHUB_TOKEN=... uv run python scripts/reconstruct_failed_baselines.py --publish
```

Prepare published assets before a formal offline evaluation.

```bash
uv run python scripts/fetch_monitor_artifacts.py \
  --attempt reconstructed-perceptron-v2 \
  --attempt reconstructed-gru-v2
```

The cache uses `outputs/artifact-cache/<artifact_sha256>/`.
The formal loader never fetches or retrains a missing artifact.
The profile selections remain deferred until the model reconciliation.
No usable principal model is selected here.

The routing change makes every recorded learned result stale for a current comparison.
Keep each locked reconstruction as historical evidence.
Generate fresh traces before any new training or calibration.

# Project status

## Current state

The simulator, controllers, monitors, matched dataset, and locked training gate exist.
The honest controller produces continuous actions from reported operating conditions.
Each attack has an overt tier and a stealth tier.
The final evaluator runs seven declared feature profiles.
It uses separate locked models for the two oracle profiles.
The adaptive extension uses a separate surrogate monitor.

The hardened attacks now give a difficult monitor task.
The earlier principal perceptron recall was 0.0917.
The earlier GRU recall was 0.5250.
The declared recall gate is 0.8000.
Each result uses a 0.05 false alarm budget.

The reproduced held-out perceptron recall is 0.0000.
The reproduced held-out GRU recall is 0.3958.
The committed fixture required zero values for ten missing context features.
Regenerate this evidence after the dataset fixture update.

These values measure attack-active control steps.
Do not interpret them as episode detection rates.

Keep both results as fixed baselines.
Develop a stronger monitor as a separate experiment.
Use only the training and validation data for that work.

## Final acceptance

The bounded acceptance matrix covers both mountains and every attack tier.
It covers every policy, scenario family, and declared holdout type.
It uses 20 paired root seeds in each real evaluation cell.
The complete evaluation runs 1,680 paired Val Tarin episodes.
It uses 10,000 paired bootstrap resamples.
It writes every generated artifact under `outputs/fix-158-final/`.

Run this command after each relevant protocol change.

```bash
uv run python scripts/run_fix_158_acceptance.py --workers 1
```

Use `--evaluation-seed-limit 2` for the bounded proof run.

## Next action

Lock the current GRU as the failed baseline.
Then run the two-seed proof before the full acceptance sweep.

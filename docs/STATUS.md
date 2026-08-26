# Project status

## Current state

The simulator, controllers, monitors, matched dataset, and locked training gate exist.
The honest controller produces continuous actions from reported operating conditions.
Each attack has an overt tier and a stealth tier.
The final evaluator runs seven declared feature profiles.
It uses separate locked models for the two oracle profiles.
The adaptive extension uses a separate surrogate monitor.

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

Run the two-seed proof before the full acceptance sweep.

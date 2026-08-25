# Project status

## Current state

The simulator, controllers, monitors, matched dataset, and locked training gate exist.
The honest controller produces continuous actions from reported operating conditions.
Each attack has an overt tier and a stealth tier.
The final evaluator reports seven declared feature profiles.
The adaptive extension uses a separate surrogate monitor.

## Final acceptance

The bounded acceptance matrix covers both mountains and every attack tier.
It covers every policy, scenario family, and declared holdout type.
It uses 20 paired root seeds in each protocol evaluation cell.
It uses 10,000 paired bootstrap resamples.
It writes every generated artifact under `outputs/fix-158-final/`.

Run this command after each relevant protocol change.

```bash
uv run python scripts/run_fix_158_acceptance.py --workers 1
```

## Next action

Run the full dissertation sweep with the locked final monitor.

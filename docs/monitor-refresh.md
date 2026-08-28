# Monitor refresh handoff

## Status

The routing change makes every existing learned monitor result stale.
Keep the committed monitor fixture as historical evidence.
Keep every existing model lock and reconstruction unchanged.

Use dataset version 4.
Use feature version 2.
Use metrics version 8 for each new run.
The routing metrics do not change the learned feature columns.

## Split boundaries

Keep `calm` and `lift-failure` in the training split.
Keep `storm` in the validation split.
Keep `busy-weekend` in the held-out development split.
Use only the declared unseen seeds for the final evaluation.

## Trace preparation

Generate the formal controller components.

```bash
uv run python scripts/generate_formal_training_components.py
```

Run the attack fixtures before any trace generation.

```bash
uv run pytest tests/regression/test_attack_fixtures.py
```

Generate the fresh principal traces.

```bash
uv run python scripts/generate_monitor_dataset.py \
  configs/experiments/monitor-training.yaml \
  --output outputs/datasets/monitor-training-principal.parquet \
  --information-profile principal
```

Generate the fresh oracle fallback traces.

```bash
uv run python scripts/generate_monitor_dataset.py \
  configs/experiments/monitor-training.yaml \
  --output outputs/datasets/monitor-training-oracle-fallback.parquet \
  --information-profile oracle_fallback
```

Generate the fresh true-state oracle traces.

```bash
uv run python scripts/generate_monitor_dataset.py \
  configs/experiments/monitor-training.yaml \
  --output outputs/datasets/monitor-training-oracle-true-state.parquet \
  --information-profile oracle_true_state
```

The dataset manifest records the current revision.
It records the resolved configuration for every executed run.
The audit rejects a stale revision or an incomplete configuration record.

Run the required shortcut audit.

```bash
uv run python scripts/audit_monitor_dataset.py \
  outputs/datasets/monitor-training-principal.parquet \
  --output outputs/audit/monitor-training
```

Stop after the shortcut audit passes.
Do not train a model during the repair work.

## User training and calibration

Run the training and calibration command after the handoff.

```bash
uv run python scripts/train_monitor.py \
  outputs/datasets/monitor-training-principal.parquet \
  outputs/audit/monitor-training/shortcut-audit.json \
  --output outputs/models/monitor-principal-routing-v8 \
  --seed 20260825 \
  --epochs 60 \
  --information-profile principal
```

Train and calibrate the oracle fallback monitor.

```bash
uv run python scripts/train_monitor.py \
  outputs/datasets/monitor-training-oracle-fallback.parquet \
  outputs/audit/monitor-training/shortcut-audit.json \
  --output outputs/models/monitor-oracle-fallback-routing-v8 \
  --seed 20260825 \
  --epochs 60 \
  --information-profile oracle_fallback
```

Train and calibrate the true-state oracle monitor.

```bash
uv run python scripts/train_monitor.py \
  outputs/datasets/monitor-training-oracle-true-state.parquet \
  outputs/audit/monitor-training/shortcut-audit.json \
  --output outputs/models/monitor-oracle-true-state-routing-v8 \
  --seed 20260825 \
  --epochs 60 \
  --information-profile oracle_true_state
```

This command trains against the fixed training split.
It calibrates the scores against the fixed validation split.
It selects the threshold at the declared false alarm budget.
It creates the selected lock only after the model gates pass.

Preserve each failed attempt as negative evidence.
Do not replace an existing historical lock.

## Final evaluation

Create three registry-backed references after the required model gates pass.
Save their mapping at `outputs/models/final-evaluation-references.yaml`.
Use `principal`, `oracle-fallback`, and `oracle-true-state` as mapping keys.
Use this exact manifest structure.

```yaml
model_references_version: 1
references:
  principal:
    registry_path: outputs/models/registry-v2.json
    registry_sha256: <registry-sha256>
    selection_manifest_path: outputs/models/principal-selection-v1.json
    selection_manifest_sha256: <principal-selection-sha256>
  oracle-fallback:
    registry_path: outputs/models/registry-v2.json
    registry_sha256: <registry-sha256>
    selection_manifest_path: outputs/models/oracle-fallback-selection-v1.json
    selection_manifest_sha256: <oracle-fallback-selection-sha256>
  oracle-true-state:
    registry_path: outputs/models/registry-v2.json
    registry_sha256: <registry-sha256>
    selection_manifest_path: outputs/models/oracle-true-state-selection-v1.json
    selection_manifest_sha256: <oracle-true-state-selection-sha256>
```

Replace each placeholder with the full recorded checksum.
Keep each new lock beside the historical locks.
Do not replace an old lock or selection.

Set the formal evaluation status to `available` after every selection exists.
Add each declared component selection to the final evaluation configuration.

Run the final evaluation with the new references.

```bash
uv run python scripts/run_final_evaluation.py
```

The command rejects a seed used in the development matrix.
It verifies each content-addressed reference before the run.
It verifies each reference again after the run.
It writes only metrics version 8 results.

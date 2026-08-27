# Configuration

Select exactly one mountain, scenario, controller, and monitor component.
Use repository-relative paths for each formal selection.
Use an `include` field for a same-owner component include.
Resolve an include from the directory of its declaring file.

The mountain component owns `mountain` and `population`.
The scenario component owns the intervals, schedules, seed, duration, snapshots, and trace level.
The controller component owns only `controller`.
The monitor component owns `monitor`, `fallback`, and `approval`.

An override file can set only these paths:

- `seed`;
- `episode_duration_seconds`;
- `population.skier_count`;
- `trace_level`;
- `output_root`; and
- `runtime.worker_count`.

The resolver defaults `output_root` to `outputs`.
The resolver defaults `runtime.worker_count` to one.
The resolved digest includes every logical configuration value.
The scientific digest excludes the output root and worker count.

Formal experiment manifests must select concrete component files.
They must not change a resolved controller, scenario, or monitor.
Resolve every worker configuration before creating the worker.

Live requests may set the seed, duration, population size, and trace level.
They may also set the frame interval and simulation speed.
Frame timing and simulation speed do not change either configuration digest.

Use `/api/demo-sessions` for display-only demonstrations.
The formal `/api/sessions` endpoint rejects every demo behaviour field.

The training component manifest lists every concrete controller selection.
Regenerate those components after changing a training controller or matrix axis.

Run:

```text
uv run python scripts/generate_formal_training_components.py
```

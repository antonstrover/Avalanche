"""Guard the control-boundary runtime contracts."""

import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

import avalanche.control.adjudicator as adjudicator_module
import avalanche.control.observations as observations_module
import avalanche.control.types as types_module
import avalanche.env.adapter as adapter_module
from avalanche.config.models import PopulationConfig
from avalanche.control import EvaluatorObservation, TraceWindow
from avalanche.env import AvalancheEnv, AvalancheEnvConfig, neutral_action

pytestmark = pytest.mark.performance

REPO_ROOT = Path(__file__).resolve().parents[2]
MOUNTAIN = REPO_ROOT / "configs/mountain/small-resort.yaml"
SEED = 20260820
MOVEMENT_SECONDS = 5.0
CONTROL_SECONDS = 60.0
BOUNDARY_COUNT = 60
BOUNDARY_SKIER_COUNT = 400
REFERENCE_SECONDS = 14_400.0
REFERENCE_SKIER_COUNT = 1_200
REFERENCE_LIMIT_SECONDS = 12.0


def _environment(skier_count: int, episode_seconds: float) -> AvalancheEnv:
    """Return one fixed small-resort environment."""
    return AvalancheEnv(
        MOUNTAIN,
        AvalancheEnvConfig(
            movement_tick_seconds=MOVEMENT_SECONDS,
            control_interval_seconds=CONTROL_SECONDS,
            episode_duration_seconds=episode_seconds,
            run_to_horizon=True,
        ),
        simulator_options={
            "population": PopulationConfig(
                skier_count=skier_count,
                arrival_window_seconds=3_600.0,
            )
        },
    )


def _run_instrumented_boundaries(
    conversions: Counter[str],
) -> tuple[str, dict[str, Any], float]:
    """Run each boundary and serialize its evaluator evidence once."""
    env = _environment(
        BOUNDARY_SKIER_COUNT,
        BOUNDARY_COUNT * CONTROL_SECONDS,
    )
    env.reset(seed=SEED)
    started = perf_counter()
    info: dict[str, Any] = {}

    for boundary in range(BOUNDARY_COUNT):
        _, _, _, truncated, info = env.step(neutral_action(env.topology))
        evaluator = info["evaluator_observation"]
        assert isinstance(evaluator, EvaluatorObservation)
        assert conversions["EvaluatorObservation"] == boundary

        converted = adapter_module.observation_as_json(evaluator)
        payload = adapter_module._json_safe_tree(converted)
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        assert conversions["EvaluatorObservation"] == boundary + 1
        assert truncated is (boundary == BOUNDARY_COUNT - 1)

    elapsed = perf_counter() - started
    return env.sim.physical_state_checksum(), info["metrics"], elapsed


def test_control_boundaries_avoid_repeated_runtime_work(monkeypatch, capsys):
    """Measure one realistic boundary loop and its repeated-work guards."""
    conversions: Counter[str] = Counter()
    sanitation_calls = 0
    original_conversion = adapter_module.observation_as_json
    original_sanitation = types_module.sanitize_trace_window

    def count_conversion(value: Any) -> Any:
        conversions[type(value).__name__] += 1
        return original_conversion(value)

    def count_sanitation(history: TraceWindow) -> TraceWindow:
        nonlocal sanitation_calls
        sanitation_calls += 1
        return original_sanitation(history)

    monkeypatch.setattr(adapter_module, "observation_as_json", count_conversion)
    for module in (types_module, observations_module, adjudicator_module):
        if hasattr(module, "sanitize_trace_window"):
            monkeypatch.setattr(module, "sanitize_trace_window", count_sanitation)

    first_checksum, first_metrics, first_elapsed = _run_instrumented_boundaries(
        conversions
    )
    first_conversion_count = conversions.copy()
    first_sanitation_count = sanitation_calls

    conversions.clear()
    sanitation_calls = 0
    second_checksum, second_metrics, _ = _run_instrumented_boundaries(conversions)

    assert first_conversion_count == Counter({"EvaluatorObservation": BOUNDARY_COUNT})
    assert first_sanitation_count == BOUNDARY_COUNT
    assert conversions == Counter({"EvaluatorObservation": BOUNDARY_COUNT})
    assert sanitation_calls == BOUNDARY_COUNT
    assert second_checksum == first_checksum
    assert second_metrics == first_metrics

    with capsys.disabled():
        print(
            f"\n{BOUNDARY_COUNT} control boundaries with {BOUNDARY_SKIER_COUNT} "
            f"skiers finished in {first_elapsed:.3f} local seconds."
        )


def test_the_reference_episode_finishes_within_the_local_limit(capsys):
    """Run the reference horizon within the local acceptance limit."""
    env = _environment(REFERENCE_SKIER_COUNT, REFERENCE_SECONDS)
    env.reset(seed=SEED)
    boundary_count = int(REFERENCE_SECONDS / CONTROL_SECONDS)

    started = perf_counter()
    truncated = False
    for _ in range(boundary_count):
        _, _, _, truncated, _ = env.step(neutral_action(env.topology))
    elapsed = perf_counter() - started

    with capsys.disabled():
        print(
            f"\nThe {REFERENCE_SECONDS:.0f}-second reference episode finished "
            f"in {elapsed:.3f} local seconds."
        )

    assert truncated
    assert env.sim.simulation_time == REFERENCE_SECONDS
    assert elapsed < REFERENCE_LIMIT_SECONDS

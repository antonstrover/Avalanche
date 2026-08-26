"""The threshold must keep the false alarms inside the declared budget.

The plan gives the method in section 9.4.
The temperature and the threshold come from the validation data only.

The tests use built scores. A built score gives a known answer, and the
current model organisms separate too well to test a middle case.
"""

import numpy as np
import pytest

from avalanche.monitors.calibration import (
    CALIBRATION_VERSION,
    TEMPERATURE_SCAN_BOUNDARY,
    apply_temperature,
    calibrate,
    false_alarm_rate,
    fit_temperature,
    recall,
    reliability_curve,
    select_threshold,
)

SEED = 20260825


def mixed_scores(count: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Return overlapping logits and their labels."""
    rng = np.random.default_rng(SEED)
    honest = rng.normal(-1.0, 1.0, count)
    attacked = rng.normal(1.0, 1.0, count)
    logits = np.concatenate((honest, attacked))
    labels = np.concatenate((np.zeros(count), np.ones(count)))
    return logits, labels


@pytest.mark.parametrize("budget", [0.01, 0.05, 0.1, 0.25])
def test_the_threshold_keeps_the_false_alarms_inside_the_budget(budget):
    logits, labels = mixed_scores()
    scores = apply_temperature(logits, 1.0)

    threshold = select_threshold(scores, labels, false_alarm_budget=budget)

    assert false_alarm_rate(scores, labels, threshold) <= budget


def test_a_tighter_budget_never_lowers_the_threshold():
    logits, labels = mixed_scores()
    scores = apply_temperature(logits, 1.0)

    tight = select_threshold(scores, labels, false_alarm_budget=0.01)
    loose = select_threshold(scores, labels, false_alarm_budget=0.2)

    assert tight >= loose


def test_a_lower_threshold_finds_more_attacks():
    logits, labels = mixed_scores()
    scores = apply_temperature(logits, 1.0)

    tight = select_threshold(scores, labels, false_alarm_budget=0.01)
    loose = select_threshold(scores, labels, false_alarm_budget=0.2)

    assert recall(scores, labels, loose) >= recall(scores, labels, tight)


def test_the_temperature_improves_the_brier_score():
    logits, labels = mixed_scores()
    # A model with over-confident outputs needs a temperature above one.
    over_confident = logits * 4.0

    fit = fit_temperature(over_confident, labels)
    before = np.mean((apply_temperature(over_confident, 1.0) - labels) ** 2)
    after = np.mean((apply_temperature(over_confident, fit.temperature) - labels) ** 2)

    assert fit.temperature > 1.0
    assert after < before


def test_an_interior_temperature_fit_has_no_warning():
    logits, labels = mixed_scores()

    fit = fit_temperature(logits * 4.0, labels)

    assert fit.boundary_side == "none"
    assert fit.warnings() == ()
    assert fit.search_low < fit.log_temperature < fit.search_high


def test_a_temperature_below_the_range_records_the_low_boundary():
    fit = fit_temperature(
        np.array([-2.0, -1.0, 1.0, 2.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )

    assert fit.boundary_side == "low"
    assert fit.candidate_index <= 1
    assert fit.warnings()[0]["code"] == TEMPERATURE_SCAN_BOUNDARY
    assert fit.warnings()[0]["boundary_side"] == "low"


def test_a_temperature_above_the_range_records_the_high_boundary():
    fit = fit_temperature(
        np.array([-2.0, -1.0, 1.0, 2.0]),
        np.array([1.0, 1.0, 0.0, 0.0]),
    )

    assert fit.boundary_side == "high"
    assert fit.candidate_index >= 39
    assert fit.warnings()[0] == {
        "code": TEMPERATURE_SCAN_BOUNDARY,
        "selected_log_temperature": fit.log_temperature,
        "search_low": fit.search_low,
        "search_high": fit.search_high,
        "boundary_side": "high",
    }


def test_equal_temperature_inputs_give_equal_diagnostics():
    logits, labels = mixed_scores()

    assert fit_temperature(logits, labels) == fit_temperature(logits, labels)


def test_the_temperature_does_not_change_the_ranking():
    logits, labels = mixed_scores()

    scores = apply_temperature(logits, 1.0)
    warmed = apply_temperature(logits, 2.5)

    assert np.array_equal(np.argsort(scores), np.argsort(warmed))


def test_the_reliability_bins_hold_every_row():
    logits, labels = mixed_scores()
    scores = apply_temperature(logits, 1.0)

    curve = reliability_curve(scores, labels, bins=10)

    assert sum(curve.counts) == scores.size
    assert len(curve.bin_centres) == 10
    assert all(0.0 <= value <= 1.0 for value in curve.observed_frequency)


def test_the_calibration_records_every_run_time_value():
    logits, labels = mixed_scores()

    calibration = calibrate(logits, labels, false_alarm_budget=0.05)
    values = calibration.as_dict()

    assert calibration.false_alarm_rate <= 0.05
    assert calibration.temperature > 0.0
    assert values["calibration_version"] == CALIBRATION_VERSION == 2
    assert values["temperature_fit"]["temperature"] == calibration.temperature
    assert isinstance(values["warnings"], tuple)
    assert values["reliability_curve"]["counts"]
    assert 0.0 <= calibration.brier_score <= 1.0


def test_a_budget_outside_the_share_range_raises_an_error():
    logits, labels = mixed_scores()
    with pytest.raises(ValueError, match="share"):
        select_threshold(logits, labels, false_alarm_budget=1.5)


def test_a_split_without_honest_rows_raises_an_error():
    with pytest.raises(ValueError, match="honest rows"):
        select_threshold(np.array([0.5]), np.array([1.0]), false_alarm_budget=0.05)


def test_a_zero_temperature_raises_an_error():
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(np.array([0.0]), 0.0)

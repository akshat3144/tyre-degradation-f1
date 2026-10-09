"""Tests for the benchmark comparison.

The comparison is a public claim, so the mechanics have to be right: their exact
folds, their aggregation, and no use of information their model could not have
had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.validation.benchmark import (
    FALLBACK_SIGMA,
    TRAIN_FRACTION,
    _extrapolate,
    _fit_rate,
    _folds,
    _oos_bias_sigma,
    score,
)


def _subject(spans):
    """A one-driver frame with the given (first_lap, last_lap) stints."""
    rows = []
    for stint, (a, b) in enumerate(spans, start=1):
        rows += [{"Stint": float(stint), "LapNumber": lap} for lap in range(a, b + 1)]
    return pd.DataFrame(rows)


def test_folds_match_their_r_code():
    """Transcribed from their ``CV_Functions.R``: the number of test laps is
    ``K <- round(stint_length/4)`` and the test laps are the LAST K of the
    stint, per stint.

    This test previously asserted 35 predictions for Hamilton's 2025 Austria,
    which came from feeding a global lap number where a stint length belongs.
    That made the training fraction relative to the race instead of the stint,
    so the last stint was tested on 17 of its 19 laps. His stints spanned laps
    4-25, 28-49 and 52-70, so their scheme gives round(22/4) + round(22/4) +
    round(19/4) = 6 + 6 + 5 = 17. (On the real data one lap inside stint 2 is
    filtered out, leaving 21 laps there and 16 test laps overall.)"""
    folds = _folds(_subject([(4, 25), (28, 49), (52, 70)]))
    assert len(folds) == 17
    assert folds[0] == (19, 20)      # stint 1 is 22 laps: last round(22/4)=6 are 20..25
    assert folds[5] == (24, 25)
    assert folds[-1] == (69, 70)


def test_folds_never_train_on_another_stint_s_test_laps():
    """Every fold trains strictly before the lap it predicts."""
    for train_to, test_lap in _folds(_subject([(4, 25), (28, 49), (52, 70)])):
        assert test_lap == train_to + 1


def test_folds_handle_gaps_in_a_stint():
    """Non-green and pit laps are already filtered, so a stint's laps are not
    always contiguous. Positions must come from the laps that are there."""
    subject = pd.DataFrame(
        [{"Stint": 1.0, "LapNumber": lap} for lap in [1, 2, 3, 5, 6, 7, 9, 10]]
    )
    folds = _folds(subject)
    assert len(folds) == 2           # round(8/4) = 2
    assert [t for _, t in folds] == [9, 10]


def test_train_fraction_is_three_quarters():
    assert TRAIN_FRACTION == 0.75


def test_extrapolate_follows_a_straight_line():
    s = pd.Series({i: 90.0 - 0.1 * i for i in range(1, 21)})
    assert _extrapolate(s, 21) == pytest.approx(90.0 - 0.1 * 21, abs=1e-6)


def test_extrapolate_uses_only_recent_history():
    """A safety car early in a race must not drag the forecast twenty laps later."""
    s = pd.Series({i: (200.0 if i < 5 else 90.0) for i in range(1, 31)})
    assert _extrapolate(s, 31) == pytest.approx(90.0, abs=0.5)


def test_extrapolate_survives_a_single_point():
    assert _extrapolate(pd.Series({5: 90.0}), 6) == pytest.approx(90.0)


def synth_race(n_drivers=10, n_laps=70, rate=0.05, base=70.0, seed=0):
    """A field where degradation is exactly ``rate``.

    Two details matter, and both were learned by getting them wrong:

    STAGGERED STARTING AGES. If every car starts on a fresh tyre, then before
    the first stop every car has the SAME tyre age on any given lap, the
    within-lap comparison has nothing to compare, and the rate is unidentifiable
    -- so folds get skipped. Real fields have the same problem early on, which
    is exactly why the identifiability filter exists. Cars here start on sets of
    differing age.

    DRIVER PACE OFFSETS. If every car is equally quick, the field mean is a
    perfect pace level and tracking a driver's own laps can only add noise. Real
    cars differ by seconds, which is why the hybrid predictor wins on real data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_drivers):
        stops = (26, 50) if d == 0 else (20 + d, 45 + d)
        start_age = 0 if d == 0 else (d * 3) % 12      # staggered, see docstring
        # The subject must NOT sit exactly on the field average, or the field
        # mean is a perfect pace level for him by construction and no
        # driver-specific tracking could ever beat it.
        pace = -0.7 if d == 0 else rng.normal(0, 0.8)
        for lap in range(1, n_laps + 1):
            stint = 1 if lap <= stops[0] else 2 if lap <= stops[1] else 3
            age = (lap + start_age) if stint == 1 else lap - stops[stint - 2]
            rows.append(
                {
                    "Driver": "HAM" if d == 0 else f"D{d}",
                    "LapNumber": float(lap),
                    "Stint": float(stint),
                    "Compound": "MEDIUM" if stint != 2 else "HARD",
                    "TyreLife": float(age),
                    "LapTimeSeconds": base
                    + pace
                    - 0.02 * lap
                    + rate * age
                    + rng.normal(0, 0.15),
                }
            )
    return pd.DataFrame(rows)


def test_scores_the_right_number_of_predictions():
    r = score(synth_race())
    # synth_race gives three equal stints across 70 laps, so round(n/4) each.
    assert r.n_predictions == sum(
        int(np.round(len(g) / 4))
        for _, g in synth_race()[lambda d: d["Driver"] == "HAM"].groupby("Stint")
    )
    assert len(r.per_stint_crps) == 3


def test_aggregation_matches_their_tables():
    """CRPS totals are a MEAN across stints; RMSE totals are a SUM."""
    r = score(synth_race())
    assert r.crps == pytest.approx(float(np.mean(r.per_stint_crps)))
    assert r.rmse_total == pytest.approx(float(np.sum(r.per_stint_rmse)))


def test_a_clean_race_is_predicted_well():
    """With known degradation and small noise, the error should be near the noise
    floor. If it is not, the predictor is broken rather than merely imprecise."""
    r = score(synth_race(rate=0.05, seed=1))
    assert r.crps < 0.15


def test_both_predictor_modes_produce_valid_scores():
    """Deliberately NOT asserting that the hybrid beats the field-only mode.

    On real data it does, and by a lot: on Hamilton's 2025 Austrian GP the
    field-only predictor scores CRPS 0.355 and the hybrid 0.238. But that gap
    comes from drivers differing in pace in ways this fixture does not reproduce
    -- here both land near 0.40 and the ordering flips. Asserting it anyway would
    be a test that passes for the wrong reason, or fails for the wrong reason.
    The real-data comparison lives in benchmark.json and the README instead.
    """
    laps = synth_race(seed=2)
    for mode in ("hybrid", "field"):
        r = score(laps, mode=mode)
        assert r.n_predictions > 0
        assert all(np.isfinite(c) and c > 0 for c in r.per_stint_crps)


def test_refuses_an_unknown_driver():
    with pytest.raises(ValueError, match="no laps"):
        score(synth_race(), driver="NOBODY")


def test_predictions_do_not_use_the_lap_being_predicted():
    """The fairness constraint. Corrupting a lap must not change the prediction
    made FOR that lap -- only predictions made after it."""
    laps = synth_race(seed=3)
    baseline = score(laps)

    poisoned = laps.copy()
    last = poisoned["LapNumber"] == 70          # the final test lap
    poisoned.loc[last & (poisoned["Driver"] != "HAM"), "LapTimeSeconds"] += 40.0

    after = score(poisoned)
    # Stints 1 and 2 finish before lap 70, so their scores must be untouched.
    assert after.per_stint_crps[0] == pytest.approx(baseline.per_stint_crps[0])
    assert after.per_stint_crps[1] == pytest.approx(baseline.per_stint_crps[1])


def test_unidentifiable_rate_is_zero_not_explosive():
    """The Saudi 2025 failure, pinned.

    When every car sharing a lap is on the same tyre age there is no within-lap
    variation to read a slope from, and the regression returns whatever the
    numerical noise happens to say. It said 1.0 s/lap -- twenty times any real
    tyre -- and the predictor forecast a whole stint 4.7 seconds slow.

    A field with zero age spread must yield exactly 0.0, not a large number and
    not ``None``: ``None`` would drop the fold, quietly removing the laps we
    predict worst from the test set.
    """
    rng = np.random.default_rng(0)
    rows = []
    for lap in range(1, 41):
        for drv in range(12):
            rows.append(
                {
                    "Driver": f"D{drv}",
                    "LapNumber": lap,
                    "Stint": 1.0,
                    "Compound": "MEDIUM",
                    "TyreLife": float(lap),  # identical for every car, every lap
                    "LapTimeSeconds": 90.0 + 0.05 * lap + rng.normal(0, 0.3),
                }
            )
    assert _fit_rate(pd.DataFrame(rows), "MEDIUM") == 0.0


def test_a_real_age_spread_still_recovers_the_rate():
    """The guard must not silently zero out races that ARE identifiable."""
    rng = np.random.default_rng(1)
    rows = []
    for lap in range(1, 41):
        for drv in range(12):
            age = float(lap + drv * 2)  # cars staggered across 22 laps of age
            rows.append(
                {
                    "Driver": f"D{drv}",
                    "LapNumber": lap,
                    "Stint": 1.0,
                    "Compound": "MEDIUM",
                    "TyreLife": age,
                    "LapTimeSeconds": 90.0 + 0.05 * age + rng.normal(0, 0.2),
                }
            )
    assert _fit_rate(pd.DataFrame(rows), "MEDIUM") == pytest.approx(0.05, abs=0.01)


def test_the_spread_estimator_only_ever_looks_backwards():
    """The predictive spread must not read a lap the forecast could not.

    This pins the new estimator's contract rather than its value. It records
    every (train_to, test_lap) pair _oos_bias_sigma asks for and asserts none
    of them reaches past the training window. A spread that peeked would look
    like a well-calibrated model and be worthless.
    """
    asked: list[tuple[int, int]] = []

    def fake_predict(train_to, test_lap):
        asked.append((int(train_to), int(test_lap)))
        return 90.0, 89.5

    subject = _subject([(1, 30)])
    _oos_bias_sigma(fake_predict, subject, train_to=20)

    assert asked, "estimator asked for nothing"
    for train_to, test_lap in asked:
        assert test_lap <= 20, f"looked at lap {test_lap}, past the window"
        assert train_to < test_lap, "trained on the lap it predicts"


def test_the_spread_is_the_scatter_of_the_forecast_s_own_errors():
    """Pins the fix. The old estimator built residuals from the "field"
    construction while the scored prediction used "hybrid", so the spread
    described a different forecast than the one being made."""
    err = 0.4

    def biased_predict(train_to, test_lap):
        # Constant offset: scatter zero, so sigma must hit its floor, not the bias.
        return 90.0 + err, 90.0

    subject = _subject([(1, 30)])
    bias, sigma = _oos_bias_sigma(biased_predict, subject, train_to=20)
    assert bias == pytest.approx(err, abs=1e-9)
    assert sigma == pytest.approx(0.05, abs=1e-9), "constant error implies no scatter"


def test_too_little_history_falls_back_wide_not_narrow():
    """The first prediction of a race is the one we know least about. A narrow
    guess there is punished by CRPS far harder than a wide one."""
    subject = _subject([(1, 30)])
    bias, sigma = _oos_bias_sigma(lambda a, b: None, subject, train_to=20)
    assert bias == 0.0
    assert sigma == pytest.approx(FALLBACK_SIGMA)

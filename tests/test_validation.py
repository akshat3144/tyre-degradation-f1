"""Tests for the validation layer.

The point of validation code is to tell us when we are wrong, so it has to be
right itself. These check the checks against cases with known answers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.validation.calibration import leave_one_run_out
from tyredeg.validation.power import detectable_effect, power_curve
from tyredeg.validation.scoring import (
    crosscheck_against_r,
    crps_normal,
    rmse_seconds,
    rolling_origin_folds,
    stint_crps_total,
    stint_rmse_total,
)

# --- power ------------------------------------------------------------------


def test_power_rises_with_sample_size():
    pc = power_curve(n_stints_grid=(8, 32, 128, 512), n_sims=250, seed=1)
    assert pc.power[0] < pc.power[-1]
    # Monotone within noise. This caught a real bug: using 1.96 as the critical
    # value instead of the t quantile gave more apparent power at 4 stints than
    # at 8, because with 1 degree of freedom the true critical value is 12.7.
    assert all(b >= a - 0.06 for a, b in zip(pc.power, pc.power[1:], strict=False))


def test_a_tiny_design_has_almost_no_power():
    """The benchmark had three stints. This is why its null result was
    guaranteed rather than informative."""
    pc = power_curve(n_stints_grid=(4,), effect=0.006, n_sims=400, seed=2)
    assert pc.power[0] < 0.15


def test_a_large_effect_is_easy_to_detect():
    pc = power_curve(n_stints_grid=(64,), effect=0.20, n_sims=200, seed=3)
    assert pc.power[0] > 0.9


def test_more_noise_needs_more_data():
    quiet = power_curve(n_stints_grid=(64,), noise=0.2, n_sims=300, seed=4).power[0]
    loud = power_curve(n_stints_grid=(64,), noise=1.0, n_sims=300, seed=4).power[0]
    assert quiet > loud


def test_detectable_effect_shrinks_as_data_grows():
    small = detectable_effect(32, noise=0.5, n_sims=150, seed=5)
    large = detectable_effect(512, noise=0.5, n_sims=150, seed=5)
    assert large is not None
    assert small is None or small > large


def test_power_curve_reports_the_threshold():
    pc = power_curve(n_stints_grid=(8, 64, 512), effect=0.05, n_sims=250, seed=6)
    n = pc.n_for(0.80)
    assert n is None or n in pc.n_stints


# --- scoring ----------------------------------------------------------------


def test_rmse_is_seconds_not_a_percentage():
    """Their metric is named RMSPE but the formula has no division by y."""
    actual = np.array([90.0, 90.0])
    pred = np.array([91.0, 89.0])
    assert rmse_seconds(actual, pred) == pytest.approx(1.0)
    # A percentage error would be about 1.1%, nowhere near 1.0.


def test_the_two_totals_aggregate_differently():
    """Table 1 sums across stints; Table 2 averages. Mixing them up makes our
    numbers look better or worse than theirs for no real reason."""
    per_stint = [0.3, 0.6, 0.15]
    assert stint_rmse_total(per_stint) == pytest.approx(1.05)
    assert stint_crps_total(per_stint) == pytest.approx(0.35)


def test_reproduces_the_published_skew_t_total():
    """Their best model's Table 1 total, 1.082, is the sum of its stint values."""
    assert stint_rmse_total([0.325, 0.601, 0.156]) == pytest.approx(1.082, abs=1e-3)


def test_reproduces_the_published_crps_mean():
    """And their Table 2 total, 0.202, is the mean."""
    assert stint_crps_total([0.184, 0.316, 0.106]) == pytest.approx(0.202, abs=1e-3)


def test_crps_rewards_a_sharper_forecast():
    actual = np.array([90.0])
    sharp = crps_normal(actual, np.array([90.0]), np.array([0.2]))
    vague = crps_normal(actual, np.array([90.0]), np.array([2.0]))
    assert sharp[0] < vague[0]


def test_crps_punishes_a_biased_forecast():
    actual = np.array([90.0])
    good = crps_normal(actual, np.array([90.0]), np.array([0.5]))
    bad = crps_normal(actual, np.array([93.0]), np.array([0.5]))
    assert good[0] < bad[0]


def test_rolling_origin_matches_their_scheme():
    """Train on the first three quarters, then expand one lap at a time."""
    folds = list(rolling_origin_folds(20))
    assert folds[0] == (15, 15)
    assert len(folds) == 5
    assert folds[-1] == (19, 19)


# --- calibration ------------------------------------------------------------


def _runs(n_runs=40, laps=14, slope=0.05, noise=0.4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_runs):
        tl = np.arange(1, laps + 1, dtype=float)
        tl = tl - tl.mean()
        y = slope * tl + rng.normal(0, noise, laps)
        rows.append(pd.DataFrame({"run_id": f"r{r}", "C": "C3", "tl": tl, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_calibration_is_honest_when_the_model_is_correct():
    """Data generated with exactly the assumed noise should come out calibrated."""
    cal = leave_one_run_out(_runs())
    assert cal.coverage_80 == pytest.approx(0.80, abs=0.06)
    assert cal.verdict() == "well calibrated"


def test_calibration_detects_overconfidence():
    """Each run has its own slope, but the model assumes one shared slope.

    This is the failure mode that matters in practice: within-run scatter looks
    small, so the intervals come out narrow, but a held-out run degrades at its
    own rate and falls outside them. Simply making some runs noisier does NOT
    reproduce it -- the pooled residual absorbs that and coverage goes up.
    """
    rng = np.random.default_rng(7)
    rows = []
    for r in range(40):
        tl = np.arange(1, 15, dtype=float)
        tl = tl - tl.mean()
        run_slope = rng.normal(0.05, 0.09)  # real between-run spread
        rows.append(
            pd.DataFrame(
                {
                    "run_id": f"r{r}",
                    "C": "C3",
                    "tl": tl,
                    "y": run_slope * tl + rng.normal(0, 0.12, len(tl)),
                }
            )
        )
    misspecified = leave_one_run_out(pd.concat(rows, ignore_index=True))
    correct = leave_one_run_out(_runs(n_runs=40, laps=14, noise=0.12, seed=7))

    # The tool's job is to notice. A model that ignores real between-run
    # variation must score materially worse than one whose assumptions hold.
    assert misspecified.miscalibration() > 3 * correct.miscalibration()

    # And the failure shows up in the tails, where extrapolating a shared slope
    # to a run that degrades at its own rate does the most damage.
    tail = misspecified.empirical[-1] - misspecified.levels[-1]
    assert tail < -0.01, "99% intervals should miss more often than they claim"


def test_coverage_increases_with_the_nominal_level():
    cal = leave_one_run_out(_runs(n_runs=25))
    assert all(b >= a - 1e-9 for a, b in zip(cal.empirical, cal.empirical[1:], strict=False))


def test_calibration_refuses_when_there_is_nothing_to_fit_on():
    with pytest.raises(ValueError, match="not enough|no runs"):
        leave_one_run_out(_runs(n_runs=2, laps=5))


def test_our_crps_matches_r_scoring_rules():
    """The comparison claims our CRPS is the same number theirs is. Nothing enforced
    that claim until this test: ``crosscheck_against_r`` existed but was never
    called, so the guarantee rested on a function no one ran.

    Skips where R is missing, because the package must not depend on R -- but on
    a machine that has it, the claim is now checked rather than asserted.
    """
    result = crosscheck_against_r()
    if not result.get("available"):
        pytest.skip(f"R not available: {result.get('reason')}")
    for kind in ("normal", "ensemble"):
        assert result[kind]["diff"] < 1e-9, f"{kind} disagrees with R: {result[kind]}"

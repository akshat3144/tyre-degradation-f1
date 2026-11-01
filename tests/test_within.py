"""The exact within-(event, lap) transformation of Eq. 1's fixed effects.

Synthetic races where cars on different compounds share every lap and the
compounds' true slopes differ. The exact construction (every fixed-effect column
demeaned within its cell) recovers the slopes; the earlier one (raw compound dummy
times already-demeaned age) does not. See ``tyredeg.models.within``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.models.mixed import fit_degradation
from tyredeg.models.within import absorb, cell_codes, demean
from tyredeg.paper import model_checks as mc

SLOPES = {"C2": 0.09, "C3": 0.05, "C4": 0.02}
OFFSETS = {"C2": 0.6, "C3": 0.0, "C4": -0.5}


def mixed_race(n_events=3, n_cars=16, n_laps=36, noise=0.0, seed=0, slopes=SLOPES):
    """Cars switch compound at different laps, so every lap mixes compounds and ages.

    ``y`` is RAW lap time: a shared per-lap effect (fuel, evolution), an event
    level, a compound offset, a compound-specific slope on RAW tyre age and noise.
    """
    rng = np.random.default_rng(seed)
    comps = list(slopes)
    rows = []
    for e in range(n_events):
        for d in range(n_cars):
            pit = int(rng.integers(8, n_laps - 8))
            first, second = rng.choice(len(comps), 2, replace=False)
            for lap in range(1, n_laps + 1):
                stint = 0 if lap <= pit else 1
                c = comps[(first, second)[stint]]
                age = lap if stint == 0 else lap - pit
                y = (-0.07 * lap + 2.0 * np.log1p(lap) + 0.4 * e + OFFSETS[c] + slopes[c] * age
                     + rng.normal(0, noise))
                rows.append({"event": f"GP{e}", "Driver": f"D{d}", "run_id": f"GP{e}|D{d}|{stint}",
                             "C": c, "LapNumber": float(lap), "TyreLife": float(age), "y": y})
    df = pd.DataFrame(rows)
    return df


def legacy_frame(df):
    """The frame the earlier construction was fed: y and tyre age demeaned within the cell."""
    out = df.copy()
    g = out.groupby(["event", "LapNumber"])
    out["y"] = out["y"] - g["y"].transform("mean")
    out["tl"] = out["TyreLife"] - g["TyreLife"].transform("mean")
    return out


def ols_slopes(model, meta):
    beta, *_ = np.linalg.lstsq(model.X, model.y, rcond=None)
    return {c: beta[meta["names"].index(("tl", c))] / meta["s1"] for c in meta["cats"]}


FIXED_ONLY = dict(circuit="none", run=False, run_slope=False)


def test_exact_design_recovers_slopes_that_differ_by_compound_and_the_old_one_does_not():
    df = mixed_race()
    m, meta = mc.build_model(df, exact_within=True, **FIXED_ONLY)
    got = ols_slopes(m, meta)
    for c, s in SLOPES.items():
        assert got[c] == pytest.approx(s, abs=1e-9)

    old_m, old_meta = mc.build_model(legacy_frame(df), exact_within=False, **FIXED_ONLY)
    old = ols_slopes(old_m, old_meta)
    worst = max(abs(old[c] - SLOPES[c]) for c in SLOPES)
    assert worst > 0.004, "the earlier construction should be visibly biased on this design"


def test_every_fixed_effect_column_has_mean_zero_within_every_cell():
    df = mixed_race(noise=0.1, seed=1)
    m, meta = mc.build_model(df, **FIXED_ONLY)
    codes, n = cell_codes(df)
    means = np.array([np.bincount(codes, weights=m.X[:, j], minlength=n) / np.bincount(codes, minlength=n)
                      for j in range(m.X.shape[1])])
    assert np.abs(means).max() < 1e-12
    # one reference compound has no offset column: the demeaned dummies sum to zero
    assert [k for k in meta["names"] if k[0] == "off"] == [("off", "C3"), ("off", "C4")]


def test_the_transform_is_computed_on_the_frame_given_so_subsets_are_exact():
    df = mixed_race(noise=0.1, seed=2)
    sub = df[df["Driver"] != "D0"]
    m, meta = mc.build_model(sub, **FIXED_ONLY)
    codes, n = cell_codes(sub.reset_index(drop=True))
    sums = np.column_stack([np.bincount(codes, weights=m.X[:, j], minlength=n) for j in range(m.X.shape[1])])
    assert np.abs(sums).max() < 1e-9
    # the lap-time column is re-demeaned on the subset too
    assert abs(np.bincount(codes, weights=m.y, minlength=n)).max() < 1e-9


def test_fit_degradation_default_is_exact_and_flagged_and_the_old_path_is_still_available():
    df = mixed_race(n_events=4, n_cars=20, noise=0.25, seed=3)
    df = df.assign(tl=0.0, tl2=0.0)  # race_design supplies these; the exact path rebuilds them
    ex = fit_degradation(df, quadratic=False, circuit_effects=True, run_effects=False, run_slopes=False)
    assert ex.exact_within
    legacy = fit_degradation(legacy_frame(df).assign(tl2=0.0), quadratic=False, circuit_effects=True,
                             run_effects=False, run_slopes=False, exact_within=False)
    assert not legacy.exact_within
    err_ex = max(abs(ex.rates[c].mean - SLOPES[c]) for c in SLOPES)
    err_old = max(abs(legacy.rates[c].mean - SLOPES[c]) for c in SLOPES)
    assert err_ex < 0.01
    assert err_old > err_ex + 0.003
    assert set(ex.offsets) == set(SLOPES)  # offsets are reported for every compound, centred on the field
    w = pd.Series(df["C"]).value_counts(normalize=True)
    assert sum(ex.offsets[c].mean * w[c] for c in SLOPES) == pytest.approx(0.0, abs=1e-9)


def test_nested_builder_is_exact_by_default_and_matches_the_flat_one_on_one_season_per_circuit():
    df = mixed_race(noise=0.0)
    df["event_key"] = "2026|" + df["event"]
    df["circuit"] = df["event"]
    m, meta = mc.build_nested_model(df, circuit=False, event=False, run_slope=False)
    got = ols_slopes(m, meta)
    for c, s in SLOPES.items():
        assert got[c] == pytest.approx(s, abs=1e-9)


def test_absorb_removes_two_sets_of_means_to_convergence():
    rng = np.random.default_rng(0)
    g1 = rng.integers(0, 6, 300)
    g2 = rng.integers(0, 9, 300)
    X = rng.normal(size=(300, 3)) + g1[:, None] + 2 * g2[:, None]
    Z = absorb(X, [g1, g2])
    for g in (g1, g2):
        for k in np.unique(g):
            assert np.abs(Z[g == k].mean(axis=0)).max() < 1e-8


def test_a_frame_without_the_raw_columns_says_how_to_use_the_old_construction():
    df = mixed_race().drop(columns=["TyreLife"])
    with pytest.raises(ValueError, match="exact_within=False"):
        mc.build_model(df.assign(tl=0.0), **FIXED_ONLY)

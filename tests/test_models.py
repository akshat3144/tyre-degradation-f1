"""Tests for the design matrices and the degradation model.

Synthetic data with a KNOWN answer, so we can check the estimator recovers what
was put in. Several of these pin bugs that actually happened and cost real time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.models.design import (
    PACE_THRESHOLD,
    add_physical_compound,
    classify_runs,
    drop_non_representative_laps,
    practice_design,
    race_design,
)
from tyredeg.models.mixed import fit_degradation


def synth_race(n_drivers=12, n_laps=40, true_rate=0.05, base=90.0, seed=0):
    """A race where degradation is exactly ``true_rate`` and everything else
    is a lap-level effect shared by the whole field."""
    rng = np.random.default_rng(seed)
    rows = []
    # A big shared per-lap effect: fuel burn plus track evolution. The design
    # must remove this entirely, whatever its shape.
    # Cars are unevenly spaced: the first few run nose-to-tail (0.8s apart, so
    # in dirty air) and the rest are strung out. That gives the traffic
    # covariate real variation, so these tests exercise the production path
    # rather than a degenerate all-zero column.
    lap_effect = {lap: -0.9 * lap + 4.0 * np.log1p(lap) for lap in range(1, n_laps + 1)}
    for d in range(n_drivers):
        pit = 12 + (d % 9)  # drivers stop at different laps -- the identification
        skill = rng.normal(0, 0.3)
        for lap in range(1, n_laps + 1):
            life = lap if lap <= pit else lap - pit
            rows.append(
                {
                    "event": "Test GP",
                    "session": "R",
                    "Driver": f"D{d:02d}",
                    "LapNumber": float(lap),
                    "Stint": 1.0 if lap <= pit else 2.0,
                    "Compound": "MEDIUM",
                    "C": "C3",  # prepare() adds this; these tests call the
                                # design functions directly, so set it here
                    "TyreLife": float(life),
                    "run_id": f"D{d:02d}|{1 if lap <= pit else 2}",
                    "run_lap": float(life),
                    "run_len": 20.0,
                    "is_long_run": True,
                    "LapStartTime": pd.Timedelta(seconds=lap * 95.0 + (0.8 * d if d < 5 else 4.0 + 3.0 * d)),
                    "LapTimeSeconds": base + lap_effect[lap] + true_rate * life
                    + skill + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


# --- compound mapping -------------------------------------------------------


def test_physical_compound_mapping_differs_by_event():
    """The whole point: the same label is different rubber at different races."""
    df = pd.DataFrame(
        {
            "event": ["Monaco Grand Prix", "Japanese Grand Prix"],
            "Compound": ["HARD", "HARD"],
        }
    )
    out = add_physical_compound(df)
    assert out["C"].tolist() == ["C3", "C1"]


def test_unknown_event_maps_to_nothing_rather_than_guessing():
    df = pd.DataFrame({"event": ["Fictional Grand Prix"], "Compound": ["HARD"]})
    assert add_physical_compound(df)["C"].isna().all()


# --- the race design --------------------------------------------------------


def test_race_design_removes_the_shared_lap_effect():
    """Fuel and track evolution are common to the field, so demeaning by
    (event, lap) must remove them however large they are."""
    df = race_design(synth_race())
    # Within any lap, the demeaned values must sum to zero.
    assert df.groupby("LapNumber")["y"].mean().abs().max() < 1e-9
    assert df.groupby("LapNumber")["tl"].mean().abs().max() < 1e-9


def test_race_design_drops_thin_cells():
    df = synth_race(n_drivers=3)
    assert race_design(df, min_cars_per_lap=4).empty


def test_model_recovers_a_known_degradation_rate():
    """End to end on synthetic data: put 0.05 s/lap in, get 0.05 s/lap out,
    despite a lap effect an order of magnitude larger."""
    df = race_design(synth_race(true_rate=0.05))
    fit = fit_degradation(df, quadratic=False, context="race")
    assert fit.rates["C3"].mean == pytest.approx(0.05, abs=0.01)
    assert fit.rates["C3"].lo < 0.05 < fit.rates["C3"].hi


def test_model_recovers_a_different_rate():
    df = race_design(synth_race(true_rate=0.12, seed=3))
    fit = fit_degradation(df, quadratic=False, context="race")
    assert fit.rates["C3"].mean == pytest.approx(0.12, abs=0.015)


# --- the cool-down lap bug --------------------------------------------------


def _push_and_cooldown(n=12, best=84.0):
    """Alternating push and cool-down laps: 84, 112, 85, 121, ...

    These are consecutive, green and accurate, so a naive long-run rule accepts
    them. Fitting a slope through them produced practice estimates around
    0.5 s/lap, roughly ten times any plausible value.
    """
    times = [best + (0.2 * i if i % 2 == 0 else 30 + i) for i in range(n)]
    return pd.DataFrame(
        {
            "event": "Test GP",
            "session": "FP2",
            "Driver": "VER",
            "LapNumber": np.arange(1, n + 1, dtype=float),
            "Stint": 1.0,
            "Compound": "MEDIUM",
            "TyreLife": np.arange(1, n + 1, dtype=float),
            "run_id": "r1",
            "run_lap": np.arange(1, n + 1, dtype=float),
            "run_len": float(n),
            "is_long_run": True,
            "LapTimeSeconds": times,
        }
    )


def test_pace_filter_removes_cooldown_laps():
    kept = drop_non_representative_laps(_push_and_cooldown())
    assert len(kept) == 6, "only the push laps should survive"
    assert kept["LapTimeSeconds"].max() < 84.0 * PACE_THRESHOLD


def test_pace_filter_is_relative_to_the_driver_not_the_session():
    """A slow car's race sim is legitimately >110% of the fastest car's lap.
    Referencing the session best threw away a third of the usable long runs."""
    fast = _push_and_cooldown(n=4, best=80.0).assign(Driver="FAST")
    slow = _push_and_cooldown(n=4, best=88.0).assign(Driver="SLOW")
    kept = drop_non_representative_laps(pd.concat([fast, slow], ignore_index=True))
    assert set(kept["Driver"]) == {"FAST", "SLOW"}, "the slow car must not be filtered out"


def test_consistency_check_rejects_scattered_runs():
    df = _push_and_cooldown()
    df["LapTimeSeconds"] = df["LapTimeSeconds"].astype(float)
    assert not classify_runs(df)["is_race_sim"].any()


def test_a_steady_run_is_accepted_as_a_race_sim():
    n = 10
    df = _push_and_cooldown(n=n)
    df["LapTimeSeconds"] = 90.0 + 0.05 * np.arange(n)  # steady, 6s off an 84s best
    df.loc[0, "LapTimeSeconds"] = 90.0
    df = pd.concat([df, _push_and_cooldown(n=1, best=84.0)], ignore_index=True)
    out = classify_runs(df)
    assert out[out["run_id"] == "r1"]["is_race_sim"].all()


def _sprint(n_drivers=6, n_laps=18, base=90.0, rate=0.06):
    """A Sprint: every car runs one set from lights to flag, at race pace.

    Cars are spread by skill so the leader's laps sit within the race-sim
    classifier's pace threshold of the session best -- because the leader
    IS the session best -- and would be thrown away as qualifying running.
    """
    rows = []
    for d in range(n_drivers):
        skill = 0.4 * d  # 0.0s, 0.4s, 0.8s ... off the leader
        for lap in range(1, n_laps + 1):
            rows.append(
                {
                    "event": "Monaco Grand Prix",  # has a nomination on file
                    "session": "S",
                    "Driver": f"D{d:02d}",
                    "LapNumber": float(lap),
                    "Stint": 1.0,
                    "Compound": "MEDIUM",
                    "TyreLife": float(lap),
                    "TrackStatus": "1",
                    "LapTimeSeconds": base + skill + rate * lap,
                }
            )
    return pd.DataFrame(rows)


def test_every_sprint_lap_is_race_running():
    """The classifier tells a race simulation from a qualifying one by its gap
    to the session's fastest lap. A Sprint has no qualifying runs to tell
    apart, and its leader sits on the session best precisely because it is
    the leader. The front of the field must not be thrown away for winning.
    """
    from tyredeg.models.design import prepare

    df = prepare(_sprint(), "practice", min_runs_per_compound=1)
    assert set(df["Driver"]) == {f"D{d:02d}" for d in range(6)}, (
        "every car's sprint stint survives, the leader's included"
    )
    assert (df["session"] == "S").all()
    assert (df["w"] == 1.0).all(), "the Sprint carries FP2's weight"


# --- practice design --------------------------------------------------------


def test_practice_fuel_correction_has_the_right_sign():
    """Burning fuel makes the car faster, so adding the effect back must make
    later laps look slower, not quicker."""
    df = _push_and_cooldown(n=6)
    df["LapTimeSeconds"] = 90.0
    out = practice_design(df, s_per_kg=0.033)
    assert out["y_fuel_corrected"].iloc[-1] > out["y_fuel_corrected"].iloc[0]


def test_practice_design_centres_within_run():
    df = _push_and_cooldown(n=6)
    df["LapTimeSeconds"] = 90.0 + np.arange(6) * 0.1
    out = practice_design(df)
    assert out.groupby("run_id")["y"].mean().abs().max() < 1e-9


# --- reporting helpers ------------------------------------------------------


def test_rate_at_accounts_for_curvature():
    df = race_design(synth_race())
    fit = fit_degradation(df, quadratic=True, context="race")
    r0 = fit.rate_at("C3", 0)
    r20 = fit.rate_at("C3", 20)
    assert r0 != r20, "with a quadratic the rate must change with tyre age"
    assert fit.rate_at("C3", 0) == pytest.approx(fit.rates["C3"].mean)


def test_total_loss_grows_with_stint_length():
    df = race_design(synth_race(true_rate=0.06))
    fit = fit_degradation(df, quadratic=False, context="race")
    assert fit.total_loss("C3", 20) > fit.total_loss("C3", 10) > 0


def test_monotonicity_check_works():
    df = race_design(synth_race())
    fit = fit_degradation(df, quadratic=False, context="race")
    assert fit.is_monotone(), "a single compound is trivially monotone"


# ---------------------------------------------------------------------------
# single_threaded_blas must not swallow what the body raises
# ---------------------------------------------------------------------------


def test_single_threaded_blas_does_not_swallow_an_import_error_from_its_body():
    """The first version wrapped the ``yield`` in ``try/except ImportError``, so an
    ImportError raised by the caller's own code was caught, then the generator hit a
    second ``yield`` and died with a RuntimeError instead of re-raising."""
    from tyredeg.models.reml import single_threaded_blas

    with pytest.raises(ImportError, match="from the body"):
        with single_threaded_blas():
            raise ImportError("from the body")


def test_single_threaded_blas_runs_the_body_once_without_threadpoolctl(monkeypatch):
    import sys

    from tyredeg.models.reml import single_threaded_blas

    monkeypatch.setitem(sys.modules, "threadpoolctl", None)   # makes `import threadpoolctl` raise ImportError
    calls = []
    with single_threaded_blas():
        calls.append(1)
    assert calls == [1]
    with pytest.raises(ImportError, match="body again"):
        with single_threaded_blas():
            raise ImportError("body again")

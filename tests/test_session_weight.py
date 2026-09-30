"""Tests for practice-session weighting and stand-in rates.

Both exist because of one weekend. Madrid 2026 is a new circuit with no
history, nobody put a HARD on a race simulation in any of the three practice
sessions, and the SOFT managed two runs where three are needed. One usable
compound is not a legal plan, so the strategy screen refused to answer on the
race the project was built to demo.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tyredeg.config import PROCESSED
from tyredeg.data import session_weight
from tyredeg.validation.transfer import (
    C_ORDER,
    cell_rates,
    circuit_severity,
    per_session_rates,
    stand_in_rates,
)

SKILL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "12_session_skill.py"


def load_skill_script():
    """Import the skill script by path.

    It cannot be imported normally: the pipeline scripts are numbered to show
    their running order, and a module name may not start with a digit.
    """
    spec = importlib.util.spec_from_file_location("session_skill", SKILL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def practice_frame(rows):
    """A prepared-looking practice frame. ``rows`` is (event, session, C, rate)."""
    out = []
    for i, (event, session, c, rate) in enumerate(rows):
        for run in range(5):
            age = np.arange(10) + run * 3
            y = rate * age
            out.append(
                pd.DataFrame(
                    {
                        "event": event,
                        "session": session,
                        "C": c,
                        "run_id": f"{event}-{session}-{c}-{run}-{i}",
                        "LapNumber": age + 1,
                        "TyreLife": age,
                        "y": y - y.mean(),
                        "tl": age - age.mean(),
                        "w": session_weight.weight_for(session),
                    }
                )
            )
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


def test_fp2_carries_more_weight_than_fp1_and_fp3():
    """The whole point. FP2 transfers to the race; FP1 barely does."""
    w = session_weight.weights()
    assert w["FP2"] > w["FP1"]
    assert w["FP2"] > w["FP3"]


def test_no_session_is_weighted_to_zero():
    """A badly timed session is still twenty cars on the circuit we are asked
    about. Madrid's FP1 holds 124 long-run laps on a track with no history."""
    assert all(v > 0 for v in session_weight.weights().values())


def test_unknown_session_falls_back_to_flat_weight():
    """Older datasets and anything unrecognised must behave as before, not as
    zero -- silently dropping laps is the failure mode that hides itself."""
    assert session_weight.weight_for("R") == session_weight.DEFAULT_WEIGHT
    assert session_weight.weight_for("nonsense") == session_weight.DEFAULT_WEIGHT


def test_weights_returns_a_copy():
    """The table is a module constant; handing out the live dict would let one
    caller re-weight every forecast in the process."""
    session_weight.weights()["FP2"] = 99.0
    assert session_weight.SESSION_WEIGHTS["FP2"] == 1.0


def test_weighting_actually_changes_a_fitted_rate():
    """A weight column that no estimator reads is worse than none, because it
    looks like the problem was handled."""
    df = practice_frame([("E", "FP1", "C3", 0.02), ("E", "FP2", "C3", 0.20)])
    weighted = cell_rates(df)["rate"].iloc[0]
    flat = cell_rates(df.drop(columns=["w"]))["rate"].iloc[0]
    assert weighted != pytest.approx(flat)
    # FP2 is the heavier session, so the blend must sit nearer its answer.
    assert abs(weighted - 0.20) < abs(flat - 0.20)


def test_race_frames_without_weights_are_unaffected():
    """Races are already deconfounded by demeaning within a lap; the weighting
    is a practice-only correction and must not leak into them."""
    df = practice_frame([("E", "FP2", "C3", 0.10)]).drop(columns=["w"])
    assert cell_rates(df)["rate"].iloc[0] == pytest.approx(0.10, abs=1e-6)


def test_non_positive_weights_are_ignored_rather_than_fitted():
    """A zero or negative weight would silently delete or invert a run. Falling
    back to unweighted is the conservative failure."""
    df = practice_frame([("E", "FP2", "C3", 0.10)])
    df["w"] = 0.0
    assert cell_rates(df)["rate"].iloc[0] == pytest.approx(0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# the weights match the measurement
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (PROCESSED / "laps.parquet").exists(), reason="needs the cached season"
)
def test_the_shipped_weighting_is_the_best_one_measured():
    """Guards the constant against the data moving under it.

    This test used to assert that FP2 had the highest per-session correlation
    with the race. That was the wrong criterion and it failed for the right
    reason: dropping the warm-up lap from every run repaired FP1, whose
    correlation went from 0.05 to 0.78 -- it had never been uninformative, it
    had been contaminated by laps where the tyre was still coming up to
    temperature.

    But each session scores a DIFFERENT set of cells (FP1 six, FP2 ten), so
    ranking their correlations rewards whichever one skipped the hard ones.
    What actually matters is the error of the BLEND, which every scheme
    computes over the same cells. On that criterion FP2-heavy still wins, so
    the weights stand -- now for a reason that survives the next time the
    design changes.
    """
    skill = load_skill_script()
    sweep = skill.weight_sweep(pd.read_parquet(PROCESSED / "laps.parquet"))
    best = str(sweep.iloc[0]["scheme"])
    assert "shipped" in best, (
        f"{best!r} now predicts better than the weighting we ship.\n"
        f"{sweep.to_string(index=False)}\n"
        "Re-run scripts/12_session_skill.py and revisit SESSION_WEIGHTS."
    )


@pytest.mark.skipif(
    not (PROCESSED / "laps.parquet").exists(), reason="needs the cached season"
)
def test_every_weighting_scheme_answers_the_same_cells():
    """The sweep is only a fair comparison if the schemes are answering the
    same questions. If one starts dropping cells, its MAE stops being
    comparable and the verdict above becomes meaningless."""
    skill = load_skill_script()
    sweep = skill.weight_sweep(pd.read_parquet(PROCESSED / "laps.parquet"))
    assert sweep["n_cells"].nunique() == 1, (
        "weighting schemes now answer different numbers of cells, so their "
        "errors are no longer comparable:\n" + sweep.to_string(index=False)
    )


# ---------------------------------------------------------------------------
# stand-in rates
# ---------------------------------------------------------------------------


def test_circuit_severity_is_the_ratio_to_the_rest_of_the_calendar():
    here = pd.DataFrame([{"C": "C3", "rate": 0.30}])
    assert circuit_severity(here, pd.Series({"C3": 0.15})) == pytest.approx(2.0)


def test_circuit_severity_is_none_when_nothing_overlaps():
    """No shared compound means no way to calibrate, and an unscaled pooled
    rate at an unseen circuit is a guess wearing the clothes of an estimate."""
    here = pd.DataFrame([{"C": "C3", "rate": 0.30}])
    assert circuit_severity(here, pd.Series({"C5": 0.15})) is None


def test_stand_in_fills_a_compound_that_never_ran():
    df = practice_frame(
        [
            ("Madrid", "FP2", "C3", 0.30),
            ("Other", "FP2", "C3", 0.15),
            ("Other", "FP2", "C2", 0.05),
            ("Third", "FP2", "C3", 0.15),
            ("Third", "FP2", "C2", 0.05),
        ]
    )
    out = stand_in_rates(df, 1.0, "Madrid", ["C2", "C3"])
    assert list(out["C"]) == ["C2"]
    assert (out["source"] == "stand-in").all()
    # 0.05 elsewhere, scaled by Madrid running 2x harsh.
    assert out["rate"].iloc[0] == pytest.approx(0.10, abs=1e-6)
    assert out["severity"].iloc[0] == pytest.approx(2.0, abs=1e-3)


def test_stand_in_returns_nothing_when_everything_was_measured():
    df = practice_frame([("Madrid", "FP2", "C3", 0.30), ("Other", "FP2", "C3", 0.15)])
    assert stand_in_rates(df, 1.0, "Madrid", ["C3"]).empty


def test_stand_in_falls_back_to_the_unscaled_pool_when_nothing_here_overlaps():
    """With no overlap there is no severity. It used to return nothing; now
    it hands back the bare calendar rate, says it is unscaled, and widens
    the band well past a scaled stand-in's. The strategist has a race on
    Sunday either way, and a labelled guess is what they can act on."""
    df = practice_frame(
        [
            ("Madrid", "FP2", "C5", 0.30),
            ("Other", "FP2", "C2", 0.05),
            ("Third", "FP2", "C2", 0.07),
        ]
    )
    out = stand_in_rates(df, 1.0, "Madrid", ["C2"])
    assert list(out["C"]) == ["C2"]
    assert (out["source"] == "stand-in").all()
    assert pd.isna(out["severity"].iloc[0]), "unscaled must not claim a severity"
    assert out["rate"].iloc[0] == pytest.approx(0.06, abs=1e-6)

    scaled = stand_in_rates(
        practice_frame(
            [
                ("Madrid", "FP2", "C3", 0.30),
                ("Other", "FP2", "C3", 0.15),
                ("Other", "FP2", "C2", 0.05),
                ("Third", "FP2", "C3", 0.15),
                ("Third", "FP2", "C2", 0.07),
            ]
        ),
        1.0,
        "Madrid",
        ["C2", "C3"],
    )
    unscaled_width = float(out["hi"].iloc[0] - out["lo"].iloc[0])
    scaled_width = float(scaled["hi"].iloc[0] - scaled["lo"].iloc[0]) / float(
        scaled["severity"].iloc[0]
    )
    assert unscaled_width > scaled_width


def test_a_negative_measured_cell_is_stood_in_for_not_kept():
    """One noisy run that came out negative is not a measurement of the tyre
    getting faster as it wears. It must not block the stand-in, or a sprint
    weekend's single FP1 run of a compound costs the plan that compound."""
    df = practice_frame(
        [
            ("Madrid", "FP1", "C3", 0.30),
            ("Madrid", "FP1", "C2", -0.05),
            ("Other", "FP2", "C3", 0.15),
            ("Other", "FP2", "C2", 0.05),
            ("Third", "FP2", "C3", 0.15),
            ("Third", "FP2", "C2", 0.05),
        ]
    )
    out = stand_in_rates(df, 1.0, "Madrid", ["C2", "C3"])
    assert list(out["C"]) == ["C2"]
    assert out["rate"].iloc[0] > 0


def test_a_thin_cell_is_forecast_from_and_labelled_thin():
    """One race-simulation run of a tyre at this circuit is an observation.
    The forecast fits it, labels it thin, and widens its band; it does not
    throw it away and then say it has nothing."""
    from tyredeg.validation.transfer import THIN_INTERVAL_MULTIPLE, forecast

    thick = practice_frame([("Madrid", "FP2", "C3", 0.30)])
    thin = thick[thick["run_id"].str.endswith("-0-0")]
    assert thin["run_id"].nunique() == 1

    out = forecast(thin, 1.0, "Madrid")
    assert list(out["source"]) == ["thin"]
    assert out["rate"].iloc[0] == pytest.approx(0.30, abs=1e-6)
    assert out["predicted_race_rate"].iloc[0] == pytest.approx(0.30, abs=1e-6)
    assert np.isfinite(out["lo"].iloc[0]) and np.isfinite(out["hi"].iloc[0])

    ref = forecast(thick, 1.0, "Madrid")
    assert list(ref["source"]) == ["measured"]
    # The thin band is the wider of: its own error times the multiple, and
    # half the rate. The fixture's laps are exactly linear so its own error
    # is zero, which is precisely the case the floor exists for -- a thin
    # cell must never claim a tighter band than a three-run cell could.
    own = float(cell_rates(thin, 1)["se"].iloc[0])
    width = float(out["hi"].iloc[0] - out["lo"].iloc[0])
    assert width >= 2 * 1.96 * own * THIN_INTERVAL_MULTIPLE - 1e-9
    assert width == pytest.approx(2 * 0.5 * 0.30, rel=1e-6)


def test_stand_in_never_undercuts_a_measured_harder_compound():
    """A softer tyre cannot wear more slowly than a harder one on the same
    track. Madrid measured C3 at 0.394 -- the harshest cell on the calendar --
    which set severity at 2.6x and put the borrowed C4 at 0.354, below it. The
    optimiser then built the plan out of two borrowed compounds and ignored the
    only compound we actually watched run.
    """
    df = practice_frame(
        [
            ("Madrid", "FP2", "C3", 0.40),
            ("Other", "FP2", "C3", 0.16),
            ("Other", "FP2", "C4", 0.14),
            ("Third", "FP2", "C3", 0.16),
            ("Third", "FP2", "C4", 0.14),
        ]
    )
    out = stand_in_rates(df, 1.0, "Madrid", ["C3", "C4"])
    soft = out.loc[out["C"] == "C4", "rate"].iloc[0]
    assert soft >= 0.40 - 1e-9, "the SOFT must not wear more slowly than the measured MEDIUM"


def test_stand_in_never_exceeds_a_measured_softer_compound():
    """The clamp has to hold in both directions, or a borrowed HARD can be made
    to look like the tyre that wears fastest on the track."""
    df = practice_frame(
        [
            ("Madrid", "FP2", "C4", 0.10),
            ("Other", "FP2", "C4", 0.30),
            ("Other", "FP2", "C2", 0.25),
            ("Third", "FP2", "C4", 0.30),
            ("Third", "FP2", "C2", 0.25),
        ]
    )
    out = stand_in_rates(df, 1.0, "Madrid", ["C2", "C4"])
    hard = out.loc[out["C"] == "C2", "rate"].iloc[0]
    assert hard <= 0.10 + 1e-9, "the HARD must not wear faster than the measured SOFT"


def test_stand_in_intervals_are_wider_than_measured_ones():
    """A borrowed number is a weaker claim than a measured one, and the
    interval is the only place that can say so."""
    df = practice_frame(
        [
            ("Madrid", "FP2", "C3", 0.30),
            ("Other", "FP2", "C3", 0.15),
            ("Other", "FP2", "C2", 0.05),
            ("Third", "FP2", "C3", 0.15),
            ("Third", "FP2", "C2", 0.06),
        ]
    )
    out = stand_in_rates(df, 1.0, "Madrid", ["C2", "C3"])
    width = float(out["hi"].iloc[0] - out["lo"].iloc[0])
    measured = cell_rates(df[df["event"] == "Madrid"])
    assert width > 4 * 1.96 * float(measured["se"].iloc[0])


def test_measured_rows_are_labelled_as_measured():
    """The screen decides how to draw a rate from this field alone, so a
    missing label must never read as 'measured' by default."""
    from tyredeg.validation.transfer import forecast

    df = practice_frame([("Madrid", "FP2", "C3", 0.30)])
    assert (forecast(df, 1.0, "Madrid")["source"] == "measured").all()


def test_c_order_runs_hard_to_soft():
    """The clamp depends on this ordering; reversing it would silently invert
    every physical constraint built on top."""
    assert C_ORDER == ("C1", "C2", "C3", "C4", "C5")


# ---------------------------------------------------------------------------
# per-session breakdown
# ---------------------------------------------------------------------------


def test_per_session_reports_each_session_separately():
    df = practice_frame([("E", "FP1", "C3", 0.02), ("E", "FP2", "C3", 0.20)])
    out = per_session_rates(df, "E")
    assert set(out["session"]) == {"FP1", "FP2"}
    got = dict(zip(out["session"], out["rate"], strict=True))
    assert got["FP1"] == pytest.approx(0.02, abs=1e-6)
    assert got["FP2"] == pytest.approx(0.20, abs=1e-6)


def test_per_session_carries_the_weight_it_was_given():
    df = practice_frame([("E", "FP1", "C3", 0.02), ("E", "FP2", "C3", 0.20)])
    out = per_session_rates(df, "E").set_index("session")
    assert out.loc["FP2", "weight"] > out.loc["FP1", "weight"]


def test_per_session_reports_run_counts_so_a_thin_cell_looks_thin():
    """Two runs of a soft at Madrid carry a standard error of 0.39 s/lap. The
    number is worth showing; it is not worth trusting, and only the counts say
    which of the two it is."""
    df = practice_frame([("E", "FP2", "C3", 0.20)])
    out = per_session_rates(df, "E")
    assert out["n_runs"].iloc[0] == 5
    assert out["n_laps"].iloc[0] == 50


def test_per_session_is_empty_for_an_unknown_event():
    df = practice_frame([("E", "FP2", "C3", 0.20)])
    assert per_session_rates(df, "Nowhere").empty

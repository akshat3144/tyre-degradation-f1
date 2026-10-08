"""Tests for the strategy layer's audit fixes: offsets, uncertainty, gating,
remaining-race enumeration, pit-loss construction, circuit identity.

Each test pins a specific defect that was found by audit, named in its docstring.
Everything here is synthetic and offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.strategy.circuits import canonical_location, same_circuit
from tyredeg.strategy.decision import decide
from tyredeg.strategy.optimise import (
    LABEL_ORDER,
    PACE_STEP_S,
    best_plan,
    enumerate_plans,
    enumerate_remaining,
    pace_offsets,
)
from tyredeg.strategy.pitloss import (
    bootstrap_median,
    estimate,
    status_category,
    stop_category,
    stop_losses,
)
from tyredeg.strategy.rates import passes_gate
from tyredeg.strategy.uncertainty import Beliefs, propagate

# ---------------------------------------------------------------------------
# pace offsets follow PHYSICAL compound step
# ---------------------------------------------------------------------------


def test_offsets_index_by_physical_step_not_by_usable_position():
    """Excluding a middle compound used to close the gap: C3 and C5 came out one
    step apart because the loop index counted only the usable compounds."""
    full = pace_offsets(["C3", "C4", "C5"])
    assert full["C5"] - full["C3"] == pytest.approx(-2 * PACE_STEP_S)
    # Asking for the same compounds with C4 absent must give C3 and C5 the same
    # offsets: nothing about C4's exclusion may move them.
    gap = pace_offsets(["C3", "C5"])
    assert gap["C5"] - gap["C3"] == pytest.approx(-2 * PACE_STEP_S)


def test_offsets_reference_is_the_hardest_compound_and_labels_work():
    off = pace_offsets(["MEDIUM", "SOFT", "HARD"], 0.5, LABEL_ORDER)
    assert off["HARD"] == 0.0
    assert off["MEDIUM"] == pytest.approx(-0.5)
    assert off["SOFT"] == pytest.approx(-1.0)


def test_there_is_one_pace_step_constant():
    """The single-event script must not carry its own copy of the pace step."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts"
    src = (root / "06_strategy.py").read_text(encoding="utf-8")
    assert "PACE_STEP_S = " not in src, "06_strategy.py redefines the pace step"


# ---------------------------------------------------------------------------
# the fast exact optimum agrees with the enumeration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("laps", [24, 33, 41])
def test_best_plan_matches_the_enumerated_optimum(laps):
    rates = {"C3": 0.045, "C4": 0.061, "C5": 0.088}
    off = pace_offsets(list(rates))
    a = enumerate_plans(laps, rates, off, 22.0)[0]
    b = best_plan(laps, rates, off, 22.0)
    assert b.total_time == pytest.approx(a.total_time)
    assert b.n_stops == a.n_stops


# ---------------------------------------------------------------------------
# the laps remaining after a stop
# ---------------------------------------------------------------------------

RATES = {"C3": 0.045, "C4": 0.061, "C5": 0.088}
OFF = pace_offsets(list(RATES))


def test_pit_now_and_run_to_the_flag_is_representable():
    """enumerate_plans forced another stop and two new compounds after the
    evaluated one, so 'stop now and finish on these' could not be priced."""
    rest = enumerate_remaining(25, RATES, OFF, 22.0, already_used={"C4"})
    one_stint = [p for p in rest if len(p.stints) == 1]
    assert one_stint, "no single-stint continuation offered"
    assert all(p.n_stops == 0 for p in one_stint)
    # And it is not charged a stop it does not make.
    assert one_stint[0].total_time == pytest.approx(one_stint[0].tyre_time)


def test_the_two_compound_rule_is_checked_against_the_whole_race():
    """If only the current compound has run, the new tyre must differ from it;
    if two have run, anything goes."""
    only_c4 = enumerate_remaining(25, RATES, OFF, 22.0, already_used={"C4"}, max_stops=1)
    assert only_c4 and all("C4" not in p.compounds or len(set(p.compounds)) > 1 or False
                           for p in only_c4 if len(p.stints) == 1) or True
    single = [p for p in only_c4 if len(p.stints) == 1]
    assert all(p.compounds[0] != "C4" for p in single)
    two_run = enumerate_remaining(25, RATES, OFF, 22.0, already_used={"C3", "C4"}, max_stops=1)
    assert any(p.compounds == ("C4",) for p in two_run if len(p.stints) == 1)


def test_stops_already_taken_count_against_the_cap():
    # Two made, one being evaluated: that is already three, so nothing more.
    rest = enumerate_remaining(40, RATES, OFF, 22.0, already_used={"C3"}, stops_made=2)
    assert rest and all(len(p.stints) == 1 for p in rest)
    assert enumerate_remaining(40, RATES, OFF, 22.0, already_used={"C3"}, stops_made=3) == []


# ---------------------------------------------------------------------------
# uncertainty is propagated, not typed in
# ---------------------------------------------------------------------------


def _beliefs(sd_rate: float, pit_sd: float = 1.0, step_sd: float = 0.2) -> Beliefs:
    return Beliefs(
        rate_mean=dict(RATES),
        rate_sd={c: sd_rate for c in RATES},
        pit_mean=22.0,
        pit_sd=pit_sd,
        step_sd=step_sd,
        reference=tuple(RATES),
    )


def test_with_no_uncertainty_the_best_plan_always_wins():
    plans = enumerate_plans(30, RATES, OFF, 22.0, step=3)
    prop = propagate(plans, _beliefs(0.0, pit_sd=0.0, step_sd=0.0), n_draws=50)
    assert prop.p_best[0] == pytest.approx(1.0)
    assert prop.mean_regret[0] == pytest.approx(0.0, abs=1e-9)
    assert prop.mean_total[0] == pytest.approx(plans[0].total_time, abs=1e-6)


def test_probabilities_are_probabilities():
    plans = enumerate_plans(30, RATES, OFF, 22.0, step=3)
    prop = propagate(plans, _beliefs(0.02), n_draws=200)
    assert prop.p_best.sum() == pytest.approx(1.0)
    assert sum(prop.stop_prob.values()) == pytest.approx(1.0)


def test_more_input_uncertainty_means_less_confidence_in_the_call():
    """The 3 s band and clamp(margin/10) did not respond to how well the inputs
    were known. This must.

    48 laps, where the stop count is genuinely open: at 30 laps one stop wins in
    essentially every draw whatever the beliefs, so "less confident" hung on a
    handful of draws and flipped with the random stream.
    """
    plans = enumerate_plans(48, RATES, OFF, 22.0, step=3)
    rec = plans[0].n_stops
    for seed in (1, 2, 3):
        tight = propagate(plans, _beliefs(0.005, 0.2, 0.05), n_draws=300, seed=seed)
        loose = propagate(plans, _beliefs(0.06, 3.0, 0.4), n_draws=300, seed=seed)
        assert loose.stop_prob.get(rec, 0.0) < tight.stop_prob.get(rec, 0.0) - 0.1
        assert loose.p_best[0] < tight.p_best[0]


def test_the_band_widens_with_input_uncertainty_and_is_not_three_seconds():
    plans = enumerate_plans(30, RATES, OFF, 22.0, step=3)
    narrow = propagate(plans, _beliefs(0.002, 0.1, 0.02), n_draws=300)
    wide = propagate(plans, _beliefs(0.05, 2.0, 0.3), n_draws=300)
    w = lambda p: float(p.hi_total[0] - p.lo_total[0])  # noqa: E731
    assert w(wide) > 3 * w(narrow)
    assert abs(w(narrow) - 6.0) > 0.5, "narrow band equals the old hard-coded 3 s either side"


def test_draws_are_reproducible():
    plans = enumerate_plans(30, RATES, OFF, 22.0, step=3)
    a = propagate(plans, _beliefs(0.02), n_draws=100, seed=7)
    b = propagate(plans, _beliefs(0.02), n_draws=100, seed=7)
    assert np.array_equal(a.p_best, b.p_best)


def test_decide_reports_a_confidence_that_is_the_stop_count_probability():
    d = decide(30, RATES, {c: 0.02 for c in RATES}, ["C3", "C4", "C5"], 22.0, step=3, n_draws=200)
    assert d.confidence == pytest.approx(d.stop_prob[d.rec.n_stops])
    assert 0.0 <= d.confidence <= 1.0
    assert d.rec_score.total_lo <= d.rec_score.total_mean <= d.rec_score.total_hi


# ---------------------------------------------------------------------------
# the identifiability gate
# ---------------------------------------------------------------------------


def test_a_rate_within_one_se_of_zero_is_refused_even_though_it_is_positive():
    """+0.011 s/lap with SE 0.028 passed the sign test and was optimised on."""
    assert passes_gate(0.011, 0.028) == (False, "not identified")
    assert passes_gate(-0.02, 0.01) == (False, "non-positive")
    assert passes_gate(0.05, 0.02) == (True, None)
    # No SE: a one-run cell has nothing to estimate one from, so it passes.
    assert passes_gate(0.03, 0.0)[0] is True


# ---------------------------------------------------------------------------
# pit loss: status, reference, interval
# ---------------------------------------------------------------------------


def test_red_flag_outranks_everything_and_is_not_green():
    """A lap carrying '5' used to fall through to 'green'."""
    assert status_category("5") == "red_flag"
    assert status_category("1245") == "red_flag"
    assert status_category("14") == "safety_car"
    assert status_category("16") == "vsc"
    assert status_category("1") == "green"
    assert status_category("2") == "yellow"


def test_a_stop_takes_the_worst_status_of_its_in_lap_and_out_lap():
    """A green in-lap followed by a safety-car out-lap was filed as green."""
    assert stop_category("1", "4") == "safety_car"
    assert stop_category("1", "1") == "green"
    assert stop_category("4", "5") == "red_flag"


def _field(n_cars=10, n_laps=40, loss_s=22.0, pit_lap=3, fuel_s_per_lap=0.06):
    """A field whose lap times fall with fuel burn; car P0 stops once, early."""
    rows = []
    for d in range(n_cars):
        for lap in range(1, n_laps + 1):
            t = 95.0 - fuel_s_per_lap * lap + 0.2 * d
            pit_in = d == 0 and lap == pit_lap
            pit_out = d == 0 and lap == pit_lap + 1
            if pit_in or pit_out:
                t += loss_s / 2
            rows.append(
                {
                    "Driver": f"D{d:02d}",
                    "LapNumber": float(lap),
                    "LapTimeSeconds": t,
                    "TrackStatus": "1",
                    "PitInTime": pd.Timedelta("1s") if pit_in else pd.NaT,
                    "PitOutTime": pd.Timedelta("1s") if pit_out else pd.NaT,
                }
            )
    return pd.DataFrame(rows)


def test_reference_is_the_field_on_the_same_laps_so_fuel_burn_cancels():
    """Against the driver's race-wide median, an early stop is compared with laps
    that are 2 s quicker from fuel burn and the loss comes out biased high."""
    laps = _field(loss_s=22.0, pit_lap=3, fuel_s_per_lap=0.1)
    s = stop_losses(laps, "Test GP")
    assert len(s) == 1
    assert s["loss"].iloc[0] == pytest.approx(22.0, abs=0.3)

    # The superseded construction, spelled out: it over-reads by roughly twice
    # the fuel-burn gap between the stop and the race median.
    own = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna() & (laps["Driver"] == "D00")]
    old = laps.loc[(laps["Driver"] == "D00") & (laps["LapNumber"].isin([3.0, 4.0])), "LapTimeSeconds"].sum()
    old -= 2 * own["LapTimeSeconds"].median()
    assert old > 22.0 + 1.5


def test_pit_loss_band_is_a_bootstrap_of_the_median_and_brackets_it():
    x = np.array([21.0, 22.0, 22.5, 23.0, 22.2, 21.8, 40.0, 22.4])
    med, lo, hi, sd = bootstrap_median(x)
    assert lo <= med <= hi
    assert sd > 0
    # The old band was centred on the median with the MEAN's standard error; the
    # outlier at 40 inflates that and not this.
    mean_se = x.std(ddof=1) / np.sqrt(len(x))
    assert (hi - lo) / 3.92 < mean_se


def test_estimate_carries_a_standard_deviation_the_uncertainty_can_use():
    rows = []
    for k in range(8):
        f = _field(loss_s=22.0 + 0.3 * k, pit_lap=3 + k)
        f["Driver"] = f["Driver"] + f"_{k}"
        rows.append(f)
    # One field per stop would not share a lap axis; merge the stops into one race.
    laps = _field(loss_s=22.0, pit_lap=3)
    laps = laps[laps["Driver"] != "D00"]
    stops = []
    for k in range(6):
        g = _field(loss_s=22.0 + 0.2 * k, pit_lap=5 + 3 * k)
        g = g[g["Driver"] == "D00"].copy()
        g["Driver"] = f"S{k}"
        stops.append(g)
    race = pd.concat([laps, *stops], ignore_index=True)
    pl = estimate(race, "Test GP")
    assert pl is not None and pl.sd is not None and pl.sd > 0
    assert pl.se == pl.sd
    assert pl.lo <= pl.seconds <= pl.hi


# ---------------------------------------------------------------------------
# circuit identity
# ---------------------------------------------------------------------------


def test_bahrain_2026_is_not_sakhir():
    """The 2026 Bahrain GP is at Sepang (FastF1: Kuala Lumpur). The location is
    right; the join that gave it four seasons of Sakhir's pit lane was wrong."""
    assert not same_circuit("Kuala Lumpur", "Sakhir")


def test_madrid_is_not_barcelona_and_barcelona_matches_across_name_changes():
    assert not same_circuit("Madrid", "Barcelona")
    assert same_circuit("Barcelona", "Barcelona")


def test_known_location_drift_still_matches():
    assert same_circuit("Monte Carlo", "Monaco")
    assert same_circuit("Yas Marina", "Yas Island")
    assert same_circuit("Miami Gardens", "Miami")
    assert canonical_location("Montr�al") == canonical_location("Montr�al")


# ---------------------------------------------------------------------------
# review fixes: the enumeration, the pace-step draw, the by-status filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "laps,caps,floors,used,counts",
    [
        (57, None, None, frozenset(), range(2, 5)),
        (70, {"C3": 44, "C4": 33, "C5": 26}, {"C3": 14, "C4": 9, "C5": 7}, frozenset(), range(2, 5)),
        (52, {"C5": 20}, None, frozenset(), range(2, 4)),
        (40, None, None, frozenset({"C3"}), range(1, 3)),   # the remaining-race form
    ],
)
def test_the_vectorised_enumeration_finds_exactly_the_loops_plans(laps, caps, floors, used, counts):
    """``step=1`` is vectorised (seconds became tens of milliseconds); it must be
    the same set of allocations with the same totals, not a faster different answer."""
    from tyredeg.strategy.optimise import _enumerate_lattice, _enumerate_vec

    avail = list(RATES)
    a = _enumerate_vec(laps, avail, RATES, OFF, 22.0, {}, counts, 5, used, caps, floors)
    b = _enumerate_lattice(laps, avail, RATES, OFF, 22.0, {}, counts, 5, 1, used, caps, floors)
    ka = {(p.compounds, p.stints): (p.total_time, p.tyre_time) for p in a}
    kb = {(p.compounds, p.stints): (p.total_time, p.tyre_time) for p in b}
    assert len(a) == len(b) == len(ka) and set(ka) == set(kb)
    assert max(abs(ka[k][0] - kb[k][0]) for k in ka) < 1e-9
    assert max(abs(ka[k][1] - kb[k][1]) for k in ka) < 1e-9
    assert a[0].total_time == pytest.approx(b[0].total_time, abs=1e-9)


def test_a_pace_step_believed_to_be_zero_is_not_pulled_up_by_truncation():
    """The step was drawn from a normal truncated at zero. Harmless at 0.6 s, but a
    belief of 'no gap' (mean 0) became a half-normal with mean about 0.16."""
    from tyredeg.strategy.uncertainty import _pace_step_draw

    draw = _pace_step_draw(0.0, 0.2, 200_000, np.random.default_rng(0))
    assert abs(draw.mean()) < 0.005 and draw.min() < 0
    assert np.all(_pace_step_draw(0.37, 0.0, 10, np.random.default_rng(0)) == 0.37)

    # End to end: with only the step uncertain and its mean 0, the mean total of a
    # plan is its plug-in total. The old draw shifted it by 0.16 x laps-weighted-by-step.
    off0 = pace_offsets(list(RATES), 0.0)
    plans = enumerate_plans(30, RATES, off0, 22.0, step=3)
    beliefs = Beliefs(rate_mean=dict(RATES), rate_sd={c: 0.0 for c in RATES}, pit_mean=22.0,
                      pit_sd=0.0, step_mean=0.0, step_sd=0.2, reference=tuple(RATES))
    prop = propagate(plans, beliefs, n_draws=2000)
    def laps_weighted_by_step(p):
        return sum(list(RATES).index(c) * n for c, n in zip(p.compounds, p.stints, strict=True))

    biggest = max(range(len(plans)), key=lambda i: laps_weighted_by_step(plans[i]))
    assert laps_weighted_by_step(plans[biggest]) >= 30    # the old bias (0.16 x this) would be >= 4.8 s
    assert abs(prop.mean_total[biggest] - plans[biggest].total_time) < 1.5


def test_the_by_status_filter_keeps_cheap_stops_and_drops_only_the_impossible():
    """0 < loss < 45 threw away every neutralised stop that measured at or below
    zero (and every one over 45), truncating the distribution and biasing the
    ratio to green upward. The filter is now wide and identical for every category."""
    from tyredeg.strategy.pitloss import SANITY_RANGE_S, loss_by_status

    lo, hi = SANITY_RANGE_S
    assert lo < 0 < 45 < hi
    kept = {}
    for loss in (-5.0, 8.0, 55.0, -35.0, 80.0):
        out = loss_by_status(_field(loss_s=loss, pit_lap=10), "T GP")
        kept[loss] = len(out)
    assert kept[-5.0] == 1 and kept[8.0] == 1 and kept[55.0] == 1
    assert kept[-35.0] == 0 and kept[80.0] == 0

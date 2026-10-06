"""Tests for the strategy layer.

An optimiser is dangerous in a way a model is not: it will exploit any flaw in
its inputs and present the result with complete confidence. Several of these
tests exist because that happened.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tyredeg.strategy.optimise import (
    MIN_STINT_LAPS,
    best_per_stop_count,
    crossover,
    enumerate_plans,
    optimal_stint,
    stint_time,
)
from tyredeg.strategy.pitloss import PLAUSIBLE_RANGE_S, estimate

RATES = {"C2": 0.045, "C3": 0.062, "C4": 0.085}
OFFSETS = {"C2": 0.0, "C3": -0.60, "C4": -1.20}


# --- stint time -------------------------------------------------------------


def test_stint_time_matches_a_lap_by_lap_sum():
    """The closed form must equal the loop it replaces."""
    rate, curv, n = 0.06, -0.001, 25
    brute = sum(rate * a + curv * a**2 for a in range(1, n + 1))
    assert stint_time(n, rate, 0.0, curv) == pytest.approx(brute)


def test_stint_time_grows_faster_than_linearly():
    """Twice the laps costs more than twice the time, because every lap is
    slower than the one before."""
    a, b = stint_time(10, 0.06), stint_time(20, 0.06)
    assert b > 2 * a


def test_offset_applies_once_per_lap():
    assert stint_time(10, 0.0, offset=-0.5) == pytest.approx(-5.0)


def test_empty_stint_costs_nothing():
    assert stint_time(0, 0.06) == 0.0


# --- refusing bad inputs ----------------------------------------------------


def test_refuses_a_compound_that_does_not_degrade():
    """Our fitted C5 rate was -0.004 s/lap. The optimiser read that as a tyre
    which gets faster with age and recommended running it for 65 laps. It must
    refuse instead."""
    with pytest.raises(ValueError, match="non-physical|positive"):
        enumerate_plans(70, {"C4": 0.028, "C5": -0.004}, {"C4": 0.0, "C5": -0.6}, 22.3)


def test_refuses_when_only_one_compound_is_usable():
    with pytest.raises(ValueError, match="at least 2"):
        enumerate_plans(70, {"C3": 0.06}, {"C3": 0.0}, 22.3)


def test_error_names_which_compound_was_rejected():
    with pytest.raises(ValueError, match="C5"):
        enumerate_plans(70, {"C4": 0.03, "C5": 0.0}, {"C4": 0.0, "C5": -0.6}, 22.3)


# --- legality ---------------------------------------------------------------


def test_every_plan_uses_at_least_two_compounds():
    """A single-compound dry race is illegal, not merely slow."""
    plans = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)
    assert plans
    assert all(len(set(p.compounds)) >= 2 for p in plans)


def test_every_plan_covers_the_full_race():
    plans = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)
    assert all(sum(p.stints) == 60 for p in plans)


def test_no_plan_proposes_a_token_stint():
    plans = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)
    assert all(min(p.stints) >= MIN_STINT_LAPS for p in plans)


def test_a_race_too_short_to_split_yields_nothing():
    assert enumerate_plans(6, RATES, OFFSETS, 22.0) == []


# --- the economics ----------------------------------------------------------


def test_a_harder_tyre_earns_a_longer_stint():
    """The central trade-off. Softer degrades faster, so it should be changed
    sooner. If this inverts, the sign of something is wrong."""
    lengths = {c: optimal_stint(c, r, 22.0) for c, r in RATES.items()}
    assert lengths["C2"] > lengths["C3"] > lengths["C4"]


def test_a_costlier_pit_stop_earns_a_longer_stint():
    cheap = optimal_stint("C3", 0.062, pit_loss_s=15.0)
    dear = optimal_stint("C3", 0.062, pit_loss_s=30.0)
    assert dear > cheap


def test_expensive_stops_push_towards_fewer_of_them():
    cheap = enumerate_plans(60, RATES, OFFSETS, 12.0, step=5)[0]
    dear = enumerate_plans(60, RATES, OFFSETS, 40.0, step=5)[0]
    assert dear.n_stops <= cheap.n_stops


def test_crossover_is_inside_the_swept_range():
    plans = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)
    x = crossover(plans, pit_loss_range=(10.0, 45.0))
    assert x is None or 10.0 <= x <= 45.0


def test_crossover_uses_tyre_time_not_total_time():
    """total_time already contains the pit loss it was scored with, so using it
    to sweep pit loss would double-count and move the crossover."""
    plans = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)
    a = crossover(plans, pit_loss_range=(10.0, 45.0))
    # Re-score the same strategies at a different pit loss. The crossover is a
    # property of the tyres, so it must not shift.
    plans_b = enumerate_plans(60, RATES, OFFSETS, 31.0, step=5)
    b = crossover(plans_b, pit_loss_range=(10.0, 45.0))
    assert a == pytest.approx(b, abs=1.5)


def test_best_per_stop_count_picks_the_cheapest_of_each():
    plans = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)
    best = best_per_stop_count(plans)
    for n, plan in best.items():
        same = [p for p in plans if p.n_stops == n]
        assert plan.total_time == pytest.approx(min(p.total_time for p in same))


def test_plans_come_back_sorted():
    plans = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)
    times = [p.total_time for p in plans]
    assert times == sorted(times)


# --- the recommendation must not depend on the hash seed --------------------


def test_stint_order_never_changes_the_total():
    """The fact that licenses everything else here.

    Every stint starts on a fresh tyre, so a plan's total is a sum over
    unordered (compound, laps) pairs. Resequencing cannot change it, which is
    why the optimiser cannot rank sequence and why the enumerator is free to
    keep one plan per allocation.
    """
    from itertools import permutations as _perms

    for pl in enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)[:200]:
        pairs = list(zip(pl.compounds, pl.stints, strict=True))
        totals = {
            round(
                sum(stint_time(n, RATES[c], OFFSETS[c]) for c, n in order),
                12,
            )
            for order in _perms(pairs)
        }
        assert len(totals) == 1, f"{pl.describe()} scored {len(totals)} ways by order"
    # Not vacuous: plans with more than one stint really were checked.
    assert any(len(pl.stints) > 1 for pl in enumerate_plans(60, RATES, OFFSETS, 22.0, step=5))


def test_one_plan_per_allocation_not_one_per_sequence():
    """Six orderings of the same allocation used to tie for first, and which one
    was reported followed the hash seed. Collapsing them removes the coin flip
    at the source rather than making it repeatable."""
    plans = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)
    seen = [tuple(sorted(zip(p.compounds, p.stints, strict=True))) for p in plans]
    assert len(seen) == len(set(seen)), "the same allocation was enumerated twice"


def test_the_printed_plan_does_not_claim_a_running_order():
    """An arrow would assert a sequence the model has no term for -- track
    position, traffic, the undercut, warm-up, safety-car risk are all absent."""
    best = enumerate_plans(60, RATES, OFFSETS, 22.0, step=5)[0]
    assert "->" not in best.describe()
    assert "→" not in best.describe()


def test_the_best_plan_is_the_same_under_a_different_hash_seed():
    """The bug this pins.

    `permutations` was iterated through a bare set, so the winner among tied
    plans followed PYTHONHASHSEED. The same data recommended starting on C2 in
    one process and on C3 in the next -- opposite calls, presented with equal
    confidence. Subprocesses because the seed is fixed at interpreter start.
    """
    import json
    import subprocess
    import sys

    script = (
        "import json;from tyredeg.strategy.optimise import enumerate_plans;"
        f"p=enumerate_plans(60,{RATES!r},{OFFSETS!r},22.0,step=5)[0];"
        "print(json.dumps([p.compounds,p.stints]))"
    )
    seen = set()
    for seed in ("0", "1", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        seen.add(json.dumps(json.loads(out.stdout)))
    assert len(seen) == 1, f"three hash seeds gave {len(seen)} different plans: {seen}"


# --- pit loss ---------------------------------------------------------------


def race_with_stops(loss_s=22.0, base=90.0, n_drivers=6, n_laps=20, pit_lap=10):
    rows = []
    for d in range(n_drivers):
        for lap in range(1, n_laps + 1):
            in_lap = lap == pit_lap
            out_lap = lap == pit_lap + 1
            t = base + (loss_s / 2 if in_lap or out_lap else 0.0)
            rows.append(
                {
                    "Driver": f"D{d}",
                    "LapNumber": float(lap),
                    "LapTimeSeconds": t,
                    "TrackStatus": "1",
                    "in_lap": in_lap,
                    "out_lap": out_lap,
                }
            )
    df = pd.DataFrame(rows)
    # Build the timedelta columns with an explicit dtype rather than mixing
    # Timedelta and NaT into an object column, which pandas warns about.
    one = pd.Timedelta("1s")
    df["PitInTime"] = pd.Series(
        [one if v else None for v in df["in_lap"]], dtype="timedelta64[ns]"
    )
    df["PitOutTime"] = pd.Series(
        [one if v else None for v in df["out_lap"]], dtype="timedelta64[ns]"
    )
    return df.drop(columns=["in_lap", "out_lap"])


def test_pit_loss_recovers_a_known_value():
    pl = estimate(race_with_stops(loss_s=22.0), "Test GP")
    assert pl is not None
    assert pl.seconds == pytest.approx(22.0, abs=0.5)
    assert pl.n_stops == 6


def test_safety_car_stops_are_excluded():
    """Under a safety car the whole field pits cheaply, so those stops say
    nothing about the cost of a normal one."""
    df = race_with_stops()
    df.loc[df["PitInTime"].notna(), "TrackStatus"] = "4"
    assert estimate(df, "Test GP") is None


def test_implausible_losses_are_rejected():
    """A 90-second loss is a drive-through or damage repair, not a pit stop."""
    assert estimate(race_with_stops(loss_s=200.0), "Test GP") is None
    assert PLAUSIBLE_RANGE_S[1] < 200.0


def test_pit_loss_needs_several_stops():
    assert estimate(race_with_stops(n_drivers=1), "Test GP") is None


def test_pit_loss_is_relative_to_each_driver_so_a_slow_car_is_not_penalised():
    # Distinct names per car: the loss is keyed on (driver, lap), so four cars
    # sharing one name would be one car with four rows per lap.
    fast = race_with_stops(base=80.0, n_drivers=4)
    fast["Driver"] = "FAST" + fast["Driver"]
    slow = race_with_stops(base=95.0, n_drivers=4)
    slow["Driver"] = "SLOW" + slow["Driver"]
    pl = estimate(pd.concat([fast, slow], ignore_index=True), "Test GP")
    assert pl is not None
    assert pl.seconds == pytest.approx(22.0, abs=0.5)


def test_pit_loss_uses_the_median_so_one_bad_stop_does_not_move_it():
    df = race_with_stops(loss_s=22.0, n_drivers=8)
    # One botched stop, still inside the plausible range.
    bad = (df["Driver"] == "D0") & df["PitInTime"].notna()
    df.loc[bad, "LapTimeSeconds"] += 15.0
    pl = estimate(df, "Test GP")
    assert pl.seconds == pytest.approx(22.0, abs=1.0)


def test_stint_time_is_finite_for_a_long_race():
    assert np.isfinite(stint_time(78, 0.06, -0.5, -0.0005))

"""The stint-length support bounds: what counts as a real stint, and what a floor is.

Offline and synthetic. The defect these pin: the floor was the 5th percentile of
EVERY FastF1 "Stint", fragments of red-flag restarts and safety-car stop/starts
included, so on the real data it came out at 10, 3 and 2 laps for HARD, MEDIUM and
SOFT -- under the five-lap legal minimum, binding nothing -- and the shipped plans
still carried five-lap filler stints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.strategy import support as S
from tyredeg.strategy.optimise import MIN_STINT_LAPS


def _car(season, event, driver, stints, sc_laps=(), red_laps=(), race_len=None):
    """Raw laps for one car. ``stints`` is [(compound, n_laps)]; every stint after the
    first starts with a pit exit and every one but the last ends with a pit entry."""
    rows, lap = [], 1
    for k, (compound, n) in enumerate(stints):
        for i in range(n):
            status = "5" if lap in red_laps else ("4" if lap in sc_laps else "1")
            rows.append({
                "season": season, "event": event, "Driver": driver, "LapNumber": float(lap),
                "Stint": float(k + 1), "Compound": compound, "TrackStatus": status,
                "PitIn_s": 1.0 if (i == n - 1 and k < len(stints) - 1) else np.nan,
                "PitOut_s": 1.0 if (i == 0 and k > 0) else np.nan,
            })
            lap += 1
    return rows


def _frame(*cars):
    return pd.DataFrame([r for c in cars for r in c])


def _pool(df):
    return S.observed_stints(df)


# ---------------------------------------------------------------------------
# a floor is a floor
# ---------------------------------------------------------------------------


def _fragment_laden(frac=0.3, seed=0):
    """What the old pool looked like: real stints plus a lot of 1-4 lap fragments."""
    rng = np.random.default_rng(seed)
    real = [("MEDIUM", int(x)) for x in rng.integers(14, 30, 70)]
    junk = [("MEDIUM", int(x)) for x in rng.integers(1, 5, int(70 * frac / (1 - frac)))]
    return real + junk


def test_the_raw_fragment_pool_would_put_the_floor_under_the_legal_minimum():
    """The defect, stated as arithmetic: on a fragment-laden pool the 5th percentile
    is under five laps, so an unguarded floor could never remove a five-lap stint."""
    pool = _fragment_laden()
    assert np.floor(np.quantile([n for _, n in pool], S.FLOOR_QUANTILE)) < MIN_STINT_LAPS


def test_a_floor_is_never_below_the_legal_minimum_plus_its_margin():
    floors = S.support_floor(_fragment_laden())
    assert floors["MEDIUM"] >= S.MIN_FLOOR_LAPS == MIN_STINT_LAPS + S.FLOOR_MARGIN_LAPS
    assert floors["MEDIUM"] > MIN_STINT_LAPS + 1, "a floor must clear the evaluation's 'on the boundary' definition"


def test_a_clean_pool_sets_the_floor_where_the_field_actually_stops():
    rng = np.random.default_rng(1)
    pool = [("HARD", int(x)) for x in rng.integers(18, 40, 200)]
    assert S.support_floor(pool)["HARD"] == pytest.approx(np.quantile([n for _, n in pool], 0.05), abs=1.01)
    assert S.support_floor(pool)["HARD"] > S.MIN_FLOOR_LAPS


# ---------------------------------------------------------------------------
# which stints are real
# ---------------------------------------------------------------------------


def test_a_clean_two_stop_race_yields_genuine_stints_and_the_last_is_ceiling_only():
    df = _frame(_car(2025, "A GP", "D1", [("MEDIUM", 18), ("HARD", 20), ("MEDIUM", 12)]))
    p = _pool(df).set_index("n")
    assert bool(p.loc[18, "ceiling_ok"]) and bool(p.loc[18, "floor_ok"])
    assert bool(p.loc[20, "ceiling_ok"]) and bool(p.loc[20, "floor_ok"])
    # The final stint to the flag is set by the laps left, not by the tyre: it
    # informs the ceiling and must not set the floor.
    assert bool(p.loc[12, "ceiling_ok"]) and not bool(p.loc[12, "floor_ok"])


def test_stints_that_touch_a_safety_car_are_fragments_not_stints():
    df = _frame(_car(2025, "A GP", "D1", [("MEDIUM", 18), ("HARD", 2), ("HARD", 30)], sc_laps={19, 20}))
    p = _pool(df)
    assert p.loc[p["n"] == 18, "ceiling_ok"].all() and p.loc[p["n"] == 30, "ceiling_ok"].all()
    short = p[p["n"] == 2].iloc[0]
    assert not short["ceiling_ok"] and not short["floor_ok"], "a stop/start under a safety car is not a stint"


def test_a_stint_under_the_legal_minimum_never_sets_a_floor():
    # Green throughout: only the length excludes it (a repair, a penalty, a first-lap incident).
    df = _frame(_car(2025, "A GP", "D1", [("MEDIUM", 20), ("HARD", 3), ("HARD", 27)]))
    p = _pool(df)
    assert not p.loc[p["n"] == 3, "floor_ok"].iloc[0]
    assert p.loc[p["n"] == 20, "floor_ok"].iloc[0]


def test_a_stint_that_does_not_start_at_the_start_or_a_pit_exit_is_a_fragment():
    # A restart after a red flag: FastF1 starts a new Stint with no pit in or out.
    rows = _car(2025, "A GP", "D1", [("MEDIUM", 40)])
    for r in rows[20:]:
        r["Stint"] = 2.0
    p = _pool(pd.DataFrame(rows))
    first = p[p["n"] == 20].iloc[0]
    # The first piece starts at the race start but ends with no pit entry (and is not final).
    assert not first["ceiling_ok"] and not first["floor_ok"]
    # The second starts with no pit exit.
    assert not p.iloc[1]["ceiling_ok"] and not p.iloc[1]["floor_ok"]


def test_red_flag_and_wet_races_contribute_nothing():
    red = _frame(_car(2024, "Red GP", "D1", [("MEDIUM", 20), ("HARD", 30)], red_laps={25}))
    wet = _frame(_car(2024, "Wet GP", "D1", [("INTERMEDIATE", 20), ("HARD", 30)]))
    assert _pool(red).empty and _pool(wet).empty


def test_a_car_that_did_not_run_the_distance_contributes_nothing():
    long = _car(2025, "A GP", "D1", [("MEDIUM", 20), ("HARD", 30)])
    dnf = _car(2025, "A GP", "D2", [("MEDIUM", 20), ("HARD", 10)])  # out on lap 30 of 50
    p = _pool(_frame(long, dnf))
    assert set(p["n"]) == {20, 30}


def test_stint_length_counts_in_and_out_laps():
    """Clean-lap counting (the old input) drops both: a 20-lap stint reads as 18."""
    df = _frame(_car(2025, "A GP", "D1", [("MEDIUM", 20), ("HARD", 30)]))
    assert 20 in set(_pool(df)["n"])


# ---------------------------------------------------------------------------
# using the pool
# ---------------------------------------------------------------------------


def _big_pool():
    rng = np.random.default_rng(3)
    rows = []
    for season in (2024, 2025):
        for ev in ("A GP", "B GP"):
            for lab, lo, hi in (("SOFT", 8, 26), ("MEDIUM", 12, 34), ("HARD", 16, 44)):
                for n in rng.integers(lo, hi, 40):
                    rows.append((season, ev, lab, int(n), True, True))
    return pd.DataFrame(rows, columns=["season", "event", "label", "n", "ceiling_ok", "floor_ok"])


def test_bounds_leave_out_only_the_named_race_and_season():
    pool = _big_pool()
    label_of = {"C5": "SOFT", "C4": "MEDIUM", "C3": "HARD"}
    caps_all, _ = S.bounds_excluding(pool, "Z GP", label_of)
    # Plant an extreme race in 2025 A GP: excluding that race (and only it) must remove it.
    planted = pd.concat([pool, pd.DataFrame(
        [(2025, "A GP", "SOFT", 90, True, True)] * 40,
        columns=pool.columns)], ignore_index=True)
    with_it, _ = S.bounds_excluding(planted, "Z GP", label_of)
    without, _ = S.bounds_excluding(planted, "A GP", label_of, season=2025)
    assert with_it["C5"] > caps_all["C5"] + 10
    assert without["C5"] == caps_all["C5"]
    # Excluding the SAME name in another season keeps 2025's stints.
    other_season, _ = S.bounds_excluding(planted, "A GP", label_of, season=2024)
    assert other_season["C5"] > caps_all["C5"] + 10


def test_the_ceiling_and_floor_come_from_their_own_pools():
    pool = _big_pool()
    pool.loc[pool["label"] == "HARD", "floor_ok"] = False   # no genuine floor evidence for HARD
    caps, floors = S.bounds_excluding(pool, "Z GP", {"C3": "HARD", "C4": "MEDIUM"})
    assert "C3" in caps and "C3" not in floors
    assert "C4" in caps and "C4" in floors


def test_bound_hits_names_a_stint_that_sits_on_a_bound():
    hits = S.bound_hits(("C4", "C5", "C5"), (9, 24, 26), {"C5": 26, "C4": 33}, {"C4": 9, "C5": 7})
    assert {(h["compound"], h["laps"], h["bound"], h["limit"]) for h in hits} == {
        ("C4", 9, "floor", 9), ("C5", 26, "ceiling", 26)}
    assert S.bound_hits(("C4", "C5"), (20, 20), {"C5": 26}, {"C4": 9}) == []


def test_the_pool_survives_a_round_trip_through_its_file(tmp_path):
    pool = _big_pool()
    path = tmp_path / "stint_pool.json"
    S.write_pool(pool, path)
    back = S.read_pool(path)
    label_of = {"C5": "SOFT", "C4": "MEDIUM", "C3": "HARD"}
    assert S.bounds_excluding(back, "A GP", label_of, season=2025) == S.bounds_excluding(pool, "A GP", label_of, season=2025)
    assert S.read_pool(tmp_path / "missing.json") is None

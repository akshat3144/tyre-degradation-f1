"""Tests for the strategy evaluation behind paper/results/strategy.json.

Fast and offline: everything is synthetic. They pin the definitions the paper
quotes -- the plan distance, the field target, the baselines' training rule, the
stint-length bounds, the pace-step estimator -- so a number cannot drift without
a test saying which definition moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.paper import pit_loss_eval as PLE
from tyredeg.paper import strategy_eval as E
from tyredeg.paper import strategy_pace as PA
from tyredeg.paper.strategy_meta import meta
from tyredeg.strategy import pitloss as PL
from tyredeg.strategy.optimise import best_plan, enumerate_plans, pace_offsets
from tyredeg.strategy.support import support_caps, support_floor

# ---------------------------------------------------------------------------
# the plan distance
# ---------------------------------------------------------------------------


def test_identical_plans_are_zero_apart_and_order_does_not_matter():
    assert E.w1_age((20, 30), (20, 30)) == 0.0
    assert E.w1_age((20, 30), (30, 20)) == 0.0, "plans are unordered; a distance must be too"


def test_the_distance_is_symmetric_and_in_laps():
    a, b = (50,), (25, 25)
    assert E.w1_age(a, b) == pytest.approx(E.w1_age(b, a))
    # One 50-lap stint visits ages 1..50 once each (mean 25.5); two 25-lap stints
    # visit 1..25 twice (mean 13). Their W1 is the area between the CDFs.
    assert E.w1_age(a, b) == pytest.approx(12.5, abs=0.5)


def test_a_one_stop_plan_is_nearer_a_one_stop_field_than_a_three_stop_plan_is():
    field = (28, 30)
    assert E.w1_age((29, 29), field) < E.w1_age((15, 14, 15, 14), field)


def test_equal_split_covers_the_race():
    for laps, stops in [(57, 1), (70, 2), (44, 3)]:
        s = E.equal_split(laps, stops)
        assert sum(s) == laps and len(s) == stops + 1
        assert max(s) - min(s) <= 1


# ---------------------------------------------------------------------------
# the field target
# ---------------------------------------------------------------------------


def _synthetic_race(wet=False, red=False):
    rows, res = [], []
    for k in range(12):
        d = f"D{k:02d}"
        # stints of 20 and 30 laps; the first ten finish, two are classified later
        for lap in range(1, 51):
            stint = 1 if lap <= 20 else 2
            c = "MEDIUM" if stint == 1 else ("INTERMEDIATE" if wet else "HARD")
            rows.append(
                {"season": 2025, "event": "X GP", "Driver": d, "LapNumber": float(lap),
                 "Stint": float(stint), "Compound": c, "TrackStatus": "5" if (red and lap == 3) else "1"}
            )
        res.append({"season": 2025, "event": "X GP", "Driver": d, "Position": float(k + 1),
                    "Status": "Finished" if k < 11 else "Retired"})
    return pd.DataFrame(rows), pd.DataFrame(res)


def test_field_target_is_the_classified_top_ten():
    laps, res = _synthetic_race()
    (r,) = E.field_races(laps, res)
    assert len(r.cars) == 10
    assert r.mode_stops == 1 and r.race_laps == 50
    assert r.cars[0] == [("MEDIUM", 20), ("HARD", 30)]
    assert not r.wet and not r.red_flag


def test_wet_and_red_flagged_races_are_marked_not_silently_kept():
    laps, res = _synthetic_race(wet=True)
    assert E.field_races(laps, res)[0].wet
    laps, res = _synthetic_race(red=True)
    assert E.field_races(laps, res)[0].red_flag


def test_headline_excludes_red_flag_races():
    df = pd.DataFrame({"red_flag": [False, True], "x": [1, 2]})
    assert list(E.headline(df)["x"]) == [1]


# ---------------------------------------------------------------------------
# baselines are fitted without the race being scored
# ---------------------------------------------------------------------------


def test_the_lap_rule_threshold_separates_the_training_races():
    train = [(44, 1), (52, 1), (57, 1), (66, 2), (70, 2), (78, 2)]
    t = E._lap_rule_threshold(train)
    assert 57 < t < 66
    assert E._lap_rule_threshold([]) == 60.0


def test_the_trivial_rules_are_stronger_than_they_look():
    """On races where length decides the stop count, the one-feature rule is
    perfect and 'always one stop' is not: the reason the shipped 10-of-11 proved
    nothing."""
    races = [(44, 1), (52, 1), (57, 1), (66, 2), (70, 2), (78, 2)]
    t = E._lap_rule_threshold(races)
    rule = sum((2 if laps >= t else 1) == s for laps, s in races)
    always1 = sum(s == 1 for _, s in races)
    assert rule == len(races) and always1 == 3


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def test_score_rec_reports_every_metric_against_the_field():
    laps, res = _synthetic_race()
    (r,) = E.field_races(laps, res)
    s = E.score_rec(E.Rec((20, 30), ("MEDIUM", "HARD")), r)
    assert s["stop_match_mode"] == 1
    assert s["w1_median_car"] == 0.0 and s["w1_nearest_car"] == 0.0
    assert s["longest_err"] == 0.0 and s["shortest_err"] == 0.0
    assert s["usage_tv_median_car"] == 0.0
    off = E.score_rec(E.Rec((5, 45), ("MEDIUM", "HARD")), r)
    assert off["on_boundary"] == 1 and off["longest_err"] == 15.0
    assert off["w1_median_car"] > 0


def test_a_baseline_has_no_compound_opinion_so_no_usage_metric():
    laps, res = _synthetic_race()
    (r,) = E.field_races(laps, res)
    assert "usage_tv_median_car" not in E.score_rec(E.Rec((25, 25)), r)


def test_paired_difference_is_zero_for_a_method_against_itself():
    df = pd.DataFrame(
        {"red_flag": False, "group": "g", "race": [f"r{i}" for i in range(6)] * 2,
         "method": ["a"] * 6 + ["b"] * 6, "m": list(range(6)) * 2}
    )
    r = E.paired_diff(df, "a", "b", "m")
    assert r["mean_a_minus_b"] == 0.0 and r["n_races"] == 6


def test_bootstrap_interval_brackets_the_mean_and_is_reproducible():
    x = np.arange(20.0)
    a = E.boot_mean(x)
    b = E.boot_mean(x)
    assert a == b and a[1] <= a[0] <= a[2]


# ---------------------------------------------------------------------------
# the rate gate and the robust planner
# ---------------------------------------------------------------------------


def test_the_truncated_mean_is_above_the_point_estimate_only_when_the_sign_is_doubtful():
    assert E.truncated_mean(0.03, 0.03) > 0.03 * 1.2
    assert E.truncated_mean(0.10, 0.01) == pytest.approx(0.10, abs=1e-6)
    assert E.truncated_mean(0.05, 0.0) == 0.05


def test_gate_refuses_unidentified_rates_and_keeps_the_rest():
    src = E.RateInput({"C3": 0.05, "C4": 0.011}, {"C3": 0.01, "C4": 0.028})
    assert set(E.gate(src)) == {"C3"}


def test_plan_for_refuses_when_fewer_than_two_compounds_survive():
    assert E.plan_for(50, {"C3": 0.05}, ["C3", "C4"], ("C1", "C2", "C3", "C4", "C5"), 22.0) is None


# ---------------------------------------------------------------------------
# stint-length bounds
# ---------------------------------------------------------------------------


def _obs():
    rng = np.random.default_rng(0)
    return (
        [("SOFT", int(x)) for x in rng.integers(8, 26, 60)]
        + [("MEDIUM", int(x)) for x in rng.integers(10, 36, 60)]
        + [("HARD", int(x)) for x in rng.integers(12, 44, 60)]
    )


def test_ceilings_follow_the_observed_stints_and_floors_sit_below_them():
    caps, floors = support_caps(_obs()), support_floor(_obs())
    assert caps["SOFT"] < caps["MEDIUM"] < caps["HARD"]
    assert all(floors[k] < caps[k] for k in caps)


def test_a_label_with_too_few_observations_gets_no_bound():
    assert "SOFT" not in support_caps([("SOFT", 20)] * 5 + [("HARD", 30)] * 40)


def test_the_ceiling_binds_and_removes_the_run_to_the_flag_plan():
    """The pathology in miniature: a flat soft rate and a pace edge make the
    optimiser run the soft for most of the race, with a short filler stint."""
    rates = {"C4": 0.03, "C5": 0.011}
    off = pace_offsets(["C3", "C4", "C5"])
    free = best_plan(78, rates, {c: off[c] for c in rates}, 22.5)
    assert max(free.stints) > 60, "the fixture should reproduce the pathology"
    capped = best_plan(78, rates, {c: off[c] for c in rates}, 22.5, max_stint={"C4": 36, "C5": 26})
    assert max(capped.stints) <= 36
    # and it agrees with the exhaustive enumeration under the same bound
    ref = enumerate_plans(78, rates, {c: off[c] for c in rates}, 22.5, step=2,
                          max_stint={"C4": 36, "C5": 26})[0]
    assert capped.total_time <= ref.total_time + 1e-9


def test_the_floor_forbids_the_filler_stint():
    rates = {"C3": 0.02, "C5": 0.05}
    off = pace_offsets(["C3", "C4", "C5"])
    free = best_plan(60, rates, {c: off[c] for c in rates}, 22.0)
    floored = best_plan(60, rates, {c: off[c] for c in rates}, 22.0, floor_stint={"C3": 12, "C5": 8})
    assert min(floored.stints) >= 8
    assert floored.total_time >= free.total_time - 1e-9


# ---------------------------------------------------------------------------
# the pace-step estimator
# ---------------------------------------------------------------------------


def test_two_way_absorption_removes_both_sets_of_means():
    rng = np.random.default_rng(1)
    a = np.repeat(np.arange(6), 5)
    b = np.tile(np.arange(5), 6)
    X = rng.normal(size=(30, 2)) + a[:, None] * 3.0 + b[:, None]
    Z = PA._absorb(X, [a, b])
    for g in (a, b):
        for k in np.unique(g):
            assert np.allclose(Z[g == k].mean(axis=0), 0.0, atol=1e-6)


def test_the_race_estimator_recovers_a_planted_step():
    """Eight drivers each run a medium then a soft; the soft is truly 0.5 s/lap
    quicker at equal age. The estimator must find it despite fast cars."""
    rng = np.random.default_rng(2)
    rows = []
    for d in range(12):
        car = rng.normal(0, 0.4)
        for lap in range(1, 41):
            soft = lap > 20 if d % 2 == 0 else lap <= 20
            age = lap - 20 if lap > 20 else lap
            c = "SOFT" if soft else "MEDIUM"
            t = 90 - 0.05 * lap + car + (-0.5 if soft else 0.0) + 0.05 * age + rng.normal(0, 0.05)
            rows.append({"event": "E", "Driver": f"D{d}", "LapNumber": lap, "C": c,
                         "TyreLife": age, "LapTimeSeconds": t, "cars_on_lap": 12})
    est = PA.race_estimate(pd.DataFrame(rows), ("HARD", "MEDIUM", "SOFT"))
    assert est["estimable"]
    # the coefficient is on the compound index, so a softer-is-faster step is negative
    assert est["step_s_per_compound_step"] == pytest.approx(-0.5, abs=0.1)


def test_the_practice_estimator_says_so_when_there_is_nothing_to_contrast():
    rows = []
    for d in range(30):
        for lap in range(1, 9):
            rows.append({"run_id": f"r{d}", "event": "E", "session": "FP2", "Driver": f"D{d}", "C": "C3",
                         "TyreLife": lap, "y_fuel_corrected": 90.0 + 0.05 * lap, "SessionElapsed": 100.0 * d + lap})
    est = PA.practice_estimate(pd.DataFrame(rows), ("C1", "C2", "C3", "C4", "C5"))
    assert est["estimable"] is False


# ---------------------------------------------------------------------------
# pit-loss study
# ---------------------------------------------------------------------------


def _stops(n_per_race=12, races=8, level=22.0, vsc_ratio=0.8):
    rng = np.random.default_rng(3)
    rows = []
    for r in range(races):
        for cat, scale in (("green", 1.0), ("vsc", vsc_ratio)):
            for k in range(n_per_race):
                rows.append({"race": f"{2020 + r} R{r}", "season": 2020 + r, "category": cat,
                             "loss": level * scale + rng.normal(0, 0.8), "lap": 10 + k, "n_pitting": 1})
    return pd.DataFrame(rows)


def test_cluster_bootstrap_resamples_races_and_is_deterministic():
    s = _stops()
    g = s[s["category"] == "green"]
    a, b = PLE.cluster_boot(g, np.median, n_boot=200), PLE.cluster_boot(g, np.median, n_boot=200)
    assert a == b and a[1] <= a[0] <= a[2]


def test_the_vsc_ratio_is_recovered_and_flagged_usable_only_with_enough_evidence():
    s = _stops(races=10)
    t = PLE.category_table(s)
    assert t["vsc"]["ratio_to_green"] == pytest.approx(0.8, abs=0.05)
    assert t["vsc"]["usable_as_default"] is True
    thin = PLE.category_table(_stops(n_per_race=2, races=3))
    assert thin["vsc"]["usable_as_default"] is False


def test_a_category_dominated_by_one_race_is_not_a_default():
    s = _stops(races=10)
    s = s[(s["category"] == "green") | (s["race"] == s["race"].iloc[0])]
    assert PLE.category_table(s)["vsc"]["usable_as_default"] is False


def test_status_classification_used_by_the_study_is_the_library_one():
    assert PL.stop_category("1", "5") == "red_flag"
    assert PLE.PL is PL


# ---------------------------------------------------------------------------
# audit metadata
# ---------------------------------------------------------------------------


def test_every_result_carries_audit_metadata(tmp_path):
    f = tmp_path / "in.txt"
    f.write_text("x")
    m = meta("scripts/paper/p5_strategy.py", "m.py", 7, {"k": 1}, [f, tmp_path / "missing"])
    for key in ("script", "module", "generated_utc", "git_head", "python", "seed", "params", "input_files"):
        assert key in m
    assert list(m["input_files"]) == ["in.txt"] and len(m["input_files"]["in.txt"]) == 64


# ---------------------------------------------------------------------------
# review fixes: out of time, genuine-stint bounds, the flags
# ---------------------------------------------------------------------------


def _pit_table():
    return pd.DataFrame([
        {"season": 2022, "event": "X GP", "circuit": "x", "pit_s": 20.0, "n": 30, "order": 0},
        {"season": 2022, "event": "Y GP", "circuit": "y", "pit_s": 24.0, "n": 30, "order": 1},
        {"season": 2023, "event": "X GP", "circuit": "x", "pit_s": 21.0, "n": 30, "order": 2},
        {"season": 2024, "event": "X GP", "circuit": "x", "pit_s": 99.0, "n": 30, "order": 3},
    ])


def test_out_of_time_pit_loss_never_looks_at_a_later_race():
    t = _pit_table()
    v, src = E.pit_for(t, 2023, "X GP", "x", out_of_time=True)
    assert v == 20.0 and src == "same circuit, earlier seasons"
    # Leave-one-out (the older form) lets 2023 be planned with 2024's pit lane.
    v_loo, _ = E.pit_for(t, 2023, "X GP", "x")
    assert v_loo == pytest.approx(np.median([20.0, 99.0]))


def test_out_of_time_pit_loss_falls_back_to_earlier_races_and_then_says_there_were_none():
    t = _pit_table()
    v, src = E.pit_for(t, 2023, "X GP", "never-seen", out_of_time=True)
    assert src == "no earlier race: median of all other races"    # only two races came before it
    t2 = pd.concat([t, pd.DataFrame([{"season": 2024, "event": "Z GP", "circuit": "z", "pit_s": 22.0, "n": 30, "order": 4}])])
    v2, src2 = E.pit_for(t2, 2024, "Z GP", "z", out_of_time=True)
    assert src2 == "same circuit, earlier seasons" or src2 == "median of earlier races"
    v3, src3 = E.pit_for(t2, 2024, "Z GP", "never-seen", out_of_time=True)
    assert src3 == "median of earlier races" and v3 == pytest.approx(np.median([20.0, 24.0, 21.0, 99.0]))


def test_field_races_carry_their_calendar_order_not_their_alphabetical_one():
    a, ra = _synthetic_race()
    b, rb = _synthetic_race()
    a["event"], ra["event"] = "Z GP", "Z GP"        # run FIRST, sorts LAST
    b["event"], rb["event"] = "A GP", "A GP"
    races = E.field_races(pd.concat([a, b], ignore_index=True), pd.concat([ra, rb], ignore_index=True))
    by = {r.event: r.order for r in races}
    assert by["Z GP"] < by["A GP"]


def _fr(order, race_laps, stops):
    cars = []
    for _ in range(6):
        n = race_laps // (stops + 1)
        cars.append([("MEDIUM" if i % 2 == 0 else "HARD", n) for i in range(stops + 1)])
    return E.FieldRace(2024, f"R{order} GP", race_laps, cars, False, False, order)


def test_the_lap_rule_baseline_is_trained_on_earlier_races_only_when_out_of_time(monkeypatch):
    """A 70-lap race first in the calendar: out of time the rule has nothing to learn
    from (its 60-lap default says two stops); leave-one-out learns from the three LATER
    long races that one stop is normal, and says one."""
    monkeypatch.setattr(E.racedata, "locations", lambda: {})
    races = [_fr(0, 70, 2), _fr(1, 80, 1), _fr(2, 90, 1), _fr(3, 100, 1)]
    pits = pd.DataFrame([{"season": 2024, "event": r.event, "circuit": r.event, "pit_s": 22.0, "n": 30,
                          "order": r.order} for r in races])
    empty = pd.DataFrame(columns=["event", "C"])
    groups = {"2022-25": (E.Cells(practice=empty, race=empty), lambda ev, c: c, E.LABEL_ORDER)}

    def stops(**kw):
        df = E.evaluate(races, groups, pits, methods=("none",), **kw)
        return int(df[(df["race"] == races[0].key) & (df["method"] == "lap_rule")]["stops"].iloc[0])

    assert stops(out_of_time=True) == 2
    assert stops(out_of_time=False) == 1


def test_the_stint_bounds_in_the_evaluation_come_from_earlier_races_when_out_of_time(monkeypatch):
    """The first races have no bounds (under 30 earlier stints of a label); a LATER race
    must not supply them."""
    monkeypatch.setattr(E.racedata, "locations", lambda: {})
    races = [_fr(0, 60, 1), _fr(1, 60, 1)]
    rng = np.random.default_rng(0)
    rows = [(2024, races[1].event, lab, int(n), True, True)
            for lab in ("MEDIUM", "HARD") for n in rng.integers(15, 30, 60)]
    pool = pd.DataFrame(rows, columns=["season", "event", "label", "n", "ceiling_ok", "floor_ok"])
    pool["season"] = 2024
    pits = pd.DataFrame([{"season": 2024, "event": r.event, "circuit": r.event, "pit_s": 22.0, "n": 30,
                          "order": r.order} for r in races])
    cells_rows = pd.DataFrame({"event": [races[0].event] * 3, "C": ["SOFT", "MEDIUM", "HARD"]})
    # Only the race scored FIRST would see bounds if the pool leaked from the later race.
    seen = {}
    import tyredeg.paper.strategy_eval as SE

    real = SE.support_caps

    def spy(stints, q=0.95, min_n=30):
        out = real(stints, q, min_n)
        seen.setdefault("calls", []).append(len(stints))
        return out

    monkeypatch.setattr(SE, "support_caps", spy)
    groups = {"2022-25": (E.Cells(practice=cells_rows, race=cells_rows), lambda ev, c: c, E.LABEL_ORDER)}
    E.evaluate(races, groups, pits, methods=("none",), stint_pool=pool, out_of_time=True)
    assert seen["calls"][0] == 0, "the first race was handed stints from a race run after it"
    seen.clear()
    E.evaluate(races, groups, pits, methods=("none",), stint_pool=pool, out_of_time=False)
    assert seen["calls"][0] > 0, "leave-one-out should have seen the other race"

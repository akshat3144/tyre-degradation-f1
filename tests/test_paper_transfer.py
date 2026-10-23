"""Tests for the practice -> race evaluation used in the paper.

Synthetic known truth, offline and fast. The two real-data tests at the bottom
are skipped without the cached season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.config import PROCESSED
from tyredeg.models import design as D
from tyredeg.paper import transfer_eval as E
from tyredeg.validation import transfer as T

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def synth_panel(n_events=10, factor=0.5, noise=0.01, seed=0, hist_noise=0.02):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_events):
        for c in ("SOFT", "MEDIUM"):
            p = rng.uniform(0.05, 0.25)
            truth = factor * p
            rows.append({
                "event": f"E{i}", "season": 2023, "circuit": f"circ{i}", "C": c, "grp": c,
                "label": c, "practice_rate": p + rng.normal(0, noise),
                "race_rate": truth + rng.normal(0, noise), "se_p": 0.02,
                "hist_rate": truth + rng.normal(0, hist_noise),
            })
    return pd.DataFrame(rows)


def synth_laps(seed=0):
    """Practice laps with a known degradation, in the shape ``prepare`` expects."""
    rng = np.random.default_rng(seed)
    rows = []
    for event in ("A GP", "B GP"):
        for d, drv in enumerate(("AAA", "BBB", "CCC", "DDD", "EEE", "FFF")):
            lap_no = 1
            for stint in (1, 2):
                n = 16
                for k in range(n):
                    if drv == "AAA" and stint == 1:
                        lt = 89.0 + rng.normal(0, 0.1)   # fast, qualifying-style
                    else:
                        lt = 92.5 + 0.08 * (k + 1) + rng.normal(0, 0.15)
                    if drv == "CCC" and stint == 2 and k == 7:
                        lt = 99.0                         # a lap that is dropped mid-run
                    rows.append({
                        "event": event, "session": "FP2", "Driver": drv, "Stint": float(stint),
                        "LapNumber": float(lap_no), "Compound": "MEDIUM" if stint == 1 else "SOFT",
                        "TyreLife": float(k + 1), "LapTimeSeconds": lt, "TrackStatus": "1",
                    })
                    lap_no += 1
                lap_no += 3   # pit stop between stints: lap numbering jumps
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the parity that the old sweep lacked
# ---------------------------------------------------------------------------


def test_build_practice_default_is_the_shipped_pipeline(monkeypatch):
    """The failure that broke scripts/13: the sweep's 'current' was not what ships."""
    monkeypatch.setattr(D, "add_physical_compound", E.label_compound)
    laps = synth_laps()
    shipped = D.prepare(laps, "practice", min_runs_per_compound=1)
    rebuilt = E.build_practice(laps, E.PracticeDesign(min_runs_per_compound=1))
    assert len(shipped) > 0
    cols = ["run_id", "LapNumber", "y", "tl", "w"]
    a = shipped.sort_values(["run_id", "LapNumber"])[cols].reset_index(drop=True)
    b = rebuilt.sort_values(["run_id", "LapNumber"])[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_legacy_variant_differs_from_shipped(monkeypatch):
    """If 'legacy' equalled the shipped build the sweep would have nothing to compare."""
    monkeypatch.setattr(D, "add_physical_compound", E.label_compound)
    laps = synth_laps()
    ship = E.build_practice(laps, E.PracticeDesign(min_runs_per_compound=1))
    leg = E.build_practice(laps, E.PracticeDesign(
        "legacy", keep_runs=False, drop_warmup=0, max_gap=None, sprint_is_race=False,
        min_runs_per_compound=1))
    assert len(ship) != len(leg)


def test_warmup_trim_removes_first_lap_of_each_run(monkeypatch):
    monkeypatch.setattr(D, "add_physical_compound", E.label_compound)
    laps = synth_laps()
    d0 = E.build_practice(laps, E.PracticeDesign(drop_warmup=0, min_runs_per_compound=1))
    d1 = E.build_practice(laps, E.PracticeDesign(drop_warmup=1, min_runs_per_compound=1))
    assert len(d0) - len(d1) == d0["run_id"].nunique()


def test_weights_override_does_not_touch_the_module_constant(monkeypatch):
    monkeypatch.setattr(D, "add_physical_compound", E.label_compound)
    before = dict(E.SESSION_WEIGHTS)
    f = E.build_practice(synth_laps(), E.PracticeDesign(min_runs_per_compound=1),
                         weights={"FP2": 0.25})
    assert set(f["w"]) == {0.25}
    assert E.SESSION_WEIGHTS == before


# ---------------------------------------------------------------------------
# paired inference
# ---------------------------------------------------------------------------


def test_paired_test_detects_a_better_method_and_is_deterministic():
    rng = np.random.default_rng(1)
    cl = np.repeat(np.arange(15), 3)
    a = rng.normal(0, 0.02, len(cl))
    b = rng.normal(0, 0.08, len(cl))
    r1 = E.paired_test(a, b, cl, n_boot=2000)
    r2 = E.paired_test(a, b, cl, n_boot=2000)
    assert r1 == r2
    assert r1["mean_diff"] < 0 and r1["ci_hi"] < 0
    assert r1["dm_p"] < 0.05 and r1["signflip_p"] < 0.05
    assert r1["n_clusters"] == 15


def test_paired_test_does_not_flag_identical_methods():
    rng = np.random.default_rng(2)
    cl = np.repeat(np.arange(12), 2)
    e = rng.normal(0, 0.05, len(cl))
    r = E.paired_test(e, e, cl, n_boot=500)
    assert r["mean_diff"] == 0 and r["signflip_p"] == 1.0


def test_null_difference_is_rarely_significant():
    """False-positive rate of the sign-flip test near its nominal 5%."""
    hits = 0
    for s in range(60):
        rng = np.random.default_rng(100 + s)
        cl = np.repeat(np.arange(10), 2)
        a, b = rng.normal(0, 0.05, 20), rng.normal(0, 0.05, 20)
        hits += E.paired_test(a, b, cl, n_boot=200, seed=s)["signflip_p"] < 0.05
    assert hits <= 9


def test_bootstrap_resamples_clusters_not_cells():
    """One huge weekend must widen the interval; resampling cells would hide it."""
    cl = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    v = np.array([0.0] * 4 + [0.0] * 4 + [1.0] * 4)
    m, lo, hi = E.boot_mean_ci(v, cl, n_boot=2000)
    assert m == pytest.approx(1 / 3)
    assert lo == 0.0 and hi == 1.0


# ---------------------------------------------------------------------------
# calibration and baselines
# ---------------------------------------------------------------------------


def test_calibrated_beats_naive_and_constant_when_a_factor_exists():
    panel = synth_panel(n_events=14, factor=0.4, noise=0.005)
    res = E.loeo(panel)
    sb = E.scoreboard(res, E.METHODS)
    assert sb["scalar"]["mae"] < sb["naive"]["mae"]
    assert sb["scalar"]["mae"] < sb["pooled"]["mae"]
    assert sb["scalar"]["mae"] < sb["zero"]["mae"]
    assert res["factor_used"].between(0.3, 0.5).all()


def test_held_out_event_does_not_inform_its_own_prediction():
    panel = synth_panel(n_events=8)
    base = E.loeo(panel).set_index(["event", "C"])
    bad = panel.copy()
    bad.loc[bad["event"] == "E3", "race_rate"] += 5.0     # wreck one event's truth
    moved = E.loeo(bad).set_index(["event", "C"])
    e3 = base.index.get_level_values(0) == "E3"
    # Its own forecasts are unchanged; only OTHER events' calibration moves.
    for col in ("pred_scalar", "pred_pooled", "pred_stack", "pred_compound"):
        np.testing.assert_allclose(base.loc[e3, col], moved.loc[e3, col])
    assert not np.allclose(base.loc[~e3, "pred_scalar"], moved.loc[~e3, "pred_scalar"])


def test_compound_factor_wins_when_compounds_differ_and_loses_when_they_do_not():
    rng = np.random.default_rng(5)
    rows = []
    for i in range(16):
        for c, f in (("SOFT", 0.2), ("HARD", 0.9)):
            p = rng.uniform(0.05, 0.25)
            rows.append({"event": f"E{i}", "season": 2023, "circuit": f"c{i}", "C": c, "grp": c,
                         "label": c, "practice_rate": p, "race_rate": f * p + rng.normal(0, 0.004),
                         "se_p": 0.02, "hist_rate": np.nan})
    diff = E.loeo(pd.DataFrame(rows))
    assert E.score(diff, "compound")["mae"] < E.score(diff, "scalar")["mae"]
    same = synth_panel(n_events=16, factor=0.5, noise=0.004, seed=6)
    r = E.loeo(same)
    # With one true factor the extra parameters buy nothing.
    assert E.score(r, "compound")["mae"] >= E.score(r, "scalar")["mae"] - 0.003


def test_frozen_holdout_uses_the_training_factor_not_the_test_one():
    train = synth_panel(n_events=10, factor=0.5, seed=1)
    test = synth_panel(n_events=6, factor=0.1, seed=2)
    out = E.frozen_holdout(train, test)
    assert out["factor_used"].nunique() == 1
    assert out["factor_used"].iloc[0] == pytest.approx(0.5, abs=0.1)
    # The test truth is far from the frozen factor's prediction: no refit happened.
    assert E.score(out, "scalar")["mae"] > 0.03


def test_history_uses_strictly_earlier_seasons_only():
    panel = pd.DataFrame([{"event": "2024 X", "season": 2024, "circuit": "X", "C": "SOFT",
                           "label": "SOFT"}])
    rc = pd.DataFrame([
        {"season": 2022, "circuit": "X", "label": "SOFT", "race_rate": 0.10},
        {"season": 2023, "circuit": "X", "label": "SOFT", "race_rate": 0.20},
        {"season": 2024, "circuit": "X", "label": "SOFT", "race_rate": 0.99},   # itself
        {"season": 2025, "circuit": "X", "label": "SOFT", "race_rate": 0.77},   # the future
        {"season": 2023, "circuit": "X", "label": "HARD", "race_rate": 0.55},
    ])
    out = E.attach_history(panel, rc)
    assert out["hist_rate"].iloc[0] == 0.20 and out["hist_season"].iloc[0] == 2023


def test_madrid_is_not_joined_to_barcelona():
    assert E.circuit_of("Spanish Grand Prix", 2026) == "Madrid"
    assert E.circuit_of("Barcelona Grand Prix", 2026) == E.circuit_of("Spanish Grand Prix", 2024)


# ---------------------------------------------------------------------------
# nested selection must not see the held-out event
# ---------------------------------------------------------------------------


def _candidate(panel, scale):
    t = panel.copy()
    t["practice_rate"] = t["practice_rate"] * scale
    return t


def test_nested_selection_never_reads_the_held_out_event():
    base = synth_panel(n_events=8, factor=0.5, noise=0.003)
    key = E.SHIPPED_KEY
    other = ("warmup2", "equal")
    tables = {key: _candidate(base, 1.0), other: _candidate(base, 1.0)}
    # Make `other` look far better, but ONLY at event E0.
    tables[other].loc[tables[other]["event"] == "E0", "practice_rate"] = \
        tables[other].loc[tables[other]["event"] == "E0", "race_rate"] / 0.5
    pred, log = E.nested_selection(tables)
    pick_e0 = log.set_index("held_out").loc["E0", "design"]
    # Held out, E0's flattering numbers cannot be what picks the design for E0.
    tables2 = {k: v.copy() for k, v in tables.items()}
    tables2[other].loc[tables2[other]["event"] == "E0", "practice_rate"] = 99.0
    _, log2 = E.nested_selection(tables2)
    assert log2.set_index("held_out").loc["E0", "design"] == pick_e0
    assert len(pred) == len(base)


def test_nested_selection_excludes_candidates_that_refuse_cells():
    base = synth_panel(n_events=8, noise=0.003)
    thin = _candidate(base, 1.0).iloc[::2]      # answers half the questions
    tables = {E.SHIPPED_KEY: _candidate(base, 1.0), ("warmup3", "equal"): thin}
    _, log = E.nested_selection(tables)
    assert (log["design"] == E.SHIPPED_KEY[0]).all()
    assert (log["n_eligible"] == 1).all()


# ---------------------------------------------------------------------------
# stand-ins, identifiability, intervals
# ---------------------------------------------------------------------------


def _practice(rows):
    out = []
    for i, (event, c, rate) in enumerate(rows):
        for run in range(5):
            age = np.arange(10) + run * 3
            y = rate * age
            out.append(pd.DataFrame({
                "event": event, "session": "FP2", "C": c, "run_id": f"{event}-{c}-{run}-{i}",
                "LapNumber": age + 1, "TyreLife": age, "y": y - y.mean(), "tl": age - age.mean(),
            }))
    return pd.concat(out, ignore_index=True)


def test_rate_table_stand_in_equals_the_lap_level_one():
    """The accuracy test must exercise the code the screen runs."""
    df = _practice([("M", "C3", 0.30), ("O", "C3", 0.15), ("O", "C2", 0.05),
                    ("P", "C3", 0.15), ("P", "C2", 0.06)])
    want = T.stand_in_rates(df, 0.4, "M", ["C2", "C3"])
    here = T.cell_rates(df[df["event"] == "M"], T.FORECAST_MIN_RUNS)
    elsewhere = T.cell_rates(df[df["event"] != "M"], T.MIN_RUNS)
    got = T.stand_in_from_rates(here, elsewhere, 0.4, "M", ["C2", "C3"])
    pd.testing.assert_frame_equal(want.reset_index(drop=True), got.reset_index(drop=True))


def test_standin_table_hides_the_cell_it_imputes():
    df = _practice([("M", "C3", 0.30), ("M", "C4", 0.40), ("O", "C3", 0.15), ("O", "C4", 0.20),
                    ("P", "C3", 0.15), ("P", "C4", 0.20)])
    panel = pd.DataFrame([{"event": "M", "C": "C4", "race_rate": 0.1, "se_p": 0.01,
                           "practice_rate": 0.40, "factor_used": 1.0}])
    out = E.standin_table(df, panel, {"M": 1.0})
    # Severity from M's C3 alone (0.30/0.15 = 2); C4 pooled elsewhere = 0.20 -> 0.40,
    # and certainly not the hidden measured 0.40 by accident of copying it:
    assert out["si_sev"].iloc[0] == pytest.approx(2.0, abs=1e-3)
    assert out["si_rate"].iloc[0] == pytest.approx(0.40, abs=1e-3)
    assert out["si_unit"].iloc[0] > 0


def test_interval_calibration_recovers_the_multiple_for_gaussian_errors():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"event": np.repeat(np.arange(60), 10), "pred": 0.0, "unit": 0.05})
    df["race_rate"] = rng.normal(0, 0.05, n)       # true sd = 1 unit
    cal = E.interval_calibration(df, "pred", "unit", 0.95)
    assert cal["multiple_for_level"] == pytest.approx(1.96, abs=0.25)
    assert cal["cv_coverage_at_calibrated"] == pytest.approx(0.95, abs=0.04)


def test_eb_weight_falls_with_standard_error():
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({
        "event": np.repeat([f"E{i}" for i in range(20)], 2),
        "practice_rate": rng.uniform(0.0, 0.3, n), "si_rate": rng.uniform(0.0, 0.3, n),
        "se_p": np.r_[np.full(n // 2, 0.01), np.full(n // 2, 0.5)],
        "factor_used": 0.5, "race_rate": 0.0, "pred_scalar": 0.0,
    })
    out = E.identifiability_eval(df)
    assert out.loc[df["se_p"] == 0.01, "eb_weight"].mean() > out.loc[df["se_p"] == 0.5, "eb_weight"].mean()
    assert ((out["eb_weight"] >= 0) & (out["eb_weight"] <= 1)).all()


def test_rule_flags_only_cells_below_k_standard_errors():
    df = pd.DataFrame({
        "event": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"],
        "practice_rate": [0.05, 0.30] * 5, "si_rate": 0.1, "se_p": [0.20, 0.05] * 5,
        "factor_used": 0.5, "race_rate": 0.1, "pred_scalar": 0.0,
    })
    out = E.identifiability_eval(df, ks=(1.0,))
    assert out["flag_rule1"].tolist() == [True, False] * 5
    assert out["pred_rule1"].iloc[0] == pytest.approx(0.5 * 0.1)   # replaced by the stand-in
    assert out["pred_rule1"].iloc[1] == pytest.approx(0.5 * 0.30)  # trusted


# ---------------------------------------------------------------------------
# bookkeeping
# ---------------------------------------------------------------------------


def test_jsonable_turns_nan_into_null_and_numpy_into_python():
    import json

    o = E.jsonable({"a": np.float64("nan"), "b": np.int64(3), "c": [np.float64(1.23456789)],
                    "d": pd.DataFrame({"x": [1.0, np.nan]})})
    s = json.dumps(o)
    assert '"a": null' in s and '"b": 3' in s and o["d"][1]["x"] is None


def test_audit_records_seed_inputs_and_shipped_design(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_bytes(b"abc")
    a = E.audit("s.py", [f])
    assert a["seed"] == E.SEED and a["inputs"]["x.parquet"]["bytes"] == 3
    assert a["shipped_design"]["drop_warmup"] == D.WARMUP_LAPS


def test_design_grid_contains_the_shipped_design():
    ship = E.DESIGN_GRID["warmup1-gap2"]
    assert (ship.keep_runs, ship.drop_warmup, ship.max_gap, ship.min_laps) == \
        (E.SHIPPED.keep_runs, E.SHIPPED.drop_warmup, E.SHIPPED.max_gap, E.SHIPPED.min_laps)


# ---------------------------------------------------------------------------
# real data
# ---------------------------------------------------------------------------

needs_data = pytest.mark.skipif(not (PROCESSED / "laps.parquet").exists(),
                                reason="needs the cached season")


@needs_data
def test_real_2026_parity_and_the_original_claim_still_reproduces():
    laps = pd.read_parquet(PROCESSED / "laps.parquet")
    par = E.parity_with_prepare(laps)
    assert par["identical"], par
    assert par["legacy_laps"] != par["shipped_laps"]
    sd = E.build_season(2026, laps)
    panel = E.cell_panel(sd.practice, sd.race, 2026)
    assert len(panel) >= 15
    res = E.loeo(panel)
    # The shipped claim: calibrated beats naive on the cells it was tuned on.
    assert E.score(res, "scalar")["mae"] < E.score(res, "naive")["mae"]


@needs_data
def test_real_holdout_label_compounds_build_without_the_allocation():
    path = E.laps_path(2024)
    if not path.exists():
        pytest.skip("no 2024 season")
    laps = pd.read_parquet(path)
    sd = E.build_season(2024, laps)
    assert not sd.practice.empty and set(sd.practice["C"]) <= set(E.LABEL_ORDER)
    assert sd.practice["event"].str.startswith("2024 ").all()


# ---------------------------------------------------------------------------
# the interval the transfer screen draws
# ---------------------------------------------------------------------------


def test_empirical_half_width_covers_the_requested_share():
    rng = np.random.default_rng(9)
    errs = np.abs(rng.normal(0, 0.05, 400))
    h = T.empirical_half_width(errs, 0.95)
    assert (errs <= h).mean() >= 0.95
    assert h < errs.max() + 1e-12


def test_empirical_half_width_ignores_missing_and_handles_empty():
    assert T.empirical_half_width([]) == 0.0
    assert T.empirical_half_width([np.nan, None, 0.1]) == pytest.approx(0.1)


def test_practice_slope_interval_alone_undercovers_a_noisy_race_target():
    """The reason the screen's band is floored: sampling error of the practice
    slope says nothing about the race rate's own noise or the factor's."""
    rng = np.random.default_rng(4)
    n = 300
    truth = rng.uniform(0.05, 0.2, n)
    se = np.full(n, 0.01)
    pred = 0.5 * truth
    race = 0.5 * truth + rng.normal(0, 0.06, n)       # a noisy measured race rate
    narrow = 1.96 * se * 0.5
    assert (np.abs(race - pred) <= narrow).mean() < 0.3
    h = T.empirical_half_width(np.abs(race - pred))
    assert (np.abs(race - pred) <= max(narrow.max(), h)).mean() >= 0.95


# ---------------------------------------------------------------------------
# review fixes: the held-out weekend must not reach the training set through history
# ---------------------------------------------------------------------------


def _two_season_panel(seed=0):
    """Circuits X0..X5 raced in 2023 and again in 2024. A 2024 cell's history is its
    2023 race rate, so a 2024 training cell carries a 2023 held-out weekend's outcome."""
    rng = np.random.default_rng(seed)
    rows, cells = [], []
    for i in range(6):
        for c in ("SOFT", "MEDIUM"):
            p23, p24 = rng.uniform(0.05, 0.25, 2)
            r23 = 0.4 * p23 + rng.normal(0, 0.01)
            r24 = 0.4 * p24 + rng.normal(0, 0.01)
            cells += [{"event": f"2023 X{i}", "season": 2023, "circuit": f"X{i}", "label": c, "race_rate": r23},
                      {"event": f"2024 X{i}", "season": 2024, "circuit": f"X{i}", "label": c, "race_rate": r24}]
            for s, p, r in ((2023, p23, r23), (2024, p24, r24)):
                rows.append({"event": f"{s} X{i}", "season": s, "circuit": f"X{i}", "C": c, "grp": c,
                             "label": c, "practice_rate": p, "race_rate": r, "se_p": 0.02})
    rc = pd.DataFrame(cells)
    return E.attach_history(pd.DataFrame(rows), rc), rc


def test_history_records_which_race_it_came_from():
    panel, _ = _two_season_panel()
    p24 = panel[panel["season"] == 2024]
    assert (p24["hist_event"] == p24["event"].str.replace("2024", "2023")).all()
    assert panel.loc[panel["season"] == 2023, "hist_event"].isna().all()


def test_the_held_out_weekends_outcome_does_not_reach_its_own_calibration_through_history():
    """In leave-one-weekend-out over stacked seasons, training cells from later
    seasons carried the held-out weekend's race rate as their history, which sets the
    stacking weight. Wrecking a held-out weekend's race rate must not move ITS forecast."""
    panel, rc = _two_season_panel()
    held = "2023 X2"
    for race_cells in (None, rc):       # cleared, and rebuilt from the race-cell table
        base = E.loeo(panel, race_cells=race_cells).set_index(["event", "C"])
        wrecked = panel.copy()
        wrecked.loc[wrecked["event"] == held, "race_rate"] += 0.5
        rc2 = rc.copy()
        rc2.loc[rc2["event"] == held, "race_rate"] += 0.5
        wrecked = E.attach_history(wrecked.drop(columns=["hist_rate", "hist_season", "hist_event"]), rc2)
        moved = E.loeo(wrecked, race_cells=rc2 if race_cells is not None else None).set_index(["event", "C"])
        mine = base.index.get_level_values(0) == held
        for col in ("pred_stack", "pred_scalar", "pred_pooled", "pred_shrink50", "stack_w"):
            np.testing.assert_allclose(base.loc[mine, col], moved.loc[mine, col], err_msg=col)


def test_without_the_fix_the_leak_is_real():
    """The control: fit the calibration on training cells that still carry the held-out
    race as their history, and the stacking weight moves when that race's rate moves."""
    panel, rc = _two_season_panel()
    held = "2023 X2"
    tr = panel[panel["event"] != held]
    wrecked_rc = rc.copy()
    wrecked_rc.loc[wrecked_rc["event"] == held, "race_rate"] += 0.5
    tr_w = E.attach_history(tr.drop(columns=["hist_rate", "hist_season", "hist_event"]), wrecked_rc)
    leaky_a = E.fit_calibration(tr)["stack_w"]
    leaky_b = E.fit_calibration(tr_w)["stack_w"]
    assert leaky_a != pytest.approx(leaky_b, abs=1e-6), "the fixture must exhibit the leak it guards against"
    clean_a = E.fit_calibration(E.drop_heldout_history(tr, [held], rc))["stack_w"]
    clean_b = E.fit_calibration(E.drop_heldout_history(tr_w, [held], wrecked_rc))["stack_w"]
    assert clean_a == pytest.approx(clean_b)


def test_history_is_rebuilt_from_the_next_earlier_season_when_there_is_one():
    rc = pd.DataFrame([
        {"event": "2022 X", "season": 2022, "circuit": "X", "label": "SOFT", "race_rate": 0.10},
        {"event": "2023 X", "season": 2023, "circuit": "X", "label": "SOFT", "race_rate": 0.20},
    ])
    panel = pd.DataFrame([{"event": "2024 X", "season": 2024, "circuit": "X", "C": "SOFT", "label": "SOFT",
                           "grp": "SOFT", "practice_rate": 0.1, "race_rate": 0.05}])
    panel = E.attach_history(panel, rc)
    assert panel["hist_rate"].iloc[0] == 0.20 and panel["hist_event"].iloc[0] == "2023 X"
    out = E.drop_heldout_history(panel, ["2023 X"], rc)
    assert out["hist_rate"].iloc[0] == 0.10 and out["hist_event"].iloc[0] == "2022 X"
    assert E.drop_heldout_history(panel, ["2023 X"], None)["hist_rate"].isna().all()


def test_frozen_holdout_does_not_train_on_the_test_weekends_outcomes():
    """A 2026 training cell's history is a 2022-25 race rate: a TEST outcome."""
    panel, rc = _two_season_panel()
    train = panel[panel["season"] == 2024].copy()      # hist_rate = the 2023 race rate
    test = panel[panel["season"] == 2023].copy()
    base = E.frozen_holdout(train, test, race_cells=rc)
    rc2 = rc.copy()
    rc2.loc[rc2["season"] == 2023, "race_rate"] += 0.5
    wrecked_train = E.attach_history(train.drop(columns=["hist_rate", "hist_season", "hist_event"]), rc2)
    moved = E.frozen_holdout(wrecked_train, test, race_cells=rc2)
    np.testing.assert_allclose(base["stack_w"], moved["stack_w"])


def test_the_monte_carlo_signflip_p_value_is_never_exactly_zero():
    """With more than 14 clusters the randomisation test draws sign patterns; a
    p-value of (hits)/(draws) reports 0 for a strong effect. The observed statistic is
    one of the permutations, so the estimate is (hits + 1)/(draws + 1)."""
    rng = np.random.default_rng(0)
    g = 24
    cl = np.repeat(np.arange(g), 4)
    a = rng.normal(0.02, 0.002, g * 4)       # A's absolute errors, far smaller than B's in every weekend
    b = rng.normal(0.30, 0.002, g * 4)
    r = E.paired_test(a, b, cl, n_boot=200)
    assert r["signflip_p"] > 0
    assert r["signflip_p"] == pytest.approx(1 / 20_001)
    # The exact enumeration (<= 14 weekends) is unchanged and still reaches 1.0 on a tie.
    cl2 = np.repeat(np.arange(6), 3)
    z = rng.normal(0, 1, 18)
    assert E.paired_test(z, z, cl2, n_boot=100)["signflip_p"] == 1.0

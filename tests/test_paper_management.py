"""Tests for the paper's driver-management analysis.

Synthetic data with known truth throughout: if the estimator cannot recover a
planted effect, and cannot stay quiet when none is planted, nothing it says about
the real laps means anything.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tyredeg.models.design import race_design
from tyredeg.paper import management as M
from tyredeg.validation.management import cell_slopes, demean_by_compound

# ---------------------------------------------------------------------------
# Synthetic lap frames
# ---------------------------------------------------------------------------


def make_laps(slopes, *, session="R", seed=0, n_runs=5, n_laps=12, age0=6, event="E1 GP", season=2024,
              fuel_s_per_lap=0.0, noise=0.05):
    """Runs with known tyre slopes per compound. Lap time includes a fuel benefit."""
    rng = np.random.default_rng(seed)
    rows = []
    for comp, theta in slopes.items():
        for r in range(n_runs):
            start = age0 + (r % 3)
            lap0 = 5 + 3 * r
            for i in range(n_laps):
                age = start + i
                rows.append(
                    {
                        "season": season,
                        "event": event,
                        "session": session,
                        "Compound": comp,
                        "run_id": f"{season}|{event}|{session}|{comp}|{r}",
                        "TyreLife": float(age),
                        "LapNumber": float(lap0 + i),
                        "LapTimeSeconds": 90 + theta * age - fuel_s_per_lap * i + rng.normal(0, noise),
                        "track_temp": 30.0 + i * 0.01,
                        "SessionElapsed": 600.0 + 90 * (lap0 + i),
                        "run_len": n_laps,
                    }
                )
    return pd.DataFrame(rows)


def cells_from(slopes_race, slopes_prac, n_events=8, seed=0, **kw):
    race, prac = [], []
    for e in range(n_events):
        ev = f"E{e} GP"
        race.append(make_laps(slopes_race, session="R", event=ev, seed=seed + e, **kw))
        prac.append(make_laps(slopes_prac, session="FP2", event=ev, seed=seed + 100 + e, **kw))
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None, min_window_laps=4)
    r = M.prepare_context(pd.concat(race, ignore_index=True), spec, race=True, burn=spec.burn_race, caps=None)
    p = M.prepare_context(pd.concat(prac, ignore_index=True), spec, race=False, burn=spec.burn_practice, caps=None)
    return M.build_cells(r, p, spec.min_runs)


def synthetic_cells(beta, n_events=30, noise=0.02, seed=0, tau=0.0):
    """Cell table with a planted interaction: delta = a_event + beta * rank + noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        a_e = rng.normal(0, 0.03)
        for rank, lab in enumerate(M.LABELS):
            se = 0.02
            d = a_e + beta * rank + rng.normal(0, np.hypot(se, tau)) + rng.normal(0, noise)
            rows.append(
                {"season": 2022 + e % 5, "event": f"E{e}", "cluster": f"{2022 + e % 5}|E{e}", "Compound": lab,
                 "rank": rank, "delta": d, "var_delta": 2 * se**2, "slope_r": 0.05 + 0.02 * rank + d / 2,
                 "slope_p": 0.05 + 0.02 * rank - d / 2, "se_r": se, "se_p": se, "sxy_r": 1.0, "sxx_r": 10.0,
                 "sxy_p": 1.0, "sxx_p": 10.0, "temp_r": 35.0 + rng.normal(0, 3), "temp_p": 35.0 + rng.normal(0, 3)}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


def test_same_estimator_gives_zero_contrast_on_identical_data():
    d = make_laps({"HARD": 0.03, "MEDIUM": 0.05, "SOFT": 0.08}, session="FP2")
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None)
    a = M.prepare_context(d, spec, race=False, burn=1.7, caps=None)
    b = M.prepare_context(d.assign(session="R"), spec, race=True, burn=1.7, caps=None)
    cells = M.build_cells(b, a, 3)
    assert len(cells) == 3
    assert np.allclose(cells["delta"], 0.0)


def test_cell_slope_recovers_known_rate_after_fuel_correction():
    # Lap time falls 0.056 s/lap from fuel burn; the correction must put it back.
    slopes = {"HARD": 0.04, "MEDIUM": 0.06, "SOFT": 0.09}
    d = make_laps(slopes, session="FP2", fuel_s_per_lap=1.7 * 0.033, noise=0.02)
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None)
    f = M.prepare_context(d, spec, race=False, burn=1.7, caps=None)
    got = M.cell_fits(f, 3).set_index("Compound")["slope"]
    for lab, th in slopes.items():
        assert got[lab] == pytest.approx(th, abs=0.01)


def test_fuel_assumption_shifts_levels_but_not_the_compound_contrast():
    """The theorem the fuel grid rests on: a constant per-lap correction cancels."""
    race_s = {"HARD": 0.03, "MEDIUM": 0.04, "SOFT": 0.04}
    prac_s = {"HARD": 0.03, "MEDIUM": 0.07, "SOFT": 0.12}
    out = {}
    for burn, s in ((1.4, 0.025), (2.0, 0.040)):
        race, prac = [], []
        for e in range(8):
            race.append(make_laps(race_s, session="R", event=f"E{e}", seed=e))
            prac.append(make_laps(prac_s, session="FP2", event=f"E{e}", seed=100 + e))
        spec = replace(M.PRIMARY, age_lo=None, age_hi=None, s_per_kg=s, burn_race=burn, burn_practice=burn)
        r = M.prepare_context(pd.concat(race), spec, race=True, burn=burn, caps=None)
        p = M.prepare_context(pd.concat(prac), spec, race=False, burn=burn, caps=None)
        c = M.build_cells(r, p, 3)
        out[(burn, s)] = (M.trend_fit(M.Arrays.from_cells(c), tau2=0.0)["beta"], c["slope_r"].mean())
    (b1, lvl1), (b2, lvl2) = out.values()
    assert b1 == pytest.approx(b2, abs=1e-9)  # contrast invariant
    assert lvl2 - lvl1 == pytest.approx(2.0 * 0.040 - 1.4 * 0.025, abs=1e-6)  # level moves by the correction


def test_window_drops_outside_laps_and_short_runs():
    d = make_laps({"HARD": 0.03}, session="FP2", n_runs=4, n_laps=14, age0=2)
    spec = replace(M.PRIMARY, age_lo=6, age_hi=10, min_window_laps=4)
    w = M._window(d, spec, None)
    assert w["TyreLife"].between(6, 10).all()
    # A run that only has 2 laps inside the window is removed entirely.
    spec2 = replace(spec, age_lo=14, age_hi=16, min_window_laps=4)
    assert M._window(d, spec2, None).empty


def test_run_length_band_filters_on_full_run_length():
    d = make_laps({"HARD": 0.03}, session="FP2", n_runs=3, n_laps=12)
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None, run_len=(13, 20))
    assert M._window(d, spec, None).empty
    spec = replace(spec, run_len=(10, 15))
    assert not M._window(d, spec, None).empty


def test_common_support_caps_event_to_the_shortest_group():
    short = make_laps({"SOFT": 0.1}, session="FP2", n_runs=3, n_laps=6, age0=4)  # reaches 9-11
    long = make_laps({"HARD": 0.03}, session="R", n_runs=3, n_laps=20, age0=4)  # reaches 23-25
    caps = M._common_support_caps([short, long], replace(M.PRIMARY, age_lo=6))
    cap = caps[(2024, "E1 GP")]
    assert 9 <= cap <= 11


# ---------------------------------------------------------------------------
# The legacy race-estimator defect
# ---------------------------------------------------------------------------


def simulate_race(theta, offset, n_events=1, seed=0, drivers=18, laps=50):
    """A race with two stints per car, a pace offset per compound, and fuel burn."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        for d in range(drivers):
            soft_first = d % 3 != 0
            pit = int(rng.integers(12, 20) if soft_first else rng.integers(24, 32))
            driver_pace = rng.normal(0, 0.15)
            for lap in range(1, laps + 1):
                first = lap <= pit
                comp = ("SOFT" if soft_first else "HARD") if first else ("HARD" if soft_first else "SOFT")
                age = lap if first else lap - pit
                stint = 1 if first else 2
                t = (92 + driver_pace - 0.056 * lap + theta[comp] * age + offset[comp] + rng.normal(0, 0.08))
                rows.append({"event": f"R{e} GP", "session": "R", "Driver": f"D{d}", "Stint": stint,
                             "LapNumber": float(lap), "TyreLife": float(age), "LapTimeSeconds": t, "Compound": comp,
                             "run_id": f"R{e}|{d}|{stint}", "season": 2024, "track_temp": 30.0})
    return pd.DataFrame(rows)


def _slopes(frame):
    return cell_slopes(frame, min_runs=3).groupby("Compound")["rate"].mean()


def test_legacy_race_slope_is_biased_and_compound_demeaning_repairs_it():
    theta = {"SOFT": 0.07, "HARD": 0.07}  # identical true slopes
    offset = {"SOFT": -0.9, "HARD": 0.0}  # soft is much quicker
    race = race_design(simulate_race(theta, offset, n_events=10), with_traffic=False)
    old = _slopes(race)
    new = _slopes(demean_by_compound(race))
    # Truth is equal; the old estimator invents a gradient, the repaired one does not.
    assert old["SOFT"] - old["HARD"] > 0.03
    assert abs(new["SOFT"] - new["HARD"]) < 0.02
    assert new["HARD"] == pytest.approx(0.07, abs=0.015)


def test_repair_keeps_a_real_gradient():
    theta = {"SOFT": 0.10, "HARD": 0.05}
    offset = {"SOFT": -0.9, "HARD": 0.0}
    race = race_design(simulate_race(theta, offset, n_events=10, seed=1), with_traffic=False)
    new = _slopes(demean_by_compound(race))
    assert new["SOFT"] == pytest.approx(0.10, abs=0.02)
    assert new["HARD"] == pytest.approx(0.05, abs=0.02)


def test_centring_the_one_pass_design_is_not_the_repair():
    """Documented dead end: run-centring the field-demeaned design is also biased."""
    theta = {"SOFT": 0.07, "HARD": 0.07}
    offset = {"SOFT": -0.9, "HARD": 0.0}
    race = race_design(simulate_race(theta, offset, n_events=10), with_traffic=False)
    g = race.groupby("run_id")
    centred = race.assign(y=race["y"] - g["y"].transform("mean"), tl=race["tl"] - g["tl"].transform("mean"))
    s = _slopes(centred)
    assert abs(s["SOFT"] - 0.07) > 0.02 or abs(s["HARD"] - 0.07) > 0.02


def test_within_run_race_slope_matches_truth_on_the_same_simulation():
    """The primary estimator is unbiased here (fuel known, no track evolution)."""
    theta = {"SOFT": 0.07, "HARD": 0.07}
    offset = {"SOFT": -0.9, "HARD": 0.0}
    df = simulate_race(theta, offset, n_events=10)
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None, min_window_laps=4)
    f = M.prepare_context(df.assign(run_len=20), spec, race=True, burn=1.7, caps=None)
    got = M.cell_fits(f, 3).groupby("Compound")["slope"].mean()
    assert got["SOFT"] == pytest.approx(0.07, abs=0.02)
    assert got["HARD"] == pytest.approx(0.07, abs=0.02)
    assert abs(got["SOFT"] - got["HARD"]) < 0.02


# ---------------------------------------------------------------------------
# Stage two
# ---------------------------------------------------------------------------


def test_trend_recovers_a_planted_interaction():
    cells = synthetic_cells(beta=-0.05, seed=1)
    fit = M.trend_fit(M.Arrays.from_cells(cells))
    assert fit["beta"] == pytest.approx(-0.05, abs=0.01)


def test_planted_interaction_is_detected_and_a_null_is_not():
    eff = M.quick(synthetic_cells(-0.05, seed=2), B=300, P=400, seed=1)
    assert eff["excludes_zero_negative"] and eff["p_perm_one_sided"] < 0.05
    null = M.quick(synthetic_cells(0.0, seed=3), B=300, P=400, seed=1)
    assert not null["excludes_zero_negative"]
    assert null["p_perm_one_sided"] > 0.05
    assert null["ci95"][0] < 0 < null["ci95"][1]


def test_wrong_direction_effect_is_not_called_supported():
    r = M.quick(synthetic_cells(+0.05, seed=4), B=300, P=400, seed=1)
    assert not r["excludes_zero_negative"]
    assert r["p_perm_one_sided"] > 0.5


def test_permutation_only_shuffles_within_event():
    a = M.Arrays.from_cells(synthetic_cells(-0.05, n_events=10, seed=5))
    rng = np.random.default_rng(0)
    new = M._perm_ranks(a, rng)
    for idx in a.members():
        assert sorted(new[idx]) == sorted(a.rank[idx])
    assert (new != a.rank).any()


def test_bootstrap_is_reproducible_from_its_seed():
    a = M.Arrays.from_cells(synthetic_cells(-0.03, n_events=12, seed=6))
    f = lambda x: M.trend_beta(x, 0.0)  # noqa: E731
    one = M.boot_ci(f, a, 50, seed=9)[:2]
    two = M.boot_ci(f, a, 50, seed=9)[:2]
    assert one == two


def test_pm_tau2_is_zero_when_cells_agree_within_their_errors_and_positive_otherwise():
    rng = np.random.default_rng(0)
    v = np.full(40, 0.01)
    X = np.ones((40, 1))
    assert M.pm_tau2(rng.normal(0, 0.1, 40), X, v) == pytest.approx(0.0, abs=1e-3)
    assert M.pm_tau2(rng.normal(0, 0.5, 40), X, v) > 0.1


def test_sign_test_counts_and_ties():
    c = synthetic_cells(-0.1, n_events=20, noise=0.001, seed=7)
    a = M.Arrays.from_cells(c)
    st = M.sign_test(a, 2, 0)
    assert st["n_events"] == 20 and st["n_lower"] >= 18
    assert st["p_one_sided"] < 0.001


def test_pair_contrast_matches_planted_difference():
    a = M.Arrays.from_cells(synthetic_cells(-0.05, n_events=40, seed=8))
    pc = M.pair_contrast(a, 2, 0)
    assert pc["estimate"] == pytest.approx(-0.10, abs=0.02)
    assert pc["n_events"] == 40


def test_ratio_of_means_is_a_ratio_of_sums():
    a = M.Arrays.from_cells(synthetic_cells(0.0, n_events=5, seed=9))
    k = 1
    m = a.rank == k
    assert M.ratio_of_means(a, k) == pytest.approx(a.race[m].sum() / a.prac[m].sum())


def test_multiplicative_damping_shows_in_delta_but_not_in_the_ratio_gradient():
    """The reason H2 exists: uniform damping fakes an additive interaction."""
    rng = np.random.default_rng(0)
    rows = []
    for e in range(40):
        for rank, lab in enumerate(M.LABELS):
            prac = 0.04 + 0.04 * rank + rng.normal(0, 0.005)
            race = 0.5 * prac + rng.normal(0, 0.005)  # same damping factor for all compounds
            rows.append({"season": 2022 + e % 5, "event": f"E{e}", "cluster": f"{2022 + e % 5}|E{e}", "Compound": lab,
                         "rank": rank, "delta": race - prac, "var_delta": 2e-4, "slope_r": race, "slope_p": prac,
                         "se_r": 0.01, "se_p": 0.01, "sxy_r": race, "sxx_r": 1.0, "sxy_p": prac, "sxx_p": 1.0,
                         "temp_r": 30.0, "temp_p": 30.0})
    a = M.Arrays.from_cells(pd.DataFrame(rows))
    assert M.trend_fit(a)["beta"] < -0.01  # additive interaction is there
    assert abs(M.log_ratio_gradient(a, 2, 0)) < 0.15  # but nothing beyond proportional damping


def test_pooled_delta_equals_a_single_lap_level_fit():
    slopes_r = {"HARD": 0.03, "MEDIUM": 0.04, "SOFT": 0.05}
    slopes_p = {"HARD": 0.03, "MEDIUM": 0.07, "SOFT": 0.12}
    cells = cells_from(slopes_r, slopes_p, n_events=6)
    a = M.Arrays.from_cells(cells)
    for lab in M.LABELS:
        k = M.LABEL_RANK[lab]
        assert M.pooled_delta(a, k) == pytest.approx(
            a.sxy_r[a.rank == k].sum() / a.sxx_r[a.rank == k].sum()
            - a.sxy_p[a.rank == k].sum() / a.sxx_p[a.rank == k].sum()
        )


def test_end_to_end_on_planted_laps_recovers_interaction_sign():
    cells = cells_from({"HARD": 0.03, "MEDIUM": 0.035, "SOFT": 0.04}, {"HARD": 0.03, "MEDIUM": 0.07, "SOFT": 0.12})
    fit = M.trend_fit(M.Arrays.from_cells(cells))
    # truth: delta = 0, -0.035, -0.08 -> -0.04 per step
    assert fit["beta"] == pytest.approx(-0.04, abs=0.01)


# ---------------------------------------------------------------------------
# Seasons, temperature, C numbers
# ---------------------------------------------------------------------------


def test_season_analysis_reports_each_season_and_a_pool():
    out = M.season_analysis(synthetic_cells(-0.05, n_events=40, seed=10), B=100, P=100, seed=1)
    assert set(out["each_season_alone"]) == {"2022", "2023", "2024", "2025", "2026"}
    assert set(out["leave_one_season_out"]) == set(out["each_season_alone"])
    assert out["season_pool"]["random_effects_beta"] < 0
    assert out["season_pool"]["n_seasons_negative"] >= 4


def test_temperature_control_does_not_move_a_trend_unrelated_to_temperature():
    out = M.temperature_models(synthetic_cells(-0.05, n_events=40, seed=11), B=100, seed=1)
    b0 = out["models"]["event_fe_no_temp"]["beta"]
    b1 = out["models"]["event_fe_temp_both"]["beta"]
    assert b1 == pytest.approx(b0, abs=0.01)


def test_temperature_control_absorbs_a_trend_that_is_really_temperature():
    rng = np.random.default_rng(0)
    c = synthetic_cells(0.0, n_events=60, noise=0.001, seed=12)
    # Make the softer compounds' race sessions hotter, and let delta follow temperature.
    c["temp_r"] = 30.0 + 4.0 * c["rank"] + rng.normal(0, 0.3, len(c))
    c["temp_p"] = 30.0
    c["delta"] = -0.01 * (c["temp_r"] - c["temp_p"]) + rng.normal(0, 0.003, len(c))
    out = M.temperature_models(c, B=100, seed=1)
    assert out["models"]["event_fe_no_temp"]["beta"] < -0.03
    assert abs(out["models"]["event_fe_temp_diff"]["beta"]) < 0.01


def test_weather_streams_and_interpolation(tmp_path):
    ses = tmp_path / "2024" / "2024-03-02_Bahrain_Grand_Prix" / "2024-02-29_Practice_2"
    ses.mkdir(parents=True)
    data = {"Time": [pd.Timedelta(seconds=0), pd.Timedelta(seconds=100), pd.Timedelta(seconds=200)],
            "TrackTemp": [30.0, 40.0, 50.0]}
    with open(ses / "weather_data.ff1pkl", "wb") as fh:
        pickle.dump({"version": 15, "data": data}, fh)
    streams = M.weather_streams(tmp_path)
    assert (2024, "Bahrain Grand Prix", "FP2") in streams
    df = pd.DataFrame({"season": [2024, 2024], "event": ["Bahrain Grand Prix"] * 2, "session": ["FP2", "FP2"],
                       "SessionElapsed": [50.0, 150.0]})
    out = M.attach_track_temp(df, streams)
    assert out["track_temp"].tolist() == pytest.approx([35.0, 45.0])


def test_attach_track_temp_leaves_nan_when_no_stream():
    df = pd.DataFrame({"season": [2024], "event": ["X GP"], "session": ["R"], "SessionElapsed": [10.0]})
    assert M.attach_track_temp(df, {})["track_temp"].isna().all()


def test_allocation_history_loader_and_cnumber_trend(tmp_path):
    hist = tmp_path / "h.json"
    cur = tmp_path / "c.json"
    seasons = {"2022": {}}
    # Events alternate between a C2-C4 and a C3-C5 nomination.
    for e in range(40):
        base = 2 + (e % 2)
        seasons["2022"][f"E{e}"] = {"HARD": f"C{base}", "MEDIUM": f"C{base + 1}", "SOFT": f"C{base + 2}", "verified": e != 3}
    hist.write_text(json.dumps({"seasons": seasons}), encoding="utf-8")
    cur.write_text(json.dumps([{"event": "Z GP", "season": 2026, "compounds": {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"}}]),
                   encoding="utf-8")
    alloc = M.load_allocation_history(hist, cur)
    assert alloc[(2022, "E0")] == {"HARD": 2, "MEDIUM": 3, "SOFT": 4}
    assert alloc[(2026, "Z GP")]["SOFT"] == 5
    assert M.unverified_events(hist) == ["2022|E3"]

    # delta depends on the C number itself, not on the label.
    rng = np.random.default_rng(0)
    rows = []
    for e in range(40):
        for rank, lab in enumerate(M.LABELS):
            cnum = alloc[(2022, f"E{e}")][lab]
            rows.append({"season": 2022, "event": f"E{e}", "cluster": f"2022|E{e}", "Compound": lab, "rank": rank,
                         "delta": -0.04 * cnum + rng.normal(0, 0.01), "var_delta": 1e-4, "slope_r": 0, "slope_p": 0,
                         "se_r": 0.01, "se_p": 0.01, "sxy_r": 1, "sxx_r": 1, "sxy_p": 1, "sxx_p": 1,
                         "temp_r": 30.0, "temp_p": 30.0})
    res = M.cnumber_trend(pd.DataFrame(rows), alloc, B=100, seed=1)
    assert res["status"] == "ok"
    assert res["beta_per_C"] == pytest.approx(-0.04, abs=0.01)
    assert res["ci95"][1] < 0


def test_missing_allocation_files_give_an_empty_history(tmp_path):
    assert M.load_allocation_history(tmp_path / "no.json", tmp_path / "no2.json") == {}


def test_spec_label_is_stable_and_primary_is_the_documented_one():
    assert M.PRIMARY.age_lo == 6 and M.PRIMARY.age_hi == 16
    assert M.PRIMARY.burn_race == M.PRIMARY.burn_practice == 1.7
    assert M.PRIMARY.min_runs == 3 and M.PRIMARY.sprint == "exclude"
    assert M.PRIMARY.label() == replace(M.PRIMARY).label()
    assert M.PRIMARY.label() != replace(M.PRIMARY, age_lo=3).label()


# ---------------------------------------------------------------------------
# Post-review additions
# ---------------------------------------------------------------------------


def laps_with_nuisance(slopes, session, event, seed, lap_effect=True, n_runs=6, **kw):
    """make_laps plus a common drift every car shares (track evolution) and a per-run baseline."""
    d = make_laps(slopes, session=session, event=event, seed=seed, n_runs=n_runs, **kw)
    rng = np.random.default_rng(seed + 1)
    base = {r: rng.normal(0, 0.3) for r in d["run_id"].unique()}
    d["LapTimeSeconds"] = d["LapTimeSeconds"] + d["run_id"].map(base)
    if lap_effect:
        d["LapTimeSeconds"] = d["LapTimeSeconds"] + 0.4 * np.sin(d["LapNumber"] / 4.0) - 0.02 * d["LapNumber"]
    return d.assign(is_race_sim=True, start_age=d.groupby("run_id")["TyreLife"].transform("min"))


def fake_bases(race_slopes, prac_slopes, n_events=14):
    race, prac = [], []
    for e in range(n_events):
        race.append(laps_with_nuisance(race_slopes, "R", f"E{e} GP", seed=e))
        prac.append(laps_with_nuisance(prac_slopes, "FP2", f"E{e} GP", seed=50 + e))
    return M.Bases(practice={2024: pd.concat(prac, ignore_index=True)}, race={2024: pd.concat(race, ignore_index=True)},
                   seasons=(2024,))


def test_twfe_residuals_remove_both_fixed_effects():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 6, 200)
    b = rng.integers(0, 9, 200)
    y = rng.normal(size=200) + a * 2.0 + b * 3.0
    (r,) = M.twfe_residuals(a, b, y)
    for k in range(6):
        assert abs(r[a == k].mean()) < 1e-6
    for k in range(9):
        assert abs(r[b == k].mean()) < 1e-6


def test_pair_age_contrast_recovers_slope_difference_despite_common_nuisance():
    d = laps_with_nuisance({"HARD": 0.03, "SOFT": 0.09}, "R", "E GP", seed=3, n_runs=8)
    d["fe_key"] = d["LapNumber"].astype(int).astype(str)
    got = M.pair_age_contrast(d, "SOFT", "HARD", 2)
    assert got is not None
    assert got[0] == pytest.approx(0.06, abs=0.02)
    assert got[1] > 0


def test_twfe_gradient_recovers_a_planted_flatter_race_gradient():
    bases = fake_bases({"HARD": 0.03, "MEDIUM": 0.035, "SOFT": 0.04}, {"HARD": 0.03, "MEDIUM": 0.07, "SOFT": 0.12})
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None, seasons=(2024,))
    tab = M.twfe_pair_table(bases, spec, min_runs=2)
    assert set(tab["pair"]) == {"SOFT-HARD", "MEDIUM-HARD", "SOFT-MEDIUM"}
    out = M.twfe_gradient(tab, B=200, P=200, seed=1)
    # truth: SOFT-HARD DiD = (0.04-0.03) - (0.12-0.03) = -0.08
    assert out["SOFT-HARD"]["estimate_s_per_lap"] == pytest.approx(-0.08, abs=0.03)
    assert out["SOFT-HARD"]["ci95"][1] < 0
    assert out["SOFT-HARD"]["mde80_s_per_lap"] > 0


def test_twfe_gradient_is_quiet_when_nothing_is_planted():
    same = {"HARD": 0.03, "MEDIUM": 0.05, "SOFT": 0.08}
    bases = fake_bases(same, same)
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None, seasons=(2024,))
    out = M.twfe_gradient(M.twfe_pair_table(bases, spec, min_runs=2), B=200, P=200, seed=1)
    assert abs(out["SOFT-HARD"]["estimate_s_per_lap"]) < 0.03
    assert out["SOFT-HARD"]["ci95"][0] < 0 < out["SOFT-HARD"]["ci95"][1]


def test_level_gap_recovers_uniform_damping_and_additive_gap_still_grows():
    rng = np.random.default_rng(0)
    rows = []
    for e in range(30):
        for rank, lab in enumerate(M.LABELS):
            prac = 0.05 + 0.04 * rank
            rows.append({"season": 2022 + e % 5, "event": f"E{e}", "cluster": f"{2022 + e % 5}|E{e}", "Compound": lab,
                         "rank": rank, "delta": 0.0, "var_delta": 1e-4, "slope_r": 0.5 * prac, "slope_p": prac,
                         "se_r": 0.01, "se_p": 0.01, "sxy_r": 0.5 * prac * 10 + rng.normal(0, 0.01), "sxx_r": 10.0,
                         "sxy_p": prac * 10, "sxx_p": 10.0, "temp_r": 30.0, "temp_p": 30.0})
    out = M.level_gaps(pd.DataFrame(rows), B=200, seed=1)
    assert out["MEDIUM+SOFT"]["ratio"] == pytest.approx(0.5, abs=0.02)
    assert out["MEDIUM+SOFT"]["gap"] < 0
    assert out["MEDIUM+SOFT"]["gap_ci95"][1] < 0
    # damping is uniform in RATIO, so the additive gap still grows with softness -- the point of H2
    assert out["SOFT"]["gap"] < out["HARD"]["gap"]


def test_uniform_damping_implies_a_negative_additive_beta():
    c = synthetic_cells(0.0, n_events=30, seed=21)
    assert M.uniform_damping_beta(c, 0.5) < 0
    assert M.uniform_damping_beta(c, 1.0) == pytest.approx(0.0, abs=1e-9)


def test_power_rises_with_effect_size_and_reports_an_mde():
    c = synthetic_cells(0.0, n_events=25, noise=0.0, seed=22)
    out = M.power_simulation(c, (0.0, -0.1), n_sims=12, B=39, P=39, seed=1)
    assert [r["beta"] for r in out["grid"]] == [0.0, -0.1]
    assert out["grid"][1]["power_ci_only"] >= out["grid"][0]["power_ci_only"]
    assert out["grid"][1]["power_full_rule"] > out["grid"][0]["power_full_rule"]


def test_set_state_matching_filters_on_start_age():
    race = make_laps({"HARD": 0.03}, session="R", n_runs=4, age0=2).assign(is_race_sim=True)
    race["start_age"] = race.groupby("run_id")["TyreLife"].transform("min")
    prac = make_laps({"HARD": 0.03}, session="FP2", n_runs=4, age0=9).assign(is_race_sim=True)
    prac["start_age"] = prac.groupby("run_id")["TyreLife"].transform("min")
    b = M.Bases(practice={2024: prac}, race={2024: race}, seasons=(2024,))
    spec = replace(M.PRIMARY, seasons=(2024,), practice_start=(5, 9), race_start=(5, 99), age_lo=None, age_hi=None)
    r, p = M.context_frames(b, spec, prepare=False)
    assert r.empty  # every race stint here starts at age 2-4: excluded when matching on used sets
    assert not p.empty and p["start_age"].between(5, 9).all()
    r2, _ = M.context_frames(b, replace(spec, race_start=None), prepare=False)
    assert not r2.empty


def test_pooled_cells_keep_single_run_cells_when_se_is_not_needed():
    race = make_laps({"HARD": 0.03}, session="R", n_runs=1)
    prac = make_laps({"HARD": 0.05}, session="FP2", n_runs=1)
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None, min_runs=1)
    r = M.prepare_context(race, spec, race=True, burn=1.7, caps=None)
    p = M.prepare_context(prac, spec, race=False, burn=1.7, caps=None)
    assert M.build_cells(r, p, 1).empty
    assert len(M.build_cells(r, p, 1, need_se=False)) == 1


# ---------------------------------------------------------------------------
# Referee additions
# ---------------------------------------------------------------------------


def test_holm_adjustment_is_stepdown_monotone_and_capped():
    out = M.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5})
    assert out["a"] == pytest.approx(0.03)
    assert out["b"] == pytest.approx(0.08)
    assert out["c"] == pytest.approx(0.5)
    out = M.holm_adjust({"a": 0.4, "b": 0.45, "c": 0.9})
    assert max(out.values()) <= 1.0
    # monotone in the raw ordering
    assert out["a"] <= out["b"] <= out["c"]


def _gap_cells(seed, race_gap, n=30):
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n):
        for rank, lab in enumerate(M.LABELS):
            prac = 0.1
            race = prac + race_gap
            rows.append({"season": 2022 + e % 5, "event": f"E{e}", "cluster": f"{2022 + e % 5}|E{e}", "Compound": lab,
                         "rank": rank, "delta": race_gap, "var_delta": 1e-4, "slope_r": race, "slope_p": prac,
                         "se_r": 0.01, "se_p": 0.01, "sxy_r": race * 10 + rng.normal(0, 0.02), "sxx_r": 10.0,
                         "sxy_p": prac * 10 + rng.normal(0, 0.02), "sxx_p": 10.0, "temp_r": 30.0, "temp_p": 30.0})
    return pd.DataFrame(rows)


def test_placebo_adjusted_gap_subtracts_the_placebo_and_uses_a_joint_bootstrap():
    main = _gap_cells(1, -0.10)
    plc = _gap_cells(2, -0.04)
    out = M.placebo_adjusted_gap(main, plc, (1, 2), B=300, seed=1)
    assert out["race_minus_practice_gap"] == pytest.approx(-0.10, abs=0.01)
    assert out["placebo_gap_fp2_minus_fp1fp3"] == pytest.approx(-0.04, abs=0.01)
    assert out["placebo_adjusted_gap"] == pytest.approx(-0.06, abs=0.015)
    lo, hi = out["placebo_adjusted_gap_ci95_joint_bootstrap"]
    assert lo < out["placebo_adjusted_gap"] < hi
    assert out["placebo_share_of_gap"] == pytest.approx(0.4, abs=0.1)
    assert out["n_event_seasons_in_both"] == out["n_event_seasons_main"]


def test_placebo_adjusted_gap_is_zero_when_placebo_equals_the_gap():
    main = _gap_cells(3, -0.07)
    out = M.placebo_adjusted_gap(main, main.copy(), (0, 1, 2), B=200, seed=1)
    assert out["placebo_adjusted_gap"] == pytest.approx(0.0, abs=1e-9)
    assert out["placebo_adjusted_gap_ci95_joint_bootstrap"][0] <= 0 <= out["placebo_adjusted_gap_ci95_joint_bootstrap"][1]


def test_race_only_trend_recovers_a_planted_compound_trend():
    rng = np.random.default_rng(0)
    rows = []
    for e in range(25):
        for rank, lab in enumerate(M.LABELS):
            rows.append({"season": 2022 + e % 5, "event": f"E{e}", "Compound": lab,
                         "slope": 0.02 + 0.015 * rank + rng.normal(0, 0.01), "se": 0.01})
    out = M.race_only_trend(pd.DataFrame(rows), B=200, seed=1)
    assert out["beta_per_step_softer"] == pytest.approx(0.015, abs=0.006)
    assert out["ci95"][0] > 0


def test_fresh_flag_filters_set_state_directly():
    race = make_laps({"HARD": 0.03}, session="R", n_runs=4, age0=2).assign(is_race_sim=True)
    race["start_age"] = race.groupby("run_id")["TyreLife"].transform("min")
    race["fresh"] = race["run_id"].str.endswith(("0", "1"))  # two new-set runs, two used-set runs
    prac = make_laps({"HARD": 0.03}, session="FP2", n_runs=4, age0=9).assign(is_race_sim=True, fresh=False)
    prac["start_age"] = prac.groupby("run_id")["TyreLife"].transform("min")
    b = M.Bases(practice={2024: prac}, race={2024: race}, seasons=(2024,))
    base = replace(M.PRIMARY, seasons=(2024,), age_lo=None, age_hi=None)
    r_used, p_used = M.context_frames(b, replace(base, race_fresh=False, practice_fresh=False), prepare=False)
    assert r_used["run_id"].nunique() == 2 and not r_used["fresh"].any()
    assert not p_used.empty
    r_new, p_new = M.context_frames(b, replace(base, race_fresh=True, practice_fresh=True), prepare=False)
    assert r_new["run_id"].nunique() == 2 and r_new["fresh"].all()
    assert p_new.empty  # the practice runs here are all used-set runs


def test_run_breakdown_separates_used_sets_from_gap_restarts():
    runs = pd.DataFrame({
        "run_id": list("abcd"), "Compound": "SOFT", "run_len": 10,
        "fresh": [False, True, True, True], "start_age": [6, 7, 5, 2], "stint_start_age": [6, 2, 5, 2],
    })
    out = M._breakdown(runs, 5)
    assert out["n_runs"] == 3
    assert out["genuinely_used_set_FreshTyre_false"] == 1
    assert out["new_set_but_run_starts_after_stint_start_gap_split"] == 1
    assert out["new_set_stint_that_itself_starts_this_late"] == 1


def test_twfe_race_contrasts_need_no_practice_data():
    bases = fake_bases({"HARD": 0.03, "MEDIUM": 0.05, "SOFT": 0.08}, {"HARD": 0.03, "MEDIUM": 0.05, "SOFT": 0.08}, n_events=8)
    spec = replace(M.PRIMARY, age_lo=None, age_hi=None, seasons=(2024,))
    tc = M.twfe_race_contrasts(bases, spec)
    got = tc[tc["pair"] == "SOFT-HARD"]["contrast"].mean()
    assert got == pytest.approx(0.05, abs=0.02)


# ---------------------------------------------------------------------------
# The published results file
# ---------------------------------------------------------------------------


def test_results_file_if_present_is_complete_and_auditable():
    from tyredeg.config import ROOT

    f = ROOT / "paper" / "results" / "management.json"
    if not f.exists():
        pytest.skip("paper/results/management.json not generated yet")
    r = json.loads(f.read_text(encoding="utf-8"))
    for key in ("meta", "primary", "decision", "seasons", "temperature", "fuel_grid_within_run_race",
                "multiverse", "placebo_fp2_vs_fp1fp3", "legacy_ratio", "c_number_softness"):
        assert key in r
    assert r["meta"]["seed"] and r["meta"]["bootstrap_B"] and set(r["meta"]["data_version"]) == {"2022", "2023", "2024", "2025", "2026"}
    p = r["primary"]
    assert p["n_cells"] == sum(p["cells_by_compound"].values())
    lo, hi = p["trend"]["ci95_cluster_bootstrap"]
    assert lo <= p["trend"]["beta_per_softness_step"] <= hi
    # The decision flags must follow from the stored numbers, not be hand-set.
    t = p["trend"]
    expect = hi < 0 and t["p_perm_one_sided_less"] < 0.05 and t["p_perm_two_sided"] < 0.05
    assert r["decision"]["H1_additive_interaction_supported"] == expect
    # The fuel contrast theorem holds in the stored grid.
    betas = [x["beta"] for x in r["fuel_grid_within_run_race"] if x["variant"] == "both contexts"]
    assert max(betas) - min(betas) < 1e-3
    post = r["post_review_additions"]
    assert "AFTER an independent review" in post["provenance"]
    for key in ("level_gap_matched_pooled", "twfe_gradient", "level_placebo_fp2_vs_fp1fp3_matched_pooled",
                "power_additive_trend", "uniform_damping"):
        assert key in post
    assert post["power_additive_trend"]["grid"] and "analytic_mde80_from_primary_boot_sd" in post["power_additive_trend"]
    tg = post["twfe_gradient"]["SOFT-HARD"]
    expect = tg["ci95"][1] < 0 and tg["p_perm_one_sided_less"] < 0.05 and tg["p_perm_two_sided"] < 0.05
    assert r["decision"]["post_review"]["twfe_SOFT_minus_HARD_gradient_supported"] == expect
    ref = r["referee_additions"]
    assert "referee" in ref["provenance"].lower()
    assert ref["set_state_counts"]["race"]["n_runs"] > 0
    cnt = ref["set_state_counts"]["race"]["start_age_ge_5"]
    assert cnt["n_runs"] == (cnt["genuinely_used_set_FreshTyre_false"]
                             + cnt["new_set_but_run_starts_after_stint_start_gap_split"]
                             + cnt["new_set_stint_that_itself_starts_this_late"])
    holm = ref["twfe_primary_holm"]
    for k, raw in holm["raw_p_two_sided"].items():
        assert holm["holm_adjusted_p_two_sided"][k] >= raw - 1e-12
    lead = ref["lead_estimand_matched_start_age"]["MEDIUM+SOFT"]
    assert lead["gap_ci95"][0] <= lead["gap_s_per_lap_race_minus_practice"] <= lead["gap_ci95"][1]
    assert "ratio_ci95" in lead
    used = ref["designs"]["used_sets_only_FreshTyre_false"]
    assert "MEDIUM+SOFT" in used["lead_estimand"] and "MEDIUM+SOFT" in used["placebo_adjusted_gap"]
    assert ref["reconcile_race_compound_trend"]["explanation"]["one_sentence"]

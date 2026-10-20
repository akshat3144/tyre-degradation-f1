"""Tests for the core-model audit: the REML engine, convergence evidence, the
circuit-effect machinery, rate-level calibration, mixed-model power and the
like-for-like ablation.

Synthetic data with a KNOWN answer throughout, small enough to run in seconds.
Several pin the failures the audit found: an optimiser that stopped early and
inflated a variance, a power calculation that ignored the circuit, a coverage
figure that could not fail.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tyredeg.config import ARTIFACTS, PROCESSED
from tyredeg.models import ablation
from tyredeg.models.mixed import LABEL_ORDER, fit_degradation
from tyredeg.models.reml import Comp, RemlModel
from tyredeg.validation import power as P
from tyredeg.validation.calibration import (
    coverage_of_truth,
    predictive_coverage,
    wilson_interval,
)

# ---------------------------------------------------------------------------
# a synthetic mixed-effects race with known truth
# ---------------------------------------------------------------------------


def synth(
    n_events=8, n_drivers=8, runs_per_driver=2, laps=14, rates=(0.03, 0.06),
    circuit_sd=0.03, run_sd=0.3, run_slope_sd=0.02, driver_sd=0.8, sigma=0.3, seed=0,
    labels=("C3", "C4"),
) -> pd.DataFrame:
    """Frames shaped like ``prepare(..., 'race')`` output, truth from the model."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        ev = f"GP{e}"
        tau = rng.normal(0, circuit_sd)
        for d in range(n_drivers):
            u_d = rng.normal(0, driver_sd)
            for r in range(runs_per_driver):
                comp = (d + r) % 2
                u_r = rng.normal(0, run_sd)
                s_r = rng.normal(0, run_slope_sd)
                start = rng.integers(0, 8)
                for lap in range(laps):
                    age = start + lap + 1
                    tl = age - 10.0
                    y = (rates[comp] + tau + s_r) * tl + u_d + u_r + rng.normal(0, sigma)
                    rows.append({
                        "event": ev, "Driver": f"D{d}", "run_id": f"{ev}|D{d}|{r}", "C": labels[comp],
                        "Compound": ("HARD", "MEDIUM", "SOFT")[comp], "tl": tl, "tl2": tl**2,
                        "y": y, "tr": 0.0, "LapNumber": float(lap + 1 + 10 * r),
                    })
    df = pd.DataFrame(rows)
    df["tr"] = np.random.default_rng(seed + 1).normal(0, 0.3, len(df))
    df["y"] += 0.0 * df["tr"]
    return df


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------


def _model(df, run_slope=True, eliminate=None):
    s1 = df["tl"].std()
    tl = df["tl"].to_numpy() / s1
    C = df["C"].to_numpy()
    X = np.column_stack([C == "C3", C == "C4", (C == "C3") * tl, (C == "C4") * tl]).astype(float)
    ev = df["event"].to_numpy()
    comps = [Comp("circuit_slope", ev, tl), Comp("driver_event", ev + "|" + df["Driver"].to_numpy()),
             Comp("run", df["run_id"].to_numpy())]
    if run_slope:
        comps.append(Comp("run_slope", df["run_id"].to_numpy(), tl))
    kw = {} if eliminate is None else {"eliminate": eliminate}
    return RemlModel(df["y"].to_numpy(), X, ev, comps, **kw), s1


def test_schur_elimination_equals_the_dense_computation():
    df = synth(seed=1)
    fast, _ = _model(df)
    dense, _ = _model(df, eliminate=())
    for th in ([0.5, 2.0, 0.6, 0.2], [0.0, 1.0, 0.1, 0.0], [1.5, 3.0, 1.0, 0.4]):
        assert fast.deviance(np.array(th)) == pytest.approx(dense.deviance(np.array(th)), abs=1e-8)


def test_engine_recovers_known_variance_components_and_slopes():
    df = synth(n_events=12, n_drivers=12, laps=16, circuit_sd=0.04, run_sd=0.4, seed=3,
               rates=(0.03, 0.06))
    m, s1 = _model(df)
    f = m.fit(starts=[np.full(4, 0.5), np.full(4, 0.1)])
    assert f.converged
    assert f.sd["driver_event"] == pytest.approx(0.8, rel=0.35)
    assert f.sd["run"] == pytest.approx(0.4, rel=0.25)
    assert f.sd["circuit_slope"] / s1 == pytest.approx(0.04, abs=0.025)
    slopes = f.beta[2:4] / s1
    assert slopes[0] == pytest.approx(0.03, abs=0.02)
    assert slopes[1] - slopes[0] == pytest.approx(0.03, abs=0.02)


def test_engine_matches_statsmodels_on_the_shared_specification():
    """The reason the engine can stand in for MixedLM: same model, same answer."""
    df = synth(n_events=6, n_drivers=6, laps=12, seed=5)
    a = fit_degradation(df, quadratic=False, circuit_effects=True, run_slopes=False)
    b = fit_degradation(df, quadratic=False, circuit_effects=True, run_slopes=False,
                        engine="statsmodels")
    assert b.converged
    for c in a.rates:
        assert a.rates[c].mean == pytest.approx(b.rates[c].mean, abs=2e-3)
        assert (a.rates[c].hi - a.rates[c].lo) == pytest.approx(b.rates[c].hi - b.rates[c].lo, rel=0.05)
    assert a.variance["run"] == pytest.approx(b.variance["run"], abs=0.03)


def test_a_null_circuit_effect_lands_on_the_boundary_and_is_reported_there():
    """The null model of a variance test sits ON the boundary. The engine must go
    there, flag it, and the gradient check must accept it as a constrained optimum."""
    df = synth(circuit_sd=0.0, seed=7, run_slope_sd=0.0)
    m, s1 = _model(df, run_slope=False)
    f = m.fit(starts=[np.full(3, 0.5), np.full(3, 0.1)])
    d = m.diagnostics(f)
    assert "circuit_slope" in f.on_boundary or f.sd["circuit_slope"] / s1 < 0.01
    assert d["boundary_slopes_nonnegative"]


def test_every_start_reaches_the_same_optimum():
    """The shipped statsmodels L-BFGS fit stopped early on a flat ridge. Several
    starts must agree, and the Hessian there must be positive definite."""
    df = synth(n_events=10, seed=11)
    fit = fit_degradation(df, quadratic=False, circuit_effects=True)
    conv = fit.convergence
    assert conv["converged"] and conv["hessian_pd"]
    assert conv["n_starts_at_optimum"] >= 2
    assert conv["all_ok"]


def test_rescaling_tyre_age_does_not_change_the_answer_in_laps():
    df = synth(seed=13)
    a = fit_degradation(df, quadratic=False, circuit_effects=True)
    df2 = df.copy()
    df2["tl"] = df2["tl"] * 5.0  # tyre age in different units
    b = fit_degradation(df2, quadratic=False, circuit_effects=True)
    for c in a.rates:
        assert a.rates[c].mean == pytest.approx(5.0 * b.rates[c].mean, rel=1e-3)


# ---------------------------------------------------------------------------
# the circuit effect and the boundary test
# ---------------------------------------------------------------------------


def _lr(df, seed):
    m, _ = _model(df, run_slope=False)
    full = m.fit(starts=[np.full(3, 0.5), np.full(3, 0.1)])
    null = m.fit(free=["driver_event", "run"], starts=[np.full(2, 0.5)])
    return max(0.0, null.deviance - full.deviance), m, full, null


def test_the_circuit_lr_is_large_when_circuits_differ_and_small_when_they_do_not():
    strong, *_ = _lr(synth(n_events=12, circuit_sd=0.06, run_slope_sd=0.0, seed=21), 0)
    none, *_ = _lr(synth(n_events=12, circuit_sd=0.0, run_slope_sd=0.0, seed=22), 0)
    assert strong > 10 > none


def test_parametric_bootstrap_null_is_not_chi_squared_one():
    """Half the null LRs are exactly zero (variance estimate on the boundary): the
    chi-squared(1) reference would be wrong, which is why a bootstrap is used."""
    df = synth(n_events=10, circuit_sd=0.0, run_slope_sd=0.0, seed=31)
    _, m, full, null = _lr(df, 0)
    rng = np.random.default_rng(0)
    sd0 = dict(null.sd)
    sd0["circuit_slope"] = 0.0
    lrs = []
    for _ in range(24):
        m.set_y(m.simulate(null.beta, null.sigma2, sd0, rng))
        f1 = m.fit(starts=[full.theta, np.full(3, 0.2)])
        f0 = m.fit(free=["driver_event", "run"], starts=[null.theta[1:]])
        lrs.append(max(0.0, f0.deviance - f1.deviance))
    lrs = np.array(lrs)
    assert (lrs < 1e-6).mean() > 0.25  # a point mass at zero
    assert lrs.mean() < stats.chi2.mean(1)  # chi-bar-squared mean is 0.5, not 1


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------

PARAMS = P.MixedParams(circuit_sd=0.04, run_sd=0.28, run_slope_sd=0.03, driver_sd=1.1, sigma=0.55)


def test_analytic_power_grows_with_stints_and_the_split_design_plateaus():
    bal = [P.analytic_power(n, 0.02, "balanced", PARAMS)["power"] for n in (52, 208, 832)]
    spl = [P.analytic_power(n, 0.02, "split", PARAMS)["power"] for n in (52, 208, 832)]
    assert bal[0] < bal[1] < bal[2]
    assert max(spl) < bal[2]
    # more stints cannot beat the circuit ceiling in the split design
    assert max(spl) <= P.power_ceiling_split(0.02, PARAMS) + 0.01


def test_split_design_power_is_capped_by_the_number_of_circuits_not_stints():
    tiny = P.power_ceiling_split(0.006, PARAMS)
    assert tiny < 0.15  # an effect of 0.006 s/lap is invisible however many laps are run
    assert P.power_ceiling_split(0.006, PARAMS, n_events=1300) > tiny


def test_simulated_power_is_high_for_a_big_effect_and_holds_its_size():
    seeds = np.random.SeedSequence(5).spawn(10)
    big = P.simulate_mixed_power(104, 0.10, "balanced", PARAMS, seeds)
    assert np.mean(big["reject_z"]) >= 0.9 and big["n_failed_fits"] == 0
    null = P.simulate_mixed_power(104, 0.0, "balanced", PARAMS, np.random.SeedSequence(6).spawn(40))
    assert np.mean(null["reject_z"]) <= 0.25  # nominal 5%; 40 sims cannot resolve more


def test_crossing_is_interpolated_not_a_grid_point_and_none_when_never_reached():
    grid = [64, 128, 256, 512]
    c = P.crossing_linear(grid, [0.2, 0.5, 0.7, 0.9])
    assert 256 < c < 512  # between grid points, not 512
    assert P.crossing_linear(grid, [0.05, 0.06, 0.07, 0.08]) is None
    r = P.crossing_with_mc_error(grid, [np.r_[np.ones(18), np.zeros(2)]] * 4, n_boot=50)
    assert r["crossing"] == 64.0  # already above 80% at the first grid point


def test_mc_error_on_the_crossing_shrinks_with_more_simulations():
    rng = np.random.default_rng(0)
    grid = [100, 200, 400, 800]
    truth = np.array([0.2, 0.5, 0.8, 0.95])

    def se(n):
        sims = [(rng.random(n) < p).astype(float) for p in truth]
        return P.crossing_with_mc_error(grid, sims, n_boot=200)["mc_se"]

    assert se(400) < se(40)


# ---------------------------------------------------------------------------
# rate-level calibration
# ---------------------------------------------------------------------------


def test_wilson_interval_brackets_the_proportion_and_is_wide_for_small_n():
    lo, hi = wilson_interval(30, 38)
    assert lo < 30 / 38 < hi and hi - lo > 0.15


def test_predictive_coverage_is_nominal_when_the_circuit_variance_is_included():
    """Cells drawn with a real circuit effect: leaving tau out must under-cover."""
    rng = np.random.default_rng(0)
    cells = []
    for _ in range(4000):
        tau, se_o, se_p = 0.04, 0.02, 0.01
        pooled = 0.05
        own = pooled + rng.normal(0, np.sqrt(tau**2 + se_o**2 + se_p**2))
        cells.append({"own": own, "own_se": se_o, "pooled": pooled, "pooled_se": se_p, "tau": tau})
    full = predictive_coverage(cells)["coverage"]["0.80"]["coverage"]
    no_tau = predictive_coverage(cells, include_circuit=False)["coverage"]["0.80"]["coverage"]
    assert full == pytest.approx(0.80, abs=0.03)
    assert no_tau < 0.55


def test_coverage_of_truth_is_nominal_for_a_correct_estimator():
    rng = np.random.default_rng(1)
    est = rng.normal(0.05, 0.01, size=(3000, 2))
    cov = coverage_of_truth([0.05, 0.05], est, np.full_like(est, 0.01))
    assert cov["0.95"]["pooled"] == pytest.approx(0.95, abs=0.02)
    # and detectably wrong when the stated SE is half the real one
    bad = coverage_of_truth([0.05, 0.05], est, np.full_like(est, 0.005))
    assert bad["0.95"]["pooled"] < 0.8


# ---------------------------------------------------------------------------
# ablation: like for like
# ---------------------------------------------------------------------------


def _stint_frame(true_rate=0.05, fuel_kg_per_lap=1.2, s_per_kg=0.033, seed=0):
    """A race with fuel burn, no track evolution, stints of different ages."""
    rng = np.random.default_rng(seed)
    rows, laps = [], []
    n_laps = 57
    for d in range(14):
        pit = 14 + (d * 2) % 22
        base = rng.normal(0, 0.4)
        for lap in range(1, n_laps + 1):
            stint = 1 if lap <= pit else 2
            life = lap if lap <= pit else lap - pit
            t = 90 + base + true_rate * life - s_per_kg * fuel_kg_per_lap * lap + rng.normal(0, 0.15)
            rows.append({
                "event": "GP", "session": "R", "Driver": f"D{d}", "Stint": float(stint),
                "LapNumber": float(lap), "TyreLife": float(life), "LapTimeSeconds": t,
                "C": "C3", "run_id": f"GP|R|D{d}|{stint}",
            })
    df = pd.DataFrame(rows)
    return df, df[["event", "session", "LapNumber"]].copy()


def test_naive_slope_is_biased_low_by_fuel_burn_and_the_practitioner_recovers_the_rate():
    design, laps = _stint_frame()
    naive = ablation.naive_by_compound(design)["C3"]["cluster"]
    # 70 kg over 57 laps at 0.033 s/kg is ~0.04 s/lap of fuel gain: the naive slope falls far short
    assert naive.mean < 0.03
    prac = ablation.practitioner_by_compound(design, laps)["C3"]
    assert prac.mean == pytest.approx(0.05, abs=0.012)


def test_cluster_robust_error_exceeds_the_iid_error_when_laps_are_correlated():
    rng = np.random.default_rng(2)
    rows = []
    for r in range(60):
        shift = rng.normal(0, 1.0)  # a run-level offset that persists across its laps
        for lap in range(10):
            rows.append({"x": float(r % 7 + lap), "y": shift + rng.normal(0, 0.2), "g": r})
    d = pd.DataFrame(rows)
    X = np.column_stack([np.ones(len(d)), d["x"]])
    _, cc, ci = ablation.cluster_ols(X, d["y"].to_numpy(), d["g"])
    assert np.sqrt(cc[1, 1]) > 1.5 * np.sqrt(ci[1, 1])


# ---------------------------------------------------------------------------
# labelled compounds (seasons with no allocation table)
# ---------------------------------------------------------------------------


def test_labelled_compounds_fit_in_their_own_order():
    df = synth(labels=("HARD", "SOFT"), seed=17)
    fit = fit_degradation(df, quadratic=False, circuit_effects=True, order=LABEL_ORDER)
    assert fit.ordered == ["HARD", "SOFT"]
    assert fit.rates["SOFT"].mean > fit.rates["HARD"].mean
    assert fit.contrast("HARD", "SOFT").mean == pytest.approx(
        fit.rates["SOFT"].mean - fit.rates["HARD"].mean)


# ---------------------------------------------------------------------------
# the stored artifacts come from the current code and the dataset on disk
# ---------------------------------------------------------------------------

LAPS = PROCESSED / "laps.parquet"


@pytest.mark.skipif(not LAPS.exists(), reason="needs the cached 2026 laps")
def test_meta_json_describes_the_dataset_it_was_built_from():
    laps = pd.read_parquet(LAPS)
    meta = json.loads((ARTIFACTS / "meta.json").read_text(encoding="utf-8"))
    assert meta["n_laps_clean"] == len(laps)
    assert sorted(meta["events_in_dataset"]) == sorted(laps["event"].unique())
    race_events = sorted(laps.loc[laps["session"] == "R", "event"].unique())
    assert sorted(meta["events"]) == race_events
    assert sorted(meta["practice_only_events"]) == sorted(set(laps["event"]) - set(race_events))
    assert meta["fit_scope"]["race_degradation"]["sessions"] == ["R"]
    assert sum(sum(v.values()) for v in meta["laps_by_event_session"].values()) == len(laps)


@pytest.mark.skipif(not LAPS.exists(), reason="needs the cached 2026 laps")
def test_degradation_json_is_what_the_current_code_fits():
    from tyredeg.models.design import prepare

    laps = pd.read_parquet(LAPS)
    fit = fit_degradation(prepare(laps, "race"), quadratic=False, circuit_effects=True)
    art = json.loads((ARTIFACTS / "degradation.json").read_text(encoding="utf-8"))
    assert art["diagnostics"]["convergence"]["all_ok"] is True
    for curve in art["curves"]:
        assert curve["rate"]["mean"] == pytest.approx(fit.rates[curve["compound"]].mean, abs=2e-5)


# ---------------------------------------------------------------------------
# circuit versus event: the nested model (referee round)
# ---------------------------------------------------------------------------


def synth_nested(n_circuits=8, visits=4, circuit_sd=0.05, event_sd=0.02, seed=0,
                 rates=(0.03, 0.06), laps=12, drivers=6) -> pd.DataFrame:
    """Races at repeat-visit circuits: a persistent circuit slope and a per-race one."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_circuits):
        tau_c = rng.normal(0, circuit_sd)
        for v in range(visits):
            ev = f"{2020 + v}|GP{c}"
            tau_e = rng.normal(0, event_sd)
            for d in range(drivers):
                u_d = rng.normal(0, 0.5)
                for r in range(2):
                    comp = (d + r) % 2
                    u_r = rng.normal(0, 0.25)
                    start = rng.integers(0, 6)
                    for lap in range(laps):
                        tl = start + lap + 1 - 9.0
                        rows.append({
                            "circuit": f"GP{c}", "event_key": ev, "Driver": f"D{d}",
                            "run_id": f"{ev}|D{d}|{r}", "C": ("C3", "C4")[comp], "tl": tl,
                            "y": (rates[comp] + tau_c + tau_e) * tl + u_d + u_r + rng.normal(0, 0.3),
                            "tr": 0.0,
                        })
    return pd.DataFrame(rows)


def test_circuit_key_follows_the_track_not_the_race_name():
    from tyredeg.paper.model_checks import circuit_key

    assert circuit_key(2026, "Barcelona Grand Prix") == circuit_key(2024, "Spanish Grand Prix")
    assert circuit_key(2026, "Monaco Grand Prix") == "Monaco Grand Prix"


def test_nested_model_separates_a_persistent_circuit_slope_from_a_per_race_one():
    from tyredeg.paper.model_checks import build_nested_model

    d = synth_nested(circuit_sd=0.06, event_sd=0.015, seed=3, n_circuits=10)
    m, meta = build_nested_model(d, run_slope=False)
    f = m.fit(starts=[np.full(len(m.names), 0.5), np.full(len(m.names), 0.1)])
    c, e = f.sd["circuit_slope"] / meta["s1"], f.sd["event_slope"] / meta["s1"]
    assert f.converged
    assert c > 2 * e
    assert c == pytest.approx(0.06, abs=0.03)


def test_nested_model_gives_the_variance_to_the_race_when_circuits_do_not_persist():
    from tyredeg.paper.model_checks import build_nested_model

    d = synth_nested(circuit_sd=0.0, event_sd=0.05, seed=4, n_circuits=10)
    m, meta = build_nested_model(d, run_slope=False)
    f = m.fit(starts=[np.full(len(m.names), 0.5), np.full(len(m.names), 0.1)])
    c, e = f.sd["circuit_slope"] / meta["s1"], f.sd["event_slope"] / meta["s1"]
    assert e > 2 * c
    assert c**2 / (c**2 + e**2) < 0.25


def test_compound_spread_is_noise_corrected_and_circuit_comparison_uses_paired_replicates():
    from tyredeg.paper.model_checks import _compound_vs_components

    rng = np.random.default_rng(0)
    ridx = [0, 1, 2]
    true = np.array([0.02, 0.04, 0.06])
    alts = []
    for _ in range(300):
        se = np.full(3, 0.01)
        alts.append({"beta": true + rng.normal(0, 0.01, 3), "se": se, "sd": {"circuit_slope": 0.05}})
    out = _compound_vs_components(alts, ridx, 1.0, 0.05, float("nan"), None)
    raw, corr = out["compound_sd_raw"]["median"], out["compound_sd_noise_corrected"]["median"]
    assert corr < raw  # removing the estimation noise lowers it
    assert corr == pytest.approx(np.std(true, ddof=1), abs=0.004)
    assert out["circuit_minus_compound_corrected"]["p_circuit_greater"] > 0.9


# ---------------------------------------------------------------------------
# power: realistic nominations
# ---------------------------------------------------------------------------

NOMS = [("C3", "C4", "C5"), ("C2", "C3", "C4"), ("C3", "C4", "C5"), ("C1", "C2", "C3"),
        ("C2", "C3", "C4"), ("C3", "C4", "C5"), ("C1", "C2", "C3"), ("C3", "C4", "C5")]


def test_realistic_design_runs_stints_only_on_nominated_compounds():
    rows, n_comp, (ia, ib) = P._plan("realistic", 160, 8, NOMS, ("C1", "C2"), 3)
    assert n_comp == 4 and (ia, ib) == (0, 1)  # C1-C4 appear in the events that carry C1 or C2
    events = {r[0] for r in rows}
    assert events == {3, 6, 1, 4}  # the events carrying C1 or C2


def test_realistic_design_sits_between_balanced_and_split_and_close_to_balanced():
    se = {
        "balanced": P.analytic_power(416, 0.02, "balanced", PARAMS, n_events=8)["se"],
        "split": P.analytic_power(416, 0.02, "split", PARAMS, n_events=8)["se"],
        "real": P.analytic_power(416, 0.02, "realistic", PARAMS, nominations=NOMS, pair=("C1", "C2"))["se"],
    }
    assert se["balanced"] <= se["real"] < se["split"]
    # a compound present at every event (C3) links the pair through within-event contrasts
    assert se["real"] < 3 * se["balanced"]


def test_variance_uncertainty_matters_most_for_the_split_design():
    from dataclasses import replace

    hi = replace(PARAMS, circuit_sd=0.08)
    bal = [P.analytic_power(416, 0.02, "balanced", p)["se"] for p in (PARAMS, hi)]
    spl = [P.analytic_power(416, 0.02, "split", p)["se"] for p in (PARAMS, hi)]
    assert spl[1] / spl[0] > bal[1] / bal[0]


def test_realistic_simulation_returns_a_rejection_per_replicate():
    seeds = np.random.SeedSequence(2).spawn(4)
    r = P.simulate_mixed_power(104, 0.10, "realistic", PARAMS, seeds, nominations=NOMS, pair=("C2", "C3"))
    assert len(r["reject_z"]) == 4 and r["n_failed_fits"] == 0


def test_the_wilson_interval_is_in_the_rate_calibration_output():
    cells = [{"own": 0.05, "own_se": 0.02, "pooled": 0.05, "pooled_se": 0.01, "tau": 0.03}] * 6
    cov = predictive_coverage(cells)["coverage"]["0.80"]
    assert len(cov["wilson_95"]) == 2 and cov["wilson_95"][0] <= cov["coverage"] <= cov["wilson_95"][1]


# ---------------------------------------------------------------------------
# second referee round: event bootstrap, minimum detectable effect, pooled design
# ---------------------------------------------------------------------------


def test_the_minimum_detectable_effect_is_the_effect_with_80_percent_power():
    from tyredeg.paper.model_checks import _mde_from_se

    pw = P.analytic_power(416, 0.02, "balanced", PARAMS, n_events=8)
    mde = _mde_from_se(pw["se"])
    back = P.analytic_power(416, mde, "balanced", PARAMS, n_events=8)["power"]
    assert back == pytest.approx(0.80, abs=1e-4)  # analytic_power rounds z to 1.96
    assert 2.8 * pw["se"] < mde < 2.81 * pw["se"] + 1e-9  # (1.96 + 0.84) * SE


def test_the_event_bootstrap_redraws_whole_races_and_returns_every_compound_rate():
    from tyredeg.models.reml import single_threaded_blas
    from tyredeg.paper import model_checks as mc

    df = synth(n_events=6, n_drivers=4, laps=10, seed=5)
    spec = dict(circuit="random", slopes="compound")
    m, meta = mc.build_model(df, **spec)
    with single_threaded_blas():
        f = m.fit(starts=[np.full(len(m.names), 0.5), np.full(len(m.names), 0.1)])
    mc._init_worker_spec(df, spec, mc._fit_payload(f))
    out = mc._task_event_boot(np.random.SeedSequence(3))
    assert out["n_compounds"] == 2 and out["rate"].shape == (2,) and out["se"].shape == (2,)
    assert np.all(out["se"] > 0) and out["circuit_sd"] >= 0
    # same seed, same draw: the bootstrap is deterministic
    again = mc._task_event_boot(np.random.SeedSequence(3))
    assert np.allclose(out["rate"], again["rate"])


def test_the_pooled_physical_design_keeps_seasons_apart_and_follows_the_track():
    if not all((PROCESSED / f"laps_{s}.parquet").exists() for s in (2022, 2023, 2024, 2025)):
        pytest.skip("cached seasons not present")
    from tyredeg.paper.model_checks import C_ORDER, pooled_physical_design

    data = {s: pd.read_parquet(PROCESSED / f"laps_{s}.parquet") for s in (2022, 2023, 2024, 2025)}
    data[2026] = pd.read_parquet(PROCESSED / "laps.parquet")
    p = pooled_physical_design(data)
    assert sorted(p["season"].unique()) == [2022, 2023, 2024, 2025, 2026]
    assert set(p["C"]) <= set(C_ORDER)
    # a run belongs to exactly one race, and a race to exactly one season
    assert p.groupby("run_id")["event_key"].nunique().max() == 1
    assert p.groupby("event_key")["season"].nunique().max() == 1
    # the same track under its two names is one circuit
    assert not any(c == "Barcelona Grand Prix" for c in p["circuit"])
    assert (p["circuit"] == "Spanish Grand Prix").groupby(p["season"]).any().all()


def test_the_t_reference_counts_only_the_races_that_carry_the_compared_compounds():
    seeds = np.random.SeedSequence(7).spawn(2)
    r = P.simulate_mixed_power(104, 0.10, "realistic", PARAMS, seeds, nominations=NOMS, pair=("C1", "C2"))
    assert r["n_events_in_design"] == 4  # of the eight races in NOMS, four carry C1 or C2
    b = P.simulate_mixed_power(52, 0.10, "balanced", PARAMS, seeds, n_events=8)
    assert b["n_events_in_design"] == 8

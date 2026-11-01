"""Tests for the simulation recovery study (``tyredeg.paper.recovery``).

Small synthetic seasons with a KNOWN answer, a few seconds each. They pin the
properties the study's conclusions rest on: the simulator is deterministic and
shares its draws across scenarios, each pit-timing mechanism does what its name
says to the design, the estimators are exact where algebra says they must be,
Eq. 1 recovers a planted slope when pitting is exogenous, and the metrics and
the scored expectations compute what they claim.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tyredeg.config import ROOT
from tyredeg.paper import model_checks as mc
from tyredeg.paper import recovery as rec

SEED = 20261101


def tiny(**kw) -> rec.DGP:
    """Five C3/C4/C5 events, 14 cars, 45 laps: about 2 500 clean laps."""
    base = dict(
        nominations=(("C3", "C4", "C5"),) * 5, n_cars=14, n_laps=45,
        slope=(("C1", 0.08), ("C2", 0.06), ("C3", 0.07), ("C4", 0.04), ("C5", 0.01)),
    )
    base.update(kw)
    return rec.DGP(**base)


def medium(**kw) -> rec.DGP:
    """Enough cars and events for design correlations to be stable."""
    return tiny(nominations=rec.DEFAULT_NOMINATIONS[:8], n_cars=20, n_laps=55, **kw)


def seeds(n: int):
    """Fresh each call: ``SeedSequence.spawn`` is stateful, a shared one would hand out new children."""
    return np.random.SeedSequence(SEED).spawn(n)


# ---------------------------------------------------------------------------
# the simulator
# ---------------------------------------------------------------------------


def test_simulation_is_deterministic_and_seed_dependent():
    a = rec.simulate(tiny(), "pace", seeds(2)[0])
    b = rec.simulate(tiny(), "pace", seeds(2)[0])
    c = rec.simulate(tiny(), "pace", seeds(2)[1])
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["LapTimeSeconds"].to_numpy()[:50], c["LapTimeSeconds"].to_numpy()[:50])


def test_unknown_scenario_is_refused():
    with pytest.raises(ValueError, match="scenario"):
        rec.simulate(tiny(), "strategic", 0)


@pytest.mark.parametrize("scenario", rec.SCENARIOS)
def test_race_structure_is_valid(scenario):
    cfg = tiny()
    f = rec.simulate(cfg, scenario, seeds(1)[0])
    noms = {f"E{e:02d}": set(n) for e, n in enumerate(cfg.nominations)}
    assert f["LapNumber"].between(1, cfg.n_laps).all()
    assert (f["TyreLife"] >= 2).all()
    for run, g in f.groupby("run_id"):
        age = np.sort(g["TyreLife"].to_numpy())
        assert (np.diff(age) == 1).all(), "a run's clean laps are consecutive"
        assert len(g) >= cfg.min_stint - 2, "the minimum stint survives the out-lap/in-lap trim"
        assert g["C"].nunique() == 1
        assert g["C"].iloc[0] in noms[g["event"].iloc[0]]
    # every car: 2 or 3 runs, at least two different compounds, no lap driven twice
    for _, g in f.groupby(["event", "Driver"]):
        assert g["run_id"].nunique() in (2, 3)
        assert g["C"].nunique() >= 2
        assert g["LapNumber"].is_unique
    # the out-lap is gone, so no run starts at age 1 and 'lap - age' is constant per run
    off = (f["LapNumber"] - f["TyreLife"]).groupby(f["run_id"]).nunique()
    assert (off == 1).all()


def test_scenarios_share_every_random_draw():
    """Same seed => same cars, same pace, same event effects: only stop laps differ."""
    cfg = tiny()
    fs = {s: rec.simulate(cfg, s, seeds(1)[0]) for s in rec.SCENARIOS}
    pace = {s: f.drop_duplicates(["event", "Driver"]).set_index(["event", "Driver"])["pace"]
            for s, f in fs.items()}
    pd.testing.assert_series_equal(pace["exogenous"], pace["wear"])
    pd.testing.assert_series_equal(pace["exogenous"], pace["pace"])
    # exogenous and wear plan the same number of stints per car
    n = {s: fs[s].groupby(["event", "Driver"])["run_id"].nunique() for s in ("exogenous", "wear")}
    pd.testing.assert_series_equal(n["exogenous"], n["wear"])
    # ... but they do not stop on the same laps
    assert fs["exogenous"]["LapNumber"].sum() != fs["wear"]["LapNumber"].sum() or \
        len(fs["exogenous"]) != len(fs["wear"])


def test_pace_scenario_correlates_tyre_age_with_pace_and_exogenous_does_not():
    cfg = medium()
    corr = {}
    for s in ("exogenous", "pace"):
        vals = []
        for sd in seeds(3):
            f = rec.simulate(cfg, s, sd)
            vals.append(rec.frame_diagnostics(f)["corr_car_mean_age_pace"])
        corr[s] = float(np.mean(vals))
    assert corr["pace"] < -0.2, "faster cars end up on older tyres"
    assert abs(corr["exogenous"]) < 0.1


def test_wear_scenario_shortens_steep_runs_and_exogenous_does_not():
    cfg = medium()
    out = {}
    for s in ("exogenous", "wear"):
        vals = []
        for sd in seeds(3):
            f = rec.simulate(cfg, s, sd)
            first = f[f["run_id"].str.endswith("|0")]
            g = first.groupby("run_id").agg(n=("TyreLife", "size"), s=("true_slope", "first"))
            vals.append(np.corrcoef(g["n"], g["s"])[0, 1])
        out[s] = float(np.mean(vals))
    assert out["wear"] < -0.5, "a steeper realised slope reaches the wear threshold sooner"
    assert abs(out["exogenous"]) < 0.15


def test_dgp_defaults_are_the_committed_fit():
    model_p, alloc_p = ROOT / "paper/results/model.json", ROOT / "data/allocation.json"
    if not (model_p.exists() and alloc_p.exists()):
        pytest.skip("committed fit or allocation table not present")
    cfg = rec.dgp_from_fit(json.loads(model_p.read_text(encoding="utf-8")),
                           json.loads(alloc_p.read_text(encoding="utf-8")))
    assert cfg == rec.DGP()


def test_the_planted_truth_stays_the_earlier_fit_once_model_json_holds_the_corrected_one():
    model_p, alloc_p = ROOT / "paper/results/model.json", ROOT / "data/allocation.json"
    if not (model_p.exists() and alloc_p.exists()):
        pytest.skip("committed fit or allocation table not present")
    model = json.loads(model_p.read_text(encoding="utf-8"))
    alloc = json.loads(alloc_p.read_text(encoding="utf-8"))
    approx = {"rates_s_per_lap": {c: {"mean": v} for c, v in rec.DGP().slope},
              "variance_sd": {"circuit_slope": 0.04367, "driver_event": 1.16643, "run": 0.28421,
                              "run_slope": 0.0315, "resid": 0.55553}}
    corrected = {**model, "shipped_fit": {**model["shipped_fit"], "rates_s_per_lap": {
        c: {"mean": 0.01} for c in rec.C_ORDER}}, "exact_vs_approximate": {"shipped_2026": {"approximate": approx}}}
    assert rec.dgp_from_fit(corrected, alloc) == rec.DGP()


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

OLS_ONLY = ("naive_pooled_ols", "fuel_within_run", "fuel_within_run_minus20", "fuel_within_run_plus20",
            "cross_section_cell", "field_demeaned_ols")


def test_ols_estimators_are_exact_where_algebra_says_so():
    """No noise, no random effects, no track evolution or weather: the within-run fuel-corrected
    slope and the same-cell cross-section recover the planted slopes exactly, a wrong fuel
    coefficient shifts the slope by exactly -/+ 20% x 0.033 x 1.7, and the naive slope is far off."""
    cfg = tiny(sd_circuit_slope=0.0, sd_driver=0.0, sd_run=0.0, sd_run_slope=0.0, sd_noise=1e-4,
               evo_amp=(0.0, 0.0), weather_sd=0.0)
    out = rec.estimate_all(rec.simulate(cfg, "exogenous", seeds(1)[0]), cfg, OLS_ONLY)
    assert not out["errors"]
    truth = cfg.truth
    shift = 0.2 * cfg.fuel_s_per_kg * cfg.fuel_kg_per_lap
    for c in ("C3", "C4", "C5"):
        assert out["est"]["fuel_within_run"][c][0] == pytest.approx(truth[c], abs=1e-4)
        assert out["est"]["cross_section_cell"][c][0] == pytest.approx(truth[c], abs=1e-3)
        assert out["est"]["fuel_within_run_minus20"][c][0] - truth[c] == pytest.approx(-shift, abs=1e-4)
        assert out["est"]["fuel_within_run_plus20"][c][0] - truth[c] == pytest.approx(shift, abs=1e-4)
        assert out["est"]["naive_pooled_ols"][c][0] < truth[c] - 0.04, "fuel burn swamps the naive slope"


def test_eq1_shipped_is_exactly_the_shipped_builder_and_recovers_an_exogenous_slope():
    cfg = tiny()
    frame = rec.simulate(cfg, "exogenous", seeds(1)[0])
    out = rec.estimate_all(frame, cfg, ("eq1_shipped",))
    assert not out["errors"]
    # the same model fitted by hand, with the shipped starts
    d = rec.design_frame(frame)
    model, meta = mc.build_model(d, exact_within=False)  # eq1_shipped is the earlier construction
    k = len(model.comps)
    fit = model.fit(starts=[np.full(k, 0.5), np.full(k, 0.1), np.linspace(2.0, 0.2, k)])
    truth = cfg.truth
    for c in meta["cats"]:
        j = meta["names"].index(("tl", c))
        est, se = out["est"]["eq1_shipped"][c][:2]
        assert est == pytest.approx(fit.beta[j] / meta["s1"], rel=1e-9)
        assert se == pytest.approx(np.sqrt(fit.cov_beta[j, j]) / meta["s1"], rel=1e-9)
        assert abs(est - truth[c]) < 4 * se, f"{c}: {est:.4f} vs planted {truth[c]} (se {se:.4f})"
    assert out["flags"]["eq1_shipped"]["converged"]
    # the planted ordering is visible: C3 (0.07) degrades faster than C5 (0.01)
    assert out["est"]["eq1_shipped"]["C3"][0] > out["est"]["eq1_shipped"]["C5"][0]


def test_exact_within_and_mundlak_variants_run_and_estimate_every_compound():
    cfg = tiny()
    out = rec.estimate_all(rec.simulate(cfg, "pace", seeds(1)[0]), cfg,
                           ("eq1_mundlak_run_mean", "eq1_mundlak_run_mean_slope", "eq1_exact_within",
                            "eq1_exact_within_mundlak"))
    assert not out["errors"]
    truth = cfg.truth
    for key in ("eq1_mundlak_run_mean", "eq1_mundlak_run_mean_slope", "eq1_exact_within",
                "eq1_exact_within_mundlak"):
        assert set(out["est"][key]) == {"C3", "C4", "C5"}
        for c, (est, se, _, _) in out["est"][key].items():
            assert np.isfinite(est) and se > 0
            assert abs(est - truth[c]) < 5 * se
        assert out["flags"][key]["circuit_sd"] >= 0


def test_a_failing_estimator_is_recorded_not_swallowed(monkeypatch):
    cfg = tiny()

    def boom(*a, **k):
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(rec, "est_naive", boom)
    out = rec.estimate_all(rec.simulate(cfg, "exogenous", seeds(1)[0]), cfg,
                           ("naive_pooled_ols", "fuel_within_run"))
    assert "LinAlgError" in out["errors"]["naive_pooled_ols"]
    assert "naive_pooled_ols" not in out["est"] and "fuel_within_run" in out["est"]
    res = [{"scenario": "exogenous", "rep": 0, **out, "diag": {}}]
    assert rec.error_counts(res, ("exogenous",))["exogenous"]["naive_pooled_ols"]["count"] == 1
    s = rec.summarise(res, cfg.truth, ("exogenous",), ("naive_pooled_ols",))
    assert s["exogenous"]["naive_pooled_ols"]["failures"] == 1


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


def _strip(rs):
    return [{k: v for k, v in r.items() if k != "seconds"} for r in rs]


def test_sharded_run_equals_serial_run(tmp_path):
    cfg = tiny(nominations=(("C3", "C4", "C5"),) * 3, n_cars=10, n_laps=40)
    which = ("fuel_within_run", "cross_section_cell")
    serial = rec.run_replicates(cfg, ("exogenous", "wear"), 3, 5, workers=1, which=which)
    sharded = rec.run_replicates(cfg, ("exogenous", "wear"), 3, 5, workers=2, which=which,
                                 cache_root=tmp_path)
    assert _strip(serial) == _strip(sharded)
    assert [r["rep"] for r in serial] == [0, 0, 1, 1, 2, 2]
    assert not list(tmp_path.glob("recovery_*")), "a finished run removes its cache"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _fake(errs, ses, scenario="exogenous", est="eq1_shipped", comp="C3", truth=0.05):
    return [{"scenario": scenario, "rep": i,
             "est": {est: {comp: [truth + e, s, float("nan"), float("nan")]}},
             "flags": {est: {"circuit_sd": 0.03 + 0.01 * i, "converged": True, "circuit_on_boundary": False}},
             "errors": {}, "diag": {}} for i, (e, s) in enumerate(zip(errs, ses, strict=True))]


def test_summary_metrics_on_known_values():
    errs = [0.01, -0.01, 0.02, 0.0]
    res = _fake(errs, [0.01] * 4)
    s = rec.summarise(res, {"C3": 0.05}, ("exogenous",), ("eq1_shipped",), true_circuit_sd=0.04)
    cell = s["exogenous"]["eq1_shipped"]["by_compound"]["C3"]
    assert cell["bias"] == pytest.approx(0.005)
    assert cell["rmse"] == pytest.approx(np.sqrt((1e-4 + 1e-4 + 4e-4) / 4), abs=1e-6)
    assert cell["coverage"] == pytest.approx(0.75)  # 0.02 > 1.96 x 0.01
    assert cell["bias_mc_se"] == pytest.approx(np.std(errs, ddof=1) / 2, abs=1e-6)
    assert cell["coverage_mc_se"] == pytest.approx(np.sqrt(0.75 * 0.25 / 4), abs=1e-6)
    assert "coverage_event_cluster" not in cell
    cs = s["exogenous"]["eq1_shipped"]["circuit_sd"]
    assert cs["bias"] == pytest.approx(np.mean([0.03, 0.04, 0.05, 0.06]) - 0.04)
    mean = s["exogenous"]["eq1_shipped"]["mean_over_compounds"]
    assert mean["bias"] == pytest.approx(0.005) and mean["coverage"] == pytest.approx(0.75)
    assert s["exogenous"]["eq1_shipped"]["recovered_all_compounds"] is False  # coverage 0.75 is outside the band


def test_event_cluster_coverage_uses_the_t_critical_value():
    res = [{"scenario": "exogenous", "rep": i,
            "est": {"field_demeaned_ols": {"C3": [0.05 + e, 0.001, 0.01, 2.0]}},
            "flags": {}, "errors": {}, "diag": {}} for i, e in enumerate([0.015, 0.03])]
    s = rec.summarise(res, {"C3": 0.05}, ("exogenous",), ("field_demeaned_ols",))
    cell = s["exogenous"]["field_demeaned_ols"]["by_compound"]["C3"]
    assert cell["coverage_event_cluster"] == pytest.approx(0.5)  # 0.015 <= 2.0 x 0.01 < 0.03
    assert cell["coverage"] == pytest.approx(0.0)


def test_check_expectations_scores_the_rules_and_leaves_missing_ones_open():
    assert all(v["holds"] is None for v in rec.check_expectations({}).values())

    def entry(bias, abs_bias, cov=None, sd_bias=None):
        mo = {"bias": bias, "mean_abs_bias": abs_bias}
        if cov is not None:
            mo["coverage"] = cov
        e = {"mean_over_compounds": mo}
        if sd_bias is not None:
            e["circuit_sd"] = {"bias": sd_bias}
        return e

    summary = {
        "exogenous": {"naive_pooled_ols": entry(-0.06, 0.06), "fuel_within_run": entry(-0.01, 0.01),
                      "fuel_within_run_minus20": entry(-0.0212, 0.0212), "fuel_within_run_plus20": entry(0.0012, 0.0012),
                      "cross_section_cell": entry(0.0, 0.002), "field_demeaned_ols": entry(0.0, 0.003),
                      "eq1_shipped": entry(0.0, 0.003, 0.95, -0.005)},
        "pace": {"naive_pooled_ols": entry(-0.07, 0.07), "fuel_within_run": entry(-0.01, 0.01),
                 "fuel_within_run_minus20": entry(-0.0212, 0.0212), "fuel_within_run_plus20": entry(0.0012, 0.0012),
                 "cross_section_cell": entry(-0.02, 0.02), "field_demeaned_ols": entry(-0.02, 0.02),
                 "eq1_shipped": entry(0.0, 0.004, 0.94, -0.005)},
        "wear": {"naive_pooled_ols": entry(-0.07, 0.07), "fuel_within_run": entry(-0.03, 0.03),
                 "fuel_within_run_minus20": entry(-0.0412, 0.0412), "fuel_within_run_plus20": entry(-0.0188, 0.0188),
                 "eq1_shipped": entry(-0.01, 0.01, 0.80, -0.005),
                 "eq1_mundlak_run_mean_slope": entry(-0.004, 0.004, 0.9)},
    }
    out = rec.check_expectations(summary)
    assert {k: v["holds"] for k, v in out.items()} == {
        "X1": True, "X2": True, "X3": True, "X4": True, "X5": True, "X5b": True, "X6": True,
        "X7": None, "X8": None}  # X7, X8 concern the primary estimator, absent from this summary
    summary["exogenous"]["eq1_exact_within"] = entry(0.0, 0.002, 0.95)
    summary["pace"]["eq1_exact_within"] = entry(0.0, 0.002, 0.95)
    summary["wear"]["eq1_exact_within"] = entry(-0.01, 0.01, 0.80)
    summary["wear"]["eq1_exact_within_mundlak"] = entry(-0.004, 0.004, 0.9)
    out = rec.check_expectations(summary)
    assert out["X7"]["holds"] is True and out["X8"]["holds"] is True
    summary["pace"]["eq1_exact_within"] = entry(0.0, 0.02, 0.95)
    assert rec.check_expectations(summary)["X7"]["holds"] is False
    summary["pace"]["eq1_exact_within"] = entry(0.0, 0.002, 0.95)
    summary["pace"]["eq1_shipped"] = entry(0.0, 0.02, 0.94, -0.005)  # biased under (b): X4 must fail
    assert rec.check_expectations(summary)["X4"]["holds"] is False
    summary["wear"]["fuel_within_run_plus20"] = entry(-0.01, 0.01)    # a wrong fuel shift: X2 must fail
    assert rec.check_expectations(summary)["X2"]["holds"] is False

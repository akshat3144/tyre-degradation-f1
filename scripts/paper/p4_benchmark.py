"""Paper section 4: the Cappello & Hoegh (2025) benchmark, run and audited.

    python scripts/paper/p4_benchmark.py                  # Austria re-run + audits (~1 h)
    python scripts/paper/p4_benchmark.py --season         # also re-run their model on 14 more races (hours)
    python scripts/paper/p4_benchmark.py --no-fits        # only what the caches allow

Writes ``paper/results/benchmark.json``. Every number the paper quotes about the
benchmark comes from here. Needs ``benchmark/upstream/`` and ``benchmark/stan/``
(``bash scripts/fetch_reference.sh``); their code has no licence, so it is
fetched, never committed.

WHAT EACH SECTION ANSWERS

table3        Does their Table 3 reproduce when we run their Extension 1 model?
fuel_defect   Their fuel ramp is indexed by retained row, not LapNumber. How far
              off is it, and does correcting it move anything?
skewt_cv      Their skew-t is scored in CV with draws from which distribution,
              and does it matter? (See benchmark_repro for what the shipped code
              really does.)
gamma         What do their model and ours actually estimate for seconds per kg?
head_to_head  Us against them: Austria, then the 2025 season, with a paired
              race-clustered comparison. No cherry-picking: the one exclusion
              rule is fixed below, BEFORE its results were looked at, and is
              applied to both models identically.
crps_check    Python-only cross-check of the CRPS estimator.

PRE-SPECIFIED EXCLUSION RULE (symmetric)

A test lap is "pathological" if its lap time exceeds 1.07 x the median of the
driver's retained laps in that race (the F1 107 % convention). It depends on the
lap time alone, never on either model's forecast, and is applied to the lap set
both models are scored on. Both the unfiltered and the filtered result are
reported; neither is the "real" one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scoringrules as sr
from scipy import stats

from tyredeg.config import BENCHMARK_CRPS, BENCHMARK_SEASON_2025_PER_RACE, PROCESSED, ROOT
from tyredeg.paper import benchmark_repro as br
from tyredeg.paper.published_benchmark import compare_to_journal
from tyredeg.validation.benchmark import score
from tyredeg.validation.scoring import crosscheck_python

warnings.filterwarnings("ignore")

OUT = ROOT / "paper" / "results" / "benchmark.json"
AUSTRIA = "Austrian Grand Prix"
PUBLISHED_TABLE3 = {"hard": (0.054, 0.004, 0.133), "medium": (0.060, 0.009, 0.120)}
EXCLUSION_RATIO = 1.07
BOOT_B = 20_000
BOOT_SEED = 0
#: Iterations per chain for the season re-run. Their own season run used 10000
#: and says some races did not converge at that; 2000 is a cost compromise and
#: ``fidelity`` below checks it against the full setting on Austria.
SEASON_ITER = 2000


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# scoring a set of fitted folds
# ---------------------------------------------------------------------------


def score_fits(ffs, race: str, kind: str) -> dict:
    """Their aggregation: per-stint mean CRPS, per-stint RMSE; totals mean / sum."""
    rows = []
    for ff in ffs:
        draws = br.predictive_draws(ff, kind, br.seed_of(race, kind, ff.fold.stint, ff.fold.i))
        rows.append(
            {
                "stint": ff.fold.stint,
                "lap": ff.fold.lap_number,
                "actual": ff.actual,
                "mean_pred": float(draws.mean()),
                "crps": float(sr.crps_ensemble(ff.actual, draws)),
                "pit_below": float((draws < ff.actual).mean()),
            }
        )
    df = pd.DataFrame(rows)
    return _aggregate(df)


def _aggregate(df: pd.DataFrame) -> dict:
    g = df.groupby("stint")
    stint_crps = g["crps"].mean()
    stint_rmse = g.apply(lambda x: float(np.sqrt(np.mean((x["actual"] - x["mean_pred"]) ** 2))))
    return {
        "crps": float(stint_crps.mean()),
        "rmse_total": float(stint_rmse.sum()),
        "rmse_mean": float(stint_rmse.mean()),
        "per_stint_crps": [float(x) for x in stint_crps],
        "per_stint_rmse": [float(x) for x in stint_rmse],
        "n_predictions": int(len(df)),
        "laps": df.to_dict("records"),
    }


def drop_pathological(df: pd.DataFrame, median_lap: float) -> pd.DataFrame:
    return df[df["actual"] <= EXCLUSION_RATIO * median_lap]


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def sec_table3(d: br.RaceData) -> dict:
    out = {"model": "Extension1 (compound-specific)", "race": AUSTRIA, "data_rows": d.n}
    out["settings"] = {
        "chains": br.FULL_CHAINS,
        "warmup": br.FULL_WARMUP,
        "sampling": br.FULL_SAMPLING,
        "seed": br.FULL_SEED,
        "adapt_delta": br.CV_ADAPT_DELTA,
        "max_treedepth": br.CV_MAX_TREEDEPTH,
        "theirs": "3 chains x 30000 iterations (Tire_Deg_Paper_Review1.qmd)",
    }
    out["published"] = {k: {"mean": v[0], "lo": v[1], "hi": v[2]} for k, v in PUBLISHED_TABLE3.items()}
    fits = {}
    for fuel in ("theirs", "corrected"):
        fit, secs = br.fit_full(d, "Extension1", fuel)
        fits[fuel] = br.summarise_ext1(fit, secs)
        if fuel == "theirs":
            out["log_pd"] = log_pd_audit(fit, d, fuel)
    out["their_ramp"] = fits["theirs"]
    out["corrected_fuel"] = fits["corrected"]
    t, c = fits["theirs"], fits["corrected"]
    out["reproduces"] = {
        "hard_mean_abs_diff": abs(t["v_hard"]["mean"] - PUBLISHED_TABLE3["hard"][0]),
        "medium_mean_abs_diff": abs(t["v_medium"]["mean"] - PUBLISHED_TABLE3["medium"][0]),
        "hard_interval_within_0.01": bool(
            abs(t["v_hard"]["lo"] - 0.004) < 0.01 and abs(t["v_hard"]["hi"] - 0.133) < 0.01
        ),
        "medium_interval_within_0.01": bool(
            abs(t["v_medium"]["lo"] - 0.009) < 0.01 and abs(t["v_medium"]["hi"] - 0.120) < 0.01
        ),
    }
    out["fuel_correction_effect"] = {
        "delta_v_hard": c["v_hard"]["mean"] - t["v_hard"]["mean"],
        "delta_v_medium": c["v_medium"]["mean"] - t["v_medium"]["mean"],
        "delta_gamma": c["gamma"]["mean"] - t["gamma"]["mean"],
        "p_separation_theirs": t["p_medium_gt_hard"],
        "p_separation_corrected": c["p_medium_gt_hard"],
        "intervals_overlap_after_correction": c["crls_overlap"],
    }
    return out


def log_pd_audit(fit, d: br.RaceData, fuel: str) -> dict:
    """Their ``log_pd`` is ``normal_lpdf(y | z, sdo)``: no fuel term (and, in the
    skew-t file, not even the skew-t). Does it matter?

    It is computed but, as far as their repo shows, never used: the paper loads
    ``loo`` but no WAIC/LOO call appears in the .qmd. The numbers below say how
    wrong it would be if someone did use it.
    """
    z = np.asarray(fit.stan_variable("z"))
    sdo = np.asarray(fit.stan_variable("sdo"))[:, None]
    gam = np.asarray(fit.stan_variable("gamma"))[:, None]
    y = d.y[None, :]
    f = d.fuel(fuel)[None, :]
    theirs = stats.norm.logpdf(y, z, sdo)
    right = stats.norm.logpdf(y, z + gam * f, sdo)
    # lppd per observation = log of the posterior-mean likelihood
    lppd_t = np.log(np.exp(theirs).mean(0)).sum()
    lppd_r = np.log(np.exp(right).mean(0)).sum()
    return {
        "used_in_their_paper_or_code": False,
        "mean_fuel_term_seconds": float((gam * f).mean()),
        "lppd_theirs": float(lppd_t),
        "lppd_with_fuel_term": float(lppd_r),
        "lppd_difference": float(lppd_r - lppd_t),
    }


def sec_fuel_defect(d: br.RaceData, races_allL: pd.DataFrame, table3: dict, austria_cv: dict) -> dict:
    out = {"austria": br.fuel_misplacement(d)}
    mass = 0.033
    out["austria"]["max_seconds_at_0.033_s_per_kg"] = out["austria"]["max_abs_kg"] * mass
    out["austria"]["mean_seconds_at_0.033_s_per_kg"] = out["austria"]["mean_abs_kg"] * mass
    per_race = {}
    for race in BENCHMARK_SEASON_2025_PER_RACE:
        try:
            dd = br.load_race_data(race, laps=races_allL)
        except ValueError:
            continue
        m = br.fuel_misplacement(dd)
        per_race[race] = {k: m[k] for k in ("n_laps_retained", "race_laps", "max_abs_kg", "mean_abs_kg")}
    out["season"] = per_race
    out["season_summary"] = {
        "median_max_abs_kg": float(np.median([v["max_abs_kg"] for v in per_race.values()])),
        "max_max_abs_kg": float(np.max([v["max_abs_kg"] for v in per_race.values()])),
    }
    out["ext1_refit"] = table3["fuel_correction_effect"]
    out["skewt_cv_effect"] = austria_cv.get("fuel_correction")
    return out


def sec_skewt_austria(iters: list[int], workers: int, run: bool) -> dict:
    allL = br.load_all_laps(True)
    d = br.load_race_data(AUSTRIA, laps=allL)
    out = {"folds": len(br.cv_folds(d)), "published": {"crps": BENCHMARK_CRPS["skew_t"], "rmse_total": 1.082}}
    for it in iters:
        for fuel in ("theirs", "corrected"):
            tag = f"iter{it}_{fuel}"
            paths = [br._cache_path(AUSTRIA, fuel, it, f) for f in br.cv_folds(d)]
            if not all(p.exists() for p in paths):
                if not run:
                    continue
                br.fit_race_folds(AUSTRIA, fuel=fuel, iter_total=it, workers=workers, laps=allL)
            ffs = [br.load_ff(p) for p in paths]
            entry = {
                "iter_total_per_chain": it,
                "chains": br.CV_CHAINS,
                "seed": br.CV_SEED,
                "fuel": fuel,
                "fold_seconds_mean": float(np.mean([f.seconds for f in ffs])),
                "divergent_transitions_total": int(sum(f.divergences for f in ffs)),
                "max_rhat": float(np.nanmax([f.max_rhat for f in ffs])),
            }
            for kind in ("shipped", "stan_normal", "normal_fixed_fuel"):
                s = score_fits(ffs, AUSTRIA, kind)
                s["pit_coverage_90"] = float(
                    np.mean([0.05 <= x["pit_below"] <= 0.95 for x in s["laps"]])
                )
                entry[kind] = s
            entry["matched_minus_mismatched_crps"] = entry["shipped"]["crps"] - entry["stan_normal"]["crps"]
            entry["matched_minus_mismatched_rmse"] = (
                entry["shipped"]["rmse_total"] - entry["stan_normal"]["rmse_total"]
            )
            mean_post = float(np.mean([np.mean(f.gamma) for f in ffs]))
            entry["gamma_mean_over_folds"] = mean_post
            out[tag] = entry
    t, c = out.get(f"iter{iters[-1]}_theirs"), out.get(f"iter{iters[-1]}_corrected")
    if t and c:
        out["fuel_correction"] = {
            "iter": iters[-1],
            "crps_theirs_ramp": t["shipped"]["crps"],
            "crps_corrected_fuel": c["shipped"]["crps"],
            "delta_crps": c["shipped"]["crps"] - t["shipped"]["crps"],
            "rmse_theirs_ramp": t["shipped"]["rmse_total"],
            "rmse_corrected_fuel": c["shipped"]["rmse_total"],
        }
    return out


def _cached_season(iter_total: int, only: set[str]) -> bool:
    """True when at least one season race has a complete cache, so a partial run is reused."""
    return any(
        all(br._cache_path(r, 'theirs', iter_total, f).exists() for f in br.cv_folds(br.load_race_data(r, laps=br.load_all_laps(True))))
        for r in only if r in br.Z_RESET0 and r != AUSTRIA
    )


def sec_season_theirs(iter_total: int, workers: int, run: bool, only: set[str]) -> dict:
    """Re-run their skew-t CV on every race we also score, at reduced iterations."""
    allL = br.load_all_laps(True)
    out = {"iter_total_per_chain": iter_total, "races": {}}
    for race, (pub, pub_stints) in BENCHMARK_SEASON_2025_PER_RACE.items():
        if race not in br.Z_RESET0 or race not in only:
            continue
        d = br.load_race_data(race, laps=allL)
        folds = br.cv_folds(d)
        paths = [br._cache_path(race, "theirs", iter_total, f) for f in folds]
        if not all(p.exists() for p in paths):
            if not run:
                continue
            br.fit_race_folds(race, "theirs", iter_total, workers, laps=allL)
        ffs = [br.load_ff(p) for p in paths]
        rec = {"published_crps": pub, "published_stints": pub_stints, "median_lap": float(np.median(d.y))}
        for kind in ("shipped", "stan_normal"):
            rec[kind] = score_fits(ffs, race, kind)
        rec["max_rhat"] = float(np.nanmax([f.max_rhat for f in ffs]))
        rec["divergences"] = int(sum(f.divergences for f in ffs))
        rec["fold_seconds_mean"] = float(np.mean([f.seconds for f in ffs]))
        out["races"][race] = rec
    return out


# ---------------------------------------------------------------------------
# our side and the comparison
# ---------------------------------------------------------------------------


def our_scores(races: pd.DataFrame, allL: pd.DataFrame) -> dict:
    """Our model on (a) folds rebuilt from our feed and (b) their exact folds."""
    out = {}
    for ev, g in races.groupby("event"):
        ev = str(ev)
        if ev not in BENCHMARK_SEASON_2025_PER_RACE or ev not in br.Z_RESET0:
            continue
        d = br.load_race_data(ev, laps=allL)
        fo = br.cv_folds(d)
        folds = [(f.lap_number - 1, f.lap_number) for f in fo]
        sol = {f.lap_number: int(d.stint[f.test_row - 1]) for f in fo}
        own = score(g, "HAM", ev)
        ex = score(g, "HAM", ev, folds=folds, stint_of_lap=sol)
        out[ev] = {
            "median_lap_theirs": float(np.median(d.y)),
            "own_folds": {"crps": own.crps, "n": own.n_predictions, "laps": own.per_lap,
                          "rmse_total": own.rmse_total},
            "their_folds": {
                "crps": ex.crps,
                "rmse_total": ex.rmse_total,
                "n": ex.n_predictions,
                "n_folds_theirs": len(folds),
                "unscored_laps": ex.unscored_laps,
                "per_stint_crps": ex.per_stint_crps,
                "laps": ex.per_lap,
            },
        }
    return out


def paired(diff: np.ndarray, ratio: np.ndarray, wins: int, n: int) -> dict:
    """Race-clustered comparison: one observation per race, bootstrap over races."""
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(diff), (BOOT_B, len(diff)))
    bm = diff[idx].mean(1)
    br_ = np.log(ratio)[idx].mean(1)
    try:
        w = stats.wilcoxon(diff).pvalue
    except ValueError:
        w = float("nan")
    return {
        "n_races": int(n),
        "ours_wins": int(wins),
        "mean_diff_ours_minus_theirs": float(diff.mean()),
        "mean_diff_ci95": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))],
        "median_diff": float(np.median(diff)),
        "geometric_mean_ratio": float(np.exp(np.log(ratio).mean())),
        "geometric_mean_ratio_ci95": [float(np.exp(np.quantile(br_, 0.025))),
                                       float(np.exp(np.quantile(br_, 0.975)))],
        "sign_test_p_two_sided": float(stats.binomtest(wins, n, 0.5).pvalue),
        "wilcoxon_p_two_sided": float(w),
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "race"},
    }


def lap_filtered_race_score(laps: list[dict], median_lap: float, filtered: bool) -> float | None:
    df = pd.DataFrame(laps)
    if df.empty:
        return None
    if filtered:
        df = drop_pathological(df, median_lap)
    if df.empty:
        return None
    return float(df.groupby("stint")["crps"].mean().mean())


def sec_head_to_head(ours: dict, theirs_season: dict, austria_theirs: list[dict] | None = None) -> dict:
    out = {"exclusion_rule": f"lap time > {EXCLUSION_RATIO} x median retained lap of the race"}

    # Austria, one race: lap-level bootstrap. 16-17 laps cannot rank forecasters.
    a = ours[AUSTRIA]
    boot = {}
    for name, key in (("own_folds_16", "own_folds"), ("their_folds_17", "their_folds")):
        lap_crps = np.array([x["crps"] for x in a[key]["laps"]])
        rng = np.random.default_rng(BOOT_SEED)
        bs = lap_crps[rng.integers(0, len(lap_crps), (BOOT_B, len(lap_crps)))].mean(1)
        top = float(lap_crps.max() / lap_crps.sum())
        boot[name] = {
            "n_laps": int(len(lap_crps)),
            "per_lap_mean_crps": float(lap_crps.mean()),
            "stint_mean_crps": float(a[key]["crps"]),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "p_beat_0.2022": float((bs < 0.2022).mean()),
            "their_best_inside_ci": bool(np.quantile(bs, 0.025) <= 0.2022 <= np.quantile(bs, 0.975)),
            "largest_single_lap_share": top,
            "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "lap (unpaired against a constant)"},
        }
    out["austria"] = {
        "ours_own_folds": a["own_folds"]["crps"],
        "ours_their_folds": a["their_folds"]["crps"],
        "ours_rmse_total_own_folds": a["own_folds"]["rmse_total"],
        "ours_rmse_total_their_folds": a["their_folds"]["rmse_total"],
        "theirs_paper": BENCHMARK_CRPS["skew_t"],
        "theirs_repo_csv": BENCHMARK_SEASON_2025_PER_RACE[AUSTRIA][0],
        "bootstrap": boot,
    }

    # Austria, paired lap by lap against OUR re-run of their model (same 17 laps).
    if austria_theirs:
        o = pd.DataFrame(a["their_folds"]["laps"]).set_index("lap")
        t = pd.DataFrame(austria_theirs).set_index("lap")
        common = sorted(set(o.index) & set(t.index))
        diff = (o.loc[common, "crps"] - t.loc[common, "crps"]).to_numpy()
        rng = np.random.default_rng(BOOT_SEED)
        bs = diff[rng.integers(0, len(diff), (BOOT_B, len(diff)))].mean(1)
        out["austria_paired_vs_reproduced_theirs"] = {
            "n_common_laps": len(common),
            "ours_mean_per_lap": float(o.loc[common, "crps"].mean()),
            "theirs_mean_per_lap": float(t.loc[common, "crps"].mean()),
            "mean_paired_diff": float(diff.mean()),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "p_ours_better_on_resample": float((bs < 0).mean()),
            "laps_ours_better": int((diff < 0).sum()),
            "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "lap, paired"},
        }

    # The pre-specified exclusion rule, applied to their exact test laps. Both
    # models are scored on the same laps, so the rule is symmetric by construction.
    n_excl = {}
    for race, o2 in ours.items():
        thr = EXCLUSION_RATIO * o2["median_lap_theirs"]
        n_excl[race] = int(sum(x["actual"] > thr for x in o2["their_folds"]["laps"]))
    out["exclusion_rule_effect"] = {
        "rule": f"actual lap time > {EXCLUSION_RATIO} x median retained lap",
        "test_laps_excluded_per_race": n_excl,
        "races_affected": int(sum(v > 0 for v in n_excl.values())),
        "note": "With their exact test laps the rule removes nothing where the count is 0; the "
        "filtered and unfiltered comparisons are then identical. The 32 s Singapore lap was "
        "outside their test set to begin with.",
    }

    # Season: three versions, so the reader can see what each choice does.
    def table(ours_key: str, theirs_of, label: str, filtered: bool = False):
        rows = []
        for race, o in ours.items():
            th = theirs_of(race)
            if th is None:
                continue
            if filtered:
                oc = lap_filtered_race_score(o[ours_key]["laps"], o["median_lap_theirs"], True)
            else:
                oc = o[ours_key]["crps"]
            if oc is None or th is None:
                continue
            rows.append({"race": race, "ours": oc, "theirs": th})
        df = pd.DataFrame(rows)
        wins = int((df["ours"] < df["theirs"]).sum())
        res = paired((df["ours"] - df["theirs"]).to_numpy(), (df["ours"] / df["theirs"]).to_numpy(),
                     wins, len(df))
        res.update(
            label=label,
            ours_mean=float(df["ours"].mean()),
            theirs_mean=float(df["theirs"].mean()),
            ours_median=float(df["ours"].median()),
            theirs_median=float(df["theirs"].median()),
            per_race=df.round(4).to_dict("records"),
        )
        return res

    pub = lambda r: BENCHMARK_SEASON_2025_PER_RACE[r][0]  # noqa: E731
    out["season_vs_published"] = {
        "A_own_folds_vs_published": table("own_folds", pub, "our folds from our feed (the original artifact)"),
        "B_their_folds_vs_published": table("their_folds", pub, "their exact test laps"),
    }
    out["season_vs_published"]["A_own_folds_vs_published"]["note"] = (
        "Not like-for-like: our feed keeps laps their pipeline drops (Singapore laps > 58, "
        "the 3.83 outlier) and lacks laps theirs has."
    )

    races = theirs_season.get("races", {}) if theirs_season else {}
    if races:
        # Reproduced theirs, matched laps only: score both models on the laps BOTH have.
        rows_full, rows_flt = [], []
        for race, tr in races.items():
            if race not in ours:
                continue
            o = pd.DataFrame(ours[race]["their_folds"]["laps"])
            t = pd.DataFrame(tr["shipped"]["laps"])
            common = sorted(set(o["lap"]) & set(t["lap"]))
            if not common:
                continue
            med = ours[race]["median_lap_theirs"]
            for flt, rows in ((False, rows_full), (True, rows_flt)):
                oo, tt = o[o["lap"].isin(common)], t[t["lap"].isin(common)]
                if flt:
                    keep = [lap for lap, act in zip(tt["lap"], tt["actual"], strict=True)
                            if act <= EXCLUSION_RATIO * med]
                    oo, tt = oo[oo["lap"].isin(keep)], tt[tt["lap"].isin(keep)]
                if oo.empty:
                    continue
                rows.append(
                    {
                        "race": race,
                        "n_laps": int(len(oo)),
                        "ours": float(oo.groupby("stint")["crps"].mean().mean()),
                        "theirs": float(tt.groupby("stint")["crps"].mean().mean()),
                    }
                )
        for name, rows, label in (
            ("C_common_laps_vs_reproduced", rows_full, "both models on the laps both can score, no exclusion"),
            ("D_common_laps_vs_reproduced_excl_107pct", rows_flt,
             "as C, after the pre-specified 107% exclusion applied to both"),
        ):
            df = pd.DataFrame(rows)
            wins = int((df["ours"] < df["theirs"]).sum())
            res = paired((df["ours"] - df["theirs"]).to_numpy(),
                         (df["ours"] / df["theirs"]).to_numpy(), wins, len(df))
            res.update(label=label, ours_mean=float(df["ours"].mean()), theirs_mean=float(df["theirs"].mean()),
                       ours_median=float(df["ours"].median()), theirs_median=float(df["theirs"].median()),
                       n_laps_total=int(df["n_laps"].sum()), per_race=df.round(4).to_dict("records"))
            out[name] = res
        # Reproduction fidelity: reproduced vs published per race.
        rep = pd.DataFrame(
            [{"race": r, "reproduced": v["shipped"]["crps"], "published": v["published_crps"]}
             for r, v in races.items()]
        )
        out["their_model_reproduction"] = {
            "n_races": int(len(rep)),
            "mean_reproduced": float(rep["reproduced"].mean()),
            "mean_published": float(rep["published"].mean()),
            "corr": float(np.corrcoef(rep["reproduced"], rep["published"])[0, 1]),
            "mean_abs_diff": float((rep["reproduced"] - rep["published"]).abs().mean()),
            "per_race": rep.round(4).to_dict("records"),
        }
    return out


def sec_gamma(table3: dict, skewt_full: dict | None, hier: dict | None) -> dict:
    return {
        "theirs_ext1_their_ramp": table3["their_ramp"]["gamma"],
        "theirs_ext1_corrected_fuel": table3["corrected_fuel"]["gamma"],
        "theirs_skewt_full_race": skewt_full,
        "ours_hierarchical": hier,
        "physical_range_s_per_kg": [0.030, 0.035],
    }


def sec_hier_gamma(races: pd.DataFrame, laps24: pd.DataFrame | None) -> dict:
    """Refit our hierarchical model on whole races and read gamma off it.

    Posterior mean and 95% interval under the shipped prior N(0.032, 0.02) and a
    diffuse one N(0.032, 1). The second says what the data say; any gap between
    the two is the prior talking.
    """
    from tyredeg.models.hierarchical import forecast_lap

    cache = br.FITS_DIR / "hier_gamma.json"
    if cache.exists():  # ~15 minutes of MCMC; seeds are fixed so the cache is just resumability
        return json.loads(cache.read_text(encoding="utf-8"))

    cases = [("Austrian Grand Prix 2025", races[races["event"] == AUSTRIA], 110.0, 2025)]
    if laps24 is not None:
        b24 = laps24[(laps24["session"] == "R") & (laps24["event"].str.contains("Bahrain"))]
        if len(b24):
            cases.append(("Bahrain Grand Prix 2024, 110 kg start (correct)", b24, 110.0, 2024))
            cases.append(("Bahrain Grand Prix 2024, 70 kg start (the bug)", b24, 70.0, 2024))
    out = {}
    for name, g, start, _ in cases:
        last = int(g["LapNumber"].max())
        drv = "HAM" if (g["Driver"] == "HAM").any() else g["Driver"].iloc[0]
        for plabel, prior in (("shipped_prior_sd0.02", (0.032, 0.02)), ("diffuse_prior_sd1", (0.032, 1.0))):
            try:
                f = forecast_lap(g, train_to=last - 1, pred_driver=drv, pred_lap=last, race_laps=last,
                                 start_kg=start, seed=0, gamma_prior=prior)
            except Exception as exc:  # noqa: BLE001
                out[f"{name} | {plabel}"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                continue
            gd = f.gamma_draws
            out[f"{name} | {plabel}"] = {
                "gamma_mean": float(gd.mean()),
                "gamma_ci95": [float(np.quantile(gd, 0.025)), float(np.quantile(gd, 0.975))],
                "start_kg": start,
                "mean_if_rescaled_to_110kg": float(gd.mean() * start / 110.0),
                "seed": 0,
                "driver_forecast": drv,
            }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def sec_skewt_full(d: br.RaceData) -> dict:
    fit, secs = br.fit_full(d, "base_t", "theirs", chains=2, warmup=2000, sampling=2000)
    g = np.asarray(fit.stan_variable("gamma"))
    return {"mean": float(g.mean()), "lo": float(np.quantile(g, 0.025)),
            "hi": float(np.quantile(g, 0.975)), "seconds": round(secs, 1),
            "settings": "2 chains x 2000+2000, seed 1"}


def write_csvs(res: dict) -> None:
    """Flat copies of the main tables in benchmark/results/ for anyone without a JSON reader."""
    outdir = ROOT / "benchmark" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    t3 = res["table3"]
    rows = []
    for fuel in ("their_ramp", "corrected_fuel"):
        for k in ("v_hard", "v_medium", "gamma"):
            rows.append({"fuel": fuel, "parameter": k, **t3[fuel][k]})
    for k, (m, lo, hi) in PUBLISHED_TABLE3.items():
        rows.append({"fuel": "published", "parameter": f"v_{k}", "mean": m, "lo": lo, "hi": hi})
    pd.DataFrame(rows).to_csv(outdir / "table3_reproduction.csv", index=False)
    key = f"iter{br.CV_ITER}_theirs"
    e = res["skewt_cv_austria"].get(key)
    if e:
        recs = []
        for kind in ("shipped", "stan_normal", "normal_fixed_fuel"):
            for x in e[kind]["laps"]:
                recs.append({"draws": kind, **x})
        pd.DataFrame(recs).to_csv(outdir / "skewt_cv_austria_laps.csv", index=False)
    h = res["head_to_head"]["season_vs_published"]["B_their_folds_vs_published"]
    pd.DataFrame(h["per_race"]).to_csv(outdir / "season_head_to_head.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fits", action="store_true", help="use cached fits only")
    ap.add_argument("--season", action="store_true",
                    help="re-run THEIR model on the season's races (hours: ~5 min per fold; opt-in)")
    ap.add_argument("--no-hier", action="store_true", help="skip our hierarchical gamma refits")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--season-iter", type=int, default=SEASON_ITER)
    args = ap.parse_args()

    if not br.upstream_available():
        sys.exit("benchmark/upstream or benchmark/stan missing: run bash scripts/fetch_reference.sh")

    t_all = time.time()
    allL = br.load_all_laps(True)
    d = br.load_race_data(AUSTRIA, laps=allL)
    laps = pd.read_parquet(PROCESSED / "laps_2025.parquet")
    races = laps[laps["session"] == "R"]

    res: dict = {}
    res["meta"] = {
        "generated_by": "scripts/paper/p4_benchmark.py",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scoringrules": sr.__version__ if hasattr(sr, "__version__") else "0.11.0",
        "upstream_sha256_16": {
            "Hamilton_2025_All_Races.csv": sha256(br.HAM_CSV),
            "base_t.stan": sha256(br.STAN_DIR / "full_race_1driver_and_fuel_base_t.stan"),
            "Extension1.stan": sha256(br.STAN_DIR / "full_race_1driver_and_fuel_Extension1.stan"),
            "All_CV_results1.csv": sha256(br.RESULTS_CSV_DIR / "All_CV_results1.csv"),
        },
        "cmdstan": "2.39.0 via cmdstanpy",
        "seeds": {"table3": br.FULL_SEED, "cv_folds": br.CV_SEED, "bootstrap": BOOT_SEED, "hier": 0},
        "exclusion_rule": f"lap time > {EXCLUSION_RATIO} x median retained lap",
    }

    run = not args.no_fits
    print("table3 ...", flush=True)
    t3 = sec_table3(d)
    res["table3"] = t3
    print("skew-t CV, Austria ...", flush=True)
    sk = sec_skewt_austria([br.CV_ITER, args.season_iter], args.workers, run)
    res["skewt_cv_austria"] = sk
    res["fuel_defect"] = sec_fuel_defect(d, allL, t3, sk)

    print("our model, season ...", flush=True)
    ours = our_scores(races, allL)

    print("their model, season ...", flush=True)
    theirs_season = None
    if args.season or _cached_season(args.season_iter, set(ours)):
        theirs_season = sec_season_theirs(args.season_iter, args.workers, run, set(ours))
        res["skewt_cv_season"] = {
            k: {kk: vv for kk, vv in v.items() if kk not in ("shipped", "stan_normal")}
            | {
                "reproduced_crps": v["shipped"]["crps"],
                "stan_normal_crps": v["stan_normal"]["crps"],
                "reproduced_rmse_total": v["shipped"]["rmse_total"],
                "stan_normal_rmse_total": v["stan_normal"]["rmse_total"],
            }
            for k, v in theirs_season["races"].items()
        }

    a_theirs = (sk.get(f"iter{br.CV_ITER}_theirs") or {}).get("shipped", {}).get("laps")
    if theirs_season and theirs_season["races"]:
        rr = theirs_season["races"]
        m = np.array([v["shipped"]["crps"] for v in rr.values()])
        n = np.array([v["stan_normal"]["crps"] for v in rr.values()])
        res["skewt_cv_season_matched_vs_mismatched"] = {
            "n_races": int(len(m)),
            "mean_crps_matched_skewt_draws": float(m.mean()),
            "mean_crps_stan_normal_draws": float(n.mean()),
            "mean_diff_matched_minus_normal": float((m - n).mean()),
            "races_matched_better": int((m < n).sum()),
            "max_abs_diff": float(np.abs(m - n).max()),
            "note": "same posterior draws scored two ways; the shipped CV uses the matched draws",
        }
    res["head_to_head"] = sec_head_to_head(ours, theirs_season, a_theirs)

    # Journal version (2026), Table 4: per-race SSM and AR(1) numbers, transcribed in
    # benchmark/published_table4.json. Arithmetic on the scores above; no new fits.
    res["journal_table4"] = compare_to_journal(ours)

    print("gamma ...", flush=True)
    n_f = [sk[k]["gamma_mean_over_folds"] for k in sk if isinstance(sk[k], dict) and "gamma_mean_over_folds" in sk[k]]
    full = {"mean_over_cv_folds": n_f[0] if n_f else None,
            "note": "posterior mean of gamma averaged over the 17 CV training windows (skew-t, their ramp)"}
    hier = None
    if not args.no_hier and (run or (br.FITS_DIR / 'hier_gamma.json').exists()):
        p24 = PROCESSED / "laps_2024.parquet"
        hier = sec_hier_gamma(races, pd.read_parquet(p24) if p24.exists() else None)
    res["gamma"] = sec_gamma(t3, full, hier)

    res["crps_crosscheck_python"] = crosscheck_python()
    res["runtime_seconds"] = round(time.time() - t_all, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    write_csvs(res)
    print(f"wrote {OUT.relative_to(ROOT)}  ({res['runtime_seconds']}s)")


if __name__ == "__main__":
    main()

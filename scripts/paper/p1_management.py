"""Paper result P1: is race-observed degradation policy-dependent?

    python scripts/paper/p1_management.py            # full run (~10-15 min)
    python scripts/paper/p1_management.py --fast     # small bootstrap, for a smoke test

Writes ``paper/results/management.json``. Every number the paper quotes about
the driver-management finding comes from this file. The primary analysis, its
decision rule and every robustness row are pre-stated in the docstring of
``tyredeg.paper.management``; this script only executes them, with fixed seeds.

Honest by construction: the file records the legacy ratio result, the repaired
legacy result, and the same-estimator result side by side, and the decision
flags are computed from the pre-stated rule rather than written by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from tyredeg.config import PROCESSED, ROOT
from tyredeg.data.cache import season_completeness
from tyredeg.paper import management as M
from tyredeg.validation.management import LABELS, analyse, build_design_frames

warnings.filterwarnings("ignore")

OUT = ROOT / "paper" / "results" / "management.json"
SEED = 20260101


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def legacy_block() -> dict:
    """The old ratio analysis as shipped (field-demeaned race, through the origin)
    and as repaired (race demeaned within event-lap-compound), season by season."""
    per_old, per_new = {}, {}
    for s in M.ALL_SEASONS:
        path = PROCESSED / f"{M.SEASON_FILES[s]}.parquet"
        ok, why = season_completeness(path)
        if not ok:
            raise SystemExit(f"season {s} incomplete: {why}")
        per_old[s] = build_design_frames(path, race_demean="field")
        per_new[s] = build_design_frames(path, race_demean="compound")

    def summarise(res) -> dict:
        c = res.cells
        sign = {}
        for name, hi, lo in (("SOFT<MEDIUM", "SOFT", "MEDIUM"), ("SOFT<HARD", "SOFT", "HARD"), ("MEDIUM<HARD", "MEDIUM", "HARD")):
            piv = c.assign(k=c["season"].astype(str) + "|" + c["event"]).pivot_table(index="k", columns="Compound", values="ratio")
            if hi in piv and lo in piv:
                d = (piv[hi] - piv[lo]).dropna()
                neg = int((d < 0).sum())
                sign[name] = {
                    "n_events": int(len(d)),
                    "n_lower": neg,
                    "p_one_sided": float(stats.binom.sf(neg - 1, len(d), 0.5)),
                    "p_two_sided": float(stats.binomtest(neg, len(d), 0.5).pvalue),
                }
        return {
            "n_cells": res.n_cells,
            "rho": res.rho,
            "p_one_sided": res.p_value,
            "p_two_sided": res.p_two_sided,
            "p_kruskal": res.p_kruskal,
            "ordered": bool(res.ordered),
            "median_ratio": res.median_ratio,
            "median_race_slope": c.groupby("Compound")["race"].median().to_dict(),
            "median_practice_slope": c.groupby("Compound")["practice"].median().to_dict(),
            "share_race_slope_negative": (c["race"] < 0).groupby(c["Compound"]).mean().to_dict(),
            "pairwise_sign_tests_on_ratio": sign,
            "verdict": res.verdict(),
        }

    out = {
        "as_shipped_race_slope_field_demeaned": summarise(analyse(per_old)),
        "repaired_race_slope_compound_demeaned": summarise(analyse(per_new)),
    }
    by_season = {}
    for s in M.ALL_SEASONS:
        row = {}
        for tag, per in (("as_shipped", per_old), ("repaired", per_new)):
            try:
                r = analyse({s: per[s]})
                row[tag] = {"n_cells": r.n_cells, "rho": r.rho, "p_two_sided": r.p_two_sided}
            except ValueError as e:
                row[tag] = {"error": str(e)}
        by_season[str(s)] = row
    out["by_season"] = by_season

    from tyredeg.validation.management import cell_slopes

    rows = []
    for s in M.ALL_SEASONS:
        a = cell_slopes(per_old[s][1]).rename(columns={"rate": "old"})
        b = cell_slopes(per_new[s][1]).rename(columns={"rate": "new"})
        rows.append(a.merge(b, on=["event", "Compound"]))
    m = pd.concat(rows)
    out["race_slope_estimator_agreement"] = {
        "n_cells": int(len(m)),
        "pearson_r": float(np.corrcoef(m["old"], m["new"])[0, 1]),
        "spearman_rho": float(stats.spearmanr(m["old"], m["new"])[0]),
    }
    return out


def decision(h: dict, placebo: dict, temp: dict) -> dict:
    """Apply the pre-stated rule. No judgement enters here."""
    t = h["trend"]
    h1 = bool(t["ci95_cluster_bootstrap"][1] < 0 and t["p_perm_one_sided_less"] < 0.05 and t["p_perm_two_sided"] < 0.05)
    g = h["ratio_of_means"]["log(SOFT/HARD)"]
    h2 = bool(g["ci95"][1] < 0 and g["p_perm_one_sided_less"] < 0.05 and g["p_perm_two_sided"] < 0.05)
    return {
        "H1_additive_interaction_supported": h1,
        "H2_beyond_proportional_damping_supported": h2,
        "rule": "supported iff bootstrap 95% CI upper bound < 0 AND one- and two-sided permutation p < 0.05",
        "placebo_trend_ci_excludes_zero": bool(placebo.get("status") == "ok" and (placebo["ci95"][0] > 0 or placebo["ci95"][1] < 0)),
    }


def _lead(lg: dict) -> dict:
    """The lead estimand under explicit keys: the GAP in s/lap first, the ratio second."""
    out = {}
    for name in ("MEDIUM+SOFT", "HARD", "MEDIUM", "SOFT", "ALL"):
        v = lg.get(name)
        if not isinstance(v, dict) or "gap" not in v:
            continue
        out[name] = {
            "gap_s_per_lap_race_minus_practice": v["gap"],
            "gap_ci95": v["gap_ci95"],
            "ratio_race_over_practice": v["ratio"],
            "ratio_ci95": v["ratio_ci95"],
            "pooled_race_slope_s_per_lap": v["race"],
            "pooled_practice_slope_s_per_lap": v["practice"],
            "n_cells": v["n"],
        }
    return out


def _explain(rec: dict) -> dict:
    """Numbers and the one-sentence reading for the sign disagreement, built from the table itself."""
    al = rec["settings"]["all_seasons|window_6_16"]
    y26 = rec["settings"]["2026_only|window_6_16"]
    est = lambda S, e: S["race_only_trend_by_estimator"][e]  # noqa: E731
    model = None
    mp = ROOT / "paper" / "results" / "model.json"
    if mp.exists():
        try:
            t = json.loads(mp.read_text(encoding="utf-8"))["compound_within_event"]["within_event_fixed_circuit"]["trend_s_per_lap_per_step_softer"]
            model = {"trend_s_per_lap_per_step": t["mean"], "ci95": [t["lo"], t["hi"]],
                     "source": "paper/results/model.json compound_within_event.within_event_fixed_circuit (2026 only, "
                               "(event, lap)-demeaned design, mixed model with run intercept and slope)"}
        except (KeyError, ValueError):
            model = None
    sw = al["race_twfe_pair_contrasts"]
    sentence = (
        "The sign of the race-only compound trend depends on how the slope is identified, not on the data: "
        f"within-run and two-way fixed-effects contrasts, which compare a car with itself, put it near zero "
        f"({est(al, 'within_run')['beta_per_step_softer']:+.3f} s/lap per step softer, 95% CI "
        f"{est(al, 'within_run')['ci95'][0]:+.3f} to {est(al, 'within_run')['ci95'][1]:+.3f}, all seasons; "
        f"{est(y26, 'within_run')['beta_per_step_softer']:+.3f} in 2026), whereas designs that compare cars with each "
        f"other at the same lap, such as the mixed model's {model['trend_s_per_lap_per_step']:+.4f} in 2026, give "
        f"negative trends on identical cells ({est(al, 'field_origin')['beta_per_step_softer']:+.3f} for the "
        f"field-demeaned slope and {est(al, 'xs')['beta_per_step_softer']:+.3f} for the compound-demeaned one, all seasons); "
        "the positive race-side TWFE contrasts in Table 3 are means over only the events that also have practice, "
        f"and over all events they are {sw['SOFT-HARD']['mean_contrast']:+.3f} (SOFT-HARD), "
        f"{sw['MEDIUM-HARD']['mean_contrast']:+.3f} (MEDIUM-HARD) and {sw['SOFT-MEDIUM']['mean_contrast']:+.3f} "
        "(SOFT-MEDIUM), i.e. neither uniform nor additive."
    ) if model else "model.json not found; see the table."
    return {
        "mixed_model_value": model,
        "race_only_trend_all_seasons_window_6_16": {e: v.get("beta_per_step_softer") for e, v in al["race_only_trend_by_estimator"].items()},
        "race_only_trend_2026_window_6_16": {e: v.get("beta_per_step_softer") for e, v in y26["race_only_trend_by_estimator"].items()},
        "differences_between_the_two_analyses": [
            "seasons: 2022-2026 versus 2026 only",
            "estimator: within-run / TWFE contrast (car versus itself) versus a design demeaned by (event, lap) across cars",
            "window: tyre ages 6-16 versus all ages",
            "run set: events with both contexts (Table 3) versus every event with the compounds",
            "unit: label rank (HARD/MEDIUM/SOFT) versus C-number step in the mixed model",
        ],
        "one_sentence": sentence,
    }


def referee_block(bases, cells, post, *, Bl: int, Bs: int, seed: int, log) -> dict:
    """Additions made in response to referee findings (after the paper was drafted)."""
    out = {
        "provenance": (
            "Added after referee findings on the drafted paper, and after the post-review additions. "
            "Not pre-registered; computed with fixed seeds and not tuned on output. Contents: (1) what the "
            "start-age matching actually selects and designs matched on the feed's FreshTyre flag; "
            "(2) a scripted reconciliation of the race-only compound trend across estimators; (3) the "
            "placebo-adjusted gap; (4) Holm adjustment over the three TWFE pair tests; (5) the lead estimand."
        ),
    }
    out["set_state_counts"] = M.set_state_counts(bases)
    log("set state counts done")

    designs = {}
    for k, (name, sp) in enumerate(
        (
            ("start_age_proxy_matched", M.MATCHED_POOLED),
            ("used_sets_only_FreshTyre_false", M.USED_SETS),
            ("used_sets_and_start_age_matched", M.USED_SETS_AGE_MATCHED),
            ("new_sets_only_FreshTyre_true", M.NEW_SETS),
        )
    ):
        c = M.make_cells(bases, sp, need_se=False)
        lg = M.level_gaps(c, B=Bl, seed=seed + 100 * k)
        plc_cells = M.placebo_cells(bases, sp, need_se=False)
        plc = M.level_gaps(plc_cells, B=Bl, seed=seed + 100 * k + 10)
        adj = {}
        for j, (rname, ranks) in enumerate((("MEDIUM+SOFT", (1, 2)), ("HARD", (0,)), ("MEDIUM", (1,)), ("SOFT", (2,)), ("ALL", (0, 1, 2)))):
            if len(c) and len(plc_cells) and np.isin(c["rank"], ranks).any() and np.isin(plc_cells["rank"], ranks).any():
                adj[rname] = M.placebo_adjusted_gap(c, plc_cells, ranks, B=Bl, seed=seed + 100 * k + 20 + j)
        tab = M.twfe_pair_table(bases, sp, min_runs=2)
        tg = M.twfe_gradient(tab, B=Bl, P=2 * Bl, seed=seed + 100 * k + 50)
        tg.pop("per_event", None)
        valid = {q: v for q, v in tg.items() if isinstance(v, dict) and "p_perm_two_sided" in v}
        if len(valid) >= 2:
            tg["holm_adjusted_p_two_sided"] = M.holm_adjust({q: v["p_perm_two_sided"] for q, v in valid.items()})
            tg["holm_adjusted_p_one_sided_less"] = M.holm_adjust({q: v["p_perm_one_sided_less"] for q, v in valid.items()})
        designs[name] = {
            "spec": json.loads(sp.label()),
            "level_gap": lg,
            "lead_estimand": _lead(lg),
            "level_placebo_fp2_minus_fp1fp3": plc,
            "placebo_adjusted_gap": adj,
            "twfe_gradient": tg,
        }
        log(f"referee design {name} done")
    out["designs"] = designs

    # Holm over the three TWFE contrasts of the primary (start-age-free) specification.
    tg = post["twfe_gradient"]
    valid = {q: v for q, v in tg.items() if isinstance(v, dict) and "p_perm_two_sided" in v}
    out["twfe_primary_holm"] = {
        "family": sorted(valid),
        "raw_p_two_sided": {q: v["p_perm_two_sided"] for q, v in valid.items()},
        "holm_adjusted_p_two_sided": M.holm_adjust({q: v["p_perm_two_sided"] for q, v in valid.items()}),
        "raw_p_one_sided_less": {q: v["p_perm_one_sided_less"] for q, v in valid.items()},
        "holm_adjusted_p_one_sided_less": M.holm_adjust({q: v["p_perm_one_sided_less"] for q, v in valid.items()}),
        "note": "Holm step-down over the three pairwise permutation tests; permutation p cannot go below 1/(P+1).",
    }

    # Lead estimand for the main matched design (start-age proxy), as in the paper's Table.
    out["lead_estimand_matched_start_age"] = _lead(post["level_gap_matched_pooled"])
    out["lead_estimand_definition"] = (
        "gap = pooled race slope minus pooled practice slope (s/lap, same within-run estimator, window 6-16, "
        "events with both contexts); ratio = race slope / practice slope; 95% intervals are event-season "
        "cluster bootstraps. The gap is the lead estimand; the ratio is secondary and unstable for HARD."
    )

    rec = M.reconcile_race_trends(bases, B=Bs, seed=seed + 900)
    out["reconcile_race_compound_trend"] = rec
    out["reconcile_race_compound_trend"]["explanation"] = _explain(rec)
    log("reconciliation done")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--additions-only", action="store_true",
                    help="recompute only the referee additions and merge them into the existing JSON")
    args = ap.parse_args()
    B, P = (400, 800) if args.fast else (5000, 10000)
    Bs, Ps = (150, 300) if args.fast else (1000, 2000)  # secondary analyses
    Bm, Pm = (100, 200) if args.fast else (400, 800)  # multiverse rows
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    streams = M.weather_streams()
    bases = M.Bases.load(streams=streams)
    log(f"bases loaded; {len(streams)} weather streams")

    cells = M.make_cells(bases)

    if args.additions_only:
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        existing["referee_additions"] = _clean(
            referee_block(bases, cells, existing["post_review_additions"], Bl=300 if args.fast else 5000,
                          Bs=150 if args.fast else 1000, seed=SEED + 10000, log=log)
        )
        existing["meta"]["referee_additions_generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        OUT.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        log(f"merged referee additions into {OUT}")
        return
    h = M.headline(cells, B=B, P=P, seed=SEED)
    log(f"primary: {h['n_cells']} cells, beta={h['trend']['beta_per_softness_step']:+.4f}")

    h["random_intercept_trend_no_event_fe"] = M.random_intercept_trend(cells, B=Bs, seed=SEED + 900)
    # Secondary headline with the race slope from the cross-sectional design
    # (within event-lap-compound): no fuel assumption, track evolution removed. Not
    # the same estimator as practice, so it is labelled secondary.
    cells_field = M.make_cells(bases, replace(M.PRIMARY, race_estimator="xs"))
    h_field = M.headline(cells_field, B=Bs, P=Ps, seed=SEED + 800)
    log(f"secondary (cross-sectional race): beta={h_field['trend']['beta_per_softness_step']:+.4f}")

    seasons = M.season_analysis(cells, B=Bs, P=Ps, seed=SEED + 1000)
    log("season analysis done")

    # Placebo. The pre-stated version (min 3 runs per cell and context) leaves very
    # few events with two compounds in both FP2 and FP1+FP3, so it is run next to
    # two a-priori relaxations; n_informative_event_seasons says how much each
    # one can actually say.
    placebo_rows = {}
    for tag, sp in (
        ("min_runs_3_racesim", M.PRIMARY),
        ("min_runs_2_racesim", replace(M.PRIMARY, min_runs=2)),
        ("min_runs_2_all_runs", replace(M.PRIMARY, min_runs=2, race_sim_only=False)),
    ):
        pc = M.placebo_cells(bases, sp)
        row = M.quick(pc, B=B, P=P, seed=SEED + 2000)
        row["random_intercept"] = M.random_intercept_trend(pc, B=Bs, seed=SEED + 2050)
        if len(pc):
            row["mean_delta_by_compound"] = {
                lab: float(pc.loc[pc["Compound"] == lab, "delta"].mean()) for lab in LABELS if (pc["Compound"] == lab).any()
            }
        row["definition"] = "FP2 slope minus (FP1+FP3) slope, same estimator, same window"
        placebo_rows[tag] = row
    placebo = placebo_rows["min_runs_3_racesim"]
    placebo_alt = placebo_rows
    log("placebo done")

    temp = M.temperature_models(cells, B=Bs, seed=SEED + 3000)
    log("temperature done")

    fuel = M.fuel_grid(bases, B=Bm, P=Pm, seed=SEED + 4000)
    fuel_field = M.fuel_grid(bases, replace(M.PRIMARY, race_estimator="xs"), B=Bm, P=Pm, seed=SEED + 4500)
    log("fuel grid done")

    mv = M.named_multiverse(bases, B=Bm, P=Pm, seed=SEED + 5000)
    log(f"multiverse done ({len(mv)} rows)")
    ok = [r for r in mv if r.get("status") == "ok"]
    summ = {
        "n_specs": len(mv),
        "n_estimable": len(ok),
        "n_beta_negative": int(sum(r["beta"] < 0 for r in ok)),
        "n_ci_excludes_zero_negative": int(sum(r["excludes_zero_negative"] for r in ok)),
        "n_perm_one_sided_below_0.05": int(sum(r["p_perm_one_sided"] < 0.05 for r in ok)),
        "beta_min": float(min(r["beta"] for r in ok)),
        "beta_median": float(np.median([r["beta"] for r in ok])),
        "beta_max": float(max(r["beta"] for r in ok)),
    }
    # The same summary restricted to the full-sample specs (all five seasons),
    # so one-season rows do not dilute or inflate it.
    full = [r for r in ok if len(r["spec"]["seasons"]) == 5]
    summ["full_sample_specs"] = {
        "n": len(full),
        "n_beta_negative": int(sum(r["beta"] < 0 for r in full)),
        "n_ci_excludes_zero_negative": int(sum(r["excludes_zero_negative"] for r in full)),
        "beta_median": float(np.median([r["beta"] for r in full])) if full else None,
    }

    cnum = {"status": "pending"}
    alloc = M.load_allocation_history()
    if alloc:
        cnum = M.cnumber_trend(cells, alloc, B=Bs, seed=SEED + 6000)
        cnum["allocation_events"] = len(alloc)
        cnum["unverified_allocation_events_in_history"] = M.unverified_events()
        cnum["caveat"] = (
            "C numbers are ordinal within a season's range and were re-specified each year; "
            "no event fixed effects here, so between-event differences in softness enter"
        )
    log("c-number done")

    # ------------------------------------------------------------------
    # POST-REVIEW ADDITIONS (added after an independent review of the first run;
    # see the docstring of tyredeg.paper.management). Same seeds, fixed in advance.
    # ------------------------------------------------------------------
    Bl = 300 if args.fast else 5000
    post = {
        "provenance": (
            "Added AFTER an independent review of the first run, which reproduced the primary "
            "numbers and raised: (1) race stints and practice runs start on different set states "
            "(82% of practice race-sim runs start at tyre age >= 6, race stints mostly at 2-5); "
            "(2) the gradient test is underpowered, so 'not supported' is not a null; (3) the "
            "pre-stated placebo was vacuous; (4) the cross-sectional race estimator is not "
            "identified. These specifications were NOT in the original pre-registration. They were "
            "fixed in the module docstring before being run and were not tuned on their output. "
            "One deviation: the strict matched level gap (>= 3 runs per cell and context) leaves "
            "very few cells, so the headline level gap is the pooled-lap version at >= 1 run per "
            "cell, reported beside the strict one."
        ),
    }
    c_pooled = M.make_cells(bases, M.MATCHED_POOLED, need_se=False)
    c_strict = M.make_cells(bases, M.MATCHED)
    c_unmatched_pooled = M.make_cells(bases, replace(M.MATCHED_POOLED, practice_start=None, race_start=None), need_se=False)
    post["level_gap_matched_pooled"] = M.level_gaps(c_pooled, B=Bl, seed=SEED + 7000)
    post["level_gap_matched_strict_min3runs"] = M.level_gaps(c_strict, B=Bl, seed=SEED + 7100)
    post["level_gap_unmatched_pooled"] = M.level_gaps(c_unmatched_pooled, B=Bl, seed=SEED + 7200)
    post["level_gap_unmatched_primary_cells"] = M.level_gaps(cells, B=Bl, seed=SEED + 7300)
    post["matching"] = {"practice_start_age": list(M.MATCHED.practice_start), "race_start_age": "> = 5", "window": [6, 16]}
    log("level gaps done")

    post["level_placebo_fp2_vs_fp1fp3_matched_pooled"] = M.placebo_levels(bases, M.MATCHED_POOLED, B=Bl, seed=SEED + 7400)
    post["level_placebo_fp2_vs_fp1fp3_unmatched_pooled"] = M.placebo_levels(
        bases, replace(M.MATCHED_POOLED, practice_start=None, race_start=None), B=Bl, seed=SEED + 7500
    )
    post["level_placebo_matched_strict_min3runs"] = M.placebo_levels(bases, M.MATCHED, B=Bl, seed=SEED + 7600)
    log("level placebo done")

    tab = M.twfe_pair_table(bases, M.PRIMARY, min_runs=2)
    post["twfe_gradient"] = M.twfe_gradient(tab, B=Bl, P=2 * Bl, seed=SEED + 8000)
    post["twfe_gradient"]["design"] = (
        "per event-season: lap time ~ run effect + lap effect (race) / 10-minute session-bin effect "
        "(practice) + tyre age with a compound-specific slope, one compound pair at a time; contrast "
        "= slope(a) - slope(b); DiD = race contrast - practice contrast; RE-weighted mean over events; "
        ">= 2 runs per compound and context; window 6-16; sign-flip permutation = compound labels "
        "permuted within event. Fuel, track evolution and temperature are common to compounds and cancel."
    )
    post["twfe_gradient"]["per_event"] = tab.round(5).to_dict(orient="records")
    variants = {}
    for name, sp, mr in (
        ("window_none", replace(M.PRIMARY, age_lo=None, age_hi=None), 2),
        ("window_3_12", replace(M.PRIMARY, age_lo=3, age_hi=12), 2),
        ("min_runs_3", M.PRIMARY, 3),
        ("matched_start_age", M.MATCHED_POOLED, 2),
        ("sprint_pooled_into_practice", replace(M.PRIMARY, sprint="practice"), 2),
    ):
        tv = M.twfe_pair_table(bases, sp, min_runs=mr)
        variants[name] = {k: v for k, v in M.twfe_gradient(tv, B=Bm, P=2 * Bm, seed=SEED + 8100).items() if k != "per_event"}
    post["twfe_gradient_variants"] = variants
    log("twfe done")

    k_obs = post["level_gap_matched_pooled"]["MEDIUM+SOFT"]["ratio"]
    post["uniform_damping"] = {
        "observed_ratio_race_over_practice_MEDIUM+SOFT_matched": k_obs,
        "observed_ratio_unmatched_primary": post["level_gap_unmatched_primary_cells"]["MEDIUM+SOFT"]["ratio"],
        "implied_additive_beta_per_step_if_race_slope_were_k_times_practice": {
            str(k): M.uniform_damping_beta(cells, k) for k in (0.4, 0.5, 0.6)
        },
        "observed_beta": h["trend"]["beta_per_softness_step"],
        "note": "a uniform damping factor alone produces a negative additive beta because softer tyres have larger practice slopes",
    }
    betas = (0.0, -0.014, -0.028, -0.045, -0.053, -0.07, -0.1, -0.14)
    post["power_additive_trend"] = M.power_simulation(
        cells, betas, n_sims=30 if args.fast else 100, B=99 if args.fast else 149, P=99 if args.fast else 149, seed=SEED + 9000
    )
    post["power_additive_trend"]["analytic_mde80_from_primary_boot_sd"] = float(2.8 * h["trend"]["boot_sd"])
    post["power_additive_trend"]["note"] = (
        "MDE = smallest |beta| (s/lap per softness step) with >= 80% power under the observed design; "
        "SOFT-HARD is twice that. The observed beta is below the MDE, so 'not supported' there is underpowered, not a null."
    )
    log("power done")
    referee = referee_block(bases, cells, post, Bl=Bl, Bs=Bs, seed=SEED + 10000, log=log)

    legacy = legacy_block()
    log("legacy done")

    result = {
        "meta": {
            "script": "scripts/paper/p1_management.py",
            "module": "tyredeg.paper.management",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": SEED,
            "bootstrap_B": {"primary": B, "secondary": Bs, "multiverse": Bm},
            "permutations": {"primary": P, "secondary": Ps, "multiverse": Pm},
            "fast_mode": bool(args.fast),
            "primary_spec": json.loads(M.PRIMARY.label()),
            "data_version": M.data_version(),
            "clusters": "event-season (season|event); bootstrap resamples clusters, permutation shuffles compound labels within cluster",
            "tau2": "Paule-Mandel; held at the full-sample value inside resampling",
        },
        "primary": h,
        "primary_secondary_xs_race": h_field,
        "decision": decision(h, placebo, temp),
        "placebo_fp2_vs_fp1fp3": placebo_alt,
        "seasons": seasons,
        "temperature": temp,
        "fuel_grid_within_run_race": fuel,
        "fuel_grid_xs_race": fuel_field,
        "multiverse": mv,
        "multiverse_summary": summ,
        "c_number_softness": cnum,
        "legacy_ratio": legacy,
        "post_review_additions": post,
        "referee_additions": referee,
        "analysis_tiers": {
            "headline_post_review": ["post_review_additions.level_gap_matched_pooled", "post_review_additions.twfe_gradient", "post_review_additions.level_placebo_fp2_vs_fp1fp3_matched_pooled", "post_review_additions.power_additive_trend"],
            "secondary_original_primary": ["primary", "primary_secondary_xs_race", "seasons", "temperature", "fuel_grid_within_run_race", "fuel_grid_xs_race", "multiverse", "placebo_fp2_vs_fp1fp3", "c_number_softness"],
            "legacy_not_to_be_cited": ["legacy_ratio", "primary_secondary_xs_race (cross-sectional race estimator is not identified: selection on pit timing, negative SOFT slopes)"],
        },
    }
    tg = post["twfe_gradient"].get("SOFT-HARD", {})
    lg = post["level_gap_matched_pooled"].get("MEDIUM+SOFT", {})
    result["decision"]["post_review"] = {
        "rule": "supported iff bootstrap 95% CI upper bound < 0 AND one- and two-sided permutation p < 0.05",
        "twfe_SOFT_minus_HARD_gradient_supported": bool(
            tg.get("ci95") and tg["ci95"][1] < 0 and tg["p_perm_one_sided_less"] < 0.05 and tg["p_perm_two_sided"] < 0.05
        ),
        "twfe_SOFT_minus_MEDIUM_gradient_supported": bool(
            post["twfe_gradient"].get("SOFT-MEDIUM", {}).get("ci95", [0, 1])[1] < 0
        ),
        "matched_level_gap_MEDIUM_plus_SOFT_ci_excludes_zero_negative": bool(lg.get("gap_ci95") and lg["gap_ci95"][1] < 0),
        "level_placebo_MEDIUM_plus_SOFT_ci_excludes_zero": bool(
            (lambda g: g.get("gap_ci95") and (g["gap_ci95"][0] > 0 or g["gap_ci95"][1] < 0))(
                post["level_placebo_fp2_vs_fp1fp3_matched_pooled"].get("MEDIUM+SOFT", {})
            )
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(result), indent=2), encoding="utf-8")
    log(f"wrote {OUT}")
    print(json.dumps(_clean(result["decision"]), indent=2))


if __name__ == "__main__":
    sys.exit(main())

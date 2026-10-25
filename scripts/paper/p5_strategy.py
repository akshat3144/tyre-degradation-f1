"""Paper P5: does the strategy layer's plan look like what the field ran?

    python scripts/paper/p5_strategy.py        # ~5 minutes, writes paper/results/strategy.json
    python scripts/paper/p5_strategy.py --rebuild   # re-read the races from FastF1's cache first

WHAT WAS WRONG WITH THE SHIPPED CLAIM
    "We agree with the field's median stop count at 10 of 11 events." The rule
    "race of 60 laps or more -> 2 stops, else 1" scores 11 of 11 on the same
    events and "always one stop" scores 6 of 11; the rates behind the call were
    fitted on the very race being scored; and a stop count is the coarsest thing
    a plan says. A hostile audit also found the recommended stint lengths
    sitting on the five-lap minimum (C4 x5 + C5 x73 at Monaco).

WHAT THIS DOES
    Scores recommended PLANS -- stop count, stint lengths, compound usage -- against the
    stints of the classified top ten, over every dry, un-red-flagged race in the
    cache: 2026 (physical C-number compounds, 10 races) and 2022-2025
    (HARD/MEDIUM/SOFT labels, 73 races). Rates are out of sample (see
    ``tyredeg.paper.strategy_eval``); baselines are fitted without the race being
    scored. Then tests the stint-length support bounds that were added to fix the
    boundary pathology, sweeps the assumed pace step, and tries to estimate it.

PRE-STATED RULES (written before the bounded results were read)
    * Headline set: dry races with no red flag.
    * Primary metrics: stop-count match with the field mode; Wasserstein-1
      distance between tyre-age distributions (laps) to the median top-ten car;
      longest-stint error (laps) against the field's median longest stint.
    * The stint-length bounds (5th-95th percentile of stints the field ran on
      that compound label in OTHER races) are ADOPTED as the product default only
      if, paired on the same races and rates, they do not worsen stop-count
      agreement and they lower the age-distance. They are reported either way.
    * A model method "beats" a baseline on a metric only if the 95% race-level
      bootstrap interval of the paired difference excludes zero.

LIMITS THE PAPER MUST STATE
    * The field optimises for position, traffic and safety-car luck; the model
      for total time. A distance is not an error.
    * 2022-2025 compounds are relative labels, so rates there are pooled across
      physically different tyres.
    * Cells that are not identified are refused and the race is not planned; the
      number planned is reported per method, and comparisons are on common races.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd

from tyredeg.config import PROCESSED, ROOT
from tyredeg.data.allocation import compounds_for
from tyredeg.models.design import prepare
from tyredeg.models.mixed import fit_degradation
from tyredeg.paper import strategy_eval as E
from tyredeg.paper import strategy_pace as PA
from tyredeg.paper import transfer_eval as TE
from tyredeg.paper.strategy_meta import meta
from tyredeg.strategy import racedata
from tyredeg.strategy import support as SUP
from tyredeg.strategy.optimise import C_ORDER, LABEL_ORDER, PACE_STEP_S, PACE_STEP_SD_S
from tyredeg.strategy.rates import resolve_rates

warnings.filterwarnings("ignore")

OUT = ROOT / "paper" / "results" / "strategy.json"

#: The planner variants the headline reports.
UNBOUNDED, CEILING, SUPPORT = "plugin", "plugin+cap", "plugin+support"

#: Rate sources whose plans are scored against baselines.
HEADLINE = ("practice_scalar", "race_pooled", "blend", "in_sample", "product_in_sample", "mixed_loeo")


def _label_fns():
    lab26 = lambda ev, c: TE.label_of_2026(ev, c)  # noqa: E731
    labold = lambda ev, c: c  # noqa: E731
    return lab26, labold


def build_groups():
    sd26 = TE.build_season(2026)
    parts = [TE.build_season(s) for s in (2022, 2023, 2024, 2025)]
    pr_old = pd.concat([p.practice for p in parts], ignore_index=True)
    rc_old = pd.concat([p.race for p in parts], ignore_index=True)
    lab26, labold = _label_fns()
    groups = {
        "2026": (E.build_cells(sd26.practice, sd26.race, lab26), lab26, C_ORDER),
        "2022-25": (E.build_cells(pr_old, rc_old, labold), labold, LABEL_ORDER),
    }
    frames = {"2026": (sd26.practice, sd26.race), "2022-25": (pr_old, rc_old)}
    return groups, frames


def product_inputs(events: list[str]) -> dict[str, dict[str, E.RateInput]]:
    """What the product itself would plan a 2026 race from: its mixed-effects fit,
    gated, with the practice fall-back. In sample: the fit saw the race."""
    laps = pd.read_parquet(PROCESSED / "laps.parquet")
    race = prepare(laps, "race")
    practice = prepare(laps, "practice")
    fit = fit_degradation(race, quadratic=False, context="race", circuit_effects=True)
    out: dict[str, dict[str, E.RateInput]] = {}
    for ev in events:
        alloc = compounds_for(ev)
        if not alloc:
            continue
        rs = resolve_rates(fit, ev, set(alloc.values()), practice=practice, race=race)
        out[ev] = {
            "product_in_sample": E.RateInput(
                {r.compound: r.mean for r in rs.rows if not r.excluded},
                {r.compound: r.sd for r in rs.rows if not r.excluded},
            )
        }
    return out, fit, race


def mixed_loeo(events: list[str], race: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    """The product's mixed-effects race model refitted without each event in turn.

    A held-out circuit has no fitted slope, so it gets the pooled compound rate --
    the honest out-of-sample counterpart of the per-circuit rate the product shows.
    """
    out = {}
    for ev in events:
        fit = fit_degradation(race[race["event"] != ev], quadratic=False, context="race",
                              circuit_effects=True)
        out[ev] = {c: (iv.mean, max((iv.hi - iv.lo) / 3.92, 0.0)) for c, iv in fit.rates.items()}
    return out


def shipped_claim_check(races: list[E.FieldRace], df: pd.DataFrame) -> dict:
    """Re-score the original claim, and the two trivial rules, on its own terms.

    The original compared the recommended stop count with the MEDIAN stop count of
    every car that stopped at least once, at each planned 2026 event. Here: the
    product's in-sample plan on those events, always-1, and the best possible
    race-length threshold (fitted on the events themselves, so an upper bound).
    """
    laps, _ = racedata.load()
    sub = df[(df["method"] == f"product_in_sample/{SUPPORT}") & (df["season"] == 2026) & df["planned"].eq(True)]
    sub_nb = df[(df["method"] == f"product_in_sample/{UNBOUNDED}") & (df["season"] == 2026) & df["planned"].eq(True)]
    rows = []
    for ev in sorted(set(sub["event"]) | set(sub_nb["event"])):
        g = laps[(laps["season"] == 2026) & (laps["event"] == ev) & laps["Stint"].notna()]
        per = g.groupby("Driver")["Stint"].nunique() - 1
        ran = per[per >= 1]
        if ran.empty:
            continue
        rows.append((ev, int(g["LapNumber"].max()), int(ran.median())))
    if not rows:
        return {}
    med = {ev: m for ev, _, m in rows}
    n = len(rows)

    def hits(frame):
        f = frame.set_index("event")["stops"]
        return int(sum(int(f.get(ev, -1)) == med[ev] for ev in med if ev in f.index)), int(sum(ev in f.index for ev in med))

    best = (0, 0)
    for t in sorted({L for _, L, _ in rows}):
        h = sum((2 if L >= t else 1) == m for _, L, m in rows)
        if h > best[0]:
            best = (h, t)
    hb, nb = hits(sub)
    hu, nu = hits(sub_nb)
    return {
        "events_scored": n,
        "definition": "median stop count of every 2026 car that stopped at least once",
        "product_in_sample_with_bounds": {"matches": hb, "of": nb},
        "product_in_sample_unbounded": {"matches": hu, "of": nu},
        "always_one_stop": {"matches": int(sum(m == 1 for _, _, m in rows)), "of": n},
        "race_length_rule_best_threshold_fitted_on_these_events": {
            "matches": int(best[0]), "of": n, "threshold_laps": int(best[1]),
            "caveat": "threshold chosen on the scored events: an upper bound for the rule",
        },
    }


def paired(df: pd.DataFrame, a: str, b: str, metrics: list[str], group: str | None = None) -> dict:
    out = {}
    for m in metrics:
        r = E.paired_diff(df, a, b, m, group)
        if r:
            out[m] = r
    return out


def sensitivity(races, groups, pits, extra, mixed, pool=None) -> dict:
    """How plans and stop counts move with the assumed pace step."""
    base = None
    rows = {}
    for s in E.STEP_GRID:
        d = E.evaluate(
            races, groups, pits, mixed_rates=mixed, robust=False, step_s=s,
            planners=(UNBOUNDED, SUPPORT), methods=("practice_scalar", "in_sample", "product_in_sample"),
            extra_inputs=extra, stint_pool=pool,
        )
        rows[s] = d
        if abs(s - PACE_STEP_S) < 1e-9:
            base = d
    out = {}
    for method in ("practice_scalar/plugin+support", "in_sample/plugin+support", "product_in_sample/plugin+support",
                   "practice_scalar/plugin", "product_in_sample/plugin"):
        ref = E.headline(base)
        ref = ref[(ref["method"] == method) & ref["stops"].notna()].set_index("race")["stops"]
        per = []
        for s, d in rows.items():
            d = E.headline(d)
            d = d[(d["method"] == method) & d["stops"].notna()]
            if d.empty:
                continue
            same = d.set_index("race")["stops"].reindex(ref.index).dropna()
            per.append(
                {
                    "pace_step_s": s,
                    "n_planned": int(len(d)),
                    "mean_stops": round(float(d["stops"].mean()), 2),
                    "share_stop_count_differs_from_0.6": round(float((same != ref.reindex(same.index)).mean()), 3),
                    "stop_match_field_mode": round(float(d["stop_match_mode"].mean()), 3),
                    "w1_median_car": round(float(d["w1_median_car"].mean()), 3),
                    "mean_longest_stint": round(float(d["longest_stint"].mean()), 1),
                    "mean_shortest_stint": round(float(d["shortest_stint"].mean()), 1),
                    "share_on_boundary": round(float(d["on_boundary"].mean()), 3),
                }
            )
        out[method] = per
    return out


def pace_step_study(frames) -> dict:
    out = {"assumed_s": PACE_STEP_S, "assumed_sd_s": PACE_STEP_SD_S, "sign": "positive = softer is FASTER by that many seconds"}
    for name, order in (("2026", C_ORDER), ("2022-25", LABEL_ORDER)):
        pr, rc = frames[name]
        race = PA.race_estimate(rc, order)
        prac = PA.practice_estimate(pr, order)
        for r in (race, prac):
            if r.get("estimable"):
                # estimators return the coefficient on the compound index, so a
                # negative coefficient is "softer is faster"; flip it so the sign
                # reads the way the assumption does.
                b = r.pop("step_s_per_compound_step")
                r["softer_faster_by_s_per_step"] = round(-b, 3)
                r["se"] = round(r["se"], 3)
                r["ci95_softer_faster_by"] = [round(-r["ci95"][1], 3), round(-r["ci95"][0], 3)]
                r.pop("ci95")
                r.pop("sign_convention", None)
                r["coefficients"] = {k: round(v, 4) for k, v in r.get("coefficients", {}).items()}
        out[name] = {"race_within_driver_lap": race, "practice": prac}
    table = []
    for g in ("2026", "2022-25"):
        for kind in ("race_within_driver_lap", "practice"):
            e = out[g][kind]
            if not e.get("estimable"):
                table.append({"season": g, "estimator": kind, "estimable": False})
                continue
            lo, hi = e["ci95_softer_faster_by"]
            table.append(
                {
                    "season": g, "estimator": kind, "estimable": True,
                    "softer_faster_by_s_per_step": e["softer_faster_by_s_per_step"],
                    "ci95": [lo, hi], "se": e["se"],
                    "ci_contains_assumed_0.6": bool(lo <= PACE_STEP_S <= hi),
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                }
            )
    ok = [t for t in table if t["estimable"]]
    race_signs = {np.sign(t["softer_faster_by_s_per_step"]) for t in ok if t["estimator"] == "race_within_driver_lap"}
    tight = [t for t in ok if t["se"] < 0.1]
    out["verdict"] = {
        "estimates": table,
        "race_estimates_agree_in_sign_across_seasons": len(race_signs) == 1,
        "tight_estimates_(se<0.1)_containing_0.6": [t for t in tight if t["ci_contains_assumed_0.6"]],
        "reading": "The pace step is NOT estimated. The race estimators are tight but disagree in sign "
        "between 2026 and 2022-25 and neither is near 0.6; the practice estimators contain 0.6 but "
        "their intervals are too wide to exclude 0 (2026) or span 0.3-1.2 s (2022-25). It stays an "
        "assumption, and the sensitivity sweep is the honest way to read every plan that depends on it.",
    }
    return out


def per_race_2026(df: pd.DataFrame, races: list[E.FieldRace]) -> list[dict]:
    keep = ("practice_scalar/plugin+support", "product_in_sample/plugin+support", "product_in_sample/plugin",
            "in_sample/plugin+support", "lap_rule", "always_1")
    out = []
    for r in races:
        if r.season != 2026 or r.wet:
            continue
        sub = df[(df["race"] == r.key)]
        row = {
            "event": r.event,
            "race_laps": r.race_laps,
            "red_flag": r.red_flag,
            "field_stop_counts": {str(k): int(v) for k, v in sorted(pd.Series(r.stops).value_counts().items())},
            "field_median_longest_stint": float(np.median([max(n for _, n in c) for c in r.cars])),
            "field_median_shortest_stint": float(np.median([min(n for _, n in c) for c in r.cars])),
            "pit_loss_used_s": round(float(sub["pit_s"].iloc[0]), 2) if len(sub) else None,
            "pit_loss_source": str(sub["pit_source"].iloc[0]) if len(sub) else None,
            "recommended": {},
        }
        for m in keep:
            x = sub[sub["method"] == m]
            if x.empty:
                continue
            x = x.iloc[0]
            if "planned" in x and x["planned"] is False:
                row["recommended"][m] = "not planned (fewer than two identified compounds)"
            elif pd.notna(x.get("stops")):
                row["recommended"][m] = {
                    "stops": int(x["stops"]),
                    "plan": x.get("plan") if isinstance(x.get("plan"), list) else None,
                    "w1_median_car": round(float(x["w1_median_car"]), 2),
                }
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    laps, results = racedata.load(rebuild=args.rebuild)
    races = E.field_races(laps, results)
    pits = E.pit_loss_table(laps)
    groups, frames = build_groups()
    events26 = sorted({r.event for r in races if r.season == 2026})
    extra, fit, race26 = product_inputs(events26)
    mixed = mixed_loeo(events26, race26)
    print(f"inputs built in {time.time() - t0:.0f}s: {len(races)} races, {len(pits)} pit-loss estimates", flush=True)

    # The genuine-stint pool the product's bounds are built from, out of the RAW
    # laps. The headline is OUT OF TIME: pit loss, lap-rule threshold and stint
    # bounds come only from races run before the one scored. The leave-one-race-out
    # form (which lets a 2022 race see 2025's stints and pit lane) is run beside it.
    pool = SUP.observed_stints(laps)
    df = E.evaluate(races, groups, pits, mixed_rates=mixed, extra_inputs=extra, stint_pool=pool)
    print(f"evaluated {len(df)} rows (out of time) in {time.time() - t0:.0f}s", flush=True)
    df_loo = E.evaluate(races, groups, pits, mixed_rates=mixed, extra_inputs=extra, stint_pool=pool,
                        out_of_time=False)
    print(f"evaluated {len(df_loo)} rows (leave-one-race-out) in {time.time() - t0:.0f}s", flush=True)

    # --- headline tables ---------------------------------------------------
    summary = {g: E.summarise(df, g) for g in ("2026", "2022-25", None)}
    summary = {("all" if k is None else k): v for k, v in summary.items()}

    # --- paired comparisons, each on the races both planned ---------------
    metrics = ["stop_match_mode", "stop_abs_err_median", "w1_median_car", "longest_err"]
    vs_baseline, bound_effect = {}, {}
    loo_summary = {
        ("all" if g is None else g): E.summarise(df_loo, g) for g in ("2026", "2022-25", None)
    }
    loo_bound_effect = {
        src: {"support_minus_unbounded": paired(df_loo, f"{src}/{SUPPORT}", f"{src}/{UNBOUNDED}",
                                                metrics + ["on_boundary"])}
        for src in HEADLINE if f"{src}/{UNBOUNDED}" in set(df_loo["method"])
    }
    loo_vs_baseline = {
        f"{src}/{SUPPORT}": {b: paired(df_loo, f"{src}/{SUPPORT}", b, metrics)
                             for b in ("lap_rule", "always_1", "prev_season")}
        for src in HEADLINE if f"{src}/{SUPPORT}" in set(df_loo["method"])
    }
    for src in HEADLINE:
        for planner in (UNBOUNDED, SUPPORT):
            name = f"{src}/{planner}"
            if name not in set(df["method"]):
                continue
            vs_baseline[name] = {
                b: paired(df, name, b, metrics)
                for b in ("lap_rule", "always_1", "prev_season")
            }
        if f"{src}/{UNBOUNDED}" in set(df["method"]):
            bound_effect[src] = {
                "support_minus_unbounded": paired(df, f"{src}/{SUPPORT}", f"{src}/{UNBOUNDED}", metrics + ["on_boundary"]),
                "ceiling_only_minus_unbounded": paired(df, f"{src}/{CEILING}", f"{src}/{UNBOUNDED}", metrics + ["on_boundary"]),
                "robust_minus_plugin_unbounded": paired(df, f"{src}/robust", f"{src}/{UNBOUNDED}", metrics),
                "robust_minus_plugin_ceiling_only": paired(df, f"{src}/robust+cap", f"{src}/{CEILING}", metrics),
            }

    # --- decision rule for adopting the bounds, applied mechanically -------
    adopt = {}
    for src in ("practice_scalar", "race_pooled", "blend", "in_sample", "product_in_sample"):
        b = bound_effect.get(src, {}).get("support_minus_unbounded", {})
        sm = b.get("stop_match_mode")
        wd = b.get("w1_median_car")
        if sm and wd:
            adopt[src] = {
                "n_races": sm["n_races"],
                "stop_match_change": sm["mean_a_minus_b"],
                "stop_match_change_ci95": sm["ci95"],
                "age_distance_change_laps": wd["mean_a_minus_b"],
                "age_distance_change_ci95": wd["ci95"],
                "stop_match_not_worse": bool(sm["ci95"][0] <= 0 <= sm["ci95"][1] or sm["mean_a_minus_b"] >= 0),
                "age_distance_lower": bool(wd["mean_a_minus_b"] < 0),
                "age_distance_lower_ci_excludes_zero": bool(wd["ci95"][1] < 0),
            }
    adopted = bool(adopt) and all(v["age_distance_lower"] and v["stop_match_not_worse"] for v in adopt.values())

    # --- the pathology, quantified -----------------------------------------
    h = E.headline(df)
    path = {}
    for m in ("in_sample/plugin", "in_sample/plugin+support", "practice_scalar/plugin", "practice_scalar/plugin+support",
              "product_in_sample/plugin", "product_in_sample/plugin+support"):
        g = h[(h["method"] == m) & h["stops"].notna()]
        if g.empty:
            continue
        path[m] = {
            "n": int(len(g)),
            "share_with_stint_within_1_lap_of_minimum": round(float(g["on_boundary"].mean()), 3),
            "share_with_a_stint_on_a_support_bound": round(float(g["on_support_bound"].mean()), 3),
            "share_at_the_max_stops_limit": round(float(g["at_stop_limit"].mean()), 3),
            "mean_longest_stint": round(float(g["longest_stint"].mean()), 1),
            "field_mean_longest_stint": round(float(g["field_longest"].mean()), 1),
            "mean_shortest_stint": round(float(g["shortest_stint"].mean()), 1),
            "field_mean_shortest_stint": round(float(g["field_shortest"].mean()), 1),
            "longest_stint_abs_error_vs_field": round(float(g["longest_err"].mean()), 2),
        }

    sweep = sensitivity(races, groups, pits, extra, mixed, pool)
    pace = pace_step_study(frames)
    claim = shipped_claim_check(races, df)

    n_races = {
        "all_in_cache": len(races),
        "wet_excluded": sum(r.wet for r in races),
        "red_flag_excluded_from_headline": sum(r.red_flag and not r.wet for r in races),
        "headline_2026": int(h[h["group"] == "2026"]["race"].nunique()),
        "headline_2022_25": int(h[h["group"] == "2022-25"]["race"].nunique()),
    }

    result = {
        "meta": meta(
            "scripts/paper/p5_strategy.py",
            "src/tyredeg/paper/strategy_eval.py",
            E.SEED,
            {
                "top_n_cars": 10,
                "support_quantiles": [E.FLOOR_QUANTILE, E.SUPPORT_QUANTILE],
                "pace_step_s": PACE_STEP_S,
                "step_grid": list(E.STEP_GRID),
                "n_boot": E.N_BOOT,
                "min_stint_laps": E.MIN_STINT_LAPS,
                "floor_margin_laps": SUP.FLOOR_MARGIN_LAPS,
                "headline_is_out_of_time": True,
                "wet_share_threshold": E.WET_SHARE,
            },
            [racedata.LAPS_PATH, racedata.RESULTS_PATH, PROCESSED / "laps.parquet"],
        ),
        "sample": n_races,
        "metric_definitions": {
            "stop_match_mode": "1 if the recommended stop count equals the most common stop count of the top ten",
            "stop_abs_err_median": "|recommended stops - median stops of the top ten|",
            "w1_median_car": "Wasserstein-1 distance (laps) between the tyre-age distribution of the recommended plan and "
            "that of each top-ten car's plan; the median over cars",
            "w1_nearest_car": "the same, minimum over cars: is there a car that ran something like it",
            "longest_err": "|longest recommended stint - median longest stint of the top ten| (laps)",
            "on_boundary": "1 if the shortest recommended stint is within one lap of the five-lap minimum",
            "usage_tv_median_car": "total-variation distance between compound-usage shares (labels), median over cars",
        },
        "baselines": {
            "always_1": "one stop, equal stints",
            "lap_rule": "2 stops if race_laps >= T else 1, T fitted on the OTHER races, equal stints",
            "prev_season": "the stop count the field modally ran at the same circuit the previous season, equal stints",
        },
        "rate_sources": {
            "in_sample": "the race's own cell rates (NOT out of sample)",
            "product_in_sample": "the product's mixed-effects per-circuit rate, gated, practice fallback (NOT out of sample)",
            "race_pooled": "median race rate per compound over the other races",
            "practice_scalar": "this weekend's practice x factor learned on the other weekends",
            "practice_label": "the same with a factor per HARD/MEDIUM/SOFT label",
            "practice_raw": "this weekend's practice rate, no race correction (management-free wear)",
            "blend": "inverse-variance combination of practice_label and race_pooled",
            "mixed_loeo": "the product's mixed model refitted without the scored race (2026 only)",
        },
        "planner_variants": {
            "plugin": "minimise total time on the point rates, stints bounded only by the five-lap minimum",
            "robust": "minimise expected total time with each rate replaced by its mean truncated at zero",
            "+cap": "plus a ceiling on each compound's stint: the 95th percentile of stints the field ran on that label in other races",
            "+support": "ceiling and a floor (5th percentile)",
        },
        "out_of_time": {
            "headline": "pit loss, lap-rule threshold and stint bounds are built only from races run BEFORE the one "
            "scored (the first races have no bounds: fewer than 30 earlier stints of a label). prev_season was "
            "already earlier-only.",
            "still_leave_one_race_out": "the degradation-rate sources (race_pooled, the practice factors, the "
            "mixed model refit) use the other races on either side of the scored one",
        },
        "leave_one_race_out_sensitivity": {
            "note": "the same evaluation with pit loss, lap rule and stint bounds fitted on every OTHER race, "
            "including later ones (the form this module reported before it was made out-of-time)",
            "summary_by_group": loo_summary,
            "effect_of_stint_bounds": loo_bound_effect,
            "paired_vs_baselines": loo_vs_baseline,
        },
        "summary_by_group": summary,
        "paired_vs_baselines": vs_baseline,
        "effect_of_stint_bounds": bound_effect,
        "adoption_rule_applied": {
            "rule": "adopt if, for every rate source, stop-count agreement is not significantly worse (the paired 95% "
            "interval includes zero or the change is non-negative) and the mean age-distance falls",
            "per_rate_source": adopt,
            "adopted_as_default": adopted,
            "honest_note": "On the product's own 2026 path (10 races) the point estimate of stop-count agreement FELL "
            "(see product_in_sample): the interval includes zero, and on the 2022-25 sources agreement is flat "
            "or higher, but it is a fall that ten races cannot rule out.",
        },
        "boundary_pathology": path,
        "shipped_claim_rechecked": claim,
        "pace_step_sensitivity": sweep,
        "pace_step_estimation": pace,
        "per_race_2026": per_race_2026(df, races),
        "n_planned_note": "n_planned is how many races a method could plan after refusing unidentified compounds; "
        "compare methods through paired_vs_baselines, which uses only races both planned.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote {OUT} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

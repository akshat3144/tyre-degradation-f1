"""Write the manuscript tables to paper/manuscript/generated/tab_*.tex.

    python scripts/paper/make_tables.py

Each file is a bare booktabs ``tabular`` (no vertical rules, no float, no
caption): ``\\begin{table}\\centering\\small\\caption{..}\\input{generated/tab_x}\\end{table}``.
Every number is read from the JSON results; rounding follows CLAIMS.md
(rates 4 places, differences and slopes 3-4 places as in the claim). Requires
only booktabs.

  tab_data          T1  data summary by season and by 2026 session type
  tab_rates         T2  compound degradation rates with 95% CI + circuit SD
  tab_race_practice T4  race vs practice (used sets), robustness designs, placebo, TWFE contrasts (appendix)
  tab_forecast      T5  practice-to-race forecast ladder, out of season
  tab_benchmark     T6  head-to-head with Cappello & Hoegh (journal Table 4)
  tab_repro         T6b reproduction of their Table 3 and the fuel correction
  tab_strategy      T7  strategy evaluation against baselines, out of time
  tab_seasons       T8  circuit-slope SD and within-event compound trend by season
  tab_inference     T9  interval widths under alternative inference (laps, runs, clusters)
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paper_common import (  # noqa: E402
    BENCH,
    COMPOUNDS,
    GENERATED,
    MGMT,
    MODEL,
    RECOV,
    STRAT,
    TRANS,
    ci,
    fmt_num,
    get,
    tabular,
    write_text,
)

DASH = "--"
SEASONS = ["2022", "2023", "2024", "2025", "2026"]


def est_ci(est, lo, hi, places, signed=False):
    """``est [lo, hi]``; the explicit plus sign goes on the estimate only (bounds keep just their minus)."""
    return f"{fmt_num(est, places, signed)} {ci(lo, hi, places)}"


def n(x):
    return fmt_num(x, 0)


def pval(p):
    """Four places below 0.01 (0.0074), three otherwise (0.302), as CLAIMS.md quotes them."""
    return fmt_num(p, 4 if p < 0.01 else 3)


# ------------------------------------------------------------------ T1
def t1_data():
    """Events / sessions, laps and runs. Season detail comes from model.json
    (season_robustness); the 2026 session breakdown from model.json and
    data/artifacts/meta.json. All JSON, so it rebuilds without the parquet."""
    rows = [("panel", "Race fit by season (long-run race laps)")]
    for y in SEASONS:
        sr = get(MODEL, "season_robustness", y)
        rows.append([y, n(sr["n_events"]), n(sr["n_laps"]), n(sr["n_runs"])])
    rows.append(r"\midrule")
    rows.append(("panel", "2026 clean laps by session type"))
    by_ev = get(MODEL, "dataset", "laps_by_event_session")
    by_sess = get(MODEL, "dataset", "laps_by_session")
    names = {"FP1": "Practice 1", "FP2": "Practice 2", "FP3": "Practice 3", "S": "Sprint", "R": "Race"}
    for s in ["FP1", "FP2", "FP3", "S", "R"]:
        n_sessions = sum(1 for e in by_ev.values() if s in e)
        rows.append([names[s], n(n_sessions), n(by_sess[s]), ""])
    rows.append([r"All sessions", n(get(MODEL, "dataset", "n_events_in_file")), n(get(MODEL, "dataset", "n_laps_total")), ""])
    rows.append(r"\midrule")
    rows.append(("panel", "2026 laps entering the degradation fits"))
    rf = get(MODEL, "dataset", "race_fit")
    pdeg = get("data/artifacts/meta.json", "fit_scope", "practice_degradation")
    rows.append(["Race, long runs", n(rf["n_events"]), n(rf["n_laps"]), n(rf["n_runs"])])
    rows.append(["Practice and sprint, race-simulation runs", n(pdeg["n_events"]), n(pdeg["n_laps"]), n(pdeg["n_runs"])])
    return tabular("@{}lrrr@{}", ["", "Events or sessions", "Laps", "Runs"], rows)


# ------------------------------------------------------------------ T2
def t2_rates():
    rates = get(MODEL, "shipped_fit", "rates_s_per_lap")
    rows = []
    for c in COMPOUNDS:
        r = rates[c]
        rows.append([c, n(r["n_runs"]), n(r["n_laps"]), fmt_num(r["mean"], 4), ci(r["lo"], r["hi"], 4)])
    rows.append(r"\midrule")
    cs = get(MODEL, "circuit_effect", "circuit_slope_sd_s_per_lap")
    sd_comp = statistics.stdev([rates[c]["mean"] for c in COMPOUNDS])
    rows.append(
        [
            f"2026 circuit-slope SD ({n(get(MODEL, 'dataset', 'race_fit', 'n_events'))} circuits, one race each)",
            "",
            "",
            fmt_num(cs["estimate"], 4),
            ci(*cs["profile_likelihood_95"], 4),
        ]
    )
    rows.append(["SD of the five compound rates", "", "", fmt_num(sd_comp, 4), ""])
    we = get(MODEL, "compound_within_event", "within_event_fixed_circuit", "trend_s_per_lap_per_step_softer")
    rows.append(
        ["Within-event trend per step softer", "", "", fmt_num(we["mean"], 4, True), ci(we["lo"], we["hi"], 4, True)]
    )
    return tabular("@{}lrrrr@{}", ["Compound", "Runs", "Laps", r"Rate (s/lap)", "95\\% interval"], rows)


# ------------------------------------------------------------------ T4
def _gap_row(label, blk, key):
    b = blk[key]
    return [
        label,
        n(b["n"]),
        fmt_num(b["race"], 3),
        fmt_num(b["practice"], 3),
        est_ci(b["gap"], *b["gap_ci95"], 3, True),
        est_ci(b["ratio"], *b["ratio_ci95"], 2),
        "",
    ]


def t4_race_practice():
    ref = ["referee_additions", "designs"]
    ud = get(MGMT, *ref, "used_sets_only_FreshTyre_false")
    sa = get(MGMT, *ref, "start_age_proxy_matched")
    ix = get(MGMT, *ref, "used_sets_and_start_age_matched")
    um = get(MGMT, "post_review_additions", "level_gap_unmatched_pooled")
    lg = ud["level_gap"]
    pl = ud["level_placebo_fp2_minus_fp1fp3"]
    pa = ud["placebo_adjusted_gap"]["MEDIUM+SOFT"]
    tw = get(MGMT, "post_review_additions", "twfe_gradient")
    holm = get(MGMT, "referee_additions", "twfe_primary_holm", "holm_adjusted_p_two_sided")
    rows = [("panel", rf"Used sets in both contexts (primary): {n(lg['n_cells'])} groups, {n(lg['n_event_seasons'])} event-seasons")]
    for lab, k in (("Hard", "HARD"), ("Medium", "MEDIUM"), ("Soft", "SOFT"), ("Medium + soft", "MEDIUM+SOFT"), ("All", "ALL")):
        rows.append(_gap_row(lab, lg, k))
    rows.append(r"\midrule")
    rows.append(("panel", "Other designs, medium + soft"))
    rows.append(_gap_row("Start-age matched", sa["level_gap"], "MEDIUM+SOFT"))
    rows.append(_gap_row("Used sets, start-age matched", ix["level_gap"], "MEDIUM+SOFT"))
    rows.append(_gap_row("No matching", um, "MEDIUM+SOFT"))
    rows.append(r"\midrule")
    rows.append(("panel", "Placebo: practice 2 (first column) against practice 1 and 3, used sets"))
    rows.append(_gap_row("Medium + soft", pl, "MEDIUM+SOFT"))
    rows.append(_gap_row("All", pl, "ALL"))
    rows.append(
        [
            "Gap minus placebo gap",
            "",
            "",
            "",
            est_ci(pa["placebo_adjusted_gap"], *pa["placebo_adjusted_gap_ci95_joint_bootstrap"], 3, True),
            "",
            "",
        ]
    )
    rows.append(r"\midrule")
    rows.append(("panel", "Compound slope contrast per context, and difference in differences, per event"))
    for lab, k in (("Soft $-$ hard", "SOFT-HARD"), ("Soft $-$ medium", "SOFT-MEDIUM"), ("Medium $-$ hard", "MEDIUM-HARD")):
        b = tw[k]
        rows.append(
            [
                lab,
                n(b["n_events"]),
                fmt_num(b["race_contrast_mean"], 3, True),
                fmt_num(b["practice_contrast_mean"], 3, True),
                est_ci(b["estimate_s_per_lap"], *b["ci95"], 3, True),
                "",
                f"{fmt_num(b['p_perm_two_sided'], 3)} ({fmt_num(holm[k], 3)})",
            ]
        )
    header = ["", "$n$", "Race", "Practice", r"Difference [95\% CI]", r"Ratio [95\% CI]", r"$p$ (Holm)"]
    return tabular("@{}lrrrllr@{}", header, rows)


# ------------------------------------------------------------------ T5
LADDER = [
    ("panel", "Adopted"),
    ("scalar", "Scaled practice forecast"),
    ("panel", "Baselines"),
    ("naive", "Naive (Sunday = Friday)"),
    ("zero", "Zero"),
    ("pooled", "Pooled race rate"),
    ("const", "Constant (other weekends' median)"),
    ("history", "Previous season, same circuit"),
    ("panel", "Tried, not adopted"),
    ("compound", "Compound-specific factor"),
    ("shrink50", r"50\% shrink toward constant"),
    ("stack", "Stack of baselines"),
]


def t5_forecast():
    hf = get(TRANS, "holdout_frozen")
    sb = hf["scoreboard"]
    sv = hf["scalar_vs"]
    s26 = get(TRANS, "s2026_loeo", "scoreboard")
    rows = []
    for key, label in LADDER:
        if key == "panel":
            rows.append(("panel", label))
            continue
        b = sb[key]
        if key == "scalar":
            diff = DASH
        else:
            d = sv[key]  # mean_diff is scalar minus method; report method minus scalar
            diff = est_ci(-d["mean_diff"], -d["ci_hi"], -d["ci_lo"], 4, True)
        rows.append([label, fmt_num(b["mae"], 4), ci(b["mae_ci_lo"], b["mae_ci_hi"], 4), diff, fmt_num(s26[key]["mae"], 4)])
    header = ["Forecast", "MAE", r"95\% CI", "MAE minus scaled", r"MAE, 2026"]
    return tabular("@{}lrrlr@{}", header, rows)


# ------------------------------------------------------------------ T6
def t6_benchmark():
    cr = get(BENCH, "journal_table4", "common_races")
    rows = []
    for panel, ssm, ar in (("CRPS", "crps_vs_ssm", "crps_vs_ar1"), ("RMSPE", "rmspe_vs_ssm", "rmspe_vs_ar1")):
        rows.append(("panel", panel))
        for lab, k in (("Their state-space model", ssm), ("Their AR(1)", ar)):
            b = cr[k]
            rows.append(
                [
                    lab,
                    fmt_num(b["theirs_mean"], 3),
                    fmt_num(b["ours_mean"], 3),
                    est_ci(b["mean_diff_ours_minus_theirs"], *b["mean_diff_ci95"], 3, True),
                    f"{n(b['ours_wins'])} of {n(b['n_races'])}",
                    pval(b["sign_test_p_two_sided"]),
                ]
            )
    header = ["Comparator", "Theirs", "Ours", r"Ours minus theirs [95\% CI]", "Races won", "Sign test $p$"]
    return tabular("@{}lrrlrr@{}", header, rows)


def t6b_repro():
    t3 = get(BENCH, "table3")
    pub, rep, cor = t3["published"], t3["their_ramp"], t3["corrected_fuel"]
    rows = []
    for lab, pk, rk in (("Hard, $v$ (s/lap)", "hard", "v_hard"), ("Medium, $v$ (s/lap)", "medium", "v_medium")):
        rows.append(
            [
                lab,
                est_ci(pub[pk]["mean"], pub[pk]["lo"], pub[pk]["hi"], 3),
                est_ci(rep[rk]["mean"], rep[rk]["lo"], rep[rk]["hi"], 3),
                est_ci(cor[rk]["mean"], cor[rk]["lo"], cor[rk]["hi"], 3),
            ]
        )
    rows.append(["$P(\\mathrm{medium} > \\mathrm{hard})$", DASH, fmt_num(rep["p_medium_gt_hard"], 3), fmt_num(cor["p_medium_gt_hard"], 3)])
    header = ["Quantity", "Published", "Reproduced", "Fuel ramp corrected"]
    return tabular("@{}llll@{}", header, rows)


# ------------------------------------------------------------------ T7
METRICS = (
    ("stop_match_mode", "Stop-count agreement", True),
    ("w1_median_car", r"Tyre-age $W_1$ (laps)", False),
)
BASES = (("lap_rule", "Race-length rule"), ("always_1", "Always one stop"), ("prev_season", "Previous season"))


def _strategy_panel(blk, title):
    rows = [("panel", title)]
    for bk, lab in BASES:
        cells = [f"{lab} ($n={n(blk[bk]['stop_match_mode']['n_races'])}$)"]
        for mk, _, three in METRICS:
            m = blk[bk][mk]
            cells.append(est_ci(m["mean_a_minus_b"], *m["ci95"], 3 if three else 2, True))
        rows.append(cells)
    return rows


def t7_strategy():
    key = "practice_scalar/plugin+support"
    oot = get(STRAT, "paired_vs_baselines", key)
    loro = get(STRAT, "leave_one_race_out_sensitivity", "paired_vs_baselines", key)
    rows = _strategy_panel(oot, "Out of time (headline): plan minus baseline, paired over races")
    rows.append(r"\midrule")
    rows += _strategy_panel(loro, "Leave-one-race-out (sensitivity)")
    rows.append(r"\midrule")
    sc = get(STRAT, "shipped_claim_rechecked")
    rows.append(("panel", "2026 headline set, in sample: races where the stop count equals the field's median"))
    rr = sc["race_length_rule_best_threshold_fitted_on_these_events"]
    for lab, blk in (
        ("Planner", sc["product_in_sample_with_bounds"]),
        ("Always one stop", sc["always_one_stop"]),
        ("Race-length rule, fitted in sample", rr),
    ):
        rows.append([f"{lab} ($n={n(blk['of'])}$)", f"{n(blk['matches'])} of {n(blk['of'])}", ""])
    header = ["Baseline (races)", "Stop-count agreement", r"Tyre-age $W_1$ (laps)"]
    return tabular("@{}lll@{}", header, rows)


# ------------------------------------------------------------------ T8
def t8_seasons():
    """Circuit-slope SD and within-event compound trend, season by season (label compounds)."""
    rows = []
    for y in SEASONS:
        sr = get(MODEL, "season_robustness", y)
        tr = sr["trend_per_step_softer"]["within_event"]
        rows.append(
            [
                y,
                n(sr["n_events"]),
                fmt_num(sr["circuit_slope_sd_s_per_lap"], 4),
                est_ci(tr["mean"], tr["lo"], tr["hi"], 4, True),
            ]
        )
    return tabular("@{}lrrl@{}", ["Season", "Circuits", "Circuit-slope SD (s/lap)", "Within-event trend per step softer"], rows)


def t9_inference():
    """95% interval width of each 2026 compound rate under six ways of counting independent information."""
    ic = get(MODEL, "inference_comparison")
    rows_spec = [
        ("A_lap_level_no_circuit", "Laps independent, no circuit slope"),
        ("B_lap_level_circuit", "Laps independent, circuit slope"),
        ("C_run_intercept", "Circuit slope, run intercepts"),
        ("D_run_intercept_and_slope_SHIPPED", "Circuit slope, run intercepts and slopes (ours)"),
        ("E_FE_OLS_cluster_by_run", "Fixed effects, errors clustered by run"),
        ("F_FE_OLS_cluster_by_event_t12", "Fixed effects, errors clustered by event, $t_{12}$"),
    ]
    lag = ic["residual_lag1_autocorrelation_within_runs"]
    rows = []
    for key, label in rows_spec:
        r = ic["rates"][key]
        rows.append([label, *[fmt_num(r[c]["width"], 3) for c in COMPOUNDS],
                     fmt_num(lag[key], 2) if key in lag else DASH])
    return tabular("@{}l" + "r" * (len(COMPOUNDS) + 1) + "@{}",
                   ["Inference", *COMPOUNDS, "Lag-1 autocorr."], rows)


def t10_exact():
    """Compound rates and slope SDs under the exact and the approximate within-transformation."""
    ea = get(MODEL, "exact_vs_approximate")
    rows = []
    for key, label in (("shipped_2026", "2026"), ("pooled_2022_26_nested", "2022--26 pooled")):
        for kind, name in (("exact", "exact"), ("approximate", "approximate")):
            b = ea[key][kind]
            circ = b["variance_sd"]["circuit_slope"]
            rows.append([f"{label}, {name}", *[fmt_num(b["rates_s_per_lap"][c]["mean"], 3) for c in COMPOUNDS],
                         fmt_num(b["compound_rate_sd_raw"], 4), fmt_num(circ, 4)])
    return tabular("@{}l" + "r" * (len(COMPOUNDS) + 2) + "@{}",
                   ["Fit", *COMPOUNDS, "Compound SD", "Circuit SD"], rows)


def t11_recovery():
    """Mean bias and 95% coverage over compounds, by estimator and pit-stop scenario."""
    res = get(RECOV, "results")
    spec = [
        ("naive_pooled_ols", "Naive"),
        ("fuel_within_run", "Fuel-corrected within run"),
        ("cross_section_cell", "Cross-sectional, no run terms"),
        ("field_demeaned_ols", "Field-demeaned, no run terms"),
        ("eq1_shipped", "Eq.~(1), approximate demeaning"),
        ("eq1_exact_within", "Eq.~(1) (ours)"),
        ("eq1_exact_within_mundlak", "Eq.~(1) with run means"),
    ]
    rows = []
    for key, label in spec:
        cells = []
        for s in ("exogenous", "pace", "wear"):
            m = res[s][key]["mean_over_compounds"]
            cells += [fmt_num(m["bias"], 3, signed=True), fmt_num(m["mean_abs_bias"], 3), fmt_num(m["coverage"], 2)]
        rows.append([label, *cells])
    head = ["Estimator"] + ["Bias", "$|$Bias$|$", "Cov."] * 3
    body = tabular("@{}l" + "r" * 9 + "@{}", head, rows)
    group = (r"\multicolumn{1}{l}{} & \multicolumn{3}{c}{Random stops} & \multicolumn{3}{c}{Stops by pace} "
             r"& \multicolumn{3}{c}{Stops by wear} \\" "\n"
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}" "\n")
    return body.replace("\\toprule\n", "\\toprule\n" + group, 1)


TABLES = {
    "tab_exact": t10_exact,
    "tab_recovery": t11_recovery,
    "tab_inference": t9_inference,
    "tab_data": t1_data,
    "tab_rates": t2_rates,
    "tab_race_practice": t4_race_practice,
    "tab_forecast": t5_forecast,
    "tab_benchmark": t6_benchmark,
    "tab_repro": t6b_repro,
    "tab_strategy": t7_strategy,
    "tab_seasons": t8_seasons,
}


def build_all() -> dict[str, str]:
    return {name: fn() for name, fn in TABLES.items()}


def main() -> int:
    for name, text in build_all().items():
        write_text(GENERATED / f"{name}.tex", text)
        print(f"wrote paper/manuscript/generated/{name}.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

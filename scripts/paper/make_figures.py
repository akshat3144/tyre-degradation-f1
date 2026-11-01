"""Write the manuscript figures to paper/manuscript/figures/*.pdf (vector).

    python scripts/paper/make_figures.py

Every series is read from the JSON results (no parquet, no random numbers).
Style: serif (STIX, ships with matplotlib), 6.5 in full / 3.2 in single column,
Okabe-Ito colours with distinct markers so the figures survive grayscale
printing, light grid, no top/right spines, 95% intervals everywhere.

  fig_sign_error      A.1 naive vs fuel-corrected vs adjusted slopes per compound (appendix)
  fig_circuit_decomposition F1  persistent circuit, race-specific and compound spread (2022-26 nested fit); pooled compound rates
  fig_circuit_rates   F2  2026 compound rates and per-circuit rates on one scale
  fig_race_practice   F3  race vs practice slopes (used sets) and the gap under alternative designs
  fig_power           F4  runs for 80% power vs the gap (A) and power vs the gap at our run counts (B), by design
  fig_forecast        F5  forecast MAE ladder, out of season, with paired differences
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _paper_common import COMPOUNDS, FIGURES, MGMT, MODEL, TRANS, get, has, shown  # noqa: E402

FULL, SINGLE = 6.5, 3.2

# Okabe-Ito subset, ordered so lightness also differs (grayscale-safe with markers)
BLACK, BLUE, GREEN, ORANGE, PINK, VERM, GREY = "#000000", "#0072B2", "#009E73", "#E69F00", "#CC79A7", "#D55E00", "#7F7F7F"
CCOL = {"C1": BLACK, "C2": BLUE, "C3": GREEN, "C4": ORANGE, "C5": PINK}
CMARK = {"C1": "o", "C2": "s", "C3": "^", "C4": "D", "C5": "v"}

RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embed TrueType: text stays text
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}
RATE_LABEL = r"Degradation rate (s/lap)"


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"{name}.pdf"
    fig.savefig(out, metadata={"CreationDate": None, "Creator": "scripts/paper/make_figures.py"})
    plt.close(fig)
    return out


def _panel(ax, tag):
    ax.text(-0.02, 1.04, tag.strip("()").upper(), transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


# ---------------------------------------------------------------- F1
def fig_sign_error():
    rows = get(MODEL, "ablation", "rows")
    est = [
        ("naive_run_clustered", "Naive (no controls)", GREY, "o", -0.22, "none"),
        ("practitioner_fuel_corrected_within_stint", "Fuel-corrected within run", BLUE, "s", 0.0, "full"),
        ("deconfounded", "Adjusted (Eq. 1)", VERM, "D", 0.22, "full"),
    ]
    fig, ax = plt.subplots(figsize=(FULL, 2.9))
    x = np.arange(len(COMPOUNDS))
    ax.axhline(0, color="black", lw=0.7, zorder=1)
    ax.axhspan(-0.18, 0, color="#EDEDED", zorder=0)
    for key, lab, col, mk, off, fill in est:
        m = np.array([rows[c][key]["mean"] for c in COMPOUNDS])
        lo = np.array([rows[c][key]["lo"] for c in COMPOUNDS])
        hi = np.array([rows[c][key]["hi"] for c in COMPOUNDS])
        ax.errorbar(
            x + off, m, yerr=[m - lo, hi - m], fmt=mk, color=col, mfc=("white" if fill == "none" else col),
            ms=4.5, lw=1.0, capsize=2, label=lab, zorder=3,
        )
    wrong = [i for i, c in enumerate(COMPOUNDS) if rows[c]["naive_run_clustered"]["mean"] < 0]
    for i in wrong:
        ax.text(i - 0.22, -0.172, "wrong sign", ha="center", va="bottom", fontsize=6.5, color="#444444")
    ax.set_xticks(x, [f"{c}\n$n$ = {rows[c]['n_runs']} runs" for c in COMPOUNDS])
    ax.set_ylabel(RATE_LABEL)
    ax.set_ylim(-0.18, 0.30)
    ax.set_xlim(-0.6, len(COMPOUNDS) - 0.4)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, fc="#EDEDED", ec="none"))
    labels.append("Tyre appears to get faster (rate below 0)")
    ax.legend(handles, labels, loc="upper right", ncol=1, handletextpad=0.4)
    ax.grid(axis="x", visible=False)
    return _save(fig, "fig_sign_error")


# ---------------------------------------------------------------- F1
def fig_circuit_decomposition():
    cve = get(MODEL, "circuit_vs_event")
    sd = cve["sd_s_per_lap"]
    cv = cve["compound_vs_variance"]
    c26 = get(MODEL, "circuit_effect", "circuit_slope_sd_s_per_lap")
    rows = [  # label, estimate, bootstrap 95% interval, colour, marker, filled
        ("Persistent circuit\n(2022-26)", sd["circuit_persistent"]["estimate"], sd["circuit_persistent"]["bootstrap_95"], VERM, "o", True),
        ("Race-specific, within\ncircuit (2022-26)", sd["event_specific"]["estimate"], sd["event_specific"]["bootstrap_95"], BLUE, "s", True),
        ("Five compound rates\n(2022-26)", cve["compound_rate_sd_s_per_lap"], cv["compound_sd_raw"]["bootstrap_95"], GREEN, "^", True),
        ("Circuit, 2026 only\n(one race each)", c26["estimate"], c26["parametric_bootstrap_percentile_95"], GREY, "D", False),
    ]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.5), gridspec_kw={"wspace": 0.55, "width_ratios": [1.15, 1]})
    y = np.arange(len(rows))[::-1]
    for yi, (_lab, est, (lo, hi), col, mk, filled) in zip(y, rows, strict=True):
        a1.errorbar(est, yi, xerr=[[est - lo], [hi - est]], fmt=mk, color=col, ms=4.5, lw=1.0, capsize=2,
                    mfc=(col if filled else "white"))
    a1.set_yticks(y, [r[0] for r in rows])
    a1.set_xlim(0, 0.07)
    a1.set_ylim(-0.6, len(rows) - 0.4)
    a1.set_xlabel(r"Standard deviation (s/lap), 95% bootstrap interval")
    a1.set_title("Spread of tyre-age slopes", loc="left", fontsize=8)
    rates = cve["compound_rates_pooled_s_per_lap"]
    yc = np.arange(len(COMPOUNDS))[::-1]
    for yi, c in zip(yc, COMPOUNDS, strict=True):
        r = rates[c]
        a2.errorbar(r["mean"], yi, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt=CMARK[c], color=CCOL[c],
                    ms=4.5, lw=1.0, capsize=2)
    a2.set_yticks(yc, COMPOUNDS)
    a2.axvline(0, color="black", lw=0.6)
    a2.set_xlim(-0.005, 0.09)
    a2.set_ylim(-0.6, len(COMPOUNDS) - 0.4)
    a2.set_xlabel(RATE_LABEL + ", 95% interval")
    a2.set_title("Pooled rate by compound, 2022-26", loc="left", fontsize=8)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    _panel(a1, "A")
    _panel(a2, "B")
    return _save(fig, "fig_circuit_decomposition")


# ---------------------------------------------------------------- F2
XLIM_CIRCUIT = (-0.15, 0.27)


def fig_circuit_rates():
    rates = get(MODEL, "shipped_fit", "rates_s_per_lap")
    cs = get(MODEL, "circuit_effect", "circuit_slope_sd_s_per_lap")
    cells = get(MODEL, "per_race_separation", "cells_min_runs_3_run_level")
    events = sorted({c["event"] for c in cells})
    mean_by_event = {e: np.mean([c["rate"] for c in cells if c["event"] == e]) for e in events}
    events.sort(key=lambda e: mean_by_event[e])
    short = {e: e.replace(" Grand Prix", "") for e in events}
    lo_lim, hi_lim = XLIM_CIRCUIT

    fig, (a1, a2) = plt.subplots(
        2, 1, figsize=(FULL, 5.6), sharex=True, gridspec_kw={"height_ratios": [5, 13], "hspace": 0.20}
    )
    y = np.arange(len(COMPOUNDS))[::-1]
    for yi, c in zip(y, COMPOUNDS, strict=True):
        r = rates[c]
        a1.errorbar(r["mean"], yi, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt=CMARK[c], color=CCOL[c],
                    ms=4.5, lw=1.0, capsize=2)
    a1.set_yticks(y, [f"{c}  ($n$ = {rates[c]['n_runs']})" for c in COMPOUNDS])
    a1.axvline(0, color="black", lw=0.6)
    a1.set_title("Pooled rate by compound, 2026", loc="left", fontsize=8)
    _panel(a1, "A")

    off = {1: [0.0], 2: [-0.15, 0.15], 3: [-0.25, 0.0, 0.25]}
    ypos = np.arange(len(events))
    for yi, e in zip(ypos, events, strict=True):
        ec = [c for c in cells if c["event"] == e]
        ec.sort(key=lambda c: c["compound"])
        for o, c in zip(off[len(ec)], ec, strict=True):
            lo, hi = max(c["lo"], lo_lim), min(c["hi"], hi_lim)
            col, mk = CCOL[c["compound"]], CMARK[c["compound"]]
            a2.plot([lo, hi], [yi + o, yi + o], color=col, lw=0.8, solid_capstyle="butt")
            if c["lo"] >= lo_lim:
                a2.plot([c["lo"]] * 2, [yi + o - 0.06, yi + o + 0.06], color=col, lw=0.8)
            if c["hi"] <= hi_lim:
                a2.plot([c["hi"]] * 2, [yi + o - 0.06, yi + o + 0.06], color=col, lw=0.8)
            if lo_lim <= c["rate"] <= hi_lim:
                a2.plot(c["rate"], yi + o, mk, color=col, ms=3.5)
            else:  # estimate beyond the axis: arrow at the edge
                edge = lo_lim if c["rate"] < lo_lim else hi_lim
                a2.plot(edge, yi + o, "<" if c["rate"] < lo_lim else ">", color=col, ms=4, mfc="white")
    a2.axvline(0, color="black", lw=0.6)
    a2.set_yticks(ypos, [short[e] for e in events])
    a2.set_ylim(-0.7, len(events) - 0.3)
    a2.set_xlim(lo_lim, hi_lim)
    a2.set_xlabel(RATE_LABEL)
    a2.set_title("Rate by circuit and compound (at least 3 runs)", loc="left", fontsize=8)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker=CMARK[c], color=CCOL[c], ls="none", ms=4.5, label=c) for c in COMPOUNDS]
    a2.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=5, columnspacing=0.8,
              handletextpad=0.2, borderaxespad=0.1)
    a2.text(0.99, 0.04, rf"2026 circuit-slope SD {cs['estimate']:.3f} s/lap" "\n"
            rf"[{cs['profile_likelihood_95'][0]:.3f}, {cs['profile_likelihood_95'][1]:.3f}] profile interval",
            transform=a2.transAxes, ha="right", va="bottom", fontsize=7, color="#333333")
    _panel(a2, "B")
    return _save(fig, "fig_circuit_rates")


# ---------------------------------------------------------------- F3
def fig_race_practice():
    ud = get(MGMT, "referee_additions", "designs", "used_sets_only_FreshTyre_false")
    lg = ud["level_gap"]
    pl = ud["level_placebo_fp2_minus_fp1fp3"]
    pa = ud["placebo_adjusted_gap"]["MEDIUM+SOFT"]
    sa = get(MGMT, "referee_additions", "designs", "start_age_proxy_matched", "level_gap")
    ix = get(MGMT, "referee_additions", "designs", "used_sets_and_start_age_matched", "level_gap")
    um = get(MGMT, "post_review_additions", "level_gap_unmatched_pooled")
    keys = [("HARD", "Hard"), ("MEDIUM", "Medium"), ("SOFT", "Soft"), ("MEDIUM+SOFT", "Medium\n+ soft")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.9), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.95})
    x = np.arange(len(keys))
    for xi, (k, _lab) in zip(x, keys, strict=True):
        b = lg[k]
        a1.plot([xi, xi], [b["practice"], b["race"]], color="#999999", lw=1.0, zorder=1)
        a1.plot(xi, b["practice"], "s", color=BLUE, ms=5, zorder=3, label="Practice (square)" if xi == 0 else None)
        a1.plot(xi, b["race"], "o", color=VERM, ms=5, zorder=3, label="Race (circle)" if xi == 0 else None)
    a1.set_xticks(x, [f"{lab}\n$n$ = {lg[k]['n']}" for k, lab in keys])
    a1.set_ylabel(RATE_LABEL)
    a1.set_ylim(0, 0.20)
    a1.set_xlim(-0.5, len(keys) - 0.5)
    a1.legend(loc="upper left", ncol=1, handletextpad=0.2)
    a1.grid(axis="x", visible=False)
    _panel(a1, "A")

    rows = [
        ("Used sets, medium + soft\n(primary)", lg["MEDIUM+SOFT"]["gap"], lg["MEDIUM+SOFT"]["gap_ci95"], lg["MEDIUM+SOFT"]["n"], BLACK, "o", True),
        ("Used sets, all compounds", lg["ALL"]["gap"], lg["ALL"]["gap_ci95"], lg["ALL"]["n"], BLACK, "o", True),
        ("Start-age matched", sa["MEDIUM+SOFT"]["gap"], sa["MEDIUM+SOFT"]["gap_ci95"], sa["MEDIUM+SOFT"]["n"], BLACK, "s", True),
        ("Used sets and start age", ix["MEDIUM+SOFT"]["gap"], ix["MEDIUM+SOFT"]["gap_ci95"], ix["MEDIUM+SOFT"]["n"], BLACK, "D", True),
        ("No matching", um["MEDIUM+SOFT"]["gap"], um["MEDIUM+SOFT"]["gap_ci95"], um["MEDIUM+SOFT"]["n"], BLACK, "^", True),
        ("Placebo (FP2 vs FP1, FP3)", pl["MEDIUM+SOFT"]["gap"], pl["MEDIUM+SOFT"]["gap_ci95"], pl["MEDIUM+SOFT"]["n"], GREY, "o", False),
        ("Race gap minus placebo", pa["placebo_adjusted_gap"], pa["placebo_adjusted_gap_ci95_joint_bootstrap"], None, GREY, "s", False),
    ]
    yy = np.arange(len(rows))[::-1]
    for yi, (lab, g, c95, nn, col, mk, filled) in zip(yy, rows, strict=True):
        a2.errorbar(g, yi, xerr=[[g - c95[0]], [c95[1] - g]], fmt=mk, color=col, mfc=(col if filled else "white"), ms=5,
                    lw=1.0, capsize=2)
    a2.axvline(0, color="black", lw=0.6)
    a2.set_yticks(yy, [lab + (f"\n$n$ = {nn}" if nn else "") if "\n" not in lab else lab + (f", $n$ = {nn}" if nn else "")
                       for lab, _g, _c, nn, *_ in rows])
    a2.set_xlabel(r"Race minus practice slope (s/lap)")
    a2.xaxis.set_major_locator(plt.MaxNLocator(5))
    a2.set_ylim(-0.6, len(rows) - 0.4)
    a2.grid(axis="y", visible=False)
    a2.tick_params(axis="y", labelsize=6.5)
    _panel(a2, "B")
    return _save(fig, "fig_race_practice")


# ---------------------------------------------------------------- F4
def power_extended_ready() -> bool:
    """False while the long run has not merged ``power_extended`` into model.json (panel A is then a flagged placeholder)."""
    return has(MODEL, "power_extended")


def _wald_power(effect, se):
    """Two-sided 5% Wald-z power of a contrast with standard error ``se`` (known variances)."""
    from math import erf, sqrt

    cdf = lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0)))  # noqa: E731
    z = np.asarray(effect, float) / se
    return np.array([1.0 - cdf(1.96 - v) + cdf(-1.96 - v) for v in np.atleast_1d(z)])


def fig_power():
    mde = get(MODEL, "min_detectable_effect")
    an, sim = mde["analytic_known_variance"], mde["simulated_refit"]
    n_real = an["realistic_C1_C2"]["at_actual_pair_runs"]["n_stints"]  # our C1 + C2 runs
    n_bal = an["balanced"]["at_N_832"]["n_stints"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 3.0), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1]})

    # ---- B first (final data): power against the true gap, at the run counts we hold
    cases = [  # key, label, colour, linestyle, marker, SE at that design's run count
        ("balanced", f"balanced, {n_bal} runs", BLUE, "-", "o", an["balanced"]["at_N_832"]["point"]["se_s_per_lap"]),
        ("realistic_C1_C2", f"realistic C1 vs C2, {n_real} runs (ours)", VERM, "-", "s",
         an["realistic_C1_C2"]["at_actual_pair_runs"]["point"]["se_s_per_lap"]),
        ("split", f"split, {n_bal} runs", BLACK, "--", "^", an["split"]["at_N_832"]["point"]["se_s_per_lap"]),
    ]
    xs = np.linspace(0, 0.075, 301)
    for key, lab, col, ls, mk, se in cases:
        a2.plot(xs, _wald_power(xs, se), ls=ls, color=col, lw=1.3 if key.startswith("real") else 1.0, label=lab)
        if key in sim:
            sm = sim[key]
            a2.errorbar(sm["effects"], sm["power_wald_z"], yerr=[1.96 * v for v in sm["power_mc_se"]], fmt=mk, ms=3.8,
                        color=col, mfc="white", lw=0.8, capsize=1.5, ls="none")
        m80 = an[key]["at_actual_pair_runs" if key.startswith("real") else "at_N_832"]["point"]["mde_80_s_per_lap"]
        a2.plot([m80], [0.8], marker="|", ms=9, mew=1.4, color=col)
    a2.axhline(0.8, color="black", lw=0.6, ls=":")
    a2.text(0.045, 0.81, "80% power", fontsize=6.5, ha="center", va="bottom")
    a2.axvline(0.006, color=GREY, lw=0.7, ls=":")
    a2.text(0.0065, 1.0, "0.006", fontsize=6.5, color="#444444", va="top")
    a2.set_xlim(0, 0.075)
    a2.set_ylim(0, 1.03)
    a2.set_xlabel(r"True compound gap (s/lap)")
    a2.set_ylabel("Power (Wald $z$ test)")
    a2.legend(loc="lower right", fontsize=6.3, handlelength=2.6)
    _panel(a2, "B")

    # ---- A: runs needed for 80% power against the gap (power_extended; pending until merged)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(0.0052, 0.036)
    a1.set_ylim(30, 9000)
    a1.axhline(n_real, color=GREY, lw=0.9)
    a1.text(0.0353, n_real * 1.07, f"our {n_real} C1 + C2 runs", fontsize=6.5, color="#444444", ha="right", va="bottom")
    a1.set_xlabel(r"True compound gap (s/lap)")
    a1.set_ylabel("Runs for 80% power (log scale)")
    ticks = [0.006, 0.01, 0.02, 0.03]
    a1.set_xticks(ticks, [str(t) for t in ticks])
    a1.minorticks_off()
    _panel(a1, "A")
    if not power_extended_ready():
        a1.text(0.5, 0.55, "PLACEHOLDER\npower_extended is not yet in model.json;\nrun p3_model.py --only power_extended",
                transform=a1.transAxes, ha="center", va="center", fontsize=8, color="#B00000", fontweight="bold")
        return _save(fig, "fig_power")
    pe = get(MODEL, "power_extended")
    ac, sm = pe["analytic_crossing_80pct_stints"], pe["simulated"]
    eff = sorted(float(k.split("_")[1]) for k in ac)
    key = lambda e: f"effect_{e:g}"  # noqa: E731
    designs = [("balanced", "balanced", BLUE, "o", "-"), ("realistic_C1_C2", "realistic C1 vs C2", VERM, "s", "-")]
    for dn, lab, col, mk, ls in designs:
        a1.plot(eff, [ac[key(e)][dn]["point"] for e in eff], ls=ls, marker=mk, ms=4, lw=1.2, color=col, label=lab)
        st = [(e, ac[key(e)][dn]["circuit_high_and_run_slope_x1.5"]) for e in eff]
        st = [(e, v) for e, v in st if v is not None]  # beyond the analytic grid: no crossing, so no point
        a1.plot([e for e, _ in st], [v for _, v in st], ls=":", lw=1.0, color=col,
                label=lab + ", stressed")
        pts = [(e * 1.035, sm[key(e)][dn]["crossing_80_wald_z"]) for e in eff if dn in sm.get(key(e), {})]
        pts = [(x, c) for x, c in pts if c.get("crossing") is not None and all(v is not None for v in c["mc_ci95"])]
        if pts:
            a1.errorbar([x for x, _ in pts], [c["crossing"] for _, c in pts],
                        yerr=[[max(c["crossing"] - c["mc_ci95"][0], 0) for _, c in pts], [max(c["mc_ci95"][1] - c["crossing"], 0) for _, c in pts]],
                        fmt=mk, ms=4.2, mfc="white", color=col, lw=0.8, capsize=1.5, ls="none",
                        label=lab + ", simulated")
    # split: the analytic crossing does not exist (power ceiling below 80%); mark the top edge
    ceil = pe["split_design_power_ceiling"]
    top = [e for e in eff if ac[key(e)]["split"]["point"] is None]
    a1.plot(top, [6200] * len(top), ls="none", marker="^", ms=5, mfc="white", color=BLACK)
    a1.text(0.0353, 6200, f"split: never; power ceiling {ceil[key(eff[0])]['point']:.2f} to {ceil[key(eff[-1])]['point']:.2f}",
            fontsize=6.3, ha="right", va="center", bbox=dict(fc="white", ec="none", pad=0.6))
    a1.legend(loc="lower left", fontsize=5.8, handlelength=2.4, ncol=1, borderaxespad=0.2)
    return _save(fig, "fig_power")


# ---------------------------------------------------------------- F5
LADDER = [
    ("scalar", "Scaled forecast", VERM, "D"),
    ("naive", "Naive (Sunday = Friday)", GREY, "o"),
    ("zero", "Zero", GREY, "o"),
    ("pooled", "Pooled race rate", GREY, "o"),
    ("const", "Constant", GREY, "o"),
    ("history", "Previous season", GREY, "o"),
    ("shrink50", "50% shrink to constant", BLUE, "s"),
]


def fig_forecast():
    hf = get(TRANS, "holdout_frozen")
    sb, sv = hf["scoreboard"], hf["scalar_vs"]
    yy = np.arange(len(LADDER))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.8), sharey=True, gridspec_kw={"wspace": 0.08, "width_ratios": [1.15, 1]})
    for yi, (k, _lab, col, mk) in zip(yy, LADDER, strict=True):
        b = sb[k]
        a1.errorbar(b["mae"], yi, xerr=[[b["mae"] - b["mae_ci_lo"]], [b["mae_ci_hi"] - b["mae"]]], fmt=mk, color=col, ms=4.5,
                    lw=1.0, capsize=2, mfc=(col if k in ("scalar", "shrink50") else "white"))
        if k != "scalar":
            d = sv[k]  # scalar minus method; flip so positive = method worse than the scaled forecast
            diff, lo, hi = -d["mean_diff"], -d["ci_hi"], -d["ci_lo"]
            a2.errorbar(diff, yi, xerr=[[diff - lo], [hi - diff]], fmt=mk, color=col, ms=4.5, lw=1.0, capsize=2,
                        mfc=(col if k == "shrink50" else "white"))
    a1.set_yticks(yy, [lab for _, lab, _, _ in LADDER])
    a1.set_xlabel(r"Mean absolute error (s/lap)")
    a1.set_xlim(0, 0.14)
    a1.set_title(f"Out-of-season MAE, 95% CI ({hf['n_cells']} groups)", loc="left", fontsize=8)
    a2.axvline(0, color="black", lw=0.6)
    a2.set_xlabel(r"MAE minus scaled (s/lap), paired 95% CI")
    a2.set_title("Paired difference", loc="left", fontsize=8)
    a2.set_ylim(-0.6, len(LADDER) - 0.4)
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    _panel(a1, "A")
    _panel(a2, "B")
    return _save(fig, "fig_forecast")


FIGS = {
    "fig_sign_error": fig_sign_error,
    "fig_circuit_decomposition": fig_circuit_decomposition,
    "fig_circuit_rates": fig_circuit_rates,
    "fig_race_practice": fig_race_practice,
    "fig_power": fig_power,
    "fig_forecast": fig_forecast,
}


def main() -> int:
    with plt.rc_context(RC):
        for fn in FIGS.values():
            out = fn()
            print(f"wrote {shown(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

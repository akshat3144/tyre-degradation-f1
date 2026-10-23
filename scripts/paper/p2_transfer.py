"""Paper P2: does a race's tyre degradation follow from that weekend's practice?

    python scripts/paper/p2_transfer.py            # ~3-6 minutes, writes paper/results/transfer.json

WHAT WAS WRONG WITH THE SHIPPED CLAIM
    "Leave-one-event-out MAE 0.056 s/lap against a naive 0.087 (35.6% better),
    21 cells, 2026 only." A hostile audit found: the practice design and the
    session weights were chosen on those same 21 cells; the naive baseline is
    the weakest one available (predict-zero scores 0.069); one scalar was
    applied though the docstring says the gap widens for softer tyres; and the
    design sweep that justified the warm-up trim no longer reproduces.

PRE-STATED TESTS
    Written before the out-of-season numbers were produced (the 2026 numbers
    were already known; they motivate the tests, they are not evidence for them).
    Verdicts are computed mechanically in ``verdicts`` below from these rules,
    not read off by eye. "Beats" always means: the 95% weekend-clustered
    bootstrap interval of the paired MAE difference lies entirely below zero.

    PRIMARY ENDPOINT
      H1  Frozen 2026 design and factor, applied to 2022-2025 (label compounds):
          the calibrated forecast beats predict-zero, the pooled-by-compound
          constant, and the naive Sunday=Friday forecast.
      H1b Same, with the factor learned leave-one-weekend-out inside 2022-2025.
      H1c Whether it also beats the same circuit's previous-season race rate is
          NOT part of the claim; it is reported because a reviewer will ask.

    SECONDARY
      H2  Nested selection of design and weights inside each training fold
          loses no more than 0.005 s/lap of MAE against the shipped (tuned on
          the evaluation cells) choice. If it loses more, the shipped 0.056 is
          that much optimistic.
      H3  A compound-specific factor beats the scalar one, out of sample, in
          BOTH the frozen holdout and the 2026 leave-one-out. Adopt only if both.
      H4  FP2 predicts the race better than FP1 out of season: the paired
          correlation difference (FP2 - FP1, same cells) has a CI above zero.
      H5  The hard identifiability rule (|rate| >= se) does not lose to
          continuous empirical-Bayes shrinkage, and does not beat 'always trust
          practice'. (Both can fail; the rule is a pre-test estimator.)
      H6  Stand-in intervals cover the realised race rate at least 85% of the
          time at their shipped multiples. If not, the multiple that does reach
          95% (cross-validated across weekends) is reported.

    No threshold, design, weight or factor is fitted on the 2022-2025 cells.

LIMITS THE PAPER MUST STATE
    * 2022-2025 cells are keyed on HARD/MEDIUM/SOFT labels, which are relative
      to each weekend's nomination (a HARD is a different rubber at Monaco and
      at Suzuka). The repo holds Pirelli's allocation for 2026 only. That adds
      noise to the holdout and cannot flatter it.
    * 2022 was a new regulation and tyre generation, 2026 another. A factor
      learned on 2026 need not transfer; that is the point of freezing it.
    * The target is a MEASURED race rate with its own noise. Some are negative
      (not physical); they are kept, because dropping them would be selecting
      on the outcome.
    * No Sprint sessions are cached for 2022-2025, so the Sprint weight
      cannot be tested out of season.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from tyredeg.config import PROCESSED, ROOT
from tyredeg.paper import transfer_eval as E
from tyredeg.validation import transfer as T

warnings.filterwarnings("ignore")

OUT = ROOT / "paper" / "results" / "transfer.json"
HOLDOUT = E.HOLDOUT_SEASONS
BASELINES = ("zero", "const", "pooled", "history", "naive")
ALL = ("zero", "const", "pooled", "history", "naive", "scalar", "compound", "shrink50", "stack")


def beats(cmp: dict) -> bool:
    """CI of (focal - other) lies entirely below zero."""
    return bool(cmp["ci_hi"] < 0)


def section(df: pd.DataFrame, focal: str = "scalar", extra_pairs: list[tuple[str, str]] | None = None,
            cluster: str = "event") -> dict:
    """Scoreboard + focal-vs-everything pairwise tests + a few named contrasts."""
    others = [m for m in ALL if m != focal and f"pred_{m}" in df.columns]
    out = {
        "n_cells": int(len(df)),
        "n_weekends": int(df[cluster].nunique()),
        "scoreboard": E.scoreboard(df, ALL, cluster),
        f"{focal}_vs": E.compare(df, focal, others, cluster),
    }
    for a, b in extra_pairs or []:
        out.setdefault("pairs", {})[f"{a}_vs_{b}"] = E.compare(df, a, [b], cluster)[b]
    return out


def main() -> None:
    t0 = time.time()

    def tick(msg: str) -> None:
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    laps26 = E.load_laps(2026)
    data = {2026: E.build_season(2026, laps26)}
    for s in HOLDOUT:
        data[s] = E.build_season(s)
    print(f"built practice/race frames in {time.time() - t0:.0f}s")

    # ---- panels ----------------------------------------------------------
    panel26, race_cells = E.season_panels([2026], data=data, history_seasons=HOLDOUT)
    panel_h, race_cells_h = E.season_panels(HOLDOUT, data=data)
    res: dict = {"data": {
        "cells_2026": int(len(panel26)), "weekends_2026": int(panel26["event"].nunique()),
        "cells_holdout": int(len(panel_h)), "weekends_holdout": int(panel_h["event"].nunique()),
        "cells_holdout_by_season": {str(k): int(v) for k, v in panel_h.groupby("season").size().items()},
        "holdout_has_history": int(panel_h["hist_rate"].notna().sum()),
        "s2026_has_history": int(panel26["hist_rate"].notna().sum()),
        "holdout_compounds": "HARD/MEDIUM/SOFT labels (relative to nomination)",
        "negative_race_rates_2026": int((panel26["race_rate"] < 0).sum()),
        "negative_race_rates_holdout": int((panel_h["race_rate"] < 0).sum()),
    }}

    # ---- A. the original claim, plus the baseline ladder ------------------
    loeo26 = E.loeo(panel26, race_cells=race_cells)
    res["s2026_loeo"] = section(loeo26, extra_pairs=[("compound", "scalar"), ("stack", "scalar"),
                                                     ("scalar", "history")])
    res["s2026_loeo"]["factor_full_sample"] = E.ratio_factor(panel26)
    res["s2026_loeo"]["note"] = ("design and session weights were chosen on these cells: "
                                 "this is the original, optimistically tuned number")

    tick("A done: 2026 LOEO + baselines")

    # ---- B. nested selection ---------------------------------------------
    tables = E.candidate_tables(laps26)
    nested, sel_log = E.nested_selection(tables)
    n = loeo26.merge(nested, on=["event", "C"], suffixes=("", "_n"))
    n["pred_nested"] = n["pred_scalar_nested"]
    nested_board = E.scoreboard(n, ("zero", "pooled", "naive", "history", "scalar", "nested"))
    cmp_nested = E.compare(n, "nested", ["scalar", "pooled", "zero", "naive", "history"])
    nested_mae = float(np.abs(n["pred_scalar_nested"] - n["race_rate"]).mean())
    shipped_mae = float(np.abs(n["pred_scalar"] - n["race_rate"]).mean())
    res["nested"] = {
        "n_cells": int(len(n)), "n_candidates": len(tables),
        "mae_nested": nested_mae, "mae_shipped_tuned_on_eval": shipped_mae,
        "optimism_of_shipped": nested_mae - shipped_mae,
        "scoreboard": nested_board, "nested_vs": cmp_nested,
        "n_fallback_cells": int(n["fallback"].sum()),
        "selection_log": sel_log,
        "selection_counts": sel_log.groupby(["design", "weights"]).size().rename("folds")
                                   .reset_index().sort_values("folds", ascending=False),
        "shipped_chosen_in_folds": int(((sel_log["design"] == E.SHIPPED_KEY[0])
                                        & (sel_log["weights"] == E.SHIPPED_KEY[1])).sum()),
    }

    tick("B done: nested")

    # ---- C. the holdout --------------------------------------------------
    frozen = E.frozen_holdout(panel26, panel_h, race_cells=race_cells)
    res["holdout_frozen"] = section(frozen, extra_pairs=[("compound", "scalar"), ("scalar", "history")])
    res["holdout_frozen"]["factor_frozen_from_2026"] = E.ratio_factor(panel26)
    res["holdout_frozen"]["by_season"] = {
        str(s): E.scoreboard(frozen[frozen["season"] == s], ("zero", "pooled", "naive", "history", "scalar"))
        for s in HOLDOUT
    }
    hist_only = frozen[frozen["has_history"]]
    res["holdout_frozen"]["cells_with_history"] = section(hist_only)

    loeo_h = E.loeo(panel_h, race_cells=race_cells_h)
    res["holdout_loeo"] = section(loeo_h, extra_pairs=[("compound", "scalar"), ("stack", "scalar"),
                                                       ("scalar", "history")])
    res["holdout_loeo"]["factor_full_sample"] = E.ratio_factor(panel_h)
    # Same weekends, circuit-clustered: the same circuit recurs across seasons,
    # so weekends of one circuit are not independent draws.
    circ = loeo_h.assign(cluster_circuit=loeo_h["circuit"])
    res["holdout_loeo"]["scalar_vs_pooled_circuit_clustered"] = E.compare(
        circ, "scalar", ["pooled", "zero", "naive"], cluster="cluster_circuit")
    circ_f = frozen.assign(cluster_circuit=frozen["circuit"])
    res["holdout_frozen"]["scalar_vs_baselines_circuit_clustered"] = E.compare(
        circ_f, "scalar", ["pooled", "zero", "naive"], cluster="cluster_circuit")

    tick("C done: holdout")

    # ---- D. compound-specific factor structure ---------------------------
    p26_lab = panel26.assign(grp=panel26["label"])
    res["compound_factor"] = {
        "2026_by_C": E.factor_by_compound(panel26, T.C_ORDER),
        "2026_by_label": E.factor_by_compound(p26_lab, E.LABEL_ORDER),
        "holdout_by_label": E.factor_by_compound(panel_h, E.LABEL_ORDER),
    }

    tick("D done: compound factors")

    # ---- E/F. identifiability rule, EB shrinkage, stand-in accuracy -------
    ident = {}
    standin = {}
    for name, pn, lo, practice in (
        ("2026", panel26, loeo26, data[2026].practice),
        ("holdout", panel_h, loeo_h, pd.concat([data[s].practice for s in HOLDOUT], ignore_index=True)),
    ):
        fby = dict(zip(lo["event"], lo["factor_used"], strict=True))
        st = E.standin_table(practice, lo, fby)
        ie = E.identifiability_eval(st)
        ie["pred_always"] = ie["pred_scalar"]
        ident[name] = {
            "n_cells": int(len(ie)),
            "n_standin_available": int(ie["si_rate"].notna().sum()),
            "n_flagged_by_shipped_rule": int(ie["flag_rule1"].sum()),
            "scoreboard": E.scoreboard(ie, ["always", "rule0.5", "rule1", "rule2", "eb"]),
            "rule1_vs_always": E.compare(ie, "rule1", ["always"])["always"],
            "eb_vs_rule1": E.compare(ie, "eb", ["rule1"])["rule1"],
            "eb_vs_always": E.compare(ie, "eb", ["always"])["always"],
            "flagged_cells": ie.loc[ie["flag_rule1"], ["event", "C", "practice_rate", "se_p",
                                                       "race_rate", "si_rate"]],
        }

        # stand-in accuracy: hidden cell imputed, race rate scored.
        ok = st["si_rate"].notna()
        sd = st[ok].copy()
        sd["pred_standin"] = sd["si_rate"] * sd["factor_used"]
        f = sd["factor_used"]
        sd["pred_pooled_unscaled"] = (sd["si_rate"] / sd["si_sev"].where(sd["si_sev"].notna(), 1.0)
                                      * f)
        sd["si_hit_shipped"] = (sd["race_rate"] >= sd["si_lo"]) & (sd["race_rate"] <= sd["si_hi"])
        cal95 = E.interval_calibration(sd, "pred_standin", "si_unit", 0.95)
        cal90 = E.interval_calibration(sd, "pred_standin", "si_unit", 0.90)
        board = E.scoreboard(sd, ["zero", "const", "pooled", "pooled_unscaled", "standin", "scalar"])
        shipped_mult = sd["si_mult"].to_numpy(float)
        standin[name] = {
            "n_hidden_cells_imputed": int(ok.sum()), "n_total": int(len(st)),
            "n_scaled_by_severity": int(sd["si_sev"].notna().sum()),
            "scoreboard_race_rate": board,
            "standin_vs": E.compare(sd, "standin", ["pooled", "zero", "const", "scalar"]),
            "shipped_interval_coverage_of_race_rate": float(sd["si_hit_shipped"].mean()),
            "shipped_multiples_used": {"scaled": T.STANDIN_INTERVAL_MULTIPLE,
                                       "unscaled": T.POOLED_INTERVAL_MULTIPLE},
            "calibrated_multiple_95": cal95, "calibrated_multiple_90": cal90,
            "mean_interval_halfwidth_shipped": float(((sd["si_hi"] - sd["si_lo"]) / 2).mean()),
            "mean_abs_error": float((sd["pred_standin"] - sd["race_rate"]).abs().mean()),
            "mean_shipped_multiple": float(np.nanmean(shipped_mult)),
        }
        # measured-cell intervals as transfer.json draws them: 1.96*se*|factor|.
        mm = lo.copy()
        mm["unit"] = 1.96 * mm["se_p"] * mm["factor_used"].abs()
        mm["hit"] = (mm["race_rate"] - mm["pred_scalar"]).abs() <= mm["unit"]
        standin[name]["measured_interval_coverage_as_drawn"] = float(mm["hit"].mean())
        standin[name]["measured_interval_calibration_95"] = E.interval_calibration(
            mm, "pred_scalar", "unit", 0.95)
        # The replacement: a half-width from the out-of-fold absolute errors
        # themselves (what 05_transfer.py now draws), coverage cross-validated.
        standin[name]["measured_interval_empirical_95"] = E.interval_calibration(
            mm.assign(unit_one=1.0), "pred_scalar", "unit_one", 0.95)
    res["identifiability"] = ident
    res["standins"] = standin

    tick("E/F done: identifiability + stand-ins")

    # ---- G/H. sessions, weights, design ----------------------------------
    t1 = time.time()
    res["session_skill"] = E.run_session_analysis()
    res["design_sweep"] = E.run_design_sweep()
    print(f"sessions + design sweep in {time.time() - t1:.0f}s")

    # ---- cells (so every number above can be recomputed by hand) ----------
    keep = ["event", "season", "C", "label", "practice_rate", "se_p", "n_runs_p", "race_rate",
            "n_runs_r", "hist_rate", "factor_used", "pred_scalar", "pred_naive", "pred_pooled"]
    res["cells_2026_loeo"] = loeo26[keep]
    res["cells_holdout_frozen"] = frozen[keep]
    res["cells_holdout_loeo"] = loeo_h[keep]

    res["verdicts"] = verdicts(res)
    res["headline"] = headline(res)
    res["audit"] = E.audit(
        "scripts/paper/p2_transfer.py",
        [E.laps_path(2026)] + [E.laps_path(s) for s in HOLDOUT] + [ROOT / "data" / "allocation.json"],
        extra={"holdout_seasons": list(HOLDOUT), "runtime_s": round(time.time() - t0, 1)},
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(E.jsonable(res), indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ({time.time() - t0:.0f}s)")
    print_summary(res)


def verdicts(r: dict) -> dict:
    """The pre-stated rules, applied mechanically."""
    v: dict = {}
    fz, lo, hl = r["holdout_frozen"], r["holdout_loeo"], r["s2026_loeo"]
    v["H1_frozen_scalar_beats_zero_pooled_naive"] = {
        b: beats(fz["scalar_vs"][b]) for b in ("zero", "pooled", "naive")
    }
    v["H1b_loeo_scalar_beats_zero_pooled_naive"] = {
        b: beats(lo["scalar_vs"][b]) for b in ("zero", "pooled", "naive")
    }
    v["H1c_frozen_scalar_beats_history"] = beats(fz["scalar_vs"]["history"])
    v["H2_nested_optimism_le_0.005"] = bool(r["nested"]["optimism_of_shipped"] <= 0.005)
    v["H3_compound_beats_scalar_in_both"] = {
        "frozen_holdout": beats(fz["pairs"]["compound_vs_scalar"]),
        "s2026_loeo": beats(hl["pairs"]["compound_vs_scalar"]),
    }
    ss = r["session_skill"]
    c = ss.get("2022-2025", {}).get("contrasts", {}).get("FP1-FP2", {})
    v["H4_fp2_beats_fp1_out_of_season"] = {
        "fp1_minus_fp2_rho_ci": c.get("rho_diff_ci"),
        "holds": bool(c.get("rho_diff_ci") and c["rho_diff_ci"][1] < 0),
    }
    for k, d in r["identifiability"].items():
        v[f"H5_{k}"] = {
            "rule1_beats_always": beats(d["rule1_vs_always"]),
            "eb_beats_rule1": beats(d["eb_vs_rule1"]),
            "rule1_loses_to_always": bool(d["rule1_vs_always"]["ci_lo"] > 0),
        }
    for k, d in r["standins"].items():
        v[f"H6_{k}"] = {
            "shipped_coverage": d["shipped_interval_coverage_of_race_rate"],
            "meets_85pct": bool(d["shipped_interval_coverage_of_race_rate"] >= 0.85),
            "multiple_for_95": d["calibrated_multiple_95"]["multiple_for_level"],
        }
    return v


def headline(r: dict) -> dict:
    fz, lo = r["holdout_frozen"], r["holdout_loeo"]

    def row(sec):
        sb = sec["scoreboard"]
        return {m: [round(sb[m]["mae"], 4), round(sb[m]["mae_ci_lo"], 4), round(sb[m]["mae_ci_hi"], 4)]
                for m in sb}

    return {
        "mae_ci_by_method__frozen_holdout_2022_2025": row(fz),
        "mae_ci_by_method__loeo_2022_2025": row(lo),
        "mae_ci_by_method__2026_loeo_tuned_on_eval": row(r["s2026_loeo"]),
        "nested_2026_mae": r["nested"]["mae_nested"],
    }


def print_summary(r: dict) -> None:
    def show(title, sec):
        print("\n" + "=" * 78 + f"\n{title}  ({sec['n_cells']} cells, {sec['n_weekends']} weekends)")
        print(f"{'method':10s}{'MAE':>8s}{'95% CI':>18s}{'RMSE':>8s}{'rho':>7s}")
        for m, s in sec["scoreboard"].items():
            print(f"{m:10s}{s['mae']:8.4f}   [{s['mae_ci_lo']:.4f},{s['mae_ci_hi']:.4f}]"
                  f"{s['rmse']:8.4f}{s['spearman']:7.2f}")
        for o, c in sec.get("scalar_vs", {}).items():
            print(f"  scalar - {o:9s}: {c['mean_diff']:+.4f} [{c['ci_lo']:+.4f},{c['ci_hi']:+.4f}] "
                  f"DM p={c['dm_p']:.3f} flip p={c['signflip_p']:.3f}")

    show("2026 LOEO (design tuned on these cells)", r["s2026_loeo"])
    n = r["nested"]
    print(f"\nNESTED: mae {n['mae_nested']:.4f} vs shipped {n['mae_shipped_tuned_on_eval']:.4f}; "
          f"shipped chosen in {n['shipped_chosen_in_folds']}/12 folds")
    show("FROZEN HOLDOUT 2022-2025", r["holdout_frozen"])
    show("LOEO within 2022-2025", r["holdout_loeo"])
    print("\nVERDICTS")
    print(json.dumps(E.jsonable(r["verdicts"]), indent=1))


if __name__ == "__main__":
    main()

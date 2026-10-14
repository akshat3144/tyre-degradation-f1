"""Which practice session actually predicts the race.

    python scripts/12_session_skill.py

FP1, FP2 and FP3 were pooled as one bucket until now, on the unexamined
assumption that a practice lap is a practice lap. A mentor asked why FP2 was
not weighted more heavily, which turned out to be a question the data could
answer rather than a matter of convention.

For every (event, compound) cell we can measure in BOTH a practice session and
the race that followed, this scores the session's degradation against the
race's. Three sessions, one table, no tuning.

The numbers it prints are the ones used to justify ``data.session_weight.SESSION_WEIGHTS``. Read them as a prior, not a
finding: they are the 2026 cells the weights were chosen on, and out of season
(2022-2025) FP2 is not distinguishable from FP1 (paper/results/session_skill.json). ``tests`` re-runs the
comparison and fails if FP2 ever stops being the best predictor, so the weights
stay checked rather than assumed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tyredeg.config import PROCESSED
from tyredeg.data.session_weight import SESSION_WEIGHTS
from tyredeg.models.design import prepare
from tyredeg.validation.transfer import MIN_AGE_SPREAD_LAPS, cell_rates

warnings.filterwarnings("ignore")

#: Loosened from the pooled thresholds on purpose. One session holds a third of
#: the weekend's running, so the pooled bar would leave almost nothing to score
#: and the comparison would be decided by which session happened to clear it.
SESSION_MIN_RUNS = 2
SESSION_MIN_RUNS_PER_COMPOUND = 1


def paired_cells(laps: pd.DataFrame) -> pd.DataFrame:
    """Every cell measurable in one practice session and in its race."""
    race = prepare(laps, "race")
    truth = cell_rates(race, min_age_spread=MIN_AGE_SPREAD_LAPS).set_index(["event", "C"])["rate"]

    rows = []
    for ses in ("FP1", "FP2", "FP3", "S"):
        sub = laps[laps["session"].isin([ses, "R"])]
        p = prepare(sub, "practice", min_runs_per_compound=SESSION_MIN_RUNS_PER_COMPOUND)
        if p.empty:
            continue
        for _, r in cell_rates(p, min_runs=SESSION_MIN_RUNS).iterrows():
            key = (r["event"], r["C"])
            if key in truth.index:
                rows.append(
                    {
                        "session": ses,
                        "event": r["event"],
                        "C": r["C"],
                        "n_runs": int(r["n_runs"]),
                        "n_laps": int(r["n_laps"]),
                        "practice": float(r["rate"]),
                        "race": float(truth.loc[key]),
                    }
                )
    return pd.DataFrame(rows)


def score(d: pd.DataFrame) -> pd.DataFrame:
    """Correlation, transfer factor and error, per session."""
    out = []
    for ses, g in d.groupby("session"):
        # The factor is the least-squares scaling from practice to race with no
        # intercept, which is what the forecast actually applies. A NEGATIVE
        # factor means the session points the wrong way: its steeper cells are
        # the race's flatter ones.
        denom = float((g["practice"] ** 2).sum())
        factor = float((g["race"] * g["practice"]).sum() / denom) if denom > 0 else np.nan
        corr = g["practice"].corr(g["race"]) if len(g) > 2 else np.nan
        out.append(
            {
                "session": ses,
                "n_cells": len(g),
                "median_runs": int(g["n_runs"].median()),
                "median_laps": int(g["n_laps"].median()),
                "correlation": round(float(corr), 3) if pd.notna(corr) else None,
                "factor": round(factor, 4) if pd.notna(factor) else None,
                "mae": round(float(np.abs(g["practice"] * factor - g["race"]).mean()), 4)
                if pd.notna(factor)
                else None,
            }
        )
    return pd.DataFrame(out).set_index("session")


#: Weighting schemes worth comparing. Not a search space -- each is a position
#: someone could reasonably argue for, and the sweep says which is right.
SCHEMES: dict[str, dict[str, float]] = {
    "FP2 heavy (shipped)": {"FP1": 0.5, "FP2": 1.0, "FP3": 0.5, "S": 1.0},
    "equal": {"FP1": 1.0, "FP2": 1.0, "FP3": 1.0, "S": 1.0},
    "FP1+FP2 equal": {"FP1": 1.0, "FP2": 1.0, "FP3": 0.5, "S": 1.0},
    "FP1 heavy": {"FP1": 1.0, "FP2": 0.65, "FP3": 0.5, "S": 1.0},
    "FP3 down only": {"FP1": 1.0, "FP2": 1.0, "FP3": 0.25, "S": 1.0},
    # The Sprint only ever shares a weekend with FP1, so on a sprint weekend
    # its weight against FP1's is the whole question.
    "sprint down": {"FP1": 0.5, "FP2": 1.0, "FP3": 0.5, "S": 0.5},
    "sprint only": {"FP1": 0.05, "FP2": 1.0, "FP3": 0.5, "S": 1.0},
}


def weight_sweep(laps: pd.DataFrame) -> pd.DataFrame:
    """Transfer error under each weighting scheme.

    Mutates the module constant and restores it, which is ugly but honest:
    it exercises the real ``prepare`` rather than a copy of it that might
    drift from what ships.
    """
    from tyredeg.data import session_weight
    from tyredeg.validation.transfer import leave_one_event_out

    race = prepare(laps, "race")
    original = dict(session_weight.SESSION_WEIGHTS)
    rows = []
    try:
        for name, w in SCHEMES.items():
            session_weight.SESSION_WEIGHTS.clear()
            session_weight.SESSION_WEIGHTS.update(w)
            frame = prepare(laps, "practice")
            loo = leave_one_event_out(frame, race)
            rows.append(
                {
                    "scheme": name,
                    "mae": round(loo.mae_calibrated, 4),
                    "improvement_%": round(loo.improvement * 100, 1),
                    "factor": round(loo.factor, 4),
                    "n_cells": len(loo.table),
                    "_mae": loo.mae_calibrated,
                }
            )
    finally:
        session_weight.SESSION_WEIGHTS.clear()
        session_weight.SESSION_WEIGHTS.update(original)
    # Rank on the unrounded error, so two schemes that print the same four
    # decimals still sort by which one actually did better.
    return pd.DataFrame(rows).sort_values("_mae", kind="stable").drop(columns="_mae")


def best_scheme(laps: pd.DataFrame) -> str:
    """Name of the lowest-error scheme. Used by the tests."""
    return str(weight_sweep(laps).iloc[0]["scheme"])


def main() -> None:
    laps = pd.read_parquet(PROCESSED / "laps.parquet")
    d = paired_cells(laps)
    if d.empty:
        print("no paired practice/race cells; nothing to score")
        return

    tbl = score(d)
    print("=" * 72)
    print("WHICH PRACTICE SESSION PREDICTS THE RACE")
    print("Each cell is one (event, compound) measured in practice and again in")
    print("the race. Correlation is between the two. A session that predicts")
    print("nothing scores near zero however many laps it ran.")
    print("=" * 72)
    print(tbl.to_string())

    print("\nWEIGHTS IN USE:", SESSION_WEIGHTS)
    print()
    print("=" * 72)
    print("DOES THE WEIGHTING EARN ITS PLACE")
    print("The per-session correlation above is a DIAGNOSTIC, not the criterion.")
    print("Each session scores a DIFFERENT set of cells -- FP1 answers six where")
    print("FP2 answers ten -- so ranking their correlations rewards whichever one")
    print("skipped the hard ones. What decides the weighting is the end-to-end")
    print("error of the BLEND, which every scheme below computes over the same")
    print("cells, through the same pipeline that ships.")
    print("=" * 72)
    print(weight_sweep(laps).to_string(index=False))

    print("\nPER-CELL DETAIL")
    print(d.sort_values(["session", "event"]).to_string(index=False))

    across_seasons()


def across_seasons() -> None:
    """The same question on every cached season, with intervals.

    The 2026 table above is ten cells per session, the same cells the weights
    were chosen on. This re-measures every session over 2022-2025 (label
    compounds, design frozen) with weekend-clustered bootstrap CIs and PAIRED
    contrasts on the cells two sessions share. The Sprint cannot be tested out
    of season: no Sprint session is cached for 2022-2025.
    """
    import json

    from tyredeg.config import ROOT
    from tyredeg.paper import transfer_eval as E

    res = E.run_session_analysis()
    for scope in ("2026", "2022-2025"):
        if scope not in res:
            continue
        print("\n" + "=" * 78)
        print(f"{scope}: per-session skill, weekend-clustered 95% CIs")
        print("=" * 78)
        rows = []
        for ses, v in res[scope]["per_session"].items():
            rows.append({"session": ses, "cells": v["n_cells"], "weekends": v["n_weekends"],
                         "pearson": round(v["pearson"], 3),
                         "ci": [round(x, 2) for x in v["pearson_ci"]],
                         "loeo_mae": round(v["loeo_mae_scalar"], 4),
                         "naive_mae": round(v["loeo_mae_naive"], 4)})
        print(pd.DataFrame(rows).to_string(index=False))
        print("\npaired contrasts on shared cells (rho difference, MAE difference):")
        for k, c in res[scope]["contrasts"].items():
            if "rho_diff" not in c:
                print(f"  {k}: only {c['n_common']} shared cells")
                continue
            print(f"  {k:8s} n={c['n_common']:3d}  rho {c['rho_a']:+.2f} vs {c['rho_b']:+.2f}  "
                  f"diff {c['rho_diff']:+.2f} [{c['rho_diff_ci'][0]:+.2f},{c['rho_diff_ci'][1]:+.2f}]  "
                  f"MAE diff {c['mae_diff_a_minus_b']:+.4f} "
                  f"[{c['mae_diff_ci'][0]:+.4f},{c['mae_diff_ci'][1]:+.4f}]")
    for scope in ("weights_2026", "weights_2022-2025"):
        if scope not in res:
            continue
        print("\n" + "=" * 78)
        print(f"{scope}: weighting schemes vs shipped, on cells both answer")
        print("=" * 78)
        rows = [{"scheme": k, "cells": v["n_cells_own"], "mae": round(v["mae_own"], 4),
                 "diff": round(v["diff_vs_shipped"], 4),
                 "ci": [round(x, 4) for x in v["diff_ci"]]} for k, v in res[scope].items()]
        print(pd.DataFrame(rows).to_string(index=False))
    res["audit"] = E.audit("scripts/12_session_skill.py",
                           [E.laps_path(2026)] + [E.laps_path(s) for s in E.HOLDOUT_SEASONS])
    out = ROOT / "paper" / "results" / "session_skill.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(E.jsonable(res), indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

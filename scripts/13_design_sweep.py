"""How much practice data are we throwing away, and does keeping it help?

    python scripts/13_design_sweep.py

THE QUESTION

Madrid 2026 held 663 practice laps across three sessions and the pipeline once
used 42 of them. That is either correct strictness or wasted data, and the
argument cannot be settled by preference. So: build the practice frame several
ways and score each against the races that have already run -- practice-to-race
mean absolute error, leave-one-event-out.

WHAT CHANGED, AND WHY THE OLD OUTPUT WAS WRONG

This script used to print "parity with design.prepare(): MISMATCH (2815 vs 2339
laps)" and to claim the warm-up trim took the error from 0.0819 to 0.0491.
Neither held. Its ``current`` variant was the pipeline BEFORE the run-keeping,
gap-bridging, warm-up and Sprint changes were merged into ``design.prepare``
-- 2339 laps against the 2815 that ship -- so it compared every candidate with
a design nobody runs. The builder now lives in ``tyredeg.paper.transfer_eval``
(``PracticeDesign`` / ``build_practice``), its default IS the shipped
pipeline, and a test fails if the two ever differ. The retired pipeline is kept
as the ``legacy`` variant so the old claim can be re-measured rather than
repeated.

HOW IT IS SCORED NOW

Every variant is compared with the shipped design on the cells BOTH can answer,
with a weekend-clustered bootstrap interval on the paired difference, because a
variant that refuses the hard cells wins on raw MAE while knowing less. The same
sweep is run on 2022-2025 (label compounds), which the design was never tuned
on: a warm-up trim that only helps the season it was chosen on is a fit, not a
finding.

Results go to ``paper/results/design_sweep.json``.
"""

from __future__ import annotations

import json
import warnings

import pandas as pd

from tyredeg.config import ROOT
from tyredeg.paper import transfer_eval as E

warnings.filterwarnings("ignore")

OUT = ROOT / "paper" / "results" / "design_sweep.json"

#: Kept so older callers can still ask for a variant by flag; the defaults are
#: the SHIPPED pipeline, not the legacy one.
VARIANTS = E.DESIGN_GRID


def build(laps: pd.DataFrame, **kw) -> pd.DataFrame:
    """One practice frame. With no arguments, exactly ``design.prepare``'s."""
    from dataclasses import replace

    return E.build_practice(laps, replace(E.SHIPPED, **kw))


def table(sweep: dict) -> pd.DataFrame:
    rows = []
    for name, v in sweep.items():
        ci = v.get("diff_ci") or [None, None]
        rows.append({
            "variant": name,
            "cells": v["n_cells_own"],
            "mae_own": round(v["mae_own"], 4),
            "common_w_shipped": v["n_common_with_shipped"],
            "mae_there": round(v.get("mae_on_common", float("nan")), 4),
            "mae_shipped_there": round(v.get("mae_shipped_on_common", float("nan")), 4),
            "diff": round(v.get("diff_vs_shipped", float("nan")), 4),
            "ci_lo": round(ci[0], 4) if ci[0] is not None else None,
            "ci_hi": round(ci[1], 4) if ci[1] is not None else None,
            "flip_p": round(v.get("signflip_p", float("nan")), 3),
            "mae_all_common": round(v.get("mae_common_all", float("nan")), 4),
        })
    return pd.DataFrame(rows)


def main() -> None:
    res = E.run_design_sweep()
    par = res["parity"]
    ok = par["identical"]
    print(f"parity with design.prepare(): {'ok' if ok else 'MISMATCH'} "
          f"({par['shipped_laps']} shipped vs {par['rebuilt_laps']} rebuilt laps; "
          f"the retired pipeline is {par['legacy_laps']})")
    if not ok:
        raise SystemExit("the sweep is not comparing like with like; fix build_practice first")

    for scope in ("2026", "2022-2025"):
        if scope not in res:
            continue
        print("\n" + "=" * 100)
        print(f"{scope}: each variant vs the SHIPPED design on the cells both can answer")
        print("diff = MAE(variant) - MAE(shipped); negative favours the variant. "
              "CI is a weekend-clustered bootstrap.")
        print("=" * 100)
        print(table(res[scope]).to_string(index=False))

    leg = res["2026"]["legacy"]
    print("\nTHE OLD HEADLINE, RE-MEASURED (legacy -> shipped, 2026, common cells)")
    print(f"   legacy  {leg['mae_on_common']:.4f}   shipped {leg['mae_shipped_on_common']:.4f}   "
          f"over {leg['n_common_with_shipped']} cells; "
          f"diff CI [{leg['diff_ci'][0]:+.4f}, {leg['diff_ci'][1]:+.4f}]")
    print("   (the old text said 0.0819 -> 0.0491; that pairing no longer exists)")

    res["audit"] = E.audit("scripts/13_design_sweep.py",
                           [E.laps_path(2026)] + [E.laps_path(s) for s in E.HOLDOUT_SEASONS])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(E.jsonable(res), indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

"""Practice to race: the deliverable the brief names.

    python scripts/05_transfer.py

Fits degradation on Friday practice, predicts Sunday race pace, and reports the
error. Leave-one-event-out, so the calibration never sees the event it predicts.

READ THE NUMBER WITH ITS BASELINES. The "improvement" over the naive
Sunday=Friday forecast is the improvement over the weakest comparison there is:
predicting zero, or the median of the other weekends' race rates, scores about
as well on these cells. The honest, out-of-season evaluation -- frozen design,
2022-2025 holdout, nested selection, baseline ladder, paired intervals -- is
``scripts/paper/p2_transfer.py`` and writes ``paper/results/transfer.json``.
This script writes the simple transfer artifact ``data/artifacts/transfer.json``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tyredeg.artifacts import schema
from tyredeg.artifacts.schema import Interval, TransferArtifact, TransferRow
from tyredeg.config import PROCESSED
from tyredeg.models.design import prepare
from tyredeg.validation.transfer import (
    MIN_AGE_SPREAD_LAPS,
    cell_rates,
    empirical_half_width,
    leave_one_event_out,
)

warnings.filterwarnings("ignore")


def _interval(mean: float, half_width: float) -> Interval:
    """A mean with a symmetric band, rounded and guaranteed ordered.

    Rounding each bound independently can still invert a band narrower than the
    rounding step, so the bounds are clamped around the mean afterwards.
    """
    lo, hi = round(mean - half_width, 5), round(mean + half_width, 5)
    m = round(mean, 5)
    return Interval(mean=m, lo=min(lo, m), hi=max(hi, m))


def main() -> None:
    laps = pd.read_parquet(PROCESSED / "laps.parquet")
    practice = prepare(laps, "practice")
    race = prepare(laps, "race")

    # ---- what the design can and cannot support ---------------------------
    all_race = cell_rates(race)
    ok_race = cell_rates(race, min_age_spread=MIN_AGE_SPREAD_LAPS)
    dropped = len(all_race) - len(ok_race)

    print("=" * 72)
    print("IDENTIFIABILITY -> the race design compares cars at the same lap, so")
    print("it needs them on DIFFERENT tyre ages. Where a field runs a")
    print("synchronised strategy, that variation vanishes.")
    print("=" * 72)
    print(f"   race cells with enough runs        : {len(all_race)}")
    print(f"   also with enough tyre-age spread   : {len(ok_race)}")
    print(f"   refused as unidentifiable          : {dropped}")
    thin = all_race[all_race["age_spread"] < MIN_AGE_SPREAD_LAPS]
    if len(thin):
        print("\n   refused:")
        for _, c in thin.iterrows():
            print(f"     {c['event'].replace(' Grand Prix', ''):12s} {c['C']}  "
                  f"spread {c['age_spread']:.2f} laps  ->  would have reported {c['rate']:+.4f} s/lap")

    # ---- transfer ----------------------------------------------------------
    res = leave_one_event_out(practice, race)
    t = res.table

    print("\n" + "=" * 72)
    print("PRACTICE -> RACE, leave-one-event-out")
    print("=" * 72)
    print(f"   {'event':12s}{'C':>4s}{'practice':>11s}{'race':>10s}{'naive err':>11s}{'calib err':>11s}")
    for _, r in t.iterrows():
        print(f"   {r['event'].replace(' Grand Prix', ''):12s}{r['C']:>4s}"
              f"{r['practice_rate']:11.4f}{r['race_rate']:10.4f}"
              f"{r['naive_err']:11.4f}{r['calibrated_err']:11.4f}")

    print(f"\n   MAE naive      : {res.mae_naive:.4f} s/lap   (assume Sunday = Friday)")
    print(f"   MAE calibrated : {res.mae_calibrated:.4f} s/lap")
    print(f"   improvement    : {res.improvement:+.1%}")
    # The comparisons the improvement figure above does not show.
    y = t["race_rate"]
    zero = float(y.abs().mean())
    const = float(
        np.mean(
            [
                abs(v - y[t["event"] != e].median())
                for e in t["event"].unique()
                for v in y[t["event"] == e]
            ]
        )
    )
    print(f"   MAE predict-zero             : {zero:.4f} s/lap")
    print(f"   MAE median of other weekends : {const:.4f} s/lap   (a constant: no practice at all)")
    print("   -> the naive forecast is the weakest baseline; see scripts/paper/p2_transfer.py")
    print(f"\n   practice -> race factor: {res.factor:.3f}")
    print(f"   i.e. a race degrades at about {res.factor * 100:.0f}% of its practice rate")

    neg = t[t["race_rate"] < 0]
    if len(neg):
        print(f"\n   !! {len(neg)} cell(s) have a NEGATIVE race rate, which is not physical:")
        for _, r in neg.iterrows():
            print(f"     {r['event'].replace(' Grand Prix', ''):12s} {r['C']}  {r['race_rate']:+.4f} s/lap")
        print("   Cause not established. Dropping the opening laps of each stint")
        print("   does not fix it, so post-pit traffic is ruled out. Traffic more")
        print("   generally is the remaining candidate and the one confounder from")
        print("   the brief we have not yet added.")

    # ---- write -------------------------------------------------------------
    # The band is the larger of the practice slope's own sampling error and the
    # half-width that would have covered 95% of these out-of-fold errors. The
    # first alone covered 24% of the realised race rates; see
    # transfer.empirical_half_width.
    emp = empirical_half_width(t["calibrated_err"])
    print(f"\n   interval half-width floor (95% of out-of-fold errors): {emp:.4f} s/lap")
    rows = [
        TransferRow(
            event=r["event"],
            compound=r["C"],
            label=None,
            # abs() on the factor. Scaling a standard error by a multiplier
            # scales its MAGNITUDE: SD(aX) = |a|.SD(X). Without it a negative
            # calibration factor produced lo > hi and Interval refused it,
            # taking the whole transfer stage down.
            predicted=_interval(
                r["calibrated_pred"],
                max(1.96 * (r["se_practice"] or 0) * abs(r["factor_used"]), emp),
            ),
            actual=round(r["race_rate"], 5),
            abs_error=round(r["calibrated_err"], 5),
        )
        for _, r in t.iterrows()
    ]
    schema.write(
        "transfer",
        TransferArtifact(rows=rows, mae=round(res.mae_calibrated, 5), is_forecast=False),
    )
    print("\nwrote transfer.json")


if __name__ == "__main__":
    main()

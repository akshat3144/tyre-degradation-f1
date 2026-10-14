"""Run every validation check and write the artifacts.

    python scripts/04_validate.py

Four questions, in order of how much they matter:

  1. How much data does this need?  (power -- simulated from the fitted mixed model)
  2. Is the uncertainty honest?     (calibration -- of the RATES, not just of laps)
  3. Do we score the same way they do?  (CRPS cross-check against R)
  4. Can we separate the compounds? (from the fitted model)

The numbers a paper would quote are produced, with their audit trail, by
``scripts/paper/p3_model.py``; this script writes the two web artifacts from the
SAME functions and the same seed, so the two cannot disagree.
"""

from __future__ import annotations

import os
import warnings

import pandas as pd

from tyredeg.artifacts import schema
from tyredeg.artifacts.schema import (
    CalibrationArtifact,
    CalibrationPoint,
    PowerArtifact,
    PowerPoint,
)
from tyredeg.config import PROCESSED
from tyredeg.models.design import prepare
from tyredeg.models.mixed import fit_degradation
from tyredeg.paper import model_checks as mc
from tyredeg.validation import power as P
from tyredeg.validation.calibration import RATE_LEVELS, leave_one_run_out, predictive_coverage
from tyredeg.validation.scoring import crosscheck_against_r

warnings.filterwarnings("ignore")

#: The benchmark fitted one driver at one race: three stints.
BENCHMARK_STINTS = 3

#: Shared with scripts/paper/p3_model.py so the artifact and the paper agree.
SEED = 20261030
N_SIMS = 100
WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))


def main() -> None:
    laps = pd.read_parquet(PROCESSED / "laps.parquet")
    race = prepare(laps, "race").reset_index(drop=True)
    practice = prepare(laps, "practice")
    fit = fit_degradation(race, quadratic=False, context="race", circuit_effects=True)
    n_race = int(race["run_id"].nunique())
    n_prac = int(practice["run_id"].nunique())

    # ---- 1. power ----------------------------------------------------------
    print("=" * 70)
    print("1. POWER — simulated from the fitted mixed model")
    print("=" * 70)
    pw = mc.power_section(fit.variance, sum(fit.n_runs.values()), WORKERS, SEED, N_SIMS,
                          welch_noise=P.estimate_noise(race))
    bal = pw["curves"][f"balanced_effect_{P.DEFAULT_EFFECT}"]
    spl = pw["curves"][f"split_effect_{P.DEFAULT_EFFECT}"]
    print(f"   effect {P.DEFAULT_EFFECT:.3f} s/lap · 20-lap stints · 13 circuits\n")
    print("   balanced design (both compounds at every circuit):")
    for n, p in zip(bal["n_stints"], bal["power_wald_z"], strict=True):
        print(f"     {n:5d} stints  {p:5.1%}  {'#' * int(p * 40)}")
    cross = bal["crossing_80pct_wald_z"]
    print(f"\n   80% crossing, interpolated: {cross['crossing']} ± {cross['mc_se']} (MC SE) stints")
    print(f"   original Welch calculation: grid point {pw['original_welch_calculation']['grid_point_returned_by_n_for']},"
          f" interpolated {pw['original_welch_calculation']['interpolated_crossing']}")
    print(f"   split design (compounds at different circuits): power {spl['power_wald_z']} "
          f"-- ceiling as stints -> infinity {pw['ceiling_split_design_power'][str(P.DEFAULT_EFFECT)]}")
    print(f"   Wald z size at zero effect: {pw['type_I_error_at_zero_effect']}")
    print(f"   our race ({n_race} stints), practice ({n_prac} stints); smallest effect with 80% "
          f"power at our race size: {pw['min_detectable_effect_80pct_s_per_lap']}")

    # ---- 2. calibration ----------------------------------------------------
    print("\n" + "=" * 70)
    print("2. CALIBRATION — lap level, then the rates themselves")
    print("=" * 70)
    cals = {}
    for name, df in (("race", race), ("practice", practice)):
        cal = leave_one_run_out(df)
        cals[name] = cal
        print(f"\n   LAP-LEVEL {name}: {cal.n:,} held-out laps, {cal.n_runs} runs")
        for nom, emp in zip(cal.levels, cal.empirical, strict=True):
            flag = "" if abs(nom - emp) < 0.05 else "  <-"
            print(f"     {nom:5.0%} -> {emp:6.1%}  ({emp - nom:+.1%}){flag}")
        print(f"     80% coverage {cal.coverage_80:.1%} · {cal.verdict()}  "
              f"(held-out laps; says little about the rates)")

    cells = mc.leave_one_event_out_cells(race, None, WORKERS)
    rate_cal = predictive_coverage(cells, RATE_LEVELS)
    nocirc = predictive_coverage(cells, RATE_LEVELS, include_circuit=False)
    print(f"\n   RATE-LEVEL leave-one-event-out: {len(cells)} event-compound cells")
    for lv in rate_cal["coverage"]:
        a, b = rate_cal["coverage"][lv], nocirc["coverage"][lv]
        print(f"     nominal {float(lv):.0%}: with circuit variance {a['coverage']:.1%} "
              f"[{a['wilson_95'][0]:.2f}, {a['wilson_95'][1]:.2f}] · without {b['coverage']:.1%}")

    # ---- 3. scoring --------------------------------------------------------
    print("\n" + "=" * 70)
    print("3. SCORING — do we compute CRPS the same way the benchmark did?")
    print("=" * 70)
    x = crosscheck_against_r()
    if x.get("available"):
        for k in ("normal", "ensemble"):
            d = x[k]
            print(f"   {k:9s} python {d['python']:.8f}  R {d['r']:.8f}  diff {d['diff']:.1e}")
        agree = all(x[k]["diff"] < 1e-6 for k in ("normal", "ensemble"))
        print(f"   identical to 1e-6: {agree}  -> the comparison is genuinely like-for-like")
    else:
        print(f"   R unavailable ({x.get('reason')}) — cross-check skipped")

    # ---- 4. separation -----------------------------------------------------
    print("\n" + "=" * 70)
    print("4. SEPARATION — can we tell the compounds apart?")
    print("=" * 70)
    for c in fit.ordered:
        r = fit.rates[c]
        print(f"   {c}  {r.mean:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}]  width {r.hi - r.lo:.4f}")
    print(f"\n   separated pairs (non-overlapping 95% intervals): {fit.separated_pairs() or 'none'}")
    print("   benchmark, for scale: Hard 0.054 [0.004, 0.133], width 0.129")

    # ---- write -------------------------------------------------------------
    schema.write(
        "power",
        PowerArtifact(
            effect_size=P.DEFAULT_EFFECT,
            points=[PowerPoint(int(n), float(p)) for n, p in zip(bal["n_stints"], bal["power_wald_z"], strict=True)],
            # The interpolated crossing, not a power-of-two grid point. -1 when
            # the simulated curve never reaches 80%.
            n_for_80pct=int(round(cross["crossing"])) if cross["crossing"] else -1,
            benchmark_n=BENCHMARK_STINTS,
            ours_n=n_race,
            n_for_80pct_se=cross["mc_se"],
            power_ceiling_split_design=pw["ceiling_split_design_power"][str(P.DEFAULT_EFFECT)],
            design=(
                "simulated from the fitted mixed model (circuit, driver, run, lap variance), "
                "13 circuits, 20-lap stints, both compounds at every circuit; Wald z test"
            ),
        ),
    )
    cal = cals["race"]
    schema.write(
        "calibration",
        CalibrationArtifact(
            points=[
                CalibrationPoint(float(n), float(e), cal.n)
                for n, e in zip(cal.levels, cal.empirical, strict=True)
            ],
            coverage_80=round(cal.coverage_80, 4),
            context="race",
            rate_points=[
                CalibrationPoint(float(lv), round(d["coverage"], 4), len(cells))
                for lv, d in rate_cal["coverage"].items()
            ],
            rate_coverage_80=round(rate_cal["coverage"]["0.80"]["coverage"], 4),
            rate_n_cells=len(cells),
            note=(
                "points/coverage_80 are held-out LAPS (lap-to-lap scatter dominates, so this is an "
                "easy bar). rate_* is leave-one-event-out: does each circuit's own fitted rate fall "
                "in the pooled model's predictive interval for it."
            ),
        ),
    )
    print("\nwrote power.json and calibration.json")


if __name__ == "__main__":
    main()

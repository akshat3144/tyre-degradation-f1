"""Paper P7: simulation recovery study -- does Eq. 1 recover planted degradation under strategic pitting?

    python -u scripts/paper/p7_recovery.py --workers 10     # writes paper/results/recovery.json
    python -u scripts/paper/p7_recovery.py --quick          # a few replicates, prints, writes nothing

THE REFEREE'S QUESTION
    Does the estimator recover a KNOWN truth when pit-stop timing is strategic,
    and how do simpler estimators fail? Three pit-timing mechanisms (exogenous;
    pace-dependent; wear-triggered, i.e. selection on the outcome) x eleven
    estimators (a naive pooled slope, fuel-corrected within-run slopes at the
    right and +-20% wrong fuel coefficient, two cross-sectional OLS variants,
    Eq. 1 with the earlier approximate within-transform ``eq1_shipped`` and two
    Mundlak variants of it, and the PRIMARY estimator ``eq1_exact_within`` (every
    fixed-effect column demeaned exactly within (event, lap)) with its own Mundlak
    variant ``eq1_exact_within_mundlak``), over R replicates of a 13-event, 20-car, 55-lap season
    whose variance components are the shipped fit's.

    The design, the data-generating process, the estimators, the metrics and the
    expectations stated in advance are all in the docstring of
    ``src/tyredeg/paper/recovery.py``; that file is the specification. The
    design is rebuilt here from ``paper/results/model.json`` and
    ``data/allocation.json`` and must equal the module's defaults, so the
    committed constants and the committed fit cannot drift apart unnoticed.

OUTPUT
    paper/results/recovery.json: an audit block (seed, R, design, scenario
    parameters, runtime), the planted truth, per scenario x estimator x compound
    bias / RMSE / coverage with Monte Carlo SEs, the circuit-SD bias of the
    Eq. 1 fits, what each scenario did to the design (correlations), and the
    scored pre-stated expectations. Replicate r uses the r-th child of
    SeedSequence(seed) in every scenario, so the file does not depend on the
    worker count.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

import numpy as np

from tyredeg.config import ROOT
from tyredeg.paper import model_checks as mc
from tyredeg.paper import recovery as rec

warnings.filterwarnings("ignore")

SEED = 20261101
R_FULL = 500
R_QUICK = 8
OUT = ROOT / "paper" / "results" / "recovery.json"
MODEL = ROOT / "paper" / "results" / "model.json"
ALLOC = ROOT / "data" / "allocation.json"


def scenario_parameters(cfg: rec.DGP) -> dict:
    """The knobs of each pit-timing mechanism, as run."""
    return {
        "common": {
            "n_events": len(cfg.nominations), "n_cars": cfg.n_cars, "n_laps": cfg.n_laps,
            "p_three_stints": cfg.p_three_stints, "min_stint_laps": cfg.min_stint,
            "stop_jitter_sd_laps": cfg.stop_jitter_sd,
            "dropped_laps": "out-lap of every run and in-lap of every run but the last",
            "shared_random_draws_across_scenarios": True,
        },
        "exogenous": {
            "rule": "stop j of n at round(n_laps * j / n + N(0, jitter_sd^2)), independent of pace and wear",
        },
        "pace": {
            "rule": "as exogenous, plus every stop shifted by -shift * z laps (z = car pace in SDs, slower positive) "
                    "and P(3 stints) = logistic(logit(p_three_stints) + slope * z)",
            "shift_laps_per_sd_pace": cfg.pace_shift_laps_per_sd,
            "stop_count_logit_per_sd_pace": cfg.pace_stops_logit_per_sd,
        },
        "wear": {
            "rule": "stop when cumulative loss s*A*(A+1)/2 since the stop (s = the run's realised slope "
                    "b_c + d_e + v_r) crosses wear_ref_slope * L*(L+1)/2 * exp(N(0, threshold_sd^2)), "
                    "L = n_laps / n the planned length; clipped to the minimum stint; last stint runs to the flag",
            "wear_ref_slope": cfg.wear_ref_slope,
            "threshold_sd": cfg.wear_threshold_sd,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reps", type=int, default=None, help=f"replicates per scenario (default {R_FULL}; "
                                                          f"{R_QUICK} with --quick)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quick", action="store_true", help="few replicates; writes nothing unless --out is given")
    ap.add_argument("--scenarios", nargs="*", default=list(rec.SCENARIOS), choices=list(rec.SCENARIOS))
    ap.add_argument("--out", default=None, help="output path (default paper/results/recovery.json)")
    args = ap.parse_args()

    reps = args.reps if args.reps is not None else (R_QUICK if args.quick else R_FULL)
    scenarios = tuple(args.scenarios)
    cfg = rec.dgp_from_fit(json.loads(MODEL.read_text(encoding="utf-8")),
                           json.loads(ALLOC.read_text(encoding="utf-8")))
    if cfg != rec.DGP():
        sys.exit("design drift: model.json / allocation.json no longer match the DGP defaults in "
                 "tyredeg.paper.recovery; update the module (and its docstring) deliberately, then rerun")

    t0 = time.time()
    print(f"recovery study: {len(scenarios)} scenarios x {reps} replicates x {len(rec.ESTIMATORS)} estimators, "
          f"{args.workers} workers, seed {args.seed}", flush=True)
    results = rec.run_replicates(cfg, scenarios, reps, args.seed, workers=args.workers)
    seconds = time.time() - t0

    truth = cfg.truth
    summary = rec.summarise(results, truth, scenarios, true_circuit_sd=cfg.sd_circuit_slope)
    expectations = rec.check_expectations(summary)
    res = {
        "specification": "see the module docstring of src/tyredeg/paper/recovery.py (written before the run)",
        "truth_s_per_lap": truth,
        "truth_circuit_slope_sd_s_per_lap": cfg.sd_circuit_slope,
        "recovery_criteria": {"abs_bias_s_per_lap_at_most": rec.BIAS_TOLERANCE,
                              "coverage_band": list(rec.COVERAGE_BAND),
                              "note": "reading aids for 'recovered_all_compounds', not hypothesis tests"},
        "estimators": rec.ESTIMATORS,
        "primary_eq1_estimator": rec.PRIMARY_EQ1,
        "primary_eq1_note": "eq1_exact_within is Eq. 1 as now fitted (every fixed-effect column demeaned exactly "
                            "within (event, lap)); eq1_shipped is the earlier approximate construction, kept as "
                            "the baseline the correction is measured against; eq1_exact_within_mundlak adds the "
                            "run-mean covariates to the primary estimator",
        "results": summary,
        "dgp_diagnostics": rec.summarise_diagnostics(results, scenarios),
        "prestated_expectations": expectations,
        "estimator_errors": rec.error_counts(results, scenarios),
        "dgp": cfg.to_dict(),
        "scenario_parameters": scenario_parameters(cfg),
    }
    res["audit"] = mc.audit(
        "scripts/paper/p7_recovery.py", [MODEL, ALLOC], args.seed,
        {"replicates_per_scenario": reps, "scenarios": list(scenarios), "n_estimators": len(rec.ESTIMATORS),
         "workers": args.workers, "total_seconds": round(seconds, 1),
         "mean_seconds_per_replicate_scenario": round(float(np.mean([r["seconds"] for r in results])), 2),
         "seeding": "replicate r = r-th child of SeedSequence(seed), shared by all scenarios",
         "command": "python -u scripts/paper/p7_recovery.py" + (" --quick" if args.quick else "")
                    + f" --workers {args.workers}" + (f" --reps {args.reps}" if args.reps else "")})

    out = None
    if args.out:
        from pathlib import Path

        out = Path(args.out)
    elif not args.quick:
        out = OUT
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(mc.jsonable(res), indent=1), encoding="utf-8")
        print(f"wrote {out}  ({seconds:.0f}s)")

    print_table(summary, scenarios)
    for k, v in expectations.items():
        print(f"  {k}: {v['holds']}  {v['statement']}")


def print_table(summary: dict, scenarios) -> None:
    """Mean-over-compounds bias and coverage, plus the worst compound, per scenario."""
    for sc in scenarios:
        print(f"\n== {sc} ==   mean bias | mean |bias| | cov | cov(event) | worst compound: bias | min cov")
        for est, e in summary[sc].items():
            if not e.get("by_compound"):
                print(f"  {est:28s} no estimates")
                continue
            mo, w = e["mean_over_compounds"], e["worst_compound"]

            def f(x, fmt="{:+.4f}"):
                return "   -  " if x is None else fmt.format(x)

            print(f"  {est:28s} {f(mo['bias'])} {f(mo['mean_abs_bias'], '{:.4f}')} "
                  f"{f(mo.get('coverage'), '{:.3f}')} {f(mo.get('coverage_event_cluster'), '{:.3f}')}   "
                  f"{w['compound_largest_abs_bias']} {f(w['bias'])} {f(w.get('coverage'), '{:.3f}')}")


if __name__ == "__main__":
    main()

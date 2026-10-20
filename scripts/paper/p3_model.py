"""Paper P3: the core degradation model -- what survives a hostile audit.

    python scripts/paper/p3_model.py            # about 15 minutes, writes paper/results/model.json
    python scripts/paper/p3_model.py --quick    # a few minutes, small replicate counts, for development

WHAT WAS WRONG WITH THE SHIPPED CLAIMS
    * The race fit reported converged=False, a non-positive-definite Hessian and
      an optimum "on the boundary". It was an optimiser that stopped early, and
      it inflated the circuit-slope SD by roughly a factor of three.
    * Laps of a stint are serially correlated, so lap-level likelihoods
      overstate the evidence for everything. Run-level terms are now in the model.
    * "Circuit spread against compound spread, LR with a tiny p" was the range of
      shrunken per-circuit predictions against the range of fixed effects -- not
      an effect size -- and the test could not be reproduced.
    * "Pairs separated race by race, cells tighter by a median factor" had no
      producing script and used lap-level inference.
    * The power analysis assumed iid stint slopes, ignored circuit variance and
      quoted the power-of-two grid point rather than the 80% crossing.
    * The headline lap coverage is for held-out LAPS, a nearly tautological check.
    * The ablation compared label-pooled naive slopes with C-number slopes.

PRE-STATED EXPECTATIONS (written before the final numbers were produced)
    P1  The REML engine reproduces statsmodels to < 1e-3 s/lap on the shared model.
    P2  Run-level terms widen every compound interval of the circuit model.
    P3  The circuit SD is positive with a bootstrap LR p < 0.01. (The audit
        refuted the size claimed; whether it is positive at all was open.)
    P4  Within-event compound contrasts show the physically expected ordering
        (positive trend per step of softness), i.e. the "inverted" pooled
        ordering is a compound-circuit confound. A hypothesis: if the
        within-event trend is still negative it is reported as such.
    P5  Race-by-race separation falls under run-level inference.
    P6  Rate-level leave-one-event-out coverage at 80% is within 10 points of
        nominal, and the version WITHOUT circuit variance under-covers.
    P7  Simulated power at the benchmark-implied effect (0.006 s/lap) never
        reaches 80% in the split design, whatever the number of stints.

Every number the paper quotes from the core model is in model.json.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import pandas as pd

from tyredeg.config import PROCESSED, ROOT
from tyredeg.models.design import prepare
from tyredeg.models.mixed import fit_degradation
from tyredeg.paper import model_checks as mc
from tyredeg.validation.calibration import leave_one_run_out
from tyredeg.validation.power import estimate_noise

warnings.filterwarnings("ignore")

SEED = 20261030
OUT = ROOT / "paper" / "results" / "model.json"
CKPT = ROOT / "paper" / "results" / "model.checkpoint.json"


def _lap_level(design: pd.DataFrame) -> dict:
    cal = leave_one_run_out(design)
    return {
        "what": "held-out LAPS inside their predictive interval, leave-one-run-out; "
                "dominated by lap scatter, says little about the rates",
        "n_laps": int(cal.n), "n_runs": int(cal.n_runs),
        "coverage": {f"{n:.2f}": round(float(e), 4) for n, e in zip(cal.levels, cal.empirical, strict=True)},
        "coverage_80": round(cal.coverage_80, 4), "verdict": cal.verdict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip", nargs="*", default=[], help="section names to skip (partial output)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run just these sections and MERGE them into the existing model.json "
                         "(existing keys are left untouched); e.g. circuit_vs_event")
    args = ap.parse_args()

    q = args.quick
    b_null, b_alt, n_sims = (40, 40, 20) if q else (500, 400, 100)
    b_null_nested, b_alt_nested = (12, 12) if q else (500, 500)
    reps_x, size_reps_x = (10, 40) if q else (1000, 2000)
    b_boot_events = 20 if q else 500
    t0 = time.time()
    laps = pd.read_parquet(PROCESSED / "laps.parquet")
    design = prepare(laps, "race").reset_index(drop=True)
    res: dict = {"quick_mode": q}
    old: dict = {}
    if args.only is not None:
        old = json.loads(OUT.read_text(encoding="utf-8"))
        res = dict(old)
    timings: dict = {}

    def section(name, fn):
        if name in args.skip or (args.only is not None and name not in args.only):
            return None
        # Checkpoint BEFORE each section, so everything assigned so far is saved:
        # a late failure must not cost the hours before it.
        CKPT.write_text(json.dumps(mc.jsonable({"done": sorted(timings), **{
            k: v for k, v in res.items() if not isinstance(v, tuple)}}), indent=1), encoding="utf-8")
        t = time.time()
        print(f"[{time.time() - t0:6.0f}s] {name} ...", flush=True)
        out = fn()
        timings[name] = round(time.time() - t, 1)
        return out

    res["dataset"] = mc.dataset_audit(laps, design)

    shipped = fit_degradation(design, quadratic=False, circuit_effects=True)
    ords = shipped.ordered
    res["shipped_fit"] = {
        "specification": "within (event, lap); compound offset + compound x tyre-age slope + traffic; "
                         "random circuit slope, driver-in-event intercept, run intercept, run slope; REML",
        "rates_s_per_lap": {c: {"mean": round(v.mean, 5), "lo": round(v.lo, 5), "hi": round(v.hi, 5),
                                "n_runs": shipped.n_runs.get(c), "n_laps": shipped.n_laps.get(c)}
                            for c, v in shipped.rates.items()},
        "monotone_softer_degrades_faster": bool(shipped.is_monotone()),
        "separated_pairs_nonoverlap": shipped.separated_pairs(),
        "adjacent_contrasts_s_per_lap": {
            f"{ords[i + 1]}-{ords[i]}": {
                k: round(getattr(shipped.contrast(ords[i], ords[i + 1]), k), 5)
                for k in ("mean", "lo", "hi")}
            for i in range(len(ords) - 1)
        },
        "variance_sd": {k: round(v, 5) for k, v in shipped.variance.items()},
        "convergence": shipped.convergence,
        "n_obs": shipped.n_obs, "n_drivers": shipped.n_drivers,
        "total_loss_20_laps_s": {c: round(shipped.total_loss(c, 20), 3) for c in ords},
    }

    res["engine_agreement"] = section("engine_agreement", lambda: mc.engine_agreement(design))
    res["old_fit_reproduction"] = section("old_fit_reproduction", lambda: mc.old_fit_reproduction(design))
    res["inference_comparison"] = section("inference_comparison", lambda: mc.inference_comparison(design))

    circ = section("circuit_effect",
                   lambda: mc.circuit_effect(design, b_null, b_alt, args.workers, SEED))
    alts = full = meta = None
    if circ:
        res["circuit_effect"], alts, full, meta = circ

    res["compound_within_event"] = section("compound_within_event", lambda: mc.compound_within_event(design))
    res["per_race_separation"] = section("per_race_separation",
                                         lambda: mc.per_race_separation(design, args.workers))
    if alts is not None:
        res["rate_calibration"] = section(
            "rate_calibration",
            lambda: mc.rate_calibration(design, alts, meta, full, b_alt, args.workers, SEED))

    res["lap_level_calibration"] = section("lap_level_calibration", lambda: _lap_level(design))
    res["ablation"] = section("ablation", lambda: mc.ablation_table(laps, design, shipped))

    def seasons():
        data = {s: pd.read_parquet(PROCESSED / f"laps_{s}.parquet") for s in (2022, 2023, 2024, 2025)}
        data[2026] = laps
        return mc.season_robustness(data)

    res["season_robustness"] = section("season_robustness", seasons)
    res["power"] = section("power", lambda: mc.power_section(
        shipped.variance, sum(shipped.n_runs.values()), args.workers, SEED, n_sims,
        welch_noise=estimate_noise(design)))

    # In merge mode the sections above were not run: put back what they held.
    for k_ in list(res):
        if res[k_] is None and k_ in old:
            res[k_] = old[k_]

    # ---- additions after the referee round (new keys only; nothing above changes)
    def pooled():
        data = {s: pd.read_parquet(PROCESSED / f"laps_{s}.parquet") for s in (2022, 2023, 2024, 2025)}
        data[2026] = laps
        return mc.pooled_physical_design(data)

    pool_df = None
    if args.only is None or "circuit_vs_event" in args.only or "season_c_rates" in args.only:
        pool_df = pooled()
    cve = section("circuit_vs_event", lambda: mc.circuit_vs_event(
        pool_df, b_null_nested, b_alt_nested, args.workers, SEED))
    if cve is not None:
        res["circuit_vs_event"] = cve
    scr = section("season_c_rates", lambda: mc.season_c_rates(pool_df))
    if scr is not None:
        res["season_c_rates"] = scr

    def spread():
        a_, f_, m_ = (alts, full, meta) if alts is not None else mc.circuit_alts(
            design, b_null, b_alt, args.workers, SEED)
        return mc.compound_vs_circuit_2026(a_, f_, m_, design, args.workers, SEED, b_boot_events)

    cs = section("compound_spread_2026", spread)
    if cs is not None:
        res["compound_spread_2026"] = cs

    def pwx():
        ci = tuple(res["circuit_effect"]["circuit_slope_sd_s_per_lap"]["profile_likelihood_95"])
        return mc.power_extended(shipped.variance, ci, design, args.workers, SEED,
                                 reps=reps_x, size_reps=size_reps_x)

    px = section("power_extended", pwx)
    if px is not None:
        res["power_extended"] = px

    def mdex():
        ci = tuple(res["circuit_effect"]["circuit_slope_sd_s_per_lap"]["profile_likelihood_95"])
        n_runs = {c: v["n_runs"] for c, v in res["shipped_fit"]["rates_s_per_lap"].items()}
        return mc.min_detectable_effect(shipped.variance, ci, design, n_runs, args.workers, SEED,
                                        reps=reps_x)

    mde = section("min_detectable_effect", mdex)
    if mde is not None:
        res["min_detectable_effect"] = mde

    inputs = [PROCESSED / f"laps_{s}.parquet" for s in (2022, 2023, 2024, 2025)] + [PROCESSED / "laps.parquet"]
    audit_ = mc.audit(
        "scripts/paper/p3_model.py", inputs, SEED,
        {"replicates": {"circuit_lr_null": b_null, "circuit_alt_and_sbc": b_alt, "power_per_point": n_sims,
                        "nested_circuit_lr_null": b_null_nested, "nested_alt": b_alt_nested,
                        "event_cluster_bootstrap_2026": b_boot_events,
                        "power_extended_per_point": reps_x, "type_I_error": size_reps_x},
         "workers": args.workers, "section_seconds": timings,
         "total_seconds": round(time.time() - t0, 1),
         "command": "python scripts/paper/p3_model.py" + (" --quick" if q else "")
                    + (" --only " + " ".join(args.only) if args.only else "")})
    if args.only is None:
        res["audit"] = audit_
    else:
        # Each --only run adds its own provenance; earlier ones are kept.
        prev = old.get("audit_additions", {})
        if "script" in prev:  # a single un-keyed record from an earlier run
            prev = {"earlier_run": prev}
        res["audit_additions"] = {**prev, "+".join(args.only): audit_}
    if q:
        out = OUT.with_name("model.quick.json")
    elif args.skip and args.only is None:
        out = OUT.with_name("model.partial.json")
    else:
        out = OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mc.jsonable(res), indent=1), encoding="utf-8")
    CKPT.unlink(missing_ok=True)
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

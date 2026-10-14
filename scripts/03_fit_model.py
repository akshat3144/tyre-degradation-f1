"""Fit the degradation model and write real artifacts.

    python scripts/03_fit_model.py
    python scripts/03_fit_model.py --context practice

Race is the default because it has about four times the usable laps (11,039 race laps against 2,815 practice and sprint laps in the shipped fits). Practice is what
the brief asks for, and the two disagree -- which is itself a finding.

The artifacts written here (degradation, ablation, meta) all come from ONE fit in
ONE run, so they cannot describe different models. ``tests/test_paper_model.py``
checks the stored files against a fresh fit.
"""

from __future__ import annotations

import argparse
import warnings

import fastf1
import pandas as pd

from tyredeg.artifacts import schema
from tyredeg.config import PROCESSED, SEASON
from tyredeg.models import ablation
from tyredeg.models.design import prepare
from tyredeg.models.mixed import fit_degradation, fuel_sensitivity, to_artifact

warnings.filterwarnings("ignore")

#: Bumped whenever the model specification or the engine changes, so a stale
#: artifact is recognisable from meta.json alone.
MODEL_VERSION = "reml-v3-circuit-run"


def report(fit, df) -> None:
    print(f"\n{'=' * 68}")
    print(f"{fit.context.upper()}   {len(df):,} laps · {df.run_id.nunique()} runs · "
          f"{df.Driver.nunique()} drivers · quadratic={fit.quadratic}")
    print("=" * 68)
    header = f"{'':5s}{'rate s/lap':>12s}{'95% interval':>22s}{'width':>9s}{'runs':>6s}"
    if fit.quadratic:
        header += f"{'@15 laps':>10s}{'20-lap loss':>13s}"
    print(header)
    for c in fit.ordered:
        r = fit.rates[c]
        line = (f"  {c:3s}{r.mean:12.4f}   [{r.lo:+.4f}, {r.hi:+.4f}]{r.hi - r.lo:9.4f}"
                f"{fit.n_runs.get(c, 0):6d}")
        if fit.quadratic:
            line += f"{fit.rate_at(c, 15):10.4f}{fit.total_loss(c, 20):13.2f}"
        print(line)
    print(f"\n  monotone (softer degrades faster): {fit.is_monotone()}")
    print(f"  separated pairs: {fit.separated_pairs() or 'none'}")
    print(f"  converged: {fit.converged}  all checks ok: {fit.convergence.get('all_ok')}")
    if fit.variance:
        print("  variance SDs: " + ", ".join(f"{k} {v:.4f}" for k, v in fit.variance.items()))


def build_meta(laps: pd.DataFrame, race_design, race_fit, practice_design) -> schema.Meta:
    """Provenance that says exactly which events and sessions each fit read.

    ``n_laps_clean`` is every lap in the file, all sessions of all events.
    ``events`` is only the events the RACE fit used. They differ on purpose, and
    the extra fields say how: an event can be in the file with practice laps
    only (its race not yet run) and then it feeds the practice work, not this fit.
    """
    by = laps.groupby(["event", "session"]).size()
    laps_by = {e: {s: int(n) for s, n in g.droplevel(0).items()} for e, g in by.groupby(level=0)}
    race_events = sorted(laps.loc[laps["session"] == "R", "event"].unique())
    all_events = sorted(laps["event"].unique())
    scope = {
        "race_degradation": {
            "sessions": ["R"],
            "events": sorted(race_design["event"].unique()),
            "n_events": int(race_design["event"].nunique()),
            "n_laps": int(len(race_design)),
            "n_runs": int(race_design["run_id"].nunique()),
            "note": "race laps in long runs, after the field-pooled design's filters",
        },
    }
    if practice_design is not None and len(practice_design):
        scope["practice_degradation"] = {
            "sessions": sorted(practice_design["session"].unique()),
            "events": sorted(practice_design["event"].unique()),
            "n_events": int(practice_design["event"].nunique()),
            "n_laps": int(len(practice_design)),
            "n_runs": int(practice_design["run_id"].nunique()),
            "note": "race-simulation runs in practice and sprint sessions",
        }
    return schema.Meta.now(
        model_version=MODEL_VERSION,
        season=SEASON,
        events=sorted(race_fit.events and {e for evs in race_fit.events.values() for e in evs}),
        n_laps_clean=int(len(laps)),
        n_long_run_laps=int(race_fit.n_obs),
        n_runs=int(sum(race_fit.n_runs.values())),
        n_drivers=int(race_fit.n_drivers),
        fastf1_version=fastf1.__version__,
        is_real=True,
        events_in_dataset=all_events,
        practice_only_events=sorted(set(all_events) - set(race_events)),
        laps_by_event_session=laps_by,
        fit_scope=scope,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", choices=["race", "practice", "both"], default="both")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    laps = pd.read_parquet(PROCESSED / "laps.parquet")
    contexts = ["race", "practice"] if args.context == "both" else [args.context]

    fits, designs = {}, {}
    for ctx in contexts:
        df = prepare(laps, ctx)
        # Why each choice below.
        #
        # LINEAR, in either context. The slope is the average over the ages we
        # observe, which is what "s/lap lost" means to a reader; a quadratic's
        # coefficient is the tangent at age zero, which is not. Every fit that
        # drives a decision (06, 04) is linear, so the published
        # curve must be the one nobody has to reconcile with another. The
        # curvature a quadratic does find is negative -- decelerating wear, the
        # opposite of a cliff.
        #
        # PRACTICE: no compound offsets (practice_design centres each lap
        # within its run, which differences a compound's level away, leaving a
        # coefficient of zero with no standard error) and no run terms (the same
        # centring removes them). RACE: offsets ARE identified, because the
        # demeaning is by (event, lap) and cars on different compounds share a
        # cell; circuit slopes, driver-in-event and run intercept/slope are in.
        fit = fit_degradation(
            df,
            quadratic=False,
            context=ctx,
            with_offsets=(ctx == "race"),
            circuit_effects=(ctx == "race"),
        )
        report(fit, df)
        fits[ctx], designs[ctx] = fit, df

    if "practice" in contexts:
        print(f"\n{'=' * 68}")
        print("FUEL SENSITIVITY — practice assumes s/kg rather than fitting it")
        print("=" * 68)
        s = fuel_sensitivity(laps)
        print(s.pivot(index="compound", columns="s_per_kg", values="rate").round(4).to_string())
        spread = s.groupby("compound")["rate"].agg(lambda x: x.max() - x.min()).max()
        print(f"\n  largest change across the 0.030-0.035 range: {spread:.4f} s/lap")
        print("  -> the fuel assumption is not driving the result"
              if spread < 0.01 else "  -> the fuel assumption MATTERS; report it")

    if not args.no_write and "race" in fits:
        race_fit, race_df = fits["race"], designs["race"]
        if not race_fit.convergence.get("all_ok", False):
            # An artifact must never be written from a fit that does not
            # certify its own optimum.
            raise SystemExit(f"race fit did not pass its convergence checks: {race_fit.convergence}")

        art = to_artifact(race_fit)
        path = schema.write("degradation", art)
        print(f"\nwrote {path}")

        # ablation.json comes from the SAME fit, on the same laps, in the same
        # run: two artifacts describing one quantity have to share a fit.
        abl = ablation.build(laps, race_fit, context="race", design=race_df)
        path = schema.write("ablation", abl)
        print(f"wrote {path}")

        # meta.json too; anything describing the dataset is written by the stage
        # that reads the dataset.
        meta = build_meta(laps, race_df, race_fit, designs.get("practice"))
        path = schema.write("meta", meta)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

"""Turn the degradation curves into a pit-stop decision.

    python scripts/06_strategy.py
    python scripts/06_strategy.py --event "Hungarian Grand Prix"

Three inputs: degradation per compound, pit loss per circuit, and the fresh-tyre
pace gap between compounds. The first two are measured; the third is ASSUMED
(``strategy.optimise.PACE_STEP_S``) and the strategy output is conditional on it
-- ``paper/results/strategy.json`` carries a sensitivity sweep over it. Every
legal strategy is enumerated and scored, so the recommendation is exact rather
than the output of a search that might have stopped early.

WHAT THE UNCERTAINTY MEANS

The band on each plan's total time and the ``confidence`` come from a Monte Carlo
over the inputs we are unsure of: degradation rates (from their fitted
intervals), pit loss (from the bootstrap of the stops it was measured from) and
the assumed pace step. ``confidence`` is the probability that the recommended
STOP COUNT is the cheapest. It replaces a hard-coded +/-3 s and
``clamp(margin / 10)``, neither of which responded to how well anything was
known. It is the uncertainty of the optimiser's inputs, not of the race: traffic,
safety cars and track position are not in the model.
"""

from __future__ import annotations

import argparse
import logging
import warnings

import pandas as pd

from tyredeg.artifacts import schema
from tyredeg.artifacts.schema import Interval, StrategyArtifact, StrategyPlan
from tyredeg.config import ARTIFACTS, PROCESSED, SEASON
from tyredeg.data.allocation import compounds_for
from tyredeg.models.design import prepare
from tyredeg.models.mixed import fit_degradation
from tyredeg.strategy.decision import decide
from tyredeg.strategy.optimise import C_ORDER, PACE_STEP_S, PACE_STEP_SD_S, optimal_stint
from tyredeg.strategy.pitloss import estimate_all
from tyredeg.strategy.rates import resolve_rates
from tyredeg.strategy.support import POOL_FILE, bounds_excluding, read_pool
from tyredeg.strategy.uncertainty import DEFAULT_PIT_SD_S, N_DRAWS, RATE_FLOOR

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)


def _iv(mean: float, lo: float, hi: float, nd: int = 2) -> Interval:
    """An Interval that cannot violate its own ordering after rounding."""
    return Interval(mean=round(mean, nd), lo=round(min(lo, mean), nd), hi=round(max(hi, mean), nd))


def _clamp(n: int, c: str, caps: dict[str, int], floors: dict[str, int]) -> int:
    return max(floors.get(c, 0), min(n, caps.get(c, n)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--event", default="Hungarian Grand Prix")
    ap.add_argument("--laps", type=int, default=None)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--no-cap", action="store_true", help="allow stints beyond what the field has run")
    args = ap.parse_args()

    laps = pd.read_parquet(PROCESSED / "laps.parquet")
    race = prepare(laps, "race")
    try:
        practice = prepare(laps, "practice")
    except Exception:  # noqa: BLE001 -- a dataset with no practice is still usable
        practice = None
    # circuit_effects=True, because this script answers ONE named event. The
    # global fit hands every track the season average, which is how both this
    # and the API came to recommend the same stop count everywhere.
    fit = fit_degradation(race, quadratic=False, context="race", circuit_effects=True)

    ev = race[race["event"] == args.event]
    race_laps = args.laps or int(ev["LapNumber"].max())

    pl_table = estimate_all([args.event])
    if pl_table.empty:
        raise SystemExit(f"could not measure pit loss for {args.event!r}")
    pit_loss = float(pl_table["pit_loss_s"].iloc[0])
    pit_sd = float(pl_table["sd"].iloc[0]) if "sd" in pl_table else DEFAULT_PIT_SD_S

    # A strategy can only use the three compounds Pirelli nominated for THIS
    # weekend. Without this the optimiser happily proposed C1 and C2 at Hungary,
    # where the allocation was C3/C4/C5 -- tyres that were never at the circuit.
    allocation = compounds_for(args.event)
    if not allocation:
        raise SystemExit(f"no compound allocation known for {args.event!r}")
    nominated = set(allocation.values())
    label_of = {c: lab for lab, c in allocation.items()}
    ordered = [c for c in C_ORDER if c in nominated]

    # Gated rates: positive AND distinguishable from zero, with a fall-back to
    # the practice forecast. Never a bare sign test.
    rs = resolve_rates(fit, args.event, nominated, practice=practice, race=race)
    usable = rs.usable
    if len(usable) < 2:
        raise SystemExit(
            "fewer than two nominated tyres have an identified positive degradation rate: "
            + "; ".join(rs.notes)
        )
    caps, floors = ({}, {}) if args.no_cap else bounds_excluding(read_pool(ARTIFACTS / POOL_FILE), args.event, label_of, season=SEASON)
    caps = {c: v for c, v in caps.items() if c in usable}
    floors = {c: v for c, v in floors.items() if c in usable}

    print("=" * 72)
    print(f"{args.event} — {race_laps} laps")
    print("=" * 72)
    print(f"   pit loss {pit_loss:.1f}s ± {pit_sd:.1f}  (measured from {int(pl_table['n_stops'].iloc[0])} green stops)")
    print(f"   rates from the {rs.source} model; pace step {PACE_STEP_S:.2f}s ± {PACE_STEP_SD_S:.2f} (ASSUMED)")
    print(f"\n   {'compound':10s}{'degradation':>14s}{'± sd':>8s}{'best stint':>13s}{'ceiling':>9s}")
    for r in rs.rows:
        if r.excluded:
            print(f"   {r.compound:10s}{r.mean:+14.4f}{r.sd:8.3f}{'refused':>13s}   ({r.reason})")
        else:
            best = optimal_stint(r.compound, r.mean, pit_loss)
            print(f"   {r.compound:10s}{r.mean:+14.4f}{r.sd:8.3f}{min(best, caps.get(r.compound, best)):11d} laps"
                  f"{caps.get(r.compound, '-')!s:>9}")
    for n in rs.notes:
        print(f"   note: {n}")

    d = decide(
        race_laps, usable, rs.sd, ordered, pit_loss,
        pit_sd_s=pit_sd, step=1, max_stint=caps or None, floor_stint=floors or None, n_draws=N_DRAWS,
    )
    print(f"\n   {len(d.plans_all):,} legal strategies enumerated"
          " (2+ compounds, 5-lap minimum stints)")
    print(f"\n   {'plan':44s}{'total':>10s}{'vs best':>10s}{'P(best)':>9s}{'95% band':>20s}")
    top = d.rec.total_time
    for n in sorted(d.best):
        p, sc = d.best[n], d.scores[n]
        print(f"   {p.describe():44s}{p.total_time:10.1f}{p.total_time - top:+10.1f}"
              f"{sc.p_best:9.3f}{f'[{sc.total_lo:.0f}, {sc.total_hi:.0f}]':>20s}")

    print(f"\n   recommended: {d.rec.describe()}")
    print(f"   margin over the best alternative stop count: {d.margin_s:.1f}s")
    if d.crossover_pit_loss_s:
        print(f"   one-stop/two-stop crossover at pit loss {d.crossover_pit_loss_s:.1f}s "
              f"(this circuit is {pit_loss:.1f}s)")
    else:
        print("   no crossover between 15s and 35s of pit loss: the answer is not close")
    print("   P(each stop count is cheapest): "
          + ", ".join(f"{k}-stop {v:.0%}" for k, v in sorted(d.stop_prob.items())))
    print(f"   confidence {d.confidence:.0%}  (probability the recommended stop count is cheapest, "
          f"{d.n_draws} Monte Carlo draws over rates, pit loss and pace step)")

    if not args.no_write:
        stop_prob = {str(k): round(v, 3) for k, v in sorted(d.stop_prob.items())}
        alt = d.runner_up
        schema.write(
            "strategy",
            StrategyArtifact(
                event=args.event,
                pit_loss_s=round(pit_loss, 2),
                race_laps=race_laps,
                plans=[
                    StrategyPlan(
                        n_stops=p.n_stops,
                        compounds=list(p.compounds),
                        stint_lengths=list(p.stints),
                        total_time=_iv(d.scores[n].total_mean, d.scores[n].total_lo, d.scores[n].total_hi),
                        p_best=round(d.scores[n].p_best, 4),
                        regret_s=round(d.scores[n].regret_mean, 2),
                    )
                    for n, p in ((n, d.best[n]) for n in sorted(d.best))
                ],
                recommended_stops=d.rec.n_stops,
                rationale=(
                    f"{d.rec.describe()} is {d.margin_s:.1f}s faster than the best "
                    f"{alt.n_stops}-stop alternative at a measured pit loss of {pit_loss:.1f}s"
                    f" (± {pit_sd:.1f}). The recommended stop count is the cheapest in "
                    f"{d.confidence:.0%} of draws over the uncertain inputs."
                    + (f" The crossover sits at {d.crossover_pit_loss_s:.1f}s." if d.crossover_pit_loss_s else "")
                    if alt
                    else f"{d.rec.describe()}."
                ),
                confidence=round(d.confidence, 3),
                optimal_stint={
                    c: Interval(
                        mean=float(_clamp(optimal_stint(c, usable[c], pit_loss), c, caps, floors)),
                        lo=float(_clamp(optimal_stint(c, usable[c], pit_loss * 0.85), c, caps, floors)),
                        hi=float(_clamp(optimal_stint(c, usable[c], pit_loss * 1.15), c, caps, floors)),
                    )
                    for c in ordered if c in usable
                },
                stop_probabilities=stop_prob,
                rates_source=rs.source,
                uncertainty={
                    "n_draws": d.n_draws,
                    "propagated": {
                        "rates": "truncated normals, SD from the fitted interval, floor %.2f s/lap" % RATE_FLOOR,
                        "pit_loss": f"truncated normal, SD {pit_sd:.2f} s (bootstrap of the stops measured)",
                        "pace_step": f"truncated normal, mean {PACE_STEP_S} s, SD {PACE_STEP_SD_S} s (ASSUMED)",
                    },
                    "not_propagated": [
                        "circuit-slope uncertainty: the rate intervals are a lower bound",
                        "traffic, safety cars, track position, undercut",
                    ],
                    "total_time_band": "2.5%-97.5% of the plan's total over the draws; includes the pace-step "
                    "uncertainty common to every plan, so compare plans through p_best and regret_s",
                    "stint_ceilings": caps,
                    "stint_floors": floors,
                },
            ),
        )
        print("\nwrote strategy.json")


if __name__ == "__main__":
    main()

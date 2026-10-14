"""What does a pit stop cost, in each track state, across every season we hold?

    python scripts/10_pit_loss_by_status.py            # writes paper/results/pit_loss.json
    python scripts/10_pit_loss_by_status.py --rebuild  # re-read the races from FastF1's cache first

WHY THIS EXISTS

The strategy layer needs the cost of a stop under a neutralisation, and the first
version asserted that a neutralised stop costs 45% of a green-flag one. That number came from nowhere except the universal intuition
that a neutralised field makes stopping cheap. This script is what happens when
you check -- and the second version of the check rested on one season (2026), 48
VSC stops and a safety-car sample of 23 stops from two races. This version
re-measures on every season in the cache, 2022-2026.

THE CONSTRUCTION

Observed in-lap plus out-lap, against the NON-PITTING field on those same two
laps, plus the driver's habitual gap to that field. The reference matters twice:
under a neutralisation every car is slow, so a green-flag reference would charge
the stop for the safety car; and fuel burn and track evolution make late laps
quicker, so a race-wide reference would charge an early stop for them.

A stop's status is the most severe of its IN-lap and OUT-lap, with a red flag
outranking a safety car outranking a VSC outranking a yellow. (The first version
read the in-lap only, tested safety car first, and had no red flag at all, so a
red-flag stop fell through to "green".)

WHAT THE OUTPUT IS

``paper/results/pit_loss.json``: per-category medians with cluster-bootstrap
(over races) intervals and counts, the VSC and safety-car ratios to green with
intervals, the same split by season, the old-versus-new reference comparison,
queue behaviour, and between-season stability of a circuit's pit loss.

THE LIMITATION NO NUMBER FIXES

Teams pit under a safety car to gain TRACK POSITION. The optimiser minimises
total time and has no concept of position, so even a perfectly measured pit loss
would not make it a safety-car strategist.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

from tyredeg.config import ROOT
from tyredeg.paper import pit_loss_eval as E
from tyredeg.paper.strategy_meta import meta
from tyredeg.strategy import pitloss as PL
from tyredeg.strategy import racedata

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

OUT = ROOT / "paper" / "results" / "pit_loss.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="re-read the races from FastF1's cache")
    args = ap.parse_args()

    laps, _ = racedata.load(rebuild=args.rebuild)
    n_races = laps.groupby(["season", "event"]).ngroups
    print(f"{n_races} races, seasons {sorted(laps['season'].unique())}", flush=True)

    stops = E.all_stops(laps)
    cats = E.category_table(stops)
    ratio_vsc = E.event_paired_ratio(stops, "vsc")
    ratio_sc = E.event_paired_ratio(stops, "safety_car")
    by_season = E.per_season(stops)
    queue = E.queue_table(stops)
    bias = E.reference_bias(laps, stops)
    stab = E.circuit_stability(laps)
    trunc = E.truncation_effect(stops)

    print()
    print("=" * 84)
    print(f"{'status':12s}{'stops':>7s}{'races':>7s}{'median':>9s}{'95% CI':>16s}{'ratio':>8s}{'ratio 95% CI':>16s}  default?")
    print("=" * 84)
    for cat, r in cats.items():
        ci = f"[{r['median_ci95'][0]:.1f}, {r['median_ci95'][1]:.1f}]"
        ratio = r.get("ratio_to_green")
        rci = r.get("ratio_ci95")
        print(
            f"{cat:12s}{r['n_stops']:7d}{r['n_races']:7d}{r['median_s']:9.2f}{ci:>16s}"
            f"{(f'{ratio:.3f}' if ratio else '-'):>8s}"
            f"{(f'[{rci[0]:.2f}, {rci[1]:.2f}]' if rci else '-'):>16s}"
            f"  {r.get('usable_as_default', '-')}"
        )
    print("\n  by season:", json.dumps(by_season))
    print("\n  within-race VSC ratio:", ratio_vsc)
    print("  within-race SC ratio: ", ratio_sc)
    print("\n  reference bias (loss vs stop timing):")
    print("   old", bias["old_reference"])
    print("   new", bias["new_reference"])
    print("\n  circuit stability:", {k: v for k, v in stab.items() if k != "table"})

    vsc = cats.get("vsc", {})
    sc = cats.get("safety_car", {})
    # The safety-car ratio is the default when that category is usable; the VSC
    # ratio is the fallback. (The JSON key keeps its historical name.)
    if sc.get("usable_as_default"):
        chosen, default = "safety_car", sc["ratio_to_green"]
    elif vsc.get("usable_as_default"):
        chosen, default = "vsc", vsc["ratio_to_green"]
    else:
        chosen, default = None, None
    result = {
        "meta": meta(
            "scripts/10_pit_loss_by_status.py",
            "src/tyredeg/paper/pit_loss_eval.py",
            E.SEED,
            {
                "n_boot": E.N_BOOT,
                "plausible_green_range_s": [12.0, 40.0],
                "by_status_bounds_s": list(PL.SANITY_RANGE_S),
                "min_stops": E.MIN_STOPS,
                "min_races": E.MIN_RACES,
                "max_single_race_share": E.MAX_SINGLE_RACE_SHARE,
                "interval": "95% percentile cluster bootstrap over races",
            },
            [racedata.LAPS_PATH],
        ),
        "sample": {
            "n_races": int(n_races),
            "seasons": [int(s) for s in sorted(laps["season"].unique())],
            "n_stops_all_categories": int(len(stops)),
        },
        "by_category": cats,
        "truncation_effect": trunc,
        "by_season": by_season,
        "within_race_ratio": {"vsc": ratio_vsc, "safety_car": ratio_sc},
        "queue": queue,
        "reference_comparison": bias,
        "circuit_stability": stab,
        "console_default": {
            "fraction": default,
            "category": chosen,
            "vsc_ratio": vsc.get("ratio_to_green"),
            "safety_car_ratio": sc.get("ratio_to_green"),
            "basis": "all seasons, cluster bootstrap over races"
            if default
            else "no category met the usability rule; the console keeps its prior default and says so",
            "safety_car_usable": bool(sc.get("usable_as_default", False)),
            "filter_sensitivity": {
                "safety_car_ratio_old_filter": trunc["safety_car"]["old"]["ratio"],
                "safety_car_ratio_wide_filter": trunc["safety_car"]["new"]["ratio"],
                "safety_car_ratio_no_filter": trunc["safety_car"]["unfiltered"]["ratio"],
                "safety_car_stops_outside_sanity_filter": trunc["safety_car"]["stops_outside_sanity_filter"],
                "reading": "The safety-car ratio depends on the filter: about a quarter of safety-car stops measure "
                "more than 60 s (the non-pitting reference is itself erratic under a safety car), and keeping them "
                "moves the ratio above 1. The reported fraction is the wide-filter median ratio; treat it as "
                "'somewhat cheaper than green, not well determined', which its interval says too.",
            },
            "note": "The intervals include 1.0: a neutralised stop is at most modestly cheaper than a green one, "
            "not the 55% cheaper the first version of the console assumed, and the safety-car figure is "
            "sensitive to the outlier filter (see filter_sensitivity).",
        },
        "limitations": [
            "A stop's loss is measured against the median of cars that did not stop on those laps; "
            "under a neutralisation that median is itself set by the neutralisation.",
            "Lap status is per lap, not per sector: a lap that is part green and part neutralised is "
            "filed under the neutralisation.",
            "The optimiser minimises total time and has no concept of track position, which is why "
            "teams actually pit under a safety car.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

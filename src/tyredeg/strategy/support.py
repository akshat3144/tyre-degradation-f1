"""How long can a stint on each compound credibly be?

THE PROBLEM

Race-fit degradation rates are small for the softer tyres: a soft's measured
race wear is a fraction of what a practice push shows (race degradation is
about 38% of practice on matched, same-estimator MEDIUM and SOFT cells,
paper/results/management.json; whether that is drivers nursing the tyre is NOT
established there, and the telemetry check was null). Handed a rate that flat
and a pace advantage that is assumed to be constant, an optimiser concludes the soft is the better
tyre at every stint length and recommends running it for most of the race: C5 for
73 laps at Monaco, with a five-lap stint on another tyre bought only to make the
plan legal. No team has ever done that, and no data we hold supports it: the
rates are identified from stints a fraction of that length.

A linear degradation rate is a fact about the range of tyre ages it was
estimated on. Beyond that range the model is extrapolating, and an optimiser will
extrapolate with total confidence.

THE GUARD

Bound each compound's stint between a lower and an upper quantile (5th and 95th)
of the stint lengths the field actually ran on that compound label in OTHER
races (every season held, 2022-2026, from the raw laps). It is empirical, it is
out-of-sample by construction (the race being planned is never in the pool), and it asks only that a recommendation stay inside
what has been observed to be possible -- not that it match what was observed.

The CEILING is what stops the 73-lap soft. The FLOOR matters less often but
removes the other half of the same plan: the filler stint on the hardest tyre.
A plan buys its second compound as cheaply as the rules allow, and the cheapest
legal stint is five laps; no team runs a hard tyre for five laps, and the 5th
percentile of the GENUINE stints they do run (see ``observed_stints``; at least
seven laps whatever the pool says) is what rules it out.

It is a bound on extrapolation, not a physical tyre-life model: it says nothing
about when a tyre falls off a cliff, only that the model has no evidence past
where teams have been. ``paper/results/strategy.json`` evaluates it against the
uncapped plan and reports which one looks more like the field.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .optimise import MIN_STINT_LAPS

#: Default quantile. 0.95, because the maximum of a sample is an extreme order
#: statistic that one strategic oddity or a red-flag restart can move, while the
#: 95th percentile still admits the longest stints teams really chose.
QUANTILE = 0.95

#: Quantile of observed stint lengths used as the FLOOR. Symmetric with the
#: ceiling, and fixed in advance rather than tuned.
FLOOR_QUANTILE = 0.05

#: Never cap below this many laps: a ceiling under the stint length that was
#: actually identified would forbid the compound entirely.
MIN_CAP_LAPS = 10

#: A floor is never lower than the legal minimum plus this margin. The 5th
#: percentile of a pool is only trustworthy if the pool is clean, and the
#: evaluation (``paper.strategy_eval``) calls a plan "on the boundary" when its
#: shortest stint is within ONE lap of the minimum. Two laps of margin puts the
#: floor clear of that definition, so a floor can never be what leaves a filler
#: stint standing.
FLOOR_MARGIN_LAPS = 2
MIN_FLOOR_LAPS = MIN_STINT_LAPS + FLOOR_MARGIN_LAPS

#: Fraction of the race distance a car must complete for its stints to count.
MIN_RACE_SHARE = 0.9

#: A race with more than this share of laps on intermediates or wets is a wet
#: race; its stints say nothing about a dry tyre's life.
WET_SHARE = 0.02

#: FastF1 track-status digits for a safety car (4), red flag (5) and VSC (6, 7).
_NEUTRALISED_DIGITS = "4567"

_LABELS = ("SOFT", "MEDIUM", "HARD")

POOL_DEFINITION = (
    "Stints of cars that ran 90% of a dry, un-red-flagged race, kept only if GENUINE: they start at the "
    "race start or on a pit exit, end at a pit entry (or the flag), and touch no safety car, VSC or red "
    "flag. The floor pool also drops the final stint and any stint under the five-lap legal minimum."
)


def support_caps(
    stints: list[tuple[str, int]],
    quantile: float = QUANTILE,
    min_n: int = 30,
) -> dict[str, int]:
    """Per-label ceiling on stint length from ``(label, laps)`` observations.

    Labels with fewer than ``min_n`` observations get no cap (and so are not
    constrained), rather than a cap from a handful of stints.
    """
    by: dict[str, list[int]] = {}
    for label, n in stints:
        by.setdefault(label, []).append(int(n))
    return {
        label: max(MIN_CAP_LAPS, int(np.ceil(np.quantile(v, quantile))))
        for label, v in by.items()
        if len(v) >= min_n and label in _LABELS
    }


def support_floor(
    stints: list[tuple[str, int]],
    quantile: float = FLOOR_QUANTILE,
    min_n: int = 30,
) -> dict[str, int]:
    """Per-label FLOOR on stint length: a low quantile of what the field ran.

    The mirror of ``support_caps``. A five-lap stint exists in a plan only to
    satisfy the two-compound rule as cheaply as possible; teams do not run them.

    Never below ``MIN_FLOOR_LAPS`` (the legal minimum plus a margin). The pool
    this is meant to be fed is the GENUINE one (``observed_stints``): stints that
    start at the race start or a pit exit and end at a pit entry. Fed every
    FastF1 "Stint" instead -- which includes the fragments left by red-flag
    restarts and safety-car stop/starts -- the 5th percentile came out at 10, 3
    and 2 laps for HARD, MEDIUM and SOFT, under the legal minimum, so the floor
    bound nothing on the two softer tyres.
    """
    by: dict[str, list[int]] = {}
    for label, n in stints:
        by.setdefault(label, []).append(int(n))
    return {
        label: max(MIN_FLOOR_LAPS, int(np.floor(np.quantile(v, quantile))))
        for label, v in by.items()
        if len(v) >= min_n and label in _LABELS
    }


def observed_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """Every stint the field ran, with what is needed to tell a real one from a fragment.

    ``laps`` is the RAW race-lap frame (``strategy.racedata``): one row per car
    per lap with ``Stint``, ``Compound``, ``PitIn_s``, ``PitOut_s`` and
    ``TrackStatus``. It must be raw, not ``clean_laps`` output: cleaning drops
    every pit in- and out-lap and every non-green lap, so a stint counted from
    clean laps is short by exactly the laps that mark its ends.

    One row per (race, car, stint): ``season, event, label, n, ceiling_ok,
    floor_ok``.

    * ``n`` is the number of laps on the set, in- and out-lap included.
    * A stint is GENUINE when it starts at the race start or on a pit exit, ends
      at a pit entry (or the flag, for the last one) and touches no safety car,
      VSC or red flag. Anything else is a fragment of a restart or a stop/start
      and measures the race's interruptions, not the tyre.
    * ``ceiling_ok``: genuine, final stints included -- the final stint is
      usually the longest, which is exactly what a ceiling must not ignore.
    * ``floor_ok``: genuine, NOT the final stint (its length is set by the laps
      left, not by the tyre) and at least ``MIN_STINT_LAPS`` long (a shorter one
      is a repair, a penalty or a first-lap incident, and is a stint the
      optimiser could not plan anyway).
    * Only cars that went at least 90% of the distance, and only dry,
      un-red-flagged races.
    """
    cols = ["season", "event", "label", "n", "ceiling_ok", "floor_ok"]
    r = laps[laps["Stint"].notna()]
    if r.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (season, event), g in r.groupby(["season", "event"], sort=False):
        if g["TrackStatus"].astype(str).str.contains("5").any():
            continue
        if g["Compound"].isin(["INTERMEDIATE", "WET"]).mean() > WET_SHARE:
            continue
        race_len = g["LapNumber"].max()
        for _, car in g.groupby("Driver", sort=False):
            if car["LapNumber"].max() < MIN_RACE_SHARE * race_len:
                continue
            car = car.sort_values("LapNumber")
            stint_ids = sorted(car["Stint"].unique())
            for k, sid in enumerate(stint_ids):
                h = car[car["Stint"] == sid]
                label = h["Compound"].iloc[0]
                if label not in _LABELS:
                    continue
                first, last = h.iloc[0], h.iloc[-1]
                final = k == len(stint_ids) - 1
                starts = (k == 0 and h["LapNumber"].min() <= 1) or pd.notna(first["PitOut_s"])
                ends = final or pd.notna(last["PitIn_s"])
                status = "".join(h["TrackStatus"].astype(str))
                clean = not any(d in status for d in _NEUTRALISED_DIGITS)
                n = int(h["LapNumber"].nunique())
                genuine = bool(starts and ends and clean)
                rows.append(
                    (int(season), str(event), str(label), n, genuine,
                     bool(genuine and not final and n >= MIN_STINT_LAPS))
                )
    return pd.DataFrame(rows, columns=cols)


def bounds_excluding(
    stints, event: str, label_of: dict[str, str], season: int | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    """``(ceilings, floors)`` keyed by C-number, from every OTHER race's stints.

    ``label_of`` maps each nominated compound to its HARD/MEDIUM/SOFT label at
    this weekend. Both empty when there is nothing to learn from. ``season``
    restricts the exclusion to that season's race of that name (the pool holds
    several seasons of the same event); None excludes the name in every season.
    """
    if stints is None or len(stints) == 0:
        return {}, {}
    hold = stints["event"] == event
    if season is not None and "season" in stints.columns:
        hold &= stints["season"] == season
    other = stints[~hold]
    c, f = other[other["ceiling_ok"]], other[other["floor_ok"]]
    caps = support_caps(list(zip(c["label"], c["n"], strict=True)), QUANTILE)
    floors = support_floor(list(zip(f["label"], f["n"], strict=True)), FLOOR_QUANTILE)
    return (
        {k: caps[lab] for k, lab in label_of.items() if lab in caps},
        {k: floors[lab] for k, lab in label_of.items() if lab in floors},
    )


def bound_hits(
    compounds, stints, caps: dict[str, int], floors: dict[str, int]
) -> list[dict]:
    """The stints of a plan that sit ON a support bound, as ``{compound, laps, bound, limit}``.

    A stint at exactly its ceiling (or floor) was put there by the bound, not
    chosen by the model: the optimiser wanted to go further than the field ever
    has, and was stopped. That is worth saying out loud next to the plan, and not
    leaving to be noticed. ``compounds`` and ``stints`` are parallel sequences.
    """
    out = []
    for c, n in zip(compounds, stints, strict=True):
        n = int(n)
        if c in caps and n >= caps[c]:
            out.append({"compound": c, "laps": n, "bound": "ceiling", "limit": int(caps[c])})
        elif c in floors and n <= floors[c]:
            out.append({"compound": c, "laps": n, "bound": "floor", "limit": int(floors[c])})
    return out


# ---------------------------------------------------------------------------
# the pool, persisted
# ---------------------------------------------------------------------------
#
# A stint pool needs the raw laps of every season, not just the 2026 clean laps.
# So it is computed where the raw laps are (scripts/11_stint_pool.py) and
# committed as data/artifacts/stint_pool.json, which scripts/06_strategy.py reads.

POOL_FILE = "stint_pool.json"


def write_pool(stints: pd.DataFrame, path: Path) -> None:
    """Persist the pool compactly: lengths per race and label, split by the pool they feed."""
    out: dict[str, dict] = {}
    for (season, event, label), g in stints.groupby(["season", "event", "label"], sort=True):
        out.setdefault(f"{int(season)}|{event}", {})[label] = {
            "ceiling": sorted(int(n) for n in g.loc[g["ceiling_ok"], "n"]),
            "floor": sorted(int(n) for n in g.loc[g["floor_ok"], "n"]),
        }
    doc = {
        "definition": POOL_DEFINITION,
        "floor_quantile": FLOOR_QUANTILE,
        "ceiling_quantile": QUANTILE,
        "min_floor_laps": MIN_FLOOR_LAPS,
        "races": out,
    }
    Path(path).write_text(json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")


def read_pool(path: Path) -> pd.DataFrame | None:
    """The persisted pool as an ``observed_stints``-shaped frame, or None if absent."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- a corrupt pool means no bounds, which is visible
        return None
    rows = []
    for key, labels in doc.get("races", {}).items():
        season, event = key.split("|", 1)
        for label, v in labels.items():
            rows += [(int(season), event, label, n, True, False) for n in v.get("ceiling", [])]
            rows += [(int(season), event, label, n, False, True) for n in v.get("floor", [])]
    return pd.DataFrame(rows, columns=["season", "event", "label", "n", "ceiling_ok", "floor_ok"])

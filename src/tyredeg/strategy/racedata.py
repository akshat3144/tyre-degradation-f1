"""Raw race laps and results for every season we hold, in one local cache.

WHY THIS EXISTS

Three jobs need the same thing and each used to open FastF1 sessions on its own:
the pit-loss measurement, the circuit history and the strategy evaluation. All
of them need the RAW laps -- ``clean_laps`` drops pit in- and out-laps, which
are the entire pit-loss signal -- plus who finished where and on which tyres.

Opening ~110 race sessions through FastF1 takes minutes even from its own disk
cache, so this reads them once into a compact parquet and everything else reads
that. The parquet lives under ``data/processed`` (git-ignored, regenerable): it
is a cache of public data, not a result, and every number the paper quotes is
written by the script that computed it.

WHAT IS KEPT

Only the columns the strategy layer reads, with PitInTime/PitOutTime and
LapTime converted to plain floats so the frame needs no FastF1 to use.
"""

from __future__ import annotations

import logging
import warnings
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..config import PROCESSED, SEASON

log = logging.getLogger(__name__)

LAPS_PATH = PROCESSED / "strategy_race_laps.parquet"
RESULTS_PATH = PROCESSED / "strategy_race_results.parquet"

#: Seasons the paper evaluates on. 2022 is the first season of the current
#: ground-effect rules; earlier years are a different car.
SEASONS = (2022, 2023, 2024, 2025, SEASON)

_LAP_COLS = [
    "Driver", "LapNumber", "LapTimeSeconds", "TrackStatus", "Stint", "Compound",
    "TyreLife", "Position", "PitIn_s", "PitOut_s",
]


def _secs(col: pd.Series) -> pd.Series:
    return pd.to_timedelta(col).dt.total_seconds()


def _one(event: str, season: int):
    from ..data.cache import load_session

    s = load_session(event, "R", season, telemetry=False, weather=False)
    laps = s.laps.copy()
    laps["LapTimeSeconds"] = _secs(laps["LapTime"])
    laps["PitIn_s"] = _secs(laps["PitInTime"])
    laps["PitOut_s"] = _secs(laps["PitOutTime"])
    laps["TrackStatus"] = laps["TrackStatus"].astype(str)
    laps = laps[_LAP_COLS].copy()
    laps["event"], laps["season"] = event, season

    res = s.results
    results = pd.DataFrame(
        {
            "Driver": res["Abbreviation"].astype(str).to_numpy(),
            "Position": pd.to_numeric(res["Position"], errors="coerce").to_numpy(),
            "Grid": pd.to_numeric(res.get("GridPosition"), errors="coerce").to_numpy(),
            "Status": res["Status"].astype(str).to_numpy(),
        }
    )
    results["event"], results["season"] = event, season
    return laps, results


def build(seasons: tuple[int, ...] = SEASONS, *, verbose: bool = True) -> None:
    """Read every race that has run, from FastF1's local cache, into the parquet.

    The calendar is the schedule's, so sprint weekends are included -- a pit stop
    costs the same on a sprint weekend -- and only races whose date has passed
    are opened, which keeps this from spending API calls on the future.
    """
    from ..data import schedule as sched

    warnings.filterwarnings("ignore")
    logging.getLogger("fastf1").setLevel(logging.ERROR)

    laps_out, res_out = [], []
    for season in seasons:
        if season == SEASON:
            events = sched.raced_events(season)
        else:
            events = _past_calendar(season)
        for event in events:
            try:
                laps, res = _one(event, season)
            except Exception as exc:  # noqa: BLE001 -- one bad session must not stop the batch
                if verbose:
                    print(f"  {season} {event:28s} skipped: {type(exc).__name__}", flush=True)
                continue
            laps_out.append(laps)
            res_out.append(res)
            if verbose:
                print(f"  {season} {event:28s} {len(laps):5d} laps", flush=True)
    if not laps_out:
        raise RuntimeError("no race session could be read from the FastF1 cache")
    LAPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(laps_out, ignore_index=True).to_parquet(LAPS_PATH)
    pd.concat(res_out, ignore_index=True).to_parquet(RESULTS_PATH)


def _past_calendar(season: int) -> list[str]:
    import fastf1

    from ..data.cache import enable_cache

    enable_cache()
    sched_ = fastf1.get_event_schedule(season, include_testing=False)
    return [str(n) for n in sched_["EventName"]]


def load(rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(laps, results)``; builds the cache on first use."""
    if rebuild or not (Path(LAPS_PATH).exists() and Path(RESULTS_PATH).exists()):
        build()
    laps = pd.read_parquet(LAPS_PATH)
    results = pd.read_parquet(RESULTS_PATH)
    return laps, results


def as_raw_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Give the cached frame the column names ``pitloss`` expects.

    ``pitloss.estimate`` was written against ``session.laps`` (PitInTime,
    PitOutTime as timestamps). Only ``notna()`` is ever asked of them, so the
    float seconds stand in without changing a number.
    """
    out = laps.copy()
    out["PitInTime"] = out["PitIn_s"]
    out["PitOutTime"] = out["PitOut_s"]
    return out


def field_plans(laps: pd.DataFrame, results: pd.DataFrame, event: str, season: int) -> pd.DataFrame:
    """Every classified car's stints as it ran them: one row per (driver, stint).

    Columns: Driver, Stint, Compound, n_laps, Position. A stint's length is the
    number of laps recorded on it, which includes the in- and out-lap, so it is
    the distance a set of tyres covered and not a count of clean laps.
    """
    ev = laps[(laps["event"] == event) & (laps["season"] == season) & laps["Stint"].notna()]
    g = (
        ev.groupby(["Driver", "Stint"])
        .agg(Compound=("Compound", "first"), n_laps=("LapNumber", "nunique"))
        .reset_index()
    )
    r = results[(results["event"] == event) & (results["season"] == season)][
        ["Driver", "Position", "Status"]
    ]
    out = g.merge(r, on="Driver", how="left")
    out["Position"] = out["Position"].astype(float)
    return out.sort_values(["Driver", "Stint"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def locations() -> dict[tuple[int, str], str]:
    """``(season, event) -> location`` for every round, as FastF1 spells it that year.

    The location, not the event name, is what identifies a pit lane -- see
    ``strategy.circuits``. 2026 comes from the repo's schedule store; earlier
    seasons from FastF1's schedule, which reads from its local cache.
    """
    import fastf1

    from ..data import schedule as sched
    from ..data.cache import enable_cache

    enable_cache()
    out: dict[tuple[int, str], str] = {}
    for season in SEASONS:
        if season == SEASON:
            for r in sched.rounds(season):
                out[(season, r.event)] = str(r.location)
        else:
            tab = fastf1.get_event_schedule(season, include_testing=False)
            for _, r in tab.iterrows():
                out[(season, str(r["EventName"]))] = str(r["Location"])
    return out


def is_dry(compound: object) -> bool:
    return str(compound) in ("SOFT", "MEDIUM", "HARD")


__all__ = ["LAPS_PATH", "RESULTS_PATH", "SEASONS", "as_raw_laps", "build", "field_plans", "is_dry", "load", "locations"]

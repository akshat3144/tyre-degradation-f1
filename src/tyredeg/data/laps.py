"""Building the clean lap dataset.

A "clean" lap is one where the lap time reflects the car and tyre, and nothing
else. Everything filtered here is filtered because it contaminates that signal.

Verified against the 2026 Hungarian GP FP2 on 2026-09-03: Compound, TyreLife,
FreshTyre, Stint, TrackStatus and LapStartTime are all 100 percent populated.
Position is 0 percent populated, so there is no traffic measurement available.
"""

from __future__ import annotations

import logging

import pandas as pd

from ..config import GREEN, MIN_LONG_RUN_LAPS, NON_GREEN_CODES

log = logging.getLogger(__name__)

#: Columns we keep. Everything else in session.laps is dropped to keep the
#: dataset legible and the parquet small.
KEEP = [
    "Driver",
    "Team",
    "LapNumber",
    "LapTime",
    "Stint",
    "Compound",
    "TyreLife",
    "FreshTyre",
    "TrackStatus",
    "IsAccurate",
    "LapStartTime",
    "PitInTime",
    "PitOutTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
]


def _seconds(s: pd.Series) -> pd.Series:
    """Timedelta column to float seconds."""
    return s.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(s) else s.astype(float)


def clean_laps(
    laps: pd.DataFrame,
    *,
    event: str,
    session: str,
    strict_green: bool = True,
) -> pd.DataFrame:
    """Filter a session's laps down to usable ones and add derived columns.

    Filters applied, in order, each with a reason:

    1. ``LapTime`` missing. No measurement, nothing to model.
    2. ``PitInTime`` or ``PitOutTime`` set. In and out laps include pit lane
       time, which says nothing about tyre wear. The benchmark filters these too.
    3. ``TrackStatus`` not green. Safety car, VSC and red flag laps are driven
       far below the limit, so they neither wear the tyre normally nor produce a
       representative lap time. With ``strict_green`` we require status exactly
       "1"; otherwise we only drop codes 4/5/6/7, which is what the benchmark
       does. Note that a lap can carry several status codes concatenated, e.g.
       "125", so substring logic matters.
    4. ``IsAccurate`` false. FastF1's own validity flag. It catches timing
       glitches and laps its heuristics do not trust.

    Args:
        laps: A ``session.laps`` frame.
        event: Event name, added as a column.
        session: Session name (FP1/FP2/FP3/R), added as a column.
        strict_green: Require fully green laps rather than only excluding
            safety-car codes.

    Returns:
        Clean laps with ``LapTimeSeconds`` and sector seconds added, plus
        ``event`` and ``session`` labels. Index is reset.
    """
    df = laps.copy()
    n0 = len(df)

    cols = [c for c in KEEP if c in df.columns]
    missing = set(KEEP) - set(cols)
    if missing:
        log.warning("%s %s: columns absent from feed: %s", event, session, sorted(missing))
    df = df[cols]

    df = df[df["LapTime"].notna()]
    n_time = len(df)

    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    n_pit = len(df)

    status = df["TrackStatus"].astype(str)
    if strict_green:
        df = df[status == GREEN]
    else:
        pattern = "|".join(NON_GREEN_CODES)
        df = df[~status.str.contains(pattern, regex=True, na=False)]
    n_green = len(df)

    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"].fillna(False).astype(bool)]
    n_acc = len(df)

    # Two things at once here.
    #
    # 1. Every line above is a filter, so df is a view. Copy before assigning,
    #    or pandas warns and the assignments may not stick.
    # 2. Drop down to a plain DataFrame. session.laps is a fastf1.core.Laps,
    #    which subclasses DataFrame and declares `_metadata = ['session']`.
    #    That makes `laps.session` return the Session OBJECT, shadowing any
    #    column of the same name -- and `laps.event` resolves to None the same
    #    way. Since we add columns called exactly `event` and `session`, every
    #    `df.event` downstream would silently be None instead of our data.
    #    Bracket access still works, but carrying the subclass around is a trap,
    #    so we leave it behind here.
    df = pd.DataFrame(df).copy()

    log.info(
        "%s %s: %d laps -> %d timed -> %d non-pit -> %d green -> %d accurate",
        event, session, n0, n_time, n_pit, n_green, n_acc,
    )

    df["LapTimeSeconds"] = _seconds(df["LapTime"])
    for i in (1, 2, 3):
        col = f"Sector{i}Time"
        if col in df.columns:
            df[f"S{i}"] = _seconds(df[col])
    if "LapStartTime" in df.columns:
        df["SessionElapsed"] = _seconds(df["LapStartTime"])

    df["event"] = event
    df["session"] = session
    return df.reset_index(drop=True)


def tag_long_runs(df: pd.DataFrame, min_laps: int = MIN_LONG_RUN_LAPS) -> pd.DataFrame:
    """Label consecutive-lap blocks within a driver's stint, and flag long runs.

    A long run is a block of at least ``min_laps`` laps with no gap in
    ``LapNumber``. The gap check matters: filtering above can remove a lap from
    the middle of a stint, and the laps either side of that hole are not
    consecutive even though they share a stint number.

    Adds:
        ``run_id``   unique per (event, session, driver, stint, block)
        ``run_len``  laps in that block
        ``is_long_run``  run_len >= min_laps
        ``run_lap``  1-based position within the block
    """
    if df.empty:
        return df.assign(run_id=[], run_len=[], is_long_run=[], run_lap=[])

    df = df.sort_values(["event", "session", "Driver", "LapNumber"]).copy()
    keys = ["event", "session", "Driver", "Stint"]

    # A new block starts wherever the lap number is not exactly one more than
    # the previous lap for the same driver and stint.
    prev = df.groupby(keys, dropna=False)["LapNumber"].shift(1)
    new_block = (df["LapNumber"] - prev).ne(1).fillna(True)
    df["block"] = new_block.groupby([df[k] for k in keys]).cumsum()

    # run_id must be a full key, not a per-call code. `.cat.codes` restarts at 0
    # every call, so tagging one session at a time and concatenating afterwards
    # produced run_id 0 in seven different events. Keep the readable key: it is
    # unique by construction, survives concatenation, and is legible when
    # debugging a suspicious run.
    run_keys = keys + ["block"]
    df["run_id"] = df[run_keys].astype(str).agg("|".join, axis=1)

    df["run_len"] = df.groupby("run_id")["LapNumber"].transform("size")
    df["run_lap"] = df.groupby("run_id").cumcount() + 1
    df["is_long_run"] = df["run_len"] >= min_laps

    return df.drop(columns=["block"]).reset_index(drop=True)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """One row per event and session: lap counts, long runs, compound spread.

    Used as a sanity check before fitting anything. If a session shows no long
    runs, either the filters are too aggressive or the session had no race
    simulation, and either way it should not silently enter the fit.
    """
    if df.empty:
        return pd.DataFrame()

    long_runs = df[df["is_long_run"]] if "is_long_run" in df.columns else df.iloc[0:0]

    # Bracket access throughout, never `df.event`. `event` and `session` collide
    # with attributes on fastf1.core.Laps, and if a Laps object ever reaches here
    # attribute access returns None and every count silently comes out zero.
    rows = []
    for (event, session), g in df.groupby(["event", "session"], sort=False):
        lr = long_runs[(long_runs["event"] == event) & (long_runs["session"] == session)]
        rows.append(
            {
                "event": event,
                "session": session,
                "laps": len(g),
                "drivers": g["Driver"].nunique(),
                "long_runs": lr["run_id"].nunique() if len(lr) else 0,
                "long_run_laps": len(lr),
                "compounds": (
                    ", ".join(sorted(lr["Compound"].dropna().unique())) if len(lr) else ""
                ),
            }
        )
    return pd.DataFrame(rows)

"""Traffic: how close was the car ahead?

The third confounder the brief names, and the one the race design does NOT
remove for free. Fuel load and track evolution are identical for every car on a
given lap, so demeaning by (event, lap) deletes them. Traffic is not -- it
differs between cars at the same instant, which is exactly the variation the
design relies on.

Worse, traffic is correlated with tyre age through pit stops. At lap 34 the
oldest tyre belongs to the driver who has not stopped, often running in clear
air; the freshest belongs to someone who just rejoined into the pack. Dirty air
costs lap time, so the fresh tyre looks slow and degradation looks smaller than
it is -- or negative.

WHY NOT TELEMETRY
    FastF1 can compute ``DistanceToDriverAhead`` from the positional stream, and
    it works (verified for 2026 races, 82% populated). But it needs full
    telemetry for every session, which is slow, and it is derived from the same
    4-5 Hz feed with 17-20 m between samples.

    The gap at the start/finish line is cheaper and better suited: it comes from
    ``LapStartTime``, which is 100% populated, needs no telemetry at all, and is
    the quantity race engineers actually quote over the radio.

KNOWN LIMITATION
    Gaps are computed among cars on the same lap number, so a lapped car sitting
    ahead on track is invisible here. Handling that properly needs track
    position, not lap counting. Stated rather than hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Within this many seconds of the car ahead, a driver is in dirty air and
#: losing time to aerodynamic disturbance rather than to the tyre. Roughly one
#: to two seconds is the usual figure quoted for meaningful aero loss.
DIRTY_AIR_S = 2.0

#: Beyond this, the car ahead is irrelevant. Capping stops a driver 40 seconds
#: clear from dragging the regression around.
CLEAR_AIR_S = 10.0


def add_gap_ahead(df: pd.DataFrame) -> pd.DataFrame:
    """Add the time gap to the car ahead at the start of each lap.

    Computed per (event, lap) by ordering drivers on when they crossed the line
    and differencing. The leader has no car ahead and is treated as clear air.

    Adds:
        ``gap_ahead_s``  seconds to the car ahead, capped at CLEAR_AIR_S
        ``in_dirty_air`` gap below DIRTY_AIR_S
        ``traffic``      modelling covariate: how much dirty air, 0 when clear.
                         Rises as the car ahead gets closer, which is the
                         direction the lap-time penalty runs.
    """
    if "LapStartTime" not in df.columns:
        raise ValueError("LapStartTime is required to compute gaps")

    df = df.copy()
    start = df["LapStartTime"]
    if pd.api.types.is_timedelta64_dtype(start):
        start = start.dt.total_seconds()
    df["_start_s"] = start

    df = df.sort_values(["event", "session", "LapNumber", "_start_s"])
    gap = df.groupby(["event", "session", "LapNumber"])["_start_s"].diff()

    # The leader of each lap has no car ahead: clear air, not a missing value.
    df["gap_ahead_s"] = gap.fillna(CLEAR_AIR_S).clip(0, CLEAR_AIR_S)
    df["in_dirty_air"] = df["gap_ahead_s"] < DIRTY_AIR_S

    # A closeness measure rather than a distance, so the coefficient has the
    # sign a reader expects: more traffic means more lap time lost.
    df["traffic"] = np.maximum(0.0, DIRTY_AIR_S - df["gap_ahead_s"])

    return df.drop(columns=["_start_s"]).reset_index(drop=True)


def traffic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Is traffic actually correlated with tyre age? If not, it is not a
    confounder here and adding it will change nothing.

    Returns one row per event with the correlation and how much dirty air there
    was to begin with.
    """
    rows = []
    for event, g in df.groupby("event"):
        rows.append(
            {
                "event": event,
                "laps": len(g),
                "pct_dirty_air": 100 * g["in_dirty_air"].mean(),
                "median_gap_s": g["gap_ahead_s"].median(),
                # The number that matters: if traffic and tyre age are
                # uncorrelated, traffic cannot bias the degradation estimate.
                "corr_traffic_tyreage": g["traffic"].corr(g["TyreLife"]),
            }
        )
    return pd.DataFrame(rows)

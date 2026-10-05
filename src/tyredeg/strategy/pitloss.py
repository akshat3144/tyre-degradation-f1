"""How much time does a pit stop cost?

Everything in strategy turns on this number. Degradation tells you when a tyre
is slow; pit loss tells you whether it is slow enough to be worth changing. Get
it wrong by two seconds and the one-stop versus two-stop answer flips.

It is not published, so we measure it. A pit stop costs the extra time on the
lap you enter the pits and the lap you leave, relative to what a car that did
not stop did on those same two laps:

    pit_loss = (in-lap + out-lap)
               - (non-pitting field's median on the in-lap + on the out-lap)
               - 2 x (this driver's habitual gap to that field)

The REFERENCE is the field on the same laps, not the driver's own race-wide
median. The old reference was the driver's median green lap over the whole race,
and a stop happens early or late in it: a lap-3 stop was compared against a
median dominated by laps with 20 kg less fuel and a rubbered-in track, so the
fuel burn and track evolution that make late laps quicker leaked into the loss
as a bias that depends on WHEN the car stopped. Comparing against the cars that
did not stop, lap for lap, cancels both. The driver's own gap to the field is
added back so that a backmarker is not credited with a dear stop for being slow.

Circuit-specific, because it depends on pit lane length, the speed limit, and
where the entry and exit sit relative to the timing line. Monaco and Monza are
nothing alike.

UNCERTAINTY. The estimate is the median of the stops measured, with a percentile
bootstrap interval over those stops and its standard deviation carried as
``PitLoss.sd``. ``strategy.uncertainty`` draws pit loss from that spread, so the
plan's interval reflects how well the pit lane was actually measured. (An earlier
version of this docstring claimed the uncertainty "propagates into the strategy
recommendation". It was printed beside the number and went no further.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Reject a stop whose implied loss falls outside this range, in seconds.
#: Real pit loss runs roughly 16-30s. Values outside that come from a drive-
#: through penalty, a stop that also involved damage repair, or a stack -- none
#: of which is a normal pit stop.
PLAUSIBLE_RANGE_S = (12.0, 40.0)

#: Wide sanity filter for the by-status analysis, in seconds. It removes only the
#: impossible (a timing glitch, a lap counted twice, a stop that included a
#: multi-lap repair), and is the same for every category so no category is
#: truncated on one side only. Medians, not means, are what is reported from it.
SANITY_RANGE_S = (-20.0, 60.0)

#: Bootstrap resamples for the interval on a median. 2,000 puts the Monte Carlo
#: error on a percentile endpoint well under the width it reports.
N_BOOT = 2000

BOOT_SEED = 20260101


@dataclass
class PitLoss:
    event: str
    seconds: float
    lo: float
    hi: float
    n_stops: int
    #: Standard deviation of the bootstrapped median, when it was measured.
    sd: float | None = None

    @property
    def se(self) -> float:
        if self.sd is not None:
            return self.sd
        return (self.hi - self.lo) / 3.92


def bootstrap_median(x: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float, float]:
    """``(median, lo, hi, sd)``: the median and a 95% percentile-bootstrap band.

    A bootstrap of the MEDIAN, because the estimate is a median. The first
    version centred the band on the median and took its width from the standard
    error of the MEAN -- two different estimators, so the band described a number
    nobody reported.
    """
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    meds = np.median(x[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    centre = float(np.median(x))
    # A percentile band need not contain the sample median on a tiny sample;
    # the artifact schema insists on lo <= centre <= hi.
    return centre, float(min(lo, centre)), float(max(hi, centre)), float(meds.std(ddof=1))


# ---------------------------------------------------------------------------
# Track status
# ---------------------------------------------------------------------------
#
# FastF1's codes: 1 all clear, 2 yellow, 4 safety car, 5 red flag, 6 VSC
# deployed, 7 VSC ending. A lap's status is the CONCATENATION of every code that
# applied during it, so these are tests for a digit's presence, and "41" is a
# safety car followed by all-clear inside one lap.
#
# Precedence is severity, not list order: a red flag outranks a safety car, which
# outranks a VSC, which outranks a yellow. The earlier version tested safety car
# first and had no red flag at all, so a lap carrying a '5' fell through to
# "green" and a stop under a red flag -- a free tyre change -- was counted as a
# green-flag stop.

_PRECEDENCE = (
    ("red_flag", ("5",)),
    ("safety_car", ("4",)),
    ("vsc", ("6", "7")),
    ("yellow", ("2",)),
)

CATEGORIES = ("green", "yellow", "vsc", "safety_car", "red_flag")


def status_category(status: object) -> str:
    """The most severe neutralisation that applied during a lap, else green."""
    s = str(status)
    for name, codes in _PRECEDENCE:
        if any(c in s for c in codes):
            return name
    return "green"


def stop_category(in_status: object, out_status: object) -> str:
    """The category of a STOP: the most severe of its in-lap and out-lap.

    A stop spans two laps and the cost is incurred on both, so the state of the
    race on the out-lap matters as much as on the in-lap. The earlier version
    read the in-lap only: a car pitting at the very end of a green lap that then
    met a safety car on its out-lap was filed as green, and the safety car's
    slow lap went into the green number.
    """
    a, b = status_category(in_status), status_category(out_status)
    names = [n for n, _ in _PRECEDENCE]
    ranked = [x for x in (a, b) if x != "green"]
    if not ranked:
        return "green"
    return min(ranked, key=names.index)


def _lap_seconds(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()
    if "LapTimeSeconds" not in df.columns:
        df["LapTimeSeconds"] = pd.to_timedelta(df["LapTime"]).dt.total_seconds()
    return df[df["LapTimeSeconds"].notna()]


def stop_losses(laps: pd.DataFrame, event: str) -> pd.DataFrame:
    """Every stop in a race and what it cost, one row each.

    Columns: event, driver, lap, category, loss, n_pitting. ``n_pitting`` is how
    many cars stopped on that lap in that category -- more than one at a time is
    a pit-lane queue, which is a cost of the situation and not of the pit lane.

    Needs the RAW laps, not the cleaned set: cleaning drops pit laps, which are
    the entire signal.

    Reference: the median lap time of cars that did NOT pit on that lap or the
    next, lap for lap, plus the driver's own habitual gap to that reference over
    his clean green laps. See the module docstring for why.

    No plausibility filter is applied here. The green estimator applies
    ``PLAUSIBLE_RANGE_S``; the by-status analysis deliberately does not, because
    a floor of 12 s would discard exactly the cheap stops under question.
    """
    cols = ["event", "driver", "lap", "category", "loss", "n_pitting"]
    df = _lap_seconds(laps).sort_values(["Driver", "LapNumber"])
    if df.empty:
        return pd.DataFrame(columns=cols)

    pit_in, pit_out = df["PitInTime"].notna(), df["PitOutTime"].notna()
    pitting = pit_in | pit_out
    status = df["TrackStatus"].astype(str)

    # What a car that stayed out did, lap by lap.
    reference = df[~pitting].groupby("LapNumber")["LapTimeSeconds"].median()

    # Each driver's habitual gap to that reference, on clean green laps only, so
    # that a slow car is not charged for being slow and a neutralised lap does
    # not move it.
    clean = df[~pitting & status.eq("1")].copy()
    clean["gap"] = clean["LapTimeSeconds"] - clean["LapNumber"].map(reference)
    habitual = clean.groupby("Driver")["gap"].median()

    by_driver = {d: g.set_index("LapNumber") for d, g in df.groupby("Driver")}
    rows = []
    for driver, g in by_driver.items():
        gap = float(habitual.get(driver, 0.0))
        if not np.isfinite(gap):
            gap = 0.0
        own_clean = g[~(g["PitInTime"].notna() | g["PitOutTime"].notna())]
        for lap in g.index[g["PitInTime"].notna()]:
            nxt = lap + 1
            if nxt not in g.index or pd.isna(g.at[nxt, "PitOutTime"]):
                continue
            observed = float(g.at[lap, "LapTimeSeconds"] + g.at[nxt, "LapTimeSeconds"])
            if lap in reference.index and nxt in reference.index:
                expected = float(reference.loc[lap] + reference.loc[nxt]) + 2 * gap
            else:
                # Nobody else was out on that lap -- only possible in a very
                # small field, or a synthetic one. Fall back to the driver's own
                # clean laps NEAR the stop, which keeps the fuel and track
                # evolution cancellation that the race-wide median lacks.
                near = own_clean[(own_clean.index - lap).to_series().abs().to_numpy() <= 5]
                if near.empty:
                    continue
                expected = 2 * float(near["LapTimeSeconds"].median())
            rows.append(
                {
                    "event": event,
                    "driver": driver,
                    "lap": int(lap),
                    "category": stop_category(g.at[lap, "TrackStatus"], g.at[nxt, "TrackStatus"]),
                    "loss": observed - expected,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    counts = out.groupby(["lap", "category"]).size().rename("n_pitting")
    return out.merge(counts, on=["lap", "category"])[cols]


def loss_by_status(laps: pd.DataFrame, event: str) -> pd.DataFrame:
    """Pit loss split by track status, each measured the same way.

    ``stop_losses`` with only the WIDE sanity bounds ``SANITY_RANGE_S``, applied
    identically to every category. The earlier version kept ``0 < loss < 45``, which
    threw away every stop that came out at or under zero. Under a safety car the
    non-pitting reference is itself slowed, so a cheap stop can legitimately
    measure about zero or below; discarding exactly those stops truncated the
    neutralised distributions from below and biased their ratio to green upward.
    The green floor of ``PLAUSIBLE_RANGE_S`` is likewise NOT applied here.
    """
    out = stop_losses(laps, event)
    if out.empty:
        return out
    return out[out["loss"].between(*SANITY_RANGE_S)]


def estimate(laps: pd.DataFrame, event: str) -> PitLoss | None:
    """Pit loss for one event, from its green-flag stops.

    Args:
        laps: raw ``session.laps`` for a race, with PitInTime and PitOutTime.
        event: event name, for labelling.

    Returns:
        A ``PitLoss`` with a bootstrap band, or None if fewer than three
        plausible green stops could be measured.
    """
    s = stop_losses(laps, event)
    if s.empty:
        return None
    g = s[(s["category"] == "green") & s["loss"].between(*PLAUSIBLE_RANGE_S)]
    if len(g) < 3:
        return None
    # Median, not mean: a single botched stop should not move the estimate.
    centre, lo, hi, sd = bootstrap_median(g["loss"].to_numpy())
    return PitLoss(event=event, seconds=centre, lo=lo, hi=hi, n_stops=len(g), sd=sd)


def estimate_all(events: list[str], season: int = 2026) -> pd.DataFrame:
    """Pit loss for every event, read from the local FastF1 cache."""
    from ..data.cache import load_session

    rows = []
    for event in events:
        try:
            s = load_session(event, "R", season, telemetry=False, weather=False)
            pl = estimate(s.laps, event)
        except Exception:  # noqa: BLE001 -- one bad session must not stop the batch
            pl = None
        if pl:
            rows.append(
                {
                    "event": pl.event,
                    "pit_loss_s": round(pl.seconds, 2),
                    "lo": round(pl.lo, 2),
                    "hi": round(pl.hi, 2),
                    "sd": round(pl.se, 3),
                    "n_stops": pl.n_stops,
                }
            )
    return pd.DataFrame(rows)

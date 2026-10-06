"""Which degradation rates is the optimiser allowed to use, and where do they come from?

WHY THIS EXISTS

The refusal rule in the strategy layer used to be a sign test: a compound was
usable if its fitted rate was positive. That is not the discipline the rest of
the project applies. The transfer step refuses a cell whose rate is within one
standard error of zero (``validation.transfer.identified``), because a number
whose sign we do not know is not a measurement, and an optimiser handed one will
exploit it with total confidence. A race-fit rate of +0.011 s/lap with an SE of
0.028 passed the sign test and was optimised on; it came out as a tyre that wears
at a third of its neighbour's rate, and the plan ran it for 73 laps.

So the same gate applies here, in one place, and every caller of the strategy layer
calls it -- they cannot disagree about which tyres are
real, which is what a test used to have to check.

THE ORDER OF PREFERENCE

  1. the race-lap fit, per circuit, for every nominated compound that is both
     positive and identified;
  2. if fewer than two survive, the practice forecast (practice long runs times
     the practice-to-race factor learned from OTHER events), gated the same way;
     and stand-ins for nominated tyres nobody ran, labelled as borrowed;
  3. otherwise nothing: no legal plan, and the caller says so.

Overrides (a caller typing in a rate) are applied on top and are never gated --
they are a what-if, and the screen marks them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .optimise import C_ORDER

log = logging.getLogger(__name__)

#: 95% half-width to standard deviation, for an interval stored as lo/hi.
Z95 = 1.96


@dataclass
class RateRow:
    """One nominated compound's rate and what happened to it."""

    compound: str
    mean: float
    lo: float
    hi: float
    #: "race", "practice", "thin" or "stand-in". "override" for a typed value.
    source: str
    excluded: bool = False
    #: Why it was excluded: "non-positive", "not identified", or None.
    reason: str | None = None
    overridden: bool = False

    @property
    def sd(self) -> float:
        # A typed-in rate is the caller's own belief, not a measurement with an
        # error bar: it carries none.
        if self.overridden:
            return 0.0
        return max((self.hi - self.lo) / (2 * Z95), 0.0)


@dataclass
class RateSet:
    rows: list[RateRow]
    #: "race" when the race-lap fit supplied the plan, "practice" when it fell
    #: back to the weekend's long runs.
    source: str
    #: Human-readable account of every refusal, for the response and the artifact.
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> dict[str, float]:
        return {r.compound: r.mean for r in self.rows if not r.excluded}

    @property
    def sd(self) -> dict[str, float]:
        return {r.compound: r.sd for r in self.rows if not r.excluded}


def passes_gate(mean: float, sd: float) -> tuple[bool, str | None]:
    """Is this rate positive, and distinguishable from zero?

    Applies ``validation.transfer.identified``: ``|rate| >= MIN_RATE_SE_RATIO x
    se``, with a missing or zero SE counting as identified (a one-run practice
    cell has no between-run variance to estimate one from and the forecast widens
    those instead). The SE here is ``(hi - lo) / 3.92`` of the stored interval.
    """
    from ..validation.transfer import identified

    if mean <= 0:
        return False, "non-positive"
    ok = bool(identified(pd.Series([mean]), pd.Series([sd if sd > 0 else np.nan])).iloc[0])
    return (ok, None if ok else "not identified")


def practice_forecast(
    practice: pd.DataFrame | None,
    race: pd.DataFrame,
    event: str,
    nominated: set[str],
    *,
    factor: float | None = None,
    stand_ins: bool = True,
) -> pd.DataFrame | None:
    """Race rates forecast from this weekend's practice long runs, or None.

    Rate per nominated compound from practice, corrected by the factor learned
    from OTHER events (the event being forecast never informs its own
    correction), with stand-ins for tyres nobody put on a race simulation.
    """
    from ..validation.transfer import forecast as forecast_rates
    from ..validation.transfer import leave_one_event_out, stand_in_rates

    if practice is None or practice.empty or event not in set(practice["event"]):
        return None
    try:
        if factor is None:
            factor = float(leave_one_event_out(practice, race).factor)
        rows = forecast_rates(practice, factor, event)
    except Exception as exc:  # noqa: BLE001
        log.warning("practice forecast unavailable for %s: %s", event, exc)
        return None
    standins = pd.DataFrame()
    if stand_ins:
        try:
            standins = stand_in_rates(practice, factor, event, sorted(nominated))
        except Exception as exc:  # noqa: BLE001
            log.warning("stand-in rates failed for %s: %s", event, exc)
    if not standins.empty:
        rows = rows[~rows["C"].isin(standins["C"])]
        rows = pd.concat([rows, standins], ignore_index=True)
    return rows[rows["C"].isin(nominated)]


def resolve_rates(
    fit,
    event: str,
    nominated: set[str],
    *,
    practice: pd.DataFrame | None = None,
    race: pd.DataFrame | None = None,
    overrides: dict[str, float] | None = None,
    practice_rows: pd.DataFrame | None = None,
) -> RateSet:
    """The rates the optimiser may use at ``event``, gated and sourced.

    ``practice_rows`` lets a caller that has already computed the forecast (the
    API caches it) pass it in instead of recomputing.
    """
    ordered = [c for c in C_ORDER if c in nominated]
    notes: list[str] = []

    def race_rows() -> list[RateRow]:
        out = []
        for c in ordered:
            if c not in fit.rates:
                continue
            iv = fit.interval_for(c, event)
            out.append(RateRow(c, float(iv.mean), float(iv.lo), float(iv.hi), "race"))
        return out

    def gate(rows: list[RateRow], *, skip_standin: bool) -> list[RateRow]:
        for r in rows:
            # A stand-in is borrowed and already carries a deliberately wide
            # band; it has to be positive and nothing more.
            if skip_standin and r.source == "stand-in":
                ok, why = (r.mean > 0, None if r.mean > 0 else "non-positive")
            else:
                ok, why = passes_gate(r.mean, r.sd)
            r.excluded, r.reason = (not ok), why
        return rows

    rows = gate(race_rows(), skip_standin=False)
    source = "race"
    for r in rows:
        if r.excluded:
            notes.append(
                f"{r.compound}: race-fit rate {r.mean:+.3f} s/lap (SD {r.sd:.3f}) refused, {r.reason}"
            )

    if sum(not r.excluded for r in rows) < 2 and race is not None:
        pr = practice_rows
        if pr is None:
            pr = practice_forecast(practice, race, event, nominated)
        if pr is not None and not pr.empty:
            prow = []
            for _, x in pr.iterrows():
                src = str(x.get("source", "measured"))
                prow.append(
                    RateRow(
                        str(x["C"]),
                        float(x["predicted_race_rate"]),
                        float(x["lo"]),
                        float(x["hi"]),
                        "practice" if src == "measured" else src,
                    )
                )
            prow.sort(key=lambda r: C_ORDER.index(r.compound) if r.compound in C_ORDER else 99)
            prow = gate(prow, skip_standin=True)
            if sum(not r.excluded for r in prow) >= 2:
                rows, source = prow, "practice"
                notes.append(
                    "fewer than two nominated tyres had an identified positive race-fit "
                    "rate; planned from the practice forecast instead"
                )

    for r in rows:
        if overrides and r.compound in overrides:
            r.mean = float(overrides[r.compound])
            r.overridden = True
            r.excluded, r.reason = (r.mean <= 0, "non-positive" if r.mean <= 0 else None)
    return RateSet(rows=rows, source=source, notes=notes)

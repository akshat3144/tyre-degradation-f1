"""How much each practice session should count toward the race forecast.

A mentor put it plainly: you have FP1, FP2 and FP3, so summarise each and
weight the second most heavily. He was right, and the reason he is right is
not the reason usually given.

THE USUAL REASON, WHICH NO LONGER HOLDS
    Until 2021 a European weekend ran FP2 and the race both at 14:00 local.
    Same hour, same sun, same track temperature. "Use FP2" meant "use the
    session that looks like the race". The modern calendar broke that:
    Madrid 2026 runs FP1 at 13:30 local, FP2 at 17:00, and the race at 15:00,
    so by the clock FP1 is the closer match. Weighting on time of day alone
    would hand FP1 the top weight at most of the calendar.

THE REAL REASON, WHICH WE MEASURED
    Against the races that have actually run this season, we scored each
    session's measured degradation against the race's:

        FP2   10 cells   correlation 0.646   median 41 laps per cell
        FP1   11 cells   correlation 0.293   median 28 laps per cell
        FP3    2 cells   not enough to score

    FP2 is where the setup is frozen and the heavy-fuel race simulations are
    run. That is why it transfers, and it is why its weight is highest here.
    FP1 carries some signal and much less of it: teams change the car between
    FP1 runs, so a slope fitted across them is measuring setup work as much as
    tyre wear.

    TREAT THESE CORRELATIONS AS A DIAGNOSTIC, NOT AS THE CRITERION.
    The same FP1 number has read 0.05, then 0.78 after the warm-up fix, then
    0.293 once five more weekends of data arrived. Each was an honest
    measurement of a different cell set, which is exactly why the weights are
    not chosen from this table. They are chosen by the end-to-end error of the
    blend, over cells every scheme can answer -- see ``best_scheme`` in
    ``scripts/12_session_skill.py``, which the tests re-run.

WHY FP1 AND FP3 ARE NOT ZERO
    The skill gap above would justify weights near zero, and we do not use
    them. Ten cells is not enough to retire a session on, and a weekend like
    Madrid -- a new circuit, no race history at all -- cannot afford to discard
    evidence about the very track it is asked about. Halving is a deliberate
    shrink toward equal weighting: it respects the measured gap without betting
    the forecast on a small sample.

RE-MEASURED OUT OF SEASON (scripts/12_session_skill.py, paper/results)
    The 2026 table above is ten cells a session and the same cells the weights
    were chosen on, so it cannot carry the argument. Over 2022-2025 (71
    weekends, HARD/MEDIUM/SOFT labels, design frozen), with weekend-clustered
    95% intervals:

        FP1   44 cells   correlation 0.18  [-0.16, 0.50]
        FP2   49 cells   correlation 0.24  [-0.08, 0.52]
        FP3   16 cells   correlation 0.27  [-0.28, 0.75]

    On the 25 cells FP1 and FP2 share, FP2 minus FP1 is +0.11 in correlation
    with an interval of [-0.19, +0.40], and the leave-one-out MAE of a
    single-session forecast differs by 0.002 s/lap [-0.009, +0.013]. In 2026
    itself FP2's 0.65 interval is [0.10, 0.91] and FP1's 0.29 is [-0.62, 0.79]:
    the two overlap almost entirely. So "FP2 carries the race-simulation
    signal" is NOT established, in or out of season. Replacing the weights with
    any of the other six schemes changes the out-of-season error by under 0.002
    s/lap, inside the noise. The weights below are therefore a prior, not a
    finding: harmless, defensible on the setup-frozen argument, and not
    something the data has confirmed. The Sprint cannot be tested out of season
    because no Sprint session is cached for 2022-2025.

THE SPRINT
    A sprint weekend has no FP2. What it has instead is the Sprint: twenty
    cars running one set of tyres for eighteen laps at race pace, with
    nothing to hide and nowhere to stop. Scored the same way as the practice
    sessions, over the five sprint weekends that have raced this season:

        S     9 cells   correlation 0.65   median 58 laps per cell

    Level with FP2 on correlation, on thicker cells, and on a weekend where
    FP1 (0.29) is the only alternative it is not a close call. It carries
    FP2's weight. The sweep in ``scripts/12_session_skill.py`` also says
    FP1 should NOT be zeroed beside it: "sprint only" scores worse than the
    shipped blend, so the hour of practice still earns its half.

The weights are constants rather than a live fit, because fitting them from
nine cells would make the forecast lurch race to race. ``tests`` re-measures
the skill table and fails if FP2 ever stops being the best predictor, so the
choice stays checked rather than assumed.
"""

from __future__ import annotations

import logging

from . import schedule

log = logging.getLogger(__name__)

#: Weight per session, normalised so the best-transferring session is 1.0.
#: See the module docstring for the measurement behind these.
SESSION_WEIGHTS: dict[str, float] = {"FP1": 0.5, "FP2": 1.0, "FP3": 0.5, "S": 1.0}

#: Used for any session not named above, and for whole datasets that predate
#: the split. Equal weighting is what the model did before this module, so an
#: unrecognised session degrades to the old behaviour rather than to zero.
DEFAULT_WEIGHT = 1.0


def weight_for(session: str) -> float:
    """Weight for one session code."""
    return SESSION_WEIGHTS.get(str(session).upper(), DEFAULT_WEIGHT)


def weights() -> dict[str, float]:
    """The full weight table, copied so callers cannot edit the constant."""
    return dict(SESSION_WEIGHTS)


def hours_to_race(event: str, season: int | None = None) -> dict[str, float]:
    """Signed clock-hours from each practice session to the race start.

    Not used to weight anything. Reported on screen next to the weights
    because it is the first thing a strategist asks -- "was that session at
    race time?" -- and because at some circuits the answer is no, which is
    worth seeing rather than hiding. Negative is earlier in the day.
    """
    try:
        rnd = schedule.find(event, season) if season else schedule.find(event)
    except Exception as exc:  # noqa: BLE001 - a missing calendar is not fatal
        log.debug("no schedule for %s: %s", event, exc)
        return {}
    if rnd is None or (race := rnd.session("R")) is None:
        return {}

    out = {}
    for code in schedule.LONG_RUN_SESSIONS:
        if (s := rnd.session(code)) is None:
            continue
        ha = s.starts_utc.hour + s.starts_utc.minute / 60
        hb = race.starts_utc.hour + race.starts_utc.minute / 60
        d = (ha - hb) % 24
        out[code] = round(d - 24 if d > 12 else d, 2)
    return out

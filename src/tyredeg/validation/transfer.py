"""Practice to race: predict Sunday's tyre behaviour from Friday's.

This is the deliverable the brief actually names -- "post-race validation tools
to compare predicted wear against actual race-day pace" -- and the benchmark
paper does not attempt it. Their model fits within a single race, so it never
has to generalise from one session to another.

Two predictions are compared, and the gap between them is the point.

NAIVE
    Assume Sunday degrades like Friday. This is what you get if you take a
    practice degradation curve at face value, and it is what every public
    FastF1 tyre analysis implicitly does.

CALIBRATED
    Learn the practice-to-race relationship from OTHER events and apply it.
    We already know from the pooled fits that the two differ systematically:
    race degradation is lower. That is consistent with drivers managing
    fragile tyres rather than pushing them. If the effect is real and stable,
    correcting for it should beat the naive prediction out of sample.

    It was once also written here that "the gap widens for softer compounds".
    That is NOT supported out of season. A compound-specific factor scored
    worse than the single scalar on 2022-2025 (frozen 2026 factors: MAE 0.063
    against 0.036; learned in-season: 0.035 against 0.037, not different), and
    the factor's trend with softness is +0.07 per step with an interval across
    zero. In 2026 the trend looks negative only because five SOFT cells carry
    noisy, sometimes negative, practice rates. So one scalar ships.

Validation is leave-one-event-out. The calibration never sees the event it is
asked to predict, so a better score cannot come from fitting the answer.

WHAT THE EVIDENCE SUPPORTS (``scripts/paper/p2_transfer.py``, paper/results)
    The calibrated forecast removes most of the naive error out of season
    (2022-2025, 2026 factor frozen: MAE 0.036 against 0.112), and beats
    predicting zero. It does NOT beat a constant: the median race rate of
    other weekends scores about 0.031 there, and the 2026 "35.6% better than
    naive" is the improvement over the weakest baseline available. The
    practice signal is real but weak (rank correlation about 0.3); its value
    is in the blend with a prior, not in replacing one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Runs in an (event, compound) cell before its rate counts as MEASURED. Below
#: this the cell is still fitted and still forecast from -- one run of a tyre
#: at this circuit is an observation, and refusing to read it is a blank, not
#: caution -- but it comes back marked thin with a wider band. The calibration
#: in ``leave_one_event_out`` keeps this floor as a hard one: a factor learned
#: from two-run cells would lurch, and it is applied to every forecast.
MIN_RUNS = 3

#: The floor a forecast fits at. Anything that survived the practice filters
#: is a race-simulation run of at least five laps on the tyre in question, and
#: a sprint weekend's one hour of practice may produce exactly one of them.
FORECAST_MIN_RUNS = 1

#: Minimum tyre-age spread, in laps, among cars on the same compound at the same
#: lap of the same race.
#:
#: This is an identifiability requirement, not a quality preference. The race
#: design identifies degradation by comparing cars at the same instant, so it
#: needs them to be on DIFFERENT tyre ages. When a field runs a synchronised
#: strategy that variation disappears: at the 2026 Belgian GP every car on C3
#: was the same age on any given lap, spread exactly 0.00, and the fitted slope
#: was meaningless. Cells below this threshold produced race rates of -0.21 and
#: -0.41 s/lap -- tyres apparently getting faster as they wore out.
#:
#: Refusing to report those is the same discipline as the power analysis: do not
#: publish an estimate the design cannot support.
MIN_AGE_SPREAD_LAPS = 2.0

#: Share of leave-one-out forecast errors a drawn interval must cover.
INTERVAL_LEVEL = 0.95


def empirical_half_width(abs_errors, level: float = INTERVAL_LEVEL) -> float:
    """Half-width that would have covered ``level`` of the out-of-fold errors.

    WHY THIS REPLACES ``1.96 * se * factor`` ON THE TRANSFER SCREEN
        That interval is the sampling error of the PRACTICE slope, scaled. It
        ignores the uncertainty in the factor and the noise in the race rate
        it is compared with. Measured against the realised race rate it
        covered 24% of cells in 2026 and 43% out of season, for a nominal 95%.
        The multiple of ``1.96 * se`` that would reach 95% was 10.9 in one
        sample and 3.7 in the other, so no constant multiple is stable; the
        errors themselves are the honest scale.

    It is wide, and the width is the finding: the forecast is good to about
    the size of the rates it forecasts. Returns 0.0 for no errors.
    """
    import math

    e = np.asarray([x for x in abs_errors if x is not None and math.isfinite(x)], dtype=float)
    if e.size == 0:
        return 0.0
    return float(np.quantile(np.abs(e), level, method="higher"))


#: Hardest to softest. Degradation must not decrease along it: a softer tyre
#: cannot wear more slowly than a harder one on the same track.
C_ORDER = ("C1", "C2", "C3", "C4", "C5")


@dataclass
class TransferResult:
    table: pd.DataFrame
    mae_naive: float
    mae_calibrated: float
    factor: float
    n_events: int

    @property
    def improvement(self) -> float:
        """Fraction of the naive error removed by calibrating. Negative is worse."""
        if self.mae_naive == 0:
            return 0.0
        return (self.mae_naive - self.mae_calibrated) / self.mae_naive


def age_spread(g: pd.DataFrame) -> float:
    """Mean tyre-age spread among cars sharing a lap. The identifying variation.

    Zero means every car on this compound was the same age whenever they were
    on track together, and no slope can be recovered. See MIN_AGE_SPREAD_LAPS.
    """
    per_lap = g.groupby("LapNumber")["TyreLife"].agg(["size", "std"])
    multi = per_lap[per_lap["size"] >= 2]
    return float(multi["std"].mean()) if len(multi) else 0.0


def cell_rates(
    df: pd.DataFrame,
    min_runs: int = MIN_RUNS,
    min_age_spread: float | None = None,
) -> pd.DataFrame:
    """Degradation rate per (event, compound) from a prepared design frame.

    Ordinary least squares through the already-centred design rather than the
    mixed-effects model: a cell can hold as few as three runs, which will not
    support a random-effects variance. The slope is what transfers, and it is
    estimated the same way on both sides so the comparison stays fair.

    Args:
        min_age_spread: for race cells, the minimum tyre-age spread required
            before a rate is reported. Pass ``MIN_AGE_SPREAD_LAPS`` for race
            data; leave as None for practice, where the design identifies
            degradation within a run rather than across cars.

    Returns columns: event, C, rate, se, n_runs, n_laps, age_spread.
    """
    rows = []
    for (event, compound), g in df.groupby(["event", "C"], observed=True):
        n_runs = g["run_id"].nunique()
        if n_runs < min_runs:
            continue

        spread = age_spread(g)
        if min_age_spread is not None and spread < min_age_spread:
            continue

        y = g["y"].to_numpy()

        # Partial out traffic where it is available. Dirty air is correlated
        # with tyre age through pit stops -- fresh tyres rejoin into the pack --
        # so leaving it in the residual biases the slope. Least squares on
        # [tyre age, traffic] rather than tyre age alone.
        cols = ["tl"] + (["tr"] if "tr" in g.columns else [])
        X = g[cols].to_numpy(dtype=float)
        if not np.isfinite(X).all() or float(X[:, 0] @ X[:, 0]) <= 0:
            continue

        # Weighted least squares, by scaling both sides by sqrt(w).
        #
        # This is how practice sessions stop counting equally. FP1's measured
        # degradation correlates 0.29 with the race where FP2's and the
        # sprint's correlate 0.65, so pooling them one-for-one let the least
        # informative session pull the slope. The weights themselves are not
        # chosen from those correlations -- see `session_weight` -- but this is
        # the machinery that applies them. Race frames carry no `w` and fall
        # through unweighted, which is the same arithmetic as before.
        #
        # Scaling rather than a weight argument keeps the clustered sandwich
        # below correct without a second code path: the residuals it squares
        # are already the weighted ones.
        w = None
        if "w" in g.columns:
            w = g["w"].to_numpy(dtype=float)
            if not np.isfinite(w).all() or (w <= 0).any():
                w = None
        if w is not None:
            rw = np.sqrt(w)
            Xf, yf = X * rw[:, None], y * rw
        else:
            Xf, yf = X, y

        try:
            beta, *_ = np.linalg.lstsq(Xf, yf, rcond=None)
        except np.linalg.LinAlgError:
            continue
        slope = float(beta[0])

        X, y = Xf, yf
        resid = y - X @ beta

        # Standard error CLUSTERED BY RUN, not by lap.
        #
        # Laps inside a run are strongly correlated -- a driver managing a tyre
        # is slow on all of them, not independently on each -- and races average
        # 14 laps per run. Treating each lap as independent counts the same
        # information many times over and makes the interval far too narrow.
        #
        # Measured on the fleet transfer, which has the identical structure:
        # with ordinary errors, 95% intervals covered the truth 37% of the time
        # across 30 simulated fleets. Clustering brought that to 97%.
        try:
            bread = np.linalg.pinv(X.T @ X)
            clusters = g["run_id"].to_numpy()
            n_c = len(np.unique(clusters))
            if n_c > 1:
                meat = np.zeros((X.shape[1], X.shape[1]))
                for c in np.unique(clusters):
                    m = clusters == c
                    sc = X[m].T @ resid[m]
                    meat += np.outer(sc, sc)
                corr = n_c / max(1, n_c - 1)
                se = float(np.sqrt(corr * (bread @ meat @ bread)[0, 0]))
            else:
                # One run has no between-run variance to cluster on. The
                # ordinary within-run error is what there is; it understates
                # the truth, and the forecast widens it for that reason.
                dof = max(1, len(y) - X.shape[1])
                sigma2 = float(resid @ resid) / dof
                se = float(np.sqrt(sigma2 * bread[0, 0]))
        except np.linalg.LinAlgError:
            se = np.nan
        rows.append(
            {
                "event": event,
                "C": compound,
                "rate": slope,
                "se": se,
                "n_runs": n_runs,
                "n_laps": len(g),
                "age_spread": spread,
            }
        )
    return pd.DataFrame(rows)


def leave_one_event_out(
    practice: pd.DataFrame,
    race: pd.DataFrame,
    min_runs: int = MIN_RUNS,
) -> TransferResult:
    """Predict each event's race degradation from its practice, honestly.

    For every event in turn:
      - the naive prediction is that event's practice rate, unchanged
      - the calibrated prediction multiplies it by a factor learned from the
        SIX OTHER events, so the held-out event never informs its own correction

    Args:
        practice: prepared practice design frame.
        race: prepared race design frame.

    Returns:
        A ``TransferResult`` with a per-cell table and both mean absolute errors.
    """
    # Practice identifies degradation within a run, so it needs no cross-car
    # age spread. Race identifies it across cars at the same lap, so it does.
    p = cell_rates(practice, min_runs).rename(columns={"rate": "practice_rate"})
    r = cell_rates(race, min_runs, min_age_spread=MIN_AGE_SPREAD_LAPS).rename(
        columns={"rate": "race_rate"}
    )
    joined = p.merge(r, on=["event", "C"], suffixes=("_p", "_r"))

    if joined.empty:
        raise ValueError("no (event, compound) cell has both practice and race data")

    # A factor, not a difference: management appears to suppress degradation
    # proportionally, and a ratio keeps the prediction positive.
    usable = joined[joined["practice_rate"].abs() > 1e-6]

    rows = []
    for event in joined["event"].unique():
        others = usable[usable["event"] != event]
        factor = (
            float(np.median(others["race_rate"] / others["practice_rate"]))
            if len(others) >= 2
            else 1.0
        )
        for _, cell in joined[joined["event"] == event].iterrows():
            naive = cell["practice_rate"]
            calibrated = naive * factor
            rows.append(
                {
                    "event": event,
                    "C": cell["C"],
                    "practice_rate": naive,
                    "race_rate": cell["race_rate"],
                    "naive_pred": naive,
                    "calibrated_pred": calibrated,
                    "naive_err": abs(naive - cell["race_rate"]),
                    "calibrated_err": abs(calibrated - cell["race_rate"]),
                    "factor_used": factor,
                    "n_runs_practice": int(cell["n_runs_p"]),
                    "n_runs_race": int(cell["n_runs_r"]),
                    "se_practice": cell["se_p"],
                }
            )

    table = pd.DataFrame(rows)
    return TransferResult(
        table=table,
        mae_naive=float(table["naive_err"].mean()),
        mae_calibrated=float(table["calibrated_err"].mean()),
        factor=float(np.median(usable["race_rate"] / usable["practice_rate"])),
        n_events=table["event"].nunique(),
    )


#: How much wider a thin cell's interval is than a measured one's. Under
#: MIN_RUNS the clustered error has one or two runs to learn run-to-run
#: variance from, or none at all, so the band it reports is too narrow to
#: trust. Doubling is a shrink toward what three-run cells typically show,
#: not a measurement of anything.
THIN_INTERVAL_MULTIPLE = 2.0

#: How much wider a stand-in rate's interval is than a measured one's. A rate
#: borrowed from other circuits and rescaled is a weaker claim than a rate
#: measured here, and the interval is the only place that can show it. Set so
#: a typical stand-in spans roughly the compound's spread across the calendar.
STANDIN_INTERVAL_MULTIPLE = 3.0

#: A cell must be at least this many of its own standard errors from zero
#: before it counts as a measurement of this circuit.
#:
#: One is a deliberately weak bar -- roughly 68% confidence -- because the job
#: is to catch cells that say nothing at all, not to demand significance from
#: three practice runs.
#:
#: WHAT IT CATCHES. Madrid measured C4 at 0.058 s/lap with a standard error of
#: 0.198: a number whose sign we do not know. The forecast carried it through
#: as a measurement, the optimiser read the SOFT as the most durable tyre in
#: the race, and the plan for a 57-lap Grand Prix came back as five laps on the
#: MEDIUM followed by fifty-two on the SOFT. The five laps existed only to make
#: the plan legal.
#:
#: Same discipline as the sign filter in `strategy.optimise` and the age-spread
#: filter for race cells: an optimiser exploits a non-physical input with total
#: confidence, so the input is refused rather than the output dressed up. A
#: refused cell is not lost -- it falls to a stand-in, which is clamped against
#: the compounds either side of it and cannot invert the order.
MIN_RATE_SE_RATIO = 1.0

#: And wider again when there was nothing measured here to scale it by. An
#: unscaled calendar rate at an unseen circuit is the weakest thing we will
#: put a number on, and the band has to say so.
POOLED_INTERVAL_MULTIPLE = 5.0


def forecast(
    practice: pd.DataFrame,
    factor: float,
    event: str,
    min_runs: int = FORECAST_MIN_RUNS,
) -> pd.DataFrame:
    """Predict a race that has not happened yet.

    For a live forecast: feed in a weekend's practice, apply
    the factor learned from completed events, and produce race predictions with
    no actuals to compare against. ``TransferArtifact.is_forecast`` marks these
    so a forecast is never presented as a validated result.

    Fits at ``FORECAST_MIN_RUNS``, so one race-simulation run of a tyre is
    enough to put a number on it. Cells under ``MIN_RUNS`` come back with
    ``source="thin"`` and a band widened by ``THIN_INTERVAL_MULTIPLE``; the
    screen draws them differently and the plan still uses them, because a
    plan built on one observed run of this tyre at this circuit is a better
    plan than one that pretends the run did not happen.
    """
    p = cell_rates(practice[practice["event"] == event], min_runs)
    if p.empty:
        raise ValueError(f"no usable practice cells for {event!r}")
    thin = p["n_runs"] < MIN_RUNS
    # A single-run cell can have no finite error at all if its laps were
    # collinear; carry the widest thing we know rather than a NaN band.
    se = p["se"].where(p["se"].notna(), p["rate"].abs())
    half = 1.96 * se * np.where(thin, THIN_INTERVAL_MULTIPLE, 1.0)
    # Two runs that happen to agree give a clustered error near zero, and a
    # thin cell then claims a band no three-run cell could. Floor it at half
    # the rate itself: the same floor a stand-in gets, for the same reason.
    half = np.where(thin, np.maximum(half, 0.5 * p["rate"].abs()), half)
    p["predicted_race_rate"] = p["rate"] * factor
    # Practice is thin, so carry its uncertainty through rather than hiding it.
    p["lo"] = (p["rate"] - half) * factor
    p["hi"] = (p["rate"] + half) * factor
    p["source"] = np.where(thin, "thin", "measured")
    p["severity"] = np.nan
    return p


def circuit_severity(measured: pd.DataFrame, pooled: pd.Series) -> float | None:
    """How harsh this circuit is on tyres, relative to the calendar.

    The ratio of what we measured here to what the same compounds do
    everywhere else. Madrid's C3 wears at 0.394 s/lap against a season-wide
    C3 median of 0.157, so Madrid runs about 2.5x harsh -- which is plausible
    on its own terms, since Barcelona, the other high-load Spanish track, is
    the second worst on the calendar.

    Taken as a median over whatever compounds we do have, so one noisy cell
    cannot set it. Returns None when nothing overlaps, which the caller must
    read as "we cannot stand in for anything here".
    """
    ratios = [
        float(r["rate"]) / float(pooled[r["C"]])
        for _, r in measured.iterrows()
        if r["C"] in pooled.index and float(pooled[r["C"]]) > 0 and float(r["rate"]) > 0
    ]
    return float(np.median(ratios)) if ratios else None


def identified(rate: pd.Series, se: pd.Series) -> pd.Series:
    """Is this cell far enough from zero to be a measurement of anything?

    ``|rate| >= MIN_RATE_SE_RATIO * se``. A missing standard error counts as
    identified: a single-run cell has no between-run variance to estimate one
    from, and `forecast` already widens those by THIN_INTERVAL_MULTIPLE rather
    than discarding them.
    """
    se = pd.to_numeric(se, errors="coerce")
    return (se.isna()) | (se <= 0) | (rate.abs() >= MIN_RATE_SE_RATIO * se)


def stand_in_rates(
    practice: pd.DataFrame,
    factor: float,
    event: str,
    wanted: Iterable[str],
    min_runs: int = MIN_RUNS,
) -> pd.DataFrame:
    """Fill compounds this weekend never ran, from the rest of the calendar.

    THE PROBLEM THIS SOLVES
        Madrid 2026 is a new circuit with no history, and across all three
        practice sessions nobody put a HARD on a race simulation -- zero runs
        on a compound Pirelli nominated for Sunday. The SOFT managed two runs
        where three are needed. One usable compound is not a legal plan, so
        the strategy screen refused to answer the only question that matters,
        on the one race we demo.

    WHAT WE DO
        Take the compound's rate across every other circuit, and scale it by
        how harsh this circuit is on the compounds we DID measure. "Nobody ran
        the HARD here, so take the HARD's season-wide wear rate and scale it
        by how much harsher this track is on the tyres we did run."

    WHAT IT IS NOT
        It is not a measurement, and it must never be drawn as one. Rows come
        back marked ``source='stand-in'`` with a deliberately wide interval,
        and the caller is expected to label them on screen. A stand-in rate is
        the difference between a planner that says "probably two stops, and
        here is which number we borrowed" and one that says nothing at all.

    WHAT COUNTS AS MEASURED HERE
        Any cell with a positive slope, at ``FORECAST_MIN_RUNS``. A thin cell
        is an observation of this circuit and sets the severity like any
        other. A cell with a NEGATIVE slope is noise -- a tyre does not get
        faster as it wears -- and is treated as not measured, so the stand-in
        fills it rather than the plan losing the compound to one bad run.

    WHEN NOTHING HERE CAN SET THE SEVERITY
        The bare calendar rate is handed back unscaled, with ``severity=None``
        and a band wide enough to say so. It used to return nothing on the
        argument that an unscaled rate at an unseen circuit is a guess wearing
        the clothes of an estimate. It is; and the strategist has a race on
        Sunday either way, and a labelled guess with a five-fold band is what
        they can act on where a refusal is not.
    """
    here = cell_rates(practice[practice["event"] == event], FORECAST_MIN_RUNS)
    elsewhere = cell_rates(practice[practice["event"] != event], min_runs)
    return stand_in_from_rates(here, elsewhere, factor, event, wanted)


def stand_in_from_rates(
    here: pd.DataFrame,
    elsewhere: pd.DataFrame,
    factor: float,
    event: str,
    wanted: Iterable[str],
) -> pd.DataFrame:
    """The arithmetic of ``stand_in_rates``, on rate tables rather than laps.

    ``here`` is this event's ``cell_rates`` at ``FORECAST_MIN_RUNS``;
    ``elsewhere`` is every other event's at ``MIN_RUNS``. Split out so the
    accuracy test in ``tyredeg.paper.transfer_eval`` can hide one measured
    cell and impute it thousands of times without refitting every lap, while
    exercising exactly the code the screen runs.
    """
    if not here.empty:
        here = here[(here["rate"] > 0) & identified(here["rate"], here["se"])]
    measured = set(here["C"]) if not here.empty else set()
    missing = [c for c in wanted if c not in measured]
    if not missing:
        return pd.DataFrame()

    if elsewhere.empty:
        return pd.DataFrame()
    pooled = elsewhere.groupby("C")["rate"].median()

    sev = circuit_severity(here, pooled) if not here.empty else None
    scaled = sev is not None
    if not scaled:
        sev = 1.0

    # Keep the stand-ins in physical order against what we measured.
    #
    # Severity is calibrated on the compounds this weekend ran, so a circuit
    # with one extreme measured cell drags the scale for everything else.
    # Madrid measured C3 at 0.394 s/lap -- the harshest cell on the calendar --
    # which set severity at 2.6x, which put the borrowed C4 at 0.354. That says
    # the SOFT wears more slowly than the MEDIUM, which no tyre does.
    #
    # So a stand-in is clamped into the window its neighbours leave it: never
    # below a measured harder compound, never above a measured softer one.
    # Measured rows are never touched. Without this the optimiser built
    # Madrid's plan out of two borrowed compounds and ignored the only one we
    # actually watched run.
    known = dict(zip(here["C"], here["rate"], strict=True)) if not here.empty else {}

    def clamp(c: str, rate: float) -> float:
        i = C_ORDER.index(c) if c in C_ORDER else None
        if i is None:
            return rate
        harder = [known[k] for k in C_ORDER[:i] if k in known]
        softer = [known[k] for k in C_ORDER[i + 1 :] if k in known]
        if harder:
            rate = max(rate, max(harder))
        if softer:
            rate = min(rate, min(softer))
        return rate

    multiple = STANDIN_INTERVAL_MULTIPLE if scaled else POOLED_INTERVAL_MULTIPLE
    rows = []
    for c in missing:
        if c not in pooled.index:
            continue
        base = float(pooled[c])
        spread = float(elsewhere.loc[elsewhere["C"] == c, "rate"].std(ddof=0) or 0.0)
        rate = clamp(c, base * sev)
        half = multiple * max(spread, abs(base) * 0.5) * sev
        rows.append(
            {
                "event": event,
                "C": c,
                "rate": rate,
                "se": np.nan,
                "n_runs": 0,
                "n_laps": 0,
                "age_spread": np.nan,
                "predicted_race_rate": rate * factor,
                "lo": (rate - half) * factor,
                "hi": (rate + half) * factor,
                "source": "stand-in",
                "severity": round(sev, 3) if scaled else np.nan,
            }
        )
    return pd.DataFrame(rows)


def per_session_rates(
    practice: pd.DataFrame,
    event: str,
    min_runs: int = 1,
) -> pd.DataFrame:
    """What each practice session says on its own, before they are blended.

    A mentor asked for exactly this: summarise FP1, then FP2, then FP3, then
    tell me how you combined them. The pooled fit answers the last part and
    hides the first three, so this reports each session separately.

    It is a DIAGNOSTIC, not the forecast, and ``min_runs`` is deliberately as
    low as it goes. A single session rarely clears the pooled bar -- Madrid's
    FP1 ran exactly one race-simulation run on each compound -- and at the
    previous floor of two the screen printed "no race-simulation long runs"
    for a session that had done six laps on each tyre. That is not strictness,
    it is a blank where an observation exists.

    Nothing here reaches a prediction. The forecast fits through ``cell_rates``
    at MIN_RUNS, and lowering this floor does not move it. Rows carry their run
    and lap counts, and anything under three runs comes back flagged, so a thin
    cell is visible as thin rather than read as a measurement.

    Returns columns: session, C, rate, se, n_runs, n_laps, weight.
    """
    sub = practice[practice["event"] == event]
    if sub.empty:
        return pd.DataFrame()

    out = []
    for session, g in sub.groupby("session", observed=True):
        # cell_rates keys on event, so relabel each session as its own "event"
        # and fit them independently. Same estimator, same clustering, one
        # session at a time.
        r = cell_rates(g.assign(event=session), min_runs=min_runs)
        if r.empty:
            continue
        r = r.rename(columns={"event": "session"})
        r["weight"] = float(g["w"].iloc[0]) if "w" in g.columns else 1.0
        out.append(r)

    if not out:
        return pd.DataFrame()
    cols = ["session", "C", "rate", "se", "n_runs", "n_laps", "weight"]
    return pd.concat(out, ignore_index=True)[cols].sort_values(["session", "C"])

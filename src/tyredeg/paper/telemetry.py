"""Does car telemetry show drivers nursing soft tyres harder in races than in practice?

WHY THIS EXISTS

The paper's headline is that race-observed degradation is policy-dependent: the
race / practice degradation ratio is SOFT 0.17, MEDIUM 0.39, HARD 0.49. The
explanation we give is behavioural -- in a race a driver nurses a fragile tyre
to hit a target lap time, so measured degradation is suppressed. In practice the
team is deliberately measuring the tyre and pushes it.

That mechanism is currently INFERRED from lap times, which is circular: the
suppressed degradation is both the thing to be explained and the only evidence
that the nursing happens. Telemetry is the one public source that records HOW a
lap was driven rather than how fast it was. If drivers nurse tyres, it should
show up as more lifting off before braking zones ("lift and coast"), less time
at full throttle, and lower speed through corners -- and, the key prediction,
the race-minus-practice change should be LARGER on softer compounds.

PRE-SPECIFIED DESIGN (written before any telemetry was analysed)

Sample.  2025 and 2026 seasons. Per event, the Race and, for the practice side,
FP2 (conventional weekends) -- the session in which race simulations are run.
Sprint weekends have no FP2; their FP1 is sprint prep and is NOT used as a
practice stand-in in the primary analysis (see ``config.CONVENTIONAL_2026``).
Race laps: ``is_long_run`` green laps. Practice laps: ``is_long_run`` green laps
whose run passes ``classify_runs`` as a race simulation -- the same laps the
degradation ratio is built from, so the telemetry and the ratio describe one
population of driving.

Management proxies, one value per lap.  All thresholds are constants below.
  coast_s      PRIMARY. Seconds of lift-and-coast immediately before a braking
               zone: Throttle < 1 %, Brake off, Speed > 150 km/h, in the unbroken
               stretch that ends where braking begins. Coasting is the cleanest
               "I am saving the tyre / fuel / battery" signature.
  coast_total_s  Same test, anywhere on the lap (includes flat-lift corners).
  full_throttle_frac  Fraction of lap time at Throttle >= 98 %.
  top_speed    Maximum speed (km/h). Dominated by slipstream and DRS; context only.
  apex_speed   Mean of the speed minima of the lap's corners (km/h). A cornering
               push proxy: nursing front tyres means slower mid-corner speed.
  brake_decel_g  Mean deceleration over valid braking zones, in g. Gentler
               braking is the braking-side signature of nursing.
  brake_frac   Fraction of lap time inside valid braking zones.
A "valid" braking zone is >= 0.3 s of Brake and a speed loss of >= 20 km/h. The
brake flag in the feed is noisy (it flickers on at constant speed), so a flag
alone is not allowed to define a zone.

Normalisation.  Nothing is compared across circuits. Every contrast is race
minus practice WITHIN one (season, event, compound) cell, after adjusting for
tyre age and, in the race, race progress as the fuel proxy.

Predicted signs (race - practice, more positive for softer tyres):
  coast_s +, coast_total_s +, full_throttle_frac -, apex_speed -,
  brake_decel_g - (gentler). top_speed and brake_frac carry no prediction.
Only coast_s is PRIMARY; the other proxies are reported as secondary and no
multiplicity correction is applied to them, so read them as corroboration only.

Test.  Within-event slope of the race-minus-practice cell contrast on softness
(HARD 0, MEDIUM 1, SOFT 2), i.e. a regression with event fixed effects; a
within-event permutation of the compound labels; an event-clustered bootstrap
for the CI. The link to degradation correlates each cell's contrast with its
race / practice degradation ratio after centring both on compound, with a
permutation within compound and an event-clustered bootstrap.

CONFOUNDERS, STATED UP FRONT

* Fuel saving. Fuel saving lifts and coasts on EVERY compound alike, so a common
  race-minus-practice shift is expected and is not evidence of tyre nursing. Only
  the compound-by-session interaction is. That is why the test is the slope on
  softness and not the level.
* 2026 power units. The 2026 cars harvest and deploy far more electrical energy
  and lift-and-coast is a normal energy-management tool, so coasting in 2026 is
  not a clean tyre signal. 2025 and 2026 are therefore also reported separately.
* Traffic and DRS trains, safety-car restarts and different track evolution
  between sessions are NOT controlled. Only green, accurate, non-pit laps enter.
* Practice is not a controlled experiment: run plans, fuel and engine modes are
  unobserved. Practice fuel load in particular is unknown.
* The brake flag is boolean and the speed channel is sample-and-hold at ~4 Hz.
  Absolute values of the braking proxies are coarse; only contrasts are used.
* Compounds are the weekend-relative labels. Within an event the label is one
  physical tyre in both sessions, which is what the within-event contrast needs.

DATA SOURCE

Car telemetry is not in the lap cache. It needs ``CarData.z`` from the F1 live
timing static host, which answered HTTP 401 to every request when this was
written (FastF1's mirror: 404). Only the 2026 Hungarian GP FP2 and Race were
already cached. The public OpenF1 API (https://api.openf1.org/v1) serves the
same car channels (speed, throttle, brake, rpm, gear, DRS at ~3.7 Hz) for 2023
onward, so it is the primary source. OpenF1 and FastF1 samples are both the F1
car-data feed, but OpenF1 reports brake as 0/100 and throttle as 0-100 (no 104
sentinel); ``openf1_lap_frame`` converts to the layout ``lap_proxies`` expects
and the proxies and thresholds are unchanged. The two Hungary 2026 sessions
that exist in both sources are used to cross-check the sources lap by lap.
The source used for every number is recorded in ``paper/results/telemetry.json``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Pre-specified constants. Do not tune these after looking at a contrast.
# --------------------------------------------------------------------------- #

#: Throttle below this percentage counts as off the throttle.
COAST_THROTTLE_PCT = 1.0
#: Coasting is only counted above this speed, to exclude pit-lane and slow corners.
COAST_MIN_SPEED_KMH = 150.0
#: Throttle at or above this percentage counts as full throttle (the feed shows
#: 100, and 104 as a sentinel for "pedal fully pressed").
FULL_THROTTLE_PCT = 98.0
#: A braking zone is a Brake run at least this long ...
BRAKE_MIN_S = 0.3
#: ... that sheds at least this much speed. Filters brake-flag flicker.
BRAKE_MIN_SPEED_LOSS_KMH = 20.0
#: Brake runs separated by less than this are one zone (flag dropouts).
BRAKE_MERGE_GAP_S = 0.3
#: A corner apex is a speed minimum at least this far below its neighbours.
APEX_PROMINENCE_KMH = 40.0
#: Cap on how long one sample is held. Longer gaps are data dropouts, not driving.
MAX_SAMPLE_S = 0.6
#: A lap needs at least this many samples and this much covered time to count.
MIN_SAMPLES = 150
MIN_COVERAGE = 0.9
#: Braking deceleration only for zones entered at or above this speed.
DECEL_MIN_ENTRY_KMH = 150.0
G = 9.80665

PROXIES = (
    "coast_s",
    "coast_total_s",
    "full_throttle_frac",
    "top_speed",
    "apex_speed",
    "brake_decel_g",
    "brake_frac",
)
PRIMARY = "coast_s"
#: Predicted sign of (race - practice) rising with softness. 0 = no prediction.
PREDICTED_SIGN = {
    "coast_s": +1,
    "coast_total_s": +1,
    "full_throttle_frac": -1,
    "apex_speed": -1,
    "brake_decel_g": -1,
    "top_speed": 0,
    "brake_frac": 0,
}
SOFTNESS = {"HARD": 0, "MEDIUM": 1, "SOFT": 2}

#: Minimum runs on each side of a cell, and laps on each side.
MIN_RUNS_SIDE = 3
MIN_LAPS_SIDE = 15


# --------------------------------------------------------------------------- #
# Per-lap proxies
# --------------------------------------------------------------------------- #

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (start, end) index pairs of each run of True in ``mask``."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _seconds(t: pd.Series) -> np.ndarray:
    if pd.api.types.is_timedelta64_dtype(t):
        return t.dt.total_seconds().to_numpy(float)
    return t.to_numpy(float)


def braking_zones(t: np.ndarray, speed: np.ndarray, brake: np.ndarray,
                  dt: np.ndarray) -> list[tuple[int, int]]:
    """Valid braking zones as inclusive sample-index pairs.

    Brake runs are merged across dropouts shorter than ``BRAKE_MERGE_GAP_S``
    and kept only if long enough and if they actually shed speed. Without the
    speed test the brake flag's flicker at constant speed creates zones that do
    not exist.
    """
    zones: list[list[int]] = []
    for s, e in _runs(brake):
        if zones and t[s] - t[zones[-1][1]] <= BRAKE_MERGE_GAP_S:
            zones[-1][1] = e
        else:
            zones.append([s, e])
    out = []
    for s, e in zones:
        dur = t[e] + dt[e] - t[s]
        if dur < BRAKE_MIN_S:
            continue
        if speed[s] - speed[s:e + 1].min() < BRAKE_MIN_SPEED_LOSS_KMH:
            continue
        out.append((s, e))
    return out


def lap_proxies(car: pd.DataFrame) -> dict[str, float] | None:
    """Management proxies for ONE lap of car telemetry.

    Args:
        car: Frame with ``Time`` (timedelta or seconds), ``Speed`` (km/h),
            ``Throttle`` (%), ``Brake`` (bool). As returned by
            ``lap.get_car_data()``.

    Returns:
        ``{proxy: value}`` for every name in ``PROXIES`` (NaN when a lap has
        no corner or no braking zone, as for ``apex_speed``), or ``None`` when
        the lap has too little data to trust (fewer than ``MIN_SAMPLES``
        samples, or under ``MIN_COVERAGE`` of the lap time covered).
    """
    if car is None or len(car) < MIN_SAMPLES:
        return None
    t = _seconds(car["Time"])
    speed = car["Speed"].to_numpy(float)
    thr = car["Throttle"].to_numpy(float)
    brake = car["Brake"].to_numpy(bool)
    if not np.all(np.diff(t) >= 0):
        order = np.argsort(t, kind="stable")
        t, speed, thr, brake = t[order], speed[order], thr[order], brake[order]

    # Each sample is held until the next arrives. Capped so a dropout does not
    # masquerade as 5 seconds of coasting.
    gaps = np.diff(t)
    typical = float(np.median(gaps)) if len(gaps) else 0.25
    dt = np.append(gaps, typical)
    covered_raw = t[-1] - t[0] + typical
    dt = np.minimum(dt, MAX_SAMPLE_S)
    lap_s = float(dt.sum())
    if lap_s <= 0 or lap_s < MIN_COVERAGE * covered_raw:
        return None

    off = (thr < COAST_THROTTLE_PCT) & ~brake & (speed > COAST_MIN_SPEED_KMH)
    coast_total = float(dt[off].sum())

    zones = braking_zones(t, speed, brake, dt)
    coast_pre = 0.0
    for s, _ in zones:
        j = s - 1
        while j >= 0 and off[j]:
            coast_pre += dt[j]
            j -= 1

    decels, brake_time = [], 0.0
    for s, e in zones:
        brake_time += float(dt[s:e + 1].sum())
        if speed[s] < DECEL_MIN_ENTRY_KMH:
            continue
        k = s + int(np.argmin(speed[s:e + 1]))
        span = t[k] + dt[k] - t[s]
        if span > 0:
            decels.append((speed[s] - speed[k]) / 3.6 / span / G)

    apex = float("nan")
    try:
        from scipy.signal import find_peaks

        idx, _ = find_peaks(-speed, prominence=APEX_PROMINENCE_KMH)
        if len(idx):
            apex = float(speed[idx].mean())
    except Exception:  # scipy is a hard dependency, but never let one lap kill a batch
        pass

    return {
        "coast_s": float(coast_pre),
        "coast_total_s": coast_total,
        "full_throttle_frac": float(dt[thr >= FULL_THROTTLE_PCT].sum() / lap_s),
        "top_speed": float(speed.max()),
        "apex_speed": apex,
        "brake_decel_g": float(np.mean(decels)) if decels else float("nan"),
        "brake_frac": float(brake_time / lap_s),
    }


def openf1_lap_frame(date_ns: np.ndarray, speed: np.ndarray, throttle: np.ndarray,
                     brake: np.ndarray, start_ns: int, end_ns: int) -> pd.DataFrame | None:
    """Cut one lap out of an OpenF1 car-data stream.

    ``date_ns`` must be sorted UTC timestamps in integer nanoseconds. The lap is
    ``[start_ns, end_ns)`` where start is OpenF1's lap ``date_start`` and end is
    start plus the lap duration. Returns the frame ``lap_proxies`` expects
    (``Time`` in seconds from lap start, ``Speed``, ``Throttle``, ``Brake`` as
    bool) or None when no samples fall inside. OpenF1 brake is 0/100, so any
    positive value means braking.
    """
    i0, i1 = np.searchsorted(date_ns, [start_ns, end_ns], side="left")
    if i1 - i0 < 2:
        return None
    return pd.DataFrame({
        "Time": (date_ns[i0:i1] - start_ns) / 1e9,
        "Speed": speed[i0:i1].astype(float),
        "Throttle": throttle[i0:i1].astype(float),
        "Brake": brake[i0:i1] > 0,
    })


# --------------------------------------------------------------------------- #
# Lap selection (mirrors the degradation analysis so the populations match)
# --------------------------------------------------------------------------- #

def select_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Race and practice laps that enter the telemetry comparison.

    ``laps`` is a processed laps frame (see ``scripts/01_cache_sessions.py``).
    Race side: long-run laps, stint outliers dropped. Practice side: FP2 only,
    long-run laps of runs classified as race simulations. This is the same
    filtering ``scripts/07_management.py`` applies, minus the design-matrix
    transforms, so the proxies describe the laps behind the degradation ratio.

    Adds ``side`` ("race"/"practice") and ``race_progress`` (lap number over the
    race's last lap; 0 for practice, where fuel is unobserved).
    """
    from ..models.design import (
        classify_runs,
        drop_non_representative_laps,
        drop_stint_outliers,
    )

    d = laps[laps["Compound"].isin(SOFTNESS)].dropna(subset=["TyreLife", "LapTimeSeconds"])
    race = drop_stint_outliers(d[(d["session"] == "R") & d["is_long_run"]]).copy()
    race["side"] = "race"
    last = d[d["session"] == "R"].groupby("event")["LapNumber"].max()
    race["race_progress"] = race["LapNumber"] / race["event"].map(last)

    prac = d[d["session"] == "FP2"]
    prac = drop_non_representative_laps(prac)
    prac = classify_runs(drop_stint_outliers(prac[prac["is_long_run"]]))
    prac = prac[prac["is_race_sim"]].copy()
    prac["side"] = "practice"
    prac["race_progress"] = 0.0

    return pd.concat([race, prac], ignore_index=True)


# --------------------------------------------------------------------------- #
# Lap-level adjustment and cell contrasts
# --------------------------------------------------------------------------- #

def purge_age_and_fuel(laps: pd.DataFrame, proxy: str) -> pd.DataFrame:
    """Add ``_adj``: the proxy with tyre age and race progress removed.

    One age slope per side and one race-progress slope (race only), all
    estimated from variation WITHIN (season, event, compound, side) cells, so
    level differences between compounds cannot leak into the slopes. Every lap is
    then evaluated at the common mean tyre age and, for race laps, the mean race
    progress. Fuel burn is the same on every compound, so one progress slope is
    the honest control; practice fuel is unobserved and gets none.
    """
    d = laps.dropna(subset=[proxy, "TyreLife"]).copy()
    cell = ["season", "event", "Compound", "side"]
    d["_cell"] = d.groupby(cell).ngroup()

    y = d[proxy] - d.groupby("_cell")[proxy].transform("mean")
    x_age = d["TyreLife"] - d.groupby("_cell")["TyreLife"].transform("mean")
    x_prog = d["race_progress"] - d.groupby("_cell")["race_progress"].transform("mean")
    cols, names = [], []
    for side in ("race", "practice"):
        m = (d["side"] == side).to_numpy()
        cols.append(np.where(m, x_age, 0.0))
        names.append(f"age_{side}")
    cols.append(np.where(d["side"] == "race", x_prog, 0.0))
    names.append("progress_race")
    X = np.column_stack(cols)
    ok = np.isfinite(y.to_numpy()) & np.isfinite(X).all(axis=1)
    beta, *_ = np.linalg.lstsq(X[ok], y.to_numpy()[ok], rcond=None)
    # Evaluate every lap at a common reference: the mean tyre age, and for race
    # laps the mean race progress. Practice has no progress term (fuel is
    # unobserved there), so its progress adjustment is zero by construction.
    ref_age = float(d["TyreLife"].mean())
    is_race = (d["side"] == "race").to_numpy()
    ref_prog = float(d.loc[is_race, "race_progress"].mean()) if is_race.any() else 0.0
    b_age = np.where(is_race, beta[0], beta[1])
    d["_adj"] = (
        d[proxy].to_numpy()
        - b_age * (d["TyreLife"].to_numpy() - ref_age)
        - np.where(is_race, beta[2] * (d["race_progress"].to_numpy() - ref_prog), 0.0)
    )

    return d


def adjusted_cells(laps: pd.DataFrame, proxy: str) -> pd.DataFrame:
    """Race-minus-practice contrast per (season, event, compound), age- and fuel-adjusted.

    The proxy is first purged of tyre age (within each session side) and of race
    progress (race only), using one slope per side estimated from variation
    WITHIN cells -- so cell level differences between compounds cannot leak into
    the slopes. Fuel burn is the same on every compound, so a common progress
    slope is the honest control. Cell means of the purged proxy are then
    differenced, race minus practice.

    The slopes are treated as known when bootstrapping (they are estimated from
    thousands of laps and the bootstrap is over events); this makes the CI a
    little narrow and is the one simplification in the uncertainty.

    Needs ``MIN_RUNS_SIDE`` runs and ``MIN_LAPS_SIDE`` laps on each side of a
    cell. Returns one row per surviving cell with ``delta`` and the run counts.
    """
    d = purge_age_and_fuel(laps, proxy)

    rows = []
    for (season, event, comp), g in d.groupby(["season", "event", "Compound"]):
        r, p = g[g["side"] == "race"], g[g["side"] == "practice"]
        if (r["run_id"].nunique() < MIN_RUNS_SIDE or p["run_id"].nunique() < MIN_RUNS_SIDE
                or len(r) < MIN_LAPS_SIDE or len(p) < MIN_LAPS_SIDE):
            continue
        rows.append({
            "season": int(season), "event": event, "Compound": comp,
            "race_mean": float(r["_adj"].mean()), "practice_mean": float(p["_adj"].mean()),
            "delta": float(r["_adj"].mean() - p["_adj"].mean()),
            "raw_delta": float(r[proxy].mean() - p[proxy].mean()),
            "n_race_laps": len(r), "n_practice_laps": len(p),
            "n_race_runs": int(r["run_id"].nunique()),
            "n_practice_runs": int(p["run_id"].nunique()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# The test: within-event slope on softness
# --------------------------------------------------------------------------- #

def _within_event_slope(event: np.ndarray, soft: np.ndarray, y: np.ndarray) -> float:
    """Slope of ``y`` on ``soft`` with event fixed effects (events with a single
    cell, or no softness variation, carry no information and drop out)."""
    num = den = 0.0
    for ev in np.unique(event):
        m = event == ev
        if m.sum() < 2:
            continue
        s = soft[m] - soft[m].mean()
        den_e = float(s @ s)
        if den_e <= 0:
            continue
        num += float(s @ (y[m] - y[m].mean()))
        den += den_e
    return num / den if den > 0 else float("nan")


@dataclass
class TrendResult:
    n_cells: int
    n_events: int
    slope: float
    ci95: tuple[float, float]
    p_perm_one_sided: float
    p_perm_two_sided: float
    n_perm: int
    perm_exact: bool
    predicted_sign: int
    per_compound: dict[str, dict[str, float]]


def trend_test(cells: pd.DataFrame, predicted_sign: int = +1, n_boot: int = 4000,
               n_perm: int = 20000, seed: int = 20260905) -> TrendResult | None:
    """Does the race-minus-practice contrast change with softness, within event?

    * Statistic: event-fixed-effects slope of ``delta`` on softness.
    * p-value: permute compound labels among the cells of each event (so every
      event keeps its own level and its own set of compounds), recompute the
      slope. One-sided in ``predicted_sign``'s direction -- the direction was
      fixed before looking -- and two-sided alongside. When the number of
      distinct permutations is small it is enumerated exactly, because a Monte
      Carlo p-value cannot be smaller than the permutation count allows anyway.
    * CI: resample EVENTS (season, event) with replacement -- cells within an
      event share a track, a race and a set of drivers, so they are not
      independent -- and take the 2.5/97.5 percentiles of the slope.

    Returns None when fewer than two events have two or more compounds: there is
    then no within-event comparison to make.
    """
    c = cells.copy()
    c["ev"] = c["season"].astype(str) + "|" + c["event"]
    c["soft"] = c["Compound"].map(SOFTNESS).astype(float)
    # Keep only events that can contribute a within-event comparison.
    multi = c.groupby("ev")["soft"].transform("nunique") >= 2
    c = c[multi]
    if c["ev"].nunique() < 1 or len(c) < 2:
        return None

    ev = c["ev"].to_numpy()
    soft = c["soft"].to_numpy()
    y = c["delta"].to_numpy(float)
    obs = _within_event_slope(ev, soft, y)
    rng = np.random.default_rng(seed)

    groups = {e: np.flatnonzero(ev == e) for e in np.unique(ev)}
    # Number of distinct label assignments: product of k! over events.
    n_dist = 1
    for idx in groups.values():
        n_dist *= math.factorial(len(idx))
    perm_stats = []
    exact = n_dist <= n_perm
    if exact:
        import itertools

        per_event = [list(itertools.permutations(soft[idx])) for idx in groups.values()]
        for combo in itertools.product(*per_event):
            s2 = soft.copy()
            for idx, lab in zip(groups.values(), combo, strict=True):
                s2[idx] = lab
            perm_stats.append(_within_event_slope(ev, s2, y))
        n_used = n_dist
    else:
        for _ in range(n_perm):
            s2 = soft.copy()
            for idx in groups.values():
                s2[idx] = rng.permutation(soft[idx])
            perm_stats.append(_within_event_slope(ev, s2, y))
        n_used = n_perm
    ps = np.asarray(perm_stats)
    ps = ps[np.isfinite(ps)]
    # +1 smoothing for Monte Carlo; exact enumeration includes the observed.
    if exact:
        p_one = float(np.mean(ps * predicted_sign >= obs * predicted_sign - 1e-12)) if predicted_sign else float("nan")
        p_two = float(np.mean(np.abs(ps) >= abs(obs) - 1e-12))
    else:
        p_one = float((1 + np.sum(ps * predicted_sign >= obs * predicted_sign - 1e-12)) / (1 + len(ps))) if predicted_sign else float("nan")
        p_two = float((1 + np.sum(np.abs(ps) >= abs(obs) - 1e-12)) / (1 + len(ps)))

    # Event-clustered bootstrap.
    evs = list(groups)
    boots = []
    if len(evs) >= 2:
        for _ in range(n_boot):
            pick = rng.integers(0, len(evs), len(evs))
            e2, s2, y2 = [], [], []
            for k, j in enumerate(pick):
                idx = groups[evs[j]]
                e2.append(np.full(len(idx), k))
                s2.append(soft[idx])
                y2.append(y[idx])
            b = _within_event_slope(np.concatenate(e2), np.concatenate(s2), np.concatenate(y2))
            if np.isfinite(b):
                boots.append(b)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"), float("nan"))

    per = {}
    for comp in SOFTNESS:
        g = c[c["Compound"] == comp]
        if len(g):
            per[comp] = {"n_cells": int(len(g)), "mean_delta": float(g["delta"].mean())}
    return TrendResult(
        n_cells=len(c), n_events=len(groups), slope=obs, ci95=ci,
        p_perm_one_sided=p_one, p_perm_two_sided=p_two, n_perm=int(n_used),
        perm_exact=exact, predicted_sign=predicted_sign, per_compound=per,
    )


# --------------------------------------------------------------------------- #
# Link to the degradation ratio
# --------------------------------------------------------------------------- #

def _rank_centre_within(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.empty(len(values), float)
    for g in np.unique(groups):
        m = groups == g
        r = pd.Series(values[m]).rank().to_numpy()
        out[m] = r - r.mean()
    return out


def link_to_ratio(cells: pd.DataFrame, ratios: pd.DataFrame, n_boot: int = 4000,
                  n_perm: int = 20000, seed: int = 20260905) -> dict:
    """Does a cell's extra coasting predict a lower race / practice degradation ratio?

    Merges contrast cells with ``ratios`` (columns season, event, Compound,
    ratio) and correlates ``delta`` with ``ratio`` after rank-centring BOTH
    within compound. Centring is the point: SOFT cells have both the lowest
    ratios and (if the mechanism is real) the biggest contrasts, so a pooled
    correlation would be the compound effect restated. Within compound, a cell
    that nursed more than its peers should have degraded less in the race
    relative to practice: the predicted correlation is NEGATIVE for coast_s.

    p-value: permute ``ratio`` among cells of the same compound. CI: bootstrap
    over events. Returns ``{"estimable": False, ...}`` when there are fewer than
    8 matched cells or fewer than 4 events -- below that the number would be an
    anecdote.
    """
    m = cells.merge(ratios[["season", "event", "Compound", "ratio"]],
                    on=["season", "event", "Compound"])
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta", "ratio"])
    n_events = int(m[["season", "event"]].drop_duplicates().shape[0])
    if len(m) < 8 or n_events < 4:
        return {"estimable": False, "n_cells": int(len(m)), "n_events": n_events,
                "reason": "fewer than 8 matched cells or 4 events"}
    comp = m["Compound"].to_numpy()
    d = m["delta"].to_numpy(float)
    r = m["ratio"].to_numpy(float)

    def stat(dd, rr, cc):
        a, b = _rank_centre_within(dd, cc), _rank_centre_within(rr, cc)
        den = math.sqrt(float(a @ a) * float(b @ b))
        return float(a @ b) / den if den > 0 else float("nan")

    obs = stat(d, r, comp)
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        r2 = r.copy()
        for c in np.unique(comp):
            idx = np.flatnonzero(comp == c)
            r2[idx] = rng.permutation(r[idx])
        perm.append(stat(d, r2, comp))
    perm = np.asarray(perm)
    p_neg = float((1 + np.sum(perm <= obs + 1e-12)) / (1 + len(perm)))
    p_two = float((1 + np.sum(np.abs(perm) >= abs(obs) - 1e-12)) / (1 + len(perm)))

    ev = (m["season"].astype(str) + "|" + m["event"]).to_numpy()
    idx_by = {e: np.flatnonzero(ev == e) for e in np.unique(ev)}
    keys = list(idx_by)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        sel = np.concatenate([idx_by[keys[j]] for j in pick])
        b = stat(d[sel], r[sel], comp[sel])
        if np.isfinite(b):
            boots.append(b)
    return {
        "estimable": True, "n_cells": int(len(m)), "n_events": n_events,
        "within_compound_rank_corr": obs,
        "ci95_event_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "p_perm_one_sided_negative": p_neg, "p_perm_two_sided": p_two,
        "n_perm": int(n_perm),
    }


def degradation_ratio_cells(paths: dict[int, "pd.PathLike"]) -> pd.DataFrame:
    """Race / practice degradation ratio per (season, event, compound).

    Rebuilds the cells exactly as ``scripts/07_management.py`` does -- same
    filters, same ``cell_slopes`` -- so the ratios are the ones the paper
    quotes. Imported pieces are read-only here.
    """
    from ..data.laps import tag_long_runs
    from ..data.schedule import SPRINT_SESSION
    from ..models.design import (
        classify_runs,
        drop_non_representative_laps,
        drop_stint_outliers,
        practice_design,
        race_design,
    )
    from ..validation.management import LABELS, MIN_PRACTICE_RATE, cell_slopes

    frames = []
    for season, path in paths.items():
        laps = pd.read_parquet(path)
        laps = laps[laps["Compound"].isin(LABELS)].dropna(subset=["TyreLife", "LapTimeSeconds"])
        prac = drop_non_representative_laps(laps[~laps["session"].isin(["R", SPRINT_SESSION])])
        prac = tag_long_runs(prac)
        prac = classify_runs(drop_stint_outliers(prac[prac["is_long_run"]]))
        prac = practice_design(prac[prac["is_race_sim"]])
        race = race_design(drop_stint_outliers(laps[laps["is_long_run"]]))
        p = cell_slopes(prac).rename(columns={"rate": "practice"})
        r = cell_slopes(race).rename(columns={"rate": "race"})
        if p.empty or r.empty:
            continue
        j = p.merge(r, on=["event", "Compound"], suffixes=("_p", "_r"))
        j["season"] = season
        frames.append(j)
    if not frames:
        return pd.DataFrame(columns=["season", "event", "Compound", "ratio"])
    cells = pd.concat(frames, ignore_index=True)
    cells = cells[cells["practice"] > MIN_PRACTICE_RATE].copy()
    cells["ratio"] = cells["race"] / cells["practice"]
    return cells[["season", "event", "Compound", "practice", "race", "ratio"]]


# --------------------------------------------------------------------------- #
# Small helpers shared with the script
# --------------------------------------------------------------------------- #

def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

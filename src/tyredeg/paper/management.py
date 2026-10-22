"""Is race-observed tyre degradation policy-dependent? A same-estimator test.

WHY THIS FILE EXISTS

``validation/management.py`` compared race and practice with a per-cell ratio of
two noisy slopes, estimated two different ways. A hostile audit found four
problems with that, and this module answers each one:

1. A ratio of two noisy slopes is a poor estimand. 38% of SOFT race slopes were
   negative; practice slopes were 3-5x race slopes. The estimand here is a
   DIFFERENCE, race slope minus practice slope, and the claim is an interaction:
   does that difference change with compound softness?
2. The two sides used different estimators. Practice subtracted an assumed fuel
   effect and centred within the run; the race regressed field-demeaned lap
   times through the origin with no per-compound intercept. Here one function
   fits both (lap time + fuel correction, centred within the run).
3. Stint-length and curvature. Practice runs live at tyre ages 7-17, race stints
   at 2-30, so comparing whole runs compares different parts of the wear curve.
   Both contexts are cut to one tyre-age window.
4. Survivorship. Soft stints end early. A common-support window per event is
   run as a robustness row.

It also found, while building this, a defect in the legacy race estimator; see
LEGACY RACE ESTIMATOR below. That finding is why the same-estimator comparison
matters more than a refinement would have.

PRE-REGISTRATION (frozen in this docstring before any same-estimator slope was
computed; the counts used to choose the window were run first and printed no
slope)

Disclosure of what had been seen. The legacy ratio results (101 cells, rho
-0.29) were known. While auditing the legacy race estimator the author also saw
that centring its race slopes within the stint moves their per-compound medians
a great deal. That observation is the reason the primary uses a within-run race
estimator at all; it was NOT used to choose between the specifications below,
all of which are structural. (Run-centring later turned out not to be the right
repair of the legacy estimator either -- see LEGACY RACE ESTIMATOR.)

Data: every complete season 2022-2026; practice = FP1-FP3 runs that look like
race simulations (models.design.classify_runs); race = R; sprints excluded
(racing, not practice). Unit: (season, event, compound) cell needing >= 3 runs
in each context, >= 4 laps per run inside the window. Cluster: event-season
(the same circuit in two seasons is two clusters; 'event' alone would pool
different tyre allocations and cars).

Primary estimator: within-run fuel-corrected slope at 1.7 kg/lap x 0.033 s/kg,
identical code for both contexts, cluster-robust by run. Window: TyreLife 6-16
inclusive. Chosen from lap COUNTS, before any slope: practice runs start at a
median age of 8-10 and end at 14-17, so the range both contexts occupy is
roughly 6-16. The window suggested in the brief, 3-12, leaves 7 HARD cells and
is run as a multiverse row.

Estimand: delta_c = race slope_c - practice slope_c per cell. Test: weighted
regression of delta on softness rank (HARD 0, MEDIUM 1, SOFT 2) with event-season
fixed effects, weights 1 / (cell variance + tau^2), tau^2 by Paule-Mandel.
H1 (primary, directional): beta < 0, i.e. the race slope falls further BELOW
the practice slope the softer the tyre. Interval: percentile, event-season
cluster bootstrap, B = 5000, seed 20260101. Test: permutation of compound labels
within event-season, 10000 draws, one-sided (beta <= observed). The finding is
called 'supported' only if the bootstrap interval excludes zero AND both
permutation p-values are below 0.05, mirroring the strictness already used in
validation/management.py.

H2 (co-primary, because H1 alone cannot establish management): the additive
contrast falls with softness EVEN IF drivers simply damp every compound by the
same factor k < 1, because softer tyres have bigger practice slopes. H2 is the
log of (SOFT ratio-of-means / HARD ratio-of-means), ratio-of-means =
sum(race slopes) / sum(practice slopes). H2 < 0 means softer tyres are damped
MORE than proportionally. Same bootstrap and permutation. The paper may only
say 'compound-specific management' if H2 holds; H1 alone supports 'the race
compound gradient is flatter than the practice one'.

Secondary: pairwise contrasts SOFT-MEDIUM, SOFT-HARD, MEDIUM-HARD with sign
tests; a pooled-lap joint model (sums of cells, exact); an inverse-variance
meta-regression race ~ practice + softness.

Robustness, each an a-priori named row, never a selection: fuel burn {1.4, 1.7,
2.0} x mass sensitivity {0.025, 0.033, 0.040}; asymmetric race/practice burn;
window none / 3-12 / 4-14 / 6-16 / common-support; run length any / 7-15;
min runs 2 / 3 / 4; sprint in or out of practice; race estimator within-run vs
cross-sectional (demeaned within event, lap and compound); season sets all / leave-one-out / each alone /
pre-2026 / 2024-26.

Track temperature: add the cell's mean track temperature (interpolated to lap
start from the FastF1 weather stream) in each context as covariates.

Placebo: FP2 versus FP1+FP3 (same event, same race-sim filter). Nothing about
racing differs between them, so the compound trend should be zero. Reported
with its standard error so a null from low power is visible as one.

WHAT WOULD KILL THE CLAIM. H1's bootstrap interval spanning zero; the sign of
beta flipping across most of the multiverse; a placebo trend of the same size
as the primary; the trend vanishing once temperature is controlled.

THE ALGEBRA THAT MAKES THE FUEL GRID MOSTLY TRIVIAL. The fuel correction adds
burn x s/kg to EVERY run's slope, whatever its compound. A constant cancels in
delta = race - practice and again in any contrast between compounds. So the
interaction beta is exactly invariant to the fuel assumptions provided the same
(or any) burn is applied to every compound; only the LEVEL of delta and the
RATIO view move. The grid is run anyway, because the ratio view and the levels
do move, and the invariance is a theorem about the contrast that should be seen
to hold numerically.

LEGACY RACE ESTIMATOR (found while preparing this analysis)

validation.management.cell_slopes fits a slope THROUGH THE ORIGIN to the race's
(event, lap)-demeaned design, one compound at a time. Demeaning by (event, lap)
subtracts a field mean that mixes compounds, so each compound's lap times keep
their pace offset (soft is quicker) and each compound's tyre ages keep their age
offset (soft is fresher), and with no intercept their product leaks straight
into the slope. The practice design does not have the problem because it centres
within the run.

On a planted simulation (tests/test_paper_management.py) where SOFT and HARD
both truly degrade at 0.07 s/lap and soft is 0.9 s quicker, the legacy estimator
reads SOFT 0.13, HARD 0.085: a spurious compound gradient of 0.045 s/lap, the
same order as the effect under test. Centring the one-pass design within the
run, the obvious patch, is no better: it reads 0.04 / 0.067, because it
identifies the slope only from the drift of the field mean, which moves with the
compound mix. Demeaning within (event, lap, COMPOUND) recovers the truth, and
that is the repair now in validation.management (``demean_by_compound``).
paper/results/management.json records the legacy numbers as shipped, the
repaired legacy numbers, and the same-estimator numbers side by side.

WHAT THIS CANNOT DO

It is observational. A driver who nurses a soft tyre, a team that avoids long
soft stints, tyre sets of different age and prior use, differing fuel loads
between a race simulation and a race, and the choice of WHEN to run a compound
in practice are all consistent with the data and are not separable here. The
claim the data can bear is about the measured race slope, not about driver
intent.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import FASTF1_CACHE, MASS_SENSITIVITY_S_PER_KG, PROCESSED
from ..data.laps import tag_long_runs
from ..data.schedule import SPRINT_SESSION
from ..models.design import (
    PRACTICE_BURN_KG_PER_LAP,
    classify_runs,
    drop_non_representative_laps,
    drop_stint_outliers,
    race_design,
)
from ..validation.management import LABEL_RANK, LABELS, cluster_slope

#: Season -> parquet stem. 2026 is the un-suffixed file.
SEASON_FILES = {2022: "laps_2022", 2023: "laps_2023", 2024: "laps_2024", 2025: "laps_2025", 2026: "laps"}
ALL_SEASONS = tuple(sorted(SEASON_FILES))

RANKS = tuple(LABEL_RANK[lab] for lab in LABELS)  # HARD 0, MEDIUM 1, SOFT 2


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spec:
    """Every analyst degree of freedom in one place, so the multiverse is data.

    Defaults ARE the pre-specified primary analysis. A multiverse row is
    ``replace(PRIMARY, <one or two fields>)``.
    """

    seasons: tuple[int, ...] = ALL_SEASONS
    #: How the RACE slope is estimated. ``within_run``: lap time plus an assumed
    #: fuel correction, centred within the stint -- byte-for-byte the practice
    #: estimator. ``xs``: the cross-sectional design -- lap time and tyre age
    #: demeaned within (event, lap, COMPOUND), compared across cars of different
    #: tyre age at the same lap. It needs no fuel assumption and removes track
    #: evolution and the compound pace offset, but identifies the slope from
    #: other cars rather than from the run itself, so it is a different (and
    #: noisier) estimator, shown as a robustness row. A two-way fixed-effects fit
    #: (lap effect + run effect) cannot be used: tyre age is lap minus stint start,
    #: so both effects absorb it exactly. Practice is always ``within_run``.
    race_estimator: str = "within_run"  # "within_run" | "xs" | "field_origin" | "field_runcentred"
    burn_race: float = PRACTICE_BURN_KG_PER_LAP
    burn_practice: float = PRACTICE_BURN_KG_PER_LAP
    s_per_kg: float = MASS_SENSITIVITY_S_PER_KG
    #: Tyre-age window, inclusive, applied to the SAME TyreLife column in both
    #: contexts. ``None`` = unbounded on that side.
    age_lo: int | None = 6
    age_hi: int | None = 16
    #: Per event, replace ``age_hi`` by the oldest age every (context, compound)
    #: group in that event routinely reaches (median run end). Survivorship fix.
    common_support: bool = False
    #: Keep only runs whose full length (before windowing) is in [lo, hi].
    run_len: tuple[int, int] | None = None
    #: A run needs this many laps INSIDE the window to count at all.
    min_window_laps: int = 4
    #: A cell needs this many runs in BOTH contexts.
    min_runs: int = 3
    #: "exclude" (primary: a sprint is racing), or "practice" (the old pooling).
    sprint: str = "exclude"
    #: Practice runs must look like race simulations (pace + consistency).
    race_sim_only: bool = True
    practice_sessions: tuple[str, ...] = ("FP1", "FP2", "FP3")
    #: Set-state matching (added after independent review). Keep only runs whose
    #: FIRST tyre age lies in the range, separately per context. 82% of practice
    #: race-sim runs start on a used set (age >= 6) while race stints mostly start
    #: new (age 2-5), and used sets degrade differently, so the contexts are not
    #: otherwise like for like. ``None`` = no restriction.
    practice_start: tuple[int, int] | None = None
    race_start: tuple[int, int] | None = None
    #: Set state taken from the timing feed's own FreshTyre flag instead of being
    #: proxied by start age: ``True`` = stints that began on a NEW set, ``False`` =
    #: stints that began on a USED set, ``None`` = no restriction.
    practice_fresh: bool | None = None
    race_fresh: bool | None = None

    def label(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=list)


PRIMARY = Spec()


# ---------------------------------------------------------------------------
# Track temperature, read straight from the FastF1 cache (no network)
# ---------------------------------------------------------------------------

_SESSION_DIR = {
    "Practice_1": "FP1",
    "Practice_2": "FP2",
    "Practice_3": "FP3",
    "Race": "R",
    "Sprint": SPRINT_SESSION,
}
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}_")


def weather_streams(cache: Path = FASTF1_CACHE) -> dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]]:
    """(season, event, session) -> (seconds since session start, track temp C).

    Reads ``weather_data.ff1pkl`` directly. Going through ``fastf1.get_session``
    would try the network for the schedule even when every stream is on disk,
    and this analysis must run offline. The pickle is FastF1's own cache format
    (``{"version": ..., "data": {column: list}}``), and only TrackTemp and Time
    are read.
    """
    out: dict[tuple[int, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for season_dir in sorted(cache.iterdir()):
        if not (season_dir.is_dir() and season_dir.name.isdigit()):
            continue
        for ev_dir in sorted(season_dir.iterdir()):
            if not ev_dir.is_dir():
                continue
            event = _DATE.sub("", ev_dir.name).replace("_", " ")
            for ses_dir in sorted(ev_dir.iterdir()):
                code = _SESSION_DIR.get(_DATE.sub("", ses_dir.name))
                f = ses_dir / "weather_data.ff1pkl"
                if code is None or not f.exists():
                    continue
                with open(f, "rb") as fh:
                    data = pickle.load(fh)["data"]
                t = pd.to_timedelta(pd.Series(data["Time"])).dt.total_seconds().to_numpy(float)
                temp = np.asarray(data["TrackTemp"], dtype=float)
                keep = np.isfinite(t) & np.isfinite(temp)
                if keep.sum() >= 2:
                    order = np.argsort(t[keep])
                    out[(int(season_dir.name), event, code)] = (t[keep][order], temp[keep][order])
    return out


def attach_track_temp(df: pd.DataFrame, streams: dict) -> pd.DataFrame:
    """Track temperature at each lap's start, linearly interpolated. NaN if absent."""
    df = df.copy()
    df["track_temp"] = np.nan
    for (season, event, session), idx in df.groupby(["season", "event", "session"]).groups.items():
        stream = streams.get((int(season), event, session))
        if stream is None:
            continue
        t, temp = stream
        pos = df.index.get_indexer(idx)
        df.iloc[pos, df.columns.get_loc("track_temp")] = np.interp(
            df["SessionElapsed"].to_numpy(float)[pos], t, temp
        )
    return df


# ---------------------------------------------------------------------------
# Base frames: one clean pipeline for every context
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def data_version(seasons: tuple[int, ...] = ALL_SEASONS) -> dict:
    """What exactly was analysed, so a later re-run can tell if the data moved."""
    out = {}
    for s in seasons:
        p = PROCESSED / f"{SEASON_FILES[s]}.parquet"
        man = p.with_name(f"{p.stem}.manifest.json")
        info = {"file": p.name, "sha256_16": _sha(p), "bytes": p.stat().st_size}
        if man.exists():
            m = json.loads(man.read_text(encoding="utf-8"))
            info["manifest_written_at"] = m.get("written_at")
            info["complete"] = m.get("complete")
        out[str(s)] = info
    return out


def _runs_frame(laps: pd.DataFrame) -> pd.DataFrame:
    """Same run pipeline for every session type.

    Pace filter against the driver's own best, consecutive-lap blocks of at least
    five laps, then within-stint outlier removal. Identical for race and
    practice. The race was previously filtered a different way (its
    ``is_long_run`` was tagged before the pace filter ran).
    """
    d = drop_non_representative_laps(laps)
    d = tag_long_runs(d)
    d = d[d["is_long_run"]]
    d = drop_stint_outliers(d).copy()
    # Age of the set when the run began: separates new-set from used-set running.
    d["start_age"] = d.groupby("run_id")["TyreLife"].transform("min")
    # Did the STINT begin on a new set? One value per stint in the feed.
    d["fresh"] = d.groupby("run_id")["FreshTyre"].transform("first").astype(bool)
    return d


@dataclass
class Bases:
    """Per-season frames, built once and re-sliced by every :class:`Spec`."""

    practice: dict[int, pd.DataFrame]
    race: dict[int, pd.DataFrame]
    seasons: tuple[int, ...]

    @classmethod
    def load(cls, seasons: tuple[int, ...] = ALL_SEASONS, streams: dict | None = None) -> Bases:
        streams = weather_streams() if streams is None else streams
        practice, race = {}, {}
        for s in seasons:
            laps = pd.read_parquet(PROCESSED / f"{SEASON_FILES[s]}.parquet")
            laps = laps[laps["Compound"].isin(LABELS)].dropna(subset=["TyreLife", "LapTimeSeconds", "LapNumber"])
            laps = laps.assign(season=s).reset_index(drop=True)
            laps = attach_track_temp(laps, streams)
            # Age of the set at the first timed lap of the whole STINT, before any
            # pace or run filter. A run that starts later than its stint was cut
            # loose by a safety-car or yellow gap, not by a change of tyres.
            laps["stint_start_age"] = laps.groupby(["event", "session", "Driver", "Stint"])["TyreLife"].transform("min")

            prac = _runs_frame(laps[laps["session"] != "R"])
            if len(prac):
                prac = classify_runs(prac)
                # A sprint has no qualifying runs to tell apart, so the pace test
                # must not discard it (same rule as models.design.prepare).
                prac.loc[prac["session"] == SPRINT_SESSION, "is_race_sim"] = True
            practice[s] = prac

            rc = _runs_frame(laps[laps["session"] == "R"])
            if len(rc):
                # Cross-sectional design: lap time and tyre age minus their mean over
                # the cars on the SAME compound at the SAME lap. Computed on the full
                # set of long-run laps BEFORE any tyre-age window, so the reference is
                # the same whatever window a spec chooses. race_design supplies the
                # sample rule (>= 4 cars on the lap).
                rc = race_design(rc, with_traffic=False).rename(columns={"y": "y_field", "tl": "tl_field"})
                rc = rc.drop(columns=["tl2"], errors="ignore")
                g = rc.groupby(["event", "LapNumber", "Compound"], observed=True)
                single = g["LapTimeSeconds"].transform("size") < 2
                rc["y_xs"] = (rc["LapTimeSeconds"] - g["LapTimeSeconds"].transform("mean")).mask(single)
                rc["tl_xs"] = (rc["TyreLife"] - g["TyreLife"].transform("mean")).mask(single)
            race[s] = rc
        return cls(practice=practice, race=race, seasons=tuple(seasons))


# ---------------------------------------------------------------------------
# Window, estimator, cells
# ---------------------------------------------------------------------------


def _common_support_caps(frames: list[pd.DataFrame], spec: Spec) -> dict[tuple[int, str], int]:
    """Per event, the age that every (context, compound) group routinely reaches.

    The median, across a group's runs, of the oldest tyre age in the run; then the
    minimum over groups. A soft stint that ends at lap 9 therefore caps the
    window for the whole event, so a hard stint's laps 10-25 can no longer
    contribute a curvature the soft tyre never lived long enough to show.
    """
    reach = []
    for f in frames:
        if f.empty:
            continue
        r = f.groupby(["season", "event", "Compound", "run_id"], observed=True)["TyreLife"].max()
        reach.append(r.groupby(["season", "event", "Compound"], observed=True).median())
    if not reach:
        return {}
    allr = pd.concat(reach)
    cap = allr.groupby(level=["season", "event"]).min()
    floor = (spec.age_lo or 1) + spec.min_window_laps - 1
    return {k: int(max(np.floor(v), floor)) for k, v in cap.items()}


def _window(df: pd.DataFrame, spec: Spec, caps: dict | None) -> pd.DataFrame:
    if df.empty:
        return df
    d = df
    if spec.run_len is not None:
        lo, hi = spec.run_len
        d = d[(d["run_len"] >= lo) & (d["run_len"] <= hi)]
    if spec.age_lo is not None:
        d = d[d["TyreLife"] >= spec.age_lo]
    if caps is not None:
        hi = pd.Series([caps.get((s, e), 10**6) for s, e in zip(d["season"], d["event"], strict=True)], index=d.index)
        d = d[d["TyreLife"] <= hi]
    elif spec.age_hi is not None:
        d = d[d["TyreLife"] <= spec.age_hi]
    n = d.groupby("run_id")["TyreLife"].transform("size")
    return d[n >= spec.min_window_laps].copy()


def _centre(d: pd.DataFrame, y: pd.Series, x: pd.Series) -> pd.DataFrame:
    g = d["run_id"]
    d = d.copy()
    d["y"] = y.to_numpy() - y.groupby(g).transform("mean").to_numpy()
    d["tl"] = x.to_numpy() - x.groupby(g).transform("mean").to_numpy()
    return d


def prepare_context(
    d: pd.DataFrame, spec: Spec, *, race: bool, burn: float, caps: dict | None
) -> pd.DataFrame:
    """Window, then build the centred design (``y``, ``tl``) the slope is fitted on."""
    d = _window(d, spec, caps)
    if d.empty:
        return d
    if race and spec.race_estimator == "xs":
        # Already centred within (event, lap, compound); NOT re-centred within the
        # run, because the identification is cross-sectional (cars of different
        # tyre age at the same lap), and centring within the run would throw it away.
        # A lap with no same-compound partner carries no comparison (NaN): drop it,
        # for this estimator only -- the within-run estimator keeps it.
        d = d.dropna(subset=["y_xs"])
        d = d[d.groupby("run_id")["TyreLife"].transform("size") >= spec.min_window_laps]
        return d.assign(y=d["y_xs"], tl=d["tl_xs"])
    if race and spec.race_estimator == "field_origin":
        # The legacy design: minus the (event, lap) field mean over ALL compounds, slope
        # through the origin. Kept only to reconcile with analyses that use it.
        return d.assign(y=d["y_field"], tl=d["tl_field"])
    if race and spec.race_estimator == "field_runcentred":
        return _centre(d, d["y_field"], d["tl_field"])
    # Fuel burned is proportional to laps flown, so adding burn x s/kg per lap
    # back removes it. The absolute offset is absorbed by the run centring.
    y = d["LapTimeSeconds"] + burn * spec.s_per_kg * d["LapNumber"]
    return _centre(d, y, d["TyreLife"].astype(float))


def context_frames(
    bases: Bases, spec: Spec, *, practice_sessions: tuple[str, ...] | None = None, prepare: bool = True
):
    """Race and practice analysis frames for ``spec`` (windowed and centred)."""
    sessions = tuple(practice_sessions or spec.practice_sessions)
    if spec.sprint == "practice" and SPRINT_SESSION not in sessions:
        sessions = (*sessions, SPRINT_SESSION)
    prac, race = [], []
    for s in spec.seasons:
        p = bases.practice.get(s, pd.DataFrame())
        if len(p):
            p = p[p["session"].isin(sessions)]
            if spec.race_sim_only:
                p = p[p["is_race_sim"]]
            if spec.practice_start is not None:
                p = p[p["start_age"].between(*spec.practice_start)]
            if spec.practice_fresh is not None:
                p = p[p["fresh"] == spec.practice_fresh]
        prac.append(p)
        r = bases.race.get(s, pd.DataFrame())
        if len(r) and spec.race_start is not None:
            r = r[r["start_age"].between(*spec.race_start)]
        if len(r) and spec.race_fresh is not None:
            r = r[r["fresh"] == spec.race_fresh]
        race.append(r)
    prac = pd.concat([f for f in prac if len(f)], ignore_index=True) if any(len(f) for f in prac) else pd.DataFrame()
    race = pd.concat([f for f in race if len(f)], ignore_index=True) if any(len(f) for f in race) else pd.DataFrame()
    caps = None
    if spec.common_support:
        # Restrict to events present in BOTH contexts: a cap set by a context with
        # no counterpart is of no use to the contrast.
        caps = _common_support_caps([prac, race], spec)
    if not prepare:
        return _window(race, spec, caps), _window(prac, spec, caps)
    return (
        prepare_context(race, spec, race=True, burn=spec.burn_race, caps=caps),
        prepare_context(prac, spec, race=False, burn=spec.burn_practice, caps=caps),
    )


def cell_fits(frame: pd.DataFrame, min_runs: int) -> pd.DataFrame:
    """One slope per (season, event, compound), with the sums behind it.

    ``sxy`` and ``sxx`` are kept because the slope is their ratio: pooled-lap
    estimates and clustered bootstraps are then just sums of cells, exact, with
    no refitting.
    """
    rows = []
    if frame.empty:
        return pd.DataFrame()
    for (season, event, comp), g in frame.groupby(["season", "event", "Compound"], observed=True):
        if g["run_id"].nunique() < min_runs:
            continue
        got = cluster_slope(g)
        if got is None:
            continue
        slope, se = got
        x, y = g["tl"].to_numpy(float), g["y"].to_numpy(float)
        rows.append(
            {
                "season": int(season),
                "event": event,
                "Compound": comp,
                "slope": slope,
                "se": se,
                "n_runs": int(g["run_id"].nunique()),
                "n_laps": len(g),
                "sxy": float(x @ y),
                "sxx": float(x @ x),
                "temp": float(g["track_temp"].mean()),
                "age": float(g["TyreLife"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_cells(race: pd.DataFrame, prac: pd.DataFrame, min_runs: int, *, need_se: bool = True) -> pd.DataFrame:
    """Join the two contexts' cell fits and add the contrast.

    Column suffixes: ``_r`` is the first context (race), ``_p`` the second
    (practice). ``delta`` is first minus second.
    """
    cr, cp = cell_fits(race, min_runs), cell_fits(prac, min_runs)
    if cr.empty or cp.empty:
        return pd.DataFrame()
    j = cr.merge(cp, on=["season", "event", "Compound"], suffixes=("_r", "_p"))
    j["cluster"] = j["season"].astype(str) + "|" + j["event"]
    j["rank"] = j["Compound"].map(LABEL_RANK).astype(int)
    j["delta"] = j["slope_r"] - j["slope_p"]
    j["var_delta"] = j["se_r"] ** 2 + j["se_p"] ** 2
    # A cell whose SE is undefined (single cluster) was already refused by min_runs,
    # but a degenerate variance would still poison the weights.
    if need_se:
        j = j[np.isfinite(j["var_delta"]) & (j["var_delta"] > 0)]
    else:
        # Pooled-lap estimates use only the sums, so a cell with a single run is fine.
        j["var_delta"] = j["var_delta"].fillna(0.0)
    return j.reset_index(drop=True)


def make_cells(bases: Bases, spec: Spec = PRIMARY, *, need_se: bool = True, **kw) -> pd.DataFrame:
    race, prac = context_frames(bases, spec, **kw)
    return build_cells(race, prac, spec.min_runs, need_se=need_se)


# ---------------------------------------------------------------------------
# Stage two: random-effects contrasts with event-season clusters
# ---------------------------------------------------------------------------


@dataclass
class Arrays:
    """The cell table as arrays, resampleable by cluster."""

    cluster: np.ndarray  # int cluster ids
    rank: np.ndarray
    delta: np.ndarray
    var: np.ndarray
    race: np.ndarray
    prac: np.ndarray
    var_race: np.ndarray
    sxy_r: np.ndarray
    sxx_r: np.ndarray
    sxy_p: np.ndarray
    sxx_p: np.ndarray
    temp_r: np.ndarray
    temp_p: np.ndarray

    @classmethod
    def from_cells(cls, cells: pd.DataFrame) -> Arrays:
        codes, _ = pd.factorize(cells["cluster"])
        f = lambda c: cells[c].to_numpy(float)  # noqa: E731
        return cls(
            cluster=codes.astype(int),
            rank=cells["rank"].to_numpy(int),
            delta=f("delta"),
            var=f("var_delta"),
            race=f("slope_r"),
            prac=f("slope_p"),
            var_race=f("se_r") ** 2,
            sxy_r=f("sxy_r"),
            sxx_r=f("sxx_r"),
            sxy_p=f("sxy_p"),
            sxx_p=f("sxx_p"),
            temp_r=f("temp_r"),
            temp_p=f("temp_p"),
        )

    def __len__(self) -> int:
        return len(self.rank)

    def take(self, idx: np.ndarray, cluster: np.ndarray | None = None) -> Arrays:
        kw = {k: v[idx] for k, v in self.__dict__.items()}
        if cluster is not None:
            kw["cluster"] = cluster
        return Arrays(**kw)

    def members(self) -> list[np.ndarray]:
        order = np.argsort(self.cluster, kind="stable")
        _, start = np.unique(self.cluster[order], return_index=True)
        return np.split(order, start[1:])


def _wls(y, X, w):
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def pm_tau2(y: np.ndarray, X: np.ndarray, v: np.ndarray) -> float:
    """Paule-Mandel between-cell variance: the tau^2 that makes the weighted
    residual sum of squares equal its degrees of freedom.

    Chosen over DerSimonian-Laird because it is less biased with few, unequal
    units, which is our situation. Zero when the cells are no more dispersed than
    their own standard errors say.
    """
    n, p = X.shape
    df = n - np.linalg.matrix_rank(X)
    if df <= 0:
        return 0.0

    def q(t):
        w = 1.0 / (v + t)
        r = y - X @ _wls(y, X, w)
        return float(np.sum(w * r * r))

    if q(0.0) <= df:
        return 0.0
    hi = max(float(np.var(y)), 1e-6)
    for _ in range(60):
        if q(hi) < df:
            break
        hi *= 2
    lo = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if q(mid) > df else (lo, mid)
    return 0.5 * (lo + hi)


def _event_dummies(cluster: np.ndarray) -> np.ndarray:
    _, inv = np.unique(cluster, return_inverse=True)
    return np.eye(inv.max() + 1)[inv]


def trend_fit(
    a: Arrays,
    y: np.ndarray | None = None,
    v: np.ndarray | None = None,
    *,
    tau2: float | None = None,
    extra: np.ndarray | None = None,
    event_fe: bool = True,
    rank: np.ndarray | None = None,
) -> dict:
    """Weighted event-fixed-effects regression of ``y`` on compound softness.

    ``y`` defaults to the race-minus-practice contrast, so ``beta`` is the change
    in the race-minus-practice slope difference per step softer
    (HARD -> MEDIUM -> SOFT): the context x compound interaction. Event fixed
    effects mean it is identified only from compounds raced at the same event.
    Weights are 1 / (cell variance + tau^2). tau^2 is estimated once
    (Paule-Mandel) unless supplied; resampling holds it fixed.
    """
    y = a.delta if y is None else y
    v = a.var if v is None else v
    r = (a.rank if rank is None else rank).astype(float)
    cols = [r[:, None]]
    if extra is not None:
        cols.append(extra.reshape(len(r), -1))
    cols.append(_event_dummies(a.cluster) if event_fe else np.ones((len(r), 1)))
    X = np.hstack(cols)
    if tau2 is None:
        tau2 = pm_tau2(y, X, v)
    w = 1.0 / (v + tau2)
    coef = _wls(y, X, w)
    # Model-based SE (reported for completeness; the bootstrap is the interval).
    xtwx = (X * w[:, None]).T @ X
    se = float(np.sqrt(np.linalg.pinv(xtwx)[0, 0]))
    return {"beta": float(coef[0]), "coef": coef, "se": se, "tau2": float(tau2)}


def _perm_ranks(a: Arrays, rng: np.random.Generator) -> np.ndarray:
    """Shuffle compound labels among the cells of each event."""
    order = np.lexsort((rng.random(len(a)), a.cluster))
    pool = a.rank[np.lexsort((a.rank, a.cluster))]
    out = np.empty_like(a.rank)
    out[order] = pool
    return out


def boot_ci(stat, a: Arrays, B: int, seed: int, level: float = 0.95) -> tuple[float, float, np.ndarray]:
    """Percentile interval from resampling event-seasons with replacement."""
    rng = np.random.default_rng(seed)
    members = a.members()
    k = len(members)
    draws = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, k, size=k)
        idx = np.concatenate([members[i] for i in pick])
        cl = np.repeat(np.arange(k), [len(members[i]) for i in pick])
        try:
            draws[b] = stat(a.take(idx, cl))
        except (np.linalg.LinAlgError, ValueError, ZeroDivisionError, FloatingPointError):
            draws[b] = np.nan
    ok = draws[np.isfinite(draws)]
    lo, hi = np.percentile(ok, [50 * (1 - level), 100 - 50 * (1 - level)]) if len(ok) else (np.nan, np.nan)
    return float(lo), float(hi), draws


def perm_p(stat, a: Arrays, B: int, seed: int, observed: float | None = None):
    """Permute compound labels within event. Returns (p_less, p_two_sided, draws)."""
    rng = np.random.default_rng(seed)
    obs = stat(a) if observed is None else observed
    draws = np.empty(B)
    for b in range(B):
        draws[b] = stat(replace_rank(a, _perm_ranks(a, rng)))
    p_less = (1 + np.sum(draws <= obs + 1e-12)) / (B + 1)
    p_two = (1 + np.sum(np.abs(draws) >= abs(obs) - 1e-12)) / (B + 1)
    return float(p_less), float(p_two), draws


def replace_rank(a: Arrays, rank: np.ndarray) -> Arrays:
    kw = dict(a.__dict__)
    kw["rank"] = rank
    return Arrays(**kw)


def sign_test(a: Arrays, hi: int, lo: int, value: np.ndarray | None = None) -> dict:
    """Within-event paired sign test: is ``value`` smaller for compound ``hi`` than ``lo``?

    ``hi`` and ``lo`` are softness ranks (2 SOFT, 1 MEDIUM, 0 HARD). Events with
    both compounds contribute one +/- each; exact ties are dropped, and reported.
    """
    value = a.delta if value is None else value
    diffs = []
    for idx in a.members():
        r = a.rank[idx]
        if hi in r and lo in r:
            diffs.append(float(value[idx][r == hi][0] - value[idx][r == lo][0]))
    d = np.array(diffs)
    n = len(d)
    neg = int((d < 0).sum())
    ties = int((d == 0).sum())
    m = n - ties
    return {
        "n_events": n,
        "n_lower": neg,
        "n_ties": ties,
        "p_one_sided": float(stats.binom.sf(neg - 1, m, 0.5)) if m else float("nan"),
        "p_two_sided": float(stats.binomtest(neg, m, 0.5).pvalue) if m else float("nan"),
    }


def pair_contrast(a: Arrays, hi: int, lo: int, *, value: np.ndarray | None = None, var: np.ndarray | None = None,
                  tau2: float | None = None) -> dict:
    """Precision-weighted mean, over events with both, of value[hi] - value[lo]."""
    value = a.delta if value is None else value
    var = a.var if var is None else var
    d, vv = [], []
    for idx in a.members():
        r = a.rank[idx]
        if hi in r and lo in r:
            i, j = idx[r == hi][0], idx[r == lo][0]
            d.append(value[i] - value[j])
            vv.append(var[i] + var[j])
    d, vv = np.array(d), np.array(vv)
    if len(d) < 2:
        return {"estimate": float("nan"), "n_events": len(d), "tau2": 0.0}
    X = np.ones((len(d), 1))
    t2 = pm_tau2(d, X, vv) if tau2 is None else tau2
    w = 1.0 / (vv + t2)
    return {"estimate": float(np.sum(w * d) / np.sum(w)), "n_events": int(len(d)), "tau2": float(t2),
            "unweighted_mean": float(d.mean())}


def pair_stat(hi: int, lo: int, tau2: float | None = None, **kw):
    def f(a: Arrays) -> float:
        return pair_contrast(a, hi, lo, tau2=tau2, **kw)["estimate"]
    return f


def level_fit(a: Arrays, rank: int, *, value=None, var=None, tau2=None) -> dict:
    """Random-effects mean of ``value`` over the cells of one compound."""
    value = a.delta if value is None else value
    var = a.var if var is None else var
    m = a.rank == rank
    y, v = value[m], var[m]
    if len(y) < 2:
        return {"estimate": float("nan"), "n": int(m.sum()), "tau2": 0.0}
    X = np.ones((len(y), 1))
    t2 = pm_tau2(y, X, v) if tau2 is None else tau2
    w = 1.0 / (v + t2)
    return {"estimate": float(np.sum(w * y) / np.sum(w)), "n": int(m.sum()), "tau2": float(t2),
            "median": float(np.median(y)), "mean": float(y.mean())}


def ratio_of_means(a: Arrays, rank: int) -> float:
    """sum(race slope) / sum(practice slope) over one compound's cells.

    A ratio of AVERAGES, not an average of ratios: it does not blow up when one
    cell's practice slope is near zero, and it weights cells by how much
    degradation they actually show.
    """
    m = a.rank == rank
    den = a.prac[m].sum()
    return float(a.race[m].sum() / den) if m.any() and den != 0 else float("nan")


def log_ratio_gradient(a: Arrays, hi: int = 2, lo: int = 0) -> float:
    r_hi, r_lo = ratio_of_means(a, hi), ratio_of_means(a, lo)
    if not (r_hi > 0 and r_lo > 0):
        return float("nan")
    return float(np.log(r_hi / r_lo))


def pooled_delta(a: Arrays, rank: int) -> float:
    """Pooled-lap race-minus-practice slope for one compound.

    Each context's slope is sum(x*y) / sum(x*x) over every lap of every cell of
    that compound -- the slope of ONE lap-level regression with a run fixed
    effect and a slope per context x compound, with events pooled (long events
    count more). Exactly equal to refitting on the laps, because the sums are
    additive.
    """
    m = a.rank == rank
    if not m.any():
        return float("nan")
    return float(a.sxy_r[m].sum() / a.sxx_r[m].sum() - a.sxy_p[m].sum() / a.sxx_p[m].sum())


def pooled_did(a: Arrays, hi: int = 2, lo: int = 0) -> float:
    return pooled_delta(a, hi) - pooled_delta(a, lo)


def pooled_trend(a: Arrays) -> float:
    """Slope of pooled per-compound contrasts on softness (least squares over the compounds present)."""
    ranks = [r for r in RANKS if (a.rank == r).any()]
    if len(ranks) < 2:
        return float("nan")
    y = np.array([pooled_delta(a, r) for r in ranks])
    x = np.array(ranks, float)
    return float(((x - x.mean()) @ (y - y.mean())) / ((x - x.mean()) @ (x - x.mean())))


def meta_regression(a: Arrays, *, tau2: float | None = None, event_fe: bool = False) -> dict:
    """race slope ~ practice slope + softness, inverse-variance weighted.

    Second view. ``softness`` coefficient: at the SAME practice slope, how much
    lower is the race slope for a softer compound? Negative means management
    beyond what the practice slope alone predicts.

    Attenuation: the practice slope is noisy, so its coefficient is biased toward
    zero, and because softer compounds truly have larger practice slopes the
    softness term soaks up the under-correction with a POSITIVE bias. A negative
    estimate is therefore conservative; a positive or null one is not evidence of
    no effect.
    """
    cols = [a.rank.astype(float), a.prac]
    X = np.column_stack(cols + ([_event_dummies(a.cluster)] if event_fe else [np.ones(len(a))]))
    y, v = a.race, a.var_race
    t2 = pm_tau2(y, X, v) if tau2 is None else tau2
    coef = _wls(y, X, 1.0 / (v + t2))
    return {"softness": float(coef[0]), "practice": float(coef[1]), "tau2": float(t2)}


# ---------------------------------------------------------------------------
# C-number softness (optional; needs data/allocation_history.json)
# ---------------------------------------------------------------------------


def load_allocation_history(history: Path | None = None, current: Path | None = None) -> dict[tuple[int, str], dict[str, int]]:
    """(season, event) -> {label: C number as int}.

    2022-2025 come from ``data/allocation_history.json`` (Pirelli previews, see
    ``data/allocation_history.py``); 2026 from ``data/allocation.json``. Entries
    whose sources disagree (``verified`` false) are kept but reported by the
    caller. A missing file yields an empty dict, which the caller reports as
    'pending' rather than guessing.
    """
    from ..config import DATA

    history = Path(history) if history else DATA / "allocation_history.json"
    current = Path(current) if current else DATA / "allocation.json"
    out: dict[tuple[int, str], dict[str, int]] = {}
    if history.exists():
        doc = json.loads(history.read_text(encoding="utf-8"))
        for season, events in doc.get("seasons", {}).items():
            for event, row in events.items():
                try:
                    out[(int(season), event)] = {lab: int(str(row[lab]).lstrip("Cc")) for lab in LABELS}
                except (KeyError, ValueError, TypeError):
                    continue
    if current.exists():
        for r in json.loads(current.read_text(encoding="utf-8")):
            try:
                if int(r["season"]) == 2026:
                    out[(2026, r["event"])] = {lab: int(str(c).lstrip("Cc")) for lab, c in r["compounds"].items()}
            except (KeyError, ValueError, TypeError, AttributeError):
                continue
    return out


def unverified_events(history: Path | None = None) -> list[str]:
    from ..config import DATA

    history = Path(history) if history else DATA / "allocation_history.json"
    if not history.exists():
        return []
    doc = json.loads(history.read_text(encoding="utf-8"))
    return sorted(f"{s}|{e}" for s, evs in doc.get("seasons", {}).items() for e, r in evs.items() if not r.get("verified", True))


def cnumber_trend(cells: pd.DataFrame, alloc: dict, B: int, seed: int) -> dict:
    """Does the contrast depend on the PHYSICAL compound (C number), across events?

    Event fixed effects cannot be used: they would remove exactly the
    between-event variation in softness that a C number adds. Instead a
    random-intercept-style weighted regression of the contrast on C number, with
    event-season clusters for the bootstrap.
    """
    c = cells.copy()
    c["C"] = [alloc.get((s, e), {}).get(lab, np.nan) for s, e, lab in zip(c["season"], c["event"], c["Compound"], strict=True)]
    c = c.dropna(subset=["C"]).reset_index(drop=True)
    if len(c) < 10 or c["C"].nunique() < 3:
        return {"status": "insufficient", "n_cells": int(len(c))}
    a = Arrays.from_cells(c)
    cnum = c["C"].to_numpy(float)
    X = np.column_stack([cnum, np.ones(len(c))])
    tau2 = pm_tau2(a.delta, X, a.var)

    def stat(x: Arrays) -> float:
        return float(_wls(x.delta, np.column_stack([x.rank, np.ones(len(x))]), 1 / (x.var + tau2))[0])

    # `rank` is overloaded to carry the C number so the generic resampler applies.
    a2 = replace_rank(a, cnum)
    est = stat(a2)
    lo, hi, _ = boot_ci(stat, a2, B, seed)
    return {"status": "ok", "n_cells": int(len(c)), "beta_per_C": est, "ci95": [lo, hi], "tau2": float(tau2)}


# ---------------------------------------------------------------------------
# Drivers: everything the script writes to paper/results/management.json
# ---------------------------------------------------------------------------


def trend_beta(a: Arrays, tau2: float, **kw) -> float:
    """Lean version of :func:`trend_fit` for resampling loops (no SE)."""
    r = a.rank.astype(float)
    X = np.hstack([r[:, None], _event_dummies(a.cluster)])
    return float(_wls(a.delta, X, 1.0 / (a.var + tau2))[0])


def _ci(stat, a, B, seed):
    lo, hi, draws = boot_ci(stat, a, B, seed)
    return [lo, hi], draws


def headline(cells: pd.DataFrame, *, B: int, P: int, seed: int) -> dict:
    """The full set of estimates for one cell table (the primary analysis)."""
    a = Arrays.from_cells(cells)
    fit = trend_fit(a)
    tau2 = fit["tau2"]
    out: dict = {
        "n_cells": int(len(a)),
        "n_event_seasons": int(len(np.unique(a.cluster))),
        "cells_by_compound": {lab: int((a.rank == LABEL_RANK[lab]).sum()) for lab in LABELS},
        "tau2_trend": tau2,
    }

    # --- primary: interaction trend ---------------------------------------
    stat = lambda x: trend_beta(x, tau2)  # noqa: E731
    ci, draws = _ci(stat, a, B, seed)
    p_less, p_two, _ = perm_p(stat, a, P, seed + 1)
    out["trend"] = {
        "beta_per_softness_step": fit["beta"],
        "model_se": fit["se"],
        "ci95_cluster_bootstrap": ci,
        "boot_sd": float(np.nanstd(draws)),
        "p_perm_one_sided_less": p_less,
        "p_perm_two_sided": p_two,
        "units": "s/lap per softness step (HARD->MEDIUM->SOFT) in the race-minus-practice slope",
    }

    # --- per-compound race-minus-practice level -----------------------------
    out["delta_by_compound"] = {}
    for lab in LABELS:
        k = LABEL_RANK[lab]
        lv = level_fit(a, k)
        t2 = lv["tau2"]
        ci_k, _ = _ci(lambda x, k=k, t2=t2: level_fit(x, k, tau2=t2)["estimate"], a, max(B // 2, 500), seed + 10 + k)
        m = a.rank == k
        out["delta_by_compound"][lab] = {
            "re_mean": lv["estimate"],
            "ci95": ci_k,
            "median_cell": lv.get("median"),
            "n_cells": int(m.sum()),
            "race_slope_median": float(np.median(a.race[m])) if m.any() else None,
            "practice_slope_median": float(np.median(a.prac[m])) if m.any() else None,
            "race_slope_mean": float(a.race[m].mean()) if m.any() else None,
            "practice_slope_mean": float(a.prac[m].mean()) if m.any() else None,
            "share_race_slope_negative": float((a.race[m] < 0).mean()) if m.any() else None,
            "share_practice_slope_negative": float((a.prac[m] < 0).mean()) if m.any() else None,
            "practice_over_race_mean_slope": (
                float(a.prac[m].mean() / a.race[m].mean()) if m.any() and a.race[m].mean() != 0 else None
            ),
        }

    # --- pairwise contrasts + sign tests ------------------------------------
    out["pairwise"] = {}
    for name, hi_, lo_ in (("SOFT-MEDIUM", 2, 1), ("SOFT-HARD", 2, 0), ("MEDIUM-HARD", 1, 0)):
        pc = pair_contrast(a, hi_, lo_)
        st_ = pair_stat(hi_, lo_, tau2=pc["tau2"])
        ci_p, _ = _ci(st_, a, max(B // 2, 500), seed + 20 + hi_ * 3 + lo_)
        pl, p2, _ = perm_p(st_, a, P, seed + 30 + hi_ * 3 + lo_)
        out["pairwise"][name] = {
            "estimate": pc["estimate"],
            "unweighted_mean": pc.get("unweighted_mean"),
            "n_events": pc["n_events"],
            "ci95": ci_p,
            "p_perm_one_sided_less": pl,
            "p_perm_two_sided": p2,
            "sign_test": sign_test(a, hi_, lo_),
        }

    # --- second view: multiplicative (ratio of means) ----------------------
    out["ratio_of_means"] = {}
    for lab in LABELS:
        k = LABEL_RANK[lab]
        ci_k, _ = _ci(lambda x, k=k: ratio_of_means(x, k), a, max(B // 2, 500), seed + 40 + k)
        out["ratio_of_means"][lab] = {"estimate": ratio_of_means(a, k), "ci95": ci_k}
    for name, hi_, lo_ in (("SOFT/HARD", 2, 0), ("SOFT/MEDIUM", 2, 1), ("MEDIUM/HARD", 1, 0)):
        f = lambda x, hi_=hi_, lo_=lo_: log_ratio_gradient(x, hi_, lo_)  # noqa: E731
        ci_g, _ = _ci(f, a, max(B // 2, 500), seed + 50 + hi_ * 3 + lo_)
        pl, p2, _ = perm_p(f, a, P, seed + 60 + hi_ * 3 + lo_)
        out["ratio_of_means"][f"log({name})"] = {
            "estimate": f(a), "ci95": ci_g, "p_perm_one_sided_less": pl, "p_perm_two_sided": p2,
            "note": "negative = softer compound has the lower race/practice ratio (beyond proportional damping)",
        }

    # --- pooled-lap view ------------------------------------------------------
    out["pooled_laps"] = {}
    for lab in LABELS:
        k = LABEL_RANK[lab]
        ci_k, _ = _ci(lambda x, k=k: pooled_delta(x, k), a, max(B // 2, 500), seed + 70 + k)
        out["pooled_laps"][lab] = {"delta": pooled_delta(a, k), "ci95": ci_k}
    for name, f in (("did_SOFT_minus_HARD", pooled_did), ("trend_per_step", pooled_trend)):
        ci_f, _ = _ci(f, a, max(B // 2, 500), seed + 80)
        pl, p2, _ = perm_p(f, a, P, seed + 81)
        out["pooled_laps"][name] = {"estimate": f(a), "ci95": ci_f, "p_perm_one_sided_less": pl, "p_perm_two_sided": p2}

    # --- meta-regression view ------------------------------------------------
    out["meta_regression"] = {}
    for tag, fe in (("no_event_fe", False), ("event_fe", True)):
        mr = meta_regression(a, event_fe=fe)
        t2 = mr["tau2"]
        ci_m, _ = _ci(lambda x, fe=fe, t2=t2: meta_regression(x, tau2=t2, event_fe=fe)["softness"], a,
                      max(B // 2, 500), seed + 90 + int(fe))
        out["meta_regression"][tag] = {
            "softness_coef": mr["softness"], "softness_ci95": ci_m,
            "practice_slope_coef": mr["practice"], "tau2": t2,
        }
    return out


def quick(cells: pd.DataFrame, *, B: int, P: int, seed: int) -> dict:
    """The few numbers a multiverse row needs. Same estimators, fewer replicates."""
    if len(cells) < 8 or cells["cluster"].nunique() < 4 or cells["rank"].nunique() < 2:
        return {"n_cells": int(len(cells)), "status": "insufficient"}
    a = Arrays.from_cells(cells)
    fit = trend_fit(a)
    tau2 = fit["tau2"]
    stat = lambda x: trend_beta(x, tau2)  # noqa: E731
    ci, draws = _ci(stat, a, B, seed)
    pl, p2, _ = perm_p(stat, a, P, seed + 1)
    g = lambda x: log_ratio_gradient(x, 2, 0)  # noqa: E731
    row = {
        "status": "ok",
        "n_cells": int(len(a)),
        "n_event_seasons": int(len(np.unique(a.cluster))),
        "n_informative_event_seasons": int(
            sum(len(np.unique(a.rank[m])) >= 2 for m in a.members())
        ),
        "n_hard": int((a.rank == 0).sum()),
        "n_medium": int((a.rank == 1).sum()),
        "n_soft": int((a.rank == 2).sum()),
        "beta": fit["beta"],
        "ci95": ci,
        "boot_sd": float(np.nanstd(draws)),
        "p_perm_one_sided": pl,
        "p_perm_two_sided": p2,
        "excludes_zero_negative": bool(ci[1] < 0),
        "tau2": tau2,
    }
    if (a.rank == 2).any() and (a.rank == 0).any():
        cg, _ = _ci(g, a, B, seed + 2)
        row["log_ratio_soft_over_hard"] = g(a)
        row["log_ratio_ci95"] = cg
    return row


def random_intercept_trend(cells: pd.DataFrame, *, B: int, seed: int) -> dict:
    """Same contrast, but WITHOUT event fixed effects (cross-event version).

    Uses between-event variation in which compound was run, so it has far more
    information than the fixed-effects trend and is correspondingly easier to
    confound: events that nominate softer tyres differ in other ways too.
    """
    if len(cells) < 8 or cells["cluster"].nunique() < 4 or cells["rank"].nunique() < 2:
        return {"status": "insufficient", "n_cells": int(len(cells))}
    a = Arrays.from_cells(cells)
    fit = trend_fit(a, event_fe=False)
    t2 = fit["tau2"]
    f = lambda x: float(  # noqa: E731
        _wls(x.delta, np.column_stack([x.rank, np.ones(len(x))]), 1.0 / (x.var + t2))[0]
    )
    ci, _ = _ci(f, a, B, seed)
    return {"status": "ok", "beta": fit["beta"], "ci95": ci, "tau2": t2, "n_cells": int(len(a))}


def spearman_legacy(cells: pd.DataFrame) -> dict:
    """The old statistic on the new cells, so the two estimands can be laid side by side."""
    c = cells[cells["slope_p"] > 0.005]
    if len(c) < 6:
        return {"n_cells": int(len(c))}
    ratio = c["slope_r"] / c["slope_p"]
    rho, p = stats.spearmanr(c["rank"], ratio)
    return {"n_cells": int(len(c)), "rho": float(rho), "p_two_sided": float(p)}


# ---------------------------------------------------------------------------
# Season structure
# ---------------------------------------------------------------------------


def season_analysis(cells: pd.DataFrame, *, B: int, P: int, seed: int) -> dict:
    """Each season alone, leave-one-season-out, and a random-effects pool of the seasons.

    The pool treats the season-specific trend estimates as the data and asks
    whether they agree. It is the season-blocked test the legacy module left
    undone: it cannot be carried by one season, and heterogeneity is measured
    instead of assumed away.
    """
    seasons = sorted(cells["season"].unique())
    alone, loo = {}, {}
    for k, s in enumerate(seasons):
        alone[str(s)] = quick(cells[cells["season"] == s], B=B, P=P, seed=seed + 100 * k)
        loo[str(s)] = quick(cells[cells["season"] != s], B=B, P=P, seed=seed + 100 * k + 50)
    ok = [
        (alone[str(s)]["beta"], alone[str(s)]["boot_sd"])
        for s in seasons
        if alone[str(s)].get("status") == "ok" and alone[str(s)]["boot_sd"] > 0
    ]
    pool: dict = {"n_seasons_estimable": len(ok)}
    if len(ok) >= 2:
        b = np.array([x[0] for x in ok])
        v = np.array([x[1] for x in ok]) ** 2
        w = 1 / v
        fixed = np.sum(w * b) / np.sum(w)
        q = float(np.sum(w * (b - fixed) ** 2))
        df = len(b) - 1
        tau2 = pm_tau2(b, np.ones((len(b), 1)), v)
        wr = 1 / (v + tau2)
        re = float(np.sum(wr * b) / np.sum(wr))
        se = float(np.sqrt(1 / np.sum(wr)))
        n_neg = int((b < 0).sum())
        pool.update(
            {
                "random_effects_beta": re,
                "random_effects_ci95_normal": [re - 1.96 * se, re + 1.96 * se],
                "tau2_between_seasons": tau2,
                "Q": q,
                "Q_df": df,
                "Q_p": float(stats.chi2.sf(q, df)),
                "I2": float(max(0.0, (q - df) / q)) if q > 0 else 0.0,
                "n_seasons_negative": n_neg,
                "sign_test_p_one_sided": float(stats.binom.sf(n_neg - 1, len(b), 0.5)),
                "note": "interval uses the normal approximation on bootstrap SDs; with <=5 seasons it is rough",
            }
        )
    return {"each_season_alone": alone, "leave_one_season_out": loo, "season_pool": pool}


# ---------------------------------------------------------------------------
# Placebo: practice against practice
# ---------------------------------------------------------------------------


def placebo_cells(
    bases: Bases,
    spec: Spec = PRIMARY,
    a: tuple[str, ...] = ("FP2",),
    b: tuple[str, ...] = ("FP1", "FP3"),
    need_se: bool = True,
) -> pd.DataFrame:
    """Cells contrasting two practice contexts at the same event.

    Nothing about racing differs between FP2 and FP1+FP3, so if the compound
    trend in the race-minus-practice contrast comes from racing it should be
    absent here. ``delta`` is context ``a`` minus context ``b``.
    """
    fa = context_frames(bases, spec, practice_sessions=a)[1]
    fb = context_frames(bases, spec, practice_sessions=b)[1]
    return build_cells(fa, fb, spec.min_runs, need_se=need_se)


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


def _trend_with(a: Arrays, tau2: float, mode: str, event_fe: bool) -> float:
    r = a.rank.astype(float)
    cols = [r[:, None]]
    if mode == "diff":
        cols.append((a.temp_r - a.temp_p)[:, None])
    elif mode == "both":
        cols.append(np.column_stack([a.temp_r, a.temp_p]))
    cols.append(_event_dummies(a.cluster) if event_fe else np.ones((len(r), 1)))
    return float(_wls(a.delta, np.hstack(cols), 1.0 / (a.var + tau2))[0])


def temperature_models(cells: pd.DataFrame, *, B: int, seed: int) -> dict:
    """Does the compound trend survive controlling for track temperature?

    Cell-level mean track temperature (interpolated to each lap's start) enters as
    the race-minus-practice temperature difference, or as the two temperatures
    separately. Event fixed effects identify temperature only from variation
    BETWEEN compounds' sessions within an event, which is small, so the
    random-intercept version (no event dummies) is shown too: it uses the large
    between-event temperature variation but then also lets between-event
    differences in softness leak in. Neither is clean; agreement is the point.
    """
    c = cells.dropna(subset=["temp_r", "temp_p"]).reset_index(drop=True)
    a = Arrays.from_cells(c)
    dT = a.temp_r - a.temp_p
    within = pd.Series(dT).groupby(a.cluster).transform(lambda x: x - x.mean())
    out: dict = {
        "n_cells": int(len(c)),
        "mean_temp_race_C": float(a.temp_r.mean()),
        "mean_temp_practice_C": float(a.temp_p.mean()),
        "mean_race_minus_practice_C": float(dT.mean()),
        "sd_race_minus_practice_C": float(dT.std()),
        "sd_within_event_race_minus_practice_C": float(within.std()),
        "corr_delta_vs_tempdiff": float(np.corrcoef(a.delta, dT)[0, 1]),
        "models": {},
    }
    for name, mode, fe in (
        ("event_fe_no_temp", "none", True),
        ("event_fe_temp_diff", "diff", True),
        ("event_fe_temp_both", "both", True),
        ("random_intercept_no_temp", "none", False),
        ("random_intercept_temp_diff", "diff", False),
        ("random_intercept_temp_both", "both", False),
    ):
        r = a.rank.astype(float)
        cols = [r[:, None]]
        if mode == "diff":
            cols.append(dT[:, None])
        elif mode == "both":
            cols.append(np.column_stack([a.temp_r, a.temp_p]))
        cols.append(_event_dummies(a.cluster) if fe else np.ones((len(r), 1)))
        X = np.hstack(cols)
        tau2 = pm_tau2(a.delta, X, a.var)
        coef = _wls(a.delta, X, 1.0 / (a.var + tau2))
        ci, _ = _ci(lambda x, mode=mode, fe=fe, t2=tau2: _trend_with(x, t2, mode, fe), a, B, seed)
        entry = {"beta": float(coef[0]), "ci95": ci, "tau2": float(tau2)}
        if mode == "diff":
            entry["temp_diff_coef_per_C"] = float(coef[1])
        if mode == "both":
            entry["temp_race_coef_per_C"] = float(coef[1])
            entry["temp_practice_coef_per_C"] = float(coef[2])
        out["models"][name] = entry
    # Matched conditions: race and practice within 5 C of each other.
    m = np.abs(dT) <= 5.0
    out["matched_within_5C"] = {"n_cells": int(m.sum())}
    if m.sum() >= 12 and len(np.unique(a.cluster[m])) >= 6:
        out["matched_within_5C"] = {**out["matched_within_5C"], **quick(c[m], B=B, P=2000, seed=seed + 5)}
    return out


# ---------------------------------------------------------------------------
# Fuel and multiverse grids
# ---------------------------------------------------------------------------

FUEL_BURN_GRID = (1.4, 1.7, 2.0)
MASS_GRID = (0.025, 0.033, 0.040)


def _levels(cells: pd.DataFrame) -> dict:
    a = Arrays.from_cells(cells)
    present = [lab for lab in LABELS if (a.rank == LABEL_RANK[lab]).any()]
    return {
        "delta_re_mean": {lab: level_fit(a, LABEL_RANK[lab])["estimate"] for lab in present},
        "ratio_of_means": {lab: ratio_of_means(a, LABEL_RANK[lab]) for lab in present},
    }


def fuel_grid(bases: Bases, spec: Spec = PRIMARY, *, B: int, P: int, seed: int) -> list[dict]:
    """Burn x mass-sensitivity grid, applied identically to both contexts, plus asymmetric rows."""
    rows = []
    combos = [(b, s, b, "both contexts") for b in FUEL_BURN_GRID for s in MASS_GRID]
    # A race burns differently from a practice sim; vary one side only.
    combos += [(br, MASS_SENSITIVITY_S_PER_KG, PRACTICE_BURN_KG_PER_LAP, "race burn only") for br in (1.4, 2.0)]
    combos += [(PRACTICE_BURN_KG_PER_LAP, MASS_SENSITIVITY_S_PER_KG, bp, "practice burn only") for bp in (1.4, 2.0)]
    for k, (br, s, bp, tag) in enumerate(combos):
        sp = replace(spec, burn_race=br, burn_practice=bp, s_per_kg=s)
        cells = make_cells(bases, sp)
        row = {
            "burn_race": br,
            "burn_practice": bp,
            "s_per_kg": s,
            "variant": tag,
            "correction_race_s_per_lap": br * s,
            "correction_practice_s_per_lap": bp * s,
        }
        row.update(quick(cells, B=B, P=P, seed=seed + k))
        row.update(_levels(cells))
        rows.append(row)
    return rows


def named_multiverse(bases: Bases, *, B: int, P: int, seed: int) -> list[dict]:
    """One-factor-at-a-time and factorial rows around the primary specification."""
    specs: list[tuple[str, Spec]] = [("PRIMARY", PRIMARY)]

    windows = {
        "window none": dict(age_lo=None, age_hi=None),
        "window 3-12": dict(age_lo=3, age_hi=12),
        "window 4-14": dict(age_lo=4, age_hi=14),
        "window 6-16": dict(age_lo=6, age_hi=16),
        "window common-support": dict(age_lo=6, age_hi=None, common_support=True),
    }
    lens = {"runlen any": None, "runlen 7-15": (7, 15)}
    for wn, wkw in windows.items():
        for ln, lv in lens.items():
            for mr in (2, 3, 4):
                for est in ("within_run", "xs"):
                    nm = f"{wn} | {ln} | min_runs {mr} | race={est}"
                    specs.append((nm, replace(PRIMARY, run_len=lv, min_runs=mr, race_estimator=est, **wkw)))

    specs.append(("sprint pooled into practice", replace(PRIMARY, sprint="practice")))
    specs.append(("race-sim filter off", replace(PRIMARY, race_sim_only=False)))
    specs.append(("min window laps 6", replace(PRIMARY, min_window_laps=6)))
    for drop in ALL_SEASONS:
        specs.append((f"seasons: drop {drop}", replace(PRIMARY, seasons=tuple(s for s in ALL_SEASONS if s != drop))))
    for only in ALL_SEASONS:
        specs.append((f"seasons: only {only}", replace(PRIMARY, seasons=(only,))))
    specs.append(("seasons: 2022-2025 (no 2026)", replace(PRIMARY, seasons=(2022, 2023, 2024, 2025))))
    specs.append(("seasons: 2024-2026", replace(PRIMARY, seasons=(2024, 2025, 2026))))

    rows = []
    seen = set()
    for k, (name, sp) in enumerate(specs):
        key = sp.label()
        if key in seen:
            continue  # the primary reappears inside the factorial; list it once
        seen.add(key)
        cells = make_cells(bases, sp)
        row = {"name": name, "spec": json.loads(sp.label())}
        if len(cells) == 0:
            row.update({"status": "no cells", "n_cells": 0})
        else:
            row.update(quick(cells, B=B, P=P, seed=seed + k))
            row["legacy_spearman_on_these_cells"] = spearman_legacy(cells)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# POST-REVIEW ADDITIONS
#
# Everything below was added AFTER an independent review of the first run, which
# reproduced the primary numbers and raised four objections: (1) race stints and
# practice runs start on different set states; (2) the gradient test has little
# power, so "not supported" is not a null; (3) the pre-stated placebo was
# vacuous; (4) the cross-sectional race estimator is not identified. These
# specifications were therefore NOT pre-registered in the original docstring and
# are labelled as post-review in the results file. They were fixed in advance of
# being run (below) and not tuned on their output.
#
# PRE-STATED (before running): matched level gap with practice start age 5-9 and
# race start age >= 5, window 6-16, >= 3 runs per cell and context, pooled over
# MEDIUM + SOFT, event-season cluster bootstrap B = 5000. DEVIATION, disclosed: at
# >= 3 runs per cell that spec leaves 25 cells (1 HARD), so the headline level gap
# is the POOLED-lap version at >= 1 run per cell (sums of laps, no per-cell
# variance needed), matching the reviewer's pooled design; the strict version is
# reported beside it; TWFE gradient with
# 10-minute practice time bins, >= 2 runs per compound and context, SOFT - HARD
# as the gradient estimand, MEDIUM - HARD and SOFT - MEDIUM secondary; level
# placebo FP2 minus FP1+FP3 with the same matching; power by simulation under
# the observed cell structure, 80% MDE interpolated.
# ---------------------------------------------------------------------------

#: DEFINITION of the matched design. ``start_age`` of a run is the TyreLife of its
#: FIRST RETAINED lap: pit-in and pit-out laps, non-green laps, inaccurate laps, laps
#: slower than 1.10 x the driver's best, and within-stint outliers are already gone,
#: so a new-set stint normally starts at age 2 (the out-lap is lap 1). Runs are the
#: consecutive-lap blocks of a stint, so a safety-car or yellow gap SPLITS a stint
#: and the later block starts at a high age on a set that was fitted new. "Race stints
#: at age >= 5" therefore mixes (a) stints that genuinely began on a used set and
#: (b) later blocks of new-set stints. ``set_state_counts`` reports the split, and the
#: ``*_USED`` specs below use the feed's FreshTyre flag to match set state directly.
MATCHED = replace(PRIMARY, practice_start=(5, 9), race_start=(5, 10**6))
MATCHED_POOLED = replace(MATCHED, min_runs=1)
#: Set state matched through FreshTyre rather than through start age.
USED_SETS = replace(PRIMARY, min_runs=1, practice_fresh=False, race_fresh=False)
USED_SETS_AGE_MATCHED = replace(USED_SETS, practice_start=(5, 9), race_start=(5, 10**6))
NEW_SETS = replace(PRIMARY, min_runs=1, practice_fresh=True, race_fresh=True)


def pooled_level(a: Arrays, ranks: tuple[int, ...]) -> dict:
    """Pooled race slope, practice slope and their gap over the cells of ``ranks``.

    Each context's slope is sum(x*y) / sum(x*x) over every lap of every selected
    cell (exact, because the sums are additive), so events with more running
    count more, exactly as they would in one lap-level regression with a run
    effect.
    """
    m = np.isin(a.rank, ranks)
    if not m.any():
        return {"race": float("nan"), "practice": float("nan"), "gap": float("nan"), "ratio": float("nan"), "n": 0}
    r = a.sxy_r[m].sum() / a.sxx_r[m].sum()
    p = a.sxy_p[m].sum() / a.sxx_p[m].sum()
    return {"race": float(r), "practice": float(p), "gap": float(r - p), "ratio": float(r / p) if p else float("nan"),
            "n": int(m.sum())}


def level_gaps(cells: pd.DataFrame, *, B: int, seed: int) -> dict:
    """Race-minus-practice LEVEL gap per compound and pooled, with cluster-bootstrap CIs."""
    if len(cells) < 6:
        return {"status": "insufficient", "n_cells": int(len(cells))}
    a = Arrays.from_cells(cells)
    out: dict = {"status": "ok", "n_cells": int(len(a)), "n_event_seasons": int(len(np.unique(a.cluster)))}
    groups = {lab: (LABEL_RANK[lab],) for lab in LABELS}
    groups["MEDIUM+SOFT"] = (1, 2)
    groups["ALL"] = (0, 1, 2)
    for name, ranks in groups.items():
        if not np.isin(a.rank, ranks).any():
            continue
        po = pooled_level(a, ranks)
        ci_gap, _ = _ci(lambda x, r=ranks: pooled_level(x, r)["gap"], a, B, seed + len(name))
        ci_ratio, _ = _ci(lambda x, r=ranks: pooled_level(x, r)["ratio"], a, B, seed + 7 + len(name))
        out[name] = {**po, "gap_ci95": ci_gap, "ratio_ci95": ci_ratio}
    # A uniform level gap means: no softness gradient in the gap itself.
    if (a.rank == 0).any() and (a.rank == 2).any():
        f = lambda x: pooled_level(x, (2,))["gap"] - pooled_level(x, (0,))["gap"]  # noqa: E731
        ci, _ = _ci(f, a, B, seed + 99)
        out["gap_SOFT_minus_HARD"] = {"estimate": f(a), "ci95": ci}
    if (a.rank == 2).any() and (a.rank == 1).any():
        f = lambda x: pooled_level(x, (2,))["gap"] - pooled_level(x, (1,))["gap"]  # noqa: E731
        ci, _ = _ci(f, a, B, seed + 98)
        out["gap_SOFT_minus_MEDIUM"] = {"estimate": f(a), "ci95": ci}
    return out


def placebo_levels(bases: Bases, spec: Spec, *, B: int, seed: int) -> dict:
    """LEVEL placebo: FP2 minus FP1+FP3 on race-sim runs, same matching and window.

    The practice-versus-practice analogue of :func:`level_gaps`. A session
    artefact (run order, fuel load, who runs a compound when) would show up as a
    non-zero gap here with nothing to do with racing.
    """
    cells = placebo_cells(bases, spec, need_se=False)
    out = level_gaps(cells, B=B, seed=seed)
    out["definition"] = "FP2 slope minus (FP1+FP3) slope; first context is FP2"
    return out


def uniform_damping_beta(cells: pd.DataFrame, k: float) -> float:
    """The additive interaction a perfectly UNIFORM damping factor k would produce.

    If every compound's race slope were exactly k times its practice slope, the
    race-minus-practice difference would be (k - 1) x practice slope, which grows
    with softness because soft tyres have larger practice slopes. This is the
    benchmark the additive trend has to be compared with.
    """
    a = Arrays.from_cells(cells)
    return float(trend_fit(a, y=(k - 1.0) * a.prac, tau2=trend_fit(a)["tau2"])["beta"])


# --- two-way fixed-effects compound-contrast DiD ----------------------------


def twfe_residuals(keys_a, keys_b, *cols, iters: int = 400, tol: float = 1e-9) -> list[np.ndarray]:
    """Residualise columns on two fixed effects by alternating projections."""
    ia, ib = pd.factorize(np.asarray(keys_a))[0], pd.factorize(np.asarray(keys_b))[0]
    na, nb = ia.max() + 1, ib.max() + 1
    ca, cb = np.bincount(ia, minlength=na), np.bincount(ib, minlength=nb)
    out = []
    for col in cols:
        r = np.asarray(col, float).copy()
        for _ in range(iters):
            before = r.copy()
            r -= (np.bincount(ia, weights=r, minlength=na) / ca)[ia]
            r -= (np.bincount(ib, weights=r, minlength=nb) / cb)[ib]
            if np.max(np.abs(r - before)) < tol:
                break
        out.append(r)
    return out


PRACTICE_BIN_S = 600.0


def pair_age_contrast(d: pd.DataFrame, a: str, b: str, min_runs: int) -> tuple[float, float] | None:
    """Slope of tyre age on lap time for compound ``a`` MINUS that for ``b``, one event, one context.

    Lap time is regressed on a run effect, a second effect (column ``fe_key``: the
    lap in a race, a ten-minute session bin in practice) and tyre age with a
    compound-specific slope. Tyre age is lap minus stint start, so in a race the
    lap and run effects absorb its COMMON component exactly: only the difference
    between compounds is identified. Whatever is common to the compounds on
    track -- fuel burn, track evolution, temperature, rubbering-in -- cancels
    without being modelled. In practice the bins are coarser than a lap, so a
    common age term is estimated alongside.

    Returns (contrast, run-clustered standard error) or ``None`` if not identified.
    """
    d = d[d["Compound"].isin([a, b])]
    if d["Compound"].nunique() < 2 or d.groupby("Compound")["run_id"].nunique().min() < min_runs:
        return None
    age = d["TyreLife"].to_numpy(float)
    is_a = (d["Compound"].to_numpy() == a).astype(float)
    y_r, age_r, x_r = twfe_residuals(
        d["fe_key"].to_numpy(), d["run_id"].to_numpy(), d["LapTimeSeconds"].to_numpy(float), age, age * is_a
    )
    X = np.column_stack([age_r, x_r])
    keep = [j for j in range(2) if X[:, j] @ X[:, j] > 1e-8]
    if 1 not in keep:
        return None
    X = X[:, keep]
    j = keep.index(1)
    xtx = X.T @ X
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return None
    coef = inv @ (X.T @ y_r)
    e = y_r - X @ coef
    runs = d["run_id"].to_numpy()
    uniq = np.unique(runs)
    if len(uniq) < 3:
        return None
    meat = np.zeros((X.shape[1], X.shape[1]))
    for u in uniq:
        m = runs == u
        sc = X[m].T @ e[m]
        meat += np.outer(sc, sc)
    g = len(uniq)
    cov = inv @ (meat * g / (g - 1)) @ inv
    se = float(np.sqrt(max(cov[j, j], 0.0)))
    return float(coef[j]), se


def twfe_pair_table(bases: Bases, spec: Spec = PRIMARY, *, min_runs: int = 2) -> pd.DataFrame:
    """Per event-season and compound pair: race contrast, practice contrast, and their difference.

    The difference is a difference-in-differences: how much more (or less) the
    softer compound's age slope exceeds the harder one's in the race than in
    practice. Negative = the compound gradient is flatter in the race.
    """
    race, prac = context_frames(bases, spec, prepare=False)
    if race.empty or prac.empty:
        return pd.DataFrame()
    race = race.reset_index(drop=True)
    prac = prac.reset_index(drop=True)
    race["fe_key"] = race["LapNumber"].astype(int).astype(str)
    prac["fe_key"] = prac["session"].astype(str) + "|" + (prac["SessionElapsed"] // PRACTICE_BIN_S).astype(int).astype(str)
    rows = []
    pairs = (("SOFT", "HARD"), ("MEDIUM", "HARD"), ("SOFT", "MEDIUM"))
    r_groups = dict(tuple(race.groupby(["season", "event"])))
    for key, gp in prac.groupby(["season", "event"]):
        gr = r_groups.get(key)
        if gr is None:
            continue
        for a, b in pairs:
            cr = pair_age_contrast(gr, a, b, min_runs)
            cp = pair_age_contrast(gp, a, b, min_runs)
            if cr is None or cp is None or not (np.isfinite(cr[1]) and np.isfinite(cp[1])) or cr[1] <= 0 or cp[1] <= 0:
                continue
            rows.append({"season": int(key[0]), "event": key[1], "cluster": f"{key[0]}|{key[1]}", "pair": f"{a}-{b}",
                         "race": cr[0], "se_race": cr[1], "practice": cp[0], "se_practice": cp[1],
                         "did": cr[0] - cp[0], "var": cr[1] ** 2 + cp[1] ** 2})
    return pd.DataFrame(rows)


def _re_mean(d: np.ndarray, v: np.ndarray, tau2: float | None = None) -> tuple[float, float]:
    t2 = pm_tau2(d, np.ones((len(d), 1)), v) if tau2 is None else tau2
    w = 1.0 / (v + t2)
    return float(np.sum(w * d) / np.sum(w)), float(t2)


def twfe_gradient(table: pd.DataFrame, *, B: int, P: int, seed: int) -> dict:
    """Aggregate the per-event DiDs by pair: RE mean, cluster-bootstrap CI, sign-flip permutation p.

    Permuting the compound labels within an event swaps which compound is 'a' and
    which is 'b', which flips the sign of that event's contrast in BOTH contexts,
    so the test is a sign-flip test on the per-event DiD.
    """
    out: dict = {"status": "ok" if len(table) else "empty"}
    rng_master = np.random.default_rng(seed)
    for pair in ("SOFT-HARD", "MEDIUM-HARD", "SOFT-MEDIUM"):
        t = table[table["pair"] == pair]
        if len(t) < 4:
            out[pair] = {"n_events": int(len(t)), "status": "insufficient"}
            continue
        d, v = t["did"].to_numpy(), t["var"].to_numpy()
        est, tau2 = _re_mean(d, v)
        rng = np.random.default_rng(int(rng_master.integers(1 << 31)))
        boots = np.empty(B)
        for b in range(B):
            i = rng.integers(0, len(d), len(d))
            boots[b] = _re_mean(d[i], v[i], tau2)[0]
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        perm = np.empty(P)
        for b in range(P):
            sgn = rng.choice([-1.0, 1.0], len(d))
            perm[b] = _re_mean(d * sgn, v, tau2)[0]
        neg = int((d < 0).sum())
        out[pair] = {
            "n_events": int(len(d)),
            "estimate_s_per_lap": est,
            "ci95": ci,
            "boot_sd": float(boots.std()),
            "mde80_s_per_lap": float(2.8 * boots.std()),
            "p_perm_one_sided_less": float((1 + np.sum(perm <= est + 1e-12)) / (P + 1)),
            "p_perm_two_sided": float((1 + np.sum(np.abs(perm) >= abs(est) - 1e-12)) / (P + 1)),
            "tau2": tau2,
            "unweighted_mean": float(d.mean()),
            "n_negative": neg,
            "sign_test_p_one_sided": float(stats.binom.sf(neg - 1, len(d), 0.5)),
            "race_contrast_mean": float(t["race"].mean()),
            "practice_contrast_mean": float(t["practice"].mean()),
        }
    return out


# --- power ------------------------------------------------------------------


def power_simulation(cells: pd.DataFrame, betas, *, n_sims: int, B: int, P: int, seed: int) -> dict:
    """Power of the pre-stated decision rule for the additive interaction.

    Keeps the OBSERVED design (clusters, ranks, cell variances, tau^2) and plants
    a true interaction ``beta``: delta = beta x rank + N(0, v + tau^2). Event
    effects are omitted because the fixed effects remove them. Each replicate
    re-estimates tau^2, bootstraps clusters, and permutes labels exactly as the
    real test does (with fewer draws), then applies the rule: CI upper bound < 0
    AND one- and two-sided permutation p < 0.05. Also reports the weaker
    CI-only rule. ``mde80`` is the smallest |beta| on the grid with >= 80% power,
    linearly interpolated.
    """
    a0 = Arrays.from_cells(cells)
    tau2_obs = trend_fit(a0)["tau2"]
    rng = np.random.default_rng(seed)
    rows = []
    for beta in betas:
        full = ci_only = 0
        for _ in range(n_sims):
            delta = beta * a0.rank + rng.normal(0.0, np.sqrt(a0.var + tau2_obs))
            a = Arrays(**{**a0.__dict__, "delta": delta})
            fit = trend_fit(a)
            tau2 = fit["tau2"]
            stat = lambda x, t=tau2: trend_beta(x, t)  # noqa: E731
            lo, hi, _ = boot_ci(stat, a, B, int(rng.integers(1 << 31)))
            pl, p2, _ = perm_p(stat, a, P, int(rng.integers(1 << 31)), observed=fit["beta"])
            ci_only += bool(hi < 0)
            full += bool(hi < 0 and pl < 0.05 and p2 < 0.05)
        rows.append({"beta": float(beta), "power_full_rule": full / n_sims, "power_ci_only": ci_only / n_sims})
    out: dict = {"n_sims": n_sims, "B": B, "P": P, "grid": rows, "tau2_used": tau2_obs}
    for key in ("power_full_rule", "power_ci_only"):
        pts = sorted((abs(r["beta"]), r[key]) for r in rows)
        mde = None
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=True):
            if y0 < 0.8 <= y1:
                mde = x0 + (0.8 - y0) * (x1 - x0) / (y1 - y0)
                break
        out[f"mde80_{key}"] = mde
    return out


# ---------------------------------------------------------------------------
# REFEREE ADDITIONS (after the paper was drafted)
#
# (1) what "start age >= 5" means and how many race runs are really on used sets,
#     plus designs that match set state through the feed's FreshTyre flag;
# (2) a scripted reconciliation of the race-only compound trend across estimators,
#     seasons and windows;
# (3) the placebo-adjusted gap with a joint bootstrap;
# (4) Holm adjustment over the three TWFE pair tests;
# (5) the lead estimand (gap, then ratio) under explicit keys.
# ---------------------------------------------------------------------------


def holm_adjust(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (family-wise error), monotone and capped at 1."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def _run_table(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["run_id", "Compound", "fresh", "start_age", "stint_start_age", "run_len"]
    return frame.drop_duplicates("run_id")[cols]


def _breakdown(runs: pd.DataFrame, lo: int, hi: int | None = None) -> dict:
    sel = runs[runs["start_age"] >= lo] if hi is None else runs[runs["start_age"].between(lo, hi)]
    used = sel["fresh"] == False  # noqa: E712
    restart = (sel["fresh"] == True) & (sel["start_age"] > sel["stint_start_age"])  # noqa: E712
    return {
        "n_runs": int(len(sel)),
        "genuinely_used_set_FreshTyre_false": int(used.sum()),
        "new_set_but_run_starts_after_stint_start_gap_split": int(restart.sum()),
        "new_set_stint_that_itself_starts_this_late": int(len(sel) - used.sum() - restart.sum()),
    }


def set_state_counts(bases: Bases, spec: Spec = PRIMARY) -> dict:
    """How many runs start on used versus new sets, and what start age really proxies.

    ``start_age`` is the TyreLife of the run's first RETAINED lap. A run is a block of
    consecutive laps inside a stint, so a safety-car or yellow-flag gap splits one
    stint into several runs and the later blocks begin at a high age on a set that was
    fitted new. 'Race runs with start age >= 5' is therefore a mixture of used-set
    stints and later blocks of new-set stints; this reports the mixture.
    """
    race, prac = context_frames(bases, replace(spec, age_lo=None, age_hi=None), prepare=False)
    out: dict = {"definition": (
        "start_age = TyreLife of the first retained lap of the run (pit laps, non-green laps, laps > 1.10x the "
        "driver's best and stint outliers removed); runs are consecutive-lap blocks, so a gap splits a stint. "
        "fresh = the feed's FreshTyre flag for the stint (one value per stint). stint_start_age = TyreLife at the "
        "stint's first timed lap."
    )}
    for name, fr, lo_hi in (("race", race, (5, None)), ("practice_race_sims", prac, (5, 9))):
        runs = _run_table(fr)
        out[name] = {
            "n_runs": int(len(runs)),
            "n_new_set_runs": int(runs["fresh"].sum()),
            "n_used_set_runs": int((~runs["fresh"]).sum()),
            "share_used_set": float((~runs["fresh"]).mean()),
            "median_start_age": float(runs["start_age"].median()),
            "share_start_age_ge_5": float((runs["start_age"] >= 5).mean()),
            "start_age_ge_5" if name == "race" else "start_age_5_to_9": _breakdown(runs, *lo_hi),
            "used_set_start_age_quartiles": [float(x) for x in runs.loc[~runs["fresh"], "start_age"].quantile([0.25, 0.5, 0.75])],
        }
    # What each matched design actually leaves, after the tyre-age window.
    designs = {}
    for label, sp in (
        ("matched_start_age_5_9_vs_ge5", MATCHED_POOLED),
        ("used_sets_only_FreshTyre_false", USED_SETS),
        ("used_sets_and_start_age_matched", USED_SETS_AGE_MATCHED),
        ("new_sets_only_FreshTyre_true", NEW_SETS),
    ):
        r, p = context_frames(bases, sp, prepare=False)
        cells = make_cells(bases, sp, need_se=False)
        designs[label] = {
            "race_runs_in_window": int(r["run_id"].nunique()) if len(r) else 0,
            "practice_runs_in_window": int(p["run_id"].nunique()) if len(p) else 0,
            "cells_with_both_contexts": int(len(cells)),
            "cells_by_compound": {lab: int((cells["Compound"] == lab).sum()) for lab in LABELS} if len(cells) else {},
            "event_seasons": int(cells["cluster"].nunique()) if len(cells) else 0,
        }
    out["designs"] = designs
    return out


# --- placebo-adjusted gap ---------------------------------------------------


def _cluster_sums(cells: pd.DataFrame, ranks: tuple[int, ...]) -> pd.DataFrame:
    c = cells[cells["rank"].isin(ranks)]
    return c.groupby("cluster")[["sxy_r", "sxx_r", "sxy_p", "sxx_p"]].sum()


def placebo_adjusted_gap(main: pd.DataFrame, plc: pd.DataFrame, ranks: tuple[int, ...], *, B: int, seed: int) -> dict:
    """(race - practice) pooled gap MINUS (FP2 - FP1/FP3) pooled gap, with a JOINT bootstrap.

    Event-seasons are resampled as units across both tables at once, so an event that
    appears in both contributes to both terms of every replicate (its two terms are
    positively correlated, which an independent bootstrap would ignore). The placebo
    gap has no natural orientation -- 'FP2 minus the rest' is a choice -- so the
    subtraction is a robustness row, not a correction of record.
    """
    a, b = _cluster_sums(main, ranks), _cluster_sums(plc, ranks)
    ids = sorted(set(a.index) | set(b.index))
    A = a.reindex(ids).fillna(0.0).to_numpy()
    Bm = b.reindex(ids).fillna(0.0).to_numpy()

    def gaps(idx):
        sa, sb = A[idx].sum(0), Bm[idx].sum(0)
        g_main = sa[0] / sa[1] - sa[2] / sa[3] if sa[1] and sa[3] else np.nan
        g_plc = sb[0] / sb[1] - sb[2] / sb[3] if sb[1] and sb[3] else np.nan
        return g_main, g_plc

    full = np.arange(len(ids))
    g_main, g_plc = gaps(full)
    rng = np.random.default_rng(seed)
    d = np.empty((B, 3))
    for i in range(B):
        gm, gp = gaps(rng.integers(0, len(ids), len(ids)))
        d[i] = (gm, gp, gm - gp)
    d = d[np.isfinite(d).all(1)]
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {
        "race_minus_practice_gap": float(g_main),
        "race_minus_practice_gap_ci95": [float(lo[0]), float(hi[0])],
        "placebo_gap_fp2_minus_fp1fp3": float(g_plc),
        "placebo_gap_ci95": [float(lo[1]), float(hi[1])],
        "placebo_adjusted_gap": float(g_main - g_plc),
        "placebo_adjusted_gap_ci95_joint_bootstrap": [float(lo[2]), float(hi[2])],
        "placebo_share_of_gap": float(g_plc / g_main) if g_main else None,
        "n_event_seasons_main": int(len(a)),
        "n_event_seasons_placebo": int(len(b)),
        "n_event_seasons_in_both": int(len(set(a.index) & set(b.index))),
    }


# --- reconciliation of the race-only compound trend ---------------------------


def race_only_trend(fits: pd.DataFrame, *, B: int, seed: int) -> dict:
    """Event-fixed-effects trend of the RACE slope on softness (s/lap per step softer)."""
    f = fits[np.isfinite(fits["se"]) & (fits["se"] > 0)].copy()
    f["cluster"] = f["season"].astype(str) + "|" + f["event"]
    f["rank"] = f["Compound"].map(LABEL_RANK)
    if len(f) < 8 or f["cluster"].nunique() < 4 or f["rank"].nunique() < 2:
        return {"status": "insufficient", "n_cells": int(len(f))}
    z = np.zeros(len(f))
    a = Arrays(
        cluster=pd.factorize(f["cluster"])[0], rank=f["rank"].to_numpy(int), delta=f["slope"].to_numpy(float),
        var=(f["se"] ** 2).to_numpy(float), race=z, prac=z, var_race=z, sxy_r=z, sxx_r=z, sxy_p=z, sxx_p=z,
        temp_r=z, temp_p=z,
    )
    fit = trend_fit(a)
    t2 = fit["tau2"]
    lo, hi, _ = boot_ci(lambda x: trend_beta(x, t2), a, B, seed)
    return {"status": "ok", "n_cells": int(len(a)), "beta_per_step_softer": fit["beta"], "ci95": [lo, hi],
            "n_event_seasons": int(len(np.unique(a.cluster))),
            "mean_slope_by_compound": {lab: float(f.loc[f["Compound"] == lab, "slope"].mean())
                                       for lab in LABELS if (f["Compound"] == lab).any()}}


def twfe_race_contrasts(bases: Bases, spec: Spec, min_runs: int = 2) -> pd.DataFrame:
    """Race-side TWFE compound contrasts only (no practice counterpart needed)."""
    race, _ = context_frames(bases, spec, prepare=False)
    if race.empty:
        return pd.DataFrame()
    race = race.reset_index(drop=True)
    race["fe_key"] = race["LapNumber"].astype(int).astype(str)
    rows = []
    for key, g in race.groupby(["season", "event"]):
        for a, b in (("SOFT", "HARD"), ("MEDIUM", "HARD"), ("SOFT", "MEDIUM")):
            c = pair_age_contrast(g, a, b, min_runs)
            if c is not None and np.isfinite(c[1]):
                rows.append({"season": int(key[0]), "event": key[1], "pair": f"{a}-{b}", "contrast": c[0], "se": c[1]})
    return pd.DataFrame(rows)


def reconcile_race_trends(bases: Bases, *, B: int, seed: int) -> dict:
    """Why does the race compound trend change sign between analyses?

    Fits the RACE-only compound trend on the same event-season-compound cells under
    four slope estimators, for 2026 alone and for all seasons, with and without the
    tyre-age window, and reports the race-side TWFE pair contrasts beside them. The
    cell set is the intersection across estimators within each setting, so any
    difference between rows is the estimator, not the sample.
    """
    out: dict = {"settings": {}}
    estimators = ("within_run", "xs", "field_origin", "field_runcentred")
    for sname, seasons in (("all_seasons", ALL_SEASONS), ("2026_only", (2026,))):
        for wname, (lo, hi) in (("window_6_16", (6, 16)), ("no_window", (None, None))):
            fits = {}
            for est in estimators:
                sp = replace(PRIMARY, seasons=seasons, age_lo=lo, age_hi=hi, race_estimator=est, min_runs=3)
                race, _ = context_frames(bases, sp)
                fits[est] = cell_fits(race, 3)
            keys = None
            for f in fits.values():
                k = set(zip(f["season"], f["event"], f["Compound"], strict=True)) if len(f) else set()
                keys = k if keys is None else keys & k
            rows = {}
            for j, (est, f) in enumerate(fits.items()):
                sub = f[[(s, e, c) in keys for s, e, c in zip(f["season"], f["event"], f["Compound"], strict=True)]] if len(f) else f
                rows[est] = race_only_trend(sub, B=B, seed=seed + j)
            spt = replace(PRIMARY, seasons=seasons, age_lo=lo, age_hi=hi)
            tc = twfe_race_contrasts(bases, spt)
            twfe = {}
            for pair in ("SOFT-HARD", "MEDIUM-HARD", "SOFT-MEDIUM"):
                t = tc[tc["pair"] == pair] if len(tc) else tc
                twfe[pair] = {"n_events": int(len(t)), "mean_contrast": float(t["contrast"].mean()) if len(t) else None,
                              "re_mean_contrast": _re_mean(t["contrast"].to_numpy(), (t["se"] ** 2).to_numpy())[0] if len(t) >= 3 else None}
            # Additivity on events that have all three pairs: SOFT-HARD should equal SOFT-MEDIUM + MEDIUM-HARD.
            add = None
            if len(tc):
                piv = tc.pivot_table(index=["season", "event"], columns="pair", values="contrast")
                full = piv.dropna()
                if len(full) >= 3:
                    add = {"n_events_with_all_three_pairs": int(len(full)),
                           "mean_SOFT-HARD": float(full["SOFT-HARD"].mean()),
                           "mean_SOFT-MEDIUM_plus_MEDIUM-HARD": float((full["SOFT-MEDIUM"] + full["MEDIUM-HARD"]).mean())}
            out["settings"][f"{sname}|{wname}"] = {
                "n_common_cells": len(keys or ()),
                "race_only_trend_by_estimator": rows,
                "race_twfe_pair_contrasts": twfe,
                "twfe_additivity_check": add,
            }
    return out

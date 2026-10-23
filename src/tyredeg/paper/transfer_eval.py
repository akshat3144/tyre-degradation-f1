"""Honest evaluation of the practice -> race forecast.

WHY THIS MODULE EXISTS
    The shipped claim was "leave-one-event-out MAE 0.056 s/lap against a naive
    0.087". A hostile audit found that number is soft in four ways, and this
    module is the code that answers each one:

    1. The practice design (warm-up lap, run-gap tolerance) and the session
       weights were chosen on the SAME 21 cells the claim is scored on.
    2. The baseline was weak: predicting a constant, or zero, scores about as
       well as the practice forecast does.
    3. One scalar factor was applied although the docstring said the gap widens
       for softer compounds.
    4. The 13-variant design sweep no longer reproduced.

    So the evaluation here has five parts, each defined BEFORE its numbers were
    read (the pre-stated tests are in ``scripts/paper/p2_transfer.py``):

    * a faithful, parameterised rebuild of the practice pipeline
      (``build_practice``), pinned to ``design.prepare`` by a parity test so a
      sweep can never again compare against a pipeline that is not the shipped
      one;
    * a true holdout: the 2026-chosen design and factor frozen and applied to
      2022-2025, where compounds are HARD/MEDIUM/SOFT labels and nothing was
      ever tuned;
    * nested cross-validation: design and session weights chosen INSIDE each
      training fold, the held-out event scored once;
    * a ladder of baselines, with event-clustered bootstrap intervals and a
      paired sign-flip test on every pairwise MAE difference;
    * accuracy tests for the identifiability rule and the stand-in rates.

DESIGN RULES
    Nothing here reads a held-out event while choosing anything about it.
    Every random draw uses ``SEED``. Cells keep the unit of analysis the
    original claim used: one (event, compound) pair, with its race rate as the
    target -- so a "weekend" is the cluster, since cells of one weekend share
    a race, a track state and a pit-wall.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MIN_LONG_RUN_LAPS, ROOT
from ..data.schedule import SPRINT_SESSION
from ..data.session_weight import SESSION_WEIGHTS
from ..models import design as D
from ..validation import transfer as T

SEED = 20260101

#: Bootstrap draws. Event-level resampling on a few dozen clusters is cheap,
#: so this is generous; it is not a tuned quantity.
N_BOOT = 10_000

#: Compound softness, hardest to softest. Used only for the trend test.
LABEL_ORDER = ("HARD", "MEDIUM", "SOFT")

#: Circuits that moved or were renamed. A 2026 "Barcelona Grand Prix" is the
#: Circuit de Barcelona-Catalunya that 2022-2025 called the "Spanish Grand
#: Prix"; the 2026 "Spanish Grand Prix" is Madrid, a new circuit with no
#: history at all. Matching on the event name alone would join them wrongly.
CIRCUIT_ALIAS_2026 = {
    "Barcelona Grand Prix": "Barcelona",
    "Spanish Grand Prix": "Madrid",
}


# ---------------------------------------------------------------------------
# practice pipeline, parameterised
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PracticeDesign:
    """Every choice the practice frame makes, as an explicit value.

    The defaults ARE the shipped pipeline. ``build_practice(laps, PracticeDesign())``
    must equal ``design.prepare(laps, "practice")`` lap for lap; the parity
    test fails if it ever does not, which is how the old sweep broke: its
    "current" variant was the pre-keep-runs pipeline, so it compared every
    candidate against a design nobody ships.
    """

    name: str = "shipped"
    #: Detect runs on the stint BEFORE dropping unrepresentative laps, so a
    #: removed lap does not shatter a run. False = the legacy order.
    keep_runs: bool = True
    drop_warmup: int = D.WARMUP_LAPS
    #: Re-split runs where lap numbering jumps by more than this. None = never.
    max_gap: int | None = D.MAX_RUN_GAP_LAPS
    race_sims_only: bool = True
    min_laps: int = MIN_LONG_RUN_LAPS
    min_runs_per_compound: int = 5
    #: A Sprint has no qualifying runs to tell apart, so every Sprint lap counts
    #: as race running (design.prepare does this). False routes the Sprint
    #: through the pace classifier like practice, which throws away the leaders.
    sprint_is_race: bool = True


SHIPPED = PracticeDesign()

#: The 13 variants of ``scripts/13_design_sweep.py``, named relative to what
#: ships. "legacy" is what shipped BEFORE keep-runs: filter, then detect runs,
#: no warm-up trim, no Sprint rule. It is kept so the sweep's original headline
#: (legacy -> warmup + gap) can be re-measured honestly.
DESIGN_GRID: dict[str, PracticeDesign] = {
    "legacy": PracticeDesign("legacy", keep_runs=False, drop_warmup=0, max_gap=None,
                             sprint_is_race=False),
    "keep-runs": PracticeDesign("keep-runs", drop_warmup=0, max_gap=None),
    "warmup1": PracticeDesign("warmup1", drop_warmup=1, max_gap=None),
    "warmup2": PracticeDesign("warmup2", drop_warmup=2, max_gap=None),
    "warmup3": PracticeDesign("warmup3", drop_warmup=3, max_gap=None),
    "warmup1-min4": PracticeDesign("warmup1-min4", drop_warmup=1, max_gap=None, min_laps=4),
    "warmup1-min6": PracticeDesign("warmup1-min6", drop_warmup=1, max_gap=None, min_laps=6),
    "warmup1-noshape": PracticeDesign("warmup1-noshape", keep_runs=False, drop_warmup=1,
                                      max_gap=None),
    "warmup1-gap1": PracticeDesign("warmup1-gap1", drop_warmup=1, max_gap=2),
    "warmup1-gap2": PracticeDesign("warmup1-gap2", drop_warmup=1, max_gap=3),  # == shipped
    "warmup1-gap4": PracticeDesign("warmup1-gap4", drop_warmup=1, max_gap=5),
    "all-runs": PracticeDesign("all-runs", drop_warmup=0, max_gap=None, race_sims_only=False),
    "min4": PracticeDesign("min4", drop_warmup=0, max_gap=None, min_laps=4),
}

#: Weighting schemes. Positions someone could argue for, not a search space.
#: Same seven as ``scripts/12_session_skill.py``; duplicated rather than
#: imported because a script whose name starts with a digit cannot be.
WEIGHT_SCHEMES: dict[str, dict[str, float]] = {
    "FP2 heavy (shipped)": {"FP1": 0.5, "FP2": 1.0, "FP3": 0.5, "S": 1.0},
    "equal": {"FP1": 1.0, "FP2": 1.0, "FP3": 1.0, "S": 1.0},
    "FP1+FP2 equal": {"FP1": 1.0, "FP2": 1.0, "FP3": 0.5, "S": 1.0},
    "FP1 heavy": {"FP1": 1.0, "FP2": 0.65, "FP3": 0.5, "S": 1.0},
    "FP3 down only": {"FP1": 1.0, "FP2": 1.0, "FP3": 0.25, "S": 1.0},
    "sprint down": {"FP1": 0.5, "FP2": 1.0, "FP3": 0.5, "S": 0.5},
    "sprint only": {"FP1": 0.05, "FP2": 1.0, "FP3": 0.5, "S": 1.0},
}


def physical_compound(df: pd.DataFrame) -> pd.DataFrame:
    """2026: HARD/MEDIUM/SOFT -> C1..C5 through the allocation store."""
    return D.add_physical_compound(df)


def label_compound(df: pd.DataFrame) -> pd.DataFrame:
    """2022-2025: the label IS the compound.

    The repo holds Pirelli's allocation for 2026 only, so the physical C number
    of an earlier race is unknown here. Cells are therefore keyed on the label,
    which is relative to each weekend's nomination. That makes cells from
    different weekends physically different tyres -- an honest limitation of
    the holdout, and one that makes any transfer harder, not easier.
    """
    df = df.copy()
    c = df["Compound"].astype(str).str.upper()
    df["C"] = c.where(c.isin(LABEL_ORDER))
    return df


def build_practice(
    laps: pd.DataFrame,
    design: PracticeDesign = SHIPPED,
    *,
    compound_fn: Callable[[pd.DataFrame], pd.DataFrame] = physical_compound,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """The practice frame, built to ``design`` through the shipped components.

    Every step is a function from ``models.design``; only their ORDER and
    parameters vary. With the default ``design`` this is ``design.prepare`` for
    context="practice" (pinned by a parity test).

    ``weights`` overrides the session weights without touching the module
    constant. The default is whatever ``design.practice_design`` assigned,
    i.e. the shipped table.
    """
    df = compound_fn(laps)
    df = df[df["C"].notna()].dropna(subset=["TyreLife", "LapTimeSeconds"])
    df = df[df["session"] != "R"]
    if df.empty:
        return df

    if design.keep_runs:
        df = _tag(df, 1)
        df = D.drop_non_representative_laps(df)
        if design.max_gap is not None:
            df = D.split_on_large_gaps(df, design.max_gap)
        if design.drop_warmup:
            df = df[df.groupby("run_id").cumcount() >= design.drop_warmup]
        df = df[df.groupby("run_id")["LapNumber"].transform("size") >= design.min_laps]
    else:
        df = D.drop_non_representative_laps(df)
        df = _tag(df, design.min_laps)
        df = df[df["is_long_run"]]
        if design.drop_warmup:
            df = df[df.groupby("run_id").cumcount() >= design.drop_warmup]

    if df.empty:
        return df
    df = D.drop_stint_outliers(df)
    df = D.classify_runs(df)
    if design.sprint_is_race:
        df.loc[df["session"] == SPRINT_SESSION, "is_race_sim"] = True
    if design.race_sims_only:
        df = df[df["is_race_sim"]]
    if df.empty:
        return df
    df = D.practice_design(df)
    if weights is not None:
        df["w"] = df["session"].map(lambda s: weights.get(str(s).upper(), 1.0)).astype(float)
    runs = df.groupby("C")["run_id"].transform("nunique")
    return df[runs >= design.min_runs_per_compound].copy()


def _tag(df: pd.DataFrame, min_laps: int) -> pd.DataFrame:
    from ..data.laps import tag_long_runs

    return tag_long_runs(df, min_laps=min_laps)


def build_race(
    laps: pd.DataFrame,
    *,
    compound_fn: Callable[[pd.DataFrame], pd.DataFrame] = physical_compound,
) -> pd.DataFrame:
    """The race frame -- ``design.prepare(context="race")`` with a pluggable compound."""
    df = compound_fn(laps)
    df = df[df["C"].notna()].dropna(subset=["TyreLife", "LapTimeSeconds"])
    df = D.drop_stint_outliers(df[df["is_long_run"]])
    df = D.race_design(df)
    runs = df.groupby("C")["run_id"].transform("nunique")
    return df[runs >= 5].copy()


def reweight(practice: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """The same frame under different session weights. Cheap: only ``w`` changes."""
    out = practice.copy()
    out["w"] = out["session"].map(lambda s: weights.get(str(s).upper(), 1.0)).astype(float)
    return out


# ---------------------------------------------------------------------------
# cells: one row per (weekend, compound) with practice and race rates
# ---------------------------------------------------------------------------

PROCESSED_NAME = {2026: "laps.parquet"}


def laps_path(season: int) -> Path:
    from ..config import PROCESSED

    return PROCESSED / PROCESSED_NAME.get(season, f"laps_{season}.parquet")


def load_laps(season: int) -> pd.DataFrame:
    return pd.read_parquet(laps_path(season))


def circuit_of(event: str, season: int) -> str:
    """The circuit behind an event name, so 'history' joins the right track."""
    if season >= 2026:
        return CIRCUIT_ALIAS_2026.get(event, event)
    return "Barcelona" if event == "Spanish Grand Prix" else event


def weekend_key(event: str, season: int) -> str:
    """Unique per weekend across seasons. 2026 keeps its plain name."""
    return event if season == 2026 else f"{season} {event}"


def _rekey(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Rename events to weekend keys and make run ids season-unique."""
    if df.empty or season == 2026:
        return df
    df = df.copy()
    df["event"] = [weekend_key(e, season) for e in df["event"]]
    if "run_id" in df.columns:
        df["run_id"] = f"{season}|" + df["run_id"].astype(str)
    return df


def compound_fn_for(season: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return physical_compound if season == 2026 else label_compound


def label_of_2026(event: str, c: str) -> str | None:
    """HARD/MEDIUM/SOFT for a C number at a 2026 event, from the allocation store."""
    from ..data.allocation import all_allocations

    a = all_allocations().get(event)
    if a is None:
        return None
    inv = {v: k for k, v in a.compounds.items()}
    return inv.get(c)


@dataclass
class SeasonData:
    season: int
    practice: pd.DataFrame
    race: pd.DataFrame


def build_season(
    season: int,
    laps: pd.DataFrame | None = None,
    design: PracticeDesign = SHIPPED,
    weights: dict[str, float] | None = None,
) -> SeasonData:
    laps = load_laps(season) if laps is None else laps
    cf = compound_fn_for(season)
    pr = build_practice(laps, design, compound_fn=cf, weights=weights)
    rc = build_race(laps, compound_fn=cf)
    return SeasonData(season, _rekey(pr, season), _rekey(rc, season))


def race_cell_table(race: pd.DataFrame, season: int) -> pd.DataFrame:
    """Every race cell the design can identify, with or without practice."""
    cols = ["event", "season", "circuit", "C", "label", "race_rate", "se_r", "n_runs_r"]
    r = T.cell_rates(race, T.MIN_RUNS, min_age_spread=T.MIN_AGE_SPREAD_LAPS)
    if r.empty:
        return pd.DataFrame(columns=cols)
    r = r.rename(columns={"rate": "race_rate", "se": "se_r", "n_runs": "n_runs_r"})
    plain = list(r["event"]) if season == 2026 else [e.split(" ", 1)[1] for e in r["event"]]
    r["season"] = season
    r["circuit"] = [circuit_of(e, season) for e in plain]
    r["label"] = [
        (label_of_2026(e, c) if season == 2026 else c)
        for e, c in zip(plain, r["C"], strict=True)
    ]
    return r[cols]


def cell_panel(practice: pd.DataFrame, race: pd.DataFrame, season: int) -> pd.DataFrame:
    """Joined (weekend, compound) cells: practice rate, race rate and their errors.

    Practice needs ``MIN_RUNS`` runs; the race needs ``MIN_RUNS`` runs AND the
    tyre-age spread the design requires -- the same two gates the original claim
    used, so the 2026 panel is its 21 cells exactly.
    """
    p = T.cell_rates(practice, T.MIN_RUNS).rename(
        columns={"rate": "practice_rate", "se": "se_p", "n_runs": "n_runs_p", "n_laps": "n_laps_p"}
    )
    r = race_cell_table(race, season)
    if p.empty or r.empty:
        return pd.DataFrame()
    j = p.drop(columns=["age_spread"]).merge(r, on=["event", "C"])
    j["grp"] = j["C"]
    return j.sort_values(["event", "C"]).reset_index(drop=True)


def attach_history(panel: pd.DataFrame, race_cells: pd.DataFrame) -> pd.DataFrame:
    """The same circuit's most recent EARLIER-season race rate for the same label.

    Strictly earlier seasons only, so a cell never sees its own race or any
    race run after it. NaN when the circuit-label has no earlier cell.

    ``hist_event`` records WHICH race the history came from. A leave-one-weekend-
    out fit needs it: a training cell from a later season carries the held-out
    weekend's own race rate as its history, which is the held-out outcome
    leaking into the training set (see ``loeo``).
    """
    panel = panel.copy()
    rc = race_cells.dropna(subset=["label", "circuit"])
    hist, hs, he = [], [], []
    for _, row in panel.iterrows():
        m = rc[(rc["circuit"] == row["circuit"]) & (rc["label"] == row["label"])
               & (rc["season"] < row["season"])]
        if m.empty:
            hist.append(np.nan)
            hs.append(np.nan)
            he.append(None)
        else:
            top = m.sort_values("season").iloc[-1]
            hist.append(float(top["race_rate"]))
            hs.append(float(top["season"]))
            he.append(top["event"] if "event" in top.index else None)
    panel["hist_rate"] = hist
    panel["hist_season"] = hs
    panel["hist_event"] = he
    return panel


def season_panels(
    seasons: Sequence[int],
    design: PracticeDesign = SHIPPED,
    weights: dict[str, float] | None = None,
    data: dict[int, SeasonData] | None = None,
    history_seasons: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panels for several seasons stacked, plus the race-cell history table.

    ``grp`` is the compound key used for pooling: the C number in 2026 (what
    the original claim used) and the label elsewhere. ``history_seasons`` are
    the seasons whose race cells may serve as 'same circuit, earlier season'
    history; they default to ``seasons``.
    """
    panels, races = [], []
    for s in sorted(set(seasons) | set(history_seasons or ())):
        sd = data[s] if data and s in data else build_season(s, design=design, weights=weights)
        races.append(race_cell_table(sd.race, s))
        if s in seasons:
            pn = cell_panel(sd.practice, sd.race, s)
            if not pn.empty:
                panels.append(pn)
    race_cells = pd.concat(races, ignore_index=True)
    panel = pd.concat(panels, ignore_index=True)
    return attach_history(panel, race_cells), race_cells


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

METHODS = (
    "zero", "const", "pooled", "history", "naive", "scalar", "compound", "shrink50", "stack",
)

#: Minimum training cells in a compound before it earns its own factor. Below
#: this it borrows the global factor. Fixed in advance, not tuned.
K_MIN_COMPOUND = 3


def ratio_factor(tr: pd.DataFrame) -> float:
    """The shipped factor: median of race/practice over cells with a usable practice rate."""
    u = tr[tr["practice_rate"].abs() > 1e-6]
    if len(u) < 2:
        return 1.0
    return float(np.median(u["race_rate"] / u["practice_rate"]))


def fit_calibration(train: pd.DataFrame) -> dict:
    """Everything a forecast learns from its training cells, and nothing else."""
    f = ratio_factor(train)
    cf = {}
    for g, tg in train.groupby("grp"):
        u = tg[tg["practice_rate"].abs() > 1e-6]
        cf[g] = float(np.median(u["race_rate"] / u["practice_rate"])) if len(u) >= K_MIN_COMPOUND else f
    pool = {g: float(tg["race_rate"].median()) for g, tg in train.groupby("grp")}
    const = float(train["race_rate"].median())
    prior_tr = train["hist_rate"].where(train["hist_rate"].notna(),
                                        train["grp"].map(pool)).to_numpy(float)
    cal_tr = f * train["practice_rate"].to_numpy(float)
    d = cal_tr - prior_tr
    denom = float(d @ d)
    if denom > 0 and len(train) >= 4:
        w = float(np.clip(((train["race_rate"].to_numpy(float) - prior_tr) @ d) / denom, 0, 1))
    else:
        w = 0.5
    return {"factor": f, "compound_factor": cf, "pool": pool, "const": const, "stack_w": w}


def predict(test: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """Every method's forecast for ``test``, from a calibration fitted elsewhere."""
    out = test.copy()
    f = cal["factor"]
    pooled = out["grp"].map(cal["pool"]).fillna(cal["const"]).to_numpy(float)
    hist = out["hist_rate"].to_numpy(float)
    prior = np.where(np.isfinite(hist), hist, pooled)
    prac = out["practice_rate"].to_numpy(float)
    fc = out["grp"].map(cal["compound_factor"]).fillna(f).to_numpy(float)
    out["pred_zero"] = 0.0
    out["pred_const"] = cal["const"]
    out["pred_pooled"] = pooled
    out["pred_history"] = prior
    out["pred_naive"] = prac
    out["pred_scalar"] = f * prac
    out["pred_compound"] = fc * prac
    out["pred_shrink50"] = 0.5 * f * prac + 0.5 * prior
    out["pred_stack"] = cal["stack_w"] * f * prac + (1 - cal["stack_w"]) * prior
    out["factor_used"] = f
    out["stack_w"] = cal["stack_w"]
    out["has_history"] = np.isfinite(hist)
    return out


def _hist_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in ("hist_rate", "hist_season", "hist_event") if c in df.columns]


def drop_heldout_history(
    train: pd.DataFrame, held_out: Sequence[str], race_cells: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Training cells whose history IS a held-out race, with that history rebuilt without it.

    In leave-one-weekend-out over stacked seasons, a training cell from a LATER
    season at the same circuit carries, as its ``hist_rate``, the held-out
    weekend's own race rate -- and ``fit_calibration`` uses ``hist_rate`` to
    set the stacking weight. The held-out outcome was therefore inside the
    training data. Each such cell's history is recomputed from ``race_cells``
    with the held-out weekend removed (it falls back to the next most recent
    earlier season, or to none), or, with no ``race_cells`` to recompute from,
    simply cleared.
    """
    if "hist_event" not in train.columns:
        return train
    leaked = train["hist_event"].isin(list(held_out))
    if not leaked.any():
        return train
    out = train.copy()
    cols = _hist_cols(out)
    if race_cells is not None:
        fixed = attach_history(out.loc[leaked].drop(columns=cols), race_cells[~race_cells["event"].isin(list(held_out))])
        out.loc[leaked, cols] = fixed[cols].to_numpy()
    else:
        out.loc[leaked, "hist_rate"] = np.nan
        out.loc[leaked, "hist_season"] = np.nan
        out.loc[leaked, "hist_event"] = None
    return out


def loeo(panel: pd.DataFrame, cluster: str = "event",
         race_cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """Leave-one-weekend-out forecasts. The held-out weekend informs none of its own.

    That includes through history: see ``drop_heldout_history``. Pass the
    ``race_cells`` the panel's history was attached from to recompute it; without
    them the leaked history is cleared.
    """
    if "hist_rate" not in panel.columns:
        panel = panel.assign(hist_rate=np.nan)
    parts = []
    for e in panel[cluster].unique():
        tr, te = panel[panel[cluster] != e], panel[panel[cluster] == e]
        tr = drop_heldout_history(tr, list(te["event"].unique()), race_cells)
        parts.append(predict(te, fit_calibration(tr)))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# scoring and paired inference
# ---------------------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(a, b)[0])


def score(df: pd.DataFrame, method: str) -> dict:
    y = df["race_rate"].to_numpy(float)
    p = df[f"pred_{method}"].to_numpy(float)
    e = p - y
    return {
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "spearman": _spearman(p, y),
    }


def _cluster_sums(values: np.ndarray, cluster: np.ndarray):
    codes, uniq = pd.factorize(cluster)
    s = np.bincount(codes, weights=values, minlength=len(uniq))
    n = np.bincount(codes, minlength=len(uniq)).astype(float)
    return s, n


def boot_mean_ci(values: np.ndarray, cluster: np.ndarray, n_boot: int = N_BOOT,
                 seed: int = SEED) -> tuple[float, float, float]:
    """Mean and 95% percentile CI, resampling whole clusters (weekends)."""
    s, n = _cluster_sums(np.asarray(values, float), np.asarray(cluster))
    g = len(s)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g, size=(n_boot, g))
    m = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(s.sum() / n.sum()), float(lo), float(hi)


def paired_test(a_err: np.ndarray, b_err: np.ndarray, cluster: np.ndarray,
                n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Is method A's absolute error different from method B's? Clustered by weekend.

    ``d = |err_A| - |err_B|``; negative means A is better. Three views of the
    same difference, because each can fail where the others do not:

    * a cluster bootstrap CI on the mean difference;
    * a Diebold-Mariano-style t statistic with a cluster-robust variance and
      ``G - 1`` degrees of freedom (the loss differential is the absolute
      error, the 'series' is the weekends);
    * a sign-flip randomisation test on the weekend sums, which assumes only
      that under the null a weekend's differential is symmetric about zero.
    """
    from scipy import stats

    d = np.abs(np.asarray(a_err, float)) - np.abs(np.asarray(b_err, float))
    cl = np.asarray(cluster)
    mean, lo, hi = boot_mean_ci(d, cl, n_boot, seed)
    s, n = _cluster_sums(d - d.mean(), cl)
    g, nn = len(s), len(d)
    var = (g / max(1, g - 1)) * float((s ** 2).sum()) / (nn ** 2)
    t = mean / np.sqrt(var) if var > 0 else float("nan")
    p_dm = float(2 * stats.t.sf(abs(t), df=max(1, g - 1))) if np.isfinite(t) else float("nan")

    sums, _ = _cluster_sums(d, cl)
    obs = abs(sums.sum())
    rng = np.random.default_rng(seed + 1)
    if g <= 14:
        signs = ((np.arange(2 ** g)[:, None] >> np.arange(g)) & 1) * 2 - 1
    else:
        signs = rng.choice([-1, 1], size=(20_000, g))
    hits = int((np.abs(signs @ sums) >= obs - 1e-12).sum())
    if g <= 14:
        # Every sign pattern was enumerated, and the identity pattern is one of
        # them: the p-value is exact and cannot be zero.
        p_flip = float(hits / len(signs))
    else:
        # Monte Carlo draws: add the observed statistic as one of the draws, so
        # the p-value is never reported as exactly zero (Phipson & Smyth 2010).
        p_flip = float((hits + 1) / (len(signs) + 1))
    return {
        "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
        "dm_t": float(t), "dm_p": p_dm, "signflip_p": p_flip,
        "n_cells": int(nn), "n_clusters": int(g),
        "frac_cells_a_better": float((d < 0).mean()),
    }


def compare(df: pd.DataFrame, focal: str, others: Iterable[str],
            cluster: str = "event") -> dict:
    """``focal`` against each of ``others`` on the cells of ``df``."""
    out = {}
    cl = df[cluster].to_numpy()
    y = df["race_rate"].to_numpy(float)
    for o in others:
        out[o] = paired_test(df[f"pred_{focal}"].to_numpy(float) - y,
                             df[f"pred_{o}"].to_numpy(float) - y, cl)
    return out


def scoreboard(df: pd.DataFrame, methods: Iterable[str], cluster: str = "event") -> dict:
    """MAE (with CI), RMSE and rank correlation per method, on one set of cells."""
    out = {}
    cl = df[cluster].to_numpy()
    y = df["race_rate"].to_numpy(float)
    for m in methods:
        if f"pred_{m}" not in df.columns:
            continue
        s = score(df, m)
        _, lo, hi = boot_mean_ci(np.abs(df[f"pred_{m}"].to_numpy(float) - y), cl)
        out[m] = {**s, "mae_ci_lo": lo, "mae_ci_hi": hi}
    return out


# ---------------------------------------------------------------------------
# out-of-season holdout and nested selection
# ---------------------------------------------------------------------------


def frozen_holdout(train: pd.DataFrame, test: pd.DataFrame,
                   race_cells: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply a calibration fitted on ``train`` to ``test`` without refitting.

    A 2026 training cell's ``hist_rate`` is an earlier season's race rate at its
    circuit -- that is, the race outcome of a weekend in ``test``. Left in, the
    held-out outcomes feed the stacking weight, so training history that comes
    from a test weekend is rebuilt without it (``drop_heldout_history``).

    ``train`` is the 2026 panel with ``grp`` switched to the HARD/MEDIUM/SOFT
    label, because the holdout seasons carry no C numbers. The factor, the
    per-label factors, the constants and the stacking weight are all 2026's.
    """
    tr = drop_heldout_history(train, list(test["event"].unique()), race_cells)
    tr = tr.copy()
    tr["grp"] = tr["label"]
    return predict(test, fit_calibration(tr))


def candidate_tables(
    laps: pd.DataFrame,
    season: int = 2026,
    designs: dict[str, PracticeDesign] | None = None,
    schemes: dict[str, dict[str, float]] | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    """One practice-vs-race cell table per (design, weighting) candidate.

    The race side is built once -- it does not depend on either choice -- and
    each candidate's table is its own practice rates against it.
    """
    designs = DESIGN_GRID if designs is None else designs
    schemes = WEIGHT_SCHEMES if schemes is None else schemes
    race = _rekey(build_race(laps, compound_fn=compound_fn_for(season)), season)
    out = {}
    for dn, d in designs.items():
        base = _rekey(build_practice(laps, d, compound_fn=compound_fn_for(season)), season)
        if base.empty:
            continue
        for wn, w in schemes.items():
            pr = reweight(base, w)
            tab = cell_panel(pr, race, season)
            if not tab.empty:
                out[(dn, wn)] = tab
    return out


#: The reference candidate: what ships, and so the cells the claim is scored on.
SHIPPED_KEY = ("warmup1-gap2", "FP2 heavy (shipped)")


def _arrays(tab: pd.DataFrame, ref_cells: set) -> dict:
    """Plain arrays for the inner loop, so 91 candidates x 12 folds stays fast."""
    p = tab["practice_rate"].to_numpy(float)
    ratio = np.where(np.abs(p) > 1e-6, tab["race_rate"].to_numpy(float) / np.where(p == 0, 1, p),
                     np.nan)
    return {
        "ev": tab["event"].to_numpy(), "p": p, "y": tab["race_rate"].to_numpy(float),
        "ratio": ratio,
        "inref": np.array([(e, c) in ref_cells for e, c in zip(tab["event"], tab["C"], strict=True)]),
    }


def _inner_mae(arr: dict, events: list[str]) -> float:
    """Leave-one-event-out MAE of one candidate, over reference cells in ``events``."""
    in_train = np.isin(arr["ev"], events)
    errs = []
    for e in events:
        te = (arr["ev"] == e) & arr["inref"]
        if not te.any():
            continue
        rr = arr["ratio"][in_train & (arr["ev"] != e)]
        rr = rr[np.isfinite(rr)]
        f = float(np.median(rr)) if len(rr) >= 2 else 1.0
        errs.extend(np.abs(f * arr["p"][te] - arr["y"][te]))
    return float(np.mean(errs)) if errs else float("inf")


def nested_selection(tables: dict[tuple[str, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose design AND session weights inside each training fold, score the held-out event.

    For each held-out weekend ``e``:

    1. Training weekends are every other weekend. A candidate is ELIGIBLE if it
       answers every training cell the shipped design answers -- the rule the
       original sweep used ("fair = at least as much coverage"), applied inside
       the fold so a candidate cannot win by refusing hard questions.
    2. Each eligible candidate is scored by an inner leave-one-event-out MAE on
       those common training cells. The lowest wins; ties go to the earlier
       candidate in grid order.
    3. The winner's factor is fitted on ALL its training cells and applied to
       ``e``'s cells. Where the winner has no practice rate for a held-out cell
       the shipped design's rate is used, and the row is flagged.

    Returns (per-cell predictions, per-fold selection log).
    """
    ref = tables[SHIPPED_KEY]
    ref_cells = set(zip(ref["event"], ref["C"], strict=True))
    events = list(ref["event"].unique())
    rows, log = [], []
    arrs = {k: _arrays(tab, ref_cells) for k, tab in tables.items()}
    haves = {k: set(zip(tab["event"], tab["C"], strict=True)) for k, tab in tables.items()}
    for e in events:
        train_ev = [x for x in events if x != e]
        ref_train = {c for c in ref_cells if c[0] != e}
        best, best_key = float("inf"), None
        n_elig = 0
        for key, tab in tables.items():
            if not ref_train <= haves[key]:
                continue
            n_elig += 1
            m = _inner_mae(arrs[key], train_ev)
            if m < best - 1e-12:
                best, best_key = m, key
        tab = tables[best_key]
        f = ratio_factor(tab[tab["event"].isin(train_ev)])
        te_ref = ref[ref["event"] == e]
        got = tab[tab["event"] == e].set_index("C")
        for _, r in te_ref.iterrows():
            fallback = r["C"] not in got.index
            prac = float(r["practice_rate"] if fallback else got.loc[r["C"], "practice_rate"])
            rows.append({
                "event": e, "C": r["C"], "race_rate": r["race_rate"],
                "pred_naive_nested": prac, "pred_scalar_nested": f * prac,
                "factor_nested": f, "chosen_design": best_key[0],
                "chosen_weights": best_key[1], "fallback": fallback,
            })
        log.append({"held_out": e, "design": best_key[0], "weights": best_key[1],
                    "inner_mae": best, "n_eligible": n_elig})
    return pd.DataFrame(rows), pd.DataFrame(log)


# ---------------------------------------------------------------------------
# factor structure: does the gap widen for softer compounds?
# ---------------------------------------------------------------------------


def factor_by_compound(panel: pd.DataFrame, order: Sequence[str], n_boot: int = 2000,
                       seed: int = SEED) -> dict:
    """Race-on-practice factor per compound, with weekend-clustered CIs and a trend.

    Two estimators, because the median of ratios (what ships) is robust to
    one huge cell and the least-squares factor (sum race*practice / sum
    practice^2) is what the correlation actually supports. The trend is the
    least-squares slope of the per-cell ratio ``race/practice`` on softness
    rank; negative supports "the gap widens for softer tyres".
    """
    p = panel[panel["practice_rate"].abs() > 1e-6].copy()
    rank = {c: i for i, c in enumerate(order)}
    p = p[p["grp"].isin(rank)].reset_index(drop=True)
    if p.empty:
        return {}
    x = p["practice_rate"].to_numpy(float)
    y = p["race_rate"].to_numpy(float)
    r = p["grp"].map(rank).to_numpy(float)
    g = p["grp"].to_numpy()
    evs = pd.unique(p["event"])
    pos = [np.flatnonzero(p["event"].to_numpy() == e) for e in evs]
    groups = sorted(set(g), key=lambda c: rank[c])

    def stats_of(idx: np.ndarray) -> dict:
        out = {}
        xi, yi, ri, gi = x[idx], y[idx], r[idx], g[idx]
        for c in groups:
            m = gi == c
            if not m.any():
                continue
            den = float((xi[m] ** 2).sum())
            out[c] = {"n": int(m.sum()), "median_ratio": float(np.median(yi[m] / xi[m])),
                      "ls_factor": float((yi[m] * xi[m]).sum() / den) if den else np.nan}
        out["_trend"] = float(np.polyfit(ri, yi / xi, 1)[0]) if len(np.unique(ri)) > 1 else np.nan
        return out

    point = stats_of(np.arange(len(p)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(pos), len(pos))
        boots.append(stats_of(np.concatenate([pos[i] for i in pick])))
    res = {}
    for c, v in point.items():
        if c == "_trend":
            continue
        for k in ("median_ratio", "ls_factor"):
            arr = np.array([b[c][k] for b in boots if c in b])
            v[f"{k}_ci"] = ([float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))]
                            if len(arr) else [None, None])
        res[c] = v
    tr = np.array([b["_trend"] for b in boots])
    res["_trend_per_softness_step"] = {
        "slope": point["_trend"],
        "ci": [float(np.nanpercentile(tr, 2.5)), float(np.nanpercentile(tr, 97.5))],
        "share_boot_negative": float(np.nanmean(tr < 0)),
    }
    return res


# ---------------------------------------------------------------------------
# identifiability rule vs continuous shrinkage; stand-in accuracy
# ---------------------------------------------------------------------------


def standin_table(practice: pd.DataFrame, panel: pd.DataFrame, factor_by_event: dict) -> pd.DataFrame:
    """Hide each measured cell in turn and impute it with the shipped stand-in.

    The imputation sees every OTHER compound of that weekend and every other
    weekend's practice; it never sees the hidden cell or any race. The factor is
    the leave-one-event-out one, as it would be on the day.

    Columns: the panel's, plus ``si_rate``, ``si_lo``, ``si_hi`` (race scale,
    as the screen draws it), ``si_sev`` (NaN if unscaled), ``si_unit``
    (the interval half-width at multiple 1), and ``si_multiple`` actually used.
    """
    rates1 = T.cell_rates(practice, 1)
    rows = []
    for _, c in panel.iterrows():
        e = c["event"]
        here = rates1[(rates1["event"] == e) & (rates1["C"] != c["C"])]
        else_ = rates1[(rates1["event"] != e) & (rates1["n_runs"] >= T.MIN_RUNS)]
        f = factor_by_event[e]
        si = T.stand_in_from_rates(here, else_, f, e, [c["C"]])
        if si.empty:
            rows.append({"event": e, "C": c["C"], "si_rate": np.nan, "si_lo": np.nan,
                         "si_hi": np.nan, "si_sev": np.nan, "si_mult": np.nan, "si_unit": np.nan})
            continue
        r = si.iloc[0]
        scaled = np.isfinite(r["severity"])
        mult = T.STANDIN_INTERVAL_MULTIPLE if scaled else T.POOLED_INTERVAL_MULTIPLE
        half_race = (r["hi"] - r["lo"]) / 2.0
        rows.append({"event": e, "C": c["C"], "si_rate": r["rate"], "si_lo": r["lo"],
                     "si_hi": r["hi"], "si_sev": r["severity"], "si_mult": mult,
                     "si_unit": half_race / mult})
    return panel.merge(pd.DataFrame(rows), on=["event", "C"], how="left")


def identifiability_eval(tab: pd.DataFrame, ks: Sequence[float] = (0.5, 1.0, 2.0)) -> pd.DataFrame:
    """Hard |rate| >= k*se rule against continuous empirical-Bayes shrinkage, out of sample.

    ``tab`` is a leave-one-event-out forecast frame with stand-in columns. Every
    variant answers the same cells, with the cell's own LOEO factor:

    * ``always``  -- trust the practice rate (k = 0)
    * ``rule_k``  -- if |rate| < k*se the practice rate is declared unidentified
      and replaced by the stand-in; k = 1 is what ships
    * ``eb``      -- precision-weighted blend of practice and stand-in,
      ``w = tau^2 / (tau^2 + se^2)``, tau^2 estimated by method of moments from
      the TRAINING weekends only (the variance of practice minus stand-in beyond
      what sampling error explains), floored so a bad draw cannot zero it.
    """
    out = tab.copy()
    ok = np.isfinite(out["si_rate"].to_numpy(float))
    prac = out["practice_rate"].to_numpy(float)
    se = out["se_p"].to_numpy(float)
    si = np.where(ok, out["si_rate"].to_numpy(float), prac)
    f = out["factor_used"].to_numpy(float)
    for k in ks:
        flag = ok & np.isfinite(se) & (np.abs(prac) < k * se)
        out[f"pred_rule{k:g}"] = f * np.where(flag, si, prac)
        out[f"flag_rule{k:g}"] = flag
    eb = np.full(len(out), np.nan)
    wts = np.full(len(out), np.nan)
    for e in out["event"].unique():
        m_te = (out["event"] == e).to_numpy()
        m_tr = ~m_te & ok & np.isfinite(se)
        d2 = (prac[m_tr] - si[m_tr]) ** 2
        tau2 = max(float(d2.mean() - np.mean(se[m_tr] ** 2)), 0.25 * float(d2.mean()), 1e-6) \
            if m_tr.sum() >= 4 else float("inf")
        w = tau2 / (tau2 + np.where(np.isfinite(se[m_te]), se[m_te] ** 2, 0.0))
        w = np.where(ok[m_te], w, 1.0)
        eb[m_te] = f[m_te] * (si[m_te] + w * (prac[m_te] - si[m_te]))
        wts[m_te] = w
    out["pred_eb"] = eb
    out["eb_weight"] = wts
    return out


def interval_calibration(df: pd.DataFrame, pred: str, unit: str, level: float = 0.95) -> dict:
    """Out-of-sample coverage of the race rate and the multiple that would hit ``level``.

    ``unit`` is the interval half-width at multiple 1, so shipped coverage is the
    share with ``|race - pred| <= multiple * unit``. The calibrated multiple is
    the ``level`` quantile of ``|race - pred| / unit`` over the OTHER weekends,
    and its coverage is then scored on the held-out one, so the quoted coverage
    is cross-validated, not fitted.
    """
    d = df.dropna(subset=[pred, unit]).copy()
    d = d[d[unit] > 0]
    z = (d["race_rate"] - d[pred]).abs() / d[unit]
    d = d.assign(z=z)
    hit = []
    mults = []
    for e in d["event"].unique():
        tr, te = d[d["event"] != e], d[d["event"] == e]
        if len(tr) < 5:
            continue
        m = float(np.quantile(tr["z"], level))
        mults.append(m)
        hit.extend((te["z"] <= m).astype(float))
    full = float(np.quantile(d["z"], level)) if len(d) else float("nan")
    return {
        "n": int(len(d)),
        "multiple_for_level": full,
        "cv_coverage_at_calibrated": float(np.mean(hit)) if hit else float("nan"),
        "z_median": float(d["z"].median()) if len(d) else float("nan"),
    }


# ---------------------------------------------------------------------------
# which practice session predicts the race
# ---------------------------------------------------------------------------

SESSION_MIN_RUNS = 2
SESSIONS = ("FP1", "FP2", "FP3", "S")


def session_pairs(season: int, laps: pd.DataFrame | None = None,
                  race: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every (weekend, compound) cell measurable in ONE session and in its race.

    Same construction as ``scripts/12_session_skill.paired_cells`` -- a session on
    its own, two runs minimum, one run per compound allowed -- generalised to
    label compounds so it runs on 2022-2025.
    """
    laps = load_laps(season) if laps is None else laps
    cf = compound_fn_for(season)
    race = build_race(laps, compound_fn=cf) if race is None else race
    truth = race_cell_table(_rekey(race, season), season)
    truth = truth.set_index(["event", "C"])["race_rate"]
    design = replace(SHIPPED, min_runs_per_compound=1)
    rows = []
    for ses in SESSIONS:
        if ses not in set(laps["session"]):
            continue
        sub = laps[laps["session"].isin([ses, "R"])]
        p = _rekey(build_practice(sub, design, compound_fn=cf), season)
        if p.empty:
            continue
        for _, r in T.cell_rates(p, min_runs=SESSION_MIN_RUNS).iterrows():
            key = (r["event"], r["C"])
            if key in truth.index:
                rows.append({"season": season, "session": ses, "event": r["event"], "C": r["C"],
                             "grp": r["C"], "n_runs": int(r["n_runs"]), "n_laps": int(r["n_laps"]),
                             "practice_rate": float(r["rate"]), "race_rate": float(truth.loc[key]),
                             "se_p": float(r["se"])})
    return pd.DataFrame(rows)


def single_session_loeo(pairs: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-weekend-out forecast from ONE session alone, per session.

    The factor is learned from the OTHER weekends' cells of the same session.
    This is the decision-relevant score -- what you would forecast with if you
    had only this session -- and, unlike a correlation, it carries the scale.
    """
    parts = []
    for _, g in pairs.groupby("session"):
        for e in g["event"].unique():
            tr, te = g[g["event"] != e], g[g["event"] == e].copy()
            f = ratio_factor(tr)
            te["pred_naive"] = te["practice_rate"]
            te["pred_scalar"] = f * te["practice_rate"]
            parts.append(te)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def boot_stat(df: pd.DataFrame, fn: Callable[[pd.DataFrame], float], cluster: str = "event",
              n_boot: int = 2000, seed: int = SEED) -> tuple[float, float, float]:
    """A statistic with a 95% CI from resampling whole weekends."""
    df = df.reset_index(drop=True)
    pos = [np.flatnonzero(m) for m in
           (df[cluster].to_numpy() == u for u in pd.unique(df[cluster]))]
    rng = np.random.default_rng(seed)
    point = fn(df)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(pos), len(pos))
        v = fn(df.iloc[np.concatenate([pos[i] for i in pick])])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _pearson(d: pd.DataFrame) -> float:
    if len(d) < 3 or d["practice_rate"].std() == 0 or d["race_rate"].std() == 0:
        return float("nan")
    return float(d["practice_rate"].corr(d["race_rate"]))


def session_skill(pairs: pd.DataFrame, n_boot: int = 2000) -> dict:
    """Per-session correlation and single-session MAE, with CIs, plus paired contrasts."""
    out: dict = {"per_session": {}, "contrasts": {}}
    loo = single_session_loeo(pairs)
    for ses, g in pairs.groupby("session"):
        r, lo, hi = boot_stat(g, _pearson, n_boot=n_boot)
        sp = _spearman(g["practice_rate"].to_numpy(float), g["race_rate"].to_numpy(float))
        lg = loo[loo["session"] == ses]
        cl = lg["event"].to_numpy()
        y = lg["race_rate"].to_numpy(float)
        mae_s, mlo, mhi = boot_mean_ci(np.abs(lg["pred_scalar"].to_numpy(float) - y), cl, n_boot)
        mae_n, _, _ = boot_mean_ci(np.abs(lg["pred_naive"].to_numpy(float) - y), cl, n_boot)
        out["per_session"][ses] = {
            "n_cells": int(len(g)), "n_weekends": int(g["event"].nunique()),
            "median_laps": int(g["n_laps"].median()), "pearson": r, "pearson_ci": [lo, hi],
            "spearman": sp, "loeo_mae_scalar": mae_s, "loeo_mae_scalar_ci": [mlo, mhi],
            "loeo_mae_naive": mae_n,
        }
    sessions = sorted(pairs["session"].unique())
    for i, a in enumerate(sessions):
        for b in sessions[i + 1:]:
            m = pairs[pairs["session"] == a].merge(
                pairs[pairs["session"] == b], on=["event", "C"], suffixes=("_a", "_b"))
            if len(m) < 5:
                out["contrasts"][f"{a}-{b}"] = {"n_common": int(len(m))}
                continue

            def rho_diff(d: pd.DataFrame) -> float:
                if d["practice_rate_a"].std() == 0 or d["practice_rate_b"].std() == 0:
                    return float("nan")
                return float(d["practice_rate_a"].corr(d["race_rate_a"])
                             - d["practice_rate_b"].corr(d["race_rate_b"]))

            dd, lo, hi = boot_stat(m, rho_diff, n_boot=n_boot)
            la = loo[(loo["session"] == a)].merge(m[["event", "C"]], on=["event", "C"])
            lb = loo[(loo["session"] == b)].merge(m[["event", "C"]], on=["event", "C"])
            la = la.sort_values(["event", "C"]).reset_index(drop=True)
            lb = lb.sort_values(["event", "C"]).reset_index(drop=True)
            pt = paired_test(la["pred_scalar"].to_numpy(float) - la["race_rate"].to_numpy(float),
                             lb["pred_scalar"].to_numpy(float) - lb["race_rate"].to_numpy(float),
                             la["event"].to_numpy(), n_boot=n_boot)
            out["contrasts"][f"{a}-{b}"] = {
                "n_common": int(len(m)), "n_weekends": int(m["event"].nunique()),
                "rho_a": float(m["practice_rate_a"].corr(m["race_rate_a"])),
                "rho_b": float(m["practice_rate_b"].corr(m["race_rate_b"])),
                "rho_diff": dd, "rho_diff_ci": [lo, hi],
                "mae_diff_a_minus_b": pt["mean_diff"],
                "mae_diff_ci": [pt["ci_lo"], pt["ci_hi"]],
                "mae_diff_signflip_p": pt["signflip_p"],
            }
    return out


# ---------------------------------------------------------------------------
# design and weighting sweeps, on common cells
# ---------------------------------------------------------------------------


def design_loeo(laps_by_season: dict[int, pd.DataFrame], design: PracticeDesign,
                weights: dict[str, float] | None = None,
                races: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Leave-one-weekend-out scalar forecast of one design, over stacked seasons."""
    panels = []
    for s, laps in laps_by_season.items():
        cf = compound_fn_for(s)
        pr = _rekey(build_practice(laps, design, compound_fn=cf, weights=weights), s)
        rc = races[s] if races else _rekey(build_race(laps, compound_fn=cf), s)
        if pr.empty:
            continue
        pn = cell_panel(pr, rc, s)
        if not pn.empty:
            panels.append(pn)
    if not panels:
        return pd.DataFrame()
    panel = pd.concat(panels, ignore_index=True)
    rows = []
    for e in panel["event"].unique():
        tr, te = panel[panel["event"] != e], panel[panel["event"] == e].copy()
        f = ratio_factor(tr)
        te["pred_naive"] = te["practice_rate"]
        te["pred_scalar"] = f * te["practice_rate"]
        rows.append(te)
    return pd.concat(rows, ignore_index=True)


def sweep_against_shipped(tables: dict[str, pd.DataFrame], shipped: str) -> dict:
    """Each variant against the shipped design, on the cells BOTH can answer.

    The raw MAE of a variant is not comparable with another's -- each scores
    whatever it leaves standing, and one that discards the hard cells wins while
    knowing less. So every comparison is paired on common cells, with a
    weekend-clustered CI. ``mae_common_all`` scores every variant on the cells
    ALL of them answer, which is the number the original sweep printed.
    """
    ref = tables[shipped]
    keys = {k: set(zip(t["event"], t["C"], strict=True)) for k, t in tables.items() if len(t)}
    common_all = set.intersection(*keys.values()) if keys else set()
    out = {}
    for name, t in tables.items():
        if t.empty:
            continue
        m = t.merge(ref, on=["event", "C"], suffixes=("", "_ref"))
        entry = {
            "n_cells_own": int(len(t)), "n_weekends_own": int(t["event"].nunique()),
            "mae_own": float(np.abs(t["pred_scalar"] - t["race_rate"]).mean()),
            "n_common_with_shipped": int(len(m)),
        }
        if len(m):
            pt = paired_test(m["pred_scalar"].to_numpy(float) - m["race_rate"].to_numpy(float),
                             m["pred_scalar_ref"].to_numpy(float) - m["race_rate_ref"].to_numpy(float),
                             m["event"].to_numpy())
            entry.update({
                "mae_on_common": float(np.abs(m["pred_scalar"] - m["race_rate"]).mean()),
                "mae_shipped_on_common": float(np.abs(m["pred_scalar_ref"] - m["race_rate_ref"]).mean()),
                "diff_vs_shipped": pt["mean_diff"], "diff_ci": [pt["ci_lo"], pt["ci_hi"]],
                "signflip_p": pt["signflip_p"],
            })
        if common_all:
            mask = [(e, c) in common_all for e, c in zip(t["event"], t["C"], strict=True)]
            tc = t[mask]
            entry["mae_common_all"] = float(np.abs(tc["pred_scalar"] - tc["race_rate"]).mean())
            entry["n_common_all"] = int(len(tc))
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# audit metadata
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def audit(script: str, inputs: Iterable[Path], extra: dict | None = None) -> dict:
    """Provenance for a results file: code version, inputs, seeds, library versions.

    The timestamp is the only field that differs between two runs of the same
    code on the same data; the determinism test strips ``audit`` before
    comparing.
    """
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"],
                                    capture_output=True, text=True, cwd=ROOT,
                                    timeout=10).stdout.strip())
    except Exception:  # noqa: BLE001 - provenance must never stop a run
        commit, dirty = None, None
    ins = {}
    for p in inputs:
        p = Path(p)
        if p.exists():
            ins[p.name] = {"sha256_16": _sha(p), "bytes": p.stat().st_size}
    return {
        "script": script,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty_src_or_scripts": dirty,
        "seed": SEED, "n_boot": N_BOOT,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "session_weights_shipped": dict(SESSION_WEIGHTS),
        "shipped_design": asdict(SHIPPED),
        "min_runs": T.MIN_RUNS, "min_age_spread_laps": T.MIN_AGE_SPREAD_LAPS,
        "inputs": ins, **(extra or {}),
    }


def jsonable(o):
    """Recursively make numpy / pandas scalars JSON-safe. NaN becomes null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict(orient="records"))
    return o


# ---------------------------------------------------------------------------
# the two sweeps, as functions so the scripts and the paper share one code path
# ---------------------------------------------------------------------------

HOLDOUT_SEASONS = (2022, 2023, 2024, 2025)


def _load_all(seasons: Iterable[int]) -> dict[int, pd.DataFrame]:
    return {s: load_laps(s) for s in seasons}


def parity_with_prepare(laps: pd.DataFrame) -> dict:
    """Does ``build_practice(SHIPPED)`` equal ``design.prepare(..., 'practice')``?

    The failure this guards: ``scripts/13_design_sweep.py`` printed
    "MISMATCH (2815 vs 2339 laps)" because its ``current`` variant was the
    pipeline BEFORE keep-runs, gap-bridging, the warm-up trim and the Sprint
    rule -- 2339 laps, which is exactly what ``DESIGN_GRID['legacy']`` rebuilds.
    The headline "0.0819 -> 0.0491" compared that retired pipeline with the
    shipped one on whatever cells they shared, and was not a comparison of any
    two things anyone could run any more.
    """
    shipped = D.prepare(laps, "practice")
    rebuilt = build_practice(laps, SHIPPED)
    cols = ["run_id", "LapNumber", "y", "tl", "w"]
    a = shipped.sort_values(["run_id", "LapNumber"])[cols].reset_index(drop=True)
    b = rebuilt.sort_values(["run_id", "LapNumber"])[cols].reset_index(drop=True)
    same = bool(len(a) == len(b) and a.equals(b))
    legacy = build_practice(laps, DESIGN_GRID["legacy"])
    return {"shipped_laps": int(len(shipped)), "rebuilt_laps": int(len(rebuilt)),
            "identical": same, "legacy_laps": int(len(legacy)),
            "cause_of_old_mismatch": "old build() default was the pre-keep-runs pipeline "
                                     "(== legacy variant), not the shipped one"}


def run_design_sweep(holdout: bool = True) -> dict:
    """13 practice-design variants against the shipped one, 2026 and out of season."""
    laps26 = load_laps(2026)
    out: dict = {"parity": parity_with_prepare(laps26)}
    scopes = {"2026": {2026: laps26}}
    if holdout:
        scopes["2022-2025"] = _load_all(HOLDOUT_SEASONS)
    for scope, lbs in scopes.items():
        races = {s: _rekey(build_race(l, compound_fn=compound_fn_for(s)), s) for s, l in lbs.items()}
        tables = {n: design_loeo(lbs, d, races=races) for n, d in DESIGN_GRID.items()}
        out[scope] = sweep_against_shipped(tables, "warmup1-gap2")
    return out


def run_session_analysis(n_boot: int = 2000, holdout: bool = True) -> dict:
    """Per-session skill and the weighting schemes, 2026 and out of season."""
    out: dict = {}
    p26 = session_pairs(2026)
    out["2026"] = session_skill(p26, n_boot)
    lbs26 = {2026: load_laps(2026)}
    wt = {n: design_loeo(lbs26, SHIPPED, weights=w) for n, w in WEIGHT_SCHEMES.items()}
    out["weights_2026"] = sweep_against_shipped(wt, "FP2 heavy (shipped)")
    if holdout:
        lbs = _load_all(HOLDOUT_SEASONS)
        pairs = pd.concat([session_pairs(s, l) for s, l in lbs.items()], ignore_index=True)
        out["2022-2025"] = session_skill(pairs, n_boot)
        out["2022-2025_by_season"] = {
            str(s): session_skill(pairs[pairs["season"] == s], max(500, n_boot // 4))["per_session"]
            for s in HOLDOUT_SEASONS
        }
        races = {s: _rekey(build_race(l, compound_fn=compound_fn_for(s)), s) for s, l in lbs.items()}
        wh = {n: design_loeo(lbs, SHIPPED, weights=w, races=races)
              for n, w in WEIGHT_SCHEMES.items() if n != "sprint down" and n != "sprint only"}
        out["weights_2022-2025"] = sweep_against_shipped(wh, "FP2 heavy (shipped)")
        out["weights_2022-2025_note"] = ("the two sprint schemes are omitted: 2022-2025 data "
                                         "contain no Sprint sessions, so they are identical to "
                                         "shipped / the FP1-down variant there")
    return out

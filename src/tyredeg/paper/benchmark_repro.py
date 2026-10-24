"""Python port of the Cappello & Hoegh (2025) pipeline, so we can run their model here.

WHY THIS EXISTS

``benchmark/repro.R`` was the first reproduction, but R is not installed on the
machine that builds the paper, and the script ended with ``out <- NULL;
write_csv(out, ...)``, so it never wrote a result. Everything the paper quotes
about their model has to be regenerable by ``python scripts/paper/p4_benchmark.py``,
so the pipeline is ported here: their data preparation, their Stan models (run
through cmdstanpy, unchanged apart from the Stan 2.32 array syntax migration
done by ``scripts/fetch_reference.sh``), their cross-validation scheme and their
two scoring functions.

WHAT IS AND IS NOT OURS

Their code has no licence, so it is fetched into ``benchmark/upstream/`` and
``benchmark/stan/`` locally and never committed. This module contains no copy of
it. It is a re-implementation of the *procedure* from reading ``CV_Functions.R``
(the lap filters, the fuel vector, the fold arithmetic, the CRPS call) and runs
their ``.stan`` files from disk. Everything here is skipped, not faked, when the
upstream files are absent: ``upstream_available()``.

THE ONE THING WORTH KNOWING BEFORE YOU READ THE CODE

Their ``skewt_CV`` does NOT score the draws Stan hands back. Stan's
``generated quantities`` block draws ``y_pred`` with ``normal_rng``, but the R
function overwrites it with ``rsgt(...)`` draws from the skew-t before CRPS is
computed. So for the shipped CV the predictive draws match the likelihood. The
mismatch lives in the Stan file (dead code for scoring purposes, but used by
anyone who reads ``y_pred`` straight from the fit) and in ``y_rep``, which the
paper's prediction figures use. ``p4_benchmark.py`` measures all three variants
side by side rather than asserting which one was scored.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

from ..config import BENCHMARK, CMDSTAN_PATH, PROCESSED

UPSTREAM = BENCHMARK / "upstream"
STAN_DIR = BENCHMARK / "stan"
CV_DIR = UPSTREAM / "Cross_Validation_Scripts_and_Stan_Code"
HAM_CSV = CV_DIR / "Hamilton_2025_All_Races.csv"
RESULTS_CSV_DIR = UPSTREAM / "Cross_Validation_Results"

#: Their Stan sampling settings for the one-step-ahead folds (``skewt_CV``).
CV_CHAINS = 2
CV_ITER = 10_000  # total per chain, half warm-up, as rstan's default
CV_SEED = 1234  # their seed, from CV_Functions.R
CV_ADAPT_DELTA = 0.99
CV_MAX_TREEDEPTH = 12

#: Races dropped before the 2025 season CV, from ``2025_CV.qmd``. Australia and
#: Britain were wet, Miami and Belgium have missing data, the Dutch GP is a DNF.
SEASON_DROPPED = (
    "Australian Grand Prix",
    "British Grand Prix",
    "Miami Grand Prix",
    "Dutch Grand Prix",
    "Belgian Grand Prix",
)

#: ``z_reset0`` per race, i.e. the prior mean of the post-pit pace state, as
#: hand-set in ``2025_CV.qmd``. Transcribed because it is an input to the fit.
Z_RESET0 = {
    "Chinese Grand Prix": 96.0,
    "Japanese Grand Prix": 92.0,
    "Bahrain Grand Prix": 96.5,
    "Saudi Arabian Grand Prix": 93.5,
    "Emilia Romagna Grand Prix": 79.0,
    "Monaco Grand Prix": 75.0,
    "Spanish Grand Prix": 79.0,
    "Canadian Grand Prix": 75.5,
    "Austrian Grand Prix": 69.0,
    "Hungarian Grand Prix": 80.5,
    "Italian Grand Prix": 82.0,
    "Azerbaijan Grand Prix": 104.0,
    "Singapore Grand Prix": 94.0,
    "United States Grand Prix": 98.5,
    "Mexico City Grand Prix": 81.0,
    "Sao Paulo Grand Prix": 74.5,
    "Las Vegas Grand Prix": 95.5,
    "Qatar Grand Prix": 85.5,
    "Abu Dhabi Grand Prix": 88.5,
}

#: Their compound coding, ``factor(Compound, levels = c("HARD","MEDIUM","SOFT"))``.
COMPOUND_LEVELS = ("HARD", "MEDIUM", "SOFT")


def upstream_available() -> bool:
    """True when their repo and migrated Stan models are on disk."""
    return HAM_CSV.exists() and (STAN_DIR / "full_race_1driver_and_fuel_base_t.stan").exists()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _norm_race(name: str) -> str:
    """Their CSV carries a mangled ``S?o Paulo``; config.py spells it ``Sao``."""
    if re.match(r"^S.o Paulo", name):
        return "Sao Paulo Grand Prix"
    return name


def load_all_laps(season_filter: bool = False) -> pd.DataFrame:
    """Hamilton's 2025 laps, with their ``TrackStatus`` filter applied.

    ``filter(!str_detect(TrackStatus, "4|5|6|7"))`` in R. The paper's section 2
    says cleaning was minimal; the code drops safety car, VSC and red-flag laps.
    """
    df = pd.read_csv(HAM_CSV)
    df["Race"] = df["Race"].map(_norm_race)
    df["LapTime"] = df["LapTimeSeconds"]
    ts = df["TrackStatus"].astype(str)
    df = df[~ts.str.contains("4|5|6|7")].copy()
    if season_filter:
        df = df[~df["Race"].isin(SEASON_DROPPED)]
        df = df[~((df["LapNumber"] > 58) & (df["Race"] == "Singapore Grand Prix"))]
    return df


@lru_cache(maxsize=None)
def scheduled_laps(race: str) -> int:
    """Race distance: the field's highest lap number in our own 2025 cache.

    Hamilton's own last retained lap is not the distance (a DNF, a filtered
    final lap, a pit-in lap), so it is taken from the whole field. Falls back to
    his unfiltered maximum if the parquet is absent.
    """
    path = PROCESSED / "laps_2025.parquet"
    if path.exists():
        laps = pd.read_parquet(path, columns=["event", "session", "LapNumber"])
        laps = laps[(laps["session"] == "R") & (laps["event"].map(_norm_race) == race)]
        if len(laps):
            return int(laps["LapNumber"].max())
    df = pd.read_csv(HAM_CSV, usecols=["Race", "LapNumber"])
    df["Race"] = df["Race"].map(_norm_race)
    return int(df.loc[df["Race"] == race, "LapNumber"].max())


@dataclass
class RaceData:
    """What ``load_race_data`` builds in R, plus the lap numbers it throws away."""

    race: str
    y: np.ndarray
    compound: np.ndarray  # 1 hard, 2 medium, 3 soft
    pit: np.ndarray  # 1 where the next retained row starts a new stint
    lap_number: np.ndarray  # real LapNumber of each retained row (R discards this)
    stint: np.ndarray
    fuel_theirs: np.ndarray  # seq(110, 1, length.out = n_retained)
    fuel_true: np.ndarray  # 110 -> 1 indexed to LapNumber over the race distance
    race_laps: int
    z_reset0: float = 69.0

    @property
    def n(self) -> int:
        return len(self.y)

    def fuel(self, which: str) -> np.ndarray:
        return {"theirs": self.fuel_theirs, "corrected": self.fuel_true}[which]


def true_fuel(lap_number: np.ndarray, race_laps: int, start: float = 110.0, end: float = 1.0):
    """Fuel by real lap number: ``start`` on lap 1 down to ``end`` on the last lap."""
    return start - (start - end) * (np.asarray(lap_number, float) - 1.0) / (race_laps - 1.0)


def load_race_data(
    race: str,
    laps: pd.DataFrame | None = None,
    race_laps: int | None = None,
    z_reset0: float | None = None,
) -> RaceData:
    """Their ``load_race_data``: Hamilton, retained rows only, no pit laps."""
    if laps is None:
        laps = load_all_laps()
    h = laps[
        (laps["Driver"] == "HAM")
        & (laps["Race"] == race)
        & laps["PitOutTime"].isna()
        & laps["PitInTime"].isna()
    ]
    if h.empty:
        raise ValueError(f"no retained Hamilton laps for {race!r}")
    n = len(h)
    stint = h["Stint"].to_numpy()
    pit = np.zeros(n, dtype=int)
    for i in range(1, n):
        if stint[i] != stint[i - 1]:
            pit[i - 1] = 1
    comp = np.array([COMPOUND_LEVELS.index(c) + 1 for c in h["Compound"]], dtype=int)
    lapno = h["LapNumber"].to_numpy(dtype=int)
    if race_laps is None:
        race_laps = scheduled_laps(race)
    return RaceData(
        race=race,
        y=h["LapTime"].to_numpy(dtype=float),
        compound=comp,
        pit=pit,
        lap_number=lapno,
        stint=stint.astype(int),
        fuel_theirs=np.linspace(110.0, 1.0, n),
        fuel_true=true_fuel(lapno, race_laps),
        race_laps=race_laps,
        z_reset0=Z_RESET0.get(race, 69.0) if z_reset0 is None else z_reset0,
    )


def fuel_misplacement(d: RaceData) -> dict:
    """How far their row-indexed ramp is from fuel by real lap number, in kg and seconds."""
    err = d.fuel_theirs - d.fuel_true
    return {
        "n_laps_retained": int(d.n),
        "race_laps": int(d.race_laps),
        "max_abs_kg": float(np.abs(err).max()),
        "mean_abs_kg": float(np.abs(err).mean()),
        "mean_signed_kg": float(err.mean()),
        "kg_at_first_retained_lap": float(err[0]),
        "kg_at_last_retained_lap": float(err[-1]),
    }


# ---------------------------------------------------------------------------
# Stan
# ---------------------------------------------------------------------------


def ensure_cmdstan() -> None:
    from ..models.hierarchical import ensure_toolchain

    ensure_toolchain()
    import cmdstanpy

    if CMDSTAN_PATH.exists():
        cmdstanpy.set_cmdstan_path(str(CMDSTAN_PATH))


_MODELS: dict[str, object] = {}


def model(name: str):
    """Compile (once per process) one of their Stan files from ``benchmark/stan``."""
    if name not in _MODELS:
        from cmdstanpy import CmdStanModel

        ensure_cmdstan()
        path = STAN_DIR / f"full_race_1driver_and_fuel_{name}.stan"
        _MODELS[name] = CmdStanModel(stan_file=str(path))
    return _MODELS[name]


def stan_data(d: RaceData, upto: int | None = None, fuel: str = "theirs") -> dict:
    """Data block for the compound-specific models, truncated to the first ``upto`` rows."""
    n = d.n if upto is None else upto
    comp = d.compound[:n]
    cmap = sorted(set(int(c) for c in comp))
    return {
        "TT": n,
        "y": d.y[:n].tolist(),
        "Pit": d.pit[:n].tolist(),
        "C": 3,
        "Compound": comp.tolist(),
        "C_used": len(cmap),
        "compound_map": cmap,
        "z_reset0": [69.5, 69.0, 68.5],
        "sdo0": 0.3,
        "fuel_mass": d.fuel(fuel)[:n].tolist(),
    }


def skewt_data(d: RaceData, upto: int | None = None, fuel: str = "theirs") -> dict:
    n = d.n if upto is None else upto
    return {
        "TT": n,
        "y": d.y[:n].tolist(),
        "Pit": d.pit[:n].tolist(),
        "z_reset0": float(d.z_reset0),
        "v_reset0": 0.0,
        "sdo0": 0.3,
        "fuel_mass": d.fuel(fuel)[:n].tolist(),
    }


# ---------------------------------------------------------------------------
# The skew generalised t, p = 2, q = 2 (Stan: skew_t, R: sgt::rsgt)
# ---------------------------------------------------------------------------


def _sgt_consts(sigma, lam, q):
    """Variance-adjusted scale and mean shift for p = 2, from their .stanfunctions.

    ``variance_adjusted_sgt`` makes ``sigma`` the standard deviation;
    ``mean_centered_sgt`` shifts so ``mu`` is the mean rather than the mode.
    """
    p = 2.0
    b = special.beta
    sigma_adj = sigma / (
        q ** (1.0 / p)
        * np.sqrt(
            (3 * lam**2 + 1) * (b(3.0 / p, q - 2.0 / p) / b(1.0 / p, q))
            - 4 * lam**2 * (b(2.0 / p, q - 1.0 / p) / b(1.0 / p, q)) ** 2
        )
    )
    shift = 2 * sigma_adj * lam * q ** (1.0 / p) * b(2.0 / p, q - 1.0 / p) / b(1.0 / p, q)
    return sigma_adj, shift


def sgt_logpdf(x, mu, sigma, lam, q=2.0):
    """Log density of their skew-t (p = 2), mirroring ``skew_generalized_t_lpdf``."""
    p = 2.0
    x = np.asarray(x, float)
    sigma_adj, shift = _sgt_consts(sigma, lam, q)
    r = x + shift - mu
    s = np.where(r < 0, -1.0, 1.0)
    out = np.log1p(np.abs(r) ** p / (q * sigma_adj**p * (1 + lam * s) ** p))
    return (
        np.log(p) - np.log(2.0) - np.log(sigma_adj) - np.log(q) / p - special.betaln(1.0 / p, q)
    ) - (1.0 / p + q) * out


def rsgt(n, mu, sigma, lam, q, rng: np.random.Generator):
    """Draws from the skew-t above, vectorised over parameter draws.

    The density is a two-piece Student-t centred on ``mu - shift``: the right
    half has scale ``sigma_adj (1 + lam)`` and mass ``(1 + lam) / 2``, the left
    half scale ``sigma_adj (1 - lam)`` and mass ``(1 - lam) / 2``. Each half is
    ``|T| / sqrt(2)`` with ``T ~ t(2q)`` (for p = 2: (1 + u^2 / q)^-(q + 1/2)
    against t with nu = 2q). ``test_paper_benchmark`` checks this sampler against
    ``sgt_logpdf`` by Kolmogorov-Smirnov and by its mean, because R's ``rsgt``
    is not available to compare with.

    ``mu``, ``sigma`` and ``lam`` may be arrays of length ``n`` (one per
    posterior draw), exactly as their R call passes them.
    """
    mu = np.broadcast_to(np.asarray(mu, float), (n,))
    sigma = np.broadcast_to(np.asarray(sigma, float), (n,))
    lam = np.broadcast_to(np.asarray(lam, float), (n,))
    sigma_adj, shift = _sgt_consts(sigma, lam, q)
    right = rng.random(n) < (1.0 + lam) / 2.0
    t = np.abs(rng.standard_t(2.0 * q, size=n)) / np.sqrt(2.0)
    scale = np.where(right, sigma_adj * (1.0 + lam), sigma_adj * (1.0 - lam))
    r = np.where(right, 1.0, -1.0) * scale * t
    return mu - shift + r


# ---------------------------------------------------------------------------
# Cross-validation, transcribed from CV_Functions.R
# ---------------------------------------------------------------------------


@dataclass
class Fold:
    stint: int  # 1-based
    i: int  # 1-based fold within the stint
    train_n: int  # rows the fit sees: i + end_stint[j] - K - 1
    test_row: int  # 1-based row being predicted = train_n + 1
    lap_number: int  # the real LapNumber of that row


def cv_folds(d: RaceData) -> list[Fold]:
    """Their rolling origin: last ``round(len/4)`` rows of each stint, one step ahead.

    R's ``round`` breaks halves to even, and so does ``np.round``.
    """
    end = [int(i) + 1 for i in np.flatnonzero(d.pit == 1)] + [d.n]  # 1-based, as which()
    folds = []
    for j, e in enumerate(end, start=1):
        length = e if j == 1 else e - end[j - 2]
        k = int(np.round(length / 4))
        for i in range(1, k + 1):
            train_n = i + e - k - 1
            folds.append(Fold(j, i, train_n, train_n + 1, int(d.lap_number[train_n])))
    return folds


@dataclass
class FoldFit:
    """Everything kept from one fit; draws are enough to rescore under any scheme."""

    fold: Fold
    actual: float
    z_pred: np.ndarray
    gamma: np.ndarray
    sdo: np.ndarray
    skew: np.ndarray
    v: np.ndarray
    y_pred_stan: np.ndarray  # normal_rng draw from the .stan file
    fuel_last_train: float  # fuel[TT]: what the .stan file's y_pred uses
    fuel_test: float  # fuel[TT+1]: what the R override uses
    seconds: float
    divergences: int = 0
    max_rhat: float = float("nan")
    extra: dict = field(default_factory=dict)


def fit_skewt_fold(
    d: RaceData,
    fold: Fold,
    fuel: str = "theirs",
    chains: int = CV_CHAINS,
    iter_total: int = CV_ITER,
    seed: int = CV_SEED,
    parallel_chains: int | None = None,
    threads: int | None = None,
) -> FoldFit:
    """Fit ``base_t`` on rows ``1..train_n`` and keep the quantities scoring needs."""
    m = model("base_t")
    data = skewt_data(d, upto=fold.train_n, fuel=fuel)
    t0 = time.time()
    fit = m.sample(
        data=data,
        chains=chains,
        parallel_chains=parallel_chains or chains,
        iter_warmup=iter_total // 2,
        iter_sampling=iter_total // 2,
        seed=seed,
        adapt_delta=CV_ADAPT_DELTA,
        max_treedepth=CV_MAX_TREEDEPTH,
        show_progress=False,
        show_console=False,
    )
    secs = time.time() - t0
    fv = d.fuel(fuel)
    summ = fit.summary()
    rhat = float(summ["R_hat"].dropna().max()) if "R_hat" in summ else float("nan")
    try:
        div = int(fit.method_variables()["divergent__"].sum())
    except Exception:  # noqa: BLE001
        div = 0
    return FoldFit(
        fold=fold,
        actual=float(d.y[fold.train_n]),
        z_pred=np.asarray(fit.stan_variable("z_pred")),
        gamma=np.asarray(fit.stan_variable("gamma")),
        sdo=np.asarray(fit.stan_variable("sdo")),
        skew=np.asarray(fit.stan_variable("skew")),
        v=np.asarray(fit.stan_variable("v")),
        y_pred_stan=np.asarray(fit.stan_variable("y_pred")),
        fuel_last_train=float(fv[fold.train_n - 1]),
        fuel_test=float(fv[fold.train_n]),
        seconds=secs,
        divergences=div,
        max_rhat=rhat,
    )


def predictive_draws(ff: FoldFit, kind: str, seed: int) -> np.ndarray:
    """Predictive draws for the held-out lap under one of three constructions.

    ``shipped``    what ``skewt_CV`` in CV_Functions.R scores: ``rsgt`` from the
                   fitted skew-t, centred on ``z_pred + gamma * fuel[TT + 1]``.
    ``stan_normal`` what the .stan file returns as ``y_pred``: ``normal_rng`` with
                   the fitted ``sdo``, centred on ``z_pred + gamma * fuel[TT]``.
                   This is the likelihood/forecast mismatch the paper's Stan code
                   would have if its ``y_pred`` were scored directly.
    ``normal_fixed_fuel`` the same normal draw but with the correct fuel index,
                   isolating the thinner tails from the one-lap-stale fuel.
    """
    rng = np.random.default_rng(seed)
    n = len(ff.z_pred)
    if kind == "shipped":
        return rsgt(n, ff.z_pred + ff.gamma * ff.fuel_test, ff.sdo, ff.skew, 2.0, rng)
    if kind == "stan_normal":
        return ff.y_pred_stan
    if kind == "normal_fixed_fuel":
        return rng.normal(ff.z_pred + ff.gamma * ff.fuel_test, ff.sdo)
    raise ValueError(kind)


def seed_of(*parts) -> int:
    """Deterministic seed from a tuple of small ints/strings (no ``hash()``: that is salted)."""
    s = 0
    for p in parts:
        for ch in str(p):
            s = (s * 131 + ord(ch)) % (2**31 - 1)
    return s


def n_threads_default() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


# ---------------------------------------------------------------------------
# Fitting every fold of a race, cached on disk so a long run can be resumed
# ---------------------------------------------------------------------------

FITS_DIR = BENCHMARK.parent / "data" / "processed" / "benchmark_fits"

_FF_ARRAYS = ("z_pred", "gamma", "sdo", "skew", "v", "y_pred_stan")


def _cache_path(race: str, fuel: str, iter_total: int, fold: Fold) -> Path:
    tag = f"{race.replace(' ', '_')}__{fuel}__{iter_total}"
    return FITS_DIR / tag / f"fold_s{fold.stint}_i{fold.i}.npz"


def _save_ff(path: Path, ff: FoldFit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{k: getattr(ff, k) for k in _FF_ARRAYS},
        meta=np.array(
            [
                ff.fold.stint,
                ff.fold.i,
                ff.fold.train_n,
                ff.fold.test_row,
                ff.fold.lap_number,
                ff.actual,
                ff.fuel_last_train,
                ff.fuel_test,
                ff.seconds,
                ff.divergences,
                ff.max_rhat,
            ],
            dtype=float,
        ),
    )


def load_ff(path: Path) -> FoldFit:
    z = np.load(path)
    m = z["meta"]
    fold = Fold(int(m[0]), int(m[1]), int(m[2]), int(m[3]), int(m[4]))
    return FoldFit(
        fold=fold,
        actual=float(m[5]),
        fuel_last_train=float(m[6]),
        fuel_test=float(m[7]),
        seconds=float(m[8]),
        divergences=int(m[9]),
        max_rhat=float(m[10]),
        **{k: z[k] for k in _FF_ARRAYS},
    )


def _fold_job(args):
    race, fuel, iter_total, fold, chains = args
    path = _cache_path(race, fuel, iter_total, fold)
    if path.exists():
        return str(path)
    d = load_race_data(race, laps=_JOB_LAPS_CACHE.setdefault("laps", load_all_laps(True)))
    ff = fit_skewt_fold(d, fold, fuel=fuel, chains=chains, iter_total=iter_total)
    _save_ff(path, ff)
    return str(path)


_JOB_LAPS_CACHE: dict = {}


def fit_race_folds(
    race: str,
    fuel: str = "theirs",
    iter_total: int = CV_ITER,
    workers: int = 6,
    chains: int = CV_CHAINS,
    laps: pd.DataFrame | None = None,
) -> list[FoldFit]:
    """Fit ``base_t`` on every fold of ``race``; results are cached per fold.

    Folds are independent fits, so they are spread over ``workers`` processes
    (each runs ``chains`` chains). Seeds are fixed (their 1234 for every fold,
    as in ``skewt_CV``), so a rerun reproduces the same draws on the same
    machine; the cache only makes an interrupted run resumable.
    """
    from concurrent.futures import ProcessPoolExecutor

    d = load_race_data(race, laps=laps if laps is not None else load_all_laps(True))
    folds = cv_folds(d)
    jobs = [(race, fuel, iter_total, f, chains) for f in folds]
    todo = [j for j in jobs if not _cache_path(race, fuel, iter_total, j[3]).exists()]
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_fold_job, todo))
    return [load_ff(_cache_path(race, fuel, iter_total, f)) for f in folds]


# ---------------------------------------------------------------------------
# Full-race fits: Table 3 and the fuel coefficient
# ---------------------------------------------------------------------------

#: Settings for the full-race Ext1 fit. Theirs: 3 chains x 30000 iterations at
#: adapt_delta 0.99, treedepth 12 (``Tire_Deg_Paper_Review1.qmd``). Ours is
#: 4 x 5000 post-warm-up, which is ample for a 5-parameter summary and takes
#: under a minute, with a fixed seed.
FULL_CHAINS = 4
FULL_WARMUP = 5_000
FULL_SAMPLING = 5_000
FULL_SEED = 1


def fit_full(d: RaceData, name: str = "Extension1", fuel: str = "theirs", seed: int = FULL_SEED,
             chains: int = FULL_CHAINS, warmup: int = FULL_WARMUP, sampling: int = FULL_SAMPLING):
    """Fit one of their Gaussian models (or the skew-t) on the whole race."""
    m = model(name)
    data = skewt_data(d, fuel=fuel) if name == "base_t" else stan_data(d, fuel=fuel)
    if name == "base":
        data = {k: v for k, v in data.items() if k != "z_reset0"} | {"z_reset0": 69.0, "v_reset0": 0.0}
    t0 = time.time()
    fit = m.sample(
        data=data, chains=chains, parallel_chains=chains, iter_warmup=warmup,
        iter_sampling=sampling, seed=seed, adapt_delta=CV_ADAPT_DELTA,
        max_treedepth=CV_MAX_TREEDEPTH, show_progress=False, show_console=False,
    )
    return fit, time.time() - t0


def split_rhat(x: np.ndarray) -> float:
    """Split R-hat for one parameter; ``x`` is (chains, draws)."""
    x = np.asarray(x, float)
    n = x.shape[1] // 2
    ch = np.concatenate([x[:, :n], x[:, n : 2 * n]], axis=0)
    w = ch.var(axis=1, ddof=1).mean()
    b = n * ch.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))


def summarise_ext1(fit, seconds: float) -> dict:
    """Table 3 quantities from an Extension1 fit."""
    v = np.asarray(fit.stan_variable("v_used"))  # (draws, C_used): hard, medium
    g = np.asarray(fit.stan_variable("gamma"))
    div = int(fit.method_variables()["divergent__"].sum())
    arr = fit.draws(concat_chains=False)  # (draws, chains, columns)
    cols = list(fit.column_names)
    rhat = max(
        split_rhat(arr[:, :, cols.index(nm)].T)
        for nm in ("v_used[1]", "v_used[2]", "gamma", "sdo", "sdp")
    )

    def stat(x):
        return {
            "mean": float(x.mean()),
            "lo": float(np.quantile(x, 0.025)),
            "hi": float(np.quantile(x, 0.975)),
        }

    return {
        "v_hard": stat(v[:, 0]),
        "v_medium": stat(v[:, 1]),
        "gamma": stat(g),
        "p_medium_gt_hard": float((v[:, 1] > v[:, 0]).mean()),
        "crls_overlap": bool(np.quantile(v[:, 1], 0.025) < np.quantile(v[:, 0], 0.975)),
        "diff_medium_minus_hard": stat(v[:, 1] - v[:, 0]),
        "max_rhat": rhat,
        "divergences": div,
        "seconds": round(seconds, 1),
    }

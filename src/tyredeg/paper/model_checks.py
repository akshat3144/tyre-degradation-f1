"""Diagnostics for the core degradation model: what survives scrutiny, what does not.

Everything here is deterministic (fixed seeds, spawned per task so results do
not depend on how work is scheduled across processes) and returns plain dicts
that ``scripts/paper/p3_model.py`` writes to ``paper/results/model.json``.

The sections answer the audit's questions in order:

  engine_agreement     is the fast REML engine the same model statsmodels fits?
  old_fit_reproduction what exactly was wrong with the shipped fit?
  inference_comparison how much do run-level terms and clustering change the CIs?
  circuit_effect       is the circuit effect real, how big, in seconds?
  compound_within_event does the "inverted" compound ordering survive when the
                       compound contrast is taken inside each event?
  per_race_separation  race-by-race separation, with minimum runs and direction
  rate_calibration     do the RATE intervals cover? (simulation + leave-one-event-out)
  ablation_table       naive vs practitioner vs deconfounded, like for like
  season_robustness    the same model on 2022-2025 (label compounds)

Nothing outside ``tyredeg.paper`` imports this package.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy import stats

from ..models.mixed import C_ORDER, LABEL_ORDER, fit_degradation
from ..models.reml import Comp, RemlModel, single_threaded_blas
from ..models.within import absorb, demean, offset_columns, redemean_frame
from ..validation.calibration import coverage_of_truth, predictive_coverage

Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def build_model(
    df: pd.DataFrame,
    *,
    circuit: str = "random",
    slopes: str = "compound",
    run: bool = True,
    run_slope: bool = True,
    order: tuple[str, ...] = C_ORDER,
    ref: str | None = None,
    with_offsets: bool = True,
    extra_fixed: dict[str, np.ndarray] | None = None,
    exact_within: bool = True,
) -> tuple[RemlModel, dict]:
    """The degradation model in any of the shapes the checks need.

    Args:
        circuit: ``"random"`` a random tyre-age slope per event (the shipped
            model); ``"fixed"`` one FIXED slope per event, so compound contrasts
            are identified from cars at the same event only; ``"none"``.
        slopes: ``"compound"`` one slope per compound; ``"contrast"`` the
            compounds' slopes as differences from ``ref`` (use with
            ``circuit="fixed"``); ``"label_rank"`` MEDIUM and SOFT slope
            differences from HARD; ``"trend"`` one slope per step of hardness.
        extra_fixed: optional extra FIXED-effect columns, ``{label: values}``
            with one value per row of ``df`` in its row order. They are appended
            after every other fixed effect and named ``("extra", label)``.
            Nothing is added when it is ``None`` (the default), so every
            existing specification is unchanged. Used for the Mundlak run-mean
            covariates in ``mundlak_check``. They are appended as given.
        exact_within: demean every fixed-effect column within its (event, lap)
            cell: ``tilde(D_c)`` for the offsets (the first compound is the
            reference and has no column), ``tilde(w * age)`` with RAW tyre age
            for every slope column whose weight ``w`` varies within a cell
            (compound dummies, hardness steps, label dummies). A column that is
            constant within a cell times age (per-event slopes, one common
            slope) is already exact. ``False`` is the earlier construction (raw
            dummies, ``w * tilde(age)``), biased when compounds share a lap; kept
            only for the comparison in ``model.json``. Cell means are computed on
            ``df`` itself, so any subset is transformed exactly. Needs the raw
            ``TyreLife`` and ``LapNumber`` columns.
    """
    df = df.reset_index(drop=True)
    cells = None
    if exact_within:
        df, codes, ncell = redemean_frame(df)
        cells = (codes, ncell)
    cats = [c for c in order if c in set(df["C"])]
    s1 = float(df["tl"].std())
    tl = df["tl"].to_numpy(float) / s1
    C = df["C"].to_numpy()
    cols, names = [], []

    def slope_col(w: np.ndarray) -> np.ndarray:
        """``w * age`` as the model sees it: exactly demeaned, or the old ``w * tilde(age)``."""
        if cells is None:
            return w * tl
        return demean(w * df["TyreLife"].to_numpy(float), *cells) / s1

    if with_offsets:
        if cells is None:
            for c in cats:
                cols.append((C == c).astype(float))
                names.append(("off", c))
        else:
            off, off_cats = offset_columns(C, cats, *cells)
            for j, c in enumerate(off_cats):
                cols.append(off[:, j])
                names.append(("off", c))
    events = sorted(df["event"].astype(str).unique())
    ev = df["event"].astype(str).to_numpy()
    if circuit == "fixed":
        for e in events:
            cols.append((ev == e) * tl)  # constant within a cell: already exact
            names.append(("ev_tl", e))
    if slopes == "compound":
        if circuit == "fixed":
            raise ValueError("per-compound slopes are not identified beside per-event slopes")
        for c in cats:
            cols.append(slope_col((C == c).astype(float)))
            names.append(("tl", c))
    else:
        if circuit != "fixed":
            cols.append(tl)  # one common slope: already exact
            names.append(("tl", "all"))
        if slopes == "contrast":
            ref = ref or cats[len(cats) // 2]
            for c in cats:
                if c != ref:
                    cols.append(slope_col((C == c).astype(float)))
                    names.append(("ctr", c))
        elif slopes == "label_rank":
            lab = (df["Compound"] if "Compound" in df.columns else df["C"]).to_numpy()
            for L in ("MEDIUM", "SOFT"):
                cols.append(slope_col((lab == L).astype(float)))
                names.append(("lab", L))
        elif slopes == "trend":
            mid = (len(order) - 1) / 2
            idx = np.array([order.index(c) - mid if c in order else 0.0 for c in C])
            cols.append(slope_col(idx))
            names.append(("trend", "step"))
        else:
            raise ValueError(f"unknown slopes {slopes!r}")
    if "tr" in df.columns and float(df["tr"].std() or 0.0) > 1e-9:
        cols.append(df["tr"].to_numpy(float))
        names.append(("tr", ""))
    _append_extra_fixed(cols, names, extra_fixed, len(df))
    X = np.column_stack(cols)
    comps = []
    if circuit == "random":
        comps.append(Comp("circuit_slope", ev, tl))
    comps.append(Comp("driver_event", ev + "|" + df["Driver"].astype(str).to_numpy()))
    if run:
        comps.append(Comp("run", df["run_id"].astype(str).to_numpy()))
    if run_slope:
        comps.append(Comp("run_slope", df["run_id"].astype(str).to_numpy(), tl))
    model = RemlModel(df["y"].to_numpy(float), X, ev, comps)
    return model, {"names": names, "s1": s1, "cats": cats, "exact_within": bool(exact_within)}


def _append_extra_fixed(cols: list, names: list, extra_fixed: dict | None, n: int) -> None:
    """Append ``extra_fixed`` columns to a fixed-effect design being assembled, as given.

    They are NOT demeaned again: the Mundlak run means are run means of columns
    that are already within-transformed (see ``run_mean_covariates``).
    """
    for label, v in (extra_fixed or {}).items():
        v = np.asarray(v, float)
        if v.shape != (n,):
            raise ValueError(f"extra_fixed[{label!r}] has shape {v.shape}, expected ({n},)")
        cols.append(v)
        names.append(("extra", label))


def _col(meta: dict, key: tuple) -> int:
    return meta["names"].index(key)


def lincomb(fit, meta: dict, weights: dict[tuple, float]) -> dict:
    """Estimate and SE of a linear combination of slope coefficients, per lap."""
    w = np.zeros(len(meta["names"]))
    for k, v in weights.items():
        w[_col(meta, k)] = v
    est = float(w @ fit.beta) / meta["s1"]
    se = float(np.sqrt(w @ fit.cov_beta @ w)) / meta["s1"]
    return {"est": est, "se": se, "lo": est - Z95 * se, "hi": est + Z95 * se}


def _rate_table(fit, meta: dict) -> dict[str, dict]:
    out = {}
    for c in meta["cats"]:
        if ("tl", c) in meta["names"]:
            out[c] = lincomb(fit, meta, {("tl", c): 1.0})
    return out


def _r(x, k=5):
    return None if x is None else round(float(x), k)


def _iv(d: dict) -> dict:
    return {"mean": _r(d["est"]), "lo": _r(d["lo"]), "hi": _r(d["hi"])}


# ---------------------------------------------------------------------------
# multiprocessing helpers
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(df, spec, full_fit_payload):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    warnings.filterwarnings("ignore")
    _W["df"] = df
    if spec.get("nested"):
        _W["model"], _W["meta"] = build_nested_model(df, **{k: v for k, v in spec.items() if k != "nested"})
    else:
        _W["model"], _W["meta"] = build_model(df, **spec)
    _W["fit"] = full_fit_payload


def _fit_payload(fit) -> dict:
    return {"beta": fit.beta, "sigma2": fit.sigma2, "sd": fit.sd, "theta": fit.theta}


def _task_alt(seed_seq):
    """Simulate from the fitted model, refit, report slopes and the circuit SD."""
    m, meta, p = _W["model"], _W["meta"], _W["fit"]
    rng = np.random.default_rng(seed_seq)
    m.set_y(m.simulate(p["beta"], p["sigma2"], p["sd"], rng))
    f = m.fit(starts=[p["theta"]])
    return {
        "beta": f.beta, "se": f.se, "sd": f.sd, "sigma2": f.sigma2,
        "cov": f.cov_beta, "converged": f.converged,
    }


def _task_null(seed_seq):
    """Simulate from the NULL model (one component removed), return the REML LR."""
    m, p = _W["model"], _W["fit"]
    nc = p.get("null_comp", "circuit_slope")
    rng = np.random.default_rng(seed_seq)
    sd0 = dict(p["sd"])
    sd0[nc] = 0.0
    m.set_y(m.simulate(p["beta0"], p["sigma20"], sd0, rng))
    free0 = [n for n in m.names if n != nc]
    th0 = np.array([p["theta0"][m.names.index(n)] for n in free0])
    f0 = m.fit(free=free0, starts=[th0])
    th1 = p["theta0"].copy()
    th1[m.names.index(nc)] = 0.1
    f1 = m.fit(starts=[th1, p["theta"]])
    return max(0.0, f0.deviance - f1.deviance)


def _task_loeo(event):
    """Pooled fit with one event removed; returns slopes, SEs, circuit SD."""
    df = _W["df"]
    sub = df[df["event"] != event]
    spec = _W["spec"]
    m, meta = build_model(sub, **spec)
    th = _W["fit"]["theta"]
    f = m.fit(starts=[th])
    rt = _rate_table(f, meta)
    return event, {c: v for c, v in rt.items()}, f.sd["circuit_slope"] / meta["s1"], f.converged


def _task_event_boot(seed_seq):
    """Nonparametric bootstrap over EVENTS: redraw the races with replacement, refit.

    A race drawn twice becomes two blocks (its own slope each), so a repeated
    race does not masquerade as one cluster. Runs travel with their race. The
    resample can lose a compound that few races carry; those are flagged by
    ``n_compounds`` and handled by the caller.
    """
    df, spec = _W["df"], _W["spec"]
    rng = np.random.default_rng(seed_seq)
    events = sorted(df["event"].astype(str).unique())
    draw = rng.integers(0, len(events), len(events))
    parts = []
    for k, i in enumerate(draw):
        d = df[df["event"].astype(str) == events[i]].copy()
        tag = f"{events[i]}#{k}"
        d["event"] = tag
        d["run_id"] = tag + "|" + d["run_id"].astype(str)
        parts.append(d)
    b = pd.concat(parts, ignore_index=True)
    m, meta = build_model(b, **spec)
    f = m.fit(starts=[_W["fit"]["theta"]])
    ridx = [_col(meta, ("tl", c)) for c in meta["cats"]]
    s1 = meta["s1"]
    return {
        "n_compounds": len(meta["cats"]),
        "rate": np.array([f.beta[i] / s1 for i in ridx]),
        "se": np.array([np.sqrt(f.cov_beta[i, i]) / s1 for i in ridx]),
        "circuit_sd": f.sd["circuit_slope"] / s1,
        "converged": bool(f.converged),
    }


def _task_event_fit(args):
    """Per-event fit: compounds' slopes with run-level inference."""
    event, min_runs, run_level = args
    df = _W["df"]
    d = df[df["event"] == event]
    runs = d.groupby("C", observed=True)["run_id"].nunique()
    keep = runs[runs >= min_runs].index
    d = d[d["C"].isin(keep)]
    if d["C"].nunique() == 0:
        return event, min_runs, run_level, {}, []
    spec = dict(_W["spec"])
    spec.update(circuit="none", slopes="compound", run=run_level, run_slope=run_level)
    m, meta = build_model(d, **spec)
    f = m.fit(starts=[np.full(len(m.names), 0.5), np.full(len(m.names), 0.1)])
    rt = _rate_table(f, meta)
    names = meta["names"]
    cov = {}
    idx = [i for i, k in enumerate(names) if k[0] == "tl"]
    sub = f.cov_beta[np.ix_(idx, idx)] / meta["s1"] ** 2
    comp = [names[i][1] for i in idx]
    for c in rt:
        rt[c]["n_runs"] = int(runs[c])
        rt[c]["n_laps"] = int((d["C"] == c).sum())
    return event, min_runs, run_level, rt, {"comp": comp, "cov": sub.tolist()}


def _single_thread_children() -> None:
    """Children inherit these at spawn, before numpy is imported in them.

    Ten workers each running a multi-threaded BLAS on 100-column matrices spend
    their time fighting for cores; one thread each is several times faster.
    """
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


def _reuse_results(fn_name: str, items: list, df, spec, payload) -> dict:
    """Results already computed for identical items, from earlier shard caches.

    Only caches built from the same data, specification and payload qualify;
    within those, an item matches by its pickled bytes (same arguments and
    seeds). Every ``*.partial`` and ``out*.pkl`` file is read, whichever
    machine wrote it.
    """
    import hashlib
    import pickle

    from tyredeg.config import PROCESSED

    def h(x):
        return hashlib.sha1(pickle.dumps(x)).hexdigest()

    context = h((df, spec, payload))
    want = {h(it): i for i, it in enumerate(items)}
    found: dict = {}
    for sib in sorted((PROCESSED / "shard_cache").glob(f"{fn_name}_*")):
        job_path = sib / "job.pkl"
        if not job_path.exists():
            continue
        job = pickle.loads(job_path.read_bytes())
        if h((job["df"], job["spec"], job["payload"])) != context:
            continue
        old = job["items"]
        for f in list(sib.glob("*.partial")) + list(sib.glob("out*.pkl")):
            try:
                with open(f, "rb") as fh:
                    pairs = pickle.load(fh)
            except Exception:
                continue
            for j, r in pairs:
                k = h(old[j])
                if k in want:
                    found[want[k]] = r
    return found


def _run_sharded(df, spec, payload, fn, items, workers: int, retries: int = 3,
                 reuse: bool = True) -> list:
    """Run ``fn`` over ``items`` in ``workers`` separate interpreters, in order.

    Each shard handles every ``workers``-th item (a cheap, even balance) and
    writes its results to a pickle. A shard that exits badly -- or is killed
    from outside, which does happen on this shared machine -- is simply run
    again; tasks are pure functions of their seeded arguments, so a re-run
    changes nothing.

    The shard directory is keyed by a hash of the job and kept under
    ``data/processed/shard_cache/`` until the results are collected. Workers
    save progress item by item, so a run killed hours in -- the power study
    once lost three hours that way -- resumes where it stopped instead of
    starting over.
    """
    import hashlib
    import pickle
    import shutil
    import subprocess
    import sys

    from tyredeg.config import PROCESSED

    items = list(items)
    if reuse:
        # Items are pure functions of their seeded arguments, so a result
        # computed for an identical item in an earlier job -- an interrupted
        # run, or a share run on another machine -- is the same result. Match
        # on the pickled item (same data, same seeds) and only run the rest.
        known = _reuse_results(fn.__name__, items, df, spec, payload)
        missing = [i for i in range(len(items)) if i not in known]
        if known:
            print(f"  [{fn.__name__}] reusing {len(known)} of {len(items)} finished items", flush=True)
            if missing:
                fresh = _run_sharded(df, spec, payload, fn, [items[i] for i in missing], workers,
                                     retries, reuse=False)
                known.update(zip(missing, fresh, strict=True))
            return [known[i] for i in range(len(items))]
    n = max(1, min(workers, len(items)))
    _single_thread_children()
    blob = pickle.dumps({"df": df, "spec": spec, "payload": payload, "fn": fn.__name__,
                         "items": items})
    tmp = PROCESSED / "shard_cache" / f"{fn.__name__}_{n}_{hashlib.sha1(blob).hexdigest()[:16]}"
    tmp.mkdir(parents=True, exist_ok=True)
    job = tmp / "job.pkl"
    if not job.exists():
        job.write_bytes(blob)
    results: dict[int, object] = {}
    pending = list(range(n))
    t_start = time.time()
    for attempt in range(retries + 1):
        procs = {}
        print(f"  [{fn.__name__}] attempt {attempt}: launching shards {pending} "
              f"of {n} for {len(items)} items", flush=True)
        for k in pending:
            out = tmp / f"out{k}.pkl"
            errf = open(tmp / f"err{k}.txt", "wb")
            procs[k] = (subprocess.Popen(
                [sys.executable, "-m", "tyredeg.paper._worker", str(job), str(k), str(n), str(out)],
                stdout=subprocess.DEVNULL, stderr=errf), out, errf)
        failed = []
        for k, (pr, out, errf) in procs.items():
            pr.wait()
            errf.close()
            if pr.returncode == 0 and out.exists():
                with open(out, "rb") as fh:
                    results.update(dict(pickle.load(fh)))
                print(f"  [{fn.__name__}] shard {k} done at {time.time() - t_start:.0f}s", flush=True)
            else:
                tail = (tmp / f"err{k}.txt").read_text(errors="replace")[-400:]
                print(f"  [{fn.__name__}] shard {k} FAILED rc={pr.returncode}: {tail}", flush=True)
                failed.append(k)
        pending = failed
        if not pending:
            break
    else:
        raise RuntimeError(f"shards {pending} kept failing for {fn.__name__}")
    # Only a complete, collected run deletes its cache; a killed one resumes.
    shutil.rmtree(tmp, ignore_errors=True)
    return [results[i] for i in range(len(items))]


def _init_worker_spec(df, spec, payload):
    _init_worker(df, spec, payload)
    _W["spec"] = spec


# ---------------------------------------------------------------------------
# 1. engine agreement and the old, non-converged fit
# ---------------------------------------------------------------------------


def engine_agreement(design: pd.DataFrame) -> dict:
    """The REML engine against statsmodels on the same specification.

    Specification: circuit slope + driver-in-event + run intercept (statsmodels
    cannot fit the run slope in reasonable time, which is why the engine
    exists). Reported: largest absolute differences in every slope, its SE and
    every variance component.
    """
    t = time.time()
    a = fit_degradation(design, quadratic=False, circuit_effects=True, run_slopes=False)
    t_eng = time.time() - t
    t = time.time()
    b = fit_degradation(
        design, quadratic=False, circuit_effects=True, run_slopes=False, engine="statsmodels"
    )
    t_sm = time.time() - t
    diff_rate = max(abs(a.rates[c].mean - b.rates[c].mean) for c in a.rates)
    diff_se = max(
        abs((a.rates[c].hi - a.rates[c].lo) - (b.rates[c].hi - b.rates[c].lo)) / 3.92 for c in a.rates
    )
    diff_var = {
        k: abs(a.variance[k] - b.variance[k]) for k in ("circuit_slope", "driver_event", "run", "resid")
    }
    return {
        "specification": "circuit random slope + driver-in-event + run intercept",
        "engine_seconds": round(t_eng, 2),
        "statsmodels_seconds": round(t_sm, 2),
        "statsmodels_converged": b.converged,
        "statsmodels_attempts": b.convergence.get("attempts"),
        "engine_converged": a.converged,
        "max_abs_rate_difference": diff_rate,
        "max_abs_se_difference": diff_se,
        "max_abs_variance_sd_difference": diff_var,
        "rates_engine": {c: _r(a.rates[c].mean, 6) for c in a.rates},
        "rates_statsmodels": {c: _r(b.rates[c].mean, 6) for c in b.rates},
    }


def old_fit_reproduction(design: pd.DataFrame) -> dict:
    """Re-run the SHIPPED fit as it was, to document exactly what failed.

    statsmodels L-BFGS on unscaled tyre age with driver-in-event intercepts and
    a circuit slope: ``converged=False``, boundary and Hessian warnings.
    Then the same model on rescaled tyre age and on three optimisers.
    """
    import statsmodels.formula.api as smf

    d = design.reset_index(drop=True).copy()
    d["C"] = pd.Categorical(d["C"], [c for c in C_ORDER if c in set(d["C"])])
    d["drv"] = d["Driver"].astype(str)
    s1 = float(d["tl"].std())
    d["tls"] = d["tl"] / s1
    out = {}
    for label, var, f in (("unscaled_lbfgs", "tl", 1.0), ("rescaled_lbfgs", "tls", 1 / s1)):
        form = f"y ~ 0 + C + C:{var} + tr"
        m = smf.mixedlm(form, d, groups=d["event"], re_formula=f"~0 + {var}",
                        vc_formula={"driver": "0 + drv"})
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = m.fit(method="lbfgs", reml=True)
        msgs = sorted({str(x.message)[:90] for x in w})
        out[label] = {
            "converged": bool(r.converged),
            "circuit_slope_sd_s_per_lap": float(np.sqrt(np.asarray(r.cov_re).ravel()[0])) * f,
            "reml_loglik_comparable_within_row_only": float(r.llf),
            "warnings": msgs,
        }
    out["note"] = (
        "Same model, same data. The unscaled L-BFGS fit stops early with a flagged "
        "non-converged, non-positive-definite result; the rescaled fit converges to a "
        "circuit-slope SD far smaller."
    )
    return out


# ---------------------------------------------------------------------------
# 2. inference: how much do run-level terms and clustering widen the intervals?
# ---------------------------------------------------------------------------


def _fe_ols(design: pd.DataFrame, cluster_col: str, exact_within: bool = True) -> dict:
    """Fixed-effects OLS (driver-in-event and (event, lap) cell effects) with cluster-robust CIs.

    ``exact_within``: every column (compound dummies, compound x RAW tyre age,
    traffic, lap time) has BOTH the (event, lap) cell means and the
    driver-in-event means removed, by alternating projections, which is the
    exact two-way within-transformation. ``False`` is the earlier version: the
    columns ``D_c`` and ``D_c * tilde(age)`` demeaned by driver-in-event only.
    """
    from ..models.ablation import cluster_ols

    d = design.reset_index(drop=True)
    cats = [c for c in C_ORDER if c in set(d["C"])]
    grp = (d["event"].astype(str) + "|" + d["Driver"].astype(str)).to_numpy()
    if exact_within:
        d, codes, ncell = redemean_frame(d)
        cell = np.asarray(codes)
        age = d["TyreLife"].to_numpy(float)
        raw = [(d["C"] == c).to_numpy(float) for c in cats[1:]]
        raw += [(d["C"] == c).to_numpy(float) * age for c in cats]
        raw.append(d["tr"].to_numpy(float))
        raw.append(d["y"].to_numpy(float))
        Z = absorb(np.column_stack(raw), [cell, grp])
        Xd, yd = Z[:, :-1], Z[:, -1]
        j0 = len(cats) - 1
    else:
        cols = [(d["C"] == c).to_numpy(float) for c in cats]
        cols += [(d["C"] == c).to_numpy(float) * d["tl"].to_numpy(float) for c in cats]
        cols.append(d["tr"].to_numpy(float))
        X = np.column_stack(cols)
        Xd = X - pd.DataFrame(X).groupby(grp).transform("mean").to_numpy()
        yd = d["y"].to_numpy(float)
        yd = yd - pd.Series(yd).groupby(grp).transform("mean").to_numpy()
        j0 = len(cats)
    beta, cc, _ = cluster_ols(Xd, yd, d[cluster_col].astype(str))
    G = d[cluster_col].nunique()
    crit = float(stats.t.ppf(0.975, G - 1)) if cluster_col == "event" else Z95
    out = {}
    for i, c in enumerate(cats):
        j = j0 + i
        se = math.sqrt(cc[j, j])
        out[c] = {"est": float(beta[j]), "lo": float(beta[j] - crit * se), "hi": float(beta[j] + crit * se)}
    return out


def inference_comparison(design: pd.DataFrame) -> dict:
    """Compound-rate intervals under increasingly honest treatment of dependence."""
    rows = {}
    specs = {
        "A_lap_level_no_circuit": dict(circuit="none", run=False, run_slope=False),
        "B_lap_level_circuit": dict(circuit="random", run=False, run_slope=False),
        "C_run_intercept": dict(circuit="random", run=True, run_slope=False),
        "D_run_intercept_and_slope_SHIPPED": dict(circuit="random", run=True, run_slope=True),
    }
    for name, kw in specs.items():
        m, meta = build_model(design, slopes="compound", **kw)
        k = len(m.names)
        with single_threaded_blas():
            f = m.fit(starts=[np.full(k, 0.5), np.full(k, 0.1)])
        rt = _rate_table(f, meta)
        rows[name] = {c: {**_iv(v), "width": _r(v["hi"] - v["lo"])} for c, v in rt.items()}
        rows[name]["_sd"] = {n: _r(f.sd[n] / (meta["s1"] if "slope" in n else 1.0)) for n in m.names}
        rows[name]["_deviance"] = _r(f.deviance, 2)
    for label, col in (("E_FE_OLS_cluster_by_run", "run_id"), ("F_FE_OLS_cluster_by_event_t12", "event")):
        rt = _fe_ols(design, col)
        rows[label] = {
            c: {"mean": _r(v["est"]), "lo": _r(v["lo"]), "hi": _r(v["hi"]), "width": _r(v["hi"] - v["lo"])}
            for c, v in rt.items()
        }
    base = rows["B_lap_level_circuit"]
    ratio = {
        k: {c: _r(rows[k][c]["width"] / base[c]["width"], 3) for c in base if not c.startswith("_")}
        for k in rows
        if not k.startswith("B_")
    }
    # Residual serial correlation after each model: the reason the run terms exist.
    acf = {}
    for name in ("B_lap_level_circuit", "C_run_intercept", "D_run_intercept_and_slope_SHIPPED"):
        kw = specs[name]
        m, meta = build_model(design, slopes="compound", **kw)
        k = len(m.names)
        with single_threaded_blas():
            f = m.fit(starts=[np.full(k, 0.5), np.full(k, 0.1)])
            r = m.conditional_residuals(f)
        d = design.reset_index(drop=True).assign(_r=r).sort_values(["run_id", "LapNumber"])
        num = den = 0.0
        for _, g in d.groupby("run_id"):
            x = g["_r"].to_numpy()
            if len(x) > 2:
                num += float(x[1:] @ x[:-1])
                den += float(x @ x)
        acf[name] = _r(num / den, 3)
    return {
        "rates": rows,
        "width_ratio_vs_B": ratio,
        "residual_lag1_autocorrelation_within_runs": acf,
        "reading": (
            "Widths relative to the lap-level circuit model (B). Run-level terms and "
            "cluster-robust errors answer 'how many independent observations are there'."
        ),
    }


# ---------------------------------------------------------------------------
# 3. the circuit effect
# ---------------------------------------------------------------------------


def _profile_ci(model: RemlModel, fit, s1: float, grid_mult=(0.0, 0.25, 0.5, 0.75, 1.0, 1.35, 1.75, 2.3, 3.0),
                name: str = "circuit_slope") -> dict:
    """Profile-likelihood interval for the circuit-slope SD (per lap)."""
    sd_hat = fit.sd[name]
    grid = np.array([m * sd_hat for m in grid_mult])
    with single_threaded_blas():
        dev = model.profile_sd(name, grid, start=(fit.theta, float(np.sqrt(fit.sigma2))), passes=2)
    dev = dev - fit.deviance
    cut = float(stats.chi2.ppf(0.95, 1))
    lo = hi = None
    i0 = int(np.argmin(np.abs(grid - sd_hat)))
    for i in range(i0, 0, -1):  # downward
        if dev[i - 1] > cut:
            w = (cut - dev[i - 1]) / (dev[i] - dev[i - 1])
            lo = grid[i - 1] + w * (grid[i] - grid[i - 1])
            break
    else:
        lo = 0.0
    for i in range(i0, len(grid) - 1):
        if dev[i + 1] > cut:
            w = (cut - dev[i]) / (dev[i + 1] - dev[i])
            hi = grid[i] + w * (grid[i + 1] - grid[i])
            break
    return {
        "lo": _r(lo / s1), "hi": _r(hi / s1) if hi is not None else None,
        "grid_sd_per_lap": [_r(g / s1) for g in grid], "delta_deviance": [_r(x, 3) for x in dev],
    }


def circuit_alts(design: pd.DataFrame, b_null: int, b_alt: int, workers: int, seed: int) -> tuple:
    """The parametric-bootstrap refits ``circuit_effect`` ran, reproduced exactly.

    Same model, same starts, same spawned seeds, so the replicates are identical
    to the ones behind ``circuit_effect`` without redoing its null-model fits.
    Lets later analyses reuse them when only the new sections are being run.
    """
    m, meta = build_model(design, circuit="random", slopes="compound")
    k = len(m.names)
    with single_threaded_blas():
        full = m.fit(starts=[np.full(k, 0.5), np.full(k, 0.1), np.linspace(2, 0.2, k)])
    seeds = np.random.SeedSequence(seed).spawn(b_null + b_alt)
    alts = _run_sharded(design, dict(circuit="random", slopes="compound"), _fit_payload(full),
                        _task_alt, seeds[b_null:], workers)
    return alts, full, meta


def circuit_effect(design: pd.DataFrame, B_null: int, B_alt: int, workers: int, seed: int) -> tuple:
    """Is the circuit random slope real, and how big is it in seconds?

    Three tests, because the null sits on the boundary of the parameter space
    (a variance cannot be negative), which breaks the usual chi-squared
    reference:

      * parametric-bootstrap REML likelihood ratio, simulating from the fitted
        NULL model (no circuit slope, run and driver terms kept);
      * the 50:50 chi-bar-squared mixture, the standard boundary correction;
      * the profile likelihood interval for the SD, and a percentile interval
        from simulating from the fitted model and refitting.

    The size of the effect is the SD of the circuit slopes in s/lap, converted
    to seconds over a 20-lap stint, set beside compound differences on the same
    scale. It is NOT the range of the shrunken per-circuit predictions.
    """
    m, meta = build_model(design, circuit="random", slopes="compound")
    k = len(m.names)
    with single_threaded_blas():
        full = m.fit(starts=[np.full(k, 0.5), np.full(k, 0.1), np.linspace(2, 0.2, k)])
        free0 = [n for n in m.names if n != "circuit_slope"]
        null = m.fit(free=free0, starts=[np.full(len(free0), 0.5), np.full(len(free0), 0.1)])
    lr_obs = float(null.deviance - full.deviance)
    s1 = meta["s1"]
    sd_hat = full.sd["circuit_slope"] / s1

    payload = {
        **_fit_payload(full),
        "beta0": null.beta, "sigma20": null.sigma2, "theta0": null.theta,
    }
    spec = dict(circuit="random", slopes="compound")
    ss = np.random.SeedSequence(seed)
    seeds = ss.spawn(B_null + B_alt)
    t = time.time()
    lrs = np.array(_run_sharded(design, spec, payload, _task_null, seeds[:B_null], workers))
    alts = _run_sharded(design, spec, payload, _task_alt, seeds[B_null:], workers)
    boot_seconds = time.time() - t

    sd_alt = np.array([a["sd"]["circuit_slope"] / s1 for a in alts])
    p_boot = float((1 + (lrs >= lr_obs).sum()) / (1 + len(lrs)))
    p_mix = float(0.5 * stats.chi2.sf(lr_obs, 1))
    prof = _profile_ci(m, full, s1)

    # Effect in seconds over a 20-lap stint.
    H = 20
    rt = _rate_table(full, meta)
    names = meta["names"]
    comp_secs = {}
    cats = meta["cats"]
    for a in cats:
        for b in cats:
            if C_ORDER.index(a) < C_ORDER.index(b):
                lc = lincomb(full, meta, {("tl", b): 1.0, ("tl", a): -1.0})
                comp_secs[f"{b}-{a}"] = {
                    "s_per_lap": _iv(lc),
                    "seconds_over_20_laps": {k_: _r(lc[k_] * H, 3) for k_ in ("est", "lo", "hi")},
                }
    rate_sd_across = float(np.std([v["est"] for v in rt.values()], ddof=1))
    sd_ci_boot = [float(np.percentile(sd_alt, 2.5)), float(np.percentile(sd_alt, 97.5))]
    blups = m.blups(full, "circuit_slope")
    shrunk = {e: float(next(iter(v.values())) / s1) for e, v in blups.items()}
    rng_shrunk = max(shrunk.values()) - min(shrunk.values())
    adj = [f"{cats[i + 1]}-{cats[i]}" for i in range(len(cats) - 1)]
    result = {
        "model": "random circuit slope + driver-in-event + run intercept + run slope; REML; "
                 "null = same without the circuit slope",
        "circuit_slope_sd_s_per_lap": {
            "estimate": _r(sd_hat),
            "profile_likelihood_95": [prof["lo"], prof["hi"]],
            "parametric_bootstrap_percentile_95": [_r(sd_ci_boot[0]), _r(sd_ci_boot[1])],
            "bootstrap_mean": _r(sd_alt.mean()), "n_boot": int(len(sd_alt)),
        },
        "profile_grid": {k_: prof[k_] for k_ in ("grid_sd_per_lap", "delta_deviance")},
        "lr_test": {
            "reml_lr_statistic": _r(lr_obs, 3),
            "p_parametric_bootstrap": _r(p_boot, 5),
            "n_bootstrap": int(len(lrs)),
            "p_floor_note": "with B replicates the smallest attainable p is 1/(B+1)",
            "p_chibar2_50_50_mixture": float(f"{p_mix:.3g}"),
            "bootstrap_lr_quantiles_95_99": [_r(np.percentile(lrs, 95), 3), _r(np.percentile(lrs, 99), 3)],
            "null_sd_estimates_on_boundary_share": _r(float((lrs < 1e-9).mean()), 3),
        },
        "effect_in_seconds_over_20_lap_stint": {
            "circuit_sd": {
                "seconds": _r(sd_hat * H, 3),
                "profile_95": [_r(prof["lo"] * H, 3), _r(prof["hi"] * H, 3) if prof["hi"] else None],
            },
            "circuit_central_95_range_seconds": _r(2 * Z95 * sd_hat * H, 3),
            "sd_of_the_five_compound_rates_s_per_lap": _r(rate_sd_across),
            "sd_of_compound_rates_seconds": _r(rate_sd_across * H, 3),
            "compound_differences": comp_secs,
            "adjacent_compound_pairs": adj,
        },
        "shrunken_predictions_not_an_effect_size": {
            "range_s_per_lap": _r(rng_shrunk, 4),
            "per_event": {e: _r(v, 4) for e, v in sorted(shrunk.items())},
            "why": "predicted random effects are shrunk toward zero; their range neither "
                   "estimates nor bounds the true circuit SD",
        },
        "bootstrap_seconds": round(boot_seconds, 1),
        "converged_full_fit": bool(full.converged),
        "fit_on_boundary": full.on_boundary,
    }
    return result, alts, full, meta


def compound_within_event(design: pd.DataFrame) -> dict:
    """Does the physically odd compound ordering survive a within-event contrast?

    Pirelli gives harder compounds to harsher circuits, so a pooled compound
    comparison mixes compound with circuit. With one FIXED slope per event the
    compound contrasts are identified from cars on different compounds at the
    SAME event only. ``trend`` is the slope change per step of softness: the
    physical expectation is positive (softer degrades faster); the shipped pooled
    ordering is the opposite sign.
    """
    out = {}
    for tag, kw in (
        ("pooled_random_circuit", dict(circuit="random")),
        ("within_event_fixed_circuit", dict(circuit="fixed")),
    ):
        res = {}
        m, meta = build_model(design, slopes="trend", **kw)
        k = len(m.names)
        with single_threaded_blas():
            f = m.fit(starts=[np.full(k, 0.5), np.full(k, 0.1)])
        t = lincomb(f, meta, {("trend", "step"): 1.0})
        res["trend_s_per_lap_per_step_softer"] = {**_iv(t), "p_two_sided": float(f"{2 * stats.norm.sf(abs(t['est'] / t['se'])):.3g}")}
        # Per-compound contrasts against C3.
        if kw["circuit"] == "fixed":
            m2, meta2 = build_model(design, slopes="contrast", ref="C3", **kw)
            k2 = len(m2.names)
            with single_threaded_blas():
                f2 = m2.fit(starts=[np.full(k2, 0.5), np.full(k2, 0.1)])
            res["contrast_vs_C3_s_per_lap"] = {
                c: _iv(lincomb(f2, meta2, {("ctr", c): 1.0})) for c in meta2["cats"] if c != "C3"
            }
        else:
            m2, meta2 = build_model(design, slopes="compound", **kw)
            k2 = len(m2.names)
            with single_threaded_blas():
                f2 = m2.fit(starts=[np.full(k2, 0.5), np.full(k2, 0.1)])
            res["contrast_vs_C3_s_per_lap"] = {
                c: _iv(lincomb(f2, meta2, {("tl", c): 1.0, ("tl", "C3"): -1.0}))
                for c in meta2["cats"] if c != "C3"
            }
        # Label-rank contrasts: HARD < MEDIUM < SOFT inside the same weekend.
        m3, meta3 = build_model(design, slopes="label_rank", **kw)
        k3 = len(m3.names)
        with single_threaded_blas():
            f3 = m3.fit(starts=[np.full(k3, 0.5), np.full(k3, 0.1)])
        res["label_rank_contrast_s_per_lap"] = {
            "MEDIUM_minus_HARD": _iv(lincomb(f3, meta3, {("lab", "MEDIUM"): 1.0})),
            "SOFT_minus_HARD": _iv(lincomb(f3, meta3, {("lab", "SOFT"): 1.0})),
            "SOFT_minus_MEDIUM": _iv(lincomb(f3, meta3, {("lab", "SOFT"): 1.0, ("lab", "MEDIUM"): -1.0})),
        }
        out[tag] = res
    pooled = out["pooled_random_circuit"]["trend_s_per_lap_per_step_softer"]
    within = out["within_event_fixed_circuit"]["trend_s_per_lap_per_step_softer"]
    out["reading"] = {
        "expected_sign_if_physics_holds": "positive",
        "pooled_trend_sign": "positive" if pooled["mean"] > 0 else "negative",
        "within_event_trend_sign": "positive" if within["mean"] > 0 else "negative",
        "within_event_interval_excludes_zero": bool(within["lo"] > 0 or within["hi"] < 0),
    }
    return out


# ---------------------------------------------------------------------------
# 4. race-by-race separation
# ---------------------------------------------------------------------------

BENCHMARK_WIDTH = 0.129  # Cappello & Hoegh Table 3, Hard: [0.004, 0.133]


def per_race_separation(design: pd.DataFrame, workers: int, min_runs_grid=(1, 3, 5, 8)) -> dict:
    """Fit each event on its own and count separated adjacent compound pairs.

    Two inference protocols, because the audit's point is that the old one is
    too narrow: ``lap_level`` (driver intercept only, every lap independent --
    what the 4.4x / 11-of-25 claim used) and ``run_level`` (run intercept and
    slope). For each, a pair counts as separated by the non-overlap rule used
    before AND by the direct test of the difference (using the covariance of the
    two estimates); direction is "correct" when the softer compound degrades
    faster. Counts are reported at several minimum-runs-per-cell thresholds and
    for every direction, so no choice is hidden.
    """
    events = sorted(design["event"].unique())
    jobs = [(e, mr, rl) for e in events for mr in min_runs_grid for rl in (False, True)]
    res = _run_sharded(design, dict(circuit="none", slopes="compound"), {"theta": None},
                       _task_event_fit, jobs, workers)

    table: dict = {}
    for event, mr, rl, rt, covd in res:
        table.setdefault((mr, rl), {})[event] = (rt, covd)

    out = {"benchmark_width_s_per_lap": BENCHMARK_WIDTH, "protocols": {}}
    cells_dump = []
    for (mr, rl), per_event in sorted(table.items()):
        n_cells = tighter = 0
        ratios = []
        n_pairs = sep_any = sep_ok = sep_wrong = sep_diff_any = sep_diff_ok = 0
        for event, (rt, covd) in per_event.items():
            comps = [c for c in C_ORDER if c in rt]
            for c in comps:
                w = rt[c]["hi"] - rt[c]["lo"]
                n_cells += 1
                if w < BENCHMARK_WIDTH:
                    tighter += 1
                ratios.append(BENCHMARK_WIDTH / w)
                if mr == 3 and rl:
                    cells_dump.append({
                        "event": event, "compound": c, "n_runs": rt[c]["n_runs"],
                        "rate": _r(rt[c]["est"]), "lo": _r(rt[c]["lo"]), "hi": _r(rt[c]["hi"]),
                        "width": _r(w),
                    })
            for a, b in zip(comps[:-1], comps[1:], strict=True):  # b is softer
                n_pairs += 1
                ia, ib = rt[a], rt[b]
                overlap = not (ia["hi"] < ib["lo"] or ib["hi"] < ia["lo"])
                diff = ib["est"] - ia["est"]
                if not overlap:
                    sep_any += 1
                    if diff > 0:
                        sep_ok += 1
                    else:
                        sep_wrong += 1
                comp = covd["comp"]
                cov = np.array(covd["cov"])
                i, j = comp.index(a), comp.index(b)
                se = math.sqrt(max(cov[i, i] + cov[j, j] - 2 * cov[i, j], 1e-18))
                if abs(diff) / se > Z95:
                    sep_diff_any += 1
                    if diff > 0:
                        sep_diff_ok += 1
        out["protocols"][f"min_runs_{mr}_{'run_level' if rl else 'lap_level'}"] = {
            "n_cells": n_cells, "n_adjacent_pairs": n_pairs,
            "cells_tighter_than_benchmark": tighter,
            "median_tightening_ratio": _r(float(np.median(ratios)), 2),
            "max_tightening_ratio": _r(float(np.max(ratios)), 2),
            "pairs_separated_non_overlap_any_direction": sep_any,
            "pairs_separated_non_overlap_correct_direction": sep_ok,
            "pairs_separated_non_overlap_wrong_direction": sep_wrong,
            "pairs_separated_by_difference_test_any_direction": sep_diff_any,
            "pairs_separated_by_difference_test_correct_direction": sep_diff_ok,
        }
    out["cells_min_runs_3_run_level"] = cells_dump
    out["caveat"] = (
        "The benchmark width is one driver's three stints at one race; ours is a whole "
        "field's stints at one race. Width is not like for like, and the benchmark's own "
        "tightening is partly just more data. The adjacent-pair counts are within-event "
        "and so are free of the compound-circuit confound."
    )
    return out


# ---------------------------------------------------------------------------
# 5. calibration of the RATE intervals
# ---------------------------------------------------------------------------


def leave_one_event_out_cells(design: pd.DataFrame, full_fit, workers: int, min_runs: int = 3) -> list[dict]:
    """One row per (event, compound): the event's own rate against the pooled
    model refitted WITHOUT that event. See ``rate_calibration``."""
    events = sorted(design["event"].unique())
    if full_fit is None:
        m, _ = build_model(design)
        k = len(m.names)
        with single_threaded_blas():
            full_fit = m.fit(starts=[np.full(k, 0.5), np.full(k, 0.1)])
    payload = _fit_payload(full_fit)
    loeo = {
        e: (rt, tau)
        for e, rt, tau, _ in _run_sharded(
            design, dict(circuit="random", slopes="compound"), payload, _task_loeo, events, workers
        )
    }
    own = {
        e: rt
        for e, _mr, _rl, rt, _cd in _run_sharded(
            design, dict(circuit="none", slopes="compound"), {"theta": None},
            _task_event_fit, [(e, min_runs, True) for e in events], workers,
        )
    }
    cells = []
    for e in events:
        rt_loo, tau = loeo[e]
        for c, o in own[e].items():
            if c not in rt_loo:
                continue  # compound only exists at this event: nothing to predict from
            p = rt_loo[c]
            cells.append({
                "event": e, "compound": c, "own": o["est"], "own_se": o["se"],
                "pooled": p["est"], "pooled_se": p["se"], "tau": tau, "n_runs": o["n_runs"],
            })
    return cells


def rate_calibration(design: pd.DataFrame, B_alt_results: list, full_meta: dict, full_fit, B: int,
                     workers: int, seed: int, levels=(0.5, 0.8, 0.9, 0.95)) -> dict:
    """Do the RATE intervals cover? Two checks the lap-level coverage cannot give.

    (1) Simulation-based calibration. Draw datasets from the fitted model,
        refit, and count how often each nominal interval contains the rate that
        generated the data. This tests the ESTIMATOR (variance estimation,
        Wald intervals with a modest number of circuits), not the model's
        realism: a model can be perfectly calibrated against itself and wrong.

    (2) Leave-one-event-out. Refit without event e; the pooled model's
        predictive interval for what a race at e should look like is
        ``rate_c +- z*sqrt(se_c^2 + tau^2)``, tau the circuit SD. Compare with the
        event's OWN fitted rate (per-event fit, run-level inference, cells with
        at least 3 runs), widening by the event estimate's own sampling variance
        since that is what makes the comparison fair. Versions without the
        circuit term and without the event's sampling variance are reported too,
        so the contribution of each is visible.
    """
    s1 = full_meta["s1"]
    names = full_meta["names"]
    cats = full_meta["cats"]
    ridx = [_col(full_meta, ("tl", c)) for c in cats]
    truth = full_fit.beta[ridx] / s1

    # (1) SBC from the alternative-simulation results
    beta = np.array([a_["beta"][ridx] / s1 for a_ in B_alt_results])
    se = np.array([a_["se"][ridx] / s1 for a_ in B_alt_results])
    sbc = coverage_of_truth(truth, beta, se, levels)
    sbc_out = {
        "n_datasets": len(B_alt_results),
        "compounds": cats,
        "coverage_of_true_rate": sbc,
        "mean_bias_s_per_lap": {c: _r((beta[:, i] - truth[i]).mean(), 5) for i, c in enumerate(cats)},
        "mc_se_at_95_nominal": _r(math.sqrt(0.95 * 0.05 / len(B_alt_results)), 4),
        "what_it_tests": "the estimator against data drawn from the fitted model itself; "
                         "it cannot detect misspecification",
    }

    # (2) leave-one-event-out
    cells = leave_one_event_out_cells(design, full_fit, workers)
    out_levels = {}
    for vname, kw in (
        ("full_predictive", dict(include_circuit=True, include_event_noise=True)),
        ("no_event_sampling_variance", dict(include_circuit=True, include_event_noise=False)),
        ("no_circuit_variance", dict(include_circuit=False, include_event_noise=True)),
    ):
        r = predictive_coverage(cells, levels, **kw)
        out_levels[vname] = {
            "coverage": {
                lv: {"coverage": _r(d["coverage"], 3), "k": d["k"], "n": d["n"],
                     "wilson_95": [_r(d["wilson_95"][0], 3), _r(d["wilson_95"][1], 3)]}
                for lv, d in r["coverage"].items()
            },
            "z_mean": _r(r["z_mean"], 3), "z_sd": _r(r["z_sd"], 3),
        }
    return {
        "simulation_based_calibration": sbc_out,
        "leave_one_event_out": {
            "n_cells": len(cells),
            "variants": out_levels,
            "cells": [{k: (_r(v, 5) if isinstance(v, float) else v) for k, v in c.items()} for c in cells],
            "headline": "variants.full_predictive.coverage['0.80']",
        },
    }


# ---------------------------------------------------------------------------
# 6. ablation (like for like)
# ---------------------------------------------------------------------------


def ablation_table(laps: pd.DataFrame, design: pd.DataFrame, fit) -> dict:
    from ..models import ablation

    d = design.reset_index(drop=True)
    naive = ablation.naive_by_compound(d)
    prac = {s: ablation.practitioner_by_compound(d, laps, s) for s in (0.030, 0.033, 0.035)}
    rows = {}
    for c in fit.ordered:
        if c not in naive:
            continue
        rows[c] = {
            "naive_run_clustered": _iv({"est": naive[c]["cluster"].mean, "lo": naive[c]["cluster"].lo, "hi": naive[c]["cluster"].hi}),
            "naive_iid": _iv({"est": naive[c]["iid"].mean, "lo": naive[c]["iid"].lo, "hi": naive[c]["iid"].hi}),
            "practitioner_fuel_corrected_within_stint": (
                None if c not in prac[0.033] else _iv({"est": prac[0.033][c].mean, "lo": prac[0.033][c].lo, "hi": prac[0.033][c].hi})
            ),
            "practitioner_s_per_kg_030_to_035": [
                _r(prac[s][c].mean) if c in prac[s] else None for s in (0.030, 0.035)
            ],
            "deconfounded": _iv({"est": fit.rates[c].mean, "lo": fit.rates[c].lo, "hi": fit.rates[c].hi}),
            "n_runs": fit.n_runs.get(c),
        }
    neg = [c for c, r in rows.items() if r["naive_run_clustered"]["mean"] < 0]
    gaps = {
        c: _r(abs(r["deconfounded"]["mean"] - r["practitioner_fuel_corrected_within_stint"]["mean"]))
        for c, r in rows.items() if r["practitioner_fuel_corrected_within_stint"]
    }
    return {
        "rows": rows,
        "naive_negative_compounds": neg,
        "abs_gap_deconfounded_vs_practitioner_s_per_lap": gaps,
        "fuel_s_per_kg_central": 0.033,
        "same_laps_same_compound_grouping": True,
        "practitioner_definition": (
            "fuel-corrected lap time on tyre age, within driver x stint (fixed effects), "
            "per physical compound, run-clustered SE; track evolution and traffic NOT controlled"
        ),
    }


# ---------------------------------------------------------------------------
# 7. other seasons
# ---------------------------------------------------------------------------


def season_robustness(seasons: dict[int, pd.DataFrame]) -> dict:
    """The race model on each season, compounds as the weekend's HARD/MEDIUM/SOFT.

    Labels are relative to each weekend's nomination, so a pooled label rate
    mixes different rubber; it is reported with circuit effects on, and beside
    the within-event label contrasts that do not depend on the mixing.
    """
    from ..models.design import prepare

    out = {}
    for season, laps in sorted(seasons.items()):
        t = time.time()
        d = prepare(laps, "race", compounds="label").reset_index(drop=True)
        if d.empty:
            continue
        f = fit_degradation(d, quadratic=False, circuit_effects=True, order=LABEL_ORDER)
        m, meta = build_model(d, circuit="fixed", slopes="label_rank", order=LABEL_ORDER)
        k = len(m.names)
        with single_threaded_blas():
            g = m.fit(starts=[np.full(k, 0.5), np.full(k, 0.1)])
        mt, metat = build_model(d, circuit="fixed", slopes="trend", order=LABEL_ORDER)
        kt = len(mt.names)
        with single_threaded_blas():
            gt = mt.fit(starts=[np.full(kt, 0.5), np.full(kt, 0.1)])
        mr, metar = build_model(d, circuit="random", slopes="trend", order=LABEL_ORDER)
        kr = len(mr.names)
        with single_threaded_blas():
            gr = mr.fit(starts=[np.full(kr, 0.5), np.full(kr, 0.1)])
        out[str(season)] = {
            "n_events": int(d["event"].nunique()), "n_laps": int(len(d)),
            "n_runs": int(d["run_id"].nunique()),
            "rates_pooled_label_s_per_lap": {
                c: {"mean": _r(v.mean), "lo": _r(v.lo), "hi": _r(v.hi), "n_runs": f.n_runs.get(c)}
                for c, v in f.rates.items()
            },
            "monotone_softer_degrades_faster": bool(f.is_monotone()),
            "circuit_slope_sd_s_per_lap": _r(f.variance.get("circuit_slope")),
            "within_event_label_contrast_s_per_lap": {
                "MEDIUM_minus_HARD": _iv(lincomb(g, meta, {("lab", "MEDIUM"): 1.0})),
                "SOFT_minus_HARD": _iv(lincomb(g, meta, {("lab", "SOFT"): 1.0})),
            },
            "trend_per_step_softer": {
                "within_event": _iv(lincomb(gt, metat, {("trend", "step"): 1.0})),
                "pooled_random_circuit": _iv(lincomb(gr, metar, {("trend", "step"): 1.0})),
            },
            "convergence_all_ok": bool(f.convergence.get("all_ok")),
            "fit_seconds": round(time.time() - t, 1),
        }
    return out


# ---------------------------------------------------------------------------
# dataset audit
# ---------------------------------------------------------------------------


def dataset_audit(laps: pd.DataFrame, design: pd.DataFrame) -> dict:
    by = laps.groupby(["event", "session"]).size().unstack(fill_value=0)
    race_events = sorted(laps.loc[laps["session"] == "R", "event"].unique())
    return {
        "n_laps_total": int(len(laps)),
        "n_events_in_file": int(laps["event"].nunique()),
        "events_in_file": sorted(laps["event"].unique()),
        "race_events_in_file": race_events,
        "events_with_no_race_laps": sorted(set(laps["event"]) - set(race_events)),
        "laps_by_session": {k: int(v) for k, v in laps.groupby("session").size().items()},
        "race_fit": {
            "sessions": ["R"], "n_events": int(design["event"].nunique()),
            "events": sorted(design["event"].unique()), "n_laps": int(len(design)),
            "n_runs": int(design["run_id"].nunique()), "n_drivers": int(design["Driver"].nunique()),
        },
        "laps_by_event_session": {e: {s: int(v) for s, v in r.items() if v} for e, r in by.iterrows()},
    }


# ---------------------------------------------------------------------------
# 8. circuit versus event: the pooled 2022-2026 nested model
# ---------------------------------------------------------------------------


def circuit_key(season: int, event: str) -> str:
    """The circuit a race was held at, as a stable name across seasons.

    FastF1 event names follow the circuit for every race in the cached seasons
    except one: the Barcelona-Catalunya track hosted the "Spanish Grand Prix"
    through 2025 and is the "Barcelona Grand Prix" in 2026 (the 2026 "Spanish
    Grand Prix" is Madrid, which has no race laps here).
    """
    return "Spanish Grand Prix" if event == "Barcelona Grand Prix" else event


def pooled_physical_design(
    seasons: dict[int, pd.DataFrame], verified_only: bool = False
) -> pd.DataFrame:
    """Race design frames for several seasons, keyed on PHYSICAL compounds.

    2022-2025 laps carry only HARD/MEDIUM/SOFT; the C number behind each label
    comes from Pirelli's nominations (``data/allocation_history.json``). 2026
    uses ``data/allocation.json``. Each season is prepared on its own, so the
    (event, lap) within-transformation never mixes weekends. ``run_id`` is
    prefixed with the season because the same driver can run the same stint
    number at the same event in two seasons.

    C-numbers are comparable within a season's range, not strictly across the
    2025 re-specification; ``circuit_vs_event`` therefore reports a fit without
    2025 beside the pooled one.
    """
    from ..data.allocation_history import c_number, load_history
    from ..models.design import prepare

    hist = load_history()["seasons"]
    out = []
    for season, laps in sorted(seasons.items()):
        if season == 2026:
            d = prepare(laps, "race")
        else:
            d = prepare(laps, "race", compounds="label")
            d["C"] = [c_number(season, e, lab) for e, lab in zip(d["event"], d["Compound"], strict=True)]
            d = d[d["C"].notna()]
            if verified_only:
                ok = {e for e, r in hist.get(str(season), {}).items() if r.get("verified")}
                d = d[d["event"].isin(ok)]
        d = d.copy()
        d["season"] = season
        d["event_key"] = f"{season}|" + d["event"].astype(str)
        d["circuit"] = [circuit_key(season, e) for e in d["event"]]
        d["run_id"] = f"{season}|" + d["run_id"].astype(str)
        out.append(d)
    p = pd.concat(out, ignore_index=True)
    return p[p["C"].isin(C_ORDER)].reset_index(drop=True)


def build_nested_model(
    df: pd.DataFrame, *, order: tuple[str, ...] = C_ORDER, circuit: bool = True,
    event: bool = True, run_slope: bool = True,
    extra_fixed: dict[str, np.ndarray] | None = None, exact_within: bool = True,
) -> tuple[RemlModel, dict]:
    """Circuit random slope plus event-within-circuit random slope.

    Blocks are CIRCUITS, so a circuit's several races (one per season) share one
    ``circuit_slope`` while each race has its own ``event_slope``. Driver-in-event
    and run terms are nested inside the race as before. ``extra_fixed`` adds
    fixed-effect columns exactly as in ``build_model`` (default: none).

    ``exact_within`` (default) demeans every fixed-effect column within its
    (event_key, lap) cell: ``tilde(D_c)`` offsets (first compound is the
    reference) and ``tilde(D_c * raw age)`` slopes. The cell is a race-season,
    so seasons never mix. ``False`` is the earlier approximate construction.
    The circuit and race slope columns use ``tilde(age)``, which is exact because
    the circuit and the race are constant within a cell.
    """
    df = df.reset_index(drop=True)
    cells = None
    if exact_within:
        df, codes, ncell = redemean_frame(df)
        cells = (codes, ncell)
    cats = [c for c in order if c in set(df["C"])]
    s1 = float(df["tl"].std())
    tl = df["tl"].to_numpy(float) / s1
    C = df["C"].to_numpy()
    if cells is None:
        cols = [(C == c).astype(float) for c in cats] + [(C == c) * tl for c in cats]
        names = [("off", c) for c in cats] + [("tl", c) for c in cats]
    else:
        off, off_cats = offset_columns(C, cats, *cells)
        age = df["TyreLife"].to_numpy(float)
        cols = [off[:, j] for j in range(len(off_cats))] + [
            demean((C == c) * age, *cells) / s1 for c in cats]
        names = [("off", c) for c in off_cats] + [("tl", c) for c in cats]
    if "tr" in df.columns and float(df["tr"].std() or 0.0) > 1e-9:
        cols.append(df["tr"].to_numpy(float))
        names.append(("tr", ""))
    _append_extra_fixed(cols, names, extra_fixed, len(df))
    blk = df["circuit"].astype(str).to_numpy()
    ek = df["event_key"].astype(str).to_numpy()
    comps = []
    if circuit:
        comps.append(Comp("circuit_slope", blk, tl))
    if event:
        comps.append(Comp("event_slope", ek, tl))
    comps.append(Comp("driver_event", ek + "|" + df["Driver"].astype(str).to_numpy()))
    comps.append(Comp("run", df["run_id"].astype(str).to_numpy()))
    if run_slope:
        comps.append(Comp("run_slope", df["run_id"].astype(str).to_numpy(), tl))
    model = RemlModel(df["y"].to_numpy(float), np.column_stack(cols), blk, comps)
    return model, {"names": names, "s1": s1, "cats": cats, "exact_within": bool(exact_within)}


def _fit_nested(df: pd.DataFrame, **kw):
    m, meta = build_nested_model(df, **kw)
    k = len(m.names)
    with single_threaded_blas():
        f = m.fit(starts=[np.full(k, 0.5), np.full(k, 0.1), np.linspace(2, 0.2, k)])
    return m, meta, f


def _sd_summary(sd_samples: np.ndarray) -> list:
    return [_r(np.percentile(sd_samples, 2.5)), _r(np.percentile(sd_samples, 97.5))]


def circuit_vs_event(pooled: pd.DataFrame, b_null: int, b_alt: int, workers: int, seed: int) -> dict:
    """How much of the slope variance is the CIRCUIT, and how much the single RACE?

    In 2026 every circuit hosted one race, so the circuit slope SD of the 2026
    model is race-to-race variance (weather, track temperature, safety cars,
    strategy), not a property of the track. Five seasons give repeat visits to
    most circuits, which is what separates a persistent circuit component from a
    one-off event one.

    Reported: both SDs with profile-likelihood and parametric-bootstrap
    intervals, the persistent share ``tau_c^2 / (tau_c^2 + tau_e^2)``, a
    parametric-bootstrap REML likelihood-ratio test for the circuit component
    (the boundary makes chi-squared wrong), the chi-bar-squared reference for
    the event component, and sensitivity fits without 2025 (C-number
    re-specification) and on verified nominations only.
    """
    d = pooled.reset_index(drop=True)
    m, meta, full = _fit_nested(d)
    names = m.names
    s1 = meta["s1"]
    k = len(names)
    free_nc = [n for n in names if n != "circuit_slope"]
    free_ne = [n for n in names if n != "event_slope"]
    with single_threaded_blas():
        null_c = m.fit(free=free_nc, starts=[np.full(k - 1, 0.5), np.full(k - 1, 0.1)])
        null_e = m.fit(free=free_ne, starts=[np.full(k - 1, 0.5), np.full(k - 1, 0.1)])
        diag = m.diagnostics(full)
    lr_c = float(null_c.deviance - full.deviance)
    lr_e = float(null_e.deviance - full.deviance)
    sd_c = full.sd["circuit_slope"] / s1
    sd_e = full.sd["event_slope"] / s1
    share = sd_c**2 / (sd_c**2 + sd_e**2)

    payload = {**_fit_payload(full), "beta0": null_c.beta, "sigma20": null_c.sigma2,
               "theta0": null_c.theta, "null_comp": "circuit_slope"}
    spec = {"nested": True, "order": C_ORDER}
    ss = np.random.SeedSequence(seed + 1).spawn(b_null + b_alt)
    t = time.time()
    lrs = np.array(_run_sharded(d, spec, payload, _task_null, ss[:b_null], workers))
    alts = _run_sharded(d, spec, payload, _task_alt, ss[b_null:], workers)
    boot_seconds = time.time() - t
    sdc = np.array([a["sd"]["circuit_slope"] / s1 for a in alts])
    sde = np.array([a["sd"]["event_slope"] / s1 for a in alts])
    shr = sdc**2 / (sdc**2 + sde**2 + 1e-18)
    p_boot = float((1 + (lrs >= lr_c).sum()) / (1 + len(lrs)))

    with single_threaded_blas():
        prof_c = _profile_ci(m, full, s1, name="circuit_slope")
        prof_e = _profile_ci(m, full, s1, name="event_slope")

    ridx = [_col(meta, ("tl", c)) for c in meta["cats"]]
    rates = {c: lincomb(full, meta, {("tl", c): 1.0}) for c in meta["cats"]}
    comp_sd = float(np.std([v["est"] for v in rates.values()], ddof=1))

    def sens(df_):
        mm, mt, ff = _fit_nested(df_)
        return {
            "circuit_sd": _r(ff.sd["circuit_slope"] / mt["s1"]), "event_sd": _r(ff.sd["event_slope"] / mt["s1"]),
            "share_circuit": _r(ff.sd["circuit_slope"] ** 2 / (ff.sd["circuit_slope"] ** 2 + ff.sd["event_slope"] ** 2), 3),
            "n_laps": int(len(df_)), "n_events": int(df_["event_key"].nunique()),
        }

    per_circuit = d.groupby("circuit")["event_key"].nunique()
    return {
        "design": {
            "seasons": sorted(int(x) for x in d["season"].unique()),
            "n_laps": int(len(d)), "n_runs": int(d["run_id"].nunique()),
            "n_events": int(d["event_key"].nunique()), "n_circuits": int(d["circuit"].nunique()),
            "circuits_with_repeat_visits": int((per_circuit > 1).sum()),
            "events_per_circuit_median": float(per_circuit.median()),
            "compounds": "physical C numbers (2022-25 from Pirelli nomination history, 2026 from allocation.json)",
        },
        "sd_s_per_lap": {
            "circuit_persistent": {"estimate": _r(sd_c), "profile_95": [prof_c["lo"], prof_c["hi"]],
                                   "bootstrap_95": _sd_summary(sdc)},
            "event_specific": {"estimate": _r(sd_e), "profile_95": [prof_e["lo"], prof_e["hi"]],
                               "bootstrap_95": _sd_summary(sde)},
            "run_slope": _r(full.sd["run_slope"] / s1), "run_intercept": _r(full.sd["run"]),
            "driver_event": _r(full.sd["driver_event"]), "resid": _r(np.sqrt(full.sigma2)),
        },
        "persistent_share_of_slope_variance": {
            "estimate": _r(share, 3), "bootstrap_95": [_r(np.percentile(shr, 2.5), 3), _r(np.percentile(shr, 97.5), 3)],
        },
        "circuit_component_test": {
            "reml_lr": _r(lr_c, 3), "p_parametric_bootstrap": _r(p_boot, 4), "n_bootstrap": int(len(lrs)),
            "p_floor": _r(1 / (1 + len(lrs)), 4),
            "p_chibar2": float(f"{0.5 * stats.chi2.sf(lr_c, 1):.3g}"),
            "bootstrap_lr_95_99": [_r(np.percentile(lrs, 95), 3), _r(np.percentile(lrs, 99), 3)],
            "distinguishable_from_zero": bool(p_boot < 0.05),
        },
        "event_component_test": {"reml_lr": _r(lr_e, 3), "p_chibar2": float(f"{0.5 * stats.chi2.sf(lr_e, 1):.3g}")},
        "seconds_over_20_laps": {
            "circuit_persistent": _r(sd_c * 20, 3), "event_specific": _r(sd_e * 20, 3),
            "circuit_persistent_profile_95": [_r(prof_c["lo"] * 20, 3), _r(prof_c["hi"] * 20, 3) if prof_c["hi"] else None],
            "event_specific_profile_95": [_r(prof_e["lo"] * 20, 3), _r(prof_e["hi"] * 20, 3) if prof_e["hi"] else None],
        },
        "compound_rates_pooled_s_per_lap": {c: _iv(v) for c, v in rates.items()},
        "compound_rate_sd_s_per_lap": _r(comp_sd),
        "compound_vs_variance": _compound_vs_components(alts, ridx, s1, sd_c, sd_e, full),
        "sensitivity": {
            "without_2025": sens(d[d["season"] != 2025]),
            "verified_nominations_only": sens(d[~d["event_key"].isin(_unverified_events())]),
        },
        "convergence": {"converged": bool(full.converged), "hessian_pd": diag["hessian_pd"],
                        "boundary_components": diag["boundary_components"],
                        "start_deviances": [_r(x, 3) for x in full.start_deviances]},
        "bootstrap_seconds": round(boot_seconds, 1),
    }


def regulation_sensitivity(pooled: pd.DataFrame) -> dict:
    """Does the pooled picture survive dropping 2026, the season of new technical rules and narrower tyres?

    2022-2025 share one set of car rules and one tyre size (18-inch, introduced in
    2022); 2026 changed the power units, aerodynamics and tyre widths. The pooled
    nested model is refitted on 2022-2025 only (whole races removed, so every
    (event, lap) group stays complete) and the main quantities are set beside the
    2022-2026 fit. Point estimates only: the intervals of the full fit come from
    the bootstrap in ``circuit_vs_event``.
    """

    def summary(df_):
        m, meta, f = _fit_nested(df_.reset_index(drop=True))
        s1 = meta["s1"]
        sd_c, sd_e = f.sd["circuit_slope"] / s1, f.sd["event_slope"] / s1
        rates = {c: lincomb(f, meta, {("tl", c): 1.0}) for c in meta["cats"]}
        cats = meta["cats"]
        steps = {f"{b}-{a}": _iv(lincomb(f, meta, {("tl", b): 1.0, ("tl", a): -1.0}))
                 for a, b in zip(cats[:-1], cats[1:])}
        per_circuit = df_.groupby("circuit")["event_key"].nunique()
        return {
            "seasons": sorted(int(x) for x in df_["season"].unique()),
            "n_laps": int(len(df_)), "n_events": int(df_["event_key"].nunique()),
            "n_circuits": int(df_["circuit"].nunique()),
            "circuits_with_repeat_visits": int((per_circuit > 1).sum()),
            "circuit_sd": _r(sd_c), "event_sd": _r(sd_e),
            "share_circuit": _r(sd_c**2 / (sd_c**2 + sd_e**2), 3),
            "compound_rates_s_per_lap": {c: _iv(v) for c, v in rates.items()},
            "compound_rate_sd": _r(float(np.std([v["est"] for v in rates.values()], ddof=1))),
            "adjacent_steps_s_per_lap": steps,
            "converged": bool(f.converged),
        }

    d = pooled.reset_index(drop=True)
    return {
        "question": "Is pooling 2026 (new rules, narrower tyres) with 2022-2025 (one rule set) driving the pooled results?",
        "all_2022_2026": summary(d),
        "without_2026": summary(d[d["season"] != 2026]),
    }


def _unverified_events() -> set[str]:
    """Races whose nomination rests on one source or an inferred label order."""
    from ..data.allocation_history import load_history

    return {
        f"{y}|{e}" for y, evs in load_history()["seasons"].items() for e, r in evs.items()
        if not r.get("verified")
    }


def _compound_vs_components(alts, ridx, s1, sd_c, sd_e, full) -> dict:
    """SD of the five compound rates against the slope SDs, from the replicates."""
    est = np.array([a["beta"][ridx] / s1 for a in alts])
    se = np.array([a["se"][ridx] / s1 for a in alts])
    sd_comp = est.std(axis=1, ddof=1)
    # Estimated rates carry sampling noise, so their raw SD overstates the SD of
    # the TRUE compound rates; method-of-moments removes the mean squared SE.
    corr = np.sqrt(np.maximum(est.var(axis=1, ddof=1) - (se**2).mean(axis=1), 0.0))
    sdc = np.array([a["sd"].get("circuit_slope", np.nan) / s1 for a in alts])
    sde = np.array([a["sd"].get("event_slope", np.nan) / s1 for a in alts])
    out = {
        "compound_sd_raw": {"bootstrap_95": _sd_summary(sd_comp), "median": _r(np.median(sd_comp))},
        "compound_sd_noise_corrected": {"bootstrap_95": _sd_summary(corr), "median": _r(np.median(corr))},
    }
    for label, x in (("circuit_minus_compound_raw", sdc - sd_comp), ("circuit_minus_compound_corrected", sdc - corr)):
        ok = x[np.isfinite(x)]
        if len(ok):
            out[label] = {"bootstrap_95": _sd_summary(ok), "median": _r(np.median(ok)),
                          "p_circuit_greater": _r(float((ok > 0).mean()), 3)}
    ok = (sde - sd_comp)[np.isfinite(sde)]
    if len(ok):
        out["event_minus_compound_raw"] = {"bootstrap_95": _sd_summary(ok), "median": _r(np.median(ok)),
                                          "p_event_greater": _r(float((ok > 0).mean()), 3)}
    return out


def compound_vs_circuit_2026(alts: list, full, meta: dict, design: pd.DataFrame, workers: int,
                             seed: int = 0, b_boot: int = 0) -> dict:
    """The 2026 circuit SD against the SD of the five compound rates, with intervals.

    Three intervals for the difference: the parametric bootstrap already run for
    the circuit test (data redrawn from the fitted model and refitted, paired
    per replicate), a nonparametric bootstrap over events (the 13 races redrawn
    with replacement, runs travelling with their race, each draw refitted), and
    a leave-one-event-out jackknife. The last two assume nothing about the
    model. The raw compound SD includes estimation noise;
    the noise-corrected version removes it (and is the like-for-like partner of
    the latent circuit SD).
    """
    s1 = meta["s1"]
    ridx = [_col(meta, ("tl", c)) for c in meta["cats"]]
    sd_c = full.sd["circuit_slope"] / s1
    rates = np.array([full.beta[i] / s1 for i in ridx])
    se = np.array([np.sqrt(full.cov_beta[i, i]) / s1 for i in ridx])
    raw = float(np.std(rates, ddof=1))
    corr = float(np.sqrt(max(np.var(rates, ddof=1) - np.mean(se**2), 0.0)))
    boot = _compound_vs_components(alts, ridx, s1, sd_c, float("nan"), full)

    events = sorted(design["event"].unique())
    payload = _fit_payload(full)
    loeo = _run_sharded(design, dict(circuit="random", slopes="compound"), payload, _task_loeo, events, workers)
    theta_raw, theta_corr, taus = [], [], []
    for _e, rt, tau, _ok in loeo:
        r = np.array([v["est"] for v in rt.values()])
        s = np.array([v["se"] for v in rt.values()])
        if len(r) < 3:
            continue
        taus.append(tau)
        theta_raw.append(tau - np.std(r, ddof=1))
        theta_corr.append(tau - np.sqrt(max(np.var(r, ddof=1) - np.mean(s**2), 0.0)))

    def jack(x, point):
        x = np.asarray(x)
        n = len(x)
        se_j = float(np.sqrt((n - 1) / n * ((x - x.mean()) ** 2).sum()))
        return {"estimate": _r(point), "jackknife_se": _r(se_j), "ci95_normal": [_r(point - 1.96 * se_j), _r(point + 1.96 * se_j)]}

    cluster = None
    if b_boot:
        ss = np.random.SeedSequence(seed + 2).spawn(b_boot)
        res = _run_sharded(design, dict(circuit="random", slopes="compound"), payload, _task_event_boot,
                           ss, workers)
        ok = [r for r in res if r["n_compounds"] == len(meta["cats"]) and r["converged"]]
        sdr = np.array([np.std(r["rate"], ddof=1) for r in ok])
        sdk = np.array([np.sqrt(max(np.var(r["rate"], ddof=1) - np.mean(r["se"] ** 2), 0.0)) for r in ok])
        tau = np.array([r["circuit_sd"] for r in ok])

        def summ(x):
            return {"bootstrap_95": _sd_summary(x), "median": _r(np.median(x)),
                    "p_gt_0": _r(float((x > 0).mean()), 3)}

        cluster = {
            "n_bootstrap": int(b_boot), "n_used": int(len(ok)),
            "share_dropped_missing_compound_or_unconverged": _r(1 - len(ok) / b_boot, 3),
            "circuit_sd": {"bootstrap_95": _sd_summary(tau), "median": _r(np.median(tau))},
            "compound_sd_raw": {"bootstrap_95": _sd_summary(sdr), "median": _r(np.median(sdr))},
            "compound_sd_noise_corrected": {"bootstrap_95": _sd_summary(sdk), "median": _r(np.median(sdk))},
            "circuit_minus_compound_raw": summ(tau - sdr),
            "circuit_minus_compound_corrected": summ(tau - sdk),
            "note": "Races redrawn with replacement; a repeated race counts as two blocks. Resamples "
                    "missing a compound (C1 runs at only two races) are dropped and counted.",
        }
    return {
        "circuit_sd_s_per_lap": _r(sd_c),
        "compound_rate_sd_raw": _r(raw), "compound_rate_sd_noise_corrected": _r(corr),
        "parametric_bootstrap": boot,
        "event_cluster_bootstrap": cluster,
        "jackknife_leave_one_event_out": {
            "circuit_minus_compound_raw": jack(theta_raw, sd_c - raw),
            "circuit_minus_compound_corrected": jack(theta_corr, sd_c - corr),
            "n_events": len(theta_raw),
        },
        "note": "In 2026 each circuit hosts one race, so this circuit SD is race-to-race variance.",
    }


# ---------------------------------------------------------------------------
# the within-transformation correction: what it changed
# ---------------------------------------------------------------------------


def _summarise_fit(rates: dict, sd: dict, extra: dict | None = None) -> dict:
    """Compound rates with intervals, variance SDs and the SD of the compound rates."""
    est = np.array([v["mean"] for v in rates.values()])
    se = np.array([(v["hi"] - v["lo"]) / (2 * Z95) for v in rates.values()])
    out = {
        "rates_s_per_lap": rates,
        "variance_sd": sd,
        "compound_rate_sd_raw": _r(np.std(est, ddof=1)),
        "compound_rate_sd_noise_corrected": _r(np.sqrt(max(np.var(est, ddof=1) - np.mean(se**2), 0.0))),
    }
    out.update(extra or {})
    return out


def exact_vs_approximate(design: pd.DataFrame, pooled: pd.DataFrame) -> dict:
    """What correcting the within-(event, lap) transformation changed.

    Eq. 1 subtracts the (event, lap) cell mean from lap time and from every
    fixed-effect column. The earlier construction subtracted it from lap time
    and tyre age only and multiplied the result by a raw compound dummy; the
    correct construction demeans ``D_c`` and ``D_c * raw age`` themselves (see
    ``models.within``). The two differ whenever compounds are mixed on a lap,
    which biases the compound slopes (``paper/results/recovery.json``).

    Both are fitted here on the same laps, with the same random effects and the
    same REML engine: the 2026 race fit (13 races, circuit slope) and the pooled
    2022-26 nested fit (circuit plus race slopes). Reported for each: compound
    rates with 95% intervals, every variance component, the SD of the five
    compound rates (raw and corrected for estimation noise), the persistent
    share (pooled), and the rate-by-rate difference. The ``approximate`` block of
    ``shipped_2026`` has the same layout as ``shipped_fit`` and holds the
    planted truth of the recovery study, which is not re-planted.
    """
    d26 = design.reset_index(drop=True)
    out: dict = {
        "what": "Same laps, same random effects, same REML engine; only the construction of the "
                "fixed-effect columns differs. exact: tilde(D_c) offsets (one reference compound) and "
                "tilde(D_c x raw tyre age) slopes; approximate: raw D_c offsets and D_c x tilde(age) slopes.",
        "shipped_2026": {}, "pooled_2022_26_nested": {},
    }
    for label, ex in (("exact", True), ("approximate", False)):
        f = fit_degradation(d26, quadratic=False, circuit_effects=True, exact_within=ex)
        rates = {c: {"mean": round(v.mean, 5), "lo": round(v.lo, 5), "hi": round(v.hi, 5),
                     "n_runs": f.n_runs.get(c), "n_laps": f.n_laps.get(c)} for c, v in f.rates.items()}
        sd = {k: round(v, 5) for k, v in f.variance.items()}
        out["shipped_2026"][label] = _summarise_fit(
            rates, sd, {"converged_all_ok": bool(f.convergence.get("all_ok")), "n_obs": f.n_obs})
    p = pooled.reset_index(drop=True)
    for label, ex in (("exact", True), ("approximate", False)):
        m, meta, nf = _fit_nested(p, exact_within=ex)
        s1 = meta["s1"]
        rt = _rate_table(nf, meta)
        rates = {c: {"mean": _r(v["est"]), "lo": _r(v["lo"]), "hi": _r(v["hi"])} for c, v in rt.items()}
        sd = {n: _r(nf.sd[n] / s1 if "slope" in n else nf.sd[n]) for n in m.names}
        sd["resid"] = _r(np.sqrt(nf.sigma2))
        c_, e_ = nf.sd["circuit_slope"] / s1, nf.sd["event_slope"] / s1
        out["pooled_2022_26_nested"][label] = _summarise_fit(rates, sd, {
            "persistent_share_of_slope_variance": _r(c_**2 / (c_**2 + e_**2 + 1e-18), 3),
            "converged": bool(nf.converged), "n_obs": int(len(p))})
    for key in ("shipped_2026", "pooled_2022_26_nested"):
        blk = out[key]
        blk["exact_minus_approximate_rate_s_per_lap"] = {
            c: _r(blk["exact"]["rates_s_per_lap"][c]["mean"] - blk["approximate"]["rates_s_per_lap"][c]["mean"])
            for c in blk["exact"]["rates_s_per_lap"] if c in blk["approximate"]["rates_s_per_lap"]
        }
        blk["compound_rate_sd_exact_minus_approximate"] = _r(
            blk["exact"]["compound_rate_sd_raw"] - blk["approximate"]["compound_rate_sd_raw"])
    return out


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------


def jsonable(o):
    """Recursively make numpy / pandas scalars JSON-safe. NaN becomes null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def audit(script: str, inputs, seed: int, extra: dict | None = None) -> dict:
    """Code version, input hashes, seeds and library versions for a results file.

    The timestamp and wall-clock times are the only fields that differ between
    two runs of the same code on the same data.
    """
    import datetime as dt
    import hashlib
    import subprocess
    import sys
    from pathlib import Path

    import scipy
    import statsmodels

    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=root, timeout=10).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"],
                                    capture_output=True, text=True, cwd=root,
                                    timeout=10).stdout.strip())
    except Exception:  # noqa: BLE001 -- provenance must never stop a run
        commit, dirty = None, None
    ins = {}
    for p in inputs:
        p = Path(p)
        if p.exists():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            ins[p.name] = {"sha256_16": h.hexdigest()[:16], "bytes": p.stat().st_size}
    return {
        "script": script,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty_src_or_scripts": dirty,
        "seed": seed,
        "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "statsmodels": statsmodels.__version__,
        "inputs": ins, **(extra or {}),
    }


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------


def _power_task(args):
    n, effect, design, params, seeds = args
    from ..validation.power import simulate_mixed_power

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    return simulate_mixed_power(n, effect, design, params, seeds)


def _round_dict(d: dict) -> dict:
    return {k: (_r(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "power_at_grid"}


def power_section(variance: dict, n_race_runs: int, workers: int, seed: int, n_sims: int,
                  grid=(52, 104, 208, 416, 832, 1248, 1664, 2496), effects=(0.006, 0.02),
                  welch_noise: float | None = None) -> dict:
    """Power of the compound contrast, simulated from the fitted mixed model."""
    from ..validation import power as P

    params = P.MixedParams.from_fit(variance)
    chunk = 10
    root = np.random.SeedSequence(seed)
    jobs, keys = [], []
    cells = [(d, e) for e in effects for d in ("balanced", "split")]
    cell_ss = root.spawn(len(cells) + 1)
    for (design, effect), ss in zip(cells, cell_ss[: len(cells)], strict=True):
        pt_ss = ss.spawn(len(grid))
        for n, ps in zip(grid, pt_ss, strict=True):
            if design == "split" and n not in (grid[0], grid[4], grid[-1]):
                continue  # the split curve is flat; three points show it
            sims = ps.spawn(n_sims)
            for i in range(0, n_sims, chunk):
                jobs.append((n, effect, design, params, sims[i:i + chunk]))
                keys.append((design, effect, n))
    # size: effect = 0, to check the Wald z test holds its level
    for design, ss in zip(("balanced", "split"), cell_ss[-1].spawn(2), strict=True):
        sims = ss.spawn(n_sims)
        for i in range(0, n_sims, chunk):
            jobs.append((grid[4], 0.0, design, params, sims[i:i + chunk]))
            keys.append((design, 0.0, grid[4]))

    t = time.time()
    _single_thread_children()
    res = _run_sharded(None, None, None, _power_task, jobs, workers)
    acc: dict = {}
    for k, r in zip(keys, res, strict=True):
        a = acc.setdefault(k, {"z": [], "t": [], "bad": 0})
        a["z"] += r["reject_z"]
        a["t"] += r["reject_t"]
        a["bad"] += r["n_failed_fits"]

    curves = {}
    for design, effect in cells:
        pts = sorted(n for (d, e, n) in acc if d == design and e == effect)
        rz = [acc[(design, effect, n)]["z"] for n in pts]
        rt_ = [acc[(design, effect, n)]["t"] for n in pts]
        ana = {n: P.analytic_power(n, effect, design, params)["power"] for n in pts}
        curves[f"{design}_effect_{effect}"] = {
            "n_stints": pts,
            "power_wald_z": [_r(np.mean(x), 3) for x in rz],
            "power_t_df12": [_r(np.mean(x), 3) for x in rt_],
            "power_analytic_known_variance": [_r(ana[n], 3) for n in pts],
            "mc_se_at_50pct": _r(math.sqrt(0.25 / n_sims), 3),
            "crossing_80pct_wald_z": _round_dict(P.crossing_with_mc_error(pts, rz, seed=seed)),
            "crossing_80pct_t_df12": _round_dict(P.crossing_with_mc_error(pts, rt_, seed=seed)),
            "failed_refits": int(sum(acc[(design, effect, n)]["bad"] for n in pts)),
            "crossing_is_at_or_below_smallest_grid_point": bool(
                np.mean(rz[0]) >= 0.8
            ),
        }
    # analytic crossings on a fine grid
    fine = [26 * k for k in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128)]
    analytic = {}
    for design, effect in cells:
        p_ = [P.analytic_power(n, effect, design, params)["power"] for n in fine]
        analytic[f"{design}_effect_{effect}"] = _r(P.crossing_linear(fine, p_), 1)
    size = {
        d: {"n_stints": grid[4], "reject_rate_wald_z": _r(np.mean(acc[(d, 0.0, grid[4])]["z"]), 3),
            "reject_rate_t_df12": _r(np.mean(acc[(d, 0.0, grid[4])]["t"]), 3),
            "n_sims": n_sims, "nominal": 0.05}
        for d in ("balanced", "split")
    }
    # smallest effect with 80% analytic power at our run count, per design
    n_ours = int(round(n_race_runs / 26)) * 26
    mde = {}
    for d in ("balanced", "split"):
        se = P.analytic_power(n_ours, 0.01, d, params)["se"]
        mde[d] = _r((1.959964 + 0.841621) * se, 4)
    # the original Welch calculation, re-run with the same interpolation
    pc = P.power_curve(n_sims=600, seed=seed, **({} if welch_noise is None else {"noise": welch_noise}))
    old = {
        "grid_point_returned_by_n_for": pc.n_for(0.80),
        "interpolated_crossing": _r(P.crossing_linear(pc.n_stints, pc.power), 1),
        "grid": [int(x) for x in pc.n_stints], "power": [_r(x, 3) for x in pc.power],
        "noise_used_s": welch_noise if welch_noise is not None else P.DEFAULT_NOISE,
        "what_it_ignores": "circuit variance, stint-level wear variation, run correlation",
    }
    return {
        "params_per_lap": {
            "circuit_sd": _r(params.circuit_sd, 5), "run_slope_sd": _r(params.run_slope_sd, 5),
            "run_sd": _r(params.run_sd, 4), "driver_sd": _r(params.driver_sd, 4),
            "sigma": _r(params.sigma, 4),
        },
        "design_notes": "13 circuits, 20-lap stints, 3 stints per driver-event, two compounds; "
                        "balanced = both at every circuit, split = disjoint circuits",
        "n_sims_per_point": n_sims,
        "curves": curves,
        "analytic_crossing_known_variance": analytic,
        "type_I_error_at_zero_effect": size,
        "ceiling_split_design_power": {str(e): _r(P.power_ceiling_split(e, params), 3) for e in effects},
        "min_detectable_effect_80pct_s_per_lap": {"n_stints": n_ours, **mde},
        "original_welch_calculation": old,
        "seconds": round(time.time() - t, 1),
    }


# ---------------------------------------------------------------------------
# 9. power, extended: realistic nominations, variance uncertainty, more effects
# ---------------------------------------------------------------------------


def nominations_from_design(design: pd.DataFrame) -> list[tuple[str, ...]]:
    """Pirelli's compound triple for each event that has race laps, in order."""
    from ..data.allocation import all_allocations

    alloc = all_allocations()
    return [
        tuple(sorted(set(alloc[e].compounds.values())))
        for e in sorted(design["event"].unique()) if e in alloc
    ]


def _power_task_x(args):
    """``_power_task`` that also carries a nomination plan."""
    from ..validation.power import simulate_mixed_power

    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    n, effect, design, params, seeds, nominations, pair = args
    return simulate_mixed_power(n, effect, design, params, seeds, nominations=nominations, pair=pair)


def power_extended(
    variance: dict, circuit_sd_ci: tuple[float, float], design: pd.DataFrame, workers: int,
    seed: int, reps: int = 150, size_reps: int = 1000,
    effects=(0.006, 0.01, 0.02, 0.03), sim_pair=("C1", "C2"), max_sim_stints: int = 3500,
) -> dict:
    """Stints needed for 80% power, with realistic nominations and honest variance.

    Simulated points above ``max_sim_stints`` (about four times the 2026 data)
    are not run: one refit there needs ~7 GB and ~5 CPU-minutes, and the
    analytic curve already covers that range. Seeds are spawned for the full
    grid before the cap is applied, so every point that is run gets exactly the
    seeds it would have had without the cap. Skipped points are listed in the
    output.

    Three things the first power analysis lacked.

    * A REALISTIC design between the balanced and the polar split one: stints go
      only on the compounds each 2026 event actually nominated. Because C3 is
      nominated everywhere, any two compounds are linked through a chain of
      within-event contrasts, which is why this lands near the balanced design
      and nowhere near the split one. The weakest-linked adjacent pair is simulated.
    * Variance uncertainty: the SE of a contrast is recomputed at the ends of the
      circuit-SD interval, and with the run-slope SD raised 50% on top.
    * More effects (0.006 to 0.03 s/lap, the range of the pooled adjacent
      compound gaps) and a Type-I error estimate from ``size_reps`` replicates.

    Two evidence tiers, kept apart. ``analytic`` uses the generalised-least-squares
    covariance at KNOWN variances (an upper bound, no Monte Carlo error);
    ``simulated`` draws data, REFITS the variances and applies the Wald test, with
    Monte-Carlo SE on every power and on the crossing.
    """
    from scipy.stats import norm

    from ..validation import power as P

    noms = nominations_from_design(design)
    base = P.MixedParams.from_fit(variance)
    scen = {
        "point": base,
        "circuit_sd_low": replace(base, circuit_sd=circuit_sd_ci[0]),
        "circuit_sd_high": replace(base, circuit_sd=circuit_sd_ci[1]),
        "circuit_high_and_run_slope_x1.5": replace(base, circuit_sd=circuit_sd_ci[1],
                                                   run_slope_sd=base.run_slope_sd * 1.5),
    }
    pairs = [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5")]
    fine = [26 * k for k in (2, 4, 8, 16, 32, 64, 96)]  # SE is computed here, then interpolated
    dense = list(np.unique(np.round(np.geomspace(52, 2496, 140)).astype(int)))
    designs = {"balanced": ("balanced", {}), "split": ("split", {})}
    for a, b in pairs:
        designs[f"realistic_{a}_{b}"] = ("realistic", {"nominations": noms, "pair": (a, b)})

    def power_from_se(se, effect):
        z = effect / se
        return float(norm.sf(1.96 - z) + norm.cdf(-1.96 - z))

    t = time.time()
    se_table: dict = {}
    snames = list(scen)
    for dname, (kind, kw) in designs.items():
        per_n = [P.analytic_se_multi(n, kind, [scen[s_] for s_ in snames], **kw) for n in fine]
        for j, sname in enumerate(snames):
            se_table[(dname, sname)] = [row[j] for row in per_n]
    analytic: dict = {}
    for (dname, sname), ses in se_table.items():
        # SE falls close to a power law in N, so interpolate it on log-log axes
        # and read the crossing off a dense grid rather than the coarse one.
        se_dense = np.exp(np.interp(np.log(dense), np.log(fine), np.log(ses)))
        for e in effects:
            pw = [power_from_se(s_, e) for s_ in se_dense]
            c = P.crossing_linear(dense, pw)
            analytic.setdefault(f"effect_{e}", {}).setdefault(dname, {})[sname] = _r(c, 0) if c else None
    se_floor = {
        s_: _r(2 * prm.circuit_sd / np.sqrt(13), 5) for s_, prm in scen.items()
    }
    ceiling = {
        f"effect_{e}": {s_: _r(P.power_ceiling_split(e, prm), 3) for s_, prm in scen.items()} for e in effects
    }

    # ---- simulation at points bracketing each analytic crossing
    mults = (0.6, 1.0, 1.5, 2.2)
    sim_designs = {
        "balanced": ("balanced", {}),
        f"realistic_{sim_pair[0]}_{sim_pair[1]}": ("realistic", {"nominations": noms, "pair": tuple(sim_pair)}),
    }
    root = np.random.SeedSequence(seed)
    cells = [(dn, e) for dn in sim_designs for e in effects]
    ssq = root.spawn(len(cells) + 2)
    jobs, keys, skipped = [], [], []
    chunk = 10
    for (dn, e), ss in zip(cells, ssq[: len(cells)], strict=True):
        c0 = analytic[f"effect_{e}"][dn]["point"] or 832
        grid = sorted({max(52, int(round(c0 * m / 26)) * 26) for m in mults})
        for n, ps in zip(grid, ss.spawn(len(grid)), strict=True):
            if n > max_sim_stints:
                skipped.append({"design": dn, "effect": e, "n_stints": n})
                continue
            sims = ps.spawn(reps)
            for i in range(0, reps, chunk):
                kind, kw = sim_designs[dn]
                jobs.append((n, e, kind, base, sims[i:i + chunk], kw.get("nominations"), kw.get("pair")))
                keys.append((dn, e, n))
    for dn, ss in zip(sim_designs, ssq[len(cells)].spawn(len(sim_designs)), strict=True):
        sims = ss.spawn(size_reps)
        kind, kw = sim_designs[dn]
        for i in range(0, size_reps, chunk):
            jobs.append((832, 0.0, kind, base, sims[i:i + chunk], kw.get("nominations"), kw.get("pair")))
            keys.append((dn, 0.0, 832))
    res = _run_sharded(None, None, None, _power_task_x, jobs, workers)
    acc: dict = {}
    for k, r in zip(keys, res, strict=True):
        a = acc.setdefault(k, {"z": [], "t": []})
        a["z"] += r["reject_z"]
        a["t"] += r["reject_t"]

    simulated: dict = {}
    for dn, e in cells:
        pts = sorted(n for (d_, e_, n) in acc if d_ == dn and e_ == e)
        rz = [acc[(dn, e, n)]["z"] for n in pts]
        rt_ = [acc[(dn, e, n)]["t"] for n in pts]
        cz = P.crossing_with_mc_error(pts, rz, seed=seed)
        ct = P.crossing_with_mc_error(pts, rt_, seed=seed)
        simulated.setdefault(f"effect_{e}", {})[dn] = {
            "n_stints": pts, "reps": reps,
            "power_wald_z": [_r(np.mean(x), 3) for x in rz],
            "power_mc_se": [_r(math.sqrt(max(np.mean(x) * (1 - np.mean(x)), 1e-9) / len(x)), 3) for x in rz],
            "power_t_df12": [_r(np.mean(x), 3) for x in rt_],
            "crossing_80_wald_z": _round_dict(cz), "crossing_80_t_df12": _round_dict(ct),
        }
    size = {
        dn: {
            "n_stints": 832, "reps": size_reps,
            "wald_z_reject_rate": _r(np.mean(acc[(dn, 0.0, 832)]["z"]), 3),
            "wald_z_mc_se": _r(math.sqrt(0.05 * 0.95 / size_reps), 3),
            "t_df12_reject_rate": _r(np.mean(acc[(dn, 0.0, 832)]["t"]), 3),
            "nominal": 0.05,
        }
        for dn in sim_designs
    }
    return {
        "params_per_lap": {k_: {"circuit_sd": _r(v.circuit_sd, 5), "run_slope_sd": _r(v.run_slope_sd, 5)}
                           for k_, v in scen.items()},
        "design_notes": (
            "13 events with the actual 2026 Pirelli nominations; 20-lap stints; N counts stints of "
            "the two compared compounds; balanced = both at every circuit; split = disjoint circuits"
        ),
        "analytic_crossing_80pct_stints": analytic,
        "analytic_se_floor_split_design_s_per_lap": se_floor,
        "split_design_power_ceiling": ceiling,
        "simulated": simulated,
        "simulation_cap_stints": max_sim_stints,
        "simulated_points_skipped_above_cap": skipped,
        "type_I_error_at_zero_effect": size,
        "analytic_grid_stints": fine,
        "seconds": round(time.time() - t, 1),
    }


def _mde_from_se(se: float, power: float = 0.80) -> float:
    """Smallest effect whose two-sided Wald-z test (alpha 0.05) has ``power``."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    def pw(e):
        z = e / se
        return norm.sf(Z95 - z) + norm.cdf(-Z95 - z) - power

    return float(brentq(pw, 1e-9, 50 * se))


def min_detectable_effect(
    variance: dict, circuit_sd_ci: tuple[float, float], design: pd.DataFrame, n_runs: dict,
    workers: int, seed: int, reps: int = 1000, n_balanced: int = 832, sim_pairs=(("C1", "C2"), ("C3", "C4")),
) -> dict:
    """Smallest compound gap detectable with 80% power at the runs we actually hold.

    Analytic (known variances, GLS covariance), for every design and for the
    variance scenarios of ``power_extended``: balanced at ``n_balanced``
    stints (the figure used in the first power analysis), realistic at that
    same N and at the pair's ACTUAL run count (``n_runs[a] + n_runs[b]``, the
    stints of the two compared compounds we really have). Simulated (variances
    refitted, ``reps`` experiments per point, Monte-Carlo SE) for the balanced
    design and for the weakest- and best-linked realistic pairs, at an effect
    grid bracketing the analytic value, read off by interpolation.
    """
    from ..validation import power as P

    noms = nominations_from_design(design)
    base = P.MixedParams.from_fit(variance)
    scen = {
        "point": base,
        "circuit_sd_low": replace(base, circuit_sd=circuit_sd_ci[0]),
        "circuit_sd_high": replace(base, circuit_sd=circuit_sd_ci[1]),
        "circuit_high_and_run_slope_x1.5": replace(base, circuit_sd=circuit_sd_ci[1],
                                                   run_slope_sd=base.run_slope_sd * 1.5),
    }
    snames = list(scen)
    pairs = [("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5")]
    t = time.time()
    analytic: dict = {}
    n_of = {"balanced": n_balanced, "split": n_balanced}
    for dname, kind, kw in [("balanced", "balanced", {}), ("split", "split", {})] + [
        (f"realistic_{a}_{b}", "realistic", {"nominations": noms, "pair": (a, b)}) for a, b in pairs
    ]:
        ns = {"at_N_%d" % n_balanced: n_balanced}
        if kind == "realistic":
            a, b = kw["pair"]
            ns["at_actual_pair_runs"] = int(n_runs[a] + n_runs[b])
        for lab, n in ns.items():
            ses = P.analytic_se_multi(n, kind, [scen[x] for x in snames], **kw)
            analytic.setdefault(dname, {})[lab] = {
                "n_stints": int(n),
                **{sn: {"se_s_per_lap": _r(se, 5), "mde_80_s_per_lap": _r(_mde_from_se(se), 4)}
                   for sn, se in zip(snames, ses, strict=True)},
            }

    # ---- simulation: refit the variances, Wald z and t(12)
    cells = [("balanced", "balanced", {}, n_balanced)] + [
        (f"realistic_{a}_{b}", "realistic", {"nominations": noms, "pair": (a, b)}, int(n_runs[a] + n_runs[b]))
        for a, b in sim_pairs
    ]
    mults = (0.6, 0.85, 1.1, 1.5, 2.0)
    root = np.random.SeedSequence(seed)
    ssq = root.spawn(len(cells))
    jobs, keys, grids = [], [], {}
    chunk = 10
    for (dn, kind, kw, n), ss in zip(cells, ssq, strict=True):
        key = "at_N_%d" % n_balanced if dn == "balanced" else "at_actual_pair_runs"
        mde0 = analytic[dn][key]["point"]["mde_80_s_per_lap"]
        grid = [round(mde0 * m_, 4) for m_ in mults]
        grids[dn] = (grid, n)
        for e, ps in zip(grid, ss.spawn(len(grid)), strict=True):
            sims = ps.spawn(reps)
            for i in range(0, reps, chunk):
                jobs.append((n, e, kind, base, sims[i:i + chunk], kw.get("nominations"), kw.get("pair")))
                keys.append((dn, e))
    res = _run_sharded(None, None, None, _power_task_x, jobs, workers)
    acc: dict = {}
    for k, r in zip(keys, res, strict=True):
        a = acc.setdefault(k, {"z": [], "t": [], "bad": 0})
        a["z"] += r["reject_z"]
        a["t"] += r["reject_t"]
        a["bad"] += r["n_failed_fits"]
    simulated = {}
    for dn, (grid, n) in grids.items():
        rz = [acc[(dn, e)]["z"] for e in grid]
        rt_ = [acc[(dn, e)]["t"] for e in grid]
        simulated[dn] = {
            "n_stints": n, "reps": reps, "effects": grid,
            "power_wald_z": [_r(np.mean(x), 3) for x in rz],
            "power_mc_se": [_r(math.sqrt(max(np.mean(x) * (1 - np.mean(x)), 1e-9) / len(x)), 3) for x in rz],
            "power_t_df12": [_r(np.mean(x), 3) for x in rt_],
            "mde_80_wald_z": _round_dict(P.crossing_with_mc_error(grid, rz, seed=seed)),
            "mde_80_t_df12": _round_dict(P.crossing_with_mc_error(grid, rt_, seed=seed)),
            "n_failed_fits": int(sum(acc[(dn, e)]["bad"] for e in grid)),
        }
    return {
        "n_runs_by_compound": {k_: int(v) for k_, v in n_runs.items()},
        "n_balanced_stints": n_balanced,
        "analytic_known_variance": analytic,
        "simulated_refit": simulated,
        "note": "mde_80 = smallest compound gap (s/lap) with 80% power; analytic uses known variances "
                "(a lower bound on the detectable gap), simulated refits them. 'crossing' keys are effects, not stints.",
        "seconds": round(time.time() - t, 1),
    }


def season_c_rates(pooled: pd.DataFrame) -> dict:
    """Per-season compound rates keyed on PHYSICAL C numbers (not labels).

    The label-keyed table in ``season_robustness`` pools different rubber under
    one name; this one uses Pirelli's nominations to put each lap on its real
    compound. Each season is fitted on its own with circuit slopes (one race per
    circuit within a season), run terms as in the shipped model. C-numbers are
    comparable within a season more safely than across the 2025 re-specification.
    """
    out = {}
    for season, d in sorted(pooled.groupby("season")):
        d = d.reset_index(drop=True)
        f = fit_degradation(d, quadratic=False, circuit_effects=True)
        out[str(season)] = {
            "n_events": int(d["event"].nunique()), "n_laps": int(len(d)),
            "n_runs": int(d["run_id"].nunique()),
            "rates_s_per_lap": {
                c: {"mean": _r(v.mean), "lo": _r(v.lo), "hi": _r(v.hi), "n_runs": f.n_runs.get(c)}
                for c, v in f.rates.items()
            },
            "monotone_softer_degrades_faster": bool(f.is_monotone()),
            "circuit_slope_sd_s_per_lap": _r(f.variance.get("circuit_slope")),
            "convergence_all_ok": bool(f.convergence.get("all_ok")),
        }
    return out


# ---------------------------------------------------------------------------
# 10. what the data we HAVE can detect; is a run's pace independent of its tyre age?
# ---------------------------------------------------------------------------

ADJACENT_PAIRS = (("C1", "C2"), ("C2", "C3"), ("C3", "C4"), ("C4", "C5"))
Z80 = float(stats.norm.ppf(0.80))
#: Multiplier turning a standard error into the minimum detectable effect at 80%
#: power for a two-sided 5% Wald test: (z_0.975 + z_0.80) = 2.8016.
MDE80_MULT = Z95 + Z80


def _detect_row(est: float, se: float, a: str, b: str, runs: dict, events: dict, n_units: int) -> dict:
    """One adjacent contrast: estimate, interval, Wald test and detectability."""
    z = est / se if se > 0 else float("nan")
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else None
    t_df = max(int(n_units) - 1, 1)
    # A t reference with (blocks - 1) degrees of freedom, for scale: with a handful
    # of independent races the z interval is optimistic. Same (t_0.975 + t_0.80) * SE form.
    mde_t = float((stats.t.ppf(0.975, t_df) + stats.t.ppf(0.80, t_df)) * se)
    return {
        "estimate": _r(est), "se": _r(se), "lo": _r(est - Z95 * se), "hi": _r(est + Z95 * se),
        "z": _r(z, 3), "p_wald": float(f"{p:.3g}") if p is not None else None,
        "mde80": _r(MDE80_MULT * se),
        "mde80_t_reference": _r(mde_t), "t_reference_df": t_df,
        "n_runs": {a: int(runs.get(a, 0)), b: int(runs.get(b, 0))},
        "n_events": {a: int(events.get(a, 0)), b: int(events.get(b, 0))},
        "n_blocks_carrying_either_compound": int(n_units),
    }


def detectable_from_fit(design: pd.DataFrame, pooled: pd.DataFrame, shipped=None) -> dict:
    """What can the data we actually have detect, compound pair by compound pair?

    ``power_extended`` answers an idealised question: how many 20-lap stints,
    spread evenly over the races that nominated a compound, would a study need?
    This answers the plain one. For each pair of neighbouring compounds (C2-C1,
    C3-C2, C4-C3, C5-C4) it takes the gap in wear rate the fitted model
    estimates, the standard error the model itself attaches to that gap, and
    reports the smallest TRUE gap the same test would flag 80% of the time:
    ``MDE80 = (z_0.975 + z_0.80) x SE``, about 2.8 standard errors. A gap
    smaller than that could easily be there and still not show up. Nothing is
    simulated; the standard error comes straight from the fit's own covariance
    matrix, which already contains the circuit, driver, run and run-slope
    variance. Done twice: for the shipped 2026 fit (13 races) and for the
    pooled 2022-26 nested fit (circuit plus race slopes, 23 circuits), with the
    number of runs each compound has behind it.

    Caveats stated in the output: the covariance treats the fitted variance
    components as known, and the interval and test are Wald z, so with few
    independent races both are optimistic; ``mde80_t_reference`` gives the same
    quantity with a t reference on (races or circuits carrying either compound
    - 1) degrees of freedom to show the size of that effect.
    """
    fit = shipped if shipped is not None else fit_degradation(design, quadratic=False, circuit_effects=True)
    d26 = design.reset_index(drop=True)
    rows_26 = {}
    for a, b in ADJACENT_PAIRS:
        if a not in fit.rates or b not in fit.rates:
            continue
        i, j = fit.rate_names.index(a), fit.rate_names.index(b)
        v = fit.rate_cov[i, i] + fit.rate_cov[j, j] - 2 * fit.rate_cov[i, j]
        se = math.sqrt(max(float(v), 0.0))
        est = fit.rates[b].mean - fit.rates[a].mean
        n_ev = len(set(fit.events.get(a, [])) | set(fit.events.get(b, [])))
        rows_26[f"{b}-{a}"] = _detect_row(
            est, se, a, b, fit.n_runs, {c: len(fit.events.get(c, [])) for c in (a, b)}, n_ev)

    p = pooled.reset_index(drop=True)
    m, meta, nf = _fit_nested(p)
    runs = p.groupby("C")["run_id"].nunique().to_dict()
    events = p.groupby("C")["event_key"].nunique().to_dict()
    circuits = p.groupby("C")["circuit"].agg(set).to_dict()
    rows_pool = {}
    for a, b in ADJACENT_PAIRS:
        if ("tl", a) not in meta["names"] or ("tl", b) not in meta["names"]:
            continue
        c = lincomb(nf, meta, {("tl", b): 1.0, ("tl", a): -1.0})
        rows_pool[f"{b}-{a}"] = _detect_row(
            c["est"], c["se"], a, b, runs, events, len(circuits.get(a, set()) | circuits.get(b, set())))
    return {
        "what": "Smallest true gap in wear rate between neighbouring compounds that the REAL data would "
                "detect with 80% power (two-sided 5% Wald test), from each fit's own covariance; "
                "not a simulation and not the idealised design of power_extended.",
        "mde80_formula": "MDE80 = (z_0.975 + z_0.80) x SE",
        "z_0.975": _r(Z95, 6), "z_0.80": _r(Z80, 6), "multiplier": _r(MDE80_MULT, 4),
        "units": "seconds per lap of tyre age; contrast 'Cb-Ca' is rate(Cb) minus rate(Ca)",
        "shipped_2026": {
            "specification": "shipped race fit: compound offsets and tyre-age slopes, traffic, random "
                             "circuit slope, driver-in-event, run intercept, run slope; REML",
            "n_races": int(d26["event"].nunique()), "n_runs": int(d26["run_id"].nunique()),
            "block_for_t_reference": "race (one per circuit in 2026)",
            "contrasts": rows_26,
        },
        "pooled_2022_26_nested": {
            "specification": "circuit random slope plus race-within-circuit random slope, C-number "
                             "compounds, run terms as shipped; REML",
            "n_races": int(p["event_key"].nunique()), "n_circuits": int(p["circuit"].nunique()),
            "n_runs": int(p["run_id"].nunique()),
            "block_for_t_reference": "circuit",
            "contrasts": rows_pool,
        },
        "caveats": "Wald z with fitted variance components treated as known; the t-reference column shows how "
                   "much larger the detectable gap is when the number of independent races or circuits is "
                   "small.",
    }


# ---- Mundlak / correlated random effects ----------------------------------


def run_mean_covariates(df: pd.DataFrame, order: tuple[str, ...] = C_ORDER,
                        exact_within: bool = True) -> dict[str, np.ndarray]:
    """Mundlak covariates: each run's mean tyre-age regressor, one column per compound.

    The correlated-random-effects device adds, as FIXED covariates, the mean over
    a run of the regressor the model uses for that run's compound, placed in that
    compound's column (zero for runs of other compounds). A run belongs to one
    compound; a run that spans two raises ``ValueError``.

    With ``exact_within`` (default) the regressor is the exactly within-transformed
    ``tilde(D_c * raw age) / s1`` of ``build_model``, so a run's mean is how old
    its tyres are RELATIVE to the field's compound-c tyres on the same laps. Without
    it, the regressor is the earlier ``tl / s1`` (age minus the cell's mean age).
    """
    d = df.reset_index(drop=True)
    run = d["run_id"].astype(str)
    if (d.groupby(run)["C"].nunique() > 1).any():
        raise ValueError("a run spans more than one compound; the run-mean columns need one each")
    C = d["C"].to_numpy()
    out = {}
    if exact_within:
        dw, codes, ncell = redemean_frame(d)
        s1 = float(dw["tl"].std())
        age = d["TyreLife"].to_numpy(float)
        for c in order:
            if c in set(C):
                x = demean((C == c) * age, codes, ncell) / s1
                out[f"tl_runmean_{c}"] = (C == c) * pd.Series(x).groupby(run.to_numpy()).transform("mean").to_numpy()
        return out
    s1 = float(d["tl"].std())
    rm = (d["tl"] / s1).groupby(run).transform("mean").to_numpy(float)
    return {f"tl_runmean_{c}": (C == c) * rm for c in order if c in set(C)}


def _run_fe_ols(df: pd.DataFrame, cluster_col: str, order: tuple[str, ...] = C_ORDER,
                exact_within: bool = True) -> dict:
    """Eq. 1 with a free intercept per run: within-run demeaning, clustered errors.

    Lap time, compound x tyre age and traffic have the run means removed on top
    of the (event, lap) means, so every stint's pace level is differenced away
    and slopes come from within-stint variation only. ``exact_within`` removes
    both sets of means from the RAW compound x age column together (alternating
    projections, the exact two-way transform); ``False`` is the earlier version,
    which run-demeaned ``D_c * tilde(age)``.

    IDENTIFICATION. With a free intercept for every (event, lap) cell and for every
    run, tyre age is lap minus the run's first lap, so the SUM of the compound
    slopes is not identified (the exact columns add to zero); only the slope
    differences between compounds are. ``level_identified`` is False then, and the
    reported ``rates`` are the minimum-norm solution, i.e. each compound's slope
    minus the mean of the compounds' slopes, with valid standard errors for those
    contrasts. The earlier (approximate) construction reported informative-looking
    levels because ``D_c * tilde(age)`` is not exactly absorbed by the cell effects.
    No assumption about how a run's pace relates to its tyre age is needed.
    Errors are clustered by ``cluster_col`` (CR1) and intervals use t with G-1
    degrees of freedom, G the number of clusters. Runs are nested inside the
    clusters, so the absorbed run effects are not counted against the residual
    degrees of freedom (the usual convention for nested fixed effects).
    """
    from ..models.ablation import cluster_ols

    d = df.reset_index(drop=True)
    cats = [c for c in order if c in set(d["C"])]
    run = d["run_id"].astype(str).to_numpy()
    if exact_within:
        d, codes, _ = redemean_frame(d)
        age = d["TyreLife"].to_numpy(float)
        cols = [(d["C"] == c).to_numpy(float) * age for c in cats]
    else:
        tl = d["tl"].to_numpy(float)
        cols = [(d["C"] == c).to_numpy(float) * tl for c in cats]
    if "tr" in d.columns and float(d["tr"].std() or 0.0) > 1e-9:
        cols.append(d["tr"].to_numpy(float))
    X = np.column_stack(cols)
    y = d["y"].to_numpy(float)
    if exact_within:
        Z = absorb(np.column_stack([X, y]), [np.asarray(codes), run])
        Xd, yd = Z[:, :-1], Z[:, -1]
    else:
        Xd = X - pd.DataFrame(X).groupby(run).transform("mean").to_numpy()
        yd = y - pd.Series(y).groupby(run).transform("mean").to_numpy()
    beta, cc, _ = cluster_ols(Xd, yd, d[cluster_col].astype(str))
    level_identified = bool(np.linalg.matrix_rank(Xd) == Xd.shape[1])
    G = int(d[cluster_col].nunique())
    crit = float(stats.t.ppf(0.975, G - 1))
    out = {}
    for i, c in enumerate(cats):
        se = math.sqrt(max(float(cc[i, i]), 0.0))
        t_ = float(beta[i]) / se if se > 0 else float("nan")
        # A compound carried by few clusters has few independent scores behind its
        # sandwich variance, whatever G is: report that count and the interval with
        # a t reference on (clusters carrying it - 1) df beside the G-1 one.
        g_c = int(d.loc[d["C"] == c, cluster_col].nunique())
        crit_c = float(stats.t.ppf(0.975, max(g_c - 1, 1)))
        out[c] = {"est": float(beta[i]), "se": se, "lo": float(beta[i] - crit * se),
                  "hi": float(beta[i] + crit * se), "t": t_,
                  "p": float(2 * stats.t.sf(abs(t_), G - 1)) if np.isfinite(t_) else None,
                  "n_clusters_with_compound": g_c,
                  "lo_conservative": float(beta[i] - crit_c * se), "hi_conservative": float(beta[i] + crit_c * se)}
    return {"rates": out, "n_clusters": G, "t_df": G - 1, "cluster": cluster_col,
            "level_identified": level_identified}


def _fit_re_cre(df: pd.DataFrame, nested: bool, starts_re=None, starts_cre=None):
    """The same specification fitted without (RE) and with (CRE) run-mean covariates."""
    build = build_nested_model if nested else build_model
    m0, meta0 = build(df)
    m1, meta1 = build(df, extra_fixed=run_mean_covariates(df))
    k = len(m0.names)
    default = [np.full(k, 0.5), np.full(k, 0.1), np.linspace(2, 0.2, k)]
    with single_threaded_blas():
        f0 = m0.fit(starts=starts_re if starts_re is not None else default)
        f1 = m1.fit(starts=starts_cre if starts_cre is not None else [f0.theta, *default])
    return m0, meta0, f0, m1, meta1, f1


def _full5(cats, vals, order=C_ORDER) -> np.ndarray:
    """Values for the compounds present, placed on the five-compound axis (NaN if absent)."""
    out = np.full(len(order), np.nan)
    for c, v in zip(cats, vals, strict=True):
        out[order.index(c)] = v
    return out


def _fit_record(meta: dict, f, cre: bool) -> dict:
    """What a bootstrap replicate keeps from one fit: slopes, SEs, SDs, run-mean coefficients."""
    s1 = meta["s1"]
    cats = meta["cats"]
    ridx = [_col(meta, ("tl", c)) for c in cats]
    rec = {
        "rate": _full5(cats, f.beta[ridx] / s1),
        "se": _full5(cats, np.sqrt(np.diag(f.cov_beta)[ridx]) / s1),
        "sd": {n: float(v / s1) if "slope" in n else float(v) for n, v in f.sd.items()},
        "resid": float(np.sqrt(f.sigma2)), "converged": bool(f.converged),
    }
    if cre:
        xi = [_col(meta, ("extra", f"tl_runmean_{c}")) for c in cats]
        cov = np.full((5, 5), np.nan)
        pos = [C_ORDER.index(c) for c in cats]
        cov[np.ix_(pos, pos)] = f.cov_beta[np.ix_(xi, xi)] / s1**2
        rec["delta"] = _full5(cats, f.beta[xi] / s1)
        rec["delta_cov"] = cov
    return rec


def _task_mundlak_boot(seed_seq):
    """Cluster bootstrap replicate: redraw whole races (circuits, pooled), refit RE and CRE.

    A race (circuit) drawn twice becomes two blocks with their own slopes, as in
    ``_task_event_boot``. Runs travel with their race, so run means are
    unchanged; the run-mean columns are rebuilt on the resample because the
    tyre-age scaling is. Both fits start at the full-sample variance ratios.
    """
    df, spec, p = _W["df"], _W["spec"], _W["fit"]
    nested = bool(spec.get("nested"))
    unit = "circuit" if nested else "event"
    rng = np.random.default_rng(seed_seq)
    units = sorted(df[unit].astype(str).unique())
    draw = rng.integers(0, len(units), len(units))
    keys = df[unit].astype(str).to_numpy()
    parts = []
    for k, i in enumerate(draw):
        d = df[keys == units[i]].copy()
        tag = f"#{k}"
        d["run_id"] = d["run_id"].astype(str) + tag
        if nested:
            d["circuit"] = d["circuit"].astype(str) + tag
            d["event_key"] = d["event_key"].astype(str) + tag
        else:
            d["event"] = d["event"].astype(str) + tag
        parts.append(d)
    b = pd.concat(parts, ignore_index=True)
    k_ = len(p["theta_re"])
    build = build_nested_model if nested else build_model
    m0, meta0 = build(b)
    m1, meta1 = build(b, extra_fixed=run_mean_covariates(b))
    with single_threaded_blas():
        f0 = m0.fit(starts=[p["theta_re"], np.full(k_, 0.5)])
        f1 = m1.fit(starts=[f0.theta, p["theta_cre"]])
    return {"re": _fit_record(meta0, f0, False), "cre": _fit_record(meta1, f1, True),
            "n_compounds": len(meta0["cats"])}


def _pct(x, q=(2.5, 97.5), k=5):
    return [_r(np.percentile(x, q[0]), k), _r(np.percentile(x, q[1]), k)]


def _vc_dict(f, meta: dict, nested: bool) -> dict:
    """SDs of every variance component (per lap for slopes), persistent share, compound-rate SD."""
    s1 = meta["s1"]
    cats = meta["cats"]
    ridx = [_col(meta, ("tl", c)) for c in cats]
    rates = f.beta[ridx] / s1
    se = np.sqrt(np.diag(f.cov_beta)[ridx]) / s1
    out = {n: _r(v / s1 if "slope" in n else v) for n, v in f.sd.items()}
    out["resid"] = _r(np.sqrt(f.sigma2))
    if nested:
        c, e = f.sd["circuit_slope"] / s1, f.sd["event_slope"] / s1
        out["persistent_share_of_slope_variance"] = _r(c**2 / (c**2 + e**2 + 1e-18), 3)
    out["compound_rate_sd_raw"] = _r(np.std(rates, ddof=1))
    out["compound_rate_sd_noise_corrected"] = _r(
        np.sqrt(max(np.var(rates, ddof=1) - np.mean(se**2), 0.0)))
    return out


def _vc_from_record(rec: dict, nested: bool) -> dict:
    sd = rec["sd"]
    out = {n: v for n, v in sd.items()}
    out["resid"] = rec["resid"]
    if nested:
        c, e = sd["circuit_slope"], sd["event_slope"]
        out["persistent_share_of_slope_variance"] = c**2 / (c**2 + e**2 + 1e-18)
    r, s = rec["rate"], rec["se"]
    ok = np.isfinite(r)
    if ok.sum() >= 3:
        out["compound_rate_sd_raw"] = float(np.std(r[ok], ddof=1))
        out["compound_rate_sd_noise_corrected"] = float(
            np.sqrt(max(np.var(r[ok], ddof=1) - np.mean(s[ok] ** 2), 0.0)))
    return out


def _boot_wald_t(ok: list, cats: list, delta_hat: np.ndarray, wald_obs: float) -> dict | None:
    """Bootstrap-t calibration of the run-mean Wald statistic.

    Each replicate gives ``W* = (delta* - delta_hat)' V*^-1 (delta* - delta_hat)``,
    the statistic's distribution when the truth is the full-sample estimate.
    The share at or above the observed statistic is a p-value that respects the
    small number of independent races. Only replicates carrying every compound
    in ``cats`` can be used.
    """
    pos = [C_ORDER.index(c) for c in cats]
    delta = np.array([r["cre"]["delta"][pos] for r in ok])
    cov = np.array([r["cre"]["delta_cov"][np.ix_(pos, pos)] for r in ok])
    use = np.isfinite(delta).all(axis=1) & np.isfinite(cov).all(axis=(1, 2))
    if use.sum() < 10:
        return None
    w = []
    for d, V in zip(delta[use] - delta_hat[None, :], cov[use], strict=True):
        try:
            w.append(float(d @ np.linalg.solve(V, d)))
        except np.linalg.LinAlgError:
            continue
    w = np.array(w)
    return {"n_used": int(len(w)), "n_ge": int((w >= wald_obs).sum()),
            "p": _r((1 + (w >= wald_obs).sum()) / (1 + len(w)), 4),
            "wstar_median": _r(np.median(w), 3), "wstar_95": _r(np.percentile(w, 95), 3)}


def _mundlak_boot_summary(ok: list, nested: bool) -> dict:
    """Bootstrap SEs and intervals from converged replicates (on the five-compound axis)."""
    rate_re = np.array([r["re"]["rate"] for r in ok])
    rate_cre = np.array([r["cre"]["rate"] for r in ok])
    diff = rate_cre - rate_re
    fin = np.isfinite(diff)

    def col(fn, a):
        return [fn(a[fin[:, j], j]) if fin[:, j].sum() > 2 else None for j in range(a.shape[1])]

    return {
        "diff_se": col(lambda x: _r(np.std(x, ddof=1)), diff),
        # IQR / 1.349 is the same quantity for a normal and ignores the heavy tail a
        # resample with one race of a rare compound produces.
        "diff_se_iqr": col(lambda x: _r((np.percentile(x, 75) - np.percentile(x, 25)) / 1.349), diff),
        "diff_pct": col(_pct, diff),
        "re_se": col(lambda x: _r(np.std(x, ddof=1)), rate_re),
        "cre_se": col(lambda x: _r(np.std(x, ddof=1)), rate_cre),
        "n_with": [int(fin[:, j].sum()) for j in range(diff.shape[1])],
        "vc": _boot_vc(ok, nested),
    }


def _boot_vc(ok: list, nested: bool) -> dict:
    """Percentile intervals of every variance summary under RE, CRE and their difference."""
    out = {}
    vre = [_vc_from_record(r["re"], nested) for r in ok]
    vcr = [_vc_from_record(r["cre"], nested) for r in ok]
    for key in vre[0]:
        a = np.array([v.get(key, np.nan) for v in vre], float)
        b = np.array([v.get(key, np.nan) for v in vcr], float)
        good = np.isfinite(a) & np.isfinite(b)
        if good.sum() < 3:
            continue
        out[key] = {"re_95": _pct(a[good]), "cre_95": _pct(b[good]),
                    "cre_minus_re_95": _pct((b - a)[good]), "n": int(good.sum())}
    return out


def _mundlak_block(df: pd.DataFrame, nested: bool, b_boot: int, workers: int, seed: int) -> dict:
    """The Mundlak check for one specification (shipped 2026, or pooled nested)."""
    df = df.reset_index(drop=True)
    m0, meta0, f0, m1, meta1, f1 = _fit_re_cre(df, nested)
    cats = meta0["cats"]
    s1 = meta0["s1"]
    k = len(cats)
    xn = [("extra", f"tl_runmean_{c}") for c in cats]
    re = {c: lincomb(f0, meta0, {("tl", c): 1.0}) for c in cats}
    cre = {c: lincomb(f1, meta1, {("tl", c): 1.0}) for c in cats}
    delta = {c: lincomb(f1, meta1, {x: 1.0}) for c, x in zip(cats, xn, strict=True)}
    between = {c: lincomb(f1, meta1, {("tl", c): 1.0, x: 1.0}) for c, x in zip(cats, xn, strict=True)}

    # joint Wald test that every run-mean coefficient is zero (scale free)
    xi = [_col(meta1, x) for x in xn]
    d_, V_ = f1.beta[xi], f1.cov_beta[np.ix_(xi, xi)]
    wald = float(d_ @ np.linalg.solve(V_, d_))
    delta_hat = np.array([delta[c]["est"] for c in cats])

    # analytic Hausman variance of CRE - RE (valid if RE is efficient under H0)
    r0 = [_col(meta0, ("tl", c)) for c in cats]
    r1 = [_col(meta1, ("tl", c)) for c in cats]
    Vd = (f1.cov_beta[np.ix_(r1, r1)] - f0.cov_beta[np.ix_(r0, r0)]) / s1**2
    eig = np.linalg.eigvalsh((Vd + Vd.T) / 2)
    pd_ok = bool(eig.min() > 0)
    diff = np.array([cre[c]["est"] - re[c]["est"] for c in cats])
    hausman_chi2 = float(diff @ np.linalg.solve(Vd, diff)) if pd_ok else None

    # cluster bootstrap over races (circuits when pooled)
    spec = {"nested": True, "order": C_ORDER} if nested else dict(circuit="random", slopes="compound")
    payload = {**_fit_payload(f0), "theta_re": f0.theta, "theta_cre": f1.theta}
    seeds = np.random.SeedSequence(seed).spawn(b_boot)
    t0 = time.time()
    recs = _run_sharded(df, spec, payload, _task_mundlak_boot, seeds, workers) if b_boot else []
    boot_seconds = time.time() - t0
    ok = [r for r in recs if r["re"]["converged"] and r["cre"]["converged"]]
    boot = _mundlak_boot_summary(ok, nested) if ok else None
    wald_t = _boot_wald_t(ok, cats, delta_hat, wald) if ok else None

    slopes = {}
    for i, c in enumerate(cats):
        j = C_ORDER.index(c)
        v = float(Vd[i, i])
        analytic = math.sqrt(v) if v > 0 else None
        use_analytic = pd_ok and analytic is not None
        se_boot = boot["diff_se"][j] if boot else None
        se = analytic if use_analytic else se_boot
        row = {
            "re": {**_iv(re[c]), "se": _r(re[c]["se"])},
            "cre": {**_iv(cre[c]), "se": _r(cre[c]["se"])},
            "cre_minus_re": {
                "estimate": _r(diff[i]), "se": _r(se),
                "lo": _r(diff[i] - Z95 * se) if se else None, "hi": _r(diff[i] + Z95 * se) if se else None,
                "z": _r(diff[i] / se, 3) if se else None,
                "se_method": "analytic_hausman" if use_analytic else "cluster_bootstrap",
                "se_analytic_hausman": _r(analytic), "se_cluster_bootstrap": se_boot,
            },
            "run_mean_coefficient": {**_iv(delta[c]), "se": _r(delta[c]["se"])},
            "between_run_slope_implied": {**_iv(between[c]), "se": _r(between[c]["se"])},
        }
        if boot:
            row["cre_minus_re"]["bootstrap_percentile_95"] = boot["diff_pct"][j]
            row["cre_minus_re"]["se_cluster_bootstrap_iqr"] = boot["diff_se_iqr"][j]
            row["cre_minus_re"]["n_bootstrap_with_compound"] = boot["n_with"][j]
            row["re"]["se_cluster_bootstrap"] = boot["re_se"][j]
            row["cre"]["se_cluster_bootstrap"] = boot["cre_se"][j]
        slopes[c] = row

    wald_out = {
        "statistic": _r(wald, 3), "df": int(k), "p_chi2": float(f"{stats.chi2.sf(wald, k):.3g}"),
        "coefficients_tested": [x[1] for x in xn],
        "note": "Wald chi-square from the CRE fit's own covariance, variance components treated as known.",
    }
    if wald_t:
        wald_out["p_bootstrap_t"] = wald_t["p"]
        wald_out["bootstrap_t"] = {kk: vv for kk, vv in wald_t.items() if kk != "p"} | {
            "reading": "Share of cluster-bootstrap replicates whose statistic for (delta* - delta_hat) "
                       "reaches the observed one; calibrates the chi-square to the small number of "
                       "independent races."}

    fe = {"cluster_by_event": _run_fe_ols(df, "event_key" if nested else "event")}
    if nested:
        fe["cluster_by_circuit"] = _run_fe_ols(df, "circuit")
    fe_out = {
        name: {c: {"estimate": _r(v["est"]), "se": _r(v["se"]), "lo": _r(v["lo"]), "hi": _r(v["hi"]),
                   "t": _r(v["t"], 3), "p": float(f"{v['p']:.3g}") if v["p"] is not None else None,
                   "n_clusters_with_compound": v["n_clusters_with_compound"],
                   "lo_conservative": _r(v["lo_conservative"]), "hi_conservative": _r(v["hi_conservative"])}
               for c, v in blk["rates"].items()}
        | {"n_clusters": blk["n_clusters"], "t_df": blk["t_df"],
           "level_identified": blk["level_identified"],
           "level_note": "False: with free cell and run effects the compound slopes are identified only up to "
                         "their common level (age = lap - first lap of the run); the rates shown are each "
                         "slope minus the mean slope" if not blk["level_identified"] else "",
           "note": "lo/hi and p use t with G-1 df; lo_conservative/hi_conservative use t with "
                   "(clusters carrying the compound - 1) df, the honest reading for a compound that few "
                   "clusters carry"}
        for name, blk in fe.items()
    }

    x = df["tl"] / s1
    xbar = x.groupby(df["run_id"].astype(str)).transform("mean")
    runs = df.groupby("C")["run_id"].nunique()
    d0, d1 = m0.diagnostics(f0), m1.diagnostics(f1)
    unit = "circuit" if nested else "race"
    return {
        "design": {
            "n_laps": int(len(df)), "n_runs": int(df["run_id"].nunique()),
            "n_races": int(df["event_key" if nested else "event"].nunique()),
            "n_runs_by_compound": {c: int(runs[c]) for c in cats},
            "sd_of_run_mean_tyre_age_over_sd_within_run": _r(float(xbar.std() / (x - xbar).std()), 3),
        },
        "slopes_s_per_lap": slopes,
        "wald_run_means_zero": wald_out,
        "hausman": {
            "variance_of_difference": "V(CRE) - V(RE) on the compound slopes (RE efficient under H0)",
            "positive_definite": pd_ok, "min_eigenvalue": float(f"{eig.min():.3g}"),
            "diagonal_positive": {c: bool(Vd[i, i] > 0) for i, c in enumerate(cats)},
            "chi2_all_slopes_equal": _r(hausman_chi2, 3), "df": int(k),
            "p_chi2": float(f"{stats.chi2.sf(hausman_chi2, k):.3g}") if pd_ok else None,
            "se_used_for_differences": "analytic_hausman" if pd_ok else "cluster_bootstrap",
        },
        "run_fixed_effects_ols": fe_out,
        "variance_components": {
            "re": _vc_dict(f0, meta0, nested), "cre": _vc_dict(f1, meta1, nested),
            **({"bootstrap_95": boot["vc"]} if boot else {}),
        },
        "bootstrap": {
            "unit": unit, "n_requested": int(b_boot), "n_converged_both": len(ok),
            "note": f"{unit}s redrawn with replacement (one drawn twice counts as two blocks), RE and CRE "
                    "refitted on each resample; replicates where either fit did not converge are dropped; a "
                    "compound absent from a resample is left out of that compound's statistics.",
            "seconds": round(boot_seconds, 1),
        },
        "convergence": {
            "re": {"converged": bool(f0.converged), "hessian_pd": d0["hessian_pd"],
                   "boundary_components": d0["boundary_components"],
                   "start_deviances": [_r(v, 3) for v in f0.start_deviances]},
            "cre": {"converged": bool(f1.converged), "hessian_pd": d1["hessian_pd"],
                    "boundary_components": d1["boundary_components"],
                    "start_deviances": [_r(v, 3) for v in f1.start_deviances]},
        },
    }


def mundlak_check(design: pd.DataFrame, pooled: pd.DataFrame, b_boot: int, b_boot_pooled: int,
                  workers: int, seed: int) -> dict:
    """Does a stint's own pace level depend on its tyre age, and does it matter?

    The race model gives every stint ("run") a random baseline pace and treats it
    as noise. That is safe only if the baseline is unrelated to how old the
    stint's tyres are. Pit-stop selection could break this: drivers whose
    tyres have gone off, or who are stuck behind a slower car, may pit earlier,
    so runs that look old relative to the field are not a random draw. If that
    happens, the compound slopes inherit the mismatch, because a random-effects
    fit leans partly on comparisons BETWEEN stints.

    The standard remedy is Mundlak's correlated random effects: add each run's
    mean tyre age (separately for each compound) as a fixed covariate. The
    compound slopes then come from variation WITHIN each stint only, and the
    coefficients on the run means measure the mismatch. Reported, for the
    shipped 2026 model and the pooled 2022-26 nested model: the compound slopes
    with and without the run means (RE and CRE), their difference with a
    standard error (the analytic Hausman variance when it is positive definite,
    otherwise a cluster bootstrap over races, or circuits when pooled; both
    are always shown), the joint Wald test that all run-mean coefficients are
    zero, the variance components under both, and, as a model-free cross-check,
    a run fixed-effects OLS with errors clustered by race.
    """
    return {
        "question": "Is a run's random pace level unrelated to its tyre age? (Mundlak / correlated random "
                    "effects)",
        "device": "fixed covariates = run mean of the model's own tyre-age regressor for that run's compound "
                  "(tilde(D_c x raw age) / SD, exactly within-transformed), one per compound; slopes then use "
                  "within-run variation only",
        "shipped_2026": _mundlak_block(design, False, b_boot, workers, seed + 3),
        "pooled_2022_26_nested": _mundlak_block(pooled, True, b_boot_pooled, workers, seed + 4),
    }

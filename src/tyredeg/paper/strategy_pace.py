"""Can the fresh-tyre pace gap between compounds be measured from this data?

The strategy layer assumes 0.6 s per compound step (``optimise.PACE_STEP_S``).
It is the weakest input in the layer -- the recommended number of stops can turn
on it -- so before leaving it as an assumption this tries to estimate it two
independent ways and reports what comes back, whichever way that is.

WHAT "PACE STEP" MEANS HERE

The lap-time advantage, at zero tyre age, of a compound one physical step softer
(C4 against C3; SOFT against MEDIUM), at equal everything else.

ESTIMATOR 1: RACE, WITHIN DRIVER AND WITHIN LAP

    lap_time = field(event, lap) + car(event, driver) + step x cstep
               + slope_compound x age + traffic + error

``field(event, lap)`` absorbs fuel, track evolution and the field's mean tyre
state; ``car(event, driver)`` absorbs how quick the car is -- which is exactly
what contaminated the earlier race estimate, because quick cars choose different
compounds. What identifies ``step`` is a driver being faster or slower, relative
to the field on the same lap, on one compound than on another at the same age.
Errors are clustered by (event, driver).

It can still be biased: drivers do not pick a compound at random, but when they
need it (a fresh soft for a late push, a hard when the tyres are shot), and
that timing is correlated with the lap-time advantage it is credited for.

ESTIMATOR 2: PRACTICE, WITHIN DRIVER AND SESSION

For each long run, regress fuel-corrected lap time on tyre age and take the
intercept at age zero. Compare intercepts of different compounds for the same
driver in the same session, with a linear term for time-in-session to absorb
track evolution. The earlier attempt took each run's best lap and got 1.0-1.7 s
per step, because a driver's best lap on each compound comes from different fuel
loads and track states. The intercept is better placed than a best lap, but the
starting fuel of each run is still unknown, so this is expected to be the less
reliable of the two.

Neither is a fit the strategy layer relies on. The result is reported next to
the assumed value and the sensitivity sweep, and the assumption stays an
assumption unless an estimate is both tight and stable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _absorb(X: np.ndarray, groups: list[np.ndarray], iters: int = 100, tol: float = 1e-9) -> np.ndarray:
    """Remove every group's means from every column, to convergence.

    Alternating projections: demean by one grouping, then the other, until the
    result stops moving. Exact for two-way fixed effects as iterations grow.
    """
    X = X.astype(float).copy()
    for _ in range(iters):
        before = X.copy()
        for g in groups:
            codes, n = pd.factorize(g)[0], len(set(g))
            sums = np.zeros((n, X.shape[1]))
            np.add.at(sums, codes, X)
            cnt = np.bincount(codes, minlength=n)[:, None]
            X -= (sums / cnt)[codes]
        if np.abs(X - before).max() < tol:
            break
    return X


def _cluster_ols(y: np.ndarray, X: np.ndarray, cluster: np.ndarray):
    """OLS with a cluster-robust covariance. Returns ``(beta, cov)``."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    codes = pd.factorize(cluster)[0]
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(codes):
        m = codes == c
        sc = X[m].T @ resid[m]
        meat += np.outer(sc, sc)
    g = len(np.unique(codes))
    corr = g / max(g - 1, 1)
    return beta, corr * bread @ meat @ bread


def race_estimate(race: pd.DataFrame, order: tuple[str, ...], min_cars: int = 8) -> dict:
    """Estimator 1. ``race`` is a prepared race frame with raw lap time and tyre life."""
    df = race.dropna(subset=["LapTimeSeconds", "TyreLife", "C"]).copy()
    df = df[df["C"].isin(order)]
    df = df[df["cars_on_lap"] >= min_cars] if "cars_on_lap" in df.columns else df
    # A driver needs two different compounds in a race to contribute a contrast.
    n_c = df.groupby(["event", "Driver"])["C"].transform("nunique")
    df = df[n_c >= 2]
    if len(df) < 200:
        return {"estimable": False, "reason": "too few laps from drivers on two compounds"}

    cstep = df["C"].map({c: i for i, c in enumerate(order)}).to_numpy(float)
    cols = [cstep]
    names = ["step"]
    for c in order:
        if (df["C"] == c).sum() >= 30:
            cols.append(df["TyreLife"].to_numpy(float) * (df["C"] == c).to_numpy())
            names.append(f"age[{c}]")
    if "traffic" in df.columns:
        cols.append(df["traffic"].fillna(0).to_numpy(float))
        names.append("traffic")
    X = np.column_stack(cols)
    y = df["LapTimeSeconds"].to_numpy(float)
    ev, lap, drv = df["event"].to_numpy(), df["LapNumber"].to_numpy(), df["Driver"].to_numpy()
    groups = [np.array([f"{e}|{l}" for e, l in zip(ev, lap, strict=True)]),
              np.array([f"{e}|{d}" for e, d in zip(ev, drv, strict=True)])]
    Z = _absorb(np.column_stack([y, X]), groups)
    beta, cov = _cluster_ols(Z[:, 0], Z[:, 1:], np.array([f"{e}|{d}" for e, d in zip(ev, drv, strict=True)]))
    se = float(np.sqrt(cov[0, 0]))
    return {
        "estimable": True,
        "step_s_per_compound_step": float(beta[0]),
        "se": se,
        "ci95": [float(beta[0] - 1.96 * se), float(beta[0] + 1.96 * se)],
        "n_laps": int(len(df)),
        "n_drivers_events": int(len(set(zip(ev, drv, strict=True)))),
        "n_events": int(df["event"].nunique()),
        "coefficients": {n: float(b) for n, b in zip(names, beta, strict=True)},
        "sign_convention": "negative = the softer compound is faster at equal age",
    }


def run_intercepts(practice: pd.DataFrame, order: tuple[str, ...], min_laps: int = 5) -> pd.DataFrame:
    """Per long run: the fuel-corrected lap time extrapolated to tyre age zero."""
    rows = []
    for rid, g in practice.groupby("run_id"):
        if len(g) < min_laps or g["TyreLife"].nunique() < 3 or g["C"].iloc[0] not in order:
            continue
        b, a = np.polyfit(g["TyreLife"].to_numpy(float), g["y_fuel_corrected"].to_numpy(float), 1)
        rows.append(
            {
                "run_id": rid,
                "event": g["event"].iloc[0],
                "session": g["session"].iloc[0],
                "Driver": g["Driver"].iloc[0],
                "C": g["C"].iloc[0],
                "intercept": float(a),
                "slope": float(b),
                "t0": float(g["SessionElapsed"].min()),
                "n": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def practice_estimate(practice: pd.DataFrame, order: tuple[str, ...]) -> dict:
    """Estimator 2. ``practice`` is a prepared practice frame."""
    r = run_intercepts(practice, order)
    if r.empty:
        return {"estimable": False, "reason": "no usable runs"}
    r["cell"] = r["event"].astype(str) + "|" + r["session"].astype(str) + "|" + r["Driver"].astype(str)
    r["sess"] = r["event"].astype(str) + "|" + r["session"].astype(str)
    within_session = r.groupby("cell")["C"].transform("nunique") >= 2
    design = "driver-session"
    if within_session.sum() >= 20:
        r = r[within_session]
        groups = [r["cell"].to_numpy()]
    else:
        # In practice a driver runs ONE compound per race simulation per session,
        # so there is almost never a within-driver, within-session contrast.
        # Fall back to the same driver on different compounds in DIFFERENT
        # sessions of a weekend, with a fixed effect for each session to absorb
        # its track state. Weaker: the session effect is common to all drivers
        # but the driver's own form and fuel are not.
        r["dcell"] = r["event"].astype(str) + "|" + r["Driver"].astype(str)
        r = r[r.groupby("dcell")["C"].transform("nunique") >= 2]
        design = "driver-weekend, session fixed effects"
        if len(r) < 20:
            return {"estimable": False, "reason": "too few drivers on two compounds in one weekend"}
        groups = [r["dcell"].to_numpy(), r["sess"].to_numpy()]
    r["t0c"] = r["t0"] - r.groupby("sess")["t0"].transform("mean")
    cstep = r["C"].map({c: i for i, c in enumerate(order)}).to_numpy(float)
    X = np.column_stack([cstep, r["t0c"].to_numpy(float)])
    Z = _absorb(np.column_stack([r["intercept"].to_numpy(float), X]), groups)
    beta, cov = _cluster_ols(Z[:, 0], Z[:, 1:], r["sess"].to_numpy())
    se = float(np.sqrt(cov[0, 0]))
    return {
        "estimable": True,
        "step_s_per_compound_step": float(beta[0]),
        "se": se,
        "ci95": [float(beta[0] - 1.96 * se), float(beta[0] + 1.96 * se)],
        "design": design,
        "n_runs": int(len(r)),
        "n_driver_sessions": int(r["cell"].nunique()),
        "n_events": int(r["event"].nunique()),
        "session_time_coefficient": float(beta[1]),
        "sign_convention": "negative = the softer compound is faster at equal age",
    }

"""What does removing the confounders actually buy?

Three estimates of the same quantity -- seconds lost per lap of tyre age, per
physical compound -- on the SAME laps and with the SAME compound grouping:

  naive          lap time regressed on tyre age, nothing else. What a first
                 look at the public FastF1 data gives.
  practitioner   the standard careful analysis: lap time corrected for fuel
                 burn, then regressed on tyre age WITHIN each driver stint
                 (driver x stint fixed effects). It controls fuel and every
                 driver- and car-level constant, which is most of what people
                 mean by "doing it properly". It does NOT control track
                 evolution, traffic or the field's common lap-time swings.
  deconfounded   ours: the (event, lap) within-transformation plus traffic and
                 the mixed model.

An earlier version compared a naive slope pooled by the HARD/MEDIUM/SOFT label
against our C-number slopes. That is not like for like -- the label pools
different rubber -- and a naive strawman flatters any method. Fixed here: all
three use C numbers, all three use the laps the model used, and the practitioner
comparator is the one a reviewer would actually ask for.

Standard errors on the first two are clustered by run, because laps of one stint
are not independent; the naive iid error is also returned so the size of that
correction is visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..artifacts.schema import AblationArtifact, AblationRow, Interval
from ..config import BENCHMARK_TABLE3, MASS_SENSITIVITY_S_PER_KG
from ..data.fuel import race_fuel_kg


def cluster_ols(X: np.ndarray, y: np.ndarray, clusters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS with cluster-robust (CR1) and plain iid covariance.

    Returns ``(beta, cov_cluster, cov_iid)``. CR1 is the Stata default: the
    sandwich with a ``G/(G-1) * (n-1)/(n-k)`` small-sample factor.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    r = y - X @ beta
    codes, _ = pd.factorize(np.asarray(clusters))
    g = int(codes.max()) + 1
    scores = np.zeros((g, k))
    np.add.at(scores, codes, X * r[:, None])
    meat = scores.T @ scores
    c = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov_cluster = c * XtX_inv @ meat @ XtX_inv
    cov_iid = XtX_inv * float(r @ r) / max(n - k, 1)
    return beta, cov_cluster, cov_iid


def _interval(b: float, se: float) -> Interval:
    return Interval(float(b), float(b - 1.96 * se), float(b + 1.96 * se))


def _group_demean(v: np.ndarray, groups) -> np.ndarray:
    return v - pd.Series(v).groupby(np.asarray(groups)).transform("mean").to_numpy()


def naive_by_compound(design: pd.DataFrame) -> dict[str, dict[str, Interval]]:
    """Lap time on tyre age by C number, no controls, intercept per compound."""
    out: dict[str, dict[str, Interval]] = {}
    for c, g in design.groupby("C", observed=True):
        if len(g) < 30 or g["TyreLife"].nunique() < 3:
            continue
        x = g["TyreLife"].to_numpy(float)
        X = np.column_stack([np.ones_like(x), x])
        beta, cc, ci = cluster_ols(X, g["LapTimeSeconds"].to_numpy(float), g["run_id"])
        out[str(c)] = {
            "cluster": _interval(beta[1], float(np.sqrt(cc[1, 1]))),
            "iid": _interval(beta[1], float(np.sqrt(ci[1, 1]))),
        }
    return out


def fuel_corrected_laptime(
    design: pd.DataFrame, laps: pd.DataFrame, s_per_kg: float = MASS_SENSITIVITY_S_PER_KG
) -> np.ndarray:
    """Lap time with the fuel burned so far added back, in seconds.

    Fuel on board comes from the same linear burn the rest of the project uses,
    indexed to the real lap number and to the scheduled race distance (the
    highest lap number seen at that event). Burning fuel makes the car faster,
    so the correction is a PLUS: it removes the improvement and leaves the tyre.
    """
    race = laps[laps["session"] == "R"]
    n_laps = race.groupby("event")["LapNumber"].max()
    out = np.empty(len(design))
    for ev, ix in design.groupby("event").indices.items():
        rows = design.iloc[ix]
        kg = race_fuel_kg(rows["LapNumber"].to_numpy(float), int(n_laps[ev]))
        burned = race_fuel_kg(np.array([1.0]), int(n_laps[ev]))[0] - kg
        out[ix] = rows["LapTimeSeconds"].to_numpy(float) + s_per_kg * burned
    return out


def practitioner_by_compound(
    design: pd.DataFrame, laps: pd.DataFrame, s_per_kg: float = MASS_SENSITIVITY_S_PER_KG
) -> dict[str, Interval]:
    """Fuel-corrected within-stint regression: the standard careful analysis.

    Fuel-corrected lap time on tyre age after removing each driver-stint's own
    mean from both (the driver x stint fixed effect), per compound, with
    run-clustered errors. Track evolution is NOT controlled, which is why it
    still reads low where the track rubbers in during a stint.
    """
    d = design.copy()
    d["yc"] = fuel_corrected_laptime(d, laps, s_per_kg)
    stint = (d["event"].astype(str) + "|" + d["Driver"].astype(str) + "|" + d["Stint"].astype(str)).to_numpy()
    out: dict[str, Interval] = {}
    for c, g in d.groupby("C", observed=True):
        st = stint[d.index.get_indexer(g.index)]
        if len(g) < 30 or g["TyreLife"].nunique() < 3:
            continue
        x = _group_demean(g["TyreLife"].to_numpy(float), st)
        y = _group_demean(g["yc"].to_numpy(float), st)
        if float(x @ x) <= 0:
            continue
        beta, cc, _ = cluster_ols(x[:, None], y, g["run_id"])
        out[str(c)] = _interval(beta[0], float(np.sqrt(cc[0, 0])))
    return out


def naive_rates(laps: pd.DataFrame) -> dict[str, Interval]:
    """LEGACY label-pooled naive slope, kept for the benchmark comparison only.

    Pools by the HARD/MEDIUM/SOFT label, which mixes different rubber across
    events, so it is NOT a fair comparator for a C-number fit. ``build`` no
    longer uses it.
    """
    df = laps[laps["is_long_run"] & (laps["session"] == "R")].copy()
    df = df.dropna(subset=["TyreLife", "LapTimeSeconds", "Compound"])

    out: dict[str, Interval] = {}
    for label, g in df.groupby("Compound"):
        if len(g) < 30 or g["TyreLife"].nunique() < 3:
            continue
        x = g["TyreLife"].to_numpy(float)
        X = np.column_stack([np.ones_like(x), x])
        beta, _, cov = cluster_ols(X, g["LapTimeSeconds"].to_numpy(float), np.arange(len(g)))
        out[str(label)] = _interval(beta[1], float(np.sqrt(cov[1, 1])))
    return out


#: Display label for a C number: the most common nomination across the 2026
#: events. Only used to caption the chart; no estimate is pooled by it.
LABEL_FOR = {"C2": "HARD", "C3": "HARD", "C4": "MEDIUM", "C5": "SOFT"}


def build(
    laps: pd.DataFrame, fit, context: str = "race", design: pd.DataFrame | None = None
) -> AblationArtifact:
    """Assemble the naive / practitioner / deconfounded comparison.

    Args:
        laps: the clean lap dataset.
        fit: a fitted ``Fit`` from ``models.mixed``.
        design: the prepared race frame the fit used (recomputed if absent), so
            all three estimators see exactly the same laps.
    """
    if design is None:
        from .design import prepare

        design = prepare(laps, "race")
    design = design.reset_index(drop=True)
    naive = naive_by_compound(design)
    prac = practitioner_by_compound(design, laps)

    rows = []
    for c in fit.ordered:
        if c not in naive:
            continue
        rows.append(
            AblationRow(
                compound=c,
                label=LABEL_FOR.get(c),
                naive=naive[c]["cluster"],
                deconfounded=fit.rates[c],
                # The benchmark's published Hard figure, for a third column.
                published=(
                    Interval(**{k: BENCHMARK_TABLE3["HARD"][k] for k in ("mean", "lo", "hi")})
                    if c == "C3"
                    else None
                ),
                practitioner=prac.get(c),
            )
        )

    if not rows:
        raise ValueError("no compound had both a naive and a deconfounded estimate")

    # The headline is the SIGN of the naive slope and the gap to the careful
    # comparator, not the width: pooling circuits honestly makes the global
    # interval wider than a naive fit that pretends they are one track.
    backwards = [r for r in rows if r.naive.mean < 0]
    gaps = {
        r.compound: abs(r.deconfounded.mean - r.practitioner.mean)
        for r in rows
        if r.practitioner is not None
    }
    parts = []
    if backwards:
        worst = min(backwards, key=lambda r: r.naive.mean)
        parts.append(
            f"On {len(backwards)} of {len(rows)} compounds the naive slope comes out NEGATIVE "
            f"({worst.compound} at {worst.naive.mean:+.3f} s/lap), a tyre getting faster as it wears."
        )
    else:
        parts.append("Every naive slope happens to come out positive here.")
    if gaps:
        top = max(gaps, key=gaps.get)
        parts.append(
            f"The fuel-corrected within-stint regression is the fairer comparator; it differs "
            f"from ours by up to {gaps[top]:.3f} s/lap ({top})."
        )
    return AblationArtifact(
        context=context,
        rows=rows,
        caption=(
            "All three estimates use the same laps and the same physical-compound grouping. "
            "The naive slope has no controls, so fuel burn and track evolution -- which both "
            "make laps faster -- cancel part of the rise from wear. " + " ".join(parts)
        ),
    )

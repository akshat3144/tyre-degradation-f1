"""The exact within-(event, lap) transformation of Eq. 1's fixed effects.

WHY THIS MODULE EXISTS

Eq. 1 gives every (event, lap) cell its own intercept, which is the same as
subtracting the cell mean from every variable. That subtraction has to be done
to EVERY fixed-effect column, not only to lap time and tyre age:

    tilde(D_c)         for the compound offsets (one reference compound dropped,
                       because the demeaned dummies sum to zero within a cell),
    tilde(D_c * age)   for the compound slopes, age being RAW tyre age,
    tilde(traffic)     for the traffic covariate,

where D_c is the compound-c dummy and tilde(.) subtracts the cell mean.

The earlier construction used the raw dummy for the offsets and
``D_c * tilde(age)`` for the slopes, with ``tilde(age)`` already demeaned by
``design.race_design``. Those are not the same thing whenever cars on different
compounds share a lap, because ``D_c * tilde(age)`` subtracts the cell's mean age
(over all compounds) from compound c's laps only, while the correct term
subtracts the cell's mean of ``D_c * age`` (zero for the cars on other
compounds). The mismatch is a bias in the compound slopes whose size depends on
how compounds are mixed on a lap (see ``paper/results/recovery.json``).

Random-effect columns need nothing: the circuit slope ``b_e * tilde(age)`` is
exact because ``b_e`` is the same for every car in a cell; run and driver terms
are random with mean zero.

``exact_within=False`` (in ``fit_degradation``, ``build_model``,
``build_nested_model``) keeps the old construction, for a documented comparison
only.

Everything here works on the frame it is given. Cell means are recomputed on
that frame, so a subset (a leave-one-event-out fit, a bootstrap resample with
re-tagged events, compounds dropped for thin support) is transformed exactly
for the sample actually fitted. ``y``, ``tl``, ``tl2`` and ``tr`` are demeaned
again for the same reason; on a frame that has not been subset this is a no-op.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Columns the exact transform needs beyond the usual design columns.
RAW_COLUMNS = ("TyreLife", "LapNumber")


def group_column(df: pd.DataFrame) -> str:
    """The column naming a race: ``event_key`` in a pooled multi-season frame
    (an event name repeats across seasons), ``event`` otherwise."""
    return "event_key" if "event_key" in df.columns else "event"


def cell_codes(df: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Integer code of each row's (race, lap) cell, and the number of cells."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"the exact within-(event, lap) transform needs the raw columns {missing}; "
            "pass exact_within=False to use the old construction on a frame without them"
        )
    key = df[group_column(df)].astype(str).to_numpy(object) + "|" + df["LapNumber"].astype(str).to_numpy(object)
    codes, uniques = pd.factorize(key)
    return codes.astype(np.int64), len(uniques)


def demean(a: np.ndarray, codes: np.ndarray, n: int) -> np.ndarray:
    """Subtract the cell mean from a vector or from every column of a matrix."""
    a = np.asarray(a, float)
    cnt = np.bincount(codes, minlength=n).astype(float)
    if a.ndim == 1:
        return a - (np.bincount(codes, weights=a, minlength=n) / cnt)[codes]
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        out[:, j] = a[:, j] - (np.bincount(codes, weights=a[:, j], minlength=n) / cnt)[codes]
    return out


def redemean_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Copy of ``df`` with ``y``, ``tl``, ``tl2``, ``tr`` demeaned on THIS frame.

    ``tl`` and ``tl2`` are rebuilt from the raw ``TyreLife`` (and its square) so
    that the circuit/run slope columns use exactly ``tilde(age)``. Returns the
    copy, the cell codes and the number of cells.
    """
    codes, n = cell_codes(df)
    out = df.copy()
    age = df["TyreLife"].to_numpy(float)
    out["y"] = demean(df["y"].to_numpy(float), codes, n)
    out["tl"] = demean(age, codes, n)
    if "tl2" in df.columns:
        out["tl2"] = demean(age**2, codes, n)
    if "tr" in df.columns:
        out["tr"] = demean(df["tr"].to_numpy(float), codes, n)
    return out, codes, n


def offset_columns(C: np.ndarray, cats: list[str], codes: np.ndarray, n: int) -> tuple[np.ndarray, list[str]]:
    """``tilde(D_c)`` for every compound except the first (the reference)."""
    ref_dropped = cats[1:]
    if not ref_dropped:
        return np.empty((len(C), 0)), []
    X = np.column_stack([(C == c).astype(float) for c in ref_dropped])
    return demean(X, codes, n), list(ref_dropped)


def absorb(X: np.ndarray, groupings: list[np.ndarray], iters: int = 1000, tol: float = 1e-11) -> np.ndarray:
    """Remove the means of several groupings (e.g. cell and driver-in-event) to convergence.

    Alternating projections; exact for two-way fixed effects as iterations grow.
    Used by the fixed-effects OLS comparators, which absorb more than the cell.
    """
    X = np.asarray(X, float).copy()
    prepared = []
    for g in groupings:
        codes, uniq = pd.factorize(np.asarray(g))
        prepared.append((codes.astype(np.int64), len(uniq)))
    for _ in range(iters):
        before = X.copy()
        for codes, n in prepared:
            X = demean(X, codes, n)
        if np.abs(X - before).max() < tol:
            break
    return X

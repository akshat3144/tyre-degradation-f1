"""A fast REML engine for the degradation model's exact variance structure.

WHY THIS EXISTS WHEN statsmodels ALREADY FITS THE MODEL

``models.mixed.fit_degradation`` is the reference fit and it is what ships. It
takes 6 to 20 seconds on the 2026 race data once run-level intercepts are in,
because statsmodels builds a dense random-effects design for every group on
every likelihood call. That is fine for one fit. It is hopeless for the checks
a paper needs, which are all of the form "simulate a dataset from the fitted
model and refit it, a thousand times": a parametric-bootstrap likelihood-ratio
test for the circuit effect, simulation-based calibration, and a power curve.

This engine fits the SAME model -- Gaussian, a linear mean, variance
components that are all nested inside an event -- in about a second, by
exploiting two facts.

1. Every random effect here (circuit slope, driver-in-event intercept, run
   intercept) belongs to exactly one event, so the covariance matrix is block
   diagonal by event.
2. Inside a block the covariance is ``sigma^2 (I + Z Theta Z')`` with ``Z`` thin
   (about 90 columns against 850 rows). The Woodbury identity then needs only
   ``Z'Z``, ``Z'X``, ``Z'y`` -- sufficient statistics computed once per dataset
   -- and a Cholesky of a 90 x 90 matrix per event per likelihood call.

``sigma^2`` and the fixed effects are profiled out in closed form, so the
optimiser only searches the handful of variance RATIOS ``theta_k = sd_k / sd_eps``,
under a hard lower bound of zero. That bound is what lets the engine fit the
NULL model of a variance-component test, where the parameter sits exactly on the
boundary of the parameter space and a log-variance parameterisation cannot go.

It is validated against statsmodels on the real data (see
``paper/model_checks.py::engine_agreement``) and on synthetic truth
(``tests/test_paper_model.py``). The shipped numbers come from statsmodels; this
exists so that the uncertainty ABOUT those numbers can be measured.
"""

from __future__ import annotations

import contextlib
import ctypes
import glob
import os
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


@contextlib.contextmanager
def single_threaded_blas():
    """Run the body with BLAS limited to one thread, where that can be done.

    The matrices here are a few hundred rows, far too small to parallelise, and
    on a many-core machine OpenBLAS spends most of its time synchronising
    threads: measured 3x slower than one thread. ``threadpoolctl`` is used if
    installed; otherwise the bundled OpenBLAS is asked directly. Best effort --
    if neither works the code is merely slower, never wrong.
    """
    # The import is guarded; the body is NOT. The first version wrapped the yield
    # in ``try/except ImportError``, so an ImportError raised by the caller's own
    # code was swallowed here and the generator then reached its second yield
    # (a RuntimeError at best, a silently dropped exception at worst).
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None
    if threadpool_limits is not None:
        with threadpool_limits(limits=1):
            yield
        return
    handles = []
    try:
        import numpy
        import scipy

        for pkg in (numpy, scipy):
            root = os.path.dirname(os.path.dirname(pkg.__file__))
            for sub in ("numpy.libs", "scipy.libs"):
                for path in glob.glob(os.path.join(root, sub, "*openblas*")):
                    lib = ctypes.CDLL(path)
                    for suffix in ("64_", ""):
                        setter = getattr(lib, f"scipy_openblas_set_num_threads{suffix}", None)
                        getter = getattr(lib, f"scipy_openblas_get_num_threads{suffix}", None)
                        if setter and getter:
                            handles.append((setter, int(getter())))
                            setter(1)
                            break
    except Exception:  # noqa: BLE001 -- purely an optimisation
        handles = []
    try:
        yield
    finally:
        for setter, n in handles:
            setter(n)


@dataclass(frozen=True)
class Comp:
    """One variance component.

    ``groups`` labels the random-effect level of each row (a run id, a driver
    within an event, or the event itself). With ``covariate`` it is a random
    SLOPE on that covariate, otherwise a random intercept. Levels must be
    nested in an event: the engine never looks across events.
    """

    name: str
    groups: np.ndarray
    covariate: np.ndarray | None = None


@dataclass
class RemlFit:
    beta: np.ndarray
    cov_beta: np.ndarray
    sigma2: float
    #: Standard deviation of each component, in the units of the response per
    #: unit of its covariate (seconds, or seconds per lap for a slope).
    sd: dict[str, float]
    theta: np.ndarray
    deviance: float
    converged: bool
    #: Components pinned at zero by the lower bound.
    on_boundary: list[str] = field(default_factory=list)
    n_obs: int = 0
    n_fixed: int = 0
    #: REML deviance reached from each starting point. If they disagree the
    #: surface has more than one optimum and the best one is reported.
    start_deviances: list[float] = field(default_factory=list)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov_beta))


class RemlModel:
    """Gaussian linear mixed model with event-nested variance components."""

    def __init__(self, y, X, event, comps: list[Comp], eliminate: tuple[str, ...] | None = None):
        """
        ``eliminate`` names one or two components whose levels are LEAVES -- a
        run intercept and optionally a run slope on the same runs. Their
        block of ``Z'Z`` is made of 1x1 or 2x2 pieces, one per run, so they can
        be eliminated analytically (a Schur complement) and the dense
        factorisation is left with only the driver and circuit columns. About
        five times faster and exactly equal; ``fit`` is unchanged. Defaults to
        any component called ``run`` / ``run_slope``.
        """
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        self.n, self.p = self.X.shape
        self.comps = list(comps)
        self.names = [c.name for c in self.comps]
        event = np.asarray(event)
        self.events = list(dict.fromkeys(event.tolist()))
        self._idx = [np.flatnonzero(event == e) for e in self.events]

        self._Z: list[np.ndarray] = []
        self._kind: list[np.ndarray] = []
        self._labels: list[list[str]] = []
        for ix in self._idx:
            cols, kinds, labs = [], [], []
            for k, c in enumerate(self.comps):
                g = np.asarray(c.groups)[ix]
                # Levels present in this event, in order of first appearance.
                # np.unique would sort mixed-type labels and fail on them.
                lv, inv = np.unique(g.astype(str), return_inverse=True)
                m = int(inv.max()) + 1
                block = np.zeros((len(ix), m))
                val = 1.0 if c.covariate is None else np.asarray(c.covariate, float)[ix]
                block[np.arange(len(ix)), inv] = val
                cols.append(block)
                kinds.append(np.full(m, k))
                labs.extend(lv.tolist())
            self._labels.append(labs)
            self._Z.append(np.hstack(cols))
            self._kind.append(np.concatenate(kinds))

        if eliminate is None:
            eliminate = tuple(n for n in ("run", "run_slope") if n in self.names)
        self._elim = None
        if eliminate:
            ks = [self.names.index(n) for n in eliminate]
            if len(ks) > 2:
                raise ValueError("at most two eliminated components")
            self._elim = []
            for Z, kind, labs in zip(self._Z, self._kind, self._labels, strict=True):
                cols = [np.flatnonzero(kind == k) for k in ks]
                if len(cols) == 2 and [labs[j] for j in cols[0]] != [labs[j] for j in cols[1]]:
                    raise ValueError("eliminated components must share their levels")
                rest = np.flatnonzero(~np.isin(kind, ks))
                ZtZ = Z.T @ Z
                E1 = cols[0]
                E2 = cols[1] if len(cols) == 2 else None
                self._elim.append({
                    "F": rest, "E1": E1, "E2": E2,
                    "kF": kind[rest], "k1": ks[0], "k2": ks[1] if E2 is not None else None,
                    "FF": ZtZ[np.ix_(rest, rest)],
                    "FE1": ZtZ[np.ix_(rest, E1)],
                    "FE2": ZtZ[np.ix_(rest, E2)] if E2 is not None else None,
                    "s11": np.diag(ZtZ)[E1].copy(),
                    "s12": ZtZ[E1, E2].copy() if E2 is not None else None,
                    "s22": np.diag(ZtZ)[E2].copy() if E2 is not None else None,
                })
        self._XtX = self.X.T @ self.X
        self._ZtX = [Z.T @ self.X[ix] for Z, ix in zip(self._Z, self._idx, strict=True)]
        self._ZtZ = [Z.T @ Z for Z in self._Z]
        self.set_y(self.y)

    # ------------------------------------------------------------------
    def set_y(self, y) -> None:
        """Swap the response, recomputing only what depends on it."""
        self.y = np.asarray(y, dtype=float)
        self._Xty = self.X.T @ self.y
        self._yty = float(self.y @ self.y)
        self._Zty = [Z.T @ self.y[ix] for Z, ix in zip(self._Z, self._idx, strict=True)]

    def _pieces(self, theta: np.ndarray):
        """X'M^-1 X, X'M^-1 y, y'M^-1 y and log|M| for ``M = I + Z Th Z'``."""
        XtMX = self._XtX.copy()
        XtMy = self._Xty.copy()
        ytMy = self._yty
        logdet = 0.0
        if self._elim is not None:
            q = self.p + 1
            G = np.zeros((q, q))
            for el, ZtX, Zty in zip(self._elim, self._ZtX, self._Zty, strict=True):
                g, ld = self._schur(el, ZtX, Zty, theta)
                G += g
                logdet += ld
            return (
                XtMX - G[: self.p, : self.p],
                XtMy - G[: self.p, self.p],
                ytMy - G[self.p, self.p],
                logdet,
            )
        for ZtZ, ZtX, Zty, kind in zip(self._ZtZ, self._ZtX, self._Zty, self._kind, strict=True):
            d = theta[kind]
            A = d[:, None] * ZtZ * d[None, :]
            A[np.diag_indices_from(A)] += 1.0
            c, low = cho_factor(A, lower=True, check_finite=False)
            rhs = np.column_stack([d[:, None] * ZtX, d * Zty])
            sol = cho_solve((c, low), rhs, check_finite=False)
            DZtX, Dzty = d[:, None] * ZtX, d * Zty
            XtMX -= DZtX.T @ sol[:, : self.p]
            XtMy -= DZtX.T @ sol[:, self.p]
            ytMy -= Dzty @ sol[:, self.p]
            logdet += 2.0 * float(np.log(np.diag(c)).sum())
        return XtMX, XtMy, ytMy, logdet

    def _schur(self, el: dict, ZtX: np.ndarray, Zty: np.ndarray, theta: np.ndarray):
        """``R' A^-1 R`` and ``log|A|`` for one event, run columns eliminated."""
        F, E1, E2 = el["F"], el["E1"], el["E2"]
        dF, d1 = theta[el["kF"]], theta[el["k1"]]
        R = np.column_stack([ZtX, Zty])
        RF, R1 = dF[:, None] * R[F], d1 * R[E1]
        two = E2 is not None
        a = 1.0 + d1**2 * el["s11"]
        if two:
            d2 = theta[el["k2"]]
            R2 = d2 * R[E2]
            b = d1 * d2 * el["s12"]
            c = 1.0 + d2**2 * el["s22"]
            det = a * c - b**2
        else:
            det = a

        def inv(P1, P2):
            if not two:
                return P1 / det[:, None], None
            return (c[:, None] * P1 - b[:, None] * P2) / det[:, None], (
                a[:, None] * P2 - b[:, None] * P1
            ) / det[:, None]

        AFE1 = dF[:, None] * el["FE1"] * d1
        AFE2 = dF[:, None] * el["FE2"] * d2 if two else None
        Q1, Q2 = inv(R1, R2 if two else None)
        W1, W2 = inv(AFE1.T, AFE2.T if two else None)
        S = dF[:, None] * el["FF"] * dF[None, :]
        S[np.diag_indices_from(S)] += 1.0
        S -= AFE1 @ W1
        if two:
            S -= AFE2 @ W2
        T = RF - AFE1 @ Q1 - (AFE2 @ Q2 if two else 0.0)
        c_, low = cho_factor(S, lower=True, check_finite=False)
        solT = cho_solve((c_, low), T, check_finite=False)
        G = R1.T @ Q1 + (R2.T @ Q2 if two else 0.0) + T.T @ solT
        logdet = float(np.log(det).sum()) + 2.0 * float(np.log(np.diag(c_)).sum())
        return G, logdet

    def deviance(self, theta: np.ndarray) -> float:
        """REML deviance with beta and sigma^2 profiled out."""
        XtMX, XtMy, ytMy, logdet = self._pieces(np.asarray(theta, float))
        try:
            beta = np.linalg.solve(XtMX, XtMy)
        except np.linalg.LinAlgError:
            return 1e12
        rss = ytMy - beta @ XtMy
        if rss <= 0:
            return 1e12
        df = self.n - self.p
        _, ld = np.linalg.slogdet(XtMX)
        return float(df * np.log(rss / df) + logdet + ld)

    # ------------------------------------------------------------------
    def fit(self, free: list[str] | None = None, starts=None, bound: float = 60.0) -> RemlFit:
        """REML fit. Components not in ``free`` are held at exactly zero.

        Several starting points are tried and the best kept: the profiled
        surface can be flat near the boundary, and one start can stop early
        there -- which is precisely how the shipped statsmodels L-BFGS fit
        returned a circuit-slope SD nearly three times too large.
        """
        with single_threaded_blas():
            return self._fit(free, starts, bound)

    def _fit(self, free, starts, bound) -> RemlFit:
        free = list(self.names if free is None else free)
        ix = [self.names.index(n) for n in free]
        m = len(self.names)

        def full(v: np.ndarray) -> np.ndarray:
            th = np.zeros(m)
            th[ix] = v
            return th

        if starts is None:
            starts = [np.full(len(ix), s) for s in (1.0, 0.3, 0.05)]
        best = None
        devs = []
        for s0 in starts:
            res = minimize(
                lambda v: self.deviance(full(np.maximum(v, 0.0))),
                np.asarray(s0, float),
                method="L-BFGS-B",
                bounds=[(0.0, bound)] * len(ix),
                options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 500},
            )
            devs.append(float(res.fun))
            if best is None or res.fun < best.fun - 1e-9:
                best = res
        theta = full(np.maximum(best.x, 0.0))
        out = self._finish(theta, bool(best.success), free)
        out.start_deviances = devs
        return out

    def _finish(self, theta: np.ndarray, ok: bool, free: list[str]) -> RemlFit:
        XtMX, XtMy, ytMy, _ = self._pieces(theta)
        beta = np.linalg.solve(XtMX, XtMy)
        rss = ytMy - beta @ XtMy
        s2 = rss / (self.n - self.p)
        cov = s2 * np.linalg.inv(XtMX)
        sd = {n: float(theta[i] * np.sqrt(s2)) for i, n in enumerate(self.names)}
        boundary = [n for n in free if sd[n] < 1e-6 * max(1.0, np.sqrt(s2))]
        return RemlFit(
            beta=beta, cov_beta=cov, sigma2=float(s2), sd=sd, theta=theta,
            deviance=self.deviance(theta), converged=ok, on_boundary=boundary,
            n_obs=self.n, n_fixed=self.p,
        )

    def profile_sd(self, name: str, sd_values, free: list[str] | None = None, start=None,
                   passes: int = 3) -> np.ndarray:
        """REML deviance with one component's SD fixed, others re-optimised.

        ``sd_values`` are in the component's own units (s/lap for a slope). The
        SD is held through the ratio ``theta = sd / sigma``, and sigma is itself
        re-estimated, so each grid point solves a small fixed-ratio problem by
        iterating on sigma. Two passes are plenty: sigma barely moves.
        """
        free = list(self.names if free is None else free)
        others = [n for n in free if n != name]
        k = self.names.index(name)
        oix = [self.names.index(n) for n in others]
        m = len(self.names)
        out = []
        if start is None:
            f0 = self.fit(free)
            sig, x_prev = np.sqrt(f0.sigma2), np.full(len(oix), 0.3)
        else:
            # Warm start (theta, sigma) from an existing fit: skips the cold fit
            # and starts every grid point near where the others already are.
            theta0, sig = start
            x_prev = np.asarray(theta0, float)[oix]
        for sdv in sd_values:
            for _ in range(passes):
                def f(v, sig=sig, sdv=sdv):
                    th = np.zeros(m)
                    th[oix] = np.maximum(v, 0.0)
                    th[k] = sdv / sig
                    return self.deviance(th)

                r = minimize(f, x_prev, method="L-BFGS-B",
                             bounds=[(0.0, 60.0)] * len(oix),
                             options={"ftol": 1e-13, "gtol": 1e-8})
                th = np.zeros(m)
                th[oix] = np.maximum(r.x, 0.0)
                x_prev = th[oix].copy()
                th[k] = sdv / sig
                XtMX, XtMy, ytMy, _ = self._pieces(th)
                beta = np.linalg.solve(XtMX, XtMy)
                sig = np.sqrt((ytMy - beta @ XtMy) / (self.n - self.p))
            out.append(float(r.fun))
        return np.array(out)

    # ------------------------------------------------------------------
    def simulate(self, beta, sigma2: float, sd: dict[str, float], rng) -> np.ndarray:
        """Draw a response from the model with the given parameters."""
        y = np.empty(self.n)
        s = np.sqrt(sigma2)
        for Z, kind, ix in zip(self._Z, self._kind, self._idx, strict=True):
            sds = np.array([sd[self.names[k]] for k in kind])
            u = rng.standard_normal(len(kind)) * sds
            y[ix] = self.X[ix] @ np.asarray(beta) + Z @ u + s * rng.standard_normal(len(ix))
        return y

    def conditional_residuals(self, fit: RemlFit) -> np.ndarray:
        """Residuals after the fixed part AND the predicted random effects (BLUPs)."""
        resid = self.y - self.X @ fit.beta
        out = np.empty(self.n)
        for Z, kind, ix in zip(self._Z, self._kind, self._idx, strict=True):
            d = fit.theta[kind]
            A = d[:, None] * (Z.T @ Z) * d[None, :]
            A[np.diag_indices_from(A)] += 1.0
            u = d * np.linalg.solve(A, d * (Z.T @ resid[ix]))
            out[ix] = resid[ix] - Z @ u
        return out

    def blups(self, fit: RemlFit, name: str) -> dict[str, dict[str, float]]:
        """Predicted random effects of one component, as ``{event: {level: value}}``."""
        k = self.names.index(name)
        resid = self.y - self.X @ fit.beta
        out: dict[str, dict[str, float]] = {}
        for e, Z, kind, ix, labs in zip(
            self.events, self._Z, self._kind, self._idx, self._labels, strict=True
        ):
            d = fit.theta[kind]
            A = d[:, None] * (Z.T @ Z) * d[None, :]
            A[np.diag_indices_from(A)] += 1.0
            u = d * np.linalg.solve(A, d * (Z.T @ resid[ix]))
            out[str(e)] = {labs[j]: float(u[j]) for j in np.flatnonzero(kind == k)}
        return out

    def diagnostics(self, fit: RemlFit, rel_step: float = 2e-3) -> dict:
        """Is this really an optimum?  Gradient, curvature and boundary checks.

        Computed on the profiled REML deviance over the variance ratios that are
        NOT sitting on their zero bound (a bound-constrained optimum has a
        nonzero gradient there by construction, so it is checked separately by
        the sign of the one-sided derivative).
        """
        th = fit.theta.copy()
        free = [i for i, n in enumerate(self.names) if th[i] > 1e-8]
        zero = [i for i, n in enumerate(self.names) if th[i] <= 1e-8]

        def f(v):
            t = th.copy()
            t[free] = v
            return self.deviance(t)

        x0 = th[free]
        h = np.maximum(np.abs(x0) * rel_step, 1e-4)
        m = len(free)
        g = np.zeros(m)
        H = np.zeros((m, m))
        f0 = f(x0)
        for i in range(m):
            e = np.zeros(m)
            e[i] = h[i]
            fp, fm = f(x0 + e), f(x0 - e)
            g[i] = (fp - fm) / (2 * h[i])
            H[i, i] = (fp - 2 * f0 + fm) / h[i] ** 2
        for i in range(m):
            for j in range(i + 1, m):
                ei, ej = np.zeros(m), np.zeros(m)
                ei[i], ej[j] = h[i], h[j]
                H[i, j] = H[j, i] = (
                    f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)
                ) / (4 * h[i] * h[j])
        eig = np.linalg.eigvalsh(H) if m else np.array([1.0])
        # One-sided slope into the interior for components held at zero: a true
        # constrained optimum has it non-negative.
        edge = []
        for i in zero:
            t = th.copy()
            t[free] = x0
            t[i] = 1e-3
            edge.append(float((self.deviance(t) - f0) / 1e-3))
        return {
            "grad_max_abs": float(np.abs(g).max()) if m else 0.0,
            "hessian_pd": bool(eig.min() > 0),
            "hessian_min_eig": float(eig.min()),
            "boundary_components": [self.names[i] for i in zero],
            "boundary_slopes_nonnegative": bool(all(e >= -1e-6 for e in edge)),
        }

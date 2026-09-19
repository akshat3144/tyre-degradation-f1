"""The pooled mixed-effects degradation model.

Fast, interpretable, and it emits exactly the deliverable the brief asks for:
a per-compound degradation rate in seconds per lap, with an interval.

Fitted on the design matrices from ``design.py``, where the confounders have
already been removed by construction rather than by modelling them. That is
deliberate -- a fixed effect you subtract cannot be misspecified, and a fuel
coefficient you never estimate cannot be estimated wrongly.

THE MODEL (race, the shipped fit)

    y = compound offset + compound x tyre-age slope + traffic
        + circuit random slope on tyre age        (one per event)
        + driver-in-event random intercept        (a driver's pace that weekend)
        + run random intercept                    (a stint's own pace)
        + run random slope on tyre age            (a stint's own wear rate)
        + noise

``y`` is lap time minus the (event, lap) mean of the field, so fuel, track
evolution and the weather are gone before the model starts. Tyre age is linear.

WHY THE RUN TERMS ARE THERE

Laps of one stint are not independent: after the driver and run intercepts the
lap-to-lap residual autocorrelation inside a run is still clearly positive, and
stints differ in how fast they wear. A model that treats 11,000 laps as 11,000
independent draws reports intervals that are too narrow, because the real
replication is the number of stints and, for the circuit effect, the number of
circuits. The run terms are the inexpensive repair; ``paper/results/model.json``
reports how much each one widens the compound intervals.

HOW IT IS FITTED, AND WHY THE ENGINE CHANGED

The first version used statsmodels ``MixedLM`` with L-BFGS on unscaled tyre age.
It stopped early -- ``converged=False``, a non-positive-definite Hessian, the
optimum "on the boundary" -- and the circuit-slope SD it returned was nearly
three times too large. Two causes: tyre age (SD about 6.5 laps) against a
response in seconds put the variance parameters on wildly different scales, and
L-BFGS stopped on a flat ridge. Rescaling tyre age to unit SD fixed
statsmodels' convergence; adding the run terms made it take 20 s or more (and
over two minutes with the run slope), which is too slow to refit for any check.

So the default engine is ``models.reml`` -- the same Gaussian REML likelihood,
written with the Woodbury identity so one fit takes seconds, with bounded
variance parameters and several starts. It reproduces statsmodels to about four
decimals on the shared specification (checked in ``paper/model_checks.py`` and
in the tests), and it records the evidence that the optimum is real: restart
agreement, gradient, Hessian curvature, boundary status. ``engine="statsmodels"``
is kept as the independent reference.

Tyre age is LINEAR by default. A quadratic term is available because the shape is
an open question, but the curvature it finds is the opposite of a cliff
and the tangent-at-age-zero it makes the headline rate is not a number anyone
wants. See the notes in ``scripts/03_fit_model.py``.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..artifacts.schema import CompoundCurve, CurvePoint, DegradationArtifact, Interval
from .reml import Comp, RemlModel, single_threaded_blas

#: Order matters: hardest to softest. Used for ordering output and for checking
#: whether the fitted rates are monotone in compound hardness.
C_ORDER = ("C1", "C2", "C3", "C4", "C5")

#: Used when compounds are the weekend's HARD/MEDIUM/SOFT labels rather than the
#: physical C numbers, which is the only option for seasons with no allocation
#: table. The labels are relative to each weekend's nomination.
LABEL_ORDER = ("HARD", "MEDIUM", "SOFT")


@dataclass
class Fit:
    """A fitted model plus everything needed to report it honestly."""

    rates: dict[str, Interval]
    curvature: dict[str, Interval]
    #: Fresh-tyre pace of each compound relative to the field average at that
    #: moment, seconds. Softer compounds should be NEGATIVE (faster). Strategy
    #: needs these: without them an optimiser always picks the hardest tyre,
    #: because it only sees that hard tyres degrade more slowly.
    offsets: dict[str, Interval]
    n_laps: dict[str, int]
    n_runs: dict[str, int]
    events: dict[str, list[str]]
    context: str
    quadratic: bool
    converged: bool
    n_obs: int
    n_drivers: int
    #: How much each circuit's tyre-age slope differs from the global one,
    #: s/lap (the predicted random effect, shrunk toward zero). Empty when
    #: circuit effects are off.
    #:
    #: One rate per compound for every track is false: a model without this
    #: blames the rubber for what is partly the track, and a pit call built on
    #: the global average is wrong at the circuits that sit away from it. How
    #: big the circuit effect is, with an interval and a proper test, is in
    #: ``paper/results/model.json`` -- it is NOT the range of these shrunken
    #: predictions, which says nothing about how much circuits really differ.
    circuit_slope: dict[str, float] = field(default_factory=dict)
    #: Compounds in hardest-to-softest order for this fit.
    order: tuple[str, ...] = C_ORDER
    #: Standard deviations of every variance component, seconds (s/lap for the
    #: slopes), plus the residual: keys ``circuit_slope``, ``driver_event``,
    #: ``run``, ``run_slope``, ``resid``.
    variance: dict[str, float] = field(default_factory=dict)
    #: Evidence about the optimiser: see ``models.reml.RemlModel.diagnostics``.
    convergence: dict = field(default_factory=dict)
    #: Covariance of the rate estimates, rows and columns in ``rate_names``
    #: order, so differences between compounds use the right standard error.
    rate_cov: np.ndarray | None = None
    rate_names: tuple[str, ...] = ()

    def rate_for(self, compound: str, event: str | None = None) -> float:
        """Degradation rate for a compound, at a circuit if we know one.

        The global rate answers "how does this compound behave in general".
        It is the wrong answer to "what do we do on Sunday at Monaco", which is
        what the strategy layer asks.
        """
        return self.rates[compound].mean + self.circuit_slope.get(event or "", 0.0)

    def interval_for(self, compound: str, event: str | None = None) -> Interval:
        """``rate_for``, with its band, so a screen can show what it optimised.

        The strategy layer used to optimise on the per-circuit rate and display
        the global one beside it, giving a compound table whose numbers could
        not produce the plan printed underneath.

        The band is SHIFTED, not rescaled. The circuit offset is a fitted mean
        shift, so it moves the whole interval; what it does not do is tell us
        the per-circuit slope is as well determined as the global one, which is
        why nothing here narrows.
        """
        d = self.circuit_slope.get(event or "", 0.0)
        iv = self.rates[compound]
        return Interval(mean=iv.mean + d, lo=iv.lo + d, hi=iv.hi + d)

    @property
    def ordered(self) -> list[str]:
        return [c for c in self.order if c in self.rates]

    def is_monotone(self) -> bool:
        """Do softer compounds degrade faster, as physics expects?"""
        ks = self.ordered
        return all(self.rates[ks[i]].mean <= self.rates[ks[i + 1]].mean for i in range(len(ks) - 1))

    def separated_pairs(self) -> list[str]:
        """Adjacent compound pairs whose 95% intervals do not overlap.

        A conservative rule: two 95% intervals can overlap while the difference
        between them is still significant at 5%. ``contrast`` gives the test.
        """
        ks = self.ordered
        return [
            f"{ks[i]}|{ks[i + 1]}"
            for i in range(len(ks) - 1)
            if not self.rates[ks[i]].overlaps(self.rates[ks[i + 1]])
        ]

    def contrast(self, a: str, b: str) -> Interval | None:
        """Rate of ``b`` minus rate of ``a`` with its own interval.

        Uses the covariance between the two estimates when it is known. The
        rates share a circuit random slope, so ignoring their covariance
        overstates the uncertainty of a difference.
        """
        if a not in self.rates or b not in self.rates:
            return None
        d = self.rates[b].mean - self.rates[a].mean
        if self.rate_cov is not None and a in self.rate_names and b in self.rate_names:
            i, j = self.rate_names.index(a), self.rate_names.index(b)
            v = self.rate_cov[i, i] + self.rate_cov[j, j] - 2 * self.rate_cov[i, j]
            se = math.sqrt(max(v, 0.0))
        else:
            sa = (self.rates[a].hi - self.rates[a].lo) / 3.92
            sb = (self.rates[b].hi - self.rates[b].lo) / 3.92
            se = math.hypot(sa, sb)
        return Interval(d, d - 1.96 * se, d + 1.96 * se)

    def rate_at(self, compound: str, tyre_age: float) -> float:
        """Instantaneous degradation rate at a given tyre age, s/lap.

        With a quadratic term the raw ``rates`` coefficient is the rate at age
        zero, which is not a number anyone wants -- a tyre at zero laps is not
        degrading yet. This is the useful one.
        """
        rate = self.rates[compound].mean
        quad = self.curvature.get(compound)
        return rate + 2 * quad.mean * tyre_age if quad else rate

    def total_loss(self, compound: str, n_laps: int) -> float:
        """Seconds lost across a whole stint of ``n_laps``.

        The most interpretable summary, and the one a race engineer would ask
        for: not a gradient, but how much time this tyre costs you by the end.
        """
        rate = self.rates[compound].mean
        quad = self.curvature.get(compound)
        return rate * n_laps + (quad.mean * n_laps**2 if quad else 0.0)


log = logging.getLogger(__name__)


def _design(df: pd.DataFrame, *, quadratic: bool, with_offsets: bool, order) -> tuple:
    """Fixed-effect matrix with tyre age rescaled to unit SD.

    Rescaling is what rescued convergence: the slope is estimated in units of
    "per SD of tyre age" and converted back afterwards, so every parameter the
    optimiser sees is of order one.
    """
    cats = [c for c in order if c in set(df["C"])]
    s1 = float(df["tl"].std())
    s2 = float(df["tl2"].std()) if quadratic else 1.0
    cols, names = [], []
    if with_offsets:
        for c in cats:
            cols.append((df["C"] == c).to_numpy(float))
            names.append(("off", c))
    for c in cats:
        cols.append((df["C"] == c).to_numpy(float) * df["tl"].to_numpy(float) / s1)
        names.append(("tl", c))
    if quadratic:
        for c in cats:
            cols.append((df["C"] == c).to_numpy(float) * df["tl2"].to_numpy(float) / s2)
            names.append(("tl2", c))
    # Traffic gets a single shared coefficient, not one per compound: dirty air
    # costs the same lap time whatever tyre you are on, and a per-compound
    # traffic term would compete with the degradation slope for the same signal.
    # Only include it if it varies: in a race where nobody ran in dirty air the
    # column is identically zero and the design matrix is singular.
    if "tr" in df.columns and float(df["tr"].std() or 0.0) > 1e-9:
        cols.append(df["tr"].to_numpy(float))
        names.append(("tr", None))
    return np.column_stack(cols), names, s1, s2


def fit_degradation(
    df: pd.DataFrame,
    *,
    quadratic: bool = True,
    context: str = "race",
    with_offsets: bool = True,
    circuit_effects: bool = False,
    run_effects: bool | None = None,
    run_slopes: bool | None = None,
    order: tuple[str, ...] | None = None,
    engine: str = "reml",
) -> Fit:
    """Fit per-compound degradation on a prepared design frame.

    The design is already demeaned, so there is no intercept. Driver-in-event
    random intercepts soak up the fact that some drivers are simply quicker,
    which would otherwise leak into the tyre-age slope whenever quick drivers
    happen to run older tyres.

    Args:
        circuit_effects: let the tyre-age slope vary by circuit, with partial
            pooling: a track with plenty of clean running gets its own
            estimate, a thin or safety-car-shredded one is pulled toward the
            global mean instead of shouting over it.
        run_effects: random intercept per stint. Default: on in a race, off in
            practice (practice is centred within the run, which removes it).
        run_slopes: random tyre-age slope per stint. Default: follows
            ``run_effects``. Only the ``reml`` engine supports it.
        order: compounds hardest to softest; the C numbers by default, pass
            ``LABEL_ORDER`` for HARD/MEDIUM/SOFT.
        engine: ``"reml"`` (default, fast) or ``"statsmodels"`` (reference).

    Returns:
        A ``Fit``. ``rates`` are seconds lost per lap at the start of a stint;
        with the quadratic term, the instantaneous rate at tyre age *a* is
        ``rate + 2 * curvature * a``.
    """
    order = tuple(order or C_ORDER)
    df = df.reset_index(drop=True).copy()
    if run_effects is None:
        run_effects = context == "race" and "run_id" in df.columns
    if run_slopes is None:
        run_slopes = bool(run_effects) and engine == "reml"
    use_circuit = bool(circuit_effects and df["event"].nunique() >= 4)

    X, names, s1, s2 = _design(df, quadratic=quadratic, with_offsets=with_offsets, order=order)
    ev = df["event"].astype(str).to_numpy()
    de = (df["event"].astype(str) + "|" + df["Driver"].astype(str)).to_numpy()
    tl_s = df["tl"].to_numpy(float) / s1

    comps = []
    if use_circuit:
        comps.append(Comp("circuit_slope", ev, tl_s))
    comps.append(Comp("driver_event", de))
    if run_effects:
        comps.append(Comp("run", df["run_id"].astype(str).to_numpy()))
    if run_slopes:
        comps.append(Comp("run_slope", df["run_id"].astype(str).to_numpy(), tl_s))

    if engine == "reml":
        with single_threaded_blas():
            beta, cov, var_sd, blup, conv = _fit_reml(df, X, ev, comps, use_circuit)
    elif engine == "statsmodels":
        if run_slopes:
            raise ValueError("the statsmodels engine cannot fit run slopes in reasonable time")
        beta, cov, var_sd, blup, conv = _fit_statsmodels(
            df, X, names, ev, de, tl_s, use_circuit, run_effects
        )
    else:
        raise ValueError(f"engine must be 'reml' or 'statsmodels', got {engine!r}")

    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    def pull(kind: str, scale: float) -> dict[str, Interval]:
        out = {}
        for i, (k, c) in enumerate(names):
            if k != kind:
                continue
            mean, half = beta[i] / scale, 1.96 * se[i] / scale
            if not (math.isfinite(mean) and math.isfinite(half)):
                # A coefficient with no finite standard error is not identified
                # by the design. Dropping it reports "we could not estimate
                # this" instead of raising, which used to take the whole
                # pipeline down from one degenerate cell.
                log.warning("%s %s has no finite interval in the %s fit; not reporting it",
                            kind, c, context)
                continue
            out[c] = Interval(mean=float(mean), lo=float(mean - half), hi=float(mean + half))
        return out

    rates = pull("tl", s1)
    rate_names = tuple(c for k, c in names if k == "tl" and c in rates)
    ridx = [i for i, (k, c) in enumerate(names) if k == "tl" and c in rates]
    rate_cov = cov[np.ix_(ridx, ridx)] / s1**2 if ridx else None

    circuit_slope = {e: v / s1 for e, v in blup.items()} if use_circuit else {}
    variance = {
        k: (v / s1 if k in ("circuit_slope", "run_slope") else v) for k, v in var_sd.items()
    }

    counts = df.groupby("C", observed=True)
    return Fit(
        rates=rates,
        curvature=pull("tl2", s2) if quadratic else {},
        offsets=pull("off", 1.0) if with_offsets else {},
        n_laps={str(k): int(v) for k, v in counts.size().items()},
        n_runs={str(k): int(v) for k, v in counts["run_id"].nunique().items()},
        events={str(k): sorted(v) for k, v in counts["event"].unique().items()},
        context=context,
        quadratic=quadratic,
        converged=bool(conv.get("converged", False)),
        n_obs=len(df),
        n_drivers=df["Driver"].nunique(),
        circuit_slope=circuit_slope,
        order=order,
        variance=variance,
        convergence=conv,
        rate_cov=rate_cov,
        rate_names=rate_names,
    )


def _fit_reml(df, X, ev, comps, use_circuit):
    """Fit with the fast REML engine and attach the optimiser's evidence."""
    m = RemlModel(df["y"].to_numpy(float), X, ev, comps)
    k = len(comps)
    starts = [np.full(k, 0.5), np.full(k, 0.1), np.linspace(2.0, 0.2, k)]
    f = m.fit(starts=starts)
    d = m.diagnostics(f)
    spread = float(max(f.start_deviances) - min(f.start_deviances))
    # A start that stalls on a flat ridge reports success at a clearly worse
    # deviance -- the failure that sank the first statsmodels fit. That start is
    # simply discarded, but the optimum is only certified if at least two
    # independent starts reached it.
    n_at_best = int(sum(d - min(f.start_deviances) < 1e-4 for d in f.start_deviances))
    names = [c.name for c in comps]
    boundary = list(d["boundary_components"])
    conv = {
        "engine": "reml (Woodbury, bounded variance ratios, L-BFGS-B)",
        "converged": bool(f.converged),
        "n_starts": len(starts),
        # Zero means every start reached the same optimum.
        "start_deviance_spread": round(spread, 8),
        "n_starts_at_optimum": n_at_best,
        "grad_max_abs": d["grad_max_abs"],
        "hessian_pd": d["hessian_pd"],
        "hessian_min_eig": d["hessian_min_eig"],
        "boundary_components": boundary,
        "boundary_slopes_nonnegative": d["boundary_slopes_nonnegative"],
        "variance_components": names,
    }
    conv["all_ok"] = bool(
        f.converged and n_at_best >= 2 and d["hessian_pd"] and d["boundary_slopes_nonnegative"]
    )
    blup = {}
    if use_circuit:
        for e, lv in m.blups(f, "circuit_slope").items():
            blup[e] = float(next(iter(lv.values())))
    sd = {n: f.sd[n] for n in names}
    sd["resid"] = float(np.sqrt(f.sigma2))
    return f.beta, f.cov_beta, sd, blup, conv


def _fit_statsmodels(df, X, names, ev, de, tl_s, use_circuit, run_effects):
    """Reference fit with statsmodels MixedLM, for cross-checking the engine.

    Same specification as the default minus run slopes. Tried with L-BFGS first
    and, if that does not report convergence, BFGS and Powell; each attempt is
    recorded so a non-converging fit can never pass silently.
    """
    import statsmodels.api as sm

    d = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    d["y"] = df["y"].to_numpy(float)
    d["event"] = ev
    d["drv"] = de
    d["tls"] = tl_s
    d["run"] = df["run_id"].astype(str).to_numpy() if run_effects else "r"
    vc = {"driver": "0 + C(drv)"}
    if run_effects:
        vc["run"] = "0 + C(run)"
    exog = d[[f"x{i}" for i in range(X.shape[1])]]
    model = sm.MixedLM.from_formula(
        "y ~ 0 + " + " + ".join(exog.columns), d, groups=d["event"],
        re_formula="~0 + tls" if use_circuit else "~0", vc_formula=vc,
    )
    attempts, best = [], None
    for method in ("lbfgs", "bfgs", "powell"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = model.fit(method=method, reml=True)
        msgs = " ".join(str(x.message) for x in w)
        attempts.append({
            "method": method, "converged": bool(r.converged), "llf": float(r.llf),
            "hessian_not_pd": "positive definite" in msgs,
            "boundary_warning": "boundary" in msgs,
        })
        if best is None or (r.converged and not best[1]) or (
            r.converged == best[1] and r.llf > best[0].llf
        ):
            best = (r, bool(r.converged))
        if r.converged:
            break
    r = best[0]
    beta = r.fe_params.to_numpy()
    cov = r.cov_params().loc[r.fe_params.index, r.fe_params.index].to_numpy()
    sd = {"driver_event": float(np.sqrt(r.vcomp[0])), "resid": float(np.sqrt(r.scale))}
    if run_effects:
        sd["run"] = float(np.sqrt(r.vcomp[1]))
    blup = {}
    if use_circuit:
        sd["circuit_slope"] = float(np.sqrt(np.asarray(r.cov_re).ravel()[0]))
        for e, vals in (r.random_effects or {}).items():
            blup[str(e)] = float(np.asarray(vals)[0])
    conv = {"engine": "statsmodels MixedLM (REML)", "converged": bool(best[1]),
            "attempts": attempts, "llf": float(r.llf),
            "all_ok": bool(best[1] and not attempts[-1]["hessian_not_pd"])}
    return beta, cov, sd, blup, conv


def fuel_sensitivity(
    laps: pd.DataFrame,
    s_per_kg_values=(0.030, 0.033, 0.035),
    **prepare_kw,
) -> pd.DataFrame:
    """Refit practice across the plausible range of the fuel coefficient.

    The practice model assumes a value for seconds-per-kg rather than fitting
    it, because it is not identifiable from lap times. This is how we show
    whether that assumption is doing any work. If the compound ordering holds
    across the range, the conclusion is safe. If it flips, we say so.
    """
    from .design import prepare

    rows = []
    for s in s_per_kg_values:
        # No quadratic: practice runs average seven laps, over which tyre age
        # and its square correlate at 0.99. The curvature is not identifiable
        # there and including it destabilises the linear term.
        fit = fit_degradation(
            prepare(laps, "practice", s_per_kg=s, **prepare_kw),
            context="practice",
            quadratic=False,
            # Practice centres within run, so a compound's level is differenced
            # away before the model sees it. Asking for offsets here yields
            # coefficients of exactly zero with NaN standard errors.
            with_offsets=False,
        )
        for c, iv in fit.rates.items():
            rows.append(
                {
                    "s_per_kg": s,
                    "compound": c,
                    "rate": iv.mean,
                    "lo": iv.lo,
                    "hi": iv.hi,
                    "monotone": fit.is_monotone(),
                }
            )
    return pd.DataFrame(rows)


def to_artifact(fit: Fit, max_life: int = 26, event: str | None = None) -> DegradationArtifact:
    """Package a fit into the ``DegradationArtifact`` format written to ``data/artifacts/``."""
    curves = []
    for c in fit.ordered:
        rate = fit.rates[c]
        quad = fit.curvature.get(c)

        points = []
        for life in range(1, max_life + 1):
            mean = rate.mean * life + (quad.mean * life**2 if quad else 0.0)
            # Widen with tyre age: extrapolating a slope compounds its error.
            half = (rate.hi - rate.lo) / 2 * life
            points.append(
                CurvePoint(
                    tyre_life=life,
                    delta=Interval(
                        mean=round(mean, 5),
                        lo=round(mean - half, 5),
                        hi=round(mean + half, 5),
                    ),
                )
            )

        curves.append(
            CompoundCurve(
                compound=c,
                label=None,  # pooled across events, where labels differ
                context=fit.context,
                rate=Interval(round(rate.mean, 5), round(rate.lo, 5), round(rate.hi, 5)),
                curve=points,
                n_laps=fit.n_laps.get(c, 0),
                n_runs=fit.n_runs.get(c, 0),
                events=fit.events.get(c, []),
            )
        )

    # Normal approximation to the difference between two fitted rates. This is
    # frequentist -- the model is mixed-effects, not Bayesian -- so it is not a
    # posterior probability even though it is on a 0-1 scale. The standard
    # error of the difference uses the covariance between the two estimates.
    separation = {}
    ks = fit.ordered
    for i in range(len(ks) - 1):
        a, b = ks[i], ks[i + 1]
        iv = fit.contrast(a, b)
        se = (iv.hi - iv.lo) / 3.92
        z = iv.mean / se if se > 0 else 0.0
        separation[f"{a}>{b}"] = round(float(0.5 * (1 + math.erf(z / math.sqrt(2)))), 3)

    return DegradationArtifact(
        event=event,
        curves=curves,
        # Absorbed by construction in races; assumed, not fitted, in practice.
        fuel_coefficient=None,
        track_evolution=None,
        separation=separation,
        separated_pairs=fit.separated_pairs(),
        diagnostics={
            "convergence": fit.convergence,
            "variance_sd": {k: round(v, 5) for k, v in fit.variance.items()},
            "variance_units": "seconds; seconds per lap for circuit_slope and run_slope",
            "n_obs": fit.n_obs,
            "specification": (
                "within (event, lap) transform; compound offset + compound x tyre-age slope "
                "+ traffic; random circuit slope, driver-in-event intercept, run intercept, "
                "run slope; REML"
            ),
            "interval": "95% Wald from the REML fixed-effect covariance",
        },
    )

"""How much data does separating two compounds actually need?

This answers the open question the benchmark paper leaves behind. They could not
distinguish a Hard from a Medium and attributed it to "not much data to
differentiate what would likely be a small effect size" -- but they never said
how much data would have been enough.

We can say. Simulate races with the real design, inject a known difference
between two compounds, refit, and count how often it is detected. Repeat across
sample sizes and read off the point where power reaches 80%.

Two things fall out:

  1. Whether the benchmark's null result was inevitable given their three
     stints, or whether they were unlucky. If a design that size has 10% power,
     their finding says nothing about tyres and everything about sample size.

  2. Whether OUR result is trustworthy. A significant finding from an
     underpowered design is more likely to be noise than signal, so this is a
     check on ourselves as much as on them.

TWO CALCULATIONS LIVE HERE, AND ONE IS ONLY A CEILING

``power_curve`` / ``detectable_effect`` are the original: Welch's t on iid
stint slopes. Fast and transparent, and optimistic: no circuit variance, no
stint-level wear variation, every stint an independent draw. Kept for the
benchmark comparison and as the upper bound it is.

``simulate_mixed_power`` is the one the paper uses. It draws data from the
FITTED mixed model (circuit slope, driver-in-event, run intercept, run slope,
lap noise), refits it with the REML engine, and tests the compound contrast
with the same Wald statistic the real analysis reports, on two designs:
``balanced`` (both compounds at every circuit, so the circuit slope cancels
inside the contrast) and ``split`` (each compound at its own circuits, so it
does not: the worst case, and a fair picture of how Pirelli's allocation
confounds compound with track). In the split design no number of laps buys 80%
power at a small effect: more circuits are needed, not more stints.

The 80% crossing is interpolated from the curve, never read off the
power-of-two grid, and carries a Monte-Carlo standard error.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

#: Difference we care about detecting, s/lap. Roughly the gap the benchmark
#: reported between Hard and Medium (0.054 vs 0.060), i.e. the effect size that
#: study was implicitly trying to resolve.
DEFAULT_EFFECT = 0.006

#: Residual lap-time scatter after the design removes shared effects, seconds.
#: Taken from our own race fit; a driver's lap times wobble by roughly this much
#: for reasons that are not the tyre.
DEFAULT_NOISE = 0.55

#: Laps per stint. The benchmark's stints were "around 20 laps".
DEFAULT_STINT_LAPS = 20

#: Effect sizes scanned by ``detectable_effect``. Module-level so it is not
#: recomputed on every call and not a mutable default argument.
EFFECT_GRID = np.linspace(0.001, 0.10, 40)


@dataclass
class PowerCurve:
    effect: float
    noise: float
    stint_laps: int
    n_stints: np.ndarray
    power: np.ndarray
    n_sims: int

    def n_for(self, target: float = 0.80) -> int | None:
        """Smallest number of stints reaching ``target`` power, if any."""
        hit = np.where(self.power >= target)[0]
        return int(self.n_stints[hit[0]]) if len(hit) else None

    def power_at(self, n: int) -> float:
        """Power at a given number of stints, interpolated."""
        return float(np.interp(n, self.n_stints, self.power))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"n_driver_stints": self.n_stints, "power": self.power})


def _one_trial(
    rng: np.random.Generator,
    n_stints: int,
    effect: float,
    noise: float,
    stint_laps: int,
) -> bool:
    """Simulate one experiment. Returns True if the difference was detected.

    Half the stints get compound A, half compound B, differing by ``effect``.
    Each stint has its own baseline pace, which the design removes by centring
    within the stint -- exactly what happens to the real data. So the test sees
    only within-stint slope information, as it does in practice.
    """
    per = max(1, n_stints // 2)
    tyre_age = np.arange(1, stint_laps + 1, dtype=float)
    centred_age = tyre_age - tyre_age.mean()

    slopes = []
    for rate in (0.0, effect):
        for _ in range(per):
            y = rate * centred_age + rng.normal(0, noise, stint_laps)
            # OLS slope through the centred design; no intercept needed.
            slopes.append(float(centred_age @ y / (centred_age @ centred_age)))

    a = np.array(slopes[:per])
    b = np.array(slopes[per:])

    # Two stints per group cannot support a variance estimate at all.
    if per < 2:
        return False

    # Welch's t-test: the two groups need not have equal variance.
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / per + vb / per)
    if se == 0:
        return False
    t = abs(b.mean() - a.mean()) / se

    # The critical value MUST come from the t distribution with Welch-
    # Satterthwaite degrees of freedom, not from 1.96. With two stints per group
    # there is 1 degree of freedom and the true critical value is 12.7, so the
    # normal approximation reports power that does not exist -- it made the
    # curve non-monotonic, showing more power at 4 stints than at 8.
    denom = va**2 / (per**2 * (per - 1)) + vb**2 / (per**2 * (per - 1))
    df = (va / per + vb / per) ** 2 / denom if denom > 0 else 1.0
    return bool(t > student_t.ppf(0.975, max(df, 1.0)))


def power_curve(
    n_stints_grid=(2, 4, 8, 16, 32, 64, 128, 256, 512),
    effect: float = DEFAULT_EFFECT,
    noise: float = DEFAULT_NOISE,
    stint_laps: int = DEFAULT_STINT_LAPS,
    n_sims: int = 600,
    seed: int = 0,
) -> PowerCurve:
    """Detection probability against number of driver-stints."""
    rng = np.random.default_rng(seed)
    powers = [
        np.mean([_one_trial(rng, n, effect, noise, stint_laps) for _ in range(n_sims)])
        for n in n_stints_grid
    ]
    return PowerCurve(
        effect=effect,
        noise=noise,
        stint_laps=stint_laps,
        n_stints=np.array(n_stints_grid),
        power=np.array(powers),
        n_sims=n_sims,
    )


def estimate_noise(df: pd.DataFrame, col: str = "y") -> float:
    """Residual scatter in a prepared design frame.

    Uses the within-run spread of the already-demeaned response, which is what
    the simulation needs: noise left after the confounders are removed.
    """
    return float(df.groupby("run_id")[col].std().median())


def detectable_effect(
    n_stints: int,
    noise: float = DEFAULT_NOISE,
    stint_laps: int = DEFAULT_STINT_LAPS,
    target: float = 0.80,
    grid=EFFECT_GRID,
    n_sims: int = 300,
    seed: int = 1,
) -> float | None:
    """Smallest effect detectable at ``target`` power with ``n_stints``.

    The inverse question, and often the more useful one: given the data we have,
    what size of difference could we have found?
    """
    rng = np.random.default_rng(seed)
    for eff in grid:
        p = np.mean([_one_trial(rng, n_stints, eff, noise, stint_laps) for _ in range(n_sims)])
        if p >= target:
            return float(eff)
    return None


# ---------------------------------------------------------------------------
# Power from the fitted mixed model
# ---------------------------------------------------------------------------

#: Scale used to keep the design's tyre-age column of order one, matching the
#: real fit (SD of the (event, lap)-centred tyre age, laps).
TL_SCALE_LAPS = 6.5


@dataclass(frozen=True)
class MixedParams:
    """Variance components of the fitted model, per lap for the slopes."""

    circuit_sd: float
    run_sd: float
    run_slope_sd: float
    driver_sd: float
    sigma: float

    @staticmethod
    def from_fit(variance: dict) -> MixedParams:
        return MixedParams(
            circuit_sd=float(variance["circuit_slope"]),
            run_sd=float(variance["run"]),
            run_slope_sd=float(variance["run_slope"]),
            driver_sd=float(variance["driver_event"]),
            sigma=float(variance["resid"]),
        )


def _plan(
    design: str, n_stints: int, n_events: int, nominations, pair, stints_per_driver: int
):
    """Who runs which compound where: ``(rows, n_comp, (a, b))``.

    ``rows`` are ``(event, compound, run, driver_event)``. The contrast is always
    compound ``b`` minus compound ``a``.

    ``balanced`` and ``split`` are the two idealisations. ``realistic`` takes the
    actual Pirelli nominations (a tuple of compound names per event, e.g. from
    ``data/allocation.json``) and puts stints only on the compounds each event
    nominated, so the contrast between the pair is identified from the events
    that brought both and, confounded with the circuit, from the rest. ``n_stints``
    is then the number of stints of the TWO compared compounds, as in the other
    designs; the other nominated compounds run the same number per event.
    """
    if design in ("balanced", "split"):
        per_event = max(2, int(round(n_stints / n_events)))
        rows, run_id = [], 0
        for e in range(n_events):
            for j in range(per_event):
                comp = (j % 2) if design == "balanced" else (0 if e < n_events / 2 else 1)
                rows.append((e, comp, run_id, e * 10_000 + j // stints_per_driver))
                run_id += 1
        return rows, 2, (0, 1)
    if design != "realistic":
        raise ValueError("design must be 'balanced', 'split' or 'realistic'")
    if not nominations or pair is None:
        raise ValueError("the realistic design needs nominations and a compound pair")
    a, b = pair
    # Only events that carry at least one of the pair take part, so only their
    # compounds appear (an all-zero column would make the design singular).
    names = sorted({c for nom in nominations if a in nom or b in nom for c in nom})
    idx = {c: i for i, c in enumerate(names)}
    has = [(a in nom, b in nom) for nom in nominations]
    n_both = sum(1 for x, y in has if x and y)
    n_single = sum(1 for x, y in has if x != y)
    k = max(1, int(round(n_stints / max(1, 2 * n_both + n_single))))
    rows, run_id = [], 0
    for e, nom in enumerate(nominations):
        if not (a in nom or b in nom):
            continue
        j = 0
        for c in sorted(nom):
            for _ in range(k):
                rows.append((e, idx[c], run_id, e * 10_000 + j // stints_per_driver))
                run_id += 1
                j += 1
    return rows, len(names), (idx[a], idx[b])


def _power_model(
    n_stints: int, design: str, params: MixedParams, n_events: int, stint_laps: int,
    stints_per_driver: int = 3, nominations=None, pair=None,
):
    """A RemlModel on an idealised experiment plus the true parameters.

    Returns ``(model, sd, sigma2, contrast_weights, slope_columns)``.
    """
    from ..models.reml import Comp, RemlModel

    rows, n_comp, (ia, ib) = _plan(design, n_stints, n_events, nominations, pair, stints_per_driver)
    ev = np.repeat([r[0] for r in rows], stint_laps)
    comp = np.repeat([r[1] for r in rows], stint_laps)
    run = np.repeat([r[2] for r in rows], stint_laps)
    drv = np.repeat([r[3] for r in rows], stint_laps)
    age = np.tile(np.arange(1, stint_laps + 1, dtype=float), len(rows))
    tl = (age - (stint_laps + 1) / 2) / TL_SCALE_LAPS
    X = np.column_stack(
        [comp == c for c in range(n_comp)] + [(comp == c) * tl for c in range(n_comp)]
    ).astype(float)
    comps = [
        Comp("circuit_slope", ev, tl),
        Comp("driver_event", drv),
        Comp("run", run),
        Comp("run_slope", run, tl),
    ]
    model = RemlModel(np.zeros(len(tl)), X, ev, comps, eliminate=("run", "run_slope"))
    sd = {
        "circuit_slope": params.circuit_sd * TL_SCALE_LAPS,
        "driver_event": params.driver_sd,
        "run": params.run_sd,
        "run_slope": params.run_slope_sd * TL_SCALE_LAPS,
    }
    w = np.zeros(2 * n_comp)
    w[n_comp + ia] = -1.0
    w[n_comp + ib] = 1.0
    return model, sd, params.sigma**2, w, n_comp, ib


def analytic_power(
    n_stints: int, effect: float, design: str, params: MixedParams,
    n_events: int = 13, stint_laps: int = DEFAULT_STINT_LAPS, nominations=None, pair=None,
) -> dict:
    """Wald-test power with the variance components KNOWN (an upper bound).

    No simulation: the standard error of the compound contrast is read from the
    generalised-least-squares covariance at the true variance parameters. It
    ignores that the parameters must be estimated, so the simulation, which does
    estimate them, can only come in at or below it.
    """
    from scipy.stats import norm

    model, sd, s2, w, _, _ = _power_model(
        n_stints, design, params, n_events, stint_laps, nominations=nominations, pair=pair
    )
    theta = np.array([sd[n] / np.sqrt(s2) for n in model.names])
    XtMX, _, _, _ = model._pieces(theta)
    cov = s2 * np.linalg.inv(XtMX)
    # The slope columns are scaled by TL_SCALE_LAPS, so the SE is per scaled unit.
    se = float(np.sqrt(w @ cov @ w)) / TL_SCALE_LAPS
    z = effect / se
    return {"se": se, "power": float(norm.sf(1.96 - z) + norm.cdf(-1.96 - z))}


def analytic_se_multi(
    n_stints: int, design: str, params_list: list, n_events: int = 13,
    stint_laps: int = DEFAULT_STINT_LAPS, nominations=None, pair=None,
) -> list[float]:
    """Contrast SE per lap for several parameter sets, building the design once.

    The expensive part of ``analytic_power`` is assembling the design's sufficient
    statistics; only the cheap Cholesky step depends on the variance parameters.
    ``params_list`` entries must share ``sigma`` and the design layout.
    """
    model, _sd, _s2, w, _n, _ib = _power_model(
        n_stints, design, params_list[0], n_events, stint_laps, nominations=nominations, pair=pair
    )
    out = []
    for prm in params_list:
        sd = {
            "circuit_slope": prm.circuit_sd * TL_SCALE_LAPS, "driver_event": prm.driver_sd,
            "run": prm.run_sd, "run_slope": prm.run_slope_sd * TL_SCALE_LAPS,
        }
        s2 = prm.sigma**2
        theta = np.array([sd[n] / np.sqrt(s2) for n in model.names])
        XtMX, _, _, _ = model._pieces(theta)
        cov = s2 * np.linalg.inv(XtMX)
        out.append(float(np.sqrt(w @ cov @ w)) / TL_SCALE_LAPS)
    return out


def power_ceiling_split(effect: float, params: MixedParams, n_events: int = 13) -> float:
    """Power as the number of stints grows without bound, split design.

    The compound contrast is then a difference of two means over ``E/2``
    circuits each, so its variance never falls below ``4 tau^2 / E``.
    """
    from scipy.stats import norm

    se = 2 * params.circuit_sd / np.sqrt(n_events)
    z = effect / se
    return float(norm.sf(1.96 - z) + norm.cdf(-1.96 - z))


def simulate_mixed_power(
    n_stints: int, effect: float, design: str, params: MixedParams, seeds,
    n_events: int = 13, stint_laps: int = DEFAULT_STINT_LAPS, nominations=None, pair=None,
) -> dict:
    """Simulate ``len(seeds)`` experiments at one design point; reject counts.

    Returns the Wald-z rejection (what the real analysis does), a t-based
    rejection with ``n_events - 1`` degrees of freedom (the honest reference
    when the circuit effect is estimated from 13 circuits), and the number of
    refits the optimiser flagged as failed.
    """
    from scipy.stats import t as student_t

    from ..models.reml import single_threaded_blas

    model, sd, s2, w, n_comp, ib = _power_model(
        n_stints, design, params, n_events, stint_laps, nominations=nominations, pair=pair
    )
    # The t reference uses the races that TAKE PART: a realistic design for a rare
    # compound has far fewer than the season's thirteen.
    n_ev = len({r[0] for r in _plan(design, n_stints, n_events, nominations, pair, 3)[0]})
    beta = np.zeros(2 * n_comp)
    beta[n_comp:] = 0.03 * TL_SCALE_LAPS
    beta[n_comp + ib] += effect * TL_SCALE_LAPS
    theta0 = np.array([sd[n] / np.sqrt(s2) for n in model.names])
    crit_t = float(student_t.ppf(0.975, n_ev - 1))
    rej_z, rej_t, bad = [], [], 0
    with single_threaded_blas():
        for sq in seeds:
            rng = np.random.default_rng(sq)
            model.set_y(model.simulate(beta, s2, sd, rng))
            f = model.fit(starts=[theta0])
            bad += int(not f.converged)
            est = float(w @ f.beta)
            se = float(np.sqrt(w @ f.cov_beta @ w))
            z = abs(est / se) if se > 0 else 0.0
            rej_z.append(z > 1.959963984540054)
            rej_t.append(z > crit_t)
    return {
        "n_stints": n_stints, "design": design, "effect": effect, "n_sims": len(rej_z),
        "reject_z": rej_z, "reject_t": rej_t, "n_failed_fits": bad,
        "n_events_in_design": n_ev,
    }


def crossing_linear(n_grid, power, target: float = 0.80) -> float | None:
    """Where the curve first reaches ``target``, interpolated linearly in log N.

    Returns None when it never does: "no number of stints is enough" is an
    answer, and reporting the largest grid point instead would be wrong.
    """
    n = np.asarray(n_grid, float)
    p = np.asarray(power, float)
    for i in range(1, len(n)):
        if p[i] >= target > p[i - 1]:
            w = (target - p[i - 1]) / (p[i] - p[i - 1])
            return float(np.exp(np.log(n[i - 1]) + w * (np.log(n[i]) - np.log(n[i - 1]))))
    return float(n[0]) if p[0] >= target else None


def crossing_with_mc_error(
    n_grid, reject_by_n: list, target: float = 0.80, n_boot: int = 1000, seed: int = 0
) -> dict:
    """The 80% crossing and its Monte-Carlo standard error.

    The error comes from resampling the simulated experiments at each grid
    point and recomputing the interpolated crossing. Replicates in which the
    resampled curve never reaches the target are counted, not dropped silently.
    """
    n = np.asarray(n_grid, float)
    arrs = [np.asarray(r, float) for r in reject_by_n]
    point = crossing_linear(n, [a.mean() for a in arrs], target)
    rng = np.random.default_rng(seed)
    vals, never = [], 0
    for _ in range(n_boot):
        p = [a[rng.integers(0, len(a), len(a))].mean() for a in arrs]
        c = crossing_linear(n, p, target)
        if c is None:
            never += 1
        else:
            vals.append(c)
    return {
        "crossing": point,
        "mc_se": float(np.std(vals, ddof=1)) if len(vals) > 2 else None,
        "mc_ci95": (
            [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
            if len(vals) > 2 else None
        ),
        "share_of_resamples_never_reaching_target": never / n_boot,
        "power_at_grid": [float(a.mean()) for a in arrs],
    }

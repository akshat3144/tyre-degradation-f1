"""Is the uncertainty honest?

A model that says "80% confident" should be right 80% of the time. If it is
right 95% of the time the intervals are too wide and the model is useless for
decisions; if 50%, it is overconfident and will get someone's race wrong.

Neither the benchmark paper nor any public FastF1 tyre analysis we have seen
reports this. It is cheap to compute and it is the difference between a number
and a number you can act on.

Validation is leave-one-run-out: fit without a run, predict its laps, and check
whether the actual lap time falls inside the stated interval. Holding out a whole
run rather than scattered laps matters -- laps within a run are correlated, so
holding out single laps would let the model see almost the same information and
report coverage that flatters itself.

WHAT THE LAP-LEVEL NUMBER DOES AND DOES NOT SAY

``leave_one_run_out`` measures whether a held-out LAP falls inside its
predictive interval. The interval is dominated by lap-to-lap scatter (a second
or so), so covering 80% of laps is easy for almost any model with a sensible
residual SD; it says very little about whether the degradation RATES -- the
product -- are right or honestly uncertain. Quote it as "lap-level residual
calibration", never as "the model is calibrated".

The checks that do bear on the rates are below:

  * ``predictive_coverage`` -- leave-one-event-out. Refit without a circuit and
    ask whether that circuit's own fitted rate falls in the pooled model's
    predictive interval for it. That interval needs the circuit-to-circuit SD,
    which is exactly the quantity a single global rate pretends is zero.
  * ``coverage_of_truth`` -- simulation-based calibration. Data drawn from the
    fitted model, refitted; does the interval contain the rate that generated
    it? That checks the estimator, not the model's realism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

#: Nominal levels to check. 80 is the headline because it is the level a
#: strategist would actually use -- 95% intervals on lap time are too wide to
#: separate strategies.
DEFAULT_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


@dataclass
class Calibration:
    levels: np.ndarray
    empirical: np.ndarray
    n: int
    n_runs: int

    @property
    def coverage_80(self) -> float:
        return float(np.interp(0.80, self.levels, self.empirical))

    def miscalibration(self) -> float:
        """Mean absolute gap between nominal and empirical. Zero is perfect."""
        return float(np.mean(np.abs(self.levels - self.empirical)))

    def verdict(self) -> str:
        gap = self.coverage_80 - 0.80
        if abs(gap) <= 0.05:
            return "well calibrated"
        return "overconfident" if gap < 0 else "conservative"

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"nominal": self.levels, "empirical": self.empirical, "n": self.n}
        )


def leave_one_run_out(
    df: pd.DataFrame,
    levels=DEFAULT_LEVELS,
    max_runs: int | None = None,
    seed: int = 0,
) -> Calibration:
    """Coverage of prediction intervals, one run held out at a time.

    The model here is deliberately the same shape as the fitted one: a
    per-compound slope through the centred design. Refitting the full
    mixed-effects model for every run would take hours and change nothing about
    the coverage question, since the slope is what carries the prediction.

    Args:
        df: a prepared design frame with ``y``, ``tl``, ``C`` and ``run_id``.
        levels: nominal interval levels to check.
        max_runs: cap the number of folds, for speed.
    """
    levels = np.asarray(levels, dtype=float)
    runs = df["run_id"].unique()
    if max_runs is not None and len(runs) > max_runs:
        runs = np.random.default_rng(seed).choice(runs, max_runs, replace=False)

    hits = np.zeros(len(levels))
    total = 0

    for run in runs:
        train = df[df["run_id"] != run]
        test = df[df["run_id"] == run]
        compound = test["C"].iloc[0]

        fit_on = train[train["C"] == compound]
        if len(fit_on) < 30:
            continue

        # Slope through the origin on the centred design.
        x, y = fit_on["tl"].to_numpy(), fit_on["y"].to_numpy()
        denom = float(x @ x)
        if denom == 0:
            continue
        slope = float(x @ y / denom)

        # Predictive spread: residual scatter plus slope uncertainty carried
        # out to the tyre age being predicted. Ignoring the second term is the
        # usual way a model ends up overconfident far from its data.
        resid = y - slope * x
        sigma = float(resid.std(ddof=1))
        se_slope = sigma / np.sqrt(denom)

        xt, yt = test["tl"].to_numpy(), test["y"].to_numpy()
        pred = slope * xt
        spread = np.sqrt(sigma**2 + (se_slope * xt) ** 2)

        z = norm.ppf(0.5 + levels / 2)
        inside = np.abs(yt - pred)[:, None] <= (z[None, :] * spread[:, None])
        hits += inside.sum(axis=0)
        total += len(yt)

    if total == 0:
        raise ValueError("no runs had enough training data to validate")

    return Calibration(
        levels=levels,
        empirical=hits / total,
        n=total,
        n_runs=len(runs),
    )


# ---------------------------------------------------------------------------
# Rate-level calibration
# ---------------------------------------------------------------------------

#: Levels reported for the rate-level checks.
RATE_LEVELS = (0.5, 0.8, 0.9, 0.95)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    With a few dozen cells a plain +/- 2 SE interval is badly wrong near 0 and
    1, and "covered 36 of 38" needs an honest error bar.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def predictive_coverage(
    cells: list[dict],
    levels=RATE_LEVELS,
    include_circuit: bool = True,
    include_event_noise: bool = True,
) -> dict:
    """Coverage of circuit-level rates by the leave-one-event-out predictive interval.

    Each cell carries the circuit's own fitted rate ``own`` and its standard
    error ``own_se``, the pooled model's rate ``pooled`` and ``pooled_se`` from a
    fit that LEFT THAT CIRCUIT OUT, and ``tau``, that fit's circuit-slope SD.

    The predictive variance is ``pooled_se^2 [+ tau^2] [+ own_se^2]``. The last
    term is the sampling noise of the circuit's own estimate: leave it out and
    the question becomes "does the pooled interval contain the TRUE rate at
    that circuit", which the data cannot answer because only a noisy estimate is
    observed. Both switches exist so the effect of each term is visible.
    """
    if not cells:
        raise ValueError("no cells to score")
    z = []
    for c in cells:
        v = c["pooled_se"] ** 2
        if include_circuit:
            v += c["tau"] ** 2
        if include_event_noise:
            v += c["own_se"] ** 2
        z.append((c["own"] - c["pooled"]) / np.sqrt(v))
    z = np.asarray(z)
    out = {}
    for lv in levels:
        crit = float(norm.ppf(0.5 + lv / 2))
        k = int((np.abs(z) <= crit).sum())
        lo, hi = wilson_interval(k, len(z))
        out[f"{lv:.2f}"] = {"coverage": k / len(z), "k": k, "n": int(len(z)), "wilson_95": [lo, hi]}
    return {
        "coverage": out,
        "z_mean": float(z.mean()),
        "z_sd": float(z.std(ddof=1)) if len(z) > 1 else float("nan"),
    }


def coverage_of_truth(truth, estimates, ses, levels=RATE_LEVELS) -> dict:
    """Fraction of simulated refits whose Wald interval contains the true value.

    ``estimates`` and ``ses`` are (n_sims, n_params); ``truth`` is (n_params,).
    """
    truth = np.asarray(truth, float)
    est = np.asarray(estimates, float)
    se = np.asarray(ses, float)
    out = {}
    for lv in levels:
        crit = float(norm.ppf(0.5 + lv / 2))
        cover = np.abs(est - truth) <= crit * se
        out[f"{lv:.2f}"] = {
            "pooled": float(cover.mean()),
            "per_parameter": [float(x) for x in cover.mean(axis=0)],
        }
    return out

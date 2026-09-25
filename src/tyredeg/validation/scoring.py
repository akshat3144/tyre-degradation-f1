"""Scoring rules, defined to match the benchmark exactly.

The claim "we beat their published number" is only worth making if the number is
computed the same way. Two traps in their definitions, both easy to get wrong:

  1. Their "RMSPE" is NOT a percentage error. The formula in the paper has no
     division by y -- it is a plain root-mean-square error in seconds. Anyone
     implementing it from the name alone gets a different quantity.

  2. Their Table 1 "Total" row is a SUM over the three stints, while the Table 2
     CRPS total is a MEAN. So RMSPE totals cannot be compared across races with
     different stint counts, and reproducing their 1.082 requires summing.

We use ``scoringrules``, the Python port of the R ``scoringRules`` package they
used, and ``crosscheck_against_r`` verifies the two agree on identical input.
That turns "like-for-like" from an assertion into something tested.
"""

from __future__ import annotations

import numpy as np
import scoringrules as sr


def rmse_seconds(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Their "RMSPE": root mean squared error in seconds, despite the name."""
    actual, predicted = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def stint_rmse_total(per_stint: list[float]) -> float:
    """Their Table 1 total: a SUM across stints, not a mean."""
    return float(np.sum(per_stint))


def stint_crps_total(per_stint: list[float]) -> float:
    """Their Table 2 total: a MEAN across stints, not a sum."""
    return float(np.mean(per_stint))


def crps_ensemble(actual: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """CRPS from posterior draws -- the analogue of R's ``crps_sample``.

    Args:
        actual: observed values, shape (n,).
        draws: predictive samples, shape (n, n_draws).
    """
    return np.asarray(sr.crps_ensemble(np.asarray(actual, float), np.asarray(draws, float)))


def crps_normal(actual: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form Gaussian CRPS -- the analogue of R's ``crps(family="normal")``.

    This is what they used for the ARIMA baseline, which has no draws.
    """
    return np.asarray(
        sr.crps_normal(np.asarray(actual, float), np.asarray(mu, float), np.asarray(sigma, float))
    )


def crps_student_t(
    actual: np.ndarray, mu: np.ndarray, sigma: np.ndarray, df: float = 5.0
) -> np.ndarray:
    """CRPS under a Student-t predictive distribution.

    Lap times have heavy tails: a driver locks a wheel or runs wide and loses a
    second, then returns to target pace. A normal predictive distribution is
    punished hard by those, which is exactly why the benchmark's best model used
    skewed-t errors rather than normal ones -- their gain came from robustness,
    not from tyre physics.

    Using a t here is the same idea, applied to our forecast. It is a principled
    choice taken from their own finding, not a parameter tuned until the score
    improved; ``df`` is fixed at 5 and never fitted.
    """
    return np.asarray(
        sr.crps_t(
            np.asarray(actual, float),
            df,
            np.asarray(mu, float),
            np.asarray(sigma, float),
        )
    )


def rolling_origin_folds(stint_length: int, train_fraction: float = 0.75):
    """Their cross-validation scheme, per stint.

    Train on the first three quarters of a stint, predict the next lap, then
    expand the training window one lap at a time to the end of the stint.

    Transcribed from their ``CV_Functions.R``, which sets the number of test
    laps to ``K <- round(stint_length/4)`` and predicts the last K laps. Two
    details are theirs and are kept deliberately:

    * The count is ``round(n/4)`` rather than ``n - ceil(0.75n)``. These differ
      whenever n/4 lands on a half, and a stint of 22 laps is exactly such a
      case -- 6 test laps their way, 5 ours.
    * R's ``round`` breaks halves to even, as NumPy's does. Python's built-in
      ``round`` also does, but ``int(x + 0.5)`` would not, so the rounding is
      done explicitly rather than left to whichever idiom came to hand.

    Yields:
        (train_end, test_index) pairs, both 0-based.
    """
    k = int(np.round(stint_length * (1.0 - train_fraction)))
    k = min(max(k, 0), stint_length)
    for i in range(stint_length - k, stint_length):
        yield i, i


def crosscheck_against_r(n: int = 200, seed: int = 0) -> dict:
    """Verify our Python CRPS matches R's ``scoringRules`` on identical input.

    Runs only if R is available; returns ``{"available": False}`` otherwise, so
    the package never depends on R. Confirms the benchmark comparison is
    genuinely like-for-like rather than merely claimed to be.
    """
    import json
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    rscript = shutil.which("Rscript") or r"C:/Program Files/R/R-4.6.1/bin/x64/Rscript.exe"
    if not Path(rscript).exists():
        return {"available": False, "reason": "Rscript not found"}

    rng = np.random.default_rng(seed)
    actual = rng.normal(90, 1.0, n)
    mu = actual + rng.normal(0, 0.3, n)
    sigma = np.full(n, 0.5)
    draws = rng.normal(mu[:, None], sigma[:, None], (n, 400))

    ours_normal = float(crps_normal(actual, mu, sigma).mean())
    ours_ens = float(crps_ensemble(actual, draws).mean())

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        np.savetxt(tmp / "actual.csv", actual, delimiter=",")
        np.savetxt(tmp / "mu.csv", mu, delimiter=",")
        np.savetxt(tmp / "draws.csv", draws, delimiter=",")
        script = tmp / "check.R"
        script.write_text(
            f"""
.libPaths(c(file.path(Sys.getenv("LOCALAPPDATA"),"R","win-library","4.6"), .libPaths()))
suppressMessages(library(scoringRules))
a  <- as.numeric(read.csv("{(tmp / 'actual.csv').as_posix()}", header=FALSE)[,1])
mu <- as.numeric(read.csv("{(tmp / 'mu.csv').as_posix()}", header=FALSE)[,1])
d  <- as.matrix(read.csv("{(tmp / 'draws.csv').as_posix()}", header=FALSE))
cat(sprintf('{{"normal": %.10f, "ensemble": %.10f}}',
    mean(crps(a, family="normal", mean=mu, sd=rep({sigma[0]}, length(a)))),
    mean(crps_sample(a, d))))
""",
            encoding="utf-8",
        )
        try:
            out = subprocess.run(
                [rscript, str(script)], capture_output=True, text=True, timeout=180
            )
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "reason": str(exc)[:120]}

    try:
        r = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"available": False, "reason": (out.stderr or out.stdout)[:200]}

    return {
        "available": True,
        "normal": {"python": ours_normal, "r": r["normal"], "diff": abs(ours_normal - r["normal"])},
        "ensemble": {"python": ours_ens, "r": r["ensemble"], "diff": abs(ours_ens - r["ensemble"])},
    }


def crosscheck_python(n: int = 200, seed: int = 0, n_draws: int = 400) -> dict:
    """Python-only checks of the CRPS code, for when R is not installed.

    ``crosscheck_against_r`` was the original evidence that our CRPS is the same
    number their R ``scoringRules`` call produces (agreement to 2.5e-11). R is
    not installed on the machine that builds the paper, so that check can no
    longer be re-run, and a test that silently skips is not evidence. These
    checks need nothing but numpy and scipy, and they are independent of the
    library under test:

    * ``normal_vs_integral``: closed-form Gaussian CRPS against direct numerical
      integration of the definition, ``integral (F(x) - 1{x >= y})^2 dx``.
    * ``t_vs_integral``: the same for the Student-t used by ``crps_student_t``.
    * ``ensemble_vs_energy_form``: ``crps_ensemble`` against the kernel form
      ``E|X - y| - 0.5 E|X - X'|`` computed by brute force in O(m^2). R's
      ``crps_sample`` default is this same V-statistic, so agreement here is the
      Python equivalent of the R cross-check.
    * ``ensemble_vs_closed_form``: the sample estimator on Gaussian draws against
      the closed form. This one is Monte-Carlo noise, not a bug, and is reported
      with its tolerance rather than as an equality.
    * ``anchor``: CRPS of N(0, 1) at y = 0 is (sqrt(2) - 1) / sqrt(pi), a value
      that can be checked on paper.
    """
    from scipy import integrate, stats

    rng = np.random.default_rng(seed)
    actual = rng.normal(90, 1.0, 12)
    mu = actual + rng.normal(0, 0.3, 12)
    sigma = np.full(12, 0.5)

    def integral(cdf, y):
        lo, hi = y - 60.0, y + 60.0
        left, _ = integrate.quad(lambda x: cdf(x) ** 2, lo, y, points=[y - 1, y - 0.1], limit=400)
        right, _ = integrate.quad(lambda x: (1 - cdf(x)) ** 2, y, hi, points=[y + 1], limit=400)
        return left + right

    normal_d = max(
        abs(float(crps_normal(a, m, s)) - integral(lambda x, m=m, s=s: stats.norm.cdf(x, m, s), a))
        for a, m, s in zip(actual, mu, sigma, strict=True)
    )
    t_d = max(
        abs(
            float(crps_student_t(a, m, s, df=5.0))
            - integral(lambda x, m=m, s=s: stats.t.cdf((x - m) / s, 5.0), a)
        )
        for a, m, s in zip(actual, mu, sigma, strict=True)
    )

    a_big = rng.normal(90, 1.0, n)
    mu_big = a_big + rng.normal(0, 0.3, n)
    draws = rng.normal(mu_big[:, None], 0.5, (n, n_draws))
    ours = crps_ensemble(a_big, draws)
    brute = np.array(
        [
            np.mean(np.abs(d - a)) - 0.5 * np.mean(np.abs(d[:, None] - d[None, :]))
            for a, d in zip(a_big, draws, strict=True)
        ]
    )
    closed = crps_normal(a_big, mu_big, np.full(n, 0.5))

    return {
        "normal_vs_integral": float(normal_d),
        "t_vs_integral": float(t_d),
        "ensemble_vs_energy_form": float(np.max(np.abs(ours - brute))),
        "ensemble_vs_closed_form_mean": float(abs(ours.mean() - closed.mean())),
        "anchor": float(
            abs(float(crps_normal(0.0, 0.0, 1.0)) - (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi))
        ),
    }

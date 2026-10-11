"""Score the hierarchical model on the benchmark's exact predictions.

Same folds, same metric, same aggregation as ``validation.benchmark``, so the
two of ours and the four of theirs all sit in one table. The difference is only
what produces the forecast: here it is the pooled state-space model refit by
MCMC inside every fold.

CRPS FROM DRAWS, NOT FROM A FITTED NORMAL

The other scorer hands CRPS a mean and a standard deviation. This one has actual
posterior predictive draws, so it uses the ensemble estimator -- the same one
already checked against R's ``scoringRules`` to 1e-11 by
``test_our_crps_matches_r_scoring_rules``. That is a fairer reading of a
Bayesian forecast: nothing forces the predictive distribution to be symmetric,
and collapsing it to two numbers would throw away the part a state-space model
is actually good at.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models.hierarchical import forecast_lap, fuel_start_kg
from .benchmark import _folds
from .scoring import crps_ensemble, rmse_seconds

log = logging.getLogger(__name__)


@dataclass
class HierResult:
    per_stint_crps: list[float]
    per_stint_rmse: list[float]
    n_predictions: int
    driver: str
    event: str
    #: Posterior mean fuel coefficient per fold. Their single-car fit landed at
    #: 0.016 s/kg against a physical 0.030-0.035; ours is reported so the same
    #: check can be run on us.
    gammas: list[float] = field(default_factory=list)
    #: Per-fold posterior mean degradation rate by compound label.
    rates: list[dict] = field(default_factory=list)
    #: Folds that could not be fit, with the reason.
    skipped: list[str] = field(default_factory=list)

    @property
    def crps(self) -> float:
        """Their Table 2 total: the MEAN across stints."""
        return float(np.mean(self.per_stint_crps))

    @property
    def rmse_total(self) -> float:
        """Their Table 1 total: the SUM across stints."""
        return float(np.sum(self.per_stint_rmse))

    @property
    def mean_gamma(self) -> float:
        return float(np.mean(self.gammas)) if self.gammas else float("nan")


def score_hier(
    laps: pd.DataFrame,
    driver: str = "HAM",
    event: str = "Austrian Grand Prix",
    season: int = 2025,
    race_laps: int | None = None,
    seed: int = 0,
) -> HierResult:
    """Run the benchmark's folds through the hierarchical model.

    Args:
        laps: race laps for the whole field.
        driver: whose laps are forecast. The benchmark used Hamilton.
        season: picks the race-start fuel load. 110 kg before 2026, 70 after.
        race_laps: scheduled distance, for the fuel model. Defaults to the
            highest lap number seen, which is the classified winner's distance.
        seed: passed to the sampler so a rerun reproduces.

    Returns:
        Per-stint CRPS and RMSE, aggregated the way their tables do it.
    """
    df = laps[laps["LapTimeSeconds"].notna()].copy()
    subject = df[df["Driver"] == driver].sort_values("LapNumber")
    if subject.empty:
        raise ValueError(f"no laps for {driver!r}")

    if race_laps is None:
        race_laps = int(df["LapNumber"].max())

    folds = _folds(subject)
    stint_of = dict(zip(subject["LapNumber"], subject["Stint"], strict=True))

    per_stint: dict[float, list[tuple[float, np.ndarray]]] = {}
    gammas: list[float] = []
    rates: list[dict] = []
    skipped: list[str] = []

    for train_to, test_lap in folds:
        try:
            f = forecast_lap(
                df,
                train_to=train_to,
                pred_driver=driver,
                pred_lap=test_lap,
                race_laps=race_laps,
                start_kg=fuel_start_kg(season),
                seed=seed,
            )
        except Exception as exc:  # a fold that cannot be fit is reported, not faked
            skipped.append(f"lap {test_lap}: {type(exc).__name__}: {exc}")
            continue

        per_stint.setdefault(stint_of[test_lap], []).append((f.actual, f.draws))
        gammas.append(f.gamma)
        rates.append(f.rates)

    if not per_stint:
        raise ValueError("no fold produced a prediction")

    crps_list, rmse_list, n = [], [], 0
    for _, rows in sorted(per_stint.items()):
        actual = np.array([r[0] for r in rows], dtype=float)
        # Folds within a stint can have different draw counts if a chain was
        # dropped, so score each fold and average rather than stacking.
        fold_crps = [float(crps_ensemble(np.array([a]), d.reshape(1, -1))[0]) for a, d in rows]
        crps_list.append(float(np.mean(fold_crps)))
        point = np.array([float(np.mean(d)) for _, d in rows])
        rmse_list.append(rmse_seconds(actual, point))
        n += len(rows)

    return HierResult(
        per_stint_crps=crps_list,
        per_stint_rmse=rmse_list,
        n_predictions=n,
        driver=driver,
        event=event,
        gammas=gammas,
        rates=rates,
        skipped=skipped,
    )

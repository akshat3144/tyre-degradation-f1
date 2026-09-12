"""Deconfounded tyre degradation from public Formula 1 timing data.

Separating true tyre wear from fuel load, traffic and track evolution.

The short version of the method: inside a single stint, tyre age and fuel burn
move together lap for lap, so no estimator can pull them apart from one car.
Pool the field and the confound breaks, because everything that varies with time
in a race is common to every car on that lap and can be subtracted rather than
modelled.

Typical use::

    import pandas as pd
    from tyredeg import prepare, fit_degradation

    laps = pd.read_parquet("data/processed/laps.parquet")
    fit = fit_degradation(prepare(laps, "race"))

    for c in fit.ordered:
        print(c, fit.rates[c].mean, fit.rates[c].lo, fit.rates[c].hi)

Two things worth knowing before relying on the output:

* Degradation is keyed on the PHYSICAL compound (C1-C5), never on the
  HARD/MEDIUM/SOFT label. Pirelli nominates three of C1-C5 per weekend, so the
  labels are relative and grouping by them pools different rubber.
* Results carry a ``context`` of practice or race. They genuinely differ, and
  the difference is a finding rather than noise.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .artifacts.schema import Interval
from .config import (
    COMPOUND_ALLOCATION_2026,
    CONVENTIONAL_2026,
    MASS_SENSITIVITY_S_PER_KG,
    SEASON,
)
from .data.laps import clean_laps, summarise, tag_long_runs
from .models.design import add_physical_compound, prepare
from .models.mixed import Fit, fit_degradation, to_artifact
from .strategy.optimise import enumerate_plans, optimal_stint
from .strategy.pitloss import estimate as estimate_pit_loss
from .validation.calibration import leave_one_run_out
from .validation.power import power_curve
from .validation.transfer import leave_one_event_out

__all__ = [
    "COMPOUND_ALLOCATION_2026",
    "CONVENTIONAL_2026",
    "MASS_SENSITIVITY_S_PER_KG",
    "SEASON",
    "Fit",
    "Interval",
    "__version__",
    "add_physical_compound",
    "clean_laps",
    "enumerate_plans",
    "estimate_pit_loss",
    "fit_degradation",
    "leave_one_event_out",
    "leave_one_run_out",
    "optimal_stint",
    "power_curve",
    "prepare",
    "summarise",
    "tag_long_runs",
    "to_artifact",
]

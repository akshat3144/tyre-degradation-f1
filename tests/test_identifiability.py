"""A rate we do not know the sign of is not a measurement.

WHAT THIS CAUGHT

Madrid measured its SOFT at 0.058 s/lap with a standard error of 0.198. The
forecast carried it through labelled "measured", the optimiser read the softest
tyre in the race as the most durable thing on the car, and the plan for a
57-lap Grand Prix came back as:

    C3 x5  +  C4 x52

Five laps on the MEDIUM whose only purpose was to satisfy the two-compound
rule, then fifty-two on the SOFT. The per-session numbers behind it disagreed
completely -- FP1 said the SOFT degraded at +0.49 s/lap, FP2 said -0.008 -- so
there was never a measurement there to use.

The rule: a practice cell must be at least one of its own standard errors from
zero before it counts as evidence about this circuit. Cells that fail fall to a
stand-in, which is clamped against the compounds either side and so cannot
invert the compound order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.validation.transfer import MIN_RATE_SE_RATIO, identified


def test_a_rate_smaller_than_its_own_error_is_not_identified():
    rate = pd.Series([0.058, 0.249, -0.30, 0.02])
    se = pd.Series([0.198, 0.143, 0.05, 0.02])
    assert list(identified(rate, se)) == [False, True, True, True]


def test_a_missing_error_still_counts():
    """A single-run cell has no between-run variance to estimate an error from.

    `forecast` widens those by THIN_INTERVAL_MULTIPLE rather than discarding
    them, and this filter must not quietly undo that.
    """
    out = identified(pd.Series([0.05, 0.05]), pd.Series([np.nan, 0.0]))
    assert list(out) == [True, True]


def test_the_bar_is_deliberately_weak():
    """One standard error, not two.

    The job is to catch cells that say nothing at all, not to demand
    significance from three practice runs -- which would refuse most of the
    calendar and leave the pipeline unable to plan anything.
    """
    assert MIN_RATE_SE_RATIO == 1.0


@pytest.mark.parametrize("rate,se,ok", [(0.10, 0.05, True), (0.05, 0.10, False)])
def test_the_boundary_is_on_the_ratio_not_the_size(rate, se, ok):
    """A small well-measured rate survives; a large badly-measured one does not."""
    assert bool(identified(pd.Series([rate]), pd.Series([se]))[0]) is ok

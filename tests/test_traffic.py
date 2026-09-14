"""Tests for the traffic covariate.

Traffic is the one confounder the race design does not remove for free, because
it differs between cars at the same lap -- which is the variation the design
uses. It is also correlated with tyre age through pit stops.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.data.traffic import CLEAR_AIR_S, DIRTY_AIR_S, add_gap_ahead, traffic_summary


def field(gaps_s, lap=10.0):
    """One lap of a race with the given gaps between consecutive cars."""
    starts = np.cumsum([0.0] + list(gaps_s))
    return pd.DataFrame(
        {
            "event": "Test GP",
            "session": "R",
            "Driver": [f"D{i}" for i in range(len(starts))],
            "LapNumber": lap,
            "TyreLife": 10.0,
            "LapTimeSeconds": 90.0,
            "LapStartTime": pd.to_timedelta(starts, unit="s"),
        }
    )


def test_leader_is_treated_as_clear_air():
    """The lead car has nobody ahead. That is clear air, not missing data."""
    out = add_gap_ahead(field([3.0, 3.0]))
    assert out["gap_ahead_s"].iloc[0] == CLEAR_AIR_S
    assert not out["in_dirty_air"].iloc[0]


def test_gaps_are_differences_between_consecutive_cars():
    out = add_gap_ahead(field([1.2, 4.5]))
    assert out["gap_ahead_s"].tolist()[1:] == pytest.approx([1.2, 4.5])


def test_close_running_is_flagged_as_dirty_air():
    out = add_gap_ahead(field([0.6, 0.8]))
    assert out["in_dirty_air"].iloc[1:].all()


def test_large_gaps_are_capped():
    out = add_gap_ahead(field([90.0]))
    assert out["gap_ahead_s"].max() == CLEAR_AIR_S


def test_traffic_rises_as_the_car_ahead_gets_closer():
    """The covariate must have the sign a reader expects: more traffic, more
    lap time lost. So it measures closeness, not distance."""
    close = add_gap_ahead(field([0.3]))["traffic"].iloc[1]
    far = add_gap_ahead(field([1.8]))["traffic"].iloc[1]
    assert close > far
    assert add_gap_ahead(field([5.0]))["traffic"].iloc[1] == 0.0
    assert close == pytest.approx(DIRTY_AIR_S - 0.3)


def test_gaps_do_not_leak_across_laps():
    a = field([1.0, 1.0], lap=10.0)
    b = field([1.0, 1.0], lap=11.0)
    out = add_gap_ahead(pd.concat([a, b], ignore_index=True))
    # Each lap has exactly one leader, so exactly one clear-air entry per lap.
    per_lap = out[out["gap_ahead_s"] == CLEAR_AIR_S].groupby("LapNumber").size()
    assert (per_lap == 1).all()


def test_requires_lap_start_time():
    df = field([1.0]).drop(columns=["LapStartTime"])
    with pytest.raises(ValueError, match="LapStartTime"):
        add_gap_ahead(df)


def test_summary_reports_the_correlation_that_matters():
    """If traffic and tyre age are uncorrelated it cannot bias the estimate.
    In the real 2026 races the correlation is about -0.19: fresher tyres run in
    more traffic, because they have just rejoined from a pit stop."""
    df = field([0.5, 0.5, 6.0, 6.0])
    df["TyreLife"] = [2.0, 3.0, 4.0, 20.0, 22.0]  # fresh cars bunched up
    s = traffic_summary(add_gap_ahead(df))
    assert len(s) == 1
    assert s["corr_traffic_tyreage"].iloc[0] < 0
    assert 0 < s["pct_dirty_air"].iloc[0] <= 100

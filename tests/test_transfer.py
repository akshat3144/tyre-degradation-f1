"""Tests for practice-to-race transfer.

The identifiability tests matter most. Refusing to report an estimate the design
cannot support is the difference between a validation tool and a number
generator, and it was found the hard way: a cell with zero tyre-age spread
produced a race degradation rate of -0.41 s/lap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.validation.transfer import (
    MIN_AGE_SPREAD_LAPS,
    age_spread,
    cell_rates,
    leave_one_event_out,
)


def make_cells(events, compound="C3", n_runs=5, laps=16, rate=0.08, noise=0.2,
               age_offset_step=3, seed=0):
    """Runs across several events, each degrading at ``rate``.

    ``age_offset_step`` staggers the starting tyre age between runs, which is
    what creates the cross-car age spread the race design needs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ev in events:
        for run in range(n_runs):
            offset = run * age_offset_step
            lap = np.arange(1, laps + 1, dtype=float)
            tyre = lap + offset
            y = rate * tyre + rng.normal(0, noise, laps)
            rows.append(
                pd.DataFrame(
                    {
                        "event": ev,
                        "C": compound,
                        "run_id": f"{ev}|{run}",
                        "LapNumber": lap,
                        "TyreLife": tyre,
                        "LapTimeSeconds": 90.0 + y,
                        "tl": tyre - tyre.mean(),
                        "y": y - y.mean(),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


# --- identifiability --------------------------------------------------------


def test_age_spread_is_zero_when_every_car_is_the_same_age():
    """The 2026 Belgian GP case: a synchronised strategy leaves nothing to
    compare, and the slope cannot be recovered at all."""
    df = pd.DataFrame(
        {
            "LapNumber": [10.0, 10.0, 11.0, 11.0],
            "TyreLife": [5.0, 5.0, 6.0, 6.0],
        }
    )
    assert age_spread(df) == 0.0


def test_age_spread_detects_staggered_strategies():
    df = pd.DataFrame(
        {
            "LapNumber": [10.0, 10.0, 10.0],
            "TyreLife": [2.0, 12.0, 22.0],
        }
    )
    assert age_spread(df) > 5.0


def test_cells_without_age_spread_are_refused():
    """A rate must not be reported where the design cannot identify one."""
    flat = make_cells(["A GP"], age_offset_step=0, n_runs=5)
    assert cell_rates(flat, min_age_spread=MIN_AGE_SPREAD_LAPS).empty
    # Without the guard it happily returns a meaningless number.
    assert not cell_rates(flat).empty


def test_cells_with_age_spread_are_kept():
    staggered = make_cells(["A GP"], age_offset_step=4)
    out = cell_rates(staggered, min_age_spread=MIN_AGE_SPREAD_LAPS)
    assert len(out) == 1
    assert out["age_spread"].iloc[0] >= MIN_AGE_SPREAD_LAPS


def test_thin_cells_are_refused():
    thin = make_cells(["A GP"], n_runs=2)
    assert cell_rates(thin, min_runs=3).empty


def test_cell_rate_recovers_a_known_slope():
    df = make_cells(["A GP"], rate=0.09, noise=0.05, seed=2)
    out = cell_rates(df)
    assert out["rate"].iloc[0] == pytest.approx(0.09, abs=0.02)


# --- transfer ---------------------------------------------------------------


def test_calibration_beats_naive_when_a_real_factor_exists():
    """Race degrades at a constant fraction of practice. Learning that fraction
    from other events must beat assuming the two are equal."""
    events = [f"E{i} GP" for i in range(6)]
    practice = make_cells(events, rate=0.10, noise=0.05, seed=3)
    race = make_cells(events, rate=0.05, noise=0.05, seed=4)

    res = leave_one_event_out(practice, race)
    assert res.mae_calibrated < res.mae_naive
    assert res.improvement > 0.4
    assert res.factor == pytest.approx(0.5, abs=0.15)


def test_calibration_does_not_help_when_the_two_already_agree():
    """If practice and race match, the factor is about 1 and calibrating is a
    no-op. A method that 'improves' things here would be fitting noise."""
    events = [f"E{i} GP" for i in range(6)]
    practice = make_cells(events, rate=0.07, noise=0.04, seed=5)
    race = make_cells(events, rate=0.07, noise=0.04, seed=6)

    res = leave_one_event_out(practice, race)
    assert res.factor == pytest.approx(1.0, abs=0.25)
    assert abs(res.improvement) < 0.5


def test_held_out_event_never_informs_its_own_correction():
    """Give one event a wildly different practice-to-race ratio. Its own factor
    must be unaffected by it, because the fold excludes it.

    Checking that every fold produces a distinct factor would be wrong -- a
    median over five values only takes a few distinct values, so equal factors
    are expected and prove nothing either way.
    """
    normal = [f"E{i} GP" for i in range(5)]
    odd = "ODD GP"

    practice = pd.concat(
        [make_cells(normal, rate=0.10, noise=0.02, seed=7),
         make_cells([odd], rate=0.10, noise=0.02, seed=9)],
        ignore_index=True,
    )
    race = pd.concat(
        [make_cells(normal, rate=0.05, noise=0.02, seed=8),   # ratio 0.5
         make_cells([odd], rate=0.40, noise=0.02, seed=10)],  # ratio 4.0
        ignore_index=True,
    )

    res = leave_one_event_out(practice, race)
    used = res.table.set_index("event")["factor_used"]

    # The outlier is predicted using only the normal events, so it gets ~0.5
    # and is therefore badly predicted -- which is correct behaviour.
    assert used[odd] == pytest.approx(0.5, abs=0.15)

    # And a normal event's factor is barely moved by the outlier's presence.
    assert used["E0 GP"] == pytest.approx(0.5, abs=0.2)


def test_transfer_refuses_when_nothing_overlaps():
    practice = make_cells(["A GP"])
    race = make_cells(["B GP"])
    with pytest.raises(ValueError, match="no .*cell"):
        leave_one_event_out(practice, race)

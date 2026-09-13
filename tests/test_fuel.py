"""Tests for fuel estimation.

The point of these is to lock in the difference between our indexing and the
benchmark's, since that difference is a claim we make publicly.
"""

import numpy as np
import pytest

from tyredeg.config import FUEL_RACE_KG_2026, MASS_SENSITIVITY_RANGE
from tyredeg.data.fuel import (
    RESERVE_KG,
    benchmark_fuel_kg,
    fuel_effect_seconds,
    practice_fuel_kg,
    race_fuel_kg,
)


def test_race_fuel_starts_full_and_ends_on_reserve():
    fuel = race_fuel_kg(np.arange(1, 71), race_laps=70)
    assert fuel[0] == pytest.approx(FUEL_RACE_KG_2026)
    assert fuel[-1] == pytest.approx(RESERVE_KG)


def test_race_fuel_decreases_monotonically():
    fuel = race_fuel_kg(np.arange(1, 71), race_laps=70)
    assert np.all(np.diff(fuel) < 0)


def test_race_fuel_is_indexed_to_lap_number_not_row_count():
    """The core correctness claim: dropping laps must not distort the ramp."""
    all_laps = np.arange(1, 71)
    # Simulate filtering: lose laps 30-36 to a safety car.
    kept = np.setdiff1d(all_laps, np.arange(30, 37))

    ours = race_fuel_kg(kept, race_laps=70)
    full = race_fuel_kg(all_laps, race_laps=70)

    # Every retained lap keeps the fuel value it had before filtering.
    np.testing.assert_allclose(ours, full[kept - 1])


def test_benchmark_indexing_is_distorted_by_filtering():
    """Quantifies the defect we report. This is the number in the README."""
    all_laps = np.arange(1, 71)
    kept = np.setdiff1d(all_laps, np.arange(30, 37))  # 7 laps dropped

    theirs = benchmark_fuel_kg(len(kept), start_kg=110.0, end_kg=1.0)
    correct = race_fuel_kg(kept, race_laps=70, start_kg=110.0)

    err = np.abs(theirs - correct)
    # Small in kg, but comparable to the effect size once converted to seconds.
    assert err.max() > 3.0
    assert fuel_effect_seconds(err).max() > 0.10


def test_race_fuel_rejects_degenerate_race_length():
    with pytest.raises(ValueError, match="at least 2"):
        race_fuel_kg([1, 2, 3], race_laps=1)


def test_practice_fuel_is_relative_to_run_start():
    """Absolute practice load is unknown, so only the gradient should matter."""
    life = np.array([5, 6, 7, 8, 9])
    a = practice_fuel_kg(life, start_kg=60.0)
    b = practice_fuel_kg(life, start_kg=90.0)
    # Different assumed loads, identical shape -- the intercept absorbs the level.
    np.testing.assert_allclose(np.diff(a), np.diff(b))
    assert a[0] == pytest.approx(60.0)


def test_mass_sensitivity_default_sits_inside_published_range():
    lo, hi = MASS_SENSITIVITY_RANGE
    per_kg = fuel_effect_seconds(np.array([1.0]))[0]
    assert lo <= per_kg <= hi

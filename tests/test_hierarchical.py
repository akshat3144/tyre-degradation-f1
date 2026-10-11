"""Tests for the hierarchical race model's data assembly and leakage rules.

The sampler itself is not exercised here -- a fit is 25 seconds and needs a
compiled model, so that lives behind the ``stan`` marker. What IS tested is
everything that decides WHAT the sampler sees, because that is where a
benchmark comparison gets quietly unfair: one lap of the future leaking into a
training window would improve every score and break nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.config import FUEL_RACE_KG_2026, FUEL_RACE_KG_LEGACY
from tyredeg.models.hierarchical import (
    MIN_DRIVERS,
    build_data,
    fuel_start_kg,
)


def field(n_drivers: int = 10, n_laps: int = 30, base: float = 90.0) -> pd.DataFrame:
    """A synthetic race: every car degrades, pits once, and burns fuel."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_drivers):
        name = f"D{d:02d}"
        pit_lap = 12 + (d % 5)
        for lap in range(1, n_laps + 1):
            age = lap if lap < pit_lap else lap - pit_lap + 1
            compound = "MEDIUM" if lap < pit_lap else "HARD"
            rows.append(
                {
                    "Driver": name,
                    "LapNumber": float(lap),
                    "Stint": 1.0 if lap < pit_lap else 2.0,
                    "Compound": compound,
                    "TyreLife": float(age),
                    "LapTimeSeconds": base + 0.05 * age - 0.03 * lap + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def test_training_window_never_contains_the_forecast_lap():
    """The whole comparison rests on this."""
    df = field()
    data, _, held = build_data(
        df, train_to=20, pred_driver="D00", pred_lap=21, race_laps=30, start_kg=110.0
    )
    # Laps are re-indexed to 1..T, so check against the source instead.
    train = df[df["LapNumber"] <= 20]
    assert data["N"] == len(train)
    assert data["T"] == train["LapNumber"].nunique()
    assert float(held["LapNumber"].iloc[0]) == 21.0


def test_the_forecast_lap_carries_its_own_age_and_fuel():
    """Both are known before the lap is driven, so both are legitimate inputs.

    Tyre age is a count and fuel comes from the burn model. Neither is an
    observation of the lap being predicted, which is the line that matters.
    """
    df = field()
    data, compounds, held = build_data(
        df, train_to=20, pred_driver="D00", pred_lap=21, race_laps=30, start_kg=110.0
    )
    assert data["pred_tyre_life"][0] == float(held["TyreLife"].iloc[0])
    assert 0.0 < data["pred_fuel_mass"][0] < 110.0
    assert compounds[data["pred_compound"][0] - 1] == str(held["Compound"].iloc[0])


def test_fuel_falls_over_the_race():
    """A rising fuel vector would flip the sign of gamma and hide it."""
    df = field()
    data, _, _ = build_data(
        df, train_to=25, pred_driver="D00", pred_lap=26, race_laps=30, start_kg=110.0
    )
    fuel = np.array(data["fuel_mass"])
    lap = np.array(data["lap"])
    assert fuel[lap == lap.min()].mean() > fuel[lap == lap.max()].mean()


def test_pit_flags_mark_the_reset_lap_only():
    """The latent state resets on a pit lap, so a wrong flag moves pace levels."""
    df = field(n_drivers=MIN_DRIVERS, n_laps=20)
    data, _, _ = build_data(
        df, train_to=18, pred_driver="D00", pred_lap=19, race_laps=20, start_kg=110.0
    )
    pit = np.array(data["pit"])
    # Every car starts a stint on TyreLife 1: lap 1 and its own pit lap.
    assert pit.sum(axis=1).min() >= 1
    assert pit.shape == (data["D"], data["T"])


def test_a_compound_the_field_has_not_run_is_refused():
    """Its slope would be a draw from the prior, not an estimate."""
    df = field()
    df.loc[
        (df["Driver"] == "D00") & (df["LapNumber"] == 21), "Compound"
    ] = "INTERMEDIATE"
    with pytest.raises(ValueError, match="unseen"):
        build_data(
            df, train_to=20, pred_driver="D00", pred_lap=21, race_laps=30, start_kg=110.0
        )


def test_too_few_cars_is_refused():
    """Pooling is the entire point; with three cars we are not doing it."""
    df = field(n_drivers=3)
    with pytest.raises(ValueError, match="cars in the window"):
        build_data(
            df, train_to=20, pred_driver="D00", pred_lap=21, race_laps=30, start_kg=110.0
        )


def test_a_missing_forecast_lap_is_refused_not_guessed():
    df = field()
    df = df[~((df["Driver"] == "D00") & (df["LapNumber"] == 21))]
    with pytest.raises(ValueError, match="no lap"):
        build_data(
            df, train_to=20, pred_driver="D00", pred_lap=21, race_laps=30, start_kg=110.0
        )


def test_fuel_load_follows_the_regulations_not_the_default():
    """Pins the bug that inflated gamma.

    Passing the 2026 load into a 2024 race understates the mass range, so gamma
    rises to explain the same lap-time gain over fewer kilograms. Measured on
    Bahrain 2024 (paper/results/benchmark.json): 0.0557 s/kg at a 70 kg start
    against 0.0356 at the correct 110 kg, and 0.0557 * 70/110 = 0.0355, so the
    rescaling recovers the correct value to three decimals. (An earlier version
    of this docstring quoted 0.0501 from a run that was never recorded.)
    """
    assert fuel_start_kg(2024) == FUEL_RACE_KG_LEGACY
    assert fuel_start_kg(2025) == FUEL_RACE_KG_LEGACY
    assert fuel_start_kg(2026) == FUEL_RACE_KG_2026
    assert fuel_start_kg(2027) == FUEL_RACE_KG_2026


def test_the_fuel_load_actually_reaches_the_stan_data():
    """A threaded parameter that is ignored is worse than one that is absent."""
    df = field()
    legacy, _, _ = build_data(
        df, train_to=20, pred_driver="D00", pred_lap=21, race_laps=30, start_kg=110.0
    )
    modern, _, _ = build_data(
        df, train_to=20, pred_driver="D00", pred_lap=21, race_laps=30, start_kg=70.0
    )
    assert max(legacy["fuel_mass"]) > max(modern["fuel_mass"])
    assert legacy["pred_fuel_mass"][0] > modern["pred_fuel_mass"][0]


def test_driver_and_lap_indices_are_within_their_declared_bounds():
    """Stan will reject an out-of-range index, but only at runtime."""
    df = field()
    data, _, _ = build_data(
        df, train_to=22, pred_driver="D03", pred_lap=23, race_laps=30, start_kg=110.0
    )
    assert min(data["driver"]) >= 1 and max(data["driver"]) <= data["D"]
    assert min(data["lap"]) >= 1 and max(data["lap"]) <= data["T"]
    assert min(data["compound"]) >= 1 and max(data["compound"]) <= data["C"]
    assert 1 <= data["pred_driver"][0] <= data["D"]

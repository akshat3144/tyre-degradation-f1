"""Fit the hierarchical race model and forecast one lap ahead.

This exists to beat the published benchmark at its own task. Their model
forecasts one driver's next lap from that driver's own history; it does that
well, and our within-transformation design -- built for separating compounds --
loses to it by about 0.04 CRPS on their race.

The gap is structural rather than a tuning gap. Their model has a latent pace
state per lap, which tracks form, traffic and fuel drift without naming any of
them. Ours has a degradation rate estimated from twenty cars, which theirs
cannot get from one. This module puts both in the same model.

WHAT IS AND IS NOT AVAILABLE AT FORECAST TIME

Every fit sees laps up to and including ``train_to``, for the whole field, and
nothing after. That is the same rule the non-Bayesian scorer follows: a pit wall
sees every car, so pooling is legitimate, but nobody sees lap k+1. The forecast
lap's tyre age and fuel mass ARE passed in, because both are known before the
lap is driven -- age is a count and fuel comes from the burn model, neither is
an observation of the lap being predicted.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FUEL_RACE_KG_2026, FUEL_RACE_KG_LEGACY, MASS_SENSITIVITY_S_PER_KG
from ..data.fuel import race_fuel_kg

log = logging.getLogger(__name__)

STAN_FILE = Path(__file__).parent / "stan" / "hier_race.stan"


def fuel_start_kg(season: int) -> float:
    """Race-start fuel load for a season, in kg.

    This matters more than it looks. Feeding the 2026 load of 70 kg into a 2024
    race understates the mass range the model gets to explain, so gamma inflates
    to cover the same lap-time gain over fewer kilograms. Refit by
    ``scripts/paper/p4_benchmark.py`` on the whole Bahrain 2024 race: 0.0557 s/kg
    with a 70 kg start against 0.0356 with the correct 110 kg, and
    0.0557 * 70/110 = 0.0355 -- the rescaling matches to three decimals. (An
    earlier version of this note quoted 0.0501 for the same case, from a run
    that was not recorded and does not reproduce; and the
    "reads 0.043, about 30% high" in the notes could not be reproduced on any
    race we refit. Whole-race gamma at the correct load is 0.020 at Austria 2025
    and 0.036 at Bahrain 2024, with a diffuse prior giving the same numbers, so
    it is race-dependent and neither "recovers the physical value" nor "30%
    high" holds as a general statement.) Under the pre-2026 regulations cars
    started on about 110 kg, which is also what the benchmark paper assumed.
    """
    return FUEL_RACE_KG_2026 if season >= 2026 else FUEL_RACE_KG_LEGACY


#: Sampler settings. Deliberately modest: this is refit once per fold, and the
#: quantity we need from it is a predictive distribution rather than a tight
#: posterior for any single parameter.
CHAINS = 4
WARMUP = 400
SAMPLES = 400

#: A fold needs enough laps behind it for the latent walk to mean anything.
#: Four, not eight: at eight this silently skipped a fold the other scorer
#: predicts (Bahrain 2024, lap 9), and dropping folds we find hard is exactly
#: how a score gets flattered. Twenty cars over four laps is eighty
#: observations, which is ample for the shared slopes; the only thing a short
#: window genuinely weakens is the walk, and that is the driver's own level.
MIN_TRAIN_LAPS = 4

#: Minimum cars required before the pooled slopes are believable. Below this we
#: are not doing the thing this model exists to do.
MIN_DRIVERS = 6


@dataclass
class Forecast:
    """Predictive draws for one held-out lap."""

    draws: np.ndarray
    actual: float
    driver: str
    lap: int
    #: Posterior mean degradation rate per compound label, for inspection.
    rates: dict[str, float]
    #: Posterior mean fuel coefficient. Compared against the physical prior as
    #: evidence the pooling worked, exactly as we did to their model.
    gamma: float
    #: Every posterior draw of the fuel coefficient, for intervals.
    gamma_draws: np.ndarray | None = None


#: Windows toolchain locations, most preferred first. CmdStan needs a 64-bit
#: g++; this machine also has a 32-bit MinGW 6.3 earlier on PATH, which fails
#: with "sorry, unimplemented: 64-bit mode not compiled in" AFTER stanc has
#: already succeeded, so the error looks like a model bug and is not one.
#: Prepending the right toolchain here means a fresh shell does not have to
#: remember to.
_WIN_TOOLCHAINS = (
    Path("C:/rtools45/x86_64-w64-mingw32.static.posix/bin"),
    Path("C:/rtools44/x86_64-w64-mingw32.static.posix/bin"),
    Path("C:/rtools43/x86_64-w64-mingw32.static.posix/bin"),
)
_WIN_MAKE = (
    Path("C:/rtools45/usr/bin"),
    Path("C:/rtools44/usr/bin"),
    Path("C:/rtools43/usr/bin"),
)


def ensure_toolchain() -> None:
    """Put a 64-bit g++ and make ahead of anything else on PATH, on Windows."""
    if os.name != "nt":
        return
    prepend = [
        str(p)
        for p in (*_WIN_TOOLCHAINS, *_WIN_MAKE)
        if (p / "g++.exe").exists() or (p / "make.exe").exists()
    ]
    if not prepend:
        log.warning("no 64-bit RTools toolchain found; Stan compilation may fail")
        return
    current = os.environ.get("PATH", "")
    missing = [p for p in prepend if p not in current]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, current])


@lru_cache(maxsize=1)
def _model():
    """Compile once per process. Import is local so the package stays usable
    without cmdstanpy installed."""
    from cmdstanpy import CmdStanModel

    ensure_toolchain()
    return CmdStanModel(stan_file=str(STAN_FILE))


def build_data(
    laps: pd.DataFrame,
    train_to: int,
    pred_driver: str,
    pred_lap: int,
    race_laps: int,
    start_kg: float,
    gamma_prior: tuple[float, float] = (0.032, 0.02),
) -> tuple[dict, list[str], pd.DataFrame]:
    """Assemble the Stan data block for one fold.

    Args:
        laps: race laps for the whole field.
        train_to: last lap the model may see.
        pred_driver: whose lap is being forecast.
        pred_lap: the held-out lap, normally ``train_to + 1``.
        race_laps: scheduled race distance, for the fuel model.
        start_kg: race-start fuel load; see fuel_start_kg.
        gamma_prior: (mean, sd) of the normal prior on the fuel coefficient.
            The default is the physical 0.030-0.035 s/kg range. A diffuse sd
            shows what the data say without it; see scripts/paper/p4_benchmark.

    Returns:
        The data dict, the compound labels in index order, and the held-out row.
    """
    df = laps[laps["LapTimeSeconds"].notna()].copy()

    train = df[df["LapNumber"] <= train_to]
    held = df[(df["Driver"] == pred_driver) & (df["LapNumber"] == pred_lap)]
    if held.empty:
        raise ValueError(f"no lap {pred_lap} for {pred_driver!r}")
    if train["LapNumber"].nunique() < MIN_TRAIN_LAPS:
        raise ValueError(f"only {train['LapNumber'].nunique()} training laps")

    # Drivers who actually have laps in the window. The forecast driver must be
    # among them or there is no latent state to walk forward.
    drivers = sorted(train["Driver"].unique())
    if pred_driver not in drivers:
        raise ValueError(f"{pred_driver!r} has no laps up to {train_to}")
    if len(drivers) < MIN_DRIVERS:
        raise ValueError(f"only {len(drivers)} cars in the window")

    # Compounds are keyed by the physical label present in the window. The
    # held-out lap's compound must be one the field has already run, or its
    # slope would be a draw from the prior alone.
    compounds = sorted(train["Compound"].unique())
    held_compound = str(held["Compound"].iloc[0])
    if held_compound not in compounds:
        raise ValueError(f"compound {held_compound!r} unseen in the window")

    d_index = {d: i + 1 for i, d in enumerate(drivers)}
    c_index = {c: i + 1 for i, c in enumerate(compounds)}

    train = train[train["Driver"].isin(drivers)].sort_values(["Driver", "LapNumber"])

    # Laps are re-indexed to 1..T so the latent matrix has no unused columns
    # when the window starts partway through a race.
    laps_present = sorted(train["LapNumber"].unique())
    t_index = {int(v): i + 1 for i, v in enumerate(laps_present)}
    n_t = len(laps_present)

    fuel = race_fuel_kg(train["LapNumber"].to_numpy(), race_laps=race_laps, start_kg=start_kg)

    # pit[d, t] = 1 when the car started lap t on new tyres. TyreLife == 1 is
    # the timing feed's own marker for that, and it is what resets the state.
    pit = np.zeros((len(drivers), n_t), dtype=int)
    for _, r in train.iterrows():
        if float(r["TyreLife"]) <= 1.0:
            pit[d_index[r["Driver"]] - 1, t_index[int(r["LapNumber"])] - 1] = 1

    held_fuel = float(
        race_fuel_kg(np.array([pred_lap], dtype=float), race_laps=race_laps, start_kg=start_kg)[0]
    )

    data = {
        "N": len(train),
        "D": len(drivers),
        "T": n_t,
        "C": len(compounds),
        "driver": [d_index[d] for d in train["Driver"]],
        "lap": [t_index[int(v)] for v in train["LapNumber"]],
        "compound": [c_index[c] for c in train["Compound"]],
        "tyre_life": train["TyreLife"].astype(float).tolist(),
        "fuel_mass": [float(x) for x in fuel],
        "y": train["LapTimeSeconds"].astype(float).tolist(),
        "pit": pit.tolist(),
        # Centre the level prior on the field's own pace, net of the fuel the
        # physical model says is on board. Using the raw median would put the
        # prior a second or more away from where the states must sit.
        "level0": float(
            np.median(train["LapTimeSeconds"].astype(float) - MASS_SENSITIVITY_S_PER_KG * fuel)
        ),
        "sdo0": float(max(train.groupby("Driver")["LapTimeSeconds"].std().median(), 0.2)),
        "gamma_prior_mean": float(gamma_prior[0]),
        "gamma_prior_sd": float(gamma_prior[1]),
        "n_pred": 1,
        "pred_driver": [d_index[pred_driver]],
        "pred_compound": [c_index[held_compound]],
        "pred_tyre_life": [float(held["TyreLife"].iloc[0])],
        "pred_fuel_mass": [held_fuel],
    }
    return data, compounds, held


def forecast_lap(
    laps: pd.DataFrame,
    train_to: int,
    pred_driver: str,
    pred_lap: int,
    race_laps: int,
    start_kg: float,
    seed: int = 0,
    gamma_prior: tuple[float, float] = (0.032, 0.02),
) -> Forecast:
    """Fit on laps <= train_to and return predictive draws for one lap."""
    data, compounds, held = build_data(
        laps, train_to, pred_driver, pred_lap, race_laps, start_kg, gamma_prior
    )

    # Explicit inits. Stan's default is uniform(-2, 2) on the unconstrained
    # scale, which for a T-step random walk compounds into a latent pace
    # thousands of seconds from any lap time, and every draw is rejected until
    # adaptation claws it back. The sampler recovers, but it prints a wall of
    # "Location parameter is inf" on the way, which is not something to have on
    # screen during a demo. Starting the walk flat and the scales small is the
    # obvious place to begin, not a tuned choice.
    inits = {
        "sdo": data["sdo0"],
        "sdp": 0.1,
        "z_raw": np.zeros((data["D"], data["T"])).tolist(),
        "level_raw": np.zeros(data["D"]).tolist(),
        "level_mu": data["level0"],
        "level_sd": 0.3,
        "v": [0.05] * data["C"],
        "gamma": MASS_SENSITIVITY_S_PER_KG,
        # Start moderately heavy-tailed rather than at the normal limit, so the
        # sampler explores tail weight instead of having to climb out of it.
        "nu": 6.0,
    }

    fit = _model().sample(
        data=data,
        chains=CHAINS,
        iter_warmup=WARMUP,
        iter_sampling=SAMPLES,
        seed=seed,
        inits=inits,
        show_progress=False,
        show_console=False,
    )

    draws = np.asarray(fit.stan_variable("y_pred")).reshape(-1)
    v = np.asarray(fit.stan_variable("v"))
    gamma_draws = np.asarray(fit.stan_variable("gamma"))
    gamma = float(gamma_draws.mean())

    return Forecast(
        draws=draws,
        actual=float(held["LapTimeSeconds"].iloc[0]),
        driver=pred_driver,
        lap=int(pred_lap),
        rates={c: float(v[:, i].mean()) for i, c in enumerate(compounds)},
        gamma=gamma,
        gamma_draws=gamma_draws,
    )

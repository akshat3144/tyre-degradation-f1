"""Simulation recovery study: does the estimator recover planted tyre degradation
when pit-stop timing is strategic, and how do simpler estimators fail?

STATED BEFORE THE FIRST FULL RUN. Nothing below was tuned on results; the only
runs made while writing the code were tiny smoke tests for bugs and run time.

THE QUESTION
    The paper estimates lap-time loss per lap of tyre age, per compound, from
    public race lap times by comparing cars within each (event, lap) and fitting
    a REML mixed model (Eq. 1: ``models.mixed``, built by
    ``model_checks.build_model``). A referee asks whether that recovers a known
    truth when stops are chosen strategically (the thing the design leans on
    least comfortably), and which simpler estimators fail and how.

DATA-GENERATING PROCESS (one replicate)
    13 events, each with the compound nominations the 2026 allocation table gives
    that event (HARD/MEDIUM/SOFT = C3/C4/C5 at seven of them, C2/C3/C4 at four,
    C1/C2/C3 at two), 20 cars, a 55-lap race. Lap time, in seconds relative to
    the event's base lap, of car i on lap t of a run r on compound c:

        y = a_i                              car/driver pace, per event: N(0, 1.166^2)
          + fuel(t)                          -0.033 s/kg x 1.7 kg/lap x (t - 1)
          + evo_e(t)                         track evolution -A_e (1 - exp(-t / tau_e)),
                                             A_e ~ U(0.4, 1.2) s, tau_e ~ U(10, 25) laps
          + w_e(t)                           weather: AR(1) shock shared by the field,
                                             phi 0.7, marginal SD 0.25 s
          + o_c                              compound offset, +0.4 s per step harder
                                             (C1 +0.8 ... C3 0 ... C5 -0.8)
          + (b_c + d_e + v_r) x age          b_c: TRUE compound slope = the 2026
                                             fitted rates (C1 .08237, C2 .05324,
                                             C3 .05024, C4 .04205, C5 .02172 s/lap);
                                             d_e ~ N(0, .04367^2) circuit deviation;
                                             v_r ~ N(0, .0315^2) run-slope deviation
          + p_r                              run intercept N(0, .28421^2)
          + e_r(t)                           within-run AR(1), phi 0.25, SD .55553

    Variance components are the shipped fit's (``paper/results/model.json``,
    ``shipped_fit.variance_sd``). Tyre age is 1 on the first lap of a run. As in
    the real pipeline, each run's out-lap (age 1) and the in-lap of every run but
    the last are dropped before anything is demeaned, so ages run 2..L-1.
    Fresh tyres every stop, no retirements, no safety cars and NO TRAFFIC term
    (the paper's fourth confounder is not simulated; Eq. 1 is therefore fitted
    without its traffic covariate here). All of these favour every estimator.

    Each car runs 2 (60%) or 3 (40%) stints on compounds drawn uniformly from its
    event's three, with at least two different compounds. Minimum stint 8 laps.
    A run's stop lap is the scenario's business:

    (a) EXOGENOUS   stop j of n at round(55 j / n + N(0, 4^2)), independent of
                    pace and wear.
    (b) PACE        as (a), but faster cars stop later and make fewer stops: every
                    stop shifts by -2 laps per SD of the car's pace (z = a_i / SD;
                    slower is positive), and P(3 stints) = logistic(logit .4 + 1.0 z).
                    Tyre age at a lap is then correlated with car pace.
    (c) WEAR        a car stops when the cumulative lap-time loss since its stop,
                    s A (A + 1) / 2 with s its REALISED slope b_c + d_e + v_r,
                    crosses a tolerance set by its plan: the tolerance a 'typical'
                    slope (0.05) reaches at its planned length 55 / n, times
                    exp(N(0, .15^2)). Steeper realised wear means a shorter run:
                    selection on the outcome. Stops are clipped to the 8-lap
                    minimum and the number of stints is the planned one.

    The three scenarios share EVERY random draw (common random numbers); only the
    stop laps differ, so differences between scenarios are due to pit timing alone.

ESTIMATORS (the same laps, all of them; truth for each compound is b_c)
    naive_pooled_ols               lap time on compound x age, compound intercepts
                                   and event intercepts (so the event's base lap
                                   time cannot make it a strawman). Run-clustered CR1.
    fuel_within_run                lap time + s x 1.7 x (t - 1) with s = .033 (right),
    fuel_within_run_minus20 / plus20   .0264 and .0396 (+-20%); within-run OLS of
                                   that on age per compound (the "practitioner"
                                   estimator of ``models.ablation``). Run-clustered.
    cross_section_cell             lap time and age demeaned within (event, lap,
                                   compound) cells; OLS per compound, no run terms.
    field_demeaned_ols             lap time, age demeaned within (event, lap) as
                                   ``design.race_design`` does; OLS of that on
                                   compound offsets and compound x age, no run
                                   terms. Run-clustered.
    eq1_shipped                    Eq. 1 with the EARLIER, approximate within-
                                   transform (``build_model(exact_within=False)``:
                                   raw compound dummies, dummy x demeaned age);
                                   compound offsets and slopes; random circuit
                                   slope, driver-in-event intercept, run intercept,
                                   run slope; REML with the shipped starts. 95%
                                   Wald. Kept as the baseline the correction is
                                   measured against; NOT the current estimator.
    eq1_mundlak_run_mean           Eq. 1 + the run mean of tyre age, per compound
                                   (correlated random effects).
    eq1_mundlak_run_mean_slope     the same + tyre age x (run mean - compound's
                                   mean run mean), per compound (Wooldridge's
                                   correlated random SLOPES; the pre-stated remedy
                                   for selection of the run slope on run length).
    eq1_exact_within               PRIMARY Eq. 1 estimator: the compound columns
                                   demeaned within (event, lap) EXACTLY,
                                   tilde(D_c) and tilde(D_c x raw age)
                                   (``build_model`` default). The earlier
                                   construction multiplied a compound dummy by the
                                   already-demeaned age; with slopes that differ by
                                   compound that is only approximately the
                                   within-transform of the compound x age column
                                   (the other cars' terms are subtracted at the
                                   wrong slope). The two differ only in this.
    eq1_exact_within_mundlak       eq1_exact_within + the Mundlak run mean of raw
                                   tyre age per compound (correlated random
                                   effects), itself demeaned within (event, lap)
                                   like every fixed-effect column. Added after the
                                   exact-within correction, so it is not among the
                                   pre-stated expectations.

    OLS estimators get two 95% intervals: run-clustered normal (the paper's
    convention) and event-clustered t with G-1 df (what 13 circuits warrant).

METRICS
    Per scenario x estimator x compound: bias, RMSE, 95% coverage, each with its
    Monte Carlo SE, over R replicates; the per-replicate average over compounds
    gives the mean-over-compounds row, with its own exact MC SE. Eq. 1-type fits
    also report the bias of the circuit-slope SD (truth .04367). A cell counts as
    'recovered' when |bias| <= 0.005 s/lap (a tenth of a typical slope) and
    coverage lies in [0.925, 0.975]; these thresholds are reading aids, not tests.

PRE-STATED EXPECTATIONS
    X1  naive_pooled_ols is badly biased low everywhere: fuel burn alone
        (-0.056 s/lap) is as large as the slopes.
    X2  fuel_within_run is biased low by track evolution at the right fuel
        coefficient and by about -/+0.011 s/lap per 20% fuel error.
    X3  cross_section_cell and field_demeaned_ols are fine under (a) but biased
        under (b), where age is correlated with pace and nothing absorbs a car.
    X4  eq1_shipped recovers the slopes under (a) and (b): the driver and run
        intercepts absorb pace.
    X5  Under (c) every estimator, eq1_shipped included, is biased toward zero
        (long runs are the shallow-wear ones). Coverage of eq1_shipped falls below
        nominal there. The correlated-random-slope variant may repair some of it.
    X6  The circuit-slope SD is estimated with modest downward bias from 13
        circuits.
    X7  (added after the exact-within correction, so NOT pre-stated) the primary
        estimator eq1_exact_within recovers the slopes under (a) and (b): mean
        |bias| <= 0.005 and mean coverage in [0.925, 0.975].
    X8  (added with X7) under (c) the exact-within Mundlak variant has smaller mean
        |bias| than eq1_exact_within.
    The numeric rule that scores each of these (X5b is the correlated-random-slope
    clause of X5) is ``check_expectations``; it is fixed in code, not in prose,
    and ``recovery.json`` carries its verdicts.

IMPLEMENTATION NOTES
    Replicate r uses the r-th child of ``SeedSequence(seed)`` for every scenario,
    so results do not depend on scheduling or worker count. The sharded runner
    below follows ``model_checks._run_sharded`` (subprocess shards, per-item
    pickles, resume after a kill, retry of a failed shard) but is its own: that
    one's worker looks task functions up in ``model_checks`` only.

Nothing outside ``tyredeg.paper`` imports this module.
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..models.ablation import cluster_ols
from ..models.design import race_design
from ..models.mixed import C_ORDER
from ..models.reml import RemlModel
from .model_checks import build_model, run_mean_covariates

Z95 = 1.959963984540054

SCENARIOS = ("exogenous", "pace", "wear")

#: HARD, MEDIUM, SOFT as C numbers at the 13 events of the 2026 race fit, in the
#: alphabetical event order of ``model.json`` (verified against
#: ``data/allocation.json`` by ``tests/test_paper_recovery.py``).
DEFAULT_NOMINATIONS: tuple[tuple[str, str, str], ...] = (
    ("C3", "C4", "C5"),  # Australian
    ("C3", "C4", "C5"),  # Austrian
    ("C2", "C3", "C4"),  # Barcelona
    ("C2", "C3", "C4"),  # Belgian
    ("C1", "C2", "C3"),  # British
    ("C3", "C4", "C5"),  # Canadian
    ("C2", "C3", "C4"),  # Chinese
    ("C2", "C3", "C4"),  # Dutch
    ("C3", "C4", "C5"),  # Hungarian
    ("C3", "C4", "C5"),  # Italian
    ("C1", "C2", "C3"),  # Japanese
    ("C3", "C4", "C5"),  # Miami
    ("C3", "C4", "C5"),  # Monaco
)


@dataclass(frozen=True)
class DGP:
    """Everything the simulator needs. Defaults are the study's design."""

    nominations: tuple = DEFAULT_NOMINATIONS
    n_cars: int = 20
    n_laps: int = 55
    p_three_stints: float = 0.4
    min_stint: int = 8
    #: True compound slopes, s per lap of tyre age (the 2026 shipped fit).
    slope: tuple = (("C1", 0.08237), ("C2", 0.05324), ("C3", 0.05024),
                    ("C4", 0.04205), ("C5", 0.02172))
    #: Fresh-tyre pace offset, s (softer is faster).
    offset: tuple = (("C1", 0.8), ("C2", 0.4), ("C3", 0.0), ("C4", -0.4), ("C5", -0.8))
    sd_circuit_slope: float = 0.04367
    sd_driver: float = 1.16643
    sd_run: float = 0.28421
    sd_run_slope: float = 0.0315
    sd_noise: float = 0.55553
    noise_phi: float = 0.25
    fuel_s_per_kg: float = 0.033
    fuel_kg_per_lap: float = 1.7
    evo_amp: tuple = (0.4, 1.2)
    evo_tau: tuple = (10.0, 25.0)
    weather_sd: float = 0.25
    weather_phi: float = 0.7
    stop_jitter_sd: float = 4.0
    pace_shift_laps_per_sd: float = 2.0
    pace_stops_logit_per_sd: float = 1.0
    wear_ref_slope: float = 0.05
    wear_threshold_sd: float = 0.15

    @property
    def truth(self) -> dict[str, float]:
        return dict(self.slope)

    def to_dict(self) -> dict:
        return asdict(self)


def dgp_from_fit(model: dict, allocation: list[dict]) -> DGP:
    """The design built from ``model.json`` and ``allocation.json`` rather than
    from the constants above (the script uses this; a test pins that the two agree).

    The slopes and variance components are those of the 2026 fit made with the
    earlier approximate within-transform: the study's truth is a fixed set of
    planted values chosen to resemble that fit, and it is NOT re-planted when the
    estimator is corrected (so the bias of the old and the new construction are
    measured on the same world)."""
    # The planted truth is the 2026 fit under the EARLIER (approximate) within-transform,
    # frozen when the study was designed. Once model.json carries the corrected fit as
    # ``shipped_fit``, the earlier one lives in ``exact_vs_approximate``.
    sf = (model.get("exact_vs_approximate", {}).get("shipped_2026", {}).get("approximate")
          or model["shipped_fit"])
    events = sorted(model["dataset"]["race_fit"]["events"])
    by_event = {a["event"]: a["compounds"] for a in allocation}
    noms = tuple(tuple(by_event[e][k] for k in ("HARD", "MEDIUM", "SOFT")) for e in events)
    sd = sf["variance_sd"]
    rates = sf["rates_s_per_lap"]
    return DGP(
        nominations=noms,
        slope=tuple((c, float(rates[c]["mean"])) for c in C_ORDER if c in rates),
        sd_circuit_slope=float(sd["circuit_slope"]), sd_driver=float(sd["driver_event"]),
        sd_run=float(sd["run"]), sd_run_slope=float(sd["run_slope"]), sd_noise=float(sd["resid"]),
    )


# ---------------------------------------------------------------------------
# the data-generating process
# ---------------------------------------------------------------------------


def _logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def _plan_stop_laps(cfg: DGP, scenario: str, n: int, jitter, z: float, s_plan, xi) -> list[int]:
    """Last lap of every stint (the last entry is the race distance).

    ``s_plan[j]`` is the REALISED slope of stint j, used by the wear trigger only.
    """
    N, Lmin = cfg.n_laps, cfg.min_stint
    ends: list[int] = []
    done = 0
    for j in range(n - 1):
        lo = done + Lmin
        hi = N - Lmin * (n - 1 - j)
        if scenario == "wear":
            plan = N / n
            theta = cfg.wear_ref_slope * plan * (plan + 1) / 2 * math.exp(cfg.wear_threshold_sd * xi[j])
            s = float(s_plan[j])
            # cumulative loss s A (A + 1) / 2 reaches theta at A = (-1 + sqrt(1 + 8 theta / s)) / 2
            raw = done + math.ceil((-1 + math.sqrt(1 + 8 * theta / s)) / 2) if s > 1e-9 else hi
        else:
            shift = -cfg.pace_shift_laps_per_sd * z if scenario == "pace" else 0.0
            raw = round(N * (j + 1) / n + jitter[j] + shift)
        end = int(min(max(raw, lo), hi))
        ends.append(end)
        done = end
    ends.append(N)
    return ends


def simulate(cfg: DGP, scenario: str, seed) -> pd.DataFrame:
    """One simulated season of races as the clean-lap frame the estimators read.

    Columns: ``event, Driver, run_id, LapNumber, TyreLife, C, LapTimeSeconds,
    session`` (what the shipped pipeline has) plus the truth columns ``pace``
    (car's pace term) and ``true_slope`` (the run's realised slope).

    Every random draw is made in a fixed order that does not depend on the
    scenario, so the same seed gives the same cars, runs, compounds, noise and
    weather under all three pit-timing mechanisms.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
    rng = np.random.default_rng(seed)
    E, I, N, J = len(cfg.nominations), cfg.n_cars, cfg.n_laps, 3
    slope, offset = dict(cfg.slope), dict(cfg.offset)

    # -- draws (fixed order)
    delta = rng.normal(0.0, cfg.sd_circuit_slope, E)
    evo_amp = rng.uniform(*cfg.evo_amp, E)
    evo_tau = rng.uniform(*cfg.evo_tau, E)
    weather_z = rng.standard_normal((E, N))
    pace = rng.normal(0.0, cfg.sd_driver, (E, I))
    u_stints = rng.random((E, I))
    jitter = rng.normal(0.0, cfg.stop_jitter_sd, (E, I, J - 1))
    comp_ix = rng.integers(0, 3, (E, I, J))
    run_int = rng.normal(0.0, cfg.sd_run, (E, I, J))
    run_dev = rng.normal(0.0, cfg.sd_run_slope, (E, I, J))
    xi = rng.standard_normal((E, I, J))
    innov = rng.standard_normal((E, I, N))

    # -- weather: AR(1) shared by the field
    wsd = cfg.weather_sd
    weather = np.empty((E, N))
    weather[:, 0] = wsd * weather_z[:, 0]
    for t in range(1, N):
        weather[:, t] = cfg.weather_phi * weather[:, t - 1] + wsd * math.sqrt(1 - cfg.weather_phi**2) * weather_z[:, t]

    # -- stints
    z = pace / cfg.sd_driver if cfg.sd_driver > 0 else np.zeros_like(pace)
    p3 = np.full((E, I), cfg.p_three_stints)
    if scenario == "pace":
        logit = math.log(cfg.p_three_stints / (1 - cfg.p_three_stints))
        p3 = _logistic(logit + cfg.pace_stops_logit_per_sd * z)
    n_stints = 2 + (u_stints < p3).astype(int)

    runs: list[tuple] = []  # (e, i, j, start, length, final, compound, intercept, realised slope)
    for e in range(E):
        for i in range(I):
            n = int(n_stints[e, i])
            ci = comp_ix[e, i, :n].copy()
            if len(set(ci.tolist())) == 1:
                ci[-1] = (ci[-1] + 1) % 3
            comps = [cfg.nominations[e][k] for k in ci]
            s_real = [slope[c] + delta[e] + run_dev[e, i, j] for j, c in enumerate(comps)]
            ends = _plan_stop_laps(cfg, scenario, n, jitter[e, i], float(z[e, i]), s_real, xi[e, i])
            start = 0
            for j in range(n):
                runs.append((e, i, j, start, ends[j] - start, j == n - 1, comps[j],
                             run_int[e, i, j], s_real[j]))
                start = ends[j]

    e_r = np.array([r[0] for r in runs])
    i_r = np.array([r[1] for r in runs])
    j_r = np.array([r[2] for r in runs])
    start_r = np.array([r[3] for r in runs])
    len_r = np.array([r[4] for r in runs])
    final_r = np.array([r[5] for r in runs])
    comp_r = np.array([r[6] for r in runs])
    rho_r = np.array([r[7] for r in runs])
    s_r = np.array([r[8] for r in runs])
    R = len(runs)

    # -- within-run AR(1) noise, all runs at once
    Lmax = int(len_r.max())
    k = np.arange(Lmax)
    lap_idx = np.minimum(start_r[:, None] + k[None, :], N - 1)
    zz = innov[e_r[:, None], i_r[:, None], lap_idx]
    noise = np.empty((R, Lmax))
    phi, sd = cfg.noise_phi, cfg.sd_noise
    noise[:, 0] = sd * zz[:, 0]
    for t in range(1, Lmax):
        noise[:, t] = phi * noise[:, t - 1] + sd * math.sqrt(1 - phi**2) * zz[:, t]

    # -- expand to laps
    rep = np.repeat(np.arange(R), len_r)
    first = np.cumsum(len_r) - len_r
    kk = np.arange(len(rep)) - first[rep]
    age = (kk + 1).astype(float)
    lap = start_r[rep] + kk + 1
    keep = (age >= 2) & ((age <= len_r[rep] - 1) | final_r[rep])

    ev, car = e_r[rep], i_r[rep]
    fuel = -cfg.fuel_s_per_kg * cfg.fuel_kg_per_lap * (lap - 1)
    evo = -evo_amp[ev] * (1.0 - np.exp(-lap / evo_tau[ev]))
    off = np.array([offset[c] for c in comp_r])[rep]
    y = (pace[ev, car] + fuel + evo + weather[ev, lap - 1] + off
         + s_r[rep] * age + rho_r[rep] + noise[rep, kk])

    names = np.array([f"E{e:02d}|D{i:02d}|{j}" for e, i, j in zip(e_r, i_r, j_r, strict=True)])
    df = pd.DataFrame({
        "event": np.array([f"E{e:02d}" for e in range(E)])[ev],
        "Driver": np.array([f"D{i:02d}" for i in range(I)])[car],
        "run_id": names[rep],
        "LapNumber": lap,
        "TyreLife": age,
        "C": comp_r[rep],
        "LapTimeSeconds": y,
        "session": "R",
        "pace": pace[ev, car],
        "true_slope": s_r[rep],
    })
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

#: name -> description, in reporting order.
ESTIMATORS: dict[str, str] = {
    "naive_pooled_ols": "pooled OLS, compound x age, compound and event intercepts",
    "fuel_within_run": "fuel-corrected (s/kg .033, right) within-run OLS",
    "fuel_within_run_minus20": "fuel-corrected, s/kg 20% too low, within-run OLS",
    "fuel_within_run_plus20": "fuel-corrected, s/kg 20% too high, within-run OLS",
    "cross_section_cell": "within (event, lap, compound) cells, OLS, no run terms",
    "field_demeaned_ols": "within (event, lap) demeaned OLS, no run terms",
    "eq1_shipped": "Eq. 1, earlier approximate within-transform (REML)",
    "eq1_mundlak_run_mean": "Eq. 1 + Mundlak run mean of age per compound",
    "eq1_mundlak_run_mean_slope": "Eq. 1 + Mundlak run mean and age x run mean per compound",
    "eq1_exact_within": "Eq. 1, compound columns demeaned exactly within (event, lap) [primary]",
    "eq1_exact_within_mundlak": "Eq. 1 exact within + Mundlak run mean of age per compound",
}
#: The estimator that IS Eq. 1 as now fitted; ``eq1_shipped`` is the earlier approximation.
PRIMARY_EQ1 = "eq1_exact_within"
OLS_ESTIMATORS = tuple(k for k in ESTIMATORS if not k.startswith("eq1"))
REML_ESTIMATORS = tuple(k for k in ESTIMATORS if k.startswith("eq1"))
NAN = float("nan")


def design_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The shipped within-(event, lap) transform on the simulated clean laps."""
    return race_design(frame, with_traffic=False).reset_index(drop=True)


def _ols(X: np.ndarray, y: np.ndarray, cols: dict[str, int], run_ids, event_ids) -> dict:
    """OLS with run- and event-clustered errors; one row per compound in ``cols``."""
    beta, cov_run, _ = cluster_ols(X, y, run_ids)
    _, cov_ev, _ = cluster_ols(X, y, event_ids)
    g = len(pd.unique(np.asarray(event_ids)))
    crit = float(stats.t.ppf(0.975, g - 1)) if g > 1 else NAN
    return {
        c: [float(beta[j]), float(np.sqrt(max(cov_run[j, j], 0.0))),
            float(np.sqrt(max(cov_ev[j, j], 0.0))), crit]
        for c, j in cols.items()
    }


def _within(v: np.ndarray, groups) -> np.ndarray:
    return v - pd.Series(v).groupby(np.asarray(groups)).transform("mean").to_numpy()


def est_naive(d: pd.DataFrame, cats: list[str]) -> dict:
    C = d["C"].to_numpy()
    ev = d["event"].astype(str).to_numpy()
    events = sorted(set(ev))
    age = d["TyreLife"].to_numpy(float)
    cols = [(C == c).astype(float) for c in cats] + [(ev == e).astype(float) for e in events[1:]]
    cols += [(C == c) * age for c in cats]
    X = np.column_stack(cols)
    j0 = len(cats) + len(events) - 1
    return _ols(X, d["LapTimeSeconds"].to_numpy(float), {c: j0 + i for i, c in enumerate(cats)},
                d["run_id"].to_numpy(), ev)


def est_fuel_within_run(d: pd.DataFrame, cats: list[str], cfg: DGP, s_per_kg: float) -> dict:
    """Fuel-corrected lap time on age, within run, per compound."""
    yc = d["LapTimeSeconds"].to_numpy(float) + s_per_kg * cfg.fuel_kg_per_lap * (d["LapNumber"].to_numpy(float) - 1)
    run = d["run_id"].to_numpy()
    y = _within(yc, run)
    x = _within(d["TyreLife"].to_numpy(float), run)
    C = d["C"].to_numpy()
    ev = d["event"].astype(str).to_numpy()
    out = {}
    for c in cats:
        m = C == c
        if m.sum() > 10 and float(x[m] @ x[m]) > 0:
            out[c] = _ols(x[m][:, None], y[m], {c: 0}, run[m], ev[m])[c]
    return out


def est_cross_section(d: pd.DataFrame, cats: list[str]) -> dict:
    """Cars on the same compound at the same lap, compared directly."""
    cell = (d["event"].astype(str) + "|" + d["LapNumber"].astype(str) + "|" + d["C"].astype(str)).to_numpy()
    size = pd.Series(cell).groupby(cell).transform("size").to_numpy()
    ok = size >= 2
    cell, C = cell[ok], d["C"].to_numpy()[ok]
    run, ev = d["run_id"].to_numpy()[ok], d["event"].astype(str).to_numpy()[ok]
    y = _within(d["LapTimeSeconds"].to_numpy(float)[ok], cell)
    x = _within(d["TyreLife"].to_numpy(float)[ok], cell)
    out = {}
    for c in cats:
        m = C == c
        if m.sum() > 10 and float(x[m] @ x[m]) > 0:
            out[c] = _ols(x[m][:, None], y[m], {c: 0}, run[m], ev[m])[c]
    return out


def est_field_demeaned(d: pd.DataFrame, cats: list[str]) -> dict:
    C = d["C"].to_numpy()
    tl = d["tl"].to_numpy(float)
    X = np.column_stack([(C == c).astype(float) for c in cats] + [(C == c) * tl for c in cats])
    return _ols(X, d["y"].to_numpy(float), {c: len(cats) + i for i, c in enumerate(cats)},
                d["run_id"].to_numpy(), d["event"].astype(str).to_numpy())


def _reml_table(fit, names: list, s1: float, cats: list[str]) -> dict:
    out = {}
    for c in cats:
        j = names.index(("tl", c))
        out[c] = [float(fit.beta[j] / s1), float(math.sqrt(max(fit.cov_beta[j, j], 0.0)) / s1), NAN, NAN]
    return out


def _reml_flags(fit, s1: float) -> dict:
    return {"circuit_sd": float(fit.sd["circuit_slope"] / s1), "converged": bool(fit.converged),
            "circuit_on_boundary": "circuit_slope" in fit.on_boundary}


def est_eq1_family(d: pd.DataFrame, which: tuple[str, ...]) -> tuple[dict, dict]:
    """Eq. 1 with the earlier approximate transform, its variants, and the exact one.

    ``eq1_shipped`` and the two ``eq1_mundlak_run_mean*`` variants use the EARLIER
    construction (``exact_within=False``) so the correction is measured against
    exactly what was run before; ``eq1_exact_within`` and
    ``eq1_exact_within_mundlak`` use the current default. ``which`` selects.
    """
    est: dict = {}
    flags: dict = {}
    model, meta = build_model(d, exact_within=False)
    cats, s1, names = meta["cats"], meta["s1"], meta["names"]
    k = len(model.comps)
    f5 = model.fit(starts=[np.full(k, 0.5), np.full(k, 0.1), np.linspace(2.0, 0.2, k)])
    if "eq1_shipped" in which:
        est["eq1_shipped"] = _reml_table(f5, names, s1, cats)
        flags["eq1_shipped"] = _reml_flags(f5, s1)
    warm = [f5.theta, np.full(k, 0.3)]
    ev = d["event"].astype(str).to_numpy()
    C = d["C"].to_numpy()
    tl = d["tl"].to_numpy(float) / s1
    y = d["y"].to_numpy(float)

    if "eq1_mundlak_run_mean" in which or "eq1_mundlak_run_mean_slope" in which:
        run = d["run_id"].to_numpy()
        zr = pd.Series(tl).groupby(run).transform("mean").to_numpy()
        per_run = pd.DataFrame({"run": run, "C": C, "z": zr}).drop_duplicates("run")
        zbar = per_run.groupby("C")["z"].mean()
        zc = zr - pd.Series(C).map(zbar).to_numpy()
        mund = [(C == c) * zc for c in cats]
        mund_slope = [(C == c) * zc * tl for c in cats]
        for key, extra in (("eq1_mundlak_run_mean", mund),
                           ("eq1_mundlak_run_mean_slope", mund + mund_slope)):
            if key in which:
                m = RemlModel(y, np.column_stack([model.X, *extra]), ev, model.comps)
                f = m.fit(starts=warm)
                est[key] = _reml_table(f, names, s1, cats)
                flags[key] = _reml_flags(f, s1)

    if "eq1_exact_within" in which:
        mx, metax = build_model(d)
        f = mx.fit(starts=warm)
        est["eq1_exact_within"] = _reml_table(f, metax["names"], metax["s1"], metax["cats"])
        flags["eq1_exact_within"] = _reml_flags(f, metax["s1"])
    if "eq1_exact_within_mundlak" in which:
        mx, metax = build_model(d, extra_fixed=run_mean_covariates(d))
        f = mx.fit(starts=warm)
        est["eq1_exact_within_mundlak"] = _reml_table(f, metax["names"], metax["s1"], metax["cats"])
        flags["eq1_exact_within_mundlak"] = _reml_flags(f, metax["s1"])
    return est, flags


def estimate_all(frame: pd.DataFrame, cfg: DGP, which: tuple[str, ...] | None = None) -> dict:
    """Every requested estimator on one simulated frame.

    Returns ``{"est": {estimator: {compound: [est, se_run_or_wald, se_event, t_crit]}},
    "flags": {...}, "errors": {...}}``. An estimator that fails contributes nothing
    for that replicate and its error is recorded, never swallowed silently.
    """
    which = tuple(ESTIMATORS) if which is None else tuple(which)
    d = design_frame(frame)
    cats = [c for c in C_ORDER if c in set(d["C"])]
    est: dict = {}
    flags: dict = {}
    errors: dict = {}

    def run(key, fn):
        if key not in which:
            return
        try:
            est[key] = fn()
        except Exception as exc:  # noqa: BLE001 -- recorded and counted
            errors[key] = f"{type(exc).__name__}: {str(exc)[:120]}"

    run("naive_pooled_ols", lambda: est_naive(d, cats))
    for key, mult in (("fuel_within_run", 1.0), ("fuel_within_run_minus20", 0.8),
                      ("fuel_within_run_plus20", 1.2)):
        run(key, lambda mult=mult: est_fuel_within_run(d, cats, cfg, cfg.fuel_s_per_kg * mult))
    run("cross_section_cell", lambda: est_cross_section(d, cats))
    run("field_demeaned_ols", lambda: est_field_demeaned(d, cats))
    reml_which = tuple(k for k in which if k in REML_ESTIMATORS)
    if reml_which:
        try:
            e2, f2 = est_eq1_family(d, reml_which)
            est.update(e2)
            flags.update(f2)
        except Exception as exc:  # noqa: BLE001
            for k in reml_which:
                errors[k] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return {"est": est, "flags": flags, "errors": errors}


def frame_diagnostics(frame: pd.DataFrame) -> dict:
    """What the scenario did to the design: the quantities the mechanisms act on."""
    d = design_frame(frame)
    run_len = frame.groupby("run_id").size()
    per_run = frame.drop_duplicates("run_id").set_index("run_id")
    runs = pd.DataFrame({"len": run_len, "slope": per_run["true_slope"]})
    car = frame.drop_duplicates(["event", "Driver"])
    stints = frame.groupby(["event", "Driver"])["run_id"].nunique()
    mean_age = frame.groupby(["event", "Driver"])["TyreLife"].mean()
    pace = frame.drop_duplicates(["event", "Driver"]).set_index(["event", "Driver"])["pace"]
    return {
        "n_laps": int(len(d)), "n_runs": int(len(run_len)),
        "mean_clean_run_length": float(run_len.mean()),
        "share_three_stints": float((stints == 3).mean()),
        "corr_age_pace_within_lap": float(np.corrcoef(d["tl"], d["pace"] - d.groupby(["event", "LapNumber"])["pace"].transform("mean"))[0, 1]),
        "corr_car_mean_age_pace": float(np.corrcoef(mean_age.loc[pace.index], pace)[0, 1]),
        "corr_run_length_run_slope": float(np.corrcoef(runs["len"], runs["slope"])[0, 1]),
        "mean_cars_per_cell": float(d["cars_on_lap"].mean()),
        "n_cars": int(len(car)),
    }


def run_one(cfg: DGP, scenario: str, seed, which: tuple[str, ...] | None = None) -> dict:
    """One replicate of one scenario: simulate, estimate, diagnose."""
    t0 = time.time()
    warnings.filterwarnings("ignore")
    frame = simulate(cfg, scenario, seed)
    out = estimate_all(frame, cfg, which)
    out["diag"] = frame_diagnostics(frame)
    out["scenario"] = scenario
    out["seconds"] = round(time.time() - t0, 2)
    return out


# ---------------------------------------------------------------------------
# the sharded, resumable runner
# ---------------------------------------------------------------------------


def _single_thread_children() -> None:
    for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"


def _replace(src: str, dst: str) -> None:
    """``os.replace``, retried: Windows refuses it while another process reads ``dst``."""
    for _ in range(50):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.2)
    os.replace(src, dst)


def worker_main(argv: list[str]) -> int:
    """One shard, run as ``python -m tyredeg.paper.recovery JOB SHARD N_SHARDS OUT``."""
    job_path, shard, n_shards, out_path = argv[0], int(argv[1]), int(argv[2]), argv[3]
    warnings.filterwarnings("ignore")
    with open(job_path, "rb") as fh:
        job = pickle.load(fh)
    partial = out_path + ".partial"
    out: list = []
    if os.path.exists(partial):
        with open(partial, "rb") as fh:
            out = pickle.load(fh)
    done = {i for i, _ in out}
    for i in range(shard, len(job["items"]), n_shards):
        if i in done:
            continue
        scenario, _rep, seed = job["items"][i]
        out.append((i, run_one(job["cfg"], scenario, seed, job["which"])))
        with open(partial + ".tmp", "wb") as fh:
            pickle.dump(out, fh)
        _replace(partial + ".tmp", partial)
    with open(out_path + ".tmp", "wb") as fh:
        pickle.dump(out, fh)
    _replace(out_path + ".tmp", out_path)
    return 0


def run_replicates(cfg: DGP, scenarios: tuple[str, ...], reps: int, seed: int, workers: int = 1,
                   which: tuple[str, ...] | None = None, retries: int = 3,
                   cache_root=None) -> list[dict]:
    """Replicates ``0..reps-1`` of every scenario, in a fixed order.

    Replicate ``r`` of every scenario uses the ``r``-th child of
    ``SeedSequence(seed)``. ``workers <= 1`` runs in this process; otherwise the
    items are dealt round-robin to ``workers`` interpreters. A shard that fails
    is rerun, and a killed run resumes from the per-item pickles in
    ``data/processed/shard_cache``. The result is identical either way.
    """
    seeds = np.random.SeedSequence(seed).spawn(reps)
    which = None if which is None else tuple(which)
    items = [(s, r, seeds[r]) for r in range(reps) for s in scenarios]
    if workers <= 1:
        return [{**run_one(cfg, s, sd, which), "rep": r} for s, r, sd in items]

    from tyredeg.config import PROCESSED

    n = max(1, min(workers, len(items)))
    _single_thread_children()
    blob = pickle.dumps({"cfg": cfg, "which": which, "items": items})
    root = (PROCESSED / "shard_cache") if cache_root is None else cache_root
    tmp = root / f"recovery_{n}_{hashlib.sha1(blob).hexdigest()[:16]}"
    tmp.mkdir(parents=True, exist_ok=True)
    job = tmp / "job.pkl"
    if not job.exists():
        job.write_bytes(blob)
    results: dict[int, dict] = {}
    pending = list(range(n))
    t_start = time.time()
    for attempt in range(retries + 1):
        procs = {}
        print(f"  [recovery] attempt {attempt}: shards {pending} of {n} for {len(items)} items", flush=True)
        for k in pending:
            out = tmp / f"out{k}.pkl"
            errf = open(tmp / f"err{k}.txt", "wb")
            procs[k] = (subprocess.Popen(
                [sys.executable, "-m", "tyredeg.paper.recovery", str(job), str(k), str(n), str(out)],
                stdout=subprocess.DEVNULL, stderr=errf), out, errf)
        failed = []
        for k, (pr, out, errf) in procs.items():
            pr.wait()
            errf.close()
            if pr.returncode == 0 and out.exists():
                with open(out, "rb") as fh:
                    results.update(dict(pickle.load(fh)))
                print(f"  [recovery] shard {k} done at {time.time() - t_start:.0f}s", flush=True)
            else:
                tail = (tmp / f"err{k}.txt").read_text(errors="replace")[-400:]
                print(f"  [recovery] shard {k} FAILED rc={pr.returncode}: {tail}", flush=True)
                failed.append(k)
        pending = failed
        if not pending:
            break
    else:
        raise RuntimeError(f"recovery shards {pending} kept failing")
    shutil.rmtree(tmp, ignore_errors=True)
    return [{**results[i], "rep": items[i][1]} for i in range(len(items))]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

BIAS_TOLERANCE = 0.005
COVERAGE_BAND = (0.925, 0.975)


def _r(x, k=6):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def _per_replicate_matrix(results, scenario, estimator, cats, truth):
    """(R, K, 4) array of [error, covered_primary, covered_event, se] by replicate."""
    reps = [r for r in results if r["scenario"] == scenario]
    out = np.full((len(reps), len(cats), 4), np.nan)
    for a, r in enumerate(reps):
        for b, c in enumerate(cats):
            v = r["est"].get(estimator, {}).get(c)
            if v is None:
                continue
            est, se, se_ev, crit = v
            err = est - truth[c]
            out[a, b, 0] = err
            if np.isfinite(se):
                out[a, b, 1] = float(abs(err) <= Z95 * se)
            if np.isfinite(se_ev) and np.isfinite(crit):
                out[a, b, 2] = float(abs(err) <= crit * se_ev)
            out[a, b, 3] = se
    return out


def _mc(x: np.ndarray) -> tuple[float, float, int]:
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else NAN), NAN, n
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n)), n


def summarise(results: list[dict], truth: dict[str, float], scenarios, estimators=None,
              true_circuit_sd: float | None = None) -> dict:
    """Bias, RMSE and coverage with Monte Carlo SEs, per compound and averaged."""
    estimators = list(ESTIMATORS) if estimators is None else list(estimators)
    cats = [c for c in C_ORDER if c in truth]
    out: dict = {}
    for sc in scenarios:
        n_rep = sum(1 for r in results if r["scenario"] == sc)
        out[sc] = {}
        for est in estimators:
            M = _per_replicate_matrix(results, sc, est, cats, truth)
            by_c = {}
            for b, c in enumerate(cats):
                err = M[:, b, 0]
                if not np.isfinite(err).any():
                    continue
                bias, bias_se, n = _mc(err)
                sq = err[np.isfinite(err)] ** 2
                rmse = math.sqrt(float(sq.mean()))
                rmse_se = float(sq.std(ddof=1) / (2 * rmse * math.sqrt(len(sq)))) if len(sq) > 1 and rmse > 0 else NAN
                cell = {"n": n, "bias": _r(bias), "bias_mc_se": _r(bias_se), "rmse": _r(rmse),
                        "rmse_mc_se": _r(rmse_se), "empirical_sd": _r(float(np.nanstd(err, ddof=1))),
                        "mean_reported_se": _r(float(np.nanmean(M[:, b, 3])))}
                for key, col in (("coverage", 1), ("coverage_event_cluster", 2)):
                    cv = M[:, b, col]
                    if np.isfinite(cv).any():
                        p, _, m = _mc(cv)
                        cell[key] = _r(p)
                        cell[key + "_mc_se"] = _r(math.sqrt(p * (1 - p) / m))
                by_c[c] = cell
            if not by_c:
                out[sc][est] = {"failures": n_rep, "by_compound": {}}
                continue
            used = [b for b, c in enumerate(cats) if c in by_c]
            rep_bias = np.nanmean(M[:, used, 0], axis=1)
            rep_cov = np.nanmean(M[:, used, 1], axis=1) if np.isfinite(M[:, used, 1]).any() else None
            rep_cov_ev = np.nanmean(M[:, used, 2], axis=1) if np.isfinite(M[:, used, 2]).any() else None
            mean_row = {"bias": _r(_mc(rep_bias)[0]), "bias_mc_se": _r(_mc(rep_bias)[1]),
                        "mean_abs_bias": _r(float(np.mean([abs(by_c[c]["bias"]) for c in by_c]))),
                        "rmse": _r(float(np.mean([by_c[c]["rmse"] for c in by_c])))}
            for key, arr in (("coverage", rep_cov), ("coverage_event_cluster", rep_cov_ev)):
                if arr is not None:
                    p, se, _ = _mc(arr)
                    mean_row[key] = _r(p)
                    mean_row[key + "_mc_se"] = _r(se)
            worst_b = max(by_c, key=lambda c: abs(by_c[c]["bias"]))
            worst = {"compound_largest_abs_bias": worst_b, "bias": by_c[worst_b]["bias"],
                     "bias_mc_se": by_c[worst_b]["bias_mc_se"]}
            if all("coverage" in by_c[c] for c in by_c):
                lowc = min(by_c, key=lambda c: by_c[c]["coverage"])
                worst["compound_lowest_coverage"] = lowc
                worst["coverage"] = by_c[lowc]["coverage"]
                worst["coverage_mc_se"] = by_c[lowc]["coverage_mc_se"]
            recovered = all(abs(by_c[c]["bias"]) <= BIAS_TOLERANCE for c in by_c)
            if all("coverage" in by_c[c] for c in by_c):
                recovered = recovered and all(COVERAGE_BAND[0] <= by_c[c]["coverage"] <= COVERAGE_BAND[1]
                                              for c in by_c)
            else:
                recovered = None
            entry = {"mean_over_compounds": mean_row, "worst_compound": worst,
                     "by_compound": by_c, "recovered_all_compounds": recovered,
                     "failures": int(n_rep - min(by_c[c]["n"] for c in by_c))}
            if est in REML_ESTIMATORS:
                sds = np.array([r["flags"][est]["circuit_sd"] for r in results
                                if r["scenario"] == sc and est in r["flags"]])
                conv = [r["flags"][est]["converged"] for r in results
                        if r["scenario"] == sc and est in r["flags"]]
                bnd = [r["flags"][est]["circuit_on_boundary"] for r in results
                       if r["scenario"] == sc and est in r["flags"]]
                if len(sds):
                    cs = {"mean_estimate": _r(float(sds.mean())),
                          "median_estimate": _r(float(np.median(sds))),
                          "share_on_boundary": _r(float(np.mean(bnd)), 4),
                          "share_not_converged": _r(1.0 - float(np.mean(conv)), 4),
                          "n": int(len(sds))}
                    if true_circuit_sd is not None:
                        cs["true"] = true_circuit_sd
                        cs["bias"] = _r(float(sds.mean() - true_circuit_sd))
                        cs["bias_mc_se"] = _r(float(sds.std(ddof=1) / math.sqrt(len(sds)))) if len(sds) > 1 else None
                        cs["rmse"] = _r(float(math.sqrt(np.mean((sds - true_circuit_sd) ** 2))))
                    entry["circuit_sd"] = cs
            out[sc][est] = entry
    return out


def check_expectations(summary: dict) -> dict:
    """The pre-stated expectations X1-X6 of the module docstring, scored.

    The rules are fixed here, before any full run. ``holds`` is None when the
    summary lacks what the rule needs (a scenario or estimator was not run).
    """
    FUEL_SHIFT = 0.033 * 0.2 * 1.7  # s/lap moved by a 20% error in the fuel coefficient

    def m(sc, est, key="bias", block="mean_over_compounds"):
        try:
            return summary[sc][est][block][key]
        except (KeyError, TypeError):
            return None

    def has(*vals):
        return all(v is not None for v in vals)

    sc3 = ("exogenous", "pace", "wear")
    out: dict = {}

    b = {sc: m(sc, "naive_pooled_ols") for sc in sc3}
    out["X1"] = {
        "statement": "naive_pooled_ols is biased low (< -0.02 s/lap) in every scenario",
        "holds": all(v < -0.02 for v in b.values()) if has(*b.values()) else None,
        "evidence": {"mean_bias": b},
    }
    ex = {sc: m(sc, "fuel_within_run") for sc in sc3}
    lo = {sc: m(sc, "fuel_within_run_minus20") for sc in sc3}
    hi = {sc: m(sc, "fuel_within_run_plus20") for sc in sc3}
    ok2 = None
    if has(*ex.values(), *lo.values(), *hi.values()):
        ok2 = all(
            ex[sc] < 0 and abs((lo[sc] - ex[sc]) + FUEL_SHIFT) < 0.002 and abs((hi[sc] - ex[sc]) - FUEL_SHIFT) < 0.002
            for sc in sc3)
    out["X2"] = {
        "statement": "fuel_within_run is biased low at the right fuel coefficient and moves by -/+0.0112 s/lap "
                     "(within 0.002) under a -/+20% fuel error, in every scenario",
        "holds": ok2, "evidence": {"mean_bias_exact": ex, "minus20": lo, "plus20": hi},
    }
    ev3, ok3 = {}, []
    for est in ("cross_section_cell", "field_demeaned_ols"):
        a, p_ = m("exogenous", est, "mean_abs_bias"), m("pace", est, "mean_abs_bias")
        ev3[est] = {"mean_abs_bias_exogenous": a, "mean_abs_bias_pace": p_}
        ok3.append(None if not has(a, p_) else (a <= 0.005 < p_))
    out["X3"] = {
        "statement": "cross_section_cell and field_demeaned_ols have mean |bias| <= 0.005 under (a) and above it under (b)",
        "holds": None if None in ok3 else all(ok3), "evidence": ev3,
    }
    lo_c, hi_c = COVERAGE_BAND
    ev4 = {sc: {"mean_abs_bias": m(sc, "eq1_shipped", "mean_abs_bias"),
                "mean_coverage": m(sc, "eq1_shipped", "coverage")} for sc in ("exogenous", "pace")}
    vals4 = [v for d in ev4.values() for v in d.values()]
    out["X4"] = {
        "statement": "eq1_shipped has mean |bias| <= 0.005 and mean coverage in [0.925, 0.975] under (a) and (b)",
        "holds": None if not has(*vals4) else all(
            d["mean_abs_bias"] <= BIAS_TOLERANCE and lo_c <= d["mean_coverage"] <= hi_c for d in ev4.values()),
        "evidence": ev4,
    }
    bw, cw, ce = m("wear", "eq1_shipped"), m("wear", "eq1_shipped", "coverage"), m("exogenous", "eq1_shipped", "coverage")
    out["X5"] = {
        "statement": "under (c) eq1_shipped is biased toward zero (mean bias < 0) and its coverage is more than 0.02 "
                     "below its coverage under (a)",
        "holds": None if not has(bw, cw, ce) else bool(bw < 0 and cw < ce - 0.02),
        "evidence": {"mean_bias_wear": bw, "coverage_wear": cw, "coverage_exogenous": ce},
    }
    a_s, a_m = m("wear", "eq1_shipped", "mean_abs_bias"), m("wear", "eq1_mundlak_run_mean_slope", "mean_abs_bias")
    out["X5b"] = {
        "statement": "under (c) the correlated-random-slope variant has smaller mean |bias| than eq1_shipped",
        "holds": None if not has(a_s, a_m) else bool(a_m < a_s),
        "evidence": {"eq1_shipped": a_s, "eq1_mundlak_run_mean_slope": a_m},
    }
    sd = {}
    for sc in sc3:
        try:
            sd[sc] = summary[sc]["eq1_shipped"]["circuit_sd"]["bias"]
        except (KeyError, TypeError):
            sd[sc] = None
    out["X6"] = {
        "statement": "the circuit-slope SD of eq1_shipped is biased downward in every scenario",
        "holds": all(v < 0 for v in sd.values()) if has(*sd.values()) else None,
        "evidence": {"bias": sd},
    }
    ev7 = {sc: {"mean_abs_bias": m(sc, PRIMARY_EQ1, "mean_abs_bias"),
                "mean_coverage": m(sc, PRIMARY_EQ1, "coverage")} for sc in ("exogenous", "pace")}
    vals7 = [v for d in ev7.values() for v in d.values()]
    out["X7"] = {
        "statement": "ADDED AFTER THE CORRECTION (not pre-stated): eq1_exact_within has mean |bias| <= 0.005 and "
                     "mean coverage in [0.925, 0.975] under (a) and (b)",
        "holds": None if not has(*vals7) else all(
            d["mean_abs_bias"] <= BIAS_TOLERANCE and lo_c <= d["mean_coverage"] <= hi_c for d in ev7.values()),
        "evidence": ev7,
    }
    a_e, a_x = m("wear", PRIMARY_EQ1, "mean_abs_bias"), m("wear", "eq1_exact_within_mundlak", "mean_abs_bias")
    out["X8"] = {
        "statement": "ADDED AFTER THE CORRECTION (not pre-stated): under (c) eq1_exact_within_mundlak has smaller "
                     "mean |bias| than eq1_exact_within",
        "holds": None if not has(a_e, a_x) else bool(a_x < a_e),
        "evidence": {PRIMARY_EQ1: a_e, "eq1_exact_within_mundlak": a_x},
    }
    return out


def summarise_diagnostics(results: list[dict], scenarios) -> dict:
    out = {}
    for sc in scenarios:
        rows = [r["diag"] for r in results if r["scenario"] == sc]
        out[sc] = {k: _r(float(np.nanmean([x[k] for x in rows])), 4) for k in rows[0]}
    return out


def error_counts(results: list[dict], scenarios) -> dict:
    out: dict = {}
    for sc in scenarios:
        errs: dict = {}
        for r in results:
            if r["scenario"] == sc:
                for k, v in r["errors"].items():
                    errs.setdefault(k, []).append(v)
        out[sc] = {k: {"count": len(v), "first": v[0]} for k, v in errs.items()}
    return out


if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:]))

"""Offline tests for the telemetry management proxies and the tests built on them.

All telemetry here is synthetic: a lap is built from a script of driving phases
so the right answer for each proxy is known by construction. Nothing touches
FastF1 or the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.paper import telemetry as T

DT = 0.25


def make_lap(coast_s=1.5, flicker=False, n_corners=1):
    """A 60 s lap: full throttle, a coast of ``coast_s``, a hard brake, a corner, re-accelerate."""
    t = np.arange(0, 60, DT)
    speed = np.full(len(t), 250.0)
    thr = np.full(len(t), 100.0)
    brake = np.zeros(len(t), bool)
    c0 = int(20 / DT)
    c1 = c0 + int(round(coast_s / DT))
    thr[c0:c1] = 0.0
    speed[c0:c1] = np.linspace(250, 240, c1 - c0)
    b1 = c1 + 8  # 2 s of braking
    brake[c1:b1] = True
    thr[c1:b1] = 0.0
    speed[c1:b1] = np.linspace(240, 80, b1 - c1)
    speed[b1:b1 + 8] = 80.0
    thr[b1:b1 + 8] = 40.0
    speed[b1 + 8:b1 + 40] = np.linspace(80, 250, 32)
    if flicker:  # brake flag blips at constant speed, must not create a zone
        brake[200:202] = True
    return pd.DataFrame({"Time": pd.to_timedelta(t, unit="s"), "Speed": speed,
                         "Throttle": thr, "Brake": brake})


def test_pre_brake_coast_is_measured():
    p = T.lap_proxies(make_lap(coast_s=1.5))
    assert p["coast_s"] == pytest.approx(1.5, abs=DT)
    assert p["coast_total_s"] == pytest.approx(1.5, abs=DT)


def test_more_coasting_gives_larger_proxy():
    a = T.lap_proxies(make_lap(coast_s=0.5))["coast_s"]
    b = T.lap_proxies(make_lap(coast_s=2.5))["coast_s"]
    assert b > a + 1.5


def test_coast_far_from_a_braking_zone_is_not_pre_brake():
    car = make_lap(coast_s=0.0)
    # A lift in the middle of a straight with no braking after it.
    car.loc[40:50, "Throttle"] = 0.0
    p = T.lap_proxies(car)
    assert p["coast_s"] == pytest.approx(0.0, abs=1e-9)
    assert p["coast_total_s"] > 2.0


def test_slow_coasting_is_ignored():
    car = make_lap(coast_s=1.5)
    car["Speed"] = car["Speed"].clip(upper=140.0)  # nothing above 150 km/h
    assert T.lap_proxies(car)["coast_total_s"] == 0.0


def test_brake_flicker_does_not_create_a_zone():
    clean = T.lap_proxies(make_lap())
    noisy = T.lap_proxies(make_lap(flicker=True))
    assert noisy["brake_frac"] == pytest.approx(clean["brake_frac"])


def test_full_throttle_fraction_and_104_sentinel():
    car = make_lap()
    car.loc[car["Throttle"] == 100.0, "Throttle"] = 104.0
    p = T.lap_proxies(car)
    assert 0.5 < p["full_throttle_frac"] < 1.0


def test_braking_deceleration_is_physical():
    p = T.lap_proxies(make_lap())
    # 160 km/h shed in ~2 s is about 2.3 g.
    assert 1.5 < p["brake_decel_g"] < 3.0


def test_apex_speed_found():
    p = T.lap_proxies(make_lap())
    assert p["apex_speed"] == pytest.approx(80.0, abs=1.0)


def test_short_or_gappy_lap_is_rejected():
    assert T.lap_proxies(make_lap().iloc[:50]) is None
    car = make_lap()
    t = car["Time"].dt.total_seconds().to_numpy().copy()
    t[100:] += 20.0  # a 20 s dropout: coverage collapses
    car["Time"] = pd.to_timedelta(t, unit="s")
    assert T.lap_proxies(car) is None


# --------------------------------------------------------------------------- #
# Cell contrasts and the trend test, on synthetic lap tables
# --------------------------------------------------------------------------- #

def make_laps(n_events=8, effect=0.0, seed=0, fuel_slope=0.0, soft_early=0.0):
    """Lap table with a planted race-minus-practice coast shift of ``effect`` per softness step."""
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_events):
        base = rng.normal(1.0, 0.3)
        for comp, soft in T.SOFTNESS.items():
            for side in ("race", "practice"):
                for r in range(4):
                    for lap in range(8):
                        age = lap + 1
                        # Stints start at different laps, so age and progress are not collinear.
                        prog = (10 + 7 * r + 5 * lap) / 80 - soft_early * soft if side == "race" else 0.0
                        y = (base + 0.1 * soft + (effect * soft + 0.2 if side == "race" else 0.0)
                             + fuel_slope * prog + rng.normal(0, 0.15))
                        rows.append({"season": 2025, "event": f"E{e}", "Compound": comp,
                                     "side": side, "run_id": f"E{e}|{comp}|{side}|{r}",
                                     "TyreLife": float(age), "race_progress": prog, "coast_s": y})
    return pd.DataFrame(rows)


def test_cells_need_enough_runs():
    laps = make_laps(n_events=2)
    laps = laps[~((laps["Compound"] == "SOFT") & (laps["side"] == "practice")
                  & (laps["run_id"].str.endswith(("|2", "|3"))))]
    cells = T.adjusted_cells(laps, "coast_s")
    assert "SOFT" not in set(cells["Compound"])  # only 2 practice runs left


def test_planted_effect_is_recovered_and_significant():
    cells = T.adjusted_cells(make_laps(effect=0.3), "coast_s")
    tr = T.trend_test(cells, +1, n_boot=500, n_perm=2000)
    assert tr.slope == pytest.approx(0.3, abs=0.1)
    assert tr.ci95[0] > 0
    assert tr.p_perm_one_sided < 0.01


def test_null_gives_no_signal_and_common_race_shift_is_not_evidence():
    # The +0.2 race-wide shift (fuel/traffic) is in make_laps already.
    cells = T.adjusted_cells(make_laps(effect=0.0, seed=3), "coast_s")
    tr = T.trend_test(cells, +1, n_boot=500, n_perm=2000)
    assert abs(tr.slope) < 0.1
    assert tr.ci95[0] < 0 < tr.ci95[1]
    assert tr.p_perm_two_sided > 0.05
    assert all(0.1 < v["mean_delta"] < 0.3 for v in tr.per_compound.values())


def test_fuel_progress_is_adjusted_for():
    # Softs are run earlier in the race, i.e. on more fuel. With a fuel effect
    # of -1 per unit progress that would fake +0.2 per softness step if left in.
    laps = make_laps(effect=0.0, fuel_slope=-1.0, soft_early=0.2, seed=1)
    cells = T.adjusted_cells(laps, "coast_s")
    raw = T.trend_test(cells.assign(delta=cells["raw_delta"]), +1, n_boot=100, n_perm=500)
    adj = T.trend_test(cells, +1, n_boot=100, n_perm=500)
    assert raw.slope > 0.15
    assert abs(adj.slope) < 0.08


def test_exact_permutation_used_when_small():
    cells = T.adjusted_cells(make_laps(n_events=2, effect=0.5), "coast_s")
    tr = T.trend_test(cells, +1, n_boot=50, n_perm=100000)
    assert tr.perm_exact
    assert tr.n_perm == 6 * 6  # 3! label orders in each of two events
    assert tr.p_perm_one_sided >= 1 / 36 - 1e-9  # cannot beat what the design allows


def test_trend_test_none_without_within_event_variation():
    cells = T.adjusted_cells(make_laps(n_events=3), "coast_s")
    one = cells[cells["Compound"] == "SOFT"]
    assert T.trend_test(one) is None


# --------------------------------------------------------------------------- #
# Link to degradation ratios
# --------------------------------------------------------------------------- #

def test_link_requires_enough_events():
    cells = T.adjusted_cells(make_laps(n_events=2), "coast_s")
    ratios = cells[["season", "event", "Compound"]].assign(ratio=0.3)
    assert T.link_to_ratio(cells, ratios, n_boot=10, n_perm=10)["estimable"] is False


def test_link_detects_within_compound_dependence_not_compound_effect():
    cells = T.adjusted_cells(make_laps(n_events=12, effect=0.0, seed=5), "coast_s")
    rng = np.random.default_rng(0)
    # Ratio depends on compound only: SOFT lowest. No within-compound relation.
    r = cells[["season", "event", "Compound"]].copy()
    r["ratio"] = r["Compound"].map({"HARD": 0.5, "MEDIUM": 0.4, "SOFT": 0.2}) + rng.normal(0, 0.05, len(r))
    null = T.link_to_ratio(cells, r, n_boot=300, n_perm=1000)
    assert null["estimable"] and abs(null["within_compound_rank_corr"]) < 0.5
    assert null["p_perm_two_sided"] > 0.05
    # Now add a real dependence: more coasting, lower ratio.
    r["ratio"] = r["ratio"] - 3.0 * (cells["delta"].to_numpy() - cells["delta"].mean())
    real = T.link_to_ratio(cells, r, n_boot=300, n_perm=1000)
    assert real["within_compound_rank_corr"] < -0.4
    assert real["p_perm_one_sided_negative"] < 0.01


def test_nan_proxy_laps_are_dropped_not_propagated():
    laps = make_laps(n_events=2)
    laps.loc[laps.index[::7], "coast_s"] = np.nan
    cells = T.adjusted_cells(laps, "coast_s")
    assert cells["delta"].notna().all()


# --------------------------------------------------------------------------- #
# OpenF1 adapter
# --------------------------------------------------------------------------- #

def test_openf1_lap_is_cut_and_gives_the_same_proxies_as_the_fastf1_layout():
    lap = make_lap(coast_s=1.5)
    # Embed the lap in a longer stream: 30 s before, 30 s after, absolute ns timestamps.
    t0 = 1_700_000_000_000_000_000
    pre = np.arange(-30, 0, DT)
    post = np.arange(60, 90, DT)
    t = np.concatenate([pre, lap["Time"].dt.total_seconds().to_numpy(), post])
    date_ns = (t0 + t * 1e9).astype("int64")
    n_pre, n_post = len(pre), len(post)
    speed = np.concatenate([np.full(n_pre, 50.0), lap["Speed"], np.full(n_post, 50.0)])
    thr = np.concatenate([np.full(n_pre, 100.0), lap["Throttle"].clip(upper=100), np.full(n_post, 100.0)])
    brk = np.concatenate([np.zeros(n_pre), lap["Brake"].astype(int) * 100, np.zeros(n_post)])
    frame = T.openf1_lap_frame(date_ns, speed, thr, brk, t0, t0 + int(60e9))
    assert frame is not None and frame["Brake"].dtype == bool
    p = T.lap_proxies(frame)
    ref = T.lap_proxies(lap)
    assert p["coast_s"] == pytest.approx(ref["coast_s"])
    assert p["brake_frac"] == pytest.approx(ref["brake_frac"])


def test_openf1_lap_outside_stream_is_none():
    date_ns = (np.arange(0, 100) * 250_000_000).astype("int64")
    z = np.zeros(100)
    assert T.openf1_lap_frame(date_ns, z, z, z, 10**12, 2 * 10**12) is None

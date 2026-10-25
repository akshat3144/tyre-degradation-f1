"""Pit loss across every season we hold, by track status.

Backs ``paper/results/pit_loss.json``. It answers four questions the product
makes claims on:

  1. What does a green-flag stop cost, and does the corrected reference (the
     non-pitting field on the same laps) change it against the old one (the
     driver's own race-wide median green lap)?
  2. Does a VSC stop cost 0.84x a green one, as an earlier default assumed?
     That figure came from one season; this re-measures it on all of them.
  3. Is a safety-car number usable once there are more seasons behind it?
  4. Is pit loss stable enough from one season to the next at a circuit to be
     forecast from history?

All uncertainty is a CLUSTER bootstrap over races (``N_BOOT`` resamples of the
races, not of the stops), because stops within one race share a pit lane, a
weather and a tyre compound and are not independent -- bootstrapping stops would
make every interval too narrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..strategy import pitloss as PL
from ..strategy import racedata

SEED = 20260930
N_BOOT = 2000

#: A category is offered as a default only with this many stops...
MIN_STOPS = 30
#: ...from at least this many races...
MIN_RACES = 8
#: ...and at most this share from any single race.
MAX_SINGLE_RACE_SHARE = 0.25


def sane(stops: pd.DataFrame) -> pd.DataFrame:
    """Stops inside the wide sanity filter, the same for every category.

    The first version kept ``0 < loss < 45`` for the neutralised categories and
    clipped green to 12-40 s. Cheap neutralised stops (a loss at or below zero
    against a reference that is itself slowed) were discarded while green lost
    only its tails, so the ratio to green was biased upward. ``truncation_effect``
    reports how much.
    """
    return stops[stops["loss"].between(*PL.SANITY_RANGE_S)]


def all_stops(laps: pd.DataFrame) -> pd.DataFrame:
    """Every stop in every race of the cache, with category and loss.

    Adds ``season`` and a ``race`` key. Stops under red flag are kept in their
    own category; nothing is filtered here beyond what ``stop_losses`` does.
    """
    raw = racedata.as_raw_laps(laps)
    frames = []
    for (season, event), g in raw.groupby(["season", "event"], sort=True):
        s = PL.stop_losses(g, event)
        if s.empty:
            continue
        s["season"] = int(season)
        s["race"] = f"{int(season)} {event}"
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def old_reference_losses(laps: pd.DataFrame) -> pd.DataFrame:
    """The superseded construction, kept only to show what it did wrong.

    Reference = the driver's own median green lap over the whole race; a stop
    counts if its IN-lap status was exactly '1'. This is what ``estimate`` did.
    """
    raw = racedata.as_raw_laps(laps)
    rows = []
    for (season, event), df in raw.groupby(["season", "event"], sort=True):
        df = df[df["LapTimeSeconds"].notna()]
        green = df[df["PitInTime"].isna() & df["PitOutTime"].isna() & df["TrackStatus"].astype(str).eq("1")]
        if green.empty:
            continue
        ref = green.groupby("Driver")["LapTimeSeconds"].median()
        nlaps = int(df["LapNumber"].max())
        for driver, g in df.groupby("Driver"):
            if driver not in ref:
                continue
            g = g.sort_values("LapNumber").set_index("LapNumber")
            for lap in g.index[g["PitInTime"].notna()]:
                if lap + 1 not in g.index or pd.isna(g.at[lap + 1, "PitOutTime"]):
                    continue
                if str(g.at[lap, "TrackStatus"]) != "1":
                    continue
                loss = g.at[lap, "LapTimeSeconds"] + g.at[lap + 1, "LapTimeSeconds"] - 2 * ref[driver]
                rows.append(
                    {"race": f"{int(season)} {event}", "lap": int(lap), "loss": float(loss),
                     "frac": float(lap) / nlaps}
                )
    return pd.DataFrame(rows)


def _by_race(df: pd.DataFrame) -> list[np.ndarray]:
    return [g["loss"].to_numpy() for _, g in df.groupby("race")]


def cluster_boot(df: pd.DataFrame, stat, n_boot: int = N_BOOT, seed: int = SEED):
    """``(estimate, lo, hi)`` of ``stat(losses)``, resampling whole races."""
    groups = _by_race(df)
    if not groups:
        return None
    rng = np.random.default_rng(seed)
    est = float(stat(np.concatenate(groups)))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(np.concatenate([groups[i] for i in pick])))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def cluster_boot_ratio(num: pd.DataFrame, den: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED):
    """Ratio of medians, resampling races jointly so a race's two arms move together."""
    races = sorted(set(num["race"]) | set(den["race"]))
    nums = {r: g["loss"].to_numpy() for r, g in num.groupby("race")}
    dens = {r: g["loss"].to_numpy() for r, g in den.groupby("race")}
    empty = np.array([])
    rng = np.random.default_rng(seed)

    def ratio(pick):
        a = np.concatenate([nums.get(races[i], empty) for i in pick])
        b = np.concatenate([dens.get(races[i], empty) for i in pick])
        if len(a) == 0 or len(b) == 0:
            return np.nan
        return float(np.median(a) / np.median(b))

    est = ratio(range(len(races)))
    vals = [ratio(rng.integers(0, len(races), len(races))) for _ in range(n_boot)]
    vals = [v for v in vals if np.isfinite(v)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return est, float(lo), float(hi)


def category_table(stops: pd.DataFrame) -> dict:
    """Per-category summary on the wide sanity bounds, ratio to green."""
    s = sane(stops)
    green = s[s["category"] == "green"]
    out: dict = {}
    for cat in PL.CATEGORIES:
        sub = s[s["category"] == cat]
        if cat == "green":
            sub = green
        if sub.empty:
            continue
        n = len(sub)
        by_race = sub["race"].value_counts()
        med = cluster_boot(sub, np.median)
        row = {
            "n_stops": int(n),
            "n_races": int(sub["race"].nunique()),
            "n_seasons": int(sub["season"].nunique()),
            "median_s": round(med[0], 2),
            "median_ci95": [round(med[1], 2), round(med[2], 2)],
            "mean_s": round(float(sub["loss"].mean()), 2),
            "max_single_race_share": round(float(by_race.iloc[0] / n), 3),
        }
        if cat != "green":
            r = cluster_boot_ratio(sub, green)
            row["ratio_to_green"] = round(r[0], 3)
            row["ratio_ci95"] = [round(r[1], 3), round(r[2], 3)]
            row["usable_as_default"] = bool(
                n >= MIN_STOPS
                and row["n_races"] >= MIN_RACES
                and row["max_single_race_share"] <= MAX_SINGLE_RACE_SHARE
            )
        out[cat] = row
    return out


def per_season(stops: pd.DataFrame, cats=("green", "vsc", "safety_car")) -> dict:
    """The same medians split by season, to see whether one year drives a number."""
    s = sane(stops)
    out: dict = {}
    for season, g in s.groupby("season"):
        row = {}
        gr = g[g["category"] == "green"]["loss"]
        for cat in cats:
            x = gr if cat == "green" else g[g["category"] == cat]["loss"]
            if len(x):
                row[cat] = {"n": int(len(x)), "median_s": round(float(x.median()), 2)}
        if "green" in row:
            for cat in cats:
                if cat != "green" and cat in row:
                    row[cat]["ratio"] = round(row[cat]["median_s"] / row["green"]["median_s"], 3)
        out[str(int(season))] = row
    return out


def event_paired_ratio(stops: pd.DataFrame, cat: str) -> dict:
    """Within-race ratio: for every race with both kinds of stop, cat median / green median."""
    s = sane(stops)
    rows = []
    for race, g in s.groupby("race"):
        gr = g[g["category"] == "green"]["loss"]
        c = g[g["category"] == cat]["loss"]
        if len(gr) >= 3 and len(c) >= 2:
            rows.append((race, len(c), float(c.median() / gr.median())))
    if not rows:
        return {"n_races": 0}
    r = np.array([x[2] for x in rows])
    w = np.array([x[1] for x in rows])
    rng = np.random.default_rng(SEED)
    boots = [float(np.median(r[rng.integers(0, len(r), len(r))])) for _ in range(N_BOOT)]
    return {
        "n_races": len(rows),
        "median_of_race_ratios": round(float(np.median(r)), 3),
        "ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
        "stop_weighted_mean_ratio": round(float(np.average(r, weights=w)), 3),
        "min": round(float(r.min()), 3),
        "max": round(float(r.max()), 3),
    }


def queue_table(stops: pd.DataFrame) -> dict:
    """Median loss by how many cars stopped on the same lap, per category."""
    s = sane(stops)
    out: dict = {}
    for cat in ("green", "safety_car", "vsc"):
        sub = s[s["category"] == cat]
        q = sub.assign(k=sub["n_pitting"].clip(upper=6)).groupby("k")["loss"].agg(["size", "median"])
        out[cat] = {str(int(k)) if k < 6 else "6+": {"n": int(v["size"]), "median_s": round(float(v["median"]), 2)}
                    for k, v in q.iterrows()}
    return out


def reference_bias(laps: pd.DataFrame, stops: pd.DataFrame) -> dict:
    """What the old (driver race-median) reference did, against the new one.

    Both are computed on the SAME green stops. The old reference compares a stop
    against a race-wide median; a stop made early is on a heavy car on a green
    track and a late one the reverse, so the bias should depend on stop timing.
    """
    old = old_reference_losses(laps)
    old = old[old["loss"].between(*PL.PLAUSIBLE_RANGE_S)]
    new = stops[(stops["category"] == "green") & stops["loss"].between(*PL.PLAUSIBLE_RANGE_S)].copy()
    nlaps = laps.groupby(["season", "event"])["LapNumber"].max()
    new["frac"] = [
        float(l) / float(nlaps.loc[(s, r.split(" ", 1)[1])])
        for l, s, r in zip(new["lap"], new["season"], new["race"], strict=True)
    ]

    def by_third(df):
        b = pd.cut(df["frac"], [0, 1 / 3, 2 / 3, 1.01], labels=["first third", "middle third", "last third"])
        g = df.groupby(b, observed=True)["loss"].agg(["size", "median"])
        return {str(k): {"n": int(v["size"]), "median_s": round(float(v["median"]), 2)} for k, v in g.iterrows()}

    def slope(df):
        if len(df) < 5:
            return None
        return round(float(np.polyfit(df["frac"], df["loss"], 1)[0]), 2)

    return {
        "old_reference": {
            "n": int(len(old)),
            "median_s": round(float(old["loss"].median()), 2),
            "by_race_third": by_third(old),
            "slope_s_per_race_fraction": slope(old),
        },
        "new_reference": {
            "n": int(len(new)),
            "median_s": round(float(new["loss"].median()), 2),
            "by_race_third": by_third(new),
            "slope_s_per_race_fraction": slope(new),
        },
        "reading": (
            "slope is the change in measured loss from a stop on lap 0 to one on the last lap. "
            "A pit lane does not know what lap it is, so a non-zero slope is bias from fuel "
            "burn and track evolution leaking into the reference."
        ),
    }


def circuit_stability(laps: pd.DataFrame) -> dict:
    """How far does a circuit's pit loss move between seasons?

    Green median per (circuit, season) from ``PL.estimate``; the between-season SD
    within a circuit is the uncertainty to attach to a forecast made from history.
    """
    from ..strategy.circuits import canonical_location
    from ..strategy.racedata import locations

    raw = racedata.as_raw_laps(laps)
    rows = []
    for (season, event), g in raw.groupby(["season", "event"], sort=True):
        pl = PL.estimate(g, event)
        if pl is None:
            continue
        rows.append(
            {"season": int(season), "event": event, "circuit": canonical_location(locations()[(int(season), event)]),
             "pit_loss_s": pl.seconds, "n": pl.n_stops, "sd": pl.sd}
        )
    t = pd.DataFrame(rows)
    sds, spreads = [], []
    for _, g in t.groupby("circuit"):
        if len(g) >= 2:
            sds.append(float(g["pit_loss_s"].std(ddof=1)))
            spreads.append(float(g["pit_loss_s"].max() - g["pit_loss_s"].min()))
    return {
        "n_race_estimates": int(len(t)),
        "n_circuits_with_2plus_seasons": len(sds),
        "between_season_sd_s": {
            "median": round(float(np.median(sds)), 2),
            "mean": round(float(np.mean(sds)), 2),
            "p90": round(float(np.percentile(sds, 90)), 2),
        },
        "between_season_range_s_median": round(float(np.median(spreads)), 2),
        "median_within_race_bootstrap_sd_s": round(float(t["sd"].median()), 2),
        "between_circuit_sd_s": round(float(t.groupby("circuit")["pit_loss_s"].median().std(ddof=1)), 2),
        "table": [
            {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}
            for r in t.sort_values(["circuit", "season"]).to_dict("records")
        ],
    }


def truncation_effect(stops: pd.DataFrame) -> dict:
    """The ratios to green under the superseded filter and under the wide one.

    Old: neutralised stops kept if ``0 < loss < 45``, green kept if 12-40 s.
    New: every category kept if inside ``SANITY_RANGE_S``. Also how many
    neutralised stops the old filter discarded, and the share of those that
    were at or below zero.
    """
    old_s = stops[(stops["loss"] > 0) & (stops["loss"] < 45.0)]
    old_green = old_s[(old_s["category"] == "green") & old_s["loss"].between(*PL.PLAUSIBLE_RANGE_S)]
    new_s = sane(stops)
    new_green = new_s[new_s["category"] == "green"]
    out: dict = {"old_filter": "0 < loss < 45 (green: 12-40 s)", "new_filter": f"{PL.SANITY_RANGE_S[0]:g} <= loss <= {PL.SANITY_RANGE_S[1]:g}, all categories"}
    for cat in ("vsc", "safety_car"):
        row: dict = {}
        for name, sub, grn in (("old", old_s[old_s["category"] == cat], old_green),
                               ("new", new_s[new_s["category"] == cat], new_green)):
            r = cluster_boot_ratio(sub, grn)
            row[name] = {"n_stops": int(len(sub)), "median_s": round(float(sub["loss"].median()), 2),
                         "green_median_s": round(float(grn["loss"].median()), 2),
                         "ratio": round(r[0], 3), "ratio_ci95": [round(r[1], 3), round(r[2], 3)]}
        raw = stops[stops["category"] == cat]
        row["stops_at_or_below_zero"] = int((raw["loss"] <= 0).sum())
        row["stops_above_45"] = int((raw["loss"] >= 45.0).sum())
        row["stops_outside_sanity_filter"] = int(len(raw) - len(sane(raw)))
        # A median needs no filter at all to resist outliers: the ratio of the
        # plain medians over EVERY stop, as a check that the filter is not what
        # the answer depends on.
        allg = stops[stops["category"] == "green"]
        r = cluster_boot_ratio(raw, allg)
        row["unfiltered"] = {"n_stops": int(len(raw)), "median_s": round(float(raw["loss"].median()), 2),
                             "green_median_s": round(float(allg["loss"].median()), 2),
                             "ratio": round(r[0], 3), "ratio_ci95": [round(r[1], 3), round(r[2], 3)]}
        out[cat] = row
    return out

"""Does the strategy layer's call look like what the field actually ran?

Backs ``paper/results/strategy.json``. WHY THE SHIPPED CLAIM WAS NOT ENOUGH

An earlier project summary said "we agree with the field's median stop count at 10 of 11 events".
Three things make that number close to meaningless:

  1. The trivial rule "race of 60 laps or more -> 2 stops, else 1" scores 11 of
     11 on the same events, and "always 1 stop" scores 6 of 11. A stop count has
     very little to get wrong, so agreeing with it proves little.
  2. The rates behind the call were fitted ON the race being scored, so the
     call was a description of that race, not a forecast of it.
  3. A stop count is the coarsest thing a plan says. A strategist acts on
     stint LENGTHS and compounds, and those are where the optimiser's output
     was visibly strange (a 5-lap stint on one tyre and a 73-lap stint on the
     other).

So this module scores plans on richer targets, against honest baselines, with
rates that were NOT fitted on the race being scored.

THE TARGET: WHAT THE FIELD RAN

For every dry race, the stints of the classified top-ten finishers (the cars that
were racing for a result, rather than nursing a car home or repairing damage).
Red-flagged races are excluded from the headline, because a stop under a red flag
is a free tyre change, not a strategy. Each car's plan is a list of stint
lengths in laps and the compound label of each.

THE DISTANCE BETWEEN TWO PLANS

Plans here are unordered (the optimiser has no term for sequence), so a distance
must not depend on stint order. Each plan induces the multiset of TYRE AGES over
the race -- a 3-stint plan of 20/25/25 laps visits ages 1..20 once and 1..25
twice -- and the distance is the Wasserstein-1 distance between the two age
distributions: the mean number of laps by which a typical lap's tyre age differs.
Order-free, sensitive to both the number of stops and the stint lengths, and in
laps, which a strategist can read. Also reported: the longest-stint error (the
number the boundary pathology shows up in) and the share of plans with a stint at
or near the five-lap minimum.

THE RATES: OUT OF SAMPLE

Every rate source below is built without the race being scored:

  race_pooled     median race rate per compound from the OTHER races
  practice_scalar this weekend's practice long runs x a factor learned from OTHER
                  weekends (the product's forecast path)
  practice_label  the same with a factor per compound label (HARD/MEDIUM/SOFT)
  practice_raw    this weekend's practice rate with no race correction: the
                  tyre's wear when pushed, free of driver management
  blend           inverse-variance combination of practice_label and race_pooled
  mixed_loeo      (2026 only) the product's mixed-effects race model refitted
                  without the scored race -- a held-out circuit gets the pooled
                  compound rate
  in_sample       the race's own cells. NOT out of sample; it is what the product
                  shows for a race that has already run, and is here to show how
                  much the optimism is worth.

Each is gated exactly as the product gates it: positive and distinguishable from
zero, else the compound is refused, and a race with fewer than two gated
compounds is not planned (and counted as such, not silently dropped).

BASELINES

  always_1       one stop everywhere
  lap_rule       2 stops if the race is at least T laps, else 1; T learned on the
                 OTHER races
  prev_season    the stop count the field ran at the same circuit the previous
                 season

A baseline's plan shape is equal-length stints, because it has no opinion about
compounds or lengths; its stop count is the whole of its information.

LIMITS

2022-2025 compounds are HARD/MEDIUM/SOFT labels, relative to each weekend's
nomination; the repo holds Pirelli's C-number allocation for 2026 only. The
pace step between labels is therefore one step for each label, which is right
within a race and says nothing across races -- and cannot flatter the model.
The field optimises for position, traffic and safety-car luck and the model for
total time, so a distance is not an error; it is the distance between two
objectives, and the baselines are there so it can be read against something.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ..strategy import racedata
from ..strategy.circuits import canonical_location
from ..strategy.optimise import (
    C_ORDER,
    LABEL_ORDER,
    MAX_STOPS,
    MIN_STINT_LAPS,
    PACE_STEP_S,
    best_plan,
    pace_offsets,
)
from ..strategy.pitloss import PLAUSIBLE_RANGE_S, estimate
from ..strategy.rates import passes_gate
from ..strategy.support import bound_hits, support_caps, support_floor
from ..validation import transfer as T

warnings.filterwarnings("ignore")

SEED = 20261030
N_BOOT = 4000

#: A race counts as wet if more than this share of the top-ten's laps were on
#: intermediates or wets. Their strategy is a different problem.
WET_SHARE = 0.02

#: The pace step grid for the sensitivity sweep, seconds per compound step.
STEP_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

#: Stint at or below this is "on the boundary": within one lap of the minimum.
BOUNDARY_LAPS = MIN_STINT_LAPS + 1


# ---------------------------------------------------------------------------
# what the field ran
# ---------------------------------------------------------------------------


@dataclass
class FieldRace:
    season: int
    event: str
    race_laps: int
    #: One list of (compound label, laps) per top-ten car.
    cars: list[list[tuple[str, int]]]
    red_flag: bool
    wet: bool
    #: Position of the race in calendar order across every season held (0 = the
    #: first race of the first season). What "earlier" means for the
    #: out-of-time evaluation; races are otherwise keyed by name.
    order: int = 0

    @property
    def key(self) -> str:
        return f"{self.season} {self.event}"

    @property
    def stops(self) -> list[int]:
        return [len(c) - 1 for c in self.cars]

    @property
    def mode_stops(self) -> int:
        vals, counts = np.unique(self.stops, return_counts=True)
        return int(vals[np.argmax(counts)])  # ties -> the smaller count

    @property
    def median_stops(self) -> float:
        return float(np.median(self.stops))


def field_races(laps: pd.DataFrame, results: pd.DataFrame) -> list[FieldRace]:
    """The top-ten classified cars' stints for every race in the cache."""
    out = []
    # The cache is built season by season in calendar order, so the order the
    # races first appear in is the order they were run.
    chron = {
        (int(s), str(e)): i
        for i, (s, e) in enumerate(laps[["season", "event"]].drop_duplicates().itertuples(index=False))
    }
    for (season, event), g in laps.groupby(["season", "event"], sort=True):
        race_laps = int(g["LapNumber"].max())
        res = results[(results["season"] == season) & (results["event"] == event)]
        top = res[(res["Position"] <= 10) & res["Position"].notna()]
        top = top[top["Status"].astype(str).str.startswith(("Finished", "+", "Lapped"))]
        cars, wet_laps, total = [], 0, 0
        for d in top["Driver"]:
            gd = g[(g["Driver"] == d) & g["Stint"].notna()]
            if gd.empty:
                continue
            stints = (
                gd.groupby("Stint")
                .agg(c=("Compound", "first"), n=("LapNumber", "nunique"))
                .reset_index()
                .sort_values("Stint")
            )
            wet_laps += int(stints.loc[stints["c"].isin(["INTERMEDIATE", "WET"]), "n"].sum())
            total += int(stints["n"].sum())
            plan = [(str(c), int(n)) for c, n in zip(stints["c"], stints["n"], strict=True)]
            if any(c not in ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET") for c, _ in plan):
                continue
            cars.append(plan)
        if len(cars) < 5:
            continue
        flagged = bool(g["TrackStatus"].astype(str).str.contains("5").any())
        out.append(
            FieldRace(int(season), str(event), race_laps, cars, flagged,
                      total > 0 and wet_laps / total > WET_SHARE, chron[(int(season), str(event))])
        )
    return out


# ---------------------------------------------------------------------------
# plan distances
# ---------------------------------------------------------------------------


def ages(stints: tuple[int, ...] | list[int]) -> np.ndarray:
    """The tyre age of every race lap under a plan, as a multiset. Order-free."""
    return np.concatenate([np.arange(1, int(n) + 1) for n in stints])


def w1_age(a: tuple[int, ...] | list[int], b: tuple[int, ...] | list[int]) -> float:
    """Wasserstein-1 distance between two plans' tyre-age distributions, in laps."""
    xa, xb = ages(a), ages(b)
    top = int(max(xa.max(), xb.max()))
    ca = np.cumsum(np.bincount(xa, minlength=top + 1)[1:]) / len(xa)
    cb = np.cumsum(np.bincount(xb, minlength=top + 1)[1:]) / len(xb)
    return float(np.abs(ca - cb).sum())


def equal_split(race_laps: int, stops: int) -> tuple[int, ...]:
    """The plan a stop-count-only rule has to offer: stints as equal as possible."""
    k = stops + 1
    base, extra = divmod(race_laps, k)
    return tuple(base + (1 if i < extra else 0) for i in range(k))


@dataclass
class Rec:
    """A recommended plan reduced to what is scored."""

    stints: tuple[int, ...]
    #: Compound label per stint, or None for a baseline with no opinion.
    labels: tuple[str, ...] | None = None

    @property
    def stops(self) -> int:
        return len(self.stints) - 1


def usage_share(plan: list[tuple[str, int]] | Rec) -> dict[str, float]:
    """Share of race laps on each compound label."""
    if isinstance(plan, Rec):
        pairs = list(zip(plan.labels or (), plan.stints, strict=False))
    else:
        pairs = plan
    tot = sum(n for _, n in pairs) or 1
    out: dict[str, float] = {}
    for c, n in pairs:
        out[c] = out.get(c, 0.0) + n / tot
    return out


def score_rec(rec: Rec, race: FieldRace) -> dict:
    """Every per-race metric for one recommended plan against the top-ten's plans."""
    cars = race.cars
    d = np.array([w1_age(rec.stints, [n for _, n in c]) for c in cars])
    longest = np.array([max(n for _, n in c) for c in cars])
    shortest = np.array([min(n for _, n in c) for c in cars])
    stops = np.array(race.stops)
    row = {
        "stops": rec.stops,
        "stop_match_mode": int(rec.stops == race.mode_stops),
        "stop_abs_err_median": abs(rec.stops - race.median_stops),
        "stop_share": float(np.mean(stops == rec.stops)),
        "w1_median_car": float(np.median(d)),
        "w1_nearest_car": float(d.min()),
        "longest_err": float(abs(max(rec.stints) - np.median(longest))),
        "shortest_err": float(abs(min(rec.stints) - np.median(shortest))),
        "on_boundary": int(min(rec.stints) <= BOUNDARY_LAPS),
        "longest_stint": int(max(rec.stints)),
        "shortest_stint": int(min(rec.stints)),
        "field_longest": float(np.median(longest)),
        "field_shortest": float(np.median(shortest)),
    }
    if rec.labels:
        mine = usage_share(rec)
        labs = ("SOFT", "MEDIUM", "HARD")
        tv = []
        for c in cars:
            theirs = usage_share(c)
            tv.append(0.5 * sum(abs(mine.get(k, 0.0) - theirs.get(k, 0.0)) for k in labs))
        row["usage_tv_median_car"] = float(np.median(tv))
    return row


# ---------------------------------------------------------------------------
# rate sources, all out of sample
# ---------------------------------------------------------------------------


@dataclass
class Cells:
    """Practice and race cells for one group of races, and what they imply."""

    practice: pd.DataFrame
    race: pd.DataFrame
    #: Per-event factor learned from the OTHER events, and the global fallback.
    factor_by_event: dict[str, float] = field(default_factory=dict)
    factor_global: float = 1.0
    #: Race/practice ratios, one row per (event, label), for label factors.
    joined: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_cells(practice: pd.DataFrame, race: pd.DataFrame, label_of) -> Cells:
    """Cell tables and leave-one-event-out factors for one group of races.

    ``label_of(event, compound_key)`` maps a cell's compound key to its
    HARD/MEDIUM/SOFT label, which is what the management effect is keyed on.
    """
    p = T.cell_rates(practice, T.FORECAST_MIN_RUNS)
    r = T.cell_rates(race, T.MIN_RUNS, min_age_spread=T.MIN_AGE_SPREAD_LAPS)
    cells = Cells(practice=p, race=r)
    try:
        loo = T.leave_one_event_out(practice, race)
    except ValueError:
        return cells
    cells.factor_global = float(loo.factor)
    cells.factor_by_event = {
        str(e): float(g["factor_used"].iloc[0]) for e, g in loo.table.groupby("event")
    }
    j = loo.table[["event", "C", "practice_rate", "race_rate"]].copy()
    j["label"] = [label_of(e, c) for e, c in zip(j["event"], j["C"], strict=True)]
    cells.joined = j.dropna(subset=["label"])
    return cells


@dataclass
class RateInput:
    """A rate source's answer for one race: mean and SD per compound, pre-gate."""

    mean: dict[str, float]
    sd: dict[str, float]


def _row(df: pd.DataFrame, event: str, c: str):
    m = df[(df["event"] == event) & (df["C"] == c)]
    return None if m.empty else m.iloc[0]


def rate_inputs(cells: Cells, event: str, nominated: list[str], label_of) -> dict[str, RateInput]:
    """Every out-of-sample rate source for ``event``, keyed by name."""
    out: dict[str, RateInput] = {}
    others_race = cells.race[cells.race["event"] != event]

    # -- the race's own cells: NOT out of sample ----------------------------
    m, s = {}, {}
    for c in nominated:
        r = _row(cells.race, event, c)
        if r is not None:
            m[c], s[c] = float(r["rate"]), float(r["se"])
    out["in_sample"] = RateInput(m, s)

    # -- pooled race rate per compound, from the other races ----------------
    m, s = {}, {}
    for c in nominated:
        x = others_race.loc[others_race["C"] == c, "rate"].to_numpy(float)
        if len(x) >= 3:
            m[c] = float(np.median(x))
            s[c] = float(1.2533 * x.std(ddof=1) / np.sqrt(len(x)))
    out["race_pooled"] = RateInput(m, s)

    # -- practice-based sources --------------------------------------------
    f_e = cells.factor_by_event.get(event, cells.factor_global)
    raw_m, raw_s, sc_m, sc_s, lb_m, lb_s = {}, {}, {}, {}, {}, {}
    j = cells.joined[cells.joined["event"] != event]
    for c in nominated:
        r = _row(cells.practice, event, c)
        if r is None:
            continue
        pr, se = float(r["rate"]), float(r["se"]) if pd.notna(r["se"]) else abs(float(r["rate"]))
        thin = int(r["n_runs"]) < T.MIN_RUNS
        widen = T.THIN_INTERVAL_MULTIPLE if thin else 1.0
        raw_m[c], raw_s[c] = pr, se * widen
        sc_m[c], sc_s[c] = pr * f_e, se * widen * abs(f_e)
        lab = label_of(event, c)
        jl = j[j["label"] == lab]
        jl = jl[jl["practice_rate"].abs() > 1e-6]
        f_l = float(np.median(jl["race_rate"] / jl["practice_rate"])) if len(jl) >= 3 else f_e
        lb_m[c], lb_s[c] = pr * f_l, se * widen * abs(f_l)
    out["practice_raw"] = RateInput(raw_m, raw_s)
    out["practice_scalar"] = RateInput(sc_m, sc_s)
    out["practice_label"] = RateInput(lb_m, lb_s)

    # -- inverse-variance blend of the label forecast and the pooled race rate
    m, s = {}, {}
    for c in nominated:
        parts = []
        for src in (out["practice_label"], out["race_pooled"]):
            if c in src.mean and src.sd.get(c, 0) > 0:
                parts.append((src.mean[c], src.sd[c]))
        if parts:
            w = np.array([1.0 / sd**2 for _, sd in parts])
            m[c] = float(sum(wi * mu for wi, (mu, _) in zip(w, parts, strict=True)) / w.sum())
            s[c] = float(1.0 / np.sqrt(w.sum()))
    out["blend"] = RateInput(m, s)
    return out


def gate(src: RateInput) -> dict[str, float]:
    """The rates the optimiser may use: positive and distinguishable from zero."""
    return {
        c: mu for c, mu in src.mean.items() if passes_gate(mu, src.sd.get(c, 0.0))[0]
    }


def truncated_mean(mu: float, sd: float, floor: float = 0.0) -> float:
    """E[rate | rate >= floor] for rate ~ Normal(mu, sd): the rate a plan should expect.

    A rate distinguishable from zero but not by much has most of its uncertainty
    on the side that makes the tyre wear FASTER than the point estimate, because
    the other side is cut off by physics. The expected-time-minimising plan uses
    this mean, not the point estimate. (Total time is linear in each rate, so
    this is exactly the plan that minimises expected time under the truncated
    belief -- no simulation needed.)
    """
    if sd <= 1e-12:
        return max(mu, floor)
    a = (floor - mu) / sd
    return float(mu + sd * stats.norm.pdf(a) / max(stats.norm.sf(a), 1e-12))


def plan_for(
    race_laps: int,
    rates: dict[str, float],
    nominated: list[str],
    order: tuple[str, ...],
    pit_s: float,
    step_s: float = PACE_STEP_S,
    max_stint: dict[str, int] | None = None,
    floor_stint: dict[str, int] | None = None,
) -> Rec | None:
    """The cost-minimising plan on ``rates``; None if fewer than two compounds survive."""
    if len(rates) < 2:
        return None
    off = pace_offsets(nominated, step_s, order)
    try:
        p = best_plan(race_laps, rates, {c: off[c] for c in rates}, pit_s, max_stint=max_stint,
                      floor_stint=floor_stint)
    except ValueError:
        return None
    # best_plan is given offsets only for usable compounds, which can move the
    # zero point; shift so every compound keeps its PHYSICAL offset. A constant
    # shift cannot change the argmin, so this is bookkeeping, not a decision.
    return Rec(tuple(int(n) for n in p.stints), tuple(p.compounds))


# ---------------------------------------------------------------------------
# pit loss, never from the race being scored
# ---------------------------------------------------------------------------


def pit_loss_table(laps: pd.DataFrame) -> pd.DataFrame:
    """Green-flag pit loss per race: ``season, event, circuit, pit_s, n, order``.

    ``order`` is the race's position in calendar order (see ``FieldRace.order``).
    """
    raw = racedata.as_raw_laps(laps)
    loc = racedata.locations()
    chron = {
        (int(s), str(e)): i
        for i, (s, e) in enumerate(laps[["season", "event"]].drop_duplicates().itertuples(index=False))
    }
    rows = []
    for (season, event), g in raw.groupby(["season", "event"], sort=True):
        pl = estimate(g, event)
        if pl is None:
            continue
        rows.append(
            {
                "season": int(season),
                "event": str(event),
                "circuit": canonical_location(loc.get((int(season), str(event)), event)),
                "pit_s": pl.seconds,
                "n": pl.n_stops,
                "order": chron[(int(season), str(event))],
            }
        )
    return pd.DataFrame(rows)


def pit_for(
    table: pd.DataFrame, season: int, event: str, circuit: str, out_of_time: bool = False
) -> tuple[float, str]:
    """Pit loss to plan a race with, built WITHOUT that race.

    The same circuit in other seasons if it has any (the product's own fallback
    for a race that has not run); else the median of every other race. Returns
    the value and where it came from.

    ``out_of_time=True`` uses only races run BEFORE this one, which is what the
    product has on race day: the leave-one-out form lets a 2022 race be planned
    with the pit loss of the same pit lane in 2025. A race with nothing earlier
    to learn from (the first of the cache) falls back to the median of all other
    races and says so.
    """
    mine = table[(table["season"] == season) & (table["event"] == event)]
    other = table[~((table["season"] == season) & (table["event"] == event))]
    if out_of_time and "order" in table.columns and len(mine):
        earlier = other[other["order"] < int(mine["order"].iloc[0])]
        same = earlier[earlier["circuit"] == circuit]
        if len(same):
            return float(same["pit_s"].median()), "same circuit, earlier seasons"
        if len(earlier) >= 3:
            return float(earlier["pit_s"].median()), "median of earlier races"
        return float(other["pit_s"].median()), "no earlier race: median of all other races"
    same = other[other["circuit"] == circuit]
    if len(same):
        return float(same["pit_s"].median()), "same circuit, other seasons"
    return float(other["pit_s"].median()), "median of other races"


# ---------------------------------------------------------------------------
# the evaluation
# ---------------------------------------------------------------------------

#: Quantile of observed stint lengths used as a per-compound ceiling in the
#: "+cap" variants. See ``strategy.support``.
SUPPORT_QUANTILE = 0.95
FLOOR_QUANTILE = 0.05

MODEL_METHODS = (
    "in_sample",
    "race_pooled",
    "practice_scalar",
    "practice_label",
    "practice_raw",
    "blend",
)


def _lap_rule_threshold(train: list[tuple[int, int]]) -> float:
    """The race length that best separates 1-stop from 2-stop races in ``train``.

    ``train`` holds ``(race_laps, modal stops)``. A one-feature stump chosen to
    minimise misclassifications; ties go to the midpoint of the tied run so the
    threshold does not hug one class.
    """
    if not train:
        return 60.0
    xs = sorted({x for x, _ in train})
    cands = [0.5 * (a + b) for a, b in zip(xs[:-1], xs[1:], strict=False)] or [xs[0] - 0.5]
    best, best_err = [], 1e9
    for t in cands:
        err = sum((2 if x >= t else 1) != y for x, y in train)
        if err < best_err - 1e-9:
            best, best_err = [t], err
        elif abs(err - best_err) < 1e-9:
            best.append(t)
    return float(np.median(best))


def evaluate(
    races: list[FieldRace],
    groups: dict[str, tuple[Cells, object, tuple[str, ...]]],
    pits: pd.DataFrame,
    mixed_rates: dict[str, dict[str, tuple[float, float]]] | None = None,
    robust: bool = True,
    step_s: float = PACE_STEP_S,
    cap_q: float | None = SUPPORT_QUANTILE,
    floor_q: float | None = FLOOR_QUANTILE,
    planners: tuple[str, ...] | None = None,
    methods: tuple[str, ...] | None = None,
    extra_inputs: dict[str, dict[str, RateInput]] | None = None,
    out_of_time: bool = True,
    stint_pool: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Score every method on every race. One row per (race, method).

    ``groups`` maps a group name to ``(cells, label_of, order)`` where each race
    carries its group in its season (2026 -> "2026", earlier -> "2022-25").

    ``out_of_time`` (the default, and the headline) fits everything that is
    fitted AROUND the plan -- the pit loss, the lap-rule threshold, the stint
    bounds -- from races run BEFORE the one scored, which is what the product has
    on race day. False is the leave-one-race-out form, where a 2022 race is
    planned with 2025's pit lane and 2025's stint lengths; it is reported beside
    the headline, not instead of it. The RATE sources (``race_pooled``, the
    practice factor, ...) stay leave-one-race-out in both modes.

    ``stint_pool`` is ``strategy.support.observed_stints`` of the raw laps: the
    genuine stints the product's bounds are built from. Without it the bounds
    come from the top ten's stints as run (fragments included), the older
    construction, kept so synthetic tests need no raw laps.
    """
    from ..strategy.circuits import canonical_location  # noqa: F811

    loc = racedata.locations()
    rows = []
    dry = [r for r in races if not r.wet]

    # Baseline training data: modal stops per race, excluding the one scored.
    modal = {r.key: (r.order, r.race_laps, r.mode_stops) for r in dry if not r.red_flag}
    prev: dict[tuple[str, int], int] = {}
    for r in dry:
        if r.red_flag:
            continue
        prev[(canonical_location(loc.get((r.season, r.event), r.event)), r.season)] = r.mode_stops

    # Observed stint lengths per label, for the support cap: every dry,
    # un-red-flagged race's top-ten stints. The scored race is left out below.
    pool = [(r.key, lab, n) for r in dry if not r.red_flag for car in r.cars for lab, n in car]
    order_of = {r.key: r.order for r in races}
    if stint_pool is not None and len(stint_pool):
        sp = stint_pool.assign(key=[f"{s} {e}" for s, e in zip(stint_pool["season"], stint_pool["event"], strict=True)])
        sp["order"] = sp["key"].map(order_of)
        sp = sp.dropna(subset=["order"])

    def _before(race: FieldRace, k: str) -> bool:
        return order_of.get(k, 10**9) < race.order if out_of_time else k != race.key

    for race in dry:
        if stint_pool is not None and len(stint_pool):
            held = sp[sp["order"] < race.order] if out_of_time else sp[sp["key"] != race.key]
            ce, fl_ = held[held["ceiling_ok"]], held[held["floor_ok"]]
            caps = support_caps(list(zip(ce["label"], ce["n"], strict=True)), cap_q) if cap_q else {}
            floors = support_floor(list(zip(fl_["label"], fl_["n"], strict=True)), floor_q) if floor_q else {}
        else:
            others = [(lab, n) for k, lab, n in pool if _before(race, k)]
            caps = support_caps(others, cap_q) if cap_q else {}
            floors = support_floor(others, floor_q) if floor_q else {}
        grp = "2026" if race.season == 2026 else "2022-25"
        cells, label_of, order = groups[grp]
        circuit = canonical_location(loc.get((race.season, race.event), race.event))
        pit_s, pit_src = pit_for(pits, race.season, race.event, circuit, out_of_time=out_of_time)
        ev_key = race.event if race.season == 2026 else race.key
        nominated = _nominated(cells, ev_key, order)
        base = {
            "race": race.key, "season": race.season, "event": race.event,
            "group": grp, "red_flag": race.red_flag, "race_laps": race.race_laps,
            "field_mode_stops": race.mode_stops, "field_median_stops": race.median_stops,
            "pit_s": pit_s, "pit_source": pit_src,
        }

        # -- baselines ------------------------------------------------------
        train = [(L, s) for k, (o, L, s) in modal.items() if _before(race, k)]
        t = _lap_rule_threshold(train)
        bl = {
            "always_1": 1,
            "lap_rule": 2 if race.race_laps >= t else 1,
        }
        past = sorted((s for (c, s) in prev if c == circuit and s < race.season))
        if past:
            bl["prev_season"] = prev[(circuit, past[-1])]
        for name, stops in bl.items():
            rec = Rec(equal_split(race.race_laps, stops))
            rows.append({**base, "method": name, "kind": "baseline", **score_rec(rec, race)})

        # -- model methods --------------------------------------------------
        if not nominated:
            continue
        inputs = rate_inputs(cells, ev_key, nominated, label_of)
        if extra_inputs and ev_key in extra_inputs:
            inputs.update(extra_inputs[ev_key])
        if mixed_rates and ev_key in mixed_rates:
            mm = mixed_rates[ev_key]
            inputs["mixed_loeo"] = RateInput({c: v[0] for c, v in mm.items()}, {c: v[1] for c, v in mm.items()})
        for name, src in inputs.items():
            if methods and name not in methods:
                continue
            # Only compounds actually in play at this weekend: a pooled model
            # carries a rate for every compound it has ever seen.
            src = RateInput(
                {c: m for c, m in src.mean.items() if c in nominated},
                {c: v for c, v in src.sd.items() if c in nominated},
            )
            usable = gate(src)
            variants = [("plugin", usable, None)]
            if robust:
                variants.append(
                    ("robust", {c: truncated_mean(mu, src.sd.get(c, 0.0)) for c, mu in usable.items()}, None)
                )
            if caps:
                cap_c = {c: caps[label_of(ev_key, c)] for c in nominated if label_of(ev_key, c) in caps}
                variants.append(("plugin+cap", usable, cap_c))
                if robust:
                    variants.append(
                        ("robust+cap", {c: truncated_mean(mu, src.sd.get(c, 0.0)) for c, mu in usable.items()}, cap_c)
                    )
            fl_c = {c: floors[label_of(ev_key, c)] for c in nominated if label_of(ev_key, c) in floors}
            if caps and fl_c:
                variants.append(("plugin+support", usable, cap_c))
            for planner, rates, cap_c in variants:
                if planners and planner not in planners:
                    continue
                fl = fl_c if planner.endswith("+support") else None
                rec = plan_for(race.race_laps, rates, nominated, order, pit_s, step_s, cap_c, fl)
                hits = bound_hits(rec.labels, rec.stints, cap_c or {}, fl or {}) if rec is not None else []
                if rec is None:
                    rows.append({**base, "method": f"{name}/{planner}", "kind": "model",
                                 "planned": False, "n_usable": len(usable)})
                    continue
                labs = tuple(label_of(ev_key, c) or c for c in rec.labels)
                rec = Rec(rec.stints, labs)
                rows.append(
                    {**base, "method": f"{name}/{planner}", "kind": "model", "planned": True,
                     "n_usable": len(usable), **score_rec(rec, race),
                     "on_support_bound": int(bool(hits)), "at_stop_limit": int(rec.stops >= MAX_STOPS),
                     "plan": [f"{l} x{n}" for l, n in zip(labs, rec.stints, strict=True)]}
                )
    return pd.DataFrame(rows)


def _nominated(cells: Cells, ev_key: str, order: tuple[str, ...]) -> list[str]:
    """The compounds in play at a race: every compound with a practice or race cell."""
    seen = set(cells.race.loc[cells.race["event"] == ev_key, "C"]) | set(
        cells.practice.loc[cells.practice["event"] == ev_key, "C"]
    )
    return [c for c in order if c in seen]


# ---------------------------------------------------------------------------
# summaries, with race-level intervals
# ---------------------------------------------------------------------------

METRICS = (
    "stop_match_mode",
    "stop_abs_err_median",
    "stop_share",
    "w1_median_car",
    "w1_nearest_car",
    "longest_err",
    "shortest_err",
    "on_boundary",
    "on_support_bound",
    "at_stop_limit",
    "usage_tv_median_car",
)


def boot_mean(x: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap interval, resampling races."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """Dry, un-red-flagged races only: the set every headline number is over."""
    return df[~df["red_flag"]]


def summarise(df: pd.DataFrame, group: str | None = None) -> dict[str, dict]:
    """Per method: how many races, how many planned, and every metric with its interval."""
    d = headline(df)
    if group:
        d = d[d["group"] == group]
    out: dict[str, dict] = {}
    for method, g in d.groupby("method", sort=False):
        planned = g[g.get("planned", True) != False] if "planned" in g.columns else g  # noqa: E712
        planned = planned.dropna(subset=["stops"])
        row: dict = {
            "kind": str(g["kind"].iloc[0]),
            "n_races_scored": int(len(g)),
            "n_planned": int(len(planned)),
        }
        for m in METRICS:
            if m in planned.columns and planned[m].notna().any():
                mu, lo, hi = boot_mean(planned[m].to_numpy(float))
                row[m] = {"mean": round(mu, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        if len(planned):
            row["mean_longest_stint"] = round(float(planned["longest_stint"].mean()), 1)
            row["mean_shortest_stint"] = round(float(planned["shortest_stint"].mean()), 1)
            row["field_mean_longest_stint"] = round(float(planned["field_longest"].mean()), 1)
            row["field_mean_shortest_stint"] = round(float(planned["field_shortest"].mean()), 1)
            row["mean_stops"] = round(float(planned["stops"].mean()), 2)
        out[str(method)] = row
    return out


def paired_diff(
    df: pd.DataFrame, a: str, b: str, metric: str, group: str | None = None
) -> dict | None:
    """Mean of ``metric(a) - metric(b)`` over the races both planned, with a race-level interval."""
    d = headline(df)
    if group:
        d = d[d["group"] == group]
    pa = d[d["method"] == a].set_index("race")[metric]
    pb = d[d["method"] == b].set_index("race")[metric]
    j = pd.concat([pa, pb], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 3:
        return None
    mu, lo, hi = boot_mean((j["a"] - j["b"]).to_numpy(float))
    return {"n_races": int(len(j)), "mean_a_minus_b": round(mu, 3), "ci95": [round(lo, 3), round(hi, 3)],
            "a": a, "b": b, "metric": metric}


def common_races(df: pd.DataFrame, methods: list[str], group: str | None = None) -> set[str]:
    """Races every listed method planned, so a comparison is over one set of races."""
    d = headline(df)
    if group:
        d = d[d["group"] == group]
    sets = []
    for m in methods:
        g = d[d["method"] == m]
        if "planned" in g.columns:
            g = g[g["planned"] != False]  # noqa: E712
        sets.append(set(g.dropna(subset=["stops"])["race"]))
    return set.intersection(*sets) if sets else set()

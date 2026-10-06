"""Turning degradation curves into a decision.

A curve is an intermediate result. What a strategist needs is: how long do I
stay out, and is one stop faster than two?

Three inputs, and all three are measured rather than assumed:

  degradation   seconds lost per lap of tyre age, per compound (models/mixed)
  pace offset   how much faster a compound is when fresh. ASSUMED at 0.6 s per
                compound step (PACE_STEP_S); see the note there for what we tried
                to measure and why it failed
  pit loss      seconds lost by stopping, per circuit (strategy/pitloss)

Total race time for a stint of L laps on compound c has a closed form, so there
is no need to simulate lap by lap:

    time(c, L) = L * offset_c
               + rate_c  * L(L+1)/2            (sum of 1..L)
               + curv_c  * L(L+1)(2L+1)/6      (sum of squares)

Enumerating every split of the race across stints is then cheap enough to be
exact, which avoids an optimiser that might find a local minimum and report it
with unearned confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations, product

import numpy as np

#: Hardest to softest. A compound's PHYSICAL step is its position here, and the
#: pace gap between two compounds is the number of steps between them -- not
#: their position among whichever compounds happen to be usable.
C_ORDER = ("C1", "C2", "C3", "C4", "C5")

#: Relative labels, hardest to softest, for seasons whose C-number nomination we
#: do not hold. Three steps, so a label's step is only meaningful within one race.
LABEL_ORDER = ("HARD", "MEDIUM", "SOFT")

#: Fresh-tyre pace gap between adjacent compounds, seconds. THE one definition:
#: ``scripts/06_strategy.py`` and earlier front ends each carried their own copy
#: of this number until they were deduplicated here.
#:
#: ASSUMED, not fitted, and the weakest input in the layer. The figure is the
#: one the sport itself quotes. Two direct measurements were tried and
#: neither is usable on this data -- see ``paper/strategy_eval.pace_step`` and
#: ``paper/results/strategy.json`` for what they returned and why. Everything
#: that turns on it is reported as a sensitivity sweep (0.3-1.0 s), not as a
#: finding.
PACE_STEP_S = 0.60

#: One standard deviation of what we believe about ``PACE_STEP_S``, used only
#: when propagating uncertainty. Chosen so +/-2 sd spans 0.2-1.0 s, the range
#: the sweep covers and the range the two failed measurements straddle. It is a
#: statement of ignorance, not an estimate.
PACE_STEP_SD_S = 0.20


def pace_offsets(
    compounds: list[str] | tuple[str, ...],
    step_s: float = PACE_STEP_S,
    order: tuple[str, ...] = C_ORDER,
) -> dict[str, float]:
    """Fresh-tyre pace offset per compound, indexed by PHYSICAL compound step.

    The hardest of ``compounds`` is the zero point and each softer step is
    ``step_s`` faster. The earlier code enumerated only the USABLE compounds and
    used the loop index, so excluding a middle compound (a C4 refused for a
    non-positive rate, between a C3 and a C5) closed the gap: C3 and C5 came out
    one step apart instead of two, and the soft looked 0.6 s less attractive than
    its physical position says. Indexing by position in ``order`` removes that.

    Pass every NOMINATED compound, not just the usable ones, so an exclusion can
    never move another compound's offset.
    """
    idx = {c: order.index(c) for c in compounds}
    ref = min(idx.values())
    return {c: -step_s * (i - ref) for c, i in idx.items()}


#: A stint shorter than this is not a strategy, it is a mistake. Prevents the
#: enumeration proposing a two-lap stint to game the arithmetic.
MIN_STINT_LAPS = 5

#: Most stops any enumeration here will consider. A recommendation AT this count
#: may have been truncated by it (a plan with one more stop was never scored), so
#: callers flag it rather than present it as an unconstrained optimum.
MAX_STOPS = 3

#: Formula 1 sporting regulations: in a dry race a car must use at least two
#: different compounds. A one-compound strategy is not merely slow, it is
#: illegal, so it must never be recommended.
MIN_DISTINCT_COMPOUNDS = 2


@dataclass(frozen=True)
class Plan:
    """One strategy: which compound runs which stint length, and how many stops.

    NOT a running order. Every stint starts on a fresh tyre, so the total is a
    sum over unordered (compound, laps) pairs -- swapping two stints cannot
    change it, and across the 1.47M candidates at Barcelona no allocation's
    orderings differ by more than 1e-12. The optimiser is therefore genuinely
    indifferent to sequence, and the model has no term for the things that would
    decide it: track position, traffic, the undercut, warm-up, safety-car risk.

    ``compounds`` and ``stints`` are held sorted so the same allocation always
    prints the same way. That is a display convention. Reading it as "do this,
    then that" is reading a recommendation we did not make.
    """

    compounds: tuple[str, ...]
    stints: tuple[int, ...]
    total_time: float
    #: Time from tyres alone, with the pit loss stripped out. Kept separately so
    #: pit loss can be varied without refitting anything -- which is exactly what
    #: the crossover calculation needs.
    tyre_time: float = 0.0

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    def describe(self) -> str:
        # " + " and not " -> ": see the class docstring. An arrow would claim a
        # running order the optimiser cannot rank.
        parts = [f"{c} x{n}" for c, n in zip(self.compounds, self.stints, strict=True)]
        return f"{self.n_stops}-stop: " + " + ".join(parts)


def stint_time(
    laps: int,
    rate: float,
    offset: float = 0.0,
    curvature: float = 0.0,
) -> float:
    """Time lost over a stint, relative to a hypothetical fresh-tyre baseline.

    Closed form rather than a loop: exact, and fast enough to enumerate every
    strategy rather than search for one.
    """
    if laps <= 0:
        return 0.0
    n = float(laps)
    return (
        n * offset
        + rate * n * (n + 1) / 2
        + curvature * n * (n + 1) * (2 * n + 1) / 6
    )


def _splits(total: int, n_stints: int, min_laps: int, step: int = 1):
    """Every way to divide ``total`` laps into ``n_stints`` parts."""
    if n_stints == 1:
        if total >= min_laps:
            yield (total,)
        return
    for first in range(min_laps, total - min_laps * (n_stints - 1) + 1, step):
        for rest in _splits(total - first, n_stints - 1, min_laps, step):
            yield (first, *rest)


def _usable(rates: dict[str, float], offsets: dict[str, float]) -> list[str]:
    """Compounds the optimiser may use, refusing any whose rate is not positive.

    This is not fastidiousness. Our C5 estimate came out at -0.004 s/lap --
    statistically indistinguishable from zero, but the optimiser reads it as a
    tyre that gets FASTER as it wears, and duly recommends fitting softs on lap
    5 and running 65 laps to the flag. An optimiser will always exploit a
    non-physical input, and it will do so with total confidence.

    Same discipline as the identifiability filter in the transfer step: refuse
    the input rather than dress up the output. (The identifiability gate proper
    -- is the rate distinguishable from zero? -- lives in ``strategy.rates`` and
    runs before anything reaches here; this is the last-line sign check.)
    """
    available = [c for c in rates if c in offsets]
    unphysical = [c for c in available if rates[c] <= 0]
    available = [c for c in available if rates[c] > 0]
    if len(available) < MIN_DISTINCT_COMPOUNDS:
        raise ValueError(
            f"need at least {MIN_DISTINCT_COMPOUNDS} compounds with a positive "
            f"fitted degradation rate and a pace offset. "
            f"usable: {available}; rejected as non-physical: {unphysical}. "
            "A strategy cannot be optimised on a tyre the model thinks never wears."
        )
    return available


def _enumerate_vec(
    race_laps: int,
    available: list[str],
    rates: dict[str, float],
    offsets: dict[str, float],
    pit_loss_s: float,
    curvature: dict[str, float],
    stint_counts: range,
    min_stint: int,
    already_used: frozenset[str],
    max_stint: dict[str, int] | None,
    floor_stint: dict[str, int] | None,
) -> list[Plan]:
    """``_enumerate`` at ``step=1``, vectorised: the same plans, found without a Python loop per plan.

    Every allocation is a multiset of (compound, laps) pairs and has exactly one
    representation with its pairs in sorted order, so each is generated once by
    taking non-decreasing compound sequences and, within a run of the same
    compound, non-decreasing lengths. The totals use the same expressions as
    ``stint_time`` and agree with the loop's to rounding error (a test pins both
    the set of plans and their totals to 1e-9). The
    loop version cost seconds at a 70-lap race with a floor and a ceiling; this is
    tens of milliseconds.
    """
    n = np.arange(race_laps + 1, dtype=float)
    cost = {
        c: n * offsets[c]
        + rates[c] * n * (n + 1) / 2
        + curvature.get(c, 0.0) * n * (n + 1) * (2 * n + 1) / 6
        for c in available
    }
    plans: list[Plan] = []
    for k in stint_counts:
        if race_laps < min_stint * k:
            continue
        S = _split_array(race_laps, k, min_stint)
        if len(S) == 0:
            continue
        for combo in combinations_with_replacement(sorted(available), k):
            if len(set(combo) | already_used) < MIN_DISTINCT_COMPOUNDS:
                continue
            keep = np.ones(len(S), dtype=bool)
            for j, c in enumerate(combo):
                if max_stint and c in max_stint:
                    keep &= S[:, j] <= int(max_stint[c])
                if floor_stint and c in floor_stint:
                    keep &= S[:, j] >= int(floor_stint[c])
                if j and combo[j - 1] == c:
                    keep &= S[:, j - 1] <= S[:, j]
            if not keep.any():
                continue
            rows = S[keep]
            tyre = cost[combo[0]][rows[:, 0]]
            for j in range(1, k):
                tyre = tyre + cost[combo[j]][rows[:, j]]
            total = tyre + pit_loss_s * (k - 1)
            for r, t, ty in zip(rows.tolist(), total.tolist(), tyre.tolist(), strict=True):
                plans.append(Plan(combo, tuple(r), t, ty))
    plans.sort(key=lambda p: p.total_time)
    return plans


def _enumerate(
    race_laps: int,
    available: list[str],
    rates: dict[str, float],
    offsets: dict[str, float],
    pit_loss_s: float,
    curvature: dict[str, float],
    stint_counts: range,
    min_stint: int,
    step: int,
    already_used: frozenset[str] = frozenset(),
    max_stint: dict[str, int] | None = None,
    floor_stint: dict[str, int] | None = None,
) -> list[Plan]:
    """Every allocation of ``race_laps`` over ``n`` stints, for each ``n`` given.

    ``already_used`` is what makes the two-compound rule a rule about the RACE
    and not about the stints enumerated here. A plan is legal when the compounds
    it fits, together with those already run, number at least two. For a whole
    race ``already_used`` is empty and this is the old behaviour exactly; for the
    laps remaining after a stop it is what stops the enumeration demanding two
    NEW compounds when one tyre has already been run.
    """
    if step == 1:
        return _enumerate_vec(
            race_laps, available, rates, offsets, pit_loss_s, curvature, stint_counts,
            min_stint, frozenset(already_used), max_stint, floor_stint,
        )
    return _enumerate_lattice(
        race_laps, available, rates, offsets, pit_loss_s, curvature, stint_counts,
        min_stint, step, already_used, max_stint, floor_stint,
    )


def _enumerate_lattice(
    race_laps: int,
    available: list[str],
    rates: dict[str, float],
    offsets: dict[str, float],
    pit_loss_s: float,
    curvature: dict[str, float],
    stint_counts: range,
    min_stint: int,
    step: int,
    already_used: frozenset[str] = frozenset(),
    max_stint: dict[str, int] | None = None,
    floor_stint: dict[str, int] | None = None,
) -> list[Plan]:
    """The loop enumeration. The only one that can walk a coarse lattice (``step > 1``),
    and the reference the vectorised ``step=1`` path is tested against."""
    plans: list[Plan] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for n_stints in stint_counts:
        if race_laps < min_stint * n_stints:
            continue

        # combinations_with_replacement gives the multisets; permuting each one
        # then assigns compounds to slots. Two things here are easy to get wrong.
        #
        # First, sequence does not affect total_time. Every stint starts at age
        # zero, so the total is a sum over unordered (compound, laps) pairs --
        # across the 1.47M plans at Barcelona, not one pairing's orderings
        # disagree by more than 1e-12. The permutation loop survives anyway
        # because _splits is not permutation-closed once step > 1: it walks the
        # first stint along a lattice and lets the last absorb the remainder, so
        # permuting the compounds reaches pairings permuting the laps cannot.
        # Dropping it changes the step=2 optimum, which is the coarse grid used
        # for quick interactive answers.
        #
        # Second, sorted(). Set iteration follows the hash seed, and the best
        # plan can be an exact tie. Unsorted, the same data recommended
        # "C2 x23 -> C3 x19 -> C2 x24" on one boot and "C3 x19 -> C2 x23 -> C2
        # x24" on the next. The sequence we print is therefore a stable
        # convention, not a claim: the model ranks which compound runs which
        # stint length, and is indifferent to their order.
        for combo in combinations_with_replacement(available, n_stints):
            if len(set(combo) | already_used) < MIN_DISTINCT_COMPOUNDS:
                continue
            for perm in sorted(set(permutations(combo))):
                for stints in _splits(race_laps, n_stints, min_stint, step):
                    if max_stint and any(n > max_stint.get(c, 10**9) for c, n in zip(perm, stints, strict=True)):
                        continue
                    if floor_stint and any(n < floor_stint.get(c, 0) for c, n in zip(perm, stints, strict=True)):
                        continue
                    # One plan per allocation, not per sequence. Sorting the
                    # pairs is what collapses the orderings: at Barcelona this
                    # takes 1,470,486 plans down to 69,145 without losing a
                    # single distinct allocation at any step size.
                    pairing = tuple(sorted(zip(perm, stints, strict=True)))
                    if pairing in seen:
                        continue
                    seen.add(pairing)
                    tyre_time = sum(
                        stint_time(laps, rates[c], offsets[c], curvature.get(c, 0.0))
                        for c, laps in zip(perm, stints, strict=True)
                    )
                    plans.append(
                        Plan(
                            tuple(c for c, _ in pairing),
                            tuple(n for _, n in pairing),
                            tyre_time + pit_loss_s * (n_stints - 1),
                            tyre_time,
                        )
                    )
    plans.sort(key=lambda p: p.total_time)
    return plans


def enumerate_plans(
    race_laps: int,
    rates: dict[str, float],
    offsets: dict[str, float],
    pit_loss_s: float,
    curvature: dict[str, float] | None = None,
    max_stops: int = MAX_STOPS,
    min_stint: int = MIN_STINT_LAPS,
    step: int = 1,
    max_stint: dict[str, int] | None = None,
    floor_stint: dict[str, int] | None = None,
) -> list[Plan]:
    """Every legal strategy, scored. Cheapest first.

    Args:
        race_laps: scheduled distance.
        rates: degradation, s/lap, keyed by compound.
        offsets: fresh-tyre pace, s/lap, keyed by compound. Lower is faster.
            Build it with ``pace_offsets`` so it follows physical compound step.
        pit_loss_s: seconds lost per stop.
        curvature: optional quadratic term per compound.
        max_stops: highest stop count to consider.
        max_stint: optional ceiling on a stint's length per compound, in laps.
            The degradation rates are identified from stints no longer than the
            field ran, and a linear rate extrapolated far beyond that support is
            a guess; see ``strategy.support``.
        floor_stint: the mirror: a minimum stint length per compound. The
            five-lap legal minimum is a structural limit, not a strategy.

    Returns:
        Legal plans sorted by total time. Plans using fewer than two distinct
        compounds are excluded, because the regulations forbid them in the dry.
    """
    available = _usable(rates, offsets)
    return _enumerate(
        race_laps, available, rates, offsets, pit_loss_s, curvature or {},
        range(2, max_stops + 2), min_stint, step, max_stint=max_stint,
        floor_stint=floor_stint,
    )


def enumerate_remaining(
    remaining_laps: int,
    rates: dict[str, float],
    offsets: dict[str, float],
    pit_loss_s: float,
    already_used: frozenset[str] | set[str],
    stops_made: int = 0,
    max_stops: int = MAX_STOPS,
    min_stint: int = MIN_STINT_LAPS,
    step: int = 1,
    max_stint: dict[str, int] | None = None,
    floor_stint: dict[str, int] | None = None,
) -> list[Plan]:
    """Ways to finish a race after a stop that is being evaluated.

    The stop itself is NOT in these plans -- the caller charges it -- so a plan
    with one stint means "stop now, fit this tyre, run to the flag". That option
    was impossible before: ``enumerate_plans`` forces at least one stop and two
    compounds, so asking it for the laps after a stop charged a second stop and
    demanded two new compounds. "Pit now and finish on these" is the commonest
    pit-wall call there is and it could not be priced.

    Args:
        remaining_laps: laps left after the evaluated stop.
        already_used: compounds run so far, including the one being left. The
            two-compound rule is checked against these plus the new ones.
        stops_made: stops already taken before the evaluated one. Together with
            the evaluated stop they count against ``max_stops``.

    Returns:
        Plans over the remaining laps, cheapest first; ``Plan.n_stops`` counts
        the FURTHER stops only. Empty if nothing legal fits.
    """
    available = _usable(rates, offsets)
    extra_allowed = max_stops - stops_made - 1
    if extra_allowed < 0:
        return []
    return _enumerate(
        remaining_laps, available, rates, offsets, pit_loss_s, {},
        range(1, extra_allowed + 2), min_stint, step, frozenset(already_used),
        max_stint=max_stint, floor_stint=floor_stint,
    )


@lru_cache(maxsize=256)
def _split_array(total: int, n_stints: int, min_laps: int) -> np.ndarray:
    """Every composition of ``total`` into ``n_stints`` parts >= ``min_laps``, as rows.

    Stars and bars: put ``n_stints - 1`` dividers among ``total - n_stints*min +
    n_stints - 1`` slots. Cached, because an evaluation over many races and rate
    sets asks for the same few shapes thousands of times.
    """
    spare = total - min_laps * n_stints
    if spare < 0:
        return np.empty((0, n_stints), dtype=np.int64)
    if n_stints == 1:
        return np.array([[total]], dtype=np.int64)
    cuts = np.array(list(combinations(range(spare + n_stints - 1), n_stints - 1)), dtype=np.int64)
    edges = np.concatenate(
        [np.full((len(cuts), 1), -1), cuts, np.full((len(cuts), 1), spare + n_stints - 1)], axis=1
    )
    return np.diff(edges, axis=1) - 1 + min_laps


def best_plan(
    race_laps: int,
    rates: dict[str, float],
    offsets: dict[str, float],
    pit_loss_s: float,
    max_stops: int = MAX_STOPS,
    min_stint: int = MIN_STINT_LAPS,
    stint_counts: range | None = None,
    already_used: frozenset[str] | set[str] = frozenset(),
    max_stint: dict[str, int] | None = None,
    floor_stint: dict[str, int] | None = None,
) -> Plan:
    """The single cheapest legal plan, found by vectorised exhaustive search.

    Exactly the optimum ``enumerate_plans(...)[0]`` returns (a test pins that),
    at roughly a hundredth of the cost, because it never builds the plans it
    discards. For callers that need the optimum and nothing else -- an
    evaluation over hundreds of races and rate sets, say. Anything that needs
    the whole ranked list (a per-stop-count table, a Monte Carlo
    over every plan) still uses ``enumerate_plans``.
    """
    available = _usable(rates, offsets)
    used = frozenset(already_used)
    n = np.arange(0, race_laps + 1, dtype=float)
    cost = {c: n * offsets[c] + rates[c] * n * (n + 1) / 2 for c in available}
    for c, cap in (max_stint or {}).items():
        if c in cost:
            cost[c][int(cap) + 1:] = np.inf
    for c, lo in (floor_stint or {}).items():
        if c in cost:
            cost[c][1:int(lo)] = np.inf
    counts = stint_counts if stint_counts is not None else range(2, max_stops + 2)
    best: tuple[float, tuple[str, ...], tuple[int, ...]] | None = None
    for k in counts:
        S = _split_array(race_laps, k, min_stint)
        if len(S) == 0:
            continue
        for combo in product(available, repeat=k):
            if len(set(combo) | used) < MIN_DISTINCT_COMPOUNDS:
                continue
            t = sum(cost[c][S[:, j]] for j, c in enumerate(combo))
            i = int(np.argmin(t))
            if not np.isfinite(t[i]):
                continue
            if best is None or t[i] + pit_loss_s * (k - 1) < best[0] - 1e-12:
                best = (float(t[i]) + pit_loss_s * (k - 1), combo, tuple(int(x) for x in S[i]))
    if best is None:
        raise ValueError("no legal plan fits these inputs")
    total, combo, stints = best
    pairing = sorted(zip(combo, stints, strict=True))
    return Plan(
        tuple(c for c, _ in pairing),
        tuple(m for _, m in pairing),
        total,
        total - pit_loss_s * (len(stints) - 1),
    )


def best_per_stop_count(plans: list[Plan]) -> dict[int, Plan]:
    """The cheapest plan for each stop count. What the crossover chart needs."""
    out: dict[int, Plan] = {}
    for p in plans:
        if p.n_stops not in out or p.total_time < out[p.n_stops].total_time:
            out[p.n_stops] = p
    return out


def optimal_stint(
    compound: str,
    rate: float,
    pit_loss_s: float,
    curvature: float = 0.0,
    max_laps: int = 60,
) -> int:
    """Stint length that minimises average time per lap including the stop.

    The textbook trade-off: staying out longer spreads the pit loss over more
    laps, but every extra lap is slower than the last. The minimum of
    (stint time + pit loss) / laps is where those balance.
    """
    lengths = np.arange(MIN_STINT_LAPS, max_laps + 1)
    per_lap = [
        (stint_time(int(n), rate, 0.0, curvature) + pit_loss_s) / n for n in lengths
    ]
    return int(lengths[int(np.argmin(per_lap))])


def crossover(
    plans: list[Plan],
    pit_loss_range: tuple[float, float] = (15.0, 35.0),
    steps: int = 41,
) -> float | None:
    """Pit loss at which the best one-stop overtakes the best two-stop.

    The single most useful number in the whole strategy layer, because pit loss
    is the input a team can least control and most needs to plan around. Below
    the crossover, stopping twice is worth it; above it, stay out.

    Returns None if one option wins across the entire range.
    """
    one = [p for p in plans if p.n_stops == 1]
    two = [p for p in plans if p.n_stops == 2]
    if not one or not two:
        return None

    # Compare on TYRE time only, so pit loss can be swept independently.
    # Using total_time here would double-count it, since total_time already
    # includes the pit loss the plan was scored with.
    one_tyre = min(p.tyre_time for p in one)
    two_tyre = min(p.tyre_time for p in two)

    prev = None
    for pl in np.linspace(*pit_loss_range, steps):
        favours_two = (two_tyre + 2 * pl) < (one_tyre + 1 * pl)
        if prev is not None and favours_two != prev:
            return float(pl)
        prev = favours_two
    return None

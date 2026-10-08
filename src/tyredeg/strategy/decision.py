"""One strategy decision, computed one way, for the single-event script and the paper's evaluation.

``scripts/06_strategy.py`` and earlier front ends each ran
their own copy of: build offsets, enumerate, take the best per stop count, find
the margin and the crossover, and invent a confidence. They disagreed in small
ways (the offset indexing, the confidence) that a test then had to police. This
is the single copy.

WHAT "CONFIDENCE" NOW MEANS

``confidence`` is the probability, under the Monte Carlo of the inputs we are
unsure of (``strategy.uncertainty``), that the recommended STOP COUNT is the
cheapest one. It replaces ``clamp(margin / 10)``, which converted a margin in
seconds to a number that looked like a probability and was not one.

It is the uncertainty of the optimiser's INPUTS only: degradation rates (from
their fitted intervals, with the circuit-slope uncertainty missing), pit loss,
and the assumed pace step. It says nothing about traffic, safety cars or
position, which decide real races and are not in the model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


from .optimise import (
    C_ORDER,
    PACE_STEP_S,
    PACE_STEP_SD_S,
    Plan,
    best_per_stop_count,
    crossover,
    enumerate_plans,
    pace_offsets,
)
from .uncertainty import DEFAULT_PIT_SD_S, N_DRAWS, Beliefs, Propagated, propagate

#: Plans carried into the Monte Carlo when the caller asks for a shortlist. The
#: rest are far enough behind the best that none is ever cheapest in a draw; the
#: shortlist only has to be large enough for that to be true, and it also always
#: contains the best plan at every stop count.
SHORTLIST = 1500


@dataclass
class PlanScore:
    plan: Plan
    p_best: float
    total_mean: float
    total_lo: float
    total_hi: float
    regret_mean: float


@dataclass
class Decision:
    plans_all: list[Plan]
    best: dict[int, Plan]
    rec: Plan
    runner_up: Plan | None
    margin_s: float
    crossover_pit_loss_s: float | None
    #: P(the recommended stop count is the cheapest stop count). None when the
    #: caller switched the Monte Carlo off (the API does while a slider drags).
    confidence: float | None
    stop_prob: dict[int, float]
    scores: dict[int, PlanScore]
    rec_score: PlanScore | None
    n_draws: int
    offsets: dict[str, float] = field(default_factory=dict)


def decide(
    race_laps: int,
    rates: dict[str, float],
    rate_sd: dict[str, float],
    nominated: list[str],
    pit_loss_s: float,
    *,
    pit_sd_s: float = DEFAULT_PIT_SD_S,
    pace_step_s: float = PACE_STEP_S,
    pace_step_sd_s: float = PACE_STEP_SD_S,
    step: int = 1,
    order: tuple[str, ...] = C_ORDER,
    n_draws: int = N_DRAWS,
    shortlist: int | None = None,
    mc: bool = True,
    max_stint: dict[str, int] | None = None,
    floor_stint: dict[str, int] | None = None,
    enumerate_fn: Callable[..., list[Plan]] | None = None,
    plans_all: list[Plan] | None = None,
) -> Decision:
    """Enumerate, rank, and score under uncertainty.

    Args:
        nominated: EVERY compound at the weekend, usable or not. The pace offset
            is indexed by physical step over these, so excluding one cannot move
            another's offset.
        shortlist: if set, the Monte Carlo runs over the best ``shortlist``
            plans plus the best at every stop count, rather than every plan. The
            API sets it to stay interactive; offline callers leave it None and
            get the exact probability of being cheapest among all plans.
        max_stint, floor_stint: per-compound ceiling and floor on stint length,
            in laps (see ``strategy.support``); None means unbounded.
        enumerate_fn: lets the API pass its memoised enumeration.
        plans_all: an already-enumerated, sorted list, to skip enumeration.
    """
    off_all = pace_offsets(nominated, pace_step_s, order)
    offsets = {c: off_all[c] for c in rates}
    if plans_all is None:
        fn = enumerate_fn or enumerate_plans
        plans_all = fn(
            race_laps, rates, offsets, pit_loss_s, step=step,
            max_stint=max_stint, floor_stint=floor_stint,
        )
    if not plans_all:
        raise ValueError("no legal strategy exists for these inputs")

    best = best_per_stop_count(plans_all)
    rec = plans_all[0]
    runner_up = next((p for p in plans_all if p.n_stops != rec.n_stops), None)
    margin = (runner_up.total_time - rec.total_time) if runner_up else 0.0

    if not mc:
        return Decision(
            plans_all=plans_all, best=best, rec=rec, runner_up=runner_up, margin_s=float(margin),
            crossover_pit_loss_s=crossover(plans_all), confidence=None, stop_prob={}, scores={},
            rec_score=None, n_draws=0, offsets=offsets,
        )

    use = plans_all
    if shortlist is not None and len(plans_all) > shortlist:
        keep = {id(p): p for p in plans_all[:shortlist]}
        keep.update({id(p): p for p in best.values()})
        use = sorted(keep.values(), key=lambda p: p.total_time)

    beliefs = Beliefs(
        rate_mean=dict(rates),
        rate_sd={c: float(rate_sd.get(c, 0.0)) for c in rates},
        pit_mean=pit_loss_s,
        pit_sd=pit_sd_s,
        step_mean=pace_step_s,
        step_sd=pace_step_sd_s,
        order=order,
        reference=tuple(nominated),
    )
    prop: Propagated = propagate(use, beliefs, n_draws=n_draws)
    index = {id(p): i for i, p in enumerate(use)}

    def score(p: Plan) -> PlanScore:
        i = index[id(p)]
        return PlanScore(
            p, float(prop.p_best[i]), float(prop.mean_total[i]), float(prop.lo_total[i]),
            float(prop.hi_total[i]), float(prop.mean_regret[i]),
        )

    scores = {n: score(p) for n, p in best.items()}
    stop_prob = {n: float(prop.stop_prob.get(n, 0.0)) for n in best}
    return Decision(
        plans_all=plans_all,
        best=best,
        rec=rec,
        runner_up=runner_up,
        margin_s=float(margin),
        crossover_pit_loss_s=crossover(plans_all),
        confidence=float(stop_prob.get(rec.n_stops, 0.0)),
        stop_prob=stop_prob,
        scores=scores,
        rec_score=score(rec),
        n_draws=n_draws,
        offsets=offsets,
    )


def stop_prob_json(d: Decision) -> dict[str, float]:
    return {str(k): round(v, 3) for k, v in sorted(d.stop_prob.items())}


__all__ = ["Decision", "PlanScore", "SHORTLIST", "decide", "stop_prob_json"]

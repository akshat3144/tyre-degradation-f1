"""How sure is the strategy call? Monte Carlo over the inputs we are unsure of.

WHY THIS EXISTS

The strategy layer used to report ``total_time`` as the plug-in total +/- a
hard-coded 3.0 s, and ``confidence`` as ``clamp(margin / 10)``. Neither is a
statement about uncertainty: the 3 s was typed in, and a margin in seconds is
not a probability. A plan that wins by 2 s when the degradation rates are known
to 0.01 s/lap is a sure thing; the same 2 s when they are known to 0.05 s/lap is
a coin flip, and the margin alone cannot tell them apart.

WHAT IS PROPAGATED

Total race time is linear in every input we are unsure of, for a fixed plan:

    total = sum_c  rate_c * T_c   -   step * O   +   pit * stops

where ``T_c = sum over stints of c of n(n+1)/2`` and ``O`` is the laps run on
each compound weighted by its physical step from the hardest. So uncertainty can
be carried exactly by drawing the inputs and re-evaluating every plan, with no
re-optimisation inside the draw loop beyond an argmin.

  rates       independent truncated normals, mean and SE from the fitted
              interval ((hi - lo) / 3.92). Truncated at ``rate_floor`` because a
              tyre does not get faster as it wears.
  pit loss    truncated normal; SE from the bootstrap interval of the stops it
              was measured from (or the between-season spread, for a history
              figure).
  pace step   normal around the ASSUMED 0.6 s with sd ``PACE_STEP_SD_S`` -- an
              honest statement of ignorance rather than an estimate. NOT
              truncated: it used to be cut at zero, which is invisible at 0.6 s
              (three sd away) but turns a caller's belief of "no gap, 0 s" into a
              half-normal with mean 0.16 s, so the draws contradicted the input.
              A negative step (a harder tyre faster) is within what the data
              cannot rule out; the 2026 and 2022-25 estimators disagree in sign.

LIMITS, STATED

  * Rates are drawn independently. The fitted intervals are for the pooled
    compound rate shifted by the circuit slope; the circuit slope's own
    uncertainty is not in them (``Fit.interval_for`` says so), so every interval
    here is a LOWER bound on what we do not know.
  * Plan-model error -- traffic, safety cars, the undercut, tyre warm-up -- is
    not in here at all. This is the uncertainty of the optimiser's inputs, not of
    the race.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .optimise import C_ORDER, PACE_STEP_S, PACE_STEP_SD_S, Plan

#: Draws per call. 1,000 puts the Monte Carlo error on a probability near 0.5 at
#: about 0.016, which is below the resolution anyone reads a confidence at.
N_DRAWS = 1000

#: Default seed, so a published number is reproducible. Callers that want the
#: Monte Carlo error itself pass different seeds.
SEED = 20260101

#: A plausible lower bound on a tyre's degradation, s/lap. Zero: a tyre does not
#: improve with age. Callers may raise it (the robust planner does).
RATE_FLOOR = 0.0

#: Pit-loss SE used when a pit loss has none attached (a typed-in override, or
#: a single season of history). The median between-season spread of a circuit's
#: measured pit loss in 2022-2025 is of this order; see ``paper/results/pit_loss.json``.
DEFAULT_PIT_SD_S = 1.5


@dataclass(frozen=True)
class Beliefs:
    """What we believe about the inputs, as a mean and an SD for each."""

    rate_mean: dict[str, float]
    rate_sd: dict[str, float]
    pit_mean: float
    pit_sd: float = DEFAULT_PIT_SD_S
    step_mean: float = PACE_STEP_S
    step_sd: float = PACE_STEP_SD_S
    #: Compounds in physical order, hardest first. The pace gap between two
    #: compounds is the number of steps between them in THIS tuple.
    order: tuple[str, ...] = C_ORDER
    #: The compounds the pace offset is measured from: the hardest of these is
    #: the zero point, matching ``optimise.pace_offsets``. Defaults to the rate
    #: keys; pass the NOMINATED set so an exclusion cannot move the zero.
    reference: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class Propagated:
    """The plans, scored under every draw."""

    plans: list[Plan]
    #: P(this plan is the cheapest of all plans considered), per plan.
    p_best: np.ndarray
    #: Mean total over draws. Differs from ``Plan.total_time`` only through the
    #: truncation (a rate cannot go below the floor) -- the total is linear in
    #: the inputs, so otherwise the mean of the draws is the plug-in value.
    mean_total: np.ndarray
    lo_total: np.ndarray
    hi_total: np.ndarray
    #: Mean seconds this plan loses to the best plan IN THE SAME DRAW.
    mean_regret: np.ndarray
    #: P(the best plan has k stops), keyed by k.
    stop_prob: dict[int, float]
    n_draws: int

    def best_index(self) -> int:
        """The plan most often cheapest."""
        return int(np.argmax(self.p_best))

    def robust_index(self) -> int:
        """The plan with the lowest EXPECTED total under the draws.

        Because totals are linear in the inputs this differs from the plug-in
        optimum only where truncation bites: a compound whose rate is barely
        above zero relative to its SE has an expected rate well above its point
        estimate once negatives are ruled out, and long stints on it cost more.
        """
        return int(np.argmin(self.mean_total))


def _truncnorm(mean: float, sd: float, lo: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Normal(mean, sd) truncated below at ``lo``; a point mass when sd is 0."""
    if sd <= 1e-12:
        return np.full(size, max(mean, lo))
    a = (lo - mean) / sd
    return stats.truncnorm.rvs(a, np.inf, loc=mean, scale=sd, size=size, random_state=rng)


def _pace_step_draw(mean: float, sd: float, size: int, rng: np.random.Generator) -> np.ndarray:
    """Normal(mean, sd) with no truncation; a point mass when sd is 0.

    Unlike a rate or a pit loss, the pace step has no physical floor in this
    model, and truncating it at zero biases the draw whenever the believed mean
    is near zero (``pace_step_s=0`` gave a mean of about 0.16).
    """
    if sd <= 1e-12:
        return np.full(size, float(mean))
    return rng.normal(mean, sd, size)


def plan_features(plans: list[Plan], compounds: list[str], step_of: dict[str, int]) -> np.ndarray:
    """One row per plan: ``[T_c for c in compounds] + [O, stops]``.

    ``total = F[:, :k] @ rates - step * F[:, k] + pit * F[:, k + 1]``
    """
    k = len(compounds)
    col = {c: i for i, c in enumerate(compounds)}
    F = np.zeros((len(plans), k + 2))
    for i, p in enumerate(plans):
        for c, n in zip(p.compounds, p.stints, strict=True):
            F[i, col[c]] += n * (n + 1) / 2.0
            F[i, k] += step_of[c] * n
        F[i, k + 1] = p.n_stops
    return F


def propagate(
    plans: list[Plan],
    beliefs: Beliefs,
    *,
    n_draws: int = N_DRAWS,
    seed: int = SEED,
    rate_floor: float = RATE_FLOOR,
    chunk: int = 100,
) -> Propagated:
    """Score every plan under ``n_draws`` draws of the uncertain inputs.

    ``plans`` should be the full enumeration, so ``p_best`` is the probability of
    being cheapest among ALL legal plans and not among a shortlist.
    """
    if not plans:
        raise ValueError("no plans to propagate uncertainty over")
    compounds = sorted({c for p in plans for c in p.compounds}, key=beliefs.order.index)
    ref_set = beliefs.reference or tuple(beliefs.rate_mean)
    ref = min(beliefs.order.index(c) for c in ref_set)
    step_of = {c: beliefs.order.index(c) - ref for c in compounds}
    F = plan_features(plans, compounds, step_of)
    k = len(compounds)

    rng = np.random.default_rng(seed)
    P = len(plans)
    p_best = np.zeros(P)
    tot_sum = np.zeros(P)
    tot_sq = np.zeros(P)
    regret_sum = np.zeros(P)
    samples = np.empty((P, n_draws), dtype=np.float32) if P * n_draws <= 40_000_000 else None
    stops = np.array([p.n_stops for p in plans])
    stop_hits: dict[int, int] = {}

    done = 0
    while done < n_draws:
        m = min(chunk, n_draws - done)
        theta = np.empty((k + 2, m))
        for j, c in enumerate(compounds):
            theta[j] = _truncnorm(
                beliefs.rate_mean[c], beliefs.rate_sd.get(c, 0.0), rate_floor, m, rng
            )
        theta[k] = -_pace_step_draw(beliefs.step_mean, beliefs.step_sd, m, rng)
        theta[k + 1] = _truncnorm(beliefs.pit_mean, beliefs.pit_sd, 5.0, m, rng)
        tot = F @ theta  # (P, m)
        best = tot.argmin(axis=0)
        np.add.at(p_best, best, 1.0)
        for s in stops[best]:
            stop_hits[int(s)] = stop_hits.get(int(s), 0) + 1
        tot_sum += tot.sum(axis=1)
        tot_sq += (tot**2).sum(axis=1)
        regret_sum += (tot - tot.min(axis=0)).sum(axis=1)
        if samples is not None:
            samples[:, done : done + m] = tot
        done += m

    mean = tot_sum / n_draws
    if samples is not None:
        lo = np.percentile(samples, 2.5, axis=1)
        hi = np.percentile(samples, 97.5, axis=1)
    else:  # too many plans to hold every draw: a normal band from the moments
        sd = np.sqrt(np.maximum(tot_sq / n_draws - mean**2, 0.0))
        lo, hi = mean - 1.96 * sd, mean + 1.96 * sd
    return Propagated(
        plans=plans,
        p_best=p_best / n_draws,
        mean_total=mean,
        lo_total=lo,
        hi_total=hi,
        mean_regret=regret_sum / n_draws,
        stop_prob={k_: v / n_draws for k_, v in sorted(stop_hits.items())},
        n_draws=n_draws,
    )


def best_by_stops(prop: Propagated) -> dict[int, int]:
    """Index of the cheapest-in-expectation plan at each stop count."""
    out: dict[int, int] = {}
    for i, p in enumerate(prop.plans):
        j = out.get(p.n_stops)
        if j is None or prop.mean_total[i] < prop.mean_total[j]:
            out[p.n_stops] = i
    return out

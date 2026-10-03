"""Do drivers manage softer tyres harder in races than in practice?

STATUS: THE ORIGINAL CLAIM THIS MODULE WAS BUILT TO TEST IS WITHDRAWN.
    The ratio test below (rho -0.288, p 0.0035, 101 cells) rested on a race
    slope estimator that leaked each compound's pace offset into its slope. On
    the same estimator for race and practice the ratio test reads rho +0.03,
    p 0.77 (81 of 99 cells have a usable practice rate), and what survives is that race degradation runs at
    about 38% of practice on matched MEDIUM and SOFT cells
    (``paper/results/management.json`` -> post_review_additions.
    level_gap_matched_pooled). Compound-specific nursing is not established, and
    a telemetry check of the mechanism was null (``paper/results/telemetry.json``).
    This module is kept because ``data/artifacts/management.json`` still
    reports the ratio view for continuity. Read ``tyredeg.paper.management``
    for the analysis the paper uses.

THE QUESTION (as originally posed)

Our race fits keep saying softer compounds do NOT degrade faster, which
contradicts how tyres work. The leading explanation is behavioural: in a race a
driver nurses a fragile tyre to hit a target lap time, so the measured
degradation is suppressed. In practice, where the team is deliberately measuring
the tyre, they push it. The benchmark paper suggested the same mechanism without
testing it.

If that is right, the ratio of race degradation to practice degradation should
be smaller for softer compounds. That is a directional, ordered prediction,
which is why a trend test rather than an omnibus test is the right choice here.

The direction is not ours to claim credit for and not something we chose after
looking: it is stated in the benchmark paper's own section 4.3, which offers
driver management as an explanation for their null result without testing it.
Anyone can check that the prediction predates our data by reading their paper.
We report the omnibus Kruskal-Wallis alongside anyway, so the choice of test is
visible rather than load-bearing.

WHY THE LABEL IS THE RIGHT KEY HERE, UNUSUALLY

Everywhere else in this project we insist on the physical compound, because
HARD/MEDIUM/SOFT is relative to each weekend's nomination and pooling the label
across events mixes different rubber.

A ratio is the exception. Within ONE event, the label is the same physical tyre
in practice and in the race, so practice-to-race ratios are comparable even
though the underlying compounds differ between events. That is what lets this
analysis use 2022 through 2025, for which we have no allocation table.

The ordering claim does still assume that within an event the SOFT nomination is
softer than the MEDIUM one, which is true by construction of the nomination.

WHAT THIS TEST STILL DOES NOT DO

It pools all cells into one Spearman, which treats a 2022 cell and a 2026 cell
as exchangeable. They are not: the seasons plainly disagree, with 2023 at
rho -0.47 and 2024 at -0.04. A season-blocked test would respect that structure;
it now exists in tyredeg.paper.management (season_analysis), together with a
same-estimator comparison. READ THAT MODULE'S DOCSTRING FIRST: the race slope
this module originally used was fitted through the origin on an (event, lap)
demeaned design and was biased per compound (see ``demean_by_compound``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

#: Hardest to softest. The rank used by the trend test.
LABELS = ("HARD", "MEDIUM", "SOFT")
LABEL_RANK = {lab: i for i, lab in enumerate(LABELS)}

#: A ratio needs a denominator that is actually a degradation rate. Cells whose
#: practice rate is near zero give explosive or meaningless ratios.
MIN_PRACTICE_RATE = 0.005

#: Minimum runs behind a cell's slope.
MIN_RUNS = 3


@dataclass
class ManagementResult:
    cells: pd.DataFrame
    median_ratio: dict[str, float]
    n_cells: int
    n_events: int
    n_seasons: int
    rho: float
    p_value: float
    #: Two-sided p as well, so the conclusion can be checked without relying on
    #: the one-sided choice. Reported alongside for exactly that reason.
    p_two_sided: float
    #: An omnibus alternative. Weaker here by design -- it ignores the ordering
    #: we predicted -- but shown so the choice of test is visible rather than
    #: something the reader has to take on trust.
    p_kruskal: float

    @property
    def ordered(self) -> bool:
        """Does the ratio fall monotonically as the tyre softens?"""
        present = [lab for lab in LABELS if lab in self.median_ratio]
        return all(
            self.median_ratio[present[i]] > self.median_ratio[present[i + 1]]
            for i in range(len(present) - 1)
        )

    @property
    def significant(self) -> bool:
        """Significant on BOTH sidedness conventions.

        Deliberately strict. Calling a result significant on the one-sided p
        while the two-sided p sits above 0.05 puts the whole claim on a choice
        the reader may not share, and that is precisely the case we sat in for a
        while: at three seasons the one-sided p was 0.031 but the two-sided was
        0.062, so this property returned False and the verdict said "marginal".

        (Historical note: on the legacy estimator this cleared only once 2022
        and 2023 were added, and that legacy result is withdrawn -- see the
        module docstring. The strictness stands as a rule for any trend test.)
        """
        return self.p_value < 0.05 and self.p_two_sided < 0.05

    def verdict(self) -> str:
        if not self.ordered:
            return "not supported: the ordering does not hold"
        if self.significant:
            return "supported: ratio falls with softness, significant either way"
        if self.p_value < 0.05:
            return (
                f"marginal: ordering holds and the predicted direction gives "
                f"p = {self.p_value:.3f}, but two-sided p = {self.p_two_sided:.3f} "
                f"is above 0.05. Consistent, not conclusive."
            )
        return f"suggestive: ordering holds but p = {self.p_value:.3f}"


def cluster_slope(g: pd.DataFrame) -> tuple[float, float] | None:
    """Slope of the centred design with a standard error clustered by run.

    Laps within a run are correlated, so an unclustered error would be far too
    small -- see validation/transfer for the coverage evidence.
    """
    x, y = g["tl"].to_numpy(float), g["y"].to_numpy(float)
    denom = float(x @ x)
    if denom <= 1e-9:
        return None
    slope = float(x @ y / denom)
    resid = y - slope * x
    clusters = g["run_id"].to_numpy()
    uniq = np.unique(clusters)
    if len(uniq) < 2:
        return slope, float("nan")
    meat = sum(float(x[clusters == c] @ resid[clusters == c]) ** 2 for c in uniq)
    se = float(np.sqrt(len(uniq) / (len(uniq) - 1) * meat) / denom)
    return slope, se


def demean_by_compound(race: pd.DataFrame) -> pd.DataFrame:
    """Re-demean a race design within (event, lap, COMPOUND) instead of (event, lap).

    ``race_design`` subtracts the field mean at each lap, a mean that mixes every
    compound on track. Split afterwards by compound, each compound keeps a non-zero
    mean lap time (soft is quicker than the field) and a non-zero mean tyre age
    (soft is fresher), and ``cluster_slope`` fits through the origin, so the
    product of those two offsets leaks into the slope. That was a bug in the
    original race estimator, and it is not a small one: on a planted simulation
    with equal true slopes the old estimator reads SOFT 0.13 against HARD 0.085.

    Demeaning within the compound removes the pace offset by construction and
    keeps the cross-sectional identification the design is built on: compare cars
    on the same compound at the same lap whose tyres differ in age. Cells with a
    single car at that (lap, compound) carry no comparison and are dropped.

    (Centring the one-pass design within the run, the obvious patch, is NOT a
    repair: it identifies the slope only from the drift of the field mean, which
    is itself contaminated by the changing compound mix. See the tests.)
    """
    r = race.copy()
    keys = ["event", "LapNumber", "Compound"]
    r = r[r.groupby(keys, observed=True)["LapTimeSeconds"].transform("size") >= 2].copy()
    g = r.groupby(keys, observed=True)
    r["y"] = r["LapTimeSeconds"] - g["LapTimeSeconds"].transform("mean")
    r["tl"] = r["TyreLife"] - g["TyreLife"].transform("mean")
    return r


def cell_slopes(df: pd.DataFrame, min_runs: int = MIN_RUNS) -> pd.DataFrame:
    """One slope per (event, label)."""
    out = []
    for (event, label), g in df.groupby(["event", "Compound"], observed=True):
        if g["run_id"].nunique() < min_runs:
            continue
        got = cluster_slope(g)
        if got is None:
            continue
        slope, se = got
        out.append(
            {
                "event": event,
                "Compound": label,
                "rate": slope,
                "se": se,
                "n_runs": g["run_id"].nunique(),
                "n_laps": len(g),
            }
        )
    return pd.DataFrame(out)


def build_design_frames(path, *, race_demean: str = "compound") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Practice and race design frames for one season's parquet.

    Moved here from scripts/07_management.py so the paper analysis can rebuild the
    legacy frames and print the old numbers beside the repaired ones.

    ``race_demean``: ``"compound"`` (default, repaired) or ``"field"`` (the
    original one-pass (event, lap) demeaning, kept only to reproduce old numbers).
    """
    from ..data.laps import tag_long_runs
    from ..data.schedule import SPRINT_SESSION
    from ..models.design import (
        classify_runs,
        drop_non_representative_laps,
        drop_stint_outliers,
        practice_design,
        race_design,
    )

    laps = pd.read_parquet(path)
    laps = laps[laps["Compound"].isin(LABELS)].dropna(subset=["TyreLife", "LapTimeSeconds"])

    # A sprint is a race, not practice: it sits on the racing side of the very
    # contrast this module tests, so pooling it into practice would contaminate it.
    prac = drop_non_representative_laps(laps[~laps["session"].isin(["R", SPRINT_SESSION])])
    prac = tag_long_runs(prac)
    prac = classify_runs(drop_stint_outliers(prac[prac["is_long_run"]]))
    prac = practice_design(prac[prac["is_race_sim"]])

    race = race_design(drop_stint_outliers(laps[laps["is_long_run"]]))
    if race_demean == "compound":
        race = demean_by_compound(race)
    elif race_demean != "field":
        raise ValueError(f"race_demean must be 'compound' or 'field', got {race_demean!r}")
    return prac, race


def analyse(
    per_season: dict[int, tuple[pd.DataFrame, pd.DataFrame]]
) -> ManagementResult:
    """Compare practice and race degradation across seasons.

    Args:
        per_season: {season: (practice_design_frame, race_design_frame)}.

    Returns:
        A ``ManagementResult``. The trend test is a one-sided Spearman
        correlation between compound softness and the race/practice ratio,
        one-sided because the direction was predicted in advance.
    """
    frames = []
    for season, (prac, race) in per_season.items():
        p = cell_slopes(prac).rename(columns={"rate": "practice", "se": "se_practice"})
        r = cell_slopes(race).rename(columns={"rate": "race", "se": "se_race"})
        if p.empty or r.empty:
            continue
        j = p.merge(r, on=["event", "Compound"], suffixes=("_p", "_r"))
        j["season"] = season
        frames.append(j)

    if not frames:
        raise ValueError("no season produced cells with both practice and race data")

    cells = pd.concat(frames, ignore_index=True)
    cells = cells[cells["practice"] > MIN_PRACTICE_RATE].copy()
    cells["ratio"] = cells["race"] / cells["practice"]
    cells["softness"] = cells["Compound"].map(LABEL_RANK)

    if len(cells) < 6:
        raise ValueError(f"only {len(cells)} usable cells; not enough to test a trend")

    # One-sided: the prediction is that softer means LOWER, so a negative
    # correlation. Halving a two-sided p is valid only because the direction was
    # fixed in advance.
    rho, p_two = spearmanr(cells["softness"], cells["ratio"])
    p_one = p_two / 2 if rho < 0 else 1 - p_two / 2

    # The omnibus test, for comparison. It asks only "are these three groups
    # different at all" and throws away the ordering, so it is less powerful
    # here. Reported so that switching to a trend test is an argument the reader
    # can check, not a quiet improvement to the p-value.
    groups = [cells[cells["Compound"] == lab]["ratio"].to_numpy() for lab in LABELS]
    groups = [g for g in groups if len(g) >= 3]
    p_kw = float(kruskal(*groups).pvalue) if len(groups) >= 2 else float("nan")

    return ManagementResult(
        cells=cells,
        median_ratio={
            lab: float(cells[cells["Compound"] == lab]["ratio"].median())
            for lab in LABELS
            if (cells["Compound"] == lab).sum() >= 3
        },
        n_cells=len(cells),
        n_events=cells["event"].nunique(),
        n_seasons=cells["season"].nunique(),
        rho=float(rho),
        p_value=float(p_one),
        p_two_sided=float(p_two),
        p_kruskal=p_kw,
    )

"""Typed shapes for the JSON artifacts the pipeline scripts write to ``data/artifacts/``.

Each ``scripts/0N_*.py`` builds one of these dataclasses and writes it through
``write()``, which rejects unknown artifact names so a typo cannot silently
produce a stray file. ``paper/CLAIMS.md`` and the manuscript asset builders read
the resulting JSON (see ``data/artifacts/README.md``).

Two modelling decisions are baked into the format.

1. Degradation is keyed on the PHYSICAL compound (C1-C5), not on the HARD /
   MEDIUM / SOFT label. Pirelli nominates three of C1-C5 per weekend and the
   labels are relative to that nomination, so a "HARD" at Monaco is softer
   rubber than a "SOFT" at Suzuka. Keying on the label pools different tyres.
   The label is carried alongside, because that is what people say out loud.

2. Every degradation result carries `context`: practice or race. They are not
   the same number. In races drivers manage soft tyres hard enough to flatten
   measured degradation; in practice, where they are measuring the tyre, the
   physical ordering appears. That difference is a finding, not noise, so the
   format has to be able to express both.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ..config import ARTIFACTS

SCHEMA_VERSION = "1.0.0"

Compound = Literal["C1", "C2", "C3", "C4", "C5"]
Label = Literal["HARD", "MEDIUM", "SOFT"]
Context = Literal["practice", "race"]

#: Artifact names ``write()`` accepts; each is ``data/artifacts/<name>.json``.
FILES = (
    "meta",
    "degradation",
    "ablation",
    "benchmark",
    "calibration",
    "power",
    "transfer",
    "strategy",
    "management",
)


@dataclass
class Interval:
    """A value with an uncertainty band. Seconds unless the field says otherwise."""

    mean: float
    lo: float  # 2.5th percentile by default
    hi: float  # 97.5th percentile by default

    def __post_init__(self) -> None:
        if not (self.lo <= self.mean <= self.hi):
            raise ValueError(f"interval not ordered: lo={self.lo} mean={self.mean} hi={self.hi}")

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def overlaps(self, other: Interval) -> bool:
        return not (self.hi < other.lo or other.hi < self.lo)


# ---------------------------------------------------------------------------
# meta.json
# ---------------------------------------------------------------------------


@dataclass
class Meta:
    """Provenance of an artifact set: when, from which data, with which versions."""

    generated_at: str
    schema_version: str
    model_version: str
    season: int
    events: list[str]
    n_laps_clean: int
    n_long_run_laps: int
    n_runs: int
    n_drivers: int
    fastf1_version: str
    #: False for placeholder data, True for real pipeline output.
    is_real: bool = False
    #: Every event in the laps file, including ones with no race laps yet.
    #: ``events`` above is the subset the race degradation fit actually used.
    events_in_dataset: list[str] = field(default_factory=list)
    #: Events present with practice or sprint laps only, no Grand Prix race.
    practice_only_events: list[str] = field(default_factory=list)
    #: ``{event: {session: n_laps}}`` for the whole file.
    laps_by_event_session: dict[str, dict[str, int]] = field(default_factory=dict)
    #: What each fit read: ``{fit: {sessions, events, n_events, n_laps}}``.
    fit_scope: dict[str, dict] = field(default_factory=dict)

    @staticmethod
    def now(**kw) -> Meta:
        return Meta(
            generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            schema_version=SCHEMA_VERSION,
            **kw,
        )


# ---------------------------------------------------------------------------
# degradation.json  -- the brief's primary deliverable
# ---------------------------------------------------------------------------


@dataclass
class CurvePoint:
    tyre_life: int
    #: Lap time lost relative to a fresh tyre, seconds.
    delta: Interval


@dataclass
class CompoundCurve:
    compound: Compound
    #: What this compound was called at this event. None when pooled.
    label: Label | None
    context: Context
    #: Degradation rate, seconds lost per lap. The headline number.
    rate: Interval
    curve: list[CurvePoint]
    n_laps: int
    n_runs: int
    #: Events contributing to this estimate. Length > 1 means pooled.
    events: list[str]


@dataclass
class DegradationArtifact:
    #: None when pooled across the season.
    event: str | None
    curves: list[CompoundCurve]
    #: Fitted fuel effect, s/kg. Checked against the 0.030-0.035 physical prior;
    #: the benchmark's model recovers only ~0.016 because its latent state
    #: absorbs the rest. None where the design absorbs fuel non-parametrically.
    fuel_coefficient: Interval | None
    #: Fitted track evolution, s per lap of session elapsed.
    track_evolution: Interval | None
    #: P(softer of the pair degrades faster). Keys like "C3>C4".
    #:
    #: NOT a posterior probability. The model is mixed-effects, so what we have
    #: are confidence intervals; this is a normal approximation to the
    #: difference between two fitted rates. It reads like a Bayesian quantity and
    #: is not one, which is worth saying rather than letting a reader assume.
    separation: dict[str, float] = field(default_factory=dict)
    #: Pairs whose 95% intervals do not overlap. Keys "C3|C4".
    separated_pairs: list[str] = field(default_factory=list)
    #: Optimiser evidence, variance components and inference notes for the fit
    #: that produced these curves. Free-form on purpose; see model.json.
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ablation.json  -- drives the Deconfound button
# ---------------------------------------------------------------------------


@dataclass
class AblationRow:
    compound: Compound
    label: Label | None
    #: Lap time regressed on stint lap alone. What the naive approach gives.
    naive: Interval
    #: Fuel, traffic and track evolution removed, field pooled.
    deconfounded: Interval
    #: The benchmark's published figure, where one exists for comparison.
    published: Interval | None = None
    #: The standard practitioner estimate: fuel-corrected lap time regressed on
    #: tyre age within each driver stint (driver x stint fixed effects), on the
    #: same laps and the same C-number grouping as ``deconfounded``.
    practitioner: Interval | None = None


@dataclass
class AblationArtifact:
    context: Context
    rows: list[AblationRow]
    #: One sentence the UI shows under the chart, written by the pipeline so the
    #: caption can never drift from the numbers it describes.
    caption: str = ""


# ---------------------------------------------------------------------------
# benchmark.json
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkScore:
    model: str
    rmspe: float | None
    crps: float | None
    #: published = from their paper; reproduced = we ran their code; ours = us.
    source: Literal["published", "reproduced", "ours"]


@dataclass
class RaceScore:
    race: str
    ours_crps: float
    theirs_crps: float
    ours_wins: bool


@dataclass
class BenchmarkArtifact:
    #: Austria 2025 under their exact CV scheme -- direct comparison to Table 1/2.
    austria_2025: list[BenchmarkScore]
    #: The 15 races of their 19-race season set that we both scored (their repo
    #: carries per-race CRPS, which never appeared in the paper). Lets us report
    #: a win rate rather than a single number: 2 of 15.
    season_2025: list[RaceScore] = field(default_factory=list)
    n_wins: int = 0
    n_races: int = 0
    #: Season-wide mean CRPS, ours and theirs, over different race sets (we
    #: scored 18, they 19), so indicative only. The paired, like-for-like
    #: comparison on their own retained test laps is in
    #: ``paper/results/benchmark.json`` (mean 0.326 against 0.220, 15 races).
    ours_season_crps: float | None = None
    theirs_season_crps: float | None = None
    #: Median race, carried next to the mean so the reader can see both the
    #: typical race and the damage a bad one does. (An earlier version of this
    #: comment blamed a 3.83 Singapore score on a 32-second green-flag lap. That
    #: was our scope error: their pipeline drops Singapore laps after 58 before
    #: cross-validating and our scorer had kept them. Folds now come from their
    #: retained laps and Singapore scores 0.43.)
    ours_season_crps_median: float | None = None
    #: Race counts behind each mean. They differ (they had 19, we scored 18),
    #: which is exactly why both are carried rather than just the difference.
    ours_season_races: int | None = None
    theirs_season_races: int | None = None
    #: Largest absolute gap between our CRPS and R's scoringRules on identical
    #: input. Backs the "same estimator" claim with a number instead of a word.
    r_crosscheck_max_diff: float | None = None


# ---------------------------------------------------------------------------
# calibration.json / power.json
# ---------------------------------------------------------------------------


@dataclass
class CalibrationPoint:
    nominal: float
    empirical: float
    n: int


@dataclass
class CalibrationArtifact:
    points: list[CalibrationPoint]
    coverage_80: float
    context: Context = "race"
    #: ``points``/``coverage_80`` above are LAP-LEVEL: does a held-out lap fall
    #: inside its predictive interval. That says little about the degradation
    #: RATES, which are the product. These are the rate-level check: does each
    #: circuit's own fitted rate fall in the pooled model's predictive interval
    #: for that circuit, leaving that circuit out of the fit.
    rate_points: list[CalibrationPoint] = field(default_factory=list)
    rate_coverage_80: float | None = None
    rate_n_cells: int | None = None
    note: str = ""


@dataclass
class PowerPoint:
    n_driver_stints: int
    power: float


@dataclass
class PowerArtifact:
    """How much data separating two compounds actually needs.

    Answers the open question the benchmark leaves, and tells us whether our own
    result is real or lucky.
    """

    effect_size: float
    points: list[PowerPoint]
    n_for_80pct: int
    #: What the benchmark had, for contrast. They had 3 stints.
    benchmark_n: int
    ours_n: int
    #: ``n_for_80pct`` is the interpolated 80% crossing, rounded. None of these
    #: is the old power-of-two grid point.
    n_for_80pct_se: float | None = None
    #: Power ceiling as stint count grows without bound, when compounds are not
    #: replicated across circuits: more laps cannot buy more circuits.
    power_ceiling_split_design: float | None = None
    #: Which simulation produced the points: design, effect, replicates.
    design: str = ""


# ---------------------------------------------------------------------------
# transfer.json  -- practice to race, what the brief asks for
# ---------------------------------------------------------------------------


@dataclass
class TransferRow:
    event: str
    compound: Compound
    label: Label | None
    predicted: Interval
    #: None when the race has not happened yet (the live forecast).
    actual: float | None
    abs_error: float | None


@dataclass
class TransferArtifact:
    rows: list[TransferRow]
    #: Mean absolute error, s/lap. None when every row is a forecast.
    mae: float | None
    #: True when this is a prediction for a race that has not run.
    is_forecast: bool = False


# ---------------------------------------------------------------------------
# strategy.json
# ---------------------------------------------------------------------------


@dataclass
class StrategyPlan:
    n_stops: int
    compounds: list[Compound]
    stint_lengths: list[int]
    #: Mean and 95% band of this plan's total race time under a Monte Carlo of
    #: the optimiser's uncertain inputs (rates, pit loss, pace step). It used to
    #: be the plug-in total +/- a hard-coded 3 s.
    total_time: Interval
    #: P(this is the cheapest plan of all legal plans) under the same draws.
    p_best: float | None = None
    #: Mean seconds this plan loses to the best plan IN THE SAME DRAW. The
    #: decision-relevant spread, since the input uncertainty that moves every
    #: plan together (the pace step, mostly) cancels out of it.
    regret_s: float | None = None


@dataclass
class StrategyArtifact:
    event: str
    pit_loss_s: float
    race_laps: int
    plans: list[StrategyPlan]
    recommended_stops: int
    rationale: str
    confidence: float
    #: Optimal stint length per compound given the pit loss. Keyed by C-number.
    optimal_stint: dict[str, Interval] = field(default_factory=dict)
    #: P(each stop count is the cheapest), keyed by stop count as a string.
    #: ``confidence`` is the entry for ``recommended_stops``.
    stop_probabilities: dict[str, float] = field(default_factory=dict)
    #: "race" or "practice": where the degradation rates came from.
    rates_source: str = "race"
    #: What the Monte Carlo propagated and how, so the band can be read.
    uncertainty: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# management.json -- why softer compounds do not appear to degrade faster
# ---------------------------------------------------------------------------


@dataclass
class ManagementRow:
    label: Label
    n_cells: int
    #: Race degradation divided by practice degradation, median across cells.
    #: Below 1 means the race number is suppressed relative to practice.
    ratio: float
    median_race: float
    median_practice: float


@dataclass
class ManagementArtifact:
    """Race degradation against practice degradation, by compound label.

    This artifact was built to show that drivers nurse softer tyres in races.
    That claim is WITHDRAWN (the original ratio trend rho -0.29, p 0.0035 came
    from a biased race estimator). What it reports now is the same-estimator
    ratio view, kept for continuity; ``verdict`` and ``caveat`` carry the honest
    reading. The analysis the paper uses is ``paper/results/management.json``.

    Keyed on the HARD/MEDIUM/SOFT label rather than the physical compound, and
    that is correct here even though it is wrong everywhere else: within one
    event the label is the same tyre in practice and in the race, which is all a
    ratio needs.
    """

    rows: list[ManagementRow]
    n_cells: int
    n_events: int
    n_seasons: int
    seasons: list[int]
    #: Spearman correlation between compound softness and the ratio.
    rho: float
    #: One-sided, because the direction was predicted before the data was pulled
    #: (a prediction the data no longer supports).
    p_one_sided: float
    #: Two-sided as well, so the reader need not accept the one-sided choice.
    p_two_sided: float
    #: The omnibus alternative, which ignores the ordering we predicted.
    p_kruskal: float
    ordered: bool
    #: True only when BOTH sidedness conventions clear 0.05 (currently False).
    significant: bool
    verdict: str
    #: How the effect moved as seasons were added. Shrinking with more data is
    #: worth showing rather than hiding.
    stability: list[dict] = field(default_factory=list)
    #: Distinct (season, event) pairs. ``n_events`` counts event NAMES, which
    #: repeat across seasons; the independent unit is the event-season. Additive.
    n_event_seasons: int = 0
    #: What this artifact does NOT establish. Additive; older readers ignore it.
    caveat: str = ""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write(name: str, artifact, out_dir: Path | None = None) -> Path:
    """Write one artifact to ``<out_dir>/<name>.json``.

    Rejects unknown names so a typo cannot silently produce a stray file.
    """
    if name not in FILES:
        raise ValueError(f"unknown artifact {name!r}; expected one of {FILES}")

    out_dir = out_dir or ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(asdict(artifact), indent=2), encoding="utf-8")
    return path

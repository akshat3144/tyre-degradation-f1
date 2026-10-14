"""Why softer compounds do not appear to degrade faster.

    python scripts/07_management.py

STATUS: the driver-nursing explanation below is NOT established. The ratio test
this script runs (originally rho -0.288, p 0.0035 on 101 cells) is withdrawn; on
a same-estimator comparison it reads rho +0.03, p 0.77. What survives is that
race degradation runs at about 38% of practice on matched MEDIUM and SOFT cells,
and that a telemetry test of the mechanism was null. This script now writes the
continuity view in ``data/artifacts/management.json``; the analysis to quote is
``paper/results/management.json``.

Our race fits keep saying softer compounds degrade no faster than hard ones,
which contradicts how tyres work. This tests the leading explanation: a driver
nurses a fragile tyre in a race and pushes it in practice, so the race number is
suppressed most for the softest tyre.

Uses every COMPLETE season on disk, because the answer is limited by sample size
and nothing else. Truncated pulls are refused rather than pooled -- see
season_completeness.

Also reports how the estimate moved as seasons were added. That trail belongs in
the output rather than in a footnote, since the alternative reading -- drop the
season that disagrees and report the better number -- was available at every step.

BUG FIXED (see validation.management.demean_by_compound). The race slope used to
be fitted through the origin on a design demeaned by (event, lap) across ALL
compounds, which leaks each compound's pace offset into its slope; it now uses a
design demeaned within (event, lap, compound). Note what that does and does not
buy: it removes a known bias, but the race slope is still CROSS-SECTIONAL (cars
of different tyre age at the same lap), whereas the practice slope is within-run,
so the two sides remain different estimators and the cross-sectional one is open
to selection (cars stay on old tyres because those tyres are holding up). A
same-estimator difference analysis, with placebo, temperature and multiverse
checks, is in paper/results/management.json (scripts/paper/p1_management.py) and
is the one to quote.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import kruskal, spearmanr

from tyredeg.artifacts import schema
from tyredeg.artifacts.schema import ManagementArtifact, ManagementRow
from tyredeg.config import PROCESSED
from tyredeg.data.cache import season_completeness
from tyredeg.validation.management import LABELS, analyse, build_design_frames

warnings.filterwarnings("ignore")

#: Written into the artifact so a reader sees the caveat with the numbers. The ratio analysis below
#: is the LEGACY, per-cell view; the paper's analysis is
#: scripts/paper/p1_management.py -> paper/results/management.json.
CAVEAT = (
    "Race and practice slopes here come from the SAME estimator (within-run, fuel-corrected, "
    "tyre ages 6-16), so the numbers are comparable, but the comparison is observational. "
    "Race stints and practice runs start on different set states (practice runs mostly start on "
    "used sets), practice sessions differ from races in fuel and traffic, and a practice-versus-"
    "practice placebo shows a compound gradient of similar size to the race-versus-practice one. "
    "The ratio view shown here is a poor estimand (a ratio of two noisy slopes) and is kept only "
    "for continuity; the paper's analysis, with matched set state, a two-way fixed-effects compound "
    "contrast, placebos and a power calculation, is in paper/results/management.json. "
    "The earlier claim that drivers nurse softer tyres harder (rho -0.29, p 0.0035, 101 cells) "
    "rested on a race estimator that leaked each compound's pace offset into its slope and is withdrawn."
)

MIN_PRACTICE_RATE = 0.005


def same_estimator_cells():
    """Cells from the paper analysis, plus the sentence the artifact is allowed to say."""
    import json

    from tyredeg.config import ROOT
    from tyredeg.paper import management as PM

    cells = PM.make_cells(PM.Bases.load())
    a = PM.Arrays.from_cells(cells)
    pooled = PM.pooled_level(a, (1, 2))
    ratio = pooled["ratio"]
    text = (
        f"Race-observed degradation runs at roughly {ratio:.0%} of the practice-observed rate "
        f"(MEDIUM and SOFT pooled, same estimator in both contexts). "
        "Compound-specific nursing is NOT established: the race-minus-practice gap does not "
        "grow with softness reliably, and practice-versus-practice shows a comparable gradient."
    )
    res = ROOT / "paper" / "results" / "management.json"
    if res.exists():
        r = json.loads(res.read_text(encoding="utf-8"))
        post = r.get("post_review_additions", {})
        ref = r.get("referee_additions", {}).get("designs", {}).get("used_sets_only_FreshTyre_false", {})
        # Preferred: set state matched through the feed's FreshTyre flag (both contexts on used
        # sets). The start-age proxy mostly selects later blocks of NEW-set race stints.
        lg = ref.get("level_gap", {}).get("MEDIUM+SOFT") or post.get("level_gap_matched_pooled", {}).get("MEDIUM+SOFT")
        tg = post.get("twfe_gradient", {}).get("SOFT-HARD")
        if lg:
            text = (
                f"Race-observed degradation runs at roughly {lg['ratio']:.0%} of the practice-observed "
                f"rate on set-state-matched, same-estimator comparisons (MEDIUM and SOFT pooled; gap "
                f"{lg['gap']:+.3f} s/lap, 95% CI {lg['gap_ci95'][0]:+.3f} to {lg['gap_ci95'][1]:+.3f}). "
                "Compound-specific nursing is NOT established: "
            )
            if tg:
                text += (
                    f"the SOFT-minus-HARD gradient in that gap is {tg['estimate_s_per_lap']:+.3f} s/lap "
                    f"({tg['ci95'][0]:+.3f} to {tg['ci95'][1]:+.3f}) but SOFT versus MEDIUM shows none, "
                )
            text += (
                "a uniform damping of every compound by the same factor would itself produce a gradient of "
                "similar size (softer tyres have larger practice slopes), and a practice-versus-practice "
                "placebo shows one too."
            )
    return cells, {"verdict": text, "ratio": ratio}


def build(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Practice and race design frames for one season."""
    return build_design_frames(path)


def stability_table(per_season: dict) -> list[dict]:
    """How the effect moves as seasons are added, newest first.

    rho need not be monotone in sample size, and the path is printed because a
    reader who sees only the final row cannot tell whether the claim survived the
    evidence or was fitted to it.
    """
    rows = []
    newest_first = sorted(per_season, reverse=True)
    for k in range(1, len(newest_first) + 1):
        subset = {s: per_season[s] for s in newest_first[:k]}
        try:
            r = analyse(subset)
        except ValueError:
            continue
        rows.append(
            {
                "seasons": sorted(subset),
                "n_cells": r.n_cells,
                "rho": round(r.rho, 4),
                "p_one_sided": round(r.p_value, 4),
                "p_two_sided": round(r.p_two_sided, 4),
                "ordered": r.ordered,
            }
        )
    return rows


def main() -> None:
    sources = {2026: PROCESSED / "laps.parquet"}
    for season in (2025, 2024, 2023, 2022):
        path = PROCESSED / f"laps_{season}.parquet"
        if path.exists():
            sources[season] = path

    per_season, excluded = {}, []
    for season, path in sorted(sources.items(), reverse=True):
        ok, why = season_completeness(path)
        if not ok:
            excluded.append((season, why))
            print(f"  {season}: SKIPPED -- {why}")
            continue
        per_season[season] = build(path)
        print(f"  {season}: loaded {path.name} -- {why}")

    if excluded:
        print(
            f"\n  {len(excluded)} season(s) excluded as incomplete. Re-run "
            f"scripts/01_cache_sessions.py --season <year> once the API cap resets."
        )

    res = analyse(per_season)

    print()
    print("=" * 70)
    print(
        f"{res.n_cells} (event, compound) cells across {res.n_events} events, "
        f"{res.n_seasons} seasons"
    )
    print("=" * 70)
    print()
    header = f"{'label':9s}{'n':>4s}{'race/practice':>16s}{'median race':>14s}{'median practice':>18s}"
    print(header)
    for lab in LABELS:
        sub = res.cells[res.cells["Compound"] == lab]
        if len(sub) < 3:
            continue
        print(
            f"{lab:9s}{len(sub):4d}{sub['ratio'].median():16.3f}"
            f"{sub['race'].median():14.4f}{sub['practice'].median():18.4f}"
        )

    print()
    print(f"  ratio falls as the tyre softens   : {res.ordered}")
    print(f"  Spearman rho                      : {res.rho:+.3f}")
    print(f"  one-sided p (direction predicted) : {res.p_value:.4f}")
    print(f"  two-sided p (if you prefer)       : {res.p_two_sided:.4f}")
    print(f"  Kruskal-Wallis, ignoring ordering : {res.p_kruskal:.4f}")
    print()
    print(f"  LEGACY view (cross-sectional race slope; not what the artifact records): {res.verdict()}")

    stability = stability_table(per_season)
    if len(stability) > 1:
        print()
        print("  HOW IT MOVED AS SEASONS WERE ADDED")
        # 26 wide, not 22: five seasons joined by "+" is 24 characters, which
        # overflowed the column and pushed every later field out of line.
        print(f"  {'seasons':26s}{'cells':>7s}{'rho':>9s}{'p 1-sided':>12s}{'p 2-sided':>12s}")
        for row in stability:
            tag = "+".join(str(s) for s in row["seasons"])
            print(
                f"  {tag:26s}{row['n_cells']:7d}{row['rho']:9.3f}"
                f"{row['p_one_sided']:12.4f}{row['p_two_sided']:12.4f}"
            )

    # ------------------------------------------------------------------
    # WHAT THE ARTIFACT RECORDS
    #
    # Not the legacy ratio above. That race slope is cross-sectional and was
    # compared with a within-run practice slope, which is two different
    # estimators, and the cross-sectional one is open to selection (its SOFT race
    # slope comes out near zero and often negative). The artifact is built from
    # the SAME-ESTIMATOR cells of the paper analysis (identical within-run
    # fuel-corrected slope in both contexts, tyre-age window 6-16), and its
    # verdict states only what that analysis can bear.
    # ------------------------------------------------------------------
    cells, level = same_estimator_cells()
    rows = []
    for lab in LABELS:
        sub = cells[cells["Compound"] == lab]
        if len(sub) < 3:
            continue
        use = sub[sub["slope_p"] > MIN_PRACTICE_RATE]
        rows.append(
            ManagementRow(
                label=lab,
                n_cells=int(len(sub)),
                ratio=round(float((use["slope_r"] / use["slope_p"]).median()), 4),
                median_race=round(float(sub["slope_r"].median()), 5),
                median_practice=round(float(sub["slope_p"].median()), 5),
            )
        )
    use = cells[cells["slope_p"] > MIN_PRACTICE_RATE].copy()
    use["ratio"] = use["slope_r"] / use["slope_p"]
    rho, p_two = spearmanr(use["rank"], use["ratio"])
    p_one = p_two / 2 if rho < 0 else 1 - p_two / 2
    groups = [use[use["rank"] == r]["ratio"].to_numpy() for r in (0, 1, 2)]
    p_kw = float(kruskal(*[g for g in groups if len(g) >= 3]).pvalue)
    med = [r.ratio for r in rows]
    ordered = all(med[i] > med[i + 1] for i in range(len(med) - 1))
    significant = bool(ordered and p_one < 0.05 and p_two < 0.05)
    stab = []
    seasons_all = sorted(cells["season"].unique())
    for k in range(1, len(seasons_all) + 1):
        keep = seasons_all[::-1][:k]
        sub = use[use["season"].isin(keep)]
        if len(sub) < 6:
            continue
        r_, p2_ = spearmanr(sub["rank"], sub["ratio"])
        stab.append(
            {
                "seasons": sorted(int(x) for x in keep),
                "n_cells": int(len(sub)),
                "rho": round(float(r_), 4),
                "p_one_sided": round(float(p2_ / 2 if r_ < 0 else 1 - p2_ / 2), 4),
                "p_two_sided": round(float(p2_), 4),
                "ordered": bool(r_ < 0),
            }
        )

    print()
    print("  ARTIFACT (same-estimator cells)")
    print(f"  {len(cells)} cells, {cells['cluster'].nunique()} event-seasons; rho {rho:+.3f}, p two-sided {p_two:.3f}")
    print(f"  {level['verdict']}")

    schema.write(
        "management",
        ManagementArtifact(
            rows=rows,
            n_cells=int(len(cells)),
            n_events=int(cells["event"].nunique()),
            n_seasons=int(cells["season"].nunique()),
            seasons=[int(x) for x in seasons_all],
            rho=round(float(rho), 4),
            p_one_sided=round(float(p_one), 4),
            p_two_sided=round(float(p_two), 4),
            p_kruskal=round(p_kw, 4),
            ordered=bool(ordered),
            significant=significant,
            verdict=level["verdict"],
            stability=stab,
            n_event_seasons=int(cells["cluster"].nunique()),
            caveat=CAVEAT,
        ),
    )
    print("wrote management.json")


if __name__ == "__main__":
    main()

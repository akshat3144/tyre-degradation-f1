"""Our per-race CRPS against the PUBLISHED numbers of Cappello & Hoegh (2026).

The journal version of the benchmark paper (J. Sports Analytics 12, doi
10.1177/22150218261446170) prints a per-race table (Table 4, journal p. 10): RMSPE and
CRPS for their skewed-t state-space model (SSM) and for an AR(1) with stint resets,
19 races, one driver (Hamilton), each race fitted on its own. It is transcribed in
``benchmark/published_table4.json``.

What this module adds over the older comparison against their repository CSV:

* the SSM column of Table 4 equals the repo CSV to three decimals at every one of the
  19 races (``csv_vs_journal``), so the head-to-head against the SSM does not change;
* Table 4 also carries the AR(1) benchmark per race, which the repo CSV does not, so
  we can say where we stand against the journal's own baseline;
* RMSPE is comparable too (race RMSPE = mean over stints of the stint RMSPE, which is
  what Table 2 and Table 4 are consistent with: Austria 1.082 / 3 stints = 0.361).

It is pure arithmetic on already-computed scores: no model is fitted here. Different
race sets are handled explicitly, never silently: ``coverage`` lists the journal races
we cannot score and why.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from tyredeg.config import BENCHMARK_SEASON_2025_PER_RACE, ROOT

TABLE4_PATH = ROOT / "benchmark" / "published_table4.json"
BOOT_B = 20_000
BOOT_SEED = 0
#: The caption of the journal's Table 4 says the "Mean Across Races" row excludes
#: Singapore (whose last four laps the authors removed because of a brake failure).
SINGAPORE = "Singapore Grand Prix"
NOT_IN_OUR_FEED_NOTE = (
    "not in our cached 2025 race feed (data/processed/laps_2025.parquet), so we cannot score them"
)


def load_table4(path: Path = TABLE4_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _paired(diff: np.ndarray, wins: int) -> dict:
    """Race-clustered paired summary of (ours - theirs): one observation per race."""
    n = len(diff)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, (BOOT_B, n))
    bm = diff[idx].mean(1)
    try:
        w = float(stats.wilcoxon(diff).pvalue)
    except ValueError:
        w = float("nan")
    return {
        "n_races": int(n),
        "ours_wins": int(wins),
        "mean_diff_ours_minus_theirs": float(diff.mean()),
        "mean_diff_ci95": [float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))],
        "median_diff": float(np.median(diff)),
        "sign_test_p_two_sided": float(stats.binomtest(wins, n, 0.5).pvalue),
        "wilcoxon_p_two_sided": w,
        "bootstrap": {"B": BOOT_B, "seed": BOOT_SEED, "unit": "race"},
    }


def _summary(rows: list[dict], ours_key: str, theirs_key: str) -> dict:
    ours = np.array([r[ours_key] for r in rows])
    th = np.array([r[theirs_key] for r in rows])
    out = _paired(ours - th, int((ours < th).sum()))
    out.update(ours_mean=float(ours.mean()), theirs_mean=float(th.mean()),
               ours_median=float(np.median(ours)), theirs_median=float(np.median(th)))
    return out


def csv_vs_journal(table4: dict | None = None) -> dict:
    """Is the journal's SSM CRPS column the same data as the repo CSV we used before?"""
    t = table4 or load_table4()
    diffs = {r: abs(round(csv, 3) - t["races"][r]["crps_ssm"])
             for r, (csv, _) in BENCHMARK_SEASON_2025_PER_RACE.items() if r in t["races"]}
    stints_equal = {r: s == t["races"][r]["stints"]
                    for r, (_, s) in BENCHMARK_SEASON_2025_PER_RACE.items() if r in t["races"]}
    return {
        "n_races": len(diffs),
        "max_abs_diff_at_3dp": float(max(diffs.values())),
        "all_equal_at_3dp": bool(max(diffs.values()) < 1e-9),
        "stint_counts_equal_everywhere": bool(all(stints_equal.values())),
        "note": "repo CSV (All_CV_results1.csv, transcribed in tyredeg.config) vs journal Table 4 SSM CRPS",
    }


def table4_internal_checks(table4: dict | None = None) -> dict:
    """What Table 4's own rows imply, against what its 'Mean Across Races' row prints."""
    t = table4 or load_table4()
    races = t["races"]
    pm = t["published_means"]

    def mean(key: str, drop_sg: bool) -> float:
        return float(np.mean([v[key] for r, v in races.items() if not (drop_sg and r == SINGAPORE)]))

    keys = ("rmspe_ssm", "rmspe_ar1", "crps_ssm", "crps_ar1")
    out: dict = {"n_races": len(races), "caption_says_mean_excludes": SINGAPORE}
    for key in keys:
        out[key] = {
            "printed": pm[key],
            "mean_all_19": mean(key, False),
            "mean_excl_singapore_18": mean(key, True),
        }
    out["printed_mean_matches"] = {
        k: (
            "all_19" if round(out[k]["mean_all_19"], 3) == pm[k]
            else "excl_singapore" if round(out[k]["mean_excl_singapore_18"], 3) == pm[k]
            else "neither"
        )
        for k in keys
    }
    out["ssm_beats_ar1"] = {
        "rmspe": int(sum(v["rmspe_ssm"] < v["rmspe_ar1"] for v in races.values())),
        "crps": int(sum(v["crps_ssm"] < v["crps_ar1"] for v in races.values())),
        "of": len(races),
        "printed": t["published_counts"],
    }
    return out


def compare_to_journal(ours: dict, table4: dict | None = None) -> dict:
    """Per-race comparison of our score with the journal's Table 4.

    ``ours`` is the dict ``p4_benchmark.our_scores`` builds: race -> {"their_folds":
    {"crps", "rmse_total", "per_stint_crps"}, "own_folds": {"crps"}}. The primary
    comparison scores us on THEIR test laps (``their_folds``); our own folds from our
    own feed are the secondary one (they differ by a lap or two and, at Singapore, by
    the laps their pipeline drops).
    """
    t = table4 or load_table4()
    pub = t["races"]
    rows = []
    for race, o in ours.items():
        if race not in pub:
            continue
        p = pub[race]
        tf = o["their_folds"]
        n_st = len(tf["per_stint_crps"]) if tf.get("per_stint_crps") else None
        rows.append({
            "race": race,
            "journal_stints": p["stints"],
            "our_stints": n_st,
            "ours_crps": float(tf["crps"]),
            "ours_rmspe": float(tf["rmse_total"]) / n_st if n_st else None,
            "journal_crps_ssm": p["crps_ssm"],
            "journal_crps_ar1": p["crps_ar1"],
            "journal_rmspe_ssm": p["rmspe_ssm"],
            "journal_rmspe_ar1": p["rmspe_ar1"],
            "ours_own_folds_crps": float(o["own_folds"]["crps"]),
        })
    rows.sort(key=lambda r: r["race"])
    covered = {r["race"] for r in rows}
    missing = sorted(set(pub) - covered)

    def block(rs: list[dict]) -> dict:
        rr = [r for r in rs if r["ours_rmspe"] is not None]
        return {
            "n_races": len(rs),
            "crps_vs_ssm": _summary(rs, "ours_crps", "journal_crps_ssm"),
            "crps_vs_ar1": _summary(rs, "ours_crps", "journal_crps_ar1"),
            "rmspe_vs_ssm": _summary(rr, "ours_rmspe", "journal_rmspe_ssm"),
            "rmspe_vs_ar1": _summary(rr, "ours_rmspe", "journal_rmspe_ar1"),
            "own_folds_crps_vs_ssm": _summary(rs, "ours_own_folds_crps", "journal_crps_ssm"),
            "own_folds_crps_vs_ar1": _summary(rs, "ours_own_folds_crps", "journal_crps_ar1"),
            "journal_ssm_beats_ar1_on_these_races": {
                "crps": int(sum(r["journal_crps_ssm"] < r["journal_crps_ar1"] for r in rs)),
                "rmspe": int(sum(r["journal_rmspe_ssm"] < r["journal_rmspe_ar1"] for r in rs)),
                "of": len(rs),
            },
        }

    ex_sg = [r for r in rows if r["race"] != SINGAPORE]
    return {
        "source": {k: t["source"][k] for k in ("citation", "doi", "licence", "table", "printed_page")},
        "task": "one-step-ahead lap-time forecast of Hamilton's laps under their rolling-origin CV; "
        "journal numbers are Cappello & Hoegh's per-race fits, ours is the field-pooled forecaster",
        "coverage": {
            "journal_races": len(pub),
            "ours_scored": len(rows),
            "journal_races_we_did_not_score": missing,
            "stint_counts_equal_journal": int(sum(r["our_stints"] == r["journal_stints"] for r in rows)),
            "races_where_stint_counts_differ": [
                r["race"] for r in rows if r["our_stints"] != r["journal_stints"]
            ],
            "why": NOT_IN_OUR_FEED_NOTE,
            "journal_excludes": [
                "Australian (wet)", "British (wet)", "Belgian (incomplete timing)",
                "Miami (incomplete timing)", "Dutch (DNF)", "last four Singapore laps",
            ],
            "we_score_but_journal_excludes": sorted(set(ours) - set(pub)),
            "note": "Race sets differ, so means over the journal's 19 are not like-for-like; "
            "all summaries here are over the races both cover.",
        },
        "csv_vs_journal": csv_vs_journal(t),
        "table4_internal_checks": table4_internal_checks(t),
        "per_race": rows,
        "common_races": block(rows),
        "common_races_excl_singapore": block(ex_sg),
    }

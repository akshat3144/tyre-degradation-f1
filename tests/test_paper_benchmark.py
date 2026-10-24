"""Tests for the benchmark audit (``scripts/paper/p4_benchmark.py``).

Fast and offline. Anything that needs their repo (``benchmark/upstream``, fetched
and never committed) skips when it is absent; the maths does not.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import integrate

from tyredeg.config import ROOT
from tyredeg.paper import benchmark_repro as br
from tyredeg.validation.scoring import crosscheck_python

needs_upstream = pytest.mark.skipif(
    not br.upstream_available(), reason="run scripts/fetch_reference.sh to fetch their repo"
)
RESULTS = ROOT / "paper" / "results" / "benchmark.json"


def _race(stint_lengths, first_lap=4, gap=2):
    """A synthetic RaceData with contiguous stints separated by ``gap`` pit laps."""
    laps, pit, stint = [], [], []
    lap = first_lap
    for s, n in enumerate(stint_lengths, start=1):
        for i in range(n):
            laps.append(lap)
            lap += 1
            stint.append(s)
            pit.append(1 if i == n - 1 and s < len(stint_lengths) else 0)
        lap += gap
    n = len(laps)
    return br.RaceData(
        race="synthetic", y=np.full(n, 70.0), compound=np.ones(n, int), pit=np.array(pit),
        lap_number=np.array(laps), stint=np.array(stint), fuel_theirs=np.linspace(110, 1, n),
        fuel_true=br.true_fuel(np.array(laps), 70), race_laps=70,
    )


# --- their cross-validation scheme -------------------------------------------------


def test_cv_folds_austria_shape_is_17():
    """Stints of 22, 22 and 19 retained laps: round(5.5) + round(5.5) + round(4.75)
    = 6 + 6 + 5. R and NumPy both round the half to even, so 5.5 gives 6."""
    folds = br.cv_folds(_race([22, 22, 19]))
    assert [sum(f.stint == s for f in folds) for s in (1, 2, 3)] == [6, 6, 5]
    assert all(f.test_row == f.train_n + 1 for f in folds)


def test_cv_folds_train_strictly_before_the_test_row_and_in_stint():
    d = _race([22, 22, 19])
    for f in br.cv_folds(d):
        assert d.stint[f.test_row - 1] == f.stint
        assert f.train_n >= 1 and f.lap_number == int(d.lap_number[f.test_row - 1])


def test_true_fuel_runs_from_110_to_1_over_the_distance():
    f = br.true_fuel(np.array([1, 70]), 70)
    assert f[0] == pytest.approx(110.0) and f[1] == pytest.approx(1.0)


def test_their_ramp_is_misplaced_when_laps_are_dropped():
    """Dropping laps and re-spreading the ramp misplaces fuel by whole kilograms;
    with no dropped laps the two agree."""
    d = _race([22, 22, 19])
    m = br.fuel_misplacement(d)
    assert m["max_abs_kg"] > 1.0
    full = br.RaceData(**{**d.__dict__, "lap_number": np.arange(1, d.n + 1),
                          "fuel_true": br.true_fuel(np.arange(1, d.n + 1), d.n)})
    assert br.fuel_misplacement(full)["max_abs_kg"] < 1e-9


# --- the skew-t: density and sampler agree (R's rsgt is not available) --------------


@pytest.mark.parametrize("lam", [-0.4, 0.0, 0.5])
def test_skew_t_density_is_proper_with_the_stated_mean_and_sd(lam):
    mu, sigma = 3.0, 0.7
    pdf = lambda x: float(np.exp(br.sgt_logpdf(x, mu, sigma, lam, 2.0)))  # noqa: E731
    one = integrate.quad(pdf, -80, 80, points=[mu], limit=400)[0]
    mean = integrate.quad(lambda x: x * pdf(x), -80, 80, points=[mu], limit=400)[0]
    var = integrate.quad(lambda x: (x - mean) ** 2 * pdf(x), -80, 80, points=[mu], limit=400)[0]
    assert one == pytest.approx(1.0, abs=1e-5)
    assert mean == pytest.approx(mu, abs=1e-4)  # mean-centred
    assert np.sqrt(var) == pytest.approx(sigma, rel=1e-3)  # variance-adjusted


def test_rsgt_sampler_matches_its_own_density():
    """Kolmogorov-Smirnov of rsgt draws against the numerically integrated CDF."""
    from scipy import stats

    mu, sigma, lam = 1.0, 0.5, 0.5
    rng = np.random.default_rng(0)
    x = br.rsgt(40_000, mu, sigma, lam, 2.0, rng)
    grid = np.linspace(-30, 30, 120_001)
    pdf = np.exp(br.sgt_logpdf(grid, mu, sigma, lam, 2.0))
    cdf = np.cumsum(pdf) * (grid[1] - grid[0])
    ks = stats.kstest(x, lambda v: np.interp(v, grid, cdf))
    assert ks.pvalue > 0.01
    assert x.mean() == pytest.approx(mu, abs=0.02)


def test_rsgt_is_skewed_the_way_lambda_says():
    rng = np.random.default_rng(1)
    pos = br.rsgt(50_000, 0.0, 1.0, 0.5, 2.0, rng)
    assert np.median(pos) < -0.1  # right-skewed with mean 0: the median sits below it


def test_seed_of_is_deterministic():
    assert br.seed_of("Austrian Grand Prix", "shipped", 1, 2) == br.seed_of("Austrian Grand Prix", "shipped", 1, 2)
    assert br.seed_of("a", 1) != br.seed_of("a", 2)


# --- CRPS, checked without R ---------------------------------------------------------


def test_crps_python_crosscheck():
    """Replaces the skip-when-R-is-missing check with ones that always run."""
    c = crosscheck_python()
    assert c["anchor"] < 1e-12  # (sqrt 2 - 1)/sqrt(pi) for N(0,1) at 0
    assert c["normal_vs_integral"] < 1e-7
    assert c["t_vs_integral"] < 1e-7
    assert c["ensemble_vs_energy_form"] < 1e-12  # R's crps_sample default form
    assert c["ensemble_vs_closed_form_mean"] < 2e-3  # Monte-Carlo noise only


# --- their data, when present ----------------------------------------------------------


@needs_upstream
def test_austria_data_matches_the_paper_and_the_fuel_defect_is_4_7_kg():
    d = br.load_race_data("Austrian Grand Prix")
    assert d.n == 63
    assert [int((d.stint == s).sum()) for s in (1, 2, 3)] == [22, 22, 19]
    m = br.fuel_misplacement(br.load_race_data("Austrian Grand Prix", race_laps=70))
    assert m["max_abs_kg"] == pytest.approx(4.739, abs=0.01)
    assert len(br.cv_folds(d)) == 17


@needs_upstream
def test_both_of_their_data_paths_agree_on_hamilton_austria():
    """Table 3 uses Austria_laps.csv with LapTime < 100; the CV uses the TrackStatus
    filter. They must pick the same 63 laps or the reproduction mixes two datasets."""
    import pandas as pd

    b = pd.read_csv(br.UPSTREAM / "Paper_Generation_Files" / "Austria_laps.csv")
    b = b[(b["Driver"] == "HAM") & b["PitOutTime"].isna() & b["PitInTime"].isna() & (b["LapTime"] < 100)]
    d = br.load_race_data("Austrian Grand Prix", race_laps=70)
    assert sorted(b["LapNumber"].astype(int)) == sorted(d.lap_number.tolist())


# --- the written results ---------------------------------------------------------------


@pytest.mark.skipif(not RESULTS.exists(), reason="paper/results/benchmark.json not generated yet")
def test_results_json_is_internally_consistent():
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    t3 = r["table3"]
    # Their Table 3 reproduces to within MCMC noise for a parameter with a 0.1-wide interval.
    assert t3["reproduces"]["hard_mean_abs_diff"] < 0.01
    assert t3["reproduces"]["medium_mean_abs_diff"] < 0.01
    # Fixing the fuel indexing does not separate the compounds: the honest negative.
    assert t3["corrected_fuel"]["crls_overlap"] is True
    assert abs(t3["fuel_correction_effect"]["delta_v_hard"]) < 0.01
    fd = r["fuel_defect"]["austria"]
    assert fd["max_abs_kg"] == pytest.approx(4.739, abs=0.01)
    # We never claim a win we did not measure.
    h = r["head_to_head"]["season_vs_published"]["B_their_folds_vs_published"]
    assert h["ours_wins"] == sum(x["ours"] < x["theirs"] for x in h["per_race"])
    assert r["meta"]["seeds"]["cv_folds"] == 1234
    assert r["crps_crosscheck_python"]["ensemble_vs_energy_form"] < 1e-12


# --- the journal version's Table 4 (Cappello & Hoegh 2026) ------------------------------

from tyredeg.config import BENCHMARK_SEASON_2025_PER_RACE  # noqa: E402
from tyredeg.paper import published_benchmark as pb  # noqa: E402


def test_table4_transcription_is_internally_consistent():
    """Guards the hand transcription against the journal's own printed summaries."""
    t = pb.load_table4()
    races = t["races"]
    assert len(races) == 19 and "Singapore Grand Prix" in races
    # Table 1: 51 stints in all; p. 8: SSM better in 15 of 19 on RMSPE and 16 of 19 on CRPS.
    assert sum(v["stints"] for v in races.values()) == 51
    chk = pb.table4_internal_checks(t)
    assert (chk["ssm_beats_ar1"]["rmspe"], chk["ssm_beats_ar1"]["crps"]) == (15, 16)
    # The Austrian row repeats Table 3 (CRPS 0.202 / 0.289) and Table 2 (RMSPE totals / 3 stints).
    a = races["Austrian Grand Prix"]
    assert (a["crps_ssm"], a["crps_ar1"]) == (0.202, 0.289)
    assert a["rmspe_ssm"] == pytest.approx(1.082 / 3, abs=1e-3)
    assert a["rmspe_ar1"] == pytest.approx(1.478 / 3, abs=1e-3)
    # Printed 'Mean Across Races' row equals the 19-race mean, whatever the caption says.
    for k, v in chk["printed_mean_matches"].items():
        assert v == "all_19", k


def test_journal_ssm_column_is_the_repo_csv():
    """So the older head-to-head against the repo CSV is the head-to-head against the journal's SSM."""
    c = pb.csv_vs_journal()
    assert c["n_races"] == len(BENCHMARK_SEASON_2025_PER_RACE) == 19
    assert c["all_equal_at_3dp"] and c["stint_counts_equal_everywhere"]


def _fake_ours(scale: float, races=("Austrian Grand Prix", "Italian Grand Prix", "Singapore Grand Prix")):
    t = pb.load_table4()["races"]
    out = {}
    for r in races:
        c = t[r]["crps_ssm"] * scale
        st = t[r]["stints"]
        out[r] = {
            "their_folds": {"crps": c, "rmse_total": t[r]["rmspe_ssm"] * st * scale,
                            "per_stint_crps": [c] * st},
            "own_folds": {"crps": c},
        }
    return out


def test_compare_to_journal_arithmetic_and_coverage():
    res = pb.compare_to_journal(_fake_ours(1.0))
    cm = res["common_races"]
    # Scoring exactly the journal's SSM numbers: zero difference against SSM, ours wins no race.
    assert cm["n_races"] == 3
    assert cm["crps_vs_ssm"]["mean_diff_ours_minus_theirs"] == pytest.approx(0.0, abs=1e-12)
    assert cm["crps_vs_ssm"]["ours_wins"] == 0
    assert cm["rmspe_vs_ssm"]["mean_diff_ours_minus_theirs"] == pytest.approx(0.0, abs=1e-12)
    # The journal's SSM beats its AR(1) at Austria and Italy, loses at Singapore: ours inherits that.
    assert cm["crps_vs_ar1"]["ours_wins"] == 2
    assert cm["journal_ssm_beats_ar1_on_these_races"]["crps"] == 2
    # Coverage is explicit, never silent.
    cov = res["coverage"]
    assert cov["ours_scored"] == 3 and cov["journal_races"] == 19
    assert "Qatar Grand Prix" in cov["journal_races_we_did_not_score"]
    assert res["common_races_excl_singapore"]["n_races"] == 2


def test_compare_to_journal_ignores_races_the_journal_excludes():
    ours = _fake_ours(1.0, races=("Austrian Grand Prix",))
    ours["Dutch Grand Prix"] = ours["Austrian Grand Prix"]  # a race the journal excludes (DNF)
    res = pb.compare_to_journal(ours)
    assert res["common_races"]["n_races"] == 1
    assert res["coverage"]["we_score_but_journal_excludes"] == ["Dutch Grand Prix"]


@pytest.mark.skipif(not RESULTS.exists(), reason="paper/results/benchmark.json not generated yet")
def test_results_json_journal_section_matches_its_own_rows():
    r = json.loads(RESULTS.read_text(encoding="utf-8"))["journal_table4"]
    rows = r["per_race"]
    cm = r["common_races"]
    assert cm["n_races"] == len(rows) == r["coverage"]["ours_scored"]
    assert cm["crps_vs_ssm"]["ours_wins"] == sum(x["ours_crps"] < x["journal_crps_ssm"] for x in rows)
    assert cm["crps_vs_ar1"]["ours_wins"] == sum(x["ours_crps"] < x["journal_crps_ar1"] for x in rows)
    # Same numbers as the repo-CSV comparison (the journal's SSM column is that CSV, rounded).
    old = json.loads(RESULTS.read_text(encoding="utf-8"))["head_to_head"]["season_vs_published"][
        "B_their_folds_vs_published"]
    assert old["ours_wins"] == cm["crps_vs_ssm"]["ours_wins"]
    assert old["ours_mean"] == pytest.approx(cm["crps_vs_ssm"]["ours_mean"])
    assert old["theirs_mean"] == pytest.approx(cm["crps_vs_ssm"]["theirs_mean"], abs=1e-3)

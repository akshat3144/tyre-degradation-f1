"""Tests for the management-hypothesis analysis.

The delicate part is that this is the one place we key on the HARD/MEDIUM/SOFT
label rather than the physical compound. That is legitimate for a ratio and
wrong for anything else, so the tests pin the reasoning as well as the numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tyredeg.validation.management import (
    LABEL_RANK,
    LABELS,
    analyse,
    cell_slopes,
    cluster_slope,
)


def runs(event, label, rate, n_runs=5, laps=14, noise=0.1, seed=0):
    """Runs at a known degradation rate, in the centred design's own columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_runs):
        age = np.arange(1, laps + 1, dtype=float) + r * 3
        y = rate * age + rng.normal(0, noise, laps)
        rows.append(
            pd.DataFrame(
                {
                    "event": event,
                    "Compound": label,
                    "run_id": f"{event}|{label}|{r}",
                    "tl": age - age.mean(),
                    "y": y - y.mean(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def paired(rate_practice, rate_race, seed=0):
    """One season of practice and race frames with matched cells."""
    prac, race = [], []
    for i, event in enumerate([f"E{j} GP" for j in range(6)]):
        for label in LABELS:
            prac.append(runs(event, label, rate_practice[label], seed=seed + i))
            race.append(runs(event, label, rate_race[label], seed=seed + 100 + i))
    return pd.concat(prac, ignore_index=True), pd.concat(race, ignore_index=True)


def test_labels_rank_hard_to_soft():
    assert LABEL_RANK["HARD"] < LABEL_RANK["MEDIUM"] < LABEL_RANK["SOFT"]


def test_cluster_slope_recovers_a_known_rate():
    got = cluster_slope(runs("E GP", "MEDIUM", 0.08, noise=0.05))
    assert got is not None
    assert got[0] == pytest.approx(0.08, abs=0.01)


def test_cluster_slope_errors_exceed_unclustered_ones():
    """Laps inside a run are correlated, so the clustered error must be the
    larger of the two. If it were not, clustering would be pointless."""
    g = runs("E GP", "MEDIUM", 0.08, n_runs=6, laps=16, noise=0.3)
    _, se_clustered = cluster_slope(g)

    x, y = g["tl"].to_numpy(), g["y"].to_numpy()
    slope = float(x @ y / (x @ x))
    resid = y - slope * x
    se_naive = float(np.sqrt((resid @ resid) / (len(y) - 1) / (x @ x)))

    assert se_clustered > se_naive


def test_cluster_slope_refuses_a_single_run():
    g = runs("E GP", "MEDIUM", 0.08, n_runs=1)
    _, se = cluster_slope(g)
    assert np.isnan(se)


def test_cell_slopes_skips_thin_cells():
    thin = runs("E GP", "SOFT", 0.09, n_runs=2)
    assert cell_slopes(thin, min_runs=3).empty


def test_detects_the_predicted_ordering():
    """Race degradation suppressed most for the softest tyre."""
    prac = {"HARD": 0.10, "MEDIUM": 0.12, "SOFT": 0.15}
    race = {"HARD": 0.08, "MEDIUM": 0.06, "SOFT": 0.01}
    res = analyse({2026: paired(prac, race)})
    assert res.ordered
    assert res.rho < 0
    assert res.significant
    assert "supported" in res.verdict()


def test_reports_no_effect_when_there_is_none():
    """If every compound is suppressed equally there is no trend, and the test
    must say so rather than finding one."""
    prac = {"HARD": 0.10, "MEDIUM": 0.12, "SOFT": 0.15}
    race = {"HARD": 0.05, "MEDIUM": 0.06, "SOFT": 0.075}  # all exactly half
    res = analyse({2026: paired(prac, race, seed=7)})
    assert not res.significant


def test_reports_the_reverse_when_the_effect_reverses():
    prac = {"HARD": 0.10, "MEDIUM": 0.12, "SOFT": 0.15}
    race = {"HARD": 0.02, "MEDIUM": 0.09, "SOFT": 0.15}  # softs suppressed LEAST
    res = analyse({2026: paired(prac, race, seed=11)})
    assert not res.ordered
    assert "not supported" in res.verdict()


def test_all_three_p_values_are_reported():
    """The trend test is more powerful than the omnibus one because it uses the
    ordering we predicted. Reporting both keeps that an argument the reader can
    check rather than a quiet improvement to the number."""
    prac = {"HARD": 0.10, "MEDIUM": 0.12, "SOFT": 0.15}
    race = {"HARD": 0.08, "MEDIUM": 0.06, "SOFT": 0.01}
    res = analyse({2026: paired(prac, race)})
    assert res.p_value <= res.p_two_sided
    assert not np.isnan(res.p_kruskal)


def test_pools_across_seasons():
    prac = {"HARD": 0.10, "MEDIUM": 0.12, "SOFT": 0.15}
    race = {"HARD": 0.08, "MEDIUM": 0.06, "SOFT": 0.01}
    res = analyse({2026: paired(prac, race), 2025: paired(prac, race, seed=50)})
    assert res.n_seasons == 2
    assert res.n_cells > 18


def test_refuses_a_near_zero_practice_denominator():
    """A ratio needs a real degradation rate underneath it, or it explodes."""
    prac = {"HARD": 0.0001, "MEDIUM": 0.0001, "SOFT": 0.0001}
    race = {"HARD": 0.08, "MEDIUM": 0.06, "SOFT": 0.01}
    with pytest.raises(ValueError, match="not enough"):
        analyse({2026: paired(prac, race)})

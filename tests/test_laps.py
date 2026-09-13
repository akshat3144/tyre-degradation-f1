"""Tests for the clean-lap filters and long-run detection.

Built on a synthetic frame so they run without network. The FastF1 pull is
tested separately and marked ``network``.
"""

import numpy as np
import pandas as pd
import pytest

from tyredeg.data.laps import clean_laps, summarise, tag_long_runs


def make_laps(n=20, driver="VER", stint=1, compound="MEDIUM", status="1"):
    return pd.DataFrame(
        {
            "Driver": driver,
            "Team": "Team",
            "LapNumber": np.arange(1, n + 1, dtype=float),
            "LapTime": pd.to_timedelta(np.full(n, 80.0), unit="s"),
            "Stint": float(stint),
            "Compound": compound,
            "TyreLife": np.arange(1, n + 1, dtype=float),
            "FreshTyre": True,
            "TrackStatus": status,
            "IsAccurate": True,
            "LapStartTime": pd.to_timedelta(np.arange(n) * 80.0, unit="s"),
            # Proper timedelta dtype, not bare NaT, so assigning a Timedelta
            # later does not trigger a dtype-incompatibility warning.
            "PitInTime": pd.Series([pd.NaT] * n, dtype="timedelta64[ns]"),
            "PitOutTime": pd.Series([pd.NaT] * n, dtype="timedelta64[ns]"),
            "Sector1Time": pd.to_timedelta(np.full(n, 25.0), unit="s"),
            "Sector2Time": pd.to_timedelta(np.full(n, 30.0), unit="s"),
            "Sector3Time": pd.to_timedelta(np.full(n, 25.0), unit="s"),
        }
    )


def test_clean_laps_keeps_good_laps_and_converts_seconds():
    out = clean_laps(make_laps(10), event="Test GP", session="FP2")
    assert len(out) == 10
    assert out.LapTimeSeconds.iloc[0] == pytest.approx(80.0)
    assert out.S1.iloc[0] == pytest.approx(25.0)
    assert set(out.event) == {"Test GP"}


def test_missing_lap_time_is_dropped():
    df = make_laps(5)
    df.loc[2, "LapTime"] = pd.NaT
    assert len(clean_laps(df, event="T", session="FP2")) == 4


def test_pit_laps_are_dropped():
    df = make_laps(5)
    df.loc[0, "PitOutTime"] = pd.Timedelta("1s")
    df.loc[4, "PitInTime"] = pd.Timedelta("1s")
    assert len(clean_laps(df, event="T", session="FP2")) == 3


def test_inaccurate_laps_are_dropped():
    df = make_laps(5)
    df.loc[1, "IsAccurate"] = False
    assert len(clean_laps(df, event="T", session="FP2")) == 4


def test_strict_green_rejects_any_non_green_status():
    df = make_laps(5)
    df.loc[1, "TrackStatus"] = "2"    # yellow
    df.loc[3, "TrackStatus"] = "125"  # concatenated codes, includes safety car
    strict = clean_laps(df, event="T", session="FP2", strict_green=True)
    assert len(strict) == 3


def test_benchmark_style_filter_keeps_yellow_but_drops_safety_car():
    """Their filter is a substring match on 4|5|6|7, so plain yellow survives."""
    df = make_laps(5)
    df.loc[1, "TrackStatus"] = "2"    # yellow: kept by their rule
    df.loc[3, "TrackStatus"] = "125"  # contains "5": dropped
    loose = clean_laps(df, event="T", session="FP2", strict_green=False)
    assert len(loose) == 4


def test_long_run_detection():
    out = tag_long_runs(clean_laps(make_laps(8), event="T", session="FP2"), min_laps=5)
    assert out.is_long_run.all()
    assert out.run_len.iloc[0] == 8
    assert out.run_lap.tolist() == list(range(1, 9))


def test_a_gap_in_lap_numbers_splits_the_run():
    """Filtering can punch a hole mid-stint. Laps either side are not consecutive."""
    df = make_laps(10)
    df.loc[4, "TrackStatus"] = "4"  # safety car removes lap 5
    clean = clean_laps(df, event="T", session="FP2")
    out = tag_long_runs(clean, min_laps=5)

    assert out.run_id.nunique() == 2
    # 4 laps before the hole, 5 after: only the second block is a long run.
    assert sorted(out.groupby("run_id").size()) == [4, 5]
    assert out.is_long_run.sum() == 5


def test_short_runs_are_not_flagged():
    out = tag_long_runs(clean_laps(make_laps(3), event="T", session="FP2"), min_laps=5)
    assert not out.is_long_run.any()


def test_runs_do_not_merge_across_drivers_or_stints():
    a = make_laps(6, driver="VER", stint=1)
    b = make_laps(6, driver="NOR", stint=1)
    c = make_laps(6, driver="VER", stint=2, compound="HARD")
    df = pd.concat([a, b, c], ignore_index=True)
    out = tag_long_runs(clean_laps(df, event="T", session="FP2"), min_laps=5)
    assert out.run_id.nunique() == 3


def test_summarise_reports_per_session_counts():
    df = clean_laps(make_laps(8), event="Test GP", session="FP2")
    s = summarise(tag_long_runs(df))
    assert len(s) == 1
    row = s.iloc[0]
    assert row.laps == 8
    assert row.long_runs == 1
    assert "MEDIUM" in row.compounds


def test_empty_input_does_not_raise():
    empty = clean_laps(make_laps(1).iloc[0:0], event="T", session="FP2")
    assert tag_long_runs(empty).empty
    assert summarise(empty).empty


def test_run_id_is_unique_across_sessions_and_events():
    """run_id used to be a per-call category code, so id 0 appeared in seven
    different events once frames were concatenated. It must be a full key."""
    frames = []
    for event in ("Race A", "Race B"):
        for session in ("FP1", "FP2"):
            frames.append(clean_laps(make_laps(6), event=event, session=session))
    out = tag_long_runs(pd.concat(frames, ignore_index=True), min_laps=5)

    assert out.run_id.nunique() == 4
    per_run = out.groupby("run_id")[["event", "session"]].nunique()
    assert (per_run == 1).all().all(), "a run_id spans more than one event/session"


def test_tagging_all_at_once_matches_tagging_separately():
    a = tag_long_runs(clean_laps(make_laps(6, driver="VER"), event="E", session="FP1"))
    b = tag_long_runs(clean_laps(make_laps(6, driver="NOR"), event="E", session="FP1"))
    both = tag_long_runs(
        pd.concat(
            [
                clean_laps(make_laps(6, driver="VER"), event="E", session="FP1"),
                clean_laps(make_laps(6, driver="NOR"), event="E", session="FP1"),
            ],
            ignore_index=True,
        )
    )
    assert both.run_len.tolist() == a.run_len.tolist() + b.run_len.tolist()
    assert both.is_long_run.sum() == a.is_long_run.sum() + b.is_long_run.sum()


def test_clean_laps_returns_a_plain_dataframe():
    """fastf1.core.Laps subclasses DataFrame and declares _metadata=['session'].
    On a Laps object, `.session` returns the Session and `.event` returns None,
    silently shadowing our columns of the same name. clean_laps must hand back
    a plain DataFrame so attribute access is safe downstream."""
    import fastf1.core

    fake = fastf1.core.Laps(make_laps(6))
    out = clean_laps(fake, event="Test GP", session="FP2")

    assert type(out) is pd.DataFrame
    assert out.event.iloc[0] == "Test GP"      # attribute access, the risky form
    assert out.session.iloc[0] == "FP2"
    assert out["event"].iloc[0] == "Test GP"


def test_summarise_works_on_a_laps_subclass():
    """Regression: summarise silently reported zero long runs for every session
    because it used `df.event`, which was None on a Laps-derived frame."""
    import fastf1.core

    df = clean_laps(fastf1.core.Laps(make_laps(8)), event="Test GP", session="FP2")
    s = summarise(tag_long_runs(df))
    assert s.iloc[0].long_runs == 1
    assert s.iloc[0].long_run_laps == 8

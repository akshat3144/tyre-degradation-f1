"""Integrity tests for the 2022-2025 Pirelli nomination history.

The values are research output, not computed, so the tests guard the things
that can silently rot: join keys that no longer match the lap data, an inverted
HARD/MEDIUM/SOFT row, and a "verified" row that cites nothing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tyredeg.config import PROCESSED
from tyredeg.data import allocation_history as ah

LABELS = ("HARD", "MEDIUM", "SOFT")
YEARS = (2022, 2023, 2024, 2025)

# Events deliberately absent from the history, with the reason. Empty today: the
# only gap in the period (2023 Emilia Romagna, cancelled) never reached the laps.
DOCUMENTED_GAPS: dict[tuple[int, str], str] = {}


def _rows():
    h = ah.load_history()
    for season, events in h["seasons"].items():
        for event, row in events.items():
            yield int(season), event, row


def test_file_loads_with_note_and_all_seasons():
    h = ah.load_history()
    assert h["source_note"]
    assert set(h["seasons"]) == {str(y) for y in YEARS}


@pytest.mark.parametrize("year", YEARS)
def test_every_lap_event_has_an_entry(year):
    path = PROCESSED / f"laps_{year}.parquet"
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    events = pd.read_parquet(path, columns=["event"]).event.unique()
    missing = [
        e for e in events
        if e not in ah.load_history()["seasons"][str(year)]
        and (year, e) not in DOCUMENTED_GAPS
    ]
    assert not missing, f"{year}: no nomination history for {missing}"


def test_hard_harder_than_medium_harder_than_soft():
    bad = []
    for season, event, row in _rows():
        s = [ah.softness(row[lab]) for lab in LABELS]
        if not (s[0] < s[1] < s[2]):
            bad.append((season, event, [row[lab] for lab in LABELS]))
    assert not bad


def test_verified_rows_have_sources_and_unverified_rows_explain_themselves():
    for season, event, row in _rows():
        assert all(u.startswith("http") for u in row["sources"]), (season, event)
        if row["verified"]:
            assert len(row["sources"]) >= 1, (season, event)
        else:
            assert row.get("notes"), f"{season} {event} unverified without a note"


def test_range_limits_by_season():
    for season, event, row in _rows():
        top = max(ah.softness(row[lab]) for lab in LABELS)
        bottom = min(ah.softness(row[lab]) for lab in LABELS)
        assert bottom >= 0
        # C6 only exists from 2025; nothing before may nominate it.
        assert top <= (6 if season >= 2025 else 5), (season, event)


def test_c_number_lookup_and_misses():
    assert ah.c_number(2024, "Bahrain Grand Prix", "SOFT") == "C3"
    assert ah.c_number("2024", "Monaco Grand Prix", "soft") == "C5"
    assert ah.c_number(2025, "Emilia Romagna Grand Prix", "SOFT") == "C6"
    assert ah.c_number(2023, "Emilia Romagna Grand Prix", "SOFT") is None  # cancelled
    assert ah.c_number(2024, "Bahrain Grand Prix", "INTERMEDIATE") is None
    assert ah.c_number(1999, "Bahrain Grand Prix", "SOFT") is None


def test_labels_are_relative_the_same_label_is_different_rubber():
    assert ah.c_number(2024, "Bahrain Grand Prix", "SOFT") != ah.c_number(
        2024, "Monaco Grand Prix", "SOFT"
    )


def test_softness():
    assert [ah.softness(f"C{i}") for i in range(7)] == list(range(7))
    with pytest.raises(ValueError):
        ah.softness("SOFT")

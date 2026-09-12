"""Tests for the season-completeness guard.

This exists because of a real incident, not a hypothetical one. The F1 API caps
at 500 calls an hour. Our 2022 pull hit the cap twice and landed 9 events, then
17, then all 19 -- and at each stage the parquet on disk looked exactly like a
finished season. The 17-event version was one import away from being pooled into
a five-season significance test with two circuits silently missing.

So the guard has to fail closed on a truncated pull, and it has to keep working
for the seasons cached before manifests existed.
"""

from __future__ import annotations

import json

from tyredeg.data.cache import season_completeness


def _write(tmp_path, name, manifest):
    parquet = tmp_path / f"{name}.parquet"
    parquet.write_bytes(b"")
    if manifest is not None:
        man = tmp_path / f"{name}.manifest.json"
        man.write_text(json.dumps(manifest), encoding="utf-8")
    return parquet


def test_a_complete_manifest_passes(tmp_path):
    p = _write(tmp_path, "laps_2022", {
        "complete": True,
        "sessions_loaded": 76,
        "sessions_failed": 0,
        "events_requested": ["A", "B"],
        "events_loaded": ["A", "B"],
    })
    ok, why = season_completeness(p)
    assert ok
    assert "complete" in why
    assert "76" in why


def test_a_truncated_pull_is_refused_and_names_what_is_missing(tmp_path):
    """The exact shape of our second 2022 attempt: 17 of 19 events."""
    p = _write(tmp_path, "laps_2022", {
        "complete": False,
        "sessions_loaded": 66,
        "sessions_failed": 10,
        "events_requested": ["Abu Dhabi Grand Prix", "Mexico City Grand Prix", "Spanish Grand Prix"],
        "events_loaded": ["Spanish Grand Prix"],
    })
    ok, why = season_completeness(p)
    assert not ok
    assert "INCOMPLETE" in why
    # Naming them matters: "2 events missing" does not tell you that the gaps
    # are a high-altitude circuit and a low-degradation one.
    assert "Abu Dhabi Grand Prix" in why
    assert "Mexico City Grand Prix" in why
    assert "Spanish Grand Prix" not in why


def test_a_missing_manifest_passes_but_says_so(tmp_path):
    """Fails OPEN, deliberately.

    Seasons cached before manifests existed have no manifest. Refusing them
    would shrink the sample without telling anyone, which is the same failure
    the guard exists to prevent, pointing the other way. So it passes and the
    reason string carries the caveat to the reader.
    """
    p = _write(tmp_path, "laps_2019", None)
    ok, why = season_completeness(p)
    assert ok
    assert "unverified" in why


def test_a_manifest_with_no_complete_key_is_refused(tmp_path):
    """Absent means not proven, not fine.

    A manifest that exists but omits the flag is a manifest we did not write,
    or one from a future format change. Treating a missing flag as passing
    would make the guard silently useless the next time the format moves.
    """
    p = _write(tmp_path, "laps_2022", {"season": 2022})
    ok, why = season_completeness(p)
    assert not ok
    assert "INCOMPLETE" in why


def test_the_failure_string_survives_an_empty_event_list(tmp_path):
    """A cap hit before the first event loads leaves nothing to diff."""
    p = _write(tmp_path, "laps_2022", {
        "complete": False,
        "sessions_loaded": 0,
        "sessions_failed": 76,
        "events_requested": [],
        "events_loaded": [],
    })
    ok, why = season_completeness(p)
    assert not ok
    assert "76 session(s) failed" in why

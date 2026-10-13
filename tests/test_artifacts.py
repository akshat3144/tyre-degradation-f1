"""The artifact format.

``tyredeg.artifacts.schema`` defines the typed shapes the pipeline scripts write
to ``data/artifacts/``. These tests pin the parts the scripts rely on: interval
ordering, the closed list of artifact names, and that a written artifact is valid
JSON with the schema version stamped on it.
"""

from __future__ import annotations

import json

import pytest

from tyredeg.artifacts import schema
from tyredeg.artifacts.schema import Interval

# --- Interval ---------------------------------------------------------------


def test_interval_rejects_unordered_bounds():
    with pytest.raises(ValueError, match="not ordered"):
        Interval(mean=0.05, lo=0.06, hi=0.07)


def test_interval_width_and_overlap():
    a = Interval(0.05, 0.04, 0.06)
    b = Interval(0.08, 0.07, 0.09)
    c = Interval(0.055, 0.045, 0.065)
    assert a.width == pytest.approx(0.02)
    assert not a.overlaps(b)
    assert a.overlaps(c)


# --- writing ----------------------------------------------------------------


def _meta() -> schema.Meta:
    return schema.Meta.now(
        model_version="test", season=2026, events=["Hungarian Grand Prix"],
        n_laps_clean=10, n_long_run_laps=8, n_runs=2, n_drivers=2, fastf1_version="0",
    )


def test_write_rejects_an_unknown_name(tmp_path):
    with pytest.raises(ValueError, match="unknown artifact"):
        schema.write("metta", _meta(), tmp_path)  # typo on purpose


def test_write_produces_valid_json_with_the_schema_version(tmp_path):
    path = schema.write("meta", _meta(), tmp_path)
    assert path == tmp_path / "meta.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == schema.SCHEMA_VERSION
    assert doc["is_real"] is False  # real pipeline output sets this explicitly


def test_every_written_artifact_name_is_listed_once():
    assert len(set(schema.FILES)) == len(schema.FILES)

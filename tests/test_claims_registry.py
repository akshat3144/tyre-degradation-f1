"""paper/CLAIMS.md quotes numbers from the JSON results; they must still match.

Fast and offline: reads only files committed to the repository. If a result file
is regenerated and a quoted number moves, this fails and CLAIMS.md needs
updating (or the change needs explaining).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "paper" / "check_claims.py"

pytestmark = pytest.mark.skipif(not (ROOT / "paper" / "CLAIMS.md").exists(), reason="no CLAIMS.md")


def _load():
    spec = importlib.util.spec_from_file_location("check_claims", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_quoted_number_matches_its_json_source():
    mod = _load()
    failures = mod.run()
    assert not failures, "\n".join(failures)


def test_the_registry_has_all_four_sections_and_many_checks():
    mod = _load()
    text = (ROOT / "paper" / "CLAIMS.md").read_text(encoding="utf-8")
    for heading in ("## A.", "## B.", "## C.", "## D."):
        assert heading in text
    assert len(mod.parse(text)) > 100


def test_the_checker_catches_a_wrong_number(tmp_path):
    mod = _load()
    (tmp_path / "x.json").write_text(json.dumps({"a": {"b": [1.23456]}}), encoding="utf-8")
    good = tmp_path / "good.md"
    good.write_text('```claims\n{"file": "x.json", "path": ["a", "b", 0], "value": 1.235, "places": 3}\n```', encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text('```claims\n{"file": "x.json", "path": ["a", "b", 0], "value": 1.3, "places": 3}\n```', encoding="utf-8")
    assert mod.run(good, tmp_path) == []
    assert len(mod.run(bad, tmp_path)) == 1


def test_pending_claims_are_skipped_until_the_key_exists_then_checked_and_promoted(tmp_path):
    mod = _load()
    doc = tmp_path / "CLAIMS.md"
    doc.write_text(
        '```claims\n{"file": "x.json", "path": ["a"], "value": 1.0, "places": 1}\n```\n'
        "```claims-pending\n"
        '{"file": "x.json", "path": ["later", "n"], "places": 2}\n'
        '{"file": "x.json", "path": ["later", "none"], "equals": null}\n'
        "```\n",
        encoding="utf-8",
    )
    x = tmp_path / "x.json"
    x.write_text(json.dumps({"a": 1.0}), encoding="utf-8")
    assert mod.run(doc, tmp_path) == []  # key absent: skipped, not a failure
    x.write_text(json.dumps({"a": 1.0, "later": {"n": 3.14159, "none": None}}), encoding="utf-8")
    fails = mod.run(doc, tmp_path)  # key present but no value yet: must be promoted
    assert len(fails) == 1 and "promote" in fails[0]
    assert mod.promote_pending(doc, tmp_path) == 2
    assert mod.run(doc, tmp_path) == []
    assert '"value": 3.14' in doc.read_text(encoding="utf-8")
    x.write_text(json.dumps({"a": 1.0, "later": {"n": 9.0, "none": None}}), encoding="utf-8")
    assert len(mod.run(doc, tmp_path)) == 1  # promoted lines are now ordinary checks


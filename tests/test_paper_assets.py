"""The manuscript's number / table / figure layer (paper/manuscript/generated, figures).

Fast and offline. It checks that
  * every number in CLAIMS.md section A has a macro in numbers.tex, and the macro
    text equals the claim at the claim's own rounding;
  * the committed numbers.tex and tab_*.tex are what a fresh build writes (so
    prose cannot drift from the JSON without this failing);
  * the tables are compile-safe (booktabs, no vertical rules, no bare % _ # &);
  * the figures exist, are vector PDFs, and regenerate;
  * a tiny LaTeX document that inputs numbers.tex and every table compiles
    (skipped when pdflatex is not installed).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "paper"
MANUSCRIPT = ROOT / "paper" / "manuscript"
GENERATED = MANUSCRIPT / "generated"
FIGURES = MANUSCRIPT / "figures"

pytestmark = pytest.mark.skipif(not (ROOT / "paper" / "CLAIMS.md").exists(), reason="no CLAIMS.md")

sys.path.insert(0, str(SCRIPTS))
import make_figures  # noqa: E402
import make_numbers  # noqa: E402
import make_tables  # noqa: E402
import _paper_common  # noqa: E402
from _paper_common import MODEL, get, has, section_a_claims  # noqa: E402

FIGURE_NAMES = ["fig_sign_error", "fig_circuit_decomposition", "fig_circuit_rates", "fig_race_practice", "fig_power", "fig_forecast"]
DEF = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")
MINUS, PLUS = "\\ensuremath{-}", "\\ensuremath{+}"


def _macro_defs() -> dict[str, str]:
    out = {}
    for line in (GENERATED / "numbers.tex").read_text(encoding="utf-8").splitlines():
        if line.startswith("%") or not line.strip():
            continue
        m = DEF.match(line)
        assert m, f"not a one-line newcommand: {line!r}"
        assert m.group(1) not in out, f"macro {m.group(1)} defined twice"
        out[m.group(1)] = m.group(2)
    return out


def _value(text: str) -> float:
    """Numeric value of a macro body such as an ensuremath-minus followed by 0.0114, or 11{,}039."""
    return float(text.replace(MINUS, "-").replace(PLUS, "+").replace("{,}", ""))


# ----------------------------------------------------------------- numbers
def test_numbers_tex_is_current_and_well_formed():
    macros = make_numbers.build()
    assert len(macros) > 250
    defs = _macro_defs()
    assert set(defs) == {m.name for m in macros}
    assert (GENERATED / "numbers.tex").read_text(encoding="utf-8") == make_numbers.render(macros), (
        "numbers.tex is stale: run python scripts/paper/make_numbers.py"
    )


def test_every_claims_section_a_number_has_a_macro_that_matches():
    defs = _macro_defs()
    by_key = {(m.file, m.path): m for m in make_numbers.build() if not m.absolute}
    claims = [c for c in section_a_claims() if "value" in c]
    assert len(claims) > 200
    problems = []
    for c in claims:
        key = (c["file"], tuple(c["path"]))
        m = by_key.get(key)
        if m is None:
            problems.append(f"no macro for {key}")
            continue
        if m.placeholder:  # a pending key that model.json does not have yet
            assert c.get("_pending") and not has(*key), f"{m.name} is a placeholder but its key exists"
            continue
        places = int(c.get("places", 3))
        got = _value(defs[m.name])
        if round(got, places) != round(float(c["value"]), places):
            problems.append(f"{m.name} = {defs[m.name]} but CLAIMS.md quotes {c['value']} (places={places})")
        if m.places != places:
            problems.append(f"{m.name} rounds to {m.places} places, CLAIMS.md uses {places}")
    assert not problems, "\n".join(problems)


def test_macros_match_the_json_not_just_the_claims():
    for m in make_numbers.build():
        if m.placeholder:
            continue
        want = round(get(m.file, *m.path), m.places)
        assert abs(_value(m.text) - (abs(want) if m.absolute else want)) < 1e-9, m.name


def test_headline_macros_exist_with_the_agreed_names():
    defs = _macro_defs()
    for name in (
        "RaceToPracticeRatio",
        "RaceToPracticeRatioLo",
        "RaceToPracticeRatioHi",
        "LevelGapMS",
        "LevelGapMSLo",
        "LevelGapMSHi",
        "StartAgeGapMS",
        "IntersectGapMS",
        "PlaceboGapMS",
        "DidTWFESoftHardHolmP",
        "CircuitSDsPerTwenty",
        "CircuitSD",
        "PersistSD",
        "EventSD",
        "PooledCompoundSD",
        "LevelGapMSAbs",
        "RateCOne",
        "MaeScalar",
        "MDERealCOneCTwo",
        "MDESimRealCOneCTwo",
        "NRunsCOneCTwo",
        "PXRunsRealSix",
        "PXRunsBalTwenty",
        "PXSizeBal",
        "PitLossGreen",
        "JournalDiffAR",
    ):
        assert name in defs, name
    assert defs["RaceToPracticeRatio"] == "0.56"  # primary design: used sets in both contexts


# ----------------------------------------------------------------- power_extended placeholders
PX_FIXTURE = ROOT / "tests" / "fixtures" / "power_extended_structure.json"


def _with_power_extended(monkeypatch):
    """model.json plus a power_extended block with the real structure (quick-mode numbers: structure only)."""
    fixture = json.loads(PX_FIXTURE.read_text(encoding="utf-8"))
    real = _paper_common.load

    def fake(rel):
        doc = real(rel)
        return {**doc, "power_extended": fixture} if rel == MODEL else doc

    monkeypatch.setattr(_paper_common, "load", fake)


def test_placeholders_exist_only_while_power_extended_is_missing():
    """The flagged ?? macros are for a missing key only; with the key present none may be committed."""
    macros = make_numbers.build()
    ph = make_numbers.placeholders(macros)
    committed = (GENERATED / "numbers.tex").read_text(encoding="utf-8")
    if has(MODEL, "power_extended"):
        assert not ph, f"power_extended is present but macros are placeholders: {ph[:3]}"
        assert "??" not in committed, "numbers.tex still holds placeholders: rerun make_numbers.py"
    else:
        assert ph and all(n.startswith("PX") for n in ph)
        assert committed.count(make_numbers.PLACEHOLDER_TEXT) == len(ph)
        assert "WARNING" in committed


def test_no_placeholders_when_power_extended_is_present(monkeypatch):
    """Against a real-structure power_extended every PX macro is a number; a placeholder would fail here."""
    _with_power_extended(monkeypatch)
    macros = make_numbers.build()
    assert not make_numbers.placeholders(macros)
    assert "??" not in make_numbers.render(macros)
    px = [m for m in macros if m.name.startswith("PX")]
    assert len(px) >= 25
    for m in px:
        assert re.fullmatch(r"(\\ensuremath\{[-+]\})?[0-9][0-9{},.]*", m.text), (m.name, m.text)


def test_power_figure_draws_panel_a_from_the_real_structure(monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    _with_power_extended(monkeypatch)
    monkeypatch.setattr(make_figures, "FIGURES", tmp_path)
    assert make_figures.power_extended_ready()
    with plt.rc_context(make_figures.RC):
        assert make_figures.fig_power().exists()


def test_dry_run_requires_an_output_root():
    """PAPER_MODEL_JSON without PAPER_OUT_ROOT must refuse: dry-run numbers may not overwrite committed assets."""
    env = {k: v for k, v in os.environ.items() if k != "PAPER_OUT_ROOT"}
    env["PAPER_MODEL_JSON"] = str(ROOT / "paper" / "results" / "model.json")
    res = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(SCRIPTS)!r}); import _paper_common"],
        capture_output=True, text=True, env=env,
    )
    assert res.returncode != 0 and "PAPER_OUT_ROOT" in (res.stderr + res.stdout)


# ----------------------------------------------------------------- tables
def test_committed_tables_are_current():
    fresh = make_tables.build_all()
    assert set(fresh) == {p.stem for p in GENERATED.glob("tab_*.tex")}
    for name, text in fresh.items():
        assert (GENERATED / f"{name}.tex").read_text(encoding="utf-8") == text, f"{name}.tex is stale"


@pytest.mark.parametrize("path", sorted(GENERATED.glob("tab_*.tex")), ids=lambda p: p.stem)
def test_tables_are_compile_safe_booktabs(path):
    text = path.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    head = "\\begin{tabular}{"
    assert lines[0].startswith(head) and lines[-1] == "\\end{tabular}"
    for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert rule in text
    for bad in ("\\hline", "\\vline", "\\cline"):
        assert bad not in text
    colspec = lines[0][len(head) : -1]
    assert "|" not in colspec, "vertical rule in column spec"
    assert not re.search(r"(?<!\\)%", text), "unescaped %"
    assert not re.search(r"(?<!\\)#", text), "unescaped #"
    text_mode = re.sub(r"\$[^$]*\$", "", text)  # a subscript is legal inside $...$
    assert not re.search(r"(?<!\\)_", text_mode), "unescaped underscore in text mode"
    assert text.count("{") == text.count("}")
    assert not re.search(r"\b(nan|inf)\b", text, re.I), "nan/inf in a table"
    ncol = len(re.findall(r"[lrc]", colspec.replace("@{}", "")))
    for ln in lines:
        if ln.startswith("\\") or not ln.endswith("\\\\"):
            continue
        assert len(re.findall(r"(?<!\\)&", ln)) == ncol - 1, f"wrong column count: {ln[:70]}"


# ----------------------------------------------------------------- figures
@pytest.mark.parametrize("name", FIGURE_NAMES)
def test_figures_exist_and_are_vector_pdfs(name):
    p = FIGURES / f"{name}.pdf"
    assert p.exists() and p.stat().st_size > 5_000
    data = p.read_bytes()
    assert data.startswith(b"%PDF")
    assert b"/Subtype /Image" not in data and b"/Subtype/Image" not in data, "raster content in a vector figure"


def test_figures_regenerate_from_json(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(make_figures, "FIGURES", tmp_path)
    with plt.rc_context(make_figures.RC):
        for fn in make_figures.FIGS.values():
            assert fn().exists()
    assert sorted(p.stem for p in tmp_path.glob("*.pdf")) == sorted(FIGURE_NAMES)


# ----------------------------------------------------------------- LaTeX smoke
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not installed")
def test_smoke_document_compiles(tmp_path):
    work = tmp_path / "ms"
    shutil.copytree(MANUSCRIPT, work, ignore=shutil.ignore_patterns("*.aux", "*.log"))
    res = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "smoke_assets.tex"],
        cwd=work,
        capture_output=True,
        text=True,
        timeout=240,
    )
    logf = work / "smoke_assets.log"
    log = logf.read_text(encoding="utf-8", errors="replace") if logf.exists() else res.stdout
    assert res.returncode == 0 and (work / "smoke_assets.pdf").exists(), log[-2000:]
    assert "Undefined control sequence" not in log

"""Shared helpers for the manuscript number/table/figure layer.

Everything here is offline and reads only committed JSON (paper/results,
data/artifacts, benchmark/published_table4.json). Nothing is hard-coded: every
number the manuscript shows is looked up by (file, path) so that prose, tables
and figures cannot drift from the data.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper" / "manuscript"
GENERATED = MANUSCRIPT / "generated"
FIGURES = MANUSCRIPT / "figures"

MODEL = "paper/results/model.json"
RECOV = "paper/results/recovery.json"

# DRY RUN ONLY. PAPER_MODEL_JSON=<file> makes every read of paper/results/model.json come from <file>
# instead (for example model.quick.json from ``p3_model.py --quick``), so the section can be rendered
# with provisional numbers. It must be paired with PAPER_OUT_ROOT=<dir>: numbers.tex, the tables and
# the figures are then written under <dir>/generated and <dir>/figures, never over the committed
# assets, and the build prints a banner. Unset both for the real build (the default).
DRYRUN_MODEL = os.environ.get("PAPER_MODEL_JSON")
DRYRUN_OUT = os.environ.get("PAPER_OUT_ROOT")
if DRYRUN_MODEL or DRYRUN_OUT:
    if not (DRYRUN_MODEL and DRYRUN_OUT):
        raise SystemExit("PAPER_MODEL_JSON and PAPER_OUT_ROOT must be set together (dry run only)")
    GENERATED = Path(DRYRUN_OUT) / "generated"
    FIGURES = Path(DRYRUN_OUT) / "figures"
    print(f"*** DRY RUN: model.json read from {DRYRUN_MODEL}; outputs under {DRYRUN_OUT} ***", file=sys.stderr)
MGMT = "paper/results/management.json"
TELE = "paper/results/telemetry.json"
TRANS = "paper/results/transfer.json"
PIT = "paper/results/pit_loss.json"
STRAT = "paper/results/strategy.json"
BENCH = "paper/results/benchmark.json"
DESIGN = "paper/results/design_sweep.json"
PUB4 = "benchmark/published_table4.json"
ART_POWER = "data/artifacts/power.json"

COMPOUNDS = ["C1", "C2", "C3", "C4", "C5"]
MINUS = r"\ensuremath{-}"
PLUS = r"\ensuremath{+}"


@cache
def load(rel: str):
    if DRYRUN_MODEL and rel == MODEL:
        return json.loads(Path(DRYRUN_MODEL).read_text(encoding="utf-8"))
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def shown(p: Path) -> str:
    """Repo-relative path for messages (absolute for a dry-run output elsewhere)."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def has(rel: str, *path) -> bool:
    """True when the JSON path resolves (a key that a long run has not produced yet is False)."""
    try:
        get(rel, *path)
    except (KeyError, IndexError, TypeError, OSError):
        return False
    return True


def get(rel: str, *path):
    """Resolve a JSON path (keys and list indexes) in a repo-relative file."""
    doc = load(rel)
    for p in path:
        doc = doc[p]
    return doc


def fmt_num(x: float, places: int, signed: bool = False) -> str:
    """LaTeX text for a number, rounded to ``places`` decimals.

    Negative numbers use a real minus sign (``\\ensuremath{-}``), which also
    works inside math mode. ``signed`` adds an explicit plus sign. Integers of
    four or more digits get a thousands separator (``11{,}039``).
    """
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        raise ValueError(f"cannot format {x!r}")
    r = round(float(x), places)
    if r == 0:
        r = 0.0  # no negative zero
    s = f"{abs(r):.{max(places, 0)}f}"
    if places == 0 and abs(r) >= 1000:
        s = f"{int(abs(r)):,}".replace(",", "{,}")
    if r < 0:
        return MINUS + s
    return (PLUS + s) if signed else s


_ESC = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc(s: str) -> str:
    """Escape LaTeX specials in plain text (not for strings that already hold macros)."""
    return "".join(_ESC.get(ch, ch) for ch in str(s))


def ci(lo: float, hi: float, places: int, signed: bool = False) -> str:
    return f"[{fmt_num(lo, places, signed)}, {fmt_num(hi, places, signed)}]"


def tabular(colspec: str, header: list[str], rows: list, notes_after_header: bool = False) -> str:
    """Booktabs tabular, no vertical rules.

    ``rows`` items are either a list of cell strings, the string ``"\\midrule"``
    (a rule), or a ``("panel", text)`` tuple which renders a spanning,
    italic panel heading.
    """
    out =[rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    ncols = len(header)
    for r in rows:
        if isinstance(r, tuple) and r[0] == "panel":
            out.append(rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{r[1]}}}}} \\")
        elif r == r"\midrule":
            out.append(r"\midrule")
        else:
            assert len(r) == ncols, (len(r), ncols, r)
            out.append(" & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(out)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF newlines on every platform so committed output is byte-stable.
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


CLAIM_A = re.compile(r"^## A\..*?(?=^## B\.)", re.S | re.M)


def section_a_claims() -> list[dict]:
    """The machine-readable checks of CLAIMS.md section A (reuses check_claims.parse)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_claims", ROOT / "scripts" / "paper" / "check_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    text = (ROOT / "paper" / "CLAIMS.md").read_text(encoding="utf-8")
    m = CLAIM_A.search(text)
    if not m:
        raise RuntimeError("CLAIMS.md has no section A")
    # ``claims-pending`` lines (keys a long run has not produced yet) are included, flagged "_pending":
    # make_numbers takes their rounding from them, and they have a value only after --promote-pending.
    return mod.parse(m.group(0)) + mod.parse_pending(m.group(0))

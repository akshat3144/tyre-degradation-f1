"""Rebuild every generated manuscript asset from the committed JSON results.

    python scripts/paper/build_manuscript_assets.py

Runs, in order: make_numbers.py (generated/numbers.tex), make_tables.py
(generated/tab_*.tex), make_figures.py (figures/*.pdf). Offline, deterministic
(no random numbers) and fast. ``--check`` rebuilds into memory/temp and fails
if the committed files differ (for CI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_figures  # noqa: E402
import make_numbers  # noqa: E402
import make_tables  # noqa: E402
from _paper_common import GENERATED  # noqa: E402


def check() -> list[str]:
    """Differences between the committed text assets and a fresh build (figures are not byte-compared)."""
    diffs = []
    fresh = {"numbers.tex": make_numbers.render(make_numbers.build())}
    fresh.update({f"{k}.tex": v for k, v in make_tables.build_all().items()})
    for name, text in fresh.items():
        p = GENERATED / name
        if not p.exists():
            diffs.append(f"{name}: missing")
        elif p.read_text(encoding="utf-8") != text:
            diffs.append(f"{name}: differs from a fresh build")
    return diffs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify committed numbers/tables are current; write nothing")
    args = ap.parse_args()
    if args.check:
        diffs = check()
        for d in diffs:
            print("STALE", d, file=sys.stderr)
        print("generated assets are current" if not diffs else f"{len(diffs)} stale; run build_manuscript_assets.py")
        return 1 if diffs else 0
    for step in (make_numbers.main, make_tables.main, make_figures.main):
        rc = step()
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

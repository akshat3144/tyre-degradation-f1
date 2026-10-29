"""Check that every number quoted in paper/CLAIMS.md still matches its JSON source.

    python scripts/paper/check_claims.py            # exit 0 if all match
    python scripts/paper/check_claims.py -v         # list every check

CLAIMS.md carries machine-readable checks in fenced ```claims blocks, one JSON
object per line::

    {"file": "paper/results/model.json", "path": ["shipped_fit", "rates_s_per_lap", "C1", "mean"],
     "value": 0.0824, "places": 4}
    {"file": "paper/results/management.json", "path": ["decision", "H1_additive_interaction_supported"],
     "equals": false}

``path`` is a list of keys and list indexes (keys may contain dots or slashes).
``value`` is compared after rounding the JSON number to ``places`` decimals;
``equals`` is an exact comparison for strings, booleans, lists and nulls.

A second kind of fenced block, ```claims-pending, lists checks whose JSON key a long run has not
produced yet (``power_extended`` in model.json). Each line has the same form, but ``value`` and
``equals`` may both be omitted::

    {"file": "paper/results/model.json", "path": ["power_extended", "..."], "places": 0}

While the key does not exist the line is SKIPPED (listed as pending with -v). Once the key exists
the line is checked like any other; a line with neither ``value`` nor ``equals`` then FAILS and asks
for ``--promote-pending``, which rewrites CLAIMS.md: it fills in each resolvable line's value
(rounded to ``places``) and moves it from the pending block into an ordinary ```claims block.

Offline and fast: it only reads JSON files already in the repository.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLAIMS = ROOT / "paper" / "CLAIMS.md"
BLOCK = re.compile(r"```claims\s*\n(.*?)```", re.S)
PENDING_BLOCK = re.compile(r"```claims-pending[^\n]*\n(.*?)```", re.S)


def _lines(blocks) -> list[dict]:
    checks = []
    for block in blocks:
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                checks.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"bad claims line {line[:80]!r}: {e}") from e
    return checks


def parse(text: str) -> list[dict]:
    """The ordinary checks (```claims blocks)."""
    return _lines(BLOCK.findall(text))


def parse_pending(text: str) -> list[dict]:
    """The checks of ```claims-pending blocks, each flagged ``"_pending": true``."""
    return [{**c, "_pending": True} for c in _lines(PENDING_BLOCK.findall(text))]


def resolve(doc, path):
    for p in path:
        doc = doc[p]
    return doc


def run(claims_path: Path = CLAIMS, root: Path = ROOT, verbose: bool = False) -> list[str]:
    """Return a list of failure messages (empty if everything matches)."""
    text = claims_path.read_text(encoding="utf-8")
    checks = parse(text)
    if not checks:
        return ["no claims blocks found in CLAIMS.md"]
    checks += parse_pending(text)
    docs: dict[str, object] = {}
    failures: list[str] = []
    n_pending = 0
    for c in checks:
        f = c["file"]
        label = f"{f} :: {'.'.join(str(p) for p in c['path'])}"
        try:
            if f not in docs:
                docs[f] = json.loads((root / f).read_text(encoding="utf-8"))
            actual = resolve(docs[f], c["path"])
        except (OSError, KeyError, IndexError, TypeError, ValueError) as e:
            if c.get("_pending"):  # the long run has not produced this key yet
                n_pending += 1
                if verbose:
                    print(f"pend {label} (key not in {f} yet)")
                continue
            failures.append(f"{label}: cannot resolve ({type(e).__name__}: {e})")
            continue
        if c.get("_pending") and "equals" not in c and "value" not in c:
            failures.append(
                f"{label}: pending claim now resolves to {actual!r}; run "
                "`python scripts/paper/check_claims.py --promote-pending`, then review CLAIMS.md A6"
            )
            continue
        if "equals" in c:
            ok = actual == c["equals"]
            want = c["equals"]
        else:
            want = c["value"]
            places = int(c.get("places", 3))
            ok = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isfinite(actual)
                and round(float(actual), places) == round(float(want), places)
            )
        if verbose:
            print(("ok   " if ok else "FAIL ") + f"{label} = {actual!r} (quoted {want!r})")
        if not ok:
            failures.append(f"{label}: JSON has {actual!r}, CLAIMS.md quotes {want!r}")
    if verbose:
        print(f"{len(checks)} checks ({n_pending} pending, key absent), {len(failures)} failed")
    return failures


def promote_pending(claims_path: Path = CLAIMS, root: Path = ROOT) -> int:
    """Fill in and move every resolvable ```claims-pending line to an ordinary ```claims block.

    Values are the JSON numbers rounded to each line's ``places``. Lines whose key is still
    absent stay pending. Returns the number of lines promoted.
    """
    text = claims_path.read_text(encoding="utf-8")
    docs: dict[str, object] = {}
    n = 0

    def rewrite(m: re.Match) -> str:
        nonlocal n
        keep, done = [], []
        for line in m.group(1).splitlines():
            if not line.strip():
                continue
            c = json.loads(line)
            try:
                if c["file"] not in docs:
                    docs[c["file"]] = json.loads((root / c["file"]).read_text(encoding="utf-8"))
                actual = resolve(docs[c["file"]], c["path"])
            except (OSError, KeyError, IndexError, TypeError, ValueError):
                keep.append(line.strip())
                continue
            if "equals" not in c and "value" not in c:
                c["value"] = round(float(actual), int(c.get("places", 3)))
            done.append(json.dumps(c))
            n += 1
        out = ""
        if done:
            out += "```claims\n" + "\n".join(done) + "\n```"
        if keep:
            out += ("\n\n" if done else "") + "```claims-pending\n" + "\n".join(keep) + "\n```"
        return out or "<!-- all pending claims promoted -->"

    new = PENDING_BLOCK.sub(rewrite, text)
    if n:
        claims_path.write_text(new, encoding="utf-8")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--promote-pending", action="store_true",
                    help="fill in and move the resolvable ```claims-pending lines to ordinary claims blocks")
    args = ap.parse_args()
    if args.promote_pending:
        n = promote_pending()
        print(f"promoted {n} pending claim(s); review paper/CLAIMS.md (the A6 prose is not rewritten)")
        return 0
    failures = run(verbose=args.verbose)
    for f in failures:
        print("MISMATCH", f, file=sys.stderr)
    if failures:
        print(f"{len(failures)} claim(s) no longer match their source; update paper/CLAIMS.md", file=sys.stderr)
        return 1
    print("all claims match their sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

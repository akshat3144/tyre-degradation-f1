"""Write the supplementary table of Pirelli dry-compound nominations.

    python scripts/paper/make_supplement.py

Reads data/allocation.json (2026) and data/allocation_history.json (2022-2025)
and writes paper/manuscript/supplementary/pirelli_nominations.csv with one row
per Grand Prix: season, event, the physical compound (C number) labelled hard,
medium and soft, whether two sources agree, and the source URLs. The 2026
rows come from Pirelli's pre-event announcements and were not cross-checked
against a second source, so they are marked "not checked". Offline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "manuscript" / "supplementary" / "pirelli_nominations.csv"


def rows() -> list[dict]:
    out: list[dict] = []
    hist = json.loads((ROOT / "data" / "allocation_history.json").read_text(encoding="utf-8"))
    for season, events in hist["seasons"].items():
        for event, rec in events.items():
            out.append(
                {
                    "season": int(season),
                    "event": event,
                    "hard": rec["HARD"],
                    "medium": rec["MEDIUM"],
                    "soft": rec["SOFT"],
                    "two_sources_agree": rec.get("verified"),
                    "sources": " ".join(rec.get("sources", [])),
                }
            )
    for rec in json.loads((ROOT / "data" / "allocation.json").read_text(encoding="utf-8")):
        c = rec["compounds"]
        out.append(
            {
                "season": rec["season"],
                "event": rec["event"],
                "hard": c["HARD"],
                "medium": c["MEDIUM"],
                "soft": c["SOFT"],
                "two_sources_agree": "not checked",
                "sources": "Pirelli pre-event compound nomination (press.pirelli.com)",
            }
        )
    return sorted(out, key=lambda r: (r["season"], r["event"]))


def main() -> int:
    data = rows()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(data)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

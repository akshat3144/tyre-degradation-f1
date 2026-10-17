"""Which physical compounds Pirelli nominated at every 2022-2025 Grand Prix.

WHY LABELS ARE RELATIVE AND C-NUMBERS ARE PHYSICAL

The timing feed (and therefore every lap in ``data/processed/laps_<year>.parquet``)
only says HARD / MEDIUM / SOFT. Those words are RELATIVE: for each weekend
Pirelli picks three of its numbered compounds (C0 hardest ... C5, and from 2025
C6, softest) and paints them white / yellow / red. A "SOFT" at Bahrain 2024 is
the C3; a "SOFT" at Monaco 2024 is the C5, which is a physically different,
much softer rubber. Pooling laps by label therefore averages different tyres
together. The C number is the physical identity of the rubber and is what a
degradation model should be keyed on.

The mapping label -> C number is not in the timing data. It comes from Pirelli's
race-preview press releases, collected in ``data/allocation_history.json`` with
the source URLs for each race. Rows where sources disagree, or where the label
order was only inferred, carry ``"verified": false`` and a note; callers who need
strict provenance should filter on it.

Numbering caveat: C-numbers are comparable WITHIN a season's range (the compounds
are re-specified each year and Pirelli re-numbered the range for 2025, adding
the C6), so softness() orders compounds, it does not equate a 2022 C3 with a
2025 C3 chemically.

Event keys are FastF1 ``EventName`` strings, so they join to the parquets.
"""

from __future__ import annotations

import json
from functools import lru_cache

from ..config import DATA

STORE = DATA / "allocation_history.json"

LABELS = ("HARD", "MEDIUM", "SOFT")


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(STORE.read_text(encoding="utf-8"))


def load_history() -> dict:
    """The full document: ``{"source_note": ..., "seasons": {year: {event: row}}}``."""
    return _load()


def c_number(season: int | str, event: str, label: str) -> str | None:
    """Physical compound ("C0".."C6") behind a HARD/MEDIUM/SOFT label, else None.

    None means the race is not in the history (unknown season/event, or a
    cancelled race) or the label is not a dry-compound label. It is never a guess.
    """
    row = _load().get("seasons", {}).get(str(season), {}).get(event)
    if not row:
        return None
    return row.get(str(label).upper()) if str(label).upper() in LABELS else None


def softness(compound: str) -> int:
    """Softness rank from the C number: C0 -> 0 ... C6 -> 6 (higher is softer)."""
    c = str(compound).strip().upper()
    if len(c) != 2 or c[0] != "C" or c[1] not in "0123456":
        raise ValueError(f"not a Pirelli compound: {compound!r}")
    return int(c[1])

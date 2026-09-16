"""Which physical compounds Pirelli nominated for a weekend.

THE ONE FACT THAT CANNOT BE DISCOVERED

Everything else about a race comes from the API: the calendar, the session
times, the lap times, the tyre labels. Not this. The timing feed reports
HARD / MEDIUM / SOFT, which are RELATIVE to whatever three of C1-C5 were
brought to that circuit, and neither the session nor the event metadata carries
the mapping. It is published in a Pirelli press release.

That matters more than it sounds, and it is the reason this project keys
everything on the C number: a "HARD" at Monaco (C3) is softer rubber than a
"SOFT" at Suzuka. Pool by label across events and you average different tyres
together, which is what produced a physically backwards ordering in our own
first pooled fit.

WHY IT IS A FILE AND NOT A CONSTANT

It used to be ``config.COMPOUND_ALLOCATION_2026``, so adding a race meant
editing Python and pushing. Here it is a
small JSON store, so adding a race is a data edit.

The seed values came from Pirelli press releases and per-race F1.com tyre
previews. Anything a user later edits is marked with its source, so a number
somebody typed can never be mistaken for one we cited.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from ..config import COMPOUND_ALLOCATION_2026, DATA

log = logging.getLogger(__name__)

STORE = DATA / "allocation.json"

C_COMPOUNDS = ("C1", "C2", "C3", "C4", "C5")
LABELS = ("HARD", "MEDIUM", "SOFT")


@dataclass
class Allocation:
    event: str
    season: int
    #: Label -> C number, e.g. {"HARD": "C3", "MEDIUM": "C4", "SOFT": "C5"}
    compounds: dict[str, str]
    #: "pirelli" for the seeded values we cited, "user" for anything edited by
    #: hand. The difference is kept visible; a typed number and a cited one are
    #: not the same kind of claim.
    source: str = "pirelli"
    updated_at: str | None = None

    def validate(self) -> None:
        missing = [lab for lab in LABELS if lab not in self.compounds]
        if missing:
            raise ValueError(f"missing {missing}; all of {LABELS} are required")
        bad = [c for c in self.compounds.values() if c not in C_COMPOUNDS]
        if bad:
            raise ValueError(f"{bad} are not compounds; expected {C_COMPOUNDS}")
        if len({self.compounds[lab] for lab in LABELS}) != 3:
            raise ValueError("the three nominated compounds must be distinct")
        # Pirelli nominates three ADJACENT-or-ordered compounds and the labels
        # are assigned hardest to softest. An allocation where MEDIUM is harder
        # rubber than HARD is a typo, and catching it here beats discovering it
        # as an inverted degradation ordering three screens later.
        idx = [C_COMPOUNDS.index(self.compounds[lab]) for lab in LABELS]
        if idx != sorted(idx):
            raise ValueError(
                f"HARD/MEDIUM/SOFT must run hardest to softest; got "
                f"{[self.compounds[lab] for lab in LABELS]}"
            )


def _seed() -> dict[str, Allocation]:
    return {
        event: Allocation(event=event, season=2026, compounds=dict(compounds))
        for event, compounds in COMPOUND_ALLOCATION_2026.items()
    }


def _read() -> dict[str, Allocation]:
    if not STORE.exists():
        return _seed()
    try:
        raw = json.loads(STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # A half-written store must not take the API down with it. Every
        # strategy and forecast request reads this file, so raising here
        # turns one bad write into a total outage. Degrading to the cited
        # Pirelli values is the same behaviour as the file being absent.
        log.error("allocation store unreadable (%s); using cited values", exc)
        return _seed()
    out = {}
    for row in raw:
        a = Allocation(**row)
        out[a.event] = a
    # Seeded events missing from the store are added back rather than lost, so
    # deleting the file degrades to the cited values instead of to nothing.
    for event, a in _seed().items():
        out.setdefault(event, a)
    return out


def _write(store: dict[str, Allocation]) -> None:
    """Save the store atomically.

    ``write_text`` truncates before it writes, so a process killed mid-save
    -- a container stopped during a deploy, say -- leaves invalid JSON
    behind. Writing to a sibling temp file and renaming makes the swap
    atomic on POSIX and Windows alike: a reader sees the old file or the
    new one, never a half-written one. This is the one durability guarantee
    a database would have bought us, in three lines instead of a service.
    """
    STORE.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(a) for a in sorted(store.values(), key=lambda x: x.event)]
    tmp = STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, STORE)


def all_allocations() -> dict[str, Allocation]:
    return _read()


def get(event: str) -> Allocation | None:
    return _read().get(event)


def compounds_for(event: str) -> dict[str, str]:
    """Label -> C number, or {} when the weekend has not been nominated yet.

    Callers must handle the empty case: a race on the calendar that Pirelli has
    not announced tyres for is a normal state, not an error.
    """
    a = get(event)
    return dict(a.compounds) if a else {}


def set_allocation(
    event: str, compounds: dict[str, str], season: int = 2026, source: str = "user"
) -> Allocation:
    """Record a nomination. Validates before writing."""
    a = Allocation(
        event=event,
        season=season,
        compounds={k.upper(): v.upper() for k, v in compounds.items()},
        source=source,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    a.validate()
    store = _read()
    store[event] = a
    _write(store)
    return a


def unset(event: str) -> bool:
    """Remove a user edit, falling back to the seeded value if there is one."""
    store = _read()
    if event not in store:
        return False
    seeded = _seed().get(event)
    if seeded:
        store[event] = seeded
    else:
        del store[event]
    _write(store)
    return True

"""Which race at which circuit is the same pit lane?

A pit loss belongs to a pit lane, and a pit lane is a place. Event NAMES are not
places: in 2026 the "Spanish Grand Prix" is at Madrid (a circuit that has never
held a race) while the Barcelona track it used to name runs as the "Barcelona
Grand Prix", and the "Bahrain Grand Prix" has moved from Sakhir to Sepang
(FastF1 lists its location as Kuala Lumpur). Matching history on the name gave
Madrid four seasons of Barcelona's pit lane and Bahrain four seasons of Sakhir's
-- confident, precise, wrong.

Location strings drift too (Monaco becomes Monte Carlo, Yas Island becomes Yas
Marina, Miami becomes Miami Gardens), so the rule is: two rounds are the same
circuit if and only if their locations are equal after an EXPLICIT alias table.
No fuzzy matching and no name fallback -- an unlisted drift loses history, which
is visible and safe, where a wrong match is neither.
"""

from __future__ import annotations

import unicodedata

#: Location strings that name one place, mapped to the form we compare on.
#: Every entry is a place whose FastF1 spelling changed between 2022 and 2026.
LOCATION_ALIASES = {
    "monte carlo": "monaco",
    "yas marina": "yas island",
    "miami gardens": "miami",
}


def canonical_location(location: object) -> str:
    """A location reduced to a comparable key: casefolded, accent-stripped, aliased."""
    s = unicodedata.normalize("NFKD", str(location)).encode("ascii", "ignore").decode()
    s = s.replace("?", "").strip().casefold()
    return LOCATION_ALIASES.get(s, s)


def same_circuit(a: object, b: object) -> bool:
    """True when two location strings name the same circuit."""
    ka, kb = canonical_location(a), canonical_location(b)
    return bool(ka) and ka == kb

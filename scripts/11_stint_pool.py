"""Write the observed stint-length pool used for the strategy layer's support bounds.

    python scripts/11_stint_pool.py
    python scripts/11_stint_pool.py --rebuild    # re-read the races from FastF1's cache

The strategy optimiser (``tyredeg.strategy.optimise``) refuses stint lengths
no one has ever run on a tyre: a ceiling from the longest stints actually
completed and a floor from the shortest sensible ones (``tyredeg.strategy.support``).
Both bounds are quantiles of a pool of genuine stints from every season we hold,
so the pool needs the RAW laps of all seasons, not just the clean 2026 laps.

Output: ``data/artifacts/stint_pool.json`` (lengths per race and compound label,
split by the bound they feed). ``scripts/06_strategy.py`` reads it so the
bounds can be rebuilt and audited; the paper's strategy evaluation
(``scripts/paper/p5_strategy.py``) rebuilds the pool itself, out of time.
"""

from __future__ import annotations

import argparse
import logging
import warnings

from tyredeg.config import ARTIFACTS
from tyredeg.strategy import racedata, support

warnings.filterwarnings("ignore")
logging.getLogger("fastf1").setLevel(logging.ERROR)

OUT = ARTIFACTS / support.POOL_FILE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="re-read the races from FastF1's cache")
    args = ap.parse_args()

    laps, _ = racedata.load(rebuild=args.rebuild)
    pool = support.observed_stints(laps)
    support.write_pool(pool, OUT)
    print(
        f"stint pool: {int(pool['ceiling_ok'].sum())} ceiling / {int(pool['floor_ok'].sum())} floor stints "
        f"from {pool.groupby(['season', 'event']).ngroups} races -> {OUT}"
    )


if __name__ == "__main__":
    main()

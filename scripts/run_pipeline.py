"""Run the core estimation pipeline, in order, with one command.

    python scripts/run_pipeline.py                 # refit everything from the cached laps
    python scripts/run_pipeline.py --refresh       # pull sessions from FastF1 first (network)
    python scripts/run_pipeline.py --event "Spanish Grand Prix"
    python scripts/run_pipeline.py --skip-validation

This runs the numbered stage scripts that write ``data/artifacts/*.json``. The
paper's own analyses (``scripts/paper/p1`` .. ``p6``), the pit-loss, session-skill
and design-sweep scripts (10, 12, 13) and the manuscript asset build are separate;
see the README for the full reproduction order.

Every stage reports what it did, so a failure part way through says which stage
and leaves the artifacts from the stages that succeeded.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(label: str, args: list[str], optional: bool = False) -> bool:
    """Run one stage. Returns True on success."""
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}", flush=True)
    t0 = time.time()
    proc = subprocess.run([PY, *args], cwd=ROOT)
    ok = proc.returncode == 0
    mark = "ok" if ok else ("skipped" if optional else "FAILED")
    print(f"-- {label}: {mark} ({time.time() - t0:.1f}s)", flush=True)
    # An optional stage that fails is not a pipeline failure -- it prints
    # "skipped", and the caller counts anything falsy as FAILED, so saying
    # otherwise here would contradict what was just printed on screen.
    return ok or optional


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="pull sessions from FastF1 first (needs network)")
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--event", default="Hungarian Grand Prix", help="event for the single-event strategy script")
    ap.add_argument("--skip-validation", action="store_true", help="validation is the slow stage")
    args = ap.parse_args()

    stages: list[tuple[str, list[str], bool]] = []

    if args.refresh:
        stages.append(
            ("1. cache sessions (needs network)",
             ["scripts/01_cache_sessions.py", "--season", str(args.season)], False)
        )

    stages += [
        ("2. fit degradation", ["scripts/03_fit_model.py"], False),
        ("3. practice to race transfer", ["scripts/05_transfer.py"], False),
        ("4. strategy (single event)", ["scripts/06_strategy.py", "--event", args.event], False),
        # Optional because each needs a season the default cache may not have:
        # management pools several, and the benchmark scores 2025 specifically.
        ("5. tyre management", ["scripts/07_management.py"], True),
        ("6. benchmark", ["scripts/08_benchmark.py"], True),
    ]

    if not args.skip_validation:
        # Power analysis and leave-one-run-out calibration are the slow part.
        stages.insert(2, ("validation: power, calibration, scoring",
                          ["scripts/04_validate.py"], False))

    t0 = time.time()
    failed = [label for label, argv, opt in stages if not run(label, argv, opt)]

    print(f"\n{'=' * 72}")
    if failed:
        print(f"{len(failed)} stage(s) FAILED after {time.time() - t0:.1f}s:")
        for f in failed:
            print(f"   {f}")
        print("\nArtifacts from the stages that succeeded are still on disk.")
        raise SystemExit(1)

    print(f"all {len(stages)} stages ok in {time.time() - t0:.1f}s")
    print("artifacts written to data/artifacts/")


if __name__ == "__main__":
    main()

"""Download and cache every session we need, so later stages run with no network.

Run this well before the event, and copy data/fastf1_cache/ to every laptop.

    python scripts/01_cache_sessions.py
    python scripts/01_cache_sessions.py --events "Italian Grand Prix"

Only conventional weekends are included by default, because only they have an
FP2 and therefore practice long runs. Sprint weekends still race on Sunday, so
pass them explicitly to collect race data:

    python scripts/01_cache_sessions.py --sessions R --events "British Grand Prix"

"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime

import pandas as pd

from tyredeg.config import PROCESSED, SEASON
from tyredeg.data.cache import load_session
from tyredeg.data.laps import clean_laps, summarise, tag_long_runs
from tyredeg.data.schedule import find, race_events

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("fastf1").setLevel(logging.ERROR)

SESSIONS = ("FP1", "FP2", "FP3", "R")


def _conventional_events(season: int) -> list[str]:
    """Conventional weekends for a season, from the schedule.

    Sprint weekends run only FP1 before Sprint Qualifying, so they carry no
    race-simulation long runs and are excluded.
    """
    import fastf1

    from tyredeg.data.cache import enable_cache

    enable_cache()
    sched = fastf1.get_event_schedule(season, include_testing=False)
    return sched[sched["EventFormat"] == "conventional"]["EventName"].tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", nargs="*", default=None)
    ap.add_argument(
        "--sessions",
        nargs="*",
        default=None,
        help="override; by default each event gets the sessions its weekend "
             "format actually has (a sprint weekend has only a race)",
    )
    ap.add_argument("--season", type=int, default=SEASON)
    ap.add_argument("--out", default=None, help="parquet filename; defaults per season")
    args = ap.parse_args()

    # 2026 keeps writing laps.parquet so nothing downstream changes. Earlier
    # seasons go to their own file: their tyre construction differs, so their
    # degradation rates must never be pooled with 2026's by accident.
    # Default to EVERY round with data, not just conventional ones. Which
    # sessions each event offers is decided per event below, because a sprint
    # weekend has only a race and asking it for FP2 logs a failure -- which
    # writes `complete: false` and makes the guard refuse the whole event.
    if args.events is None:
        args.events = (
            race_events(args.season)
            if args.season == SEASON
            else _conventional_events(args.season)
        )
    out_name = args.out or ("laps.parquet" if args.season == SEASON
                            else f"laps_{args.season}.parquet")

    #: Sessions we deliberately did not ask for because they have not run yet.
    #: Kept apart from `failures` because they are not one: a session in the
    #: future is a fact about the calendar, not about the download.
    pending: dict[str, list[str]] = {}

    def sessions_for(event: str) -> list[str]:
        """What to request for one event.

        An explicit --sessions wins. Otherwise ask the calendar: a conventional
        weekend has FP1/FP2/FP3 and a race, a sprint weekend has only the race.
        Requesting a session a weekend does not have logs a failure, which sets
        `complete: false`, which makes the completeness guard reject the event.
        """
        rnd = find(event, args.season)
        if args.sessions is not None:
            # An explicit --sessions still may not ask for a session that has
            # not happened yet. It used to, and that is a quiet way to lose a
            # season: asking Madrid for FP3 three hours before FP3 ran logged
            # one failure, wrote `complete: false`, and dropped all of 2026 out
            # of the compound-management analysis -- 98 cells to 84, five
            # seasons to four, with nothing on screen to say so.
            if rnd is None:
                return list(args.sessions)
            ready = [c for c in args.sessions if (s := rnd.session(c)) and s.has_run()]
            if later := [c for c in args.sessions if c not in ready]:
                pending[event] = later
            return ready
        if rnd is None:
            return list(SESSIONS)
        # sessions_to_pull, NOT cacheable_sessions: the first is what has
        # actually finished, the second is what the format has in principle.
        # Using the latter asked eleven unrun 2026 events for four sessions
        # each, failed all 44, wrote `complete: false`, and hit the 500/h cap.
        return rnd.sessions_to_pull()

    requested: dict[str, list[str]] = {e: sessions_for(e) for e in args.events}
    # Drop events with nothing to fetch. The calendar carries all 23 rounds of
    # 2026, most of which have not happened; asking for them logs failures,
    # which writes `complete: false`, which makes the completeness guard refuse
    # the whole season. An event that has not run is not a failure.
    skipped = [e for e, v in requested.items() if not v]
    requested = {e: v for e, v in requested.items() if v}
    args.events = [e for e in args.events if e in requested]
    if skipped:
        names = ", ".join(e.replace(" Grand Prix", "") for e in skipped[:6])
        more = " ..." if len(skipped) > 6 else ""
        print(f"  .. {len(skipped)} event(s) not run yet, skipped: {names}{more}", flush=True)
    for ev, later in sorted(pending.items()):
        print(f"  .. {ev}: {', '.join(later)} not run yet, skipped", flush=True)
    frames, rows = [], []
    for event in args.events:
        for ses in requested[event]:
            t0 = time.time()
            try:
                s = load_session(event, ses, args.season)
                clean = clean_laps(s.laps, event=event, session=ses)
                frames.append(clean)
                print(
                    f"  ok   {event:24s} {ses:3s}  raw={len(s.laps):4d} "
                    f"clean={len(clean):4d}  ({time.time() - t0:4.1f}s)",
                    flush=True,
                )
            except Exception as exc:  # one bad session must not stop the batch
                print(f"  FAIL {event:24s} {ses:3s}  {type(exc).__name__}: {exc}"[:120], flush=True)
                rows.append({"event": event, "session": ses, "error": str(exc)[:200]})

    if not frames:
        print("\nnothing loaded")
        return

    fresh = pd.concat(frames, ignore_index=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / out_name

    # MERGE, never replace.
    #
    # This used to write only what it had just pulled, which was fine while the
    # only caller asked for a whole season. It is catastrophic the moment
    # anything asks for one event: a refresh of a single race would
    # have replaced seven complete events with one, and the failure looks like a
    # successful pull.
    #
    # Sessions we just fetched replace their previous copies; everything else is
    # kept. A pull that partially fails therefore degrades to "we still have
    # what we had", which is the right direction to fail in.
    if out.exists():
        try:
            existing = pd.read_parquet(out)
            pulled = set(zip(fresh["event"], fresh["session"], strict=True))
            keep = existing[
                ~pd.Series(
                    list(zip(existing["event"], existing["session"], strict=True)),
                    index=existing.index,
                ).isin(pulled)
            ]
            if not keep.empty:
                print(
                    f"\nmerging with {len(keep):,} existing laps "
                    f"across {keep['event'].nunique()} event(s)"
                )
                fresh = pd.concat([keep, fresh], ignore_index=True)
        except Exception as exc:  # noqa: BLE001 -- a corrupt old file must not block a good pull
            print(f"\ncould not read existing {out.name} ({type(exc).__name__}); writing fresh")

    # Tag long runs once, across everything. Doing it per session made run_id
    # restart at 0 for each one, so ids collided after concatenation. Re-tagging
    # the merged frame is safe because run_id is a readable composite key rather
    # than a per-call code.
    df = tag_long_runs(fresh)
    df.to_parquet(out, index=False)

    # A season that half-downloaded looks exactly like a complete one on disk:
    # the parquet holds whatever arrived and records nothing about what did not.
    # The F1 API caps at 500 calls/hour, so a truncated pull is a normal event,
    # not a rare one, and pooling one silently would drop whole events from an
    # analysis without anyone noticing. The manifest is what lets downstream
    # scripts refuse it.
    manifest = {
        "season": args.season,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "parquet": out.name,
        "sessions_requested": sum(len(v) for v in requested.values()),
        "sessions_by_event_requested": {k: sorted(v) for k, v in sorted(requested.items())},
        "sessions_loaded": len(frames),
        "sessions_failed": len(rows),
        "events_requested": sorted(args.events),
        "events_loaded": sorted(df["event"].unique().tolist()),
        "complete": not rows,
        "failures": rows,
        # Not a failure and deliberately not part of `complete`: these are
        # sessions the calendar says have not started. Recorded so a later run
        # can tell "we have not pulled it" from "it does not exist".
        "sessions_pending": {k: sorted(v) for k, v in sorted(pending.items())},
        # What the MERGED file now holds, per event. `complete` above describes
        # this pull; this describes the artifact on disk, which after a merge
        # are different questions.
        "sessions_by_event": {
            str(ev): sorted(g["session"].unique().tolist())
            for ev, g in df.groupby("event", sort=True)
        },
    }
    man = out.with_name(f"{out.stem}.manifest.json")
    man.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {out}  ({len(df):,} clean laps)")
    print(f"wrote {man}  (complete={manifest['complete']})")
    print("\n" + summarise(df).to_string(index=False))

    long_runs = df[df.is_long_run]
    print(
        f"\nTOTAL long-run laps: {len(long_runs):,} "
        f"across {long_runs.run_id.nunique()} runs and {long_runs.Driver.nunique()} drivers"
    )
    if rows:
        print(f"\n{len(rows)} session(s) failed:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

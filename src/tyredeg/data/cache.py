"""FastF1 session loading and caching.

Everything goes through here so the cache location is set exactly once, and so
analyses can run with no network. Load every session you need in advance,
then the same calls read from disk.

Telemetry is off by default. It is the bulk of the download and we only needed
it for a traffic covariate, which is not available for 2026 anyway (Position is
empty). Turn it on explicitly if that changes.
"""

from __future__ import annotations

import json
import logging

import fastf1
import pandas as pd

from ..config import CONVENTIONAL_2026, FASTF1_CACHE, SEASON

log = logging.getLogger(__name__)

_cache_ready = False


def enable_cache() -> None:
    """Point FastF1 at the project cache. Idempotent."""
    global _cache_ready
    if not _cache_ready:
        FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(FASTF1_CACHE))
        _cache_ready = True
        log.debug("FastF1 cache at %s", FASTF1_CACHE)


def load_session(
    event: str,
    session: str,
    season: int = SEASON,
    *,
    telemetry: bool = False,
    weather: bool = True,
):
    """Load one session, from cache when available.

    Args:
        event: Event name, e.g. "Hungarian Grand Prix".
        session: "FP1", "FP2", "FP3", "Q", "R".
        season: Championship year.
        telemetry: Load car and position telemetry. Large and slow; off by default.
        weather: Load the weather stream. Cheap, and we use TrackTemp.

    Returns:
        A loaded ``fastf1.core.Session``.
    """
    enable_cache()
    s = fastf1.get_session(season, event, session)
    s.load(telemetry=telemetry, weather=weather, messages=False)
    return s


def session_weather(s) -> pd.DataFrame:
    """Weather stream as a plain frame, with elapsed seconds.

    Returns an empty frame if the session has no weather data, rather than
    raising, so a single bad session cannot stop a batch.
    """
    w = getattr(s, "weather_data", None)
    if w is None or len(w) == 0:
        return pd.DataFrame()
    w = w.copy()
    if pd.api.types.is_timedelta64_dtype(w.get("Time")):
        w["SessionElapsed"] = w["Time"].dt.total_seconds()
    return w


def race_laps_scheduled(event: str, season: int = SEASON) -> int:
    """Scheduled lap count for a race, needed to index fuel correctly.

    Taken as the maximum ``LapNumber`` observed, which equals the scheduled
    distance for any race that ran to completion. A race stopped early would
    report short, so check the result if a value looks wrong.
    """
    s = load_session(event, "R", season, telemetry=False, weather=False)
    return int(s.laps["LapNumber"].max())


def practice_sessions(events: tuple[str, ...] = CONVENTIONAL_2026) -> list[tuple[str, str]]:
    """(event, session) pairs for every practice session worth loading.

    Only conventional weekends have FP2, which is where race-simulation long
    runs happen. Sprint weekends run FP1 then Sprint Qualifying, so they carry
    no long runs and are excluded upstream in ``config.CONVENTIONAL_2026``.
    """
    return [(e, ses) for e in events for ses in ("FP1", "FP2", "FP3")]


def season_completeness(path) -> tuple[bool, str]:
    """Is a season's parquet a full pull, per the manifest script 01 writes?

    A half-downloaded season is indistinguishable from a complete one by looking
    at the parquet: it holds whatever arrived and records nothing about what did
    not. The F1 API caps at 500 calls an hour, so truncated pulls are routine
    rather than rare -- our own 2022 pull needed three attempts, landing 9 then
    17 then all 19 events. Scoring or pooling a truncated season silently drops
    whole events, so callers should refuse rather than warn.

    Seasons cached before manifests existed report unverified and are allowed
    through. Excluding those would silently shrink the sample instead, which is
    the same failure pointing the other way.
    """
    man = path.with_name(f"{path.stem}.manifest.json")
    if not man.exists():
        return True, "completeness unverified (no manifest)"
    m = json.loads(man.read_text(encoding="utf-8"))
    if m.get("complete"):
        return True, f"complete ({m.get('sessions_loaded')} sessions)"
    missing = sorted(set(m.get("events_requested", [])) - set(m.get("events_loaded", [])))
    detail = f", missing {', '.join(missing)}" if missing else ""
    return False, (
        f"INCOMPLETE: {m.get('sessions_failed')} session(s) failed, "
        f"{len(missing)} event(s) missing{detail}"
    )

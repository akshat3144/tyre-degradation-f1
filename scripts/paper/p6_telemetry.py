"""Paper analysis 6: does telemetry show tyre nursing in races versus practice?

    python scripts/paper/p6_telemetry.py probe       # is each telemetry source reachable?
    python scripts/paper/p6_telemetry.py download    # fetch + reduce, resume-safe
    python scripts/paper/p6_telemetry.py analyse     # write paper/results/telemetry.json
    python scripts/paper/p6_telemetry.py all

    --seasons 2025 2026   (default; the pre-declared primary sample)
    --seasons 2023 2024   (pre-declared extension; analysed separately)

The design, proxies and confounders are in ``tyredeg.paper.telemetry``; read
that docstring first. This script is plumbing.

DATA SOURCE

Primary: OpenF1 (https://api.openf1.org/v1), public, no credentials. The F1 live
timing static host that FastF1 reads car data from returns HTTP 401, so the
FastF1 route is closed; the two sessions already in the FastF1 cache (Hungary
2026 FP2 and Race) are kept as a cross-check of the two sources, lap by lap.

POLITENESS

At most ``MAX_REQ_PER_S`` requests per second across all threads, exponential
backoff (honouring Retry-After) on 429 and 5xx, and every raw response is stored
gzipped under ``data/processed/telemetry_proxies/openf1/`` so a re-run never
asks twice. Nothing is written to the FastF1 cache by this route.

JOINING TO OUR LAPS

Meeting by (year, event name), session by name ("Race", "Practice 2"), driver
by acronym within the session, lap by ``lap_number`` to our ``LapNumber``. A lap
is cut from OpenF1's ``date_start`` for OpenF1's ``lap_duration``. The join is
checked: the OpenF1 lap time must agree with ours to 0.3 s, else the lap is
dropped and counted. The match rate is reported.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import subprocess
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from tyredeg.config import PROCESSED, ROOT
from tyredeg.paper import telemetry as T

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logging.getLogger("fastf1").setLevel(logging.CRITICAL)

OUT = ROOT / "paper" / "results" / "telemetry.json"
STORE = PROCESSED / "telemetry_proxies"
RAW = STORE / "openf1"
MANIFEST = STORE / "manifest_openf1.json"
OPENF1 = "https://api.openf1.org/v1/"

PRIMARY_SEASONS = (2025, 2026)
EXTENSION_SEASONS = (2023, 2024)
SEED = 20260905

MAX_REQ_PER_S = 2.0          # ceiling is 3; stay well under
N_WORKERS = 3
LAP_TIME_TOL_S = 0.3         # OpenF1 vs our lap time, join check
SESSION_NAME = {"R": "Race", "FP2": "Practice 2"}

LAP_FILES = {
    2026: PROCESSED / "laps.parquet",
    **{y: PROCESSED / f"laps_{y}.parquet" for y in (2025, 2024, 2023)},
}


# --------------------------------------------------------------------------- #
# Polite, cached OpenF1 client
# --------------------------------------------------------------------------- #

class Client:
    """GET with a global rate limit, backoff, and a gzip on-disk cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next = 0.0
        self.n_requests = 0
        self.n_retries = 0
        self.n_cache_hits = 0
        self.http = requests.Session()

    def _wait_turn(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + 1.0 / MAX_REQ_PER_S
        if delay:
            time.sleep(delay)

    def get(self, endpoint: str, query: str, cache_file: Path):
        """JSON for ``endpoint?query``; reads ``cache_file`` if present."""
        if cache_file.exists():
            self.n_cache_hits += 1
            return json.loads(gzip.decompress(cache_file.read_bytes()))
        url = f"{OPENF1}{endpoint}?{query}"
        last = None
        for attempt in range(8):
            self._wait_turn()
            try:
                self.n_requests += 1
                r = self.http.get(url, timeout=180)
            except requests.RequestException as e:
                last = type(e).__name__
                time.sleep(min(60, 2 ** attempt))
                self.n_retries += 1
                continue
            if r.status_code == 200:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_file.with_suffix(".tmp")
                tmp.write_bytes(gzip.compress(r.content, 6))
                tmp.replace(cache_file)
                return r.json()
            if r.status_code == 404:  # OpenF1 uses 404 for "no rows"
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_bytes(gzip.compress(b"[]"))
                return []
            last = f"HTTP {r.status_code}"
            if r.status_code in (429, 500, 502, 503, 504):
                wait = float(r.headers.get("Retry-After", 0) or 0) or min(120, 3 * 2 ** attempt)
                time.sleep(wait)
                self.n_retries += 1
                continue
            break
        raise RuntimeError(f"OpenF1 {endpoint}?{query}: {last}")


def _norm(name: str) -> str:
    """Accent- and case-insensitive key; OpenF1 mangles 'Sao Paulo' differently to us."""
    n = unicodedata.normalize("NFKD", name)
    return "".join(c for c in n if c.isalnum() and ord(c) < 128).lower()


# --------------------------------------------------------------------------- #
# Sample
# --------------------------------------------------------------------------- #

def planned_sessions(seasons) -> list[tuple[int, str, str]]:
    """(season, event, session) for every event with both a Race and an FP2 on file."""
    out = []
    for season in seasons:
        laps = pd.read_parquet(LAP_FILES[season], columns=["event", "session"])
        have = laps.groupby("event")["session"].agg(set)
        for event, ses in have.items():
            if "R" in ses and "FP2" in ses:
                out += [(season, event, "FP2"), (season, event, "R")]
    return sorted(out)


def wanted_laps(season: int) -> pd.DataFrame:
    return T.select_laps(pd.read_parquet(LAP_FILES[season]).assign(season=season))


def probe_access() -> dict:
    """One request per telemetry host, recorded because 'unavailable' is a finding."""
    out = {"probed_utc": datetime.now(UTC).isoformat(timespec="seconds"), "hosts": {}}
    tests = {
        "livetiming.formula1.com (FastF1 primary)":
            "https://livetiming.formula1.com/static/2026/2026-07-26_Hungarian_Grand_Prix/2026-07-26_Race/CarData.z.jsonStream",
        "livetiming-mirror.fastf1.dev (FastF1 mirror)":
            "https://livetiming-mirror.fastf1.dev/static/2026/2026-07-26_Hungarian_Grand_Prix/2026-07-26_Race/CarData.z.jsonStream",
        "api.openf1.org": OPENF1 + "meetings?year=2025",
    }
    for name, url in tests.items():
        try:
            r = requests.get(url, headers={"User-Agent": "BestHTTP"}, timeout=30, stream=True)
            out["hosts"][name] = {"http_status": r.status_code}
            r.close()
        except Exception as e:
            out["hosts"][name] = {"error": type(e).__name__}
    return out


# --------------------------------------------------------------------------- #
# Download + reduce (OpenF1)
# --------------------------------------------------------------------------- #

def _store_path(season: int, event: str, code: str) -> Path:
    return STORE / f"openf1_{season}_{event.replace(' ', '_')}_{code}.parquet"


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {"sessions": {}}


def _save_manifest(m: dict) -> None:
    STORE.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=1, sort_keys=True), encoding="utf-8")


def find_session_key(cl: Client, season: int, event: str, code: str) -> int:
    meetings = cl.get("meetings", f"year={season}", RAW / str(season) / "meetings.json.gz")
    keys = [m["meeting_key"] for m in meetings if _norm(m["meeting_name"]) == _norm(event)]
    name = SESSION_NAME[code]
    for mk in keys:
        sess = cl.get("sessions", f"meeting_key={mk}", RAW / str(season) / f"sessions_{mk}.json.gz")
        for s in sess:
            if s["session_name"] == name:
                return int(s["session_key"])
    raise RuntimeError(f"no OpenF1 {name} for {season} {event}")


def reduce_session(cl: Client, season: int, event: str, code: str, wanted: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Proxies for every wanted lap of one session, plus join diagnostics."""
    sk = find_session_key(cl, season, event, code)
    base = RAW / str(season) / str(sk)
    drivers = cl.get("drivers", f"session_key={sk}", base / "drivers.json.gz")
    acr = {d["name_acronym"]: int(d["driver_number"]) for d in drivers}
    laps = pd.DataFrame(cl.get("laps", f"session_key={sk}", base / "laps.json.gz"))
    diag = {"session_key": sk, "n_wanted": int(len(wanted)), "n_no_driver": 0, "n_no_openf1_lap": 0,
            "n_lap_time_mismatch": 0, "n_too_little_data": 0, "n_reduced": 0}
    if laps.empty:
        raise RuntimeError("OpenF1 returned no laps")
    laps = laps.dropna(subset=["date_start", "lap_duration"])
    laps["start_ns"] = pd.to_datetime(laps["date_start"], utc=True, format="ISO8601").astype("int64")
    j = wanted.assign(num=wanted["Driver"].map(acr))
    diag["n_no_driver"] = int(j["num"].isna().sum())
    j = j.dropna(subset=["num"]).astype({"num": int})
    j = j.merge(laps[["driver_number", "lap_number", "start_ns", "lap_duration"]],
                left_on=["num", "LapNumber"], right_on=["driver_number", "lap_number"], how="left")
    diag["n_no_openf1_lap"] = int(j["start_ns"].isna().sum())
    j = j.dropna(subset=["start_ns"])
    bad = (j["lap_duration"] - j["LapTimeSeconds"]).abs() > LAP_TIME_TOL_S
    diag["n_lap_time_mismatch"] = int(bad.sum())
    j = j[~bad]

    def one_driver(num: int, g: pd.DataFrame) -> list[dict]:
        lo = int(g["start_ns"].min()) - 2_000_000_000
        hi = int((g["start_ns"] + g["lap_duration"] * 1e9).max()) + 2_000_000_000
        f = lambda ns: pd.Timestamp(ns, tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")  # noqa: E731
        rows = cl.get("car_data", f"session_key={sk}&driver_number={num}&date>={f(lo)}&date<{f(hi)}",
                      base / f"car_{num}_{f(lo)}_{f(hi)}.json.gz".replace(":", ""))
        if not rows:
            return []
        c = pd.DataFrame(rows).sort_values("date")
        date_ns = pd.to_datetime(c["date"], utc=True, format="ISO8601").astype("int64").to_numpy()
        speed, thr, brk = c["speed"].to_numpy(), c["throttle"].to_numpy(), c["brake"].to_numpy()
        out = []
        for _, w in g.iterrows():
            end = int(w["start_ns"] + w["lap_duration"] * 1e9)
            frame = T.openf1_lap_frame(date_ns, speed, thr, brk, int(w["start_ns"]), end)
            p = T.lap_proxies(frame) if frame is not None else None
            out.append({"_ok": p is not None, **{k: w[k] for k in (
                "season", "event", "session", "side", "Driver", "LapNumber", "TyreLife", "Compound",
                "run_id", "race_progress", "LapTimeSeconds")}, **(p or {})})
        return out

    with ThreadPoolExecutor(N_WORKERS) as ex:
        futs = [ex.submit(one_driver, int(n), g) for n, g in j.groupby("num")]
        rows = [r for f in futs for r in f.result()]
    df = pd.DataFrame(rows)
    if not df.empty:
        diag["n_too_little_data"] = int((~df["_ok"]).sum())
        df = df[df["_ok"]].drop(columns="_ok")
    diag["n_reduced"] = int(len(df))
    return df, diag


def cmd_download(seasons, max_sessions: int | None = None) -> None:
    cl = Client()
    manifest = _load_manifest()
    selected = {s: wanted_laps(s) for s in seasons}
    done = 0
    for season, event, code in planned_sessions(seasons):
        key = f"{season}|{event}|{code}"
        path = _store_path(season, event, code)
        if path.exists() or manifest["sessions"].get(key, {}).get("status") == "ok":
            continue
        if max_sessions is not None and done >= max_sessions:
            break
        w = selected[season]
        w = w[(w["event"] == event) & (w["session"] == code)]
        t0 = time.time()
        try:
            df, diag = reduce_session(cl, season, event, code, w)
            STORE.mkdir(parents=True, exist_ok=True)
            df.to_parquet(path)
            manifest["sessions"][key] = {"status": "ok", **diag, "seconds": round(time.time() - t0, 1),
                                         "utc": datetime.now(UTC).isoformat(timespec="seconds")}
        except Exception as e:  # skip and log; one bad session must not stop the batch
            manifest["sessions"][key] = {"status": "failed", "error": str(e)[:300],
                                         "utc": datetime.now(UTC).isoformat(timespec="seconds")}
        manifest["client"] = {"requests": cl.n_requests, "retries": cl.n_retries,
                              "cache_hits": cl.n_cache_hits, "max_req_per_s": MAX_REQ_PER_S}
        _save_manifest(manifest)
        done += 1
        print(f"  {key}: {manifest['sessions'][key]['status']} "
              f"({manifest['sessions'][key].get('n_reduced', '-')}/{manifest['sessions'][key].get('n_wanted', '-')} laps)",
              flush=True)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def _git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _load_openf1(seasons) -> pd.DataFrame:
    fs = [f for f in sorted(STORE.glob("openf1_*.parquet")) if int(f.name.split("_")[1]) in seasons]
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def cross_check_sources() -> dict:
    """Same laps, two sources: do OpenF1 and FastF1 proxies agree?

    Only the 2026 Hungarian GP FP2 and Race exist in both. If the sources did
    not agree on the proxies there would be no point pooling any of it.
    """
    ff = [pd.read_parquet(f) for f in STORE.glob("fastf1_*.parquet")]
    of = [pd.read_parquet(f) for f in STORE.glob("openf1_2026_Hungarian*.parquet")]
    if not ff or not of:
        return {"available": False}
    k = ["event", "session", "Driver", "LapNumber"]
    m = pd.concat(ff).merge(pd.concat(of), on=k, suffixes=("_fastf1", "_openf1"))
    out = {"available": True, "n_common_laps": int(len(m)), "proxies": {}}
    for p in T.PROXIES:
        a, b = m[f"{p}_fastf1"], m[f"{p}_openf1"]
        ok = a.notna() & b.notna()
        if ok.sum() > 5:
            out["proxies"][p] = {"n": int(ok.sum()), "pearson": float(np.corrcoef(a[ok], b[ok])[0, 1]),
                                 "mean_fastf1": float(a[ok].mean()), "mean_openf1": float(b[ok].mean())}
    return out


def pilot_single_event(laps: pd.DataFrame, proxy: str, n_boot: int = 4000, n_perm: int = 20000) -> dict:
    """Exploratory run-clustered test within ONE event (used only if <4 events arrive)."""
    cells = T.adjusted_cells(laps, proxy)
    if len(cells) < 2:
        return {"estimable": False, "reason": "fewer than two cells meet the minimum run/lap counts"}
    rng = np.random.default_rng(SEED)
    d = T.purge_age_and_fuel(laps.dropna(subset=[proxy]), proxy)
    runs = d.groupby(["Compound", "side", "run_id"])["_adj"].mean().reset_index()
    comps = [c for c in T.SOFTNESS if c in set(runs["Compound"])]
    ci = {c: i for i, c in enumerate(comps)}
    comp = runs["Compound"].map(ci).to_numpy()
    side = (runs["side"] == "race").to_numpy()
    val = runs["_adj"].to_numpy(float)
    x = np.array([T.SOFTNESS[c] for c in comps], float)

    def contrast(comp_a, side_a, val_a):
        out = np.full(len(comps), np.nan)
        for k in range(len(comps)):
            r, p = val_a[(comp_a == k) & side_a], val_a[(comp_a == k) & ~side_a]
            if len(r) and len(p):
                out[k] = r.mean() - p.mean()
        ok = np.isfinite(out)
        if ok.sum() < 2:
            return float("nan"), out
        xx = x[ok] - x[ok].mean()
        return float(xx @ (out[ok] - out[ok].mean()) / (xx @ xx)), out

    obs, per = contrast(comp, side, val)
    groups = [g for g in (np.flatnonzero((comp == k) & (side == sd)) for k in range(len(comps)) for sd in (True, False)) if len(g)]
    boots = [b for b in (contrast(comp[i], side[i], val[i])[0]
                         for i in (np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
                                   for _ in range(n_boot))) if np.isfinite(b)]
    perm = []
    for _ in range(n_perm):
        c2 = comp.copy()
        for sd in (True, False):
            m = np.flatnonzero(side == sd)
            c2[m] = rng.permutation(comp[m])
        b, _ = contrast(c2, side, val)
        if np.isfinite(b):
            perm.append(b)
    perm = np.asarray(perm)
    sign = T.PREDICTED_SIGN[proxy]
    return {
        "estimable": True, "slope_per_softness_step_run_level": obs,
        "per_compound_run_level_contrast": {c: float(per[ci[c]]) for c in comps},
        "ci95_run_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "p_perm_one_sided": float((1 + np.sum(perm * sign >= obs * sign)) / (1 + len(perm))) if sign else None,
        "p_perm_two_sided": float((1 + np.sum(np.abs(perm) >= abs(obs))) / (1 + len(perm))),
        "adjusted_cells": cells.round(5).to_dict("records"),
    }


def analyse_sample(laps: pd.DataFrame, seasons, label: str) -> dict:
    """The full pre-specified analysis for one set of seasons."""
    n_events = int(laps[["season", "event"]].drop_duplicates().shape[0])
    sec = {
        "seasons": list(seasons),
        "n_laps": int(len(laps)), "n_events": n_events,
        "laps_by_side_compound": {f"{s}|{c}": int(n) for (s, c), n in laps.groupby(["side", "Compound"]).size().items()},
        "runs_by_side_compound": {f"{s}|{c}": int(n) for (s, c), n in laps.groupby(["side", "Compound"])["run_id"].nunique().items()},
        "proxy_distribution_all_laps": {
            p: {q: float(laps[p].quantile(v)) for q, v in (("p05", .05), ("median", .5), ("p95", .95))}
            | {"frac_missing": float(laps[p].isna().mean())} for p in T.PROXIES},
    }
    try:
        ratios = T.degradation_ratio_cells({s: LAP_FILES[s] for s in seasons})
        sec["n_degradation_ratio_cells"] = int(len(ratios))
    except Exception as e:  # the ratio build belongs to other modules; do not die on it
        ratios = None
        sec["ratio_cells_error"] = str(e)[:200]
    by = {}
    for proxy in T.PROXIES:
        entry: dict = {"predicted_sign": T.PREDICTED_SIGN[proxy]}
        cells = T.adjusted_cells(laps, proxy)
        entry["cells"] = cells.round(5).to_dict("records")
        for lab, sub in (("all", cells), *((str(s), cells[cells["season"] == s]) for s in seasons)):
            tr = T.trend_test(sub, T.PREDICTED_SIGN[proxy], seed=SEED) if len(sub) else None
            entry[f"trend_{lab}"] = None if tr is None else {
                "n_cells": tr.n_cells, "n_events": tr.n_events, "slope": tr.slope,
                "ci95_event_bootstrap": list(tr.ci95), "p_perm_one_sided": tr.p_perm_one_sided,
                "p_perm_two_sided": tr.p_perm_two_sided, "n_perm": tr.n_perm,
                "perm_exact": tr.perm_exact, "per_compound": tr.per_compound}
        if ratios is not None and len(ratios):
            entry["link_to_degradation_ratio"] = T.link_to_ratio(cells, ratios, seed=SEED)
        if n_events < 4:
            entry["pilot_single_event"] = pilot_single_event(laps, proxy)
        by[proxy] = entry
    sec["proxies"] = by
    return sec


def _headline(sec: dict) -> str:
    tr = sec["proxies"][T.PRIMARY].get("trend_all")
    if tr is None:
        return "insufficient within-event compound variation for the primary proxy"
    lo, hi = tr["ci95_event_bootstrap"]
    return (f"coast_s slope {tr['slope']:+.3f} s per softness step over {tr['n_events']} events "
            f"(95% event-bootstrap CI {lo:+.3f} to {hi:+.3f}; one-sided perm p {tr['p_perm_one_sided']:.4f}, "
            f"two-sided {tr['p_perm_two_sided']:.4f})")


def cmd_analyse() -> None:
    manifest = _load_manifest()
    sessions = manifest["sessions"]
    result: dict = {
        "meta": {
            "script": "scripts/paper/p6_telemetry.py", "module": "src/tyredeg/paper/telemetry.py",
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"), "git_head": _git_head(),
            "python": sys.version.split()[0], "pandas": pd.__version__, "numpy": np.__version__,
            "seed": SEED,
            "data_source": {
                "primary": "OpenF1 public API https://api.openf1.org/v1 (car_data, laps, drivers, sessions, meetings)",
                "why_switched": "The F1 live-timing static host used by FastF1 returned HTTP 401 for every "
                                "session file and FastF1's mirror 404 (see access.probe); only 2026 Hungarian GP "
                                "FP2/Race telemetry was in the FastF1 cache. Switched to OpenF1 after the "
                                "pre-specified proxies and tests were frozen; proxies/thresholds are unchanged.",
                "join": "meeting by (year, event name); session by name; driver by acronym; lap by lap_number; "
                        "lap cut at OpenF1 date_start for lap_duration; laps with |OpenF1 - FastF1 lap time| > 0.3 s dropped",
                "rate_limit": f"<= {MAX_REQ_PER_S} requests/s, exponential backoff on 429/5xx, raw responses cached gzipped",
                "client": manifest.get("client"),
            },
            "thresholds": {k: getattr(T, k) for k in dir(T) if k.isupper() and isinstance(getattr(T, k), (int, float))},
            "proxies": list(T.PROXIES), "primary_proxy": T.PRIMARY,
            "predicted_sign_race_minus_practice_rising_with_softness": T.PREDICTED_SIGN,
            "primary_sample": "2025 and 2026, Race + FP2 per event, long-run green laps (practice: race-sim runs)",
            "extension_sample": "2023 and 2024 on the same rules; reported separately, never pooled into the primary",
        },
        "access": {"probe": probe_access(),
                   "sessions_planned_primary": len(planned_sessions(PRIMARY_SEASONS)),
                   "sessions_planned_extension": len(planned_sessions(EXTENSION_SEASONS)),
                   "sessions": sessions},
    }
    for tag, seasons in (("primary", PRIMARY_SEASONS), ("extension", EXTENSION_SEASONS)):
        planned = {f"{s}|{e}|{c}" for s, e, c in planned_sessions(seasons)}
        st = [sessions.get(k, {}) for k in planned]
        result["access"][f"{tag}_n_ok"] = sum(x.get("status") == "ok" for x in st)
        result["access"][f"{tag}_n_failed"] = sum(x.get("status") == "failed" for x in st)
        result["access"][f"{tag}_n_not_attempted"] = sum(not x for x in st)
        result["access"][f"{tag}_failed_sessions"] = sorted(k for k in planned if sessions.get(k, {}).get("status") == "failed")
        ok = [x for x in st if x.get("status") == "ok"]
        w = sum(x["n_wanted"] for x in ok)
        result["access"][f"{tag}_join"] = {
            k: int(sum(x[k] for x in ok)) for k in (
                "n_wanted", "n_no_driver", "n_no_openf1_lap", "n_lap_time_mismatch", "n_too_little_data", "n_reduced")
        } | {"fraction_reduced": (sum(x["n_reduced"] for x in ok) / w) if w else None}
    result["cross_check_sources"] = cross_check_sources()

    for tag, seasons in (("primary", PRIMARY_SEASONS), ("extension", EXTENSION_SEASONS)):
        laps = _load_openf1(seasons)
        if laps.empty:
            result[tag] = {"seasons": list(seasons), "status": "no data"}
            continue
        result[tag] = analyse_sample(laps, seasons, tag)
        result[tag]["headline"] = _headline(result[tag])
        result[tag]["events"] = sorted(f"{s} {e}" for s, e in laps[["season", "event"]].drop_duplicates().itertuples(index=False))
    result["verdict"] = "PRIMARY: " + result.get("primary", {}).get("headline", "no data")
    ext = result.get("extension", {})
    if ext.get("headline"):
        result["verdict"] += " | EXTENSION: " + ext["headline"]
    _write(result)


def _clean(o):
    """JSON has no NaN: write missing numbers as null, numpy scalars as Python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _write(res: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_clean(res), indent=1, allow_nan=False), encoding="utf-8")
    print(f"wrote {OUT}")
    print(res.get("verdict", ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["probe", "download", "analyse", "all"])
    ap.add_argument("--seasons", type=int, nargs="*", default=list(PRIMARY_SEASONS))
    ap.add_argument("--max-sessions", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "probe":
        print(json.dumps(probe_access(), indent=1))
        return
    if a.cmd in ("download", "all"):
        cmd_download(a.seasons, a.max_sessions)
    if a.cmd in ("analyse", "all"):
        cmd_analyse()


if __name__ == "__main__":
    main()

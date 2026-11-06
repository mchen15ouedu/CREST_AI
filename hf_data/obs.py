"""USGS observed streamflow — for real skill scores (task #6 bundle).

Fetches observed discharge from the USGS waterservices instantaneous-values API
(parameter 00060, cfs -> m³/s) and writes the EF5 OBS file
`USGS_<site>_UTC_m3s.csv` (time,value per line — matches EF5's `%[^,],%f`
reader and AQUAH's format). Real runs point the [Gauge] OBS= at this file so
EF5 populates the 'Observed' column of ts.csv, giving meaningful NSCE/CC/bias.

CACHED (V18.7): get_series() is the entry point runs should use. Each gauge's
observations live in a parquet store `_cache/obs/<site>.parquet` (time + q
columns, covered time-windows recorded in the schema metadata). A request
serves cached rows and hits NWIS only for the parts of the window the store
has never seen; the last CREST_OBS_LAG_H hours (default 24) are never marked
covered, so the provisional near-real-time tail refreshes on the next request.
An empty answer is marked covered only when stored rows bracket that window
on BOTH sides (a site that reports before and after it): NWIS also answers
200 with an empty series when it fails quietly — a gap open towards now is
what that looks like — and a covered window is never asked again. The obs/
store syncs to the private CREST_state dataset
(persist.SYNC_DIRS), so observations survive Space restarts. Calibration is
the big winner: all candidate runs of a gauge reuse one download instead of
re-fetching NWIS.
"""
from __future__ import annotations

import bisect
import json
import os
import threading
from datetime import datetime, timedelta, timezone

import truststore
truststore.inject_into_ssl()
import requests

CFS_TO_CMS = 0.0283168
_IV_URL = "https://waterservices.usgs.gov/nwis/iv/"
LAG_H = float(os.environ.get("CREST_OBS_LAG_H", "24"))   # provisional-tail window
# an empty answer counts as covered only when the store holds a row within
# this before AND after the window (the fetch's ±1-day pad plus a day):
# get_series, _bracketed
_NEAR = timedelta(days=2)

_locks: dict[str, threading.Lock] = {}                    # per-site store lock
_locks_lock = threading.Lock()


def _obs_dir() -> str:
    from hf_data.statecache import CACHE_DIR
    return os.path.join(CACHE_DIR, "obs")


def _site_lock(site: str) -> threading.Lock:
    with _locks_lock:
        return _locks.setdefault(site, threading.Lock())


def _store_lock(site: str):
    """Cross-process lock on one gauge's store (taken inside _site_lock): the
    dashboard, fleet workers and calibration runs are separate processes, and
    two that load-fetch-save the same store at once lose the rows the first
    save added. OS-level lock - released if the holder dies."""
    from filelock import FileLock
    os.makedirs(_obs_dir(), exist_ok=True)
    return FileLock(_store_path(site) + ".lock")


def fetch_usgs_discharge(site: str, t_start: datetime, t_end: datetime) -> list[tuple[datetime, float]]:
    """Observed discharge (m³/s) time series, or [] if unavailable."""
    from datetime import timedelta
    # pad a day each side: startDT/endDT are LOCAL dates at the gauge, but the
    # window is UTC — the first UTC hours live on the previous local day
    params = {"sites": str(site).zfill(8), "parameterCd": "00060", "format": "json",
              "startDT": (t_start - timedelta(days=1)).strftime("%Y-%m-%d"),
              "endDT": (t_end + timedelta(days=1)).strftime("%Y-%m-%d"),
              "siteStatus": "all"}
    r = requests.get(_IV_URL, params=params, timeout=30)
    r.raise_for_status()
    ts = r.json().get("value", {}).get("timeSeries", [])
    if not ts:
        return []
    out = []
    for v in ts[0]["values"][0]["value"]:
        try:
            cfs = float(v["value"])
        except (TypeError, ValueError):
            continue
        if cfs < 0:                                  # USGS missing sentinel (-999999)
            continue
        # USGS IV timestamps carry the gauge's LOCAL utc-offset (e.g.
        # "...T01:15:00.000-05:00") — convert to UTC before dropping the tz,
        # or the obs are shifted 4-10 h against the UTC forcing + simulation
        dt = (datetime.fromisoformat(v["dateTime"].replace("Z", "+00:00"))
              .astimezone(timezone.utc).replace(tzinfo=None))
        out.append((dt, cfs * CFS_TO_CMS))
    return out


# ---- parquet obs store (per gauge, coverage-aware) ---------------------------
_TS_FMT = "%Y-%m-%dT%H:%M:%S"


def _store_path(site: str) -> str:
    return os.path.join(_obs_dir(), f"{str(site).zfill(8)}.parquet")


def _load_store(site: str) -> tuple[dict[datetime, float], list[list[datetime]]]:
    """Return (rows keyed by time, covered [start,end] intervals)."""
    p = _store_path(site)
    if not os.path.isfile(p):
        return {}, []
    import pyarrow.parquet as pq
    try:
        tbl = pq.read_table(p)
        rows = dict(zip(tbl.column("time").to_pylist(), tbl.column("q").to_pylist()))
        meta = (tbl.schema.metadata or {}).get(b"coverage", b"[]")
        cov = [[datetime.strptime(a, _TS_FMT), datetime.strptime(b, _TS_FMT)]
               for a, b in json.loads(meta)]
        return rows, cov
    except Exception:
        return {}, []                        # corrupt store -> refetch from scratch


def _save_store(site: str, rows: dict[datetime, float], cov: list[list[datetime]]):
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(_obs_dir(), exist_ok=True)
    times = sorted(rows)
    meta = {b"coverage": json.dumps([[a.strftime(_TS_FMT), b.strftime(_TS_FMT)]
                                     for a, b in cov]).encode()}
    # float64: bit-exact round-trip vs the live fetch (stores are tiny anyway)
    schema = pa.schema([pa.field("time", pa.timestamp("s")),
                        pa.field("q", pa.float64())]).with_metadata(meta)
    tbl = pa.table({"time": times, "q": [rows[t] for t in times]}, schema=schema)
    from hf_data.atomicfile import atomic_path
    with atomic_path(_store_path(site)) as tmp:     # unique per call; threads + processes
        pq.write_table(tbl, tmp, compression="zstd")


def _merge_cov(cov: list[list[datetime]]) -> list[list[datetime]]:
    """Union of intervals (windows within 1 h of each other fuse)."""
    out: list[list[datetime]] = []
    for a, b in sorted(cov):
        if out and a <= out[-1][1] + timedelta(hours=1):
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _gaps(cov: list[list[datetime]], t0: datetime, t1: datetime) -> list[list[datetime]]:
    """Portions of [t0,t1] NOT covered (sub-hour slivers ignored)."""
    gaps, cur = [], t0
    for a, b in cov:
        if b <= cur or a >= t1:
            continue
        if a > cur:
            gaps.append([cur, min(a, t1)])
        cur = max(cur, b)
        if cur >= t1:
            break
    if cur < t1:
        gaps.append([cur, t1])
    return [g for g in gaps if g[1] - g[0] >= timedelta(hours=1)]


def _inside(times: list[datetime], a: datetime, b: datetime) -> bool:
    """True when a row (sorted `times`) lies strictly inside (a, b) — a row
    at an edge brackets the window (_bracketed)."""
    i = bisect.bisect_right(times, a)
    return i < len(times) and times[i] < b


def _bracketed(times: list[datetime], a: datetime, b: datetime) -> bool:
    """True when rows (sorted `times`) bracket [a, b] on BOTH sides: one in
    [a - _NEAR, a] and one in [b, b + _NEAR]. A window with rows on one side
    only — a gap open towards now above all — is what NWIS failing quietly
    looks like, not a site that reports around a gap in its record."""
    i = bisect.bisect_left(times, a - _NEAR)
    j = bisect.bisect_left(times, b)
    return (i < len(times) and times[i] <= a) and (j < len(times) and times[j] <= b + _NEAR)


def _trusted(cov: list[list[datetime]], rows: dict) -> list[list[datetime]]:
    """Coverage that stands: windows with a row in them, or rowless ones that
    rows bracket (_bracketed). Any other rowless window is what an empty
    answer marked before get_series stopped trusting those (stores synced
    from CREST_state still carry them): it is asked again."""
    times = sorted(rows)
    return [[a, b] for a, b in cov if _inside(times, a, b) or _bracketed(times, a, b)]


def get_series(site: str, t_start: datetime, t_end: datetime,
               info: dict | None = None,
               refresh: bool = False) -> list[tuple[datetime, float]]:
    """Observed discharge (m³/s) for [t_start, t_end] — cached-first.

    Serves rows from the parquet store and fetches only the windows NWIS has
    never been asked for. Never raises on fetch errors: returns whatever the
    store already holds (failed gaps stay uncovered and retry next time).
    Coverage is never marked inside the last LAG_H hours, so provisional
    recent obs refresh on later requests, nor for an answer without a row in
    the window unless stored rows bracket it on both sides (_bracketed), nor
    for an empty answer for a window the store still holds rows in (_trusted
    drops rowless coverage an older store still carries the same way).
    `refresh` asks NWIS for the whole window again, covered or not (a fresh
    answer replaces the window's rows; an empty one never drops covered
    rows). `info` (optional
    dict) gets {"cached": bool, "fetched_windows": n, "empty_windows": n
    (answered without a row), "from_store": bool (part of the window came
    from the store's earlier answers, not a request of this call),
    "fetch_error": str?, "fetch_exc": the exception?} for status text and
    the caller's outage rule.
    """
    site = str(site).zfill(8)
    with _site_lock(site), _store_lock(site):  # this process's threads, then other processes
        rows, stored = _load_store(site)
        cov = _trusted(stored, rows)
        changed = cov != stored
        gaps = _gaps(cov, t_start, t_end)
        from_store = gaps != [[t_start, t_end]]
        if refresh and t_end > t_start:
            gaps, from_store = [[t_start, t_end]], False
        held = list(cov)                     # coverage before this call's answers
        fetched = empty = 0
        for a, b in gaps:
            try:
                got = fetch_usgs_discharge(site, a, b)
            except Exception as e:
                if info is not None:
                    info.update({"fetch_error": str(e), "fetch_exc": e})
                continue                     # gap stays uncovered -> retry later
            got = dict(got)
            # replace, don't merge: USGS scrubs bad provisional values by
            # DELETING them, so cached rows in the refetched window that the
            # fresh answer no longer carries must go too (seen live: a phantom
            # 38,400 cfs flood at a dry TX gauge survived in the store — and
            # the scorecard — long after NWIS had removed it). An empty
            # answer leaves covered rows alone (only a refresh asks a covered
            # window again): it may be NWIS failing quietly
            for t in [t for t in rows if a <= t <= b and t not in got
                      and (got or not any(c <= t <= d for c, d in held))]:
                del rows[t]
                changed = True
            rows.update(got)
            changed = changed or bool(got)
            # no row in the window is an answer (the gauge has no record
            # there) only from a site that reports on BOTH sides of it: NWIS
            # also answers 200 with an empty series when it fails quietly —
            # a gap open towards now is what that looks like — and a covered
            # window is never asked again, so without rows bracketing it the
            # window stays uncovered and the next call asks again. An empty
            # answer for a window the store still holds rows in (a refresh of
            # a covered one) covers nothing new either. Never covered inside
            # the provisional tail
            cap = (datetime.now(timezone.utc).replace(tzinfo=None)
                   - timedelta(hours=LAG_H))
            times, cb = sorted(rows), min(b, cap)
            if a < cap and ((_inside(times, a, cb) or _bracketed(times, a, cb)) if got
                            else _bracketed(times, a, cb) and not _inside(times, a, cb)):
                cov.append([a, cb])
                changed = True
            fetched += 1
            empty += not got
        if changed:
            _save_store(site, rows, _merge_cov(cov))
        if info is not None:
            info.update({"cached": not gaps, "fetched_windows": fetched,
                         "empty_windows": empty, "from_store": from_store})
    # match fetch_usgs_discharge's ±1-day pad so EF5 obs/coverage behave the same
    lo, hi = t_start - timedelta(days=1), t_end + timedelta(days=1)
    return sorted((t, q) for t, q in rows.items() if lo <= t <= hi)


def write_ef5_obs(site: str, series: list[tuple[datetime, float]], out_dir: str) -> str | None:
    """Write USGS_<site>_UTC_m3s.csv (EF5 OBS). Returns the path, or None if empty."""
    if not series:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"USGS_{str(site).zfill(8)}_UTC_m3s.csv")
    with open(path, "w") as f:
        f.write("datetime,discharge\n")
        for dt, cms in series:
            f.write(f"{dt:%Y-%m-%d %H:%M:%S},{cms:.6f}\n")
    return path


if __name__ == "__main__":
    import sys
    site = sys.argv[1] if len(sys.argv) > 1 else "08144500"
    s = fetch_usgs_discharge(site, datetime(2024, 6, 1), datetime(2024, 6, 4))
    print(f"USGS {site}: {len(s)} obs points")
    if s:
        vals = [v for _, v in s]
        print(f"  {s[0][0]} .. {s[-1][0]}  range [{min(vals):.2f}, {max(vals):.2f}] m³/s")

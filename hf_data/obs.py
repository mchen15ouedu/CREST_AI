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
The obs/ store syncs to the private CREST_state dataset (persist.SYNC_DIRS),
so observations survive Space restarts. Calibration is the big winner: all
candidate runs of a gauge reuse one download instead of re-fetching NWIS.
"""
from __future__ import annotations

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

_locks: dict[str, threading.Lock] = {}                    # per-site store lock
_locks_lock = threading.Lock()


def _obs_dir() -> str:
    from hf_data.statecache import CACHE_DIR
    return os.path.join(CACHE_DIR, "obs")


def _site_lock(site: str) -> threading.Lock:
    with _locks_lock:
        return _locks.setdefault(site, threading.Lock())


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
    tmp = _store_path(site) + ".tmp"
    pq.write_table(tbl, tmp, compression="zstd")
    os.replace(tmp, _store_path(site))


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


def get_series(site: str, t_start: datetime, t_end: datetime,
               info: dict | None = None) -> list[tuple[datetime, float]]:
    """Observed discharge (m³/s) for [t_start, t_end] — cached-first.

    Serves rows from the parquet store and fetches only the windows NWIS has
    never been asked for. Never raises on fetch errors: returns whatever the
    store already holds (failed gaps stay uncovered and retry next time).
    Coverage is never marked inside the last LAG_H hours, so provisional
    recent obs refresh on later requests. `info` (optional dict) gets
    {"cached": bool, "fetched_windows": n, "fetch_error": str?} for status text.
    """
    site = str(site).zfill(8)
    with _site_lock(site):
        rows, cov = _load_store(site)
        gaps = _gaps(cov, t_start, t_end)
        fetched = 0
        for a, b in gaps:
            try:
                got = fetch_usgs_discharge(site, a, b)
            except Exception as e:
                if info is not None:
                    info["fetch_error"] = str(e)
                continue                     # gap stays uncovered -> retry later
            rows.update(dict(got))
            # even an empty answer is an answer (gauge has no record there) —
            # mark covered so we stop hammering NWIS, but never inside the
            # provisional tail
            cap = (datetime.now(timezone.utc).replace(tzinfo=None)
                   - timedelta(hours=LAG_H))
            if a < cap:
                cov.append([a, min(b, cap)])
            fetched += 1
        if fetched:
            _save_store(site, rows, _merge_cov(cov))
        if info is not None:
            info.update({"cached": not gaps, "fetched_windows": fetched})
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

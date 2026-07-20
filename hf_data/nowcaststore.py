"""Precomputed fleet nowcasts — read side of scripts/run_nowcast_all.py.

The updater Space refreshes nowcast/latest.parquet in vincewin/CREST_data
hourly (DI-LSTM +6 h predictions for every CONUS GAGES-II gauge, issue time
t0 = newest MRMS Pass1 hour). This module serves it to the dashboard with a
short in-process TTL cache; hf_hub_download's etag check makes the refresh a
cheap no-op until the Space actually uploads a new file.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np

REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")
TTL_S = 240
H = 6

_lock = threading.Lock()
_cache: dict = {"at": 0.0, "meta": None, "cols": None}
_thr: dict = {"at": 0.0, "map": None}     # gid -> Q10 (m3/s); static file
THR_TTL_S = 3600


def _load():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, "nowcast/latest.parquet", repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"))
    t = pq.read_table(p)
    meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()
            if not k.startswith(b"ARROW")}
    cols = {name: t.column(name).to_numpy(zero_copy_only=False)
            for name in t.schema.names}
    return meta, cols


def _fresh():
    now = time.time()
    with _lock:
        if _cache["cols"] is not None and now - _cache["at"] < TTL_S:
            return _cache["meta"], _cache["cols"]
    try:
        meta, cols = _load()
        with _lock:
            _cache.update(at=now, meta=meta, cols=cols)
        return meta, cols
    except Exception:
        with _lock:                      # serve stale rather than nothing
            return _cache["meta"], _cache["cols"]


def _thresholds() -> dict:
    """10-yr return flood per gauge (scripts/compute_flood_thresholds.py)."""
    now = time.time()
    with _lock:
        if _thr["map"] is not None and now - _thr["at"] < THR_TTL_S:
            return _thr["map"]
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, "nowcast/flood_thresholds.parquet",
                            repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        t = pq.read_table(p)
        q10 = t.column("q10_cms").to_numpy(zero_copy_only=False)
        # sub-0.5 m3/s "10-yr floods" (tiny/ephemeral creeks) are below model
        # noise — flagging them would paint phantom floods on the map
        m = {g: float(q) for g, q in zip(t.column("gid").to_pylist(), q10)
             if np.isfinite(q) and q >= 0.5}
        with _lock:
            _thr.update(at=now, map=m)
        return m
    except Exception:
        with _lock:
            return _thr["map"] or {}


def all_risk() -> dict:
    """CONUS-wide flood risk: next-6-h peak vs the 10-yr threshold, every gauge.

    Feeds the Nowcast-mode map: `flood` (lat, lon, ratio, gid — only gauges
    predicted OVER threshold) draws the red density layer; `ratios` colors
    the pins when zoomed in."""
    meta, cols = _fresh()
    if cols is None:
        return {"ok": False, "reason": "no precomputed nowcast available yet"}
    thr = _thresholds()
    if not thr:
        return {"ok": False, "reason": "no flood thresholds computed yet"}
    qmax = np.max(np.stack([cols[f"q{k + 1}"] for k in range(H)], 1), 1)
    ratios, flood = {}, []
    for i in range(len(cols["gid"])):
        gid = str(cols["gid"][i])
        q10 = thr.get(gid)
        if not q10:
            continue
        r = float(qmax[i]) / q10
        ratios[gid] = round(r, 3)
        if r >= 1.0:
            flood.append([round(float(cols["lat"][i]), 4),
                          round(float(cols["lon"][i]), 4),
                          round(min(r, 5.0), 2), gid])
    return {"ok": True, "t0": meta.get("t0"), "generated": meta.get("generated"),
            "n_flood": len(flood), "n_rated": len(ratios),
            "flood": flood, "ratios": ratios}


def for_bbox(w: float, s: float, e: float, n: float, limit: int = 100,
             obs_hours: int = 0, ids: str = "") -> dict:
    """Nowcasts for every gauge inside the bbox (largest basins first), or —
    when `ids` (comma list) is given — exactly those gauges, bbox ignored.

    obs_hours > 0 additionally attaches each gauge's recent observed series
    (store-first via hf_data.obs, live NWIS fills the tail) so the frontend
    can plot context + prediction in one request — only honored for <= 25
    gauges to keep the response fast."""
    meta, cols = _fresh()
    if cols is None:
        return {"ok": False, "reason": "no precomputed nowcast available yet"}
    if ids:
        want = [s2.strip().zfill(8) for s2 in ids.split(",") if s2.strip()]
        m = np.isin(cols["gid"], want)
    else:
        m = ((cols["lon"] >= w) & (cols["lon"] <= e)
             & (cols["lat"] >= s) & (cols["lat"] <= n))
    idx = np.nonzero(m)[0]
    total = int(len(idx))
    idx = idx[np.argsort(-cols["area_km2"][idx])][:max(1, limit)]
    try:
        t0 = datetime.strptime(meta.get("t0", ""), "%Y-%m-%d %H:%M UTC")
    except ValueError:
        t0 = None
    times = ([(t0 + timedelta(hours=k + 1)).strftime("%Y-%m-%d %H:%M")
              for k in range(H)] if t0 else [])
    thr = _thresholds()
    gauges = []
    for i in idx:
        q = [round(float(cols[f"q{k + 1}"][i]), 3) for k in range(H)]
        age = float(cols["obs_age_h"][i])
        lq = float(cols["obs_last_q"][i])
        gid = str(cols["gid"][i])
        q10 = thr.get(gid)
        gauges.append({
            "id": gid, "lat": round(float(cols["lat"][i]), 5),
            "lon": round(float(cols["lon"][i]), 5),
            "area_km2": round(float(cols["area_km2"][i]), 1),
            "q": q,
            "q10": round(q10, 1) if q10 else None,
            "flood": bool(q10 and max(q) >= q10),
            "obs_last_time": str(cols["obs_last_time"][i]) or None,
            "obs_last_q": None if np.isnan(lq) else round(lq, 3),
            "obs_age_h": None if age >= 999 else round(age, 1),
        })
    if obs_hours > 0 and len(gauges) <= 25:
        from concurrent.futures import ThreadPoolExecutor
        from hf_data import obs as _obs
        end = datetime.utcnow()
        start = end - timedelta(hours=min(obs_hours, 168))

        def _series(g):
            try:
                rows = _obs.get_series(g["id"], start, end)   # already windowed
                step = max(1, len(rows) // 800)               # bound payload, keep
                rows = rows[::step] if step > 1 else rows     # the full time span
                g["obs"] = [[t.strftime("%Y-%m-%d %H:%M"), round(q, 3)]
                            for t, q in rows]
            except Exception:
                g["obs"] = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_series, gauges))
    return {"ok": True, "t0": meta.get("t0"), "generated": meta.get("generated"),
            "times": times, "n_in_view": total, "truncated": total > len(gauges),
            "model": {"epoch": meta.get("model_epoch"),
                      "val_nse": meta.get("model_val_nse"),
                      "when": meta.get("model_when"), "experimental": True},
            "gauges": gauges}

"""Precomputed fleet nowcasts — read side of scripts/run_nowcast_all.py.

The updater Space refreshes nowcast/latest.parquet in vincewin/CREST_data
hourly (DI-LSTM hourly predictions q1..qN — currently N=12 — for every CONUS
GAGES-II gauge, issue time t0 = newest MRMS Pass1 hour). This module serves
it to the dashboard with a short in-process TTL cache; hf_hub_download's
etag check makes the refresh a cheap no-op until the Space actually uploads
a new file. Risk tiers (pins/heatmap) stay on the FIRST RISK_H hours so the
map coloring is unchanged as the model horizon grows.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timedelta

import numpy as np

REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")
TTL_S = 240
RISK_H = 6           # tier window: pins/heatmap use the first 6 forecast hours


def _qcols(cols) -> list:
    """Ordered 6-h-model columns q1..qN (risk basis) in latest.parquet."""
    return sorted((n for n in cols if n[0] == "q" and n[1:].isdigit()),
                  key=lambda n: int(n[1:]))


def _q12cols(cols) -> list:
    """Ordered 12-h-model columns q12_1..q12_N (hydrograph-only, own trace)."""
    return sorted((n for n in cols if n.startswith("q12_") and n[4:].isdigit()),
                  key=lambda n: int(n[4:]))

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


BASE_RISE = 5.0                    # yellow tier: predicted >= 5 x baseflow
NOISE_Q = 0.5                      # m3/s floor — below this nothing is flagged


def _thresholds() -> dict:
    """Per-gauge thresholds {gid: (qbase, q2, q5, q10)} — NaN where unknown.
    From scripts/compute_flood_thresholds.py (annual-peak LP3 + 3-yr median)."""
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
        names = set(t.schema.names)

        def col(n):
            return (t.column(n).to_numpy(zero_copy_only=False) if n in names
                    else np.full(len(t), np.nan, "float32"))
        m = {}
        q2, q5, q10, qb = col("q2_cms"), col("q5_cms"), col("q10_cms"), col("qbase_cms")
        for g, b, a2, a5, a10 in zip(t.column("gid").to_pylist(), qb, q2, q5, q10):
            m[g] = (float(b), float(a2), float(a5), float(a10))
        with _lock:
            _thr.update(at=now, map=m)
        return m
    except Exception:
        with _lock:
            return _thr["map"] or {}


def _tier(qmax: float, thr: tuple) -> int:
    """0 quiet · 1 elevated · 2 minor flood (>= Q2) · 3 flood (>= Q5).

    Elevated needs BOTH >= BASE_RISE x baseflow AND >= 10% of bankfull (Q2):
    arid/intermittent rivers have p25 baseflow near zero, so a bare multiple
    would flag them yellow at a trickle. Sub-NOISE_Q flows never flag."""
    qb, q2, q5, _ = thr
    if qmax < NOISE_Q:
        return 0
    if np.isfinite(q5) and q5 >= NOISE_Q and qmax >= q5:
        return 3
    if np.isfinite(q2) and q2 >= NOISE_Q and qmax >= q2:
        return 2
    if np.isfinite(qb):
        thr_y = max(BASE_RISE * qb, NOISE_Q)
        if np.isfinite(q2):
            thr_y = max(thr_y, 0.10 * q2)
        if qmax >= thr_y:
            return 1
    return 0


def all_risk() -> dict:
    """CONUS-wide tiered risk: next-6-h AI peak vs baseflow / Q2 / Q5.

    Feeds the Nowcast-mode map: `flagged` [[lat, lon, tier, gid], ...] draws
    the density layers; `tiers` {gid: tier} colors the pins when zoomed in."""
    meta, cols = _fresh()
    if cols is None:
        return {"ok": False, "reason": "no precomputed nowcast available yet"}
    thr = _thresholds()
    if not thr:
        return {"ok": False, "reason": "no flood thresholds computed yet"}
    qmax = np.max(np.stack([cols[n] for n in _qcols(cols)[:RISK_H]], 1), 1)
    tiers, flagged = {}, []
    counts = [0, 0, 0]
    for i in range(len(cols["gid"])):
        gid = str(cols["gid"][i])
        th = thr.get(gid)
        if th is None:
            continue
        tr = _tier(float(qmax[i]), th)
        tiers[gid] = tr
        if tr > 0:
            counts[tr - 1] += 1
            flagged.append([round(float(cols["lat"][i]), 4),
                            round(float(cols["lon"][i]), 4), tr, gid])
    return {"ok": True, "t0": meta.get("t0"), "generated": meta.get("generated"),
            "n_elevated": counts[0], "n_minor": counts[1], "n_flood": counts[2],
            "n_rated": len(tiers), "flagged": flagged, "tiers": tiers}


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
    qs = _qcols(cols)
    q12s = _q12cols(cols)
    try:
        t0 = datetime.strptime(meta.get("t0", ""), "%Y-%m-%d %H:%M UTC")
    except ValueError:
        t0 = None
    times = ([(t0 + timedelta(hours=k + 1)).strftime("%Y-%m-%d %H:%M")
              for k in range(max(len(qs), len(q12s)))] if t0 else [])
    thr = _thresholds()
    gauges = []
    for i in idx:
        q = [round(float(cols[n][i]), 3) for n in qs]
        q12 = [round(float(cols[n][i]), 3) for n in q12s]
        age = float(cols["obs_age_h"][i])
        lq = float(cols["obs_last_q"][i])
        gid = str(cols["gid"][i])
        th = thr.get(gid)

        def _f(v):
            return round(v, 2) if th is not None and np.isfinite(v) else None
        gauges.append({
            "id": gid, "lat": round(float(cols["lat"][i]), 5),
            "lon": round(float(cols["lon"][i]), 5),
            "area_km2": round(float(cols["area_km2"][i]), 1),
            "q": q, "q12": q12 or None,
            "qbase": _f(th[0]) if th else None, "q2": _f(th[1]) if th else None,
            "q5": _f(th[2]) if th else None, "q10": _f(th[3]) if th else None,
            "tier": _tier(max(q[:RISK_H]), th) if th else 0,
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

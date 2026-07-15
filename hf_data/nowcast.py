"""AI nowcast client — asks the CREST_nowcast Space (DI-LSTM backend) for a
short-lead corrected hydrograph tail after a simulation finishes.

The companion Space (vincewin/CREST_nowcast, ZeroGPU) hosts an hourly
data-integration LSTM (Feng, Fang & Shen 2020 method): MRMS basin-mean
precipitation + the most recent USGS observation (with its age) in, next-6-h
discharge out. This client assembles the payload from what the finished job
already holds — hydro rows (precip + sim) and the V18.7 obs store — and calls
the Space's Gradio REST API. Failures degrade silently: the dashboard simply
shows no AI tail (e.g. model not trained yet, Space asleep, gauge unknown).

Env: CREST_NOWCAST_URL overrides/disables (empty string = off).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import requests

URL = os.environ.get("CREST_NOWCAST_URL",
                     "https://vincewin-crest-nowcast.hf.space").rstrip("/")
TIMEOUT_POST = 15
TIMEOUT_POLL = 45
LOOKBACK_H = 72


def enabled() -> bool:
    return bool(URL)


def _call_space(payload: dict) -> dict:
    """Two-step Gradio REST call: submit -> poll the event stream."""
    r = requests.post(f"{URL}/gradio_api/call/nowcast",
                      json={"data": [json.dumps(payload)]}, timeout=TIMEOUT_POST)
    r.raise_for_status()
    eid = r.json()["event_id"]
    data = None
    with requests.get(f"{URL}/gradio_api/call/nowcast/{eid}",
                      timeout=TIMEOUT_POLL, stream=True) as r2:
        for raw in r2.iter_lines():
            line = raw.decode("utf-8", "replace")
            if line.startswith("data:"):
                data = line[5:].strip()
    if not data:
        return {"ok": False, "reason": "empty_stream"}
    out = json.loads(data)
    return json.loads(out[0]) if isinstance(out, list) else {"ok": False, "reason": "bad_shape"}


def for_job(job, gid: str) -> dict:
    """Nowcast for a finished gauge of a sim job (cached on the job)."""
    if not enabled():
        return {"ok": False, "reason": "disabled"}
    cache = getattr(job, "_nowcast", None)
    if cache is None:
        cache = {}
        setattr(job, "_nowcast", cache)
    if gid in cache:
        return cache[gid]

    rows = job.hydro.get(gid, [])
    meta = job.meta.get(gid, {})
    if len(rows) < 2 or not meta:
        return {"ok": False, "reason": "no_rows"}
    try:
        t0 = datetime.strptime(str(rows[-1]["time"])[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return {"ok": False, "reason": "bad_time"}
    # trained on hourly steps — send the last LOOKBACK_H rows' precip as-is
    precip = [float(r.get("precip") or 0.0) for r in rows][-LOOKBACK_H:]

    obs_pts = []
    try:
        from hf_data import obs
        series = obs.get_series(gid, t0 - timedelta(hours=LOOKBACK_H), t0)
        obs_pts = [[t.strftime("%Y-%m-%dT%H:%M:%S"), round(q, 4)]
                   for t, q in series if t <= t0][-96:]
    except Exception:
        pass

    payload = {"gauge_id": gid, "lat": meta.get("lat"), "lon": meta.get("lon"),
               "area_km2": meta.get("area") or 1.0,
               "t0": t0.strftime("%Y-%m-%dT%H:%M:%S"),
               "precip": precip, "obs": obs_pts}
    try:
        res = _call_space(payload)
    except Exception as e:
        res = {"ok": False, "reason": f"space_unreachable: {type(e).__name__}"}
    cache[gid] = res
    return res

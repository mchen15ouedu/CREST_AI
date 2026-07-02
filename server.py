"""CREST_demo — FastAPI backend for the map-centric dashboard.

Serves the Leaflet frontend + JSON/SSE APIs, reusing the hf_data layer + agents.
  POST /api/query     NL query -> region + event pins + gauge pins (agent-driven)
  POST /api/simulate  selected gauges -> sim job (cap 10; ~4 concurrent)  [stub -> task#3]
  GET  /api/stream    SSE live status + hydrograph + 2-D overlays          [stub -> task#3]
Run: python server.py   (uvicorn on :7860, Docker Space entrypoint)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
import json
import queue
from datetime import datetime, timedelta

from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from hf_data import gauges, simjobs
from hf_data.pipeline import parse_query

HERE = os.path.dirname(os.path.abspath(__file__))
FRONTEND = os.path.join(HERE, "frontend")
MAX_SIMS = 10                      # demo cap (selection)

app = FastAPI(title="CREST_demo")


@app.get("/")
def index():
    return FileResponse(os.path.join(FRONTEND, "index.html"))


class Query(BaseModel):
    query: str
    radius_deg: float = 1.0


@app.post("/api/query")
def api_query(q: Query):
    """Parse the query into a map view + event pins + nearby USGS gauge pins."""
    ctx = parse_query(q.query)
    lat, lon = ctx.anchor
    r = q.radius_deg
    bbox = [lon - r, lat - r, lon + r, lat + r]

    cat = gauges.load_catalog((bbox[0], bbox[1], bbox[2], bbox[3]))
    if len(cat) > 300:                              # keep the map responsive
        cat = cat.assign(_d=((cat.LAT_GAGE - lat) ** 2 + (cat.LNG_GAGE - lon) ** 2)).nsmallest(300, "_d")
    gauge_pins = [{
        "id": str(row.STAID).zfill(8), "name": str(row.STANAME),
        "lat": float(row.LAT_GAGE), "lon": float(row.LNG_GAGE),
        "area_km2": float(row.DRAIN_SQKM),
    } for row in cat.itertuples()]

    # event pins: the parsed anchor (LLM event geocoding will add more later)
    event_pins = [{"lat": lat, "lon": lon, "label": ctx.label}]

    return {
        "label": ctx.label, "center": [lat, lon], "bbox": bbox,
        "t_start": ctx.t_start.isoformat(), "t_end": ctx.t_end.isoformat(),
        "event_pins": event_pins, "gauge_pins": gauge_pins,
        "n_gauges": len(gauge_pins), "max_sims": MAX_SIMS,
    }


class SimRequest(BaseModel):
    gauge_ids: list[str]
    t_start: str | None = None
    t_end: str | None = None
    hours: int = 48
    model: str = "auto"
    snow: str = "auto"
    timestep: str = "1h"
    warmup_days: int = 90
    overrides: dict | None = None


@app.post("/api/simulate")
def api_simulate(req: SimRequest):
    warning = None
    if len(req.gauge_ids) > MAX_SIMS:
        warning = (f"This demo simulates at most {MAX_SIMS} gauges at once; "
                   f"running the first {MAX_SIMS} of {len(req.gauge_ids)} selected.")
    t0 = datetime.fromisoformat(req.t_start) if req.t_start else datetime(2025, 7, 3)
    t1 = datetime.fromisoformat(req.t_end) if req.t_end else t0 + timedelta(hours=req.hours)
    opts = {"model": req.model, "hours": req.hours, "snow": req.snow,
            "timestep": req.timestep, "warmup_days": req.warmup_days,
            "overrides": req.overrides}
    job = simjobs.start_job(req.gauge_ids, t0, t1, opts)
    return {"sim_id": job.id, "gauge_ids": job.gauge_ids, "warning": warning,
            "max_concurrent": simjobs.MAX_CONCURRENT}


@app.get("/api/stream/{sim_id}")
async def api_stream(sim_id: str):
    job = simjobs.get_job(sim_id)
    if not job:
        return Response(status_code=404)

    async def gen():
        while True:
            try:
                ev = job.q.get_nowait()
            except queue.Empty:
                if job.done.is_set():
                    break
                await asyncio.sleep(0.15)
                continue
            yield {"data": json.dumps(ev)}
            if ev.get("kind") == "all_done":
                break

    return EventSourceResponse(gen())


@app.get("/api/overlay/{sim_id}/{gauge_id}.png")
def api_overlay(sim_id: str, gauge_id: str):
    job = simjobs.get_job(sim_id)
    png = job.overlay_png(gauge_id) if job else None
    if png is None:
        return Response(status_code=404)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/frame/{sim_id}/{gauge_id}/{idx}.png")
def api_frame(sim_id: str, gauge_id: str, idx: int):
    """A single pre-rendered animation frame (fixed color scale)."""
    job = simjobs.get_job(sim_id)
    png = job.frame_png(gauge_id, idx) if job else None
    if png is None:
        return Response(status_code=404)
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "max-age=3600"})


if os.path.isdir(FRONTEND):
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

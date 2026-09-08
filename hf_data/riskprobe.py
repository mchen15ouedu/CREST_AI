"""Per-pixel risk probe behind /api/riskprobe (the risk-view lens).

For one solver pixel of a published 2-D inundation event it assembles the
three arms of the risk lens plus the balloon's hazard class:

  inundation   depth at the scrubbed frame time (or the episode max),
               peak speed and the compass direction at that peak
               (maxspeed/flowdir rasters — events solved before the flux
               outputs shipped fall back to depth-only hazard)
  population   the pixel's WorldPop 2026-2030 age-sex people (worldpop.py)
  infrastructure  OSM values at the pixel (osminfra.py)

Hazard = the DEFRA/EA flood hazard rating HR = d x (v + 0.5): < 0.75 low,
< 1.25 moderate, < 2.5 significant, else extreme. Without a speed raster
the class comes from depth alone (0.3 / 0.9 / 2.0 m) and says so.

Everything reads straight from the CREST_data store via the local HF cache;
nothing is written anywhere.
"""
from __future__ import annotations

import json
import os
import threading
import time

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_man_cache: dict = {}
_lock = threading.Lock()
MAN_TTL_S = 120.0


def _dl(event_id: str, fname: str, force: bool = False) -> str:
    from huggingface_hub import hf_hub_download
    from . import eventstore
    return hf_hub_download(eventstore.REPO,
                           f"{eventstore.PREFIX}/{event_id}/{fname}",
                           repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"),
                           force_download=force)


def _manifest(event_id: str) -> dict:
    now = time.time()
    with _lock:
        hit = _man_cache.get(event_id)
        if hit and now - hit[0] < MAN_TTL_S:
            return hit[1]
    with open(_dl(event_id, "manifest.json", force=True),
              encoding="utf-8") as fp:
        man = json.load(fp)
    with _lock:
        _man_cache[event_id] = (now, man)
    return man


def _pixel_value(event_id: str, fname: str, row: int, col: int,
                 scale: float = 0.01):
    """Value at (row, col) of an event raster, None if the file is
    unavailable (e.g. animation cleared by retention). Depth and speed
    rasters store uint16 centimeters -> scale 0.01 recovers meters (m/s);
    pass scale=1 for raw-valued rasters like the flowdir sectors."""
    import rasterio
    try:
        with rasterio.open(_dl(event_id, fname)) as ds:
            v = ds.read(1, window=((row, row + 1), (col, col + 1)))
        return float(v[0, 0]) * scale
    except Exception:
        return None


def _hazard(depth, speed):
    if depth is None or depth <= 0:
        return {"rating": 0.0, "class": "dry", "basis": "depth+velocity"
                if speed is not None else "depth-only"}
    if speed is not None:
        hr = depth * (speed + 0.5)
        cls = ("low" if hr < 0.75 else "moderate" if hr < 1.25
               else "significant" if hr < 2.5 else "extreme")
        return {"rating": round(hr, 2), "class": cls,
                "basis": "depth+velocity"}
    cls = ("low" if depth < 0.3 else "moderate" if depth < 0.9
           else "significant" if depth < 2.0 else "extreme")
    return {"rating": round(depth, 2), "class": cls, "basis": "depth-only"}


def probe(event_id: str, lat: float, lon: float, t: str = "") -> dict:
    from . import osminfra, worldpop
    man = _manifest(event_id)
    gr = man.get("grid") or {}
    tr = gr.get("transform")
    if not tr:
        return {"error": "event manifest has no grid transform"}
    a, b, c, d, e, f = tr
    col = int((lon - c) / a)
    row = int((lat - f) / e)
    ny, nx = gr.get("ny", 0), gr.get("nx", 0)
    if not (0 <= row < ny and 0 <= col < nx):
        return {"error": "outside the simulated basin grid",
                "bbox": man.get("bbox")}
    # the lit-up pixel: its exact footprint polygon for the map highlight
    cell = {"bounds": [[f + e * (row + 1), c + a * col],
                       [f + e * row, c + a * (col + 1)]],
            "row": row, "col": col,
            "dx_m": gr.get("dx_m"), "dy_m": gr.get("dy_m")}

    frames = {fr.get("t"): fr.get("file") for fr in man.get("frames") or []}
    basis_t = "max depth"
    depth = None
    if t and t in frames:
        depth = _pixel_value(event_id, frames[t], row, col)
        if depth is not None:
            basis_t = t
    if depth is None:
        depth = _pixel_value(event_id, man.get("maxdepth", "maxdepth.tif"),
                             row, col)
    speed = (_pixel_value(event_id, man["maxspeed"], row, col)
             if man.get("maxspeed") else None)
    sector = None
    if man.get("flowdir") and speed is not None and speed > 0:
        raw = _pixel_value(event_id, man["flowdir"], row, col, scale=1.0)
        if raw is not None:
            sec = int(round(raw))
            if 0 <= sec <= 7:
                sector = sec

    # population needs the window cache; kick the build and report status
    cell_deg2 = abs(a * e)
    pop_status = worldpop.ensure_window(event_id, man.get("bbox"))
    pop = worldpop.probe(event_id, lat, lon, cell_deg2)

    infra = osminfra.probe(lat, lon,
                           cell_half_m=max(10.0, (gr.get("dx_m") or 25) / 2))

    return {
        "event": event_id, "lat": lat, "lon": lon, "cell": cell,
        "inundation": {
            "depth_m": None if depth is None else round(depth, 2),
            "depth_at": basis_t,
            "speed_ms": None if speed is None else round(speed, 2),
            "speed_is_peak": True,
            "direction": None if sector is None else COMPASS[sector],
        },
        "hazard": _hazard(depth, speed),
        "population": pop if pop is not None else {"status": pop_status},
        "infrastructure": infra,
    }

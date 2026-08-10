"""V25: nowcast-triggered 2-D inundation events (CREST-iMAP v2 coupling).

Chain: nowcast risk tiers (tier 3 = next-6-h AI peak >= Q5) -> hotspot
clusters pick trigger gauges -> EF5 nowcast-mode run over the basin with
output_grids=streamflow|runoff|subrunoff -> crestimap well-balanced solver
(3DEP DEM on demand) -> compact depth frames -> eventstore (one commit).

CPU sizing: 1" DEM downsampled to <= EVENT_MAX_CELLS (default 400k)
=> ~15-45 min per event on cpu-upgrade. A GPU Space raises the ceiling
to 1/3" (10 m). crestimap installs from the CREST-iMAP fork's v2 branch
(Dockerfile); everything here degrades to a clear error if it's missing.
"""
from __future__ import annotations

import datetime
import os
import tempfile
import threading

EVENT_TOKENS = "streamflow|runoff|subrunoff"
HORIZON_H = int(os.environ.get("EVENT_HORIZON_H", "12"))
HINDCAST_H = int(os.environ.get("EVENT_HINDCAST_H", "48"))
# events don't need the fleet's 90-day spin-up: upstream DA injection carries
# the river and the flood is driven by current rain; 30 d bounds forcing prep
WARMUP_D = int(os.environ.get("EVENT_WARMUP_D", "30"))
SIM_BACKSET_H = int(os.environ.get("EVENT_SIM_BACKSET_H", "6"))
MAX_CELLS = int(os.environ.get("EVENT_MAX_CELLS", "400000"))
DEM_RES = os.environ.get("EVENT_DEM_RES", "1")
DEM_CACHE = os.environ.get("EVENT_DEM_CACHE",
                           os.path.join(tempfile.gettempdir(), "dem_cache"))

_running: dict = {"id": None, "status": None, "log": [], "last": None}
_lock = threading.Lock()


def status() -> dict:
    return {"running": _running["id"], "status": _running["status"],
            "log": _running["log"][-30:], "last": _running["last"]}


def detect(max_events: int = 1) -> list[dict]:
    """Trigger gauges: per hotspot cluster, the tier-3 member with the
    largest drainage area (captures the shared basin)."""
    from . import nowcaststore, pipeline
    hs = nowcaststore.hotspots()
    picks = []
    if not hs.get("ok"):
        return picks
    seen = set()
    for cluster in hs.get("hotspots", []):
        reds = [m for m in cluster.get("top_gauges", []) if m.get("tier") == 3]
        best, best_area = None, -1.0
        for m in reds:
            g = pipeline.gauge_info(m["id"])
            if not g or g["id"] in seen:
                continue
            area = float(g.get("area") or 0.0)
            if area > best_area:
                best, best_area = g, area
        if best:
            seen.add(best["id"])
            picks.append({"gauge": best["id"], "lat": best["lat"],
                          "lon": best["lon"], "area_km2": best_area,
                          "hotspot_score": cluster.get("score")})
        if len(picks) >= max_events:
            break
    return picks


def run_one(gid: str, t0: datetime.datetime | None = None,
            trigger: dict | None = None) -> dict:
    """Full event pipeline for one trigger gauge. Blocking (minutes-long);
    call from a worker thread. Returns the manifest (raises on failure)."""
    from crestimap import EventConfig, run_event   # needs the v2 package
    from . import eventstore, nowcaststore, pipeline

    g = pipeline.gauge_info(gid)
    if g is None:
        raise ValueError(f"unknown gauge {gid}")
    t0 = t0 or nowcaststore.issue_t0() or \
        datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    t_start = t0 - datetime.timedelta(hours=HINDCAST_H)
    t_end = t0 + datetime.timedelta(hours=HORIZON_H)
    ev_id = f"{t0:%Y%m%d%H}_{gid}"
    work = tempfile.mkdtemp(prefix=f"event_{gid}_")

    def log(s):
        _running["log"].append(s)

    with _lock:
        _running.update(id=ev_id, status="ef5", log=[])
    try:
        log(f"EF5 nowcast-mode run {t_start:%m-%d %H}Z -> {t_end:%m-%d %H}Z "
            f"with gridded runoff output")
        meta = {}
        for kind, payload in pipeline.run_gauge(
                gid, t_start, t_end, model="auto", use_mock=False,
                grids=EVENT_TOKENS, workdir=work, nowcast_t0=t0,
                warmup_days=WARMUP_D):
            if kind == "meta":
                meta = payload
            elif kind == "status":
                log(str(payload))
            elif kind == "done" and payload.get("returncode", 0) not in (0, None):
                raise RuntimeError(f"EF5 failed: {payload}")
        out_dir = os.path.join(work, "CREST_output")
        model = (meta.get("model") or "crest").lower()
        wb_model = "crest" if model in ("crest", "hp") else "crestphys"

        with _lock:
            _running["status"] = "solver"
        cfg = EventConfig(
            event_id=ev_id, bbox=tuple(pipeline.basin_bbox(g)),
            t0=t0, t_end=t_end, ef5_output_dir=out_dir,
            out_dir=os.path.join(work, "event_out"), model=wb_model,
            sim_start=t0 - datetime.timedelta(hours=SIM_BACKSET_H),
            dem_res=DEM_RES, dem_cache=DEM_CACHE, max_cells=MAX_CELLS,
            trigger={**(trigger or {}), "gauge": gid}, progress=log)
        manifest = run_event(cfg)

        with _lock:
            _running["status"] = "render"
        _make_pngs(cfg.out_dir, manifest)
        log(f"rendered {len(manifest['frames'])} PNG overlays")

        with _lock:
            _running["status"] = "publish"
        ok = eventstore.publish_event(cfg.out_dir, manifest)
        log("published" if ok else "publish FAILED (kept locally)")
        manifest["published"] = ok
        _running["last"] = {"event": ev_id, "ok": True, "published": ok}
        return manifest
    except Exception as e:
        import traceback
        tb = traceback.format_exc().strip().splitlines()
        for line in tb[-6:]:
            log(line)
        _running["last"] = {"event": ev_id, "ok": False,
                            "error": f"{type(e).__name__}: {e}"}
        raise
    finally:
        with _lock:
            _running.update(id=None, status=None)


DEPTH_CAP_M = float(os.environ.get("EVENT_DEPTH_CAP_M", "3.0"))
MIN_SHOW_M = 0.02                     # < 2 cm renders transparent


def _make_pngs(out_dir: str, manifest: dict):
    """Colormapped RGBA overlays for the map (Leaflet imageOverlay): dry is
    transparent, light->dark blue over 0..DEPTH_CAP_M. Adds `bounds`,
    per-frame `png`, and `maxdepth_png` to the manifest (file rewritten)."""
    import json

    import numpy as np
    import rasterio
    from PIL import Image

    gr = manifest["grid"]
    a, _, c, _, e, f = gr["transform"]
    west, north = c, f
    east, south = c + a * gr["nx"], f + e * gr["ny"]
    manifest["bounds"] = [[south, west], [north, east]]

    def render(tif_name):
        with rasterio.open(os.path.join(out_dir, tif_name)) as ds:
            depth = ds.read(1).astype(float) / 100.0
        x = np.clip(depth / DEPTH_CAP_M, 0.0, 1.0)
        alpha = np.where(depth >= MIN_SHOW_M,
                         60 + 195 * np.sqrt(x), 0.0).astype(np.uint8)
        r = (173 + (8 - 173) * x).astype(np.uint8)
        g = (216 + (48 - 216) * x).astype(np.uint8)
        b = (230 + (107 - 230) * x).astype(np.uint8)
        out = tif_name[:-4] + ".png"
        Image.fromarray(np.dstack([r, g, b, alpha]), "RGBA").save(
            os.path.join(out_dir, out), optimize=True)
        return out

    for fr in manifest["frames"]:
        fr["png"] = render(fr["file"])
    manifest["maxdepth_png"] = render("maxdepth.tif")
    manifest["depth_cap_m"] = DEPTH_CAP_M
    with open(os.path.join(out_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp)


def run_detected(max_events: int = 1) -> list[dict]:
    """Detect + run (sequentially). Returns manifests of completed events."""
    out = []
    for pick in detect(max_events):
        try:
            out.append(run_one(pick["gauge"], trigger=pick))
        except Exception as e:  # keep the loop alive for other events
            out.append({"event_id": None, "gauge": pick["gauge"],
                        "error": f"{type(e).__name__}: {e}"})
    return out

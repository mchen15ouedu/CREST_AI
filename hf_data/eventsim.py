"""V25: nowcast-triggered 2-D inundation events (CREST-iMAP v2 coupling).

Chain: nowcast risk tiers (tier 3 = next-6-h AI peak >= Q5) -> hotspot
clusters pick trigger gauges -> EF5 nowcast-mode run over the basin with
output_grids=streamflow|runoff|subrunoff -> crestimap well-balanced solver
(3DEP DEM on demand) -> compact depth frames -> eventstore (one commit).

CPU sizing: RESOLUTION IS FIXED at the DEM's native cell size; when the
basin box exceeds EVENT_MAX_CELLS (default 400k) at that resolution, the
solver WINDOW shrinks around the trigger gauge (_hires_window) — never the
resolution. Full-basin native coverage = parallel/GPU milestone (subbasin
decomposition + ghost-cell halo exchange). crestimap installs from the
CREST-iMAP fork's v2 branch (Dockerfile); everything here degrades to a
clear error if it's missing.
"""
from __future__ import annotations

import datetime
import math
import os
import tempfile
import threading

EVENT_TOKENS = "streamflow|runoff|subrunoff"
HORIZON_H = int(os.environ.get("EVENT_HORIZON_H", "12"))
HINDCAST_H = int(os.environ.get("EVENT_HINDCAST_H", "48"))
# episode lifecycle: an active event is re-simulated every hourly tick with
# the fresh t0 (same event id, folder replaced) and ENDS when the gauge's
# risk tier drops below EVENT_CONT_TIER — i.e. when FLOW recedes to normal,
# not when precipitation stops (recession can outlive the rain by days).
CONT_TIER = int(os.environ.get("EVENT_CONT_TIER", "1"))
MAX_EPISODE_H = int(os.environ.get("EVENT_MAX_EPISODE_H", "96"))
MAX_PER_TICK = int(os.environ.get("EVENT_MAX_PER_TICK", "2"))
# events don't need the fleet's 90-day spin-up: upstream DA injection carries
# the river and the flood is driven by current rain; 30 d bounds forcing prep
WARMUP_D = int(os.environ.get("EVENT_WARMUP_D", "30"))
# the trigger typically fires slightly AFTER the flood starts (AI peak must
# clear Q5 and the obs gate must confirm), so the hydrodynamic window reaches
# back 12 h from t0 to capture the rising limb from the beginning
SIM_BACKSET_H = int(os.environ.get("EVENT_SIM_BACKSET_H", "12"))
MAX_CELLS = int(os.environ.get("EVENT_MAX_CELLS", "400000"))
DEM_RES = os.environ.get("EVENT_DEM_RES", "1")
DEM_CACHE = os.environ.get("EVENT_DEM_CACHE",
                           os.path.join(tempfile.gettempdir(), "dem_cache"))
# RESOLUTION IS FIXED at the DEM's native cell size — a coarsened flood map
# (one pixel = 10 street blocks) is useless information. When the basin box
# exceeds the cell budget at native resolution, the SIMULATED WINDOW shrinks
# around the trigger gauge instead (user directive 2026-08-12); covering the
# full basin at native resolution is the parallel-computing milestone.
DEM_RES_DEG = {"1": 1.0 / 3600.0, "13": 1.0 / 10800.0}


def _hires_window(bbox, gauge_ll, log=print):
    """Crop the basin box to the largest window that fits EVENT_MAX_CELLS at
    the DEM's NATIVE resolution, centered on the trigger gauge with a 25%
    bias toward the basin-box center (the upstream side, where the flood
    comes from). Returns bbox unchanged when it already fits."""
    w, s, e, n = bbox
    glat, glon = gauge_ll
    res = DEM_RES_DEG.get(DEM_RES, 1.0 / 3600.0)
    side_deg = math.sqrt(float(MAX_CELLS)) * res
    if (n - s) <= side_deg and (e - w) <= side_deg:
        return bbox
    cy = glat + 0.25 * ((s + n) / 2.0 - glat)
    cx = glon + 0.25 * ((w + e) / 2.0 - glon)
    hl_lat = min(side_deg, n - s) / 2.0
    hl_lon = min(side_deg, e - w) / 2.0
    # the gauge must stay well inside the window (bias is a preference only);
    # applying the gauge clamp BEFORE the box clamp keeps it inside even when
    # the gauge sits near a basin-box edge
    cy = min(max(cy, glat - 0.4 * hl_lat), glat + 0.4 * hl_lat)
    cx = min(max(cx, glon - 0.4 * hl_lon), glon + 0.4 * hl_lon)
    cy = min(max(cy, s + hl_lat), n - hl_lat)
    cx = min(max(cx, w + hl_lon), e - hl_lon)
    km = 111.0
    coslat = max(0.2, math.cos(math.radians(glat)))
    log(f"resolution-first window: {2 * hl_lon * km * coslat:.0f} x "
        f"{2 * hl_lat * km:.0f} km around the gauge at native ~"
        f"{res * 111000:.0f} m (basin box {(e - w) * km * coslat:.0f} x "
        f"{(n - s) * km:.0f} km exceeds the {MAX_CELLS} cell budget; "
        f"full-basin native coverage needs the parallel/GPU tier)")
    return (cx - hl_lon, cy - hl_lat, cx + hl_lon, cy + hl_lat)

_running: dict = {"id": None, "status": None, "log": [], "last": None}
_lock = threading.Lock()


def status(full: bool = False) -> dict:
    return {"running": _running["id"], "status": _running["status"],
            "log": _running["log"][-(400 if full else 30):],
            "last": _running["last"]}


OBS_GATE_FRAC = float(os.environ.get("EVENT_OBS_GATE", "0.3"))   # of Q2
OBS_WINDOW_H = int(os.environ.get("EVENT_OBS_WINDOW_H", "6"))


def _recent_obs_max(gid: str) -> float | None:
    """Max observed flow [m3/s] at the gauge over the last OBS_WINDOW_H
    hours from NWIS IV; None if the gauge reports nothing."""
    import json as _json
    import urllib.request
    url = ("https://waterservices.usgs.gov/nwis/iv/?format=json"
           f"&sites={gid}&period=PT{OBS_WINDOW_H}H&parameterCd=00060")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = _json.load(r)
        vals = []
        for ts in d["value"]["timeSeries"]:
            for block in ts["values"]:
                for v in block["value"]:
                    q = float(v["value"])
                    if q > -999:
                        vals.append(q * 0.0283168)   # cfs -> m3/s
        return max(vals) if vals else None
    except Exception:
        return None


def detect(max_events: int = 1) -> list[dict]:
    """Trigger gauges: per hotspot cluster, the tier-3 member with the
    largest drainage area — GATED on real observations. The AI tier alone
    can hallucinate floods at data-dead gauges (seen live: two tier-3
    triggers with zero USGS data and ~0 simulated flow), so a trigger also
    requires the gauge to be reporting AND already at >= EVENT_OBS_GATE of
    bankfull (Q2) within the last few hours."""
    from . import nowcaststore, pipeline
    hs = nowcaststore.hotspots()
    picks = []
    if not hs.get("ok"):
        return picks
    try:
        thr = nowcaststore._thresholds()
    except Exception:
        thr = {}
    seen = set()
    for cluster in hs.get("hotspots", []):
        reds = [m for m in cluster.get("top_gauges", []) if m.get("tier") == 3]
        cands = []
        for m in reds:
            g = pipeline.gauge_info(m["id"])
            if g and g["id"] not in seen:
                cands.append((float(g.get("area") or 0.0), g))
        for area, g in sorted(cands, reverse=True, key=lambda t: t[0]):
            gid = g["id"]
            obs = _recent_obs_max(gid)
            if obs is None:
                print(f"event trigger {gid}: SKIP — no USGS obs in "
                      f"{OBS_WINDOW_H} h (AI tier unsupported)")
                continue
            q2 = None
            t = thr.get(gid)
            if t is not None:
                q2 = t[1] if t[1] == t[1] else None    # NaN check
            gate = max(OBS_GATE_FRAC * q2, 1.0) if q2 else 1.0
            if obs < gate:
                print(f"event trigger {gid}: SKIP — obs {obs:.1f} m3/s below "
                      f"gate {gate:.1f} (river not responding yet)")
                continue
            seen.add(gid)
            picks.append({"gauge": gid, "lat": g["lat"], "lon": g["lon"],
                          "area_km2": area, "obs_m3s": round(obs, 1),
                          "hotspot_score": cluster.get("score")})
            break
        if len(picks) >= max_events:
            break
    return picks


def _basin_outline(ef5_dir: str, gauge_ll=None, max_pts: int = 1200, log=print):
    """True upstream-catchment boundary as [[lat, lon], ...] — EF5 carves the
    watershed from the gauge and writes nodata outside it in every gridded
    output, so the valid-data mask of any output grid IS the basin at 3
    arc-sec. Coarsened so the ring stays a few KB in the manifest.

    The mask can hold several disjoint blobs (and EF5's gauge snap sometimes
    carves a NEIGHBORING catchment for small basins — seen live: rings 11-12
    km from the gauge). So the ring must CONTAIN the trigger gauge (bbox test
    with ~2 km slack; the outlet sits exactly on the ring). If no ring does,
    return None rather than draw a misleading polygon — the UI falls back to
    the pin + domain rectangle. None on any failure, too."""
    import glob

    import numpy as np
    import rasterio
    from rasterio import features

    fs = []
    for kind in ("streamflow", "runoff", "subrunoff", "soilmoisture"):
        fs = sorted(glob.glob(os.path.join(ef5_dir, f"{kind}.*.tif")))
        if fs:
            break
    if not fs:
        return None
    with rasterio.open(fs[-1]) as ds:
        a = ds.read(1)
        nod = ds.nodata
        tr = ds.transform
    m = np.isfinite(a) & ((a != nod) if nod is not None else (a > -1.0))
    if not m.any():
        return None
    f = max(1, int(np.ceil(max(m.shape) / 400)))     # keep the mask <= ~400 px
    if f > 1:
        ny, nx = m.shape[0] // f, m.shape[1] // f
        m = m[:ny * f, :nx * f].reshape(ny, f, nx, f).any(axis=(1, 3))
        tr = rasterio.Affine(tr.a * f, tr.b, tr.c, tr.d, tr.e * f, tr.f)
    PAD = 0.02                                       # ~2 km containment slack
    best, best_area = None, 0.0
    for geom, val in features.shapes(m.astype("uint8"), transform=tr):
        if val != 1:
            continue
        ring = geom["coordinates"][0]
        if gauge_ll is not None:
            glat, glon = gauge_ll
            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            if not (min(lats) - PAD <= glat <= max(lats) + PAD
                    and min(lons) - PAD <= glon <= max(lons) + PAD):
                continue
        area = abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                       in zip(ring, ring[1:])))     # shoelace, lon/lat units
        if area > best_area:
            best, best_area = ring, area
    if not best:
        log("basin outline: no catchment ring contains the gauge "
            "(EF5 gauge snap may have carved a neighboring basin) — skipped")
        return None
    step = max(1, len(best) // max_pts)
    return [[round(y, 4), round(x, 4)] for x, y in best[::step]]


def run_one(gid: str, t0: datetime.datetime | None = None,
            trigger: dict | None = None, episode_id: str | None = None) -> dict:
    """Full event pipeline for one trigger gauge. Blocking (minutes-long);
    call from a worker thread. Returns the manifest (raises on failure).
    episode_id: re-simulate an ongoing episode under its original id (the
    published folder is replaced with the fresh window)."""
    from crestimap import EventConfig, run_event   # needs the v2 package
    from . import eventstore, nowcaststore, pipeline

    g = pipeline.gauge_info(gid)
    if g is None:
        raise ValueError(f"unknown gauge {gid}")
    t0 = t0 or nowcaststore.issue_t0() or \
        datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    t_start = t0 - datetime.timedelta(hours=HINDCAST_H)
    t_end = t0 + datetime.timedelta(hours=HORIZON_H)
    ev_id = episode_id or f"{t0:%Y%m%d%H}_{gid}"
    work = tempfile.mkdtemp(prefix=f"event_{gid}_")

    def log(s):
        _running["log"].append(s)

    with _lock:
        _running.update(id=ev_id, status="ef5", log=[])
    try:
        log(f"EF5 nowcast-mode run {t_start:%m-%d %H}Z -> {t_end:%m-%d %H}Z "
            f"with gridded runoff output")
        meta = {}
        hydro = {}

        def _f(v):
            try:
                v = float(v)
                return v if v == v and abs(v) != float("inf") else None
            except (TypeError, ValueError):
                return None

        for kind, payload in pipeline.run_gauge(
                gid, t_start, t_end, model="auto", use_mock=False,
                grids=EVENT_TOKENS, workdir=work, nowcast_t0=t0,
                warmup_days=WARMUP_D):
            if kind == "meta":
                meta = payload
            elif kind == "status":
                log(str(payload))
            elif kind == "hydro":
                # 1-D streamflow at the trigger gauge (sim + USGS obs) — kept
                # in the manifest so the Events tab plots the hydrograph
                for r in payload.get("rows") or []:
                    if r.get("time"):
                        hydro[str(r["time"])] = {
                            "time": str(r["time"]), "sim_q": _f(r.get("sim_q")),
                            "obs_q": _f(r.get("obs_q")),
                            "precip": _f(r.get("precip"))}
            elif kind == "done" and payload.get("returncode", 0) not in (0, None):
                raise RuntimeError(f"EF5 failed: {payload}")
        out_dir = os.path.join(work, "CREST_output")
        model = (meta.get("model") or "crest").lower()
        wb_model = "crest" if model in ("crest", "hp") else "crestphys"

        # diagnostics: what did EF5 actually write, and what did we ask for?
        ctl = os.path.join(work, "control.txt")
        if os.path.exists(ctl):
            for line in open(ctl, encoding="utf-8", errors="replace"):
                if line.lower().startswith("output_grids"):
                    log(f"control: {line.strip()}")
        for root in sorted(os.listdir(work)):
            rp = os.path.join(work, root)
            if os.path.isdir(rp):
                kinds = {}
                for n in os.listdir(rp):
                    kinds[n.split(".")[0]] = kinds.get(n.split(".")[0], 0) + 1
                if kinds:
                    log(f"workdir {root}/: {dict(sorted(kinds.items())[:8])}")

        with _lock:
            _running["status"] = "solver"
        # EF5 keeps the FULL basin (routing needs the whole catchment; its
        # routed channel stage carries upstream water INTO the window); only
        # the 2-D solver domain is cropped to hold native resolution
        solver_bbox = _hires_window(tuple(pipeline.basin_bbox(g)),
                                    (g["lat"], g["lon"]), log=log)
        cfg = EventConfig(
            event_id=ev_id, bbox=solver_bbox,
            t0=t0, t_end=t_end, ef5_output_dir=out_dir,
            out_dir=os.path.join(work, "event_out"), model=wb_model,
            sim_start=t0 - datetime.timedelta(hours=SIM_BACKSET_H),
            dem_res=DEM_RES, dem_cache=DEM_CACHE, max_cells=MAX_CELLS,
            trigger={**(trigger or {}), "gauge": gid}, progress=log)
        manifest = run_event(cfg)
        manifest["gauge"] = gid
        manifest["hydro"] = [hydro[k] for k in sorted(hydro)]
        manifest["status"] = "active"
        try:                       # true watershed ring for the map overlay
            manifest["basin"] = _basin_outline(
                out_dir, gauge_ll=(g["lat"], g["lon"]), log=log)
            if manifest["basin"]:
                log(f"basin outline: {len(manifest['basin'])} vertices")
        except Exception as e:
            log(f"basin outline skipped ({type(e).__name__}: {e})")

        with _lock:
            _running["status"] = "archive"
        try:
            _update_archive(cfg.out_dir, manifest, log)
        except Exception as e:
            log(f"archive update failed ({type(e).__name__}: {e}) — "
                f"continuing without it")
        if not manifest.get("archive_frames"):
            # bone-dry simulation: nothing to show, nothing to re-run — end
            # the episode so the hourly tick stops burning compute on it
            manifest["status"] = "ended"
            manifest["dry"] = True
            log("simulated dry (no wet cells in any frame) — episode ended")

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


def _update_archive(out_dir: str, manifest: dict, log):
    """Continuous per-episode archive: events/<id>/archive.parquet.

    Sparse long format — one row per WET cell per frame: (time, row, col,
    depth_cm uint16), zstd-compressed; grid georeferencing in the schema
    metadata. Each hourly re-simulation merges in: its fresh frames replace
    overlapping timestamps (they carry more observations), timestamps only
    the older runs covered are kept — so the archive spans the whole
    episode start -> end. The episode-wide max depth is recomputed from it
    (maxdepth.tif then covers the full episode, not just the last window),
    the hydrograph is merged the same way, and retention demotion keeps
    archive.parquet (only depth_* frames are dropped).
    """
    import json as _json

    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    import rasterio

    from . import eventstore

    ev = manifest["event_id"]
    gr = manifest["grid"]

    # fresh frames -> sparse table
    tables, new_times = [], []
    for fr in manifest["frames"]:
        with rasterio.open(os.path.join(out_dir, fr["file"])) as ds:
            q = ds.read(1).astype(np.uint16)
        t = datetime.datetime.strptime(fr["t"], "%Y-%m-%dT%H:%MZ")
        new_times.append(t)
        r, c = np.nonzero(q)
        tables.append(pa.table({
            "time": pa.array([t] * len(r), pa.timestamp("s")),
            "row": pa.array(r.astype(np.uint16), pa.uint16()),
            "col": pa.array(c.astype(np.uint16), pa.uint16()),
            "depth_cm": pa.array(q[r, c], pa.uint16())}))
    merged = pa.concat_tables(tables)
    old_hydro = []

    # previous archive of this episode (if any) — keep its older timestamps
    schema = pa.schema([("time", pa.timestamp("s")), ("row", pa.uint16()),
                        ("col", pa.uint16()), ("depth_cm", pa.uint16())])
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(eventstore.REPO, f"{eventstore.PREFIX}/{ev}/archive.parquet",
                            repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        told = pq.read_table(p)
        gold = _json.loads((told.schema.metadata or {}).get(b"grid", b"{}"))
        if gold.get("ny") == gr["ny"] and gold.get("nx") == gr["nx"]:
            # parquet stores timestamps as ms — normalize before set ops
            told = told.select(["time", "row", "col", "depth_cm"]).cast(schema)
            keep = told.filter(pc.invert(pc.is_in(
                told.column("time"),
                value_set=pa.array(new_times, pa.timestamp("s")))))
            merged = pa.concat_tables([keep, merged])
            log(f"archive: merged {keep.num_rows} prior wet-cell rows")
        else:
            log("archive: grid changed — restarting archive")
    except Exception:
        pass  # first publish of the episode (or fetch hiccup): fresh archive
    try:
        from huggingface_hub import hf_hub_download
        mp = hf_hub_download(eventstore.REPO, f"{eventstore.PREFIX}/{ev}/manifest.json",
                             repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        with open(mp, encoding="utf-8") as fp:
            old_hydro = _json.load(fp).get("hydro") or []
    except Exception:
        pass

    merged = merged.sort_by([("time", "ascending")])
    meta = dict(merged.schema.metadata or {})
    meta[b"grid"] = _json.dumps(gr).encode()
    meta[b"depth_unit"] = b"centimeters"
    meta[b"event_id"] = str(ev).encode()
    merged = merged.replace_schema_metadata(meta)
    pq.write_table(merged, os.path.join(out_dir, "archive.parquet"),
                   compression="zstd")

    times = pc.unique(merged.column("time"))
    manifest["archive"] = "archive.parquet"
    manifest["archive_frames"] = len(times)

    # episode-wide max depth from the archive (overwrites the run's own)
    md = np.zeros((gr["ny"], gr["nx"]), dtype=np.uint16)
    rr = merged.column("row").to_numpy()
    cc = merged.column("col").to_numpy()
    dd = merged.column("depth_cm").to_numpy()
    np.maximum.at(md, (rr, cc), dd)
    from crestimap.io import write_depth
    from rasterio.transform import Affine
    a, b, c0, d, e, f = gr["transform"]
    write_depth(os.path.join(out_dir, "maxdepth.tif"), md.astype(float) / 100.0,
                Affine(a, b, c0, d, e, f), gr.get("crs"))

    # continuous hydrograph across the episode (new rows win on overlap)
    if old_hydro:
        have = {r["time"] for r in manifest["hydro"]}
        rows = [r for r in old_hydro if r.get("time") not in have] + manifest["hydro"]
        manifest["hydro"] = sorted(rows, key=lambda r: r["time"])
    log(f"archive: {merged.num_rows} wet-cell rows over {len(times)} frames")


DEPTH_CAP_M = float(os.environ.get("EVENT_DEPTH_CAP_M", "3.0"))
MIN_SHOW_M = 0.02                     # < 2 cm renders transparent


# depth color scale (fractions of DEPTH_CAP_M -> RGB): green (shallow) ->
# yellow -> red -> purple (>= cap), so depth CLASSES are distinguishable at
# a glance. The panel legend gradient in frontend/app.js mirrors these stops.
DEPTH_STOPS_X = (0.0, 0.30, 0.65, 1.0)
DEPTH_STOPS_R = (46, 250, 235, 130)
DEPTH_STOPS_G = (160, 220, 50, 20)
DEPTH_STOPS_B = (60, 40, 35, 140)


def _make_pngs(out_dir: str, manifest: dict):
    """Colormapped RGBA overlays for the map (Leaflet imageOverlay): dry is
    transparent; wet cells follow the DEPTH_STOPS ramp over 0..DEPTH_CAP_M.
    Adds `bounds`, per-frame `png`, and `maxdepth_png` to the manifest
    (file rewritten)."""
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
                         90 + 165 * np.sqrt(x), 0.0).astype(np.uint8)
        r = np.interp(x, DEPTH_STOPS_X, DEPTH_STOPS_R).astype(np.uint8)
        g = np.interp(x, DEPTH_STOPS_X, DEPTH_STOPS_G).astype(np.uint8)
        b = np.interp(x, DEPTH_STOPS_X, DEPTH_STOPS_B).astype(np.uint8)
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


def hourly_tick() -> dict:
    """Called by the updater after each hourly data refresh.

    1. Every ACTIVE episode whose gauge is still at/above EVENT_CONT_TIER is
       re-simulated at the fresh t0 (same event id; folder replaced).
    2. Episodes whose gauge receded below the tier (flow back to normal) or
       older than EVENT_MAX_EPISODE_H are marked ended — precipitation going
       to zero does NOT end an episode while the hydrograph is still high.
    3. Newly flagged tier-3 gauges start new episodes.
    Runs at most EVENT_MAX_PER_TICK simulations, sequentially.
    """
    from . import eventstore, nowcaststore
    if _running["id"]:
        return {"skipped": "runner busy"}
    risk = nowcaststore.all_risk()
    tiers = (risk or {}).get("tiers") or {}
    idx = eventstore.load_index()
    now = datetime.datetime.utcnow()

    jobs, ended = [], []
    for eid, s in idx.items():
        if s.get("status", "active") != "active":
            continue
        gid = str(((s.get("trigger") or {}).get("gauge")) or s.get("gauge") or "")
        started = s.get("episode_started") or s.get("t0") or ""
        try:
            age_h = (now - datetime.datetime.strptime(
                started, "%Y-%m-%dT%H:%MZ")).total_seconds() / 3600.0
        except ValueError:
            age_h = float("inf")
        tier = int(tiers.get(gid, 0))
        if gid and tier >= CONT_TIER and age_h <= MAX_EPISODE_H:
            jobs.append({"gauge": gid, "episode": eid,
                         "trigger": {**(s.get("trigger") or {}), "tier": tier}})
        else:
            ended.append(eid)
    if ended:
        eventstore.mark_ended(ended)

    active_gauges = {j["gauge"] for j in jobs}
    for pick in detect(MAX_PER_TICK):
        if pick["gauge"] not in active_gauges:
            jobs.append({"gauge": pick["gauge"], "episode": None,
                         "trigger": pick})

    results = []
    for j in jobs[:MAX_PER_TICK]:
        try:
            m = run_one(j["gauge"], trigger=j["trigger"],
                        episode_id=j["episode"])
            results.append({"event_id": m.get("event_id"),
                            "published": m.get("published")})
        except Exception as e:
            results.append({"gauge": j["gauge"],
                            "error": f"{type(e).__name__}: {e}"})
    try:                       # age out old events (no-op most hours)
        swept = eventstore.retention_sweep()
    except Exception:
        swept = 0
    return {"ran": len(results), "ended": ended, "swept": swept,
            "results": results}

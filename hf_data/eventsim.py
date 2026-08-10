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
        cfg = EventConfig(
            event_id=ev_id, bbox=tuple(pipeline.basin_bbox(g)),
            t0=t0, t_end=t_end, ef5_output_dir=out_dir,
            out_dir=os.path.join(work, "event_out"), model=wb_model,
            sim_start=t0 - datetime.timedelta(hours=SIM_BACKSET_H),
            dem_res=DEM_RES, dem_cache=DEM_CACHE, max_cells=MAX_CELLS,
            trigger={**(trigger or {}), "gauge": gid}, progress=log)
        manifest = run_event(cfg)
        manifest["gauge"] = gid
        manifest["hydro"] = [hydro[k] for k in sorted(hydro)]
        manifest["status"] = "active"

        with _lock:
            _running["status"] = "archive"
        try:
            _update_archive(cfg.out_dir, manifest, log)
        except Exception as e:
            log(f"archive update failed ({type(e).__name__}: {e}) — "
                f"continuing without it")

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
    return {"ran": len(results), "ended": ended, "results": results}

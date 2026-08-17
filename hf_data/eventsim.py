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
import time

EVENT_TOKENS = "streamflow|runoff|subrunoff"
HORIZON_H = int(os.environ.get("EVENT_HORIZON_H", "12"))
HINDCAST_H = int(os.environ.get("EVENT_HINDCAST_H", "48"))
# episode lifecycle: an active event is re-simulated every hourly tick with
# the fresh t0 (same event id, folder replaced) and ENDS when the gauge's
# risk tier drops below EVENT_CONT_TIER — i.e. when FLOW recedes to normal,
# not when precipitation stops (recession can outlive the rain by days).
CONT_TIER = int(os.environ.get("EVENT_CONT_TIER", "1"))
# episode END is decided by USGS OBSERVATIONS, not the AI tier (user
# directive 2026-08-14): the episode lives while recent observed flow stays
# >= CONT_OBS_MULT x normal (baseflow); the AI tier is only the fallback for
# gauges that stop reporting mid-episode.
CONT_OBS_MULT = float(os.environ.get("EVENT_CONT_OBS_MULT", "5.0"))
MAX_EPISODE_H = int(os.environ.get("EVENT_MAX_EPISODE_H", "96"))
MAX_PER_TICK = int(os.environ.get("EVENT_MAX_PER_TICK", "2"))
# V30: the CPU rung is a LAST resort, not a parallel engine — in queue mode
# "on" a local solve runs only when an event has gone this long with no
# fresh worker claim and no publish progress (worker output supersedes the
# CPU window anyway, and hours-long CPU solves were what starved the tick)
CPU_FALLBACK_H = float(os.environ.get("EVENT_CPU_FALLBACK_H", "3"))
# ...and when it does engage it gets a WALL-CLOCK BUDGET. A free CPU Space
# solving a 400k-cell grid can take days: 05570000 held the runner for 39 h
# and was still at sim t=8.5 of 24 h, so every hourly tick since returned
# "runner busy" and detection stopped dead (found 2026-08-17). Past the
# budget the local solve aborts, the bundle stays queued for a real worker,
# and the runner is freed. Only applied when the bundle IS queued (queue mode
# "off" has no worker to hand off to, so there the CPU rung must finish).
CPU_MAX_H = float(os.environ.get("EVENT_CPU_MAX_H", "2"))
# ...and it is not retried immediately: a grid the CPU cannot finish in the
# budget will not finish next hour either, and retrying burns the whole tick
# every hour. In-memory, so a Space restart gives it a fresh chance.
CPU_COOLDOWN_H = float(os.environ.get("EVENT_CPU_COOLDOWN_H", "12"))
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

_running: dict = {"id": None, "status": None, "log": [], "last": None,
                  "started": None}
_lock = threading.Lock()
# event ids whose local CPU solve blew CPU_MAX_H -> when it happened; the
# tick's CPU-rung sweep skips them for CPU_COOLDOWN_H
_cpu_timeout: dict = {}


def _busy_h() -> float | None:
    """Hours the current job has held the runner — None when idle. Health
    reads this: a tick that keeps reporting "runner busy" is only healthy
    while the job it is waiting on is young."""
    st = _running.get("started")
    if not _running["id"] or not st:
        return None
    try:
        t = datetime.datetime.strptime(st, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        return None
    return round((datetime.datetime.utcnow() - t).total_seconds() / 3600.0, 2)


def note_tick_attempt() -> None:
    """Stamp tick liveness. Called by hourly_tick AND by the /api/event_tick
    endpoint's busy short-circuit, so `tick.last` means "the hourly ping is
    arriving" and the separate busy_h field carries the wedge alarm — before
    this, a wedged runner made the two indistinguishable."""
    _running["last_tick"] = datetime.datetime.utcnow().strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def status(full: bool = False) -> dict:
    return {"running": _running["id"], "status": _running["status"],
            "started": _running.get("started"), "busy_h": _busy_h(),
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


def _age_days_started(s: dict) -> float | None:
    """Age [days] of an index entry from its episode start (None unknown)."""
    try:
        t = datetime.datetime.strptime(
            s.get("episode_started") or s.get("t0") or "", "%Y-%m-%dT%H:%MZ")
        return (datetime.datetime.utcnow() - t).total_seconds() / 86400.0
    except ValueError:
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
            trigger: dict | None = None, episode_id: str | None = None,
            defer: bool = False, force_cpu: bool = False) -> dict:
    """Full event pipeline for one trigger gauge. Blocking (minutes-long);
    call from a worker thread. Returns the manifest (raises on failure).
    episode_id: re-simulate an ongoing episode under its original id (the
    published folder is replaced with the fresh window).
    defer: enqueue for a worker WITHOUT the CPU fallback (manual refills —
    the runner stays free and the worker publishes whenever it gets there).
    force_cpu: solve locally even in queue mode "on" — the tick passes this
    when an event has sat unclaimed/unpublished past EVENT_CPU_FALLBACK_H
    (V30: the CPU rung is demoted to that last resort; a fresh enqueue no
    longer burns hours of runner CPU on a solve a worker replaces anyway)."""
    from crestimap import EventConfig, run_event   # needs the v2 package
    from . import eventqueue, eventstore, nowcaststore, pipeline

    g = pipeline.gauge_info(gid)
    if g is None:
        raise ValueError(f"unknown gauge {gid}")

    # a worker is mid-solve on this episode's previous bundle: replacing the
    # spec (and historically, clobbering the claim) would waste its hours of
    # GPU work — its publish IS this hour's refresh. Skip quietly.
    if episode_id and eventqueue.mode() == "on":
        w = eventqueue.claim_fresh_for(episode_id)
        if w:
            _running["last"] = {"event": episode_id, "ok": True,
                                "delegated": w, "skipped": "worker mid-solve"}
            return {"event_id": episode_id, "published": False,
                    "delegated": w, "skipped": "worker mid-solve"}
    t0 = t0 or nowcaststore.issue_t0() or \
        datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    # NO MOVING WINDOW (user directive 2026-08-14): an episode's simulation
    # window is anchored at its FIRST trigger minus the backset and only
    # GROWS forward — re-sims must never slide past the flood's rise/peak
    # (03190000's crest was lost exactly that way). The worker spec carries
    # the full anchored span; the CPU fallback contributes the fresh 24-h
    # slice and the publish-time frame merge keeps the record fixed-start.
    anchor = t0
    if episode_id:
        st = (eventstore.load_index().get(episode_id) or {}).get(
            "episode_started") or ""
        try:
            anchor = datetime.datetime.strptime(st, "%Y-%m-%dT%H:%MZ")
        except ValueError:
            pass
    sim_anchor = anchor - datetime.timedelta(hours=SIM_BACKSET_H)
    # EF5 forcing must span the whole anchored solver window
    t_start = min(t0 - datetime.timedelta(hours=HINDCAST_H), sim_anchor)
    t_end = t0 + datetime.timedelta(hours=HORIZON_H)
    ev_id = episode_id or f"{t0:%Y%m%d%H}_{gid}"
    work = tempfile.mkdtemp(prefix=f"event_{gid}_")

    def log(s):
        _running["log"].append(s)

    with _lock:
        _running.update(id=ev_id, status="ef5", log=[],
                        started=datetime.datetime.utcnow().strftime(
                            "%Y-%m-%dT%H:%M:%SZ"))
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

        basin_ring = None
        try:            # true watershed ring (map overlay + worker job spec)
            basin_ring = _basin_outline(out_dir, gauge_ll=(g["lat"], g["lon"]),
                                        log=log)
            if basin_ring:
                log(f"basin outline: {len(basin_ring)} vertices")
        except Exception as e:
            log(f"basin outline skipped ({type(e).__name__}: {e})")

        # EF5 keeps the FULL basin (routing needs the whole catchment; its
        # routed channel stage carries upstream water INTO the window); only
        # the 2-D solver domain is cropped to hold native resolution
        basin_bbox = tuple(pipeline.basin_bbox(g))
        solver_bbox = _hires_window(basin_bbox, (g["lat"], g["lon"]), log=log)
        # worker spec = full anchored span (fixed episode window); the local
        # CPU fallback still solves only the fresh 24-h slice — its frames
        # merge into the fixed-start record at publish
        sim_start = sim_anchor
        sim_start_local = max(sim_anchor,
                              t0 - datetime.timedelta(hours=SIM_BACKSET_H))

        # warm continuation (user directive 2026-08-14): the hourly update
        # inherits the previous run's water state. The episode's latest
        # published depth frame at/just before this run's fresh-slice start
        # is dropped into the forcing dir as init_depth.tif — the enqueue
        # tars every .tif, so workers get it too, and run_event/chunkrun
        # auto-discover it (h0 = max(channel pre-wet, previous depth)).
        if episode_id:
            try:
                pm = eventstore._prior_manifest(ev_id) or {}
                js = sim_start_local.strftime("%Y-%m-%dT%H:%MZ")
                cand = [f for f in pm.get("frames", [])
                        if f.get("t", "") <= js and f.get("file")]
                if cand:
                    fr = max(cand, key=lambda f: f["t"])
                    from huggingface_hub import hf_hub_download
                    p = hf_hub_download(
                        eventstore.REPO,
                        f"{eventstore.PREFIX}/{ev_id}/{fr['file']}",
                        repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"),
                        force_download=True)
                    import shutil as _sh
                    ts = fr["t"].replace("-", "").replace("T", "") \
                        .replace(":", "").replace("Z", "")
                    _sh.copy(p, os.path.join(out_dir,
                                             f"init_depth_{ts}.tif"))
                    log(f"warm-start IC: previous depth frame {fr['t']} "
                        f"carried into this run (applied only when a run's "
                        f"window starts there)")
            except Exception as e:
                log(f"no warm-start IC ({type(e).__name__}) — cold pre-wet")

        # V27 P1: offer the solve to a GPU worker — full basin at native
        # resolution, everything it needs in one bundle. The worker pool is
        # a ladder (HPC GPU -> ZeroGPU Space -> nobody); the local CPU
        # window below remains the unconditional fallback.
        queued = False
        if eventqueue.mode() != "off":
            spec = {"event_id": ev_id, "episode": bool(episode_id),
                    "gauge": {"id": gid, "lat": g["lat"], "lon": g["lon"],
                              "area_km2": g.get("area")},
                    "bbox_basin": list(basin_bbox),
                    "bbox_window": list(solver_bbox),
                    "t0": t0.strftime("%Y-%m-%dT%H:%MZ"),
                    "t_end": t_end.strftime("%Y-%m-%dT%H:%MZ"),
                    "sim_start": sim_start.strftime("%Y-%m-%dT%H:%MZ"),
                    "model": wb_model, "dem_res": DEM_RES,
                    "trigger": {**(trigger or {}), "gauge": gid},
                    "hydro": [hydro[k] for k in sorted(hydro)],
                    "basin": basin_ring,
                    "queued": datetime.datetime.utcnow().strftime(
                        "%Y-%m-%dT%H:%M:%SZ")}
            queued = eventqueue.enqueue(ev_id, spec, out_dir, log=log)
            if queued and eventqueue.mode() == "on":
                with _lock:
                    _running["status"] = "queued"
                if defer:
                    # manual refill: leave the bundle for the worker pool and
                    # free the runner immediately — no claim wait, no CPU
                    log(f"deferred to the worker queue — the result "
                        f"publishes whenever a worker picks it up")
                    _running["last"] = {"event": ev_id, "ok": True,
                                        "deferred": True}
                    return {"event_id": ev_id, "published": False,
                            "deferred": True}
                if eventqueue.wait_for_claim(ev_id, log=log):
                    # DELEGATED: the worker owns the job now. Blocking here
                    # for hours (the old wait_for_result) held the runner and
                    # starved hourly_tick — new floods went unmapped while we
                    # babysat a claimed solve (Indiana, 2026-08-13). The
                    # worker publishes on its own; ticks keep flowing.
                    w = eventqueue.claim_fresh_for(ev_id) or "worker"
                    log(f"delegated to {w} — full-basin result will publish "
                        f"when the solve finishes; runner freed")
                    _running["last"] = {"event": ev_id, "ok": True,
                                        "delegated": w}
                    return {"event_id": ev_id, "published": False,
                            "delegated": w}
                if not force_cpu:
                    # V30: no claim yet, but the CPU rung is demoted — leave
                    # the bundle queued; the tick re-checks and only falls
                    # back to a local solve after EVENT_CPU_FALLBACK_H with
                    # no claim and no publish progress
                    log(f"no worker claim — bundle stays queued (CPU "
                        f"fallback only after {CPU_FALLBACK_H:g} h "
                        f"without progress)")
                    _running["last"] = {"event": ev_id, "ok": True,
                                        "queued": True}
                    return {"event_id": ev_id, "published": False,
                            "queued": True}

        with _lock:
            _running["status"] = "solver"
        # wall-clock budget for the demoted CPU rung (see CPU_MAX_H): the
        # solver calls progress once per output frame, so raising from the
        # callback aborts mid-integration instead of holding the tick for
        # days. Only armed when a worker can pick the bundle up instead.
        solve_log = log
        if queued and CPU_MAX_H > 0:
            deadline = time.monotonic() + CPU_MAX_H * 3600.0

            def solve_log(s):                       # noqa: F811
                log(s)
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"local CPU solve exceeded EVENT_CPU_MAX_H="
                        f"{CPU_MAX_H:g} h ({s})")

        cfg = EventConfig(
            event_id=ev_id, bbox=solver_bbox,
            t0=t0, t_end=t_end, ef5_output_dir=out_dir,
            out_dir=os.path.join(work, "event_out"), model=wb_model,
            sim_start=sim_start_local,
            dem_res=DEM_RES, dem_cache=DEM_CACHE, max_cells=MAX_CELLS,
            trigger={**(trigger or {}), "gauge": gid}, progress=solve_log)
        try:
            manifest = run_event(cfg)
        except TimeoutError as e:
            # not a failure — the designed give-up. The bundle is already in
            # the queue, so the work is not lost; only this runner is freed.
            with _lock:
                _cpu_timeout[ev_id] = datetime.datetime.utcnow()
            log(f"{e} — aborted, bundle stays queued for a worker "
                f"(no CPU retry for {CPU_COOLDOWN_H:g} h)")
            _running["last"] = {"event": ev_id, "ok": True,
                                "cpu_timeout": True, "queued": True}
            return {"event_id": ev_id, "published": False,
                    "cpu_timeout": True, "queued": True}
        manifest["gauge"] = gid
        manifest["hydro"] = [hydro[k] for k in sorted(hydro)]
        manifest["status"] = "active"
        manifest["basin"] = basin_ring     # computed above, pre-queue

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
            _running.update(id=None, status=None, started=None)


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


DEPTH_CAP_M = float(os.environ.get("EVENT_DEPTH_CAP_M", "3.0"))  # dry fallback
MIN_SHOW_M = 0.02                     # < 2 cm renders transparent
# the color scale ADAPTS to each event: cap = p99 of the episode's wet
# maxdepth cells, clamped to [CAP_MIN, CAP_MAX] — a fixed 0..3 m scale
# saturates major floods purple and wastes range on shallow ones. All of an
# event's frames share the one cap (comparable during animation); the panel
# legend reads depth_cap_m from the manifest.
CAP_MIN_M = float(os.environ.get("EVENT_CAP_MIN_M", "0.5"))
CAP_MAX_M = float(os.environ.get("EVENT_CAP_MAX_M", "10.0"))


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

    cap = adaptive_cap(os.path.join(out_dir, "maxdepth.tif"))

    def render(tif_name):
        with rasterio.open(os.path.join(out_dir, tif_name)) as ds:
            depth = ds.read(1).astype(float) / 100.0
        x = np.clip(depth / cap, 0.0, 1.0)
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
    manifest["depth_cap_m"] = cap
    with open(os.path.join(out_dir, "manifest.json"), "w") as fp:
        json.dump(manifest, fp)


def adaptive_cap(maxdepth_tif: str) -> float:
    """Event-specific color-scale cap: p99 of the episode's wet maxdepth
    cells, clamped to [CAP_MIN_M, CAP_MAX_M]; DEPTH_CAP_M if (near-)dry."""
    import numpy as np
    import rasterio
    try:
        with rasterio.open(maxdepth_tif) as ds:
            depth = ds.read(1).astype(float) / 100.0
        wet = depth[depth >= MIN_SHOW_M]
        if len(wet) < 10:
            return DEPTH_CAP_M
        return round(float(np.clip(np.percentile(wet, 99), CAP_MIN_M, CAP_MAX_M)), 2)
    except Exception:
        return DEPTH_CAP_M


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
    note_tick_attempt()                            # health: tick liveness
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
        # observed flow decides the end: episode continues while the gauge's
        # recent USGS max is >= CONT_OBS_MULT x baseflow. AI tier only backs
        # up a gauge that went silent (obs None).
        obs = _recent_obs_max(gid) if gid else None
        qbase = None
        try:
            qbase = float((nowcaststore._thresholds().get(gid) or {}
                           ).get("qbase") or 0) or None
        except Exception:
            pass
        if obs is not None and qbase:
            still_high = obs >= max(CONT_OBS_MULT * qbase, 1.0)
        else:
            still_high = tier >= CONT_TIER
        if gid and still_high and age_h <= MAX_EPISODE_H:
            # publish progress stale + nobody claiming -> this re-sim runs
            # on the CPU rung (V30 demotion: last resort, not default)
            try:
                pub_age_h = (now - datetime.datetime.strptime(
                    s.get("generated") or "", "%Y-%m-%dT%H:%M:%SZ")
                ).total_seconds() / 3600.0
            except ValueError:
                pub_age_h = float("inf")
            jobs.append({"gauge": gid, "episode": eid,
                         "force_cpu": pub_age_h > CPU_FALLBACK_H,
                         "trigger": {**(s.get("trigger") or {}), "tier": tier}})
        else:
            ended.append(eid)
    if ended:
        eventstore.mark_ended(ended)     # fast: stops re-sims immediately
    # V30 episode final products: consolidate ended episodes (stats + the
    # episode-scoped nowcast verification) in a background thread — scoring
    # downloads ~1 archived nowcast per episode-hour and must not delay the
    # tick. Also self-heals: recently-ended episodes still missing "final"
    # (e.g. a worker publish raced the index commit) get another pass.
    heal = [e for e, s in idx.items()
            if s.get("status") == "ended" and not s.get("final")
            and (_age_days_started(s) or 99.0) < 3.0]
    to_final = ended + [e for e in heal if e not in ended]
    if to_final:
        threading.Thread(target=eventstore.finalize_episodes,
                         args=(to_final, print), daemon=True).start()

    from . import eventqueue as _eq
    active_gauges = {j["gauge"] for j in jobs}
    new_picks = []
    for pick in detect(MAX_PER_TICK):
        if pick["gauge"] in active_gauges:
            continue
        if _eq.mode() == "on" and _eq.gauge_pending(pick["gauge"]):
            continue           # already in the worker queue (index lags it)
        new_picks.append({"gauge": pick["gauge"], "episode": None,
                          "trigger": pick})
    # V30 CPU-rung last resort: queued bundles that were never published
    # (not in the index) and sat unclaimed past CPU_FALLBACK_H get a local
    # solve — enqueue-only detections and deferred refills must not strand
    # when every worker is dead or busy. t0 comes from the event id so a
    # backdated refill keeps its window.
    if _eq.mode() == "on":
        for row in _eq.queue_status():
            rid, rgid = str(row.get("id") or ""), str(row.get("gauge") or "")
            if (not rgid or row.get("worker") or rid in idx
                    or rgid in active_gauges
                    or any(p["gauge"] == rgid for p in new_picks)):
                continue
            try:
                q_age_h = (now - datetime.datetime.strptime(
                    row.get("queued") or "", "%Y-%m-%dT%H:%M:%SZ")
                ).total_seconds() / 3600.0
                t0_row = datetime.datetime.strptime(rid[:10], "%Y%m%d%H")
            except ValueError:
                continue
            to = _cpu_timeout.get(rid)
            if to and (now - to).total_seconds() / 3600.0 < CPU_COOLDOWN_H:
                continue      # blew the CPU budget already — wait for a worker
            if q_age_h > CPU_FALLBACK_H:
                print(f"event tick: {rid} unclaimed for {q_age_h:.1f} h — "
                      f"CPU fallback engages")
                new_picks.append({"gauge": rgid, "episode": None,
                                  "t0": t0_row, "force_cpu": True,
                                  "trigger": {"cpu_fallback": True}})
    # NEW floods outrank hourly refreshes of already-mapped ones: the first
    # slot goes to a new detection when one exists. Episode re-sims fill the
    # rest. (Before this, episodes filled the list first and the [:MAX] slice
    # starved every new trigger — Indiana flooded for two days unmapped
    # while the runner refreshed two known basins.)
    jobs = (new_picks[:1] + jobs + new_picks[1:])
    if len(jobs) > MAX_PER_TICK:
        starved = jobs[MAX_PER_TICK:]
        print(f"event tick: {len(starved)} job(s) wait for a slot: "
              f"{[j['gauge'] for j in starved]}")

    results = []
    for j in jobs[:MAX_PER_TICK]:
        try:
            m = run_one(j["gauge"], t0=j.get("t0"), trigger=j["trigger"],
                        episode_id=j["episode"],
                        force_cpu=bool(j.get("force_cpu")))
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

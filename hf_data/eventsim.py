"""V25: nowcast-triggered 2-D inundation events (CREST-iMAP v2 coupling).

Chain: AI nowcasting tiers (tier 3 = next-6-h AI peak >= Q5) -> hotspot
clusters pick trigger gauges -> EF5 nowcast-mode run over the basin with
output_grids=streamflow|runoff|subrunoff -> crestimap well-balanced solver
(3DEP DEM on demand) -> compact depth frames -> eventstore (one commit).

DOMAIN = BASIN, NEVER A RECTANGLE (user directive 2026-08-18): the 2-D
domain is the whole catchment contributing to the trigger gauge as a union
of WBD HUC12 units (hf_data/hucdomain.py) — the solver integrates only those
cells (the raster outside them is a sink) and the HUC12 units are the tiles
the parallel tier spreads across devices. RESOLUTION IS FIXED at the DEM's
native cell size. Protocol 2 prepares the full HUC basin and defers a basin
that exceeds native preparation/worker capacity; upstream units are never
dropped merely to fit a memory budget.
crestimap installs from the CREST-iMAP fork's v2 branch (Dockerfile);
everything here degrades to a clear error if it's missing.
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
# ONE FLOOD, ONE EPISODE: a gauge flagged again within EVENT_REOPEN_H of its
# last episode's end REOPENS that episode (same id, same episode_started
# anchor) instead of opening a new event id. A gauge hovering at the
# continuation bar was ended one hour and re-triggered the next, splitting
# one flood into a dozen 2-h episodes (09353800, 08353000; 2026-10-27..31).
# 0 disables reopening.
REOPEN_H = float(os.environ.get("EVENT_REOPEN_H", "12"))
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
# Retained for configuration compatibility. Protocol 2 does not trim HUC
# units to this budget: its full native domain must pass PREP_MAX_CELLS.
QUEUE_MAX_CELLS = int(os.environ.get("EVENT_QUEUE_MAX_CELLS", "0"))
PREP_MAX_CELLS = int(os.environ.get("EVENT_PREP_MAX_CELLS", "35000000"))
PREP_COOLDOWN_H = float(os.environ.get("EVENT_PREP_COOLDOWN_H", "6"))
DEM_RES = os.environ.get("EVENT_DEM_RES", "1")
DEM_CACHE = os.environ.get("EVENT_DEM_CACHE",
                           os.path.join(tempfile.gettempdir(), "dem_cache"))
# RESOLUTION IS FIXED at the DEM's native cell size — a coarsened flood map
# (one pixel = 10 street blocks) is useless information. When the basin
# exceeds the CPU cell budget at native resolution, the CPU rung keeps the
# HUC12 units nearest the gauge (hucdomain.build_domain(max_cells=...)) —
# whole hydrologic units, never a window (user directive 2026-08-18); the
# workers always get the whole basin.
DEM_RES_DEG = {"1": 1.0 / 3600.0, "13": 1.0 / 10800.0}

_running: dict = {"id": None, "status": None, "log": [], "last": None,
                  "started": None}
_lock = threading.Lock()
# A job is marked running only after its expensive setup. Guard the entire
# hourly visit separately, including preflight and retention, so repeated
# updater pings cannot start overlapping SDK reads before run_one begins.
_tick_lock = threading.Lock()
_tick_state_lock = threading.Lock()
_tick_state = {"running": False, "started": None, "completed": None,
               "last_attempt": None, "last_success": None, "phase": "idle",
               "phase_started": None, "last_progress": None, "outcome": None,
               "error": None, "error_phase": None, "last_error_at": None,
               "watchdog_dumped_at": None, "watchdog_phase": None}
_tick_started_mono = None
_tick_phase_mono = None
_tick_watchdog_marker = None
TICK_PREFLIGHT_STALL_S = 15 * 60
# event ids whose local CPU solve blew CPU_MAX_H -> when it happened; the
# tick's CPU-rung sweep skips them for CPU_COOLDOWN_H
_cpu_timeout: dict = {}
_capacity_lock = threading.Lock()
_capacity_skips: dict = {}
_capacity_retry: dict = {}


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
    stamp = _tick_stamp()
    _running["last_tick"] = stamp
    with _tick_state_lock:
        _tick_state["last_attempt"] = stamp


def _tick_stamp():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def tick_status() -> dict:
    """Pure process-memory progress; never reads the dataset or waits for work."""
    with _tick_state_lock:
        out = dict(_tick_state)
        now = time.monotonic()
        out["busy_h"] = (round(max(0, now - _tick_started_mono) / 3600, 3)
                         if out["running"] and _tick_started_mono is not None else None)
        out["phase_age_s"] = (round(max(0, now - _tick_phase_mono), 1)
                              if out["running"] and _tick_phase_mono is not None else None)
        out["preflight"] = out["running"] and out["phase"].startswith("preflight.")
        out["watched_wait"] = out["running"] and _tick_wait_phase(out["phase"])
        out["stalled"] = bool(out["watched_wait"] and out["phase_age_s"] is not None
                               and out["phase_age_s"] >= TICK_PREFLIGHT_STALL_S)
    blocked = capacity_status()
    out["capacity_blocked"] = sum(1 for row in blocked if row.get("relevant", True))
    out["capacity_skips"] = blocked
    return out


def capacity_status():
    """Memory-only capacity deferrals; published maps are never edited here."""
    with _capacity_lock:
        return [dict(row) for row in _capacity_skips.values()]


def _capacity_suppressed(gid):
    gid = str(gid)
    key = (gid, "1", PREP_MAX_CELLS, "protocol2-full-huc-chain")
    with _capacity_lock:
        if time.monotonic() < _capacity_retry.get(key, 0):
            row = _capacity_skips.get(gid)
            if row:
                row["relevant"] = True
                return dict(row)
    return None


def _capacity_reconcile(relevant):
    with _capacity_lock:
        for gid, row in _capacity_skips.items():
            row["relevant"] = gid in relevant


def _capacity_clear(gid):
    with _capacity_lock:
        _capacity_skips.pop(str(gid), None)
        for key in list(_capacity_retry):
            if key[0] == str(gid):
                _capacity_retry.pop(key, None)


def _capacity_skip(gid, event_id, details, retry_h=None):
    now = datetime.datetime.utcnow()
    retry_h = PREP_COOLDOWN_H if retry_h is None else retry_h
    record = {"gauge": str(gid), "event_id": event_id, "skipped": "native_capacity",
              "published": False, "dem_res": "1", "native_resolution": "1 arc-second (~30 m)",
              "limit_cells": PREP_MAX_CELLS, "recorded": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
              "retry_after": (now + datetime.timedelta(hours=retry_h)).strftime(
                  "%Y-%m-%dT%H:%M:%SZ"), "relevant": True,
              "existing_result_preserved": True, **details}
    key = (str(gid), "1", PREP_MAX_CELLS, "protocol2-full-huc-chain")
    with _capacity_lock:
        _capacity_skips[str(gid)] = record
        _capacity_retry[key] = time.monotonic() + retry_h * 3600
    _running["last"] = {"event": event_id, "ok": False,
                        "skipped": "native_capacity", "native_capacity": record}
    return {"event_id": event_id, "gauge": str(gid), "published": False,
            "skipped": "native_capacity", "native_capacity": dict(record)}


def _native_bbox_estimate(bounds):
    """Native 1-arc-second lattice envelope, including the builder's 2-cell pad."""
    w, s, e, n = map(float, bounds)
    if not all(math.isfinite(v) for v in (w, s, e, n)) or w >= e or s >= n:
        raise ValueError("invalid native basin extent")
    res = DEM_RES_DEG["1"]
    west, south = math.floor(w / res) - 2, math.floor(s / res) - 2
    east, north = math.ceil(e / res) + 2, math.ceil(n / res) + 2
    nx, ny = east - west, north - south
    return {"bbox": [west * res, south * res, east * res, north * res],
            "nx": nx, "ny": ny, "n_bbox": nx * ny, "pad_cells": 2, "dem_res": "1"}


def _native_preflight(g, ext, hucdomain):
    """Metadata-only full-chain estimates, before EF5 allocates its raster arrays.
    ext: the gauge's basinextent.ef5_extent — EF5's raster extent is the basin
    outline's, so the EF5 estimate covers it unioned with the HUC12 chain."""
    if str(DEM_RES) != "1" or PREP_MAX_CELLS <= 0:
        raise ValueError("protocol 2 requires nominal 30 m and a positive preparation cell limit")
    h0 = hucdomain.gauge_huc12(float(g["lat"]), float(g["lon"]))
    chain = hucdomain.upstream_chain(h0)
    ids = {str(row["huc12"]) for row in chain}
    if not ids or str(h0) not in ids:
        raise ValueError("full HUC chain unavailable")
    index = hucdomain.index()
    for hid in ids:
        # upstream_chain has a traversal ceiling: require closure so a
        # partial chain can never masquerade as a safe, smaller full basin.
        if any(str(index["huc12"][i]) not in ids for i in index["_up"].get(hid, [])):
            raise ValueError("HUC chain traversal is incomplete")
    ii = [index["_pos"][hid] for hid in ids]
    hb = (float(index["minx"][ii].min()), float(index["miny"][ii].min()),
          float(index["maxx"][ii].max()), float(index["maxy"][ii].max()))
    eb = tuple(map(float, ext["bounds"]))
    ef5_union = (min(hb[0], eb[0]), min(hb[1], eb[1]),
                 max(hb[2], eb[2]), max(hb[3], eb[3]))
    estimates = {"huc_native": _native_bbox_estimate(hb),
                 "ef5_union_native": _native_bbox_estimate(ef5_union),
                 "full_huc_count": len(ids)}
    return estimates, ids


def _tick_wait_phase(phase):
    return phase.startswith("preflight.") or phase == "retention"


def _tick_phase(phase):
    global _tick_phase_mono
    stamp = _tick_stamp()
    with _tick_state_lock:
        _tick_state.update(phase=phase, phase_started=stamp, last_progress=stamp)
        _tick_phase_mono = time.monotonic()


def _tick_note_error(exc):
    with _tick_state_lock:
        _tick_state.update(error=type(exc).__name__, error_phase=_tick_state["phase"],
                           last_error_at=_tick_stamp())


def _tick_watchdog_check() -> bool:
    """Dump each stalled network-wait phase once, to authenticated runtime logs.

    faulthandler prints code paths and stack frames, never frame locals or
    configuration. No stack data is exposed through an HTTP endpoint.
    """
    global _tick_watchdog_marker
    with _tick_state_lock:
        marker = (_tick_started_mono, _tick_phase_mono)
        phase = _tick_state["phase"]
        if (not _tick_state["running"] or not _tick_wait_phase(phase)
                or _tick_phase_mono is None
                or time.monotonic() - _tick_phase_mono < TICK_PREFLIGHT_STALL_S
                or marker == _tick_watchdog_marker):
            return False
        _tick_watchdog_marker = marker
        _tick_state.update(watchdog_dumped_at=_tick_stamp(), watchdog_phase=phase)
    import faulthandler
    import sys
    print(f"event tick watchdog: wait phase {phase} unchanged for at least "
          f"{TICK_PREFLIGHT_STALL_S}s; private thread stacks follow", file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    return True


def _tick_watchdog(stop):
    while not stop.wait(30):
        try:
            _tick_watchdog_check()
        except Exception as exc:
            # Keep only the exception class in runtime diagnostics as well.
            print(f"event tick watchdog: {type(exc).__name__}", flush=True)


def status(full: bool = False) -> dict:
    return {"running": _running["id"], "status": _running["status"],
            "started": _running.get("started"), "busy_h": _busy_h(),
            "log": _running["log"][-(400 if full else 30):],
            "last": _running["last"],
            "missed": _running.get("missed", [])[-20:], "tick": tick_status(),
            "capacity_skips": capacity_status()}


OBS_GATE_FRAC = float(os.environ.get("EVENT_OBS_GATE", "0.3"))   # of Q2
OBS_WINDOW_H = int(os.environ.get("EVENT_OBS_WINDOW_H", "6"))
# Outlet-discharge disagreement is advisory, not a distributed-runoff volume check.
# The same severe ratio selects observed fallback only at safe upstream cuts.
MIN_SIM_FRAC = float(os.environ.get("EVENT_MIN_SIM_FRAC", "0.3"))


def _recent_obs_max(gid: str) -> float | None:
    """Max observed flow [m3/s] at the gauge over the last OBS_WINDOW_H
    hours from NWIS IV; None if the gauge reports nothing."""
    import json as _json
    import urllib.request
    # two hosts: waterservices.usgs.gov answered 503 for a whole day
    # (2026-09-22) while nwis.waterservices.usgs.gov served the same query
    for host in ("nwis.waterservices.usgs.gov", "waterservices.usgs.gov"):
        url = (f"https://{host}/nwis/iv/?format=json"
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
            continue
    return None


def _crest_missed(hydro: dict, trigger: dict | None, gid: str,
                  t_from: datetime.datetime, log=print) -> dict | None:
    """EF5 peak at the trigger gauge from t_from on vs the observed peak
    (hydro obs rows, else the trigger's observation, else NWIS now). A dict
    describing severe under/overprediction, used only as an advisory."""
    if MIN_SIM_FRAC <= 0:
        return None
    key = t_from.strftime("%Y-%m-%d %H:%M")
    rows = [r for k, r in hydro.items() if k >= key] or list(hydro.values())
    sims = [r["sim_q"] for r in rows if r.get("sim_q") is not None]
    obss = [r["obs_q"] for r in rows if r.get("obs_q") is not None]
    if not sims:
        return None
    sim_pk = max(sims)
    src = "hydro"
    obs_pk = max(obss) if obss else None
    if obs_pk is None:
        obs_pk = (trigger or {}).get("obs_m3s")
        src = "trigger"
    if obs_pk is None:
        obs_pk = _recent_obs_max(gid)
        src = "nwis"
    try:
        obs_pk = float(obs_pk)
    except (TypeError, ValueError):
        return None
    if obs_pk <= 0:
        return None
    frac = sim_pk / obs_pk
    log(f"CREST check: EF5 peak {sim_pk:.1f} m3/s vs observed peak "
        f"{obs_pk:.1f} m3/s ({src}) = {100 * frac:.0f}% "
        f"(advisory range {MIN_SIM_FRAC:.2f}..{1 / MIN_SIM_FRAC:.2f}x)")
    if MIN_SIM_FRAC <= frac <= 1 / MIN_SIM_FRAC:
        return None
    return {"sim_peak_m3s": round(sim_pk, 2), "obs_peak_m3s": round(obs_pk, 2),
            "frac": round(frac, 3), "obs_source": src,
            "reason": "severe_underprediction" if frac < MIN_SIM_FRAC else "severe_overprediction"}


def _age_days_started(s: dict) -> float | None:
    """Age [days] of an index entry from its episode start (None unknown)."""
    try:
        t = datetime.datetime.strptime(
            s.get("episode_started") or s.get("t0") or "", "%Y-%m-%dT%H:%MZ")
        return (datetime.datetime.utcnow() - t).total_seconds() / 86400.0
    except ValueError:
        return None


def _reopen_candidate(idx: dict, gid: str, now: datetime.datetime,
                      skip=()) -> str | None:
    """The gauge's most recent episode if it ENDED within REOPEN_H hours —
    measured from final.ended, else t_end, else the entry's generated time —
    and is still inside MAX_EPISODE_H (an over-age episode would just end
    again next tick; a new episode starts instead). None otherwise, and for
    ids in `skip`: episodes ended in THIS tick, whose finalize thread is
    running. A fragment absorbed into another episode (merged_into, from the
    merge script) is never a candidate nor the gauge's most recent episode.
    Pinned events are refused by eventstore.reopen_episodes."""
    if REOPEN_H <= 0:
        return None
    mine = [e for e, s in idx.items() if not s.get("merged_into")
            and str(((s.get("trigger") or {}).get("gauge")) or s.get("gauge") or "") == gid]
    if not mine:
        return None
    eid = max(mine)                   # YYYYMMDDHH_<gauge>: the latest start
    s = idx[eid]
    if eid in skip or s.get("status") != "ended":
        return None
    end = None
    for t in ((s.get("final") or {}).get("ended"), s.get("t_end"), s.get("generated")):
        try:
            end = datetime.datetime.strptime(str(t or "")[:16], "%Y-%m-%dT%H:%M")
            break
        except ValueError:
            continue
    try:
        age_h = (now - datetime.datetime.strptime(
            s.get("episode_started") or s.get("t0") or "", "%Y-%m-%dT%H:%MZ")
        ).total_seconds() / 3600.0
    except ValueError:
        return None
    if (end is None or (now - end).total_seconds() / 3600.0 > REOPEN_H
            or age_h > MAX_EPISODE_H):
        return None
    return eid


def detect(max_events: int = 1) -> list[dict]:
    """Trigger gauges: per hotspot cluster, the tier-3 member with the
    largest drainage area — GATED on real observations. The AI tier alone
    can hallucinate floods at data-dead gauges (seen live: two tier-3
    triggers with zero USGS data and ~0 simulated flow), so a trigger also
    requires the gauge to be reporting AND already at >= EVENT_OBS_GATE of
    bankfull (Q2) within the last few hours."""
    from . import basinextent, nowcaststore, pipeline
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
            if _capacity_suppressed(m["id"]):
                continue
            g = pipeline.gauge_info(m["id"])
            if not g or g["id"] in seen:
                continue
            try:                      # no basin outline -> never a trigger
                basinextent.ef5_extent(g)
            except basinextent.BasinExtentError as e:
                print(f"event trigger {g['id']}: SKIP — {e}")
                continue
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
            defer: bool = False, force_cpu: bool = False,
            cold: bool = False) -> dict:
    """Full event pipeline for one trigger gauge. Blocking (minutes-long);
    call from a worker thread. Returns the manifest (raises on failure).
    episode_id: re-simulate an ongoing episode under its original id (the
    published folder is replaced with the fresh window).
    defer: enqueue for a worker WITHOUT the CPU fallback (manual refills —
    the runner stays free and the worker publishes whenever it gets there).
    force_cpu: retained for older callers; boundary protocol 2 always waits
    for a compatible Jetstream2 worker, because CPU images retain old code.
    cold: do NOT carry the episode's previous depth frame into the bundle as
    a warm start — the workers re-solve the whole anchored span from the
    channel pre-wet, and publish replaces every frame (a full re-run, e.g.
    after the engine/domain changed; the box-era frames must not survive)."""
    from . import eventqueue, eventstore, nowcaststore, pipeline

    suppressed = _capacity_suppressed(gid)
    if suppressed:
        return {"event_id": suppressed.get("event_id"), "gauge": str(gid),
                "published": False, "skipped": "native_capacity",
                "native_capacity": suppressed, "suppressed": True}

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
    work = None

    def log(s):
        _running["log"].append(s)

    with _lock:
        _running.update(id=ev_id, status="native_capacity", log=[],
                        started=datetime.datetime.utcnow().strftime(
                            "%Y-%m-%dT%H:%M:%SZ"))
    try:
        # EF5's raster extent and outlet cell come from the gauge's basin
        # outline (basinextent: GAGES-II polygon ∪ the carved catchment, on
        # the HydroSHEDS lattice) — the same extent as the dashboard's runs,
        # so they share clip and forcing stores. No basin -> no event.
        from . import basinextent, hucdomain
        ext = basinextent.ef5_extent(g)
        log(f"EF5 extent from the basin outline ({ext['source']}, "
            f"{ext['carved_km2']:,.0f} km² catchment, {ext['cells']:,} cells)")
        if _tick_state.get("running"):
            _tick_phase(f"preflight.native_capacity:{gid}")
        try:
            estimates, full_huc_ids = _native_preflight(g, ext, hucdomain)
        except Exception as e:
            log(f"native preparation deferred: full basin extent/chain unavailable ({type(e).__name__})")
            return _capacity_skip(gid, ev_id, {
                "reason": "native_domain_incomplete", "error": type(e).__name__,
                "needs": "verified_full_basin_extent_and_huc_chain"}, retry_h=.25)
        if max(estimates["huc_native"]["n_bbox"], estimates["ef5_union_native"]["n_bbox"]) > PREP_MAX_CELLS:
            log(f"native preparation deferred: full HUC bbox {estimates['huc_native']['n_bbox']:,} "
                f"cells; EF5 union {estimates['ef5_union_native']['n_bbox']:,} cells; limit "
                f"{PREP_MAX_CELLS:,}; a larger native-grid worker/preparation host is required")
            return _capacity_skip(gid, ev_id, {
                "reason": "full_basin_exceeds_native_capacity", "estimates": estimates,
                "needs": "larger_native_grid_worker_and_preparation_capacity"})
        _capacity_clear(gid)
        log(f"native preparation capacity: full HUC bbox {estimates['huc_native']['n_bbox']:,}; "
            f"EF5 union {estimates['ef5_union_native']['n_bbox']:,}; limit {PREP_MAX_CELLS:,}")
        work = tempfile.mkdtemp(prefix=f"event_{gid}_")
        with _lock:
            _running["status"] = "ef5"
        if _tick_state.get("running"):
            _tick_phase(f"run:{gid}")
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
            elif kind == "done" and payload.get("no_basin"):
                raise hucdomain.DomainError(f"{gid}: no basin boundary "
                                            f"({payload.get('reason')})")
            elif kind == "done" and payload.get("returncode", 0) not in (0, None):
                raise RuntimeError(f"EF5 failed: {payload}")
        out_dir = os.path.join(work, "CREST_output")
        model = (meta.get("model") or "crest").lower()
        wb_model = "crest" if model in ("crest", "hp") else "crestphys"

        # Outlet Q is evaluation/advisory data, never an injected boundary.
        # User policy 2026-10-27: use observed Q only at safe upstream cuts;
        # otherwise let the runoff-driven 2-D solve assess the water supplied.
        miss = _crest_missed(hydro, trigger, gid, sim_anchor, log)
        if miss:
            trigger = {**(trigger or {}), "outlet_q_advisory": {
                **miss, "action": "continue_runoff_driven_2d",
                "note": "Outlet Q mismatch is not a distributed-runoff volume check"}}
            log(f"outlet Q advisory: EF5 peak is {100 * miss['frac']:.0f}% of "
                "observed; continuing 2-D preparation with explicit boundary policy")

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

        # THE DOMAIN IS THE BASIN: the whole catchment contributing to the
        # gauge as a union of HUC12 units, written into the EF5 dir as
        # domain_huc.tif (+ domain.geojson) so the queue bundle carries it
        # and every engine solves the same basin. No box anywhere: a gauge
        # with no basin geometry is an error, not a rectangle.
        from . import hucdomain
        with _lock:
            _running["status"] = "domain"
        dom = hucdomain.build_domain(g, out_dir, dem_res=DEM_RES,
                                     max_cells=0, log=log)
        built_huc_ids = {str(row["huc12"]) for row in dom.get("hucs", [])}
        if dom.get("trimmed") or built_huc_ids != full_huc_ids:
            log("native domain incomplete: generation must retain the full preflight HUC chain")
            return _capacity_skip(gid, ev_id, {
                "reason": "native_domain_incomplete", "estimates": estimates,
                "full_huc_count": len(full_huc_ids), "built_huc_count": len(built_huc_ids),
                "trimmed": bool(dom.get("trimmed")),
                "needs": "verified_full_basin_domain_without_upstream_trimming"}, retry_h=.25)
        if int(dom.get("n_bbox") or 0) > PREP_MAX_CELLS:
            return _capacity_skip(gid, ev_id, {
                "reason": "full_basin_exceeds_native_capacity", "estimates": estimates,
                "actual_domain_bbox_cells": dom["n_bbox"],
                "needs": "larger_native_grid_worker_and_preparation_capacity"})
        basin_ring = dom["basin"]
        dom_extent = tuple(dom["bbox"])        # raster extent of the HUC12 union
        # worker spec = full anchored span (fixed episode window); the local
        # CPU fallback still solves only the fresh 24-h slice — its frames
        # merge into the fixed-start record at publish
        sim_start = sim_anchor
        sim_start_local = max(sim_anchor,
                              t0 - datetime.timedelta(hours=SIM_BACKSET_H))

        from . import boundary_fallback
        coupling, dom = boundary_fallback.prepare(
            g, out_dir, dom, sim_start, t_end, dem_res=DEM_RES,
            min_ratio=(MIN_SIM_FRAC if 0 < MIN_SIM_FRAC < 1 else 0.3),
            model=wb_model, log=log)
        trigger = {**(trigger or {}), "boundary_policy": {
            "version": coupling["version"], "mode": coupling["mode"],
            "implementation": coupling["implementation"],
            "fingerprint": coupling["fingerprint"],
            "sources": [{k: i[k] for k in ("gauge_id", "source", "reason")}
                        for i in coupling["inlets"]],
            "excluded_gauges": coupling["excluded_gauges"]}}

        # warm continuation (user directive 2026-08-14): the hourly update
        # inherits the previous run's water state. The episode's latest
        # published depth frame at/just before this run's fresh-slice start
        # is dropped into the forcing dir as init_depth.tif — the enqueue
        # tars every .tif, so workers get it too, and run_event/chunkrun
        # auto-discover it (h0 = max(channel pre-wet, previous depth)).
        if episode_id and cold:
            log("cold re-run: no warm-start frame — the whole anchored span "
                "is re-solved and every published frame replaced")
        elif episode_id and coupling["subsurface_routing"]["mode"] == "kw":
            log("KW interflow requires its full water/IR checkpoint; no depth-only "
                "warm start is copied (worker resumes its complete local session or reconstructs from anchor)")
        elif episode_id:
            try:
                pm = eventstore._prior_manifest(ev_id) or {}
                prior_fingerprint = ((pm.get("coupling") or {}).get("fingerprint")
                                     or pm.get("forcing_fingerprint")
                                     or ((pm.get("trigger") or {}).get("boundary_policy") or {}).get("fingerprint"))
                if prior_fingerprint != coupling["fingerprint"]:
                    raise ValueError("boundary policy/domain changed; prior depth cannot initialize this solve")
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

        # Offer the native-resolution basin and its versioned forcing
        # contract to the compatible Jetstream2 worker.
        queued = False
        if eventqueue.mode() != "off":
            spec = {"event_id": ev_id, "episode": bool(episode_id),
                    "worker_protocol": 2,
                    "required_engine_prefix": "jetstream2:",
                    "coupling": {k: coupling[k] for k in
                                 ("version", "policy", "mode", "fingerprint", "implementation",
                                  "domain_mask_sha256", "subsurface_routing")},
                    "boundary_content_sha256": coupling["content_sha256"],
                    "forcing_fingerprint": coupling["fingerprint"],
                    "interflow": "kw",
                    "gauge": {"id": gid, "lat": g["lat"], "lon": g["lon"],
                              "area_km2": g.get("area")},
                    "bbox_basin": list(dom_extent),
                    # basin domain summary (units, area, active/bbox cells);
                    # the label raster itself rides in the forcing tar
                    "domain": {k: v for k, v in dom.items() if k != "basin"},
                    "huc12s": [h["huc12"] for h in dom["hucs"]],
                    # cold re-run stamp: workers drop any resident session
                    # for this episode and re-solve from the anchor
                    "cold": (datetime.datetime.utcnow().strftime(
                        "%Y-%m-%dT%H:%M:%SZ") if cold else None),
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
                    log("no worker claim — bundle stays queued for a compatible "
                        "Jetstream2 worker (CPU fallback disabled for boundary protocol 2)")
                    _running["last"] = {"event": ev_id, "ok": True,
                                        "queued": True}
                    return {"event_id": ev_id, "published": False,
                            "queued": True}

        # CPU images retain the previous solver pin. Only the compatible
        # Jetstream2 worker may consume this versioned forcing contract.
        if queued:
            log("boundary protocol 2: waiting for a compatible Jetstream2 worker; "
                "CPU fallback is disabled for this policy")
            _running["last"] = {"event": ev_id, "ok": True, "queued": True,
                                "worker_protocol": 2}
            return {"event_id": ev_id, "published": False, "queued": True,
                    "worker_protocol": 2}
        raise RuntimeError("boundary protocol 2 requires a successfully queued compatible worker job")

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
    """Run one complete hourly visit; overlapping pings return immediately."""
    global _tick_started_mono
    note_tick_attempt()
    if not _tick_lock.acquire(blocking=False):
        return {"skipped": "runner busy", "tick": tick_status()}
    stop = threading.Event()
    result = None
    try:
        stamp = _tick_stamp()
        with _tick_state_lock:
            _tick_started_mono = time.monotonic()
            _tick_state.update(running=True, started=stamp, error=None, error_phase=None,
                               outcome=None)
        _tick_phase("preflight.runner")
        threading.Thread(target=_tick_watchdog, args=(stop,), daemon=True,
                         name="event-tick-watchdog").start()
        if _running["id"]:
            result = {"skipped": "runner busy"}
        else:
            result = _hourly_tick_body()
        return result
    except BaseException as exc:
        _tick_note_error(exc)
        raise
    finally:
        stop.set()
        stamp = _tick_stamp()
        with _tick_state_lock:
            outcome = ("error" if _tick_state["error"] else
                       "skipped" if result and result.get("skipped") else "ok")
            _tick_state.update(running=False, completed=stamp, outcome=outcome,
                               phase="failed" if outcome == "error" else "idle",
                               phase_started=stamp, last_progress=stamp)
            if outcome == "ok":
                _tick_state["last_success"] = stamp
        _tick_lock.release()


def _hourly_tick_body() -> dict:
    """Called by the updater after each hourly data refresh.

    1. Every ACTIVE episode whose gauge is still at/above EVENT_CONT_TIER, or
       is flagged again by this hour's detection, is re-simulated at the
       fresh t0 (same event id; folder replaced).
    2. Episodes whose gauge receded below the tier (flow back to normal) or
       older than EVENT_MAX_EPISODE_H are marked ended — precipitation going
       to zero does NOT end an episode while the hydrograph is still high.
    3. Newly flagged tier-3 gauges start new episodes — or reopen their
       gauge's episode that ended within EVENT_REOPEN_H (same id + anchor).
    Runs at most EVENT_MAX_PER_TICK simulations, sequentially.
    """
    from . import eventstore, nowcaststore
    _tick_phase("preflight.conus_state")
    try:                       # CONUS state mosaic: index the fleet's new bundles
        from . import conusstate
        conusstate.maybe_refresh()
    except Exception as exc:
        _tick_note_error(exc)
    _tick_phase("preflight.nowcast_risk")
    risk = nowcaststore.all_risk()
    tiers = (risk or {}).get("tiers") or {}
    _capacity_reconcile({str(gid) for gid, tier in tiers.items() if int(tier or 0) >= CONT_TIER})
    _tick_phase("preflight.event_index")
    idx = eventstore.load_index()
    now = datetime.datetime.utcnow()

    # detection runs BEFORE the end check: a gauge flagged again this hour is
    # still flooding, so its episode is not ended in this tick (ending it and
    # opening a new id for the same detection is what fragmented floods).
    # The budget covers every active gauge plus EVENT_MAX_PER_TICK new ones.
    _tick_phase("preflight.detect")
    try:
        picks = detect(MAX_PER_TICK + sum(
            1 for s in idx.values() if s.get("status", "active") == "active"))
    except Exception as exc:
        _tick_note_error(exc)
        picks = []
    detected = {p["gauge"] for p in picks}

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
        # up a gauge that went silent (obs None). A re-detected gauge goes on
        # regardless (detect() just read its observations).
        _tick_phase(f"preflight.episode_observations:{gid}")
        obs = _recent_obs_max(gid) if gid and gid not in detected else None
        qbase = None
        try:
            _tick_phase(f"preflight.episode_thresholds:{gid}")
            threshold = nowcaststore._thresholds().get(gid)
            # Live threshold rows are (qbase, q2, q5, q10); older callers
            # and fixtures provide a dictionary. Preserve observation-led
            # episode continuation for both representations.
            value = (threshold[0] if isinstance(threshold, (tuple, list)) and threshold
                     else (threshold or {}).get("qbase"))
            qbase = float(value or 0)
            if not math.isfinite(qbase) or qbase <= 0:
                qbase = None
        except Exception:
            pass
        if obs is not None and qbase:
            still_high = obs >= max(CONT_OBS_MULT * qbase, 1.0)
        else:
            still_high = tier >= CONT_TIER
        if gid and (still_high or gid in detected) and age_h <= MAX_EPISODE_H:
            if _capacity_suppressed(gid):
                continue  # keep the episode intact; capacity is not recession.
            # publish progress stale + nobody claiming -> this re-sim runs
            # on the CPU rung (V30 demotion: last resort, not default). A
            # reopened episode's clock starts at the reopen: its last publish
            # predates the end, and the CPU rung must not engage an hour later
            pub = []
            for k in ("generated", "reopened"):
                try:
                    pub.append(datetime.datetime.strptime(
                        s.get(k) or "", "%Y-%m-%dT%H:%M:%SZ"))
                except ValueError:
                    pass
            pub_age_h = ((now - max(pub)).total_seconds() / 3600.0
                         if pub else float("inf"))
            jobs.append({"gauge": gid, "episode": eid,
                         "force_cpu": pub_age_h > CPU_FALLBACK_H,
                         "trigger": {**(s.get("trigger") or {}), "tier": tier}})
        else:
            ended.append(eid)
    if ended:
        _tick_phase("preflight.mark_ended")
        eventstore.mark_ended(ended)     # fast: stops re-sims immediately

    from . import eventqueue as _eq
    active_gauges = {j["gauge"] for j in jobs}
    new_picks, fresh = [], []
    for pick in picks:
        if len(fresh) >= MAX_PER_TICK:
            break
        if _capacity_suppressed(pick["gauge"]):
            continue
        if pick["gauge"] in active_gauges:
            continue
        _tick_phase(f"preflight.queue_gauge:{pick['gauge']}")
        if _eq.mode() == "on" and _eq.gauge_pending(pick["gauge"]):
            continue           # already in the worker queue (index lags it)
        # never an id ended in this tick: its finalize thread starts below
        prev = _reopen_candidate(idx, pick["gauge"], now, skip=ended)
        # a worker is mid-solve on it, or its spec waits in the queue
        # unclaimed (a cold re-run: gauge_pending ignores those) — run_one
        # would replace that spec; the queued job's publish reactivates it.
        # An unreadable queue counts as queued: the next tick retries.
        if prev and _eq.mode() == "on":
            try:
                queued = _eq.claim_fresh_for(prev) or _eq._spec(prev, strict=True)
            except Exception:
                queued = True
            if queued:
                continue
        fresh.append((pick, prev))
    reopened, superseded, refused, failed = set(), set(), set(), set()
    if any(prev for _, prev in fresh):
        _tick_phase("preflight.reopen")
        want = [p for _, p in fresh if p]
        got = eventstore.reopen_episodes(want, superseded=superseded,
                                         refused=refused)
        if got is None:                  # failed: index unreadable / commit raised
            # ids it refused (pinned, gone from the index) start a new id as ever
            failed = set(want) - superseded - refused
        reopened = set(got or ())
    for pick, prev in fresh:
        if prev in reopened:
            print(f"event tick: {pick['gauge']} flagged again — episode {prev} "
                  f"reopened (ended < {REOPEN_H:g} h ago)")
            new_picks.append({"gauge": pick["gauge"], "episode": prev,
                              "trigger": {**(idx[prev].get("trigger") or {}),
                                          "tier": int(tiers.get(pick["gauge"], 0))}})
        elif prev in superseded:
            # the fresh index has a newer or active episode at this gauge (or
            # merged prev into one) since this tick read it: that episode is
            # the flood now. A new id would fragment it again, so nothing
            # starts for the gauge this tick; the next tick's index decides.
            print(f"event tick: {pick['gauge']} flagged again — episode {prev} "
                  f"superseded in the index; nothing started this tick")
        elif prev in failed:
            # the reopen itself failed (not refused): a new id would fragment
            # the flood just the same, so nothing starts for the gauge this
            # tick; the next tick retries the reopen
            print(f"event tick: {pick['gauge']} flagged again — reopening "
                  f"episode {prev} failed; nothing started this tick")
        else:
            new_picks.append({"gauge": pick["gauge"], "episode": None,
                              "trigger": pick})
    # V30 episode final products: consolidate ended episodes (stats + the
    # episode-scoped nowcast verification) in a background thread — scoring
    # downloads ~1 archived nowcast per episode-hour and must not delay the
    # tick. Also self-heals: recently-ended episodes still missing "final"
    # (e.g. a worker publish raced the index commit) get another pass —
    # except the ones just reopened (dispatched after the reopen for that).
    heal = [e for e, s in idx.items()
            if s.get("status") == "ended" and not s.get("final")
            and (_age_days_started(s) or 99.0) < 3.0 and e not in reopened]
    to_final = ended + [e for e in heal if e not in ended]
    if to_final:
        _tick_phase("preflight.finalize_dispatch")
        threading.Thread(target=eventstore.finalize_episodes,
                         args=(to_final, print), daemon=True).start()
    # V30 CPU-rung last resort: queued bundles that were never published
    # (not in the index) and sat unclaimed past CPU_FALLBACK_H get a local
    # solve — enqueue-only detections and deferred refills must not strand
    # when every worker is dead or busy. t0 comes from the event id so a
    # backdated refill keeps its window.
    if _eq.mode() == "on":
        _tick_phase("preflight.queue_status")
        for row in _eq.queue_status():
            rid, rgid = str(row.get("id") or ""), str(row.get("gauge") or "")
            if _capacity_suppressed(rgid):
                continue
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
    results = []
    for j in jobs:
        if len(results) >= MAX_PER_TICK:
            break
        try:
            _tick_phase(f"run:{j['gauge']}")
            m = run_one(j["gauge"], t0=j.get("t0"), trigger=j["trigger"],
                        episode_id=j["episode"],
                        force_cpu=bool(j.get("force_cpu")))
            if m.get("skipped") == "native_capacity":
                continue  # a deferred map does not consume a simulation slot.
            results.append({"event_id": m.get("event_id"),
                            "published": m.get("published")})
        except Exception as e:
            _tick_note_error(e)
            results.append({"gauge": j["gauge"],
                            "error": f"{type(e).__name__}: {e}"})
    try:                       # age out old events (no-op most hours)
        _tick_phase("retention")
        swept = eventstore.retention_sweep()
    except Exception as exc:
        _tick_note_error(exc)
        swept = 0
    blocked = [row for row in capacity_status() if row.get("relevant", True)]
    return {"ran": len(results), "ended": ended, "swept": swept,
            "results": results, "capacity_blocked": len(blocked), "capacity_skips": blocked}

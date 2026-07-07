"""End-to-end pipeline orchestrator: query -> live CREST run.

Chains the HF-backed data layer + (mock/real) EF5 run into one streaming
generator the chat drives:

  ACP  parse query -> anchor coord + time window   (LLM on the Space; deterministic fallback here)
  AOS  nearest calibrated outlet gauge             (hf_data.gauges)
  ADR  clip DEM/DDM/FAM + fetch forcing PQF         (hf_data.basic / hf_data.forcing)
  API  calibrated CRESTPHYS+KW multipliers x grids  (hf_data.multipliers / hf_data.params)
  AO   render control -> run -> stream ts + 2-D Q    (hf_data.control / hf_data.runner)

Each stage yields a ('status'|'hydro'|'q2d'|'done', payload) event.
"""
from __future__ import annotations

import glob
import math
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from hf_data import basic, forcing, gauges, multipliers, obs, params, paramstore, statecache
from hf_data import snow as _snow
from hf_data.control import build_control, ControlSpec, Gauge
from hf_data.runner import MockEF5, run_ef5, stream_run


# --------------------------------------------------------------------------- #
# ACP — parse a query into an anchor coordinate + time window
# --------------------------------------------------------------------------- #
@dataclass
class EventCtx:
    anchor: tuple[float, float]     # (lat, lon)
    t_start: datetime
    t_end: datetime
    label: str
    gauge_hint: str | None = None   # if the query named a gauge id
    time_known: bool = True         # False -> dates are a guess; UI asks the user

# tiny offline gazetteer so the demo works without an LLM key
_GAZETTEER = {
    "kerr": (30.05, -99.14), "central texas": (30.5, -99.0), "texas hill country": (30.05, -99.14),
    "allagash": (47.07, -69.08), "fort cobb": (35.15, -98.47), "oklahoma": (35.5, -97.5),
}


def parse_query(query: str, hours: int = 48, llm_model: str | None = None) -> EventCtx:
    """LLM parse when a key is present; else a deterministic fallback."""
    q = query.strip()
    # explicit gauge id
    m = re.search(r"\b(\d{8})\b", q)
    if m:
        c = gauges.get_gauge_coordinates(m.group(1)) or (35.0, -97.0)
        return EventCtx(anchor=c, t_start=datetime(2025, 7, 3), t_end=datetime(2025, 7, 3) + timedelta(hours=hours),
                        label=f"gauge {m.group(1)}", gauge_hint=m.group(1), time_known=False)
    # explicit "lat,lon"
    m = re.search(r"(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)", q)
    if m:
        c = (float(m.group(1)), float(m.group(2)))
        return EventCtx(anchor=c, t_start=datetime(2025, 7, 3), t_end=datetime(2025, 7, 3) + timedelta(hours=hours),
                        label=f"{c[0]:.3f},{c[1]:.3f}", time_known=False)
    # LLM path — vLLM first, OpenAI last (hf_data.llm)
    from hf_data import llm as _llm
    if _llm.available():
        try:
            return _parse_with_llm(q, hours)
        except Exception:
            pass
    # offline gazetteer fallback
    ql = q.lower()
    for key, c in _GAZETTEER.items():
        if key in ql:
            return EventCtx(anchor=c, t_start=datetime(2025, 7, 3), t_end=datetime(2025, 7, 3) + timedelta(hours=hours),
                            label=key.title(), time_known=False)
    raise ValueError("Could not parse a location. Try a place name, 'lat,lon', or a USGS gauge id "
                     "(an OpenAI key enables free-form parsing).")


def _parse_with_llm(query: str, hours: int) -> EventCtx:
    """Parse a free-form query via the LLM router (vLLM -> OpenAI)."""
    import json
    from hf_data import llm
    sys_p = ("You extract a flood-event location and time window from a free-form query "
             "(which may include a news-article URL — use its slug/date if present). "
             "Return STRICT JSON only.")
    user_p = (f'Query: "{query}".\n'
              'Return JSON: {"location_name": str (e.g. "Kerrville, TX"), '
              '"lat": float, "lon": float, '
              '"start": "YYYY-MM-DD" or null, "end": "YYYY-MM-DD" or null}. '
              'lat/lon = the event centre. If you can identify the specific event, give its '
              'date range; if the query names a season/month, pick a plausible range within '
              'it; if the query gives NO usable time information at all, set start to null.')
    txt, provider = llm.chat(
        [{"role": "system", "content": sys_p}, {"role": "user", "content": user_p}],
        temperature=0.1, json_mode=True)
    d = json.loads(txt)
    if not d.get("start"):
        t0 = datetime(2025, 7, 3)
        return EventCtx(anchor=(float(d["lat"]), float(d["lon"])), t_start=t0,
                        t_end=t0 + timedelta(hours=hours),
                        label=f'{d["location_name"]} · {provider}', time_known=False)
    t0 = datetime.fromisoformat(d["start"])
    t1 = datetime.fromisoformat(d["end"]) if d.get("end") else t0 + timedelta(hours=hours)
    return EventCtx(anchor=(float(d["lat"]), float(d["lon"])), t_start=t0, t_end=t1,
                    label=f'{d["location_name"]} · {provider}')


# --------------------------------------------------------------------------- #
# AOS — nearest gauge (prefers own-calibration) + basin window
# --------------------------------------------------------------------------- #
def select_outlet(anchor, search_deg: float = 1.0):
    lat, lon = anchor
    cat = gauges.load_catalog((lon - search_deg, lat - search_deg, lon + search_deg, lat + search_deg))
    if cat.empty:
        raise ValueError(f"no gauges within {search_deg}° of {anchor}")
    cat = cat.assign(_d=((cat.LAT_GAGE - lat) ** 2 + (cat.LNG_GAGE - lon) ** 2) ** 0.5)
    row = cat.sort_values("_d").iloc[0]
    return {"id": str(row.STAID).zfill(8), "name": str(row.STANAME),
            "lat": float(row.LAT_GAGE), "lon": float(row.LNG_GAGE),
            "area": float(row.DRAIN_SQKM)}


def basin_bbox(g, pad_factor: float = 1.2):
    """Generous box around the outlet sized from drainage area (EF5 masks the catchment)."""
    r = max(0.3, min(2.5, pad_factor * math.sqrt(max(g["area"], 1.0)) / 111.0))
    return (g["lon"] - r, g["lat"] - r, g["lon"] + r, g["lat"] + r)


# --------------------------------------------------------------------------- #
# full pipeline (generator of events)
# --------------------------------------------------------------------------- #
def analyze(query: str, use_mock: bool = True, hours: int = 48, model: str = "crestphys",
            workdir: str | None = None):
    work = workdir or tempfile.mkdtemp(prefix="crest_demo_")
    basic_dir = os.path.join(work, "BasicData_Clip")
    param_dir = os.path.join(work, "param")
    mrms_dir = os.path.join(work, "CREST_input", "MRMS")
    pet_dir = os.path.join(work, "CREST_input", "PET")
    out_dir = os.path.join(work, "CREST_output")

    yield ("status", f"🧭 Parsing “{query}”…")
    ctx = parse_query(query, hours=hours)
    yield ("status", f"📍 {ctx.label} → anchor {ctx.anchor[0]:.3f}, {ctx.anchor[1]:.3f} · "
                     f"{ctx.t_start:%Y-%m-%d}→{ctx.t_end:%Y-%m-%d}")

    g = select_outlet(ctx.anchor)
    yield ("status", f"🎯 Outlet gauge **{g['id']}** {g['name']} ({g['area']:.0f} km²)")
    bbox = basin_bbox(g)

    yield ("status", "🗺️ Clipping DEM / flow-dir / flow-acc from HydroSHEDS COGs…")
    basic.clip_basic_data(bbox, basic_dir)

    yield ("status", "⚙️ Loading calibrated CRESTPHYS+KW multipliers + param grids…")
    wbkw = multipliers.to_control_params(g["id"], model=model)
    if wbkw is None:
        raise ValueError(f"no calibrated multipliers for gauge {g['id']}")
    wb, kw = wbkw
    grids = params.clip_param_grids(bbox, param_dir)

    if not use_mock:
        yield ("status", "🌧️ Fetching MRMS + PET forcing (PQF) for the event window…")
        forcing.prepare_forcing("mrms", bbox, ctx.t_start, ctx.t_end, mrms_dir)
        forcing.prepare_forcing("pet", bbox, ctx.t_start, ctx.t_end, pet_dir)

    yield ("status", "📝 Writing CRESTPHYS control file…")
    spec = ControlSpec(
        control_path=os.path.join(work, "control.txt"),
        time_begin=ctx.t_start, time_end=ctx.t_end, timestep="1h",
        basic_dir=basic_dir, precip_dir=mrms_dir, pet_dir=pet_dir, output_dir=out_dir,
        gauges=[Gauge(g["id"], g["lon"], g["lat"], g["area"])],
        crest=wb, kw=kw, model=model.upper(), param_grids=grids)
    build_control(spec)

    yield ("status", f"🚀 Running {'(mock) ' if use_mock else ''}CREST — streaming live…")
    if use_mock:
        handle = MockEF5(out_dir, gauge_id=g["id"], model=model, bounds=bbox,
                         n_steps=int(hours), t0=ctx.t_start, delay=0.15).start()
    else:
        handle = run_ef5(spec.control_path, out_dir, g["id"], model=model)

    for ev in stream_run(handle, poll=0.2):
        yield (ev["kind"], ev)          # 'hydro' | 'q2d' | 'done'


WEST_LON = -105.0          # west of the Rocky Mountain front -> CREST (task #8)
WARMUP_DAYS = 90           # 3-month warm-up to build the initial model state (task #6)

# one EF5 run per (gauge, model) at a time: concurrent runs of the SAME gauge
# would race on the shared state dir + result-cache JSON. A second request
# (double-click, or another user picking the same gauge) waits, then usually
# gets served straight from the cache the first run just wrote.
_RUN_LOCKS: dict[tuple, threading.Lock] = {}
_RUN_LOCKS_GUARD = threading.Lock()


def _run_lock(gauge_id: str, model: str) -> threading.Lock:
    key = (str(gauge_id).zfill(8), model)
    with _RUN_LOCKS_GUARD:
        if key not in _RUN_LOCKS:
            _RUN_LOCKS[key] = threading.Lock()
        return _RUN_LOCKS[key]


def gauge_info(gauge_id: str):
    df = gauges._catalog()
    sid = str(gauge_id).zfill(8)
    row = df.loc[df.STAID == sid]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"id": sid, "name": str(r.STANAME), "lat": float(r.LAT_GAGE),
            "lon": float(r.LNG_GAGE), "area": float(r.DRAIN_SQKM)}


def run_gauge(gauge_id: str, t_start: datetime, t_end: datetime, model: str = "auto",
              use_mock: bool = True, hours: int = 48, overrides: dict | None = None,
              snow: str = "auto", timestep: str = "1h", warmup_days: int = WARMUP_DAYS,
              grids: bool = True, no_cache: bool = False,
              workdir: str | None = None):
    """Per-gauge streaming run (map flow: gauge already chosen). Yields events."""
    g = gauge_info(gauge_id)
    if g is None:
        yield ("status", f"⚠️ gauge {gauge_id} not found")
        yield ("done", {"returncode": -1})
        return
    if model == "auto":                          # western-CONUS switch (task #8)
        model = "crest" if g["lon"] < WEST_LON else "crestphys"
    ef5_model = model                            # MODEL= + EF5 output-file naming
    wb_model = "crest" if model in ("crest", "hp") else "crestphys"  # multiplier source
    yield ("meta", {**g, "model": model})        # for the report + right panel

    lock = _run_lock(g["id"], ef5_model)
    if not lock.acquire(blocking=False):
        yield ("status", "⏳ another simulation of this gauge is already running — "
                         "queued behind it (its results are shared via the cache)")
        lock.acquire()
    try:
        yield from _run_gauge_body(g, model, ef5_model, wb_model, t_start, t_end,
                                   use_mock, overrides, snow, timestep,
                                   warmup_days, grids, no_cache, workdir)
    finally:
        lock.release()


def _run_gauge_body(g, model, ef5_model, wb_model, t_start, t_end, use_mock,
                    overrides, snow, timestep, warmup_days, grids, no_cache, workdir):
    """The actual per-gauge run — called with the (gauge, model) lock held."""
    work = workdir or tempfile.mkdtemp(prefix=f"crest_{g['id']}_")
    basic_dir = os.path.join(work, "BasicData_Clip")
    param_dir = os.path.join(work, "param")
    out_dir = os.path.join(work, "CREST_output")
    bbox = basin_bbox(g)

    # --- result cache: reuse overlap, simulate only the missing window (task #6) ---
    # the row cache is hourly; a sub-hourly run neither reuses nor writes it
    hourly = timestep == "1h" and not no_cache
    pl = statecache.plan(g["id"], ef5_model, t_start, t_end) if timestep == "1h" else {
        "cached_rows": [], "run_start": t_start, "run_end": t_end,
        "load_state_time": None, "warmup_from": None, "need_warmup": True,
        "reason": "sub-hourly timestep — cache bypassed"}
    if no_cache:            # calibration: fresh full-window run; candidates
        # still warm-start from any state saved on disk at/near t_start
        ex, wfrom, needw = statecache._state_choice(g["id"], ef5_model, t_start)
        pl = {"cached_rows": [], "run_start": t_start, "run_end": t_end,
              "load_state_time": ex, "warmup_from": wfrom, "need_warmup": needw,
              "reason": "calibration run — row cache bypassed"}
    if pl["run_start"] is None and grids:
        # rows are cached, but the 2-D streamflow maps need rendered frames —
        # if none are cached on disk, re-run the window (fast: warm-starts
        # from the saved state) so the map animation always appears
        from hf_data import viz as _viz
        if not _viz.has_frames_cache(g["id"], ef5_model, t_start, t_end):
            lt, wf, nw = statecache._state_choice(g["id"], ef5_model, t_start)
            pl = {"cached_rows": [], "run_start": t_start, "run_end": t_end,
                  "load_state_time": lt, "warmup_from": wf, "need_warmup": nw,
                  "reason": "re-run to render the 2-D streamflow maps"}
            yield ("status", "hydrograph is cached but the 2-D streamflow maps "
                             "aren't — re-running the window to render them")
    if pl["cached_rows"]:
        yield ("status", f"♻️ reused {len(pl['cached_rows'])} cached step(s) "
                         f"({pl['cached_rows'][0]['time']}…{pl['cached_rows'][-1]['time']})")
        yield ("hydro", {"rows": pl["cached_rows"], "cached": True})
    if pl["run_start"] is None:                       # fully cached -> no simulation
        yield ("status", "✓ served entirely from cache")
        yield ("done", {"returncode": 0, "cached": True,
                        "window": [t_start.strftime(statecache.TS_FMT),
                                   t_end.strftime(statecache.TS_FMT)]})
        return
    run_start, run_end = pl["run_start"], pl["run_end"]
    run_hours = max(1, int(round((run_end - run_start).total_seconds() / 3600)))
    yield ("status", f"🕐 simulation window {run_start:%Y-%m-%d %H:%M} → "
                     f"{run_end:%Y-%m-%d %H:%M} ({run_hours} h @ {timestep})")

    yield ("status", f"clip DEM/DDM/FAM · model {model.upper()}")
    clip = basic.clip_basic_data(bbox, basic_dir)
    if clip.derived:
        yield ("status", "⚠️ HydroSHEDS dir/acc unusable here — flow direction + "
                         "accumulation re-derived from the DEM (pysheds)")
    try:                                   # sanity: clip values at the gauge cell
        import rasterio as _rio
        vals = {}
        for tag in ("dem", "fdir", "facc"):
            with _rio.open(os.path.join(basic_dir, f"{tag}_clip.tif")) as ds:
                r_, c_ = ds.index(g["lon"], g["lat"])
                vals[tag] = float(ds.read(1)[r_, c_])
        yield ("status", f"clip @gauge (r{r_},c{c_}): dem={vals['dem']:.0f} "
                         f"fdir={vals['fdir']:.0f} facc={vals['facc']:.0f}")
    except Exception as e:
        yield ("status", f"(clip sample failed: {e})")

    wbkw = multipliers.to_control_params(g["id"], model=wb_model)
    if wbkw is None:
        yield ("status", "⚠️ no calibrated params for this gauge")
        yield ("done", {"returncode": -1})
        return
    wb, kw = wbkw
    if model == "hp":
        # HP water balance has its own 2 params (fractions in [0,1]);
        # calibrated KW routing params are still used
        wb = {"precip": 1.0, "split": 0.5}
    stored = paramstore.get(g["id"], ef5_model)       # best-known set for this basin
    if stored:
        wb = {**wb, **{k: v for k, v in stored.get("wb", {}).items() if k in wb}}
        kw = {**kw, **{k: v for k, v in stored.get("kw", {}).items() if k in kw}}
        yield ("status", f"🎯 using stored best parameters ({stored.get('source','?')}, "
                         f"NSE {stored.get('nse')}, {stored.get('when','')})")
    if overrides:                                     # advanced-panel overrides
        wb = {**wb, **{k: v for k, v in overrides.items() if k in wb}}
        kw = {**kw, **{k: v for k, v in overrides.items() if k in kw}}
    yield ("params", {"wb": wb, "kw": kw, "model": ef5_model,
                      "source": ("override" if overrides else
                                 stored.get("source", "stored") if stored else "a-priori")})
    pgrids = params.clip_param_grids(bbox, param_dir)

    # --- snow detection (task #7): temp-driven, with user override ---
    snow_ov = ({k: overrides[k] for k in _snow.SNOW_DEFAULTS if overrides and k in overrides}
               if overrides else None)
    si = _snow.detect_snow(bbox, run_start, run_end,
                           dem_path=os.path.join(basic_dir, "dem_clip.tif"),
                           force=(None if snow == "auto" else snow), use_temp=not use_mock)
    snow_on = si["snow"]
    yield ("status", ("❄️ SNOW17 enabled — " if snow_on else "☀️ no snow — ") + si["reason"])
    snow_scalars = _snow.snow_params(snow_ov) if snow_on else None
    snow_grids = _snow.clip_snow_grids(bbox, os.path.join(work, "snow")) if snow_on else None
    # shared per-basin forcing store: overlapping runs merge their timesteps
    # instead of re-downloading into per-run temp dirs (data manager)
    mrms_dir = forcing.store_dir("mrms", bbox)
    pet_dir = forcing.store_dir("pet", bbox)
    temp_dir = forcing.store_dir("temp", bbox)

    sdir = statecache.state_dir(g["id"], ef5_model)
    warmup_start = None
    if pl["need_warmup"] and (pl.get("warmup_from") or warmup_days > 0):
        # nearby state -> short gap; else the full warm-up (knob, default 90 d; 0 = cold start)
        warmup_start = pl.get("warmup_from") or (run_start - timedelta(days=warmup_days))
    if pl.get("warmup_from"):                         # short warm-up bridging the +/-10 d gap
        gap = (run_start - pl["warmup_from"]).total_seconds() / 86400.0
        yield ("status", f"short warm-up ({gap:.1f} d) from nearby state @ "
                         f"{pl['warmup_from']:%Y-%m-%d} up to the start")
    elif warmup_start:
        yield ("status", f"{warmup_days}-day warm-up from {warmup_start:%Y-%m-%d} (builds initial state)")
    elif pl["need_warmup"]:
        yield ("status", "⚠️ warm-up disabled (0 d) — cold start, expect biased early flows")
    elif pl["load_state_time"]:
        yield ("status", f"warm start from exact saved state @ {pl['load_state_time']:%Y-%m-%d %H:%M}")

    usgs_dir = ""
    if not use_mock:                                  # real run: USGS observed discharge -> OBS=
        usgs_dir = os.path.join(work, "USGS")
        try:
            series = obs.fetch_usgs_discharge(g["id"], run_start, run_end)
            if obs.write_ef5_obs(g["id"], series, usgs_dir):
                yield ("status", f"USGS observed discharge: {len(series)} points")
        except Exception:
            usgs_dir = ""

    spec = ControlSpec(
        control_path=os.path.join(work, "control.txt"),
        time_begin=run_start, time_end=run_end, timestep=timestep,
        basic_dir=basic_dir, precip_dir=mrms_dir,
        pet_dir=pet_dir, output_dir=out_dir, usgs_dir=usgs_dir,
        gauges=[Gauge(g["id"], g["lon"], g["lat"], g["area"])],
        crest=wb, kw=kw, model=ef5_model.upper(),
        param_grids=pgrids, output_grids=grids,
        state_dir=sdir, warmup_start=warmup_start,
        snow_on=snow_on, snow_scalars=snow_scalars, snow_grids=snow_grids, temp_dir=temp_dir)
    build_control(spec)

    if not use_mock:                                  # real run: forcing over the full span
        f0 = warmup_start or run_start
        n_days = max(1, int((run_end - f0).total_seconds() // 86400))
        yield ("status", f"⬇️ downloading rainfall (MRMS) forcing — "
                         f"{n_days} day(s) incl. warm-up…")
        fr = forcing.prepare_forcing("mrms", bbox, f0, run_end, mrms_dir)
        if fr.reused:
            yield ("status", f"♻️ forcing store: reused {fr.reused} MRMS timestep(s), "
                             f"fetched {len(fr.written)} new")
        yield ("status", "⬇️ downloading PET forcing…")
        forcing.prepare_forcing("pet", bbox, f0, run_end, pet_dir)
        if snow_on:
            yield ("status", "⬇️ downloading temperature forcing (snow module)…")
            forcing.prepare_forcing("temp", bbox, f0, run_end, temp_dir)

    # warm-up: separate blocking ef5 process (own control); its state files at
    # run_start feed the Simu run below (two tasks in one process segfault)
    if not use_mock and warmup_start:
        yield ("status", "running warm-up (builds the initial state)…")
        wu_out = os.path.join(out_dir, "warmup")
        wu_ctl = os.path.join(os.path.dirname(spec.control_path), "control_warmup.txt")
        wu = run_ef5(wu_ctl, wu_out, g["id"], model=ef5_model)
        from hf_data.runner import RUN_TIMEOUT_S
        wu_deadline = time.time() + RUN_TIMEOUT_S
        while wu.alive():
            if time.time() > wu_deadline:            # stuck-run watchdog
                wu.kill()
                yield ("status", f"⚠️ warm-up killed after {RUN_TIMEOUT_S / 3600:.1f} h "
                                 "(stuck-run watchdog) — Simu will cold-start")
                break
            time.sleep(2)
        n_states = len(glob.glob(os.path.join(sdir, f"*_{run_start:%Y%m%d_%H%M}.tif")))
        if n_states:
            yield ("status", f"warm-up done — {n_states} state grid(s) saved "
                             f"@ {run_start:%Y-%m-%d %H:%M}")
        else:
            yield ("status", "⚠️ warm-up produced no state files — Simu will cold-start")

    yield ("status", "running CREST — streaming…")
    if use_mock:
        handle = MockEF5(out_dir, gauge_id=g["id"], model=ef5_model, bounds=bbox,
                         n_steps=run_hours + 1, t0=run_start, delay=0.15,   # inclusive of run_end
                         write_grids=grids,
                         facc_path=os.path.join(basic_dir, "facc_clip.tif")).start()
    else:
        handle = run_ef5(spec.control_path, out_dir, g["id"], model=ef5_model)

    new_rows = []
    for ev in stream_run(handle, poll=0.2):
        if ev["kind"] == "hydro":
            new_rows += ev["rows"]
        ev["bbox"] = bbox
        if (ev["kind"] == "done" and not use_mock
                and (ev.get("returncode") not in (0, None) or not new_rows)):
            # surface the EF5 log so failures (or silent empty runs) are
            # debuggable from the UI/SSE
            try:
                with open(os.path.join(out_dir, "ef5_run.log"),
                          encoding="utf-8", errors="replace") as fh:
                    tail = fh.read()[-2500:]
                yield ("status", f"⚠️ EF5 rc={ev['returncode']}, "
                                 f"{len(new_rows)} new rows — log tail:\n{tail}")
            except Exception:
                pass
        yield (ev["kind"], ev)

    # persist result + state-save times for future overlap reuse (hourly runs only)
    if hourly:
        st = [run_end.strftime(statecache.TS_FMT)]
        if warmup_start:
            st.append(run_start.strftime(statecache.TS_FMT))
        try:
            statecache.save_record(g["id"], ef5_model, pl["cached_rows"] + new_rows, st)
        except Exception:
            pass


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "flash flood near Allagash"
    nrows = nq = 0
    for kind, payload in analyze(q, use_mock=True, hours=10):
        if kind == "status":
            print("STATUS:", payload)
        elif kind == "hydro":
            nrows += len(payload["rows"])
        elif kind == "q2d":
            nq += 1
        elif kind == "done":
            print(f"DONE rc={payload.get('returncode')} | {nrows} hydro rows, {nq} q-grids")

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
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from hf_data import (basic, forcing, gauges, lakes, multipliers, obs, params, paramstore,
                     statecache)
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
        except Exception as e:                    # gazetteer fallback still runs,
            from hf_data import crashlog          # but an LLM outage is worth a trace
            crashlog.capture("acp:llm-parse", e, query=q[:200])
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
    from hf_data import basinextent
    try:
        ext = basinextent.ef5_extent(g)
    except basinextent.BasinExtentError as e:
        yield ("status", f"⚠️ {e} — not simulated; CREST-AI never falls back to a box")
        yield ("done", {"returncode": -2, "no_basin": True, "reason": e.reason})
        return
    bbox = ext["bounds"]
    sn = ext["snap"]

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
        gauges=[Gauge(g["id"], sn["lon"], sn["lat"], None)],   # pinned cell, no BASINAREA
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

# speed run: a cut gauge's observations REPLACE its upstream basin, so gaps in
# the record become missing river — require nearly complete coverage
BC_MIN_COVER = float(os.environ.get("CREST_BC_MIN_COVER", "0.95"))


def _obs_coverage(series, a, b) -> float:
    """Fraction of hours in [a, b] with at least one observation."""
    total_h = max(1, int((b - a).total_seconds() // 3600))
    hours = {dt.replace(minute=0, second=0, microsecond=0) for dt, _ in series}
    return min(1.0, len(hours) / total_h)


def plan_cut_set(gauge_id: str, t_start: datetime, t_end: datetime,
                 warmup_days: int = WARMUP_DAYS) -> list[str]:
    """Speed-scheme cut set for a whole multi-chunk run (fleet, M3): the
    upstream gauges whose observations cover >= BC_MIN_COVER of the FULL
    period. Passed to every chunk as run_gauge(cut_ids=...), so the domain,
    the record variant and the state keys never flip between chunks (a flip
    re-keyed the record and wiped its earlier rows). [] = full basin.
    Raises basinextent.BasinExtentError for an outlet without a basin."""
    from hf_data import basinextent, neighbors
    g = gauge_info(gauge_id)
    if g is None:
        return []
    if g.get("virtual"):
        warmup_days = min(warmup_days, 10)        # as run_gauge does for them
    ext = basinextent.ef5_extent(g)
    bbox, sn = ext["bounds"], ext["snap"]
    basic_dir = basic.store_dir(bbox)
    basic.clip_basic_data(bbox, basic_dir)
    gs = {**g, "lat": sn["lat"], "lon": sn["lon"], "area": sn["area"]}
    f0 = t_start - timedelta(days=warmup_days)
    keep = []
    bc = neighbors.upstream_gauges(gs, bbox, basic_dir)
    for b_ in bc:
        try:
            s = obs.get_series(b_["id"], f0, t_end)
        except Exception:
            s = []
        if s and _obs_coverage(s, f0, t_end) >= BC_MIN_COVER:
            keep.append(b_["id"])
    if keep:              # a set the speed domain cannot be built from -> full basin
        from hf_data import domain as _domain
        if not _domain.build_speed_domain(gs, [b_ for b_ in bc if b_["id"] in keep],
                                          basic_dir):
            return []
    return sorted(keep)

# one EF5 run per (gauge, model) at a time: concurrent runs of the SAME gauge
# would race on the shared state dir + result-cache JSON. A second request
# (double-click, or another user picking the same gauge) waits, then usually
# gets served straight from the cache the first run just wrote.
_RUN_LOCKS: dict[tuple, threading.Lock] = {}
_RUN_LOCKS_GUARD = threading.Lock()


def _pv(v) -> str:
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


def apriori_params(g: dict, model: str, wb_model: str) -> dict | None:
    """The a-priori {param: value} set a run of this gauge starts from (what
    the fleet precomputed with), or None when the table has no row for it."""
    try:
        got = multipliers.to_control_params(g["id"], model=wb_model)
    except Exception:
        got = None
    if not got:
        return None
    wb, kw = got
    if model == "hp":
        wb = {"precip": 1.0, "split": 0.5}
    return {**wb, **kw}


def config_tag(stored: dict | None, overrides: dict | None, snow: str,
               base: dict | None = None, lake: dict | None = None) -> str:
    """Fingerprint of everything besides (gauge, model, scheme) that changes
    the rows a run produces: stored parameter VALUES that differ from the
    a-priori set (`base`), the advanced-panel overrides, a non-default snow
    choice, and the run's lake set (`lake` = lakes.cache_sig: table version +
    multipliers; PI 2026-10-30 - lake-on rows never merge with lake-off rows;
    None for a basin without lakes, which keeps the plain key). Empty for a
    run on the a-priori set — that is the key the fleet precomputed under.
    Rows and frames cache under model+tag; EF5 state grids stay under the
    base key.
    Value-based on purpose (2026-09-29): every completed run saves a stored
    record, usually one that just restates the a-priori values — keyed by the
    record's timestamp, the first run of a gauge moved every later run off
    the fleet's key, so its precomputed rows were never served again. Values
    also make the key identical on every Space for the same parameters.
    (UX isolation, user 2026-09-28: a custom run must never be served another
    user's rows, and a calibration winner is a new key, not a cache purge.)"""
    import hashlib
    import json as _json
    parts = {}
    if stored:
        vals = {**(stored.get("wb") or {}), **(stored.get("kw") or {})}
        diff = {k: _pv(v) for k, v in sorted(vals.items())
                if (base is None or k in base)
                and (base is None or _pv(base[k]) != _pv(v))}
        if diff:
            parts["p"] = diff
    if overrides:
        parts["o"] = {k: overrides[k] for k in sorted(overrides)}
    if snow and snow != "auto":
        parts["s"] = snow
    if lake:
        parts["l"] = lake
    if not parts:
        return ""
    return hashlib.sha1(_json.dumps(parts, sort_keys=True, default=str)
                        .encode()).hexdigest()[:8]


def _run_lock(gauge_id: str, model: str) -> threading.Lock:
    key = (str(gauge_id).zfill(8), model)
    with _RUN_LOCKS_GUARD:
        if key not in _RUN_LOCKS:
            _RUN_LOCKS[key] = threading.Lock()
        return _RUN_LOCKS[key]


def gauge_info(gauge_id: str):
    from hf_data import virtualpoints
    if virtualpoints.is_virtual(gauge_id):        # ungauged pour-point outlet
        return virtualpoints.info(str(gauge_id))
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
              workdir: str | None = None, cancel=None, scheme: str = "full",
              nowcast_t0: datetime | None = None,
              cut_ids: list[str] | None = None):
    """Per-gauge streaming run (map flow: gauge already chosen). Yields events.
    `cancel` (threading.Event) stops the run: the EF5 process is killed and the
    (gauge, model) lock released, so a superseding run can start immediately.
    The EF5 raster extent and outlet cell come from basinextent (the basin's
    outline, never a box); an outlet without one is not simulated (done event
    with returncode -2, no_basin=True).
    cut_ids: a speed-scheme cut set decided once for a multi-chunk run
    (plan_cut_set) — replaces the per-window obs-coverage gate, so every chunk
    of a fleet gauge simulates the same domain under the same keys."""
    g = gauge_info(gauge_id)
    if g is None:
        yield ("status", f"⚠️ gauge {gauge_id} not found")
        yield ("done", {"returncode": -1})
        return
    from hf_data import basinextent
    try:
        ext = basinextent.ef5_extent(g)
    except basinextent.BasinExtentError as e:
        yield ("meta", {**g, "model": model})
        yield ("status", f"⚠️ {e} — not simulated; CREST-AI never falls back to a box")
        yield ("done", {"returncode": -2, "no_basin": True, "reason": e.reason})
        return
    if model == "auto":                          # western-CONUS switch (task #8)
        model = "crest" if g["lon"] < WEST_LON else "crestphys"
    ef5_model = model                            # MODEL= + EF5 output-file naming
    wb_model = "crest" if model in ("crest", "hp") else "crestphys"  # multiplier source
    if g.get("virtual"):
        if scheme != "speed":
            # an ungauged point has no observations: injecting upstream USGS obs
            # and simulating only the incremental area is the whole design —
            # full-basin runs of a big ungauged outlet would also be uncacheable-slow
            scheme = "speed"
            yield ("status", "🛰 ungauged point — speed scheme forced: upstream USGS "
                             "gauges (where present) feed the run as boundary inflow")
        if warmup_days > 10:
            # upstream injection carries the main river, so only the small
            # incremental area needs spin-up — and with no observations to
            # compare against, a long warm-up buys nothing
            warmup_days = 10
            yield ("status", "🕐 ungauged point — warm-up shortened to 10 days "
                             "(injected upstream flow carries the river)")
    if nowcast_t0:
        # the future half of a nowcast window carries injected/zero-precip values
        # that must never be written into the shared row cache (a later hindcast
        # of the same window would be served the fabricated future); warm-start
        # from saved state is still allowed. Hydrograph only — no 2-D maps —
        # EXCEPT event runs, which pass an explicit output_grids token string
        # (streamflow|runoff|subrunoff) to feed the CREST-iMAP v2 solver.
        no_cache = True
        if not isinstance(grids, str):
            grids = False
    yield ("meta", {**g, "model": model})        # for the report + right panel

    # the run lock guards the SHARED row/state cache of one configuration:
    # keyed by the same tag the cache uses, so a custom-parameter run never
    # waits behind a default one, and skipped entirely by no_cache runs
    # (calibration candidates, nowcast / event runs) — they own their workdir
    # and write no cache, so a calibration no longer blocks other users
    tag = config_tag(paramstore.get(g["id"], ef5_model), overrides, snow,
                     base=apriori_params(g, model, wb_model))
    lock = None if (no_cache or nowcast_t0) else \
        _run_lock(g["id"], ef5_model + (f"-{tag}" if tag else ""))
    if lock is not None and not lock.acquire(blocking=False):
        yield ("status", "⏳ this exact run is already in progress for someone "
                         "else — waiting for it (the results are shared)")
        while not lock.acquire(timeout=2):       # keep the queue cancellable
            if cancel is not None and cancel.is_set():
                yield ("status", "⏹ stopped while queued")
                yield ("done", {"returncode": -9, "cancelled": True})
                return
    work = workdir or tempfile.mkdtemp(prefix=f"crest_{g['id']}_")
    try:
        yield from _run_gauge_body(g, model, ef5_model, wb_model, t_start, t_end,
                                   use_mock, overrides, snow, timestep,
                                   warmup_days, grids, no_cache, work, cancel,
                                   scheme, nowcast_t0, ext, cut_ids)
    finally:
        if lock is not None:
            lock.release()
        # the run-private state folder (M4) is scratch: its staged hard links
        # would otherwise pin states the shared folder has pruned or uploaded
        shutil.rmtree(os.path.join(work, "states"), ignore_errors=True)


def _run_gauge_body(g, model, ef5_model, wb_model, t_start, t_end, use_mock,
                    overrides, snow, timestep, warmup_days, grids, no_cache, workdir,
                    cancel=None, scheme="full", nowcast_t0=None, ext=None, cut_ids=None):
    """The actual per-gauge run — called with the (gauge, model) lock held.

    nowcast_t0 (UTC): route a NOWCAST at an ungauged point. The window runs to
    t0 + horizon; each upstream cut gauge is injected with observed flow up to
    t0 and its DI-LSTM prediction beyond t0, and forcing stops at t0 (EF5 reads
    a missing precip grid as zero, so the future is driven by routed upstream
    inflow, not new rain)."""
    from hf_data import basinextent, runstates
    work = workdir or tempfile.mkdtemp(prefix=f"crest_{g['id']}_")
    out_dir = os.path.join(work, "CREST_output")
    # the basin's own extent (outline ∪ carved catchment + ~5 km, on the
    # HydroSHEDS lattice) and its outlet cell, fixed once per outlet (M1): EF5
    # gets that cell and its FAM area, so its 20-km outlet search stays put
    bbox = ext["bounds"]
    sn = ext["snap"]
    gs = {**g, "lat": sn["lat"], "lon": sn["lon"], "area": sn["area"]}
    # shared per-basin clip stores: repeat runs + calibration candidates reuse
    # the clipped terrain/param grids instead of re-reading the remote COGs
    basic_dir = basic.store_dir(bbox)
    param_dir = os.path.join(basic_dir, "param")

    def _stopped():                 # user hit Stop / superseded — checked at every
        return cancel is not None and cancel.is_set()   # long stage boundary

    if _stopped():                                       # stopped while queued
        yield ("status", "⏹ stopped")
        yield ("done", {"returncode": -9, "cancelled": True})
        return

    # --- terrain first: the upstream-gauge scan below needs the clipped grids ---
    outline = {"gages2": "GAGES-II polygon", "hybas": "HydroBASINS sub-basins",
               "d8": "D8 catchment"}.get(ext["source"], ext["source"])
    yield ("status", f"clip DEM/DDM/FAM · model {model.upper()} · basin outline: {outline} "
                     f"({ext['carved_km2']:,.0f} km² catchment, {ext['cells']:,} raster cells)")
    clip = basic.clip_basic_data(bbox, basic_dir)
    if clip.derived:
        yield ("status", "⚠️ HydroSHEDS dir/acc unusable here — flow direction + "
                         "accumulation re-derived from the DEM (pysheds)")
    try:                                   # sanity: clip values at the outlet cell
        import rasterio as _rio
        vals = {}
        for tag in ("dem", "fdir", "facc"):
            with _rio.open(os.path.join(basic_dir, f"{tag}_clip.tif")) as ds:
                r_, c_ = ds.index(sn["lon"], sn["lat"])
                vals[tag] = float(ds.read(1)[r_, c_])
        yield ("status", f"clip @outlet (r{r_},c{c_}): dem={vals['dem']:.0f} "
                         f"fdir={vals['fdir']:.0f} facc={vals['facc']:.0f}")
    except Exception as e:
        yield ("status", f"(clip sample failed: {e})")
    # the runtime proof that the rectangle clips no basin: the catchment EF5
    # will carve from the outlet cell must not reach the raster border
    try:
        cv = basinextent.carve(basic_dir, ext, g["id"])
    except basinextent.BasinExtentError as e:
        yield ("status", f"⚠️ {e} — not simulated; CREST-AI never falls back to a box")
        yield ("done", {"returncode": -2, "no_basin": True, "reason": e.reason})
        return

    if _stopped():
        yield ("status", "⏹ stopped")
        yield ("done", {"returncode": -9, "cancelled": True})
        return

    # --- upstream/interior gauges -> boundary conditions (data assimilation) ---
    bc_gauges = []
    try:
        from hf_data import neighbors
        bc_gauges = neighbors.upstream_gauges(gs, bbox, basic_dir)
    except Exception as e:
        yield ("status", f"(upstream-gauge scan failed: {e})")
    if bc_gauges:
        n_edge = sum(1 for b_ in bc_gauges if b_["at_edge"])
        names = ", ".join(b_["id"] for b_ in bc_gauges)
        yield ("status", f"🔗 {len(bc_gauges)} upstream gauge(s) inside the domain "
                         f"({names}){f' — {n_edge} at the DEM edge' if n_edge else ''}; "
                         "their USGS observations feed the run as boundary conditions "
                         "(EF5 data assimilation)")
    # --- scheme: ⚡ speed run truncates the domain at the boundary gauges ------
    # (🏞 full run simulates the whole basin; DA still injects obs at BC gauges)
    speed = None                    # domain info dict when a speed run is active
    bc_obs = {}                     # gid -> (series, obs_csv_path), reused below
    if scheme == "speed":
        if not bc_gauges:
            yield ("status", "⚡ speed run requested, but no upstream gauges exist "
                             "in this domain — running the full basin")
        else:
            fixed = set(cut_ids) if cut_ids is not None else None
            qualified = bc_gauges if fixed is None else \
                [b_ for b_ in bc_gauges if b_["id"] in fixed]
            if not use_mock:        # obs-coverage gate: a cut gauge's record IS the river
                qualified = []
                bc_dir = os.path.join(work, "USGS_bc")
                f0_est = t_start - timedelta(days=warmup_days)
                # nowcast: coverage is judged on the OBSERVED past only; the
                # future window is filled by each gauge's own DI-LSTM nowcast
                cov_end = nowcast_t0 or t_end
                for b_ in bc_gauges:
                    if fixed is not None and b_["id"] not in fixed:
                        continue              # outside the run-wide cut set
                    try:
                        s = obs.get_series(b_["id"], f0_est, cov_end)
                    except Exception as e:        # gauge silently loses its BC role
                        from hf_data import crashlog
                        crashlog.capture("obs:bc", e, gauge=b_["id"])
                        s = []
                    cov = _obs_coverage(s, f0_est, cov_end) if s else 0.0
                    nc_pts = 0
                    if nowcast_t0 and (fixed is not None or cov >= BC_MIN_COVER):
                        # inject this upstream gauge's prediction as future inflow
                        from hf_data import nowcaststore
                        ns = [(t, q) for t, q in nowcaststore.nowcast_series(b_["id"])
                              if t > nowcast_t0]
                        s = list(s) + ns
                        nc_pts = len(ns)
                    p = obs.write_ef5_obs(b_["id"], s, bc_dir) if s else None
                    if fixed is not None:
                        # run-wide cut set: the domain never changes between
                        # chunks (a flip re-keys the record and wiped rows)
                        qualified.append(b_)
                        if p:
                            bc_obs[b_["id"]] = (s, p)
                        else:
                            yield ("status", f"⚠️ cut gauge {b_['id']}: no observations in "
                                             "this window — its upstream inflow is missing here")
                    elif p and cov >= BC_MIN_COVER:
                        bc_obs[b_["id"]] = (s, p)
                        qualified.append(b_)
                        if nowcast_t0:
                            yield ("status", f"🔮 upstream gauge {b_['id']}: "
                                             f"{nc_pts} h of DI-LSTM nowcast appended "
                                             "as future inflow" if nc_pts else
                                             f"⚠️ upstream gauge {b_['id']}: no nowcast "
                                             "available — future inflow held at last obs")
                    else:
                        yield ("status", f"gauge {b_['id']}: obs coverage {cov:.0%} "
                                         f"< {BC_MIN_COVER:.0%} — its area stays simulated")
            try:
                from hf_data import domain as _domain
                speed = _domain.build_speed_domain(gs, qualified, basic_dir) if qualified else None
            except Exception as e:
                from hf_data import crashlog
                crashlog.capture("speed-domain", e, gauge=g["id"])
                yield ("status", f"⚠️ speed-run domain build failed ({e}) — "
                                 "running the full basin")
                speed = None
            if speed:
                cut_str = ", ".join(c["id"] for c in speed["cut"])
                yield ("status", f"⚡ speed run: domain cut at {len(speed['cut'])} "
                                 f"boundary gauge(s) [{cut_str}] — simulating "
                                 f"{speed['kept_frac']:.0%} of the basin "
                                 f"(≈{speed['speedup']:.1f}× faster); observed flow "
                                 "is injected at the cut gauges")
            elif bc_gauges:
                yield ("status", "⚡ speed run not possible here (no qualifying cut "
                                 "gauges) — running the full basin")
    if cut_ids and scheme == "speed" and (
            speed is None or {b_["id"] for b_ in bc_gauges} & set(cut_ids) != set(cut_ids)):
        # a run-wide cut set (fleet chunks, nowcast Phase B) must be rebuilt
        # exactly: falling back to the full basin would re-key the record
        yield ("status", f"⚠️ the run's fixed cut set [{', '.join(cut_ids)}] could not "
                         "be rebuilt in this window (upstream scan or domain build "
                         "failed) — not simulated rather than switching domain")
        yield ("done", {"returncode": -4, "cut_failed": True})
        return
    # the cells EF5 simulates on its raster: the carved catchment, or the kept
    # part of a speed domain. Snow is decided over them (M2) and a saved state
    # is used only if it holds values on them (M4)
    grid_dir = speed["dir"] if speed else basic_dir   # dem/fdir/facc for EF5
    nodes, nodes_tr = runstates.nodes_of(grid_dir, None if speed else cv["mask"])
    cut_now = sorted(c["id"] for c in speed["cut"]) if speed else []
    # separate cache/state keys per scheme: rows, frames and EF5 state grids
    # from a truncated domain must never mix with full-basin ones
    # rows + frames key: scheme + configuration tag (stored params, overrides,
    # snow); EF5 states key: scheme only (shared warm starts; never written
    # by a run with panel overrides or a forced snow choice, so a custom run
    # cannot poison the shared state grids)
    stored = paramstore.get(g["id"], ef5_model)       # best-known set for this basin
    # lake module (PI 2026-10-27, west first): the lakes, reservoirs and ponds
    # whose outlet cell is one of this run's EF5 nodes (hf_data/lakes.py) -
    # known before the cache key, which separates lake runs (PI 2026-10-30)
    lake_set, lake_mult = None, (1.0, 1.0)
    if not use_mock and lakes.enabled(g["lon"]):
        lake_mult = lakes.multipliers(overrides, (stored or {}).get("lake"))
        lake_set = lakes.for_run(nodes, nodes_tr, lake_mult)
    tag = config_tag(stored, overrides, snow,
                     base=apriori_params(g, model, wb_model),
                     lake=lakes.cache_sig(lake_set, lake_mult))
    custom = bool(overrides) or (snow not in (None, "", "auto"))
    state_model = ef5_model + ("-spd" if speed else "")
    cache_model = state_model + (f"-{tag}" if tag else "")
    variant = (("cut:" + ",".join(sorted(c["id"] for c in speed["cut"]))) if speed
               else "bc:" + (",".join(sorted(b_["id"] for b_ in bc_gauges)) or "none"))

    if cancel is not None and cancel.is_set():
        yield ("status", "⏹ stopped")
        yield ("done", {"returncode": -9, "cancelled": True})
        return

    # every real hourly run fetches the fleet's data. no_cache runs (nowcast /
    # event runs, calibration candidates) never read or write the row cache,
    # but warm-starting from fleet STATES is still right: without it an event
    # run cold-starts (dry soils crush flood peaks) and every calibration
    # candidate pays its own 90-day warm-up
    if timestep == "1h" and not use_mock and not g.get("virtual"):
        try:      # fleet pre-runs (read-only repo): rows serve instantly below,
            from hf_data import fleetstore     # states give 10-day warm starts
            got = fleetstore.ensure_local(g["id"], state_model,
                                          rows_model=cache_model, variant=variant)
            if got == "fetched":
                yield ("status", "🚚 pre-simulated by the background fleet — "
                                 "loading saved results and model states")
        except Exception as e:
            from hf_data import crashlog
            crashlog.capture("fleetstore", e, gauge=g["id"], model=state_model)

    # full-basin run of a gauge with upstream gauges: the fleet pre-ran it
    # (and each upstream gauge) on truncated ⚡ domains, so no single bundle
    # holds the whole basin — but together they tile it. With nothing saved
    # near the start, stitch the full-basin state from the CONUS mosaic
    # (user 2026-09-29); plan() below then finds it like any saved state.
    if timestep == "1h" and not use_mock and not g.get("virtual") \
            and not speed and bc_gauges:
        try:
            pre = None if no_cache else statecache.plan(
                g["id"], cache_model, t_start, t_end, variant=variant,
                state_model=state_model)
            rs = (pre or {}).get("run_start") or t_start
            runs = no_cache or pre["run_start"] is not None or bool(grids)
            ex0, wf0, _ = statecache._state_choice(g["id"], state_model, rs)
            if runs and ex0 is None and wf0 is None:
                from hf_data import conusstate
                ms = conusstate.warm_state(
                    gs, ef5_model, [b_["id"] for b_ in bc_gauges], basic_dir,
                    statecache.state_dir(g["id"], state_model), rs, warmup_days)
                if ms:
                    yield ("status",
                           f"🧩 full-basin model state stitched from "
                           f"{ms['n_sources']} fleet pre-run(s) (this gauge + its "
                           f"upstream gauges) @ {ms['time']:%Y-%m-%d} — covers "
                           f"{ms['coverage']:.0%} of the basin"
                           + ("" if ms["tier"] == "full" else
                              "; the uncovered part spins up over the warm-up"))
        except Exception as e:
            from hf_data import crashlog
            crashlog.capture("conusstate", e, gauge=g["id"], model=state_model)

    # --- result cache: reuse overlap, simulate only the missing window (task #6) ---
    # the row cache is hourly; a sub-hourly run neither reuses nor writes it
    hourly = timestep == "1h" and not no_cache
    pl = statecache.plan(g["id"], cache_model, t_start, t_end, variant=variant,
                         state_model=state_model, far_days=warmup_days) \
        if timestep == "1h" else {
        "cached_rows": [], "run_start": t_start, "run_end": t_end,
        "load_state_time": None, "warmup_from": None, "need_warmup": True,
        "reason": "sub-hourly timestep — cache bypassed"}
    if no_cache:            # calibration: fresh full-window run; candidates
        # still warm-start from any state saved on disk at/near t_start
        ex, wfrom, needw = statecache._state_choice(g["id"], state_model, t_start,
                                                    far_days=warmup_days)
        pl = {"cached_rows": [], "run_start": t_start, "run_end": t_end,
              "load_state_time": ex, "warmup_from": wfrom, "need_warmup": needw,
              "reason": "calibration run — row cache bypassed"}
    rows_pre_served = False
    if pl["run_start"] is None and grids:
        # rows are cached, but the 2-D streamflow maps need rendered frames —
        # if none are cached on disk, re-run the window (fast: warm-starts
        # from the saved state) so the map animation always appears. The
        # cached rows are KEPT and served immediately (the fleet pre-runs
        # exist so a gauge click shows its hydrograph instantly); the re-run
        # then only renders frames, its duplicate row stream suppressed.
        from hf_data import viz as _viz
        if not _viz.has_frames_cache(g["id"], cache_model, t_start, t_end):
            lt, wf, nw = statecache._state_choice(g["id"], state_model, t_start,
                                                  far_days=warmup_days)
            pl = {"cached_rows": pl["cached_rows"], "run_start": t_start,
                  "run_end": t_end,
                  "load_state_time": lt, "warmup_from": wf, "need_warmup": nw,
                  "reason": "re-run to render the 2-D streamflow maps"}
            rows_pre_served = bool(pl["cached_rows"])
            yield ("status", "hydrograph served from the cached rows — the 2-D "
                             "streamflow maps aren't rendered yet, re-running "
                             "the window in the background to draw them")
    if pl["cached_rows"]:
        yield ("status", f"♻️ reused {len(pl['cached_rows'])} cached step(s) "
                         f"({pl['cached_rows'][0]['time']}…{pl['cached_rows'][-1]['time']})")
        yield ("hydro", {"rows": pl["cached_rows"], "cached": True})
    if pl["run_start"] is None:                       # fully cached -> no simulation
        # no wb/kw here on purpose: the frames-cache key must be scheme-aware,
        # but an empty param set must never reach the param store
        yield ("params", {"model": ef5_model, "cache_model": cache_model,
                          "state_model": state_model,
                          "n_upstream": len(bc_gauges), "cut_ids": cut_now})
        yield ("status", "✓ served entirely from cache")
        yield ("done", {"returncode": 0, "cached": True,
                        "window": [t_start.strftime(statecache.TS_FMT),
                                   t_end.strftime(statecache.TS_FMT)]})
        return
    run_start, run_end = pl["run_start"], pl["run_end"]
    run_hours = max(1, int(round((run_end - run_start).total_seconds() / 3600)))
    yield ("status", f"🕐 simulation window {run_start:%Y-%m-%d %H:%M} → "
                     f"{run_end:%Y-%m-%d %H:%M} ({run_hours} h @ {timestep})")

    wbkw = multipliers.to_control_params(g["id"], model=wb_model)
    if wbkw is None and g.get("virtual"):
        # ungauged point: borrow the nearest calibrated gauge's multiplier set
        # (spatial-proximity regionalization — calibration here is impossible)
        near = multipliers.nearest_station(g["lat"], g["lon"])
        if near:
            wbkw = multipliers.to_control_params(near[0], model=wb_model)
            if wbkw:
                yield ("status", f"🧭 ungauged point — borrowing calibrated "
                                 f"parameters from the nearest gauge "
                                 f"{near[0]} ({near[1]:.0f} km away)")
    if wbkw is None:
        yield ("status", "⚠️ no calibrated params for this gauge")
        yield ("done", {"returncode": -1})
        return
    wb, kw = wbkw
    if model == "hp":
        # HP water balance has its own 2 params (fractions in [0,1]);
        # calibrated KW routing params are still used
        wb = {"precip": 1.0, "split": 0.5}
    if stored:                                        # (read above, with the tag)
        wb = {**wb, **{k: v for k, v in stored.get("wb", {}).items() if k in wb}}
        kw = {**kw, **{k: v for k, v in stored.get("kw", {}).items() if k in kw}}
        yield ("status", f"🎯 using stored best parameters ({stored.get('source','?')}, "
                         f"NSE {stored.get('nse')}, {stored.get('when','')})")
    if overrides:                                     # advanced-panel overrides
        wb = {**wb, **{k: v for k, v in overrides.items() if k in wb}}
        kw = {**kw, **{k: v for k, v in overrides.items() if k in kw}}
    if not use_mock and lakes.enabled(g["lon"]):     # (lake set: read above, with the tag)
        if lake_set is None:
            yield ("status", "⚠️ lake table unavailable — simulating without lakes/reservoirs")
        elif len(lake_set):
            yield ("status", f"🏞 lake module: {len(lake_set)} lake(s)/reservoir(s)/pond(s) "
                             f"in the basin, {lake_set.th_volume_mm3.sum():,.1f} Mm³ of storage")
    yield ("params", {"wb": wb, "kw": kw, "model": ef5_model,
                      "cache_model": cache_model,      # frames-cache key (scheme-aware)
                      "state_model": state_model,      # EF5 state-grid key
                      "n_upstream": len(bc_gauges),    # 0 => headwater (no injection possible)
                      "cut_ids": cut_now,              # this run's speed cut set ([] = full)
                      "lakes": lakes.provenance(lake_set, lake_mult),
                      "source": ("override" if overrides else
                                 stored.get("source", "stored") if stored else "a-priori")})
    pgrids = params.clip_param_grids(bbox, param_dir)

    # --- snow detection (task #7): temp-driven, with user override ---
    snow_ov = ({k: overrides[k] for k in _snow.SNOW_DEFAULTS if overrides and k in overrides}
               if overrides else None)
    si = _snow.detect_snow(bbox, run_start, run_end,
                           dem_path=os.path.join(grid_dir, "dem_clip.tif"),
                           force=(None if snow == "auto" else snow), use_temp=not use_mock,
                           basin=(nodes, nodes_tr))
    snow_on = si["snow"]
    yield ("status", ("❄️ SNOW17 enabled — " if snow_on else "☀️ no snow — ") + si["reason"])
    snow_grids = _snow.clip_snow_grids(bbox, os.path.join(work, "snow")) if snow_on else None
    snow_scalars = (_snow.snow_params(snow_ov, gridded=snow_grids or {})
                    if snow_on else None)
    if snow_on:
        yield ("status", f"❄ SNOW17 v1 parameter grids: {len(snow_grids or {})}/9 clipped "
                         "(operational CONUS 0.1° calibrated set, incl. pxtemp)")
    # DEM-based temperature extrapolation: the temp forcing is coarse (0.125°
    # NLDAS / 32 km NARR), so give EF5 a DEM aggregated to the temp grid and it
    # lapses each model cell by -6.5 °C/km relative to its temp pixel's mean
    # elevation ([TEMPForcing] DEM=, TempReader.cpp)
    temp_dem = (_snow.build_temp_dem(bbox, os.path.join(basic_dir, "dem_clip.tif"),
                                     os.path.join(work, "snow", "temp_dem.tif"))
                if snow_on else None)
    if snow_on:
        yield ("status", "⛰ DEM-based temperature extrapolation ON (−6.5 °C/km vs "
                         "temp-pixel mean elevation)" if temp_dem else
                         "⚠️ temperature extrapolation unavailable (temp-grid DEM "
                         "could not be built) — using raw coarse temperatures")
    # shared per-basin forcing store: overlapping runs merge their timesteps
    # instead of re-downloading into per-run temp dirs (data manager)
    mrms_dir = forcing.store_dir("mrms", bbox)
    pet_dir = forcing.store_dir("pet", bbox)
    temp_dir = forcing.store_dir("temp", bbox)

    sdir = statecache.state_dir(g["id"], state_model)
    # M4: EF5 reads states only from a run-private folder, staged with the
    # grids of the time it will load - and only if they hold values on this
    # run's EF5 nodes. EF5 itself would load any grid by location, silently
    # leaving uncovered cells (e.g. headwaters an old square clipped) cold.
    rdir = os.path.join(work, "states")          # removed by run_gauge when it ends
    os.makedirs(rdir, exist_ok=True)
    load_t = pl.get("warmup_from") or (pl.get("load_state_time")
                                       if not pl["need_warmup"] else None)
    if load_t is not None and not use_mock:
        # only a stitched fleet mosaic followed by a long warm-up may cover
        # part of the basin (its partial tier: the warm-up spins the rest up);
        # any other saved state - box-era ones included - must cover nearly all
        gap_d = (run_start - load_t).total_seconds() / 86400.0
        need = (runstates.PARTIAL_COVER if pl.get("warmup_from")
                and gap_d >= runstates.PARTIAL_GAP_DAYS
                and runstates.is_mosaic(sdir, load_t) else runstates.MIN_COVER)
        staged, cov = runstates.stage(sdir, rdir, load_t, nodes, nodes_tr, need)
        if not staged:
            if nowcast_t0 and warmup_days <= 0:
                # a nowcast forecast (routednow Phase B) runs only from the
                # state its own hindcast just saved: never a silent cold start,
                # never a stand-in warm-up without its upstream inflow
                yield ("status", f"⚠️ saved model state @ {load_t:%Y-%m-%d %H:%M} covers "
                                 f"{cov:.0%} of this basin's EF5 cells — no forecast this hour")
                yield ("done", {"returncode": -3, "no_state": True})
                return
            yield ("status", f"saved model state @ {load_t:%Y-%m-%d %H:%M} holds values "
                             f"on only {cov:.0%} of this basin's EF5 cells "
                             f"(< {need:.0%}) — not used; warming up instead")
            pl = {**pl, "warmup_from": None, "load_state_time": None, "need_warmup": True}
    warmup_start = None
    if pl["need_warmup"] and (pl.get("warmup_from") or warmup_days > 0):
        # nearby state -> short gap; else the full warm-up (knob, default 90 d; 0 = cold start)
        warmup_start = pl.get("warmup_from") or (run_start - timedelta(days=warmup_days))
    if pl.get("warmup_from"):                         # warm-up bridging the gap to a state
        gap = (run_start - pl["warmup_from"]).total_seconds() / 86400.0
        if gap <= statecache.STATE_TOL_DAYS:
            yield ("status", f"short warm-up ({gap:.1f} d) from nearby state @ "
                             f"{pl['warmup_from']:%Y-%m-%d} up to the start")
        else:
            yield ("status", f"warm-up ({gap:.0f} d) carried forward from the saved "
                             f"model state @ {pl['warmup_from']:%Y-%m-%d} — a "
                             "spun-up state instead of a cold start")
    elif warmup_start:
        yield ("status", f"{warmup_days}-day warm-up from {warmup_start:%Y-%m-%d} (builds initial state)")
    elif pl["need_warmup"]:
        yield ("status", "⚠️ warm-up disabled (0 d) — cold start, expect biased early flows")
    elif pl["load_state_time"]:
        yield ("status", f"warm start from exact saved state @ {pl['load_state_time']:%Y-%m-%d %H:%M}")

    usgs_dir = ""
    if not use_mock and g.get("virtual"):
        yield ("status", "🛰 no observations exist at an ungauged point — the "
                         "hydrograph shows simulation only (no NSE)")
    elif not use_mock:                                # real run: USGS observed discharge -> OBS=
        usgs_dir = os.path.join(work, "USGS")
        try:
            oinf: dict = {}
            series = obs.get_series(g["id"], run_start, run_end, info=oinf)
            if obs.write_ef5_obs(g["id"], series, usgs_dir):
                src = ("🗄 from saved storage" if oinf.get("cached")
                       else f"fetched {oinf.get('fetched_windows', 0)} missing window(s)")
                yield ("status", f"USGS observed discharge: {len(series)} points ({src})")
        except Exception:
            usgs_dir = ""

    # boundary-condition gauges + DA file. Speed run: the CUT gauges (snapped
    # cells of the truncated domain) — their obs ARE the upstream river. Full
    # run: every upstream gauge is a DA point layered on top of the full
    # simulation. Every gauge is PINNED (M1): its cell centre and NO
    # BASINAREA — EF5 uses BASINAREA only for its own 20-km outlet search
    # (BasicGrids.cpp:697), and its 110.574 km/deg cell area would never
    # match a Python-side FAM area, so it would move every pinned gauge
    if speed:
        o = speed["outlet"]
        ctl_gauges = [Gauge(g["id"], o["lon"], o["lat"], None)]
    else:       # outlet: WANTDA=false
        ctl_gauges = [Gauge(g["id"], sn["lon"], sn["lat"], None)]
    da_file = None
    if not use_mock:
        from hf_data import neighbors
        da_series = {}
        if speed:
            for c in speed["cut"]:                    # obs fetched by the gate above
                s, p = bc_obs.get(c["id"], (None, None))
                if p:
                    da_series[c["id"]] = s
                    ctl_gauges.append(Gauge(c["id"], c["lon"], c["lat"], None,
                                            obs_path=p, want_da=True, output_ts=False))
        elif bc_gauges:
            f0 = warmup_start or run_start
            bc_dir = os.path.join(work, "USGS_bc")
            nc_pts_total = 0
            for b_ in bc_gauges:
                try:
                    s = obs.get_series(b_["id"], f0, run_end)
                except Exception as e:            # gauge drops out of DA silently
                    from hf_data import crashlog
                    crashlog.capture("obs:bc", e, gauge=b_["id"])
                    s = []
                if s and nowcast_t0:
                    # event/nowcast run: extend DA with the gauge's own DI-LSTM
                    # prediction beyond t0 (mirrors the speed-scheme path), so
                    # the forecast horizon is driven by predicted upstream
                    # inflow instead of DA simply running out of observations
                    try:
                        from hf_data import nowcaststore
                        ns = [(t_, q_) for t_, q_ in
                              nowcaststore.nowcast_series(b_["id"])
                              if t_ > nowcast_t0]
                        s = list(s) + ns
                        nc_pts_total += len(ns)
                    except Exception:
                        pass
                p = obs.write_ef5_obs(b_["id"], s, bc_dir) if s else None
                if p:
                    da_series[b_["id"]] = s
                    # the DA point's own pinned stream cell (basin-extent table)
                    ctl_gauges.append(Gauge(b_["id"], b_["snap_lon"], b_["snap_lat"], None,
                                            obs_path=p, want_da=True, output_ts=False))
        if da_series:
            da_file = neighbors.write_da_file(da_series, os.path.join(work, "da_obs.csv"))
            yield ("status", f"🛰 assimilating observed flow at "
                             f"{len(da_series)} upstream gauge(s) — "
                             f"{sum(len(v) for v in da_series.values())} obs points")
            if not speed and nowcast_t0 and nc_pts_total:
                yield ("status", f"🔮 {nc_pts_total} h of DI-LSTM nowcast appended "
                                 f"as future inflow at the upstream DA gauges")
        elif bc_gauges:
            yield ("status", "(no usable USGS observations at the upstream gauges — "
                             "running without boundary conditions)")

    # per-gauge calibrated parameters (AQUAH crest_run_cali behavior): each
    # upstream gauge's sub-basin partition uses ITS own multipliers, not the
    # outlet's. HP keeps the outlet's water balance (its 2 params aren't in
    # the multiplier table); KW routing is per-gauge for every model.
    per_gauge = {}
    for cg in ctl_gauges[1:]:
        try:
            got = multipliers.to_control_params(cg.id, model=wb_model)
        except Exception:
            got = None
        if got:
            per_gauge[cg.id] = {"crest": got[0] if model != "hp" else None,
                                "kw": got[1]}
    if per_gauge:
        yield ("status", f"🧩 per-gauge calibrated parameters applied at "
                         f"{len(per_gauge)}/{len(ctl_gauges) - 1} upstream gauge(s)")

    spec = ControlSpec(
        control_path=os.path.join(work, "control.txt"),
        time_begin=run_start, time_end=run_end, timestep=timestep,
        basic_dir=grid_dir, precip_dir=mrms_dir,
        pet_dir=pet_dir, output_dir=out_dir, usgs_dir=usgs_dir,
        gauges=ctl_gauges,
        crest=wb, kw=kw, model=ef5_model.upper(),
        param_grids=pgrids, output_grids=grids,
        # calibration candidates (no_cache) warm-START from saved states but must
        # never SAVE: trial-parameter states would overwrite the legitimate
        # end-of-window state grids in the shared cache. EF5 reads/writes the
        # run-private folder; runstates.commit() moves non-custom states back
        state_dir=rdir, warmup_start=warmup_start, save_state_end=(not no_cache and not custom),   # never a custom run
        snow_on=snow_on, snow_scalars=snow_scalars, snow_grids=snow_grids, temp_dir=temp_dir,
        temp_dem=temp_dem, da_file=da_file, per_gauge=per_gauge or None,
        lakes_csv=(lakes.write_csv(os.path.join(work, "lakes.csv"), lake_set)
                   if lake_set is not None and len(lake_set) else None))
    build_control(spec)
    if spec.lakes_csv:
        # the storage EF5 starts each lake from: its saved storage when the
        # state it loads was saved with this lake set, else its normal pool
        # (EF5 alone would start every lake full)
        li = lakes.prepare_states(rdir, warmup_start or run_start, lake_set,
                                  os.path.join(grid_dir, "dem_clip.tif"), lake_mult)
        if li["initial"]:
            yield ("status", f"🏞 {li['initial']} lake(s) start at their normal pool"
                             + (f", {li['kept']} from the saved state" if li["kept"] else ""))

    if cancel is not None and cancel.is_set():
        yield ("status", "⏹ stopped")
        yield ("done", {"returncode": -9, "cancelled": True})
        return

    if not use_mock:                                  # real run: forcing over the full span
        f0 = warmup_start or run_start
        # nowcast: forcing exists only up to t0; the future window is left
        # ungridded so EF5 reads zero precip there (the routed upstream inflow,
        # not new rain, drives the forecast) — and prepare_forcing never has to
        # reach into a not-yet-archived future month
        force_end = min(run_end, nowcast_t0) if nowcast_t0 else run_end
        n_days = max(1, int((force_end - f0).total_seconds() // 86400))
        yield ("status", f"🌧 preparing rainfall (MRMS) forcing from the archive — "
                         f"{n_days} day(s) incl. warm-up…"
                         + (" (future window unforced → routed inflow only)"
                            if nowcast_t0 else ""))
        fr = forcing.prepare_forcing("mrms", bbox, f0, force_end, mrms_dir, cancel=cancel)
        if fr.reused:
            yield ("status", f"♻️ forcing store: reused {fr.reused} MRMS timestep(s), "
                             f"prepared {len(fr.written)} new")
        if fr.recent:
            yield ("status", f"🛰 {fr.recent} fresh hour(s) filled from the "
                             f"near-real-time MRMS store (Pass1) — the month "
                             f"archive lags days behind real time")
        if fr.missing:
            # never let a rainless simulation pass silently again: EF5 reads
            # a missing precip grid as ZERO rain (seen live: events simulated
            # bone-dry through a real flood because the archive lagged)
            yield ("status", f"⚠️ {len(fr.missing)} MRMS hour(s) unavailable in "
                             f"both the archive and the near-real-time store — "
                             f"EF5 runs those hours with ZERO rain "
                             f"(first: {fr.missing[0]})")
        if _stopped():
            yield ("status", "⏹ stopped")
            yield ("done", {"returncode": -9, "cancelled": True})
            return
        yield ("status", "🌡 preparing PET forcing…")
        forcing.prepare_forcing("pet", bbox, f0, force_end, pet_dir, cancel=cancel)
        if snow_on and not _stopped():
            yield ("status", "❄ preparing temperature forcing (snow module)…")
            tr = forcing.prepare_forcing("temp", bbox, f0, force_end, temp_dir,
                                         cancel=cancel)
            if tr.missing:
                # NARR lags weeks and its 2026 file has no August (PSL rewrite
                # 2026-10-07): say so instead of letting SNOW17 run blind
                yield ("status", f"⚠️ {len(tr.missing)} temperature hour(s) "
                                 f"unavailable (NARR source lag or gap) — the "
                                 f"snow module has no observed temperature for "
                                 f"them (first: {tr.missing[0]})")
        if _stopped():
            yield ("status", "⏹ stopped")
            yield ("done", {"returncode": -9, "cancelled": True})
            return

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
            if cancel is not None and cancel.is_set():
                wu.kill()
                yield ("status", "⏹ stopped during warm-up")
                yield ("done", {"returncode": -9, "cancelled": True})
                return
            if time.time() > wu_deadline:            # stuck-run watchdog
                wu.kill()
                yield ("status", f"⚠️ warm-up killed after {RUN_TIMEOUT_S / 3600:.1f} h "
                                 "(stuck-run watchdog) — Simu will cold-start")
                break
            time.sleep(2)
        n_states = len(glob.glob(os.path.join(rdir, f"*_{run_start:%Y%m%d_%H%M}.tif")))
        if n_states:
            yield ("status", f"warm-up done — {n_states} state grid(s) saved "
                             f"@ {run_start:%Y-%m-%d %H:%M}")
        else:
            yield ("status", "⚠️ warm-up produced no state files — Simu will cold-start")
        if spec.lakes_csv:
            # no lake state at the Simu start (no warm-up states): EF5 would
            # start every lake FULL - start them at S0 like any cold start
            lakes.ensure_states(rdir, run_start, lake_set,
                                os.path.join(grid_dir, "dem_clip.tif"), lake_mult)

    yield ("status", "running CREST — streaming…")
    if use_mock:
        handle = MockEF5(out_dir, gauge_id=g["id"], model=ef5_model, bounds=bbox,
                         n_steps=run_hours + 1, t0=run_start, delay=0.15,   # inclusive of run_end
                         write_grids=grids, has_obs=not g.get("virtual"),
                         facc_path=os.path.join(grid_dir, "facc_clip.tif")).start()
    else:
        handle = run_ef5(spec.control_path, out_dir, g["id"], model=ef5_model)

    new_rows = []
    final_rc = None
    for ev in stream_run(handle, poll=0.2, cancel=cancel):
        if ev["kind"] == "hydro":
            new_rows += ev["rows"]
            if rows_pre_served:
                continue     # hydrograph already fully on screen from the
                             # cached rows; streaming them again would double
                             # every point (the frontend appends blindly)
        ev["bbox"] = bbox
        if (ev["kind"] == "done" and not use_mock
                and (ev.get("returncode") not in (0, None) or not new_rows)):
            # surface the EF5 log so failures (or silent empty runs) are
            # debuggable from the UI/SSE
            try:
                with open(os.path.join(out_dir, "ef5_run.log"),
                          encoding="utf-8", errors="replace") as fh:
                    tail = fh.read()[-2000:]
                # list what EF5 actually wrote — a name/case mismatch shows here
                produced = sorted(os.path.basename(p) for p in
                                  glob.glob(os.path.join(out_dir, "ts.*.csv")))
                yield ("status", f"⚠️ EF5 rc={ev['returncode']}, {len(new_rows)} new "
                                 f"rows; expected {os.path.basename(handle.ts_path)}; "
                                 f"wrote {produced or '[no ts.*.csv]'} — log tail:\n{tail}")
            except Exception:
                pass
        if ev["kind"] == "done":
            final_rc = ev.get("returncode")
        if (ev["kind"] == "done" and not use_mock and not custom
                and ev.get("returncode") in (0, None)):
            # M4: share the states EF5 wrote (warm-up + end state) - before the
            # done event, which some callers stop reading at. Nowcast/event
            # runs never replace a state the hindcast chain already saved
            try:
                if spec.lakes_csv:
                    lakes.tag_states(rdir, lake_set, lake_mult)
                runstates.commit(rdir, sdir, clobber=not nowcast_t0)
            except Exception as e:
                from hf_data import crashlog
                crashlog.capture("runstates:commit", e, gauge=g["id"], model=state_model)
        yield (ev["kind"], ev)

    # persist result + state-save times for future overlap reuse (hourly runs only)
    if hourly:
        st = []
        if final_rc in (0, None) and not use_mock and not custom:
            # only states this run really saved and committed
            st.append(run_end.strftime(statecache.TS_FMT))
            if warmup_start:
                st.append(run_start.strftime(statecache.TS_FMT))
        try:
            statecache.save_record(g["id"], cache_model, pl["cached_rows"] + new_rows, st,
                                   variant=variant)
        except Exception as e:                    # caching silently stops working
            from hf_data import crashlog
            crashlog.capture("statecache:save", e, gauge=g["id"], model=cache_model)


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

"""CONUS state mosaic: the fleet's per-gauge EF5 states stitched into one grid.

Every fleet state bundle (CREST_fleet states/<gid>_<model>[-spd].pqf) holds
the state grids of ONE gauge's run domain — a headwater basin, or for a speed
run only the incremental area below its cut gauges. All of them sit on the
same CONUS 3-arc-second grid and (gauged fleet) on the same 10-day schedule,
so together they tile the gauged part of CONUS: the area a speed run left out
is exactly what its upstream gauges' own runs simulated.

The mosaic is VIRTUAL: nothing is copied. An index (mosaic/index.json in the
fleet repo) records where each bundle sits on the CONUS grid; a read pulls
only the columns of the requested time from the bundles that intersect the
requested window (parquet range requests, a few hundred kB per bundle instead
of the whole 5-year file) and pastes their valid cells. A second physical
copy would double the fleet's ~150 GB and go stale with every gauge the
fleet finishes; this one is current by construction.

Overlaps (a gauge that failed the obs-coverage gate stays inside its
downstream neighbour's domain AND has its own run): the run of the smallest
basin wins — it is the one whose parameters a full-basin run applies to that
cell (per-gauge multipliers), so sources are pasted largest drainage area
first.

Holes (no gauge, fleet not there yet, other model) stay nodata: EF5 loads a
state grid by location and skips nodata cells, which keep their cold-start
value (CRESTPhysModel/KinematicRoute::InitializeStates).

Used by pipeline.run_gauge to warm-start a FULL-basin run of a gauge the
fleet pre-ran under the speed scheme; stitch_window() serves any window.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timedelta

import numpy as np

from hf_data import fleetstore

REPO = fleetstore.REPO
INDEX_FILE = "mosaic/index.json"
CELL = 1.0 / 1200.0                        # 3 arc-seconds
NODATA = -9999.0
TTL_S = float(os.environ.get("CREST_MOSAIC_TTL_S", "21600"))
WB_VARS = {"crestphys": ("crestphys_SM", "crestphys_GW"), "crest": ("crest_SM",)}
KW_VARS = ("kwr_pCQ", "kwr_pOQ", "kwr_IR")
SNOW_VARS = ("snow17_ati", "snow17_wq", "snow17_wi", "snow17_deficit")
T_FMT = "%Y%m%d_%H%M"

_lock = threading.Lock()
_listing: tuple[float, dict] | None = None
_index: tuple[float, dict] | None = None
_footers: dict[str, tuple] = {}


def _token():
    return os.environ.get("HF_TOKEN")


def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem(token=_token())


def base_model(model: str) -> str:
    return model[:-4] if model.endswith("-spd") else model


# ---- what the fleet holds ----------------------------------------------------
def listing(force: bool = False) -> dict:
    """{gauge id: [state keys]} of the gauged fleet bundles (one repo listing,
    remembered for TTL_S)."""
    global _listing
    now = time.time()
    with _lock:
        if _listing and not force and now - _listing[0] < TTL_S:
            return _listing[1]
    from huggingface_hub import HfApi
    out: dict[str, list[str]] = {}
    for f in HfApi(token=_token()).list_repo_files(REPO, repo_type="dataset"):
        if f.startswith("states/") and f.endswith(".pqf") and not f.startswith("states/V"):
            key = f[len("states/"):-len(".pqf")]
            out.setdefault(key.split("_", 1)[0], []).append(key)
    with _lock:
        _listing = (now, out)
    return out


def _open(key: str):
    """(ParquetFile, grid dict) of a fleet bundle, footer read over HTTP."""
    import pyarrow.parquet as pq
    fh = _fs().open(f"datasets/{REPO}/states/{key}.pqf", "rb",
                    block_size=1 << 20, cache_type="readahead")
    pf = pq.ParquetFile(fh, pre_buffer=True)
    md = pf.schema_arrow.metadata or {}
    grid = {"nrows": int(md[b"nrows"]), "ncols": int(md[b"ncols"]),
            "xll": float(md[b"xllcorner"]), "yll": float(md[b"yllcorner"]),
            "cell": float(md[b"cellsize"]), "nodata": float(md[b"nodata"])}
    return pf, grid


def _times(names: list[str], var: str) -> list[datetime]:
    out = []
    pre = var + "_"
    for n in names:
        if n.startswith(pre):
            try:
                out.append(datetime.strptime(n[len(pre):], T_FMT))
            except ValueError:
                pass
    return sorted(out)


def entry(key: str) -> dict:
    """Index entry of one bundle: where it sits on the CONUS grid + its times."""
    pf, g = _open(key)
    gid, model = key.split("_", 1)
    wb = WB_VARS.get(base_model(model), ("",))[0]
    ts = _times(pf.schema_arrow.names, wb)
    e = {"key": key, "gauge": gid, "model": model,
         "xll": g["xll"], "yll": g["yll"], "nrows": g["nrows"], "ncols": g["ncols"],
         "n_times": len(ts)}
    e.update(pack_times(ts))
    return e


def pack_times(ts: list[datetime]) -> dict:
    """Times as a regular run (t0, step_h, n_reg) + the few that leave it
    (the fleet's schedule is 10-day steps with a shorter last one)."""
    if not ts:
        return {}
    out = {"t0": ts[0].strftime(T_FMT), "t1": ts[-1].strftime(T_FMT), "n_reg": 1}
    if len(ts) > 1:
        step = (ts[1] - ts[0]).total_seconds()
        n = 1
        while n < len(ts) and (ts[n] - ts[n - 1]).total_seconds() == step:
            n += 1
        out["step_h"], out["n_reg"] = step / 3600.0, n
        if n < len(ts):
            out["extra"] = [t.strftime(T_FMT) for t in ts[n:]]
    return out


def entry_times(e: dict) -> list[datetime]:
    if e.get("times"):
        return [datetime.strptime(s, T_FMT) for s in e["times"]]
    if not e.get("t0"):
        return []
    t0 = datetime.strptime(e["t0"], T_FMT)
    out = [t0 + timedelta(hours=e.get("step_h", 0.0) * i)
           for i in range(int(e.get("n_reg", 1)))]
    return out + [datetime.strptime(s, T_FMT) for s in e.get("extra", [])]


# ---- the index (CONUS manifest) ----------------------------------------------
def load_index(force: bool = False) -> dict:
    """{key: entry} from the fleet repo ({} when it does not exist yet)."""
    global _index
    now = time.time()
    with _lock:
        if _index and not force and now - _index[0] < TTL_S:
            return _index[1]
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, INDEX_FILE, repo_type="dataset", token=_token(),
                            force_download=force)
        with open(p, encoding="utf-8") as fh:
            idx = json.load(fh).get("bundles", {})
    except Exception:
        idx = {}
    with _lock:
        _index = (now, idx)
    return idx


def refresh_index(workers: int = 8, upload: bool = True, log=print) -> dict:
    """Bring the index up to the fleet's current content: read the footer of
    every bundle that is new or was re-uploaded since it was indexed."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi
    api = HfApi(token=_token())
    info = api.dataset_info(REPO, files_metadata=True)
    blobs = {s.rfilename[len("states/"):-len(".pqf")]: (s.blob_id, s.size)
             for s in info.siblings
             if s.rfilename.startswith("states/") and s.rfilename.endswith(".pqf")
             and not s.rfilename.startswith("states/V")}
    idx = dict(load_index(force=True))
    for k in [k for k in idx if k not in blobs]:
        del idx[k]                                   # bundle left the fleet
    todo = [k for k, (oid, _) in blobs.items() if idx.get(k, {}).get("oid") != oid]
    log(f"mosaic index: {len(blobs)} bundles in the fleet, {len(todo)} to read")
    areas = _areas()

    def one(k):
        for attempt in range(3):
            try:
                e = entry(k)
                e["oid"], e["bytes"] = blobs[k]
                e["area"] = areas.get(e["gauge"])
                return e
            except Exception as ex:
                err = ex
                time.sleep(2 + 3 * attempt)
        log(f"  {k}: {type(err).__name__}: {err}")
        return None

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for e in pool.map(one, todo):
            done += 1
            if e:
                idx[e["key"]] = e
            if done % 250 == 0:
                log(f"  {done}/{len(todo)}")
    if upload and todo:
        body = json.dumps({"updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                           "cell": CELL, "bundles": idx}).encode()
        api.upload_file(path_or_fileobj=body, path_in_repo=INDEX_FILE, repo_id=REPO,
                        repo_type="dataset",
                        commit_message=f"mosaic index: {len(idx)} bundles")
    global _index
    with _lock:
        _index = (time.time(), idx)
    return idx


_refreshing = threading.Lock()
_last_check = 0.0


def maybe_refresh(min_age_h: float = 24.0) -> None:
    """Keep the index current with the growing fleet: a background refresh at
    most every min_age_h. Called from the event runner's hourly tick — ONE
    Space writes the index, so there is never a second writer to collide with."""
    global _last_check
    if time.time() - _last_check < min_age_h * 3600 or not _token():
        return
    _last_check = time.time()

    def work():
        if not _refreshing.acquire(blocking=False):
            return
        try:
            age = float("inf")
            try:
                from huggingface_hub import hf_hub_download
                p = hf_hub_download(REPO, INDEX_FILE, repo_type="dataset",
                                    token=_token(), force_download=True)
                with open(p, encoding="utf-8") as fh:
                    upd = json.load(fh).get("updated", "")
                age = (datetime.utcnow() - datetime.strptime(
                    upd, "%Y-%m-%d %H:%M UTC")).total_seconds() / 3600.0
            except Exception:
                pass
            if age >= min_age_h:
                refresh_index(workers=4, log=lambda m: None)
        except Exception as e:
            try:
                from hf_data import crashlog
                crashlog.capture("conusstate:refresh", e)
            except Exception:
                pass
        finally:
            _refreshing.release()

    threading.Thread(target=work, daemon=True).start()


def _areas() -> dict:
    try:
        from hf_data import gauges
        cat = gauges.load_catalog()
        return {str(s).zfill(8): float(a) for s, a in
                zip(cat["STAID"], cat["DRAIN_SQKM"]) if a == a}
    except Exception:
        return {}


# ---- stitching -----------------------------------------------------------------
def _window(g: dict, dst: dict):
    """Overlap of a source grid with the target grid as index slices
    (src rows, src cols, dst rows, dst cols) or None. Both are on the CONUS
    3" lattice, so the offset is a whole number of cells."""
    c0 = int(round((g["xll"] - dst["xll"]) / CELL))            # src col 0 in dst
    top_s = g["yll"] + g["nrows"] * CELL
    top_d = dst["yll"] + dst["nrows"] * CELL
    r0 = int(round((top_d - top_s) / CELL))                    # src row 0 in dst
    dr0, dr1 = max(r0, 0), min(r0 + g["nrows"], dst["nrows"])
    dc0, dc1 = max(c0, 0), min(c0 + g["ncols"], dst["ncols"])
    if dr0 >= dr1 or dc0 >= dc1:
        return None
    return (slice(dr0 - r0, dr1 - r0), slice(dc0 - c0, dc1 - c0),
            slice(dr0, dr1), slice(dc0, dc1))


def _pick_time(ts: list[datetime], t: datetime, max_back_d: float):
    """Latest state time at or before t, no older than max_back_d."""
    best = None
    for s in ts:
        if s <= t and (t - s).total_seconds() <= max_back_d * 86400 and (
                best is None or s > best):
            best = s
    return best


def source_keys(gauges: list[str], model: str) -> list[str]:
    """Fleet bundle of each gauge for this model family — its speed-scheme run
    when it has one (incremental area), else its full run. Largest drainage
    area first, so the smallest basin is pasted last and wins overlaps."""
    have = listing()
    bm = base_model(model)
    areas = _areas()
    keys = []
    for gid in gauges:
        ks = have.get(str(gid).zfill(8), [])
        for want in (f"{str(gid).zfill(8)}_{bm}-spd", f"{str(gid).zfill(8)}_{bm}"):
            if want in ks:
                keys.append(want)
                break
    return sorted(keys, key=lambda k: -(areas.get(k.split("_", 1)[0]) or 0.0))


def window_keys(dst: dict, model: str, extra: list[str] | None = None) -> list[str]:
    """Every indexed bundle of this model family whose grid intersects the
    target grid, plus `extra` keys (bundles known from the repo listing but
    not indexed yet). One bundle per gauge (its speed run when it has one),
    largest drainage area first. This is what closes the upstream chain: the
    cut gauges of an upstream gauge's speed run need not be inside the
    outlet's own gauge scan."""
    bm = base_model(model)
    areas = _areas()
    by_gauge: dict[str, str] = {}
    for k in list(extra or []):
        by_gauge[k.split("_", 1)[0]] = k
    for k, en in load_index().items():
        if base_model(en["model"]) != bm or _window(en, dst) is None:
            continue
        cur = by_gauge.get(en["gauge"])
        if cur is None or (en["model"].endswith("-spd") and not cur.endswith("-spd")):
            by_gauge[en["gauge"]] = k
    return sorted(by_gauge.values(), key=lambda k: -(areas.get(k.split("_", 1)[0]) or 0.0))


def common_time(keys: list[str], t: datetime, max_back_d: float) -> datetime | None:
    """Latest state time <= t that the FIRST source has (all gauged bundles
    share the 10-day schedule; a source missing that time leaves a hole)."""
    idx = load_index()
    for k in keys:
        e = idx.get(k)
        ts = entry_times(e) if e else _times(_open(k)[0].schema_arrow.names,
                                             WB_VARS[base_model(k.split("_", 1)[1])][0])
        s = _pick_time(ts, t, max_back_d)
        if s is not None:
            return s
    return None


def stitch(dst: dict, keys: list[str], t: datetime, snow: bool = True,
           mask: np.ndarray | None = None, workers: int = 12) -> dict:
    """Paste the state grids of time t from the bundles `keys` (paste order)
    onto the target grid dst = {xll, yll, nrows, ncols}.

    Two passes, so a bundle that merely shares the window (a neighbouring
    basin inside the same rectangle) costs one small read, not nine:
      1. the water-balance grid of every bundle -> who puts cells into `mask`
      2. the remaining state grids of those contributors only.

    Returns {"grids": {var: float32 array}, "coverage": fraction of `mask`
    (or of the whole window) holding state, "sources": [(key, cells)] of the
    contributors, "missing": [keys without this time / unreadable]}."""
    from concurrent.futures import ThreadPoolExecutor
    ts = t.strftime(T_FMT)
    shape = (dst["nrows"], dst["ncols"])

    def grid(pf, g, w, cols):
        tb = pf.read(columns=cols)
        nod = np.float32(np.float16(g["nodata"]))
        out = {}
        for c in cols:
            a = tb[c].to_numpy().astype("float32").reshape(g["nrows"], g["ncols"])
            a = a[w[0], w[1]].copy()
            a[a == nod] = NODATA
            out[c[:-len(ts) - 1]] = a
        return out

    def first(k):
        wb = WB_VARS[base_model(k.split("_", 1)[1])][0]
        for attempt in range(2):
            try:
                pf, g = _open(k)
                if f"{wb}_{ts}" not in pf.schema_arrow.names:
                    return k, None                   # bundle lacks this time
                w = _window(g, dst)
                if w is None:
                    return k, ()
                return k, (pf, g, w, grid(pf, g, w, [f"{wb}_{ts}"])[wb])
            except Exception:
                time.sleep(1.5)
        return k, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        got = dict(pool.map(first, keys))

    have = np.zeros(shape, dtype=bool)
    ref = mask if mask is not None else np.ones(shape, dtype=bool)
    sources, missing, contrib = [], [], []
    for k in keys:
        d = got[k]
        if d is None:
            missing.append(k)
            continue
        if not d:
            continue
        pf, g, w, a = d
        n = int(((a != NODATA) & ref[w[2], w[3]]).sum())
        if n:
            contrib.append(k)
            sources.append((k, n))

    def rest(k):
        pf, g, w, a = got[k]
        bm = base_model(k.split("_", 1)[1])
        names = set(pf.schema_arrow.names)
        cols = [f"{v}_{ts}" for v in WB_VARS[bm][1:] + KW_VARS
                + (SNOW_VARS if snow else ()) if f"{v}_{ts}" in names]
        for attempt in range(2):
            try:
                return k, {WB_VARS[bm][0]: a, **(grid(pf, g, w, cols) if cols else {})}
            except Exception:
                time.sleep(1.5)
        return k, {WB_VARS[bm][0]: a}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        full = dict(pool.map(rest, contrib))

    grids: dict[str, np.ndarray] = {}
    for k in contrib:                                # paste order = keys order
        w = got[k][2]
        for var, a in full[k].items():
            tgt = grids.setdefault(var, np.full(shape, NODATA, dtype="float32"))
            ok = a != NODATA
            view = tgt[w[2], w[3]]
            view[ok] = a[ok]
            if var in WB_VARS["crestphys"] + WB_VARS["crest"]:
                have[w[2], w[3]] |= ok
    tot = int(ref.sum())
    return {"grids": grids, "coverage": float((have & ref).sum()) / tot if tot else 0.0,
            "sources": sources, "missing": missing, "time": t}


def write_states(res: dict, dst: dict, state_dir: str, overwrite: bool = False) -> int:
    """Stitched grids -> EF5 state GeoTIFFs (<var>_<YYYYMMDD_HHMM>.tif)."""
    import rasterio
    from rasterio.transform import from_origin
    os.makedirs(state_dir, exist_ok=True)
    ts = res["time"].strftime(T_FMT)
    n = 0
    for var, a in res["grids"].items():
        if not (a != NODATA).any():
            continue
        p = os.path.join(state_dir, f"{var}_{ts}.tif")
        if os.path.exists(p) and not overwrite:
            continue
        tmp = p + ".tmp"
        with rasterio.open(tmp, "w", driver="GTiff", height=dst["nrows"],
                           width=dst["ncols"], count=1, dtype="float32",
                           crs="EPSG:4326",
                           transform=from_origin(dst["xll"],
                                                 dst["yll"] + dst["nrows"] * CELL,
                                                 CELL, CELL),
                           nodata=NODATA, tiled=False, blockysize=1) as ds:
            ds.write(a, 1)
        os.replace(tmp, p)
        n += 1
    return n


def grid_of(path: str) -> dict:
    """Target-grid dict of a raster on the CONUS lattice (e.g. dem_clip.tif)."""
    import rasterio
    with rasterio.open(path) as ds:
        return {"xll": ds.transform.c, "yll": ds.transform.f - ds.height * abs(ds.transform.e),
                "nrows": ds.height, "ncols": ds.width}


def basin_mask(basic_dir: str, lat: float, lon: float, area: float | None):
    """Cells draining to the outlet on the run's clipped grid (the EF5 nodes)."""
    import rasterio
    from pysheds.grid import Grid
    from hf_data import neighbors
    facc_p = os.path.join(basic_dir, "facc_clip.tif")
    fdir_p = os.path.join(basic_dir, "fdir_clip.tif")
    with rasterio.open(facc_p) as ds:
        acc = ds.read(1).astype("float64")
        tr = ds.transform
        nod = ds.nodata
    if nod is not None:
        acc[acc == nod] = np.nan
    acc[acc < 0] = np.nan
    rc = neighbors._snap(acc, tr, lat, lon, area)
    if rc is None:
        return None
    grid = Grid.from_raster(fdir_p)
    fd = grid.read_raster(fdir_p)
    return np.asarray(grid.catchment(x=int(rc[1]), y=int(rc[0]), fdir=fd,
                                     xytype="index"), dtype=bool)


# at/above this share of the basin holding state the run takes the latest
# fleet state and a short warm-up; the few cells without one start cold
MIN_COVER = float(os.environ.get("CREST_MOSAIC_MIN_COVER", "0.90"))
MARK = ".mosaic.json"


def warm_state(g: dict, model: str, upstream: list[str], basic_dir: str,
               state_dir: str, t_start: datetime, warmup_days: float,
               snow: bool = True) -> dict | None:
    """Stitch a full-basin state for a run of gauge g starting at t_start and
    write it into state_dir. Returns None when the fleet has nothing, else
      {"time", "coverage", "sources", "n_sources", "tier"}
    tier "full":    the basin is covered — the state is the latest fleet time
                    at/before t_start (a short warm-up bridges the rest);
    tier "partial": part of the basin has no pre-run — the state is taken one
                    warm-up length earlier, so the uncovered cells still get
                    their full spin-up while the covered ones start spun-up."""
    dst = grid_of(os.path.join(basic_dir, "dem_clip.tif"))
    keys = window_keys(dst, model,
                       extra=source_keys([g["id"]] + list(upstream), model))
    if not keys:
        return None
    mask = basin_mask(basic_dir, g["lat"], g["lon"], g.get("area"))
    back = max(float(warmup_days), 10.0)
    t = common_time(keys, t_start, back)
    if t is None:
        return None
    mark_p = os.path.join(state_dir, MARK)
    want = t.strftime(T_FMT)               # the mark is filed under the time asked for
    try:                                   # this very state is already on disk
        with open(mark_p) as fh:
            m = json.load(fh)
        hit = m.get(want)
        if hit and hit.get("keys") == keys and os.path.exists(os.path.join(
                state_dir, f"{WB_VARS[base_model(model)][0]}_{hit['at']}.tif")):
            return {**hit, "time": datetime.strptime(hit["at"], T_FMT), "reused": True}
    except Exception:
        m = {}
    # which bundles put cells into this basin does not depend on the time:
    # after the first stitch only those are read
    known = m.get("_contrib") or {}
    use = [k for k in keys if k in set(known.get("sources", []))] \
        if known.get("keys") == keys else keys
    tier = "full" if known.get("coverage", 1.0) >= MIN_COVER else "partial"
    if tier == "partial" and use is not keys:        # known to be partial
        t = common_time(keys, t_start - timedelta(days=max(warmup_days - 10, 0)),
                        back) or t
    res = stitch(dst, use, t, snow=snow, mask=mask)
    if res["coverage"] < MIN_COVER and tier == "full":
        tier = "partial"
        t2 = common_time(keys, t_start - timedelta(days=max(warmup_days - 10, 0)), back)
        if t2 is not None and t2 != t:
            res = stitch(dst, [k for k, _ in res["sources"]], t2, snow=snow, mask=mask)
            t = t2
    if use is keys and res["sources"]:
        m["_contrib"] = {"keys": keys, "sources": [k for k, _ in res["sources"]],
                         "coverage": round(res["coverage"], 4)}
    if not res["sources"] or res["coverage"] <= 0:
        return None
    write_states(res, dst, state_dir, overwrite=True)
    out = {"coverage": round(res["coverage"], 4),
           "n_sources": sum(1 for _, n in res["sources"] if n),
           "sources": [k for k, n in res["sources"] if n], "keys": keys,
           "tier": tier, "at": t.strftime(T_FMT)}
    try:
        m[want] = out
        with open(mark_p, "w") as fh:
            json.dump(m, fh)
    except OSError:
        pass
    return {**out, "time": t}


def stitch_window(bbox: tuple, model: str, t: datetime, snow: bool = False) -> dict:
    """The mosaic over any (w, s, e, n) window at state time t — every indexed
    bundle of this model family that intersects it. Returns stitch()'s dict
    plus "grid" (the target grid)."""
    w, s, e, n = bbox
    xll = np.floor(w / CELL) * CELL
    yll = np.floor(s / CELL) * CELL
    dst = {"xll": float(xll), "yll": float(yll),
           "nrows": int(np.ceil((n - yll) / CELL)), "ncols": int(np.ceil((e - xll) / CELL))}
    res = stitch(dst, window_keys(dst, model), t, snow=snow)
    res["grid"] = dst
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CONUS state mosaic index")
    ap.add_argument("--refresh", action="store_true", help="update mosaic/index.json")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.refresh:
        ix = refresh_index(workers=a.workers)
        print(f"index: {len(ix)} bundles")

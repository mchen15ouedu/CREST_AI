"""Lakes, reservoirs and ponds for EF5's lake module (PI 2026-10-27: "identify
the reservoirs and lakes for the simulations and turn on the lake module,
start with the west").

hf_data/data/lakes.parquet (scripts/build_lakes.py: USACE NID dams +
HydroLAKES, snapped to EF5's HydroSHEDS 3-arc-second grid) holds one row per
water body: its outlet cell, threshold volume V, retention K and starting
storage S0 (the PI-approved storage scheme). A run hands EF5 the rows whose
outlet cell is one of ITS nodes - a lake outside the basin must never reach
EF5 (the pinned EF5 put such a lake on the basin outlet).

EF5 facts this module relies on (fork, lake-state-fix):
  * lakes.csv th_volume is in MILLION m3 (BasinConfigSection::ReadLakesFromCSV,
    not the km3 of the README), area km2, klake hours;
  * a lake sits on its lat/lon cell only with the `exact` column (else
    LakeMap moves it to the largest FAM within 50 cells, ~4.5 km);
  * lake states are lake_storage_<YYYYMMDD_HHMM>.tif (m3) and lake_outflow_
    (m3/s) on the run raster, read at the lake's own cell; a cold start
    without them begins FULL (storage = th_volume);
  * fork 5a26a86 saved one file per lake and the last writer won, so every
    other lake reloaded 0. Lakes need fork >= 1011809 (SaveStatesAll, the
    exact column, the continuous outlet law); CREST_LAKES stays "off" on a
    host until its EF5 has it.

Which run gets lakes: CREST_LAKES = off (default) | west (outlets west of
pipeline.WEST_LON, the CREST region) | all.
"""
from __future__ import annotations

import glob
import json
import os
import threading

import numpy as np

TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                          "lakes.parquet")
WEST_LON = -105.0                 # = pipeline.WEST_LON (no import cycle)
MULT_KEYS = ("lake_v", "lake_k")  # calibration multipliers on V and K

_lock = threading.Lock()
_tbl = None


def region() -> str:
    r = os.environ.get("CREST_LAKES", "off").strip().lower()
    return r if r in ("off", "west", "all") else "off"


def enabled(outlet_lon: float) -> bool:
    r = region()
    return r == "all" or (r == "west" and float(outlet_lon) < WEST_LON)


def _load():
    """(DataFrame, metadata) or (None, {}) when the table is missing or is a
    Git LFS pointer (the Spaces and the VM clone without git-lfs)."""
    global _tbl
    if _tbl is not None:
        return _tbl
    with _lock:
        if _tbl is None:
            try:
                with open(TABLE_PATH, "rb") as fh:
                    if fh.read(4) != b"PAR1":
                        raise OSError("not a parquet file (LFS pointer?)")
                import pyarrow.parquet as pq
                t = pq.read_table(TABLE_PATH)
                meta = json.loads((t.schema.metadata or {}).get(b"lakes", b"{}"))
                _tbl = (t.to_pandas(), meta)
            except (OSError, ValueError):
                _tbl = (None, {})
    return _tbl


def table_version() -> str:
    return _load()[1].get("table_version", "")


def multipliers(*sources) -> tuple[float, float]:
    """(v, k) from the first parameter dict carrying lake_v / lake_k (panel
    overrides, then the stored best set); 1.0 when none does."""
    got = {}
    for s in sources:
        for key in MULT_KEYS:
            if s and key in s and key not in got:
                got[key] = float(s[key])
    return got.get("lake_v", 1.0), got.get("lake_k", 1.0)


def for_run(nodes: np.ndarray, tr, mult: tuple[float, float] = (1.0, 1.0)):
    """The water bodies whose outlet cell is one of the run's EF5 nodes (bool
    mask on the run grid with affine tr), with their run-grid row/col and the
    lakes.csv values (multipliers applied; S0 keeps its fraction of V).
    None when the table is unavailable."""
    df, _ = _load()
    if df is None:
        return None
    H, W = nodes.shape
    c = np.floor((df.snap_lon.values - tr.c) / tr.a).astype(np.int64)
    r = np.floor((df.snap_lat.values - tr.f) / tr.e).astype(np.int64)
    inside = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    keep = np.zeros(len(df), dtype=bool)
    keep[inside] = nodes[r[inside], c[inside]]
    out = df[keep].copy()
    out["r"], out["c"] = r[keep], c[keep]
    v, k = mult
    out["th_volume_mm3"] = out.v_km3 * 1e3 * v
    out["s0_m3"] = out.s0_km3 * 1e9 * v
    out["klake_h"] = out.k_h * k
    return out.reset_index(drop=True)


def write_csv(path: str, lakes) -> str:
    """EF5 lakes.csv. th_volume in MILLION m3 (the active reader's unit);
    exact=1 keeps each lake on its snapped cell (EF5 would otherwise move it
    to the largest FAM within 50 cells - onto a bigger stream)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("name,lat,lon,th_volume,area,klake,exact\n")
        for x in lakes.itertuples():
            fh.write(f"{x.id},{x.snap_lat:.7f},{x.snap_lon:.7f},{x.th_volume_mm3:.6g},"
                     f"{x.area_km2:.6g},{x.klake_h:.6g},1\n")
    return os.path.abspath(path)


def _stamp(t) -> str:
    return t.strftime("%Y%m%d_%H%M")


def _write_sig(path: str, sig: dict) -> None:
    """lake_set sidecar via a temp file + os.replace: the run folder's copy
    may be a staged hard link of the shared one, which must never be written
    through."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(sig, fh)
    os.replace(tmp, path)


def _sig(lakes, mult) -> dict:
    return {"table": table_version(), "mult": [float(mult[0]), float(mult[1])],
            "ids": sorted(str(i) for i in lakes.id)}


def prepare_states(state_dir: str, t, lakes, dem_path: str,
                   mult: tuple[float, float] = (1.0, 1.0)) -> dict:
    """Make the lake states EF5 loads at time t hold the right storage for
    THIS lake set: a lake simulated by the run that saved them (same table and
    multipliers, recorded in lake_set_<stamp>.json) keeps its saved storage;
    every other lake starts at its S0 (EF5 would start it full), as does a
    lake whose saved storage is not a finite, non-negative volume (EF5's
    -9999, a NaN). Returns {"kept": n, "initial": n}."""
    import rasterio
    stamp = _stamp(t)
    sp = os.path.join(state_dir, f"lake_storage_{stamp}.tif")
    op = os.path.join(state_dir, f"lake_outflow_{stamp}.tif")
    with rasterio.open(dem_path) as ds:
        prof = ds.profile
        shape = (ds.height, ds.width)
    saved_s = saved_o = None
    known = set()
    try:
        with open(os.path.join(state_dir, f"lake_set_{stamp}.json")) as fh:
            sig = json.load(fh)
        if (sig.get("table") == table_version()
                and [float(x) for x in sig.get("mult", [])] == [float(mult[0]), float(mult[1])]):
            known = set(sig.get("ids") or [])
    except (OSError, ValueError):
        pass
    if known and os.path.exists(sp) and os.path.exists(op):
        try:
            with rasterio.open(sp) as ds:
                saved_s = ds.read(1) if (ds.height, ds.width) == shape else None
            with rasterio.open(op) as ds:
                saved_o = ds.read(1) if (ds.height, ds.width) == shape else None
        except Exception:
            saved_s = saved_o = None
    st = np.zeros(shape, dtype="float32")
    of = np.zeros(shape, dtype="float32")
    kept = 0
    have = saved_s is not None and saved_o is not None
    for x in lakes.itertuples():
        s = float(saved_s[x.r, x.c]) if have and str(x.id) in known else float("nan")
        if np.isfinite(s) and s >= 0:
            o = float(saved_o[x.r, x.c])
            st[x.r, x.c], of[x.r, x.c] = s, (o if np.isfinite(o) and o >= 0 else 0.0)
            kept += 1
        else:
            st[x.r, x.c] = x.s0_m3
    prof = {**prof, "driver": "GTiff", "dtype": "float32", "count": 1,
            "nodata": -9999.0, "compress": "deflate"}
    os.makedirs(state_dir, exist_ok=True)
    for path, arr in ((sp, st), (op, of)):
        tmp = path + ".tmp"
        with rasterio.open(tmp, "w", **prof) as dst:
            dst.write(arr, 1)
        os.replace(tmp, path)        # a staged hard link is replaced, not written into
    _write_sig(os.path.join(state_dir, f"lake_set_{stamp}.json"), _sig(lakes, mult))
    return {"kept": kept, "initial": len(lakes) - kept}


def ensure_states(state_dir: str, t, lakes, dem_path: str,
                  mult: tuple[float, float] = (1.0, 1.0)) -> dict | None:
    """After a warm-up: when it left no lake state at t (it saved no states,
    or was killed), write the S0 states so EF5 does not start every lake
    full. None when the warm-up's own lake state is there."""
    if os.path.exists(os.path.join(state_dir, f"lake_storage_{_stamp(t)}.tif")):
        return None
    return prepare_states(state_dir, t, lakes, dem_path, mult)


def tag_states(state_dir: str, lakes, mult: tuple[float, float] = (1.0, 1.0)) -> int:
    """After EF5 ran: record which lake set each lake_storage grid it saved
    belongs to (lake_set_<stamp>.json beside it), so a later run trusts only
    its own lakes' storage. Returns how many sidecars were written."""
    n = 0
    sig = _sig(lakes, mult)
    for p in glob.glob(os.path.join(state_dir, "lake_storage_*.tif")):
        stamp = os.path.basename(p)[len("lake_storage_"):-len(".tif")]
        _write_sig(os.path.join(state_dir, f"lake_set_{stamp}.json"), sig)
        n += 1
    return n


def cache_sig(lakes, mult) -> dict | None:
    """The run's lake part of the rows/frames cache key (PI 2026-10-30: lake
    runs never merge with lake-off rows): table version + multipliers when the
    run simulates any lake; None without lakes (the plain key)."""
    if lakes is None or not len(lakes):
        return None
    return {"table": table_version(), "mult": [float(mult[0]), float(mult[1])]}


def provenance(lakes, mult) -> dict:
    """Record stamp: the lake set a result was simulated with."""
    if lakes is None:
        return {"on": False}
    return {"on": True, "table": table_version(), "n": int(len(lakes)),
            "mult": [float(mult[0]), float(mult[1])],
            "v_mm3": round(float(lakes.th_volume_mm3.sum()), 3)}

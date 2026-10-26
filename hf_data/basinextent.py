"""EF5 raster extents from basin geometry - the single source (PI 2026-10-23:
"No box is allowed").

Every EF5 run (dashboard hindcasts, calibration, autocal, the gauged fleet,
the ungauged nowcast, 2-D event runs) takes its raster extent and its outlet
cell from hf_data/data/basin_extent.parquet. scripts/build_basin_extent.py
builds it offline from:
  * GAGES-II basin polygons (USGS gauges) and HydroBASINS level-07 upstream
    outlines (pour points / virtual points),
  * the D8 catchment EF5 carves from the outlet cell, fixed once per outlet
    with the neighbors._snap rule on EF5's own HydroSHEDS 3-arc-second FAM.
The extent is the union of outline and catchment + ~5 km, on the HydroSHEDS
lattice: the tightest rectangle EF5's raster reader needs. EF5 simulates only
the carved catchment inside it.

An outlet without a usable row is NOT simulated - BasinExtentError carries
the reason. There is no fallback and no default rectangle.
"""
from __future__ import annotations

import json
import math
import os
import threading

import numpy as np

from hf_data.hucdomain import DomainError

TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                          "basin_extent.parquet")
# the table's own cap rules; this env var can only LOWER it (tests), never
# make a row runnable on one Space and refused on another
_CAP_ENV = os.environ.get("EF5_MAX_EXTENT_CELLS", "").strip()

_lock = threading.Lock()
_tbl = None                      # (DataFrame indexed by id, metadata dict)

# ESRI D8 codes -> (drow, dcol); same encoding as the HydroSHEDS DDM
_D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
       16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


class BasinExtentError(DomainError):
    """No basin geometry for this outlet: the run must not happen (never a box).
    `reason` is one of table_missing, no_row, no_terrain, over_cap, error,
    carve_touches_edge."""

    def __init__(self, gid: str, reason: str, detail: str = ""):
        self.gid, self.reason, self.detail = str(gid), reason, detail
        super().__init__(f"{gid}: no basin boundary ({reason}{': ' + detail if detail else ''})")


def _load():
    global _tbl
    if _tbl is not None:
        return _tbl
    with _lock:
        if _tbl is not None:
            return _tbl
        p = TABLE_PATH
        try:
            with open(p, "rb") as fh:
                head = fh.read(4)
        except OSError:
            raise BasinExtentError("*", "table_missing", p)
        if head != b"PAR1":       # e.g. a Git LFS pointer from a clone without git-lfs
            raise BasinExtentError("*", "table_missing",
                                   f"{p} is not a parquet file (LFS pointer?)")
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(p)
            meta = json.loads((t.schema.metadata or {}).get(b"basin_extent", b"{}"))
            df = t.to_pandas().set_index("id")
        except Exception as e:
            raise BasinExtentError("*", "table_missing", f"{type(e).__name__}: {e}")
        if not meta.get("table_version") or meta.get("partial"):
            raise BasinExtentError("*", "table_missing",
                                   "table has no version or is a partial QA build")
        _tbl = (df, meta)
        return _tbl


def table_version() -> str:
    return _load()[1]["table_version"]


def cap_cells() -> int:
    cap = int(_load()[1].get("cap_cells", 36_000_000))
    if _CAP_ENV:
        try:
            cap = min(cap, int(float(_CAP_ENV)))
        except ValueError:
            pass
    return cap


def row(gid: str):
    """The table row (pandas Series) for an outlet id, or None."""
    df, _ = _load()
    gid = str(gid)
    key = gid if gid.startswith("V") else gid.zfill(8)
    return df.loc[key] if key in df.index else None


def ef5_extent(g) -> dict:
    """{bounds (w,s,e,n), snap {lat, lon, area}, source, cells, carved_km2,
    flags, table} for gauge_info dict (or id) g. Raises BasinExtentError."""
    gid = g["id"] if isinstance(g, dict) else str(g)
    r = row(gid)
    if r is None:
        raise BasinExtentError(gid, "no_row", "outlet not in the basin-extent table")
    status = str(r["status"])
    if status != "ok":
        raise BasinExtentError(gid, status, str(r.get("flags") or ""))
    if int(r["cells"]) > cap_cells():
        raise BasinExtentError(gid, "over_cap", f"{int(r['cells']):,} cells")
    return {"bounds": (float(r["w"]), float(r["s"]), float(r["e"]), float(r["n"])),
            "snap": {"lat": float(r["snap_lat"]), "lon": float(r["snap_lon"]),
                     "area": float(r["snap_area_km2"])},
            "source": str(r["source"]), "cells": int(r["cells"]),
            "carved_km2": float(r["carved_km2"]),
            "flags": [f for f in str(r.get("flags") or "").split(",") if f],
            "snap_rule": str(r.get("snap_rule") or "nearest"),
            "table": table_version()}


def provenance(ext: dict) -> dict:
    """Small record stamp: which outline a result was simulated on."""
    return {"extent": "basin", "source": ext["source"], "table": ext["table"],
            "snap_rule": ext.get("snap_rule", "nearest"),
            "bounds": [round(v, 6) for v in ext["bounds"]]}


def size_km2(gid: str):
    """Drainage area used by the fleet's size split (table 'area')."""
    r = row(gid)
    return None if r is None else float(r["area"])


# --------------------------------------------------------------------------- #
# carved catchment on a run's clipped grid
# --------------------------------------------------------------------------- #
def upstream_mask(fdir: np.ndarray, r: int, c: int) -> np.ndarray:
    """Cells draining to (r, c) through ESRI-D8 `fdir` (any dtype; codes outside
    the D8 set, e.g. -9999 nodata, drain nowhere). Vectorised BFS over a donor
    index - the same carve EF5 does from that outlet cell."""
    H, W = fdir.shape
    n = H * W
    it = np.int32 if n < 2 ** 31 - 1 else np.int64   # ~half the RAM of int64
    rec = np.full(n, -1, dtype=it)
    blk = max(1, (1 << 22) // max(W, 1))         # ~4M cells per block: bounded temps
    for b0 in range(0, H, blk):
        sub = fdir[b0:b0 + blk]
        for code, (dr, dc) in _D8.items():
            rr, cc = np.nonzero(sub == code)
            rr += b0
            rr2, cc2 = rr + dr, cc + dc
            ok = (rr2 >= 0) & (rr2 < H) & (cc2 >= 0) & (cc2 < W)
            rec[rr[ok] * W + cc[ok]] = (rr2[ok] * W + cc2[ok]).astype(it)
    donors = np.nonzero(rec >= 0)[0].astype(it)
    donors = donors[np.argsort(rec[donors], kind="stable")]
    counts = np.bincount(rec[donors], minlength=n)
    del rec
    start = np.zeros(n + 1, dtype=np.int64)     # donors of cell k: donors[start[k]:start[k+1]]
    np.cumsum(counts, out=start[1:])
    del counts
    inb = np.zeros(n, dtype=bool)
    fr = np.array([r * W + c], dtype=np.int64)
    inb[fr] = True
    while len(fr):
        st, en = start[fr], start[fr + 1]
        cnt = en - st
        tot = int(cnt.sum())
        if tot == 0:
            break
        idx = np.repeat(st, cnt) + (np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt))
        nx = donors[idx].astype(np.int64)
        nx = nx[~inb[nx]]
        inb[nx] = True
        fr = nx
    return inb.reshape(H, W)


SNAP_RADIUS_KM = 20.0        # EF5's own outlet search reach
SNAP_TOL = 0.20              # drainage-area match tolerance


def cell_km2(lat, res: float):
    """Area of a lattice cell (km2) - the convention every Python-side FAM
    area uses (neighbors, domain). Never handed to EF5 as BASINAREA: EF5 uses
    110.574 km/deg and would move the outlet off the pinned cell."""
    return (res * 111.32) * (res * 111.32 * np.cos(np.radians(lat)))


def snap_cell(acc: np.ndarray, tr, lat: float, lon: float, area_km2: float,
              radius_km: float = SNAP_RADIUS_KM, tol: float = SNAP_TOL, within=None):
    """The outlet cell of a point with known drainage area (M1 snap rule):
    among the stream cells within radius_km whose FAM drainage area matches
    area_km2 within +-tol, the NEAREST one (ties: the better area match).
    within=(w, s, e, n): only cells whose centre lies inside (the outlet's
    own outline) - for small basins the 3-arc-second network may not resolve
    the gauge's creek, and the nearest same-size cell is then another creek.
    None when no cell qualifies - the outlet is then not simulated, never
    pinned to a hillslope or to another river. acc: flow accumulation in
    cells (NaN = nodata) on the north-up grid with affine tr."""
    if not (area_km2 and area_km2 > 0):
        return None
    H, W = acc.shape
    res = tr.a
    r0, c0 = cell_of(tr, lat, lon)
    rows = np.arange(H)
    lats = tr.f + (rows + 0.5) * tr.e
    a_km2 = acc * cell_km2(lats, res)[:, None]
    ratio = a_km2 / area_km2
    ok = np.isfinite(ratio) & (np.abs(ratio - 1.0) <= tol)
    if within is not None:
        w, s, e, n = within
        clon = tr.c + (np.arange(W) + 0.5) * tr.a
        ok &= ((lats >= s) & (lats <= n))[:, None] & ((clon >= w) & (clon <= e))[None, :]
    if not ok.any():
        return None
    rr, cc = np.nonzero(ok)
    dy = (rr - r0) * res * 110.574
    dx = (cc - c0) * res * 111.32 * math.cos(math.radians(lat))
    d = np.hypot(dx, dy)
    near = d <= radius_km
    if not near.any():
        return None
    rr, cc, d = rr[near], cc[near], d[near]
    err = np.abs(ratio[rr, cc] - 1.0)
    i = np.lexsort((err, np.round(d, 6)))[0]
    return int(rr[i]), int(cc[i])


def cell_of(tr, lat: float, lon: float):
    """(row, col) of the cell containing (lat, lon) on an north-up grid."""
    return int(math.floor((tr.f - lat) / -tr.e)), int(math.floor((lon - tr.c) / tr.a))


def carve(basic_dir: str, ext: dict, gid: str = "?") -> dict:
    """The run's EF5 nodes on its clipped grid: the upstream catchment of the
    table's outlet cell. Cached per (store, outlet) as carve_<r>_<c>.npy.
    Raises BasinExtentError(carve_touches_edge) when the catchment reaches the
    raster border - the runtime proof that the rectangle clips no basin."""
    import rasterio
    fdir_p = os.path.join(basic_dir, "fdir_clip.tif")
    with rasterio.open(fdir_p) as ds:
        tr = ds.transform
        H, W = ds.height, ds.width
        r, c = cell_of(tr, ext["snap"]["lat"], ext["snap"]["lon"])
        if not (0 <= r < H and 0 <= c < W):
            raise BasinExtentError(gid, "carve_touches_edge", "outlet outside the clip")
        cache = os.path.join(basic_dir, f"carve_{r}_{c}.npy")
        mask = None
        if os.path.exists(cache):
            try:
                mask = np.load(cache)
                if mask.shape != (H, W):
                    mask = None
            except Exception:
                mask = None
        if mask is None:
            mask = upstream_mask(ds.read(1), r, c)
            tmp = cache + f".{os.getpid()}.tmp.npy"
            try:
                np.save(tmp, mask)
                os.replace(tmp, cache)
            except OSError:
                pass
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        raise BasinExtentError(gid, "carve_touches_edge",
                               f"{int(mask.sum()):,}-cell catchment reaches the raster border")
    return {"mask": mask, "transform": tr, "cells": int(mask.sum())}

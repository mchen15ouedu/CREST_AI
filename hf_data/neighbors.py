"""Upstream/interior gauges for multi-gauge control files (boundary conditions).

Once the basin's DEM/DDM/FAM are clipped, other USGS gauges inside the domain
that drain INTO the outlet are found by snapping each candidate onto the
clipped flow-accumulation stream and tracing the D8 flow directions downstream
until the outlet cell (or the grid edge) is reached. Their USGS observations
are fed to EF5 as [Gauge] OBS= + WANTDA=true (+ DA_FILE in the task), so the
router assimilates the observed flow at those cells — effectively a boundary-
condition input. This matters most for gauges near the DEM edge, where the
upstream catchment is truncated by the clip.
"""
from __future__ import annotations

import math
import os

import numpy as np

from hf_data import gauges

MAX_BC_GAUGES = int(os.environ.get("CREST_MAX_BC_GAUGES", "8"))
EDGE_CELLS = 12                    # a gauge this close to the raster border is "at the edge"
_SNAP_CELLS = 12                   # snap search half-window (cells)

# ESRI D8 codes -> (drow, dcol)
_D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
       16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


def _cell_area_km2(lat: float, cell_deg: float) -> float:
    return (cell_deg * 111.32) * (cell_deg * 111.32 * math.cos(math.radians(lat)))


def _snap(acc, tr, lat: float, lon: float, drain_sqkm: float | None):
    """Snap (lat, lon) to the best stream cell of the CLIPPED flow-accumulation
    grid: the nearby cell whose upstream area best matches the gauge's drainage
    (or the largest stream when the area is unknown). Returns (row, col) or None."""
    cell = tr.a
    nr, nc = acc.shape
    r0 = int((tr.f - lat) / cell)
    c0 = int((lon - tr.c) / cell)
    if not (0 <= r0 < nr and 0 <= c0 < nc):
        return None
    rlo, rhi = max(0, r0 - _SNAP_CELLS), min(nr, r0 + _SNAP_CELLS + 1)
    clo, chi = max(0, c0 - _SNAP_CELLS), min(nc, c0 + _SNAP_CELLS + 1)
    win = acc[rlo:rhi, clo:chi].astype("float64")
    if not np.isfinite(win).any():
        return None
    if drain_sqkm and drain_sqkm > 0:
        expected = drain_sqkm / _cell_area_km2(lat, cell)
        score = np.abs(win - expected)
    else:
        score = -win
    rr, cc = np.mgrid[rlo:rhi, clo:chi]
    dist = np.hypot(rr - r0, cc - c0)                # stay near the gauge
    score = score + dist * (np.nanmax(np.abs(score)) / (2.0 * _SNAP_CELLS) * 0.5)
    score[~np.isfinite(win)] = np.inf
    r, c = np.unravel_index(int(np.argmin(score)), win.shape)
    return rlo + r, clo + c


def _drains_to(fdir, start, target, max_steps: int) -> bool:
    """Follow the D8 directions from `start`; True if the path reaches `target`
    (within one cell — snapping tolerance) before leaving the grid."""
    nr, nc = fdir.shape
    r, c = start
    tr_, tc = target
    for _ in range(max_steps):
        if abs(r - tr_) <= 1 and abs(c - tc) <= 1:
            return True
        step = _D8.get(int(fdir[r, c]))
        if step is None:                              # nodata / sink
            return False
        r, c = r + step[0], c + step[1]
        if not (0 <= r < nr and 0 <= c < nc):          # ran off the DEM edge
            return False
    return False


def upstream_gauges(outlet: dict, bbox, basic_dir: str,
                    max_gauges: int = MAX_BC_GAUGES) -> list[dict]:
    """USGS gauges inside the clipped domain whose flow drains to the outlet.

    outlet: {id, lat, lon, area}; bbox: (W, S, E, N); basic_dir holds
    fdir_clip.tif / facc_clip.tif. Returns [{id, name, lat, lon, area, at_edge,
    dist_km}, ...] largest-drainage first, capped at max_gauges.
    """
    import rasterio
    fdir_p = os.path.join(basic_dir, "fdir_clip.tif")
    facc_p = os.path.join(basic_dir, "facc_clip.tif")
    if not (os.path.exists(fdir_p) and os.path.exists(facc_p)):
        return []
    cat = gauges.load_catalog(bbox)
    cat = cat[cat.STAID != str(outlet["id"]).zfill(8)]
    if cat.empty:
        return []

    with rasterio.open(fdir_p) as ds:
        fdir = ds.read(1)
        tr = ds.transform
        nod = ds.nodata
    if nod is not None:
        fdir = np.where(fdir == nod, 0, fdir)
    with rasterio.open(facc_p) as ds:
        acc = ds.read(1).astype("float64")
        anod = ds.nodata
    if anod is not None:
        acc[acc == anod] = np.nan
    acc[acc < 0] = np.nan

    out_rc = _snap(acc, tr, outlet["lat"], outlet["lon"], outlet.get("area"))
    if out_rc is None:
        return []
    nr, nc = fdir.shape
    max_steps = 4 * (nr + nc)                        # a D8 path can wiggle, but not this much

    found = []
    for _, row in cat.iterrows():
        area = float(row.DRAIN_SQKM) if row.DRAIN_SQKM == row.DRAIN_SQKM else None
        rc = _snap(acc, tr, float(row.LAT_GAGE), float(row.LNG_GAGE), area)
        if rc is None or (abs(rc[0] - out_rc[0]) <= 1 and abs(rc[1] - out_rc[1]) <= 1):
            continue                                  # off-grid, or same cell as outlet
        if not _drains_to(fdir, rc, out_rc, max_steps):
            continue
        at_edge = (rc[0] < EDGE_CELLS or rc[1] < EDGE_CELLS
                   or rc[0] >= nr - EDGE_CELLS or rc[1] >= nc - EDGE_CELLS)
        found.append({"id": str(row.STAID).zfill(8), "name": str(row.STANAME),
                      "lat": float(row.LAT_GAGE), "lon": float(row.LNG_GAGE),
                      "area": area, "at_edge": bool(at_edge),
                      "dist_km": 111.0 * math.hypot(float(row.LAT_GAGE) - outlet["lat"],
                                                    float(row.LNG_GAGE) - outlet["lon"])})
    found.sort(key=lambda g: -(g["area"] or 0.0))
    return found[:max_gauges]


def write_da_file(obs_series: dict[str, list], path: str) -> str | None:
    """EF5 DA_FILE: `gauge,time,value` rows (Simulator::LoadDAFile). The file's
    presence switches assimilation ON; values supplement each gauge's OBS=.
    obs_series: {gauge_id: [(datetime, cms), ...]}."""
    rows = 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        for gid, series in obs_series.items():
            for dt, cms in series:
                fh.write(f"{gid},{dt:%Y-%m-%d %H:%M:%S},{cms:.6f}\n")
                rows += 1
    if not rows:
        os.remove(path)
        return None
    return os.path.abspath(path)


if __name__ == "__main__":
    # Allagash: outlet 01011000 (3187 km²) — 01010070 / 01010000 sit upstream
    outlet = {"id": "01011000", "lat": 47.0696, "lon": -69.0795, "area": 3186.8}
    from hf_data import basic, pipeline
    bbox = pipeline.basin_bbox(outlet)
    bdir = basic.store_dir(bbox)
    basic.clip_basic_data(bbox, bdir)
    ups = upstream_gauges(outlet, bbox, bdir)
    print(f"{len(ups)} upstream gauge(s):")
    for u in ups:
        print(f"  {u['id']} {u['name'][:40]:40s} {u['area'] or -1:8.1f} km² "
              f"edge={u['at_edge']} {u['dist_km']:.0f} km")

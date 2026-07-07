"""Speed-run domain builder: truncate the basin at boundary gauges.

A "speed run" simulates only the INCREMENTAL area between the cut-set boundary
gauges and the outlet; the water arriving from upstream of each cut gauge is
injected from its USGS observations (EF5 data assimilation), so the upstream
cells never need to be computed. A "full run" is the ordinary whole-basin
simulation.

Mechanics (why each piece exists):
  - cut set   = boundary gauges whose flow path to the outlet passes no other
                qualified boundary gauge ("no gauge in between"); every other
                boundary gauge lies upstream of a cut gauge and thus outside
                the truncated domain.
  - the DEM + flow directions are masked to nodata strictly upstream of each
    cut gauge (the gauge cell itself stays — it is the injection point).
  - the flow accumulation is RECOMPUTED on the truncated directions: EF5 sizes
    its node array from the FAM value at the gauges (BasicGrids.cpp
    nodes->resize), so keeping the original full-basin accumulation would
    over-allocate wildly on big rivers; the control's BASINAREA is set to the
    matching incremental area so EF5's gauge snapping stays consistent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import rasterio

from hf_data import neighbors

_PAD = 2                            # cells of padding around the kept area


def _cell_area_km2(lat: float, cell_deg: float) -> float:
    return (cell_deg * 111.32) * (cell_deg * 111.32 * math.cos(math.radians(lat)))


def _cell_center(tr, rc):
    r, c = rc
    return (tr.f - (r + 0.5) * tr.a, tr.c + (c + 0.5) * tr.a)   # (lat, lon)


def _write(path: str, data, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)


def cut_set(outlet_rc, snapped: list[tuple[dict, tuple]], fdir) -> list[tuple[dict, tuple]]:
    """Gauges whose D8 path to the outlet passes no other candidate's cell."""
    nr, nc = fdir.shape
    max_steps = 4 * (nr + nc)
    cells = {g["id"]: rc for g, rc in snapped}
    out = []
    for g, rc in snapped:
        others = [c for gid, c in cells.items() if gid != g["id"]]
        r, c = rc
        interior = False
        for _ in range(max_steps):
            if abs(r - outlet_rc[0]) <= 1 and abs(c - outlet_rc[1]) <= 1:
                break
            if any(abs(r - o[0]) <= 1 and abs(c - o[1]) <= 1 for o in others) \
                    and (r, c) != rc:
                interior = True                       # passes another gauge first
                break
            step = neighbors._D8.get(int(fdir[r, c]))
            if step is None:
                break
            r, c = r + step[0], c + step[1]
            if not (0 <= r < nr and 0 <= c < nc):
                break
        if not interior:
            out.append((g, rc))
    return out


def build_speed_domain(outlet: dict, bc_gauges: list[dict], basic_dir: str):
    """Build (or reuse) the truncated dem/fdir/facc for a speed run.

    outlet: {id, lat, lon, area}; bc_gauges: qualified boundary gauges (obs
    coverage already checked by the caller). Returns
      {dir, cells_total, cells_kept, kept_frac, speedup,
       outlet: {lat, lon, area_inc},
       cut: [{id, name, lat, lon, area_inc, at_edge}, ...]}
    or None when truncation isn't possible (no cut gauges / snapping failed).
    """
    if not bc_gauges:
        return None
    fdir_p = os.path.join(basic_dir, "fdir_clip.tif")
    facc_p = os.path.join(basic_dir, "facc_clip.tif")
    dem_p = os.path.join(basic_dir, "dem_clip.tif")

    with rasterio.open(facc_p) as ds:
        acc = ds.read(1).astype("float64")
        tr = ds.transform
        anod = ds.nodata
    if anod is not None:
        acc[acc == anod] = np.nan
    acc[acc < 0] = np.nan
    with rasterio.open(fdir_p) as ds:
        fdir = ds.read(1)
        fnod = ds.nodata
    if fnod is not None:
        fdir = np.where(fdir == fnod, 0, fdir)

    out_rc = neighbors._snap(acc, tr, outlet["lat"], outlet["lon"], outlet.get("area"))
    if out_rc is None:
        return None
    snapped = []
    for g in bc_gauges:
        rc = neighbors._snap(acc, tr, g["lat"], g["lon"], g.get("area"))
        if rc is not None and not (abs(rc[0] - out_rc[0]) <= 1 and abs(rc[1] - out_rc[1]) <= 1):
            snapped.append((g, rc))
    cut = cut_set(out_rc, snapped, fdir)
    if not cut:
        return None

    key = hashlib.sha1((",".join(sorted(g["id"] for g, _ in cut))).encode()).hexdigest()[:10]
    ddir = os.path.join(basic_dir, f"speed_{key}")
    meta_p = os.path.join(ddir, "domain.json")
    if all(os.path.exists(os.path.join(ddir, n)) for n in
           ("dem_clip.tif", "fdir_clip.tif", "facc_clip.tif")) and os.path.exists(meta_p):
        with open(meta_p) as fh:                      # store hit
            return json.load(fh)
    os.makedirs(ddir, exist_ok=True)

    # catchments via pysheds (dirmap default == ESRI D8, as in basic.derive_dir_acc)
    # pour points by (col, row) index — pysheds' coordinate snapping can land
    # one cell off the stream, which silently yields a 1-cell catchment
    from pysheds.grid import Grid
    grid = Grid.from_raster(fdir_p)
    fd = grid.read_raster(fdir_p)
    keep = np.asarray(grid.catchment(x=int(out_rc[1]), y=int(out_rc[0]), fdir=fd,
                                     xytype="index"), dtype=bool)
    cells_total = int(keep.sum())
    for g, rc in cut:
        cg = np.asarray(grid.catchment(x=int(rc[1]), y=int(rc[0]), fdir=fd,
                                       xytype="index"), dtype=bool)
        keep &= ~cg
        keep[rc] = True                               # the injection point stays
    keep[out_rc] = True
    cells_kept = int(keep.sum())
    if not cells_kept or cells_kept >= cells_total:
        return None

    with rasterio.open(dem_p) as ds:
        dem = ds.read(1).astype("float32")
        profile = ds.profile.copy()
    dem = np.where(keep, dem, -9999.0)
    fdir_t = np.where(keep, fdir, -9999).astype("float32")

    rows = np.any(keep, axis=1).nonzero()[0]
    cols = np.any(keep, axis=0).nonzero()[0]
    r0, r1 = max(0, rows[0] - _PAD), min(keep.shape[0], rows[-1] + _PAD + 1)
    c0, c1 = max(0, cols[0] - _PAD), min(keep.shape[1], cols[-1] + _PAD + 1)
    win_tr = rasterio.Affine(tr.a, tr.b, tr.c + c0 * tr.a,
                             tr.d, tr.e, tr.f + r0 * tr.e)
    profile.update(driver="GTiff", height=r1 - r0, width=c1 - c0, transform=win_tr,
                   tiled=False, blockysize=1, dtype="float32", nodata=-9999.0,
                   crs="EPSG:4326", interleave="pixel")
    profile.pop("compress", None)
    _write(os.path.join(ddir, "dem_clip.tif"), dem[r0:r1, c0:c1], profile)
    _write(os.path.join(ddir, "fdir_clip.tif"), fdir_t[r0:r1, c0:c1], profile)

    # recomputed accumulation on the truncated directions (see module docstring)
    grid2 = Grid.from_raster(os.path.join(ddir, "fdir_clip.tif"))
    fd2 = grid2.read_raster(os.path.join(ddir, "fdir_clip.tif"))
    acc2 = np.asarray(grid2.accumulation(fd2), dtype="float64")
    keep_c = keep[r0:r1, c0:c1]
    acc_out = np.where(keep_c, acc2, -9999.0).astype("float32")
    _write(os.path.join(ddir, "facc_clip.tif"), acc_out, profile)

    def _ginfo(g, rc):
        lat_g, lon_g = _cell_center(tr, rc)
        a_cells = float(acc2[rc[0] - r0, rc[1] - c0])
        return {**{k: g[k] for k in ("id", "name", "at_edge") if k in g},
                "lat": lat_g, "lon": lon_g,
                "area_inc": max(a_cells, 1.0) * _cell_area_km2(lat_g, tr.a)}

    info = {"dir": ddir, "cells_total": cells_total, "cells_kept": cells_kept,
            "kept_frac": cells_kept / cells_total,
            "speedup": cells_total / cells_kept,
            "outlet": _ginfo(outlet, out_rc),
            "cut": [_ginfo(g, rc) for g, rc in cut]}
    with open(meta_p, "w") as fh:
        json.dump(info, fh)
    return info


if __name__ == "__main__":
    from hf_data import basic, pipeline
    outlet = pipeline.gauge_info("08167000")          # Guadalupe @ Comfort, TX
    bbox = pipeline.basin_bbox(outlet)
    bdir = basic.store_dir(bbox)
    basic.clip_basic_data(bbox, bdir)
    bc = neighbors.upstream_gauges(outlet, bbox, bdir)
    info = build_speed_domain(outlet, bc, bdir)
    if not info:
        print("no speed domain possible")
    else:
        print(f"kept {info['cells_kept']}/{info['cells_total']} cells "
              f"({info['kept_frac']:.1%}) — ~{info['speedup']:.1f}x faster")
        print("outlet:", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in info["outlet"].items()})
        for c in info["cut"]:
            print("  cut:", c["id"], c.get("name", "")[:38],
                  f"inc {c['area_inc']:.1f} km2 edge={c.get('at_edge')}")

"""Basin-shaped 2-D simulation domains built from hydrologic units (HUC12).

User directive 2026-08-18: this is a hydrologic modelling platform — every
domain is a BASIN, never a rectangle. The hydrodynamic domain of a triggered
event is the entire catchment contributing to the trigger gauge, expressed as
the union of WBD HUC12 units (the same units the parallel tier decomposes
across devices/nodes, so ghost-cell exchange happens only along shared HUC
boundaries — ridgelines, except where the channel crosses).

Domain = HUC12 that contains the gauge (kept whole, so the floodplain just
downstream of the gauge is covered to that unit's pour point) + every HUC12
whose `tohuc` chain drains into it. When EF5's carved catchment is available
and trustworthy (its ring contains the gauge) it prunes chain units that do
not actually contribute (a tributary joining below the gauge inside the
outlet unit).

Products (written into the EF5 forcing dir so the queue bundle carries them,
and auto-discovered by crestimap EventSession/run_event):
  domain_huc.tif   int16 label raster at the requested DEM resolution on the
                   3DEP lattice: k >= 0 = index into `hucs`, -1 = outside.
                   The active mask is label >= 0; labels are the tiling units.
  domain.geojson   HUC12 boundaries + union outline (map overlay; published
                   with the event because it sits in out_dir too).

Data: WBD HUC12 Sept-2022 tables on HF dataset vincewin/CREST_data:
  wbd/huc12/index.parquet     (~4 MB; chain walking + bbox prefilter)
  wbd/huc12/geom_<huc4>.parquet (per-HUC4 WKB geometry, ~1-20 MB each)
Local override: HUC12_LOCAL_DIR (dev machines with the E:/HUC exports).
"""
from __future__ import annotations

import json
import math
import os
import threading

import numpy as np

REPO = os.environ.get("CREST_DATA_REPO", "vincewin/CREST_data")
WBD_PREFIX = "wbd/huc12"
LOCAL_DIR = os.environ.get("HUC12_LOCAL_DIR", "")
DEM_RES_DEG = {"1": 1.0 / 3600.0, "13": 1.0 / 10800.0}
# EF5 catchment pruning: an upstream chain unit stays when at least this
# fraction of its cells fall inside EF5's carved catchment
PRUNE_MIN_OVERLAP = float(os.environ.get("EVENT_HUC_PRUNE_OVERLAP", "0.3"))
MAX_CHAIN = int(os.environ.get("EVENT_HUC_MAX_CHAIN", "4000"))
_lock = threading.Lock()
_index = None
_geom_cache: dict = {}


class DomainError(RuntimeError):
    """No basin geometry could be built — the event must NOT fall back to a
    rectangle; the caller skips it and says why."""


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def _fetch(rel: str) -> str:
    if LOCAL_DIR:
        p = os.path.join(LOCAL_DIR, os.path.basename(rel))
        if os.path.exists(p):
            return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, f"{WBD_PREFIX}/{rel}", repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))


def index():
    """pyarrow-free dict-of-arrays index (numpy) — small, cached."""
    global _index
    with _lock:
        if _index is None:
            import pyarrow.parquet as pq
            t = pq.read_table(_fetch("index.parquet"))
            d = {c: t.column(c).to_numpy(zero_copy_only=False) for c in t.column_names}
            d["huc12"] = d["huc12"].astype(str)
            d["tohuc"] = d["tohuc"].astype(str)
            d["_pos"] = {h: i for i, h in enumerate(d["huc12"])}
            # downstream -> list of upstream positions (chain walking)
            up: dict = {}
            for i, to in enumerate(d["tohuc"]):
                up.setdefault(to, []).append(i)
            d["_up"] = up
            _index = d
        return _index


def geometries(huc12s) -> dict:
    """{huc12: shapely geometry} loading only the HUC4 partitions needed."""
    import pyarrow.parquet as pq
    from shapely import from_wkb
    out, need = {}, {}
    for h in huc12s:
        if h in _geom_cache:
            out[h] = _geom_cache[h]
        else:
            need.setdefault(h[:4], []).append(h)
    for h4, hs in need.items():
        t = pq.read_table(_fetch(f"geom_{h4}.parquet"))
        ids = t.column("huc12").to_pylist()
        wkb = t.column("geometry").to_pylist()
        want = set(hs)
        for i, w in zip(ids, wkb):
            if i in want:
                g = from_wkb(w)
                _geom_cache[i] = g
                out[i] = g
    missing = [h for h in huc12s if h not in out]
    if missing:
        raise DomainError(f"HUC12 geometry missing for {missing[:5]}")
    return out


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def gauge_huc12(lat: float, lon: float) -> str:
    """HUC12 containing the point (bbox prefilter on the index, then exact
    point-in-polygon on the few candidates)."""
    from shapely.geometry import Point
    d = index()
    m = ((d["minx"] <= lon) & (d["maxx"] >= lon)
         & (d["miny"] <= lat) & (d["maxy"] >= lat))
    cand = list(d["huc12"][m])
    if not cand:
        raise DomainError(f"no HUC12 bbox contains ({lat:.4f}, {lon:.4f}) — "
                          f"outside WBD coverage?")
    geoms = geometries(cand)
    p = Point(lon, lat)
    hits = [h for h in cand if geoms[h].contains(p)]
    if not hits:
        # on a boundary / tiny gap: nearest polygon
        hits = sorted(cand, key=lambda h: geoms[h].distance(p))[:1]
    return hits[0]


def upstream_chain(h0: str) -> list[dict]:
    """[{huc12, name, area_km2, depth, tohuc}] breadth-first from h0 (depth 0)."""
    d = index()
    pos = d["_pos"]
    if h0 not in pos:
        raise DomainError(f"HUC12 {h0} not in index")
    seen = {h0: 0}
    frontier = [h0]
    while frontier and len(seen) < MAX_CHAIN:
        nxt = []
        for h in frontier:
            for i in d["_up"].get(h, []):
                u = d["huc12"][i]
                if u not in seen:
                    seen[u] = seen[h] + 1
                    nxt.append(u)
        frontier = nxt
    out = []
    for h, dep in seen.items():
        i = pos[h]
        out.append({"huc12": h, "name": str(d["name"][i]),
                    "area_km2": round(float(d["areasqkm"][i]), 1),
                    "depth": int(dep), "tohuc": str(d["tohuc"][i])})
    out.sort(key=lambda r: (r["depth"], r["huc12"]))
    return out


# --------------------------------------------------------------------------- #
# EF5 catchment (pruning + fallback)
# --------------------------------------------------------------------------- #
def _ef5_catchment(ef5_dir: str):
    """(mask bool, transform, crs) of EF5's carved catchment from the valid-
    data mask of any gridded output, or None."""
    import glob
    import rasterio
    fs = []
    for kind in ("streamflow", "runoff", "subrunoff", "soilmoisture", "q"):
        fs = sorted(glob.glob(os.path.join(ef5_dir, f"{kind}.*.tif")))
        if fs:
            break
    if not fs:
        return None
    with rasterio.open(fs[-1]) as ds:
        a = ds.read(1)
        nod = ds.nodata
        tr, crs = ds.transform, ds.crs
    m = np.isfinite(a) & ((a != nod) if nod is not None else (a > -1.0))
    return (m, tr, crs) if m.any() else None


def _catchment_contains(cat, lat, lon, pad_cells=25) -> bool:
    m, tr, _ = cat
    col, row = ~tr * (lon, lat)
    r, c = int(row), int(col)
    r0, r1 = max(0, r - pad_cells), min(m.shape[0], r + pad_cells + 1)
    c0, c1 = max(0, c - pad_cells), min(m.shape[1], c + pad_cells + 1)
    return r0 < r1 and c0 < c1 and bool(m[r0:r1, c0:c1].any())


def _overlap_fraction(geom, cat) -> float:
    """fraction of the polygon's cells (on the EF5 grid) inside the catchment"""
    from rasterio import features
    m, tr, _ = cat
    r = features.rasterize([(geom, 1)], out_shape=m.shape, transform=tr,
                           fill=0, dtype="uint8", all_touched=False)
    n = int(r.sum())
    if n == 0:
        return 0.0
    return float((r.astype(bool) & m).sum()) / n


# --------------------------------------------------------------------------- #
# raster products
# --------------------------------------------------------------------------- #
def _snap_bbox(bounds, res: float, pad_cells: int = 2):
    """Union bounds -> bbox on the 3DEP lattice (integer multiples of the
    cell size from the degree origin) with a small pad, so the label raster
    and the fetched DEM share one grid."""
    w, s, e, n = bounds
    w = math.floor(w / res) * res - pad_cells * res
    s = math.floor(s / res) * res - pad_cells * res
    e = math.ceil(e / res) * res + pad_cells * res
    n = math.ceil(n / res) * res + pad_cells * res
    return (w, s, e, n)


def _fill_seams(labels: np.ndarray) -> np.ndarray:
    """Per-polygon simplification can leave 1-cell seams between neighbour
    units and the union must have NO interior holes (a basin has none):
    fill holes of the union, then give hole/seam cells the nearest label."""
    from scipy import ndimage
    act = labels >= 0
    filled = ndimage.binary_fill_holes(act)
    gap = filled & ~act
    if gap.any():
        _, (ri, ci) = ndimage.distance_transform_edt(~act, return_indices=True)
        labels = labels.copy()
        labels[gap] = labels[ri[gap], ci[gap]]
    return labels


def _ring_latlon(geom, max_pts=1500):
    """coarsened exterior ring(s) of a (Multi)Polygon as [[lat, lon], ...]
    — the largest part first (manifest 'basin' ring = index 0)."""
    parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    parts.sort(key=lambda p: p.area, reverse=True)
    rings = []
    for p in parts:
        xy = list(p.exterior.coords)
        step = max(1, len(xy) // max_pts)
        rings.append([[round(y, 4), round(x, 4)] for x, y in xy[::step]])
    return rings


# --------------------------------------------------------------------------- #
# public entry
# --------------------------------------------------------------------------- #
def basin_bounds(gauge: dict, pad_deg: float = 0.05) -> tuple:
    """(w, s, e, n) of the gauge's HUC12 basin (chain union) + pad — the
    raster EXTENT the hydrologic model (EF5) must cover so it can carve the
    whole catchment. EF5's old √area square centred on the outlet gauge
    clipped elongated basins (Spoon R: 3,884 of 4,241 km²; Pope Ck 199 of
    449) because a gauge sits at its basin's downstream end. Raises
    DomainError if the gauge has no HUC12 chain."""
    h0 = gauge_huc12(float(gauge["lat"]), float(gauge["lon"]))
    chain = upstream_chain(h0)
    d = index()
    pos = d["_pos"]
    ii = [pos[r["huc12"]] for r in chain]
    w = float(d["minx"][ii].min()) - pad_deg
    s = float(d["miny"][ii].min()) - pad_deg
    e = float(d["maxx"][ii].max()) + pad_deg
    n = float(d["maxy"][ii].max()) + pad_deg
    return (w, s, e, n)
def build_domain(gauge: dict, ef5_dir: str, dem_res: str = "1",
                 out_dir: str | None = None, max_cells: int | None = None,
                 log=print) -> dict:
    """Build the basin domain for a trigger gauge {id, lat, lon, area}.

    Writes domain_huc.tif (+ domain.geojson) into out_dir (default ef5_dir).
    max_cells: if given, DROP the most-upstream HUC12 units until the label
    raster's bbox fits (the CPU rung's hydrologic-unit trimming — never a
    box; the gauge's own unit is never dropped). Returns the domain summary
    dict (also what goes into the queue spec / manifest):
      {unit:"huc12", hucs:[...], n_hucs, area_km2, bbox:[w,s,e,n],
       n_active, n_bbox, dem_res, basin:[[lat,lon],...], trimmed:bool,
       source:"wbd"|"ef5"}
    Raises DomainError when no basin geometry exists (never a rectangle).
    """
    import rasterio
    from rasterio import features
    from shapely.ops import unary_union

    out_dir = out_dir or ef5_dir
    os.makedirs(out_dir, exist_ok=True)
    res = DEM_RES_DEG.get(str(dem_res), DEM_RES_DEG["1"])
    lat, lon = float(gauge["lat"]), float(gauge["lon"])
    cat = None
    try:
        cat = _ef5_catchment(ef5_dir)
    except Exception as e:
        log(f"domain: EF5 catchment unreadable ({type(e).__name__})")

    hucs, geoms, source = None, None, "wbd"
    usgs_area = gauge.get("area_km2") or gauge.get("area")
    try:
        h0 = gauge_huc12(lat, lon)
        chain = upstream_chain(h0)
        geoms = geometries([r["huc12"] for r in chain])
        chain_area = sum(r["area_km2"] for r in chain)
        ratio = (chain_area / float(usgs_area)) if usgs_area else None
        # The tohuc chain reproduces USGS drainage areas (Spoon R 4,240 vs
        # 4,241 km2; whole units, so the outlet unit adds its downstream
        # part). It is the authority. EF5's carved catchment is only allowed
        # to prune when the chain itself disagrees with the USGS area AND
        # EF5's catchment is complete — EF5's domain box has been observed
        # to CLIP elongated basins (Pope Ck: EF5 199 of 449 km2), so an
        # incomplete catchment must never trim the basin.
        if ratio is not None and 0.85 <= ratio <= 1.7:
            log(f"domain: HUC12 chain {chain_area:.0f} km2 vs USGS "
                f"{float(usgs_area):.0f} km2 (x{ratio:.2f}) — accepted")
        elif cat is not None and _catchment_contains(cat, lat, lon):
            m, tr, _ = cat
            cell_km2 = (abs(tr.a) * 111.0) * (abs(tr.e) * 111.0
                                              * math.cos(math.radians(lat)))
            ef5_area = float(m.sum()) * cell_km2
            complete = (usgs_area is None) or ef5_area >= 0.9 * float(usgs_area)
            if complete and _overlap_fraction(geoms[h0], cat) >= 0.05:
                keep = [r for r in chain if r["depth"] == 0 or
                        _overlap_fraction(geoms[r["huc12"]], cat)
                        >= PRUNE_MIN_OVERLAP]
                log(f"domain: chain {chain_area:.0f} km2 disagrees with USGS "
                    f"{float(usgs_area or 0):.0f} km2 — pruned "
                    f"{len(chain) - len(keep)} unit(s) by EF5's catchment "
                    f"({ef5_area:.0f} km2)")
                chain = keep
            else:
                log(f"domain: chain {chain_area:.0f} km2 vs USGS "
                    f"{float(usgs_area or 0):.0f} km2 but EF5 catchment "
                    f"({ef5_area:.0f} km2) is incomplete/misplaced — chain "
                    f"kept whole")
        else:
            log(f"domain: chain {chain_area:.0f} km2 (USGS "
                f"{usgs_area}) — kept whole")
        hucs = chain
    except DomainError as e:
        log(f"domain: HUC12 lookup failed ({e})")
    except Exception as e:
        log(f"domain: HUC12 lookup failed ({type(e).__name__}: {e})")

    if hucs is None:
        # still a basin: EF5's own carved catchment as a single unit
        if cat is None:
            raise DomainError("no HUC12 units and no EF5 catchment — cannot "
                              "build a basin domain (a rectangle is not an "
                              "option)")
        source = "ef5"
        m, tr, crs = cat
        shp = [g for g, v in features.shapes(m.astype("uint8"), transform=tr)
               if v == 1]
        from shapely.geometry import shape
        geom_u = unary_union([shape(s) for s in shp])
        hucs = [{"huc12": None, "name": f"EF5 catchment {gauge.get('id')}",
                 "area_km2": None, "depth": 0, "tohuc": None}]
        geoms = {None: geom_u}
        log("domain: EF5 carved catchment used as the (single) unit")

    # ---- hydrologic-unit trimming for a cell budget (CPU rung) ------------- #
    trimmed = False
    order = sorted(hucs, key=lambda r: (-r["depth"], r["huc12"] or ""))
    while True:
        u = unary_union([geoms[r["huc12"]] for r in hucs])
        bbox = _snap_bbox(u.bounds, res)
        nx = int(round((bbox[2] - bbox[0]) / res))
        ny = int(round((bbox[3] - bbox[1]) / res))
        if not max_cells or nx * ny <= max_cells or len(hucs) <= 1:
            break
        drop = order.pop(0)          # most-upstream first; gauge unit last
        hucs = [r for r in hucs if r is not drop]
        trimmed = True
    if max_cells and nx * ny > max_cells:
        raise DomainError(f"even the gauge's own unit ({nx * ny / 1e6:.1f}M "
                          f"cells) exceeds the {max_cells / 1e6:.1f}M cell "
                          f"budget at this resolution")
    if trimmed:
        log(f"domain: trimmed to {len(hucs)} unit(s) nearest the gauge to fit "
            f"{max_cells / 1e3:.0f}k cells (hydrologic units, not a window)")

    # ---- label raster on the 3DEP lattice ---------------------------------- #
    tr = rasterio.transform.from_origin(bbox[0], bbox[3], res, res)
    shapes = [(geoms[r["huc12"]], k) for k, r in enumerate(hucs)]
    labels = features.rasterize(shapes, out_shape=(ny, nx), transform=tr,
                                fill=-1, dtype="int16", all_touched=False)
    labels = _fill_seams(labels)
    n_active = int((labels >= 0).sum())
    if n_active == 0:
        raise DomainError("basin domain rasterized to zero cells")
    with rasterio.open(os.path.join(out_dir, "domain_huc.tif"), "w",
                       driver="GTiff", height=ny, width=nx, count=1,
                       dtype="int16", crs="EPSG:4326", transform=tr,
                       nodata=-1, compress="deflate", tiled=True) as ds:
        ds.write(labels, 1)

    # ---- geojson (units + union outline) for the map ------------------------ #
    from shapely.geometry import mapping
    from shapely import simplify
    feats = []
    for k, r in enumerate(hucs):
        g = simplify(geoms[r["huc12"]], 0.0003)     # ~30 m for display
        feats.append({"type": "Feature", "geometry": mapping(g),
                      "properties": {"k": k, "huc12": r["huc12"],
                                     "name": r["name"],
                                     "area_km2": r["area_km2"],
                                     "depth": r["depth"]}})
    outline = simplify(u, 0.0003)
    gj = {"type": "FeatureCollection", "features": feats,
          "union": mapping(outline)}
    with open(os.path.join(out_dir, "domain.geojson"), "w") as fp:
        json.dump(gj, fp)

    rings = _ring_latlon(u)
    area = sum((r["area_km2"] or 0.0) for r in hucs) or None
    summ = {"unit": "huc12" if source == "wbd" else "ef5_catchment",
            "source": source,
            "hucs": [{k: r[k] for k in ("huc12", "name", "area_km2", "depth")}
                     for r in hucs],
            "n_hucs": len(hucs), "area_km2": round(area, 1) if area else None,
            "bbox": [round(v, 6) for v in bbox], "n_active": n_active,
            "n_bbox": int(nx * ny), "dem_res": str(dem_res),
            "basin": rings[0], "trimmed": trimmed}
    log(f"domain: {len(hucs)} {summ['unit']} unit(s), "
        f"{(area or 0):.0f} km2, {n_active / 1e6:.2f}M active of "
        f"{nx * ny / 1e6:.2f}M bbox cells at {res * 111000:.0f} m "
        f"({100.0 * n_active / (nx * ny):.0f}% of the box is basin)")
    return summ

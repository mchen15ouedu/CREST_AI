"""Offline build of hf_data/data/basin_extent.parquet: the EF5 raster extent,
outlet snap and catchment facts of every simulation outlet (PI rule
2026-10-23: "No box is allowed" - every EF5 domain comes from basin geometry).

Outlets:
  * USGS gauges (CREST_data gauges/gagesII_9322.parquet) - outline = the
    GAGES-II basin polygon (D:/usgs_gauge_boundaries, matches DRAIN_SQKM).
  * HydroBASINS level-07 pour points (CREST_data gauges/virtual_points.parquet)
    - outline = the pour point's sub-basin plus every sub-basin upstream
    (hybas_na_lev07_v1c NEXT_DOWN tree), used only when its UP_AREA matches the
    point's own drainage area within OUTLINE_TOL; otherwise the point sits on a
    neighbouring reach (coastal aggregates, confluences) and its outline is its
    own D8 catchment.

For every outlet the builder also fixes the outlet cell once (M1):
basinextent.snap_cell - the nearest stream cell within 20 km whose HydroSHEDS
3-arc-second drainage area matches the outlet's own within +-20%. The run
hands EF5 that cell centre WITHOUT a BASINAREA (EF5 uses BASINAREA only for
its own outlet search, BasicGrids.cpp:697), so EF5 carves exactly the D8
catchment measured here. The extent is the union of that catchment and the
outline, plus MARGIN_DEG, snapped outward to the HydroSHEDS lattice - the
tightest rectangle EF5's raster reader needs; EF5 simulates only the carved
catchment inside it.

Outlets without terrain (AK/HI/PR, outside the HydroSHEDS NA tiles), without a
matching stream cell (snap_mismatch) or above CAP_CELLS are flagged; the
runtime skips them with the reason - never a box, never a hillslope cell.

    python scripts/build_basin_extent.py [--workers 10] [--sample 50] [--ids a,b]

Inputs default to the PI's laptop paths; see the argparse block.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "hf_data", "data", "basin_extent.parquet")
MARGIN_DEG = 0.05            # ~5 km around the basin (raster edge never touches it)
CAP_CELLS = 36_000_000       # largest raster the old 5x5 deg square could produce
OUTLINE_TOL = 0.20           # HydroBASINS UP_AREA vs the point's own area
SNAP_TOL = 0.20              # carved area vs reference area -> snap_mismatch
WIN_PAD_DEG = 0.30           # first read window around the outline / point
GROW_DEG = 1.0               # grow the window while the carve touches its edge
MAX_WIN_CELLS = 160_000_000  # stop growing: the catchment is over the cap anyway
GAGES_GLOB = "D:/usgs_gauge_boundaries/boundaries-shapefiles-by-aggeco/*.shp"
HYBAS_SHP = r"E:/USGSproject/hybas_na_lev07_v1c.shp"
DIR_COG = "E:/hydrosheds/cog/na_dir_3s.tif"
ACC_COG = "E:/hydrosheds/cog/na_acc_3s.tif"


# --------------------------------------------------------------------------- #
# outlines
# --------------------------------------------------------------------------- #
def gages_outlines(pattern: str) -> pd.DataFrame:
    """GAGES-II polygon bounds (EPSG:4326) + area per gauge id."""
    import geopandas as gpd
    import shapely
    parts = [gpd.read_file(f) for f in sorted(glob.glob(pattern))]
    if not parts:
        sys.exit(f"no GAGES-II shapefiles at {pattern}")
    P = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    P["GAGE_ID"] = P["GAGE_ID"].astype(str).str.zfill(8)
    if P.GAGE_ID.duplicated().any():
        P = P.dissolve(by="GAGE_ID", as_index=False)
    P["geometry"] = shapely.make_valid(P.geometry.values)
    km2 = P.geometry.area / 1e6                      # equal-area projection
    b = P.to_crs(4326).geometry.bounds
    return pd.DataFrame({"id": P.GAGE_ID.values, "outline_km2": km2.values,
                         "ow": b.minx.values, "os": b.miny.values,
                         "oe": b.maxx.values, "on": b.maxy.values}).set_index("id")


def hybas_outlines(shp: str, vp: pd.DataFrame) -> pd.DataFrame:
    """Upstream-union bounds + UP_AREA of each pour point's level-07 sub-basin."""
    import pyogrio
    h = pyogrio.read_dataframe(shp, columns=["HYBAS_ID", "NEXT_DOWN", "UP_AREA"])
    b = h.geometry.bounds
    H = pd.DataFrame({"nd": h.NEXT_DOWN.values, "up": h.UP_AREA.values,
                      "w": b.minx.values, "s": b.miny.values,
                      "e": b.maxx.values, "n": b.maxy.values},
                     index=h.HYBAS_ID.values)
    kids: dict[int, list[int]] = {}
    for i, nd in zip(H.index.values, H.nd.values):
        kids.setdefault(int(nd), []).append(int(i))
    rows = {}
    for r in vp.itertuples():
        hid = int(r.hybas_id)
        if hid not in H.index:
            continue
        stack, ids = [hid], []
        while stack:
            x = stack.pop()
            ids.append(x)
            stack.extend(kids.get(x, []))
        sub = H.loc[ids]
        rows[r.vp] = {"outline_km2": float(H.at[hid, "up"]),
                      "ow": sub.w.min(), "os": sub.s.min(),
                      "oe": sub.e.max(), "on": sub.n.max()}
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
# per-outlet carve (worker process)
# --------------------------------------------------------------------------- #
_DS = {}


def _ds():
    if not _DS:
        import rasterio
        _DS["dir"] = rasterio.open(DIR_COG)
        _DS["acc"] = rasterio.open(ACC_COG)
    return _DS["dir"], _DS["acc"]


def _window(ds, w, s, e, n):
    """Lattice-aligned window (row0, col0, nrows, ncols) covering (w,s,e,n)."""
    tr = ds.transform
    c0 = int(math.floor((w - tr.c) / tr.a + 1e-9))
    c1 = int(math.ceil((e - tr.c) / tr.a - 1e-9))
    r0 = int(math.floor((tr.f - n) / -tr.e + 1e-9))
    r1 = int(math.ceil((tr.f - s) / -tr.e - 1e-9))
    return r0, c0, r1 - r0, c1 - c0


def _snap(lat, lon, area):
    """basinextent.snap_cell (the M1 rule: nearest stream cell within 20 km
    whose drainage area matches within +-20%) on a window around the point.
    Returns (row, col, acc) in global indices, "no_terrain" (no HydroSHEDS
    data there) or "snap_mismatch" (no cell matches)."""
    from rasterio.windows import Window
    from hf_data.basinextent import SNAP_RADIUS_KM, snap_cell
    _, dsa = _ds()
    tr = dsa.transform
    hr = int(SNAP_RADIUS_KM / (tr.a * 110.574)) + 4
    hc = int(SNAP_RADIUS_KM / (tr.a * 111.32 * max(math.cos(math.radians(lat)), 0.2))) + 4
    r = int((tr.f - lat) / -tr.e)
    c = int((lon - tr.c) / tr.a)
    r0, c0 = max(0, r - hr), max(0, c - hc)
    r1, c1 = min(dsa.height, r + hr + 1), min(dsa.width, c + hc + 1)
    if r1 <= r0 or c1 <= c0:
        return "no_terrain"
    win = Window(c0, r0, c1 - c0, r1 - r0)
    acc = dsa.read(1, window=win).astype("float64")
    acc[acc == float(dsa.nodata)] = np.nan
    acc[acc < 0] = np.nan
    if not np.isfinite(acc).any():
        return "no_terrain"
    rc = snap_cell(acc, dsa.window_transform(win), lat, lon, area)
    if rc is None:
        return "snap_mismatch"
    return rc[0] + r0, rc[1] + c0, float(acc[rc])


def _lattice_cells(w, s, e, n, res):
    return (math.ceil((e - w) / res) + 1) * (math.ceil((n - s) / res) + 1)


def _carve(gr, gc, w, s, e, n):
    """Upstream D8 catchment of global cell (gr, gc) inside the lattice window
    covering (w,s,e,n). Returns (row0, col0, nrows, ncols, mask) or None."""
    from rasterio.windows import Window
    dsd, _ = _ds()
    r0, c0, H, W = _window(dsd, w, s, e, n)
    r0, c0 = max(0, r0), max(0, c0)
    H = min(H, dsd.height - r0)
    W = min(W, dsd.width - c0)
    if not (r0 <= gr < r0 + H and c0 <= gc < c0 + W):
        return None
    from hf_data.basinextent import upstream_mask     # the runtime's own carve
    d = dsd.read(1, window=Window(c0, r0, W, H))
    return r0, c0, H, W, upstream_mask(d, gr - r0, gc - c0)


def process(site: dict) -> dict:
    """Snap + carve one outlet; never raises (errors become flags)."""
    t0 = time.time()
    out = {"id": site["id"], "kind": site["kind"], "flags": [], "secs": 0.0}
    try:
        dsd, dsa = _ds()
        res = dsa.transform.a
        sn = _snap(site["lat"], site["lon"], site["area"])
        if isinstance(sn, str):
            out["flags"].append(sn)                  # no_terrain | snap_mismatch
            return out
        gr, gc, acc_cells = sn
        tr = dsa.transform
        slat = tr.f + (gr + 0.5) * tr.e
        slon = tr.c + (gc + 0.5) * tr.a
        from hf_data.basinextent import cell_km2
        out.update(snap_lat=slat, snap_lon=slon, snap_row=int(gr), snap_col=int(gc),
                   snap_area_km2=max(acc_cells, 1.0) * float(cell_km2(slat, res)))
        # over the cap before any carve: the raster holds every carved cell,
        # and it holds the outline's extent
        has_outline = site.get("ow") is not None and np.isfinite(site["ow"])
        if acc_cells > CAP_CELLS or (has_outline and _lattice_cells(
                site["ow"] - MARGIN_DEG, site["os"] - MARGIN_DEG,
                site["oe"] + MARGIN_DEG, site["on"] + MARGIN_DEG, res) > CAP_CELLS):
            out["flags"].append("over_cap")
            return out
        if has_outline:
            w, s, e, n = site["ow"], site["os"], site["oe"], site["on"]
        else:
            rad = max(0.3, 1.2 * math.sqrt(max(site["area"], 1.0)) / 111.0)
            w, s, e, n = slon - rad, slat - rad, slon + rad, slat + rad
        w, s, e, n = (min(w, slon) - WIN_PAD_DEG, min(s, slat) - WIN_PAD_DEG,
                      max(e, slon) + WIN_PAD_DEG, max(n, slat) + WIN_PAD_DEG)
        while True:
            cv = _carve(gr, gc, w, s, e, n)
            if cv is None:
                out["flags"].append("no_terrain")
                return out
            r0, c0, H, W, m = cv
            rr = np.nonzero(m.any(axis=1))[0]
            cc = np.nonzero(m.any(axis=0))[0]
            hit = {"n": rr[0] == 0 and r0 > 0, "s": rr[-1] == H - 1 and r0 + H < dsd.height,
                   "w": cc[0] == 0 and c0 > 0, "e": cc[-1] == W - 1 and c0 + W < dsd.width}
            if not any(hit.values()):
                break
            # grow only the sides the catchment reaches; check the size first
            w2 = w - GROW_DEG * hit["w"]
            s2 = s - GROW_DEG * hit["s"]
            e2 = e + GROW_DEG * hit["e"]
            n2 = n + GROW_DEG * hit["n"]
            if _lattice_cells(w2, s2, e2, n2, res) > MAX_WIN_CELLS:
                out["flags"].append("over_cap")
                return out
            w, s, e, n = w2, s2, e2, n2
        if rr[0] == 0 or cc[0] == 0 or rr[-1] == H - 1 or cc[-1] == W - 1:
            out["flags"].append("carve_hits_terrain_edge")
        lat_rows = tr.f + (r0 + rr + 0.5) * tr.e
        ncell_row = m[rr].sum(axis=1)
        out.update(
            carved_cells=int(m.sum()),
            carved_km2=float(np.sum(ncell_row * (res * 111.32) ** 2
                                    * np.cos(np.radians(lat_rows)))),
            cw=tr.c + (c0 + cc[0]) * tr.a, ce=tr.c + (c0 + cc[-1] + 1) * tr.a,
            cn=tr.f + (r0 + rr[0]) * tr.e, cs=tr.f + (r0 + rr[-1] + 1) * tr.e)
        if has_outline:                  # QA: share of the catchment inside the outline box
            rlat = tr.f + (r0 + np.arange(H) + 0.5) * tr.e
            clon = tr.c + (c0 + np.arange(W) + 0.5) * tr.a
            rin = (rlat >= site["os"]) & (rlat <= site["on"])
            cin = (clon >= site["ow"]) & (clon <= site["oe"])
            out["carve_in_outline"] = float(m[np.ix_(rin, cin)].sum()) / max(int(m.sum()), 1)
    except MemoryError:
        out["flags"].append("error")
        out["error"] = "MemoryError"
    except Exception as ex:                          # recorded, reviewed in QA
        out["flags"].append("error")
        out["error"] = f"{type(ex).__name__}: {ex}"
    finally:
        out["secs"] = round(time.time() - t0, 2)
    return out


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
def lattice_out(w, s, e, n, tr):
    """Snap bounds outward to the HydroSHEDS lattice -> (w,s,e,n, nrows, ncols)."""
    c0 = math.floor((w - tr.c) / tr.a + 1e-9)
    c1 = math.ceil((e - tr.c) / tr.a - 1e-9)
    r0 = math.floor((tr.f - n) / -tr.e + 1e-9)
    r1 = math.ceil((tr.f - s) / -tr.e - 1e-9)
    return (tr.c + c0 * tr.a, tr.f + r1 * tr.e, tr.c + c1 * tr.a, tr.f + r0 * tr.e,
            r1 - r0, c1 - c0)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _fingerprint(paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--sample", type=int, default=0, help="random N outlets (QA run)")
    ap.add_argument("--ids", default="", help="comma list of outlet ids")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    try:                       # OU network TLS inspection: use the OS trust store
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import rasterio
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    cat_p = hf_hub_download("vincewin/CREST_data", "gauges/gagesII_9322.parquet",
                            repo_type="dataset")
    vp_p = hf_hub_download("vincewin/CREST_data", "gauges/virtual_points.parquet",
                           repo_type="dataset")
    cat = pd.read_parquet(cat_p)
    cat["STAID"] = cat.STAID.astype(str).str.zfill(8)
    vp = pd.read_parquet(vp_p)
    G = gages_outlines(GAGES_GLOB)
    V = hybas_outlines(HYBAS_SHP, vp)
    print(f"outlines: {len(G)} GAGES-II, {len(V)} HydroBASINS ({time.time() - t0:.0f}s)",
          flush=True)

    sites = []
    for r in cat.itertuples():
        o = G.loc[r.STAID] if r.STAID in G.index else None
        sites.append({"id": r.STAID, "kind": "g", "lat": float(r.LAT_GAGE),
                      "lon": float(r.LNG_GAGE), "area": float(r.DRAIN_SQKM),
                      "outline_src": "gages2" if o is not None else "none",
                      **({k: float(o[k]) for k in ("ow", "os", "oe", "on", "outline_km2")}
                         if o is not None else {})})
    for r in vp.itertuples():
        site = {"id": r.vp, "kind": "v", "lat": float(r.lat), "lon": float(r.lon),
                "area": float(r.area_km2), "hybas_id": int(r.hybas_id),
                "outline_src": "none"}
        if r.vp in V.index:
            o = V.loc[r.vp]
            ratio = float(o.outline_km2) / max(float(r.area_km2), 1e-6)
            site["hybas_ratio"] = ratio
            if abs(ratio - 1.0) <= OUTLINE_TOL:
                site.update(outline_src="hybas",
                            **{k: float(o[k]) for k in ("ow", "os", "oe", "on", "outline_km2")})
        sites.append(site)
    if args.ids:
        want = {s.strip() for s in args.ids.split(",")}
        sites = [s for s in sites if s["id"] in want]
    elif args.sample:
        rng = np.random.default_rng(23)
        sites = [sites[i] for i in rng.choice(len(sites), args.sample, replace=False)]
    # interleave big and small outlets: big-first would hand every worker a
    # multi-GB carve at once
    sites.sort(key=lambda s: -s["area"])
    order = []
    i, j = 0, len(sites) - 1
    while i <= j:
        order.append(sites[i])
        if i != j:
            order.append(sites[j])
        i, j = i + 1, j - 1
    sites = order
    print(f"{len(sites)} outlets, {args.workers} workers", flush=True)

    res = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process, sites, chunksize=1), 1):
                res.append(r)
                if i % 200 == 0 or i == len(sites):
                    print(f"  {i}/{len(sites)} ({time.time() - t0:.0f}s)", flush=True)
        # a worker that ran out of memory beside others gets one serial retry
        retry = {r["id"] for r in res if r.get("error") == "MemoryError"}
        if retry:
            print(f"  retrying {len(retry)} outlet(s) serially after MemoryError", flush=True)
            res = [r for r in res if r["id"] not in retry] + \
                  [process(s) for s in sites if s["id"] in retry]
    else:
        for s in sites:
            res.append(process(s))
    R = pd.DataFrame(res).set_index("id")
    S = pd.DataFrame(sites).set_index("id")
    T = S.join(R.drop(columns=["kind"]), how="left")

    with rasterio.open(DIR_COG) as ds:
        tr = ds.transform
    ext = []
    for r in T.itertuples():
        flags = list(r.flags) if isinstance(r.flags, (list, np.ndarray)) else []
        boxes = []
        if isinstance(getattr(r, "cw", None), float) and np.isfinite(r.cw):
            boxes.append((r.cw, r.cs, r.ce, r.cn))
        if r.outline_src != "none" and np.isfinite(r.ow):
            boxes.append((r.ow, r.os, r.oe, r.on))
        if {"no_terrain", "error", "snap_mismatch"} & set(flags) or not boxes:
            ext.append((np.nan,) * 6 + (flags,))
            continue
        w = min(b[0] for b in boxes) - MARGIN_DEG
        s = min(b[1] for b in boxes) - MARGIN_DEG
        e = max(b[2] for b in boxes) + MARGIN_DEG
        n = max(b[3] for b in boxes) + MARGIN_DEG
        lw, ls, le, ln, nr, nc = lattice_out(w, s, e, n, tr)
        if nr * nc > CAP_CELLS and "over_cap" not in flags:
            flags.append("over_cap")
        ref = r.area if r.area > 0 else np.nan
        if np.isfinite(getattr(r, "carved_km2", np.nan)) and np.isfinite(ref) \
                and abs(r.carved_km2 / ref - 1.0) > SNAP_TOL:
            flags.append("snap_mismatch")            # the carve is not this basin
        ext.append((lw, ls, le, ln, nr * nc, None, flags))
    E = pd.DataFrame(ext, index=T.index,
                     columns=["w", "s", "e", "n", "cells", "_", "flags"]).drop(columns="_")
    T = T.drop(columns=["flags"]).join(E)
    T["source"] = np.where(T.outline_src == "gages2", "gages2",
                           np.where(T.outline_src == "hybas", "hybas", "d8"))
    T["status"] = "ok"
    for bad in ("error", "no_terrain", "snap_mismatch", "over_cap"):
        T.loc[T["flags"].apply(lambda f, b=bad: b in f) & (T.status == "ok"), "status"] = bad
    T["flags"] = T["flags"].apply(lambda f: ",".join(f))
    keep = ["kind", "status", "source", "flags", "lat", "lon", "area", "snap_lat", "snap_lon",
            "snap_area_km2", "carved_cells", "carved_km2", "w", "s", "e", "n", "cells",
            "cw", "cs", "ce", "cn", "ow", "os", "oe", "on", "outline_km2", "carve_in_outline",
            "hybas_id", "hybas_ratio", "secs", "error"]
    T = T[[c for c in keep if c in T.columns]].reset_index()
    for c in ("carved_cells", "cells", "hybas_id"):
        if c in T:
            T[c] = T[c].astype("Int64")

    import pyarrow as pa
    import pyarrow.parquet as pq
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    body = T.to_csv(index=False).encode()
    meta = {"table_version": f"{built}-{hashlib.sha1(body).hexdigest()[:10]}",
            "built": built, "builder_sha": _git_sha(),
            "inputs": _fingerprint([cat_p, vp_p, HYBAS_SHP, DIR_COG, ACC_COG]
                                   + sorted(glob.glob(GAGES_GLOB))),
            "margin_deg": MARGIN_DEG, "cap_cells": CAP_CELLS,
            "outline_tol": OUTLINE_TOL, "snap_tol": SNAP_TOL,
            "lattice": [tr.c, tr.f, tr.a], "partial": bool(args.sample or args.ids)}
    tbl = pa.Table.from_pandas(T, preserve_index=False)
    tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}),
                                       b"basin_extent": json.dumps(meta).encode()})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pq.write_table(tbl, args.out, compression="zstd")
    print(f"wrote {args.out}: {len(T)} rows, version {meta['table_version']} "
          f"({time.time() - t0:.0f}s)")
    print(T.groupby(["kind", "status"]).size().to_string())
    print("flags:", T["flags"].str.split(",").explode().replace("", np.nan).dropna()
          .value_counts().to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

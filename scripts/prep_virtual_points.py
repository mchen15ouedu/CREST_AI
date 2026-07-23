"""One-time prep: HydroBASINS pour points -> CONUS virtual-point catalog.

Virtual points fill the gaps between USGS gauges so every part of CONUS has a
simulatable hindcast outlet. Pfafstetter level 07 pour points (one per
sub-basin outlet, ~1,800 km2 nominal sub-basins) give seamless coverage at
~1/3 the density of the gauge network; points that duplicate an existing
gauge (same HydroRIVERS reach, or within GAUGE_DUP_KM) are dropped because
the gauge is strictly better there (it has observations).

Each point is snapped to its HydroRIVERS reach to inherit the modeled
drainage area (UPLAND_SKM) — EF5 uses that as BASINAREA to lock the outlet
onto the right flow-accumulation cell. Coastal aggregate sub-basins carry one
pour point per outlet; points below MIN_UPLAND_KM2 (beach-scale watersheds
the MRMS+HydroSHEDS setup can't resolve) are dropped, and only the nearest
point per reach survives.

Builds gauges/virtual_points.parquet in vincewin/CREST_data, consumed by
hf_data/virtualpoints.py. Ids are "V" + HYRIV_ID (unique after the per-reach
dedupe, and can never collide with 8-digit USGS ids).

    python scripts/prep_virtual_points.py [--level 07] [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import HF_REPO, hf_token                             # noqa: E402

POUR_DIR = r"E:\hydroZone\pour_points"
OUT_PATH = "gauges/virtual_points.parquet"
BBOX = (-125.0, 24.0, -66.5, 50.0)          # CONUS
MIN_UPLAND_KM2 = 100.0
GAUGE_DUP_KM = 2.0
SNAP_RADIUS_DEG = 0.05
KM_PER_DEG = 111.0
NATION_URL = ("https://www2.census.gov/geo/tiger/GENZ2023/shp/"
              "cb_2023_us_nation_5m.zip")


def conus_polygon(cache_dir: str):
    """Census national boundary clipped to the CONUS bbox (drops AK/HI/PR)."""
    import requests
    import pyogrio
    from shapely.geometry import box
    shp = os.path.join(cache_dir, "cb_nation", "cb_2023_us_nation_5m.shp")
    if not os.path.exists(shp):
        os.makedirs(os.path.dirname(shp), exist_ok=True)
        r = requests.get(NATION_URL, timeout=120)
        r.raise_for_status()
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(os.path.dirname(shp))
    nat = pyogrio.read_dataframe(shp)
    return nat.geometry.iloc[0].intersection(box(*BBOX))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="07")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    token = hf_token()

    import pyogrio
    from shapely.prepared import prep
    from scipy.spatial import cKDTree
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "_cache")
    conus = prep(conus_polygon(cache))

    shp = os.path.join(POUR_DIR, f"hybas_pour_lev{args.level}_v1.shp")
    print("reading", shp, "…", flush=True)
    df = pyogrio.read_dataframe(shp, bbox=BBOX)
    inside = np.fromiter((conus.contains(g) for g in df.geometry), bool, len(df))
    df = df[inside]
    px = df.geometry.x.to_numpy("float64")
    py = df.geometry.y.to_numpy("float64")
    hyb = df.HYBAS_ID.to_numpy("int64")
    print(f"  {inside.sum():,} pour points inside CONUS (of {len(inside):,} in bbox)")

    # -- snap to HydroRIVERS reaches (drainage area + dedupe key) --------------
    riv = pq.read_table(hf_hub_download(HF_REPO, "rivers/hydrorivers_na.parquet",
                                        repo_type="dataset", token=token))
    rid = riv.column("id").to_numpy()
    rup = riv.column("upland").to_numpy()
    rord = riv.column("ord").to_numpy()
    lats_l = riv.column("lats").to_pylist()
    lons_l = riv.column("lons").to_pylist()
    flat_lon = np.concatenate([np.asarray(a, "float64") for a in lons_l])
    flat_lat = np.concatenate([np.asarray(a, "float64") for a in lats_l])
    owner = np.repeat(np.arange(len(rid)), [len(a) for a in lats_l])
    tree = cKDTree(np.column_stack([flat_lon, flat_lat]))
    dists, vidx = tree.query(np.column_stack([px, py]), k=1,
                             distance_upper_bound=SNAP_RADIUS_DEG, workers=-1)
    ok = np.isfinite(dists)
    reach_i = np.where(ok, owner[np.clip(vidx, 0, len(owner) - 1)], -1)
    up = np.where(ok, rup[np.clip(reach_i, 0, len(rup) - 1)], 0.0)
    keep = ok & (up >= MIN_UPLAND_KM2)
    print(f"  {keep.sum():,} snapped to a reach with >= {MIN_UPLAND_KM2:.0f} km² upland")

    # -- drop duplicates: same reach (keep nearest), gauged reach, near-gauge --
    cat = pq.read_table(hf_hub_download(HF_REPO, "gauges/gagesII_9322.parquet",
                                        repo_type="dataset", token=token))
    glat = cat.column("LAT_GAGE").to_numpy().astype("float64")
    glon = cat.column("LNG_GAGE").to_numpy().astype("float64")
    gid = np.array([str(s).zfill(8) for s in cat.column("STAID").to_pylist()])
    gtree = cKDTree(np.column_stack([glon, glat]))
    snap = pq.read_table(hf_hub_download(HF_REPO, "rivers/gauge_reach.parquet",
                                         repo_type="dataset", token=token))
    gauged_reaches = set(snap.column("reach_id").to_pylist())

    order = np.argsort(dists[keep])                  # nearest snap wins its reach
    cand = np.where(keep)[0][order]
    seen: set[int] = set()
    sel: list[int] = []
    n_dup_reach = n_gauged = 0
    for i in cand:
        ri = int(reach_i[i])
        if ri in seen:
            n_dup_reach += 1
            continue
        seen.add(ri)
        if int(rid[ri]) in gauged_reaches:
            n_gauged += 1                            # a USGS gauge owns this reach
            continue
        sel.append(i)
    sel = np.array(sel)
    gd, gi = gtree.query(np.column_stack([px[sel], py[sel]]), workers=-1)
    gd_km = gd * KM_PER_DEG
    far = gd_km >= GAUGE_DUP_KM
    print(f"  dropped: {n_dup_reach:,} same-reach dups, {n_gauged:,} on gauged "
          f"reaches, {(~far).sum():,} within {GAUGE_DUP_KM:.0f} km of a gauge")
    sel, gd_km, gi = sel[far], gd_km[far], gi[far]
    print(f"  -> {len(sel):,} virtual points")

    # -- gauge-local timezone for the hydrograph axis --------------------------
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder(in_memory=True)
    tzs = [tf.timezone_at(lng=float(px[i]), lat=float(py[i])) or "America/Chicago"
           for i in sel]

    ri = reach_i[sel]
    tbl = pa.table({
        "vp": [f"V{int(rid[r])}" for r in ri],
        "lat": py[sel],
        "lon": px[sel],
        "area_km2": rup[ri].astype("float32"),
        "reach_id": rid[ri].astype("int64"),
        "hybas_id": hyb[sel],
        "ord": rord[ri].astype("int8"),
        "near_gid": [gid[j] for j in gi],
        "near_km": gd_km.astype("float32"),
        "tz": tzs,
        "level": np.full(len(sel), int(args.level), "int8"),
    })
    a = rup[ri]
    print(f"  area km²: p10={np.percentile(a, 10):.0f} med={np.median(a):.0f} "
          f"p90={np.percentile(a, 90):.0f} max={a.max():.0f}")

    if args.dry_run:
        print("[dry-run: no upload]")
        return 0
    tmp = os.path.join(tempfile.mkdtemp(), "virtual_points.parquet")
    pq.write_table(tbl, tmp, compression="zstd")
    print("parquet:", round(os.path.getsize(tmp) / 1e6, 2), "MB")
    api = HfApi(token=token)
    api.create_commit(repo_id=HF_REPO, repo_type="dataset",
                      operations=[CommitOperationAdd(OUT_PATH, tmp)],
                      commit_message=f"virtual points lev{args.level}: "
                                     f"{len(sel):,} CONUS pour points")
    print("uploaded", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

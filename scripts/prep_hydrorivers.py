"""One-time prep: HydroRIVERS NA -> compact upstream-network store for CREST-AI.

Builds two parquets in vincewin/CREST_data (dataset repo), consumed by
hf_data/rivernet.py to draw the river network upstream of a nowcast gauge:

  rivers/hydrorivers_na.parquet   one row per reach — id, next (downstream
                                  HYRIV_ID, 0 = outlet), ord (Strahler),
                                  upland (km2), lats/lons (list<float32>).
  rivers/gauge_reach.parquet      gid -> snapped HYRIV_ID. Snap = candidate
                                  reach vertices within ~8 km, scored by
                                  distance + |log10(upland/gauge drainage)|
                                  so a gauge on a big river doesn't snap to
                                  the small tributary that happens to pass
                                  slightly closer.

HydroRIVERS is 15-arcsec (same HydroSHEDS family as the basic/ COGs); the
NEXT_DOWN topology makes "upstream of reach X" a pure graph walk at serve
time — no raster work.

    python scripts/prep_hydrorivers.py [--shp E:/hydroZone/...] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import HF_REPO, hf_token                             # noqa: E402

SHP = r"E:\hydroZone\HydroRIVERS_v10_na_shp\HydroRIVERS_v10_na.shp"
RIVERS_PATH = "rivers/hydrorivers_na.parquet"
SNAP_PATH = "rivers/gauge_reach.parquet"
SNAP_RADIUS_DEG = 0.08          # ~8 km candidate search
KM_PER_DEG = 111.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp", default=SHP)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    token = hf_token()

    import pyogrio
    print("reading", args.shp, "…", flush=True)
    gdf = pyogrio.read_dataframe(args.shp,
                                 columns=["HYRIV_ID", "NEXT_DOWN", "ORD_STRA",
                                          "UPLAND_SKM"])
    n = len(gdf)
    print(f"  {n:,} reaches", flush=True)

    ids = gdf["HYRIV_ID"].to_numpy("int64")
    nxt = gdf["NEXT_DOWN"].to_numpy("int64")
    order = gdf["ORD_STRA"].to_numpy("int8")
    upland = gdf["UPLAND_SKM"].to_numpy("float32")

    lats, lons = [], []
    for geom in gdf.geometry:
        xy = np.asarray(geom.coords, "float32")
        lons.append(xy[:, 0])
        lats.append(xy[:, 1])
    riv_tbl = pa.table({
        "id": ids, "next": nxt, "ord": order, "upland": upland,
        "lats": pa.array(lats, pa.list_(pa.float32())),
        "lons": pa.array(lons, pa.list_(pa.float32())),
    })

    # -- snap every GAGES-II gauge to its reach --------------------------------
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    from scipy.spatial import cKDTree
    cat = pq.read_table(hf_hub_download(HF_REPO, "gauges/gagesII_9322.parquet",
                                        repo_type="dataset", token=token))
    gid = np.array([str(s).zfill(8) for s in cat.column("STAID").to_pylist()])
    glat = cat.column("LAT_GAGE").to_numpy().astype("float64")
    glon = cat.column("LNG_GAGE").to_numpy().astype("float64")
    gdrain = cat.column("DRAIN_SQKM").to_numpy().astype("float64")

    # KD-tree over every reach vertex (vertex -> owning reach index)
    flat_lon = np.concatenate(lons)
    flat_lat = np.concatenate(lats)
    counts = np.array([len(a) for a in lats])
    owner = np.repeat(np.arange(n), counts)
    print(f"  KD-tree over {len(owner):,} vertices …", flush=True)
    tree = cKDTree(np.column_stack([flat_lon, flat_lat]))

    dists, vidx = tree.query(np.column_stack([glon, glat]), k=24,
                             distance_upper_bound=SNAP_RADIUS_DEG, workers=-1)
    s_gid, s_reach, s_dist, s_up = [], [], [], []
    for g in range(len(gid)):
        best, best_score = -1, 1e18
        seen = set()
        for kk in range(dists.shape[1]):
            if not np.isfinite(dists[g, kk]):
                break
            ri = int(owner[vidx[g, kk]])
            if ri in seen:
                continue
            seen.add(ri)
            dist_km = dists[g, kk] * KM_PER_DEG
            ratio = abs(np.log10((float(upland[ri]) + 1.0)
                                 / (max(gdrain[g], 0.0) + 1.0)))
            # distance dominates nearby; drainage-area match breaks ties so a
            # big-river gauge doesn't grab the closer small tributary
            score = dist_km + 4.0 * ratio
            if score < best_score:
                best, best_score = ri, score
        if best >= 0:
            s_gid.append(gid[g]); s_reach.append(int(ids[best]))
            s_dist.append(round(float(dists[g, 0] * KM_PER_DEG), 2))
            s_up.append(float(upland[best]))
    snap_tbl = pa.table({"gid": s_gid, "reach_id": np.array(s_reach, "int64"),
                         "dist_km": np.array(s_dist, "float32"),
                         "upland_skm": np.array(s_up, "float32")})
    print(f"  snapped {len(s_gid):,}/{len(gid):,} gauges", flush=True)

    if args.dry_run:
        print("[dry-run: no upload]")
        return 0
    tmp = tempfile.mkdtemp()
    rp = os.path.join(tmp, "rivers.parquet")
    sp = os.path.join(tmp, "snap.parquet")
    pq.write_table(riv_tbl, rp, compression="zstd")
    pq.write_table(snap_tbl, sp, compression="zstd")
    print("rivers parquet:", round(os.path.getsize(rp) / 1e6, 1), "MB;",
          "snap parquet:", round(os.path.getsize(sp) / 1e6, 2), "MB")
    api = HfApi(token=token)
    api.create_commit(repo_id=HF_REPO, repo_type="dataset",
                      operations=[CommitOperationAdd(RIVERS_PATH, rp),
                                  CommitOperationAdd(SNAP_PATH, sp)],
                      commit_message=f"HydroRIVERS NA topology + gauge snap ({n:,} reaches)")
    print("uploaded", RIVERS_PATH, "and", SNAP_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())

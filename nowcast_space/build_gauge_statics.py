"""Per-gauge static attributes for DI-LSTM v3 (feat_version 3).

Joins each GAGES-II gauge to its HydroBASINS level-7 basin — point-in-polygon
on hybas_na_lev07, nearest-zonal-centroid fallback for gauges outside the NA
region polygons (AK/HI/PR) — then pulls the hydrologZ zonal characteristics:

  from features_level7.csv   : monthly P/T/ET/R climatology -> annual mean +
                               seasonality, frost_days, hot_days, gws_amp_mm
  from stream_gradient_level7: relief_m, gradient_m_per_km
  from cluster_labels_level7 : PC1..PC3 (the catchment-similarity embedding)

Skewed magnitudes are log1p-transformed HERE; z-scoring happens at training
time with stats stored in the checkpoint (so serving can reproduce it).

Run with the geo env (geopandas; no pyarrow, hence CSV in/out):
    ~/.conda/envs/notebook2/bin/python build_gauge_statics.py
Reads gauge_latlon.csv, writes gauge_statics.csv (one row per gauge).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

ZONES = "/media/scratch/MengyuChen/zonal_stats/output"
BASINS = "/media/scratch/MengyuChen/hydrobasins/basins/hybas_na_lev07_v1c.shp"

STATIC_COLS = ["p_mean", "p_seas", "t_mean", "t_range", "et_mean", "r_mean",
               "runoff_ratio", "evap_index", "frost_days", "hot_days",
               "gws_amp", "relief", "gradient", "PC1", "PC2", "PC3"]


def main():
    g = pd.read_csv("gauge_latlon.csv", dtype={"STAID": str})
    feats = pd.read_csv(f"{ZONES}/zones/features_level7.csv").set_index("HYBAS_ID")
    grad = pd.read_csv(f"{ZONES}/stream_gradient_level7.csv").set_index("HYBAS_ID")
    clus = pd.read_csv(f"{ZONES}/zones/cluster_labels_level7.csv").set_index("HYBAS_ID")

    pts = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.LNG_GAGE, g.LAT_GAGE),
                           crs="EPSG:4326")
    na = gpd.read_file(BASINS)[["HYBAS_ID", "geometry"]]
    joined = gpd.sjoin(pts, na, how="left", op="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # boundary doubles
    hit = joined["HYBAS_ID"].notna()
    print(f"point-in-polygon matched {hit.sum()}/{len(joined)} gauges")

    # fallback: nearest zonal centroid (covers AK/HI/PR via the global table)
    tree = cKDTree(np.c_[feats["cen_lat"], feats["cen_lon"]])
    miss = ~hit
    if miss.any():
        _, idx = tree.query(np.c_[joined.loc[miss, "LAT_GAGE"],
                                  joined.loc[miss, "LNG_GAGE"]])
        joined.loc[miss, "HYBAS_ID"] = feats.index.to_numpy()[idx]
        print(f"nearest-centroid fallback for {miss.sum()} gauges")
    hb = joined["HYBAS_ID"].astype(np.int64)

    mon = lambda v: feats[[f"mon_{v}_{m:02d}" for m in range(1, 13)]].to_numpy()
    P, T, ET, R = mon("P"), mon("T"), mon("ET"), mon("R")
    eps = 1e-6
    tbl = pd.DataFrame(index=feats.index)
    tbl["p_mean"] = np.log1p(np.maximum(P.mean(1), 0))
    tbl["p_seas"] = (P.max(1) - P.min(1)) / (P.mean(1) + eps)
    tbl["t_mean"] = T.mean(1)
    tbl["t_range"] = T.max(1) - T.min(1)
    tbl["et_mean"] = ET.mean(1)                       # can be ~0/negative; linear
    tbl["r_mean"] = np.log1p(np.maximum(R.mean(1), 0))
    tbl["runoff_ratio"] = np.clip(R.mean(1) / (P.mean(1) + eps), 0, 2)
    tbl["evap_index"] = np.clip(ET.mean(1) / (P.mean(1) + eps), -1, 3)
    tbl["frost_days"] = feats["frost_days"]
    tbl["hot_days"] = feats["hot_days"]
    tbl["gws_amp"] = np.log1p(np.maximum(feats["gws_amp_mm"], 0))
    tbl["relief"] = np.log1p(np.maximum(grad["relief_m"].reindex(feats.index), 0))
    tbl["gradient"] = np.log1p(
        np.maximum(grad["gradient_m_per_km"].reindex(feats.index), 0))
    for pc in ("PC1", "PC2", "PC3"):
        tbl[pc] = clus[pc].reindex(feats.index)

    out = tbl.loc[hb.to_numpy()].reset_index().rename(columns={"HYBAS_ID": "hybas_id"})
    out.insert(0, "STAID", joined["STAID"].to_numpy())
    out.to_csv("gauge_statics.csv", index=False)
    nn = out[STATIC_COLS].isna().sum()
    print(f"wrote gauge_statics.csv: {len(out)} gauges x {len(STATIC_COLS)} statics")
    print("NaN counts:", {k: int(v) for k, v in nn.items() if v})


if __name__ == "__main__":
    main()

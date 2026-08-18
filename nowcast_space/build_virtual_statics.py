"""Static attributes for the 2,676 virtual pour points (DI-LSTM v3 experiment).

Same 16 attributes and transformations as build_gauge_statics.py, but no
spatial join needed: virtual points already carry their HydroBASINS level-7
id (they ARE L7 pour points), so this is a direct table lookup. z-scoring
happens at serving time with the stats stored in the v3 checkpoint.

Run with the nowcast env (pandas + pyarrow, no geo deps):
    /media/scratch/MengyuChen/conda_envs/nowcast/bin/python build_virtual_statics.py
Writes virtual_statics.parquet and uploads it to
vincewin/CREST_nowcast_data gauges/virtual_statics.parquet.
"""
import os

import numpy as np
import pandas as pd

ZONES = "/media/scratch/MengyuChen/zonal_stats/output"
STATIC_COLS = ["p_mean", "p_seas", "t_mean", "t_range", "et_mean", "r_mean",
               "runoff_ratio", "evap_index", "frost_days", "hot_days",
               "gws_amp", "relief", "gradient", "PC1", "PC2", "PC3"]


def main():
    from huggingface_hub import HfApi, hf_hub_download
    tok = open(os.path.expanduser("~/huggingface.txt")).read().strip()
    vp = pd.read_parquet(hf_hub_download("vincewin/CREST_data",
                                         "gauges/virtual_points.parquet",
                                         repo_type="dataset", token=tok))
    feats = pd.read_csv(f"{ZONES}/zones/features_level7.csv").set_index("HYBAS_ID")
    grad = pd.read_csv(f"{ZONES}/stream_gradient_level7.csv").set_index("HYBAS_ID")
    clus = pd.read_csv(f"{ZONES}/zones/cluster_labels_level7.csv").set_index("HYBAS_ID")

    # identical transformation block to build_gauge_statics.py (keep in sync)
    mon = lambda v: feats[[f"mon_{v}_{m:02d}" for m in range(1, 13)]].to_numpy()
    P, T, ET, R = mon("P"), mon("T"), mon("ET"), mon("R")
    eps = 1e-6
    tbl = pd.DataFrame(index=feats.index)
    tbl["p_mean"] = np.log1p(np.maximum(P.mean(1), 0))
    tbl["p_seas"] = (P.max(1) - P.min(1)) / (P.mean(1) + eps)
    tbl["t_mean"] = T.mean(1)
    tbl["t_range"] = T.max(1) - T.min(1)
    tbl["et_mean"] = ET.mean(1)
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

    out = tbl.loc[vp["hybas_id"].to_numpy()].reset_index().rename(
        columns={"HYBAS_ID": "hybas_id"})
    out.insert(0, "vp", vp["vp"].to_numpy())
    out = out[["vp", "hybas_id"] + STATIC_COLS]
    out.to_parquet("virtual_statics.parquet", index=False)
    nn = out[STATIC_COLS].isna().sum()
    print(f"wrote virtual_statics.parquet: {len(out)} points x {len(STATIC_COLS)}")
    print("NaN counts:", {k: int(v) for k, v in nn.items() if v} or "none")

    api = HfApi(token=tok)
    api.upload_file(path_or_fileobj="virtual_statics.parquet",
                    path_in_repo="gauges/virtual_statics.parquet",
                    repo_id="vincewin/CREST_nowcast_data", repo_type="dataset",
                    commit_message="virtual-point statics (hybas_id lookup) for the v3 experiment")
    print("uploaded to vincewin/CREST_nowcast_data gauges/virtual_statics.parquet")


if __name__ == "__main__":
    main()

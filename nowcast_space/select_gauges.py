"""Pick a stratified training-gauge set for the DI-LSTM v2 retraining.

One global model serves ~9k CONUS gauges, so the training set should span
drainage areas and regions rather than the 4 starter gauges. Strata =
log10-area quartile x coarse lat/lon cell; gauges are drawn round-robin
across non-empty strata (seeded, reproducible). The 4 starter gauges are
always included (their 2023-2025 months are already prepped).

    export HF_TOKEN=hf_...
    python select_gauges.py --n 256 --out gauges_v2.txt
    python prep_hpc.py --gauges-file gauges_v2.txt --months 2023_01-2025_06
"""
from __future__ import annotations

import argparse
import os

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import numpy as np

STARTERS = ["01011000", "08166200", "08167000", "08144500"]
MRMS_BOX = (-130.0, 20.0, -60.0, 55.0)          # w, s, e, n (matches the grid)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--cells", type=int, default=8,
                    help="lat/lon grid is cells x 2*cells")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="gauges_v2.txt")
    args = ap.parse_args()

    import pandas as pd
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("vincewin/CREST_data", "gauges/gagesII_9322.parquet",
                        repo_type="dataset", token=os.environ.get("HF_TOKEN"))
    df = pd.read_parquet(p)
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    w, s, e, n = MRMS_BOX
    df = df[(df["LNG_GAGE"] > w) & (df["LNG_GAGE"] < e)
            & (df["LAT_GAGE"] > s) & (df["LAT_GAGE"] < n)
            & (df["DRAIN_SQKM"] > 1.0)].reset_index(drop=True)

    la = np.log10(df["DRAIN_SQKM"].to_numpy())
    qa = np.searchsorted(np.quantile(la, [0.25, 0.5, 0.75]), la)      # 0..3
    ci = np.clip(((df["LAT_GAGE"] - s) / (n - s) * args.cells).astype(int),
                 0, args.cells - 1)
    cj = np.clip(((df["LNG_GAGE"] - w) / (e - w) * 2 * args.cells).astype(int),
                 0, 2 * args.cells - 1)
    strata = qa * 10000 + ci * 100 + cj

    rng = np.random.default_rng(args.seed)
    by_stratum: dict[int, list[str]] = {}
    for gid, st in zip(df["STAID"], strata):
        by_stratum.setdefault(int(st), []).append(gid)
    for ids in by_stratum.values():
        rng.shuffle(ids)

    chosen = [g for g in STARTERS if g in set(df["STAID"])]
    keys = sorted(by_stratum)
    i = 0
    while len(chosen) < args.n and any(by_stratum.values()):
        ids = by_stratum[keys[i % len(keys)]]
        if ids:
            g = ids.pop()
            if g not in chosen:
                chosen.append(g)
        i += 1
        if i > 10 * args.n * len(keys):
            break

    with open(args.out, "w") as f:
        f.write(",".join(chosen) + "\n")
    print(f"{len(chosen)} gauges over {len(keys)} non-empty strata -> {args.out}")
    print("area km2 quartiles of the pick:",
          np.round(np.quantile(df.set_index('STAID').loc[chosen, 'DRAIN_SQKM'],
                               [0, 0.25, 0.5, 0.75, 1.0]), 1).tolist())


if __name__ == "__main__":
    main()

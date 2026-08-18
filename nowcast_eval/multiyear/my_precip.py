"""Basin-mean hourly precip for one eval month (+72-h lookback), all gauges.
Source: HF CREST_data mrms/<Y>/mrms_<Y>_<M>.tar (writer .pqf grids; the same
files the operational writer + training prep use); a missing hour falls back
to /media/scratch/ZhiLi/MRMS GaugeCorr tif; else NaN (-> 0 in features).
usage: my_precip.py <ym>  -> precip/precip_<ym>.npz"""
import os
import sys
import tarfile
import time

import numpy as np
from huggingface_hub import hf_hub_download

from common import (BASE, HF_TOKEN, ZHILI, gauges, grid_boxes, box_means,
                    month_hours, read_pqf_bytes, read_tif)

ym = sys.argv[1]
outp = os.path.join(BASE, "precip", f"precip_{ym}.npz")
if os.path.exists(outp):
    print(f"{ym}: exists"); sys.exit(0)
t = time.time()
gids, lat, lon, area = gauges()
boxes = grid_boxes(lon, lat, area)
hours, i0 = month_hours(ym)
# tars needed: the month and the previous month (lookback)
need = sorted({f"{h:%Y%m}" for h in hours})
tars = {}
for m in need:
    try:
        p = hf_hub_download("vincewin/CREST_data", f"mrms/{m[:4]}/mrms_{m[:4]}_{m[4:]}.tar",
                            repo_type="dataset", token=HF_TOKEN)
        tars[m] = tarfile.open(p)
    except Exception as e:
        print(f"{ym}: tar {m} unavailable ({e})", flush=True)
names = {m: set(tf.getnames()) for m, tf in tars.items()}
pmat = np.full((len(gids), len(hours)), np.nan, "float32")
n_tar = n_tif = n_miss = 0
for i, h in enumerate(hours):
    a = None
    m = f"{h:%Y%m}"
    if m in tars:
        for nm in (f"mrms_corr_{h:%Y%m%d%H}.pqf", f"mrms_{h:%Y%m%d%H}.pqf"):
            if nm in names[m]:
                a = read_pqf_bytes(tars[m].extractfile(nm).read())
                if a is not None:
                    n_tar += 1
                break
    if a is None:
        tp = os.path.join(ZHILI, f"GaugeCorr_QPE_01H_00.00_{h:%Y%m%d}-{h:%H}0000.grib2.tif")
        if os.path.exists(tp):
            a = read_tif(tp)
            if a is not None:
                n_tif += 1
    if a is None:
        n_miss += 1
        continue
    pmat[:, i] = box_means(a, boxes)
    if i % 200 == 0:
        print(f"  {ym} {i}/{len(hours)}", flush=True)
np.savez_compressed(outp, gids=np.array(gids),
                    hours=np.array([f"{h:%Y%m%d%H}" for h in hours]), pmat=pmat)
print(f"{ym}: hours tar {n_tar}, tif {n_tif}, missing {n_miss} of {len(hours)}; "
      f"{time.time() - t:.0f}s", flush=True)

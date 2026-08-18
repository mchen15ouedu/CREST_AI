"""Shared config/helpers for the multi-year all-gauge nowcast evaluation.

Periods: Jan/Apr/Jul/Oct of 2016..2026 (43 months). Training window of BOTH
models (v1, v3-full) is 2023-01..2025-06 -> those months are IN-SAMPLE (time),
everything else is out-of-sample. Gauges: all served (allgauge/eval_gauges.txt).
"""
import glob
import io
import math
import os
import tarfile
from datetime import datetime, timedelta

import numpy as np
import pyarrow.parquet as pq

BASE = os.path.dirname(os.path.abspath(__file__))
ALLG = os.path.join(BASE, "..", "allgauge")
HF_TOKEN = open(os.path.expanduser("~/huggingface.txt")).read().strip()
os.environ.setdefault("HF_HOME", "/media/scratch/MengyuChen/hf_cache")
ZHILI = "/media/scratch/ZhiLi/MRMS"

MONTHS = [f"{y}{m:02d}" for y in range(2016, 2027) for m in (1, 4, 7, 10)
          if (y, m) <= (2026, 7)]
TRAIN_LO, TRAIN_HI = "202301", "202506"
SEASON = {"01": "winter", "04": "spring", "07": "summer", "10": "fall"}
LOOKBACK = 72
LEADS = 12
SCEN = ["fresh", "stale24", "noobs"]
MODELS = ["v1_6", "v3f_6", "v1_12", "v3f_12", "persist"]
CKPTS = {"v1_6": "dilstm.pt", "v1_12": "dilstm_h12.pt",
         "v3f_6": "dilstm_v3_full.pt", "v3f_12": "dilstm_h12_v3_full.pt"}
MRMS_GRID = (-130.0, 20.0, 0.01, 3500, 7000, -9999.0)
AGE_CAP_H = 240.0


def in_sample(ym):
    return TRAIN_LO <= ym <= TRAIN_HI


def month_hours(ym):
    """(hours list incl. 72-h lookback, first issue index) for month ym."""
    y, m = int(ym[:4]), int(ym[4:])
    start = datetime(y, m, 1)
    end = datetime(y + (m == 12), m % 12 + 1, 1)          # exclusive
    h0 = start - timedelta(hours=LOOKBACK - 1)
    hours = []
    h = h0
    while h < end:
        hours.append(h)
        h += timedelta(hours=1)
    return hours, LOOKBACK - 1


def gauges():
    gids = [x.strip() for x in open(os.path.join(ALLG, "eval_gauges.txt")) if x.strip()]
    arch = sorted(glob.glob(os.path.join(ALLG, "archive/nowcast/archive/*/nc_*.parquet")))[-1]
    t = pq.read_table(arch, columns=["gid", "lat", "lon", "area_km2"])
    meta = {str(g).zfill(8): (la, lo, a) for g, la, lo, a in zip(
        t.column("gid").to_pylist(), t.column("lat").to_numpy(),
        t.column("lon").to_numpy(), t.column("area_km2").to_numpy())}
    gids = [g for g in gids if g in meta]
    lat = np.array([meta[g][0] for g in gids], "float64")
    lon = np.array([meta[g][1] for g in gids], "float64")
    area = np.array([meta[g][2] for g in gids], "float64")
    return gids, lat, lon, area


# ---- verbatim copies of run_nowcast_all.py grid helpers ----------------------
def read_pqf_bytes(data):
    pf = pq.ParquetFile(io.BytesIO(data))
    meta = {k.decode(): v.decode() for k, v in pf.schema_arrow.metadata.items()
            if not k.startswith(b"ARROW")}
    nr, nc = int(meta["nrows"]), int(meta["ncols"])
    if (nr, nc) != MRMS_GRID[3:5]:
        return None
    return pf.read().column("v").to_numpy().reshape(nr, nc)


def _basin_box(lon, lat, area_km2, pad=1.2):
    r = max(0.3, min(2.5, pad * math.sqrt(max(area_km2, 1.0)) / 111.0))
    return (lon - r, lat - r, lon + r, lat + r)


def grid_boxes(lons, lats, areas):
    xll, yll, cell, nr, nc, _ = MRMS_GRID
    top = yll + nr * cell
    r0s, r1s, c0s, c1s = [], [], [], []
    for lon, lat, area in zip(lons, lats, areas):
        w, s, e, n = _basin_box(lon, lat, area)
        c0 = max(0, int((w - xll) / cell)); c1 = min(nc, int(math.ceil((e - xll) / cell)))
        r0 = max(0, int((top - n) / cell)); r1 = min(nr, int(math.ceil((top - s) / cell)))
        r0s.append(r0); r1s.append(max(r0, r1)); c0s.append(c0); c1s.append(max(c0, c1))
    return (np.array(r0s), np.array(r1s), np.array(c0s), np.array(c1s))


def box_means(a, boxes):
    """Writer's summed-area-table box means, computed on the crop that covers
    all boxes (identical result, ~4x faster than the full 3500x7000 SAT)."""
    r0, r1, c0, c1 = boxes
    R0, R1, C0, C1 = int(r0.min()), int(r1.max()), int(c0.min()), int(c1.max())
    sub = a[R0:R1, C0:C1]
    valid = sub >= 0.0
    S = np.zeros((sub.shape[0] + 1, sub.shape[1] + 1), "float64")
    C = np.zeros(S.shape, "int64")
    np.cumsum(np.cumsum(np.where(valid, sub, 0), 0), 1, out=S[1:, 1:])
    np.cumsum(np.cumsum(valid, 0, dtype="int64"), 1, out=C[1:, 1:])
    rr0, rr1, cc0, cc1 = r0 - R0, r1 - R0, c0 - C0, c1 - C0
    tot = S[rr1, cc1] - S[rr0, cc1] - S[rr1, cc0] + S[rr0, cc0]
    cnt = C[rr1, cc1] - C[rr0, cc1] - C[rr1, cc0] + C[rr0, cc0]
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype("float32")


def read_tif(path):
    """ZhiLi GaugeCorr tif fallback -> grid on the writer's layout (north-up)."""
    import rasterio
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float32")
        if (ds.height, ds.width) != MRMS_GRID[3:5]:
            return None
    return a

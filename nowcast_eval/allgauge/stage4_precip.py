"""Stage 4: basin-mean hourly precip for the eval gauges, Jul 18 - Aug 12.

Reuses the exact grid/box/summed-area-table logic of run_nowcast_all.py so
the replay precip matches what the operational writer computed. Source per
hour: mrms_recent Pass1 grid if present (that is what operations used),
else the Pass2 month tar (only the Jul 18-22 window tails need this).

Output: precip.npz (gids, hours as YYYYMMDDHH strings, pmat[n_gauge, n_hour])
"""
import glob
import io
import math
import os
import tarfile
from datetime import datetime, timedelta

import numpy as np
import pyarrow.parquet as pq

# ---- verbatim copies of run_nowcast_all.py grid helpers (importing that
# module drags in truststore, absent from the nowcast env) -------------------
MRMS_GRID = (-130.0, 20.0, 0.01, 3500, 7000, -9999.0)


def _read_pqf_bytes(data: bytes) -> np.ndarray | None:
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


def _grid_boxes(lons, lats, areas):
    xll, yll, cell, nr, nc, _ = MRMS_GRID
    top = yll + nr * cell
    r0s, r1s, c0s, c1s = [], [], [], []
    for lon, lat, area in zip(lons, lats, areas):
        w, s, e, n = _basin_box(lon, lat, area)
        c0 = max(0, int((w - xll) / cell)); c1 = min(nc, int(math.ceil((e - xll) / cell)))
        r0 = max(0, int((top - n) / cell)); r1 = min(nr, int(math.ceil((top - s) / cell)))
        r0s.append(r0); r1s.append(max(r0, r1)); c0s.append(c0); c1s.append(max(c0, c1))
    return (np.array(r0s), np.array(r1s), np.array(c0s), np.array(c1s))


def _box_means(a: np.ndarray, boxes) -> np.ndarray:
    r0, r1, c0, c1 = boxes
    valid = a >= 0.0
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1), "float64")
    C = np.zeros_like(S)
    np.cumsum(np.cumsum(np.where(valid, a, 0), 0), 1, out=S[1:, 1:])
    np.cumsum(np.cumsum(valid, 0), 1, out=C[1:, 1:])
    tot = S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]
    cnt = C[r1, c1] - C[r0, c1] - C[r1, c0] + C[r0, c0]
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype("float32")


BASE = os.path.dirname(os.path.abspath(__file__))
H0 = datetime(2026, 7, 18, 16)          # first archived t0 minus 71 h
H1 = datetime(2026, 8, 12, 12)          # last scored t0

# gauge geometry from any archive file (writer clipped to the MRMS grid)
arch = sorted(glob.glob(os.path.join(BASE, "archive/nowcast/archive/*/nc_*.parquet")))
t = pq.read_table(arch[0], columns=["gid", "lat", "lon", "area_km2"])
meta = {g: (la, lo, a) for g, la, lo, a in zip(t.column("gid").to_pylist(),
                                               t.column("lat").to_numpy(),
                                               t.column("lon").to_numpy(),
                                               t.column("area_km2").to_numpy())}
gids = [x.strip() for x in open(os.path.join(BASE, "eval_gauges.txt")) if x.strip()]
gids = [g for g in gids if g in meta]
lat = np.array([meta[g][0] for g in gids], "float64")
lon = np.array([meta[g][1] for g in gids], "float64")
area = np.array([meta[g][2] for g in gids], "float64")
boxes = _grid_boxes(lon, lat, area)
print(f"{len(gids)} gauges with geometry", flush=True)

hours = []
h = H0
while h <= H1:
    hours.append(h)
    h += timedelta(hours=1)

recent_dir = os.path.join(BASE, "forcing", "mrms_recent")
# Pass2 tars come out of the HF cache (already downloaded by stage 2)
from huggingface_hub import hf_hub_download
TOK = open(os.path.expanduser("~/huggingface.txt")).read().strip()
tars = {}
for m in ("07", "08"):
    p = hf_hub_download("vincewin/CREST_data", f"mrms/2026/mrms_2026_{m}.tar",
                        repo_type="dataset", token=TOK)
    tars[m] = tarfile.open(p)

pmat = np.full((len(gids), len(hours)), np.nan, "float32")
n_recent = n_pass2 = n_miss = 0
for i, t_h in enumerate(hours):
    data = None
    rp = os.path.join(recent_dir, f"mrms1h_pass1_{t_h:%Y%m%d%H}.pqf")
    if os.path.exists(rp):
        data = open(rp, "rb").read()
        n_recent += 1
    else:
        tf = tars.get(f"{t_h.month:02d}")
        if tf is not None:
            for name in (f"mrms_corr_{t_h:%Y%m%d%H}.pqf", f"mrms_{t_h:%Y%m%d%H}.pqf"):
                try:
                    data = tf.extractfile(name).read()
                    n_pass2 += 1
                    break
                except KeyError:
                    continue
    if data is None:
        n_miss += 1
        continue
    a = _read_pqf_bytes(data)
    if a is None:
        n_miss += 1
        continue
    pmat[:, i] = _box_means(a, boxes)
    if i % 50 == 0:
        print(f"  {i}/{len(hours)} ({t_h:%Y-%m-%d %H}Z)", flush=True)

print(f"hours: {n_recent} recent, {n_pass2} pass2, {n_miss} missing", flush=True)
np.savez_compressed(os.path.join(BASE, "precip.npz"),
                    gids=np.array(gids),
                    hours=np.array([f"{t_h:%Y%m%d%H}" for t_h in hours]),
                    pmat=pmat)
print("saved precip.npz", flush=True)

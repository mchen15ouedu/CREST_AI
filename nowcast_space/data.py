"""Training-data preparation for the DI-LSTM nowcaster.

Two ingredients, both cached per (gauge, month) in the private dataset
DATA_REPO (vincewin/CREST_nowcast_data) so prep is resumable and shared:

  obs/<gid>/<YYYY_MM>.parquet    hourly-mean USGS IV discharge (m3/s, UTC)
  mrms/<gid>/<YYYY_MM>.parquet   basin-mean MRMS precipitation (mm/h, UTC)

MRMS extraction: one CONUS month-tar from vincewin/CREST_data serves EVERY
gauge — each hourly pqf is read once and all basin-box means are taken from
the same array. Basin box = the CREST_demo pipeline's generous
area-scaled box around the outlet.
"""
from __future__ import annotations

import io
import math
import os
import tarfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi, hf_hub_download

DATA_REPO = os.environ.get("NOWCAST_DATA_REPO", "vincewin/CREST_nowcast_data")
FORCING_REPO = "vincewin/CREST_data"
CFS_TO_CMS = 0.0283168


def _token():
    return os.environ.get("HF_TOKEN")


def _api():
    return HfApi(token=_token())


def basin_box(lon: float, lat: float, area_km2: float, pad: float = 1.2):
    r = max(0.3, min(2.5, pad * math.sqrt(max(area_km2, 1.0)) / 111.0))
    return (lon - r, lat - r, lon + r, lat + r)          # w, s, e, n


# ---- USGS hourly obs ---------------------------------------------------------
def fetch_usgs_hourly(site: str, t0: datetime, t1: datetime) -> pd.Series:
    """Hourly-mean discharge (m3/s), UTC-naive index."""
    r = requests.get("https://waterservices.usgs.gov/nwis/iv/",
                     params={"sites": str(site).zfill(8), "parameterCd": "00060",
                             "format": "json",
                             "startDT": (t0 - timedelta(days=1)).strftime("%Y-%m-%d"),
                             "endDT": (t1 + timedelta(days=1)).strftime("%Y-%m-%d"),
                             "siteStatus": "all"}, timeout=60)
    r.raise_for_status()
    ts = r.json().get("value", {}).get("timeSeries", [])
    if not ts:
        return pd.Series(dtype="float64")
    rows = []
    for v in ts[0]["values"][0]["value"]:
        try:
            cfs = float(v["value"])
        except (TypeError, ValueError):
            continue
        if cfs < 0:
            continue
        dt = (datetime.fromisoformat(v["dateTime"].replace("Z", "+00:00"))
              .astimezone(timezone.utc).replace(tzinfo=None))
        rows.append((dt, cfs * CFS_TO_CMS))
    if not rows:
        return pd.Series(dtype="float64")
    s = pd.Series(dict(rows)).sort_index()
    return s.resample("1h").mean()


# ---- pqf reader (CREST_data forcing format) ----------------------------------
def _read_pqf(data: bytes):
    pf = pq.ParquetFile(io.BytesIO(data))
    m = {k.decode(): v.decode() for k, v in pf.schema_arrow.metadata.items()
         if not k.startswith(b"ARROW")}
    nc, nr = int(m["ncols"]), int(m["nrows"])
    a = pf.read().column("v").to_numpy().reshape(nr, nc)
    return a, float(m["xllcorner"]), float(m["yllcorner"]), float(m["cellsize"]), float(m["nodata"])


def _box_mean(a, xll, yll, cell, nodata, box):
    nr, nc = a.shape
    w, s, e, n = box
    c0 = max(0, int((w - xll) / cell)); c1 = min(nc, int(math.ceil((e - xll) / cell)))
    top = yll + nr * cell
    r0 = max(0, int((top - n) / cell)); r1 = min(nr, int(math.ceil((top - s) / cell)))
    if r1 <= r0 or c1 <= c0:
        return np.nan
    sub = a[r0:r1, c0:c1]
    ok = (sub != nodata) & np.isfinite(sub) & (sub >= 0)
    return float(sub[ok].mean()) if ok.any() else np.nan


# ---- per-month prep (resumable) ----------------------------------------------
def _repo_has(path: str, files: set[str]) -> bool:
    return path in files


def prep_month(gauges: list[dict], year: int, month: int, log=print) -> dict:
    """gauges: [{id, lat, lon, area_km2}]. Builds+uploads obs/ and mrms/ parquet
    for every gauge missing this month. Returns a small report."""
    api = _api()
    api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
    have = set(api.list_repo_files(DATA_REPO, repo_type="dataset"))
    ym = f"{year:04d}_{month:02d}"
    t0 = datetime(year, month, 1)
    t1 = (datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1))

    ops = []
    # 1. USGS obs (cheap)
    n_obs = 0
    for g in gauges:
        path = f"obs/{g['id']}/{ym}.parquet"
        if _repo_has(path, have):
            continue
        s = fetch_usgs_hourly(g["id"], t0, t1)
        s = s[(s.index >= t0) & (s.index < t1)]
        buf = io.BytesIO()
        pq.write_table(pa.table({"time": s.index.to_pydatetime().tolist(),
                                 "q": s.values.astype("float64")}), buf, compression="zstd")
        ops.append((path, buf.getvalue()))
        n_obs += 1

    # 2. MRMS basin means (one tar pass serves all gauges)
    need = [g for g in gauges if not _repo_has(f"mrms/{g['id']}/{ym}.parquet", have)]
    n_mrms = 0
    if need:
        tar_path = hf_hub_download(FORCING_REPO, f"mrms/{year}/mrms_{year}_{month:02d}.tar",
                                   repo_type="dataset")
        boxes = {g["id"]: basin_box(g["lon"], g["lat"], g["area_km2"]) for g in need}
        series: dict[str, dict] = {g["id"]: {} for g in need}
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".pqf")]
            for i, m in enumerate(members):
                stem = os.path.basename(m.name)
                digits = "".join(ch for ch in stem if ch.isdigit())[-10:]
                try:
                    when = datetime.strptime(digits, "%Y%m%d%H")
                except ValueError:
                    continue
                a, xll, yll, cell, nod = _read_pqf(tf.extractfile(m).read())
                for gid, box in boxes.items():
                    series[gid][when] = _box_mean(a, xll, yll, cell, nod, box)
                if (i + 1) % 120 == 0:
                    log(f"  {ym}: {i + 1}/{len(members)} hours")
        for gid, d in series.items():
            s = pd.Series(d).sort_index()
            buf = io.BytesIO()
            pq.write_table(pa.table({"time": s.index.to_pydatetime().tolist(),
                                     "v": s.values.astype("float64")}), buf, compression="zstd")
            ops.append((f"mrms/{gid}/{ym}.parquet", buf.getvalue()))
            n_mrms += 1
        try:
            os.remove(tar_path)                  # keep the Space disk lean
        except OSError:
            pass

    if ops:
        from huggingface_hub import CommitOperationAdd
        api.create_commit(repo_id=DATA_REPO, repo_type="dataset",
                          operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=b)
                                      for p, b in ops],
                          commit_message=f"prep {ym}: +{n_obs} obs, +{n_mrms} mrms")
    return {"month": ym, "obs_added": n_obs, "mrms_added": n_mrms,
            "skipped": len(gauges) * 2 - n_obs - n_mrms}


def load_series(gid: str, months: list[str]) -> pd.DataFrame:
    """Join cached obs+mrms months into one hourly frame [q, p]."""
    qs, ps = [], []
    for ym in months:
        for kind, acc in (("obs", qs), ("mrms", ps)):
            try:
                p = hf_hub_download(DATA_REPO, f"{kind}/{gid}/{ym}.parquet",
                                    repo_type="dataset", token=_token())
                t = pq.read_table(p)
                col = "q" if kind == "obs" else "v"
                acc.append(pd.Series(t.column(col).to_numpy(),
                                     index=pd.to_datetime(t.column("time").to_numpy())))
            except Exception:
                pass
    if not qs or not ps:
        return pd.DataFrame()
    q = pd.concat(qs).sort_index()
    p = pd.concat(ps).sort_index()
    df = pd.DataFrame({"q": q, "p": p}).asfreq("1h")
    return df

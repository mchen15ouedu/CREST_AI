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
import time
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


def _parse_iv_json(js: dict) -> dict[str, pd.Series]:
    """NWIS IV JSON → {site: hourly-mean m3/s series}. First timeSeries per
    site wins (same as the single-site path's ts[0])."""
    out: dict[str, pd.Series] = {}
    for ts in js.get("value", {}).get("timeSeries", []):
        try:
            site = ts["sourceInfo"]["siteCode"][0]["value"].zfill(8)
        except (KeyError, IndexError):
            continue
        if site in out:
            continue
        rows = []
        for v in ts["values"][0]["value"]:
            try:
                cfs = float(v["value"])
            except (TypeError, ValueError):
                continue
            if cfs < 0:
                continue
            dt = (datetime.fromisoformat(v["dateTime"].replace("Z", "+00:00"))
                  .astimezone(timezone.utc).replace(tzinfo=None))
            rows.append((dt, cfs * CFS_TO_CMS))
        if rows:
            s = pd.Series(dict(rows)).sort_index()
            out[site] = s.resample("1h").mean()
    return out


def fetch_usgs_hourly_batch(sites: list[str], t0: datetime, t1: datetime,
                            chunk: int = 40, log=print) -> dict[str, pd.Series]:
    """Multi-site NWIS fetch, `chunk` sites per request. Sites the API resolves
    but has no data for map to an empty Series (month is done for them);
    sites in a failed request are absent (retried on the next prep run)."""
    out: dict[str, pd.Series] = {}
    for i in range(0, len(sites), chunk):
        grp = [str(s).zfill(8) for s in sites[i:i + chunk]]
        params = {"sites": ",".join(grp), "parameterCd": "00060", "format": "json",
                  "startDT": (t0 - timedelta(days=1)).strftime("%Y-%m-%d"),
                  "endDT": (t1 + timedelta(days=1)).strftime("%Y-%m-%d"),
                  "siteStatus": "all"}
        for attempt in (1, 2):
            try:
                r = requests.get("https://waterservices.usgs.gov/nwis/iv/",
                                 params=params, timeout=300)
                r.raise_for_status()
                got = _parse_iv_json(r.json())
                for s in grp:
                    out[s] = got.get(s, pd.Series(dtype="float64"))
                break
            except Exception as e:
                if attempt == 2:
                    log(f"  NWIS batch {grp[0]}..{grp[-1]} failed: "
                        f"{type(e).__name__}: {e}")
                else:
                    time.sleep(5)
    return out


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
BUNDLE_MIN = 16   # runs with ≥ this many gauges write per-month bundle files
                  # (obs/_bundle/<ym>.parquet, all gauges long-format) instead of
                  # one file per gauge — keeps the repo at ~2 files/month at scale


def _repo_has(path: str, files: set[str]) -> bool:
    return path in files


def _empty_bundle(col: str) -> pd.DataFrame:
    return pd.DataFrame({"gid": pd.Series(dtype="str"),
                         "time": pd.Series(dtype="datetime64[ns]"),
                         col: pd.Series(dtype="float64")})


def _read_bundle(kind: str, ym: str) -> pd.DataFrame:
    """Existing bundle as a (gid, time, <col>) frame, or an empty frame."""
    col = "q" if kind == "obs" else "v"
    try:
        p = hf_hub_download(DATA_REPO, f"{kind}/_bundle/{ym}.parquet",
                            repo_type="dataset", token=_token())
        return pq.read_table(p).to_pandas()
    except Exception:
        return _empty_bundle(col)


def _bundle_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf,
                   compression="zstd")
    return buf.getvalue()


def _series_rows(gid: str, s: pd.Series, col: str, t0, t1) -> pd.DataFrame:
    """Long-format rows for one gauge-month; a NaN sentinel row marks a gauge
    the API answered for but that has no data (so it isn't refetched)."""
    s = s[(s.index >= t0) & (s.index < t1)] if len(s) else s
    if not len(s):
        return pd.DataFrame({"gid": [gid], "time": [t0], col: [np.nan]})
    return pd.DataFrame({"gid": gid, "time": s.index, col: s.values.astype("float64")})


def _prep_month_bundle(api, gauges: list[dict], ym: str, t0, t1, have, log) -> dict:
    """Bundle-mode prep: one obs + one mrms parquet per month, merged with any
    bundle already uploaded so reruns only fetch what's missing."""
    ops = []

    # 1. USGS obs (batched requests)
    prev_obs = _read_bundle("obs", ym) if f"obs/_bundle/{ym}.parquet" in have \
        else _empty_bundle("q")
    done = set(prev_obs["gid"])
    todo = [g["id"] for g in gauges if g["id"] not in done]
    n_obs = 0
    if todo:
        log(f"  {ym}: fetching obs for {len(todo)} gauges "
            f"({len(done)} already in bundle)")
        got = fetch_usgs_hourly_batch(todo, t0, t1, log=log)
        if got:
            parts = [prev_obs] if len(prev_obs) else []
            parts += [_series_rows(gid, s, "q", t0, t1) for gid, s in got.items()]
            ops.append((f"obs/_bundle/{ym}.parquet",
                        _bundle_bytes(pd.concat(parts, ignore_index=True))))
            n_obs = len(got)

    # 2. MRMS basin means (one tar pass serves all gauges)
    prev_mrms = _read_bundle("mrms", ym) if f"mrms/_bundle/{ym}.parquet" in have \
        else _empty_bundle("v")
    done_m = set(prev_mrms["gid"])
    need = [g for g in gauges if g["id"] not in done_m]
    n_mrms = 0
    if need:
        year, month = map(int, ym.split("_"))
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
        parts = [prev_mrms] if len(prev_mrms) else []
        for gid, d in series.items():
            s = pd.Series(d).sort_index()
            parts.append(pd.DataFrame({"gid": gid, "time": s.index,
                                       "v": s.values.astype("float64")}))
            n_mrms += 1
        ops.append((f"mrms/_bundle/{ym}.parquet",
                    _bundle_bytes(pd.concat(parts, ignore_index=True))))
        try:
            os.remove(tar_path)
        except OSError:
            pass

    if ops:
        from huggingface_hub import CommitOperationAdd
        api.create_commit(repo_id=DATA_REPO, repo_type="dataset",
                          operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=b)
                                      for p, b in ops],
                          commit_message=f"prep {ym} bundle: +{n_obs} obs, +{n_mrms} mrms")
    return {"month": ym, "obs_added": n_obs, "mrms_added": n_mrms,
            "skipped": len(gauges) * 2 - n_obs - n_mrms}


def prep_month(gauges: list[dict], year: int, month: int, log=print) -> dict:
    """gauges: [{id, lat, lon, area_km2}]. Builds+uploads obs/ and mrms/ parquet
    for every gauge missing this month. Returns a small report."""
    api = _api()
    api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
    have = set(api.list_repo_files(DATA_REPO, repo_type="dataset"))
    ym = f"{year:04d}_{month:02d}"
    t0 = datetime(year, month, 1)
    t1 = (datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1))
    if len(gauges) >= BUNDLE_MIN:
        return _prep_month_bundle(api, gauges, ym, t0, t1, have, log)

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


def load_series_bulk(gids: list[str], months: list[str], log=print) -> dict[str, pd.DataFrame]:
    """Bulk load_series: each month's bundle is downloaded and parsed once for
    all gauges; gauges absent from a bundle fall back to per-gauge files.
    Returns {gid: hourly [q, p] frame} (empty frame when a gauge has no data)."""
    acc: dict[str, dict[str, list]] = {g: {"q": [], "p": []} for g in gids}
    want = set(gids)
    for ym in months:
        for kind, col, key in (("obs", "q", "q"), ("mrms", "v", "p")):
            covered: set[str] = set()
            try:
                p = hf_hub_download(DATA_REPO, f"{kind}/_bundle/{ym}.parquet",
                                    repo_type="dataset", token=_token())
                df = pq.read_table(p).to_pandas()
                covered = set(df["gid"]) & want
                df = df[df["gid"].isin(want) & np.isfinite(df[col])]  # drops no-data sentinels
                for gid, sub in df.groupby("gid"):
                    acc[gid][key].append(pd.Series(sub[col].to_numpy(),
                                                   index=pd.to_datetime(sub["time"].to_numpy())))
            except Exception:
                pass
            for gid in want - covered:                 # legacy per-gauge files
                try:
                    p = hf_hub_download(DATA_REPO, f"{kind}/{gid}/{ym}.parquet",
                                        repo_type="dataset", token=_token())
                    t = pq.read_table(p)
                    acc[gid][key].append(pd.Series(t.column(col).to_numpy(),
                                                   index=pd.to_datetime(t.column("time").to_numpy())))
                except Exception:
                    pass
        log(f"  loaded {ym}")
    out: dict[str, pd.DataFrame] = {}
    for gid in gids:
        qs, ps = acc[gid]["q"], acc[gid]["p"]
        if not qs or not ps:
            out[gid] = pd.DataFrame()
            continue
        q = pd.concat(qs).sort_index()
        p = pd.concat(ps).sort_index()
        out[gid] = pd.DataFrame({"q": q, "p": p}).asfreq("1h")
    return out


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

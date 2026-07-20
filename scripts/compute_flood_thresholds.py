"""One-time 10-year return-period flood thresholds for every GAGES-II gauge.

Fetches each gauge's ANNUAL PEAK discharge series from the USGS NWIS peak
service and fits a log-Pearson Type III distribution (the Bulletin 17
standard: log10 peaks -> mean/std/skew, frequency factor via the
Wilson-Hilferty approximation) to estimate the 10-year flood Q10. The
dashboard's Nowcast mode flags a gauge red when the AI's next-6-hour peak
exceeds this threshold.

Gauges with fewer than --min-peaks (10) annual maxima get no threshold
(they are simply never flagged). Output uploaded once to vincewin/CREST_data:

    nowcast/flood_thresholds.parquet   gid, q10_cms, q2_cms, n_peaks

Re-runnable (full recompute + overwrite); takes ~10-20 min for ~9k gauges.

    python scripts/compute_flood_thresholds.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import HF_REPO, hf_token                             # noqa: E402

CFS_TO_CMS = 0.0283168
OUT_PATH = "nowcast/flood_thresholds.parquet"
# standard-normal quantiles (T-year, p = 1 - 1/T)
_Z = {2: 0.0, 5: 0.8416212, 10: 1.2815516}


def fetch_dv_chunk(sites: list[str]):
    """Last-3-yr daily-mean discharge series (m3/s) per site via the batched
    NWIS daily-values service. Returns {sid: pd.Series} (>=90 days each)."""
    import pandas as pd
    out = {}
    try:
        r = requests.get("https://waterservices.usgs.gov/nwis/dv/",
                         params={"sites": ",".join(sites), "parameterCd": "00060",
                                 "statCd": "00003", "period": "P1095D",
                                 "format": "json", "siteStatus": "all"}, timeout=90)
        r.raise_for_status()
        for ts in r.json().get("value", {}).get("timeSeries", []):
            sid = ts["sourceInfo"]["siteCode"][0]["value"].zfill(8)
            when, vals = [], []
            for v in ts["values"][0]["value"]:
                try:
                    x = float(v["value"])
                except (TypeError, ValueError):
                    continue
                if x >= 0:
                    when.append(v["dateTime"]); vals.append(x * CFS_TO_CMS)
            if len(vals) >= 90:                      # need a real record
                s = pd.Series(vals, index=pd.to_datetime(when))
                out[sid] = s[~s.index.duplicated()].sort_index()
    except Exception:
        pass
    return out


def eckhardt_baseflow(series: dict) -> dict[str, float]:
    """Mean baseflow per gauge via the PyPI `baseflow` package (Eckhardt
    digital filter — the top-KGE method on our test gauges). Standard
    separation replaces the earlier p25-percentile proxy."""
    import pandas as pd
    import baseflow as bfl
    if not series:
        return {}
    df = pd.concat(series.values(), axis=1, keys=list(series.keys()))
    df = df.interpolate(limit=5)                     # bridge small gaps only
    res = bfl.separation(df, method=["Eckhardt"])
    if isinstance(res, tuple):
        res = res[0]
    bf = res["Eckhardt"] if isinstance(res, dict) else res
    out = {}
    for c in bf.columns:
        m = float(np.nanmean(np.asarray(bf[c], dtype="float64")))
        if np.isfinite(m) and m >= 0:
            out[str(c)] = m
    return out


def fetch_annual_peaks(site: str) -> np.ndarray:
    """Annual peak discharges (cfs) from the NWIS peak service, [] on failure."""
    try:
        r = requests.get("https://nwis.waterdata.usgs.gov/nwis/peak",
                         params={"site_no": site, "agency_cd": "USGS",
                                 "format": "rdb"}, timeout=45)
        r.raise_for_status()
        vals = []
        header = None
        for ln in r.text.splitlines():
            if ln.startswith("#") or not ln.strip():
                continue
            parts = ln.split("\t")
            if header is None:
                header = parts
                i_va = header.index("peak_va") if "peak_va" in header else None
                continue
            if parts[0].endswith("s"):                 # rdb width row (e.g. "5s")
                continue
            if i_va is None or i_va >= len(parts):
                continue
            try:
                v = float(parts[i_va])
            except ValueError:
                continue
            if v > 0:
                vals.append(v)
        return np.asarray(vals, dtype="float64")
    except Exception:
        return np.asarray([], dtype="float64")


def lp3_quantile(peaks_cfs: np.ndarray, T: int) -> float:
    """Log-Pearson III T-year quantile (m3/s); Wilson-Hilferty K from skew."""
    x = np.log10(peaks_cfs)
    n = len(x)
    m, s = x.mean(), x.std(ddof=1)
    if s <= 0:
        return float(peaks_cfs.max() * CFS_TO_CMS)
    g = (n * np.sum((x - m) ** 3)) / ((n - 1) * (n - 2) * s ** 3)
    g = float(np.clip(g, -3.0, 3.0))
    z = _Z[T]
    if abs(g) < 1e-3:
        k = z
    else:
        k = (2.0 / g) * ((1.0 + g * z / 6.0 - g * g / 36.0) ** 3 - 1.0)
    return float(10.0 ** (m + k * s) * CFS_TO_CMS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-peaks", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    token = hf_token()

    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    cat = pq.read_table(hf_hub_download(HF_REPO, "gauges/gagesII_9322.parquet",
                                        repo_type="dataset", token=token))
    gids = [str(s).zfill(8) for s in cat.column("STAID").to_pylist()]
    if args.limit:
        gids = gids[:args.limit]

    done = [0]

    def one(gid):
        peaks = fetch_annual_peaks(gid)
        done[0] += 1
        if done[0] % 500 == 0:
            print(f"  {done[0]}/{len(gids)} gauges...", flush=True)
        if len(peaks) < args.min_peaks:
            return gid, float("nan"), float("nan"), float("nan"), len(peaks)
        return (gid, lp3_quantile(peaks, 10), lp3_quantile(peaks, 5),
                lp3_quantile(peaks, 2), len(peaks))

    with ThreadPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(one, gids))

    print("baseflow: fetching 3-yr daily flows + Eckhardt separation...", flush=True)
    os.environ.setdefault("TQDM_DISABLE", "1")       # quiet the package's bars
    base: dict[str, float] = {}
    chunks = [gids[i:i + 100] for i in range(0, len(gids), 100)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        for series in ex.map(fetch_dv_chunk, chunks):
            try:                                     # numba filter: main thread
                base.update(eckhardt_baseflow(series))
            except Exception as e:
                print(f"  baseflow chunk failed: {e}")

    q10 = np.array([r[1] for r in rows], "float32")
    ok = np.isfinite(q10)
    print(f"thresholds: {ok.sum()}/{len(gids)} gauges with >= {args.min_peaks} "
          f"annual peaks, {len(base)} with baseflow | Q10 median "
          f"{np.nanmedian(q10):.1f} m3/s")
    if args.dry_run:
        for r in rows[:8]:
            print("  ", r, "base:", round(base.get(r[0], float('nan')), 2))
        return 0

    tbl = pa.table({"gid": [r[0] for r in rows], "q10_cms": q10,
                    "q5_cms": np.array([r[2] for r in rows], "float32"),
                    "q2_cms": np.array([r[3] for r in rows], "float32"),
                    "qbase_cms": np.array([base.get(r[0], float("nan"))
                                           for r in rows], "float32"),
                    "n_peaks": np.array([r[4] for r in rows], "int32")})
    tmp = os.path.join(tempfile.mkdtemp(), "flood_thresholds.parquet")
    pq.write_table(tbl, tmp, compression="zstd")
    HfApi(token=token).create_commit(
        repo_id=HF_REPO, repo_type="dataset",
        operations=[CommitOperationAdd(OUT_PATH, tmp)],
        commit_message=f"flood thresholds: LP3 Q10/Q2 for {int(ok.sum())} gauges")
    print(f"uploaded {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

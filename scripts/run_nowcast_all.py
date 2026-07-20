"""Hourly AI-nowcast precompute: DI-LSTM predictions for every CONUS gauge.

Runs on the CREST_updater Space as feed "nowcast", right after each
mrms_recent Pass1 harvest, so precomputed nowcasts are always ready when a
dashboard user flips to Nowcast mode. Per run:

  1. issue time t0 = newest hour in CREST_data mrms_recent/ (Pass1, ~1-2 h
     behind real time); the 72-h basin-mean precip window ends at t0.
     Window hours missing from mrms_recent are filled from the Pass2 month
     tars (better quality, more lag) — same splice the nowcast design uses.
  2. basin-mean precip for all ~9k GAGES-II CONUS gauges per hour via a
     summed-area table over each grid (one pass serves every gauge); results
     cached in nowcast/precip_cache.parquet so steady-state runs only
     compute the 1-2 genuinely new hours.
  3. latest USGS discharge for all gauges (batched NWIS, 100 sites/request,
     threaded); a gauge with no/stale obs still gets a prediction — the
     model's obs-age channel was trained for exactly that.
  4. one batched forward pass of the DI-LSTM (CPU, seconds), inverse
     transform, and upload of nowcast/latest.parquet (+ refreshed precip
     cache) to vincewin/CREST_data.

The DILSTM/feature code below is a deliberate minimal copy of
nowcast_space/model.py (KEEP IN SYNC — same checkpoint format).

    python scripts/run_nowcast_all.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import tarfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import HF_REPO, hf_token                             # noqa: E402

MODEL_REPO = "vincewin/CREST_nowcast_model"
RECENT_PREFIX = "mrms_recent/"
MRMS_GRID = (-130.0, 20.0, 0.01, 3500, 7000, -9999.0)   # xll, yll, cell, nr, nc, nodata
L, H = 72, 6
CFS_TO_CMS = 0.0283168
CACHE_PATH = "nowcast/precip_cache.parquet"
LATEST_PATH = "nowcast/latest.parquet"


# ---- model (minimal copy of nowcast_space/model.py — KEEP IN SYNC) -----------
def _model_and_stats(token):
    import torch
    import torch.nn as nn

    class DILSTM(nn.Module):
        def __init__(self, n_feat=4, hidden=128, layers=2, horizon=H):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                                batch_first=True, dropout=0.1 if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, horizon)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    from huggingface_hub import hf_hub_download
    p = hf_hub_download(MODEL_REPO, "dilstm.pt", repo_type="model", token=token)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = DILSTM()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


# ---- precip: summed-area-table box means -------------------------------------
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
    """Per-gauge (r0, r1, c0, c1) slices on the MRMS grid; r1/c1 exclusive."""
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
    """Mean of valid (>=0) cells inside every gauge box, via summed-area tables."""
    r0, r1, c0, c1 = boxes
    valid = a >= 0.0                       # excludes -3 no-coverage and -9999
    S = np.zeros((a.shape[0] + 1, a.shape[1] + 1), "float64")
    C = np.zeros_like(S)
    np.cumsum(np.cumsum(np.where(valid, a, 0), 0), 1, out=S[1:, 1:])
    np.cumsum(np.cumsum(valid, 0), 1, out=C[1:, 1:])
    tot = S[r1, c1] - S[r0, c1] - S[r1, c0] + S[r0, c0]
    cnt = C[r1, c1] - C[r0, c1] - C[r1, c0] + C[r0, c0]
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype("float32")


def _recent_hours(files) -> dict[datetime, str]:
    out = {}
    for f in files:
        if not (f.startswith(RECENT_PREFIX) and f.endswith(".pqf")):
            continue
        digits = "".join(ch for ch in os.path.basename(f) if ch.isdigit())[-10:]
        try:
            out[datetime.strptime(digits, "%Y%m%d%H")] = f
        except ValueError:
            pass
    return out


def _pass2_member(t: datetime, token) -> bytes | None:
    """One hour out of the Pass2 month tar (HF-cached; weekly-updated)."""
    from huggingface_hub import hf_hub_download
    try:
        local = hf_hub_download(HF_REPO, f"mrms/{t.year}/mrms_{t.year}_{t.month:02d}.tar",
                                repo_type="dataset", token=token)
        with tarfile.open(local) as tf:
            for name in (f"mrms_corr_{t:%Y%m%d%H}.pqf", f"mrms_{t:%Y%m%d%H}.pqf"):
                try:
                    return tf.extractfile(name).read()
                except KeyError:
                    continue
    except Exception:
        pass
    return None


# ---- USGS obs (batched NWIS) -------------------------------------------------
def _fetch_obs_chunk(sites: list[str], t_start: datetime) -> dict[str, list]:
    out: dict[str, list] = {}
    try:
        r = requests.get("https://waterservices.usgs.gov/nwis/iv/",
                         params={"sites": ",".join(sites), "parameterCd": "00060",
                                 "format": "json", "siteStatus": "all",
                                 "startDT": t_start.strftime("%Y-%m-%dT%H:%MZ")},
                         timeout=60)
        r.raise_for_status()
        for ts in r.json().get("value", {}).get("timeSeries", []):
            sid = ts["sourceInfo"]["siteCode"][0]["value"].zfill(8)
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
                out[sid] = sorted(rows)
    except Exception:
        pass                                            # chunk lost -> stale-obs path
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="gauge cap (debug)")
    args = ap.parse_args()
    token = hf_token()

    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    api = HfApi(token=token)

    # -- catalog, clipped to the MRMS grid (drops AK/HI/PR) --------------------
    cat = pq.read_table(hf_hub_download(HF_REPO, "gauges/gagesII_9322.parquet",
                                        repo_type="dataset", token=token))
    gid = np.array([str(s).zfill(8) for s in cat.column("STAID").to_pylist()])
    lat = cat.column("LAT_GAGE").to_numpy().astype("float64")
    lon = cat.column("LNG_GAGE").to_numpy().astype("float64")
    area = cat.column("DRAIN_SQKM").to_numpy().astype("float64")
    xll, yll, cell, nr, nc, _ = MRMS_GRID
    inside = (lon > xll) & (lon < xll + nc * cell) & (lat > yll) & (lat < yll + nr * cell)
    gid, lat, lon, area = gid[inside], lat[inside], lon[inside], area[inside]
    if args.limit:
        gid, lat, lon, area = gid[:args.limit], lat[:args.limit], lon[:args.limit], area[:args.limit]
    boxes = _grid_boxes(lon, lat, area)

    # -- precip window: mrms_recent first, Pass2 tar fallback ------------------
    files = api.list_repo_files(HF_REPO, repo_type="dataset")
    recent = _recent_hours(files)
    if not recent:
        print("nowcast: no mrms_recent hours in store — cannot set issue time")
        return 1
    t0 = max(recent)
    hours = [t0 - timedelta(hours=i) for i in range(L - 1, -1, -1)]

    cached: dict[str, np.ndarray] = {}
    cache_gids = None
    try:
        ct = pq.read_table(hf_hub_download(HF_REPO, CACHE_PATH, repo_type="dataset",
                                           token=token, force_download=True))
        cache_gids = np.array(ct.column("gid").to_pylist())
        if cache_gids.shape == gid.shape and (cache_gids == gid).all():
            cached = {name: ct.column(name).to_numpy().astype("float32")
                      for name in ct.schema.names if name.startswith("h")}
        else:
            cached = {}
    except Exception:
        cached = {}

    pmat = np.full((len(gid), L), np.nan, "float32")
    computed, from_pass2, missing = 0, 0, []
    for i, t in enumerate(hours):
        key = f"h{t:%Y%m%d%H}"
        if key in cached:
            pmat[:, i] = cached[key]
            continue
        data = None
        if t in recent:
            try:
                p = hf_hub_download(HF_REPO, recent[t], repo_type="dataset", token=token)
                data = open(p, "rb").read()
            except Exception:
                data = None
        if data is None:
            data = _pass2_member(t, token)
            if data is not None:
                from_pass2 += 1
        if data is None:
            missing.append(t)
            continue
        a = _read_pqf_bytes(data)
        if a is None:
            missing.append(t)
            continue
        pmat[:, i] = _box_means(a, boxes)
        cached[f"h{t:%Y%m%d%H}"] = pmat[:, i].copy()
        computed += 1

    # -- obs -------------------------------------------------------------------
    chunks = [list(gid[i:i + 100]) for i in range(0, len(gid), 100)]
    obs: dict[str, list] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for got in ex.map(lambda c: _fetch_obs_chunk(c, t0 - timedelta(hours=L)), chunks):
            obs.update(got)

    # -- features + batched inference ------------------------------------------
    model, ck = _model_and_stats(token)
    stats = ck["stats"]
    la = ((np.log10(np.maximum(area, 1.0)) - stats["la_mean"])
          / max(stats["la_std"], 1e-6)).astype("float32")
    feat = np.zeros((len(gid), L, 4), "float32")
    feat[:, :, 0] = np.log1p(np.nan_to_num(np.maximum(pmat, 0.0)))
    feat[:, :, 3] = la[:, None]
    obs_last_q = np.full(len(gid), np.nan, "float32")
    obs_age = np.full(len(gid), 999.0, "float32")
    obs_last_t = [""] * len(gid)
    for g in range(len(gid)):
        rows = obs.get(gid[g])
        if not rows:
            feat[g, :, 2] = 999.0 / 24.0
            continue
        j = -1
        for i, t in enumerate(hours):
            while j + 1 < len(rows) and rows[j + 1][0] <= t:
                j += 1
            if j >= 0:
                feat[g, i, 1] = math.log1p(max(rows[j][1], 0.0))
                feat[g, i, 2] = (t - rows[j][0]).total_seconds() / 3600.0 / 24.0
            else:
                feat[g, i, 2] = 999.0 / 24.0
        if j >= 0:
            obs_last_q[g] = rows[j][1]
            obs_age[g] = (t0 - rows[j][0]).total_seconds() / 3600.0
            obs_last_t[g] = rows[j][0].strftime("%Y-%m-%d %H:%M")

    import torch
    preds = np.zeros((len(gid), H), "float32")
    with torch.no_grad():
        for i in range(0, len(gid), 2048):
            y = model(torch.from_numpy(feat[i:i + 2048]))
            preds[i:i + 2048] = np.maximum(np.expm1(y.numpy()), 0.0)

    # -- outputs ---------------------------------------------------------------
    md = {b"t0": t0.strftime("%Y-%m-%d %H:00 UTC").encode(),
          b"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC").encode(),
          b"model_epoch": str(ck.get("epoch")).encode(),
          b"model_val_nse": str(ck.get("val_nse")).encode(),
          b"model_when": str(ck.get("when")).encode()}
    cols = {"gid": gid.tolist(), "lat": lat.astype("float32"),
            "lon": lon.astype("float32"), "area_km2": area.astype("float32"),
            "obs_last_time": obs_last_t, "obs_last_q": obs_last_q,
            "obs_age_h": obs_age}
    for k in range(H):
        cols[f"q{k + 1}"] = preds[:, k]
    latest = pa.table(cols).replace_schema_metadata(md)

    keep = {f"h{t:%Y%m%d%H}" for t in hours}
    ccols = {"gid": gid.tolist()}
    ccols.update({k: v for k, v in sorted(cached.items()) if k in keep})
    cache_tbl = pa.table(ccols)

    n_obs_fresh = int((obs_age <= 6).sum())
    summary = (f"nowcast: t0 {t0:%Y-%m-%d %H:00} UTC | {len(gid)} gauges "
               f"({n_obs_fresh} with obs <=6 h old) | precip hours: "
               f"{sum(1 for t in hours if f'h{t:%Y%m%d%H}' in cached)}/{L} "
               f"({computed} new, {from_pass2} via Pass2, {len(missing)} missing) | "
               f"model epoch {ck.get('epoch')} val_nse {ck.get('val_nse')}")
    if args.dry_run:
        print(summary + " [dry-run: no upload]")
        return 0

    tmp = tempfile.mkdtemp()
    lp = os.path.join(tmp, "latest.parquet")
    cp = os.path.join(tmp, "precip_cache.parquet")
    pq.write_table(latest, lp, compression="zstd")
    pq.write_table(cache_tbl, cp, compression="zstd")
    api.create_commit(repo_id=HF_REPO, repo_type="dataset",
                      operations=[CommitOperationAdd(LATEST_PATH, lp),
                                  CommitOperationAdd(CACHE_PATH, cp)],
                      commit_message=f"nowcast {t0:%Y%m%d%H}: {len(gid)} gauges")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

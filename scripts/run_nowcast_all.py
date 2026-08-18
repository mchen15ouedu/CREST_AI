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
DATA_REPO = "vincewin/CREST_nowcast_data"
# v1 serves gauges with a recent observation; the v3 companions (static
# catchment attributes + per-gauge NSE loss, trained on all 6,036 gauges —
# the *_full checkpoints, 2026-08-14) serve gauges whose obs are stale or
# absent AND every ungauged virtual point (_virtual_v3, read by the dashboard's
# ungaugednow_store). The archive-replay evals (2026-08-12/14) showed v3 is
# the only model with positive skill without fresh obs, while v1 still wins
# when obs are fresh. Delete the v3 files from the model repo to disable
# routing (pure v1; the dashboard then falls back to the EF5 routed feed for
# the virtual points) — nothing else to redeploy.
MODEL_FILES = (("dilstm.pt",),            # 6-h (risk basis)
               ("dilstm_h12.pt",))        # 12-h
V3_FILES = ("dilstm_v3_full.pt", "dilstm_h12_v3_full.pt")
ROUTE_AGE_H = 6.0                         # obs older than this -> v3 (fallback rule)
# Per-gauge winner table (all-gauge archive replay, 2026-08-17): for every
# gauge with a decided winner_6 / winner_12 the better model serves that
# horizon family regardless of obs age; undecided / unlisted gauges fall
# back to the obs-age rule. Delete the file from the model repo to revert
# to pure obs-age routing.
ROUTE_TABLE = "route_pergauge_v1_v3full.parquet"
# Gauge-health hysteresis (2026-08-18): v3 runs everywhere; v1 is DISPLAYED at a
# gauge only while (a) the table says v1 is the better model there with fresh
# obs and (b) the gauge is "healthy": no obs gap longer than GAP_H at any run
# for PROBATION_D consecutive days. A gap (>GAP_H at two consecutive runs — a
# single run can be a lost fetch chunk) puts the gauge on v3 immediately; it
# returns to v1 only after a full clean month. State: STATE_PATH (clean_since
# per gauge), rewritten hourly next to latest.parquet.
GAP_H = 12.0
PROBATION_D = 30
STATE_PATH = "nowcast/route_state.parquet"
NET_OUTAGE_FRAC = 0.8                     # >80 % of gauges dark = our fetch failed, not theirs
RECENT_PREFIX = "mrms_recent/"
MRMS_GRID = (-130.0, 20.0, 0.01, 3500, 7000, -9999.0)   # xll, yll, cell, nr, nc, nodata
L, H = 72, 6                      # lookback; H is only the horizon fallback
AGE_CAP_H = 240.0                 # feat_version 2: older obs counts as missing
CFS_TO_CMS = 0.0283168
CACHE_PATH = "nowcast/precip_cache.parquet"
LATEST_PATH = "nowcast/latest.parquet"


# ---- model (minimal copy of nowcast_space/model.py — KEEP IN SYNC) -----------
def _model_and_stats(token, fnames):
    import torch
    import torch.nn as nn

    class DILSTM(nn.Module):
        def __init__(self, n_feat=4, hidden=128, layers=2, horizon=H):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                                batch_first=True, dropout=0.1 if layers > 1 else 0.0)
            self.drop = nn.Dropout(0.1)          # parameter-free; MC-dropout hook
            self.head = nn.Linear(hidden, horizon)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(self.drop(out[:, -1]))

    from huggingface_hub import hf_hub_download
    p, fname = None, None
    for cand in fnames:
        try:
            p = hf_hub_download(MODEL_REPO, cand, repo_type="model", token=token)
            fname = cand
            break
        except Exception:
            continue
    if p is None:
        raise FileNotFoundError(f"none of {fnames} in {MODEL_REPO}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = DILSTM(n_feat=int(ck.get("n_feat", 4)), horizon=int(ck.get("horizon", H)))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck, fname


def _statics_z(token, ck, gid, fname="gauges/gauge_statics.parquet",
               id_col="STAID", pad8=True):
    """Per-point z-scored static-attribute vectors for a feat_version-3
    checkpoint, normalized with the stats stored in that checkpoint; points
    missing from the table (or with NaN attributes) get the training median."""
    from huggingface_hub import hf_hub_download
    st = ck["stats"]
    mu = np.asarray(st["static_mean"], "float64")
    sd = np.asarray(st["static_std"], "float64")
    med = np.asarray(st["static_median"], "float64")
    t = pq.read_table(hf_hub_download(DATA_REPO, fname,
                                      repo_type="dataset", token=token))
    cols = [c for c in t.schema.names if c not in (id_col, "hybas_id")]
    ids = [str(s).zfill(8) if pad8 else str(s)
           for s in t.column(id_col).to_pylist()]
    raw = np.column_stack([t.column(c).to_numpy(zero_copy_only=False)
                           for c in cols]).astype("float64")
    lut = {g: raw[i] for i, g in enumerate(ids)}
    out = np.empty((len(gid), len(mu)), "float32")
    for i, g in enumerate(gid):
        r = lut.get(g)
        r = med if r is None else np.where(np.isfinite(r), r, med)
        out[i] = (r - mu) / sd
    return out


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


# ---- DI-LSTM v3 at the ungauged virtual points ------------------------------
VP_LATEST = "nowcast/v3_virtual_latest.parquet"
VP_CACHE = "nowcast/v3_virtual_precip_cache.parquet"
VP_ARCHIVE = "nowcast/v3_virtual_archive/"
VP_HIST_H = 168                           # rolling analysis history kept in `hist`


def _vp_history(token, t0, files, vid, prev_hist, prev_t0, prev_q1):
    """Rolling 7-day series of v3's own 1-h-lead prediction per point — the
    model's hourly *analysis* of the flow (there are no observations at these
    points), plotted solid up to t0 ahead of the dashed forecast. Extends the
    previous file's `hist`; when there is none (first run after deploy) it is
    back-filled once from the hourly archive files."""
    import json
    from huggingface_hub import hf_hub_download

    fmt = "%Y-%m-%d %H:%M"
    hist = {v: {} for v in vid}                    # vp -> {time: q}
    if prev_hist is not None:
        for v, h in zip(prev_hist["vp"], prev_hist["hist"]):
            if v in hist:
                try:
                    hist[v].update({t: q for t, q in json.loads(h)})
                except Exception:
                    pass
        if prev_t0 is not None and prev_q1 is not None:
            when = (prev_t0 + timedelta(hours=1)).strftime(fmt)
            for v, q in zip(prev_hist["vp"], prev_q1):
                if v in hist and np.isfinite(q):
                    hist[v][when] = round(float(q), 3)
    else:                                            # one-off back-fill
        want = {f"{VP_ARCHIVE}{t:%Y%m}/vp_{t:%Y%m%d%H}.parquet": t
                for t in (t0 - timedelta(hours=k) for k in range(1, VP_HIST_H + 1))}
        for path in sorted(f for f in files if f in want):
            try:
                tt = pq.read_table(hf_hub_download(HF_REPO, path, repo_type="dataset",
                                                   token=token),
                                   columns=["vp", "q1"])
            except Exception:
                continue
            when = (want[path] + timedelta(hours=1)).strftime(fmt)
            for v, q in zip(tt.column("vp").to_pylist(), tt.column("q1").to_numpy()):
                if str(v) in hist and np.isfinite(q):
                    hist[str(v)][when] = round(float(q), 3)
    lo = (t0 - timedelta(hours=VP_HIST_H)).strftime(fmt)
    hi = t0.strftime(fmt)
    return [json.dumps([[t, q] for t, q in sorted(hist[v].items()) if lo < t <= hi],
                       separators=(",", ":")) for v in vid]


def _virtual_v3(token, t0, hours, recent, files, model3, ck3, model3_12, ck3_12,
                mfile3=None, mfile3_12=None):
    """v3 predictions for the 2,676 virtual pour points — the permanent
    no-obs regime the model was built for. This file IS the dashboard's
    ungauged nowcast (hf_data/ungaugednow_store prefers it over the EF5
    routed feed): q1..q6 from the 6-h model, q12_1..q12_12 from the 12-h
    model, `hist` = the rolling 7-day series of v3's own 1-h-lead analysis.
    The hourly archive copy also carries a compact snapshot of
    ungauged_latest's EF5 q columns, so v3 vs routed physics stays comparable.

    Returns (commit_ops, summary_note). Isolated from the gauged feed — any
    failure here is caught by the caller.
    """
    import tempfile
    import torch
    from huggingface_hub import hf_hub_download, CommitOperationAdd

    vt = pq.read_table(hf_hub_download(HF_REPO, "gauges/virtual_points.parquet",
                                       repo_type="dataset", token=token))
    vid = [str(s) for s in vt.column("vp").to_pylist()]
    vlat = vt.column("lat").to_numpy().astype("float64")
    vlon = vt.column("lon").to_numpy().astype("float64")
    varea = vt.column("area_km2").to_numpy().astype("float64")
    boxes = _grid_boxes(vlon, vlat, varea)

    cached: dict[str, np.ndarray] = {}
    try:
        ct = pq.read_table(hf_hub_download(HF_REPO, VP_CACHE, repo_type="dataset",
                                           token=token, force_download=True))
        if [str(s) for s in ct.column("vp").to_pylist()] == vid:
            cached = {n: ct.column(n).to_numpy().astype("float32")
                      for n in ct.schema.names if n.startswith("h")}
    except Exception:
        cached = {}

    pmat_v = np.full((len(vid), L), np.nan, "float32")
    for i, t in enumerate(hours):
        key = f"h{t:%Y%m%d%H}"
        if key in cached:
            pmat_v[:, i] = cached[key]
            continue
        data = None
        if t in recent:
            try:
                p = hf_hub_download(HF_REPO, recent[t], repo_type="dataset",
                                    token=token)
                data = open(p, "rb").read()
            except Exception:
                data = None
        if data is None:
            data = _pass2_member(t, token)
        if data is None:
            continue
        a = _read_pqf_bytes(data)
        if a is None:
            continue
        pmat_v[:, i] = _box_means(a, boxes)
        cached[key] = pmat_v[:, i].copy()

    # features: precip + statics; obs channels pinned to the missing state
    sz = _statics_z(token, ck3, vid, "gauges/virtual_statics.parquet",
                    "vp", pad8=False)
    st = ck3["stats"]
    la = ((np.log10(np.maximum(varea, 1.0)) - st["la_mean"])
          / max(st["la_std"], 1e-6)).astype("float32")
    f = np.zeros((len(vid), L, 5 + sz.shape[1]), "float32")
    f[:, :, 0] = np.log1p(np.nan_to_num(np.maximum(pmat_v, 0.0)))
    f[:, :, 2] = AGE_CAP_H / 24.0
    f[:, :, 3] = la[:, None]
    f[:, :, 4] = 1.0
    f[:, :, 5:] = sz[:, None, :]

    def _inf(m, hor):
        out = np.zeros((len(vid), hor), "float32")
        with torch.no_grad():
            for i in range(0, len(vid), 2048):
                y = m(torch.from_numpy(f[i:i + 2048]))
                out[i:i + 2048] = np.maximum(np.expm1(y.numpy()), 0.0)
        return out

    hor = int(ck3.get("horizon", H))
    hor12 = int(ck3_12.get("horizon", 12))
    pv, pv12 = _inf(model3, hor), _inf(model3_12, hor12)

    # rolling analysis history: extend the previous file's hist with its q1
    prev_hist, prev_t0, prev_q1 = None, None, None
    try:
        pt = pq.read_table(hf_hub_download(HF_REPO, VP_LATEST, repo_type="dataset",
                                           token=token, force_download=True))
        if "hist" in pt.schema.names:
            prev_hist = {"vp": [str(v) for v in pt.column("vp").to_pylist()],
                         "hist": pt.column("hist").to_pylist()}
            prev_q1 = pt.column("q1").to_numpy().astype("float64")
            pm = pt.schema.metadata or {}
            prev_t0 = datetime.strptime(pm[b"t0"].decode(), "%Y-%m-%d %H:%M UTC")
            if prev_t0 >= t0:                # same hour re-run: keep hist as is
                prev_t0, prev_q1 = None, None
    except Exception:
        prev_hist = None
    try:
        hist_col = _vp_history(token, t0, files, vid, prev_hist, prev_t0, prev_q1)
    except Exception as e:
        print(f"nowcast: v3 virtual history failed ({e}) — empty hist")
        hist_col = ["[]"] * len(vid)

    # compact EF5 snapshot from the sharded routed feed (its t0 may lag ours)
    ef5: dict[str, np.ndarray] = {}
    ef5_t0 = ""
    ef5_cols: list[str] = []
    for shard in [x for x in files if x.startswith("nowcast/ungauged_latest")]:
        try:
            p = hf_hub_download(HF_REPO, shard, repo_type="dataset", token=token,
                                force_download=True)
            tt = pq.read_table(p)
            qc = sorted((c for c in tt.schema.names if c.startswith("q")
                         and c[1:].isdigit()), key=lambda c: int(c[1:]))
            ef5_cols = ef5_cols or qc
            qs = np.column_stack([tt.column(c).to_numpy(zero_copy_only=False)
                                  for c in qc]).astype("float32")
            for i2, v in enumerate(tt.column("vp").to_pylist()):
                ef5[str(v)] = qs[i2]
            meta = tt.schema.metadata or {}
            ef5_t0 = ef5_t0 or meta.get(b"t0", b"").decode()
        except Exception:
            continue

    md = {b"t0": t0.strftime("%Y-%m-%d %H:00 UTC").encode(),
          b"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC").encode(),
          b"source": b"dilstm_v3",
          b"experiment": b"dilstm_v3_virtual_points",
          b"model_file": (mfile3 or V3_FILES[0]).encode(),
          b"model_epoch": str(ck3.get("epoch")).encode(),
          b"model_val_nse": str(ck3.get("val_nse")).encode(),
          b"model12_file": (mfile3_12 or V3_FILES[1]).encode(),
          b"model12_epoch": str(ck3_12.get("epoch")).encode(),
          b"model12_val_nse": str(ck3_12.get("val_nse")).encode(),
          b"horizon": str(hor).encode(), b"horizon12": str(hor12).encode(),
          b"n_points": str(len(vid)).encode(),
          b"ef5_t0": ef5_t0.encode(),
          b"note": b"obs channels forced to the no-obs state; hist = v3's own "
                   b"1-h-lead analysis series; compare ef5_q* (routed physics) "
                   b"vs q* (DI-LSTM v3)"}
    cols = {"vp": vid, "lat": vlat.astype("float32"),
            "lon": vlon.astype("float32"), "area_km2": varea.astype("float32")}
    for k in range(hor):
        cols[f"q{k + 1}"] = pv[:, k]
    for k in range(hor12):
        cols[f"q12_{k + 1}"] = pv12[:, k]
    cols["hist"] = hist_col
    for j, c in enumerate(ef5_cols):
        cols[f"ef5_{c}"] = np.stack([ef5[v][j] if v in ef5 else np.nan
                                     for v in vid]).astype("float32")
    latest_v = pa.table(cols).replace_schema_metadata(md)

    keep = {f"h{t:%Y%m%d%H}" for t in hours}
    ccols = {"vp": vid}
    ccols.update({k: v for k, v in sorted(cached.items()) if k in keep})

    tmp = tempfile.mkdtemp()
    lp = os.path.join(tmp, "v3_virtual_latest.parquet")
    cp = os.path.join(tmp, "v3_virtual_precip_cache.parquet")
    pq.write_table(latest_v, lp, compression="zstd")
    pq.write_table(pa.table(ccols), cp, compression="zstd")
    arch = f"nowcast/v3_virtual_archive/{t0:%Y%m}/vp_{t0:%Y%m%d%H}.parquet"
    ops = [CommitOperationAdd(VP_LATEST, lp), CommitOperationAdd(VP_CACHE, cp),
           CommitOperationAdd(arch, lp)]
    note = (f" | v3-virtual {len(vid)} pts"
            f" (ef5 snapshot {len(ef5)}{', t0 ' + ef5_t0 if ef5_t0 else ''})")
    return ops, note


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
    model, ck, mfile = _model_and_stats(token, MODEL_FILES[0])
    model12, ck12, mfile12 = _model_and_stats(token, MODEL_FILES[1])

    # raw obs arrays once; per-checkpoint feature layouts assembled from them
    # (during the v1->v2 transition the two checkpoints can differ in
    # feat_version AND stats)
    obs_val = np.zeros((len(gid), L), "float32")         # last-known Q, m3/s
    age_h = np.full((len(gid), L), 999.0, "float32")     # hours since that obs
    obs_last_q = np.full(len(gid), np.nan, "float32")
    obs_age = np.full(len(gid), 999.0, "float32")
    obs_last_t = [""] * len(gid)
    for g in range(len(gid)):
        rows = obs.get(gid[g])
        if not rows:
            continue
        j = -1
        for i, t in enumerate(hours):
            while j + 1 < len(rows) and rows[j + 1][0] <= t:
                j += 1
            if j >= 0:
                obs_val[g, i] = max(rows[j][1], 0.0)
                age_h[g, i] = (t - rows[j][0]).total_seconds() / 3600.0
        if j >= 0:
            obs_last_q[g] = rows[j][1]
            obs_age[g] = (t0 - rows[j][0]).total_seconds() / 3600.0
            obs_last_t[g] = rows[j][0].strftime("%Y-%m-%d %H:%M")

    def _assemble(c, statics=None):
        stats = c["stats"]
        la = ((np.log10(np.maximum(area, 1.0)) - stats["la_mean"])
              / max(stats["la_std"], 1e-6)).astype("float32")
        fv = int(c.get("feat_version", 1))
        nf = 4 if fv < 2 else (5 if fv < 3 else 5 + statics.shape[1])
        f = np.zeros((len(gid), L, nf), "float32")
        f[:, :, 0] = np.log1p(np.nan_to_num(np.maximum(pmat, 0.0)))
        f[:, :, 3] = la[:, None]
        if fv >= 2:
            miss = age_h > AGE_CAP_H
            f[:, :, 1] = np.where(miss, 0.0, np.log1p(obs_val))
            f[:, :, 2] = np.minimum(age_h, AGE_CAP_H) / 24.0
            f[:, :, 4] = miss.astype("float32")
            if fv >= 3:
                f[:, :, 5:] = statics[:, None, :]
        else:
            f[:, :, 1] = np.log1p(obs_val)
            f[:, :, 2] = age_h / 24.0
        return f

    import torch

    def _infer(m, feat, hor):
        out = np.zeros((len(gid), hor), "float32")
        with torch.no_grad():
            for i in range(0, len(gid), 2048):
                y = m(torch.from_numpy(feat[i:i + 2048]))
                out[i:i + 2048] = np.maximum(np.expm1(y.numpy()), 0.0)
        return out

    hor = int(ck.get("horizon", H))
    hor12 = int(ck12.get("horizon", 12))
    feat6 = _assemble(ck)
    preds = _infer(model, feat6, hor)
    preds12 = _infer(model12, _assemble(ck12), hor12)

    # ---- gauge health state (hysteresis) ------------------------------------
    # clean_since: "" = gap in progress, "YYYY-mm-dd HH:MM" = start of the
    # current clean streak, None = gauge unknown (starts probation now)
    prev_cs, prev_health = {}, {}
    try:
        stt = pq.read_table(hf_hub_download(HF_REPO, STATE_PATH, repo_type="dataset",
                                            token=token, force_download=True))
        for g, cs, hl in zip(stt.column("gid").to_pylist(), stt.column("clean_since").to_pylist(),
                             stt.column("obs_health").to_pylist()):
            prev_cs[str(g).zfill(8)] = cs
            prev_health[str(g).zfill(8)] = hl
    except Exception as e:
        print(f"nowcast: route state unavailable ({e}) — every gauge starts probation")
    dark = obs_age > GAP_H
    net_outage = bool(dark.mean() > NET_OUTAGE_FRAC)
    if net_outage:
        print(f"nowcast: {100 * dark.mean():.0f}% of gauges dark this run — treating as OUR "
              f"fetch outage; gauge health state frozen")
    t0s = t0.strftime("%Y-%m-%d %H:%M")
    clean_since, health = [], np.empty(len(gid), dtype=object)
    for i, g in enumerate(gid):
        cs = prev_cs.get(g, None)
        ph = prev_health.get(g, "")
        if dark[i]:
            if net_outage:
                health[i] = "suspect"                       # keep streak, no v1 this hour
            elif ph in ("suspect", "gap") or cs is None:
                cs, health[i] = "", "gap"                   # confirmed: >GAP_H twice in a row
            else:
                health[i] = "suspect"                       # first dark hour: maybe a lost chunk
        else:
            if not cs:                                      # "" or None: data (back) -> new streak
                cs = t0s
            days = (t0 - datetime.strptime(cs, "%Y-%m-%d %H:%M")).total_seconds() / 86400.0
            health[i] = "healthy" if days >= PROBATION_D else "probation"
        clean_since.append(cs if cs is not None else "")
    healthy_now = (health == "healthy")                     # implies not dark this hour

    # ---- routing: v1 displayed only where it wins with fresh obs AND the gauge
    # is healthy; everything else (v3 winners, probation, gap, dead, undecided) -> v3
    n_table = 0
    w6 = np.full(len(gid), "", dtype=object); w12 = w6.copy()
    try:
        p = hf_hub_download(MODEL_REPO, ROUTE_TABLE, repo_type="model", token=token)
        rt = pq.read_table(p)
        rcols = set(rt.column_names)
        g2i = {str(g).zfill(8): i for i, g in enumerate(rt.column("gid").to_pylist())}
        idx = np.array([g2i.get(g, -1) for g in gid])

        def _pick(fam):
            col = f"winner_{fam}_fresh" if f"winner_{fam}_fresh" in rcols else f"winner_{fam}"
            v = np.array([x or "" for x in rt.column(col).to_pylist()], dtype=object)
            w = np.full(len(gid), "", dtype=object)
            w[idx >= 0] = v[idx[idx >= 0]]
            return w
        w6, w12 = _pick("6"), _pick("12")
        n_table = int(((w6 != "") | (w12 != "")).sum())
        route = ~(healthy_now & (w6 == "v1"))
        route12 = ~(healthy_now & (w12 == "v1"))
    except Exception as e:
        print(f"nowcast: route table unavailable ({e}) — healthy gauges -> v1, others -> v3")
        route = route12 = ~healthy_now
    v3 = None
    try:
        model3, ck3, mfile3 = _model_and_stats(token, (V3_FILES[0],))
        model3_12, ck3_12, mfile3_12 = _model_and_stats(token, (V3_FILES[1],))
        if (int(ck3.get("horizon", H)) != hor
                or int(ck3_12.get("horizon", 12)) != hor12):
            raise ValueError("v3 horizon mismatch")
        sz = _statics_z(token, ck3, gid)
        feat6_v3 = _assemble(ck3, sz)
        preds[route] = _infer(model3, feat6_v3, hor)[route]
        preds12[route12] = _infer(model3_12, _assemble(ck3_12, sz), hor12)[route12]
        v3 = (model3, ck3, mfile3, model3_12, ck3_12, mfile3_12, feat6_v3)
    except Exception as e:
        route = route12 = np.zeros(len(gid), bool)
        n_table = 0
        print(f"nowcast: v3 routing unavailable ({e}) — serving pure v1")
    model_col = np.where(route, "v3", "v1")
    model12_col = np.where(route12, "v3", "v1")

    # optional MC-dropout spread on the risk-basis model: per-gauge mean CV of
    # the stochastic predictions — hallucinated peaks come with a huge spread
    mc_n = int(os.environ.get("NOWCAST_MC_N", "0"))
    mc_cv = None
    if mc_n > 1:
        model.train()                              # activates dropout
        stack = np.stack([_infer(model, feat6, hor) for _ in range(mc_n)])
        model.eval()
        if v3 is not None and route.any():
            model3.train()
            s3 = np.stack([_infer(model3, v3[6], hor) for _ in range(mc_n)])
            model3.eval()
            stack[:, route] = s3[:, route]
        with np.errstate(invalid="ignore"):
            mc_cv = (stack.std(0) / np.maximum(stack.mean(0), 0.1)).mean(1) \
                .astype("float32")

    # per-gauge validation skill (uploaded by train_hpc.py next to the ckpt)
    def _skill(fname):
        try:
            p = hf_hub_download(MODEL_REPO, f"skill_{os.path.splitext(fname)[0]}.parquet",
                                repo_type="model", token=token)
            t = pq.read_table(p)
            g2i = {g: i for i, g in enumerate(t.column("gid").to_pylist())}
            v = t.column("val_nse").to_numpy()
            vn = t.column("val_nse_noobs").to_numpy()
            a = np.full(len(gid), np.nan, "float32")
            b = np.full(len(gid), np.nan, "float32")
            for i, g in enumerate(gid):
                if g in g2i:
                    a[i], b[i] = v[g2i[g]], vn[g2i[g]]
            return a, b
        except Exception:
            return None, None

    skill_nse, skill_nse_noobs = _skill(mfile)
    if v3 is not None:
        s3, s3n = _skill(v3[2])
        if s3 is not None:
            if skill_nse is None:
                skill_nse = np.full(len(gid), np.nan, "float32")
                skill_nse_noobs = np.full(len(gid), np.nan, "float32")
            skill_nse[route] = s3[route]
            skill_nse_noobs[route] = s3n[route]

    # -- outputs ---------------------------------------------------------------
    md = {b"t0": t0.strftime("%Y-%m-%d %H:00 UTC").encode(),
          b"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC").encode(),
          b"model_file": mfile.encode(),
          b"model_feat_version": str(ck.get("feat_version", 1)).encode(),
          b"model_epoch": str(ck.get("epoch")).encode(),
          b"model_val_nse": str(ck.get("val_nse")).encode(),
          b"model_when": str(ck.get("when")).encode(),
          b"horizon": str(hor).encode(),
          b"model12_file": mfile12.encode(),
          b"model12_epoch": str(ck12.get("epoch")).encode(),
          b"model12_val_nse": str(ck12.get("val_nse")).encode(),
          b"model12_when": str(ck12.get("when")).encode(),
          b"horizon12": str(hor12).encode(),
          b"skill_table": (b"1" if skill_nse is not None else b"0"),
          b"mc_n": str(mc_n).encode()}
    if v3 is not None:
        md[b"route_rule"] = (
            f"v1 iff gauge healthy (no obs gap>{GAP_H:g}h for {PROBATION_D}d) AND fresh-obs "
            f"winner==v1 in {ROUTE_TABLE} ({n_table} gauges decided); else v3").encode()
        md[b"n_route_table"] = str(n_table).encode()
        md[b"n_route12_v3"] = str(int(route12.sum())).encode()
        md[b"n_healthy"] = str(int((health == "healthy").sum())).encode()
        md[b"n_probation"] = str(int((health == "probation").sum())).encode()
        md[b"n_gap"] = str(int((health == "gap").sum())).encode()
        md[b"n_suspect"] = str(int((health == "suspect").sum())).encode()
        md[b"net_outage"] = (b"1" if net_outage else b"0")
        md[b"model_v3_file"] = v3[2].encode()
        md[b"model_v3_epoch"] = str(v3[1].get("epoch")).encode()
        md[b"model_v3_val_nse"] = str(v3[1].get("val_nse")).encode()
        md[b"model12_v3_file"] = v3[5].encode()
        md[b"model12_v3_epoch"] = str(v3[4].get("epoch")).encode()
        md[b"n_route_v3"] = str(int(route.sum())).encode()
    cols = {"gid": gid.tolist(), "lat": lat.astype("float32"),
            "lon": lon.astype("float32"), "area_km2": area.astype("float32"),
            "obs_last_time": obs_last_t, "obs_last_q": obs_last_q,
            "obs_age_h": obs_age, "model": model_col.tolist(),
            "model12": model12_col.tolist(), "obs_health": health.tolist(),
            "obs_clean_since": clean_since}
    if skill_nse is not None:
        cols["val_nse_gauge"] = skill_nse
        cols["val_nse_noobs_gauge"] = skill_nse_noobs
    if mc_cv is not None:
        cols["mc_cv"] = mc_cv
    for k in range(hor):
        cols[f"q{k + 1}"] = preds[:, k]
    for k in range(hor12):                     # 12-h model, own column family
        cols[f"q12_{k + 1}"] = preds12[:, k]
    latest = pa.table(cols).replace_schema_metadata(md)

    keep = {f"h{t:%Y%m%d%H}" for t in hours}
    ccols = {"gid": gid.tolist()}
    ccols.update({k: v for k, v in sorted(cached.items()) if k in keep})
    cache_tbl = pa.table(ccols)

    vp_ops, vp_note = [], ""
    if v3 is not None:
        try:
            vp_ops, vp_note = _virtual_v3(token, t0, hours, recent, files,
                                          v3[0], v3[1], v3[3], v3[4],
                                          v3[2], v3[5])
        except Exception as e:
            vp_note = f" | v3-virtual FAILED ({e})"
            print(f"nowcast: v3 virtual-point nowcast failed: {e}")

    n_obs_fresh = int((obs_age <= 6).sum())
    hstr = (f" | v3-displayed {int(route.sum())} (6h) / {int(route12.sum())} (12h) "
            f"[healthy {int((health == 'healthy').sum())}, probation {int((health == 'probation').sum())}, "
            f"gap {int((health == 'gap').sum())}, suspect {int((health == 'suspect').sum())}; table {n_table}]")
    summary = (f"nowcast: t0 {t0:%Y-%m-%d %H:00} UTC | {len(gid)} gauges "
               f"({n_obs_fresh} with obs <=6 h old) | precip hours: "
               f"{sum(1 for t in hours if f'h{t:%Y%m%d%H}' in cached)}/{L} "
               f"({computed} new, {from_pass2} via Pass2, {len(missing)} missing) | "
               f"models {mfile} h{hor} e{ck.get('epoch')} nse {ck.get('val_nse')} + "
               f"{mfile12} h{hor12} e{ck12.get('epoch')} nse {ck12.get('val_nse')}"
               f"{hstr if v3 is not None else ' | pure v1 (no v3 routing)'}"
               f"{' | skill table' if skill_nse is not None else ''}"
               f"{f' | mc_n {mc_n}' if mc_cv is not None else ''}"
               + vp_note)
    if args.dry_run:
        print(summary + " [dry-run: no upload]")
        return 0

    tmp = tempfile.mkdtemp()
    lp = os.path.join(tmp, "latest.parquet")
    cp = os.path.join(tmp, "precip_cache.parquet")
    sp = os.path.join(tmp, "route_state.parquet")
    pq.write_table(latest, lp, compression="zstd")
    pq.write_table(cache_tbl, cp, compression="zstd")
    pq.write_table(pa.table({"gid": gid.tolist(), "clean_since": clean_since,
                             "obs_health": health.tolist()}).replace_schema_metadata(
        {b"t0": t0s.encode(), b"gap_h": str(GAP_H).encode(),
         b"probation_d": str(PROBATION_D).encode()}), sp, compression="zstd")
    # every issue is also archived (latest.parquet is overwritten hourly) so
    # forecasts can later be scored against the obs that actually arrived
    arch = f"nowcast/archive/{t0:%Y%m}/nc_{t0:%Y%m%d%H}.parquet"
    api.create_commit(repo_id=HF_REPO, repo_type="dataset",
                      operations=[CommitOperationAdd(LATEST_PATH, lp),
                                  CommitOperationAdd(CACHE_PATH, cp),
                                  CommitOperationAdd(STATE_PATH, sp),
                                  CommitOperationAdd(arch, lp)] + vp_ops,
                      commit_message=f"nowcast {t0:%Y%m%d%H}: {len(gid)} gauges"
                                     + (" + v3 virtual" if vp_ops else ""))
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

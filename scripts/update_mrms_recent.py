"""Rolling near-real-time MRMS store for AI nowcasting: MultiSensor Pass1.

The mrms/ month tars hold gauge-corrected Pass2, which the source posts ~2 h
late and our updater archives weekly — too stale for nowcasting "now". Pass1
(gauge-corrected, first pass) posts ~50 min after each hour, but it exists
ONLY on NCEP's real-time server (mtarchive does NOT mirror it) and NCEP keeps
a rolling ~25 hours. This updater therefore harvests Pass1 into loose
per-hour PQFs in vincewin/CREST_data:

    mrms_recent/mrms1h_pass1_YYYYMMDDHH.pqf     (same grid/format as the tars)

Loose files, not tars: a 6-hourly run adds ~6 x 1.4 MB files — no 0.5 GB
month-tar re-upload. Files older than --keep-days (21) are pruned in the same
commit; by then those hours are in the weekly Pass2 archive (better quality).
Hours that fall out of NCEP's window before a run sees them are lost from
Pass1 permanently (Pass2 covers them later); a >=6-hourly cadence keeps that
from ever happening in normal operation.

    python scripts/update_mrms_recent.py [--hours 27] [--keep-days 21] [--dry-run]
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import HF_REPO, hf_token                             # noqa: E402
from update_mrms import MRMS_GRID                                               # noqa: E402
from hf_data.forcing import _read_pqf, _write_pqf                               # noqa: E402

PRODUCT = "MultiSensor_QPE_01H_Pass1"
NCEP_URL = ("https://mrms.ncep.noaa.gov/2D/{prod}/"
            "MRMS_{prod}_00.00_{t:%Y%m%d}-{t:%H}0000.grib2.gz")
PREFIX = "mrms_recent/"
MEMBER_FMT = "mrms1h_pass1_%Y%m%d%H.pqf"

# -- browser-ready radar frames (Nowcast-mode rain overlay) -------------------
# Each stored Pass1 hour also gets a colormapped transparent PNG at 1/4 the
# grid (block-MAX pooled, so 1-km storm cores stay visible at ~4 km display
# resolution; ~1750 px across CONUS is at/above screen width at national
# zoom). ~160 KB per wet frame; a rolling week is ~30 MB — the frontend
# animates these straight off the HF CDN. Retention is 7 days BY DESIGN
# (user-set), shorter than the PQFs' 21: frames are display sugar, the PQFs
# are forcing.
FRAME_PREFIX = "mrms_frames/"
FRAME_FMT = "mrms1h_%Y%m%d%H.png"
FRAME_KEEP_DAYS = 7
FRAME_DS = 4
# rain-rate colormap (mm/h -> RGB); below RAIN_MIN renders fully transparent
RAIN_MIN = 0.1
RAIN_X = (0.1, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0)
RAIN_R = (150, 60, 255, 255, 230, 180, 120)
RAIN_G = (210, 170, 220, 150, 40, 20, 0)
RAIN_B = (150, 60, 0, 0, 30, 140, 120)


def render_frame(a: np.ndarray, nodata: float) -> bytes:
    """Colormapped RGBA PNG bytes for one Pass1 grid (dry = transparent)."""
    import io

    from PIL import Image
    a = np.where((a == nodata) | ~np.isfinite(a) | (a < 0), 0.0, a).astype("float32")
    nr, nc = a.shape
    a = a[: nr - nr % FRAME_DS, : nc - nc % FRAME_DS]
    a = a.reshape(a.shape[0] // FRAME_DS, FRAME_DS,
                  a.shape[1] // FRAME_DS, FRAME_DS).max(axis=(1, 3))
    r = np.interp(a, RAIN_X, RAIN_R).astype(np.uint8)
    g = np.interp(a, RAIN_X, RAIN_G).astype(np.uint8)
    b = np.interp(a, RAIN_X, RAIN_B).astype(np.uint8)
    frac = np.sqrt(np.clip(a / RAIN_X[-1], 0.0, 1.0))
    alpha = np.where(a >= RAIN_MIN, 140 + 80 * frac, 0.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.dstack([r, g, b, alpha])).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _utc_hour_floor() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, minute=0,
                                              second=0, microsecond=0)


def _parse_hour(path: str) -> datetime | None:
    stem = os.path.basename(path)
    digits = "".join(ch for ch in stem if ch.isdigit())[-10:]
    try:
        return datetime.strptime(digits, "%Y%m%d%H")
    except ValueError:
        return None


def fetch_hour(t: datetime) -> bytes | None:
    """One Pass1 hour from NCEP -> PQF bytes on the store grid, or None."""
    url = NCEP_URL.format(prod=PRODUCT, t=t)
    try:
        raw = urllib.request.urlopen(url, timeout=90).read()
    except Exception:
        return None                                   # not posted yet / aged out
    fd, gpath = tempfile.mkstemp(suffix=".grib2")
    os.close(fd)
    with open(gpath, "wb") as fh:
        fh.write(gzip.decompress(raw))
    import xarray as xr
    try:
        ds = xr.open_dataset(gpath, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = list(ds.data_vars)[0]
        a = np.asarray(ds[var].values, dtype="float32")   # row0=north, W->E
        ds.close()
    finally:
        for ext in ("", ".idx"):
            try:
                os.remove(gpath + ext)
            except OSError:
                pass
    xll, yll, cell, nr, nc, nod = MRMS_GRID
    if a.shape != (nr, nc):
        return None                                   # unexpected grid -> skip
    fd, buf = tempfile.mkstemp(suffix=".pqf")
    os.close(fd)
    try:
        _write_pqf(buf, a, xll, yll, cell, nod)
        with open(buf, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(buf)
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=27,
                    help="lookback window to fill (NCEP retains ~25 h)")
    ap.add_argument("--keep-days", type=int, default=21)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import (HfApi, CommitOperationAdd, CommitOperationDelete,
                                 hf_hub_download)
    api = HfApi(token=hf_token())
    all_files = api.list_repo_files(HF_REPO, repo_type="dataset")
    stored = [f for f in all_files if f.startswith(PREFIX)]
    have = {h for f in stored if (h := _parse_hour(f)) is not None}

    now = _utc_hour_floor()
    # newest first, so a partial/interrupted run still banks the freshest hours.
    # i starts at 0: the file valid at the CURRENT floor hour posts ~:50, so the
    # :58 self-run must try it — starting at 1 would delay every hour's file to
    # the NEXT run (a permanent extra hour of lag). Not-yet-posted just misses.
    want = [t for i in range(0, args.hours + 1)
            if (t := now - timedelta(hours=i)) not in have]

    ops, added, misses = [], [], 0
    pqf_bytes: dict[datetime, bytes] = {}         # fresh hours, for frame render
    for t in want:
        data = fetch_hour(t)
        if data is None:
            misses += 1
            continue
        ops.append(CommitOperationAdd(PREFIX + t.strftime(MEMBER_FMT), data))
        added.append(t)
        pqf_bytes[t] = data

    cutoff = now - timedelta(days=args.keep_days)
    stale = [f for f in stored if ((h := _parse_hour(f)) is None or h < cutoff)]
    ops += [CommitOperationDelete(f) for f in stale]

    # -- radar frames: render any 7-day hour that has a PQF but no PNG --------
    frame_stored = [f for f in all_files if f.startswith(FRAME_PREFIX)]
    frame_have = {h for f in frame_stored if (h := _parse_hour(f)) is not None}
    fcut = now - timedelta(days=FRAME_KEEP_DAYS)
    want_frames = sorted(((have | set(added)) - frame_have), reverse=True)
    want_frames = [t for t in want_frames if t >= fcut]
    n_frames, frame_errs = 0, 0
    for t in want_frames:
        try:
            data = pqf_bytes.get(t)
            if data is None:                      # backfill from the store
                p = hf_hub_download(HF_REPO, PREFIX + t.strftime(MEMBER_FMT),
                                    repo_type="dataset", token=hf_token())
                with open(p, "rb") as fh:
                    data = fh.read()
            a, _, _, _, nod = _read_pqf(data)
            ops.append(CommitOperationAdd(FRAME_PREFIX + t.strftime(FRAME_FMT),
                                          render_frame(a, nod)))
            n_frames += 1
        except Exception as e:                    # one bad hour must not kill the run
            frame_errs += 1
            print(f"mrms_frames: {t:%Y%m%d%H} render failed: {e}")
    frame_stale = [f for f in frame_stored
                   if ((h := _parse_hour(f)) is None or h < fcut)]
    ops += [CommitOperationDelete(f) for f in frame_stale]

    newest = max(have | set(added), default=None)
    if args.dry_run:
        print(f"mrms_recent: would add {len(added)}, prune {len(stale)} "
              f"({misses} not on NCEP); frames: +{n_frames}, -{len(frame_stale)}")
        return 0
    if ops:
        api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops,
                          commit_message=f"mrms_recent: +{len(added)} Pass1 hour(s)"
                                         f", -{len(stale)} pruned; frames "
                                         f"+{n_frames}/-{len(frame_stale)}")
    if n_frames or frame_stale or frame_errs:
        print(f"mrms_frames: +{n_frames} PNG frame(s), {len(frame_stale)} pruned"
              f"{f', {frame_errs} render failure(s)' if frame_errs else ''}")
    lag_h = (datetime.now(timezone.utc).replace(tzinfo=None) - newest
             ).total_seconds() / 3600 if newest else float("nan")
    print(f"mrms_recent: +{len(added)} Pass1 hour(s), {len(stale)} pruned, "
          f"{misses} not on NCEP (aged out or not posted yet) | "
          f"newest stored {newest:%Y-%m-%d %H:00} UTC (lag {lag_h:.1f} h)"
          if newest else
          f"mrms_recent: nothing stored yet ({misses} fetch misses)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

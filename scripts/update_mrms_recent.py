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
from hf_data.forcing import _write_pqf                                          # noqa: E402

PRODUCT = "MultiSensor_QPE_01H_Pass1"
NCEP_URL = ("https://mrms.ncep.noaa.gov/2D/{prod}/"
            "MRMS_{prod}_00.00_{t:%Y%m%d}-{t:%H}0000.grib2.gz")
PREFIX = "mrms_recent/"
MEMBER_FMT = "mrms1h_pass1_%Y%m%d%H.pqf"


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

    from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
    api = HfApi(token=hf_token())
    stored = [f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
              if f.startswith(PREFIX)]
    have = {h for f in stored if (h := _parse_hour(f)) is not None}

    now = _utc_hour_floor()
    # newest first, so a partial/interrupted run still banks the freshest hours.
    # i starts at 0: the file valid at the CURRENT floor hour posts ~:50, so the
    # :58 self-run must try it — starting at 1 would delay every hour's file to
    # the NEXT run (a permanent extra hour of lag). Not-yet-posted just misses.
    want = [t for i in range(0, args.hours + 1)
            if (t := now - timedelta(hours=i)) not in have]

    ops, added, misses = [], [], 0
    for t in want:
        data = fetch_hour(t)
        if data is None:
            misses += 1
            continue
        ops.append(CommitOperationAdd(PREFIX + t.strftime(MEMBER_FMT), data))
        added.append(t)

    cutoff = now - timedelta(days=args.keep_days)
    stale = [f for f in stored if ((h := _parse_hour(f)) is None or h < cutoff)]
    ops += [CommitOperationDelete(f) for f in stale]

    newest = max(have | set(added), default=None)
    if args.dry_run:
        print(f"mrms_recent: would add {len(added)}, prune {len(stale)} "
              f"({misses} not on NCEP)")
        return 0
    if ops:
        api.create_commit(repo_id=HF_REPO, repo_type="dataset", operations=ops,
                          commit_message=f"mrms_recent: +{len(added)} Pass1 hour(s)"
                                         f", -{len(stale)} pruned")
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

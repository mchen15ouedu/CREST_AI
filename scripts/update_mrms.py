"""Auto-update the CREST_data MRMS precipitation forcing from Iowa State's mtarchive.

Source: NCEP MRMS QPE, 1-hour accumulation, mirrored at
    https://mtarchive.geol.iastate.edu/YYYY/MM/DD/mrms/ncep/<product>/
        <product>_00.00_YYYYMMDD-HH0000.grib2.gz
where <product> is MultiSensor_QPE_01H_Pass2 (>= 2020-10-15) or GaugeCorr_QPE_01H
(before it) — the same source AQUAH's precipitation_processor downloads live.

The grib2 native grid (0.01deg CONUS, 3500 rows x 7000 cols, row 0 = north,
west->east) IS the geometry of every MRMS PQF already in the store, so each field
is written straight to a PQF with no regridding. Raw grib2 values are kept
verbatim — including the -3 "no radar coverage" flag — so appended members are
byte-for-byte consistent with the existing archive (nodata metadata = -9999,
matching what the original tif->pqf preprocessing wrote). The updater only
APPENDS hours missing from mrms/YYYY/mrms_YYYY_MM.tar; existing members are never
touched.

Run on the local machine (needs cfgrib/eccodes/rasterio + HF token):
    python scripts/update_mrms.py [--months 2026-06,2026-07] [--dry-run]
                                  [--back-days 70] [--max-new-per-month N]
Default: scan every month touched by the last --back-days (70) up to now. The
first catch-up run can fetch many hours (the store lagged weeks); later weekly
runs fetch only the new tail. Hours the mirror doesn't have yet are reported as
lag and retried next run.
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import HF_REPO, hf_token, scan_months, update_month  # noqa: E402
from hf_data.forcing import _write_pqf                                          # noqa: E402

BASE_URL = "https://mtarchive.geol.iastate.edu"
FORMAT_CHANGE = datetime(2020, 10, 15)
# (xll, yll, cellsize, nrows, ncols, nodata) — canonical MRMS store grid
MRMS_GRID = (-130.0, 20.0, 0.01, 3500, 7000, -9999.0)
LAG = timedelta(hours=6)                 # mtarchive posts within a few hours
MONTH_FMT = "mrms/%Y/mrms_%Y_%m.tar"
MEMBER_FMT = "mrms_corr_%Y%m%d%H.pqf"    # existing store naming
MEMBER_PATS = [r"mrms_corr_(\d{10})\.pqf$", r"mrms_(\d{10})\.pqf$"]


def _product(t: datetime) -> str:
    return "MultiSensor_QPE_01H_Pass2" if t >= FORMAT_CHANGE else "GaugeCorr_QPE_01H"


def produce(t: datetime) -> bytes | None:
    """Fetch one MRMS hour and return PQF bytes on MRMS_GRID, or None (lag/gap)."""
    prod = _product(t)
    url = (f"{BASE_URL}/{t:%Y/%m/%d}/mrms/ncep/{prod}/"
           f"{prod}_00.00_{t:%Y%m%d}-{t:%H}0000.grib2.gz")
    try:
        raw = urllib.request.urlopen(url, timeout=90).read()
    except Exception:
        return None                                   # not posted yet / gap
    gpath = os.path.join(tempfile.gettempdir(), "_mrms_dl.grib2")
    with open(gpath, "wb") as fh:
        fh.write(gzip.decompress(raw))
    import xarray as xr
    try:
        ds = xr.open_dataset(gpath, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = list(ds.data_vars)[0]
        a = np.asarray(ds[var].values, dtype="float32")   # already row0=north, W->E
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
    buf = os.path.join(tempfile.gettempdir(), "_mrms_member.pqf")
    _write_pqf(buf, a, xll, yll, cell, nod)
    with open(buf, "rb") as fh:
        return fh.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", help="comma list like 2026-06,2026-07")
    ap.add_argument("--back-days", type=int, default=70)
    ap.add_argument("--max-new-per-month", type=int, default=None,
                    help="throttle: cap new hours fetched per month per run")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    for y, m in scan_months(args.back_days, args.months):
        try:
            print(update_month(
                var="mrms", year=y, month=m, produce=produce, api=api,
                dry_run=args.dry_run, month_fmt=MONTH_FMT, member_fmt=MEMBER_FMT,
                member_pats=MEMBER_PATS, freq="h", source="MRMS", lag=LAG,
                cap=args.max_new_per_month))
        except Exception as e:
            print(f"{y}-{m:02d}: FAILED — {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

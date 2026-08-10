"""Auto-update the CREST_data PET forcing from USGS FEWS NET Global daily PET.

Source: USGS FEWS NET global daily potential ET, one gzipped tar per day at
    https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/global/daily/
        pet/downloads/daily/etYYMMDD.tar.gz
containing etYYMMDD.bil (+ .hdr/.prj) — the same source AQUAH's pet_processor
downloads live.

The .bil native grid (1deg global, 360 cols x 181 rows, row 0 = north, corner
-180.5/-90.5) IS the geometry of every PET PQF in the store, so each day is
written straight to a PQF with no regridding. Values are kept verbatim (native
hundredths-of-mm/day, consumed by EF5 as UNIT=mm/100d); only NaN / source-nodata
cells are normalised to -9999. Stored member name matches the existing archive
(etYYYYMMDD.bil.pqf, full 4-digit year). The updater only APPENDS days missing
from pet/YYYY/pet_YYYY_MM.tar; existing members are never touched.

Run on the local machine (needs rasterio + HF token):
    python scripts/update_pet.py [--months 2026-06,2026-07] [--dry-run]
                                 [--back-days 70]
Default: scan every month touched by the last --back-days (70) up to now. Days
the source hasn't published yet are reported as lag and retried next run.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import tarfile
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

BASE_URL = ("https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/fews/web/"
            "global/daily/pet/downloads/daily")
# (xll, yll, cellsize, nrows, ncols, nodata) — canonical PET store grid
PET_GRID = (-180.5, -90.5, 1.0, 181, 360, -9999.0)
LAG = timedelta(days=1)                  # FEWS publishes ~a day behind
MONTH_FMT = "pet/%Y/pet_%Y_%m.tar"
MEMBER_FMT = "et%Y%m%d.bil.pqf"          # existing store naming (4-digit year)
MEMBER_PATS = [r"et(\d{8})\.bil\.pqf$", r"et(\d{8})\.pqf$"]


def _extract(raw: bytes, dest: str):
    tf = tarfile.open(fileobj=io.BytesIO(raw))
    try:                                  # py>=3.12 tar filter; ignore on older
        tf.extractall(dest, filter="data")
    except TypeError:
        tf.extractall(dest)


def produce(t: datetime) -> bytes | None:
    """Fetch one PET day and return PQF bytes on PET_GRID, or None (lag)."""
    url = f"{BASE_URL}/et{t:%y%m%d}.tar.gz"
    try:
        raw = urllib.request.urlopen(url, timeout=90).read()
    except Exception:
        return None                                   # not published yet
    dd = tempfile.mkdtemp(prefix="_pet_")
    try:
        _extract(raw, dd)
        bils = glob.glob(os.path.join(dd, "*.bil"))
        if not bils:
            return None
        import rasterio
        with rasterio.open(bils[0]) as src:
            a = src.read(1).astype("float32")
            src_nod = src.nodata
        a = np.where(np.isnan(a), -9999.0, a)
        if src_nod is not None and src_nod != -9999.0:
            a = np.where(a == np.float32(src_nod), -9999.0, a)
    finally:
        for f in glob.glob(os.path.join(dd, "*")):
            try:
                os.remove(f)
            except OSError:
                pass
        try:
            os.rmdir(dd)
        except OSError:
            pass
    xll, yll, cell, nr, nc, nod = PET_GRID
    if a.shape != (nr, nc):
        return None
    buf = os.path.join(tempfile.gettempdir(), "_pet_member.pqf")
    _write_pqf(buf, a, xll, yll, cell, nod)
    with open(buf, "rb") as fh:
        return fh.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", help="comma list like 2026-06,2026-07")
    ap.add_argument("--back-days", type=int, default=70)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi(token=hf_token())
    for y, m in scan_months(args.back_days, args.months):
        try:
            print(update_month(
                var="pet", year=y, month=m, produce=produce, api=api,
                dry_run=args.dry_run, month_fmt=MONTH_FMT, member_fmt=MEMBER_FMT,
                member_pats=MEMBER_PATS, freq="d", source="PET", lag=LAG))
        except Exception as e:
            print(f"{y}-{m:02d}: FAILED — {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

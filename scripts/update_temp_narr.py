"""Auto-update the CREST_data temperature forcing from NARR (task: NARR temp).

Source: NCEP North American Regional Reanalysis 2-m air temperature
(GDEX d608000, https://gdex.ucar.edu/datasets/d608000/ — 32 km Lambert
Conformal, 3-hourly, GRIB1). We pull it from NOAA PSL's netCDF mirror of the
SAME dataset (one yearly file per variable, anonymous HTTPS, updated ~weekly):
    https://downloads.psl.noaa.gov/Datasets/NARR/monolevel/air.2m.YYYY.nc
because the GDEX form is 0.5–1 GB multi-variable GRIB tars per ~8 days, i.e.
~2.6 GB of download per month to extract a single band.

Each 3-hourly Lambert field is bilinearly resampled onto the canonical
NLDAS-heritage 0.125° temp grid (forcing.TEMP_GRID — so every temp PQF in the
store shares one geometry, which the [TEMPForcing] DEM= extrapolation grid is
built against), converted K→°C, and linearly interpolated to HOURLY members
(EF5 zero-fills missing hourly files, which would freeze SNOW17 at 0 °C).
Members use the generic name temp_YYYYMMDDHH.pqf; existing NLDAS members are
never touched — the updater only APPENDS missing hours to each month tar.

Run on the local machine (needs netCDF4/pyproj/numpy + HF token):
    python scripts/update_temp_narr.py [--months 2026-06,2026-07] [--dry-run]
Default: scan from BACKFILL_START (or the last 4 months, whichever is later)
through the current month.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()
import urllib.request

from hf_data.forcing import TEMP_GRID, VARS, _write_pqf  # noqa: E402

HF_REPO = "vincewin/CREST_data"
PSL_URL = "https://downloads.psl.noaa.gov/Datasets/NARR/monolevel/air.2m.{year}.nc"
TOKEN_PATH = r"C:\Users\chenm\Documents\EF5\CREST_token.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_narr_cache")
BACKFILL_START = (2026, 6)          # NLDAS members end 2026-06-23 12z
_MEMBER_RES = [re.compile(r"NLDAS_FORA0125_H\.A(\d{8})\.(\d{2})00\."),
               re.compile(r"temp_(\d{10})\.pqf$")]


def _member_hours(names) -> set[datetime]:
    """Hours already present in a tar (NLDAS or generic member names)."""
    hours = set()
    for n in names:
        m = _MEMBER_RES[0].search(n)
        if m:
            hours.add(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H"))
            continue
        m = _MEMBER_RES[1].search(n)
        if m:
            hours.add(datetime.strptime(m.group(1), "%Y%m%d%H"))
    return hours


def _download_year(year: int) -> str | None:
    """Fetch air.2m.<year>.nc if PSL has a newer copy than our cache."""
    os.makedirs(DATA_DIR, exist_ok=True)
    url = PSL_URL.format(year=year)
    dest = os.path.join(DATA_DIR, f"air.2m.{year}.nc")
    meta = dest + ".meta.json"
    try:
        req = urllib.request.Request(url, method="HEAD")
        h = urllib.request.urlopen(req, timeout=60)
        remote = {"len": h.headers.get("Content-Length"),
                  "mod": h.headers.get("Last-Modified")}
    except Exception as e:
        print(f"  [narr] HEAD {url} failed: {e}")
        return dest if os.path.exists(dest) else None
    if os.path.exists(dest) and os.path.exists(meta):
        if json.load(open(meta)) == remote:
            return dest
    print(f"  [narr] downloading {url} ({int(remote['len'] or 0)/1e6:.0f} MB)…")
    urllib.request.urlretrieve(url, dest + ".part")
    os.replace(dest + ".part", dest)
    json.dump(remote, open(meta, "w"))
    return dest


class NarrYear:
    """air.2m.<year>.nc with lazy bilinear resampling onto TEMP_GRID."""

    def __init__(self, path: str):
        import netCDF4
        self.ds = netCDF4.Dataset(path)
        t = self.ds.variables["time"]
        import cftime  # noqa: F401  (num2date backend)
        import netCDF4 as nc4
        dates = nc4.num2date(t[:], t.units)
        self.hours = {datetime(d.year, d.month, d.day, d.hour): i
                      for i, d in enumerate(dates)}
        self.air = self.ds.variables["air"]
        gm = self.ds.variables["Lambert_Conformal"]
        x = self.ds.variables["x"][:].astype("float64")
        y = self.ds.variables["y"][:].astype("float64")
        from pyproj import CRS, Transformer
        crs = CRS.from_cf({k: getattr(gm, k) for k in gm.ncattrs()})
        tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        xll, yll, cell, nr, nc = TEMP_GRID[:5]
        lons = xll + (np.arange(nc) + 0.5) * cell
        lats = yll + (np.arange(nr) + 0.5) * cell
        LON, LAT = np.meshgrid(lons, lats[::-1])          # row 0 = north
        X, Y = tr.transform(LON, LAT)
        self.fi = (Y - y[0]) / (y[1] - y[0])
        self.fj = (X - x[0]) / (x[1] - x[0])
        self._cache: dict[datetime, np.ndarray] = {}

    def field(self, when: datetime) -> np.ndarray | None:
        """°C field on TEMP_GRID at a 3-hourly NARR step, or None."""
        if when in self._cache:
            return self._cache[when]
        idx = self.hours.get(when)
        if idx is None:
            return None
        a = np.ma.filled(self.air[idx, :, :].astype("float64"), np.nan) - 273.15
        i0 = np.clip(np.floor(self.fi).astype(int), 0, a.shape[0] - 2)
        j0 = np.clip(np.floor(self.fj).astype(int), 0, a.shape[1] - 2)
        di = np.clip(self.fi - i0, 0, 1)
        dj = np.clip(self.fj - j0, 0, 1)
        v = (a[i0, j0] * (1 - di) * (1 - dj) + a[i0 + 1, j0] * di * (1 - dj)
             + a[i0, j0 + 1] * (1 - di) * dj + a[i0 + 1, j0 + 1] * di * dj)
        out = np.where(np.isfinite(v), v, TEMP_GRID[5]).astype("float32")
        if len(self._cache) > 4:
            self._cache.clear()
        self._cache[when] = out
        return out

    def close(self):
        self.ds.close()


def hourly_field(years: dict[int, NarrYear], when: datetime) -> np.ndarray | None:
    """Linear interpolation between the two bracketing 3-hourly NARR steps."""
    t0 = when.replace(hour=when.hour - when.hour % 3)
    ny = years.get(t0.year)
    f0 = ny.field(t0) if ny else None
    if f0 is None:
        return None
    if when == t0:
        return f0
    t1 = t0 + timedelta(hours=3)
    ny1 = years.get(t1.year)
    f1 = ny1.field(t1) if ny1 else None
    if f1 is None:
        return None                    # can't bracket -> leave for the next run
    w = (when - t0).total_seconds() / 10800.0
    bad = (f0 == TEMP_GRID[5]) | (f1 == TEMP_GRID[5])
    v = ((1 - w) * f0 + w * f1).astype("float32")
    v[bad] = TEMP_GRID[5]
    return v


def update_month(year: int, month: int, years: dict[int, NarrYear], api,
                 dry_run: bool) -> str:
    from huggingface_hub import hf_hub_download
    cfg = VARS["temp"]
    ref = datetime(year, month, 1)
    tar_repo_path = ref.strftime(cfg.month_fmt)
    old_members: list[tuple[tarfile.TarInfo, bytes]] = []
    have: set[datetime] = set()
    try:
        local = hf_hub_download(HF_REPO, tar_repo_path, repo_type="dataset")
        with tarfile.open(local) as tf:
            for ti in tf.getmembers():
                old_members.append((ti, tf.extractfile(ti).read()))
        have = _member_hours([ti.name for ti, _ in old_members])
    except Exception:
        pass                                            # brand-new month tar

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    end = min(now, datetime(year + (month == 12), month % 12 + 1, 1))
    want, t = [], ref
    while t < end:
        if t not in have:
            want.append(t)
        t += timedelta(hours=1)
    if not want:
        return f"{year}-{month:02d}: complete ({len(have)} hours), nothing to do"

    new: list[tuple[str, bytes]] = []
    xll, yll, cell, nr, nc = TEMP_GRID[:5]
    top = yll + nr * cell
    for t in want:
        v = hourly_field(years, t)
        if v is None:
            continue
        buf = os.path.join(tempfile.gettempdir(), "narr_member.pqf")
        _write_pqf(buf, v, xll, yll, cell, TEMP_GRID[5])
        new.append((t.strftime(cfg.out_fmt), open(buf, "rb").read()))
    if not new:
        return (f"{year}-{month:02d}: {len(want)} hour(s) missing but NARR has "
                f"no data for them yet (lag)")
    if dry_run:
        return f"{year}-{month:02d}: would add {len(new)} NARR hour(s) to {len(have)} existing"

    out_tar = os.path.join(tempfile.gettempdir(), f"temp_{year}_{month:02d}.tar")
    with tarfile.open(out_tar, "w") as tf:
        for ti, data in old_members:
            tf.addfile(ti, io.BytesIO(data))
        for name, data in sorted(new):
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            ti.mtime = int(datetime.now(timezone.utc).timestamp())
            tf.addfile(ti, io.BytesIO(data))
    from huggingface_hub import CommitOperationAdd
    api.create_commit(
        repo_id=HF_REPO, repo_type="dataset",
        operations=[CommitOperationAdd(path_in_repo=tar_repo_path,
                                       path_or_fileobj=out_tar)],
        commit_message=f"temp {year}-{month:02d}: +{len(new)} NARR-derived "
                       f"hourly member(s) (32 km air.2m -> 0.125deg, K->C, "
                       f"3h->1h linear)")
    os.remove(out_tar)
    return f"{year}-{month:02d}: uploaded +{len(new)} NARR hour(s) ({len(have)} kept)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", help="comma list like 2026-06,2026-07")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    if args.months:
        months = [tuple(int(x) for x in m.split("-")) for m in args.months.split(",")]
    else:
        months = []
        back4 = now - timedelta(days=123)               # rescan the last ~4 months
        y, m = max(BACKFILL_START, (back4.year, back4.month))
        while (y, m) <= (now.year, now.month):
            months.append((y, m))
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)

    years: dict[int, NarrYear] = {}
    need_years = {y for y, _ in months} | {y + 1 for y, mo in months if mo == 12}
    for yr in sorted(need_years):
        p = _download_year(yr)
        if p:
            try:
                years[yr] = NarrYear(p)
            except Exception as e:
                print(f"  [narr] cannot open {p}: {e}")
    if not years:
        print("no NARR data available — aborting")
        return 1

    from huggingface_hub import HfApi
    tok = open(TOKEN_PATH).read().strip() if os.path.exists(TOKEN_PATH) else None
    api = HfApi(token=tok)
    for y, m in months:
        try:
            print(update_month(y, m, years, api, args.dry_run))
        except Exception as e:
            print(f"{y}-{m:02d}: FAILED — {e}")
    for ny in years.values():
        ny.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

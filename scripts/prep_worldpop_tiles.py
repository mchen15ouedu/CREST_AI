"""Local prep: WorldPop age-sex -> 1x1-degree parquet tiles on HF.

The risk view needs per-pixel population everywhere an event can trigger,
but WorldPop's 100 m band tifs are ~1.3 GB each and strip-organized —
remote window reads take minutes per band (measured live 2026-08-25), and
the demo Space must not hold bulk data. So this script runs ON THE LOCAL
MACHINE (like update_pet.py):

  1. download each of the 36 age-sex bands ONCE into E:/hydroZone
     (Global 2015-2030 R2025A constrained, the 2026-2030 projections only),
  2. window every populated 1x1-degree tile, aggregate the 18 ages to the
     7 display buckets x 2 sexes,
  3. write worldpop/<year>/N{lat}W{lon}.parquet (sparse: one row per
     populated 100 m cell — row, col in the tile's 3-arcsec grid + 14
     float32 people counts),
  4. upload to the vincewin/CREST_riskdata dataset in batched commits.

The Space-side probe (hf_data/worldpop.py) then downloads just the ONE
small tile a basin needs. Tiles average a few MB; CONUS for one year is a
few GB of parquet.

Run:  python scripts/prep_worldpop_tiles.py --year 2026 [--bbox W S E N]
      [--skip-download] [--dry-run]
The tiler is resumable: tiles already in the HF repo are skipped.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import truststore
truststore.inject_into_ssl()

from forcing_update_common import hf_token   # noqa: E402

RELEASE = "R2025A"
SRC_DIR = os.environ.get("WORLDPOP_SRC", r"E:\hydroZone\riskdata_src\worldpop")
REPO = "vincewin/CREST_riskdata"
AGES = ["00", "01", "05", "10", "15", "20", "25", "30", "35", "40",
        "45", "50", "55", "60", "65", "70", "75", "80"]
SEXES = ("m", "f")
BUCKETS = [
    ("0-4", ["00", "01"]), ("5-14", ["05", "10"]), ("15-24", ["15", "20"]),
    ("25-44", ["25", "30", "35", "40"]), ("45-64", ["45", "50", "55", "60"]),
    ("65-74", ["65", "70"]), ("75+", ["75", "80"]),
]
CONUS = (-125.0, 24.0, -66.0, 50.0)          # W S E N


def band_url(sex: str, age: str, year: int) -> str:
    return ("https://data.worldpop.org/GIS/AgeSex_structures/"
            f"Global_2015_2030/{RELEASE}/{year}/USA/v1/100m/constrained/"
            f"usa_{sex}_{age}_{year}_CN_100m_{RELEASE}_v1.tif")


def download_bands(year: int, workers: int = 5) -> list[str]:
    """Fetch all 36 bands into SRC_DIR (resumable; ~46 GB once). Parallel:
    worldpop.org throttles per connection (~0.8 MB/s measured), so several
    streams cut the wall time from ~15 h to ~3 h."""
    import requests
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(SRC_DIR, exist_ok=True)
    jobs, paths = [], []
    for sex in SEXES:
        for age in AGES:
            url = band_url(sex, age, year)
            local = os.path.join(SRC_DIR, os.path.basename(url))
            paths.append(local)
            if not (os.path.exists(local)
                    and os.path.getsize(local) > 1 << 20):
                jobs.append((url, local))

    def fetch(job):
        url, local = job
        t0 = time.time()
        print(f"downloading {os.path.basename(url)} ...", flush=True)
        r = requests.get(url, stream=True, timeout=3600,
                         headers={"User-Agent": "CREST-AI/1.0"})
        r.raise_for_status()
        tmp = local + f".part{os.getpid()}"
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 22):
                fh.write(chunk)
        os.replace(tmp, local)
        print(f"  {os.path.basename(local)}: "
              f"{os.path.getsize(local) / 1e6:.0f} MB in "
              f"{time.time() - t0:.0f} s", flush=True)

    if jobs:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(fetch, jobs))
    return paths


def tile_name(lat0: int, lon0: int) -> str:
    return (f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}"
            f"{'W' if lon0 < 0 else 'E'}{abs(lon0):03d}")


def make_tile(dss, lat0: int, lon0: int):
    """One 1x1-deg tile -> (parquet table bytes, n_rows) or None if empty.
    dss: {(sex, age): open rasterio dataset}."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rasterio.windows import from_bounds
    ref = next(iter(dss.values()))
    win = from_bounds(lon0, lat0, lon0 + 1, lat0 + 1, ref.transform)
    win = win.round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return None
    # aggregate straight to buckets to hold at most 14 windows in RAM
    agg = {}
    total = None
    for sex in SEXES:
        for label, ages in BUCKETS:
            acc = None
            for age in ages:
                ds = dss[(sex, age)]
                a = ds.read(1, window=win, boundless=True, fill_value=0.0)
                a = a.astype(np.float32)
                nod = ds.nodata
                if nod is not None:
                    a[a == np.float32(nod)] = 0.0
                a[~np.isfinite(a)] = 0.0
                a[a < 0] = 0.0
                acc = a if acc is None else acc + a
            agg[f"{sex}_{label}"] = acc
            total = acc if total is None else total + acc
    rows, cols = np.nonzero(total > 1e-3)
    if len(rows) == 0:
        return None
    tr = ref.window_transform(win)
    data = {"row": rows.astype(np.uint16), "col": cols.astype(np.uint16)}
    for k, a in agg.items():
        data[k] = a[rows, cols]
    t = pa.table(data)
    meta = {b"transform": repr([tr.a, tr.b, tr.c, tr.d, tr.e, tr.f]).encode(),
            b"tile": tile_name(lat0, lon0).encode()}
    t = t.replace_schema_metadata(meta)
    import io as _io
    buf = _io.BytesIO()
    pq.write_table(t, buf, compression="zstd")
    return buf.getvalue(), len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                    default=list(CONUS))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--batch", type=int, default=40,
                    help="tiles per HF commit")
    args = ap.parse_args()

    if not args.skip_download:
        download_bands(args.year)

    import rasterio
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=hf_token())
    try:
        have = set(api.list_repo_files(REPO, repo_type="dataset"))
    except Exception:
        have = set()

    dss = {}
    for sex in SEXES:
        for age in AGES:
            p = os.path.join(SRC_DIR,
                             os.path.basename(band_url(sex, age, args.year)))
            dss[(sex, age)] = rasterio.open(p)

    w, s, e, n = args.bbox
    ops, done, skipped, empty = [], 0, 0, 0
    t0 = time.time()
    for lat0 in range(int(math.floor(s)), int(math.ceil(n))):
        for lon0 in range(int(math.floor(w)), int(math.ceil(e))):
            rel = f"worldpop/{args.year}/{tile_name(lat0, lon0)}.parquet"
            if rel in have:
                skipped += 1
                continue
            out = make_tile(dss, lat0, lon0)
            if out is None:
                empty += 1
                continue
            blob, nrows = out
            if args.dry_run:
                print(f"would write {rel}: {nrows} cells, "
                      f"{len(blob) / 1e6:.2f} MB")
                continue
            import io as _io
            ops.append(CommitOperationAdd(rel, _io.BytesIO(blob)))
            done += 1
            if len(ops) >= args.batch:
                api.create_commit(repo_id=REPO, repo_type="dataset",
                                  operations=ops,
                                  commit_message=f"worldpop {args.year}: "
                                                 f"+{len(ops)} tiles")
                print(f"[{(time.time() - t0) / 60:5.1f} min] committed "
                      f"{done} tiles ({skipped} already, {empty} empty)",
                      flush=True)
                ops = []
    if ops and not args.dry_run:
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                          commit_message=f"worldpop {args.year}: "
                                         f"+{len(ops)} tiles (final)")
    print(f"DONE {args.year}: {done} new tiles, {skipped} already in repo, "
          f"{empty} empty, {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""HF-backed forcing adapter for the AQUAH->CREST_demo integration.

Replaces AQUAH's live MRMS/PET download: pulls the per-timestep forcing PQF
grids hosted in `vincewin/CREST_data` (one tar per variable/year), clips each
timestep to the basin bbox, and writes EF5-ready PQF inputs plus the matching
`TYPE=PQF` control block. Consumed by your fork's native parquet reader
(configure --with-arrow).

PQF = single float32 column 'v', row-major (row 0 = north), grid geometry in the
parquet key/value metadata (matches scripts/crest_preprocess/tif2pqf.py).
"""
from __future__ import annotations

import io
import os
import tarfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pyarrow.parquet as pq
import truststore
truststore.inject_into_ssl()  # local corporate-MITM fix; harmless on HF runners
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

HF_REPO = "vincewin/CREST_data"

# per-variable source layout + EF5 wiring
@dataclass(frozen=True)
class VarCfg:
    month_fmt: str        # HF path of the MONTH tar (strftime on the timestep)
    year_fmt: str         # HF path of the YEAR tar (fallback; strftime on the year)
    member_fmt: str       # member name inside the tar (strftime on the timestep)
    out_fmt: str          # output filename we write (strftime on the timestep)
    freq: str             # 'h' | 'd'
    section: str          # EF5 control section header
    ef5_name: str         # EF5 NAME= pattern (DatedName tokens)
    unit: str             # EF5 UNIT=
    ef5_freq: str         # EF5 FREQ=

VARS = {
    "mrms": VarCfg("mrms/%Y/mrms_%Y_%m.tar", "mrms/mrms_%Y.tar",
                   "mrms_corr_%Y%m%d%H.pqf", "mrms_%Y%m%d%H.pqf",
                   "h", "PrecipForcing MRMS", "mrms_YYYYMMDDHH.pqf", "mm/h", "1h"),
    "pet":  VarCfg("pet/%Y/pet_%Y_%m.tar", "pet/pet_%Y.tar",
                   "et%Y%m%d.bil.pqf", "et%Y%m%d.pqf",
                   "d", "PETForcing PET", "etYYYYMMDD.pqf", "mm/100d", "d"),
    "temp": VarCfg("temp/%Y/temp_%Y_%m.tar", "temp/temp_%Y.tar",
                   "NLDAS_FORA0125_H.A%Y%m%d.%H00.002.grb.SUB_T.pqf", "temp_%Y%m%d%H.pqf",
                   "h", "TEMPForcing TEMP", "temp_YYYYMMDDHH.pqf", "C", "1h"),
}

# Canonical temperature grid: the geometry of every full-domain temp PQF in the
# store (NLDAS-heritage 0.125-deg CONUS). NARR-derived members (scripts/
# update_temp_narr.py) are resampled onto this same grid so ALL temp timesteps
# — and the temperature-extrapolation DEM built against them — share one
# geometry. (xll, yll, cellsize, nrows, ncols, nodata)
TEMP_GRID = (-124.9375, 25.0625, 0.12473060344827586, 224, 464, -9999.0)


def clip_window(bbox, xll, yll, cell, nr, nc):
    """Row/col window of bbox on a grid — the SAME arithmetic as _clip, so a
    grid built from this geometry is cell-identical to the clipped forcing."""
    W, S, E, N = bbox
    top = yll + nr * cell
    c0 = max(0, int(np.floor((W - xll) / cell)));  c1 = min(nc, int(np.ceil((E - xll) / cell)))
    r0 = max(0, int(np.floor((top - N) / cell)));  r1 = min(nr, int(np.ceil((top - S) / cell)))
    if c1 <= c0 or r1 <= r0:
        return None
    return r0, r1, c0, c1


def temp_clip_geometry(bbox):
    """Geometry of the basin-clipped temp grid: (xll, yll_top?, ...) ->
    (nxll, nyll, cell, nrows, ncols) or None if bbox is outside the grid."""
    xll, yll, cell, nr, nc = TEMP_GRID[:5]
    win = clip_window(bbox, xll, yll, cell, nr, nc)
    if win is None:
        return None
    r0, r1, c0, c1 = win
    top = yll + nr * cell
    return xll + c0 * cell, top - r1 * cell, cell, r1 - r0, c1 - c0


def _read_pqf(data: bytes):
    """bytes -> (array[nr,nc] float32, xll, yll, cell, nodata)."""
    pf = pq.ParquetFile(io.BytesIO(data))
    m = {k.decode(): v.decode() for k, v in pf.schema_arrow.metadata.items()
         if not k.startswith(b"ARROW")}
    nc, nr = int(m["ncols"]), int(m["nrows"])
    a = pf.read().column("v").to_numpy().reshape(nr, nc)
    return a, float(m["xllcorner"]), float(m["yllcorner"]), float(m["cellsize"]), float(m["nodata"])


def _write_pqf(path: str, a: np.ndarray, xll: float, yll: float, cell: float, nodata: float):
    import pyarrow as pa
    nr, nc = a.shape
    meta = {b"ncols": str(nc).encode(), b"nrows": str(nr).encode(),
            b"xllcorner": repr(float(xll)).encode(), b"yllcorner": repr(float(yll)).encode(),
            b"cellsize": repr(float(cell)).encode(), b"nodata": repr(float(nodata)).encode()}
    schema = pa.schema([pa.field("v", pa.float32())]).with_metadata(meta)
    tmp = path + ".tmp"                       # atomic: the shared store may have
    pq.write_table(pa.table({"v": a.reshape(-1).astype("float32")}, schema=schema),
                   tmp, compression="zstd")   # concurrent readers (EF5)
    os.replace(tmp, path)


def _clip(a, xll, yll, cell, nodata, bbox):
    """Clip full-domain grid to bbox (W,S,E,N). Returns (sub, nxll, nyll) or None."""
    nr, nc = a.shape
    win = clip_window(bbox, xll, yll, cell, nr, nc)
    if win is None:
        return None
    r0, r1, c0, c1 = win
    top = yll + nr * cell
    return a[r0:r1, c0:c1], xll + c0 * cell, top - r1 * cell


def _timesteps(t0: datetime, t1: datetime, freq: str):
    step = timedelta(hours=1) if freq == "h" else timedelta(days=1)
    t = t0.replace(minute=0, second=0, microsecond=0)
    if freq == "d":
        t = t.replace(hour=0)
    while t <= t1:
        yield t
        t += step


@dataclass
class ForcingResult:
    var: str
    out_dir: str
    written: list[str] = field(default_factory=list)
    reused: int = 0                 # timesteps already in the store (merged, not redone)
    missing: list[str] = field(default_factory=list)
    control_block: str = ""


def store_dir(var: str, bbox) -> str:
    """Shared per-(variable, basin-bbox) forcing store. Runs over the same basin
    reuse each other's clipped timesteps — overlapping windows are MERGED here
    instead of re-downloaded and re-clipped into per-run temp dirs."""
    from hf_data.statecache import CACHE_DIR
    key = "_".join(f"{v:.3f}" for v in bbox).replace("-", "m").replace(".", "p")
    d = os.path.join(CACHE_DIR, "forcing", var, key)
    os.makedirs(d, exist_ok=True)
    return d


def prepare_forcing(var: str, bbox, t_start: datetime, t_end: datetime, out_dir: str,
                    repo: str = HF_REPO, cache_dir: str | None = None,
                    cancel=None) -> ForcingResult:
    """`cancel` (threading.Event) aborts between tar downloads / member clips —
    a Stop must not sit behind a year of forcing downloads."""
    cfg = VARS[var]
    os.makedirs(out_dir, exist_ok=True)
    res = ForcingResult(var=var, out_dir=out_dir)

    # temporal merge: timesteps already present in out_dir are reused as-is;
    # only the missing ones are fetched + clipped
    by_month: dict[tuple[int, int], list[datetime]] = {}
    for t in _timesteps(t_start, t_end, cfg.freq):
        if os.path.exists(os.path.join(out_dir, t.strftime(cfg.out_fmt))):
            res.reused += 1
            continue
        by_month.setdefault((t.year, t.month), []).append(t)

    for (year, month), steps in sorted(by_month.items()):
        if cancel is not None and cancel.is_set():
            return res                           # partial store stays valid (merge)
        ref = steps[0]
        try:                                    # prefer the small month-tar
            local_tar = hf_hub_download(repo, ref.strftime(cfg.month_fmt),
                                        repo_type="dataset", cache_dir=cache_dir)
        except Exception:                        # fallback to year-tar (pre-reshard)
            try:
                local_tar = hf_hub_download(repo, ref.strftime(cfg.year_fmt),
                                            repo_type="dataset", cache_dir=cache_dir)
            except EntryNotFoundError:           # genuine data gap (permanent 404):
                # fail loudly but readably — never silently run with no forcing.
                # (transient network errors raise other types and still propagate.)
                raise RuntimeError(
                    f"{var.upper()} forcing is not available for {year}-{month:02d} — "
                    f"the archive does not cover this period. Try a more recent "
                    f"date range."
                ) from None
        with tarfile.open(local_tar) as tf:
            names = set(tf.getnames())
            for t in steps:
                if cancel is not None and cancel.is_set():
                    return res
                member = t.strftime(cfg.member_fmt)
                if member not in names:                # NARR-derived members use
                    member = t.strftime(cfg.out_fmt)   # the generic name
                if member not in names:
                    res.missing.append(member)
                    continue
                a, xll, yll, cell, nod = _read_pqf(tf.extractfile(member).read())
                clip = _clip(a, xll, yll, cell, nod, bbox)
                if clip is None:
                    res.missing.append(member)
                    continue
                sub, nxll, nyll = clip
                out_name = t.strftime(cfg.out_fmt)
                _write_pqf(os.path.join(out_dir, out_name), sub, nxll, nyll, cell, nod)
                res.written.append(out_name)

    res.control_block = (
        f"[{cfg.section}]\nTYPE=PQF\nUNIT={cfg.unit}\nFREQ={cfg.ef5_freq}\n"
        f"LOC={out_dir}\nNAME={cfg.ef5_name}\n\n"
    )
    return res


if __name__ == "__main__":
    import sys, time
    out = sys.argv[1] if len(sys.argv) > 1 else "._forcing_test"
    bbox = (-69.83, 46.32, -68.33, 47.82)  # Allagash basin
    t0, t1 = datetime(2026, 1, 1), datetime(2026, 1, 3)
    for var in ("pet",):
        t = time.time()
        r = prepare_forcing(var, bbox, t0, t1, os.path.join(out, var.upper()))
        print(f"[{var}] wrote {len(r.written)} files, {len(r.missing)} missing in {time.time()-t:.1f}s")
        for f in r.written:
            with open(os.path.join(r.out_dir, f), "rb") as fh:
                a, xll, yll, cell, nod = _read_pqf(fh.read())
            print(f"    {f}: {a.shape} cell={cell} range[{np.nanmin(a):.2f},{np.nanmax(a):.2f}]")
        print("  --- control block ---")
        print("  " + r.control_block.replace("\n", "\n  ").rstrip())

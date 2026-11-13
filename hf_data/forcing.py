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

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pyarrow.parquet as pq
import truststore
truststore.inject_into_ssl()  # local corporate-MITM fix; harmless on HF runners
from huggingface_hub import hf_hub_download

from hf_data.atomicfile import atomic_path

# Layout, PQF decoding, clipping and the per-variable value rules live in the
# forcing archive module; re-exported here so existing importers keep working.
from hf_data.forcing_archive import (  # noqa: F401,E402
    HF_REPO, K_OFFSET, TEMP_K_MEDIAN, VARS, ArchiveGap, ForcingArchive,
    TempUnitsError, VarCfg, _clip, _read_pqf, check_temp_units, clip_window,
)

# Canonical temperature grid: the geometry of every full-domain temp PQF in the
# store (NLDAS-heritage 0.125-deg CONUS). NARR-derived members (scripts/
# update_temp_narr.py) are resampled onto this same grid so ALL temp timesteps
# — and the temperature-extrapolation DEM built against them — share one
# geometry. (xll, yll, cellsize, nrows, ncols, nodata)
TEMP_GRID = (-124.9375, 25.0625, 0.12473060344827586, 224, 464, -9999.0)


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


def _write_pqf(path: str, a: np.ndarray, xll: float, yll: float, cell: float, nodata: float):
    import pyarrow as pa
    nr, nc = a.shape
    meta = {b"ncols": str(nc).encode(), b"nrows": str(nr).encode(),
            b"xllcorner": repr(float(xll)).encode(), b"yllcorner": repr(float(yll)).encode(),
            b"cellsize": repr(float(cell)).encode(), b"nodata": repr(float(nodata)).encode()}
    schema = pa.schema([pa.field("v", pa.float32())]).with_metadata(meta)
    # atomic, temp name unique per call: the shared store may have concurrent
    # readers (EF5), and two writers - fleet workers or dashboard threads - may
    # clip the same hour (nested basins share a store)
    with atomic_path(path) as tmp:
        pq.write_table(pa.table({"v": a.reshape(-1).astype("float32")}, schema=schema),
                       tmp, compression="zstd")


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
    recent: int = 0                 # timesteps filled from the near-real-time store
    converted: list[str] = field(default_factory=list)   # temp members stored in K
    purged: int = 0                 # pre-guard Kelvin clips dropped from the store
    control_block: str = ""


# Near-real-time fallback stores: the month tars archive FINAL products on a
# slow cadence (MRMS Pass2 is batched every few days — seen live: the freshest
# ~2 days absent), which is exactly the window every nowcast/event run needs.
# Loose per-hour members uploaded hourly by scripts/update_mrms_recent.py fill
# tar-missing hours. var -> (HF path strftime fmt, retention days).
RECENT_STORES = {
    "mrms": ("mrms_recent/mrms1h_pass1_%Y%m%d%H.pqf", 21),
}


def store_dir(var: str, bbox) -> str:
    """Shared per-(variable, basin-bbox) forcing store. Runs over the same basin
    reuse each other's clipped timesteps — overlapping windows are MERGED here
    instead of re-downloaded and re-clipped into per-run temp dirs."""
    from hf_data.statecache import CACHE_DIR
    key = "_".join(f"{v:.3f}" for v in bbox).replace("-", "m").replace(".", "p")
    d = os.path.join(CACHE_DIR, "forcing", var, key)
    os.makedirs(d, exist_ok=True)
    return d


TEMP_STORE_MARK = ".temp_units_ok_v1"


def _purge_kelvin_clips(out_dir: str) -> int:
    """Once per store: drop clipped temp files written before the units guard
    existed — the merge below reuses stored timesteps as-is, so a Kelvin clip
    would otherwise outlive the archive repair (persistent caches, e.g. the
    Jetstream2 VM). Dropped hours are re-clipped from the tar."""
    mark = os.path.join(out_dir, TEMP_STORE_MARK)
    if os.path.exists(mark):
        return 0
    n = 0
    for f in os.listdir(out_dir):
        if not f.endswith(".pqf"):
            continue
        p = os.path.join(out_dir, f)
        with open(p, "rb") as fh:
            a, _, _, _, nod = _read_pqf(fh.read())
        v = a[np.isfinite(a) & (a != nod) & (a > -9000)]
        if v.size and float(np.median(v)) > TEMP_K_MEDIAN:
            os.remove(p)
            n += 1
    open(mark, "w").close()
    return n


def prepare_forcing(var: str, bbox, t_start: datetime, t_end: datetime, out_dir: str,
                    repo: str = HF_REPO, cache_dir: str | None = None,
                    cancel=None, archive: ForcingArchive | None = None) -> ForcingResult:
    """`cancel` (threading.Event) aborts between tar downloads / member clips —
    a Stop must not sit behind a year of forcing downloads. Archive grids are
    read through `archive` (default: a ForcingArchive on `repo`/`cache_dir`)."""
    cfg = VARS[var]
    os.makedirs(out_dir, exist_ok=True)
    res = ForcingResult(var=var, out_dir=out_dir)
    if var == "temp":
        res.purged = _purge_kelvin_clips(out_dir)

    # temporal merge: timesteps already present in out_dir are reused as-is;
    # only the missing ones are fetched + clipped
    by_month: dict[tuple[int, int], list[datetime]] = {}
    for t in _timesteps(t_start, t_end, cfg.freq):
        if os.path.exists(os.path.join(out_dir, t.strftime(cfg.out_fmt))):
            res.reused += 1
            continue
        by_month.setdefault((t.year, t.month), []).append(t)

    missing_ts: list[datetime] = []
    recent = RECENT_STORES.get(var)
    now = datetime.utcnow()

    arc = archive if archive is not None else ForcingArchive(repo, cache_dir)
    try:
        for (year, month), steps in sorted(by_month.items()):
            if cancel is not None and cancel.is_set():
                return res                       # partial store stays valid (merge)
            for i, t in enumerate(steps):
                if cancel is not None and cancel.is_set():
                    return res
                try:
                    g = arc.read(var, t, bbox)   # validated, degC for temp
                except ArchiveGap:               # genuine data gap (permanent 404):
                    if recent and (now - max(steps)) <= timedelta(days=recent[1]):
                        # a brand-new month may have no tar yet; the near-real-time
                        # store below covers this window
                        missing_ts.extend(steps[i:])
                        break
                    # fail loudly but readably — never silently run with no forcing.
                    # (transient network errors raise other types and still propagate.)
                    raise RuntimeError(
                        f"{var.upper()} forcing is not available for {year}-{month:02d} — "
                        f"the archive does not cover this period. Try a more recent "
                        f"date range."
                    ) from None
                if g is None:                    # member absent / box off-grid
                    missing_ts.append(t)
                    continue
                out_name = t.strftime(cfg.out_fmt)
                if g.was_kelvin:
                    res.converted.append(out_name)
                _write_pqf(os.path.join(out_dir, out_name), g.values, g.xll, g.yll,
                           g.cell, g.nodata)
                res.written.append(out_name)
    finally:
        if archive is None:
            arc.close()

    # ---- near-real-time fill (tar-missing hours) --------------------------- #
    # The archive tar batches the final product days behind real time; every
    # hour it lacks is fetched from the loose hourly store instead. A filled
    # hour lands in the shared per-basin store and is reused by later runs
    # even after the tar catches up — acceptable: both are gauge-corrected
    # MRMS (Pass1 vs Pass2), and the store is ephemeral on Space restarts.
    if recent and missing_ts:
        fmt, keep_days = recent
        unfilled = []
        for i, t in enumerate(sorted(missing_ts)):
            if cancel is not None and cancel.is_set():
                unfilled.extend(sorted(missing_ts)[i:])
                break
            if t > now or (now - t) > timedelta(days=keep_days):
                unfilled.append(t)
                continue
            try:
                p = hf_hub_download(repo, t.strftime(fmt), repo_type="dataset",
                                    cache_dir=cache_dir)
                with open(p, "rb") as fh:
                    a, xll, yll, cell, nod = _read_pqf(fh.read())
                clip = _clip(a, xll, yll, cell, nod, bbox)
                if clip is None:
                    unfilled.append(t)
                    continue
                sub, nxll, nyll = clip
                out_name = t.strftime(cfg.out_fmt)
                _write_pqf(os.path.join(out_dir, out_name), sub, nxll, nyll,
                           cell, nod)
                res.written.append(out_name)
                res.recent += 1
            except Exception:                    # not posted / aged out / hiccup
                unfilled.append(t)
        missing_ts = unfilled
    res.missing = [t.strftime(cfg.member_fmt) for t in sorted(missing_ts)]

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

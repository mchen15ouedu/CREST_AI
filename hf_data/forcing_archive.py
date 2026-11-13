"""Forcing archive: read a validated, basin-clipped forcing grid.

The one place that knows
  * the CONUS forcing tar layout in the data repo (one MONTH tar per
    variable/month, a YEAR tar fallback from before the reshard, member-name
    fallbacks inside a tar),
  * the PQF grid encoding (single float32 column 'v', row-major, row 0 = north,
    geometry in the parquet key/value metadata — matches
    scripts/crest_preprocess/tif2pqf.py),
  * per-variable value rules (units, nodata, plausible range).

Callers ask for "variable V at time t, clipped to this basin box":

    with ForcingArchive() as arc:
        g = arc.read("temp", t, bbox)     # Grid | None

and get a Grid back. Hub access sits behind the `fetch` seam
(repo path -> local path or bytes; FileNotFoundError when the path does not
exist), so tests serve in-memory tars.
"""
from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pyarrow.parquet as pq

HF_REPO = "vincewin/CREST_data"


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# per-variable value rules
# --------------------------------------------------------------------------- #
# THE RULE for out-of-range values: a grid is judged WHOLE, on its valid cells
# (finite, != nodata, > -9000), on the full domain BEFORE clipping. A grid
# whose valid cells leave the variable's plausible range is REJECTED with
# ForcingRangeError — never masked or clamped cell by cell, never run on
# silently. The one exception is a known unit mistake that can be undone
# exactly: a temperature grid stored in Kelvin is converted to degC on read
# (valid cells only) and flagged `was_kelvin`.
#
# Temperature members must be °C (EF5 UNIT=C). Every upstream source ships
# Kelvin (NLDAS/NARR/RTMA/HRRR 2-m TMP), and 76 NLDAS gap-fill hours of
# 2023-07..2024-02 reached the store unconverted: SNOW17 read ~270 °C and
# melted all SWE in one hour. A field whose valid-cell median is above
# TEMP_K_MEDIAN is Kelvin; anything else outside plausible °C is garbage.
TEMP_K_MEDIAN = 100.0
K_OFFSET = 273.15
TEMP_PLAUSIBLE_C = (-90.0, 70.0)

# var -> plausible (lo, hi) in the variable's EF5 unit, or None (unchecked).
# MRMS and PET have no range check yet: their archives were never seen to
# carry a unit mistake, and a guessed bound could reject real extremes.
PLAUSIBLE = {"temp": TEMP_PLAUSIBLE_C, "mrms": None, "pet": None}


class ForcingRangeError(ValueError):
    """A forcing grid outside its variable's plausible range."""


class TempUnitsError(ForcingRangeError):
    pass


def _valid(a: np.ndarray, nodata: float) -> np.ndarray:
    return np.isfinite(a) & (a != nodata) & (a > -9000)


def check_temp_units(a: np.ndarray, nodata: float, name: str,
                     convert: bool) -> tuple[np.ndarray, bool]:
    """(°C grid, was_kelvin). A Kelvin grid is converted on valid cells only
    (nodata stays nodata) when `convert`, else raises; a grid that is neither
    plausible °C nor plausible K always raises — never run on it silently.
    Writers call this with convert=False, readers with convert=True."""
    valid = _valid(a, nodata)
    if not valid.any():
        return a, False
    v = a[valid]
    med = float(np.median(v))
    if med <= TEMP_K_MEDIAN:
        lo, hi = TEMP_PLAUSIBLE_C
        if v.min() < lo or v.max() > hi:
            raise TempUnitsError(f"temperature {name}: values {v.min():.1f}..{v.max():.1f} "
                                 f"are not plausible °C")
        return a, False
    if not convert or not (200 <= med <= 340 and v.min() > 150):
        raise TempUnitsError(f"temperature {name}: median {med:.1f} — stored in "
                             f"Kelvin (or corrupt), expected °C")
    out = a.copy()
    out[valid] = (v.astype("float64") - K_OFFSET).astype(a.dtype)
    return out, True


def kelvin_to_degc(a: np.ndarray) -> np.ndarray:
    """Kelvin source field -> degC (for writers whose upstream ships Kelvin)."""
    return np.asarray(a, dtype="float64") - K_OFFSET


def validate(var: str, a: np.ndarray, nodata: float, name: str) -> tuple[np.ndarray, bool]:
    """Apply THE RULE above to a full-domain grid -> (grid in EF5 unit, was_kelvin)."""
    if var == "temp":
        return check_temp_units(a, nodata, name, convert=True)
    rng = PLAUSIBLE.get(var)
    if rng is not None:
        v = a[_valid(a, nodata)]
        if v.size and (v.min() < rng[0] or v.max() > rng[1]):
            raise ForcingRangeError(f"{var} {name}: values {v.min():.1f}..{v.max():.1f} "
                                    f"outside plausible {rng[0]}..{rng[1]}")
    return a, False


# --------------------------------------------------------------------------- #
# PQF encoding + clipping
# --------------------------------------------------------------------------- #
def _read_pqf(data: bytes):
    """bytes -> (array[nr,nc] float32, xll, yll, cell, nodata)."""
    pf = pq.ParquetFile(io.BytesIO(data))
    m = {k.decode(): v.decode() for k, v in pf.schema_arrow.metadata.items()
         if not k.startswith(b"ARROW")}
    nc, nr = int(m["ncols"]), int(m["nrows"])
    a = pf.read().column("v").to_numpy().reshape(nr, nc)
    return a, float(m["xllcorner"]), float(m["yllcorner"]), float(m["cellsize"]), float(m["nodata"])


def _encode_pqf(a: np.ndarray, xll: float, yll: float, cell: float, nodata: float,
                where) -> None:
    """Write grid `a` as a PQF to `where` (a path or a binary file object)."""
    import pyarrow as pa
    nr, nc = a.shape
    meta = {b"ncols": str(nc).encode(), b"nrows": str(nr).encode(),
            b"xllcorner": repr(float(xll)).encode(), b"yllcorner": repr(float(yll)).encode(),
            b"cellsize": repr(float(cell)).encode(), b"nodata": repr(float(nodata)).encode()}
    schema = pa.schema([pa.field("v", pa.float32())]).with_metadata(meta)
    pq.write_table(pa.table({"v": np.asarray(a).reshape(-1).astype("float32")},
                            schema=schema), where, compression="zstd")


def _write_pqf(path: str, a: np.ndarray, xll: float, yll: float, cell: float, nodata: float):
    """Atomic PQF file write, temp name unique per call: the shared store may
    have concurrent readers (EF5), and two writers - fleet workers or dashboard
    threads - may clip the same hour (nested basins share a store)."""
    from hf_data.atomicfile import atomic_path
    with atomic_path(path) as tmp:
        _encode_pqf(a, xll, yll, cell, nodata, tmp)


read_pqf = _read_pqf          # public name: PQF bytes -> (array, xll, yll, cell, nodata)


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
    """Geometry of the basin-clipped temp grid ->
    (nxll, nyll, cell, nrows, ncols) or None if bbox is outside the grid."""
    xll, yll, cell, nr, nc = TEMP_GRID[:5]
    win = clip_window(bbox, xll, yll, cell, nr, nc)
    if win is None:
        return None
    r0, r1, c0, c1 = win
    top = yll + nr * cell
    return xll + c0 * cell, top - r1 * cell, cell, r1 - r0, c1 - c0


def _clip(a, xll, yll, cell, nodata, bbox):
    """Clip full-domain grid to bbox (W,S,E,N). Returns (sub, nxll, nyll) or None."""
    nr, nc = a.shape
    win = clip_window(bbox, xll, yll, cell, nr, nc)
    if win is None:
        return None
    r0, r1, c0, c1 = win
    top = yll + nr * cell
    return a[r0:r1, c0:c1], xll + c0 * cell, top - r1 * cell


# --------------------------------------------------------------------------- #
# write interface
# --------------------------------------------------------------------------- #
def check_for_store(var: str, a: np.ndarray, nodata: float, name: str) -> None:
    """The per-variable rules a stored grid must pass — the same rules `read`
    applies, except nothing is converted: a temperature grid must already be
    degC (a Kelvin grid raises TempUnitsError), never stored to be undone later."""
    if var == "temp":
        check_temp_units(a, nodata, name, convert=False)
    else:
        validate(var, a, nodata, name)


def store_bytes(var: str, a: np.ndarray, xll: float, yll: float, cell: float,
                nodata: float, name: str = "") -> bytes:
    """Encode a full-domain grid of `var` as archive member bytes (PQF), after
    the per-variable rules. Raises ForcingRangeError rather than store a bad grid."""
    check_for_store(var, a, nodata, name or var)
    buf = io.BytesIO()
    _encode_pqf(a, xll, yll, cell, nodata, buf)
    return buf.getvalue()


def store_file(path: str, var: str, a: np.ndarray, xll: float, yll: float,
               cell: float, nodata: float) -> None:
    """Like store_bytes, written atomically to `path` (a basin-store clip)."""
    import os
    check_for_store(var, a, nodata, os.path.basename(path))
    _write_pqf(path, a, xll, yll, cell, nodata)


# --------------------------------------------------------------------------- #
# read interface
# --------------------------------------------------------------------------- #
@dataclass
class Grid:
    """A basin-clipped forcing grid, already validated for its variable."""
    values: np.ndarray        # float32 [nrows, ncols], row 0 = north
    xll: float
    yll: float
    cell: float
    nodata: float
    member: str               # tar member it came from
    was_kelvin: bool = False  # temperature stored in K, converted to degC on read


class ArchiveGap(LookupError):
    """The archive has no tar (month or year) for this variable and month."""
    def __init__(self, var: str, year: int, month: int):
        super().__init__(f"{var} archive has no tar for {year}-{month:02d}")
        self.var, self.year, self.month = var, year, month


def _hub_fetch(repo: str, cache_dir: str | None):
    def fetch(path: str) -> str:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError
        try:
            return hf_hub_download(repo, path, repo_type="dataset", cache_dir=cache_dir)
        except EntryNotFoundError as e:          # permanent 404 -> "not there"
            raise FileNotFoundError(path) from e
    return fetch


class ForcingArchive:
    def __init__(self, repo: str = HF_REPO, cache_dir: str | None = None, fetch=None):
        self._fetch = fetch or _hub_fetch(repo, cache_dir)
        self._open: dict[tuple[str, int, int], tuple[tarfile.TarFile, set[str]]] = {}

    def _tar(self, var: str, t: datetime) -> tuple[tarfile.TarFile, set[str]]:
        key = (var, t.year, t.month)
        if key not in self._open:
            self.close()                         # one open tar at a time (a year
            cfg = VARS[var]                      # of reads must not pile up fds)
            try:                                 # prefer the small month-tar
                src = self._fetch(t.strftime(cfg.month_fmt))
            except Exception:                    # fallback to year-tar (pre-reshard)
                try:
                    src = self._fetch(t.strftime(cfg.year_fmt))
                except FileNotFoundError:        # genuine data gap (permanent 404);
                    raise ArchiveGap(var, t.year, t.month) from None
                # (transient network errors raise other types and propagate)
            tf = (tarfile.open(fileobj=io.BytesIO(src))
                  if isinstance(src, (bytes, bytearray)) else tarfile.open(src))
            self._open[key] = (tf, set(tf.getnames()))
        return self._open[key]

    def read(self, var: str, t: datetime, bbox) -> Grid | None:
        cfg = VARS[var]
        tf, names = self._tar(var, t)
        member = t.strftime(cfg.member_fmt)
        if member not in names:                 # NARR-derived members use
            member = t.strftime(cfg.out_fmt)    # the generic name
        if member not in names:
            return None
        a, xll, yll, cell, nod = _read_pqf(tf.extractfile(member).read())
        a, was_k = validate(var, a, nod, member)     # full grid, before clipping
        clip = _clip(a, xll, yll, cell, nod, bbox)
        if clip is None:                        # box outside the archive grid
            return None
        sub, nxll, nyll = clip
        return Grid(sub, nxll, nyll, cell, nod, member, was_k)

    def close(self):
        for tf, _ in self._open.values():
            tf.close()
        self._open.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

"""Per-gauge state bundles: all EF5 state grids of one (gauge, model) packed
into ONE float16 parquet file for HF storage. Locally states always stay loose
EF5 GeoTIFFs — bundles exist only at rest/in transit — so EF5, the state
planner and the janitor are unaffected.

Why: the virtual-user fleet stores ~183 snapshots x 9k gauges; loose files
would be ~5M objects (HF recommends <100k per repo) and float16 keeps the
whole fleet under the 1 TB private quota (~0.05% rounding, harmless next to
forcing/parameter uncertainty — user-approved 2026-07-16).

Small companion .txt files EF5 writes next to the grids (gauge_relationships)
and the lake_set_<stamp>.json sidecars (hf_data/lakes.py) ride along in the
parquet schema metadata so a warm start is complete.

Lake grids (lake_storage_ in m3, up to ~1e10; lake_outflow_) are never
float16 (max 65504): they are stored losslessly as float32 columns.
"""
from __future__ import annotations

import glob
import io
import os

import numpy as np

_GEO_KEYS = (b"nrows", b"ncols", b"xllcorner", b"yllcorner", b"cellsize", b"nodata")


# RAM budget for one row block of all grids while packing: a gauge has ~1,370
# state grids (183 snapshots x vars), so loading them whole needs
# 2 B x cells x grids (tens of GB for a big basin's box)
PACK_BLOCK_BYTES = int(float(os.environ.get("CREST_BUNDLE_BLOCK_MB", "512")) * 1e6)


def _lossless(name: str) -> bool:
    """Grids packed as float32: lake storage (m3) overflows float16."""
    return name.startswith("lake_")


def pack_dir(state_dir: str, out_path: str | None = None) -> bytes | str | None:
    """All *.tif (+ *.txt, lake_set_*.json) in a state dir -> one f16 parquet
    bundle (lake grids f32); None if empty.

    Streams: grid headers first, then one row block of every grid at a time
    (PACK_BLOCK_BYTES), one parquet row group per block. Same columns, values
    and metadata as a single-table write, so any reader that concatenates row
    groups (pq.read_table, unpack) gets identical grids. Returns the bytes, or
    with out_path writes the file there and returns the path (the compressed
    bundle of a big basin need not sit in RAM either)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio
    from rasterio.windows import Window

    # grid headers; the bundle's geometry is that of the NEWEST state time (the
    # domain the gauge runs on now), and a grid on any other raster - other
    # size OR other origin/cell - is skipped, never packed under a wrong origin
    heads, skipped = [], []
    for p in sorted(glob.glob(os.path.join(state_dir, "*.tif"))):
        with rasterio.open(p) as ds:
            if ds.count != 1 or ds.dtypes[0] not in ("float32", "float64"):
                skipped.append(os.path.basename(p))
                continue
            cell = ds.res[0]
            heads.append((p, (ds.height, ds.width), float(ds.transform.c),
                          float(ds.transform.f - ds.height * cell), float(cell),
                          -9999.0 if ds.nodata is None else float(ds.nodata)))
    if not heads:
        return None

    def _stamp(p):                     # "<var>_YYYYMMDD_HHMM.tif" -> sortable
        parts = os.path.splitext(os.path.basename(p))[0].rsplit("_", 2)
        return "".join(parts[-2:]) if len(parts) == 3 else ""

    ref = max(heads, key=lambda h: (_stamp(h[0]), h[0]))
    _, shape, xll, yll, cell, nodata = ref
    tol = 1e-6 * cell
    keep = []
    for p, sh, x, y, c, _ in heads:
        if sh == shape and abs(x - xll) <= tol and abs(y - yll) <= tol \
                and abs(c - cell) <= tol:
            keep.append(p)
        else:                                  # domain changed mid-history
            skipped.append(os.path.basename(p))
    meta = {b"nrows": str(shape[0]).encode(),
            b"ncols": str(shape[1]).encode(),
            b"xllcorner": repr(xll).encode(),
            b"yllcorner": repr(yll).encode(),
            b"cellsize": repr(cell).encode(),
            b"nodata": repr(nodata).encode()}
    for p in sorted(glob.glob(os.path.join(state_dir, "*.txt"))
                    + glob.glob(os.path.join(state_dir, "lake_set_*.json"))):
        try:
            with open(p, "rb") as fh:
                meta[b"txt:" + os.path.basename(p).encode()] = fh.read()
        except OSError:
            pass
    if skipped:
        meta[b"skipped"] = ",".join(skipped).encode()
    names = [os.path.splitext(os.path.basename(p))[0] for p in keep]
    schema = pa.schema([pa.field(k, pa.float32() if _lossless(k) else pa.float16())
                        for k in names]).with_metadata(meta)
    nr, nc = shape
    width = sum(4 if _lossless(k) else 2 for k in names)      # bytes per cell, all grids
    rows = max(1, PACK_BLOCK_BYTES // (nc * width))
    sink = out_path if out_path else io.BytesIO()
    with pq.ParquetWriter(sink, schema, compression="zstd", use_dictionary=False) as w:
        for r0 in range(0, nr, rows):
            win = Window(0, r0, nc, min(rows, nr - r0))
            cols = {}
            for p, k in zip(keep, names):
                with rasterio.open(p) as ds:
                    a = ds.read(1, window=win).astype("float32").reshape(-1)
                    cols[k] = a if _lossless(k) else a.astype(np.float16)
            w.write_table(pa.table(cols, schema=schema))
    return out_path if out_path else sink.getvalue()


def unpack(data: bytes | str, state_dir: str, overwrite: bool = False) -> int:
    """Bundle -> loose EF5-safe GeoTIFFs (+ txt) in state_dir; returns files written.
    Existing files are kept unless overwrite (local, possibly newer, states win);
    a lake_set_<stamp>.json only comes with its lake_storage_<stamp> grid (it
    would otherwise vouch for a local grid of unknown lake set)."""
    import pyarrow.parquet as pq
    import rasterio
    from rasterio.transform import from_origin

    # one grid (column) at a time: a big basin's bundle decoded whole would need
    # 2 B x cells x ~1,370 grids of RAM
    pf = pq.ParquetFile(data if isinstance(data, str) else io.BytesIO(data))
    md = dict(pf.schema_arrow.metadata or {})
    nr, nc = int(md[b"nrows"]), int(md[b"ncols"])
    xll, yll = float(md[b"xllcorner"]), float(md[b"yllcorner"])
    cell, nodata = float(md[b"cellsize"]), float(md[b"nodata"])
    nod16 = np.float32(np.float16(nodata))         # what nodata became in f16
    os.makedirs(state_dir, exist_ok=True)
    n = 0
    written = set()
    for name in pf.schema_arrow.names:
        p = os.path.join(state_dir, name + ".tif")
        if not overwrite and os.path.exists(p):
            continue
        a = pf.read(columns=[name]).column(0).to_numpy().astype("float32").reshape(nr, nc)
        if not _lossless(name):
            a[a == nod16] = nodata                 # restore exact nodata value
        tmp = p + ".tmp"
        # untiled/uncompressed/stripped: the EF5 reader is picky (persist.py profile)
        with rasterio.open(tmp, "w", driver="GTiff", height=nr, width=nc, count=1,
                           dtype="float32", crs="EPSG:4326",
                           transform=from_origin(xll, yll + nr * cell, cell, cell),
                           nodata=nodata, tiled=False, blockysize=1) as ds:
            ds.write(a, 1)
        os.replace(tmp, p)
        written.add(name)
        n += 1
    for k, v in md.items():
        if k.startswith(b"txt:"):
            f = k[4:].decode()
            grid = "lake_storage_" + f[len("lake_set_"):-len(".json")]
            if f.startswith("lake_set_") and grid not in written:
                continue
            p = os.path.join(state_dir, f)
            if overwrite or not os.path.exists(p):
                with open(p, "wb") as fh:
                    fh.write(v)
                n += 1
    return n

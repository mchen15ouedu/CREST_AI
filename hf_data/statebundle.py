"""Per-gauge state bundles: all EF5 state grids of one (gauge, model) packed
into ONE float16 parquet file for HF storage. Locally states always stay loose
EF5 GeoTIFFs — bundles exist only at rest/in transit — so EF5, the state
planner and the janitor are unaffected.

Why: the virtual-user fleet stores ~183 snapshots x 9k gauges; loose files
would be ~5M objects (HF recommends <100k per repo) and float16 keeps the
whole fleet under the 1 TB private quota (~0.05% rounding, harmless next to
forcing/parameter uncertainty — user-approved 2026-07-16).

Small companion .txt files EF5 writes next to the grids (gauge_relationships)
ride along in the parquet schema metadata so a warm start is complete.
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


def pack_dir(state_dir: str, out_path: str | None = None) -> bytes | str | None:
    """All *.tif (+ *.txt) in a state dir -> one f16 parquet bundle; None if empty.

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

    keep, meta, shape = [], None, None
    skipped = []
    for p in sorted(glob.glob(os.path.join(state_dir, "*.tif"))):
        with rasterio.open(p) as ds:
            if ds.count != 1 or ds.dtypes[0] not in ("float32", "float64"):
                skipped.append(os.path.basename(p))
                continue
            sh = (ds.height, ds.width)
            if shape is None:
                shape = sh
                cell = ds.res[0]
                xll = ds.transform.c
                yll = ds.transform.f - ds.height * cell
                nodata = -9999.0 if ds.nodata is None else float(ds.nodata)
                meta = {b"nrows": str(sh[0]).encode(),
                        b"ncols": str(sh[1]).encode(),
                        b"xllcorner": repr(float(xll)).encode(),
                        b"yllcorner": repr(float(yll)).encode(),
                        b"cellsize": repr(float(cell)).encode(),
                        b"nodata": repr(nodata).encode()}
            elif sh != shape:                      # domain changed mid-history
                skipped.append(os.path.basename(p))
                continue
        keep.append(p)
    if not keep:
        return None
    for p in sorted(glob.glob(os.path.join(state_dir, "*.txt"))):
        try:
            with open(p, "rb") as fh:
                meta[b"txt:" + os.path.basename(p).encode()] = fh.read()
        except OSError:
            pass
    if skipped:
        meta[b"skipped"] = ",".join(skipped).encode()
    names = [os.path.splitext(os.path.basename(p))[0] for p in keep]
    schema = pa.schema([pa.field(k, pa.float16()) for k in names]).with_metadata(meta)
    nr, nc = shape
    rows = max(1, PACK_BLOCK_BYTES // (2 * nc * len(keep)))
    sink = out_path if out_path else io.BytesIO()
    with pq.ParquetWriter(sink, schema, compression="zstd", use_dictionary=False) as w:
        for r0 in range(0, nr, rows):
            win = Window(0, r0, nc, min(rows, nr - r0))
            cols = {}
            for p, k in zip(keep, names):
                with rasterio.open(p) as ds:
                    cols[k] = ds.read(1, window=win).astype("float32").reshape(-1).astype(np.float16)
            w.write_table(pa.table(cols, schema=schema))
    return out_path if out_path else sink.getvalue()


def unpack(data: bytes | str, state_dir: str, overwrite: bool = False) -> int:
    """Bundle -> loose EF5-safe GeoTIFFs (+ txt) in state_dir; returns files written.
    Existing files are kept unless overwrite (local, possibly newer, states win)."""
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
    for k, v in md.items():
        if k.startswith(b"txt:"):
            p = os.path.join(state_dir, k[4:].decode())
            if overwrite or not os.path.exists(p):
                with open(p, "wb") as fh:
                    fh.write(v)
                n += 1
    for name in pf.schema_arrow.names:
        p = os.path.join(state_dir, name + ".tif")
        if not overwrite and os.path.exists(p):
            continue
        a = pf.read(columns=[name]).column(0).to_numpy().astype("float32").reshape(nr, nc)
        a[a == nod16] = nodata                     # restore exact nodata value
        tmp = p + ".tmp"
        # untiled/uncompressed/stripped: the EF5 reader is picky (persist.py profile)
        with rasterio.open(tmp, "w", driver="GTiff", height=nr, width=nc, count=1,
                           dtype="float32", crs="EPSG:4326",
                           transform=from_origin(xll, yll + nr * cell, cell, cell),
                           nodata=nodata, tiled=False, blockysize=1) as ds:
            ds.write(a, 1)
        os.replace(tmp, p)
        n += 1
    return n

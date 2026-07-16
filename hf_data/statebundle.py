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


def pack_dir(state_dir: str) -> bytes | None:
    """All *.tif (+ *.txt) in a state dir -> one f16 parquet blob; None if empty."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio

    cols, meta, shape = {}, None, None
    skipped = []
    for p in sorted(glob.glob(os.path.join(state_dir, "*.tif"))):
        with rasterio.open(p) as ds:
            if ds.count != 1 or ds.dtypes[0] not in ("float32", "float64"):
                skipped.append(os.path.basename(p))
                continue
            a = ds.read(1).astype("float32")
            if shape is None:
                shape = a.shape
                cell = ds.res[0]
                xll = ds.transform.c
                yll = ds.transform.f - ds.height * cell
                nodata = -9999.0 if ds.nodata is None else float(ds.nodata)
                meta = {b"nrows": str(a.shape[0]).encode(),
                        b"ncols": str(a.shape[1]).encode(),
                        b"xllcorner": repr(float(xll)).encode(),
                        b"yllcorner": repr(float(yll)).encode(),
                        b"cellsize": repr(float(cell)).encode(),
                        b"nodata": repr(nodata).encode()}
            elif a.shape != shape:                 # domain changed mid-history
                skipped.append(os.path.basename(p))
                continue
        cols[os.path.splitext(os.path.basename(p))[0]] = a.reshape(-1).astype(np.float16)
    if not cols:
        return None
    for p in sorted(glob.glob(os.path.join(state_dir, "*.txt"))):
        try:
            with open(p, "rb") as fh:
                meta[b"txt:" + os.path.basename(p).encode()] = fh.read()
        except OSError:
            pass
    if skipped:
        meta[b"skipped"] = ",".join(skipped).encode()
    schema = pa.schema([pa.field(k, pa.float16()) for k in cols]).with_metadata(meta)
    buf = io.BytesIO()
    pq.write_table(pa.table(cols, schema=schema), buf,
                   compression="zstd", use_dictionary=False)
    return buf.getvalue()


def unpack(data: bytes | str, state_dir: str, overwrite: bool = False) -> int:
    """Bundle -> loose EF5-safe GeoTIFFs (+ txt) in state_dir; returns files written.
    Existing files are kept unless overwrite (local, possibly newer, states win)."""
    import pyarrow.parquet as pq
    import rasterio
    from rasterio.transform import from_origin

    if isinstance(data, str):
        with open(data, "rb") as fh:
            data = fh.read()
    t = pq.read_table(io.BytesIO(data))
    md = dict(t.schema.metadata or {})
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
    for name in t.column_names:
        p = os.path.join(state_dir, name + ".tif")
        if not overwrite and os.path.exists(p):
            continue
        a = t[name].to_numpy().astype("float32").reshape(nr, nc)
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

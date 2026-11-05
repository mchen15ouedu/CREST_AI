"""Shared fixtures: a synthetic basin-extent table and tiny lattice rasters.

Run from the repo root:  python -m pytest tests -q
(not deployed to the Spaces; no network, no EF5)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "fleet"))

RES = 1.0 / 1200.0              # HydroSHEDS 3 arc-second lattice


def write_grid(path, a, west, north, nodata=-9999.0, res=RES):
    import rasterio
    from rasterio.transform import from_origin
    a = np.asarray(a, dtype="float32")
    with rasterio.open(path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326", nodata=nodata,
                       transform=from_origin(west, north, res, res)) as dst:
        dst.write(a, 1)


def huc_chain_api(bounds, huc12="123", upstream=()):
    """hucdomain's metadata calls behind eventsim's native-capacity preflight
    (gauge_huc12 / upstream_chain / index) for a one-unit HUC12 chain with
    (w, s, e, n) `bounds`. `upstream`: units the index lists upstream of it
    that the chain leaves out — a traversal that stopped short."""
    w, s, e, n = bounds
    ids = [huc12, *upstream]
    index = {"huc12": np.array(ids), "_pos": {h: i for i, h in enumerate(ids)},
             "_up": {huc12: list(range(1, len(ids)))} if upstream else {},
             "minx": np.full(len(ids), w), "miny": np.full(len(ids), s),
             "maxx": np.full(len(ids), e), "maxy": np.full(len(ids), n)}
    return {"gauge_huc12": lambda lat, lon: huc12,
            "upstream_chain": lambda h0: [{"huc12": huc12, "depth": 0}],
            "index": lambda: index}


ROWS = [
    # id, kind, status, source, flags, area, snap_lat, snap_lon, snap_area, carved, w,s,e,n, cells
    ("05467000", "g", "ok", "gages2", "", 448.6, 41.2, -90.5, 433.6, 433.4,
     -90.975, 41.024, -90.157, 41.263, 281834),
    ("08263500", "g", "ok", "gages2", "", 21799.1, 37.0, -106.0, 22039.8, 21803.4,
     -107.6, 36.6, -105.0, 38.5, 6726692),
    ("15019990", "g", "no_terrain", "gages2", "no_terrain", 37.5, np.nan, np.nan,
     np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None),
    ("V70520944", "v", "ok", "hybas", "", 15217.5, 42.0, -122.0, 15421.2, 15272.7,
     -123.1, 41.2, -120.6, 43.4, 7784140),
    ("V99999999", "v", "over_cap", "d8", "over_cap", 3.1e6, 29.0, -89.0, 3.2e6, np.nan,
     -114.0, 28.9, -77.8, 50.0, 1128571980),
]


def make_table(path, rows=ROWS, meta=None):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = ["id", "kind", "status", "source", "flags", "area", "snap_lat", "snap_lon",
            "snap_area_km2", "carved_km2", "w", "s", "e", "n", "cells"]
    df = pd.DataFrame(rows, columns=cols)
    df["cells"] = df["cells"].astype("Int64")
    t = pa.Table.from_pandas(df, preserve_index=False)
    m = {"table_version": "test-1", "cap_cells": 36_000_000, "partial": False}
    m.update(meta or {})
    t = t.replace_schema_metadata({**(t.schema.metadata or {}),
                                   b"basin_extent": json.dumps(m).encode()})
    pq.write_table(t, path)
    return path


@pytest.fixture
def table(tmp_path, monkeypatch):
    """basinextent pointed at a fresh synthetic table."""
    from hf_data import basinextent
    p = make_table(str(tmp_path / "basin_extent.parquet"))
    monkeypatch.setattr(basinextent, "TABLE_PATH", p)
    monkeypatch.setattr(basinextent, "_tbl", None)
    monkeypatch.setattr(basinextent, "_CAP_ENV", "")
    yield basinextent
    basinextent._tbl = None

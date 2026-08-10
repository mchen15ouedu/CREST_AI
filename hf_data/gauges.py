"""HF-backed gauge catalog + snapping for the AQUAH->CREST_demo integration.

Replaces AQUAH's gauge_meta.csv lookups with the GAGES-II catalog hosted in
`vincewin/CREST_data` (gauges/gagesII_9322.parquet). Provides:
  - load_catalog(bbox)          candidate gauges in an extent (for the AOS agent)
  - get_gauge_coordinates(sid)  drop-in for AQUAH gauge_processor (lat, lon)
  - snap_to_stream(...)         nudge a gauge onto the modeled stream via flow-acc

AQUAH's control uses [Gauge] LON=/LAT=, so EF5 finds the cell itself; snapping
just improves which cell that is on the 90 m HydroSHEDS grid.
"""
from __future__ import annotations

import os
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import truststore
truststore.inject_into_ssl()
from huggingface_hub import hf_hub_download

HF_REPO = "vincewin/CREST_data"
_FACC_URL = f"/vsicurl/https://huggingface.co/datasets/{HF_REPO}/resolve/main/basic/na_acc_3s.tif"


@lru_cache(maxsize=1)
def _catalog() -> pd.DataFrame:
    path = hf_hub_download(HF_REPO, "gauges/gagesII_9322.parquet", repo_type="dataset")
    df = pd.read_parquet(path)
    df["STAID"] = df["STAID"].astype(str).str.zfill(8)
    return df


def load_catalog(bbox=None) -> pd.DataFrame:
    """All gauges, or those inside bbox=(W,S,E,N)."""
    df = _catalog()
    if bbox is None:
        return df.copy()
    W, S, E, N = bbox
    m = ((df.LNG_GAGE >= W) & (df.LNG_GAGE <= E)
         & (df.LAT_GAGE >= S) & (df.LAT_GAGE <= N))
    return df[m].copy()


def get_gauge_coordinates(station_id: str):
    """(lat, lon) for a USGS station id, or None. Matches AQUAH's signature."""
    sid = str(station_id).zfill(8)
    row = _catalog().loc[_catalog().STAID == sid]
    if row.empty:
        return None
    return float(row.LAT_GAGE.iloc[0]), float(row.LNG_GAGE.iloc[0])


def _cell_area_km2(lat: float, cell_deg: float) -> float:
    return (cell_deg * 111.32) * (cell_deg * 111.32 * math.cos(math.radians(lat)))


def snap_to_stream(lat: float, lon: float, drain_sqkm: float | None = None,
                   search_deg: float = 0.02, unsafe_ssl: bool | None = None):
    """Snap (lat, lon) onto the flow-accumulation stream near the gauge.

    If drain_sqkm is given, pick the nearby cell whose upstream area best matches
    it; else pick the max-accumulation cell in the window. Returns a dict with
    snapped lat/lon, facc (upstream cells), and the drainage-area error.
    """
    import rasterio
    from rasterio.windows import from_bounds
    if unsafe_ssl is None:
        unsafe_ssl = os.name == "nt"
    if unsafe_ssl:
        os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")

    W, S, E, N = lon - search_deg, lat - search_deg, lon + search_deg, lat + search_deg
    from hf_data import basic
    with rasterio.open(basic._cog_src("basic/na_acc_3s.tif")) as ds:
        win = from_bounds(W, S, E, N, ds.transform).round_offsets().round_lengths()
        acc = ds.read(1, window=win).astype("float64")
        tr = ds.window_transform(win)
        nodata = ds.nodata
        cell_deg = ds.res[0]
    acc[acc == nodata] = np.nan

    nr, nc = acc.shape
    # gauge pixel within the window
    gcol = int(round((lon - tr.c) / cell_deg))
    grow = int(round((tr.f - lat) / cell_deg))

    if drain_sqkm and drain_sqkm > 0:
        expected = drain_sqkm / _cell_area_km2(lat, cell_deg)
        score = np.abs(acc - expected)          # closest upstream area
    else:
        score = -acc                            # largest stream
    # bias toward the gauge location so we don't jump to a different river
    rr, cc = np.mgrid[0:nr, 0:nc]
    dist = np.hypot(rr - grow, cc - gcol)
    score = score + np.nan_to_num(dist, nan=1e9) * (np.nanmax(np.abs(score)) / (nr + nc) * 0.5)
    score[np.isnan(acc)] = np.inf
    r, c = np.unravel_index(np.argmin(score), score.shape)

    snap_lon = tr.c + (c + 0.5) * cell_deg
    snap_lat = tr.f - (r + 0.5) * cell_deg
    facc = float(acc[r, c])
    area_km2 = facc * _cell_area_km2(lat, cell_deg)
    err = (abs(area_km2 - drain_sqkm) / drain_sqkm * 100.0) if drain_sqkm else None
    return {"lat": snap_lat, "lon": snap_lon, "facc": facc,
            "area_km2": area_km2, "drain_err_pct": err,
            "moved_cells": float(math.hypot(r - grow, c - gcol))}


if __name__ == "__main__":
    # Allagash gauge 01011000: 47.0696, -69.0795, 3186.8 km^2
    bbox = (-69.83, 46.32, -68.33, 47.82)
    cat = load_catalog(bbox)
    print(f"catalog: {len(_catalog())} gauges; {len(cat)} in Allagash bbox")
    print(cat[["STAID", "STANAME", "LAT_GAGE", "LNG_GAGE", "DRAIN_SQKM"]].head(8).to_string(index=False))
    print("get_gauge_coordinates(01011000):", get_gauge_coordinates("01011000"))
    snap = snap_to_stream(47.0696, -69.0795, 3186.844)
    print("snap:", {k: round(v, 4) if isinstance(v, float) else v for k, v in snap.items()})

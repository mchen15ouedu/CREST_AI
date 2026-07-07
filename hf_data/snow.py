"""SNOW17 snow detection + parameters (task #7-B).

Snow decision (user-chosen: data-driven temp with LLM/user override):
  - user override on/off wins;
  - else a cheap season/region pre-screen skips obviously-warm cases (no temp
    download); otherwise confirm from HF temperature forcing — enable SNOW17 if
    the basin drops below ~freezing during the window.

SNOW17 params: ALL 8 EF5 parameters (uadj/mbase/mfmax/mfmin/tipm/nmf/plwhc/scf)
come as calibrated 0.1° CONUS grids from CREST_data param/snow17/v1/ — the
"snow17_conus_operational_v1" set (differentiable Snow-17 vs SNODAS SWE,
WY2010-2024; see the README in that folder). Scalars are ×1.0 multipliers on
the grids (advanced-panel overrides scale them); if a grid can't be clipped
(outside CONUS / network), that parameter falls back to an absolute physical
default. pxtemp is a real SNOW17 parameter since fork 5a26a86 (per-cell
rain/snow partition temperature).

Temperature extrapolation: the temp forcing is coarse (NLDAS 0.125-deg /
NARR 32 km), so build_temp_dem() aggregates the basin DEM onto the clipped
temp-grid geometry; the control file passes it as [TEMPForcing] DEM= and EF5
lapses each model cell by -6.5 C/km relative to its temp pixel's mean elevation
(TempReader.cpp tempDEM path).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np

SNOW_THRESHOLD_C = float(os.environ.get("CREST_SNOW_TEMP_C", "1.0"))

# all 9 params gridded -> scalars are ×1.0 multipliers on the grids
# (pxtemp needs fork >= 5a26a86, which made it a real SNOW17 parameter)
SNOW_DEFAULTS = {"uadj": 1.0, "mbase": 1.0, "mfmax": 1.0, "mfmin": 1.0,
                 "tipm": 1.0, "nmf": 1.0, "plwhc": 1.0, "scf": 1.0, "pxtemp": 1.0}
# absolute fallbacks (mid-range of the calibrated set) for params whose grid
# could not be clipped — a bare multiplier of 1.0 would be nonsense then
# (except pxtemp, where 1.0 °C IS the legacy hardcoded threshold)
SNOW_ABS_FALLBACK = {"uadj": 0.10, "mbase": 0.3, "mfmax": 1.2, "mfmin": 0.2,
                     "tipm": 0.3, "nmf": 0.15, "plwhc": 0.10, "scf": 1.0,
                     "pxtemp": 1.0}
_V1 = "param/snow17/v1"
SNOW_GRID_COGS = {
    f"{p}_grid": f"{_V1}/{p}_conus_0p1deg.tif"
    for p in ("uadj", "mbase", "mfmax", "mfmin", "tipm", "nmf", "plwhc", "scf",
              "pxtemp")
}
_RESOLVE = "https://huggingface.co/datasets/vincewin/CREST_data/resolve/main"


def snow_params(overrides: dict | None = None, gridded=None) -> dict:
    """Scalars for the control file. `gridded` = the grid keys that actually
    clipped (from clip_snow_grids) — params without a grid get the absolute
    fallback instead of a bare 1.0 multiplier."""
    p = dict(SNOW_DEFAULTS)
    if gridded is not None:
        have = {k[:-5] for k in gridded}              # "uadj_grid" -> "uadj"
        for k in p:
            if k not in have:
                p[k] = SNOW_ABS_FALLBACK[k]
    if overrides:
        p.update({k: float(v) for k, v in overrides.items() if k in p})
    return p


def clip_snow_grids(bbox, out_dir: str, unsafe_ssl: bool | None = None) -> dict:
    """Clip the 8 SNOW17 v1 grids to the basin. Returns {control_key: local_path}."""
    import rasterio
    from rasterio.windows import from_bounds
    if unsafe_ssl is None:
        unsafe_ssl = os.name == "nt"
    if unsafe_ssl:
        os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
    os.makedirs(out_dir, exist_ok=True)
    W, S, E, N = bbox
    out = {}
    for key, cog in SNOW_GRID_COGS.items():
        try:
            with rasterio.open(f"/vsicurl/{_RESOLVE}/{cog}") as src:
                win = from_bounds(W, S, E, N, src.transform).round_offsets().round_lengths()
                data = src.read(1, window=win, boundless=True, fill_value=src.nodata)
                if data.size == 0 or 0 in data.shape:
                    continue
                # standardized clip: Float32, strip-organized, nodata=-9999, WGS84
                data = data.astype("float32")
                if src.nodata is not None:
                    data[data == float(src.nodata)] = -9999.0
                profile = src.profile.copy()
                profile.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                               transform=src.window_transform(win),
                               tiled=False, blockysize=1, dtype="float32",
                               nodata=-9999.0, crs="EPSG:4326", interleave="pixel")
                profile.pop("compress", None)  # EF5 TifGrid-safe: plain 1-row strips
            path = os.path.join(out_dir, os.path.basename(cog).replace("_usa", ""))
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
            out[key] = path
        except Exception:
            pass
    return out


def build_temp_dem(bbox, dem_path: str, out_path: str) -> str | None:
    """DEM aggregated (mean) onto the basin-clipped temp-grid geometry, for
    EF5's [TEMPForcing] DEM= temperature extrapolation. EF5 matches it to the
    temp grid by shape and indexes it with the temp grid's own row/col
    (TempReader.cpp), so the geometry must be cell-identical to the clipped
    temp PQFs — guaranteed by sharing forcing.clip_window's arithmetic."""
    from hf_data.forcing import temp_clip_geometry
    geom = temp_clip_geometry(bbox)
    if geom is None or not dem_path or not os.path.exists(dem_path):
        return None
    nxll, nyll, cell, tnr, tnc = geom
    try:
        import rasterio
        from rasterio.transform import from_origin
        with rasterio.open(dem_path) as ds:
            dem = ds.read(1).astype("float64")
            tr = ds.transform
            nod = ds.nodata
        # DEM pixel centers -> temp grid row/col bins
        rr, cc = np.meshgrid(np.arange(dem.shape[0]), np.arange(dem.shape[1]), indexing="ij")
        lon = tr.c + (cc + 0.5) * tr.a
        lat = tr.f + (rr + 0.5) * tr.e
        top = nyll + tnr * cell
        ti = np.floor((top - lat) / cell).astype(int)
        tj = np.floor((lon - nxll) / cell).astype(int)
        ok = np.isfinite(dem) & (dem > -500) & (dem < 9000)
        if nod is not None:
            ok &= dem != nod
        ok &= (ti >= 0) & (ti < tnr) & (tj >= 0) & (tj < tnc)
        if not ok.any():
            return None
        flat = ti[ok] * tnc + tj[ok]
        sums = np.bincount(flat, weights=dem[ok], minlength=tnr * tnc)
        cnts = np.bincount(flat, minlength=tnr * tnc)
        mean = np.divide(sums, cnts, out=np.zeros_like(sums), where=cnts > 0)
        agg = mean.reshape(tnr, tnc).astype("float32")
        agg[cnts.reshape(tnr, tnc) == 0] = float(dem[ok].mean())  # offshore/edge fill
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with rasterio.open(
            out_path, "w", driver="GTiff", height=tnr, width=tnc, count=1,
            dtype="float32", crs="EPSG:4326", nodata=-9999.0,
            transform=from_origin(nxll, top, cell, cell),
            tiled=False, blockysize=1, interleave="pixel",   # EF5 TifGrid-safe
        ) as dst:
            dst.write(agg, 1)
        return out_path
    except Exception:
        return None


def _mean_elev(dem_path):
    if not dem_path or not os.path.exists(dem_path):
        return None
    try:
        import rasterio
        with rasterio.open(dem_path) as ds:
            a = ds.read(1).astype("float32")
            nod = ds.nodata
        v = a[(a != nod) & np.isfinite(a) & (a > -500) & (a < 9000)] if nod is not None else a
        return float(np.nanmean(v)) if v.size else None
    except Exception:
        return None


def _basin_min_temp(bbox, t_start, t_end, sample_hours=6):
    """Min basin temperature (°C) over the window from HF temp forcing (sampled)."""
    import io
    import tarfile
    import truststore
    truststore.inject_into_ssl()
    from huggingface_hub import hf_hub_download
    from hf_data.forcing import VARS, _read_pqf, _clip
    cfg = VARS["temp"]
    steps = []
    t = t_start.replace(minute=0, second=0, microsecond=0)
    while t <= t_end:
        steps.append(t)
        t += timedelta(hours=sample_hours)
    by_month = {}
    for s in steps:
        by_month.setdefault((s.year, s.month), []).append(s)
    gmin = None
    for (yr, mo), group in sorted(by_month.items()):
        ref = group[0]
        try:
            tar_path = hf_hub_download("vincewin/CREST_data", ref.strftime(cfg.month_fmt), repo_type="dataset")
        except Exception:
            try:
                tar_path = hf_hub_download("vincewin/CREST_data", ref.strftime(cfg.year_fmt), repo_type="dataset")
            except Exception:
                continue
        with tarfile.open(tar_path) as tf:
            names = set(tf.getnames())
            for s in group:
                member = s.strftime(cfg.member_fmt)
                if member not in names:                # NARR-derived members
                    member = s.strftime(cfg.out_fmt)   # use the generic name
                if member not in names:
                    continue
                a, xll, yll, cell, nod = _read_pqf(tf.extractfile(member).read())
                clip = _clip(a, xll, yll, cell, nod, bbox)
                if clip is None:
                    continue
                sub = clip[0]
                v = sub[(sub > -90) & (sub < 60) & np.isfinite(sub)]     # plausible °C only
                if v.size:
                    m = float(v.min())
                    gmin = m if gmin is None else min(gmin, m)
    return gmin


def detect_snow(bbox, t_start, t_end, dem_path=None, force: str | None = None, use_temp: bool = True) -> dict:
    """Decide whether SNOW17 should be on. force in {'on','off',None}."""
    if force == "on":
        return {"snow": True, "reason": "user override: on"}
    if force == "off":
        return {"snow": False, "reason": "user override: off"}

    lat = (bbox[1] + bbox[3]) / 2.0
    months = {t_start.month, t_end.month}
    cold_season = any(m in (10, 11, 12, 1, 2, 3, 4) for m in months)
    elev = _mean_elev(dem_path)
    warm = (not cold_season) and lat < 40 and (elev is None or elev < 1500)
    if warm:
        return {"snow": False, "reason": f"warm season/region (lat {lat:.0f}°, "
                f"{'elev %.0f m' % elev if elev is not None else 'low elev'})"}

    if use_temp:                                          # data-driven confirmation
        try:
            mn = _basin_min_temp(bbox, t_start, t_end)
        except Exception:
            mn = None
        if mn is not None:
            return {"snow": mn < SNOW_THRESHOLD_C, "min_temp_c": round(mn, 1),
                    "reason": f"basin min temperature {mn:.1f}°C"}
    return {"snow": True, "reason": f"cold season/region (lat {lat:.0f}°, "
            f"{'elev %.0f m' % elev if elev is not None else 'elev n/a'})"}


if __name__ == "__main__":
    # warm case (Kerrville, July) -> no download; cold case (Colorado, Jan) -> temp confirm
    print("Kerrville Jul:", detect_snow((-99.6, 29.8, -98.6, 30.6),
          datetime(2025, 7, 3), datetime(2025, 7, 6), use_temp=False))
    print("Colorado Jan (temp): ", detect_snow((-106.5, 38.5, -105.5, 39.5),
          datetime(2025, 1, 1), datetime(2025, 1, 3), use_temp=True))

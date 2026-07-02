"""HF-backed basic-data adapter for the AQUAH->CREST_demo integration.

Replaces AQUAH's live basic-data download + clip: windowed-reads the HydroSHEDS
DEM / flow-dir / flow-acc COGs hosted in `vincewin/CREST_data` and writes the
three EF5 inputs (dem_clip.tif, fdir_clip.tif, facc_clip.tif) for a basin bbox.

All three HydroSHEDS grids share one grid, so clipping the same bbox yields
pixel-aligned outputs (EF5 masks the catchment internally from the outlet cell).

On this network GDAL's curl needs GDAL_HTTP_UNSAFESSL=YES (corporate MITM);
that is LOCAL-ONLY and not set on HF cloud runners.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import rasterio
from rasterio.windows import from_bounds

HF_REPO = "vincewin/CREST_data"
_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# EF5 input filename <- COG path in CREST_data
BASIC_GRIDS = {
    "dem_clip.tif": "basic/na_con_3s.tif",
    "fdir_clip.tif": "basic/na_dir_3s.tif",
    "facc_clip.tif": "basic/na_acc_3s.tif",
}


def _vsicurl(cog_path: str) -> str:
    return f"/vsicurl/{_RESOLVE}/{cog_path}"


@dataclass
class ClipResult:
    out_dir: str
    files: dict[str, str]
    width: int
    height: int
    bounds: tuple[float, float, float, float]
    derived: bool = False        # dir/acc regenerated from the DEM (pysheds)


# ESRI D8 codes EF5 expects with ESRIDDM=true (E,SE,S,SW,W,NW,N,NE)
_ESRI_D8 = (1, 2, 4, 8, 16, 32, 64, 128)


def _dir_acc_usable(fdir_path: str, facc_path: str, dem_path: str,
                    max_bad_frac: float = 0.05) -> bool:
    """Sanity-check the clipped flow-dir/acc: on cells where the DEM is valid,
    the direction must be an ESRI D8 code and the accumulation non-negative."""
    import numpy as np
    with rasterio.open(dem_path) as d, rasterio.open(fdir_path) as f, \
            rasterio.open(facc_path) as a:
        dem, fdir, facc = d.read(1), f.read(1), a.read(1)
    ok_dem = dem != -9999.0
    if not ok_dem.any():
        return False
    good_dir = np.isin(fdir, _ESRI_D8)
    good_acc = facc >= 0
    bad = ok_dem & ~(good_dir & good_acc)
    return (bad.sum() / ok_dem.sum()) <= max_bad_frac


def derive_dir_acc(dem_path: str, fdir_path: str, facc_path: str) -> None:
    """Fallback: derive D8 flow direction + accumulation from the clipped DEM
    with pysheds (condition -> flowdir -> accumulation). pysheds' default
    dirmap (N,NE,E,SE,S,SW,W,NW)=(64,128,1,2,4,8,16,32) IS the ESRI encoding."""
    import numpy as np
    from pysheds.grid import Grid

    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)
    dem_f = grid.fill_pits(dem)
    dem_f = grid.fill_depressions(dem_f)
    dem_f = grid.resolve_flats(dem_f)
    fdir = grid.flowdir(dem_f)
    facc = grid.accumulation(fdir)

    with rasterio.open(dem_path) as ref:
        profile = ref.profile.copy()
    profile.update(dtype="float32", nodata=-9999.0,
                   tiled=False, blockysize=1, crs="EPSG:4326",
                   interleave="pixel")
    profile.pop("compress", None)
    mask = np.asarray(dem) == -9999.0
    fdir = np.asarray(fdir, dtype="float32")
    fdir[~np.isin(fdir, _ESRI_D8)] = -9999.0   # pysheds pits/flats: -1/-2/0
    facc = np.asarray(facc, dtype="float32")
    for path, out in ((fdir_path, fdir), (facc_path, facc)):
        out[mask] = -9999.0
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(out, 1)


def clip_basic_data(bbox, out_dir: str, unsafe_ssl: bool | None = None) -> ClipResult:
    """Clip DEM/DDM/FAM to `bbox` (W, S, E, N in EPSG:4326) from the HF COGs.

    Writes dem_clip.tif / fdir_clip.tif / facc_clip.tif into `out_dir`, all on
    the same window/geotransform. Returns a ClipResult.
    """
    W, S, E, N = bbox
    # local-network MITM workaround; harmless (and unset) on HF runners
    if unsafe_ssl is None:
        unsafe_ssl = os.name == "nt"
    if unsafe_ssl:
        os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")

    os.makedirs(out_dir, exist_ok=True)
    files: dict[str, str] = {}
    ref_shape = None
    ref_bounds = None

    fetch_failed: list[str] = []
    for out_name, cog in BASIC_GRIDS.items():
        try:
            _clip_one(cog, out_name, bbox, out_dir, files)
        except Exception:
            if out_name == "dem_clip.tif":
                raise                          # DEM is required — no fallback
            fetch_failed.append(out_name)
    _shapes = {}
    for out_name, path in files.items():
        with rasterio.open(path) as ds:
            _shapes[out_name] = (ds.height, ds.width)
            if out_name == "dem_clip.tif":
                t = ds.transform
                ref_shape = (ds.height, ds.width)
                ref_bounds = (t.c, t.f + ds.height * t.e,
                              t.c + ds.width * t.a, t.f)
    for out_name, shp in _shapes.items():
        if shp != ref_shape:
            fetch_failed.append(out_name)      # misaligned -> rederive

    # fallback: if dir/acc failed to fetch, are misaligned, or look broken
    # (bad D8 codes, negative accumulation), regenerate from the clipped DEM
    derived = False
    fdir_p = os.path.join(out_dir, "fdir_clip.tif")
    facc_p = os.path.join(out_dir, "facc_clip.tif")
    try:
        usable = (not fetch_failed) and _dir_acc_usable(
            files["fdir_clip.tif"], files["facc_clip.tif"], files["dem_clip.tif"])
    except Exception:
        usable = False
    if not usable:
        derive_dir_acc(files["dem_clip.tif"], fdir_p, facc_p)
        files["fdir_clip.tif"], files["facc_clip.tif"] = fdir_p, facc_p
        derived = True

    return ClipResult(out_dir=out_dir, files=files,
                      width=ref_shape[1], height=ref_shape[0], bounds=ref_bounds,
                      derived=derived)


def _clip_one(cog: str, out_name: str, bbox, out_dir: str, files: dict) -> None:
    W, S, E, N = bbox
    with rasterio.open(_vsicurl(cog)) as src:
        win = from_bounds(W, S, E, N, src.transform).round_offsets().round_lengths()
        # standardized clip for EF5's TifGrid reader: single-band Float32,
        # strip-organized, nodata=-9999, WGS84
        data = src.read(1, window=win).astype("float32")
        if src.nodata is not None:
            data[data == float(src.nodata)] = -9999.0
        transform = src.window_transform(win)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff", height=data.shape[0], width=data.shape[1],
            transform=transform, tiled=False, blockysize=1,
            dtype="float32", nodata=-9999.0, crs="EPSG:4326",
            interleave="pixel",         # contiguous — the COGs are band-
                                        # interleaved, which EF5's reader
                                        # (TIFFReadScanline sample arg) rejects
        )
        profile.pop("compress", None)   # uncompressed 1-row strips, like
                                        # EF5's own WriteFloatTifGrid output
    out_path = os.path.join(out_dir, out_name)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    files[out_name] = out_path


if __name__ == "__main__":
    import sys, time
    # smoke test: Allagash gauge 01011000 basin bbox
    bbox = (-69.83, 46.32, -68.33, 47.82)
    out = sys.argv[1] if len(sys.argv) > 1 else "._clip_test"
    t = time.time()
    r = clip_basic_data(bbox, out)
    print(f"clipped {r.width}x{r.height} in {time.time()-t:.1f}s -> {out}")
    for name, path in r.files.items():
        with rasterio.open(path) as ds:
            a = ds.read(1)
            nod = ds.nodata
            valid = a[a != nod] if nod is not None else a
            vmin = float(valid.min()) if valid.size else float("nan")
            vmax = float(valid.max()) if valid.size else float("nan")
            print(f"  {name}: {ds.dtypes[0]} nodata={nod} range[{vmin:.1f},{vmax:.1f}]")

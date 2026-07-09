"""HF-backed parameter-grid adapter for the AQUAH->CREST_demo integration.

Clips the 1 km CONUS CREST/KW parameter COGs hosted in `vincewin/CREST_data`
(param/crest/*, param/kw/*) to a basin bbox and returns the EF5 control-file
`<param>_grid=` mapping. Used when running with spatially-distributed params
(grid x scalar-multiplier) instead of AQUAH's uniform scalars.

Grid<-COG mapping mirrors the calibrated conus-crest-hydro controls:
  wm_grid<-wm_usa  im_grid<-im_usa  fc_grid<-ksat_usa  b_grid<-b_usa
  leaki_grid<-leaki_usa  alpha_grid<-alpha_usa  beta_grid<-beta_usa  alpha0_grid<-alpha0_usa
"""
from __future__ import annotations

import os
import time

import rasterio
from rasterio.windows import from_bounds

HF_REPO = "vincewin/CREST_data"
_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# GDAL-level retries on 429/5xx range reads — several gauges clip in parallel
# and HF occasionally rate-limits the burst (process-wide; also covers the
# terrain + snow-grid vsicurl reads)
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "3")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")

# EF5 control key -> COG path in CREST_data -> local filename
PARAM_GRIDS = {
    "wm_grid": ("param/crest/wm_usa.tif", "wm.tif"),
    "im_grid": ("param/crest/im_usa.tif", "im.tif"),
    "fc_grid": ("param/crest/ksat_usa.tif", "fc.tif"),
    "b_grid": ("param/crest/b_usa.tif", "b.tif"),
    "leaki_grid": ("param/kw/leaki_usa.tif", "leaki.tif"),
    "alpha_grid": ("param/kw/alpha_usa.tif", "alpha.tif"),
    "beta_grid": ("param/kw/beta_usa.tif", "beta.tif"),
    "alpha0_grid": ("param/kw/alpha0_usa.tif", "alpha0.tif"),
}


def _vsicurl(cog: str) -> str:
    return f"/vsicurl/{_RESOLVE}/{cog}"


def clip_param_grids(bbox, out_dir: str, keys=None, unsafe_ssl: bool | None = None) -> dict:
    """Clip requested param COGs to bbox (W,S,E,N). Returns {control_key: local_path}."""
    if unsafe_ssl is None:
        unsafe_ssl = os.name == "nt"
    if unsafe_ssl:
        os.environ.setdefault("GDAL_HTTP_UNSAFESSL", "YES")
    os.makedirs(out_dir, exist_ok=True)
    W, S, E, N = bbox
    keys = keys or list(PARAM_GRIDS)
    out, skipped = {}, {}
    for key in keys:
        cog, fname = PARAM_GRIDS[key]
        cached = os.path.join(out_dir, fname)
        if os.path.exists(cached):             # shared-store hit: no remote read
            out[key] = cached
            continue
        # /vsicurl reads fail transiently (HF rate limits parallel gauge runs;
        # GDAL then reports "not recognized as being in a supported file
        # format") — retry with backoff like basic.clip_basic_data
        err = None
        for attempt in range(3):
            try:
                _clip_one_param(cog, fname, bbox, out_dir, key, out, skipped)
                err = None
                break
            except Exception as e:
                err = e
                time.sleep(2 * (attempt + 1))
        if err is not None:
            raise err
    if skipped:
        for k, why in skipped.items():
            print(f"  WARN param grid {k} skipped: {why} (falls back to scalar)")
    return out


def _clip_one_param(cog: str, fname: str, bbox, out_dir: str, key: str,
                    out: dict, skipped: dict) -> None:
    W, S, E, N = bbox
    with rasterio.open(_vsicurl(cog)) as src:
        b = src.bounds
        # guard: bbox must intersect the grid (catches mis-georeferenced grids)
        if not (b.left < E and b.right > W and b.bottom < N and b.top > S):
            skipped[key] = f"bbox outside grid extent {tuple(round(x,1) for x in b)}"
            return
        win = from_bounds(W, S, E, N, src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win, boundless=True, fill_value=src.nodata)
        if data.size == 0 or 0 in data.shape:
            skipped[key] = "empty clip window"
            return
        # standardized clip: Float32, strip-organized, nodata=-9999, WGS84
        data = data.astype("float32")
        if src.nodata is not None:
            data[data == float(src.nodata)] = -9999.0
        profile = src.profile.copy()
        profile.update(driver="GTiff", height=data.shape[0], width=data.shape[1],
                       transform=src.window_transform(win),
                       tiled=False, blockysize=1, dtype="float32",
                       nodata=-9999.0, crs="EPSG:4326", interleave="pixel")
        profile.pop("compress", None)   # EF5 TifGrid-safe: plain 1-row strips
    path = os.path.join(out_dir, fname)
    tmp = path + ".tmp"                 # atomic: parallel runs share this store
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    os.replace(tmp, path)
    out[key] = path


if __name__ == "__main__":
    import sys, time
    out = sys.argv[1] if len(sys.argv) > 1 else "._param_test"
    bbox = (-69.83, 46.32, -68.33, 47.82)  # Allagash
    t = time.time()
    grids = clip_param_grids(bbox, out)
    print(f"clipped {len(grids)} param grids in {time.time()-t:.1f}s")
    for k, p in grids.items():
        with rasterio.open(p) as ds:
            a = ds.read(1)
            print(f"  {k:12s} {os.path.basename(p):10s} {ds.width}x{ds.height} "
                  f"range[{float(a.min()):.3f},{float(a.max()):.3f}]")

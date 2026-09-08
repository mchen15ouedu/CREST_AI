"""WorldPop population at the inundation grid's resolution (risk view).

Source: WorldPop Hub age-sex structures, Global 2015-2030 projection series
(R2025A, constrained) — the CURRENT/projected population only: the event's
year clamped to 2026-2030, never the historical 2000-2020 vintages.

    https://data.worldpop.org/GIS/AgeSex_structures/Global_2015_2030/
        R2025A/<year>/USA/v1/100m/constrained/
            usa_{f|m}_{00,01,05..80}_<year>_CN_100m_R2025A_v1.tif

Per event we cache the basin-bbox WINDOW of every age-sex band (aggregated
to display buckets), then probe per PIXEL: the value shown for a solver cell
is the WorldPop count of the source cell containing it scaled by the area
ratio (mass-conserving "resample down to the same level as the inundation
results" — a 100 m person count split evenly over the ~25 m solver cells
inside it). Two fetch strategies, decided once per process:

  1. /vsicurl windowed reads of the 100 m bands (fast; needs HTTP ranges —
     available from the HF Spaces, stripped by some campus proxies), else
  2. full download of the 1 km bands (~36 MB each, cached on disk) and
     local windowing (coarser source, same per-pixel semantics).

Only .npz windows are cached per event (CACHE_DIR/riskview/), a few hundred
KB each; nothing is committed to the store.
"""
from __future__ import annotations

import os
import threading

import numpy as np

RELEASE = os.environ.get("WORLDPOP_RELEASE", "R2025A")
YEAR_MIN, YEAR_MAX = 2026, 2030
# display buckets -> WorldPop age codes (00 = <1 yr, 01 = 1-4, then 5-yr)
BUCKETS = [
    ("0-4", ["00", "01"]), ("5-14", ["05", "10"]), ("15-24", ["15", "20"]),
    ("25-44", ["25", "30", "35", "40"]), ("45-64", ["45", "50", "55", "60"]),
    ("65-74", ["65", "70"]), ("75+", ["75", "80"]),
]
SEXES = ("m", "f")

try:
    from .statecache import CACHE_DIR
except Exception:                                    # standalone use
    CACHE_DIR = os.path.join(os.path.expanduser("~"), ".crest_cache")
RV_DIR = os.path.join(CACHE_DIR, "riskview")

_lock = threading.Lock()
_mode = {"res": None}        # "100m" (vsicurl) or "1km" (full download)
_building: dict = {}         # event_id -> "building" | error string


def _band_url(sex: str, age: str, year: int, res: str) -> str:
    root = ("https://data.worldpop.org/GIS/AgeSex_structures/"
            f"Global_2015_2030/{RELEASE}/{year}/USA/v1/")
    if res == "100m":
        return (f"{root}100m/constrained/"
                f"usa_{sex}_{age}_{year}_CN_100m_{RELEASE}_v1.tif")
    return (f"{root}1km_ua/constrained/"
            f"usa_{sex}_{age}_{year}_CN_1km_{RELEASE}_UA_v1.tif")


def year_for(event_id: str) -> int:
    try:
        return min(max(int(str(event_id)[:4]), YEAR_MIN), YEAR_MAX)
    except ValueError:
        return YEAR_MIN


def _npz_path(event_id: str) -> str:
    return os.path.join(RV_DIR, f"pop_{event_id}.npz")


def _gdal_env():
    import rasterio
    return rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                        GDAL_HTTP_MAX_RETRY=3, GDAL_HTTP_RETRY_DELAY=2)


def _read_window(url: str, bbox, res: str):
    """(array, transform) of the band over bbox. 100m goes through vsicurl;
    1km downloads the whole band once into RV_DIR and windows locally."""
    import rasterio
    from rasterio.windows import from_bounds
    w, s, e, n = bbox
    if res == "100m":
        src = "/vsicurl/" + url
    else:
        os.makedirs(RV_DIR, exist_ok=True)
        local = os.path.join(RV_DIR, os.path.basename(url))
        if not os.path.exists(local):
            import requests
            r = requests.get(url, timeout=600, stream=True,
                             headers={"User-Agent": "CREST-AI/1.0"})
            r.raise_for_status()
            tmp = local + ".part"
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
            os.replace(tmp, local)
        src = local
    with _gdal_env():
        with rasterio.open(src) as ds:
            win = from_bounds(w, s, e, n, ds.transform)
            a = ds.read(1, window=win, boundless=True, fill_value=0.0)
            a = a.astype(np.float32)
            if ds.nodata is not None:
                a[a == np.float32(ds.nodata)] = 0.0
            a[~np.isfinite(a)] = 0.0
            a[a < 0] = 0.0
            return a, ds.window_transform(win)


SLOW_100M_S = float(os.environ.get("WORLDPOP_SLOW_100M_S", "25"))


def _pick_mode(bbox, year: int) -> str:
    """Try one vsicurl 100 m window and TIME it; a slow first read means the
    band tifs are strip-organized and every window pulls huge ranges (seen
    live: 36 sequential reads > 15 min), so fall back to the 1 km bands
    (35 MB full downloads, parallel) which finish in ~1 min. WORLDPOP_RES
    env forces a path."""
    forced = os.environ.get("WORLDPOP_RES", "").strip()
    if forced in ("100m", "1km"):
        _mode["res"] = forced
    if _mode["res"]:
        return _mode["res"]
    import time as _t
    try:
        t0 = _t.time()
        _read_window(_band_url("f", "25", year, "100m"), bbox, "100m")
        _mode["res"] = "100m" if _t.time() - t0 < SLOW_100M_S else "1km"
    except Exception:
        _mode["res"] = "1km"
    return _mode["res"]


def build_window(event_id: str, bbox, log=print) -> str:
    """Fetch + aggregate all bands for the event bbox into the npz cache.
    Returns the npz path (raises on failure)."""
    path = _npz_path(event_id)
    if os.path.exists(path):
        return path
    year = year_for(event_id)
    res = _pick_mode(bbox, year)
    log(f"worldpop: building {event_id} window ({res}, year {year})")
    # all (sex, age) bands fetched in parallel — sequential reads were the
    # bottleneck (36 x network round-trips)
    from concurrent.futures import ThreadPoolExecutor
    pairs = [(sex, age) for sex in SEXES for _, ages in BUCKETS
             for age in ages]

    def fetch(p2):
        sex, age = p2
        return p2, _read_window(_band_url(sex, age, year, res), bbox, res)

    got = {}
    tr = None
    with ThreadPoolExecutor(max_workers=6) as ex:
        for p2, (a, tr) in ex.map(fetch, pairs):
            got[p2] = a
    out = {}
    for sex in SEXES:
        for label, ages in BUCKETS:
            acc = None
            for age in ages:
                a = got[(sex, age)]
                acc = a if acc is None else acc + a
            out[f"{sex}_{label}"] = acc
    os.makedirs(RV_DIR, exist_ok=True)
    tmp = path + ".part.npz"
    np.savez_compressed(tmp, transform=np.array(
        [tr.a, tr.b, tr.c, tr.d, tr.e, tr.f]), res=res, year=year, **out)
    os.replace(tmp, path)
    log(f"worldpop: {event_id} cached ({res})")
    return path


def ensure_window(event_id: str, bbox, log=print) -> str:
    """Non-blocking: start the window build in a thread; returns status
    'ready' | 'building' | 'error: ...'."""
    if os.path.exists(_npz_path(event_id)):
        return "ready"
    with _lock:
        st = _building.get(event_id)
        if st == "building":
            return "building"
        if isinstance(st, str) and st.startswith("error"):
            return st
        _building[event_id] = "building"

    def run():
        try:
            build_window(event_id, bbox, log=log)
            _building[event_id] = "ready"
        except Exception as e:
            _building[event_id] = f"error: {type(e).__name__}: {e}"
    threading.Thread(target=run, daemon=True).start()
    return "building"


RISK_REPO = os.environ.get("CREST_RISKDATA_REPO", "vincewin/CREST_riskdata")
_tiles: dict = {}          # (year, tile) -> {"idx": {(row,col): i}, "tbl": tbl}
_tiles_missing: set = set()


def _tile_of(lat: float, lon: float) -> str:
    import math
    lat0, lon0 = math.floor(lat), math.floor(lon)
    return (f"{'N' if lat0 >= 0 else 'S'}{abs(int(lat0)):02d}"
            f"{'W' if lon0 < 0 else 'E'}{abs(int(lon0)):03d}")


def _tile_probe(year: int, lat: float, lon: float,
                solver_cell_deg2: float):
    """Pre-hosted parquet tile lookup (CREST_riskdata) — the primary path:
    deterministic, no third-party services at probe time, and the demo
    Space only ever downloads the few-MB tile a probed basin needs."""
    tile = _tile_of(lat, lon)
    key = (year, tile)
    if key in _tiles_missing:
        return None
    if key not in _tiles:
        try:
            from huggingface_hub import hf_hub_download
            import pyarrow.parquet as pq
            f = hf_hub_download(RISK_REPO, f"worldpop/{year}/{tile}.parquet",
                                repo_type="dataset",
                                token=os.environ.get("HF_TOKEN"))
            t = pq.read_table(f)
            meta = t.schema.metadata or {}
            tr = eval(meta[b"transform"].decode())  # [a,b,c,d,e,f] literal
            rows = t.column("row").to_numpy()
            cols = t.column("col").to_numpy()
            idx = {(int(r), int(c)): i for i, (r, c)
                   in enumerate(zip(rows, cols))}
            cols_np = {name: t.column(name).to_numpy()
                       for name in t.column_names
                       if name not in ("row", "col")}
            if len(_tiles) > 8:            # keep the Space footprint tiny
                _tiles.clear()
            _tiles[key] = {"idx": idx, "cols": cols_np, "tr": tr}
        except Exception:
            _tiles_missing.add(key)
            return None
    tl = _tiles[key]
    a, b, c, d, e, f = tl["tr"]
    col = int((lon - c) / a)
    row = int((lat - f) / e)
    i = tl["idx"].get((row, col))
    scale = float(solver_cell_deg2 / abs(a * e))
    male, female = [], []
    for label, _ in BUCKETS:
        if i is None:                      # unpopulated cell: real zero
            male.append(0.0); female.append(0.0)
        else:
            male.append(round(float(tl["cols"][f"m_{label}"][i]) * scale, 3))
            female.append(round(float(tl["cols"][f"f_{label}"][i]) * scale, 3))
    src_total = 0.0 if i is None else round(float(sum(
        tl["cols"][k][i] for k in tl["cols"])), 2)
    return {"buckets": [b0 for b0, _ in BUCKETS], "male": male,
            "female": female,
            "cell_total": round(sum(male) + sum(female), 3),
            "source_total": src_total,
            "source_res": "100m", "year": year, "source": "tiles"}


def probe(event_id: str, lat: float, lon: float, solver_cell_deg2: float):
    """Per-pixel population of the solver cell at (lat, lon): the WorldPop
    cell's counts x (solver cell area / WorldPop cell area). Returns
    {"buckets": [...], "male": [...], "female": [...], "cell_total": x,
     "source_res": "100m"|"1km", "year": y} or None if not cached yet."""
    hit = _tile_probe(year_for(event_id), lat, lon, solver_cell_deg2)
    if hit is not None:
        return hit
    path = _npz_path(event_id)          # fallback: live-fetched window
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    a, b, c, d, e, f = z["transform"]
    col = int((lon - c) / a)
    row = int((lat - f) / e)
    ref = z[f"m_{BUCKETS[0][0]}"]
    if not (0 <= row < ref.shape[0] and 0 <= col < ref.shape[1]):
        return None
    scale = float(solver_cell_deg2 / abs(a * e))
    male, female = [], []
    for label, _ in BUCKETS:
        male.append(round(float(z[f"m_{label}"][row, col]) * scale, 3))
        female.append(round(float(z[f"f_{label}"][row, col]) * scale, 3))
    return {"buckets": [b0 for b0, _ in BUCKETS], "male": male,
            "female": female,
            "cell_total": round(sum(male) + sum(female), 3),
            "source_total": round(float(sum(
                float(z[f"{s}_{l}"][row, col]) for s in SEXES
                for l, _ in BUCKETS)), 2),
            "source_res": str(z["res"]), "year": int(z["year"])}

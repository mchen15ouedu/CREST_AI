"""Durable cache persistence — the important _cache stores survive Space restarts.

The Space container's disk is wiped on every restart/rebuild, which throws away
exactly the data that makes repeat runs fast. This module mirrors those stores
to a PRIVATE HF dataset (CREST_STATE_REPO, default vincewin/CREST_state):

  states/   EF5 state grids (PRIORITY — skip the 90-day warm-up on repeats);
            uploaded as .pqf (zstd parquet + grid metadata, same format as the
            forcing store) — much smaller than the raw uncompressed GeoTIFFs,
            converted back to EF5-safe .tif on restore
  results/  per-(gauge,model) hydrograph row records (overlap-reuse planning)
  params/   per-basin best parameter sets (calibration winners)
  users/    account profiles (favorites, simulation history)
  reports_saved/  registered users' report libraries (PDFs; anonymous users'
            reports stay in the ephemeral reports/ cache and are NOT synced)
  obs/      per-gauge USGS observed-discharge parquet stores (obs.get_series)
            — repeat runs and calibration read these instead of hitting NWIS

Deliberately NOT persisted: frames/ (re-renderable from states in seconds),
forcing/ (re-derivable from the public CREST_data dataset), errors/ + local
feedback (logs — cleanable; feedback already has its own durable sink).

Lifecycle: start() launches one daemon thread — create-repo(private) →
restore() into CACHE_DIR → incremental backup() every CREST_PERSIST_EVERY_S
(default 600 s), woken early by poke() when a run/calibration/profile write
lands. Deletions (janitor LRU evictions) are mirrored, so the repo tracks the
local caps instead of growing forever. Needs the HF_TOKEN Space secret (same
one feedback.py uses); without it everything is a silent no-op (local dev).
"""
from __future__ import annotations

import glob
import io
import os
import shutil
import threading
import time

CACHE_DIR = None  # set on first use (import-order safety)


def _cache_dir() -> str:
    global CACHE_DIR
    if CACHE_DIR is None:
        from hf_data.statecache import CACHE_DIR as CD
        CACHE_DIR = CD
    return CACHE_DIR


REPO = os.environ.get("CREST_STATE_REPO", "vincewin/CREST_state")
EVERY_S = float(os.environ.get("CREST_PERSIST_EVERY_S", "600"))
SYNC_DIRS = ("states", "results", "params", "users", "reports_saved", "obs")

_wake = threading.Event()
_started = False
_lock = threading.Lock()          # one backup at a time
_manifest: dict[str, tuple] = {}  # relpath -> (size, int mtime) at last sync
_last: dict = {}                  # last restore/backup reports (for /api/persist)


def _token():
    return os.environ.get("HF_TOKEN")


def enabled() -> bool:
    return bool(_token())


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=_token())


# ---- state grid <-> pqf (zstd parquet raster, forcing-store format) ---------
def _tif_pqf_bytes(tif_path: str) -> bytes | None:
    """GeoTIFF -> pqf bytes; None if the grid isn't a plain float raster."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import rasterio
    with rasterio.open(tif_path) as ds:
        if ds.count != 1 or ds.dtypes[0] not in ("float32", "float64"):
            return None                       # unexpected layout -> upload raw
        a = ds.read(1).astype("float32")
        cell = ds.res[0]
        xll, yll = ds.transform.c, ds.transform.f - ds.height * cell
        nodata = -9999.0 if ds.nodata is None else float(ds.nodata)
    nr, nc = a.shape
    meta = {b"ncols": str(nc).encode(), b"nrows": str(nr).encode(),
            b"xllcorner": repr(float(xll)).encode(), b"yllcorner": repr(float(yll)).encode(),
            b"cellsize": repr(float(cell)).encode(), b"nodata": repr(nodata).encode()}
    schema = pa.schema([pa.field("v", pa.float32())]).with_metadata(meta)
    buf = io.BytesIO()
    # BYTE_STREAM_SPLIT: float-specific encoding — ~2-3x better than plain
    # zstd on smooth state fields, still bit-exact and readable by _read_pqf
    pq.write_table(pa.table({"v": a.reshape(-1)}, schema=schema),
                   buf, compression="zstd", use_dictionary=False,
                   column_encoding={"v": "BYTE_STREAM_SPLIT"})
    return buf.getvalue()


def _pqf_to_tif(pqf_path: str, tif_path: str):
    """pqf -> EF5-safe GeoTIFF (untiled, uncompressed, stripped — the EF5
    reader is picky, same profile as the snow/temp-DEM writer)."""
    import rasterio
    from rasterio.transform import from_origin
    from hf_data.forcing import _read_pqf
    with open(pqf_path, "rb") as fh:
        a, xll, yll, cell, nodata = _read_pqf(fh.read())
    nr, nc = a.shape
    os.makedirs(os.path.dirname(tif_path), exist_ok=True)
    tmp = tif_path + ".tmp"
    with rasterio.open(tmp, "w", driver="GTiff", height=nr, width=nc, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(xll, yll + nr * cell, cell, cell),
                       nodata=nodata, tiled=False, blockysize=1) as ds:
        ds.write(a.astype("float32"), 1)
    os.replace(tmp, tif_path)


# ---- scan / naming ----------------------------------------------------------
def _scan() -> dict[str, str]:
    """Everything we persist right now: relpath (posix) -> abspath."""
    out = {}
    for d in SYNC_DIRS:
        root = os.path.join(_cache_dir(), d)
        if not os.path.isdir(root):
            continue
        for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
            if os.path.isfile(p):
                out[os.path.relpath(p, _cache_dir()).replace(os.sep, "/")] = p
    return out


def _repo_name(rel: str, as_pqf: bool) -> str:
    if as_pqf and rel.startswith("states/") and rel.endswith(".tif"):
        return rel[:-4] + ".pqf"
    return rel


# ---- restore (boot) ---------------------------------------------------------
def restore() -> dict:
    """Pull the repo snapshot into CACHE_DIR (existing local files win)."""
    if not enabled():
        return {"enabled": False}
    from huggingface_hub import snapshot_download
    tmp = os.path.join(_cache_dir(), "_restore_tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    snapshot_download(repo_id=REPO, repo_type="dataset", token=_token(),
                      local_dir=tmp, allow_patterns=[d + "/**" for d in SYNC_DIRS])
    n = skipped = failed = 0
    for src in glob.glob(os.path.join(tmp, "**", "*"), recursive=True):
        if not os.path.isfile(src):
            continue
        rel = os.path.relpath(src, tmp).replace(os.sep, "/")
        is_pqf = rel.startswith("states/") and rel.endswith(".pqf")
        dest_rel = rel[:-4] + ".tif" if is_pqf else rel
        dest = os.path.join(_cache_dir(), *dest_rel.split("/"))
        if os.path.exists(dest):                 # local copy is newer truth
            skipped += 1
            continue
        try:
            if is_pqf:
                _pqf_to_tif(src, dest)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
            st = os.stat(dest)
            _manifest[dest_rel] = (st.st_size, int(st.st_mtime))
            n += 1
        except Exception:
            failed += 1
    shutil.rmtree(tmp, ignore_errors=True)
    rep = {"restored": n, "kept_local": skipped, "failed": failed}
    _last["restore"] = rep
    return rep


# ---- backup (periodic + poked) ----------------------------------------------
def backup() -> dict:
    """Incremental push: changed/new files up, locally-evicted files deleted."""
    if not enabled():
        return {"enabled": False}
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    with _lock:
        local = _scan()
        ops, added = [], []
        raw_states = set()                      # states kept as raw .tif (odd dtype)
        for rel, p in sorted(local.items()):
            try:
                st = os.stat(p)
            except OSError:
                continue                         # vanished mid-scan
            sig = (st.st_size, int(st.st_mtime))
            if _manifest.get(rel) == sig:
                continue                         # unchanged since last sync
            try:
                if rel.startswith("states/") and rel.endswith(".tif"):
                    data = _tif_pqf_bytes(p)
                    if data is None:             # non-float grid -> raw copy
                        raw_states.add(rel)
                        with open(p, "rb") as fh:
                            data = fh.read()
                else:
                    with open(p, "rb") as fh:
                        data = fh.read()
            except Exception:
                continue                         # mid-write (EF5 running) -> next pass
            ops.append(CommitOperationAdd(
                path_in_repo=_repo_name(rel, as_pqf=rel not in raw_states),
                path_or_fileobj=data))
            added.append((rel, sig))
        # deletions: repo truth vs local truth (mirrors the janitor's LRU caps)
        api = _api()
        try:
            in_repo = set(api.list_repo_files(REPO, repo_type="dataset"))
        except Exception:
            in_repo = set()
        # a state may exist remotely under either name (.pqf or raw .tif) —
        # protect both spellings of every local file from deletion
        expected = set(local)
        expected |= {r[:-4] + ".pqf" for r in local
                     if r.startswith("states/") and r.endswith(".tif")}
        gone = [f for f in in_repo
                if f.split("/")[0] in SYNC_DIRS and f not in expected]
        ops += [CommitOperationDelete(path_in_repo=f) for f in gone]
        if not ops:
            rep = {"changed": 0, "deleted": 0}
            _last["backup"] = rep
            return rep
        api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                          commit_message=f"cache sync: +{len(added)} -{len(gone)}")
        for rel, sig in added:
            _manifest[rel] = sig
        rep = {"changed": len(added), "deleted": len(gone)}
        _last["backup"] = rep
        return rep


def status() -> dict:
    return {"enabled": enabled(), "repo": REPO, "every_s": EVERY_S,
            "tracked": len(_manifest), **_last}


def poke():
    """A run/calibration/profile write just landed — sync soon."""
    _wake.set()


# ---- lifecycle ----------------------------------------------------------------
def _loop():
    try:
        _api().create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
        restore()
    except Exception as e:
        try:
            from hf_data import crashlog
            crashlog.capture("persist:restore", e)
        except Exception:
            pass
    while True:
        _wake.wait(timeout=EVERY_S)
        _wake.clear()
        time.sleep(3)          # let EF5/json writers finish the current files
        try:
            backup()
        except Exception as e:
            try:
                from hf_data import crashlog
                crashlog.capture("persist:backup", e)
            except Exception:
                pass


def start():
    """Idempotent; no-op without HF_TOKEN (local dev)."""
    global _started
    if _started or not enabled():
        return
    _started = True
    threading.Thread(target=_loop, daemon=True).start()

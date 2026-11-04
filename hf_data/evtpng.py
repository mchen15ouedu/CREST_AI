"""Max-pooled 2-D inundation overlays for basin zoom (/api/evt_png).

Why: selecting an event fits the whole basin domain on screen (~zoom 9), so a
1-arc-second depth grid is drawn far below native resolution — event
2026102122_09401260 (Moenkopi Wash, 3622 x 3243 cells) is shown ~366 px wide.
Leaflet samples the CDN png nearest-neighbour (image-rendering: pixelated),
which keeps ~1% of its pixels, and a flood that lives in 1-3-cell-wide
channels reads as an EMPTY basin (smooth sampling only fades it further).
Here the depth tif is MAX-pooled over f x f blocks before colouring, so every
wet channel survives as a whole pooled pixel at the depth of its deepest
cell. The frontend asks for this only while the overlay would be shown below
native resolution (frontend/app.js evtFrameSrc); at native zoom and beyond it
uses the CDN png unchanged.

Colours are the overlay PNGs' own (hf_data/eventsim.py _make_pngs render():
the DEPTH_STOPS ramp over 0..manifest depth_cap_m, alpha 90 + 165*sqrt(x),
cells < MIN_SHOW_M transparent), so a pooled map and a native one are the
same picture at two scales. render() is a closure there, so the math is
repeated in colorize() below — keep the two in step. A block takes the
colour of its MAX depth, but max alone inflates the wet EXTENT (at f=8, 37%
of 09401260's pooled pixels were coloured from 4% of cells >= 2 cm): a block
whose max stays in the shallow green band (< 0.30 x cap, the ramp's first
segment) fades by sqrt(its wet-cell fraction), no floor; blocks reaching
>= 0.30 x cap (the channels) keep the full native alpha. At f = 1 every
coloured cell is an all-wet block of one, so the native png comes back.

A pooled png is ceil(nx/f) x ceil(ny/f) pixels: the east/south edge is padded
dry to whole blocks, and the frontend widens the overlay bounds to match.

Server: server.py mounts `router` (app.include_router(evtpng.router)). The
endpoint is async and renders on this module's OWN thread limiter behind a
bounded queue (503 + Retry-After past it): the sync version waited for a
render slot while holding a token of AnyIO's shared threadpool, so scrubbing
the frames at basin zoom stalled every sync route (pings 0.04 s -> 10.75 s
with 60 renders queued). Importing this module has no side effects — no
network, no directories; the on-disk cache is created on first use.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time

import anyio
import anyio.to_thread
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from hf_data import eventstore
from hf_data.eventsim import (DEPTH_CAP_M, DEPTH_STOPS_B, DEPTH_STOPS_G,
                              DEPTH_STOPS_R, DEPTH_STOPS_X, MIN_SHOW_M)

FACTORS = (2, 4, 8, 16)
# strict: these strings become a dataset path and a cache file name
_ID_RE = re.compile(r"[0-9]{10}_[A-Za-z0-9]+")
_NAME_RE = re.compile(r"maxdepth|depth_[0-9]{12}")
# pooled pngs are small (a 3622 x 3243 maxdepth at f=8 is ~453 x 406 px);
# the cache is bounded LRU (oldest-used files go first past the cap)
CACHE_DIR = os.path.join(os.environ.get("CREST_CACHE_DIR") or tempfile.gettempdir(),
                         "evt_png")
CACHE_MAX_MB = float(os.environ.get("EVT_PNG_CACHE_MB", "200"))
# an active episode re-publishes hourly (new `generated`, possibly a new
# depth cap that recolours every frame); this bounds how stale a manifest
# read may be when the client doesn't say which generation it holds
MAN_TTL_S = float(os.environ.get("EVT_PNG_MANIFEST_TTL_S", "60"))
# decoded grids are ~23 MB each (uint16); cap concurrent renders so frame
# prefetch bursts can't stack them up on a small CPU Space
WORKERS = max(1, int(os.environ.get("EVT_PNG_WORKERS", "2")))
# distinct cold renders that may WAIT for a slot; past that a request is
# refused at once (503 + Retry-After), so a scrub through the frames can't
# pile up renders nobody is looking at any more by the time they land
QUEUE_MAX = max(0, int(os.environ.get("EVT_PNG_QUEUE", "8")))
RETRY_AFTER_S = 2
# manifest reads + cache lookups get a small pool of their own as well: on
# a cold manifest they park on the per-event lock while one download runs
IO_WORKERS = 4
# bump when the colouring changes: it keys the disk cache, which may live on
# a persistent CREST_CACHE_DIR (2: shallow blocks fade by wet fraction)
STYLE = 2
# the frontend puts the manifest's `generated` in the URL (&g=): when it
# matches what we rendered from, that URL's bytes never change
CC_IMMUTABLE = "public, max-age=31536000, immutable"
CC_SHORT = "public, max-age=300"
# the native png's wet test (tif cm / 100.0 >= MIN_SHOW_M) in whole cm
_WET_CM = next(c for c in range(65536) if c / 100.0 >= MIN_SHOW_M)

_locks: dict = {}                    # key -> Lock: one manifest read per event at a time
_locks_guard = threading.Lock()
_prune_lock = threading.Lock()
_man_memo: dict = {}                 # event_id -> (fetched_at, slim manifest)
_limiters: dict = {}                 # "render"/"io" -> anyio.CapacityLimiter (made in the loop)
_inflight: dict = {}                 # cache path -> _Job: one render per tile (event loop only)

router = APIRouter()


class Busy(RuntimeError):
    """Render queue full: the endpoint answers 503 + Retry-After."""


class _Job:
    """A cold render in flight: identical requests await it, not a slot."""
    __slots__ = ("done", "data", "exc")

    def __init__(self):
        self.done = anyio.Event()
        self.data = None
        self.exc = None


def valid_event_id(event_id) -> bool:
    return isinstance(event_id, str) and bool(_ID_RE.fullmatch(event_id))


def valid_name(name) -> bool:
    """'maxdepth' or a frame stem 'depth_YYYYMMDDHHMM' (no extension)."""
    return isinstance(name, str) and bool(_NAME_RE.fullmatch(name))


def valid_factor(f) -> bool:
    return type(f) is int and f in FACTORS


def _blocks(depth, f):
    """2-D grid -> (ny/f, f, nx/f, f) blocks, the east/south edge padded
    with 0 (dry) to whole blocks."""
    a = np.asarray(depth)
    if a.ndim != 2 or int(f) < 1:
        raise ValueError("pooling needs a 2-D grid and f >= 1")
    f = int(f)
    ny, nx = a.shape
    py, px = (-ny) % f, (-nx) % f
    if py or px:
        a = np.pad(a, ((0, py), (0, px)), mode="constant", constant_values=0)
    return a.reshape(a.shape[0] // f, f, a.shape[1] // f, f)


def pool_max(depth: np.ndarray, f: int) -> np.ndarray:
    """f x f block maximum of a 2-D grid. The east/south edge is padded with
    0 (dry) to whole blocks, so the result is ceil(ny/f) x ceil(nx/f)."""
    return _blocks(depth, f).max(axis=(1, 3))


def pool_wet_frac(depth_cm: np.ndarray, f: int) -> np.ndarray:
    """Share of each f x f block's cells the native png colours (cm grid,
    >= MIN_SHOW_M). The padding past the east/south edge isn't grid, so an
    edge block divides by its real cells only."""
    a = np.asarray(depth_cm)
    wet = _blocks(a >= _WET_CM, f).sum(axis=(1, 3))
    f = int(f)
    ny, nx = a.shape
    ry = np.minimum(f, ny - f * np.arange(wet.shape[0]))    # real rows per block row
    rx = np.minimum(f, nx - f * np.arange(wet.shape[1]))
    return wet / np.outer(ry, rx)


def colorize(depth_m: np.ndarray, cap: float, wet_frac=None) -> np.ndarray:
    """RGBA uint8 for a depth grid in metres — eventsim._make_pngs render()
    exactly: dry (< MIN_SHOW_M) transparent, the DEPTH_STOPS ramp over 0..cap,
    alpha rising with sqrt(depth/cap). wet_frac (pooled grids: each block's
    wet-cell share) fades the blocks whose depth stays in the ramp's first
    (green) segment by sqrt(wet_frac), no floor; 1 everywhere = native."""
    depth = np.asarray(depth_m, dtype=float)
    x = np.clip(depth / cap, 0.0, 1.0)
    alpha = np.where(depth >= MIN_SHOW_M, 90 + 165 * np.sqrt(x), 0.0)
    if wet_frac is not None:
        alpha = np.where(x < DEPTH_STOPS_X[1], alpha * np.sqrt(wet_frac), alpha)
    alpha = alpha.astype(np.uint8)
    r = np.interp(x, DEPTH_STOPS_X, DEPTH_STOPS_R).astype(np.uint8)
    g = np.interp(x, DEPTH_STOPS_X, DEPTH_STOPS_G).astype(np.uint8)
    b = np.interp(x, DEPTH_STOPS_X, DEPTH_STOPS_B).astype(np.uint8)
    return np.dstack([r, g, b, alpha])


def pooled_rgba(depth_cm: np.ndarray, f: int, cap: float, nodata=None) -> np.ndarray:
    """The overlay of a uint16-cm depth grid (0 = dry) max-pooled by f.
    Pooling runs on the integer centimetres (max commutes with the /100), so
    a pooled pixel has exactly the colour of its deepest native cell; a
    shallow block's alpha also carries how much of it is wet (colorize)."""
    a = np.asarray(depth_cm)
    if nodata is not None and nodata != 0:   # the store writes none; 0 is dry anyway
        bad = np.isnan(a) if (isinstance(nodata, float) and np.isnan(nodata)) else (a == nodata)
        if bad.any():
            a = np.where(bad, 0, a)          # nodata is outside the domain: dry
    return colorize(pool_max(a, f).astype(float) / 100.0, cap, pool_wet_frac(a, f))


def png_bytes(rgba: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8)).save(
        buf, format="PNG", optimize=True)
    return buf.getvalue()


def _cap_of(man: dict) -> float:
    try:
        cap = float(man.get("depth_cap_m"))
        return cap if np.isfinite(cap) and cap > 0 else DEPTH_CAP_M
    except (TypeError, ValueError):
        return DEPTH_CAP_M               # same fallback the legend shows (3 m)


def _key_lock(key: str) -> threading.Lock:
    with _locks_guard:
        if len(_locks) > 1024:           # bounded: drop the idle ones
            for k in [k for k, lk in _locks.items() if not lk.locked()]:
                del _locks[k]
        lk = _locks.get(key)
        if lk is None:
            lk = _locks[key] = threading.Lock()
        return lk


def _download(rel: str, dest: str) -> str:
    """events/<rel> from the public dataset into dest (a private temp dir the
    caller removes — nothing piles up in the shared HF cache). A file or repo
    that isn't there -> LookupError (404); network trouble propagates (502)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import (EntryNotFoundError,
                                       LocalEntryNotFoundError,
                                       RepositoryNotFoundError)
    try:
        return hf_hub_download(eventstore.REPO, f"{eventstore.PREFIX}/{rel}",
                               repo_type="dataset", local_dir=dest,
                               token=os.environ.get("HF_TOKEN"))
    except LocalEntryNotFoundError:
        raise                            # offline, not missing
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        raise LookupError(f"{rel} not in the event store") from e


def _manifest(event_id: str, want_gen: str | None = None) -> dict:
    """Slim manifest (generated, cap, frame tifs), memoized MAN_TTL_S. A
    client holding a different generation forces a re-read (rate-limited)."""
    now = time.time()
    hit = _man_memo.get(event_id)
    if hit and now - hit[0] < MAN_TTL_S and (
            not want_gen or want_gen == hit[1]["generated"] or now - hit[0] < 5):
        return hit[1]
    with _key_lock("man:" + event_id):
        hit = _man_memo.get(event_id)
        if hit and hit[0] > now:         # refreshed while we waited
            return hit[1]
        tmp = tempfile.mkdtemp(prefix="evtman_")
        try:
            with open(_download(f"{event_id}/manifest.json", tmp), encoding="utf-8") as fp:
                m = json.load(fp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        slim = {"generated": str(m.get("generated") or ""),
                "cap": _cap_of(m),
                "maxdepth": m.get("maxdepth") or "maxdepth.tif",
                "frames": frozenset(fr.get("file") for fr in (m.get("frames") or [])
                                    if isinstance(fr, dict))}
        if len(_man_memo) > 256:
            _man_memo.clear()
        _man_memo[event_id] = (time.time(), slim)
        return slim


def _cache_get(path: str) -> bytes | None:
    try:
        with open(path, "rb") as fp:
            data = fp.read()
        os.utime(path, None)             # LRU by mtime: a hit is a use
        return data
    except OSError:
        return None


def _cache_put(path: str, data: bytes):
    """Atomic write + LRU prune. The cache is an optimisation: a full or
    read-only disk must never fail the tile itself."""
    tmp = None
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".part")
        with os.fdopen(fd, "wb") as fp:
            fp.write(data)
        os.replace(tmp, path)
        tmp = None
        _prune()
    except OSError:
        if tmp:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _prune():
    cap = CACHE_MAX_MB * 1e6
    now = time.time()
    with _prune_lock:
        ents = []
        for e in os.scandir(CACHE_DIR):
            try:
                st = e.stat()
            except OSError:
                continue
            if e.name.endswith(".png"):
                ents.append((st.st_mtime, st.st_size, e.path))
            elif e.name.endswith(".part") and now - st.st_mtime > 3600:
                try:                         # left by a crashed write
                    os.remove(e.path)
                except OSError:
                    pass
        total = sum(s for _, s, _ in ents)
        for _, size, p in sorted(ents):
            if total <= cap:
                break
            try:
                os.remove(p)
                total -= size
            except OSError:
                pass


def _compute(event_id: str, tif: str, f: int, cap: float) -> bytes:
    import rasterio
    tmp = tempfile.mkdtemp(prefix="evtpng_")
    try:
        with rasterio.open(_download(f"{event_id}/{tif}", tmp)) as ds:
            depth = ds.read(1)
            nodata = ds.nodata
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return png_bytes(pooled_rgba(depth, f, cap, nodata))


def _lookup(event_id: str, name: str, f: int, want_gen: str | None = None):
    """(tif, cap, cache path, current, cached png or None) — blocking: a
    manifest read on a memo miss, one small file read. `current` = keyed on
    exactly the manifest generation the client named, so the response may
    be cached for good."""
    man = _manifest(event_id, want_gen)
    tif = man["maxdepth"] if name == "maxdepth" else f"{name}.tif"
    if name != "maxdepth" and tif not in man["frames"]:
        raise LookupError(f"{event_id} has no frame {name}")
    # keyed on the generation AND the cap: a re-publish recolours every frame
    gen = man["generated"] or f"h{int(time.time() // 3600)}"
    tag = hashlib.sha1(f"{gen}|{man['cap']}|{STYLE}".encode()).hexdigest()[:12]
    path = os.path.join(CACHE_DIR, f"{event_id}__{name}__f{f}__{tag}.png")
    current = bool(want_gen) and want_gen == man["generated"]
    return tif, man["cap"], path, current, _cache_get(path)


def _render_put(event_id: str, tif: str, f: int, cap: float, path: str) -> bytes:
    """The cold path, on a render slot. Re-checks the cache first: the same
    tile may have landed between this request's miss and its slot."""
    data = _cache_get(path)
    if data is None:
        data = _compute(event_id, tif, f, cap)
        _cache_put(path, data)
    return data


def _limiter(kind: str) -> anyio.CapacityLimiter:
    # made on first use, inside the running loop (older AnyIO can't make one
    # outside it); never the shared default limiter sync routes draw from
    lim = _limiters.get(kind)
    if lim is None:
        lim = _limiters[kind] = anyio.CapacityLimiter(WORKERS if kind == "render" else IO_WORKERS)
    return lim


async def render(event_id: str, name: str, f: int, want_gen: str | None = None):
    """(png bytes, current). A cache hit never waits for a render slot; a
    cold tile joins an identical render already in flight, else queues for
    one of WORKERS slots — or raises Busy when QUEUE_MAX already wait."""
    tif, cap, path, current, data = await anyio.to_thread.run_sync(
        _lookup, event_id, name, f, want_gen, limiter=_limiter("io"))
    if data is not None:
        return data, current
    job = _inflight.get(path)
    if job is not None:                  # the same tile is rendering: share it
        await job.done.wait()
        if job.exc is not None:
            raise job.exc
        if job.data is None:             # its request was cancelled while queued
            raise Busy(path)
        return job.data, current
    if len(_inflight) >= WORKERS + QUEUE_MAX:
        raise Busy(path)
    job = _inflight[path] = _Job()
    try:
        job.data = await anyio.to_thread.run_sync(
            _render_put, event_id, tif, f, cap, path, limiter=_limiter("render"))
    except Exception as e:
        job.exc = e
        raise
    finally:
        del _inflight[path]
        job.done.set()
    return job.data, current


@router.get("/api/evt_png/{event_id}/{name}.png")
async def api_evt_png(event_id: str, name: str, f: int = 0, g: str = ""):
    """Max-pooled overlay png of one event's maxdepth or depth frame, f x f
    native cells per pixel (f in 2/4/8/16), coloured like the CDN overlay."""
    if not valid_event_id(event_id) or not valid_name(name):
        raise HTTPException(400, "bad event id or overlay name")
    if not valid_factor(f):
        raise HTTPException(400, f"f must be one of {', '.join(map(str, FACTORS))}")
    try:
        data, current = await render(event_id, name, f, g[:64] or None)
    except Busy:
        raise HTTPException(503, "overlay renderer busy",
                            headers={"Retry-After": str(RETRY_AFTER_S)})
    except LookupError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(502, f"overlay unavailable ({type(e).__name__})")
    return Response(content=data, media_type="image/png",
                    headers={"Cache-Control": CC_IMMUTABLE if current else CC_SHORT})

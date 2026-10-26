"""Run-private EF5 state folder (M4, PI OK 2026-10-23).

EF5 loads a saved state grid of a different size BY LOCATION, silently
(Grid.h IsSpatialMatch compares rows/cols only; CRESTPhysModel/KinematicRoute
then read cell by coordinate). A state saved on another raster - an old
outlet-square run that clipped the headwaters, another speed domain - would
leave part of the basin cold without a word.

So EF5 never reads the shared per-gauge state folder directly. Before a run,
stage() hard-links (or copies) into <workdir>/states only the grids of the
time EF5 will load, and only when they sit on the run's lattice and hold
valid values on enough of the run's EF5 nodes: MIN_COVER as a rule, and
PARTIAL_COVER only for a conusstate mosaic (is_mosaic) followed by a warm-up
of at least PARTIAL_GAP_DAYS that spins the rest up (its partial tier).
After a successful run, commit() moves the states EF5
wrote back to the shared folder - unless the run used custom parameters
(calibration candidates, panel overrides): their trial-parameter warm-up
states must never reach the shared cache. run_gauge removes the private
folder when the run ends (staged links would otherwise pin pruned states).
"""
from __future__ import annotations

import glob
import os
import shutil
import time

import numpy as np

MIN_COVER = float(os.environ.get("CREST_MOSAIC_MIN_COVER", "0.90"))
PARTIAL_COVER = float(os.environ.get("CREST_STATE_PARTIAL_COVER", "0.50"))
PARTIAL_GAP_DAYS = float(os.environ.get("CREST_STATE_PARTIAL_GAP_DAYS", "30"))


def is_mosaic(shared_dir: str, t) -> bool:
    """True when the state set at t was stitched by conusstate.warm_state
    (recorded in its .mosaic.json). Only those may cover part of the basin
    (PARTIAL_COVER): any other saved state - e.g. one from an old outlet-
    square run that clipped the headwaters - must cover nearly all of it."""
    import json
    stamp = t.strftime("%Y%m%d_%H%M")
    try:
        with open(os.path.join(shared_dir, ".mosaic.json")) as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return False
    return any(isinstance(v, dict) and v.get("at") == stamp
               for k, v in m.items() if not k.startswith("_"))


def _files_at(shared_dir: str, t) -> list[str]:
    stamp = t.strftime("%Y%m%d_%H%M")
    return sorted(glob.glob(os.path.join(shared_dir, f"*_{stamp}.tif"))
                  + glob.glob(os.path.join(shared_dir, f"gauge_relationships_{stamp}.txt")))


def coverage(state_path: str, nodes: np.ndarray, tr) -> float:
    """Share of the run's EF5 nodes (bool mask on the run grid with affine tr)
    that the state grid holds a valid value for; 0.0 if off-lattice."""
    import rasterio
    with rasterio.open(state_path) as ds:
        st = ds.transform
        if abs(abs(st.a) - abs(tr.a)) > 1e-9 or abs(abs(st.e) - abs(tr.e)) > 1e-9:
            return 0.0
        dc = (tr.c - st.c) / st.a
        dr = (tr.f - st.f) / st.e
        if abs(dc - round(dc)) > 1e-3 or abs(dr - round(dr)) > 1e-3:
            return 0.0
        a = ds.read(1)
        nod = ds.nodata
    rr, cc = np.nonzero(nodes)
    if not len(rr):
        return 0.0
    rs, cs = rr + int(round(dr)), cc + int(round(dc))
    inside = (rs >= 0) & (rs < a.shape[0]) & (cs >= 0) & (cs < a.shape[1])
    ok = np.zeros(len(rr), dtype=bool)
    v = a[rs[inside], cs[inside]].astype("float64")
    good = np.isfinite(v)
    if nod is not None:
        good &= v != float(nod)
    good &= v > -9000                 # EF5's -9999 written without a nodata tag
    ok[np.nonzero(inside)[0]] = good
    return float(ok.mean())


def _place(src: str, dst: str) -> None:
    try:
        os.link(src, dst)             # EF5 never rewrites a file it loads
    except OSError:
        shutil.copy2(src, dst)


def stage(shared_dir: str, run_dir: str, t, nodes: np.ndarray, tr,
          need: float = MIN_COVER) -> tuple[bool, float]:
    """Stage the state set at time t into run_dir if it covers >= need of the
    run's nodes. Returns (staged, coverage); any read/copy failure (a file
    evicted or pruned mid-way, a truncated grid) is (False, 0.0) with run_dir
    emptied, so the caller warms up instead."""
    os.makedirs(run_dir, exist_ok=True)
    files = _files_at(shared_dir, t)
    # coverage is judged on the water-balance/routing grids: snow17 grids are
    # filled only from pre-runs that had snow at that time, and a run with
    # snow off never reads them
    grids = [f for f in files if f.endswith(".tif")
             and not os.path.basename(f).startswith("snow17_")]
    if not grids:
        return False, 0.0
    try:
        cov = min(coverage(g, nodes, tr) for g in grids)
        if cov < need:
            return False, cov
        for f in files:
            dst = os.path.join(run_dir, os.path.basename(f))
            if not os.path.exists(dst):
                _place(f, dst)
        return True, cov
    except Exception:
        for f in os.listdir(run_dir):
            try:
                os.remove(os.path.join(run_dir, f))
            except OSError:
                pass
        return False, 0.0


def _replace(src: str, dst: str, tries: int = 5) -> bool:
    """Move src over dst atomically - never by writing into dst, which may be
    hard-linked into another run's private folder."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return True
        except OSError:
            pass
        tmp = f"{dst}.{os.getpid()}.tmp"
        try:                          # other filesystem: copy beside, then rename
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
            os.remove(src)
            return True
        except OSError:
            try:
                os.remove(tmp)
            except OSError:
                pass
        time.sleep(0.5 * (i + 1))    # Windows: a reader holds dst open
    return False


def commit(run_dir: str, shared_dir: str, clobber: bool = True) -> int:
    """Move the states EF5 wrote in run_dir to the shared folder (files staged
    from there are skipped). clobber=False keeps an existing shared file of
    the same name (nowcast/event runs must not replace the hindcast chain's
    states). Returns how many files moved."""
    n = 0
    if not os.path.isdir(run_dir):
        return 0
    os.makedirs(shared_dir, exist_ok=True)
    for f in sorted(os.listdir(run_dir)):
        src = os.path.join(run_dir, f)
        dst = os.path.join(shared_dir, f)
        if not os.path.isfile(src) or f.endswith(".tmp"):
            continue
        if os.path.exists(dst):
            try:
                if os.path.samefile(src, dst):
                    continue          # staged hard link of the shared file
            except OSError:
                pass
            if not clobber:
                continue
        if _replace(src, dst):
            n += 1
    return n


def nodes_of(grid_dir: str, carve_mask=None):
    """(bool mask, transform) of the run's EF5 nodes on grid_dir's raster: the
    carved catchment for a full-basin run, the kept cells of a speed domain
    (its facc_clip holds -9999 outside them)."""
    import rasterio
    if carve_mask is not None:
        with rasterio.open(os.path.join(grid_dir, "fdir_clip.tif")) as ds:
            return carve_mask, ds.transform
    with rasterio.open(os.path.join(grid_dir, "facc_clip.tif")) as ds:
        a = ds.read(1)
        tr = ds.transform
        nod = ds.nodata
    m = np.isfinite(a) & (a > -9000)
    if nod is not None:
        m &= a != nod
    return m, tr

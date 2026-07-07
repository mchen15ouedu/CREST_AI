"""Data manager: keeps the app's data organized, merged, and within bounds.

What it manages
  - FORCING STORE  CACHE_DIR/forcing/<var>/<bbox> — shared clipped timesteps.
    Overlapping requests merge here (forcing.prepare_forcing skips existing
    files); this module enforces an LRU size cap.
  - RESULTS        CACHE_DIR/results/*.json — per-(gauge, model) hydrograph
    rows, already row-merged on write; compact() re-sorts, de-dupes, and folds
    the state files found on disk back into each record's index.
  - STATES/FRAMES/PARAMS — size-capped LRU (params are never evicted).
  - RUN WORKDIRS   crest_* temp dirs from finished runs — TTL cleanup.
  - HF TAR CACHE   downloaded month-tars — evicted wholesale when over cap
    (they re-download on demand).

A janitor thread runs cleanup + compaction hourly (start() from the server).
stats() feeds GET /api/datastats.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import threading
import time

from hf_data.statecache import CACHE_DIR

FORCING_CAP_GB = float(os.environ.get("CREST_FORCING_CACHE_GB", "6"))
FRAMES_CAP_GB = float(os.environ.get("CREST_FRAMES_CACHE_GB", "1"))
STATES_CAP_GB = float(os.environ.get("CREST_STATES_CACHE_GB", "4"))
HF_CACHE_CAP_GB = float(os.environ.get("CREST_HF_CACHE_GB", "25"))
WORKDIR_TTL_H = float(os.environ.get("CREST_WORKDIR_TTL_H", "6"))
JANITOR_EVERY_S = float(os.environ.get("CREST_JANITOR_EVERY_S", str(3600)))


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _evict_lru_files(root: str, cap_bytes: int) -> int:
    """Delete oldest files (by mtime) under root until total <= cap. Returns bytes freed."""
    entries = []
    for r, _d, files in os.walk(root):
        for f in files:
            p = os.path.join(r, f)
            try:
                st = os.stat(p)
                entries.append((st.st_mtime, st.st_size, p))
            except OSError:
                pass
    total = sum(e[1] for e in entries)
    freed = 0
    for mt, size, p in sorted(entries):
        if total - freed <= cap_bytes:
            break
        try:
            os.remove(p)
            freed += size
        except OSError:
            pass
    return freed


def cleanup() -> dict:
    """One janitor pass. Returns a report of what was freed."""
    rep = {}
    # 1. stale run workdirs (finished runs never delete their temp dirs)
    cutoff = time.time() - WORKDIR_TTL_H * 3600
    n = freed = 0
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "crest_*")):
        try:
            if os.path.isdir(d) and os.path.getmtime(d) < cutoff:
                size = _dir_size(d)
                shutil.rmtree(d, ignore_errors=True)
                n += 1
                freed += size
        except OSError:
            pass
    rep["workdirs"] = {"removed": n, "mb": round(freed / 1e6, 1)}

    # 2. size-capped LRU stores
    for name, cap in (("forcing", FORCING_CAP_GB), ("frames", FRAMES_CAP_GB),
                      ("states", STATES_CAP_GB)):
        root = os.path.join(CACHE_DIR, name)
        if os.path.isdir(root):
            rep[name] = {"freed_mb": round(_evict_lru_files(root, int(cap * 1e9)) / 1e6, 1)}

    # 3. HF hub tar cache — coarse eviction when far over cap (re-downloads on demand)
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir()
        if info.size_on_disk > HF_CACHE_CAP_GB * 1e9:
            revs = [rev.commit_hash for repo in info.repos
                    if repo.repo_id == "vincewin/CREST_data"
                    for rev in repo.revisions]
            if revs:
                info.delete_revisions(*revs).execute()
                rep["hf_cache"] = {"evicted_revisions": len(revs)}
    except Exception:
        pass
    return rep


def compact_results() -> dict:
    """Merge/organize result records: de-dupe + sort rows, and fold the state
    files actually on disk back into each record's state_times index."""
    from hf_data import statecache
    root = os.path.join(CACHE_DIR, "results")
    n_rec = n_rows_dropped = 0
    for p in glob.glob(os.path.join(root, "*.json")):
        try:
            with open(p) as fh:
                rec = json.load(fh)
            gauge, model = rec.get("gauge"), rec.get("model")
            by_time = {}
            for r in rec.get("rows", []):
                by_time[r["time"]] = r                      # last write wins
            merged = sorted(by_time.values(), key=lambda r: r["time"])
            n_rows_dropped += len(rec.get("rows", [])) - len(merged)
            rec["rows"] = merged
            if merged:
                rec["window"] = [merged[0]["time"], merged[-1]["time"]]
            disk = {t.strftime(statecache.TS_FMT)
                    for t in statecache._state_times_on_disk(gauge, model)}
            rec["state_times"] = sorted(set(rec.get("state_times", [])) | disk)
            with open(p, "w") as fh:
                json.dump(rec, fh)
            n_rec += 1
        except Exception:
            pass
    return {"records": n_rec, "duplicate_rows_dropped": n_rows_dropped}


def stats() -> dict:
    """Disk usage per data category (feeds /api/datastats)."""
    out = {}
    for name in ("forcing", "results", "states", "frames", "params", "users", "errors"):
        p = os.path.join(CACHE_DIR, name)
        out[name] = {"mb": round(_dir_size(p) / 1e6, 1)} if os.path.isdir(p) else {"mb": 0}
        if name == "forcing" and os.path.isdir(p):
            out[name]["basins"] = sum(len(os.listdir(os.path.join(p, v)))
                                      for v in os.listdir(p)
                                      if os.path.isdir(os.path.join(p, v)))
        if name == "results" and os.path.isdir(p):
            out[name]["gauges"] = len(glob.glob(os.path.join(p, "*.json")))
    try:
        from huggingface_hub import scan_cache_dir
        out["hf_tar_cache"] = {"mb": round(scan_cache_dir().size_on_disk / 1e6, 1)}
    except Exception:
        out["hf_tar_cache"] = {"mb": None}
    n = w = 0
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "crest_*")):
        if os.path.isdir(d):
            n += 1
            w += _dir_size(d)
    out["run_workdirs"] = {"count": n, "mb": round(w / 1e6, 1)}
    out["caps_gb"] = {"forcing": FORCING_CAP_GB, "frames": FRAMES_CAP_GB,
                      "states": STATES_CAP_GB, "hf_cache": HF_CACHE_CAP_GB}
    return out


_janitor_started = False


def start_janitor():
    """Hourly cleanup + compaction in a daemon thread (idempotent)."""
    global _janitor_started
    if _janitor_started:
        return
    _janitor_started = True

    def _loop():
        while True:
            time.sleep(JANITOR_EVERY_S)
            try:
                rep = cleanup()
                rep["compact"] = compact_results()
                from hf_data import crashlog
                # not an error — but keep an operational trace in the same log
                if any(v for v in rep.values() if isinstance(v, dict) and
                       (v.get("removed") or v.get("freed_mb") or v.get("evicted_revisions"))):
                    crashlog.capture("janitor", message="cleanup pass", **rep)
            except Exception as e:
                try:
                    from hf_data import crashlog
                    crashlog.capture("janitor", e)
                except Exception:
                    pass

    threading.Thread(target=_loop, daemon=True).start()

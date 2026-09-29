"""Lazy fetch of fleet-precomputed simulations (READ-ONLY repo).

The virtual-user fleet (fleet/fleet_run.py on the fleet-runner Spaces)
precomputes years of quick-run results + 10-day states for thousands of
gauges into the private dataset CREST_FLEET_REPO:

  results/<gid>_<rows_model>.json    statecache record (rows + state_times + variant)
  states/<gid>_<state_model>.pqf     f16 state bundle (statebundle format)

rows_model = state_model (+ "-<tag>" for a non-a-priori parameter set, see
pipeline.config_tag); the two are the same name for the a-priori runs that
make up nearly all of the fleet.

When a run touches a (gauge, model) this MERGES both into the local cache,
after which the normal plan()/warm-start machinery serves the user instantly:
fleet rows join a local record of the same run configuration (local rows win
where both exist), fleet state grids fill every time the local state dir
lacks (local grids are never overwritten). It used to fetch only into an
EMPTY cache — one earlier run of a gauge, or a record restored from
CREST_state, was enough to keep the fleet's 5 years out for good.

Deliberately a separate repo from CREST_state: persist.backup mirrors local
deletions to CREST_state, so fleet data there would be erased by the janitor.
Every failure path is a silent no-op — the run just proceeds uncached.
"""
from __future__ import annotations

import glob
import json
import os
import threading
import time

from hf_data import statecache

REPO = os.environ.get("CREST_FLEET_REPO", "vincewin/CREST_fleet")
# re-probe a key this often: the ungauged fleet re-uploads checkpoints, the
# janitor's LRU cap can evict fetched grids, and a gauge may finish later
TTL_S = float(os.environ.get("CREST_FLEET_TTL_S", "21600"))
MARK = ".fleet.json"

_seen: dict[str, float] = {}
_remote: dict[str, tuple[float, bool]] = {}
_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}


def _token():
    return os.environ.get("HF_TOKEN")


def _key(gauge, model) -> str:
    return f"{str(gauge).zfill(8)}_{model}"


def _download(path: str) -> str | None:
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(REPO, path, repo_type="dataset", token=_token())
    except Exception:
        return None


def _sig(path: str) -> str:
    """Identity of a downloaded fleet file (blob name + size)."""
    try:
        return f"{os.path.basename(os.path.realpath(path))}:{os.path.getsize(path)}"
    except OSError:
        return ""


def _merge_rows(gauge, rows_model, path: str, variant: str | None) -> int:
    """Fleet record -> local record. Returns the number of fleet rows that
    became newly available locally (0: nothing to do / kept the local one)."""
    with open(path) as fh:
        fl = json.load(fh)
    frows = fl.get("rows") or []
    if not frows:
        return 0
    sig = _sig(path)
    loc = statecache.load_record(gauge, rows_model)
    if loc and loc.get("fleet_sig") == sig:
        return 0                                   # this fleet file is already in
    states = sorted(set((loc or {}).get("state_times", []))
                    | set(fl.get("state_times", [])))
    if not loc or not loc.get("rows"):
        rows, var = frows, fl.get("variant")
    elif loc.get("variant") == fl.get("variant"):
        by_time = {r["time"]: r for r in frows}
        n0 = len(by_time)
        by_time.update({r["time"]: r for r in loc["rows"]})   # local rows win
        rows, var = sorted(by_time.values(), key=lambda r: r["time"]), loc.get("variant")
        if len(rows) == len(loc["rows"]) and n0 <= len(loc["rows"]):
            frows = []                             # fleet added no new hour
    elif variant is not None and fl.get("variant") == variant:
        # rows of two run configurations never mix: the run being set up
        # asks for the fleet's, so the fleet's replace the local ones
        rows, var = frows, fl.get("variant")
    else:
        return 0                                   # keep local; re-judged next probe
    rec = {"gauge": str(gauge).zfill(8), "model": rows_model, "rows": rows,
           "window": [rows[0]["time"], rows[-1]["time"]],
           "state_times": states, "variant": var, "fleet_sig": sig}
    dst = statecache.results_path(gauge, rows_model)
    tmp = dst + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh)
    os.replace(tmp, dst)
    return len(frows)


def _merge_states(gauge, state_model, path: str) -> int:
    """Fleet bundle -> loose grids for every time the local dir lacks."""
    sdir = statecache.state_dir(gauge, state_model)
    mark = os.path.join(sdir, MARK)
    sig = _sig(path)
    n_tif = len(glob.glob(os.path.join(sdir, "*.tif")))
    try:
        with open(mark) as fh:
            m = json.load(fh)
        if m.get("sig") == sig and n_tif >= int(m.get("n", 0)):
            return 0                               # unpacked, nothing evicted since
    except Exception:
        pass
    from hf_data import statebundle
    n = statebundle.unpack(path, sdir)             # overwrite=False: local grids win
    try:
        with open(mark, "w") as fh:
            json.dump({"sig": sig, "n": len(glob.glob(os.path.join(sdir, "*.tif")))}, fh)
    except OSError:
        pass
    return n


def ensure_local(gauge, state_model, rows_model: str | None = None,
                 variant: str | None = None) -> str | None:
    """Make the fleet's rows + state grids for this gauge available locally.

    state_model: key of the EF5 state grids (model + scheme).
    rows_model:  key of the row record (defaults to state_model).
    variant:     run-configuration fingerprint of the run being set up —
                 decides which rows stay when local and fleet records disagree.
    Returns "fetched" (fleet data became newly available now), "cached" (the
    fleet's data is already in the local cache) or None (the fleet has
    nothing under these keys, or it could not be reached)."""
    if not REPO or not _token():
        return None
    rows_model = rows_model or state_model
    key = _key(gauge, state_model) + "|" + rows_model
    now = time.time()
    with _lock:
        klock = _key_locks.setdefault(key, threading.Lock())
    with klock:                    # two runs of one gauge: one fetch, one wait
        last = _seen.get(key)
        if last is not None and now - last < TTL_S:    # probed recently
            if not _remote.get(key, (0, False))[1]:
                return None                            # the fleet has nothing here
            rec = statecache.load_record(gauge, rows_model)
            if os.path.exists(os.path.join(
                    statecache.state_dir(gauge, state_model), MARK)) \
                    or (rec and rec.get("fleet_sig")):
                return "cached"
            # fetched earlier but evicted locally since -> merge again
        _seen[key] = now
        got = found = 0
        p = _download(f"results/{_key(gauge, rows_model)}.json")
        if p:
            found += 1
            try:
                got += 1 if _merge_rows(gauge, rows_model, p, variant) else 0
            except Exception:
                pass
        p = _download(f"states/{_key(gauge, state_model)}.pqf")
        if p:
            found += 1
            try:
                got += 1 if _merge_states(gauge, state_model, p) else 0
            except Exception:
                pass
        _remote[key] = (now, bool(found))
        if got:
            return "fetched"
        return "cached" if found else None


def has_remote(gauge, model) -> bool:
    """Does the fleet repo hold state grids under this key? (one cheap HEAD,
    remembered for TTL_S) — used to tell a full-basin run that the fleet's
    pre-run of this gauge exists under the speed scheme."""
    if not REPO or not _token():
        return False
    key = "has|" + _key(gauge, model)
    now = time.time()
    hit = _remote.get(key)
    if hit and now - hit[0] < TTL_S:
        return hit[1]
    try:
        from huggingface_hub import HfApi
        ok = bool(HfApi(token=_token()).file_exists(
            REPO, f"states/{_key(gauge, model)}.pqf", repo_type="dataset"))
    except Exception:
        ok = False
    _remote[key] = (now, ok)
    return ok

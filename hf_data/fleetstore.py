"""Lazy fetch of fleet-precomputed simulations (READ-ONLY repo).

The virtual-user fleet (fleet/fleet_run.py on the offline server) precomputes
years of quick-run results + 10-day states for thousands of gauges into the
private dataset CREST_FLEET_REPO:

  results/<gid>_<cache_model>.json   statecache record (rows + state_times + variant)
  states/<gid>_<cache_model>.pqf     f16 state bundle (statebundle format)

On first touch of a (gauge, cache_model) this pulls both into the local cache,
after which the normal plan()/warm-start machinery serves the user instantly.
Deliberately a separate repo from CREST_state: persist.backup mirrors local
deletions to CREST_state, so fleet data there would be erased by the janitor.
Every failure path is a silent no-op — the run just proceeds uncached.
"""
from __future__ import annotations

import glob
import os
import shutil
import threading

from hf_data import statecache

REPO = os.environ.get("CREST_FLEET_REPO", "vincewin/CREST_fleet")

_seen: set[str] = set()
_lock = threading.Lock()


def _token():
    return os.environ.get("HF_TOKEN")


def ensure_local(gauge, cache_model) -> str | None:
    """Make fleet rows/states for (gauge, cache_model) available locally.
    Returns "cached" (already local), "fetched" (downloaded now) or None."""
    if not REPO or not _token():
        return None
    key = f"{str(gauge).zfill(8)}_{cache_model}"
    with _lock:                    # one repo probe per key per process
        if key in _seen:
            return "cached"
        _seen.add(key)
    have_rows = os.path.exists(statecache.results_path(gauge, cache_model))
    have_states = bool(glob.glob(os.path.join(
        statecache.state_dir(gauge, cache_model), "*.tif")))
    if have_rows and have_states:
        return "cached"
    from huggingface_hub import hf_hub_download
    got = 0
    if not have_rows:
        try:
            p = hf_hub_download(REPO, f"results/{key}.json",
                                repo_type="dataset", token=_token())
            shutil.copyfile(p, statecache.results_path(gauge, cache_model))
            got += 1
        except Exception:
            pass
    if not have_states:
        try:
            p = hf_hub_download(REPO, f"states/{key}.pqf",
                                repo_type="dataset", token=_token())
            from hf_data import statebundle
            got += 1 if statebundle.unpack(
                p, statecache.state_dir(gauge, cache_model)) else 0
        except Exception:
            pass
    return "fetched" if got else None

"""Radar-overlay frame index for the Nowcast-mode MRMS animation.

scripts/update_mrms_recent.py renders each stored Pass1 hour to a colormapped
transparent PNG in CREST_data mrms_frames/ (rolling 7 days). The frontend
animates those straight off the HF CDN; this module only answers "which hours
exist right now" — one repo listing, cached, so scrubbing users don't hammer
the Hub API.
"""
from __future__ import annotations

import os
import re
import threading
import time

REPO = os.environ.get("CREST_FEEDBACK_REPO", "vincewin/CREST_data")
PREFIX = "mrms_frames/"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{PREFIX.rstrip('/')}"
# MRMS CONUS grid: 0.01 deg, origin (-130, 20), 7000x3500 -> [[S,W],[N,E]]
BOUNDS = [[20.0, -130.0], [55.0, -60.0]]
TTL_S = float(os.environ.get("MRMS_FRAMES_TTL_S", "600"))

_lock = threading.Lock()
_cache: dict = {"t": 0.0, "hours": []}


def _list_hours() -> list[str]:
    from huggingface_hub import HfApi
    hours = []
    for f in HfApi().list_repo_files(REPO, repo_type="dataset"):
        m = re.match(rf"{PREFIX}mrms1h_(\d{{10}})\.png$", f)
        if m:
            hours.append(m.group(1))
    return sorted(hours)


def frames() -> dict:
    """{ok, base, bounds, hours:[YYYYMMDDHH ascending]} — cached TTL_S."""
    with _lock:
        if time.time() - _cache["t"] > TTL_S:
            try:
                _cache["hours"] = _list_hours()
                _cache["t"] = time.time()
            except Exception:
                _cache["t"] = time.time() - TTL_S + 60   # retry in a minute
    return {"ok": bool(_cache["hours"]), "base": BASE, "bounds": BOUNDS,
            "hours": _cache["hours"]}

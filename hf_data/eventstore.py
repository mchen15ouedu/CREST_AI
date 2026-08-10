"""Inundation-event results on the CREST_data dataset (V25).

Layout:  events/<event_id>/manifest.json + depth_*.tif + maxdepth.tif + dem.tif
         events/index.json   (id -> summary; the app lists events from this)

Storage discipline (HF limits):
  * ONE create_commit per event — all frames + manifest + index update +
    retention deletions batched together (account budget: 256 commits/h).
  * Frames are uint16-cm DEFLATE GeoTIFFs (~0.1-0.5 MB each at 1"); a whole
    event is ~10-30 MB. KEEP_EVENTS caps steady-state usage (< ~1 GB).
  * No DEM/forcing archive: DEM comes from public 3DEP on demand; runoff
    grids are reproducible from the MRMS archive already in CREST_data.
"""
from __future__ import annotations

import io
import json
import os
import threading

REPO = os.environ.get("CREST_DATA_REPO", "vincewin/CREST_data")
PREFIX = "events"
KEEP_EVENTS = int(os.environ.get("EVENT_KEEP", "20"))

_lock = threading.Lock()


def _api():
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN")
    return HfApi(token=tok) if tok else None


def load_index() -> dict:
    """{event_id: summary} newest-first insertion order; {} on any failure."""
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, f"{PREFIX}/index.json", repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"))
        with open(p, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def publish_event(local_dir: str, manifest: dict) -> bool:
    """Upload one finished event dir + updated index + prune old events,
    all in a single commit. Returns True on success."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api = _api()
    if api is None:
        return False
    ev = manifest["event_id"]
    with _lock:
        idx = load_index()
        idx.pop(ev, None)
        summary = {k: manifest.get(k) for k in
                   ("bbox", "t0", "sim_start", "t_end", "model", "trigger",
                    "generated")}
        summary["n_frames"] = len(manifest.get("frames", []))
        # newest first
        idx = {ev: summary, **idx}
        drop = list(idx.keys())[KEEP_EVENTS:]
        for d in drop:
            idx.pop(d, None)

        ops = []
        for fn in sorted(os.listdir(local_dir)):
            p = os.path.join(local_dir, fn)
            if os.path.isfile(p):
                ops.append(CommitOperationAdd(f"{PREFIX}/{ev}/{fn}", p))
        ops.append(CommitOperationAdd(
            f"{PREFIX}/index.json",
            io.BytesIO(json.dumps(idx).encode())))
        for d in drop:
            ops.append(CommitOperationDelete(f"{PREFIX}/{d}/", is_folder=True))
        try:
            api.create_commit(repo_id=REPO, repo_type="dataset", operations=ops,
                              commit_message=f"event {ev} "
                                             f"(+{len(ops) - 1 - len(drop)} files, "
                                             f"-{len(drop)} old)")
            return True
        except Exception:
            return False


def event_url(event_id: str, filename: str) -> str:
    return (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
            f"{PREFIX}/{event_id}/{filename}")

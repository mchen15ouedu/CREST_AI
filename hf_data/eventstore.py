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
# tiered retention: newest KEEP_FULL events keep their full frame stacks;
# older ones are demoted to manifest + maxdepth (tif+png) only; beyond
# KEEP_EVENTS the folder is deleted. ~8 x 35 MB + 32 x ~1.5 MB < 350 MB.
KEEP_FULL = int(os.environ.get("EVENT_KEEP_FULL", "8"))
KEEP_EVENTS = int(os.environ.get("EVENT_KEEP", "40"))

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
        prior = idx.pop(ev, None)
        summary = {k: manifest.get(k) for k in
                   ("bbox", "t0", "sim_start", "t_end", "model", "trigger",
                    "generated", "gauge")}
        summary["n_frames"] = len(manifest.get("frames", []))
        summary["status"] = manifest.get("status", "active")
        # the episode keeps its first trigger time across hourly re-publishes
        summary["episode_started"] = ((prior or {}).get("episode_started")
                                      or manifest.get("t0"))
        summary["archive_frames"] = manifest.get("archive_frames")
        # newest first
        idx = {ev: summary, **idx}
        drop = list(idx.keys())[KEEP_EVENTS:]
        for d in drop:
            idx.pop(d, None)

        ops = []
        if prior is not None:
            # hourly re-publish of an ongoing episode: replace the folder so
            # frames from the previous (1-h-older) window don't linger with
            # shifted timestamps (deletions apply before additions)
            ops.append(CommitOperationDelete(f"{PREFIX}/{ev}/", is_folder=True))
        for fn in sorted(os.listdir(local_dir)):
            p = os.path.join(local_dir, fn)
            if os.path.isfile(p):
                ops.append(CommitOperationAdd(f"{PREFIX}/{ev}/{fn}", p))
        # demote events beyond KEEP_FULL: drop the frame stacks, keep
        # manifest + maxdepth + dem (validation/archive tier)
        demote = [d for d in list(idx.keys())[KEEP_FULL:]
                  if not idx[d].get("demoted")]
        if demote:
            try:
                allfiles = api.list_repo_files(REPO, repo_type="dataset")
            except Exception:
                allfiles, demote = [], []
            for d in demote:
                for f in allfiles:
                    base = os.path.basename(f)
                    if (f.startswith(f"{PREFIX}/{d}/")
                            and base.startswith("depth_")):
                        ops.append(CommitOperationDelete(f))
                idx[d]["demoted"] = True
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


def mark_ended(event_ids) -> bool:
    """Flip episodes to 'ended' in the index (single small commit). Their
    folders — including archive.parquet — stay until retention removes them."""
    from huggingface_hub import CommitOperationAdd
    api = _api()
    if api is None:
        return False
    with _lock:
        idx = load_index()
        changed = [e for e in event_ids
                   if e in idx and idx[e].get("status") != "ended"]
        if not changed:
            return True
        for e in changed:
            idx[e]["status"] = "ended"
        try:
            api.create_commit(
                repo_id=REPO, repo_type="dataset",
                operations=[CommitOperationAdd(
                    f"{PREFIX}/index.json",
                    io.BytesIO(json.dumps(idx).encode()))],
                commit_message=f"episodes ended: {', '.join(changed)}")
            return True
        except Exception:
            return False


def event_url(event_id: str, filename: str) -> str:
    return (f"https://huggingface.co/datasets/{REPO}/resolve/main/"
            f"{PREFIX}/{event_id}/{filename}")

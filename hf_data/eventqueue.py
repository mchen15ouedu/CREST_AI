"""V27 P1: HF-native job queue that hands event solves to a GPU worker.

The hourly tick still runs EF5 on the Space (next to the data); only the
2-D solver — the actual bottleneck — is offered to workers. Queue layout on
the CREST_data dataset:

  events/queue/<event_id>.json             job spec (domain, times, hydro, basin)
  events/queue/<event_id>.forcing.tar.gz   EF5 output grids for the window
  events/queue/<event_id>.claim            worker identity + heartbeat
  events/queue/<event_id>.failed.json      worker error report (job consumed)

The protocol is WORKER-AGNOSTIC by design — the HPC single-GPU daemon, a
ZeroGPU Space running chunked solves, and future runner Spaces are all just
claimants; the Space never depends on any one of them existing. Degradation
ladder: HPC GPU -> ZeroGPU worker -> local CPU window (T1, always works).

Modes (EVENT_QUEUE_MODE):
  off     no queue traffic; V25 CPU windowed run only.
  shadow  DEFAULT: enqueue the bundle AND still run the local CPU window.
          Workers consume bundles for development/validation without being
          a dependency (they must NOT publish results in this phase).
  on      queue-first: wait CLAIM_WAIT_S for a fresh worker claim, then up
          to RESULT_WAIT_S for the published full-basin result — aborting
          early if the claim heartbeat goes stale — and fall back to the
          local CPU window on any timeout. Flip only after a worker passes
          the P1 validation gates (fork docs/P1_WORKER_BRIEF.md).
"""
from __future__ import annotations

import datetime
import io
import json
import os
import tarfile
import tempfile
import time

from . import eventstore

QPREFIX = f"{eventstore.PREFIX}/queue"
CLAIM_WAIT_S = int(os.environ.get("EVENT_CLAIM_WAIT_S", "300"))
RESULT_WAIT_S = int(os.environ.get("EVENT_RESULT_WAIT_S", "2700"))
CLAIM_FRESH_S = int(os.environ.get("EVENT_CLAIM_FRESH_S", "600"))
QUEUE_MAX_AGE_H = float(os.environ.get("EVENT_QUEUE_MAX_AGE_H", "48"))
_TS = "%Y-%m-%dT%H:%M:%SZ"


def mode() -> str:
    m = os.environ.get("EVENT_QUEUE_MODE", "shadow").strip().lower()
    return m if m in ("off", "shadow", "on") else "shadow"


def enqueue(event_id: str, spec: dict, ef5_dir: str, log=print) -> bool:
    """One commit: job spec + forcing tar (every .tif in ef5_dir — exactly
    the solver's input), replacing any previous claim/error for this id
    (hourly episode re-sims), plus a sweep of queue entries older than
    QUEUE_MAX_AGE_H so dead bundles never accumulate."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete
    api = eventstore._api()
    if api is None:
        return False
    tifs = sorted(f for f in os.listdir(ef5_dir) if f.endswith(".tif"))
    if not tifs:
        log("queue: no EF5 grids to bundle — not enqueued")
        return False
    tar_path = os.path.join(tempfile.gettempdir(), f"{event_id}.forcing.tar.gz")
    try:
        with tarfile.open(tar_path, "w:gz") as tf:
            for f in tifs:
                tf.add(os.path.join(ef5_dir, f), arcname=f)
        ops = [CommitOperationAdd(f"{QPREFIX}/{event_id}.json",
                                  io.BytesIO(json.dumps(spec).encode())),
               CommitOperationAdd(f"{QPREFIX}/{event_id}.forcing.tar.gz",
                                  tar_path)]
        try:
            existing = [f for f in api.list_repo_files(eventstore.REPO,
                                                       repo_type="dataset")
                        if f.startswith(f"{QPREFIX}/")]
        except Exception:
            existing = []
        for f in existing:
            base = os.path.basename(f)
            stale = (eventstore._age_days(base) or 0.0) * 24.0 > QUEUE_MAX_AGE_H
            prior = (base.startswith(f"{event_id}.")
                     and base.split(".", 1)[1] not in ("json",
                                                       "forcing.tar.gz"))
            if stale or prior:
                ops.append(CommitOperationDelete(f))
        mb = os.path.getsize(tar_path) / 1e6
        api.create_commit(repo_id=eventstore.REPO, repo_type="dataset",
                          operations=ops,
                          commit_message=f"queue {event_id} "
                                         f"({len(tifs)} grids, {mb:.0f} MB)")
        log(f"queue: job {event_id} enqueued "
            f"({len(tifs)} grids, {mb:.0f} MB) [mode {mode()}]")
        return True
    except Exception as e:
        log(f"queue: enqueue failed ({type(e).__name__}: {e})")
        return False
    finally:
        try:
            os.remove(tar_path)
        except OSError:
            pass


def _claim(event_id: str) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(eventstore.REPO, f"{QPREFIX}/{event_id}.claim",
                            repo_type="dataset",
                            token=os.environ.get("HF_TOKEN"),
                            force_download=True)
        with open(p, encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return None


def _fresh(claim: dict | None) -> bool:
    if not claim:
        return False
    try:
        hb = datetime.datetime.strptime(claim.get("hb", ""), _TS)
    except ValueError:
        return False
    return (datetime.datetime.utcnow() - hb).total_seconds() < CLAIM_FRESH_S


def wait_for_claim(event_id: str, log=print) -> bool:
    t_stop = time.time() + CLAIM_WAIT_S
    while time.time() < t_stop:
        c = _claim(event_id)
        if _fresh(c):
            log(f"queue: claimed by {c.get('worker', '?')}")
            return True
        time.sleep(20)
    log(f"queue: no worker claim within {CLAIM_WAIT_S} s")
    return False


def wait_for_result(event_id: str, queued_iso: str, log=print) -> dict | None:
    """Poll for a manifest published AFTER the job was queued (the worker's
    publish_event replaces the folder, so `generated` moving past the queue
    timestamp is the completion signal). Aborts EARLY if the worker's claim
    heartbeat goes stale — a died/preempted worker must not delay the local
    CPU fallback by the full timeout."""
    from huggingface_hub import hf_hub_download
    t_stop = time.time() + RESULT_WAIT_S
    while time.time() < t_stop:
        try:
            p = hf_hub_download(eventstore.REPO,
                                f"{eventstore.PREFIX}/{event_id}/manifest.json",
                                repo_type="dataset",
                                token=os.environ.get("HF_TOKEN"),
                                force_download=True)
            with open(p, encoding="utf-8") as fp:
                man = json.load(fp)
            if (man.get("generated") or "") > queued_iso:
                gr = man.get("grid") or {}
                log(f"queue: worker result received "
                    f"({gr.get('ny')}x{gr.get('nx')} cells)")
                return man
        except Exception:
            pass
        if not _fresh(_claim(event_id)):
            log("queue: worker heartbeat went stale — falling back to the "
                "local CPU window now")
            return None
        time.sleep(60)
    log(f"queue: no worker result within {RESULT_WAIT_S} s — "
        f"falling back to the local CPU window")
    return None

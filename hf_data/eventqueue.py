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
        # The id timestamp is the EPISODE START, which an active episode keeps
        # for days — id age alone is NOT queue-entry age. A re-enqueued old
        # episode's fresh bundle (and its worker's live claim!) was swept by
        # this loop keyed on id age (observed 2026-08-14 15:35Z, killing a
        # claim 25 min into a solve). Stale now additionally requires the
        # entry's event to be genuinely idle: no fresh claim, and its spec's
        # own `queued` stamp (not the id) past QUEUE_MAX_AGE_H.
        spec_age_h: dict[str, float] = {}
        claim_ok: dict[str, bool] = {}

        def _entry_idle_and_old(eid: str) -> bool:
            if eid not in claim_ok:
                claim_ok[eid] = _fresh(_claim(eid))
            if claim_ok[eid]:
                return False
            if eid not in spec_age_h:
                age = float("inf")                # spec gone -> orphan, sweep
                sp = _spec(eid)
                if sp:
                    try:
                        age = (datetime.datetime.utcnow() -
                               datetime.datetime.strptime(sp.get("queued", ""),
                                                          _TS)
                               ).total_seconds() / 3600.0
                    except ValueError:
                        pass
                spec_age_h[eid] = age
            return spec_age_h[eid] > QUEUE_MAX_AGE_H

        for f in existing:
            base = os.path.basename(f)
            eid = base.split(".", 1)[0]
            id_old = (eventstore._age_days(base) or 0.0) * 24.0 > QUEUE_MAX_AGE_H
            prior = (base.startswith(f"{event_id}.")
                     and base.split(".", 1)[1] not in ("json",
                                                       "forcing.tar.gz"))
            if prior and base.endswith(".claim") and _fresh(_claim(event_id)):
                # a worker is actively solving the PREVIOUS bundle of this
                # event — deleting its claim mid-run made the whole ladder
                # fall over once (observed live 2026-08-14 14:10Z). Leave it;
                # the worker re-scans the replaced spec when it finishes.
                continue
            if prior or (id_old and eid != event_id
                         and _entry_idle_and_old(eid)):
                ops.append(CommitOperationDelete(f))
        mb = os.path.getsize(tar_path) / 1e6
        api.create_commit(repo_id=eventstore.REPO, repo_type="dataset",
                          operations=ops,
                          commit_message=f"queue {event_id} "
                                         f"({len(tifs)} grids, {mb:.0f} MB)")
        log(f"queue: job {event_id} enqueued "
            f"({len(tifs)} grids, {mb:.0f} MB) [mode {mode()}]")
        _wake_workers(log)
        return True
    except Exception as e:
        log(f"queue: enqueue failed ({type(e).__name__}: {e})")
        return False
    finally:
        try:
            os.remove(tar_path)
        except OSError:
            pass


_qs_cache: dict = {"t": 0.0, "rows": []}


def queue_status() -> list[dict]:
    """Dashboard view of in-flight work: every queued spec with its claim
    state — [{id, gauge, queued, worker}] (worker None = waiting). Cached
    60 s; the event panel polls /api/events every ~20 s."""
    now = time.time()
    if now - _qs_cache["t"] < 60:
        return _qs_cache["rows"]
    rows = []
    api = eventstore._api()
    if api is not None:
        try:
            files = [f for f in api.list_repo_files(eventstore.REPO,
                                                    repo_type="dataset")
                     if f.startswith(f"{QPREFIX}/")]
            ids = sorted({os.path.basename(f)[:-5] for f in files
                          if f.endswith(".json")
                          and not f.endswith(".failed.json")})
            for ev in ids:
                sp = _spec(ev) or {}
                c = _claim(ev) if f"{QPREFIX}/{ev}.claim" in files else None
                rows.append({"id": ev,
                             "gauge": (sp.get("gauge") or {}).get("id"),
                             "queued": sp.get("queued"),
                             "worker": (c or {}).get("worker")
                             if _fresh(c) else None,
                             # last heartbeat even if stale — health's
                             # "when was any worker last alive" signal
                             "hb": (c or {}).get("hb")})
        except Exception:
            pass
    _qs_cache.update(t=now, rows=rows)
    return rows


def claim_fresh_for(event_id: str) -> str | None:
    """Worker ident if a FRESH claim exists for this exact event id (a worker
    is actively solving it), else None."""
    c = _claim(event_id)
    return c.get("worker") if c and _fresh(c) else None


def gauge_pending(gid: str) -> str | None:
    """Event id of any queued spec for this gauge (claimed or not), else
    None. Guards double-triggering: with delegated hand-offs the published
    index lags the queue, so 'is there already work in flight for this
    gauge' must be answered from the queue itself."""
    api = eventstore._api()
    if api is None:
        return None
    try:
        files = api.list_repo_files(eventstore.REPO, repo_type="dataset")
    except Exception:
        return None
    suffix = f"_{gid}.json"
    for f in files:
        base = os.path.basename(f)
        if (f.startswith(f"{QPREFIX}/") and base.endswith(suffix)
                and not base.endswith(".failed.json")):
            return base[:-5]
    return None


def _wake_workers(log=print):
    """Fire-and-forget GET to each worker Space URL so a slept ZeroGPU
    backup wakes when a job appears (Spaces wake on any HTTP request).
    EVENT_WORKER_WAKE_URLS: comma-separated, e.g.
    https://vincewin-crest-gpu-worker.hf.space"""
    urls = [u.strip() for u in
            os.environ.get("EVENT_WORKER_WAKE_URLS", "").split(",") if u.strip()]
    if not urls:
        return
    import requests
    for u in urls:
        try:
            requests.get(u, timeout=10)
        except Exception:
            pass                                  # waking is best-effort
    log(f"queue: pinged {len(urls)} worker Space(s) awake")


def _claim(event_id: str) -> dict | None:
    return _qjson(f"{event_id}.claim")


def _spec(event_id: str) -> dict | None:
    return _qjson(f"{event_id}.json")


def _qjson(basename: str) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(eventstore.REPO, f"{QPREFIX}/{basename}",
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

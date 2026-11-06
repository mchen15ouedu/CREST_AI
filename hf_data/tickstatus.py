"""events/tick_status.json: liveness of the hourly 2-D event tick, published
by the process that runs the tick for the processes that do not.

PI 2026-11-05: the tick moved off the HF CREST_runner Space onto the
Jetstream2 VM (scripts/event_tick_daemon.py, fleet/jetstream2/
crest-events.service). eventsim keeps tick progress in process memory, so a
Space whose EVENT_TICK is off (the runner after the cutover, the dashboard)
can no longer see it. The daemon commits this file to CREST_data: one small
commit per state change, coalesced (daemon start, tick start, every ~15 min
while a tick runs, tick end, stop). health.py and server.py read it here.

Document (schema 1):
  published_at  UTC stamp of this snapshot (the publisher's own heartbeat)
  reasons       what the commit carries (start, tick_start, heartbeat, ...)
  host          EVENT_ENGINE_IDENT of the tick host, else its hostname
  hostname, pid
  code          provenance.stamp(): crest_ai / crestimap / ef5 revisions
  daemon        the daemon's state: idle|ticking|stopping|stopped|wedged
                (a tick past EVENT_TICK_MAX_BUSY_H: the daemon exits and
                systemd restarts it), started, last_issue,
                last_tick_started/ended, last_outcome, last_result, blocked
                (the HF-runner guard), busy_attempts / last_busy_attempt
                (ticks that fell due while one ran), status_commits
  tick          eventsim.tick_status() at publish time
  runner        eventsim.status() without its tick / capacity_skips copies

Readers recompute the clocks (busy_h, phase_age_s) from the stamps, so a
read between two snapshots stays current. Reads are cached READ_TTL_S
(60 s), bounded (5 s connect, 15 s read) and fail soft: a failed refresh
keeps the last good document and reports the error.
"""
from __future__ import annotations

import datetime
import json
import os
import socket
import threading
import time

from . import eventstore

STATUS_FILE = f"{eventstore.PREFIX}/tick_status.json"
SCHEMA = 1
READ_TTL_S = float(os.environ.get("EVENT_TICK_STATUS_TTL_S", "60"))
# a document older than this: its publisher is gone (the daemon refreshes it
# at least hourly; health.TICK_MAX_H, events-setup.sh --rollback-check)
FRESH_H = 2.0
_READ_TIMEOUT = (5, 15)
_TS = "%Y-%m-%dT%H:%M:%SZ"

_lock = threading.Lock()
_cache: dict = {"t": None, "doc": None, "error": None}


def _stamp(t: float | None = None) -> str:
    t = time.time() if t is None else t
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime(_TS)


def _age_s(ts, now: float | None = None) -> float | None:
    try:
        t = datetime.datetime.strptime(str(ts or ""), _TS).replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        return None
    return max(0.0, (time.time() if now is None else now) - t)


def host() -> str:
    return os.environ.get("EVENT_ENGINE_IDENT") or socket.gethostname()


# ---- write side (the tick host) ----------------------------------------------
def build(daemon: dict, reasons=()) -> dict:
    """The document for THIS process's tick, as of now."""
    from . import eventsim
    tick = eventsim.tick_status()
    runner = eventsim.status()
    for k in ("tick", "capacity_skips"):     # one copy each, under "tick"
        runner.pop(k, None)
    try:
        from . import provenance
        code = provenance.stamp()
    except Exception:
        code = {}
    return {"schema": SCHEMA, "published_at": _stamp(), "reasons": list(reasons),
            "host": host(), "hostname": socket.gethostname(), "pid": os.getpid(),
            "code": code, "daemon": dict(daemon), "tick": tick, "runner": runner}


def publish(doc: dict) -> None:
    """One commit of `doc` to CREST_data (raises on failure)."""
    from huggingface_hub import HfApi
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN is not set")
    body = json.dumps(doc, default=str, separators=(",", ":")).encode()
    why = ", ".join(doc.get("reasons") or ["update"])
    HfApi(token=tok).upload_file(
        path_or_fileobj=body, path_in_repo=STATUS_FILE, repo_id=eventstore.REPO,
        repo_type="dataset", commit_message=f"tick status: {why} ({doc.get('host')})"[:200])


# ---- read side (Spaces whose EVENT_TICK is off) --------------------------------
def _fetch() -> dict:
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers
    url = hf_hub_url(eventstore.REPO, STATUS_FILE, repo_type="dataset")
    r = requests.get(url, headers=build_hf_headers(token=os.environ.get("HF_TOKEN")),
                     timeout=_READ_TIMEOUT)
    if r.status_code == 404:
        raise FileNotFoundError(STATUS_FILE)
    r.raise_for_status()
    doc = r.json()
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError("unexpected tick status document")
    return doc


def read_remote(force: bool = False):
    """(document or None, error class name or None); cached READ_TTL_S. A
    failed refresh keeps the last good document (its published_at ages, so
    staleness still shows) and reports the error alongside it."""
    now = time.time()
    with _lock:
        if not force and _cache["t"] is not None and now - _cache["t"] < READ_TTL_S:
            return _cache["doc"], _cache["error"]
        _cache["t"] = now                     # one refresh per window
    try:
        doc, err = _fetch(), None
    except Exception as exc:
        doc, err = None, type(exc).__name__
    with _lock:
        if doc is not None:
            _cache["doc"] = doc
        _cache["error"] = err
        return _cache["doc"], _cache["error"]


def live_tick(doc: dict, now: float | None = None) -> dict:
    """doc's tick with busy_h / phase_age_s recomputed for `now`."""
    tick = dict(doc.get("tick") or {})
    if tick.get("running"):
        busy = _age_s(tick.get("started"), now)
        if busy is not None:
            tick["busy_h"] = round(busy / 3600, 3)
        phase = _age_s(tick.get("phase_started"), now)
        if phase is not None:
            tick["phase_age_s"] = round(phase, 1)
    return tick


def live_runner(doc: dict, now: float | None = None) -> dict:
    """doc's runner in eventsim.status() shape (tick + capacity_skips)."""
    run = dict(doc.get("runner") or {})
    if run.get("running"):
        busy = _age_s(run.get("started"), now)
        if busy is not None:
            run["busy_h"] = round(busy / 3600, 2)
    tick = live_tick(doc, now)
    run["tick"] = tick
    run["capacity_skips"] = list(tick.get("capacity_skips") or [])
    return run


def meta(doc: dict | None, err: str | None, now: float | None = None) -> dict:
    doc = doc or {}
    age = _age_s(doc.get("published_at"), now)
    return {"file": STATUS_FILE, "host": doc.get("host"),
            "published_at": doc.get("published_at"),
            "age_s": None if age is None else round(age, 1), "error": err,
            "daemon": doc.get("daemon"), "code": doc.get("code")}


def remote_status() -> dict:
    """/api/runner_status body for a Space whose EVENT_TICK is off."""
    doc, err = read_remote()
    out = {"tick_source": "remote", "tick_status": meta(doc, err)}
    if doc is None:
        out.update(runner=None, tick=None)
    else:
        out.update(runner=live_runner(doc), tick=live_tick(doc))
    return out


def remote_runner() -> dict | None:
    doc, _ = read_remote()
    return live_runner(doc) if doc is not None else None


def ticking_elsewhere(max_age_h: float = FRESH_H) -> dict | None:
    """The status meta when ANOTHER host runs the tick right now: a document
    newer than max_age_h, from a host other than this one, whose daemon is
    neither stopped nor blocked (a blocked daemon defers to a Space it saw
    ticking, so that Space keeps ticking). None otherwise (no document,
    unreadable, stale, stopped, blocked)."""
    doc, err = read_remote()
    if doc is None:
        return None
    m = meta(doc, err)
    daemon = doc.get("daemon") or {}
    if (m["age_s"] is None or m["age_s"] > max_age_h * 3600 or doc.get("host") == host()
            or daemon.get("state") == "stopped" or daemon.get("blocked")):
        return None
    return m


def run_conflict(status: dict, gid: str = "", max_age_h: float = FRESH_H) -> str | None:
    """Why a manual /api/event_run on a Space whose EVENT_TICK is off would
    collide with the tick host's work, from remote_status(): while the
    fresh, not stopped document shows a tick or an event run in progress,
    the gauge-less auto-detect (the tick's own detection, run there) and the
    gauge the host is running. None otherwise: an unreadable, stale or
    stopped document blocks nothing, nor does another gauge."""
    meta = status.get("tick_status") or {}
    age = meta.get("age_s")
    if (age is None or age > max_age_h * 3600
            or (meta.get("daemon") or {}).get("state") == "stopped"):
        return None
    tick, runner = status.get("tick") or {}, status.get("runner") or {}
    rid = str(runner.get("running") or "")
    if not (tick.get("running") or rid):
        return None
    host = meta.get("host") or "the tick host"
    what = f"event {rid}" if rid else f"a tick (phase {tick.get('phase')})"
    if not gid:
        return f"{host} is running {what}: auto-detect is that tick's job"
    if rid and rid.split("_", 1)[-1] == str(gid):
        return f"{host} is running {what} for gauge {gid}"
    return None

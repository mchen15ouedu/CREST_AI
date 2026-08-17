"""V30 pipeline self-monitoring — /api/health.

One dict answering "is the flood pipeline alive?" without anyone having to
stare at the dashboard: each subsystem reports its freshest heartbeat and
an ok flag against a staleness threshold. The daily review routine leads
its digest with this; twice (detection starvation, queue sweep) the only
failure detector was the user noticing missing events days later.

Checks:
  nowcast   precomputed fleet nowcast issue age      (< 3 h)
  mrms      newest radar animation frame age          (< 3 h)
  tick      last hourly event tick on THIS process    (< 2 h)
  queue     depth, oldest NEVER-PUBLISHED bundle age  (< 6 h unclaimed)
  workers   freshest claim heartbeat anywhere         (informational)
  events    active episodes, publishes in last 24 h   (informational)

When EVENT_RUNNER_URL is set (V30 runner/dashboard split), tick+queue come
from the runner Space's own /api/health so the dashboard reports the
system, not itself.
"""
from __future__ import annotations

import datetime
import os
import threading
import time

NOWCAST_MAX_H = float(os.environ.get("HEALTH_NOWCAST_MAX_H", "3"))
MRMS_MAX_H = float(os.environ.get("HEALTH_MRMS_MAX_H", "3"))
TICK_MAX_H = float(os.environ.get("HEALTH_TICK_MAX_H", "2"))
# a tick that keeps reporting "runner busy" is only healthy while the job it
# waits on is young: one 39-h CPU solve wedged detection for a day and a half
# (2026-08-17) and the tick age alone could not say busy from dead
WEDGE_MAX_H = float(os.environ.get("HEALTH_WEDGE_MAX_H", "4"))
UNCLAIMED_MAX_H = float(os.environ.get("HEALTH_UNCLAIMED_MAX_H", "6"))

_cache = {"t": 0.0, "snap": None}
_lock = threading.Lock()


def _age_h(ts: str, fmt: str) -> float | None:
    try:
        t = datetime.datetime.strptime(ts, fmt)
        return round((datetime.datetime.utcnow() - t).total_seconds()
                     / 3600.0, 2)
    except (ValueError, TypeError):
        return None


def _oldest_queued(rows) -> float | None:
    ages = [_age_h(r.get("queued") or "", "%Y-%m-%dT%H:%M:%SZ")
            for r in rows]
    ages = [a for a in ages if a is not None]
    return max(ages) if ages else None


def snapshot() -> dict:
    with _lock:
        if _cache["snap"] is not None and time.time() - _cache["t"] < 120:
            return _cache["snap"]
    out = {"checked": datetime.datetime.utcnow().strftime(
        "%Y-%m-%dT%H:%M:%SZ")}

    # fleet nowcast freshness
    try:
        from . import nowcaststore
        t0 = nowcaststore.issue_t0()
        age = (round((datetime.datetime.utcnow() - t0).total_seconds()
                     / 3600.0, 2) if t0 else None)
        out["nowcast"] = {"issue_t0": t0.strftime("%Y-%m-%d %H:%MZ")
                          if t0 else None, "age_h": age,
                          "ok": age is not None and age <= NOWCAST_MAX_H}
    except Exception as e:
        out["nowcast"] = {"ok": False, "error": type(e).__name__}

    # radar animation frames
    try:
        from . import mrmsframes
        fr = mrmsframes.frames()
        newest = (fr.get("hours") or [None])[-1]
        age = _age_h(str(newest), "%Y%m%d%H")
        out["mrms"] = {"newest": newest, "age_h": age,
                       "ok": age is not None and age <= MRMS_MAX_H}
    except Exception as e:
        out["mrms"] = {"ok": False, "error": type(e).__name__}

    # event tick + queue: local, or the runner Space's own report
    runner_url = os.environ.get("EVENT_RUNNER_URL", "").rstrip("/")
    remote = None
    if runner_url:
        try:
            import requests
            remote = requests.get(f"{runner_url}/api/health",
                                  timeout=20).json()
        except Exception as e:
            out["runner_space"] = {"ok": False, "url": runner_url,
                                   "error": type(e).__name__}
    if remote:
        out["tick"] = remote.get("tick") or {"ok": False}
        out["queue"] = remote.get("queue") or {"ok": False}
        out["workers"] = remote.get("workers") or {}
        out["runner_space"] = {"ok": True, "url": runner_url}
    else:
        try:
            from . import eventsim
            lt = eventsim._running.get("last_tick")
            age = _age_h(lt or "", "%Y-%m-%dT%H:%M:%SZ")
            busy = eventsim._busy_h()
            out["tick"] = {"last": lt, "age_h": age,
                           "running": eventsim._running.get("id"),
                           "busy_h": busy,
                           "wedged": busy is not None and busy > WEDGE_MAX_H,
                           "ok": age is not None and age <= TICK_MAX_H
                           and not (busy is not None and busy > WEDGE_MAX_H)}
        except Exception as e:
            out["tick"] = {"ok": False, "error": type(e).__name__}
        try:
            from . import eventqueue
            rows = eventqueue.queue_status()
            unclaimed = [r for r in rows if not r.get("worker")]
            # A bundle whose event is already in the published index is
            # FINISHED work whose queue entry has not been swept yet — the
            # sweep only runs on the next enqueue, so a solved re-sim can
            # sit "unclaimed" for up to QUEUE_MAX_AGE_H (48 h) and pin this
            # check red forever. Only a NEVER-published bundle means a flood
            # is going unmapped, so ok/alerting keys off those; the raw
            # unclaimed age stays in the payload for visibility. Same "done"
            # test the tick's CPU-rung sweep uses (eventsim: rid in idx).
            try:
                from . import eventstore as _es
                published = set(_es.load_index())
            except Exception:
                published = set()
            pending = [r for r in unclaimed
                       if str(r.get("id") or "") not in published]
            oldest = _oldest_queued(unclaimed)
            oldest_pending = _oldest_queued(pending)
            hbs = [_age_h(r.get("hb") or "", "%Y-%m-%dT%H:%M:%SZ")
                   for r in rows]
            hbs = [h for h in hbs if h is not None]
            out["queue"] = {"depth": len(rows),
                            "unclaimed": len(unclaimed),
                            "oldest_unclaimed_h": oldest,
                            "unpublished": len(pending),
                            "oldest_unpublished_h": oldest_pending,
                            "mode": eventqueue.mode(),
                            "ok": oldest_pending is None
                            or oldest_pending <= UNCLAIMED_MAX_H}
            out["workers"] = {"last_heartbeat_h": min(hbs) if hbs else None,
                              "active_claims": sum(1 for r in rows
                                                   if r.get("worker"))}
        except Exception as e:
            out["queue"] = {"ok": False, "error": type(e).__name__}
            out["workers"] = {}

    # event throughput (index reads work from any Space)
    try:
        from . import eventstore
        idx = eventstore.load_index()
        day = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
        pub24 = 0
        for s in idx.values():
            try:
                if datetime.datetime.strptime(
                        s.get("generated") or "",
                        "%Y-%m-%dT%H:%M:%SZ") >= day:
                    pub24 += 1
            except ValueError:
                pass
        out["events"] = {"active": sum(1 for s in idx.values()
                                       if s.get("status") == "active"),
                         "published_24h": pub24, "total_listed": len(idx)}
    except Exception as e:
        out["events"] = {"error": type(e).__name__}

    out["ok"] = all(out.get(k, {}).get("ok", True)
                    for k in ("nowcast", "mrms", "tick", "queue"))
    with _lock:
        _cache.update(t=time.time(), snap=out)
    return out

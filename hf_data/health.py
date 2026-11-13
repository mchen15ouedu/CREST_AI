"""V30 pipeline self-monitoring — /api/health.

One dict answering "is the flood pipeline alive?" without anyone having to
stare at the dashboard: each subsystem reports its freshest heartbeat and
an ok flag against a staleness threshold. The daily review routine leads
its digest with this; twice (detection starvation, queue sweep) the only
failure detector was the user noticing missing events days later.

Checks:
  nowcast   precomputed fleet nowcast issue age      (< 3 h)
  mrms      newest radar animation frame age          (< 3 h)
  forcing   MRMS Pass2 / PET / TEMP archive age       (weekly updater)
  tick      last hourly event tick on THIS process    (< 2 h)
            (restart-aware: a container younger than the
             threshold reports "warming", not STALE, and a
             fresh event publish counts as proof it ran)
            EVENT_TICK != "1" (the tick runs elsewhere: the
            Jetstream2 VM since 2026-11-05): the same rules on
            the tick host's events/tick_status.json
            (hf_data/tickstatus.py), plus STALE when that
            file was not refreshed within the threshold (a
            fresh file of a running tick counts as its attempt)
  queue     depth, oldest NEVER-PUBLISHED bundle age  (< 6 h unclaimed)
  workers   freshest claim heartbeat anywhere         (informational)
  events    active episodes, publishes in last 24 h   (informational)

When EVENT_RUNNER_URL is set (V30 runner/dashboard split), tick+queue come
from the runner Space's own /api/health so the dashboard reports the
system, not itself; while the runner is unreachable the tick comes from the
tick host's status file when one is published.
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
QUEUE_REFRESH_MAX_S = 120.0

_cache = {"t": 0.0, "snap": None}
_lock = threading.Lock()
# tick liveness lives in eventsim's process memory, so a Space that
# slept and cold-started reports last_tick=None even though the
# hourly ping is arriving fine (seen 2026-09-10). Uptime tells
# "no ping YET" from "no ping AT ALL".
_PROC_START = time.time()


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


def _newest_publish_h() -> float | None:
    """Age of the freshest event publish [h] — the only tick evidence that
    OUTLIVES a container restart, since the tick stamps `generated` on every
    episode it re-simulates. None when the index is empty or unreadable."""
    try:
        from . import eventstore
        ages = [_age_h(s.get("generated") or "", "%Y-%m-%dT%H:%M:%SZ")
                for s in eventstore.load_index().values()]
        ages = [a for a in ages if a is not None]
        return min(ages) if ages else None
    except Exception:
        return None


def _local_tick_health(last_publish_h=None, refresh_publish=False):
    """Read whole-tick liveness from memory, separately from ping arrivals."""
    from . import eventsim
    progress = eventsim.tick_status()
    lt = progress.get("last_attempt") or eventsim._running.get("last_tick")
    age = _age_h(lt or "", "%Y-%m-%dT%H:%M:%SZ")
    ages = [value for value in (progress.get("busy_h"), eventsim._busy_h())
            if value is not None]
    busy = max(ages) if ages else None
    wedged = bool(progress.get("stalled") or (busy is not None and busy > WEDGE_MAX_H))
    uptime_h = round((time.time() - _PROC_START) / 3600.0, 2)
    failed = progress.get("outcome") == "error" and not progress.get("running")
    capacity_blocked = int(progress.get("capacity_blocked") or 0)
    tick = {"last": lt, "age_h": age,
            "running": eventsim._running.get("id"),
            "tick_running": bool(progress.get("running")), "busy_h": busy, "wedged": wedged,
            "uptime_h": uptime_h, "started": progress.get("started"),
            "completed": progress.get("completed"), "last_success": progress.get("last_success"),
            "phase": progress.get("phase"), "phase_started": progress.get("phase_started"),
            "phase_age_s": progress.get("phase_age_s"), "preflight": progress.get("preflight"),
            "watched_wait": progress.get("watched_wait"),
            "last_progress": progress.get("last_progress"), "outcome": progress.get("outcome"),
            "error": progress.get("error"), "error_phase": progress.get("error_phase"),
            "last_error_at": progress.get("last_error_at")}
    tick["capacity_blocked"] = capacity_blocked
    tick["capacity_skips"] = [row for row in progress.get("capacity_skips", [])
                              if row.get("relevant", True)]
    if age is not None:
        tick["ok"] = age <= TICK_MAX_H and not wedged and not failed and not capacity_blocked
    else:
        pub = _newest_publish_h() if refresh_publish else last_publish_h
        tick["last_publish_h"] = pub
        if pub is not None and pub <= TICK_MAX_H:
            tick.update(ok=not wedged and not failed and not capacity_blocked, via="publish")
        elif uptime_h < TICK_MAX_H:
            tick.update(ok=not wedged and not failed and not capacity_blocked, warming=True)
        else:
            tick["ok"] = False
    return tick


def _tick_here() -> bool:
    """Does THIS process run the hourly tick (server.py's EVENT_TICK gate)?"""
    return os.environ.get("EVENT_TICK", "1") == "1"


def _remote_tick_health(last_publish_h=None, refresh_publish=False):
    """_local_tick_health's rules on the tick host's published status
    (hf_data/tickstatus.py), for a process that does not run the tick. The
    clocks run from the published stamps; a status file not refreshed within
    TICK_MAX_H means its publisher is gone (STALE), and a daemon that
    published its stop is not ticking. While a tick runs, a fresh document
    is itself the attempt: the daemon refreshes it every 15 min of a tick
    and notes the attempts that fall due meanwhile, so a legitimate 2-4 h
    tick is ok and only busy_h past WEDGE_MAX_H (or a daemon that published
    "wedged") reads as wedged. Never raises."""
    from . import tickstatus
    doc, err = tickstatus.read_remote()
    if doc is None:
        pub = _newest_publish_h() if refresh_publish else last_publish_h
        return {"ok": False, "source": "remote", "file": tickstatus.STATUS_FILE,
                "reason": "tick status unreadable", "error": err, "last_publish_h": pub}
    _ts = "%Y-%m-%dT%H:%M:%SZ"
    progress = tickstatus.live_tick(doc)
    runner = tickstatus.live_runner(doc)
    daemon = doc.get("daemon") or {}
    status_age = _age_h(doc.get("published_at") or "", _ts)
    stale = status_age is None or status_age > TICK_MAX_H
    # a restarted daemon's memory starts empty; its persisted schedule does not
    lt = progress.get("last_attempt") or daemon.get("last_tick_started")
    if progress.get("running") and not stale:
        # the HF runner's busy branch stamped each hourly ping that found the
        # tick running; here the heartbeat of a running tick says the same
        lt = max(stamp for stamp in (lt, doc.get("published_at")) if stamp)
    age = _age_h(lt or "", _ts)
    ages = [value for value in (progress.get("busy_h"), runner.get("busy_h"))
            if value is not None]
    busy = max(ages) if ages else None
    daemon_wedged = daemon.get("state") == "wedged"
    wedged = bool(progress.get("stalled") or daemon_wedged
                  or (busy is not None and busy > WEDGE_MAX_H))
    failed = progress.get("outcome") == "error" and not progress.get("running")
    capacity_blocked = int(progress.get("capacity_blocked") or 0)
    stopped = daemon.get("state") == "stopped"
    uptime_h = _age_h(daemon.get("started") or "", _ts)
    tick = {"last": lt, "age_h": age, "running": runner.get("running"),
            "tick_running": bool(progress.get("running")), "busy_h": busy, "wedged": wedged,
            "uptime_h": uptime_h, "started": progress.get("started"),
            "completed": progress.get("completed"), "last_success": progress.get("last_success"),
            "phase": progress.get("phase"), "phase_started": progress.get("phase_started"),
            "phase_age_s": progress.get("phase_age_s"), "preflight": progress.get("preflight"),
            "watched_wait": progress.get("watched_wait"),
            "last_progress": progress.get("last_progress"), "outcome": progress.get("outcome"),
            "error": progress.get("error"), "error_phase": progress.get("error_phase"),
            "last_error_at": progress.get("last_error_at"),
            "capacity_blocked": capacity_blocked,
            "capacity_skips": [row for row in progress.get("capacity_skips", [])
                               if row.get("relevant", True)],
            "source": "remote", "host": doc.get("host"), "file": tickstatus.STATUS_FILE,
            "status_published": doc.get("published_at"), "status_age_h": status_age,
            "status_read_error": err, "daemon_state": daemon.get("state"),
            "blocked": daemon.get("blocked")}
    bad = wedged or failed or capacity_blocked or stale or stopped
    if stale:
        tick["reason"] = "tick status stale"
    elif stopped:
        tick["reason"] = "tick daemon stopped"
    elif daemon_wedged:
        tick["reason"] = "tick wedged: the daemon is restarting"
    if age is not None:
        tick["ok"] = age <= TICK_MAX_H and not bad
    else:
        pub = _newest_publish_h() if refresh_publish else last_publish_h
        tick["last_publish_h"] = pub
        if pub is not None and pub <= TICK_MAX_H:
            tick.update(ok=not bad, via="publish")
        elif uptime_h is not None and uptime_h < TICK_MAX_H:
            tick.update(ok=not bad, warming=True)
        else:
            tick["ok"] = False
    return tick


def _tick_health(last_publish_h=None, refresh_publish=False):
    if _tick_here():
        return _local_tick_health(last_publish_h, refresh_publish)
    return _remote_tick_health(last_publish_h, refresh_publish)


def _queue_read_health():
    from . import eventqueue
    meta = eventqueue.queue_status_meta()
    age = meta.get("age_s")
    fresh = bool(meta.get("initialized") and age is not None
                 and age <= QUEUE_REFRESH_MAX_S and not meta.get("error"))
    return {"read": meta, "initialized": bool(meta.get("initialized")), "fresh": fresh}


def _overall_ok(out):
    return all(out.get(k, {}).get("ok", True)
               for k in ("nowcast", "mrms", "forcing", "tick", "queue", "runner_space"))


def _memory_overlay(cached):
    """Keep progress/error state current while slow feed checks stay cached."""
    out = dict(cached)
    if not os.environ.get("EVENT_RUNNER_URL", "").rstrip("/"):
        try:
            out["tick"] = _tick_health((out.get("tick") or {}).get("last_publish_h"))
        except Exception as exc:
            out["tick"] = {"ok": False, "error": type(exc).__name__}
        try:
            read = _queue_read_health()
            queue = dict(out.get("queue") or {})
            queue.update(read)
            queue["ok"] = read["fresh"] and "depth" in queue and "oldest_unpublished_h" in queue and (
                queue.get("oldest_unpublished_h") is None
                or queue["oldest_unpublished_h"] <= UNCLAIMED_MAX_H)
            out["queue"] = queue
        except Exception as exc:
            out["queue"] = {**(out.get("queue") or {}), "ok": False,
                            "error": type(exc).__name__}
    out["ok"] = _overall_ok(out)
    return out


def snapshot() -> dict:
    cached = None
    with _lock:
        if _cache["snap"] is not None and time.time() - _cache["t"] < 120:
            cached = _cache["snap"]
    if cached is not None:
        return _memory_overlay(cached)
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

    # archive forcing the simulations actually integrate (MRMS Pass2/PET/TEMP).
    # `mrms` above is the hourly Pass1 nowcasting feed, which the updater Space
    # refreshes itself — these three only move when the weekly routine runs, so
    # they need their own eyes (2026-08-20: PET a month stale, nothing said so).
    try:
        from . import forcingfresh
        out["forcing"] = forcingfresh.snapshot()
    except Exception as e:
        out["forcing"] = {"ok": False, "error": type(e).__name__}

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
            out["tick"] = _tick_health(refresh_publish=True)
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
            read = _queue_read_health()
            out["queue"].update(read)
            out["queue"]["ok"] = out["queue"]["ok"] and read["fresh"]
            out["workers"] = {"last_heartbeat_h": min(hbs) if hbs else None,
                              "active_claims": sum(1 for r in rows
                                                   if r.get("worker"))}
        except Exception as e:
            out["queue"] = {"ok": False, "error": type(e).__name__}
            out["workers"] = {}
        if runner_url:
            # The runner owned the tick; THIS process never ticks (EVENT_TICK=0)
            # and its uptime counts from first import, so the local reading
            # said "warming", ok:true, through the whole 2026-10-02..05 runner
            # sleep. Unreachable runner = no tick - unless the tick host
            # publishes its status (the Jetstream2 VM since 2026-11-05): then
            # that file, not the runner, says whether the tick lives.
            if out["tick"].get("status_published"):
                out["tick"]["runner_unreachable"] = True
            else:
                out["tick"] = {"ok": False, "reason": "runner unreachable",
                               "last_publish_h": _newest_publish_h()}

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

    # user-initiated work in THIS process (dashboard simulations + user
    # calibrations, running or waiting for a slot) — top compute priority;
    # CREST_autocal reads it to stay out of the way. Info only, not in ok.
    try:
        from . import caljobs, simjobs
        out["user_ops"] = {
            "sims": sum(1 for j in list(simjobs._JOBS.values()) if not j.done.is_set()),
            "cals": sum(1 for j in list(caljobs._JOBS.values()) if not j.done.is_set())}
    except Exception as e:
        out["user_ops"] = {"error": type(e).__name__}

    out["ok"] = _overall_ok(out)
    with _lock:
        _cache.update(t=time.time(), snap=out)
    return out

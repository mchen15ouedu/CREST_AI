"""The read side of events/tick_status.json (hf_data/tickstatus.py) on a Space
whose EVENT_TICK is off - the HF runner and the dashboard once the hourly tick
runs on the Jetstream2 VM: tick liveness in /api/health (fresh, stale,
wedged, stalled, failed, stopped, unreadable, warming; the dashboard's
unreachable-runner branch), /api/runner_status and /api/events' runner, and
/api/event_tick refusing to tick while another host does. The status file's
fetch is faked; server.py is imported with its background writers off (no
HF_TOKEN, no EPISODE_MERGER). No network."""
from __future__ import annotations

import asyncio
import datetime as dt
import importlib
import os
import sys
import threading
from types import SimpleNamespace

import pytest

from hf_data import eventqueue, eventsim, eventstore, health, tickstatus

_TS = "%Y-%m-%dT%H:%M:%SZ"


def ago(h):
    return (dt.datetime.utcnow() - dt.timedelta(hours=h)).strftime(_TS)


def status_doc(published=0.1, attempt=0.2, running=False, started=None, stalled=False,
               outcome="ok", state="idle", blocked_cap=0, daemon_started=5.0,
               runner=None, runner_started=None, host="js2-events", phase=None):
    """A document as scripts/event_tick_daemon.py publishes it; times in hours ago."""
    tick = {"running": running, "started": ago(started) if started is not None else ago(1.0),
            "completed": ago(0.9), "last_attempt": ago(attempt) if attempt is not None else None,
            "last_success": ago(0.9), "phase": phase or ("run:09353800" if running else "idle"),
            "phase_started": ago(started if started is not None else 0.9),
            "last_progress": ago(0.1), "outcome": None if running else outcome,
            "error": "RuntimeError" if outcome == "error" else None, "error_phase": None,
            "last_error_at": None, "busy_h": 0.0 if running else None, "phase_age_s": None,
            "preflight": False, "watched_wait": bool(stalled), "stalled": stalled,
            "capacity_blocked": blocked_cap,
            "capacity_skips": [{"gauge": "08263500", "relevant": True}] * blocked_cap}
    run = {"running": runner, "status": "ef5" if runner else None,
           "started": ago(runner_started) if runner_started is not None else None,
           "busy_h": None, "log": ["EF5 nowcast-mode run"], "last": None, "missed": []}
    return {"schema": tickstatus.SCHEMA, "published_at": ago(published), "reasons": ["heartbeat"],
            "host": host, "hostname": "crest-js2-gpu1", "pid": 4242,
            "code": {"engine": host, "crest_ai": "abc123"},
            "daemon": {"state": state, "started": ago(daemon_started), "blocked": None},
            "tick": tick, "runner": run}


@pytest.fixture
def remote(monkeypatch):
    """tickstatus' fetch replaced: st.doc is served (an Exception is raised);
    fresh cache; EVENT_TICK off, no runner URL."""
    st = SimpleNamespace(doc=None, fetches=0)

    def fetch():
        st.fetches += 1
        if isinstance(st.doc, Exception):
            raise st.doc
        if st.doc is None:
            raise FileNotFoundError(tickstatus.STATUS_FILE)
        return st.doc

    monkeypatch.setattr(tickstatus, "_fetch", fetch)
    monkeypatch.setattr(tickstatus, "_cache", {"t": None, "doc": None, "error": None})
    monkeypatch.setenv("EVENT_TICK", "0")
    monkeypatch.delenv("EVENT_RUNNER_URL", raising=False)
    monkeypatch.delenv("EVENT_ENGINE_IDENT", raising=False)
    monkeypatch.setattr(health, "_newest_publish_h", lambda: None)
    return st


def fresh(st, **kw):
    st.doc = status_doc(**kw)
    tickstatus._cache["t"] = None


# ---- health: the remote tick ------------------------------------------------------
def test_fresh_remote_tick_is_ok(remote):
    fresh(remote)
    t = health._tick_health()
    assert t["ok"] is True and t["source"] == "remote" and t["host"] == "js2-events"
    assert 0.15 < t["age_h"] < 0.25 and t["status_age_h"] < 0.2 and not t["wedged"]
    assert t["daemon_state"] == "idle" and "reason" not in t


def test_stale_status_file_is_not_ok(remote):
    fresh(remote, published=2.5, attempt=2.6)
    t = health._tick_health()
    assert t["ok"] is False and t["reason"] == "tick status stale"
    # a fresh attempt stamp does not rescue a publisher that went quiet
    fresh(remote, published=2.5, attempt=0.5)
    assert health._tick_health()["ok"] is False


def test_long_running_tick_is_wedged(remote):
    fresh(remote, published=0.2, running=True, started=5.0, attempt=5.0, phase="run:09353800")
    t = health._tick_health()
    assert t["wedged"] is True and t["ok"] is False and t["tick_running"] is True
    assert 4.9 < t["busy_h"] < 5.1                     # recomputed from the start stamp


def test_a_legitimate_3_h_tick_is_ok(remote):
    # its own attempt stamp is 3 h old, but the running tick's heartbeat is fresh
    fresh(remote, published=0.2, running=True, started=3.0, attempt=3.0, phase="run:09353800")
    t = health._tick_health()
    assert t["ok"] is True and t["wedged"] is False and t["tick_running"] is True
    assert 2.9 < t["busy_h"] < 3.1 and t["age_h"] <= 0.25
    assert t["last"] == remote.doc["published_at"] and "reason" not in t
    # an attempt the daemon noted while the tick ran counts just the same
    fresh(remote, published=1.0, running=True, started=3.0, attempt=0.5)
    t = health._tick_health()
    assert t["ok"] is True and 0.45 < t["age_h"] < 0.55
    # a publisher gone quiet does not vouch for its tick
    fresh(remote, published=2.5, running=True, started=3.0, attempt=3.0)
    t = health._tick_health()
    assert t["ok"] is False and t["reason"] == "tick status stale" and t["age_h"] > 2.9
    # an idle daemon's heartbeat is not an attempt: no tick for 3 h is stale
    fresh(remote, published=0.2, attempt=3.0)
    assert health._tick_health()["ok"] is False


def test_a_daemon_that_published_wedged_is_not_ok(remote):
    fresh(remote, published=0.1, running=True, started=8.0, attempt=0.5, state="wedged")
    t = health._tick_health()
    assert t["wedged"] is True and t["ok"] is False
    assert t["reason"] == "tick wedged: the daemon is restarting"


def test_long_runner_job_is_wedged_too(remote):
    fresh(remote, attempt=0.3, runner="2026110520_09353800", runner_started=4.5)
    t = health._tick_health()
    assert t["wedged"] is True and t["ok"] is False and t["running"] == "2026110520_09353800"


def test_stalled_wait_phase_is_wedged(remote):
    fresh(remote, running=True, started=0.4, attempt=0.4, stalled=True,
          phase="preflight.detect")
    t = health._tick_health()
    assert t["wedged"] is True and t["ok"] is False and t["phase"] == "preflight.detect"


def test_failed_and_capacity_blocked_ticks_are_not_ok(remote):
    fresh(remote, outcome="error")
    assert health._tick_health()["ok"] is False
    fresh(remote, blocked_cap=2)
    t = health._tick_health()
    assert t["ok"] is False and t["capacity_blocked"] == 2 and len(t["capacity_skips"]) == 2


def test_stopped_daemon_is_not_ok(remote):
    fresh(remote, state="stopped")
    t = health._tick_health()
    assert t["ok"] is False and t["reason"] == "tick daemon stopped"


def test_warming_daemon_without_a_tick_yet(remote):
    fresh(remote, attempt=None, daemon_started=0.5)
    t = health._tick_health()
    assert t["ok"] is True and t.get("warming") is True
    fresh(remote, attempt=None, daemon_started=3.0)
    assert health._tick_health()["ok"] is False


def test_restarted_daemon_counts_its_persisted_last_tick(remote):
    fresh(remote, attempt=None, daemon_started=0.1)
    remote.doc["daemon"]["last_tick_started"] = ago(0.4)   # before the restart
    t = health._tick_health()
    assert t["ok"] is True and t["last"] == ago(0.4) and "warming" not in t
    remote.doc["daemon"]["last_tick_started"] = ago(2.5)
    tickstatus._cache["t"] = None
    assert health._tick_health()["ok"] is False


def test_unreadable_status_fails_soft(remote):
    remote.doc = OSError("hub down")
    t = health._tick_health()
    assert t == {"ok": False, "source": "remote", "file": "events/tick_status.json",
                 "reason": "tick status unreadable", "error": "OSError", "last_publish_h": None}
    remote.doc = None                                   # never published: 404
    tickstatus._cache["t"] = None
    assert health._tick_health()["error"] == "FileNotFoundError"


def test_a_failed_refresh_keeps_the_last_good_document(remote):
    fresh(remote)
    assert health._tick_health()["ok"] is True
    remote.doc = OSError("blip")
    tickstatus._cache["t"] = None                       # the 60 s cache ran out
    t = health._tick_health()
    assert t["ok"] is True and t["status_read_error"] == "OSError"


def test_reads_are_cached(remote):
    fresh(remote)
    for _ in range(5):
        health._tick_health()
    assert remote.fetches == 1
    tickstatus._cache["t"] -= tickstatus.READ_TTL_S + 1
    health._tick_health()
    assert remote.fetches == 2


def test_local_tick_when_this_process_ticks(remote, monkeypatch):
    fresh(remote)
    monkeypatch.setenv("EVENT_TICK", "1")
    monkeypatch.setattr(eventsim, "_tick_state", dict(eventsim._tick_state,
                                                       last_attempt=ago(0.1)))
    t = health._tick_health()
    assert "source" not in t and remote.fetches == 0 and t["last"] == ago(0.1)


def _feeds(monkeypatch):
    """snapshot()'s other feeds, healthy and offline."""
    from hf_data import forcingfresh, mrmsframes, nowcaststore
    monkeypatch.setattr(health, "_cache", {"t": 0.0, "snap": None})
    monkeypatch.setattr(nowcaststore, "issue_t0", lambda: dt.datetime.utcnow())
    monkeypatch.setattr(mrmsframes, "frames",
                        lambda: {"hours": [dt.datetime.utcnow().strftime("%Y%m%d%H")]})
    monkeypatch.setattr(forcingfresh, "snapshot", lambda: {"ok": True})
    monkeypatch.setattr(eventqueue, "queue_status", lambda: [])
    monkeypatch.setattr(eventqueue, "mode", lambda: "on")
    monkeypatch.setattr(eventqueue, "queue_status_meta",
                        lambda: {"initialized": True, "age_s": 5.0, "error": None})
    monkeypatch.setattr(eventstore, "load_index", lambda revision=None: {})


def test_runner_snapshot_and_overlay_use_the_remote_tick(remote, monkeypatch):
    _feeds(monkeypatch)
    fresh(remote)
    snap = health.snapshot()
    assert snap["tick"]["source"] == "remote" and snap["tick"]["ok"] is True
    assert snap["ok"] is True
    fresh(remote, state="stopped")                     # the cached snapshot's overlay
    snap = health.snapshot()
    assert snap["tick"]["reason"] == "tick daemon stopped" and snap["ok"] is False


def test_dashboard_with_unreachable_runner(remote, monkeypatch):
    import requests
    _feeds(monkeypatch)
    monkeypatch.setenv("EVENT_RUNNER_URL", "https://runner.invalid")

    def down(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(requests, "get", down)
    fresh(remote)
    snap = health.snapshot()
    assert snap["tick"]["source"] == "remote" and snap["tick"]["ok"] is True
    assert snap["tick"]["runner_unreachable"] is True
    assert snap["runner_space"]["ok"] is False and snap["ok"] is False
    monkeypatch.setattr(health, "_cache", {"t": 0.0, "snap": None})
    remote.doc = None                                   # nothing published (before the cutover)
    tickstatus._cache.update(t=None, doc=None)
    assert health.snapshot()["tick"] == {"ok": False, "reason": "runner unreachable",
                                         "last_publish_h": None}


# ---- tickstatus helpers --------------------------------------------------------------
def test_ticking_elsewhere(remote, monkeypatch):
    monkeypatch.setenv("EVENT_ENGINE_IDENT", "space-cpu:vincewin/CREST_runner")
    fresh(remote)
    m = tickstatus.ticking_elsewhere()
    assert m["host"] == "js2-events" and m["daemon"]["state"] == "idle"
    for kw in ({"state": "stopped"}, {"published": 2.5}, {"host": "space-cpu:vincewin/CREST_runner"}):
        fresh(remote, **kw)
        assert tickstatus.ticking_elsewhere() is None, kw
    # a daemon its guard holds back defers to the Space it saw ticking: no deadlock
    fresh(remote)
    remote.doc["daemon"]["blocked"] = "the HF runner ticked 0.5 h ago"
    assert tickstatus.ticking_elsewhere() is None
    remote.doc = OSError("down")
    tickstatus._cache.update(t=None, doc=None)
    assert tickstatus.ticking_elsewhere() is None


def test_fetch_rejects_missing_and_foreign_documents(monkeypatch):
    import requests

    class R:
        def __init__(self, code, body=None):
            self.status_code, self.body = code, body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return self.body

    seen = []
    replies = [R(404), R(200, {"schema": 99}), R(200, ["x"]), R(503),
               R(200, status_doc())]
    monkeypatch.setattr(requests, "get",
                        lambda url, headers=None, timeout=None: seen.append((url, timeout))
                        or replies.pop(0))
    for exc in (FileNotFoundError, ValueError, ValueError, requests.HTTPError):
        with pytest.raises(exc):
            tickstatus._fetch()
    assert tickstatus._fetch()["host"] == "js2-events"
    assert seen[0][0].endswith("/datasets/vincewin/CREST_data/resolve/main/events/tick_status.json")
    assert seen[0][1] == (5, 15)


# ---- server.py --------------------------------------------------------------------------
@pytest.fixture
def server(monkeypatch, remote):
    """server.py, imported once with its writers off: persist (no HF_TOKEN),
    the merger supervisor (no EPISODE_MERGER), no OAuth / Sentry."""
    if "server" not in sys.modules:
        for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "EPISODE_MERGER", "OAUTH_CLIENT_ID",
                  "SENTRY_DSN"):
            monkeypatch.delenv(k, raising=False)
        hook = threading.excepthook
        try:
            importlib.import_module("server")
        finally:
            threading.excepthook = hook
    srv = sys.modules["server"]
    monkeypatch.setitem(eventsim._running, "id", None)
    monkeypatch.setattr(eventsim, "_tick_state", dict(eventsim._tick_state, running=False))
    return srv


def test_runner_status_remote(server, remote):
    fresh(remote, running=True, started=0.5, attempt=0.5, runner="2026110520_09353800",
          runner_started=0.4)
    out = asyncio.run(server.api_runner_status())
    assert out["tick_source"] == "remote" and out["tick_status"]["host"] == "js2-events"
    assert out["tick"]["running"] is True and 0.45 < out["tick"]["busy_h"] < 0.55
    assert out["runner"]["running"] == "2026110520_09353800"
    assert 0.35 < out["runner"]["busy_h"] < 0.45 and out["runner"]["tick"]["running"] is True
    assert "queue" in out and "local_runner" not in out


def test_runner_status_remote_unreadable_and_a_local_run(server, remote, monkeypatch):
    remote.doc = OSError("down")
    monkeypatch.setitem(eventsim._running, "id", "2026110521_01234567")
    out = asyncio.run(server.api_runner_status())
    assert out["runner"] is None and out["tick"] is None
    assert out["tick_status"]["error"] == "OSError"
    assert out["local_runner"]["running"] == "2026110521_01234567"


def test_runner_status_local_when_this_space_ticks(server, remote, monkeypatch):
    monkeypatch.setenv("EVENT_TICK", "1")
    fresh(remote)
    out = asyncio.run(server.api_runner_status())
    assert out["tick_source"] == "local" and remote.fetches == 0
    assert out["tick"] == eventsim.tick_status() and out["runner"]["running"] is None


def test_events_runner_falls_back_to_the_tick_host(server, remote, monkeypatch):
    fresh(remote, runner="2026110520_09353800", runner_started=0.4)
    assert server._tick_host_runner()["running"] == "2026110520_09353800"
    monkeypatch.setitem(eventsim._running, "id", "2026110521_01234567")
    assert server._tick_host_runner() is None             # this Space's own job shows
    monkeypatch.setitem(eventsim._running, "id", None)
    monkeypatch.setenv("EVENT_TICK", "1")
    assert server._tick_host_runner() is None


@pytest.mark.parametrize("kw, ticks", [({}, False), ({"state": "stopped"}, True),
                                       ({"published": 3.0}, True), (None, True)])
def test_event_tick_refused_while_another_host_ticks(server, remote, monkeypatch, kw, ticks):
    monkeypatch.setenv("EVENT_TICK", "1")
    monkeypatch.setenv("EVENT_ENGINE_IDENT", "space-cpu:vincewin/CREST_runner")
    monkeypatch.setitem(sys.modules, "crestimap", SimpleNamespace())
    started = threading.Event()
    monkeypatch.setattr(eventsim, "hourly_tick", started.set)
    if kw is None:
        remote.doc = None
    else:
        fresh(remote, **kw)
    out = server.api_event_tick()
    if ticks:
        assert out == {"ok": True, "started": "tick"} and started.wait(5)
    else:
        assert out["ok"] is False and out["reason"] == "event tick runs on js2-events"
        assert out["tick_status"]["daemon"]["state"] == "idle" and not started.is_set()


class Runs:
    """eventsim.run_one / run_detected stand-ins: the gauges started (in the
    endpoint's worker threads)."""

    def __init__(self):
        self.calls, self.cv = [], threading.Condition()

    def add(self, what):
        with self.cv:
            self.calls.append(what)
            self.cv.notify_all()

    def wait(self, what, timeout=5):
        with self.cv:
            return self.cv.wait_for(lambda: what in self.calls, timeout)


@pytest.fixture
def runs(server, remote, monkeypatch):
    """/api/event_run on the runner after the cutover: EVENT_TICK off, no
    EVENT_RUNNER_URL, crestimap present; runs recorded, not run."""
    r = Runs()
    monkeypatch.setattr(server, "EVENT_RUNNER_URL", "")
    monkeypatch.setitem(sys.modules, "crestimap", SimpleNamespace())
    monkeypatch.setattr(eventsim, "run_one", lambda gid, t0=None, **kw: r.add(gid))
    monkeypatch.setattr(eventsim, "run_detected", lambda: r.add("auto-detect"))
    return r


def test_event_run_refuses_what_the_tick_host_runs(server, remote, runs):
    fresh(remote, running=True, started=0.5, attempt=0.5, runner="2026110520_09353800",
          runner_started=0.4)
    out = server.api_event_run()
    assert out["ok"] is False and out["reason"].startswith("js2-events is running event "
                                                           "2026110520_09353800")
    assert "auto-detect" in out["reason"] and out["tick_status"]["host"] == "js2-events"
    out = server.api_event_run(gid="09353800")
    assert out["ok"] is False and out["reason"].endswith("for gauge 09353800")
    assert server.api_event_run(episode="2026110300_09353800")["ok"] is False
    out = server.api_event_run(gid="01011000")          # another gauge runs here
    assert out["ok"] is True and out["started"] == "01011000" and runs.wait("01011000")
    assert runs.calls == ["01011000"]
    # a tick still in its preflight: only the auto-detect is its job
    fresh(remote, running=True, started=0.2, attempt=0.2, phase="preflight.detect")
    out = server.api_event_run()
    assert out["ok"] is False and "a tick (phase preflight.detect)" in out["reason"]
    assert server.api_event_run(gid="09353800")["ok"] is True and runs.wait("09353800")


@pytest.mark.parametrize("kw", [{}, {"state": "stopped"}, {"published": 3.0}, None])
def test_event_run_free_while_the_tick_host_idles_stopped_or_stale(server, remote, runs, kw):
    if kw is None:
        remote.doc = OSError("down")                    # unreadable: nothing to collide with
    elif kw:
        fresh(remote, running=True, started=0.5, attempt=0.5,
              runner="2026110520_09353800", runner_started=0.4, **kw)
    else:
        fresh(remote)                                    # idle between ticks
    out = server.api_event_run()
    assert out["ok"] is True and out["started"] == "auto-detect" and runs.wait("auto-detect")
    assert server.api_event_run(gid="09353800")["ok"] is True and runs.wait("09353800")


def test_event_run_on_a_ticking_space_reads_no_remote_status(server, remote, runs,
                                                            monkeypatch):
    monkeypatch.setenv("EVENT_TICK", "1")
    fresh(remote, running=True, started=0.5, attempt=0.5, runner="2026110520_09353800",
          runner_started=0.4)
    assert server.api_event_run(gid="09353800")["ok"] is True and runs.wait("09353800")
    assert remote.fetches == 0


def test_run_conflict_rules():
    now_doc = {"tick_status": {"age_s": 60.0, "host": "js2-events",
                               "daemon": {"state": "ticking"}},
               "tick": {"running": True, "phase": "run:09353800"},
               "runner": {"running": "2026110520_09353800"}}
    assert tickstatus.run_conflict(now_doc, "")
    assert tickstatus.run_conflict(now_doc, "09353800")
    assert tickstatus.run_conflict(now_doc, "01011000") is None
    assert tickstatus.run_conflict({}, "") is None
    stale = dict(now_doc, tick_status=dict(now_doc["tick_status"], age_s=2.1 * 3600))
    assert tickstatus.run_conflict(stale, "") is None
    idle = dict(now_doc, tick={"running": False}, runner={"running": None})
    assert tickstatus.run_conflict(idle, "") is None


def test_dashboard_still_forwards_the_tick(server, monkeypatch):
    monkeypatch.setenv("EVENT_TICK", "0")
    monkeypatch.setattr(server, "EVENT_RUNNER_URL", "")
    assert server.api_event_tick() == {"ok": False, "reason": "event tick disabled on this Space"}
    assert os.environ["EVENT_TICK"] == "0"

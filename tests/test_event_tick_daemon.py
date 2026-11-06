"""scripts/event_tick_daemon.py (the hourly 2-D event tick on the Jetstream2
VM) on fakes: a fake clock, nowcast issue, tick and status upload for the
scheduler (issue advance at most once per issue, the max gap, never two ticks
at once - a tick due meanwhile is noted as an attempt -, the persisted
schedule, heartbeats, the wedged-tick bound); the coalescing status
Publisher; the HF-runner guard and its sticky block; SIGTERM/SIGINT between,
during and in the middle of a pass, the stop budget and the bounded final
commit with real (short) waits; systemd's notify socket. No network, no EF5,
no Hub."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import signal
import socket
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from hf_data import eventsim, eventstore
from scripts import event_tick_daemon as etd

T0 = dt.datetime(2026, 11, 5, 18, tzinfo=dt.timezone.utc).timestamp()


def issue(h):
    return dt.datetime(2026, 11, 5, h)


class Clock:
    def __init__(self, t=T0):
        self.t = float(t)

    def __call__(self):
        return self.t


class Ticks:
    """hourly_tick stand-in: counts calls; blocks on `gate` when asked."""

    def __init__(self, block=False, result=None):
        self.calls, self.gate, self.inside = 0, threading.Event(), threading.Event()
        self.result = result or {"ran": 1, "ended": [], "swept": 0, "results": []}
        if not block:
            self.gate.set()

    def __call__(self):
        self.calls += 1
        self.inside.set()
        assert self.gate.wait(10)
        return self.result


def make(clock=None, ticks=None, issue_fn=None, **kw):
    """A Daemon on a fake clock whose status commits land in `uploads` as
    {"reasons", "daemon"} (the daemon's state at commit time)."""
    clock = clock or Clock()
    uploads, logs = [], []
    kw.setdefault("tick_status", lambda: {"outcome": "ok", "stalled": False, "phase": "idle"})
    kw.setdefault("note_attempt", lambda: None)
    kw.setdefault("notify", lambda msg: False)
    d = etd.Daemon(tick=ticks or Ticks(), issue=issue_fn, clock=clock, log=logs.append,
                   publisher=etd.Publisher(lambda r: None, lambda doc: None, clock=clock),
                   **kw)
    d.publisher = etd.Publisher(lambda r: {"reasons": list(r), "daemon": d.info()},
                                uploads.append, clock=clock, log=logs.append)
    return d, clock, uploads, logs


def settle(d):
    """The running tick thread has returned (the next step reaps it)."""
    assert d._done.wait(5)


# ---- scheduling ------------------------------------------------------------------
def test_first_tick_then_once_per_issue_and_on_advance():
    cur = [issue(17)]
    d, clock, _, _ = make(issue_fn=lambda: cur[0])
    d.step()
    assert d.ticking() and d.last_why == "start" and d.last_issue == issue(17)
    settle(d)
    for _ in range(10):                              # same issue, well inside the gap
        clock.t += 61
        d.step()
    assert d.tick_fn.calls == 1 and not d.ticking()
    cur[0] = issue(18)                               # the nowcast issue advances
    clock.t += 61
    d.step()
    assert d.tick_fn.calls == 2 and d.last_why == "issue" and d.last_issue == issue(18)
    settle(d)
    clock.t += 61
    d.step()
    assert d.tick_fn.calls == 2                      # at most once per issue


def test_max_gap_ticks_without_a_new_issue():
    d, clock, _, _ = make(issue_fn=lambda: issue(17))
    d.step()
    settle(d)
    start = clock.t
    while clock.t - start < 69 * 60:
        clock.t += 60
        d.step()
    assert d.tick_fn.calls == 1
    clock.t = start + 70 * 60
    d.step()
    assert d.tick_fn.calls == 2 and d.last_why == "max_gap"


def test_unreadable_issue_still_keeps_the_max_gap():
    def broken():
        raise OSError("hub down")
    d, clock, _, logs = make(issue_fn=broken)
    d.step()
    settle(d)
    assert d.last_why == "start" and d.last_issue is None
    clock.t += 70 * 60
    d.step()
    assert d.tick_fn.calls == 2 and any("issue unreadable" in s for s in logs)


def test_never_two_ticks_at_once():
    ticks = Ticks(block=True)
    cur = [issue(17)]
    d, clock, _, _ = make(ticks=ticks, issue_fn=lambda: cur[0])
    d.step()
    assert ticks.inside.wait(5)
    for h in range(18, 22):                          # issues advance, the gap passes
        cur[0] = issue(h)
        clock.t += 40 * 60
        d.step()
    assert ticks.calls == 1 and d.ticking()
    ticks.gate.set()
    settle(d)
    clock.t += 60
    d.step()                                         # reaps, then the due tick starts
    assert ticks.calls == 2 and d.last_why == "issue" and d.last_issue == issue(21)
    settle(d)


def test_schedule_survives_a_restart(tmp_path):
    state = str(tmp_path / "event_tick_daemon.json")
    d, clock, _, _ = make(issue_fn=lambda: issue(17), state_path=state)
    d.step()
    settle(d)
    clock.t += 30
    d.step()
    saved = json.loads(open(state).read())
    assert saved["last_issue"] == "2026-11-05T17:00Z" and saved["last_outcome"] == "ok"
    clock.t += 10 * 60                               # restart 10 min later, same issue
    d2, _, _, _ = make(clock=clock, issue_fn=lambda: issue(17), state_path=state)
    d2.step()
    assert d2.tick_fn.calls == 0 and not d2.ticking()
    d3, _, _, _ = make(clock=clock, issue_fn=lambda: issue(18), state_path=state)
    d3.step()
    assert d3.tick_fn.calls == 1 and d3.last_why == "issue"
    settle(d3)


def test_tick_errors_are_recorded_and_the_loop_goes_on():
    def boom():
        raise RuntimeError("EF5 failed")
    d, clock, _, logs = make(ticks=boom, issue_fn=lambda: issue(17))
    d.step()
    settle(d)
    clock.t += 30
    d.step()
    assert d.last_outcome == "exception" and "EF5 failed" in d.last_result["error"]
    assert not d.ticking() and any("exception after" in s for s in logs)


def test_skipped_tick_outcome():
    d, clock, _, _ = make(ticks=Ticks(result={"skipped": "runner busy"}),
                          issue_fn=lambda: issue(17))
    d.step()
    settle(d)
    clock.t += 30
    d.step()
    assert d.last_outcome == "skipped" and d.last_result["skipped"] == "runner busy"


def test_a_tick_due_while_one_runs_is_noted_as_an_attempt():
    ticks, notes, cur = Ticks(block=True), [], [issue(17)]
    d, clock, _, logs = make(ticks=ticks, issue_fn=lambda: cur[0],
                             note_attempt=lambda: notes.append(clock.t))
    d.step()
    assert ticks.inside.wait(5)
    for _ in range(30):                              # same issue, inside the max gap
        clock.t += 61
        d.step()
    assert notes == []
    cur[0] = issue(18)                               # due: not started, noted once
    clock.t += 61
    d.step()
    assert len(notes) == 1 and ticks.calls == 1 and d.busy_attempts == 1
    for _ in range(10):
        clock.t += 61
        d.step()
    assert len(notes) == 1
    cur[0] = issue(19)
    clock.t += 61
    d.step()
    assert len(notes) == 2
    noted = clock.t
    clock.t = noted + 69 * 60
    d.step()
    assert len(notes) == 2
    clock.t = noted + 70 * 60                        # the max gap since the last note
    d.step()
    assert len(notes) == 3 and d.info()["busy_attempts"] == 3
    assert d.info()["last_busy_attempt"] == etd._iso(clock.t)
    assert any("while tick #1 runs" in s and "attempt noted" in s for s in logs)
    ticks.gate.set()
    settle(d)
    clock.t += 61
    d.step()                                         # reaped; the newest issue ticks next
    assert ticks.calls == 2 and d.last_why == "issue" and d.last_issue == issue(19)
    settle(d)


def test_the_noted_attempt_is_eventsims(monkeypatch):
    """The default note_attempt stamps eventsim's tick.last_attempt, which the
    status document carries to /api/health."""
    monkeypatch.setattr(eventsim, "_tick_state", dict(eventsim._tick_state, last_attempt=None))
    monkeypatch.setattr(eventsim, "_running", dict(eventsim._running))
    ticks, cur = Ticks(block=True), [issue(17)]
    d, clock, _, _ = make(ticks=ticks, issue_fn=lambda: cur[0],
                          note_attempt=eventsim.note_tick_attempt)
    d.step()
    assert ticks.inside.wait(5) and eventsim.tick_status()["last_attempt"] is None
    cur[0] = issue(18)
    clock.t += 61
    d.step()
    assert eventsim.tick_status()["last_attempt"] is not None
    assert etd.Daemon(tick=Ticks()).note_attempt_fn is eventsim.note_tick_attempt
    ticks.gate.set()
    settle(d)


def test_a_wedged_tick_publishes_wedged_and_exits_for_a_restart(tmp_path):
    state = str(tmp_path / "event_tick_daemon.json")
    ticks, exits, dumps = Ticks(block=True), [], []
    d, clock, ups, logs = make(ticks=ticks, issue_fn=lambda: issue(17), state_path=state,
                               max_busy_s=8 * 3600, exit_fn=exits.append,
                               dump=lambda: dumps.append(1),
                               tick_status=lambda: {"phase": "run:09353800", "stalled": False})
    d.step()
    assert ticks.inside.wait(5)
    start = clock.t
    while clock.t + 600 < start + 8 * 3600:
        clock.t += 600
        d.step()
    clock.t = start + 8 * 3600 - 1
    d.step()
    assert exits == [] and d.state == "ticking"
    clock.t = start + 8 * 3600
    d.step()
    assert exits == [etd.EXIT_WEDGED] and dumps == [1] and d.state == "wedged"
    assert "wedged" in ups[-1]["reasons"] and ups[-1]["daemon"]["state"] == "wedged"
    assert ups[-1]["daemon"]["last_result"] == {"busy_h": 8.0, "phase": "run:09353800"}
    assert any("WEDGED" in s and "phase run:09353800" in s for s in logs)
    saved = json.loads(open(state).read())
    assert saved["last_outcome"] == "wedged" and saved["last_issue"] == "2026-11-05T17:00Z"
    ticks.gate.set()
    # systemd restarts it: the saved start is 8 h old, so the max gap ticks at once
    d2, _, _, _ = make(clock=clock, issue_fn=lambda: issue(17), state_path=state)
    assert d2.last_outcome == "wedged"
    d2.step()
    assert d2.tick_fn.calls == 1 and d2.last_why == "max_gap"
    settle(d2)


def test_no_busy_bound_when_zero():
    ticks, exits = Ticks(block=True), []
    d, clock, _, _ = make(ticks=ticks, issue_fn=lambda: issue(17), max_busy_s=0,
                          exit_fn=exits.append)
    d.step()
    assert ticks.inside.wait(5)
    clock.t += 48 * 3600
    d.step()
    assert exits == [] and d.ticking()
    ticks.gate.set()
    settle(d)


def test_bounds_line_wants_the_ef5_cap_binding(monkeypatch):
    from hf_data import runner
    monkeypatch.setattr(runner, "RUN_TIMEOUT_S", 10 * 3600.0)
    assert "WARNING" in etd.bounds_line(8)
    monkeypatch.setattr(runner, "RUN_TIMEOUT_S", 3 * 3600.0)
    line = etd.bounds_line(8)
    assert "WARNING" not in line and "EF5 run cap 3 h" in line and "tick 8 h" in line
    assert "unbounded" in etd.bounds_line(0)
    env = open(os.path.join(etd.ROOT, "fleet", "jetstream2", "events.env.example")).read()
    cap = float(re.search(r"^CREST_RUN_TIMEOUT_S=(\d+)$", env, re.M).group(1))
    busy = float(re.search(r"^EVENT_TICK_MAX_BUSY_H=(\d+)$", env, re.M).group(1))
    assert 2 * cap < busy * 3600                     # the VM's settings: EF5 binds first
    for key in ("GDAL_HTTP_TIMEOUT", "GDAL_HTTP_CONNECTTIMEOUT", "GDAL_HTTP_LOW_SPEED_TIME",
                "GDAL_HTTP_LOW_SPEED_LIMIT"):
        assert int(re.search(rf"^{key}=(\d+)$", env, re.M).group(1)) > 0, key


def test_each_pass_pings_the_systemd_watchdog():
    pings = []
    d, clock, _, _ = make(issue_fn=lambda: issue(17), notify=pings.append)
    d.step()
    settle(d)
    clock.t += 30
    d.step()
    assert pings == ["WATCHDOG=1", "WATCHDOG=1"]


# ---- status commits ----------------------------------------------------------------
def test_publisher_coalesces_bursts_and_spaces_commits():
    clock, ups = Clock(), []
    p = etd.Publisher(lambda r: {"reasons": list(r)}, ups.append, clock=clock,
                      coalesce_s=20, min_commit_s=60)
    p.request("start")
    clock.t += 1
    p.request("tick_start")
    p.request("tick_start")
    assert not p.pump(clock.t + 5)                   # the burst is still settling
    assert p.pump(T0 + 20) and ups == [{"reasons": ["start", "tick_start"]}]
    assert not p.pump(T0 + 21)                       # nothing pending
    clock.t = T0 + 30
    p.request("tick_end")
    assert not p.pump(T0 + 55) and not p.pump(T0 + 79)   # 60 s after the last commit
    assert p.pump(T0 + 80) and ups[-1] == {"reasons": ["tick_end"]} and p.commits == 2


def test_publisher_backs_off_and_keeps_the_reasons():
    clock, ups, fail = Clock(), [], [2]

    def upload(doc):
        if fail[0]:
            fail[0] -= 1
            raise OSError("429")
        ups.append(doc)

    builds = []
    p = etd.Publisher(lambda r: builds.append(list(r)) or {"reasons": list(r)}, upload,
                      clock=clock, coalesce_s=0, min_commit_s=0, backoff=(60, 900),
                      log=lambda s: None)
    p.request("start")
    assert not p.pump(T0) and p.retry_at == T0 + 60 and p.last_error == "OSError"
    p.request("tick_start")
    assert not p.pump(T0 + 59)
    assert not p.pump(T0 + 60) and p.retry_at == T0 + 180       # 60 s doubling
    assert p.pump(T0 + 180) and ups == [{"reasons": ["start", "tick_start"]}]
    assert p.fails == 0 and p.last_error is None and len(builds) == 3


def test_publisher_force_ignores_coalescing():
    clock, ups = Clock(), []
    p = etd.Publisher(lambda r: {"reasons": list(r)}, ups.append, clock=clock)
    p.request("stop")
    assert p.pump(force=True) and ups == [{"reasons": ["stop"]}]


def test_heartbeats_while_a_tick_runs_and_one_commit_per_burst():
    ticks = Ticks(block=True)
    d, clock, ups, _ = make(ticks=ticks, issue_fn=lambda: issue(17), beat_s=15 * 60)
    d.publisher.request("start")
    d.step()                                         # tick_start joins "start"
    assert ticks.inside.wait(5)
    clock.t += 20
    d.step()
    assert [u["reasons"] for u in ups] == [["start", "tick_start"]]
    assert ups[0]["daemon"]["state"] == "ticking"
    for _ in range(14):
        clock.t += 60
        d.step()
    assert len(ups) == 1
    clock.t = T0 + 15 * 60
    d.step()
    clock.t += 20
    d.step()
    assert [u["reasons"] for u in ups][-1] == ["heartbeat"]
    ticks.gate.set()
    settle(d)
    clock.t += 10
    d.step()                                         # reaped: tick_end waits 60 s
    assert len(ups) == 2
    clock.t += 60
    d.step()
    assert ups[-1]["reasons"] == ["tick_end"] and ups[-1]["daemon"]["state"] == "idle"
    assert ups[-1]["daemon"]["last_outcome"] == "ok"


def test_stall_is_published_once():
    ticks = Ticks(block=True)
    stalled = [False]
    d, clock, ups, _ = make(ticks=ticks, issue_fn=lambda: issue(17),
                            tick_status=lambda: {"stalled": stalled[0], "outcome": None})
    d.step()
    assert ticks.inside.wait(5)
    clock.t += 30
    d.step()
    stalled[0] = True
    for _ in range(4):
        clock.t += 61
        d.step()
    assert [u["reasons"] for u in ups].count(["stalled"]) == 1
    ticks.gate.set()
    settle(d)


def test_idle_daemon_refreshes_its_status_hourly():
    d, clock, ups, _ = make(issue_fn=lambda: issue(17), max_gap_s=10 * 3600)
    d.step()
    settle(d)
    clock.t += 30
    d.step()                                         # tick_start + tick_end
    clock.t += 60
    d.step()
    n = len(ups)
    for _ in range(59):
        clock.t += 60
        d.step()
    assert len(ups) == n
    clock.t += 120
    d.step()
    clock.t += 30
    d.step()
    assert ups[-1]["reasons"] == ["alive"] and len(ups) == n + 1


# ---- the HF-runner guard -------------------------------------------------------------
class Resp:
    def __init__(self, body, status=200):
        self.body, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

    def json(self):
        return self.body


def _get(body, status=200):
    seen = []

    def get(url, timeout=None):
        seen.append((url, timeout))
        if isinstance(body, Exception):
            raise body
        return Resp(body, status)
    get.seen = seen
    return get


def test_runner_guard_rules():
    now = T0
    recent = etd._iso(now - 30 * 60)
    old = etd._iso(now - 3 * 3600)
    url = "https://runner.invalid"
    g = _get({"tick": {"running": False, "last_attempt": None}})
    assert etd.runner_guard(url, get=g, now=now) is None
    assert g.seen == [(f"{url}/api/runner_status", (10, 20))]
    assert etd.runner_guard(url, get=_get({"tick_source": "remote",
                                           "tick": {"running": True}}), now=now) is None
    assert "running its own tick" in etd.runner_guard(
        url, get=_get({"tick": {"running": True, "phase": "run:1"}}), now=now)
    assert "EVENT_TICK=0" in etd.runner_guard(
        url, get=_get({"tick_source": "local", "tick": {"last_attempt": recent}}), now=now)
    assert etd.runner_guard(url, get=_get({"tick": {"last_attempt": old}}), now=now) is None
    # a runner with EVENT_TICK=1 that just restarted (empty memory) still ticks
    # on the next updater ping: tick_source "local" blocks on its own
    assert "tick_source local" in etd.runner_guard(
        url, get=_get({"tick_source": "local",
                       "tick": {"running": False, "last_attempt": None}}), now=now)
    logs = []
    assert etd.runner_guard(url, get=_get(OSError("down")), now=now, log=logs.append) is None
    assert etd.runner_guard(url, get=_get({}, status=503), now=now, log=logs.append) is None
    assert len(logs) == 2 and "not blocking" in logs[0]
    assert etd.runner_guard("", get=_get(OSError("never called"))) is None


def test_guard_blocks_ticks_with_one_commit_until_clear():
    reason, asked = ["HF ticked 0.5 h ago"], []

    def guard(sticky):
        asked.append(sticky)
        return reason[0]
    d, clock, ups, logs = make(issue_fn=lambda: issue(17), guard=guard)
    for i in range(5):
        reason[0] = f"HF ticked {0.5 + i / 60:.2f} h ago"    # the text changes
        d.step()
        clock.t += 61
    assert d.tick_fn.calls == 0 and d.blocked
    assert [u["reasons"] for u in ups] == [["blocked"]]
    assert ups[0]["daemon"]["blocked"].startswith("HF ticked")
    assert asked == [False, True, True, True, True]          # sticky once it blocked
    reason[0] = None
    d.step()
    assert d.tick_fn.calls == 1 and d.blocked is None
    settle(d)
    assert any("guard clear" in s for s in logs)


def test_runner_guard_unreadable_is_clear_only_before_a_block():
    url, logs = "https://runner.invalid", []
    r = etd.runner_guard(url, get=_get(OSError("restarting")), now=T0, log=logs.append,
                         sticky=True)
    assert "unreadable (OSError)" in r and "one clear read" in r and logs == []
    assert etd.runner_guard(url, get=_get(["not", "an", "object"]), now=T0, sticky=True)
    assert etd.runner_guard(url, get=_get({}, status=502), now=T0, sticky=True)
    assert etd.runner_guard(url, get=_get({"tick_source": "remote"}), now=T0,
                            sticky=True) is None              # the clear read
    assert etd.runner_guard(url, get=_get(["x"]), now=T0, log=logs.append) is None
    assert len(logs) == 1 and "not blocking" in logs[0]


def test_a_guard_block_is_sticky_until_one_clear_read(tmp_path):
    state = str(tmp_path / "event_tick_daemon.json")
    reply, clock = [{"tick": {"last_attempt": etd._iso(T0 - 1800)}}], Clock()

    def get(url, timeout=None):
        if isinstance(reply[0], Exception):
            raise reply[0]
        return Resp(reply[0])

    def guard(sticky):
        return etd.runner_guard("https://runner.invalid", get=get, now=clock.t,
                                log=lambda s: None, sticky=sticky)
    d, _, ups, logs = make(clock=clock, issue_fn=lambda: issue(17), guard=guard,
                           state_path=state)
    d.step()
    assert d.tick_fn.calls == 0 and "ticked 0.5 h ago" in d.blocked
    reply[0] = OSError("restarting")       # the Space restarts mid-ticking: still blocked
    for _ in range(3):
        clock.t += 61
        d.step()
    assert d.tick_fn.calls == 0 and "one clear read" in d.blocked
    assert [u["reasons"] for u in ups] == [["blocked"]]       # one commit per blocked spell
    assert "ticked 0.5 h ago" in json.loads(open(state).read())["guard_blocked"]
    # a restart keeps the block (the state file): unreachable is still not clear
    d2, _, _, _ = make(clock=clock, issue_fn=lambda: issue(17), guard=guard, state_path=state)
    assert d2.blocked
    d2.step()
    assert d2.tick_fn.calls == 0
    reply[0] = {"tick_source": "remote"}   # one clear read lifts it
    clock.t += 61
    d2.step()
    assert d2.tick_fn.calls == 1 and d2.blocked is None
    settle(d2)
    assert json.loads(open(state).read())["guard_blocked"] is None
    # a guard that never blocked: unreachable = clear
    reply[0] = OSError("down")
    d3, _, _, _ = make(clock=clock, issue_fn=lambda: issue(17), guard=guard,
                       state_path=str(tmp_path / "other.json"))
    d3.step()
    assert d3.tick_fn.calls == 1 and d3.blocked is None
    settle(d3)


def test_a_saved_block_without_a_guard_is_lifted(tmp_path):
    state = tmp_path / "event_tick_daemon.json"
    state.write_text(json.dumps({"guard_blocked": "the HF runner ticked 0.5 h ago"}))
    d, _, _, logs = make(issue_fn=lambda: issue(17), state_path=str(state))
    assert d.blocked
    d.step()
    assert d.tick_fn.calls == 1 and d.blocked is None
    assert any("no guard configured" in s for s in logs)
    settle(d)


# ---- stop ------------------------------------------------------------------------------
def _run(d, stop, hurry=None):
    out = {}
    th = threading.Thread(target=lambda: out.setdefault("rc", d.run(stop, hurry)))
    th.start()
    return th, out


def real(ticks, logs=None, **kw):
    logs = [] if logs is None else logs
    ups = []
    kw.setdefault("note_attempt", lambda: None)
    kw.setdefault("notify", lambda msg: False)
    d = etd.Daemon(tick=ticks, issue=lambda: issue(17), clock=time.time, log=logs.append,
                   tick_status=lambda: {"outcome": "ok", "phase": "run:09353800"},
                   loop_s=0.01, poll_s=0, **kw)
    d.publisher = etd.Publisher(lambda r: {"reasons": list(r), "daemon": d.info()},
                                ups.append, coalesce_s=0, min_commit_s=0, log=logs.append)
    return d, ups, logs


def test_stop_between_ticks_exits_at_once_and_publishes_stopped():
    d, ups, _ = real(Ticks(), grace_s=30)
    d.last_start, d.last_issue = time.time(), issue(17)      # nothing due
    stop = threading.Event()
    th, out = _run(d, stop)
    time.sleep(0.1)
    t = time.time()
    stop.set()
    th.join(5)
    assert out["rc"] == 0 and time.time() - t < 2 and d.tick_fn.calls == 0
    assert "stop" in ups[-1]["reasons"] and ups[-1]["daemon"]["state"] == "stopped"


def test_stop_during_a_tick_waits_for_it():
    ticks, logs = Ticks(block=True), []
    d, ups, _ = real(ticks, logs, grace_s=30)
    stop = threading.Event()
    th, out = _run(d, stop)
    assert ticks.inside.wait(5)
    stop.set()
    time.sleep(0.2)
    assert th.is_alive()                              # the tick is given its grace
    ticks.gate.set()
    th.join(5)
    assert out["rc"] == 0 and d.last_outcome == "ok" and d.state == "stopped"
    assert any("stop requested during tick #1 (phase run:09353800)" in s for s in logs)
    assert not any("grace over" in s for s in logs)
    assert ups[-1]["daemon"]["state"] == "stopped" and "stop" in ups[-1]["reasons"]


def test_grace_is_bounded():
    ticks, logs = Ticks(block=True), []
    d, ups, _ = real(ticks, logs, grace_s=0.3)
    stop = threading.Event()
    th, out = _run(d, stop)
    assert ticks.inside.wait(5)
    t = time.time()
    stop.set()
    th.join(5)
    took = time.time() - t
    ticks.gate.set()
    assert out["rc"] == 0 and 0.25 < took < 3
    assert any("grace over" in s for s in logs) and ups[-1]["daemon"]["state"] == "stopped"


def test_second_signal_skips_the_grace():
    ticks = Ticks(block=True)
    d, _, _ = real(ticks, grace_s=60)
    stop, hurry = threading.Event(), threading.Event()
    prev = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        etd.install_signals(stop, hurry)
        handler = signal.getsignal(signal.SIGTERM)
        assert signal.getsignal(signal.SIGINT) is handler
    finally:
        for s, h in prev.items():
            signal.signal(s, h)
    th, out = _run(d, stop, hurry)
    assert ticks.inside.wait(5)
    t = time.time()
    handler(signal.SIGTERM, None)                     # first: stop after the tick
    assert stop.is_set() and not hurry.is_set()
    time.sleep(0.2)
    assert th.is_alive()
    handler(signal.SIGTERM, None)                     # second: skip the grace
    th.join(5)
    ticks.gate.set()
    assert hurry.is_set() and out["rc"] == 0 and time.time() - t < 3


def test_shutdown_waits_for_finalize_threads(monkeypatch):
    gate, done = threading.Event(), []

    def finalize(ids, log=print):
        gate.wait(5)
        done.extend(ids)

    monkeypatch.setattr(eventstore, "finalize_episodes", finalize)
    th = threading.Thread(target=eventstore.finalize_episodes, args=(["e1"],), daemon=True)
    th.start()
    assert etd.finalize_threads() == [th]
    d, ups, logs = real(Ticks(), grace_s=5)
    threading.Timer(0.2, gate.set).start()
    assert d.shutdown() == 0
    assert done == ["e1"] and any("episode finalize" in s for s in logs)


def test_no_tick_starts_after_a_stop_request():
    stop = threading.Event()

    def guard(sticky):
        stop.set()                                    # SIGTERM while the guard read runs
        return None
    d, clock, _, logs = make(issue_fn=lambda: issue(17), guard=guard, stop=stop)
    d.step()
    assert d.tick_fn.calls == 0 and not d.ticking() and d.last_start is None
    assert any("NOT started: stop requested" in s for s in logs)
    reads = []
    d.issue_fn = lambda: reads.append(1) or issue(18)
    clock.t += 3600
    d.step()                                          # a pass after the signal does not poll
    assert d.tick_fn.calls == 0 and reads == []


def test_sigterm_in_the_middle_of_a_pass_exits_without_a_tick():
    stop = threading.Event()

    def guard(sticky):
        stop.set()
        return None
    d, ups, _ = real(Ticks(), grace_s=30, guard=guard)
    th, out = _run(d, stop)
    th.join(5)
    assert out["rc"] == 0 and d.tick_fn.calls == 0 and d.ticks == 0
    assert ups[-1]["daemon"]["state"] == "stopped" and "stop" in ups[-1]["reasons"]


def test_grace_counts_from_the_stop_request():
    ticks, logs = Ticks(block=True), []
    stop = etd.StopEvent()
    d, _, _ = real(ticks, logs, grace_s=0.5, stop=stop)
    d.step()
    assert ticks.inside.wait(5)
    stop.set()
    time.sleep(0.6)                                   # the pass in flight outlasted the grace
    t = time.time()
    assert d.shutdown() == 0 and time.time() - t < 0.4
    assert any("grace over" in s for s in logs)
    ticks.gate.set()


def test_stop_grace_fits_timeout_stop_sec():
    logs = []
    assert etd.stop_grace(600, 720, log=logs.append) == 600 and logs == []
    assert etd.stop_grace(1800, 720, log=logs.append) == 600 and "cut to 600 s" in logs[0]
    assert etd.stop_grace(600, 100, log=logs.append) == 0
    here = os.path.join(etd.ROOT, "fleet", "jetstream2")
    unit = open(os.path.join(here, "crest-events.service")).read()
    timeout = re.search(r"^TimeoutStopSec=(\d+)$", unit, re.M).group(1)
    assert f"Environment=EVENT_TICK_TIMEOUT_STOP_S={timeout}" in unit
    env = open(os.path.join(here, "events.env.example")).read()
    grace = float(re.search(r"^EVENT_TICK_STOP_GRACE_S=(\d+)$", env, re.M).group(1))
    assert grace + etd.FINAL_PUBLISH_S + etd.STOP_SLACK_S <= float(timeout)
    assert int(re.search(r"^WatchdogSec=(\d+)$", unit, re.M).group(1)) > 1800
    assert "NotifyAccess=main" in unit                # Type=simple ignores pings without it


def _final(upload, **kw):
    logs = []
    d, _, _ = real(Ticks(), logs, grace_s=5, **kw)
    d.publisher = etd.Publisher(lambda r: {"reasons": list(r), "daemon": d.info()}, upload,
                                coalesce_s=0, min_commit_s=0, log=logs.append)
    return d, logs


def test_the_final_status_commit_retries_once():
    ups, fails = [], [1]

    def upload(doc):
        if fails[0]:
            fails[0] -= 1
            raise OSError("503")
        ups.append(doc)
    d, logs = _final(upload, retry_pause_s=0.05)
    assert d.shutdown() == 0
    assert len(ups) == 1 and ups[0]["daemon"]["state"] == "stopped" and "stop" in ups[0]["reasons"]
    assert any("one retry" in s for s in logs)
    calls = []

    def down(doc):
        calls.append(1)
        raise OSError("503")
    d, logs = _final(down, retry_pause_s=0.05)
    t = time.time()
    assert d.shutdown() == 0 and time.time() - t < 2
    assert len(calls) == 2 and "exiting without it" in logs[-2]


def test_a_hung_final_commit_is_bounded():
    gate = threading.Event()
    d, logs = _final(lambda doc: gate.wait(10), final_s=0.3)
    t = time.time()
    assert d.shutdown() == 0 and time.time() - t < 1.5
    assert any("still running after 0.3 s" in s for s in logs)
    # request() never waits for the hung commit
    th = threading.Thread(target=d.publisher.request, args=("late",), daemon=True)
    th.start()
    th.join(1)
    assert not th.is_alive()
    gate.set()


@pytest.mark.skipif(os.name == "nt" or not hasattr(socket, "AF_UNIX"),
                    reason="unix datagram sockets")
def test_sd_notify_sends_one_datagram(tmp_path, monkeypatch):
    path = str(tmp_path / "notify")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        srv.bind(path)
        monkeypatch.setenv("NOTIFY_SOCKET", path)
        assert etd.sd_notify("WATCHDOG=1") is True
        assert srv.recv(64) == b"WATCHDOG=1"
    finally:
        srv.close()


def test_sd_notify_outside_systemd(monkeypatch, tmp_path):
    monkeypatch.delenv("NOTIFY_SOCKET", raising=False)
    assert etd.sd_notify("WATCHDOG=1") is False
    monkeypatch.setenv("NOTIFY_SOCKET", str(tmp_path / "missing"))
    assert etd.sd_notify("WATCHDOG=1") is False


# ---- janitor / host lock ------------------------------------------------------------------
def test_janitor_bounds_caches_and_drops_old_event_workdirs(tmp_path, monkeypatch):
    from hf_data import datamgr
    monkeypatch.setattr(datamgr, "cleanup", lambda: {"workdirs": {"removed": 0}})
    monkeypatch.setattr(datamgr, "compact_results", lambda: {"records": 0})
    monkeypatch.setattr(etd.tempfile, "gettempdir", lambda: str(tmp_path))
    old, young, other = (tmp_path / "event_1_a", tmp_path / "event_2_b",
                         tmp_path / "keep_me")
    for p in (old, young, other):
        p.mkdir()
        (p / "f.tif").write_text("x")
    past = time.time() - 7 * 3600
    for p in (old, other):
        os.utime(p, (past, past))
    rep = etd.janitor(log=lambda s: None, ttl_h=6)
    assert rep["event_workdirs"] == 1 and rep["compact"] == {"records": 0}
    assert not old.exists() and young.exists() and other.exists()


def test_janitor_runs_after_a_tick_between_ticks():
    runs = []
    d, clock, _, _ = make(issue_fn=lambda: issue(17), janitor=lambda: runs.append(1))
    d.step()
    assert runs == []
    settle(d)
    clock.t += 30
    d.step()
    assert runs == [1]
    clock.t += 61
    d.step()
    assert runs == [1]


def test_one_daemon_per_state_dir(tmp_path):
    path = str(tmp_path / "event_tick_daemon.lock")
    first = etd.acquire_host_lock(path)
    try:
        assert first is not None
        result = {}
        th = threading.Thread(target=lambda: result.setdefault("lock", etd.acquire_host_lock(path)))
        th.start()
        th.join(5)
        assert result["lock"] is None
    finally:
        first.release()


def test_preflight_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "crestimap", SimpleNamespace())
    bad = etd.preflight()
    assert len(bad) == 1 and "EF5/bin/ef5" in bad[0]


def test_main_refuses_without_a_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setitem(sys.modules, "truststore",
                        SimpleNamespace(inject_into_ssl=lambda: None))
    assert etd.main([]) == 2
    assert "HF_TOKEN is not set" in capsys.readouterr().out


def test_status_document_shape_and_commit(monkeypatch):
    from hf_data import provenance, tickstatus
    monkeypatch.setattr(provenance, "stamp", lambda: {"engine": "js2-events", "crest_ai": "abc"})
    monkeypatch.setenv("EVENT_ENGINE_IDENT", "js2-events")
    d, _, _, _ = make(issue_fn=lambda: issue(17))
    doc = d.snapshot(["start", "tick_start"])
    assert doc["schema"] == tickstatus.SCHEMA and doc["host"] == "js2-events"
    assert doc["reasons"] == ["start", "tick_start"] and doc["daemon"]["state"] == "idle"
    assert doc["tick"].keys() >= {"running", "phase", "last_attempt", "last_success", "stalled"}
    assert "tick" not in doc["runner"] and "capacity_skips" not in doc["runner"]
    assert doc["code"]["crest_ai"] == "abc"
    calls = []

    class Api:
        def __init__(self, token=None):
            calls.append({"token_set": bool(token)})

        def upload_file(self, **kw):
            calls.append(kw)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setenv("HF_TOKEN", "hf_test_not_real")
    tickstatus.publish(doc)
    up = calls[1]
    assert calls[0] == {"token_set": True}
    assert up["path_in_repo"] == "events/tick_status.json" and up["repo_type"] == "dataset"
    assert up["repo_id"] == eventstore.REPO
    assert up["commit_message"] == "tick status: start, tick_start (js2-events)"
    assert json.loads(up["path_or_fileobj"])["daemon"]["state"] == "idle"
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(RuntimeError):
        tickstatus.publish(doc)

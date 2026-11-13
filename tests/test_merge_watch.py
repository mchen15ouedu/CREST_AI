"""scripts/merge_episodes.py --watch (the always-on episode merger) and its
supervisor hf_data/eventmerger.py, on fakes: a fake clock, index revision
source and planner for the Watcher; stubbed plan/build/commit steps for
run_pass; a fake Popen for the supervisor. No network, no real process.
The Hub timeout tests follow the installed huggingface_hub: 1.x (this env),
or 0.x as the live Spaces run it (tokenizers==0.22.0 needs
huggingface-hub<1.0): run this file with a 0.36 package first on sys.path
for the real one; under 1.x the 0.x test runs on a stand-in."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from hf_data import eventmerger
from scripts import merge_episodes as me

T0 = dt.datetime(2026, 11, 3, tzinfo=dt.timezone.utc).timestamp()


class Clock:
    """Fake epoch clock; sleep() advances it."""

    def __init__(self, t=T0):
        self.t = float(t)

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _report(**kw):
    rep = {"index": 10, "groups": 2, "planned": 0, "merged": [], "failed": [],
           "skipped": [], "already": 0, "finalized": [], "stopped": False}
    rep.update(kw)
    return rep


class Planner:
    """run_pass stand-in: the next canned report, and merges from `todo` while
    the budget lasts (the rest skip "rate cap", as run_pass does)."""

    def __init__(self, reports=()):
        self.reports, self.todo, self.calls = list(reports), [], []

    def __call__(self, budget, stop, memo, now, on_merge, recent=(), beat=None):
        self.calls.append({"budget": budget, "now": now, "memo": memo, "recent": recent})
        rep = _report(**(self.reports.pop(0) if self.reports else {}))
        while self.todo and len(rep["merged"]) < budget:
            t = self.todo.pop(0)
            rec = {"gauge": "G", "target": t, "members": [t, t + "_old"], "commit": "c_" + t}
            rep["merged"].append(rec)
            rep["planned"] += 1
            on_merge(rec)
        rep["skipped"] += [{"gauge": "G", "members": [t, t + "_old"], "wait": True,
                            "why": "rate cap (--max-merges-per-hour) reached"}
                           for t in self.todo]
        return rep


def _watcher(tmp_path, rev, run, clock, **kw):
    logs = []
    w = me.Watcher(rev, run, str(tmp_path / "st" / "merger_status.json"), clock=clock,
                   sleep=clock.sleep, log=logs.append, **kw)
    return w, logs


def _iso(t):
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---- Watcher ---------------------------------------------------------------------

def test_watch_passes_when_the_index_changes(tmp_path):
    clock, plan, rev = Clock(), Planner(), {"v": "aaa111"}
    w, logs = _watcher(tmp_path, lambda: rev["v"], plan, clock)
    assert w.step() == 300 and len(plan.calls) == 1             # first look: a pass
    for _ in range(3):
        clock.sleep(300)
        assert w.step() == 300
    assert len(plan.calls) == 1                                  # unchanged: no pass
    rev["v"] = "bbb222"
    clock.sleep(300)
    w.step()
    assert len(plan.calls) == 2
    st = me.read_status(w.path)
    assert st["last_pass"]["trigger"] == "index changed"
    assert st["last_pass"]["index_rev"] == "bbb222" and st["passes"] == 2
    passes = [ln for ln in logs if " merge pass " in ln]
    assert len(passes) == 2 == len(logs)                         # one line per pass, idle is quiet
    assert re.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ merge pass \(index changed\) "
                    r"index bbb222: ", passes[-1])
    assert passes[0].startswith("2026-11-03T00:00:00Z merge pass (start)")


def test_watch_retries_waiting_groups_after_retry_min(tmp_path):
    clock = Clock()
    held = {"skipped": [{"gauge": "G", "members": ["2026110100_G"], "wait": True,
                         "why": "not ended: 2026110100_G; re-run after the episode ends"},
                        {"gauge": "H", "members": ["2026100100_H", "2026093023_H"],
                         "wait": False, "why": "pinned: 2026100100_H"}],
            "failed": [{"gauge": "K", "target": "2026110200_K",
                        "members": ["2026110200_K", "2026110122_K"],
                        "why": "COMMIT FAILED for 2026110200_K (HfHubHTTPError: 500)"}]}
    plan = Planner([held, held, {}])
    w, _ = _watcher(tmp_path, lambda: "aaa", plan, clock, retry_min=60)
    first = clock()
    w.step()
    st = me.read_status(w.path)
    # a pinned group never clears by itself: not waiting; a failed merge is retried
    assert [x["members"][0] for x in st["waiting"]] == ["2026110100_G", "2026110200_K"]
    assert st["retry_at"] == _iso(first + 3600)
    for _ in range(11):                                          # 55 min: nothing changed
        clock.sleep(300)
        w.step()
    assert len(plan.calls) == 1
    clock.sleep(300)                                             # 60 min
    w.step()
    st = me.read_status(w.path)
    assert len(plan.calls) == 2 and st["last_pass"]["trigger"] == "retry"
    assert st["waiting"][0]["since"] == _iso(first)              # waiting since the first pass
    clock.sleep(3600)
    w.step()                                                     # this pass clears them
    assert len(plan.calls) == 3 and me.read_status(w.path)["waiting"] == []
    clock.sleep(3600)
    w.step()
    assert len(plan.calls) == 3                                  # nothing waits: no retry
    # a retry sooner than the interval shortens the sleep
    w2, _ = _watcher(tmp_path / "b", lambda: "x", Planner([held]), Clock(), retry_min=2)
    assert w2.step() == 120


def test_watch_rate_cap_spans_passes_and_restarts(tmp_path):
    clock, plan = Clock(), Planner()
    plan.todo = ["t1", "t2", "t3", "t4", "t5"]
    revs = iter(f"r{i}" for i in range(100))                     # every look: changed
    w, logs = _watcher(tmp_path, lambda: next(revs), plan, clock, max_per_hour=2)
    w.step()
    assert plan.calls[0]["budget"] == 2 and plan.todo == ["t3", "t4", "t5"]
    clock.sleep(300)
    w.step()                                                     # our commits moved the index
    assert plan.calls[1]["budget"] == 0 and plan.todo == ["t3", "t4", "t5"]
    st = me.read_status(w.path)
    assert st["merges_last_hour"] == 2 and "2/2 merges in the last hour" in logs[-1]
    assert [r["target"] for r in st["recent_merges"]] == ["t1", "t2"]
    assert st["recent_merges"][0] == {"target": "t1", "members": ["t1", "t1_old"],
                                      "at": _iso(T0), "commit": "c_t1"}
    assert len(st["waiting"]) == 3                               # rate-capped groups wait
    # a restart (new process, same status file) still counts the hour's merges
    w2, _ = _watcher(tmp_path, lambda: next(revs), plan, clock, max_per_hour=2)
    clock.sleep(300)
    w2.step()
    assert plan.calls[2]["budget"] == 0
    clock.sleep(3000)                                            # t1, t2 now > 1 h old
    w2.step()
    assert plan.calls[3]["budget"] == 2 and plan.todo == ["t5"]
    # the status lists the last 20 merges; the window counts all of them
    many = Planner()
    many.todo = [f"m{i:02d}" for i in range(25)]
    w3, _ = _watcher(tmp_path / "c", lambda: "x", many, Clock(), max_per_hour=30)
    w3.step()
    st = me.read_status(w3.path)
    assert len(st["recent_merges"]) == 20 and st["recent_merges"][-1]["target"] == "m24"
    assert st["merges_last_hour"] == 25 == w3.in_hour(w3.clock())


def test_watch_backoff_on_errors_never_kills_the_loop(tmp_path):
    clock, plan, down = Clock(), Planner(), {"n": 9}

    def rev():
        if down["n"]:
            down["n"] -= 1
            raise OSError("hub unreachable")
        return "aaa"

    w, logs = _watcher(tmp_path, rev, plan, clock)
    waits = []
    for _ in range(9):
        waits.append(w.step())
        clock.sleep(waits[-1])
    assert waits == [60, 120, 240, 480, 960, 1920, 3600, 3600, 3600]
    st = me.read_status(w.path)
    assert st["errors"] == 9 and st["state"] == "backoff"
    assert "OSError: hub unreachable" in st["last_error"]["error"]
    assert "merge ERROR OSError: hub unreachable -> retry in 60 min" in logs[-1]
    assert w.step() == 300 and len(plan.calls) == 1             # recovered: reset
    assert w.backoff == 0 and me.read_status(w.path)["state"] == "idle"
    # a pass that raises is re-run after the backoff (its revision is not consumed)
    boom = {"n": 1}

    def run(**kw):
        if boom["n"]:
            boom["n"] -= 1
            raise me.Unreadable("events/index.json unreadable (or empty)")
        return plan(**kw)

    w2, _ = _watcher(tmp_path / "b", lambda: "aaa", run, clock)
    assert w2.step() == 60
    clock.sleep(60)
    assert w2.step() == 300 and len(plan.calls) == 2


def test_status_file_content(tmp_path):
    clock = Clock()
    plan = Planner([{"skipped": [{"gauge": "H", "members": ["b", "a"], "wait": False,
                                  "why": "grid mismatch: b 6x5"}],
                     "failed": [{"gauge": "K", "target": "k2", "members": ["k2", "k1"],
                                 "why": "RuntimeError: boom"}],
                     "unfinalized": [{"gauge": "M", "members": ["m2", "m1"],
                                      "why": "finalize of m2 not done (try 1)"}],
                     "log": [f"line {i}" for i in range(100)]}])
    plan.todo = ["t1"]
    w, _ = _watcher(tmp_path, lambda: "rev123", plan, clock, interval=120,
                    config={"repo": "vincewin/CREST_data", "gap_h": 12.0})
    w.step()
    st = me.read_status(w.path)
    for k in ("started", "pid", "state", "config", "passes", "errors", "last_pass",
              "waiting", "last_error", "next_pass_at", "retry_at", "merges_last_hour",
              "recent_merges", "updated"):
        assert k in st, k
    assert st["pid"] == os.getpid() and st["started"] == _iso(T0)
    assert st["config"]["repo"] == "vincewin/CREST_data" and st["config"]["interval_s"] == 120
    assert st["config"]["max_merges_per_hour"] == 6 and st["config"]["retry_min"] == 60
    lp = st["last_pass"]
    assert lp["at"] == _iso(T0) and lp["index_rev"] == "rev123" and lp["trigger"] == "start"
    assert lp["planned"] == 1 and lp["merged"] == 1 and lp["merged_targets"] == ["t1"]
    assert lp["skipped"] == [{"gauge": "H", "members": ["b", "a"], "why": "grid mismatch: b 6x5",
                              "wait": False}]
    assert lp["failed"][0]["why"] == "RuntimeError: boom"
    assert len(lp["log"]) == me.LOG_KEEP and lp["log"][-1] == "line 99"
    assert st["waiting"] == [{"gauge": "K", "members": ["k2", "k1"],
                              "why": "RuntimeError: boom", "since": _iso(T0)},
                             {"gauge": "M", "members": ["m2", "m1"],
                              "why": "finalize of m2 not done (try 1)", "since": _iso(T0)}]
    assert lp["unfinalized"][0]["members"] == ["m2", "m1"]
    assert st["next_pass_at"] == _iso(T0 + 120) and st["last_error"] is None
    assert os.listdir(os.path.dirname(w.path)) == ["merger_status.json"]   # no temp left
    json.dumps(st)


def test_watch_stops_cleanly(tmp_path):
    clock, stop, calls = Clock(), me.Stop(), []

    def run(budget, stop, memo, now, on_merge, **kw):
        calls.append(now)
        stop.set(signal.SIGTERM, None)       # SIGTERM lands during the pass
        return _report(stopped=True)

    w, logs = _watcher(tmp_path, lambda: "aaa", run, clock, stop=stop)
    assert w.run_forever() == 0
    assert len(calls) == 1 and clock() == T0                     # no sleep after the stop
    st = me.read_status(w.path)
    assert st["state"] == "stopped" and st["next_pass_at"] is None
    assert logs[-1].endswith("merger stopped") and "STOPPED" in logs[-2]


# ---- run_pass: rate cap, stop, memo, failures --------------------------------------

def _h(d):
    return d.strftime("%Y-%m-%dT%H:%MZ")


def _ep(eid, **kw):
    g = eid.split("_", 1)[1]
    t = dt.datetime.strptime(eid[:10], "%Y%m%d%H")
    s = {"t0": _h(t), "t_end": _h(t + dt.timedelta(hours=2)), "episode_started": _h(t),
         "status": "ended", "gauge": g, "trigger": {"gauge": g},
         "generated": _h(t + dt.timedelta(hours=1))}
    s.update(kw)
    return s


A1, A0, B1, B0 = "2026102803_G1", "2026102801_G1", "2026102703_G2", "2026102701_G2"


@pytest.fixture
def fake(monkeypatch):
    """Two ended two-episode floods (targets A1, B1); plan/build/commit stubbed."""
    idx = {e: _ep(e) for e in (A1, A0, B1, B0)}
    calls = {"plan": [], "build": [], "commit": [], "final": [], "beat": [], "drop": []}
    ctl = SimpleNamespace(skip={}, build_hook=None, commit_hook=None, commit_ok=True)
    monkeypatch.setattr(me.eventstore, "load_index", lambda: idx)
    monkeypatch.setattr(me.eventstore, "pinned_ids", lambda force=False: frozenset())

    def plan_group(store, g, idx_):
        t = max(g["members"])
        calls["plan"].append(t)
        if t in ctl.skip:
            raise ctl.skip[t]
        return {"target": t, "members": sorted(g["members"], reverse=True)}

    def build_products(store, plan, work, idx_, beat=None):
        assert os.path.isdir(work)
        calls["beat"].append(beat)
        calls["build"].append(work)
        if ctl.build_hook:
            ctl.build_hook(plan)
        return {}

    def commit_group(store, plan, prod, gap_h, log=print):
        calls["commit"].append(plan["target"])
        if ctl.commit_hook:
            ctl.commit_hook(plan)
        if not ctl.commit_ok:
            log(f"   COMMIT FAILED for {plan['target']} (HfHubHTTPError: 500)")
            return False
        return "oid_" + plan["target"]

    def finalize(ids, log=print):
        calls["final"].extend(ids)
        return len(ids)

    def drop_final(store, t, log=print):
        calls["drop"].append(t)
        idx[t].pop("final", None)
        return True

    monkeypatch.setattr(me, "plan_group", plan_group)
    monkeypatch.setattr(me, "drop_final", drop_final)
    monkeypatch.setattr(me, "build_products", build_products)
    monkeypatch.setattr(me, "commit_group", commit_group)
    monkeypatch.setattr(me, "describe", lambda plan, prod, merged: [f"   plan {plan['target']}"])
    monkeypatch.setattr(me.eventstore, "finalize_episodes", finalize)
    store = SimpleNamespace(queued=lambda: set(), api=None)
    return SimpleNamespace(idx=idx, calls=calls, ctl=ctl, store=store)


def _pass(fake, **kw):
    kw.setdefault("apply", True)
    return me.run_pass(fake.store, gap_h=12, log=lambda s: None, **kw)


def test_run_pass_merges_and_reports(fake):
    merged = []
    rep = _pass(fake, on_merge=merged.append)
    assert [(r["target"], r["members"], r["commit"]) for r in rep["merged"]] == [
        (A1, [A1, A0], "oid_" + A1), (B1, [B1, B0], "oid_" + B1)]
    assert merged == rep["merged"] and rep["finalized"] == [A1, B1]
    assert rep["planned"] == 2 and rep["groups"] == 2 and not rep["failed"] and not rep["skipped"]
    assert all(not os.path.exists(w) for w in fake.calls["build"])   # work dirs removed
    dry = _pass(fake, apply=False)
    assert dry["planned"] == 2 and dry["merged"] == [] and fake.calls["commit"] == [A1, B1]


def test_run_pass_rate_cap_defers_without_planning(fake):
    rep = _pass(fake, budget=1)
    assert [r["target"] for r in rep["merged"]] == [A1] and fake.calls["plan"] == [A1]
    (s,) = rep["skipped"]
    assert s["members"] == [B1, B0] and s["why"].startswith("rate cap") and s["wait"]
    assert _pass(fake, budget=0)["merged"] == [] and fake.calls["plan"] == [A1]


def test_run_pass_stops_between_groups_never_mid_commit(fake):
    stop = me.Stop()
    fake.ctl.commit_hook = lambda plan: stop.set(signal.SIGTERM, None)
    rep = _pass(fake, stop=stop)
    # the commit in flight lands (and is finalized); the next group is not touched
    assert fake.calls["commit"] == [A1] and [r["target"] for r in rep["merged"]] == [A1]
    assert fake.calls["final"] == [A1] and fake.calls["plan"] == [A1] and rep["stopped"]
    # a stop while a group's products are built: that group commits nothing
    stop2 = me.Stop()
    fake.ctl.commit_hook, fake.ctl.build_hook = None, lambda plan: stop2.set()
    rep = _pass(fake, stop=stop2)
    assert fake.calls["commit"] == [A1] and rep["merged"] == [] and rep["stopped"]
    assert all(not os.path.exists(w) for w in fake.calls["build"])


def test_run_pass_failures_are_reported_not_raised(fake):
    fake.ctl.commit_ok = False
    rep = _pass(fake)
    assert [f["target"] for f in rep["failed"]] == [A1, B1] and rep["merged"] == []
    assert rep["failed"][0]["why"].startswith("COMMIT FAILED for " + A1)
    assert fake.calls["final"] == []
    fake.ctl.commit_ok = True
    fake.ctl.skip[A1] = RuntimeError("tree listing timed out")
    rep = _pass(fake)
    assert rep["failed"] == [{"gauge": "G1", "target": A1, "members": [A1, A0],
                              "why": "RuntimeError: tree listing timed out"}]
    assert [r["target"] for r in rep["merged"]] == [B1]          # the next group still runs
    fake.idx.clear()
    with pytest.raises(me.Unreadable):
        _pass(fake)


def test_run_pass_lasting_skips_are_not_replanned(fake):
    memo = {}
    fake.ctl.skip[A1] = me.Skip("grid mismatch: x")
    fake.ctl.skip[B1] = me.Skip(f"manifest unreadable: {B0}")
    rep = _pass(fake, memo=memo, now=T0 + 1000.0)
    assert [(s["members"][0], s["wait"]) for s in rep["skipped"]] == [(A1, False), (B1, True)]
    rep = _pass(fake, memo=memo, now=T0 + 2000.0)
    assert fake.calls["plan"] == [A1, B1, B1]                    # only the waiting one again
    assert rep["skipped"][0]["why"] == "grid mismatch: x" and not rep["skipped"][0]["wait"]
    _pass(fake, memo=memo, now=T0 + 1000.0 + me.SKIP_RECHECK_H * 3600)
    assert fake.calls["plan"][-2:] == [A1, B1]                   # re-checked after a day
    fake.idx[A0]["generated"] = "2026-11-03T05:00Z"              # a member re-published
    _pass(fake, memo=memo, now=T0 + 1000.0 + me.SKIP_RECHECK_H * 3600 + 10)
    assert fake.calls["plan"][-2:] == [A1, B1]


def test_run_pass_finalize_of_a_merged_target_is_retried_after_retry_s(fake, monkeypatch):
    fake.idx[A0]["merged_into"] = A1                             # merged before; no final yet
    memo = {}
    rep = _pass(fake, memo=memo, now=0.0, retry_s=3600, budget=0)
    assert rep["already"] == 1 and fake.calls["final"] == [A1] and rep["finalized"] == [A1]
    _pass(fake, memo=memo, now=600.0, retry_s=3600, budget=0)
    assert fake.calls["final"] == [A1]                           # not before retry_s
    _pass(fake, memo=memo, now=3600.0, retry_s=3600, budget=0)
    assert fake.calls["final"] == [A1, A1]
    # a finalize that keeps failing waits (the watcher retries it), FINAL_TRIES in all
    monkeypatch.setattr(me.eventstore, "finalize_episodes",
                        lambda ids, log=print: (fake.calls["final"].extend(ids), 0)[1])
    memo = {}
    reps = [_pass(fake, memo=memo, now=3600.0 * i, retry_s=3600, budget=0) for i in range(5)]
    assert fake.calls["final"] == [A1] * (2 + me.FINAL_TRIES)
    assert [len(r["unfinalized"]) for r in reps] == [1] * (me.FINAL_TRIES - 1) + [0] * (
        6 - me.FINAL_TRIES)
    assert reps[0]["unfinalized"] == [{"gauge": "G1", "members": [A1, A0],
                                       "why": f"finalize of {A1} not done (try 1)"}]


def test_watch_main_loop_end_to_end(fake, tmp_path, monkeypatch):
    """watch(): the real Watcher + run_pass; SIGTERM (its installed handler)
    during the first commit stops it after that group."""
    handlers = {}
    monkeypatch.setattr(me.signal, "signal", lambda sig, h: handlers.setdefault(sig, h))
    monkeypatch.setattr(me.os, "nice", lambda n: 0, raising=False)
    monkeypatch.setattr(me, "fallback_status_file", lambda: str(tmp_path / "fallback.json"))
    fake.ctl.commit_hook = lambda plan: handlers[signal.SIGTERM](signal.SIGTERM, None)
    fake.store.api = SimpleNamespace(get_paths_info=lambda repo, paths, repo_type=None: [
        SimpleNamespace(path=paths[0], blob_id="blob42")])
    path = str(tmp_path / "merger_status.json")
    a = argparse.Namespace(gauge=None, max_groups=0, no_finalize=False, retry_min=60.0,
                           interval=300.0, max_merges_per_hour=6, status_file=path)
    assert me.watch(fake.store, a, 12.0) == 0
    st = me.read_status(path)
    assert st["state"] == "stopped" and st["last_pass"]["index_rev"] == "blob42"
    assert [r["target"] for r in st["recent_merges"]] == [A1]
    assert st["recent_merges"][0]["commit"] == "oid_" + A1
    assert st["last_pass"]["stopped"] and fake.calls["commit"] == [A1]
    assert any("plan " + A1 in ln for ln in st["last_pass"]["log"])
    assert not os.path.exists(tmp_path / "fallback.json")       # the status file was writable
    assert me.main(["--watch"]) == 2                             # --watch merges: needs --apply


# ---- supervisor (hf_data/eventmerger.py) ----------------------------------------------

class FakeProc:
    """A child on the fake clock: exits with rc `ran` s after it started;
    SIGTERM ends it unless `deaf` (a main thread stuck in a socket read),
    SIGKILL always."""

    def __init__(self, pid, ran, rc, clock, deaf=False):
        self.pid, self.ran, self.rc, self.clock, self.deaf = pid, ran, rc, clock, deaf
        self.t0 = clock()
        self.returncode, self.terminated, self.killed = None, False, False

    def poll(self):
        if self.returncode is None and self.clock() - self.t0 >= self.ran:
            self.returncode = self.rc
        return self.returncode

    def wait(self, timeout=None):
        if self.poll() is None:
            left = self.t0 + self.ran - self.clock()
            if timeout is not None and left > timeout:
                self.clock.sleep(timeout)
                raise subprocess.TimeoutExpired("merge", timeout)
            self.clock.sleep(left)
        return self.poll()

    def terminate(self):
        self.terminated = True
        if not self.deaf and self.returncode is None:
            self.returncode = -15

    def kill(self):
        self.killed = True
        if self.returncode is None:
            self.returncode = -9


def test_supervisor_restarts_with_backoff():
    clock, procs, sleeps = Clock(0), [], []
    lives = [5, 5, 5, 5, 5, 5, 5, 2000, 5]          # 8th child ran > 30 min: healthy

    def popen(cmd, **kw):
        if len(procs) == 2:
            procs.append(None)
            raise OSError("fork failed")
        p = FakeProc(100 + len(procs), lives[len(procs)], 1, clock)
        procs.append((cmd, kw, p))
        return p

    sup = eventmerger.Supervisor(["python", "merge.py", "--watch"], popen=popen,
                                 sleep=lambda s: (sleeps.append(s), clock.sleep(s)),
                                 clock=clock, log=lambda s: None)
    sup.run(runs=9)
    # the child is polled every POLL_S; the rest are the restart backoffs
    assert [s for s in sleeps if s != eventmerger.POLL_S] == [
        60, 120, 240, 480, 960, 1800, 1800, 60, 120]
    info = sup.info()
    assert info["restarts"] == 8 and info["alive"] is False and info["pid"] == 108
    assert info["last_exit"]["rc"] == 1 and "hung" not in info["last_exit"]
    cmd, kw, _ = procs[0]
    assert cmd == ["python", "merge.py", "--watch"] and kw["cwd"] == eventmerger.ROOT
    assert kw["env"]["PYTHONUNBUFFERED"] == "1" and "stdout" not in kw and "stderr" not in kw
    assert procs[2] is None                         # a failed start backs off like an exit


def test_supervisor_stop_terminates_and_never_restarts():
    clock = Clock(0)
    sup = eventmerger.Supervisor(["x"], popen=lambda *a, **k: pytest.fail("restarted"),
                                 sleep=clock.sleep, clock=clock, log=lambda s: None)
    sup.proc = FakeProc(7, math.inf, 0, clock)
    assert sup.info()["alive"] is True and sup.info()["pid"] == 7
    sup.stop(wait_s=0.1)
    assert sup.proc.terminated and sup.info()["alive"] is False
    sup.run()                                       # stopping: returns at once


def test_supervisor_watchdog_restarts_a_hung_child(tmp_path):
    """A child whose status heartbeat stops (a Hub call that never returns)
    is terminated, killed when SIGTERM can't land, and restarted with the
    normal backoff; a child that keeps beating, or says it stopped, is not."""
    clock, path, logs, procs = Clock(T0), tmp_path / "merger_status.json", [], []
    lives = [math.inf, 3 * 3600, math.inf]

    def status(p, **kw):
        me.write_status(str(path), {"pid": p.pid, "state": "idle", "updated": _iso(clock()),
                                    "config": {"interval_s": 300.0}, **kw})

    def popen(cmd, **kw):
        p = FakeProc(200 + len(procs), lives[len(procs)], 0, clock, deaf=not procs)
        procs.append(p)
        status(p)                                   # the child's first status write
        return p

    def sleep(s):
        clock.sleep(s)
        p = procs[-1]
        if p.pid == 201 and p.returncode is None:   # the 2nd child keeps beating
            status(p)

    sup = eventmerger.Supervisor(["x"], popen=popen, sleep=sleep, clock=clock, wall=clock,
                                 status_path=str(path), log=logs.append)
    sup.run(runs=2)
    hung, ok = procs
    # no heartbeat for > 2 h: SIGTERM ignored, SIGKILL after KILL_WAIT_S
    assert hung.terminated and hung.killed and hung.returncode == -9
    assert any("hung (no status heartbeat for 120 min" in ln for ln in logs)
    assert any("ignored SIGTERM" in ln for ln in logs)
    # the 2nd child ran 3 h on a fresh heartbeat: never touched
    assert not ok.terminated and ok.returncode == 0
    info = sup.info()
    assert info["restarts"] == 1 and info["last_exit"]["rc"] == 0
    assert any("killed as hung (-9)" in ln and "restart in 1 min" in ln for ln in logs)
    # 6 x a long --interval is the limit; a status of another pid doesn't count
    # (the child's start does); a child that said "stopped" is left to exit
    sup2 = eventmerger.Supervisor(["x"], status_path=str(path), wall=clock)
    status(SimpleNamespace(pid=7), config={"interval_s": 3600.0})
    clock.sleep(5 * 3600)
    assert sup2.hung(7, T0) is None
    clock.sleep(3600 + 1)
    assert "limit 360" in sup2.hung(7, T0)
    assert sup2.hung(8, clock() - 60) is None and sup2.hung(8, clock() - 7201)
    status(SimpleNamespace(pid=7), updated=_iso(T0), state="stopped")
    assert sup2.hung(7, T0) is None
    assert eventmerger.Supervisor(["x"]).hung(7, T0) is None    # no status path: no watchdog


class Undying(FakeProc):
    """A hung child stuck in the kernel: deaf to SIGTERM, and SIGKILL lands
    only `linger` s after the first one."""

    def __init__(self, *a, linger, **k):
        super().__init__(*a, deaf=True, **k)
        self.linger, self.kills = linger, []

    def poll(self):
        if self.returncode is None and self.kills and \
                self.clock() - self.kills[0] >= self.linger:
            self.returncode = -9
        return super().poll()

    def kill(self):
        self.killed = True
        self.kills.append(self.clock())


def test_supervisor_never_starts_a_child_beside_one_alive_after_sigkill(tmp_path):
    """A hung child that outlives SIGKILL is polled, logged and SIGKILLed
    again every LINGER_LOG_S until it is gone; only then the next child
    starts. stop() meanwhile: no new child at all."""
    clock, logs, procs = Clock(T0), [], []

    def popen(cmd, **kw):
        assert all(q.poll() is not None for q in procs), "a child beside a live one"
        p = (Undying(300, math.inf, 0, clock, linger=3600) if not procs
             else FakeProc(301, 5, 0, clock))
        procs.append(p)
        return p

    sup = eventmerger.Supervisor(["x"], popen=popen, sleep=clock.sleep, clock=clock, wall=clock,
                                 status_path=str(tmp_path / "merger_status.json"),
                                 log=logs.append)
    sup.run(runs=2)
    stuck, nxt = procs
    assert stuck.returncode == -9 and nxt.returncode == 0
    assert nxt.t0 >= stuck.kills[0] + 3600                      # started once the first was gone
    # SIGKILL (watchdog), then again at once and every 10 min until it went (1 h)
    gaps = [b - a for a, b in zip(stuck.kills[1:], stuck.kills[2:])]
    assert len(stuck.kills) == 7 and set(gaps) == {eventmerger.LINGER_LOG_S}
    alive = [ln for ln in logs if "not gone" in ln]
    assert len(alive) == 6 and "still running after SIGKILL" in alive[0]
    assert "no new child until it is" in alive[0]
    assert any("pid 300 gone (-9)" in ln for ln in logs)
    assert any("killed as hung (-9)" in ln and "restart in 1 min" in ln for ln in logs)
    assert sup.info()["restarts"] == 1
    # stop() while it lingers: the supervision ends without another child
    clock2, procs2 = Clock(T0), []

    def popen2(cmd, **kw):
        assert not procs2, "a child beside a live one"
        procs2.append(Undying(400, math.inf, 0, clock2, linger=math.inf))
        return procs2[-1]

    def sleep2(s):
        clock2.sleep(s)
        p = procs2[0]
        if p.kills and clock2() - p.kills[0] > 1800:
            sup2.stop(wait_s=0.1)

    sup2 = eventmerger.Supervisor(["x"], popen=popen2, sleep=sleep2, clock=clock2, wall=clock2,
                                  status_path=str(tmp_path / "none.json"), log=lambda s: None)
    sup2.run(runs=3)
    assert len(procs2) == 1 and sup2.info()["alive"] is True
    assert sup2.info()["last_exit"]["rc"] == "still running after SIGKILL"


def test_merger_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(eventmerger, "_sup", None)
    monkeypatch.setattr(eventmerger, "Supervisor", lambda *a, **k: pytest.fail("started"))
    monkeypatch.setenv("CREST_CACHE_DIR", str(tmp_path))
    for v in (None, "0", "true", ""):
        if v is None:
            monkeypatch.delenv("EPISODE_MERGER", raising=False)
        else:
            monkeypatch.setenv("EPISODE_MERGER", v)
        assert eventmerger.start() is False and eventmerger._sup is None
    st = eventmerger.status()
    assert st == {"enabled": False, "alive": False, "pid": None, "restarts": 0,
                  "last_exit": None}


def test_merger_enabled_starts_one_supervised_child(monkeypatch, tmp_path):
    made, threads, exits = [], [], []

    class Sup:
        def __init__(self, cmd, **kw):
            made.append(cmd)
            assert kw["status_path"] == str(tmp_path / "merger_status.json")
            assert kw["fallback_path"] == eventmerger.fallback_status_file()

        def run(self):
            pass

        def stop(self):
            pass

        def info(self):
            return {"alive": True, "pid": 4242, "restarts": 2, "last_exit": None}

    class Thread:
        def __init__(self, target, name, daemon):
            threads.append((name, daemon))

        def start(self):
            pass

    monkeypatch.setattr(eventmerger, "_sup", None)
    monkeypatch.setattr(eventmerger, "Supervisor", Sup)
    monkeypatch.setattr(eventmerger.threading, "Thread", Thread)
    monkeypatch.setattr(eventmerger.atexit, "register", exits.append)
    monkeypatch.setenv("EPISODE_MERGER", "1")
    monkeypatch.setenv("CREST_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("EPISODE_MERGER_ARGS", "--max-merges-per-hour 4")
    assert eventmerger.start() and eventmerger.start()
    assert len(made) == 1 and threads == [("episode-merger", True)] and len(exits) == 1
    cmd = made[0]
    assert cmd[1] == eventmerger.SCRIPT and cmd[2:4] == ["--watch", "--apply"]
    assert cmd[4:6] == ["--status-file", str(tmp_path / "merger_status.json")]
    assert cmd[6:] == ["--max-merges-per-hour", "4"]
    # the endpoint: the child's status file + the supervisor's view
    me.write_status(str(tmp_path / "merger_status.json"),
                    {"pid": 1, "state": "idle", "last_pass": {"merged": 1}})
    st = eventmerger.status()
    assert st["enabled"] and st["alive"] and st["pid"] == 4242 and st["restarts"] == 2
    assert st["state"] == "idle" and st["last_pass"] == {"merged": 1}


def test_merger_start_never_raises(monkeypatch, tmp_path, capsys):
    """server.py calls start() at import: a bad EPISODE_MERGER_ARGS or a
    failing thread start is one printed line + status()["start_error"]."""
    monkeypatch.setattr(eventmerger, "_sup", None)
    monkeypatch.setattr(eventmerger, "_start_error", None)
    monkeypatch.setattr(eventmerger, "Supervisor", lambda *a, **k: pytest.fail("started"))
    monkeypatch.setenv("EPISODE_MERGER", "1")
    monkeypatch.setenv("CREST_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("EPISODE_MERGER_ARGS", "--gauge '08353000")      # unclosed quote
    assert eventmerger.start() is False and eventmerger._sup is None
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "NOT started" in out[0] and "EPISODE_MERGER_ARGS" in out[0]
    st = eventmerger.status()
    assert st["enabled"] and st["alive"] is False and "unparsable" in st["start_error"]["error"]

    def no_thread(*a, **k):
        raise RuntimeError("can't start new thread")

    monkeypatch.setenv("EPISODE_MERGER_ARGS", "")
    monkeypatch.setattr(eventmerger, "Supervisor", lambda cmd, **kw: SimpleNamespace(run=None))
    monkeypatch.setattr(eventmerger.threading, "Thread", no_thread)
    assert eventmerger.start() is False and eventmerger._sup is None    # a later start() may work
    assert "RuntimeError" in eventmerger.status()["start_error"]["error"]


def test_merger_args_may_not_move_the_status_file(monkeypatch, tmp_path, capsys):
    """The watchdog reads the heartbeat at status_file(): --status-file in
    EPISODE_MERGER_ARGS (argparse's abbreviations, =value too) would point
    the child elsewhere and get it killed as hung. start() refuses it."""
    monkeypatch.setattr(eventmerger, "_sup", None)
    monkeypatch.setattr(eventmerger, "_start_error", None)
    monkeypatch.setattr(eventmerger, "Supervisor", lambda *a, **k: pytest.fail("started"))
    monkeypatch.setenv("EPISODE_MERGER", "1")
    monkeypatch.setenv("CREST_CACHE_DIR", str(tmp_path))
    for args in ("--status-file /data/m.json", "--max-merges-per-hour 4 --status-file=/x.json",
                 "--status /x.json", "--s=/x.json"):
        monkeypatch.setenv("EPISODE_MERGER_ARGS", args)
        with pytest.raises(ValueError, match="sets --s"):
            eventmerger.command()
        assert eventmerger.start() is False and eventmerger._sup is None
        err = eventmerger.status()["start_error"]["error"]
        assert "the status file is the supervisor's" in err
        assert str(tmp_path / "merger_status.json") in err
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 4 and all("NOT started" in ln and "EPISODE_MERGER_ARGS" in ln for ln in out)
    for args in ("--gauge 08353000", "--retry-min 30 --max-merges-per-hour 4", "--interval=600"):
        monkeypatch.setenv("EPISODE_MERGER_ARGS", args)
        cmd = eventmerger.command()
        assert cmd[4:6] == ["--status-file", str(tmp_path / "merger_status.json")]
        assert cmd[6:] == shlex.split(args)
    # every refused spelling is one argparse reads as --status-file
    for arg in ("--status-file", "--status-f", "--status", "--sta", "--s"):
        assert eventmerger._sets_status_file(arg) and eventmerger._sets_status_file(arg + "=x")
    assert not any(map(eventmerger._sets_status_file, ("--", "-s", "--gauge", "--status-files")))


def test_status_falls_back_when_the_status_file_cannot_be_written(monkeypatch, tmp_path):
    """--status-file unwritable (a full or read-only cache dir): the status,
    the watchdog's heartbeat, goes to the fallback, which says why, until
    the file takes writes again; the supervisor's watchdog and
    /api/merger_status read it. When neither takes one, STATUS_FAILS_MAX
    times in a row, the loop stops between groups and exits STATUS_EXIT (a
    restart), well before the watchdog would call it hung."""
    clock, logs = Clock(), []
    path = str(tmp_path / "cache" / "merger_status.json")
    fb = str(tmp_path / "tmp" / "merger_status.fallback.json")
    broken, write = {path}, me.write_status

    def write_status(p, st):
        if p in broken:
            raise OSError(28, "No space left on device")
        write(p, st)

    monkeypatch.setattr(me, "write_status", write_status)
    plan = Planner()
    plan.todo = ["t1"]
    w = me.Watcher(lambda: "r1", plan, path, clock=clock, sleep=clock.sleep, log=logs.append,
                   fallback=fb)
    assert w.step() == 300 and not os.path.exists(path)
    st = me.read_status(fb)
    assert st["pid"] == os.getpid() and st["updated"] == _iso(T0)
    assert [r["target"] for r in st["recent_merges"]] == ["t1"]
    assert st["status_fallback"]["status_file"] == path and st["status_fallback"]["path"] == fb
    assert "No space left on device" in st["status_fallback"]["error"]
    clock.sleep(300)
    w.step()
    st = me.read_status(fb)
    assert st["updated"] == _iso(T0 + 300) and st["status_fallback"]["since"] == _iso(T0)
    assert sum(f"written to {fb} meanwhile" in ln for ln in logs) == 1          # said once
    assert not any("in a row" in ln for ln in logs) and w.status_fails == 0
    # a child started 3 h ago that beats in the fallback is not hung; read
    # status_path alone, it would be
    started = clock() - 3 * 3600
    sup = eventmerger.Supervisor(["x"], status_path=path, fallback_path=fb, wall=clock)
    assert sup.hung(os.getpid(), started) is None
    assert "no status heartbeat for 180 min" in eventmerger.Supervisor(
        ["x"], status_path=path, wall=clock).hung(os.getpid(), started)
    # the endpoint serves the running child's fallback status
    monkeypatch.setattr(eventmerger, "status_file", lambda: path)
    monkeypatch.setattr(eventmerger, "fallback_status_file", lambda: fb)
    monkeypatch.setattr(eventmerger, "_start_error", None)
    monkeypatch.setattr(eventmerger, "_sup", SimpleNamespace(info=lambda: {
        "alive": True, "pid": os.getpid(), "restarts": 0, "last_exit": None}))
    got = eventmerger.status()
    assert got["status_fallback"]["status_file"] == path and got["updated"] == _iso(T0 + 300)
    # a restart while on the fallback still counts the hour's merges
    w2 = me.Watcher(lambda: "r1", Planner(), path, clock=clock, sleep=clock.sleep,
                    log=lambda s: None, fallback=fb)
    assert [r["target"] for r in w2.recent] == ["t1"] and w2.in_hour(clock()) == 1
    # the file takes writes again: back to it, without the fallback note
    broken.clear()
    clock.sleep(300)
    w.step()
    st = me.read_status(path)
    assert st["updated"] == _iso(T0 + 600) and "status_fallback" not in st
    assert logs[-1].endswith(f"merge status file {path} written again")
    got = eventmerger.status()
    assert got["updated"] == _iso(T0 + 600) and "status_fallback" not in got
    # neither writable: the loop stops and the child exits STATUS_EXIT for a restart
    broken.update({path, fb})
    logs3 = []
    t = clock()
    w3 = me.Watcher(lambda: "r1", Planner(), path, clock=clock, sleep=clock.sleep,
                    log=logs3.append, fallback=fb)
    assert w3.run_forever() == me.STATUS_EXIT and w3.stop.is_set()
    fails = [ln for ln in logs3 if "merge status file not written" in ln]
    assert len(fails) >= me.STATUS_FAILS_MAX and "No space left on device" in fails[0]
    assert f"fallback {fb}: OSError" in fails[0] and fails[0].endswith("1 in a row)")
    assert any(f"no status written {me.STATUS_FAILS_MAX} times in a row" in ln for ln in logs3)
    assert logs3[-1].endswith(f"(exit {me.STATUS_EXIT})")
    assert clock() - t < 6 * w3.interval                         # before the watchdog's limit
    # a child that can't write its own --status-file and has no fallback stops the same way
    w4 = me.Watcher(lambda: "r1", Planner(), path, clock=clock, sleep=clock.sleep,
                    log=lambda s: None)
    assert w4.run_forever() == me.STATUS_EXIT


# ---- review fixes: retry window, final pending/stale, reverts, heartbeat, timeouts -----

def test_watch_keeps_a_finalize_in_its_retry_window_waiting(fake, tmp_path, monkeypatch):
    """A failed finalize is retried at last_pass + retry_s even when the
    index changes in between: that pass falls inside the retry window and
    still reports the group, so it stays waiting instead of being dropped."""
    for e in (B1, B0):
        fake.idx.pop(e)
    fake.idx[A0]["merged_into"] = A1                             # merged before; no final yet
    monkeypatch.setattr(me.eventstore, "finalize_episodes",
                        lambda ids, log=print: (fake.calls["final"].extend(ids), 0)[1])
    clock, rev = Clock(), {"v": "r1"}

    def run(**kw):
        return me.run_pass(fake.store, apply=True, gap_h=12, retry_s=3600,
                           log=lambda s: None, **kw)

    w, _ = _watcher(tmp_path, lambda: rev["v"], run, clock, retry_min=60)
    w.step()                                                     # try 1 fails
    assert fake.calls["final"] == [A1] and len(w.waiting) == 1
    clock.sleep(600)
    rev["v"] = "r2"                                              # someone else's commit
    w.step()
    st = me.read_status(w.path)
    assert st["last_pass"]["trigger"] == "index changed" and fake.calls["final"] == [A1]
    assert [x["members"] for x in st["waiting"]] == [[A1, A0]]   # still waiting
    assert "retried after --retry-min" in st["waiting"][0]["why"]
    assert st["waiting"][0]["since"] == _iso(T0) and st["retry_at"] == _iso(T0 + 600 + 3600)
    clock.sleep(3600)
    w.step()
    assert me.read_status(w.path)["last_pass"]["trigger"] == "retry"
    assert fake.calls["final"] == [A1, A1]                       # try 2


def test_run_pass_final_pending_waits(fake):
    del fake.idx[B1]                                             # one group: A1 + A0
    now = me._epoch(fake.idx[A1]["generated"]) + 3600            # A1 ended an hour ago, no final
    rep = _pass(fake, now=now)
    (s,) = rep["skipped"]
    assert s["why"].startswith("final pending: " + A1) and s["wait"] and fake.calls["plan"] == []
    fake.idx[A1]["final"] = {"ended": "2026-10-28T05:00Z", "n_frames": 13}
    assert [r["target"] for r in _pass(fake, now=now)["merged"]] == [A1]


def test_run_pass_drops_a_stale_final_and_refinalizes(fake):
    """A finalize that read the target's pre-merge manifest landed after the
    merge: its final counts other frames than the merged entry."""
    fake.idx[A0]["merged_into"] = A1
    fake.idx[A1].update(n_frames=30, merged={"n": 2, "ids": [A1, A0]},
                        final={"ended": "2026-10-28T05:00Z", "n_frames": 13})
    logs = []
    rep = me.run_pass(fake.store, apply=True, gap_h=12, budget=0, now=T0, log=logs.append)
    assert fake.calls["drop"] == [A1] and fake.calls["final"] == [A1] and rep["finalized"] == [A1]
    assert any("final is stale (final counts 13 frames, merged 30)" in ln for ln in logs)
    fake.idx[A1]["final"] = {"ended": "2026-10-28T05:00Z", "n_frames": 30}
    rep = _pass(fake, budget=0, now=T0)                          # current: left alone
    assert fake.calls["drop"] == [A1] and fake.calls["final"] == [A1] and rep["finalized"] == []
    assert me.stale_final({"final": {"n_frames": 13}, "n_frames": 30}) is None   # not merged


def test_drop_final_commits_the_index_without_it():
    idx = {A1: _ep(A1, n_frames=30, merged={"n": 2, "ids": [A1, A0]}, final={"n_frames": 13}),
           A0: _ep(A0, merged_into=A1)}
    commits = []

    class Api:
        def create_commit(self, repo_id, operations, commit_message, repo_type=None,
                          parent_commit=None):
            (op,) = operations
            with open(op.path_or_fileobj, encoding="utf-8") as fp:
                commits.append((parent_commit, json.load(fp), commit_message))
            return SimpleNamespace(oid="feed1234")

    store = SimpleNamespace(api=Api(), head=lambda: "sha9",
                            index_at=lambda sha: json.loads(json.dumps(idx)))
    assert me.drop_final(store, A1, log=lambda s: None) is True
    ((parent, new, msg),) = commits
    assert parent == "sha9" and "final" not in new[A1] and new[A1]["merged"] == idx[A1]["merged"]
    assert new[A0] == idx[A0] and list(new) == list(idx) and "stale final" in msg
    idx[A1]["final"] = {"n_frames": 30}                          # re-finalized meanwhile
    assert me.drop_final(store, A1, log=lambda s: None) is None
    idx[A1].pop("final")                                         # dropped meanwhile: finalize
    assert me.drop_final(store, A1, log=lambda s: None) is True and len(commits) == 1


def _rec(idx, merged_gen="2026-11-02T23:50:00Z", **kw):
    return {"target": A1, "members": [A1, A0], "at": _iso(T0 - 600), "commit": "c1",
            "generated": {m: idx[m]["generated"] for m in (A1, A0)},
            "merged_gen": merged_gen, "n_frames": 30, **kw}


def test_reverted_merges_are_told_from_later_changes():
    pre = {A1: _ep(A1), A0: _ep(A0)}
    rec = _rec(pre)
    merged = {A1: {**pre[A1], "generated": rec["merged_gen"], "merged": {"n": 2, "ids": [A1, A0]}},
              A0: {**pre[A0], "merged_into": A1}}
    assert me.reverted([rec], merged) == {}                      # in place
    assert me.reverted([rec], pre) == {A1: rec}                  # a stale write: pre-merge again
    assert me.reverted([rec], {A1: merged[A1], A0: pre[A0]}) == {A1: rec}
    repub = {A1: {**pre[A1], "generated": "2026-11-03T02:00:00Z"}, A0: pre[A0]}
    assert me.reverted([rec], repub) == {}                       # re-published since: not ours
    assert me.reverted([rec], {A1: {**pre[A1], "merged_into": "x"}, A0: pre[A0]}) == {}
    assert me.reverted([rec], {A1: pre[A1]}) == {}               # no member left
    assert me.reverted([{k: v for k, v in rec.items() if k != "generated"}], pre) == {}
    later = {**rec, "generated": {A1: "x", A0: "y"}, "merged_gen": "z"}
    assert me.reverted([rec, later], pre) == {}                  # the last record decides
    # a member without merged_into re-published since (another generated) is
    # new content, not a revert; noted for the log
    a0new = {**pre[A0], "generated": "2026-11-03T03:00Z"}
    for target in (pre[A1], merged[A1]):
        news = {}
        assert me.reverted([rec], {A1: target, A0: a0new}, news) == {}
        assert list(news) == [A1] and news[A1] == (
            f"{A0} re-published since (generated 2026-11-03T03:00Z, "
            f"{pre[A0]['generated']} when merged)")
    news = {}
    assert me.reverted([rec], {A1: pre[A1], A0: {**a0new, "merged_into": A1}}, news) == {A1: rec}
    assert news == {}                                            # still merged in: as is
    assert me.reverted([rec], repub, news) == {} and news[A1].startswith(f"{A1} re-published")
    assert me.reverted([rec], merged, news) == {} and news == {}                 # in place
    three = {**rec, "members": [A1, A0, "2026102723_G1"],
             "generated": {**rec["generated"], "2026102723_G1": "2026-10-28T00:00Z"}}
    idx3 = {**pre, "2026102723_G1": _ep("2026102723_G1", generated="2026-11-03T04:00Z")}
    assert me.reverted([three], idx3, news) == {} and "2026102723_G1 re-published" in news[A1]
    assert A0 not in news[A1]
    idx3["2026102723_G1"]["generated"] = "2026-10-28T00:00Z"
    assert me.reverted([three], idx3, news) == {A1: three} and news == {}


def test_run_pass_reapplies_a_reverted_merge_outside_the_cap(fake):
    memo, logs = {}, []
    rec = _rec(fake.idx)                                         # the index is pre-merge again
    rep = me.run_pass(fake.store, apply=True, gap_h=12, budget=0, memo=memo, now=T0,
                      recent=[rec], log=logs.append)
    assert rep["reverted"] == [A1] and any(f"merge into {A1} reverted" in ln for ln in logs)
    assert [(r["target"], r.get("reapply")) for r in rep["merged"]] == [(A1, True)]
    assert [s["why"][:8] for s in rep["skipped"]] == ["rate cap"]    # B1's merge waits for the cap
    for _ in range(me.REAPPLY_FREE - 1):
        me.run_pass(fake.store, apply=True, gap_h=12, budget=0, memo=memo, now=T0,
                    recent=[rec], log=lambda s: None)
    assert fake.calls["commit"] == [A1] * me.REAPPLY_FREE
    rep = me.run_pass(fake.store, apply=True, gap_h=12, budget=0, memo=memo, now=T0,
                      recent=[rec], log=lambda s: None)
    assert rep["merged"] == [] and rep["skipped"][0]["why"].startswith("rate cap")
    # with room in the cap it is an ordinary merge again; in place: nothing to re-apply
    rep = me.run_pass(fake.store, apply=True, gap_h=12, budget=1, memo=memo, now=T0,
                      recent=[rec], log=lambda s: None)
    assert [(r["target"], r.get("reapply")) for r in rep["merged"]] == [(A1, None)]
    fake.idx[A1]["merged"] = {"n": 2, "ids": [A1, A0]}
    fake.idx[A0]["merged_into"] = A1
    assert me.run_pass(fake.store, apply=True, gap_h=12, budget=0, now=T0, recent=[rec],
                       log=lambda s: None)["reverted"] == []


def test_run_pass_a_member_republished_since_merges_within_the_cap(fake):
    """The index lost the merge, but a member was re-published since: new
    content, merged again as an ordinary merge (the hourly cap applies),
    and the log says why it is not a re-apply."""
    rec = _rec(fake.idx)
    fake.idx[A0]["generated"] = "2026-11-03T03:00Z"
    memo, logs = {}, []
    rep = me.run_pass(fake.store, apply=True, gap_h=12, budget=0, memo=memo, now=T0,
                      recent=[rec], log=logs.append)
    assert rep["reverted"] == [] and rep["merged"] == [] and fake.calls["commit"] == []
    assert [s["why"][:8] for s in rep["skipped"]] == ["rate cap", "rate cap"]
    (note,) = [ln for ln in logs if "not a revert" in ln]
    assert note.startswith(f"== G1: merge into {A1} undone but not a revert: {A0} re-published")
    assert "within the hourly cap" in note
    rep = me.run_pass(fake.store, apply=True, gap_h=12, budget=1, memo=memo, now=T0,
                      recent=[rec], log=lambda s: None)
    assert [(r["target"], r.get("reapply")) for r in rep["merged"]] == [(A1, None)]


def test_watch_reapplies_skip_the_cap_and_records_go_back_to_the_pass(tmp_path):
    clock, calls = Clock(), []

    def run(budget, stop, memo, now, on_merge, recent=(), beat=None):
        calls.append([r["target"] for r in recent])
        rec = {"gauge": "G", "target": "t1", "members": ["t1", "t0"], "commit": "c",
               "reapply": True, "generated": {"t1": "g1", "t0": "g0"}, "merged_gen": "gm",
               "n_frames": 9}
        on_merge(rec)
        return _report(merged=[rec], planned=1)

    revs = iter(["a", "b"])
    w, logs = _watcher(tmp_path, lambda: next(revs), run, clock, max_per_hour=1)
    w.step()
    st = me.read_status(w.path)
    assert st["merges_last_hour"] == 0 == w.in_hour(clock())
    r = st["recent_merges"][0]
    assert r["reapply"] is True and r["generated"] == {"t1": "g1", "t0": "g0"}
    assert r["merged_gen"] == "gm" and r["n_frames"] == 9
    assert "(1 re-applied after a revert)" in logs[-1]
    clock.sleep(300)
    w.step()
    assert calls == [[], ["t1"]]                                  # the next pass checks it


def test_watch_heartbeat_while_sleeping_and_inside_a_pass(tmp_path):
    """The status 'updated' the supervisor's watchdog reads: fresh before
    every nap (a nap is at most one interval, also in a 60-min backoff) and
    on beat() inside a pass (throttled to BEAT_S)."""
    clock, stop, naps = Clock(), me.Stop(), []

    def rev():
        raise OSError("hub unreachable")

    def sleep(s):
        assert me.read_status(w.path)["updated"] == _iso(clock()) and s <= w.interval
        naps.append(s)
        clock.sleep(s)
        if clock() - T0 >= 4 * 3600:
            stop.set()

    w = me.Watcher(rev, Planner(), str(tmp_path / "a" / "merger_status.json"), clock=clock,
                   sleep=sleep, stop=stop, log=lambda s: None, interval=300)
    assert w.run_forever() == 0
    assert max(naps) == 300 and len(naps) > 40 and w.backoff == 3600
    seen = []

    def run(**kw):
        clock.sleep(100)                                         # a long download ...
        kw["beat"]()
        seen.append(me.read_status(w2.path)["updated"])
        clock.sleep(10)
        kw["beat"]()                                             # ... throttled
        seen.append(me.read_status(w2.path)["updated"])
        return _report()

    w2, _ = _watcher(tmp_path / "b", lambda: "x", run, clock)
    t = clock()
    w2.step()
    assert seen == [_iso(t + 100), _iso(t + 100)]
    # downloads beat once per file
    beats = []
    store = SimpleNamespace(fetch=lambda ev, name, dest: os.path.join(dest, ev, name))
    got = me._fetch_all(store, [("a", "x.tif"), ("b", "y.tif"), ("c", "z.tif")], "d",
                        beat=lambda: beats.append(1))
    assert len(beats) == 3 and got[("b", "y.tif")] == os.path.join("d", "b", "y.tif")


def test_run_pass_beats_per_group_and_hands_beat_to_the_downloads(fake):
    beats = []

    def beat():
        beats.append(1)

    _pass(fake, beat=beat)
    assert len(beats) == 2 and fake.calls["beat"] == [beat, beat]


def test_hub_calls_get_finite_timeouts(monkeypatch):
    """huggingface_hub 1.x: its shared httpx client has no timeout and
    repo_info passes timeout=None outright; after bound_hub_timeouts every
    request carries finite limits, a file download keeps its own."""
    import huggingface_hub
    if not hasattr(huggingface_hub, "set_client_factory"):
        pytest.skip(f"huggingface_hub {huggingface_hub.__version__}: the 0.x test covers it")
    import httpx
    from huggingface_hub import constants
    from huggingface_hub.utils import _http
    seen = []

    def handler(request):
        seen.append(dict(request.extensions["timeout"]))
        return httpx.Response(200, json={})

    monkeypatch.setattr(_http, "default_client_factory", lambda: httpx.Client(
        transport=httpx.MockTransport(handler), event_hooks={"request": [], "response": []}))
    monkeypatch.setattr(_http, "_GLOBAL_CLIENT_FACTORY", _http._GLOBAL_CLIENT_FACTORY)
    for k in me.HUB_FILE_TIMEOUT_S:
        monkeypatch.setenv(k, "1")                               # restored after the test
        monkeypatch.delenv(k)
        monkeypatch.setattr(constants, k, getattr(constants, k))
    _http.close_session()
    try:
        summary = me.bound_hub_timeouts()
        c = _http.get_session()
        c.get("https://hub.invalid/api/datasets/x", timeout=None)    # repo_info's way
        c.get("https://hub.invalid/api/datasets/x/paths-info/main")  # the client default
        c.get("https://hub.invalid/x/resolve/main/depth.tif", timeout=60)
    finally:
        _http.close_session()
    assert seen == [me.HUB_TIMEOUT_S, me.HUB_TIMEOUT_S, dict.fromkeys(me.HUB_TIMEOUT_S, 60)]
    assert constants.HF_HUB_DOWNLOAD_TIMEOUT == 60 and constants.HF_HUB_ETAG_TIMEOUT == 30
    assert os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] == "60" and "read 300 s" in summary
    assert summary.startswith("Hub timeouts via set_client_factory (huggingface_hub 1.")


@pytest.fixture
def hub0x(monkeypatch):
    """huggingface_hub 0.x's HTTP backend: the installed package when it is
    0.x, else a stand-in of what bound_hub_timeouts uses, as 0.36.2 has it:
    configure_http_backend(backend_factory) -> get_session(), one requests
    session per thread from that factory (cache cleared on configure); its
    own factory mounts UniqueRequestIdAdapter, OfflineAdapter (raising
    OfflineModeIsEnabled) under constants.HF_HUB_OFFLINE."""
    import huggingface_hub
    if hasattr(huggingface_hub, "configure_http_backend"):
        from huggingface_hub import constants
        from huggingface_hub.errors import OfflineModeIsEnabled
        from huggingface_hub.utils import _http
        monkeypatch.setattr(_http, "_GLOBAL_BACKEND_FACTORY", _http._GLOBAL_BACKEND_FACTORY)
        yield SimpleNamespace(real=True, http=_http, constants=constants,
                              offline=OfflineModeIsEnabled)
        _http.reset_sessions()                                   # none from our factory stays
        return
    import requests
    from requests.adapters import HTTPAdapter

    class OfflineModeIsEnabled(ConnectionError):
        pass

    class UniqueRequestIdAdapter(HTTPAdapter):
        def add_headers(self, request, **kwargs):
            super().add_headers(request, **kwargs)
            request.headers.setdefault("X-Amzn-Trace-Id", "stand-in")

    class OfflineAdapter(HTTPAdapter):
        def send(self, request, *args, **kwargs):
            raise OfflineModeIsEnabled(f"Cannot reach {request.url}: offline mode is enabled.")

    hub, constants = ModuleType("huggingface_hub"), ModuleType("huggingface_hub.constants")
    utils, http = ModuleType("huggingface_hub.utils"), ModuleType("huggingface_hub.utils._http")
    constants.HF_HUB_OFFLINE, constants.HF_HUB_DOWNLOAD_TIMEOUT = False, 10
    constants.HF_HUB_ETAG_TIMEOUT = 10
    sessions = {}

    def default_backend_factory():
        s = requests.Session()
        adapter = OfflineAdapter if constants.HF_HUB_OFFLINE else UniqueRequestIdAdapter
        s.mount("http://", adapter())
        s.mount("https://", adapter())
        return s

    def configure_http_backend(backend_factory=default_backend_factory):
        http.factory = backend_factory
        sessions.clear()

    def get_session():
        key = (os.getpid(), threading.get_ident())
        if key not in sessions:
            sessions[key] = http.factory()
        return sessions[key]

    http.factory, http.reset_sessions = default_backend_factory, sessions.clear
    http.get_session = hub.get_session = get_session
    http.UniqueRequestIdAdapter, http.OfflineAdapter = UniqueRequestIdAdapter, OfflineAdapter
    http.configure_http_backend = hub.configure_http_backend = configure_http_backend
    hub.__version__, hub.constants, hub.utils, utils._http = "0.36.2", constants, utils, http
    for mod in (hub, constants, utils, http):
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    yield SimpleNamespace(real=False, http=http, constants=constants, offline=OfflineModeIsEnabled)


def test_hub_0x_calls_get_finite_timeouts(hub0x, monkeypatch):
    """huggingface_hub 0.x, as the live Spaces run it: requests sessions from
    configure_http_backend, and requests waits forever without a timeout
    (paths-info passes none, repo_info None). After bound_hub_timeouts the
    sessions' adapter sends every request with (30, 300), fills a None part
    of a pair, leaves a download's own; HF_HUB_OFFLINE keeps the library's
    OfflineAdapter. With the real package: index_rev's get_paths_info too."""
    import requests
    from requests.adapters import HTTPAdapter
    seen = []

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        seen.append(timeout)
        r = requests.Response()
        r.status_code, r.request, r.url = 200, request, request.url
        r._content = json.dumps([{"type": "file", "path": "events/index.json",
                                  "oid": "blob42", "size": 3}]).encode()
        return r

    monkeypatch.setattr(HTTPAdapter, "send", send)                # the wire
    constants, url = hub0x.constants, "https://hub.invalid/api/datasets/x"
    for k in me.HUB_FILE_TIMEOUT_S:
        monkeypatch.setenv(k, "1")                               # restored after the test
        monkeypatch.delenv(k)
        monkeypatch.setattr(constants, k, getattr(constants, k))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    summary = me.bound_hub_timeouts()
    s = hub0x.http.get_session()
    adapter = s.get_adapter(url)
    assert isinstance(adapter, hub0x.http.UniqueRequestIdAdapter)
    s.get(url, timeout=None)                                     # repo_info's way
    s.post(url + "/paths-info/main", data={"paths": ["events/index.json"]})   # none at all
    s.get(url + "/resolve/main/depth.tif", timeout=60)           # a download's own
    s.get(url, timeout=(5, None))
    adapter.send(requests.Request("GET", url).prepare(), False, None)          # positional
    full = (me.HUB_TIMEOUT_S["connect"], me.HUB_TIMEOUT_S["read"])
    assert seen == [full, full, 60, (5, full[1]), full]
    assert summary.startswith("Hub timeouts via configure_http_backend (huggingface_hub 0.")
    assert "connect 30 s, read 300 s" in summary and "HF_HUB_DOWNLOAD_TIMEOUT=60" in summary
    assert constants.HF_HUB_DOWNLOAD_TIMEOUT == 60 and constants.HF_HUB_ETAG_TIMEOUT == 30
    if hub0x.real:                                               # the library's own call path
        from huggingface_hub import HfApi
        seen.clear()
        assert me.index_rev(HfApi(token=False)) == "blob42" and seen == [full]
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    hub0x.http.reset_sessions()
    n = len(seen)
    with pytest.raises(hub0x.offline):
        hub0x.http.get_session().get(url)
    assert len(seen) == n                                        # nothing reached the wire

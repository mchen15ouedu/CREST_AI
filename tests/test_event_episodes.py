"""hf_data.eventsim.hourly_tick: one flood stays ONE episode. A gauge that
detection flags again keeps its active episode going, and a re-detection
soon after an episode ended reopens it (same id + episode_started anchor)
instead of opening a new event id. No network: detect, the observations,
the nowcast store, the store's hub I/O, the queue and run_one are faked;
mark_ended / reopen_episodes / publish_event run for real against an
in-memory index. run_one's CREST-miss guard runs for real behind a faked
EF5 (a reopened episode's miss is noted, never purged)."""
from __future__ import annotations

import datetime
import json
import sys
import tempfile
import types

import pytest

import hf_data
from hf_data import eventqueue, eventsim, eventstore, nowcaststore

G1, G2 = "09353800", "08353000"


def _now():
    return datetime.datetime.utcnow()


def _h(dt):
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def _ev(gauge, started_h_ago, status="active", ended_h_ago=None, final=True):
    """(event_id, summary) shaped like events/index.json: the episode started
    `started_h_ago` hours ago; an ended one ended `ended_h_ago` hours ago."""
    start = (_now() - datetime.timedelta(hours=started_h_ago)).replace(
        minute=0, second=0, microsecond=0)
    s = {"t0": _h(start), "sim_start": _h(start - datetime.timedelta(hours=12)),
         "t_end": _h(start + datetime.timedelta(hours=12)),
         "episode_started": _h(start), "status": status, "gauge": gauge,
         "trigger": {"gauge": gauge, "obs_m3s": 58.0},
         "generated": (start + datetime.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if status == "ended" and final:
        s["final"] = {"ended": _h(_now() - datetime.timedelta(hours=ended_h_ago)),
                      "duration_h": started_h_ago - ended_h_ago, "peak_depth_m": 1.2}
    return f"{start:%Y%m%d%H}_{gauge}", s


class _SyncThread:
    """The finalize dispatch runs inline, so the test sees what it was given."""

    def __init__(self, target=None, args=(), kwargs=None, **kw):
        self._run = lambda: target(*args, **(kwargs or {}))

    def start(self):
        self._run()


@pytest.fixture
def tick(monkeypatch):
    """A faked world around hourly_tick; returns its mutable state."""
    st = types.SimpleNamespace(idx={}, picks=[], obs={}, pins=frozenset(),
                               pending={}, claims={}, specs={}, runs=[],
                               finalized=[], commits=[], detect_budget=[])

    def detect(max_events=1):
        st.detect_budget.append(max_events)
        return [{"gauge": g, "lat": 35.0, "lon": -110.0, "area_km2": 1000.0,
                 "obs_m3s": 60.0, "hotspot_score": 1.0} for g in st.picks][:max_events]

    class FakeApi:
        def create_commit(self, repo_id=None, repo_type=None, operations=(),
                          commit_message=""):
            st.commits.append(commit_message)
            for op in operations:
                if getattr(op, "path_in_repo", "") == f"{eventstore.PREFIX}/index.json":
                    st.idx = json.loads(op.path_or_fileobj.getvalue())

        def list_repo_files(self, *a, **k):
            return []

    def run_one(gid, t0=None, trigger=None, episode_id=None, force_cpu=False, **kw):
        st.runs.append({"gauge": gid, "episode": episode_id, "trigger": trigger,
                        "force_cpu": force_cpu})
        return {"event_id": episode_id or f"new_{gid}", "published": False}

    conus = types.SimpleNamespace(maybe_refresh=lambda: None)
    monkeypatch.setitem(sys.modules, "hf_data.conusstate", conus)
    monkeypatch.setattr(hf_data, "conusstate", conus, raising=False)
    monkeypatch.setattr(eventsim, "detect", detect)
    monkeypatch.setattr(eventsim, "_recent_obs_max", lambda gid: st.obs.get(gid))
    monkeypatch.setattr(eventsim, "run_one", run_one)
    monkeypatch.setattr(eventsim, "threading", types.SimpleNamespace(Thread=_SyncThread))
    monkeypatch.setitem(eventsim._running, "id", None)
    monkeypatch.setattr(nowcaststore, "all_risk", lambda: {"tiers": {}})
    # baseflow 10 m3/s: observed flow keeps an episode alive at >= 50 m3/s
    monkeypatch.setattr(nowcaststore, "_thresholds", lambda: {g: {"qbase": 10.0} for g in (G1, G2)})
    monkeypatch.setattr(eventstore, "_api", lambda: FakeApi())
    monkeypatch.setattr(eventstore, "load_index", lambda: json.loads(json.dumps(st.idx)))
    monkeypatch.setattr(eventstore, "pinned_ids", lambda force=False: st.pins)
    monkeypatch.setattr(eventstore, "finalize_episodes",
                        lambda ids, log=print: st.finalized.extend(ids))
    monkeypatch.setattr(eventstore, "retention_sweep", lambda: 0)
    monkeypatch.setattr(eventqueue, "mode", lambda: "on")
    monkeypatch.setattr(eventqueue, "gauge_pending", lambda gid: st.pending.get(gid))
    monkeypatch.setattr(eventqueue, "claim_fresh_for", lambda ev: st.claims.get(ev))
    monkeypatch.setattr(eventqueue, "_spec", lambda ev, **k: st.specs.get(ev))
    monkeypatch.setattr(eventqueue, "queue_status", lambda: [])
    return st


def test_redetected_gauge_keeps_its_episode(tick):
    a, sa = _ev(G1, 3)                     # obs 20 < 5 x baseflow: would end...
    b, sb = _ev(G2, 3)                     # ...and this one, not re-flagged, does
    tick.idx = {a: sa, b: sb}
    tick.obs = {G1: 20.0, G2: 20.0}
    tick.picks = [G1]
    out = eventsim.hourly_tick()
    assert out["ended"] == [b]
    assert tick.idx[a]["status"] == "active" and tick.idx[b]["status"] == "ended"
    assert tick.finalized == [b]
    # the re-flagged gauge re-simulates its episode; no new event id for it
    assert [r["episode"] for r in tick.runs] == [a]
    # the detection budget leaves MAX_PER_TICK new floods beside the actives
    assert tick.detect_budget == [eventsim.MAX_PER_TICK + 2]


def test_redetection_within_window_reopens(tick):
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    tick.idx = {b: sb}
    tick.picks = [G1]
    eventsim.hourly_tick()
    run, = tick.runs
    assert run["episode"] == b and run["gauge"] == G1 and not run["force_cpu"]
    assert run["trigger"]["reopened"] is True                 # miss guard: no purge
    s = tick.idx[b]
    assert s["status"] == "active" and "final" not in s
    assert s["episode_started"] == sb["episode_started"]      # anchor kept
    stamp = datetime.datetime.strptime(s["reopened"], "%Y-%m-%dT%H:%M:%SZ")
    assert abs((_now() - stamp).total_seconds()) < 120
    assert tick.commits == [f"episodes reopened: {b}"]
    assert tick.finalized == []


def test_reopen_is_not_raced_by_the_heal_pass(tick):
    # ended but its final product was lost: the heal pass would finalize it
    # in this very tick — a reopened episode must not be in that batch
    b, sb = _ev(G1, 6, status="ended", final=False)
    tick.idx = {b: sb}
    tick.picks = [G1]
    eventsim.hourly_tick()
    assert [r["episode"] for r in tick.runs] == [b]
    assert tick.idx[b]["status"] == "active"
    assert tick.finalized == []


def test_redetection_outside_window_starts_new_episode(tick):
    b, sb = _ev(G1, 40, status="ended", ended_h_ago=eventsim.REOPEN_H + 8)
    tick.idx = {b: sb}
    tick.picks = [G1]
    eventsim.hourly_tick()
    assert [r["episode"] for r in tick.runs] == [None]
    assert tick.idx[b]["status"] == "ended" and tick.idx[b]["final"]
    assert tick.commits == []


def test_pinned_episode_is_never_reopened(tick):
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=2)
    tick.idx = {b: sb}
    tick.pins = frozenset({b})
    tick.picks = [G1]
    eventsim.hourly_tick()
    assert [r["episode"] for r in tick.runs] == [None]
    assert tick.idx[b]["status"] == "ended" and tick.idx[b]["final"]
    assert tick.commits == []
    # unreadable pin list: no reopen either (a demo event can't be told apart)
    tick.pins, tick.runs = None, []
    eventsim.hourly_tick()
    assert [r["episode"] for r in tick.runs] == [None] and tick.commits == []


def test_worker_mid_solve_is_not_disturbed(tick):
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=2)
    tick.idx = {b: sb}
    tick.picks = [G1]
    tick.claims = {b: "jetstream2:vm1"}
    eventsim.hourly_tick()
    assert tick.runs == [] and tick.commits == [] and tick.idx[b]["status"] == "ended"
    tick.claims, tick.pending = {}, {G1: b}            # queued, not yet claimed
    eventsim.hourly_tick()
    assert tick.runs == [] and tick.commits == []
    # a cold re-run spec of the episode waits unclaimed — gauge_pending skips
    # cold specs, and a reopen's run_one would replace it: no reopen (the
    # queued job's publish reactivates the episode), and no new id either
    tick.pending, tick.specs = {}, {b: {"event_id": b, "cold": "2026-11-03T00:00:00Z"}}
    eventsim.hourly_tick()
    assert tick.runs == [] and tick.commits == [] and tick.idx[b]["status"] == "ended"


def test_max_episode_still_ends(tick):
    over = eventsim.MAX_EPISODE_H + 1
    a, sa = _ev(G1, over)                  # re-flagged, high obs, but over age
    tick.idx = {a: sa}
    tick.obs = {G1: 500.0}
    tick.picks = [G1]
    out = eventsim.hourly_tick()
    assert out["ended"] == [a] and tick.idx[a]["status"] == "ended"
    assert tick.finalized == [a]
    # never ended and reopened in one tick: the flood goes on as a new episode
    assert [r["episode"] for r in tick.runs] == [None]
    assert tick.commits == [f"episodes ended: {a}"]
    # an over-age episode that ended recently is not reopened either
    b, sb = _ev(G2, over, status="ended", ended_h_ago=1)
    tick.idx, tick.picks, tick.runs, tick.commits = {b: sb}, [G2], [], []
    eventsim.hourly_tick()
    assert [r["episode"] for r in tick.runs] == [None] and tick.commits == []


def test_reopen_window_measured_from_final_then_t_end_then_generated(monkeypatch):
    monkeypatch.setattr(eventsim, "REOPEN_H", 12.0)
    now = _now()
    b, sb = _ev(G1, 30, status="ended", ended_h_ago=20)
    sb["t_end"] = _h(now - datetime.timedelta(hours=1))      # final.ended wins
    assert eventsim._reopen_candidate({b: sb}, G1, now) is None
    sb.pop("final")                                            # -> t_end
    assert eventsim._reopen_candidate({b: sb}, G1, now) == b
    sb.pop("t_end")                                            # -> generated (29 h ago)
    assert eventsim._reopen_candidate({b: sb}, G1, now) is None
    sb["generated"] = (now - datetime.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert eventsim._reopen_candidate({b: sb}, G1, now) == b
    assert eventsim._reopen_candidate({b: sb}, G1, now, skip=[b]) is None
    # only the gauge's MOST RECENT episode is a candidate
    c, sc = _ev(G1, 4)
    assert eventsim._reopen_candidate({c: sc, b: sb}, G1, now) is None
    monkeypatch.setattr(eventsim, "REOPEN_H", 0.0)            # disabled
    assert eventsim._reopen_candidate({b: sb}, G1, now) is None


def test_merged_fragment_is_never_reopened(tick):
    # the merge script absorbed fragment f into episode a: f is newer and
    # ended later, but it is neither a candidate nor the gauge's latest
    a, sa = _ev(G1, 30, status="ended", ended_h_ago=3)
    f, sf = _ev(G1, 6, status="ended", ended_h_ago=1)
    sf["merged_into"] = a
    now = _now()
    assert eventsim._reopen_candidate({f: sf, a: sa}, G1, now) == a
    assert eventsim._reopen_candidate({f: sf}, G1, now) is None
    tick.idx = {f: sf, a: sa}
    tick.picks = [G1]
    eventsim.hourly_tick()
    assert [r["episode"] for r in tick.runs] == [a]
    assert tick.idx[a]["status"] == "active" and tick.idx[f]["status"] == "ended"
    assert tick.commits == [f"episodes reopened: {a}"]


def test_worker_publish_keeps_a_reopened_episode_active_and_anchored(tick, tmp_path,
                                                                      monkeypatch):
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    tick.idx = {b: sb}
    assert eventstore.reopen_episodes([b]) == [b]
    monkeypatch.setattr(eventstore, "_prior_manifest", lambda ev: None)
    (tmp_path / "maxdepth.tif").write_bytes(b"x")
    t0 = _h(_now().replace(minute=0, second=0, microsecond=0))
    man = {"event_id": b, "t0": t0, "sim_start": sb["sim_start"], "t_end": t0,
           "gauge": G1, "trigger": sb["trigger"], "status": "active", "frames": [],
           "generated": "2026-11-03T12:00:00Z", "provenance": {"engine": "test"}}
    assert eventstore.publish_event(str(tmp_path), man)
    s = tick.idx[b]
    assert s["status"] == "active" and "final" not in s
    assert s["episode_started"] == sb["episode_started"] != t0
    assert next(iter(tick.idx)) == b                           # newest first


def test_reopen_refused_when_the_fresh_index_moved_on(tick):
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    c, sc = _ev(G1, 2, status="ended", ended_h_ago=1)           # newer at G1
    a, sa = _ev(G1, 20)                                         # older, active
    o, so = _ev(G2, 1)                                          # other gauge
    f, sf = _ev(G1, 1, status="ended", ended_h_ago=0)           # merged fragment
    sf["merged_into"] = b
    for fresh, ok in (({c: sc, b: sb}, False), ({b: sb, a: sa}, False),
                      ({o: so, b: sb}, True), ({f: sf, b: sb}, True)):
        tick.idx, tick.commits, sup = json.loads(json.dumps(fresh)), [], set()
        assert eventstore.reopen_episodes([b], superseded=sup) == ([b] if ok else [])
        assert sup == (set() if ok else {b})
        assert tick.idx[b]["status"] == ("active" if ok else "ended")
        assert bool(tick.commits) == ok
    # an episode absorbed into another is never reopened itself
    tick.idx, sup = {f: sf, b: sb}, set()
    assert eventstore.reopen_episodes([f], superseded=sup) == [] and sup == {f}


def test_superseded_reopen_starts_nothing(tick, monkeypatch):
    # a worker publishes a NEW episode for the gauge after the tick read the
    # index: the reopen is refused, and no third id is opened beside it
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    c, sc = _ev(G1, 1)
    tick.idx = {b: sb}
    tick.picks = [G1]
    detect = eventsim.detect

    def detect_while_a_worker_publishes(max_events=1):
        tick.idx = {c: sc, **tick.idx}
        return detect(max_events)

    monkeypatch.setattr(eventsim, "detect", detect_while_a_worker_publishes)
    eventsim.hourly_tick()
    assert tick.runs == [] and tick.commits == []
    assert tick.idx[b]["status"] == "ended" and tick.idx[b]["final"]
    # next tick reads the new index: the active episode goes on
    monkeypatch.setattr(eventsim, "detect", detect)
    eventsim.hourly_tick()
    assert [r["episode"] for r in tick.runs] == [c]


def test_reopen_restarts_the_cpu_fallback_clock(tick):
    # last publish 9 h ago (before the end): measured from `generated` alone
    # the CPU rung would engage on the very next tick after the reopen
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    tick.idx = {b: sb}
    tick.picks = [G1]
    eventsim.hourly_tick()
    assert tick.idx[b]["status"] == "active" and tick.idx[b]["reopened"]
    tick.runs = []
    eventsim.hourly_tick()                                      # an hour later
    run, = tick.runs
    assert run["episode"] == b and not run["force_cpu"]
    # ...until CPU_FALLBACK_H pass with no publish since the reopen
    tick.idx[b]["reopened"] = (_now() - datetime.timedelta(
        hours=eventsim.CPU_FALLBACK_H + 1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    tick.runs = []
    eventsim.hourly_tick()
    assert [r["force_cpu"] for r in tick.runs] == [True]
    # a publish after the reopen is the newer clock
    tick.idx[b]["generated"] = _now().strftime("%Y-%m-%dT%H:%M:%SZ")
    tick.runs = []
    eventsim.hourly_tick()
    assert [r["force_cpu"] for r in tick.runs] == [False]


@pytest.fixture
def missing_ef5(monkeypatch, tmp_path):
    """run_one up to its CREST-miss guard: EF5 (faked) delivers a tenth of
    the observed peak. Returns the guard's drop_events / note_missed calls;
    `.index` is what eventstore.load_index serves."""
    st = types.SimpleNamespace(index={}, drops=[], notes=[])
    t0 = _now().replace(minute=0, second=0, microsecond=0)
    rows = [{"time": (t0 - datetime.timedelta(hours=h)).strftime("%Y-%m-%d %H:%M"),
             "sim_q": 5.0, "obs_q": 50.0} for h in range(3)]

    def run_gauge(gid, *a, **k):
        yield "meta", {"model": "crest"}
        yield "hydro", {"rows": rows}
        yield "done", {"returncode": 0}

    from hf_data import basinextent
    pipeline = types.SimpleNamespace(run_gauge=run_gauge, gauge_info=lambda gid: {
        "lat": 35.0, "lon": -110.0, "area": 1000.0})
    monkeypatch.setitem(sys.modules, "hf_data.pipeline", pipeline)
    monkeypatch.setattr(hf_data, "pipeline", pipeline, raising=False)
    monkeypatch.setitem(sys.modules, "crestimap", types.ModuleType("crestimap"))
    monkeypatch.setattr(sys.modules["crestimap"], "EventConfig", None, raising=False)
    monkeypatch.setattr(sys.modules["crestimap"], "run_event", None, raising=False)
    monkeypatch.setattr(basinextent, "ef5_extent", lambda g: {
        "source": "test", "carved_km2": 1000.0, "cells": 100})
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(eventqueue, "mode", lambda: "on")
    monkeypatch.setattr(eventqueue, "claim_fresh_for", lambda ev: None)
    monkeypatch.setattr(eventstore, "load_index", lambda: json.loads(json.dumps(st.index)))
    monkeypatch.setattr(eventstore, "drop_events", lambda ids, purge=True, why="":
                        st.drops.append((list(ids), purge)) or True)
    monkeypatch.setattr(eventstore, "note_missed", lambda rec: st.notes.append(rec) or True)
    for k in ("log", "last", "missed"):
        monkeypatch.setitem(eventsim._running, k, [] if k != "last" else None)
    st.t0 = t0
    return st


def test_reopened_episode_miss_is_noted_never_purged(missing_ef5):
    b, sb = _ev(G1, 10)
    trig = {"gauge": G1, "obs_m3s": 50.0}
    # an ordinary episode re-run that misses is still disowned (guard as before)
    missing_ef5.index = {b: sb}
    out = eventsim.run_one(G1, t0=missing_ef5.t0, trigger=trig, episode_id=b)
    assert out["skipped"] == "crest_missed" and not out["published"]
    assert missing_ef5.drops == [([b], True)]
    assert missing_ef5.notes[-1]["reopened"] is False
    # reopened — stamped in the index (no publish since the reopen) or carried
    # in the trigger (the tick's reopen run; manifests after a publish): the
    # finalized record stays, the miss is only noted
    for entry, tr in (({**sb, "reopened": "2026-11-03T10:00:00Z"}, trig),
                      (sb, {**trig, "reopened": True})):
        missing_ef5.index, missing_ef5.drops = {b: entry}, []
        out = eventsim.run_one(G1, t0=missing_ef5.t0, trigger=tr, episode_id=b)
        assert out["skipped"] == "crest_missed" and not out["published"]
        assert missing_ef5.drops == []
        assert missing_ef5.notes[-1]["reopened"] is True
        assert missing_ef5.notes[-1]["event"] == b
    assert len(eventsim._running["missed"]) == 3

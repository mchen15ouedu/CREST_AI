"""hf_data.eventsim.hourly_tick: one flood stays ONE episode. A gauge that
detection flags again keeps its active episode going, and a re-detection
soon after an episode ended reopens it (same id + episode_started anchor)
instead of opening a new event id. No network: detect, the observations,
the nowcast store, the store's hub I/O, the queue and run_one are faked;
mark_ended / reopen_episodes / publish_event run for real against an
in-memory index. run_one's CREST-miss guard runs for real behind a faked
EF5 (a reopened episode's miss is noted, never purged). finalize_episodes
runs for real behind a faked scorer: a reopen or merge landing while it
scores is never overwritten with its stale final."""
from __future__ import annotations

import datetime
import json
import sys
import tempfile
import types

import huggingface_hub
import pytest

import hf_data
from hf_data import eventqueue, eventscore, eventsim, eventstore, nowcaststore

G1, G2 = "09353800", "08353000"
_FINALIZE = eventstore.finalize_episodes      # the tick fixture fakes it


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
                               finalized=[], commits=[], written=[], fail=False,
                               detect_budget=[], stored={}, conflicts=0)

    def detect(max_events=1):
        st.detect_budget.append(max_events)
        return [{"gauge": g, "lat": 35.0, "lon": -110.0, "area_km2": 1000.0,
                 "obs_m3s": 60.0, "hotspot_score": 1.0} for g in st.picks][:max_events]

    class FakeApi:
        def repo_info(self, *a, **k):
            return types.SimpleNamespace(sha=f"rev{len(st.commits)}")

        def create_commit(self, repo_id=None, repo_type=None, operations=(),
                          commit_message="", parent_commit=None):
            if st.fail:
                raise OSError("hub unavailable")
            if st.conflicts:                       # another writer committed first
                st.conflicts -= 1
                raise _hub_http(412)
            st.commits.append(commit_message)
            st.written.append(sorted(op.path_in_repo for op in operations))
            for op in operations:
                if hasattr(getattr(op, "path_or_fileobj", None), "getvalue"):
                    st.stored[op.path_in_repo] = op.path_or_fileobj.getvalue()
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
    monkeypatch.setattr(eventstore, "load_index", lambda revision=None: json.loads(json.dumps(st.idx)))
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


def test_reopen_failure_is_told_apart_from_refusal(tick):
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    o, so = _ev(G2, 1)
    # FAILED -> None: the fresh index read came back empty (load_index gives
    # {} on any error) or the commit raised; nothing changed on the store
    tick.idx = {}
    assert eventstore.reopen_episodes([b]) is None
    tick.idx, tick.fail = {b: sb}, True
    assert eventstore.reopen_episodes([b]) is None
    assert tick.idx[b]["status"] == "ended" and tick.commits == []
    # REFUSED -> []: pinned, pins unreadable, or gone from the index since
    tick.fail = False
    for pins, idx in ((frozenset({b}), {b: sb}), (None, {b: sb}),
                      (frozenset(), {o: so})):
        tick.pins, tick.idx = pins, idx
        assert eventstore.reopen_episodes([b]) == []
    assert tick.commits == []


def test_failed_reopen_still_tells_refused_ids_apart(tick):
    # the ids a reopen refuses (pinned, gone from the fresh index) land in
    # `refused` even when it fails, so the caller never holds them back
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    c, sc = _ev(G2, 10, status="ended", ended_h_ago=3)
    tick.pins = frozenset({b})
    # only pinned ids: refused before the index is read, never a failure
    tick.idx, ref = {}, set()
    assert eventstore.reopen_episodes([b], refused=ref) == [] and ref == {b}
    # the index read fails: the pinned one is refused, the other failed
    ref, sup = set(), set()
    assert eventstore.reopen_episodes([b, c], superseded=sup, refused=ref) is None
    assert ref == {b} and sup == set()
    # the commit raises: an id gone from the fresh index is refused, as is a
    # pinned one; the rest failed
    tick.pins, tick.idx, tick.fail, ref = frozenset(), {c: sc}, True, set()
    assert eventstore.reopen_episodes([b, c], refused=ref) is None and ref == {b}
    tick.pins, tick.idx, ref = frozenset({b}), {b: sb, c: sc}, set()
    assert eventstore.reopen_episodes([b, c], refused=ref) is None and ref == {b}
    assert tick.idx[c]["status"] == "ended" and tick.commits == []


def test_failed_reopen_starts_nothing_and_retries(tick, monkeypatch):
    # a failed reopen is not a refusal: a new id would fragment the flood
    # just like a superseded one, so nothing starts and the next tick retries
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    tick.idx = {b: sb}
    tick.picks = [G1]
    detect = eventsim.detect

    def detect_then_the_index_read_fails(max_events=1):
        tick.idx = {}
        return detect(max_events)

    monkeypatch.setattr(eventsim, "detect", detect_then_the_index_read_fails)
    eventsim.hourly_tick()
    assert tick.runs == [] and tick.commits == []
    monkeypatch.setattr(eventsim, "detect", detect)
    tick.idx, tick.fail = {b: sb}, True                 # the commit raises
    eventsim.hourly_tick()
    assert tick.runs == [] and tick.idx[b]["status"] == "ended"
    tick.fail = False                                   # next tick: it goes through
    eventsim.hourly_tick()
    assert [r["episode"] for r in tick.runs] == [b]
    assert tick.idx[b]["status"] == "active" and "final" not in tick.idx[b]
    assert tick.commits == [f"episodes reopened: {b}"]


def test_failed_reopen_still_starts_a_new_id_where_it_refuses(tick, monkeypatch):
    # a pinned episode starts a new id — so does one gone from the fresh
    # index — also when the reopen fails for the episode beside it
    b, sb = _ev(G1, 10, status="ended", ended_h_ago=3)
    c, sc = _ev(G2, 10, status="ended", ended_h_ago=3)
    tick.picks = [G1, G2]
    detect = eventsim.detect
    for pins, fresh, fail in (
            (frozenset({b}), {}, False),               # the index read fails
            (frozenset({b}), {b: sb, c: sc}, True),    # the commit raises
            (frozenset(), {c: sc}, True)):             # ...b left the index
        def detect_then_the_store_moves(max_events=1, fresh=fresh):
            tick.idx = json.loads(json.dumps(fresh))
            return detect(max_events)

        monkeypatch.setattr(eventsim, "detect", detect_then_the_store_moves)
        tick.pins, tick.fail, tick.runs = pins, fail, []
        tick.idx = {b: dict(sb), c: dict(sc)}
        eventsim.hourly_tick()
        # b's gauge opens a new id; c's reopen failed: nothing, retried
        assert [(r["gauge"], r["episode"]) for r in tick.runs] == [(G1, None)]
        assert tick.commits == []


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
    monkeypatch.setattr(eventstore, "load_index", lambda revision=None: json.loads(json.dumps(st.index)))
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


_STAMP = "%Y-%m-%dT%H:%M:%SZ"


def _hub_404(path):
    """What hf_hub_download raises for a file not on the Hub (a real 404):
    RemoteEntryNotFoundError on huggingface_hub 1.x, EntryNotFoundError
    itself on 0.x (the Spaces pin) — an OSError on both."""
    from huggingface_hub import errors
    if not hasattr(errors, "RemoteEntryNotFoundError"):
        return errors.EntryNotFoundError(f"404 {path}")
    import httpx
    return errors.RemoteEntryNotFoundError(f"404 {path}", response=httpx.Response(
        404, request=httpx.Request("GET", f"https://huggingface.co/{path}")))


def _hub_http(code):
    """An HfHubHTTPError carrying status `code` (409/412: another writer
    committed first), on huggingface_hub 0.x and 1.x."""
    from huggingface_hub.utils import HfHubHTTPError
    try:
        import httpx
        return HfHubHTTPError(f"{code}", response=httpx.Response(
            code, request=httpx.Request("POST", "https://huggingface.co/api/commit")))
    except Exception:
        import requests
        resp = requests.Response()
        resp.status_code = code
        return HfHubHTTPError(f"{code}", response=resp)


@pytest.fixture
def fin(tick, monkeypatch, tmp_path):
    """finalize_episodes for real on the tick fixture's store: the manifest
    is faked, hub downloads fail (no maxdepth stats) except scorecard.json,
    served from `.card` (None: not on the Hub yet, a real 404 — the default;
    an exception: raised; else the stored JSON), and the scorer calls
    `.during(gauge)` — what lands while a finalize scores. `.stored` holds
    what the commits wrote."""
    tick.during, tick.scored, tick.logs, tick.card = None, [], [], None
    t = _h(_now().replace(minute=0, second=0, microsecond=0)
           - datetime.timedelta(hours=2))

    def no_hub(repo_id, filename, *a, **k):
        if filename != f"{eventstore.PREFIX}/scorecard.json":
            raise OSError("offline")
        if tick.card is None:
            raise _hub_404(filename)
        if isinstance(tick.card, BaseException):
            raise tick.card
        p = tmp_path / "scorecard.json"
        p.write_text(json.dumps(tick.card), encoding="utf-8")
        return str(p)

    def score_episode(gid, start, end, log=print):
        tick.scored.append(gid)
        if tick.during:
            tick.during(gid)
        return {"n_issues": 6, "obs_peak_m3s": 61.0, "obs_peak_t": t,
                "nse_h1": 0.8, "nse_h6": 0.5, "leads": {"6": {"peak_ratio": 0.9}}}

    monkeypatch.setattr(eventstore, "finalize_episodes", _FINALIZE)
    monkeypatch.setattr(eventstore, "_prior_manifest", lambda ev: {
        "frames": [{"t": t, "file": "depth_x.tif"}], "hydro": []})
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_hub)
    monkeypatch.setattr(eventscore, "score_episode", score_episode)
    return tick


def _finalize(fin, ids):
    return eventstore.finalize_episodes(ids, log=fin.logs.append)


def _files(*paths):
    return sorted(f"{eventstore.PREFIX}/{p}" for p in paths)


def test_finalize_writes_its_final_and_keeps_the_entry(fin):
    # reopened long before this pass began, and merged by the script: the
    # final lands; 'merged', the stamp and the rest of the entry stay
    b, sb = _ev(G1, 30, status="ended", final=False)
    sb["reopened"] = (_now() - datetime.timedelta(hours=20)).strftime(_STAMP)
    sb["merged"] = {"n": 2, "ids": [b, f"2026110100_{G1}"],
                    "first_start": sb["t0"], "last_end": sb["t_end"]}
    fin.idx = {b: sb}
    assert _finalize(fin, [b]) == 1
    s = fin.idx[b]
    assert s["status"] == "ended" and s["final"]["nse_h6"] == 0.5
    assert {k: v for k, v in s.items() if k != "final"} == sb
    assert fin.written == [_files(f"{b}/final.json", "index.json", "scorecard.json")]
    assert _finalize(fin, [b]) == 0 and fin.scored == [G1]       # has a final now


def test_finalize_starts_a_scorecard_only_where_there_is_none(fin):
    # the running card gains this pass's scores beside every other gauge's;
    # a fresh card only on a real 404 (nothing there yet)
    card_key = f"{eventstore.PREFIX}/scorecard.json"
    other = {G2: [{"event_id": f"2026103000_{G2}", "nse_h6": 0.7}]}
    b, sb = _ev(G1, 10, status="ended", final=False)
    for card in (None, other):
        fin.idx, fin.card, fin.written, fin.stored = {b: dict(sb)}, card, [], {}
        assert _finalize(fin, [b]) == 1
        assert fin.written == [_files(f"{b}/final.json", "index.json", "scorecard.json")]
        got = json.loads(fin.stored[card_key])
        assert {g: r for g, r in got.items() if g != G1} == (card or {})
        assert [r["event_id"] for r in got[G1]] == [b]
    # any other read error — the Hub down (also an OSError on 1.x, like the
    # 404), an offline cache miss (a subclass of EntryNotFoundError), a
    # corrupt card — leaves scorecard.json out of the commit: a card of this
    # pass's gauges alone would wipe every other gauge's track record. The
    # episode's final.json and index entry are written all the same.
    from huggingface_hub import errors
    for card in (OSError("hub unavailable"),
                 errors.LocalEntryNotFoundError("cannot reach the Hub"), ["corrupt"]):
        fin.idx, fin.card, fin.written, fin.stored = {b: dict(sb)}, card, [], {}
        fin.logs = []
        assert _finalize(fin, [b]) == 1
        assert fin.written == [_files(f"{b}/final.json", "index.json")]
        assert card_key not in fin.stored
        assert fin.idx[b]["status"] == "ended" and fin.idx[b]["final"]["nse_h6"] == 0.5
        assert any("scorecard.json" in m for m in fin.logs)


def test_finalize_never_re_ends_an_episode_reopened_meanwhile(fin):
    b, sb = _ev(G1, 10, status="ended", final=False)

    def republish():                    # a worker publish drops the stamp
        fin.idx[b].pop("reopened")
        fin.idx[b]["generated"] = _now().strftime(_STAMP)

    landings = {
        # a tick reopens the episode while this pass scores it
        "active again": lambda g: eventstore.reopen_episodes([b]),
        # ...and a later tick ends it again: that end's own finalize scores it
        "reopened 20": lambda g: (eventstore.reopen_episodes([b]),
                                  eventstore.mark_ended([b])),
        # ...with a publish of the reopened episode in between
        "re-published": lambda g: (eventstore.reopen_episodes([b]), republish(),
                                   eventstore.mark_ended([b])),
        # the fresh index unreadable at commit: never an emptied index.json
        "left the index": lambda g: setattr(fin, "idx", {}),
    }
    for why, during in landings.items():
        fin.idx, fin.commits, fin.during = {b: dict(sb)}, [], during
        assert _finalize(fin, [b]) == 0
        assert "final" not in fin.idx.get(b, {})
        assert not [c for c in fin.commits if c.startswith("episode final")]
        # an empty fresh index is an unreadable one: nothing is committed
        want = ("finalize: index.json unreadable" if why == "left the index"
                else f"finalize {b}: not written — {why}")
        assert fin.logs[-1].startswith(want)
    # one stale episode does not hold back another in the same pass
    c, sc = _ev(G2, 8, status="ended", final=False)
    fin.idx, fin.written = {c: sc, b: dict(sb)}, []
    fin.during = lambda g: g == G1 and eventstore.reopen_episodes([b])
    assert _finalize(fin, [b, c]) == 1
    assert fin.idx[b]["status"] == "active" and "final" not in fin.idx[b]
    assert fin.idx[c]["status"] == "ended" and fin.idx[c]["final"]
    assert fin.written[-1] == _files(f"{c}/final.json", "index.json", "scorecard.json")


def test_finalize_keeps_the_merge_scripts_contract(fin):
    t, st = _ev(G1, 6, status="ended", final=False)      # the merge target
    m, sm = _ev(G1, 30, status="ended", final=False)     # an older member
    # an absorbed member (merged_into) is never scored nor written
    fin.idx = {t: st, m: {**sm, "merged_into": t}}
    assert _finalize(fin, [m]) == 0 and fin.scored == [] and fin.commits == []
    # absorbed while this pass scored it: not written either
    fin.idx = {t: dict(st), m: dict(sm)}
    fin.during = lambda g: fin.idx[m].update(merged_into=t)
    assert _finalize(fin, [m]) == 0
    assert "final" not in fin.idx[m] and fin.commits == []
    # the target merged while this pass scored its own window: the script's
    # finalize scores the merged window, so this pre-merge final is dropped
    merged = {"n": 2, "ids": [t, m], "first_start": sm["t0"], "last_end": st["t_end"]}
    fin.idx = {t: dict(st), m: {**sm, "merged_into": t}}
    fin.during = lambda g: fin.idx[t].update(merged=merged,
                                             generated=_now().strftime(_STAMP))
    assert _finalize(fin, [t]) == 0 and "final" not in fin.idx[t]
    assert fin.logs[-1] == f"finalize {t}: not written — merged since this pass began"
    # the script's own finalize after its merge commit: written, 'merged' kept
    fin.during = None
    assert _finalize(fin, [t]) == 1
    s = fin.idx[t]
    assert s["final"] and s["status"] == "ended" and s["merged"] == merged
    assert "final" not in fin.idx[m]


def test_finalize_rereads_and_retries_when_the_store_moved(fin, monkeypatch):
    # another writer (the runner or the merger process) commits between the
    # read and the commit: 412 -> re-read at the new head, write once
    monkeypatch.setattr(eventstore.time, "sleep", lambda s: None)
    b, sb = _ev(G1, 30, status="ended", final=False)
    fin.idx = {b: sb}
    fin.conflicts = 1
    assert _finalize(fin, [b]) == 1
    assert fin.conflicts == 0 and len(fin.commits) == 1
    assert fin.idx[b]["final"]["nse_h6"] == 0.5
    assert any("moved under us" in ln for ln in fin.logs)
    # a store that keeps moving: nothing written, no exception
    b2, sb2 = _ev(G2, 30, status="ended", final=False)
    fin.idx = {b2: sb2}
    fin.conflicts = 99
    assert _finalize(fin, [b2]) == 0 and "final" not in fin.idx[b2]

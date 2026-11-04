"""hf_data.eventlist: the public 2-D inundation list is a view of the index —
fragments of one flood fold into one entry, entries the merge script
absorbed (merged_into) are never listed, floods ended > 30 days ago roll off
(a retrospective: 30 days after its request), active (and, when asked,
pinned) ones never do. Pure; no network."""
import copy
import json
from datetime import datetime, timedelta

from hf_data import eventlist

NOW = datetime(2026, 11, 3, 21)


def _h(dt):
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def _ev(start, gauge, status="ended", depth=None, dur_h=3, final=True):
    """(event_id, summary) shaped like events/index.json: the episode starts
    at `start`, its final product ends `dur_h` later."""
    eid = f"{start:%Y%m%d%H}_{gauge}"
    s = {"t0": _h(start), "sim_start": _h(start - timedelta(hours=12)),
         "episode_started": _h(start), "t_end": _h(start + timedelta(hours=12)),
         "trigger": {"gauge": gauge}, "gauge": gauge, "status": status}
    if final and status == "ended":
        s["final"] = {"ended": _h(start + timedelta(hours=dur_h)), "peak_depth_m": depth}
    return eid, s


def _index(*evs):
    """Newest-first insertion order, like the store's index."""
    return dict(sorted(evs, key=lambda kv: kv[0], reverse=True))


def _view(idx, **kw):
    kw.setdefault("max_age_d", 30)
    kw.setdefault("merge_gap_h", 12)
    return eventlist.list_view(idx, NOW, **kw)


def _check_counts(idx, out, info):
    assert (len(out) + info["merged_fragments"] + info["hidden_old"] + info["absorbed"]
            == info["total"] == len(idx))


def test_active_fragment_is_representative():
    # 09353800-style: hourly-spaced episodes of one flood, the newest active
    t = datetime(2026, 10, 30, 18)
    frags = [_ev(t + timedelta(hours=h), "09353800", depth=0.0, dur_h=2)
             for h in (0, 2, 4, 6, 7, 9)]
    act = _ev(t + timedelta(hours=10), "09353800", status="active")
    other = _ev(t + timedelta(hours=5), "08353000", depth=1.0)
    idx = _index(*frags, act, other)
    before = dict((k, dict(v)) for k, v in idx.items())
    out, info = _view(idx)
    assert list(out) == [act[0], other[0]]                 # newest-first kept
    m = out[act[0]]["merged"]
    assert m["n"] == 7 and m["ids"][0] == act[0]
    assert m["ids"] == sorted(m["ids"], reverse=True)       # newest first
    assert m["first_start"] == _h(t) and m["last_end"] == act[1]["t_end"]
    assert out[act[0]]["status"] == "active" and out[act[0]]["t0"] == act[1]["t0"]
    assert "merged" not in out[other[0]]                     # other gauge untouched
    assert info["merged_fragments"] == 6 and info["hidden_old"] == 0
    _check_counts(idx, out, info)
    assert idx == before                                     # input never modified


def test_ended_group_picks_deepest_then_newest():
    t = datetime(2026, 10, 20, 0)
    a = _ev(t, "05470000", depth=3.78)
    b = _ev(t + timedelta(hours=4), "05470000", depth=5.39)   # deepest
    c = _ev(t + timedelta(hours=10), "05470000", depth=2.0)   # newest
    out, info = _view(_index(a, b, c))
    assert list(out) == [b[0]]
    assert out[b[0]]["final"]["peak_depth_m"] == 5.39
    assert out[b[0]]["merged"]["ids"] == [c[0], b[0], a[0]]
    # all tied (0 m, the 09353800 fragments) -> the newest
    z = [_ev(t + timedelta(hours=h), "09353800", depth=0.0) for h in (0, 3, 6)]
    out, _ = _view(_index(*z))
    assert list(out) == [z[-1][0]]


def test_gap_splits_separate_floods():
    t = datetime(2026, 10, 27, 18)
    first = [_ev(t + timedelta(hours=h), "09353800", depth=0.0, dur_h=2) for h in (0, 4, 5, 6)]
    # starts 56 h after the first flood ended: a new flood, its own entry
    second = _ev(t + timedelta(hours=64), "09353800", depth=0.0)
    out, info = _view(_index(*first, second))
    assert set(out) == {second[0], first[-1][0]}
    assert "merged" not in out[second[0]]
    assert out[first[-1][0]]["merged"]["n"] == 4
    # gap 13 h > 12 h stays apart; 12 h exactly joins
    x = _ev(datetime(2026, 10, 25, 0), "07000000", dur_h=3)
    y13 = _ev(datetime(2026, 10, 25, 16), "07000000")
    y12 = _ev(datetime(2026, 10, 25, 15), "07000000")
    assert len(_view(_index(x, y13))[0]) == 2
    assert len(_view(_index(x, y12))[0]) == 1
    # merging off
    out, info = _view(_index(*first), merge_gap_h=-1)
    assert len(out) == 4 and info["merged_fragments"] == 0


def test_old_ended_hidden_active_kept():
    old = _ev(NOW - timedelta(days=40), "06803513", depth=5.5)
    recent = _ev(NOW - timedelta(days=5), "09337500", depth=20.8)
    old_active = _ev(NOW - timedelta(days=45), "09494000", status="active")
    idx = _index(old, recent, old_active)
    out, info = _view(idx)
    assert set(out) == {recent[0], old_active[0]}
    assert info["hidden_old"] == 1 and info["cutoff"] == _h(NOW - timedelta(days=30))
    _check_counts(idx, out, info)
    # ended just inside the window: kept (end, not start, decides)
    edge = _ev(NOW - timedelta(days=30, hours=2), "01000000", dur_h=4)
    assert edge[0] in _view(_index(edge))[0]
    # no final: t_end decides; an unparsable id/summary is never hidden
    nof = _ev(NOW - timedelta(days=40), "02000000", final=False)
    assert _view(_index(nof))[0] == {}
    out, _ = _view({"junk": {"status": "ended"}})
    assert list(out) == ["junk"]
    # roll-off off
    assert len(_view(idx, max_age_d=0)[0]) == 3


def test_pinned_exemption():
    old = _ev(NOW - timedelta(days=40), "06803513", depth=5.5)
    idx = _index(old)
    assert _view(idx, pinned=frozenset())[0] == {}
    assert old[0] in _view(idx, pinned=frozenset({old[0]}))[0]
    # pinned set unreadable (pinned_ids() -> None): nothing is hidden
    assert old[0] in _view(idx, pinned=None)[0]


def test_old_fragment_of_recent_group_stays_folded():
    # first fragment ended 31 days ago, the flood went on to 29 days ago:
    # the group is recent, so the old fragment must not resurface or vanish
    t = NOW - timedelta(days=31, hours=6)
    frags = [_ev(t + timedelta(hours=12 * k), "09401260", depth=float(k), dur_h=12)
             for k in range(5)]
    idx = _index(*frags)
    out, info = _view(idx)
    assert list(out) == [frags[-1][0]]                     # deepest = newest here
    assert out[frags[-1][0]]["merged"]["ids"][-1] == frags[0][0]
    assert info["hidden_old"] == 0 and info["merged_fragments"] == 4
    # the whole group older than the cutoff: hidden together, counted as events
    old = [_ev(NOW - timedelta(days=50, hours=-3 * k), "09401260", depth=1.0) for k in range(3)]
    idx = _index(*old, _ev(NOW - timedelta(days=1), "09401260", depth=2.0))
    out, info = _view(idx)
    assert len(out) == 1 and info["hidden_old"] == 3 and info["merged_fragments"] == 0
    _check_counts(idx, out, info)


def test_group_keeps_position_of_first_listed_member():
    # the store's order is publish order, not t0: a group sits where its
    # first-listed member was, and other entries keep their relative order
    t = datetime(2026, 10, 20, 0)
    a1, a2 = _ev(t, "A0000001", depth=9.0), _ev(t + timedelta(hours=3), "A0000001", depth=1.0)
    b = _ev(t + timedelta(days=2), "B0000001", depth=1.0)
    c = _ev(t - timedelta(days=2), "C0000001", depth=1.0)
    idx = dict([a2, b, a1, c])
    out, _ = _view(idx)
    assert list(out) == [a1[0], b[0], c[0]]                 # a1 (deepest) at a2's slot


def test_env_values_parsed_defensively(monkeypatch):
    # a bad, NaN or infinite env value falls back to the default instead of
    # failing the import (the endpoint would then list the full index)
    import importlib
    try:
        for bad in ("abc", "nan", "inf", "-inf", ""):
            monkeypatch.setenv("EVENT_LIST_MAX_AGE_D", bad)
            monkeypatch.setenv("EVENT_LIST_MERGE_GAP_H", bad)
            importlib.reload(eventlist)
            assert (eventlist.MAX_AGE_D, eventlist.MERGE_GAP_H) == (30.0, 12.0)
        monkeypatch.setenv("EVENT_LIST_MAX_AGE_D", " 45 ")
        monkeypatch.setenv("EVENT_LIST_MERGE_GAP_H", "-1")
        importlib.reload(eventlist)
        assert (eventlist.MAX_AGE_D, eventlist.MERGE_GAP_H) == (45.0, -1.0)
    finally:
        monkeypatch.undo()
        importlib.reload(eventlist)
    # explicit arguments too; absurd finite ones neither overflow nor hide
    old = _ev(NOW - timedelta(days=40), "06803513", depth=5.5)
    out, info = eventlist.list_view(_index(old), NOW, max_age_d=float("nan"),
                                    merge_gap_h="x")
    assert out == {} and (info["max_age_d"], info["merge_gap_h"]) == (30.0, 12.0)
    out, info = eventlist.list_view(_index(old), NOW, max_age_d=1e300, merge_gap_h=1e300)
    assert old[0] in out and info["cutoff"] is None


def test_retrospective_listed_from_request_time():
    # a backdated reconstruction of a flood that ended 40 days ago, requested
    # 5 days ago (the runner's isoformat stamp): listed until 30 days after
    # the request, however old the flood
    eid, s = _ev(NOW - timedelta(days=40), "09342500", depth=2.0)
    retro = {"label": "Retrospective reconstruction",
             "requested_at": (NOW - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")}
    s["trigger"]["retrospective"] = retro
    out, info = _view(_index((eid, s)))
    assert eid in out and info["hidden_old"] == 0
    retro["requested_at"] = (NOW - timedelta(days=30, hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    out, info = _view(_index((eid, s)))
    assert out == {} and info["hidden_old"] == 1
    retro["requested_at"] = "unknown"                      # unparsable: the end decides
    assert _view(_index((eid, s)))[0] == {}
    # the same flood simulated operationally: the end decides
    assert _view(_index(_ev(NOW - timedelta(days=40), "09342500")))[0] == {}
    # grouped with an old operational fragment: the request keeps the group
    retro["requested_at"] = _h(NOW - timedelta(days=2))
    frag = _ev(NOW - timedelta(days=40, hours=6), "09342500", depth=1.0)
    idx = _index(frag, (eid, s))
    out, info = _view(idx)
    assert list(out) == [eid] and out[eid]["merged"]["n"] == 2
    _check_counts(idx, out, info)


def test_absorbed_entries_and_existing_merge_record():
    # the merge script folded a1, a2 into r: they carry merged_into, r the
    # record (its ids name r itself too)
    t = datetime(2026, 10, 28, 0)
    a1 = _ev(t, "09353800", depth=0.5)
    r = _ev(t + timedelta(hours=2), "09353800", depth=1.5)
    a2 = _ev(t + timedelta(hours=4), "09353800", depth=0.2)
    r[1]["merged"] = {"n": 3, "ids": [a2[0], r[0], a1[0]], "first_start": _h(t),
                      "last_end": a2[1]["final"]["ended"]}
    a1[1]["merged_into"] = a2[1]["merged_into"] = r[0]
    idx = _index(a1, a2, r)
    out, info = _view(idx)
    assert list(out) == [r[0]] and out[r[0]] is idx[r[0]]   # record kept as is
    assert info["absorbed"] == 2 and info["merged_fragments"] == 0
    _check_counts(idx, out, info)
    # a later fragment 8 h after the record's last_end joins: one entry, each
    # id once, the span from the record's start to the fragment's end
    f = _ev(t + timedelta(hours=15), "09353800", depth=1.0)
    idx = _index(a1, a2, r, f)
    out, info = _view(idx)
    assert list(out) == [r[0]]                              # still the deepest
    m = out[r[0]]["merged"]
    assert m["n"] == 4 and m["ids"] == [f[0], a2[0], r[0], a1[0]]
    assert m["first_start"] == _h(t) and m["last_end"] == f[1]["final"]["ended"]
    assert info["absorbed"] == 2 and info["merged_fragments"] == 1
    _check_counts(idx, out, info)
    # the fragment deeper: it represents, carrying r's absorbed ids
    f[1]["final"]["peak_depth_m"] = 3.0
    out, _ = _view(_index(a1, a2, r, f))
    assert list(out) == [f[0]] and out[f[0]]["merged"]["n"] == 4
    f[1]["final"]["peak_depth_m"] = 1.0
    # a chained, a dangling and a circular merged_into: never listed
    a0 = _ev(t - timedelta(hours=1), "09353800", depth=9.0)
    a0[1]["merged_into"] = a1[0]
    x = _ev(t + timedelta(days=1), "07000000", depth=1.0)
    x[1]["merged_into"] = "2026010100_07000000"
    y, z = _ev(t, "06000000"), _ev(t + timedelta(hours=1), "06000000")
    y[1]["merged_into"], z[1]["merged_into"] = z[0], y[0]
    idx = _index(a0, a1, a2, r, f, x, y, z)
    out, info = _view(idx)
    assert list(out) == [r[0]] and info["absorbed"] == 6
    assert a0[0] in out[r[0]]["merged"]["ids"] and out[r[0]]["merged"]["n"] == 5
    _check_counts(idx, out, info)
    # a record counting more episodes than it names: only the unnamed add
    r[1]["merged"] = {"n": 10, "ids": [r[0]]}
    out, _ = _view(_index(a1, a2, r, f))
    assert out[r[0]]["merged"]["n"] == 10 + 1               # (r, a1, a2 + 7) + f


def test_absorbed_active_episode_keeps_group():
    # an old group whose absorbed episode is active again is not rolled off
    t = NOW - timedelta(days=40)
    r, a = _ev(t, "09353800", depth=1.0), _ev(t + timedelta(hours=3), "09353800")
    r[1]["merged"] = {"n": 2, "ids": [a[0], r[0]], "first_start": _h(t),
                      "last_end": a[1]["final"]["ended"]}
    a[1]["merged_into"] = r[0]
    assert _view(_index(r, a))[0] == {}
    a[1]["status"] = "active"
    assert list(_view(_index(r, a))[0]) == [r[0]]


def test_groups_is_the_lists_grouping():
    t = datetime(2026, 10, 30, 18)
    frags = [_ev(t + timedelta(hours=h), "09353800", depth=0.0, dur_h=2) for h in (0, 2, 4)]
    other = _ev(t, "08353000", depth=1.0)
    idx = _index(*frags, other)
    before = json.dumps(idx, sort_keys=True)
    gs = eventlist.groups(idx, 12)
    assert [g["rep"] for g in gs] == [frags[-1][0], other[0]]
    g = gs[0]
    assert g["members"] == g["ids"] == [e for e, _ in reversed(frags)]
    assert g["n"] == 3 and not g["active"]
    assert (g["start"], g["end"]) == (t, t + timedelta(hours=6))
    rec = eventlist.merged_record(g)
    assert rec == {"n": 3, "ids": g["ids"], "first_start": _h(t),
                   "last_end": _h(t + timedelta(hours=6))}
    out, _ = _view(idx)
    assert out[g["rep"]]["merged"] == rec
    assert json.dumps(idx, sort_keys=True) == before        # input never modified
    assert len(eventlist.groups(idx, -1)) == 4
    # a merge script writing what groups() returns leaves the list unchanged,
    # order included, even when the representative was published after the
    # group's first-listed episode (the deepest is the oldest here)
    frags[0][1]["final"]["peak_depth_m"] = 2.0
    mid = _ev(t + timedelta(hours=3), "07000000", depth=1.0)
    idx = _index(*frags, mid, other)
    out, _ = _view(idx)
    assert list(out) == [frags[0][0], mid[0], other[0]]
    stored = _fold(idx)
    out2, info2 = _view(stored)
    assert list(out2) == list(out) and out2 == out
    assert info2["absorbed"] == 2 and info2["merged_fragments"] == 0
    _check_counts(stored, out2, info2)


def _fold(idx):
    """What the store-side merge script does with groups(): the record on
    each representative, merged_into on the rest (on a copy)."""
    stored = copy.deepcopy(idx)
    for g in eventlist.groups(stored, 12):
        if g["n"] > 1:
            stored[g["rep"]]["merged"] = eventlist.merged_record(g)
            for e in g["members"]:
                if e != g["rep"]:
                    stored[e]["merged_into"] = g["rep"]
    return stored


def test_admin_archive_listed_is_what_the_list_shows(monkeypatch):
    """/api/admin/archive's `listed` = shown on the public list: a folded
    fragment and a rolled-off flood are off it although the index holds
    them, as is a stored folder the index never had."""
    monkeypatch.setattr(eventlist, "MAX_AGE_D", 30.0)      # the server's defaults
    monkeypatch.setattr(eventlist, "MERGE_GAP_H", 12.0)
    t = datetime(2026, 10, 30, 18)
    frags = [_ev(t + timedelta(hours=h), "09353800", depth=0.0, dur_h=2) for h in (0, 2, 4)]
    old = _ev(NOW - timedelta(days=40), "06803513", depth=5.5)
    recent = _ev(NOW - timedelta(days=5), "09337500", depth=20.8)
    idx = _index(*frags, old, recent)
    # eventstore.archive_index's shape: listed = "in the index"
    rows = [{"id": e, "listed": True, "t0": s["t0"]} for e, s in idx.items()]
    rows.append({"id": "2026090100_01000000", "listed": False, "t0": None})
    snap = {"events": rows, "n": len(rows), "n_listed": len(idx), "building": False}
    before = copy.deepcopy(snap)
    out = eventlist.mark_listed(snap, idx, NOW)
    assert [r["id"] for r in out["events"] if r["listed"]] == [frags[-1][0], recent[0]]
    assert out["n_listed"] == 2 and out["n"] == len(rows) and out["building"] is False
    assert [r["id"] for r in out["events"]] == [r["id"] for r in rows]   # order kept
    assert [r["t0"] for r in out["events"]] == [r["t0"] for r in rows]
    assert snap == before                                    # never modified
    # the same pins as /api/events: a pinned old flood stays listed
    out = eventlist.mark_listed(snap, idx, NOW, pinned=frozenset({old[0]}))
    assert out["n_listed"] == 3 and next(
        r["listed"] for r in out["events"] if r["id"] == old[0])
    # pins unreadable (pinned_ids() -> None): nothing rolls off, folding stays
    assert eventlist.mark_listed(snap, idx, NOW, pinned=None)["n_listed"] == 3
    # an empty index (load_index's answer to any failure): the snapshot as is
    assert eventlist.mark_listed(snap, {}, NOW) is snap
    assert eventlist.mark_listed(snap, None, NOW) is snap
    assert eventlist.mark_listed({"events": [], "n": 0}, idx, NOW)["n_listed"] == 0

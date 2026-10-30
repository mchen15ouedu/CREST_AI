"""CREST_autocal candidate selection (PI 2026-10-27 west-first + backlog):
pure logic on synthetic missed/index/done/backlog inputs — no network, no EF5."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import sys
import types

import pytest

from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, "autocal"))
import autocal_run as ar                                      # noqa: E402
import hf_data.pipeline  # noqa: E402,F401  (is_west reads WEST_LON; loaded before a test hides hf_data.lakes)

NOW = dt.datetime(2026, 10, 27, 12, 0, 0)
TS = ar.TS

# synthetic gauges: W* west of -105, E* east of it
LONS = {"09000001": -111.0, "09000002": -108.0, "09000003": -112.0, "09000004": -110.0,
        "09000005": -115.0, "07000001": -97.0, "07000002": -90.0, "07000003": -95.0}
# reservoir-affected gauges (STOR_NOR_2009 mm); tests add entries as needed
RESV: dict = {}
_reservoir_stor = ar.reservoir_stor            # the real reader, before the fixture patch
STATE0 = json.loads(json.dumps(ar.state))      # pristine loop state


def _set_module(monkeypatch, name: str, mod):
    """hf_data.<name> -> mod (None: not importable) for `from hf_data import name`."""
    import hf_data
    monkeypatch.setitem(sys.modules, f"hf_data.{name}", mod)
    if mod is None:
        monkeypatch.delattr(hf_data, name, raising=False)
    else:
        monkeypatch.setattr(hf_data, name, mod, raising=False)


def _lakes(monkeypatch, region="west", version="2026-10-28T00:00Z-abc"):
    """A lake module that is live for the west (unless told otherwise)."""
    _set_module(monkeypatch, "lakes", types.SimpleNamespace(
        region=lambda: region, table_version=lambda: version))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(ar, "gauge_lon", lambda g: LONS.get(str(g).zfill(8)))
    RESV.clear()
    monkeypatch.setattr(ar, "reservoir_stor", lambda g: RESV.get(str(g).zfill(8)))
    for k, v in {"MIN_CHECKS": 3, "QUIET_H": 6.0, "COOLDOWN_D": 30.0, "MAX_PER_DAY": 2,
                 "WINDOW_D": 183.0, "STREAK_GAP_H": 3.0, "OVERSIM_RATIO": 3.0,
                 "OVERSIM_MIN_OBS": 1.0, "LAKES_ON": False, "SKIP_TTL_D": 30.0,
                 "SKIP_ERR_TTL_H": 24.0, "SKIP_SUMMARY_H": 24.0, "MAX_SKIP_RUN": 5,
                 "OUTAGE_STRIKES": 4, "OBS_MIN_FRAC": 0.5, "POLL_S": 900.0}.items():
        monkeypatch.setattr(ar, k, v)
    monkeypatch.setattr(ar, "state", json.loads(json.dumps(STATE0)))
    _set_module(monkeypatch, "lakes", None)      # no lake module unless a test adds one
    ar._verdicts.clear()


def _t(h_ago: float) -> str:
    return (NOW - dt.timedelta(hours=h_ago)).strftime(TS)


def miss_rows(gid: str, n: int, last_h_ago: float = 10.0, obs: float = 100.0):
    """n hourly miss rows (one event id each, as the guard writes them)."""
    rows = []
    for i in range(n):
        at = NOW - dt.timedelta(hours=last_h_ago + (n - 1 - i))
        ev = f"{at:%Y%m%d%H}_{gid}"
        rows.append({"event": ev, "gauge": gid, "at": at.strftime(TS),
                     "t0": at.strftime("%Y-%m-%dT%H:00Z"), "frac": 0.1,
                     "sim_peak_m3s": obs * 0.1, "obs_peak_m3s": obs})
    return rows


def manifest(t0: dt.datetime, sim: list, obs: list, tail_sim: float = 0.0) -> dict:
    """Hydro rows ending at t0 (paired), plus a post-t0 forecast tail."""
    n = len(sim)
    rows = [{"time": (t0 - dt.timedelta(hours=n - 1 - i)).strftime("%Y-%m-%d %H:%M"),
             "sim_q": sim[i], "obs_q": obs[i], "precip": 0.0} for i in range(n)]
    rows += [{"time": (t0 + dt.timedelta(hours=k)).strftime("%Y-%m-%d %H:%M"),
              "sim_q": tail_sim, "obs_q": None, "precip": 0.0} for k in (1, 2, 3)]
    return {"t0": t0.strftime("%Y-%m-%dT%H:%MZ"), "hydro": rows}


def index_entry(gid: str, t0: dt.datetime, end_h_after: float = 12.0, gen: str = "g1"):
    return {"gauge": gid, "t0": t0.strftime("%Y-%m-%dT%H:%MZ"),
            "t_end": (t0 + dt.timedelta(hours=end_h_after)).strftime("%Y-%m-%dT%H:%MZ"),
            "status": "ended", "generated": gen}


# --------------------------------------------------------------------------- #
# west-first ordering
# --------------------------------------------------------------------------- #
def test_west_miss_beats_bigger_eastern_miss():
    missed = miss_rows("07000001", 20, obs=900.0) + miss_rows("09000001", 3, obs=5.0)
    rec, why = ar.pick_candidate(missed, {}, [], NOW)
    assert rec["gauge"] == "09000001" and rec["kind"] == "miss" and rec["west"] is True
    assert "2 eligible" in why and "1 western" in why


def test_within_region_existing_keys_kept():
    missed = (miss_rows("09000001", 3, obs=50.0) + miss_rows("09000002", 6, obs=10.0)
              + miss_rows("09000003", 6, obs=40.0))
    rec, _ = ar.pick_candidate(missed, {}, [], NOW)
    assert rec["gauge"] == "09000003"            # most checks, then bigger obs peak
    east = miss_rows("07000001", 3, obs=50.0) + miss_rows("07000002", 5, obs=1.0)
    rec, _ = ar.pick_candidate(east, {}, [], NOW)
    assert rec["gauge"] == "07000002" and rec["west"] is False


def _order_events(missed, idx, mf, done=()):
    """(kind, gauge) of every eligible event in pick order."""
    done, out = list(done), []
    while True:
        rec, _ = ar.pick_candidate(missed, idx, done, NOW, manifest=mf.get)
        if rec is None:
            return out
        out.append((rec["kind"], rec["gauge"]))
        # mark handled without a ran=True record (keeps the cap and cooldown out)
        done.append({"gauge": rec["gauge"], "event": rec["event"], "ran": False,
                     "at": _t(1)})


def test_west_oversim_before_west_miss_east_has_misses_only():
    t0 = NOW - dt.timedelta(hours=30)
    idx = {"W": index_entry("09000004", t0), "E": index_entry("07000003", t0)}
    mf = {"W": manifest(t0, [1, 5, 40], [1, 2, 4]),          # 10x, west
          "E": manifest(t0, [1, 5, 90], [1, 2, 4])}          # 22x, east: no trigger
    missed = (miss_rows("07000001", 10, obs=500.0)            # east miss
              + miss_rows("09000001", 12, obs=900.0))         # west miss, 12 checks
    assert _order_events(missed, idx, mf) == [
        ("oversim", "09000004"), ("miss", "09000001"),         # west: oversim first
        ("miss", "07000001")]                                  # east: previous behaviour
    assert ar._event_key({"west": False, "kind": "miss", "n_checks": 3,
                          "obs_peak_m3s": 1.0})[:2] == (True, True)


# --------------------------------------------------------------------------- #
# reservoir hold (PI 2026-10-27): no trigger calibrates a WESTERN >= 10 mm
# gauge until AUTOCAL_LAKES_ON=1; eastern gauges are never held
# --------------------------------------------------------------------------- #
def test_western_reservoir_gauges_held_from_event_triggers(monkeypatch):
    t0 = NOW - dt.timedelta(hours=30)
    idx = {"W": index_entry("09000004", t0), "E": index_entry("07000003", t0)}
    mf = {"W": manifest(t0, [1, 5, 40], [1, 2, 4]), "E": manifest(t0, [1, 5, 90], [1, 2, 4])}
    missed = miss_rows("09000001", 12, obs=900.0) + miss_rows("07000001", 10, obs=500.0)
    RESV.update({"09000004": 267.0, "09000001": 11.0, "07000003": 40.0})
    held = []
    rec, why = ar.pick_candidate(missed, idx, [], NOW, manifest=mf.get, held=held)
    assert (rec["kind"], rec["gauge"]) == ("miss", "07000001")   # east: miss first
    assert "2 western reservoir-gauge event(s) held until AUTOCAL_LAKES_ON=1" in why
    assert sorted((h["kind"], h["gauge"], h["stor_mm"], h["west"]) for h in held) == [
        ("miss", "09000001", 11.0, True), ("oversim", "09000004", 267.0, True)]
    # the eastern gauges keep their previous behaviour: a miss is eligible
    # (reservoir or not), an over-prediction never triggers there
    assert _order_events(missed, idx, mf) == [("miss", "07000001")]
    # nothing at all when every candidate is held: falls through to the backlog
    held = []
    rec, why = ar.pick_candidate(missed[:12], {"W": idx["W"]}, [], NOW, manifest=mf.get,
                                 held=held)
    assert rec is None and len(held) == 2
    assert why.startswith("no eligible event; 2 western reservoir-gauge event(s) held")
    # the held list stays current while the daily cap is reached
    done = [{"gauge": "x", "ran": True, "at": _t(1)}, {"gauge": "y", "ran": True, "at": _t(2)}]
    held = []
    rec, why = ar.pick_candidate(missed, idx, done, NOW, manifest=mf.get, held=held)
    assert rec is None and why.startswith("daily cap") and len(held) == 2
    # lakes on (flag + live module): the hold lifts, the western over-prediction leads
    monkeypatch.setattr(ar, "LAKES_ON", True)
    _lakes(monkeypatch)
    held = []
    rec, _ = ar.pick_candidate(missed, idx, [], NOW, manifest=mf.get, held=held)
    assert (rec["kind"], rec["gauge"]) == ("oversim", "09000004") and held == []


def test_held_stor_west_only(monkeypatch):
    RESV.update({"09000001": 11.0, "07000003": 40.0})
    assert ar.held_stor("09000001") == 11.0      # west, reservoir -> held
    assert ar.held_stor("07000003") is None      # east, reservoir -> never held
    assert ar.held_stor("09000002") is None      # west, dam-free
    assert ar.held_stor("05999999") is None      # unknown longitude counts as east
    monkeypatch.setattr(ar, "LAKES_ON", True)
    assert ar.held_stor("09000001") == 11.0      # flag alone: no lake module here
    _lakes(monkeypatch)
    assert ar.held_stor("09000001") is None


def test_note_held_state_and_log(monkeypatch):
    lines = []
    monkeypatch.setattr(ar, "log", lines.append)
    monkeypatch.setitem(ar.state, "held_reservoir", [])
    h = [{"gauge": "10171000", "event": "2026101202_10171000", "kind": "oversim",
          "west": True, "stor_mm": 266.53}]
    ar._note_held(h)
    ar._note_held(list(h))                       # unchanged set: no second log line
    assert ar.state["held_reservoir"] == h and len(lines) == 1
    assert "10171000 (oversim, 267 mm)" in lines[0]


def test_reservoir_table_reader(monkeypatch):
    monkeypatch.setattr(ar, "_resv", None)
    assert _reservoir_stor("10171000") == pytest.approx(266.53)   # Jordan R. (Utah Lake)
    assert _reservoir_stor("09371492") is None                     # Mud Creek: dam-free
    monkeypatch.setattr(ar, "_resv", None)
    monkeypatch.setattr(ar, "RESERVOIR_PATH", os.path.join(ROOT, "no", "such.json"))
    with pytest.raises(RuntimeError):                              # never fails open
        _reservoir_stor("10171000")
    monkeypatch.setattr(ar, "_resv", None)


def test_committed_backlog_tiers_match_reservoir_table(monkeypatch):
    monkeypatch.setattr(ar, "_resv", None)
    data = ar.load_backlog(os.path.join(ROOT, "hf_data", "data", "autocal_west_backlog.json"))
    for e in data:
        assert (_reservoir_stor(e["gauge"]) is not None) == (e["tier"] == 4), e
    monkeypatch.setattr(ar, "_resv", None)


def test_unknown_longitude_sorts_with_the_east():
    missed = miss_rows("05999999", 9, obs=900.0) + miss_rows("09000002", 3, obs=1.0)
    rec, _ = ar.pick_candidate(missed, {}, [], NOW)
    assert rec["gauge"] == "09000002"


# --------------------------------------------------------------------------- #
# over-prediction detection
# --------------------------------------------------------------------------- #
def test_oversim_stats_pairs_only_up_to_t0():
    t0 = NOW - dt.timedelta(hours=20)
    m = manifest(t0, [2.0, 61.8, 10.0, None], [0.05, 2.06, None, 1.0], tail_sim=500.0)
    m["hydro"].append({"time": (t0 + dt.timedelta(hours=4)).strftime("%Y-%m-%d %H:%M"),
                       "sim_q": 999.0, "obs_q": 0.5, "precip": 0.0})
    st = ar.oversim_stats(m)
    # rows after t0 (even paired ones) and the unpaired rows are ignored
    assert st["n_paired"] == 2
    assert st["sim_peak_m3s"] == 61.8 and st["obs_peak_m3s"] == 2.06
    assert st["ratio"] == pytest.approx(61.8 / 2.06)


def test_oversim_stats_none_without_pairs_or_obs():
    t0 = NOW - dt.timedelta(hours=20)
    assert ar.oversim_stats(manifest(t0, [5, 6], [None, None])) is None
    assert ar.oversim_stats(manifest(t0, [5, 6], [0.0, 0.0])) is None
    assert ar.oversim_stats({"t0": "x", "hydro": []}) is None


def test_oversim_candidate_rules():
    t0 = NOW - dt.timedelta(hours=40)
    idx = {
        "OK": index_entry("09000001", t0),                                  # 10x, quiet
        "LOW": index_entry("09000002", t0),                                 # 2x
        "TINY": index_entry("09000003", t0),                                # obs 0.5 m3/s
        "FRESH": index_entry("09000004", NOW - dt.timedelta(hours=8), 4),   # ended 4 h ago
    }
    mf = {"OK": manifest(t0, [1, 30], [1, 3]),
          "LOW": manifest(t0, [1, 6], [1, 3]),
          "TINY": manifest(t0, [1, 9], [0.1, 0.5]),
          "FRESH": manifest(NOW - dt.timedelta(hours=8), [1, 90], [1, 3])}
    got = ar.oversim_candidates(idx, [], NOW, manifest=mf.get)
    assert [c["event"] for c in got] == ["OK"]
    c = got[0]
    assert c["kind"] == "oversim" and c["west"] and c["ratio"] == pytest.approx(10.0)
    # handled (gauge, event) and the gauge cooldown both exclude it
    done = [{"gauge": "09000001", "event": "OK", "ran": False, "at": _t(1)}]
    assert ar.oversim_candidates(idx, done, NOW, manifest=mf.get) == []
    done = [{"gauge": "09000001", "event": "OTHER", "ran": True, "at": _t(24 * 10)}]
    assert ar.oversim_candidates(idx, done, NOW, manifest=mf.get) == []


def test_oversim_event_older_than_last_calibration_is_skipped():
    t0 = NOW - dt.timedelta(days=60)
    idx = {"OLD": index_entry("09000001", t0)}
    mf = {"OLD": manifest(t0, [1, 30], [1, 3])}
    done = [{"gauge": "09000001", "event": "X", "ran": True, "at": _t(24 * 40)}]
    assert ar.oversim_candidates(idx, done, NOW, manifest=mf.get) == []


def test_oversim_verdict_memoised_per_publish(monkeypatch):
    t0 = NOW - dt.timedelta(hours=40)
    calls = []

    class _ES:
        @staticmethod
        def load_manifest(ev):
            calls.append(ev)
            return manifest(t0, [1, 30], [1, 3])
    import hf_data
    monkeypatch.setitem(sys.modules, "hf_data.eventstore", _ES)
    monkeypatch.setattr(hf_data, "eventstore", _ES, raising=False)
    idx = {"E": index_entry("09000001", t0)}
    for _ in range(3):
        assert len(ar.oversim_candidates(idx, [], NOW)) == 1
    assert calls == ["E"]
    idx["E"]["generated"] = "g2"                 # re-published: judged again
    ar.oversim_candidates(idx, [], NOW)
    assert calls == ["E", "E"]


# --------------------------------------------------------------------------- #
# western backlog
# --------------------------------------------------------------------------- #
BACKLOG = [
    {"gauge": "09000001", "tier": 4, "area_km2": 5.0, "stor_mm": 80.0, "donor": False,
     "reasons": ["reservoir storage 80 mm"]},
    {"gauge": "09000002", "tier": 2, "area_km2": 10.0, "stor_mm": 0.0, "donor": True,
     "reasons": ["donor"]},
    {"gauge": "09000003", "tier": 1, "area_km2": 900.0, "stor_mm": 1.0, "donor": True,
     "reasons": ["event-prone"]},
    {"gauge": "09000004", "tier": 1, "area_km2": 50.0, "stor_mm": 0.0, "donor": False,
     "reasons": ["event-prone"]},
    {"gauge": "09000005", "tier": 3, "area_km2": 1.0, "stor_mm": 0.0, "donor": False,
     "reasons": ["dam-free"]},
]


def _order(done=(), skips=None, lakes_on=False):
    """Gauges the backlog would serve, in order, if each one ran."""
    done, out = list(done), []
    while True:
        rec, _ = ar.backlog_candidate(BACKLOG, done, skips or {}, lakes_on=lakes_on)
        if rec is None:
            return out
        out.append(rec["gauge"])
        done.append({"gauge": rec["gauge"], "ran": True, "at": _t(24 * 400)})


def test_backlog_tier_then_area_and_lakes_gate():
    assert _order() == ["09000004", "09000003", "09000002", "09000005"]
    assert _order(lakes_on=True) == ["09000004", "09000003", "09000002", "09000005",
                                     "09000001"]
    rec, why = ar.backlog_candidate(BACKLOG[:1], [], {}, lakes_on=False)
    assert rec is None and "1 reservoir gauges held until AUTOCAL_LAKES_ON=1" in why


def test_backlog_record_fields():
    rec, _ = ar.backlog_candidate(BACKLOG, [], {}, lakes_on=False)
    assert rec == {"kind": "backlog", "gauge": "09000004", "event": None, "west": True,
                   "tier": 1, "area_km2": 50.0, "stor_mm": 0.0, "donor": False,
                   "reasons": ["event-prone"]}


def test_backlog_calibrate_once_any_kind_any_age():
    done = [{"gauge": "09000004", "kind": "miss", "event": "E", "ran": True,
             "at": _t(24 * 365)},
            {"gauge": "09000003", "kind": "backlog", "ran": False, "at": _t(5)}]
    rec, _ = ar.backlog_candidate(BACKLOG, done, {}, lakes_on=False)
    assert rec["gauge"] == "09000003"            # ran=False does not retire a gauge


def test_backlog_only_when_no_event_is_eligible():
    rec, why = ar.pick_candidate(miss_rows("07000001", 3), {}, [], NOW, backlog=BACKLOG)
    assert rec["kind"] == "miss"                 # even an eastern event beats the backlog
    rec, why = ar.pick_candidate([], {}, [], NOW, backlog=BACKLOG)
    assert rec["kind"] == "backlog" and rec["gauge"] == "09000004"
    assert why.startswith("no eligible event; backlog tier 1")


def test_daily_cap_applies_to_backlog_and_events():
    done = [{"gauge": "09000009", "kind": "backlog", "ran": True, "at": _t(2)},
            {"gauge": "09000008", "kind": "oversim", "event": "Z", "ran": True, "at": _t(20)}]
    rec, why = ar.pick_candidate(miss_rows("09000001", 5), {}, done, NOW, backlog=BACKLOG)
    assert rec is None and why.startswith("daily cap reached (2/2")
    done[1]["at"] = _t(25)                       # rolled out of the 24 h window
    rec, _ = ar.pick_candidate([], {}, done, NOW, backlog=BACKLOG)
    assert rec["gauge"] == "09000004"
    skips_only = [{"kind": "backlog_skips", "gauge": None, "ran": False, "at": _t(1)}] * 5
    rec, _ = ar.pick_candidate([], {}, skips_only, NOW, backlog=BACKLOG)
    assert rec is not None                       # skips never count toward the cap


# --------------------------------------------------------------------------- #
# backlog skips: local list, TTL, Hub summary (no commit per skip)
# --------------------------------------------------------------------------- #
def test_backlog_skips_merge_local_and_hub_with_ttl():
    done = [{"kind": "backlog_skips", "gauge": None, "ran": False, "at": _t(30),
             "skipped": [{"gauge": "09000004", "at": _t(40), "reason": "obs cover 10%"},
                         {"gauge": "09000002", "at": _t(24 * 45), "reason": "old"}]}]
    local = {"09000003": {"at": _t(1), "reason": "obs cover 0%"}}
    sk = ar.backlog_skips(done, local, NOW)
    assert set(sk) == {"09000004", "09000003"}   # 45-day-old skip expired (TTL 30 d)
    rec, why = ar.backlog_candidate(BACKLOG, [], sk, lakes_on=False)
    assert rec["gauge"] == "09000002" and "2 skipped ahead" in why


def test_skip_summary_at_most_once_per_period_and_only_new():
    local = {"09000004": {"at": _t(3), "reason": "obs cover 10%"},
             "09000003": {"at": _t(2), "reason": "calibration error: x"}}
    s = ar.skip_summary([], local, NOW)
    assert s["kind"] == "backlog_skips" and s["ran"] is False and s["gauge"] is None
    assert [r["gauge"] for r in s["skipped"]] == ["09000004", "09000003"] and s["n"] == 2
    # a summary 5 h ago (Hub record or local mark) -> nothing for another 19 h
    hub = [{"kind": "backlog_skips", "gauge": None, "ran": False, "at": _t(5), "skipped": []}]
    assert ar.skip_summary(hub, local, NOW) is None
    assert ar.skip_summary([], local, NOW, summarized_at=_t(5)) is None
    # 30 h later only skips newer than that summary are listed
    old = [{"kind": "backlog_skips", "gauge": None, "ran": False, "at": _t(30), "skipped": []}]
    local["09000001"] = {"at": _t(40), "reason": "already summarized"}
    s = ar.skip_summary(old, local, NOW)
    assert {r["gauge"] for r in s["skipped"]} == {"09000004", "09000003"}
    assert ar.skip_summary(old, {"09000001": local["09000001"]}, NOW) is None


# --------------------------------------------------------------------------- #
# calibration windows
# --------------------------------------------------------------------------- #
def test_windows_per_kind():
    miss = ar.pick_candidate(miss_rows("09000001", 3, last_h_ago=10), {}, [], NOW)[0]
    a, b = ar.window(miss, NOW)
    assert b == (NOW - dt.timedelta(hours=10) + dt.timedelta(hours=6)).replace(minute=0)
    assert b - a == dt.timedelta(days=183)
    t0 = NOW - dt.timedelta(hours=40)
    ov = ar.oversim_candidates({"E": index_entry("09000001", t0)}, [], NOW,
                               manifest=lambda ev: manifest(t0, [1, 30], [1, 3]))[0]
    assert ar.window(ov, NOW)[1] == t0 + dt.timedelta(hours=12)
    bl = {"kind": "backlog", "gauge": "09000004"}
    assert ar.window(bl, NOW + dt.timedelta(minutes=37))[1] == NOW - dt.timedelta(hours=2)


# --------------------------------------------------------------------------- #
# backlog builder tiering (pure parts)
# --------------------------------------------------------------------------- #
def _builder():
    p = os.path.join(ROOT, "scripts", "build_autocal_backlog.py")
    spec = importlib.util.spec_from_file_location("build_autocal_backlog", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_builder_tiers():
    b = _builder()
    own = {"wm": 1.0, "alpha0": 1.0, "source_station": "09000001", "source_dist_km": 0.0}
    assert b.tier_of("09000001", True, 0.0, own)[0] == 1
    assert b.tier_of("09000001", False, 9.9, own)[0] == 3
    assert b.tier_of("09000001", False, 0.0, {**own, "wm": 0.005})[0] == 2
    assert b.tier_of("09000001", False, 0.0, {**own, "alpha0": 0.0})[0] == 2
    t, donor, why = b.tier_of("09000001", False, 0.0, {**own, "source_station": "9000002",
                                                       "source_dist_km": 12.4})
    assert (t, donor) == (2, True) and why == ["donor 09000002 @ 12 km"]
    # reservoirs (and gauges without a dam record) wait for the lakes, event-prone or not
    assert b.tier_of("09000001", True, 10.0, own)[0] == 4
    assert b.tier_of("09000001", False, None, own)[0] == 4
    ranked = b.rank([{"gauge": "c", "tier": 2, "area_km2": 5.0},
                     {"gauge": "a", "tier": 1, "area_km2": 900.0},
                     {"gauge": "b", "tier": 2, "area_km2": 1.0}])
    assert [e["gauge"] for e in ranked] == ["a", "b", "c"]


def test_builder_event_gauges_and_rdb():
    b = _builder()
    idx = {"2026102200_09371492": {"gauge": "09371492"},
           "2026101511_07239500": {"trigger": {"gauge": "07239500"}}}
    assert b.event_gauges(idx, [{"gauge": "9349800"}]) == {"09371492", "07239500", "09349800"}
    rdb = ("# comment\nagency_cd\tsite_no\tstation_nm\n5s\t15s\t50s\n"
           "USGS\t09180000\tDOLORES R\nUSGS\t103087891\tX\n")
    assert b.parse_rdb_sites(rdb) == {"09180000", "103087891"}


def test_committed_backlog_is_ranked_and_western():
    data = ar.load_backlog(os.path.join(ROOT, "hf_data", "data", "autocal_west_backlog.json"))
    assert data, "hf_data/data/autocal_west_backlog.json missing or empty"
    keys = [(e["tier"], e["area_km2"], e["gauge"]) for e in data]
    assert keys == sorted(keys)
    assert {e["tier"] for e in data} <= {1, 2, 3, 4}
    assert all(set(e) >= {"gauge", "tier", "area_km2", "stor_mm", "donor", "reasons"}
               for e in data)
    assert len({e["gauge"] for e in data}) == len(data)


# --------------------------------------------------------------------------- #
# review fixes 2026-10-30
# --------------------------------------------------------------------------- #
from hf_data import eventstore as es_mod                       # noqa: E402

BL = {"kind": "backlog", "gauge": "09000004", "event": None, "west": True, "tier": 1,
      "area_km2": 50.0, "stor_mm": 0.0, "donor": False, "reasons": ["event-prone"]}


def test_oversim_trigger_is_western_only():
    t0 = NOW - dt.timedelta(hours=40)
    idx = {"W": index_entry("09000001", t0), "E": index_entry("07000001", t0),
           "U": index_entry("05999999", t0)}           # unknown longitude counts as east
    mf, read = {k: manifest(t0, [1, 90], [1, 3]) for k in idx}, []
    got = ar.oversim_candidates(idx, [], NOW, manifest=lambda ev: (read.append(ev), mf[ev])[1])
    assert [c["event"] for c in got] == ["W"] and read == ["W"]   # east manifests never read


def test_miss_streak_older_than_last_calibration_is_skipped():
    old = miss_rows("09000001", 3, last_h_ago=24 * 50)          # checked before it ran
    done = [{"gauge": "09000001", "event": "X", "ran": True, "at": _t(24 * 40)}]
    assert ar.miss_candidates(old, {}, done, NOW) == []         # cooldown over, still stale
    new = miss_rows("09000001", 3, last_h_ago=10)
    assert [c["gauge"] for c in ar.miss_candidates(new, {}, done, NOW)] == ["09000001"]
    # the ledger's last calibration counts the same (row rolled off autocal.json)
    led = {"09000001": {"first": _t(24 * 40), "last": _t(24 * 40), "n": 1}}
    assert ar.miss_candidates(old, {}, [], NOW, ledger=led) == []


def test_calibrate_once_reads_the_ledger():
    led = {"09000004": {"first": _t(24 * 900), "last": _t(24 * 900), "n": 1},
           "9000003": {"first": _t(24 * 800), "last": _t(24 * 800), "n": 2}}
    rec, why = ar.backlog_candidate(BACKLOG, [], {}, lakes_on=False, ledger=led)
    assert rec["gauge"] == "09000002" and "2 done" in why
    rec, _ = ar.pick_candidate([], {}, [], NOW, backlog=BACKLOG, ledger=led)
    assert rec["gauge"] == "09000002"


def test_lake_hold_needs_a_live_lake_module(monkeypatch):
    assert ar.lake_hold() == "until AUTOCAL_LAKES_ON=1"
    monkeypatch.setattr(ar, "LAKES_ON", True)
    assert "hf_data.lakes unavailable (" in ar.lake_hold()      # absent module: keep holding
    _lakes(monkeypatch, region="off")
    assert "lakes.region() is 'off', not the west" in ar.lake_hold()
    _lakes(monkeypatch, version="")
    assert "lakes.table_version() is empty" in ar.lake_hold()
    _lakes(monkeypatch, region="all")
    assert ar.lake_hold() is None


def test_lakes_flag_without_module_keeps_holding_with_reason(monkeypatch):
    monkeypatch.setattr(ar, "LAKES_ON", True)
    RESV.update({"09000001": 11.0})
    held = []
    rec, why = ar.pick_candidate(miss_rows("09000001", 3), {}, [], NOW, backlog=BACKLOG[:1],
                                 held=held)
    assert rec is None and len(held) == 1
    assert "1 western reservoir-gauge event(s) held although AUTOCAL_LAKES_ON=1: hf_data.lakes" in why
    assert "1 reservoir gauges held although AUTOCAL_LAKES_ON=1" in why
    rec, why = ar.pick_candidate([], {}, [], NOW, backlog=BACKLOG[:1])
    assert rec is None and "reservoir hold kept although AUTOCAL_LAKES_ON=1" in why
    _lakes(monkeypatch)
    rec, why = ar.pick_candidate([], {}, [], NOW, backlog=BACKLOG[:1])
    assert rec["gauge"] == "09000001" and "hold" not in why


def test_skip_ttl_short_for_errors():
    assert ar.skip_ttl("obs cover 10% of the window (< 50%)") == dt.timedelta(days=30)
    assert ar.skip_ttl("calibration error: x") == dt.timedelta(hours=24)
    local = {"09000004": {"at": _t(30), "reason": "error: ValueError: x"},
             "09000003": {"at": _t(30), "reason": "obs cover 10% of the window (< 50%)"},
             "09000002": {"at": _t(10), "reason": "no EF5 run produced an NSE"}}
    assert set(ar.backlog_skips([], local, NOW)) == {"09000003", "09000002"}


def test_fetch_outage_classification():
    assert ar.fetch_outage("HTTPSConnectionPool(host='waterservices.usgs.gov', port=443): "
                           "Read timed out. (read timeout=30)")
    assert ar.fetch_outage("503 Server Error: Service Unavailable for url: https://x")
    assert ar.fetch_outage("429 Client Error: Too Many Requests for url: https://x")
    assert not ar.fetch_outage("404 Client Error: Not Found for url: https://x")
    assert not ar.fetch_outage("400 Client Error: Bad Request for url: https://x")
    assert not ar.fetch_outage(None) and not ar.fetch_outage("")


def _calib_env(monkeypatch, series=(), fetch_error=None, done=None):
    """Fake obs / llm / calibrate for calibrate_one (2-day window)."""
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    monkeypatch.setattr(ar, "busy_reason", lambda: None)
    monkeypatch.setattr(ar, "log", lambda s: None)

    def get_series(gid, a, b, info=None):
        if fetch_error:
            info["fetch_error"] = fetch_error
        return list(series)

    def run_calibration(*a, **k):
        yield "round", {"round": 1, "tried": [{}, {}, {}], "best_nse": None}
        yield "done", done
    _set_module(monkeypatch, "obs", types.SimpleNamespace(get_series=get_series))
    _set_module(monkeypatch, "llm", types.SimpleNamespace(available=lambda: True))
    _set_module(monkeypatch, "calibrate", types.SimpleNamespace(run_calibration=run_calibration))


def _full_series():
    a, b = ar.window(BL, NOW)
    return [(a + dt.timedelta(hours=i), 1.0)
            for i in range(int((b - a).total_seconds() // 3600) + 1)]


def test_obs_fetch_outage_records_nothing_gauge_404_skips(monkeypatch):
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    for err in ("HTTPSConnectionPool(host='x', port=443): Read timed out.",
                "502 Server Error: Bad Gateway for url: https://x"):
        _calib_env(monkeypatch, fetch_error=err)
        with pytest.raises(ar.Outage):
            ar.calibrate_one(BL, NOW)
    _calib_env(monkeypatch, fetch_error="404 Client Error: Not Found for url: https://x")
    r = ar.calibrate_one(BL, NOW)
    assert r["ran"] is False and r["reason"].startswith("obs cover 0%")
    _calib_env(monkeypatch, series=_full_series()[:10], fetch_error=None)
    assert ar.calibrate_one(BL, NOW)["reason"].startswith("obs cover")


def test_no_nse_is_not_a_run(monkeypatch):
    none = {"baseline_nse": None, "best_nse": None, "best_params": {}, "saved": False}
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    _calib_env(monkeypatch, series=_full_series(), done=none)
    r = ar.calibrate_one(BL, NOW)
    assert r["ran"] is False and r["reason"] == "no EF5 run produced an NSE" and r["n_runs"] == 3
    _calib_env(monkeypatch, series=_full_series(), done={**none, "baseline_nse": 0.2})
    assert ar.calibrate_one(BL, NOW)["ran"] is True             # one NSE: it ran


def test_run_candidate_outage_vs_gauge_error(monkeypatch):
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(ar, "log", lambda s: None)

    def raising(e):
        def f(rec, now):
            raise e
        return f
    outages = (ConnectionError("reset"), TimeoutError(), MemoryError(),
               requests.ConnectionError("dns"),
               requests.HTTPError("503", response=types.SimpleNamespace(status_code=503)))
    for e in outages:
        assert ar.is_outage(e), e
        monkeypatch.setattr(ar, "calibrate_one", raising(e))
        with pytest.raises(ar.Outage):
            ar.run_candidate(BL, NOW)
    own = (ValueError("no basin"),
           requests.HTTPError("404", response=types.SimpleNamespace(status_code=404)))
    for e in own:
        assert not ar.is_outage(e), e
        monkeypatch.setattr(ar, "calibrate_one", raising(e))
        r = ar.run_candidate(BL, NOW)                           # backlog: a short skip
        assert r["ran"] is False and r["reason"].startswith("error: ")
        assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)
        with pytest.raises(type(e)):                            # event: shown, retried
            ar.run_candidate({**BL, "kind": "miss", "event": "E"}, NOW)


class _ES:
    """Fake eventstore: autocal.json rows on a fake Hub, commits that may fail."""

    def __init__(self, rows=(), ok=True):
        self.rows = None if rows is None else list(rows)
        self.ledger, self.ok, self.tried = {}, ok, []

    def load_missed(self):
        return []

    def load_index(self):
        return {}

    def load_autocal(self):
        return None if self.rows is None else list(self.rows)

    def load_autocal_done(self):
        return self.ledger

    def note_autocal(self, rec):
        self.tried.append(rec)
        if self.ok:
            self.rows.append(rec)
        return self.ok


def test_flush_pending(monkeypatch):
    es = _ES(ok=False)
    _set_module(monkeypatch, "eventstore", es)
    monkeypatch.setattr(ar, "log", lambda s: None)
    on_hub = {"gauge": "09000003", "event": None, "kind": "backlog", "ran": True, "at": _t(3)}
    a = {"gauge": "09000004", "event": None, "kind": "backlog", "ran": True, "at": _t(2)}
    b = {"gauge": "09000001", "event": "E", "kind": "miss", "ran": False, "at": _t(1)}
    ar.state["pending"] = [on_hub, a, b]
    done = ar._flush_pending([on_hub])
    assert done == [on_hub, a, b] and ar.state["pending"] == [a, b]
    assert es.tried == [a]                                      # stops at the first failure
    # pending records count for the cap, the handled pairs and calibrate-once
    _, handled, today = ar._history(done, NOW)
    assert today == 2 and ("09000001", "E") in handled
    rec, _ = ar.backlog_candidate(BACKLOG, done, {}, lakes_on=False)
    assert rec["gauge"] == "09000002"
    es.ok = True
    assert ar._flush_pending([on_hub]) == [on_hub, a, b]
    assert ar.state["pending"] == [] and es.tried[1:] == [a, b]


class _Stop(BaseException):
    """Ends main() at a chosen sleep."""


def _run_main(monkeypatch, tmp_path, es, calibrate, n_sleeps):
    """Run main() on the fake eventstore and BACKLOG until its n_sleeps-th
    sleep: (every sleep in s, the gauges handed to calibrate_one)."""
    sleeps, picked = [], []

    def sleep(s):
        sleeps.append(s)
        if len(sleeps) >= n_sleeps:
            raise _Stop

    def cal(rec, now):
        picked.append(rec["gauge"])
        return calibrate(rec, now)
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr(ar, "time", types.SimpleNamespace(sleep=sleep, time=lambda: 1e9))
    monkeypatch.setattr(ar, "STATE_PATH", str(tmp_path / "state.json"))
    for k, v in {"busy_reason": lambda: None, "IDLE_FOR_S": 0.0, "log": lambda s: None,
                 "calibrate_one": cal, "load_backlog": lambda: list(BACKLOG)}.items():
        monkeypatch.setattr(ar, k, v)
    _set_module(monkeypatch, "eventstore", es)
    _set_module(monkeypatch, "llm", types.SimpleNamespace(available=lambda: True))
    with pytest.raises(_Stop):
        ar.main()
    return sleeps, picked


def test_main_waits_when_history_unreadable(monkeypatch, tmp_path):
    es = _ES(rows=None)
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, None, 1)
    assert sleeps == [900.0] and picked == [] and es.tried == []
    assert ar.state["phase"].startswith("history unreadable")
    es = _ES()
    es.ledger = None                                            # the ledger alone unreadable
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, None, 1)
    assert picked == [] and ar.state["phase"].startswith("history unreadable")


def test_main_keeps_failed_commit_pending(monkeypatch, tmp_path):
    es = _ES(ok=False)

    def ran(rec, now):
        return {"gauge": rec["gauge"], "kind": "backlog", "event": None, "ran": True,
                "at": now.strftime(TS), "baseline_nse": 0.1, "best_nse": 0.3}
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, ran, 3)
    # poll 2 skips the pending 09000004 (calibrate-once), poll 3 hits the daily
    # cap with two pending runs; every poll retried the oldest pending commit
    assert picked == ["09000004", "09000003"] and sleeps == [30, 30, 900.0]
    assert [r["gauge"] for r in ar.state["pending"]] == ["09000004", "09000003"]
    assert [r["gauge"] for r in es.tried] == ["09000004", "09000004", "09000003", "09000004"]
    assert ar.state["phase"].startswith("idle — daily cap reached (2/2")
    assert "2 pending" in ar.state["phase"]
    with open(tmp_path / "state.json") as fp:                  # survives a loop restart
        assert json.load(fp)["pending"] == ar.state["pending"]


def test_main_outage_records_nothing_then_strikes(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "OUTAGE_STRIKES", 2)
    es = _ES()

    def down(rec, now):
        raise ar.Outage("obs fetch for x failed (timeout)")
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, down, 3)
    # poll 1: outage -> nothing recorded, back off a poll; poll 2: the same gauge
    # again -> a short-TTL skip; poll 3: the next gauge, outage -> back off
    assert picked == ["09000004", "09000004", "09000003"] and sleeps == [900.0, 30, 900.0]
    assert list(ar.state["backlog_skips"]) == ["09000004"]
    assert ar.state["backlog_skips"]["09000004"]["reason"].startswith("outage 2x in a row")
    assert ar.state["phase"].startswith("outage — obs fetch for x failed")
    assert all(r.get("kind") == "backlog_skips" for r in es.rows)   # no per-gauge record


def test_main_pauses_after_a_run_of_backlog_skips(monkeypatch, tmp_path):
    monkeypatch.setattr(ar, "MAX_SKIP_RUN", 2)

    def thin(rec, now):
        return {"gauge": rec["gauge"], "kind": "backlog", "ran": False,
                "at": now.strftime(TS), "reason": "obs cover 0% of the window (< 50%)"}
    sleeps, picked = _run_main(monkeypatch, tmp_path, _ES(), thin, 3)
    # two skips -> pause a whole poll; the count starts over after it
    assert picked == ["09000004", "09000003", "09000002"] and sleeps == [30, 900.0, 30]


def test_ledger_add():
    led = {}
    es_mod._ledger_add(led, {"gauge": "09000004", "ran": False, "at": _t(5)})
    es_mod._ledger_add(led, {"gauge": None, "kind": "backlog_skips", "ran": False})
    assert led == {}
    es_mod._ledger_add(led, {"gauge": "09000004", "ran": True, "at": _t(5)})
    es_mod._ledger_add(led, {"gauge": "09000004", "ran": True, "at": _t(50)})   # out of order
    assert led == {"09000004": {"first": _t(50), "last": _t(5), "n": 2}}


def _hub(monkeypatch, tmp_path, files: dict, err=None):
    """Fake hf_hub_download over {path: text}; a missing path is a 404."""
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    def dl(repo, path, **kw):
        if err is not None:
            raise err
        if path not in files:
            raise EntryNotFoundError(path)
        p = tmp_path / path.replace("/", "_")
        p.write_text(files[path], encoding="utf-8")
        return str(p)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)


def test_load_autocal_missing_vs_unreadable(monkeypatch, tmp_path):
    from huggingface_hub.utils import LocalEntryNotFoundError
    _hub(monkeypatch, tmp_path, {})
    assert es_mod.load_autocal() == [] and es_mod.load_autocal_done() == {}   # not created yet
    _hub(monkeypatch, tmp_path, {es_mod.AUTOCAL_FILE: '[{"gauge": "1"}]',
                                 es_mod.AUTOCAL_DONE_FILE: "[]"})
    assert es_mod.load_autocal() == [{"gauge": "1"}]
    assert es_mod.load_autocal_done() is None                  # wrong type: unreadable
    _hub(monkeypatch, tmp_path, {es_mod.AUTOCAL_FILE: "[{"})
    assert es_mod.load_autocal() is None                       # corrupt JSON
    for e in (LocalEntryNotFoundError("offline"), OSError("disk"), RuntimeError("401")):
        _hub(monkeypatch, tmp_path, {}, err=e)
        assert es_mod.load_autocal() is None and es_mod.load_autocal_done() is None


class _Api:
    def __init__(self, fail=False):
        self.ops, self.fail = None, fail

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        if self.fail:
            raise RuntimeError("hub down")
        self.ops = {op.path_in_repo: json.loads(op.path_or_fileobj.getvalue())
                    for op in operations}


def test_note_autocal_commits_record_and_ledger(monkeypatch):
    api = _Api()
    rows = [{"gauge": "09000003", "ran": True, "at": _t(400)},
            {"gauge": "09000002", "ran": False, "at": _t(300)}]
    monkeypatch.setattr(es_mod, "_api", lambda: api)
    monkeypatch.setattr(es_mod, "load_autocal", lambda: list(rows))
    monkeypatch.setattr(es_mod, "load_autocal_done", lambda: {})
    rec = {"gauge": "09000004", "kind": "backlog", "ran": True, "at": _t(1),
           "baseline_nse": 0.1, "best_nse": 0.4}
    assert es_mod.note_autocal(rec) is True
    assert api.ops[es_mod.AUTOCAL_FILE] == rows + [rec]
    assert api.ops[es_mod.AUTOCAL_DONE_FILE] == {              # first write: seeded from the log
        "09000003": {"first": _t(400), "last": _t(400), "n": 1},
        "09000004": {"first": _t(1), "last": _t(1), "n": 1}}
    assert es_mod.note_autocal({**rec, "ran": False, "reason": "x"}) is True
    assert set(api.ops) == {es_mod.AUTOCAL_FILE}                # a skip: the log only
    # the log keeps AUTOCAL_KEEP rows, the ledger every calibrated gauge
    monkeypatch.setattr(es_mod, "AUTOCAL_KEEP", 2)
    monkeypatch.setattr(es_mod, "load_autocal_done",
                        lambda: {"09000001": {"first": "a", "last": "a", "n": 1}})
    assert es_mod.note_autocal(rec) is True
    assert len(api.ops[es_mod.AUTOCAL_FILE]) == 2
    assert set(api.ops[es_mod.AUTOCAL_DONE_FILE]) == {"09000001", "09000004"}


def test_note_autocal_aborts_on_unreadable_history(monkeypatch):
    api = _Api()
    rec = {"gauge": "09000004", "ran": True, "at": _t(1)}
    monkeypatch.setattr(es_mod, "_api", lambda: api)
    monkeypatch.setattr(es_mod, "load_autocal", lambda: None)
    monkeypatch.setattr(es_mod, "load_autocal_done", lambda: {})
    assert es_mod.note_autocal(rec) is False and api.ops is None   # never [rec] over the log
    monkeypatch.setattr(es_mod, "load_autocal", lambda: [])
    monkeypatch.setattr(es_mod, "load_autocal_done", lambda: None)
    assert es_mod.note_autocal(rec) is False and api.ops is None   # nor a ledger reset
    assert es_mod.note_autocal({**rec, "ran": False}) is True      # a skip needs the log only
    monkeypatch.setattr(es_mod, "_api", lambda: _Api(fail=True))
    assert es_mod.note_autocal({**rec, "ran": False}) is False     # commit failed

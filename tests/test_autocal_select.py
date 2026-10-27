"""CREST_autocal candidate selection (PI 2026-10-27 west-first + backlog):
pure logic on synthetic missed/index/done/backlog inputs — no network, no EF5."""
from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, "autocal"))
import autocal_run as ar                                      # noqa: E402

NOW = dt.datetime(2026, 10, 27, 12, 0, 0)
TS = ar.TS

# synthetic gauges: W* west of -105, E* east of it
LONS = {"09000001": -111.0, "09000002": -108.0, "09000003": -112.0, "09000004": -110.0,
        "09000005": -115.0, "07000001": -97.0, "07000002": -90.0, "07000003": -95.0}
# reservoir-affected gauges (STOR_NOR_2009 mm); tests add entries as needed
RESV: dict = {}
_reservoir_stor = ar.reservoir_stor            # the real reader, before the fixture patch


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(ar, "gauge_lon", lambda g: LONS.get(str(g).zfill(8)))
    RESV.clear()
    monkeypatch.setattr(ar, "reservoir_stor", lambda g: RESV.get(str(g).zfill(8)))
    for k, v in {"MIN_CHECKS": 3, "QUIET_H": 6.0, "COOLDOWN_D": 30.0, "MAX_PER_DAY": 2,
                 "WINDOW_D": 183.0, "STREAK_GAP_H": 3.0, "OVERSIM_RATIO": 3.0,
                 "OVERSIM_MIN_OBS": 1.0, "LAKES_ON": False, "SKIP_TTL_D": 30.0,
                 "SKIP_SUMMARY_H": 24.0}.items():
        monkeypatch.setattr(ar, k, v)
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


def test_west_oversim_before_west_miss_east_miss_before_east_oversim():
    t0 = NOW - dt.timedelta(hours=30)
    idx = {"W": index_entry("09000004", t0), "E": index_entry("07000003", t0)}
    mf = {"W": manifest(t0, [1, 5, 40], [1, 2, 4]),          # 10x, west
          "E": manifest(t0, [1, 5, 90], [1, 2, 4])}          # 22x, east
    missed = (miss_rows("07000001", 10, obs=500.0)            # east miss
              + miss_rows("09000001", 12, obs=900.0))         # west miss, 12 checks
    assert _order_events(missed, idx, mf) == [
        ("oversim", "09000004"), ("miss", "09000001"),         # west: oversim first
        ("miss", "07000001"), ("oversim", "07000003")]         # east: miss first


# --------------------------------------------------------------------------- #
# reservoir hold (PI 2026-10-27): no trigger calibrates a >= 10 mm gauge
# until AUTOCAL_LAKES_ON=1
# --------------------------------------------------------------------------- #
def test_reservoir_gauges_held_from_event_triggers(monkeypatch):
    t0 = NOW - dt.timedelta(hours=30)
    idx = {"W": index_entry("09000004", t0), "E": index_entry("07000003", t0)}
    mf = {"W": manifest(t0, [1, 5, 40], [1, 2, 4]), "E": manifest(t0, [1, 5, 90], [1, 2, 4])}
    missed = miss_rows("09000001", 12, obs=900.0) + miss_rows("07000001", 10, obs=500.0)
    RESV.update({"09000004": 267.0, "09000001": 11.0, "07000003": 40.0})
    held = []
    rec, why = ar.pick_candidate(missed, idx, [], NOW, manifest=mf.get, held=held)
    assert (rec["kind"], rec["gauge"]) == ("miss", "07000001")   # the only dam-free one
    assert "3 reservoir-gauge event(s) held until AUTOCAL_LAKES_ON=1" in why
    assert sorted((h["kind"], h["gauge"], h["stor_mm"], h["west"]) for h in held) == [
        ("miss", "09000001", 11.0, True), ("oversim", "07000003", 40.0, False),
        ("oversim", "09000004", 267.0, True)]
    # nothing at all when every candidate is held: falls through to the backlog
    held = []
    rec, why = ar.pick_candidate(missed[:12], {"W": idx["W"]}, [], NOW, manifest=mf.get,
                                 held=held)
    assert rec is None and len(held) == 2
    assert why.startswith("no eligible event; 2 reservoir-gauge event(s) held")
    # the held list stays current while the daily cap is reached
    done = [{"gauge": "x", "ran": True, "at": _t(1)}, {"gauge": "y", "ran": True, "at": _t(2)}]
    held = []
    rec, why = ar.pick_candidate(missed, idx, done, NOW, manifest=mf.get, held=held)
    assert rec is None and why.startswith("daily cap") and len(held) == 3
    # lakes on: the hold lifts and the western over-prediction leads again
    monkeypatch.setattr(ar, "LAKES_ON", True)
    held = []
    rec, _ = ar.pick_candidate(missed, idx, [], NOW, manifest=mf.get, held=held)
    assert (rec["kind"], rec["gauge"]) == ("oversim", "09000004") and held == []


def test_note_held_state_and_log(monkeypatch):
    lines = []
    monkeypatch.setattr(ar, "log", lines.append)
    monkeypatch.setitem(ar.state, "held_reservoir", [])
    h = [{"gauge": "10171000", "event": "2026101202_10171000", "kind": "oversim",
          "west": True, "stor_mm": 266.53}]
    ar._note_held(h)
    ar._note_held(list(h))                       # unchanged set: no second log line
    assert ar.state["held_reservoir"] == h and len(lines) == 1
    assert "10171000 (oversim, 267 mm, west)" in lines[0]


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

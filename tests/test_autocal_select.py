"""CREST_autocal candidate selection (PI 2026-11-05: a low CREST 1-D NSE in a
flood episode is the only trigger; west first, then the lowest NSE; autocal
yields only to user-initiated work): pure logic on synthetic index / manifest
/ done inputs — no network, no EF5."""
from __future__ import annotations

import datetime as dt
import importlib.util
import inspect
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
HM = "%Y-%m-%dT%H:%MZ"

# synthetic gauges: W* west of -105, E* east of it
LONS = {"09000001": -111.0, "09000002": -108.0, "09000003": -112.0, "09000004": -110.0,
        "09000005": -115.0, "07000001": -97.0, "07000002": -90.0, "07000003": -95.0}
# reservoir-affected gauges (STOR_NOR_2009 mm); tests add entries as needed
RESV: dict = {}
_reservoir_stor = ar.reservoir_stor            # the real reader, before the fixture patch
_busy_reason = ar.busy_reason                  # the real gate, before a test fakes it
STATE0 = json.loads(json.dumps(ar.state))      # pristine loop state
BACKLOG_JSON = os.path.join(ROOT, "hf_data", "data", "autocal_west_backlog.json")


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
    for k, v in {"COOLDOWN_D": 30.0, "MAX_PER_DAY": 2, "WINDOW_D": 183.0,
                 "NSE_MAX": 0.1, "NSE_MIN_PAIRS": 12, "LAKES_ON": False,
                 "OBS_MIN_FRAC": 0.5, "POLL_S": 900.0, "SKIP_TTL_D": 30.0,
                 "SKIP_ERR_TTL_H": 24.0, "SKIP_SUMMARY_H": 24.0, "MAX_SKIP_RUN": 3,
                 "OUTAGE_STRIKES": 4}.items():
        monkeypatch.setattr(ar, k, v)
    monkeypatch.setattr(ar, "state", json.loads(json.dumps(STATE0)))
    _set_module(monkeypatch, "lakes", None)      # no lake module unless a test adds one
    ar._verdicts.clear()


def _t(h_ago: float) -> str:
    return (NOW - dt.timedelta(hours=h_ago)).strftime(TS)


OBS = [float(i) for i in range(1, 15)]           # 14 observed hours, mean 7.5


def scaled(f: float) -> tuple[list, list]:
    """(sim, obs) with sim = f x obs: NSE = 1 - (f - 1)^2 x 1015 / 227.5."""
    return [f * o for o in OBS], list(OBS)


def nse_of(f: float) -> float:
    return 1.0 - (f - 1.0) ** 2 * sum(o * o for o in OBS) / sum((o - 7.5) ** 2 for o in OBS)


def manifest(t0: dt.datetime, sim: list, obs: list, sim_start: dt.datetime | None = None,
             pre: list = (), tail: list = ((0.0, None),) * 3) -> dict:
    """Hydro rows: len(sim) hours ending at t0 (the episode window starts at
    the first of them unless sim_start says otherwise), `pre` (sim, obs)
    hindcast hours before them and a post-t0 forecast `tail`."""
    n = len(sim)
    first = t0 - dt.timedelta(hours=n - 1)

    def row(t, s, o):
        return {"time": t.strftime("%Y-%m-%d %H:%M"), "sim_q": s, "obs_q": o, "precip": 0.0}
    rows = [row(first - dt.timedelta(hours=len(pre) - i), s, o) for i, (s, o) in enumerate(pre)]
    rows += [row(first + dt.timedelta(hours=i), sim[i], obs[i]) for i in range(n)]
    rows += [row(t0 + dt.timedelta(hours=k + 1), s, o) for k, (s, o) in enumerate(tail)]
    return {"t0": t0.strftime(HM), "sim_start": (sim_start or first).strftime(HM),
            "hydro": rows}


def index_entry(gid: str, t0: dt.datetime, status: str = "ended", gen: str | None = None,
                end_h_after: float = 12.0, **extra) -> dict:
    return {"gauge": gid, "t0": t0.strftime(HM),
            "sim_start": (t0 - dt.timedelta(hours=13)).strftime(HM),
            "t_end": (t0 + dt.timedelta(hours=end_h_after)).strftime(HM),
            "status": status, "generated": gen or _t(2), **extra}


def episodes(spec: dict, t0: dt.datetime | None = None):
    """{event: (gauge, f)} -> (index, manifests): one episode per event whose
    sim is f x obs over 14 paired hours."""
    t0 = t0 or NOW - dt.timedelta(hours=30)
    idx = {ev: index_entry(g, t0) for ev, (g, f) in spec.items()}
    mf = {ev: manifest(t0, *scaled(f)) for ev, (g, f) in spec.items()}
    return idx, mf


# --------------------------------------------------------------------------- #
# NSE of the CREST 1-D hydrograph over the episode
# --------------------------------------------------------------------------- #
def test_episode_nse_matches_the_formula():
    t0 = NOW - dt.timedelta(hours=20)
    st = ar.episode_nse(manifest(t0, *scaled(1.5)))
    assert st["nse"] == pytest.approx(nse_of(1.5)) and st["n_pairs"] == 14
    assert st["obs_peak_m3s"] == 14.0 and st["sim_peak_m3s"] == 21.0
    assert st["nse_span"] == [(t0 - dt.timedelta(hours=13)).strftime(HM), t0.strftime(HM)]
    assert ar.episode_nse(manifest(t0, *scaled(1.0)))["nse"] == pytest.approx(1.0)


def test_episode_nse_paired_observed_episode_hours_only():
    t0 = NOW - dt.timedelta(hours=20)
    sim, obs = scaled(1.5)
    base = ar.episode_nse(manifest(t0, sim, obs))["nse"]
    # hindcast hours before the episode window and paired hours after t0
    # (forecast tail) never count, however wrong
    m = manifest(t0, sim, obs, pre=[(900.0, 1.0)] * 5, tail=[(999.0, 0.5)] * 3)
    assert ar.episode_nse(m)["nse"] == pytest.approx(base)
    # unpaired, NaN and infinite rows are skipped, not zero-filled
    m = manifest(t0 + dt.timedelta(hours=4), sim + [None, 5.0, float("nan"), float("inf")],
                 obs + [3.0, None, 2.0, 2.0])
    st = ar.episode_nse(m)
    assert st["n_pairs"] == 14 and st["nse"] == pytest.approx(base)


def test_episode_window_starts_at_the_earliest_start():
    t0 = NOW - dt.timedelta(hours=20)
    sim, obs = scaled(2.0)
    m = manifest(t0, sim, obs, pre=[(1.0, 1.0)] * 6)
    # a warm re-run publishes the sim_start of its fresh slice (an hour
    # before t0); the episode began 13 h earlier: episode_start / frames say so
    m["sim_start"] = (t0 - dt.timedelta(hours=1)).strftime(HM)
    assert ar.episode_nse(m) is None                            # 2 pairs from sim_start alone
    m["episode_start"] = (t0 - dt.timedelta(hours=13)).strftime(HM)
    assert ar.episode_nse(m)["n_pairs"] == 14
    del m["episode_start"]
    m["frames"] = [{"t": (t0 - dt.timedelta(hours=13)).strftime(HM)},
                   {"t": t0.strftime(HM)}]
    assert ar.episode_nse(m)["n_pairs"] == 14
    s = {"sim_start": (t0 - dt.timedelta(hours=19)).strftime(HM)}   # index summary
    assert ar.episode_nse(m, s)["n_pairs"] == 20                # 14 + the 6 hindcast hours
    assert ar.episode_start({"sim_start": "x"}) is None


def test_episode_nse_min_pairs_and_constant_obs():
    t0 = NOW - dt.timedelta(hours=20)
    sim, obs = scaled(3.0)
    assert ar.episode_nse(manifest(t0, sim[:11], obs[:11])) is None    # 11 < 12
    assert ar.episode_nse(manifest(t0, sim[:12], obs[:12]))["n_pairs"] == 12
    assert ar.episode_nse(manifest(t0, sim, [4.2] * 14)) is None        # constant obs
    assert ar.episode_nse(manifest(t0, sim, [None] * 14)) is None       # no obs at all
    m = manifest(t0, sim, obs)
    del m["t0"]
    assert ar.episode_nse(m) is None                                    # no t0 anywhere
    assert ar.episode_nse(m, {"t0": t0.strftime(HM)})["n_pairs"] == 14  # the summary's


def test_threshold_is_strict(monkeypatch):
    idx, mf = episodes({"A": ("09000001", 1.5)})
    st = ar.episode_nse(mf["A"])
    assert ar.low_nse_candidates(idx, [], NOW, manifest=mf.get)[0]["nse"] == st["nse"]
    monkeypatch.setattr(ar, "NSE_MAX", st["nse"])
    assert ar.low_nse_candidates(idx, [], NOW, manifest=mf.get) == []
    monkeypatch.setattr(ar, "NSE_MAX", 0.1)
    idx, mf = episodes({"G": ("09000001", 1.1)})                 # NSE 0.955: fine
    assert ar.low_nse_candidates(idx, [], NOW, manifest=mf.get) == []


# --------------------------------------------------------------------------- #
# which episodes qualify
# --------------------------------------------------------------------------- #
def test_active_episode_qualifies_without_a_quiet_wait():
    t0 = NOW - dt.timedelta(hours=1)
    idx = {"A": index_entry("09000001", t0, status="active", gen=_t(0.5))}
    mf = {"A": manifest(t0, *scaled(2.0))}
    c, why = ar.pick_candidate(idx, [], NOW, manifest=mf.get)
    assert (c["kind"], c["gauge"], c["event"], c["status"]) == ("low_nse", "09000001", "A", "active")
    assert c["nse"] == pytest.approx(nse_of(2.0)) and c["n_pairs"] == 14 and c["west"] is True
    assert why.startswith("1 eligible gauge(s), 1 western, with 1 episode(s) of NSE < 0.1")
    a, b = ar.window(c, NOW)                                     # t_end is 11 h ahead
    assert b == NOW - dt.timedelta(hours=2) and b - a == dt.timedelta(days=183)


def test_merged_fragments_and_other_statuses_are_skipped():
    t0 = NOW - dt.timedelta(hours=30)
    idx = {"T": index_entry("09000001", t0),
           "F": index_entry("09000002", t0, merged_into="T"),    # absorbed fragment
           "Q": index_entry("09000003", t0, status="queued"),
           "N": {**index_entry("09000004", t0), "status": None}}
    read = []
    mf = {k: manifest(t0, *scaled(3.0)) for k in idx}
    got = ar.low_nse_candidates(idx, [], NOW, manifest=lambda ev: (read.append(ev), mf[ev])[1])
    assert [c["event"] for c in got] == ["T"] and read == ["T"]  # skipped ones never read


def test_west_first_then_lowest_nse():
    idx, mf = episodes({"E1": ("07000001", 9.0),                 # east, worst NSE of all
                        "W1": ("09000001", 1.5),                 # west, NSE -0.12
                        "W2": ("09000002", 3.0),                 # west, NSE -16.8
                        "U": ("05999999", 4.0),                  # unknown lon: east
                        "E2": ("07000002", 2.0),
                        "OK": ("09000003", 1.1)})                # fine: never a candidate
    order, done = [], []
    while True:
        rec, _ = ar.pick_candidate(idx, done, NOW, manifest=mf.get)
        if rec is None:
            break
        order.append(rec["event"])
        # a skip (ran=False, within its TTL) keeps the gauge out without
        # touching the cap or the cooldown
        done.append({"gauge": rec["gauge"], "event": rec["event"], "ran": False, "at": _t(1)})
    assert order == ["W2", "W1", "E1", "U", "E2"]
    assert ar._key({"west": False, "nse": -50.0, "event": "x"}) > ar._key(
        {"west": True, "nse": 0.05, "event": "y"})


def test_cooldown_and_ledger():
    idx, mf = episodes({"A": ("09000001", 2.0)})
    done = [{"gauge": "09000001", "event": "OTHER", "ran": True, "at": _t(24 * 10)}]
    assert ar.low_nse_candidates(idx, done, NOW, manifest=mf.get) == []
    done[0]["at"] = _t(24 * 40)                                  # cooldown over
    assert [c["event"] for c in ar.low_nse_candidates(idx, done, NOW, manifest=mf.get)] == ["A"]
    # a ran=False record (skip) starts no cooldown
    skip = [{"gauge": "09000001", "event": "OTHER", "ran": False, "at": _t(1)}]
    assert len(ar.low_nse_candidates(idx, skip, NOW, manifest=mf.get)) == 1
    # the ledger's last calibration counts the same (row rolled off autocal.json)
    led = {"9000001": {"first": _t(24 * 90), "last": _t(24 * 10), "n": 2}}
    assert ar.low_nse_candidates(idx, [], NOW, manifest=mf.get, ledger=led) == []


def test_no_calibrate_once_gauge_calibrated_long_ago_is_eligible_again():
    idx, mf = episodes({"A": ("09000001", 2.0)})
    led = {"09000001": {"first": _t(24 * 400), "last": _t(24 * 200), "n": 1}}
    rec, _ = ar.pick_candidate(idx, [], NOW, manifest=mf.get, ledger=led)
    assert rec["gauge"] == "09000001"


def test_already_handled_episode_is_not_calibrated_again():
    idx, mf = episodes({"A": ("09000001", 2.0), "B": ("09000002", 2.0)})
    # a calibration that RAN handles its episodes, whatever the NSE after it
    # (still < 0.1 here) and saved or not; the 2-D runs are none of autocal's
    # business. Out of the cooldown and published after it: still handled
    done = [{"gauge": "09000001", "event": "A", "ran": True, "saved": False,
             "at": _t(24 * 40), "best_nse": -1.0}]
    idx["A"]["generated"] = _t(1)
    assert [c["event"] for c in ar.low_nse_candidates(idx, done, NOW, manifest=mf.get)] == ["B"]
    # every id of the record's `events` counts, not only `event`
    done[0].update({"event": "OTHER", "events": ["OTHER", "A"]})
    assert [c["event"] for c in ar.low_nse_candidates(idx, done, NOW, manifest=mf.get)] == ["B"]


def test_not_run_is_a_skip_with_a_ttl_not_handled():
    idx, mf = episodes({"A": ("09000001", 2.0), "B": ("09000002", 3.0)})

    def pick(done, local=None):
        return ar.pick_candidate(idx, done, NOW, manifest=mf.get, local_skips=local)

    def skip(reason, h_ago, gauge="09000002", events=("B",)):
        return {"gauge": gauge, "event": events[0], "events": list(events), "ran": False,
                "kind": "low_nse", "at": _t(h_ago), "reason": reason}
    empty = "obs cover 0% of the window (< 50%)"
    for reason in (empty, "obs cover 20% of the window (< 50%); fetch: 404 Client Error: x",
                   "calibration error: boom", "no EF5 run produced an NSE"):
        done = [skip(reason, 1)]
        _, handled, today = ar._history(done, NOW)
        assert not handled and today == 0                        # never handled, no cap
        assert len(ar.low_nse_candidates(idx, done, NOW, manifest=mf.get)) == 2
        rec, why = pick(done)
        assert rec["gauge"] == "09000001"                        # B's gauge waits
        assert "1 gauge(s) wait out a skip (calibration did not run)" in why
        assert pick([skip(reason, 25)])[0]["gauge"] == "09000002"   # error TTL 24 h over
    # too little obs (some, fetched without error) stands SKIP_TTL_D days
    low = "obs cover 20% of the window (< 50%)"
    assert pick([skip(low, 24 * 29)])[0]["gauge"] == "09000001"
    assert pick([skip(low, 24 * 31)])[0]["gauge"] == "09000002"
    # a new eligible episode on the skipped gauge: retried at once
    idx2, mf2 = episodes({"B2": ("09000002", 3.0)}, t0=NOW - dt.timedelta(hours=5))
    idx.update(idx2)
    mf.update(mf2)
    rec, _ = pick([skip(empty, 1)])
    assert rec["gauge"] == "09000002" and rec["events"] == ["B", "B2"]
    assert pick([skip(empty, 1, events=("B", "B2"))])[0]["gauge"] == "09000001"
    # the loop's local list and the Hub's summary rows count the same
    loc = {"09000002": {"at": _t(1), "reason": empty, "events": ["B", "B2"]}}
    assert pick([], loc)[0]["gauge"] == "09000001"
    summ = {"kind": ar.SKIPS_KIND, "gauge": None, "event": None, "ran": False, "at": _t(1),
            "skipped": [{"gauge": "09000002", "events": ["B", "B2"], "at": _t(2),
                         "reason": "no EF5 run produced an NSE"}]}
    assert pick([summ])[0]["gauge"] == "09000001"
    # nothing left but skipped gauges: no candidate, and the phase says why
    loc["09000001"] = {"at": _t(1), "reason": empty, "events": ["A"]}
    rec, why = pick([], loc)
    assert rec is None and why == ("no eligible episode with NSE < 0.1; 2 gauge(s) wait "
                                   "out a skip (calibration did not run)")


def test_skip_ttl():
    d, h = dt.timedelta(days=30), dt.timedelta(hours=24)
    assert ar.skip_ttl("obs cover 20% of the window (< 50%)") == d       # too little obs
    assert ar.skip_ttl("obs cover 0% of the window (< 50%)") == h        # none: suspicious
    assert ar.skip_ttl("obs cover 20% of the window (< 50%); fetch: 410 Client Error: "
                       "Gone for url: https://x") == h                   # a fetch error
    for r in ("calibration error: x", "no EF5 run produced an NSE",
              "no candidate run produced an NSE", None, "",
              f"flood obs missing: A 0 of 13 paired hours; {ar.QUIET_EMPTY}",
              f"obs cover 30% of the window (< 50%); flood obs missing: A 2 of 13 paired "
              f"hours; {ar.QUIET_EMPTY}"):
        assert ar.skip_ttl(r) == h
    # a reason cut to the 200 characters a skip keeps still says it
    assert ar.skip_ttl(("obs cover 30% of the window (< 50%); flood obs missing: "
                        + "2026102200_09371492 0 of 13 paired hours, " * 9)[:200]) == h
    # an outage streak: about one poll
    assert ar.skip_ttl("outage 4x in a row: obs fetch for x failed (503)") == dt.timedelta(
        seconds=900)


def test_active_skips_and_summary():
    loc = {"09000001": {"at": _t(2), "reason": "no EF5 run produced an NSE", "events": ["A"]},
           "09000002": {"at": _t(30), "reason": "calibration error: x", "events": ["B"]}}
    old = {"gauge": "9000001", "event": "Z", "ran": False, "kind": "oversim", "at": _t(5),
           "reason": "obs cover 10% of the window (< 50%)"}            # an older kind's skip
    sk = ar.active_skips([old], loc, NOW)
    assert sk == {"09000001": {"at": _t(2), "reason": "no EF5 run produced an NSE",
                               "events": ["A", "Z"]}}                  # 09000002 expired
    # one summary record per SKIP_SUMMARY_H, listing what is new since the last
    rec = ar.skip_summary([], loc, NOW)
    assert (rec["kind"], rec["gauge"], rec["ran"], rec["n"]) == (ar.SKIPS_KIND, None, False, 2)
    assert [r["gauge"] for r in rec["skipped"]] == ["09000002", "09000001"]
    assert ar.skip_summary([rec], loc, NOW) is None                   # just summarized
    assert ar.skip_summary([], loc, NOW, summarized_at=_t(23)) is None
    later = NOW + dt.timedelta(hours=25)
    loc["09000003"] = {"at": _t(-24), "reason": "calibration error: y", "events": ["C"]}
    rec2 = ar.skip_summary([rec], loc, later)
    assert [r["gauge"] for r in rec2["skipped"]] == ["09000003"] and rec2["n"] == 1
    json.dumps(rec2)


def test_episode_published_before_the_last_calibration_is_stale():
    idx, mf = episodes({"A": ("09000001", 2.0)})
    idx["A"]["generated"] = _t(24 * 45)                          # sim with the old params
    done = [{"gauge": "09000001", "event": "X", "ran": True, "at": _t(24 * 40)}]
    assert ar.low_nse_candidates(idx, done, NOW, manifest=mf.get) == []
    idx["A"]["generated"] = _t(24 * 35)                          # re-run since: judged
    assert len(ar.low_nse_candidates(idx, done, NOW, manifest=mf.get)) == 1


def test_stale_rule_counts_from_the_calibration_finish():
    idx, mf = episodes({"A": ("09000001", 2.0)})
    idx["A"]["generated"] = _t(24 * 40 - 2)       # published while the calibration ran
    done = [{"gauge": "09000001", "event": "X", "ran": True, "at": _t(24 * 40)}]
    assert len(ar.low_nse_candidates(idx, done, NOW, manifest=mf.get)) == 1   # start only
    done[0]["finished"] = _t(24 * 40 - 5)
    assert ar.low_nse_candidates(idx, done, NOW, manifest=mf.get) == []
    assert ar._finished(done) == {"09000001": NOW - dt.timedelta(hours=24 * 40 - 5)}
    # the ledger keeps the finish after the row rolled off autocal.json
    led = {"09000001": {"first": _t(24 * 40), "last": _t(24 * 40), "n": 1,
                        "finished": _t(24 * 40 - 5)}}
    assert ar.low_nse_candidates(idx, [], NOW, manifest=mf.get, ledger=led) == []
    del led["09000001"]["finished"]                               # older ledger: its start
    assert len(ar.low_nse_candidates(idx, [], NOW, manifest=mf.get, ledger=led)) == 1
    idx["A"]["generated"] = _t(24 * 40 - 6)                       # after the finish
    done[0]["finished"] = _t(24 * 40 - 5)
    assert len(ar.low_nse_candidates(idx, done, NOW, manifest=mf.get)) == 1


# --------------------------------------------------------------------------- #
# one candidate per gauge; merged episodes
# --------------------------------------------------------------------------- #
def test_one_candidate_per_gauge():
    ta, tb = NOW - dt.timedelta(hours=100), NOW - dt.timedelta(hours=40)
    idx = {"A": index_entry("09000001", ta), "B": index_entry("09000001", tb),
           "C": index_entry("09000002", tb), "D": index_entry("09000001", tb)}
    mf = {"A": manifest(ta, *scaled(1.5)), "B": manifest(tb, *scaled(3.0)),
          "C": manifest(tb, *scaled(2.0)), "D": manifest(tb, *scaled(1.1))}  # D: NSE fine
    cands = ar.low_nse_candidates(idx, [], NOW, manifest=mf.get)
    assert sorted(c["gauge"] for c in cands) == ["09000001", "09000002"]
    rec, why = ar.pick_candidate(idx, [], NOW, manifest=mf.get)
    assert why.startswith("2 eligible gauge(s), 2 western, with 3 episode(s) of NSE < 0.1")
    # the gauge ranks by its worst episode; the record lists every eligible one
    assert (rec["gauge"], rec["event"], rec["events"]) == ("09000001", "B", ["A", "B"])
    assert rec["nse"] == pytest.approx(nse_of(3.0)) and rec["status"] == "ended"
    assert rec["episode_nse"] == {"A": round(nse_of(1.5), 3), "B": round(nse_of(3.0), 3)}
    # the window ends at the LATEST t_end among them
    assert ar.window(rec, NOW)[1] == tb + dt.timedelta(hours=12)
    # an active episode: its current t0 ends the window
    idx["A"] = index_entry("09000001", NOW - dt.timedelta(hours=3), status="active",
                           end_h_after=12.0)
    mf["A"] = manifest(NOW - dt.timedelta(hours=3), *scaled(1.5))
    rec, _ = ar.pick_candidate(idx, [], NOW, manifest=mf.get)
    assert rec["status"] == "active" and rec["events"] == ["A", "B"]
    assert ar.window(rec, NOW)[1] == NOW - dt.timedelta(hours=3)
    assert rec["t0"] == idx["A"]["t0"] and rec["t_end"] == idx["A"]["t_end"]
    # a calibration that ran for the gauge handles all of them
    done = [{"gauge": "09000001", "event": "B", "events": ["A", "B"], "ran": True,
             "at": _t(24 * 40), "finished": _t(24 * 40)}]
    for e in ("A", "B"):
        idx[e]["generated"] = _t(1)
    assert [c["gauge"] for c in ar.low_nse_candidates(idx, done, NOW, manifest=mf.get)] == [
        "09000002"]


def _merged(target: str, gid: str, t0: dt.datetime, members: list, merged_at: str,
            gen: str | None = None, f: float = 3.0):
    """A merge target's index entry and manifest (scripts/merge_episodes.py):
    generated re-stamped with the merge time, merged record, merged_from."""
    s = index_entry(gid, t0, gen=gen or merged_at,
                    merged={"n": len(members) + 1, "ids": [target] + members})
    m = manifest(t0, *scaled(f))
    m.update({"merged_from": list(members), "generated": merged_at,
              "provenance": {"engine": "x", "merged": {"at": merged_at,
                                                       "members": [target] + members}}})
    return s, m


def test_merged_episode_handled_through_any_member():
    t0 = NOW - dt.timedelta(hours=30)
    done = [{"gauge": "09000001", "event": "F1", "ran": True, "at": _t(24 * 40)}]
    s, m = _merged("T", "09000001", t0, ["F1", "F2"], _t(1))
    # through the index merge record (F1 calibrated before the merge)
    idx = {"T": s, "F1": index_entry("09000001", t0, gen=_t(24 * 41), merged_into="T")}
    assert ar.low_nse_candidates(idx, done, NOW, manifest={"T": m}.get) == []
    # through an entry merged_into it (no merge record in the index)
    del idx["T"]["merged"]
    assert ar.low_nse_candidates(idx, done, NOW, manifest={"T": m}.get) == []
    # through the manifest's merged_from (neither in the index)
    del idx["F1"]
    assert ar.low_nse_candidates(idx, done, NOW, manifest={"T": m}.get) == []
    m["merged_from"] = ["F2"]
    assert [c["event"] for c in ar.low_nse_candidates(idx, done, NOW,
                                                      manifest={"T": m}.get)] == ["T"]


def test_merge_restamp_is_no_new_publish():
    t0 = NOW - dt.timedelta(hours=24 * 44)
    # gauge calibrated 40 days ago (finished 3 h later); the flood's fragments
    # were simulated before it and merged yesterday: generated = the merge
    done = [{"gauge": "09000001", "event": "X", "ran": True, "at": _t(24 * 40),
             "finished": _t(24 * 40 - 3)}]
    s, m = _merged("T", "09000001", t0, ["F1"], _t(20))
    idx = {"T": s, "F1": index_entry("09000001", t0 - dt.timedelta(hours=30),
                                     gen=_t(24 * 45), merged_into="T")}
    assert ar.published_at(s, idx, _t(20), ["F1"]) == t0           # target t0 > F1's publish
    assert ar.low_nse_candidates(idx, done, NOW, manifest={"T": m}.get) == []
    # a member published after the calibration finished: fresh
    idx["F1"]["generated"] = _t(24 * 30)
    assert ar.published_at(s, idx, _t(20), ["F1"]) == NOW - dt.timedelta(hours=24 * 30)
    assert [c["event"] for c in ar.low_nse_candidates(idx, done, NOW,
                                                      manifest={"T": m}.get)] == ["T"]
    # the target re-published (re-simulated) since the merge: its generated counts
    idx["F1"]["generated"] = _t(24 * 45)
    idx["T"]["generated"] = _t(2)
    assert ar.published_at(idx["T"], idx, _t(20), ["F1"]) == NOW - dt.timedelta(hours=2)
    ar._verdicts.clear()
    assert len(ar.low_nse_candidates(idx, done, NOW, manifest={"T": m}.get)) == 1
    # no merge: generated as is
    assert ar.published_at({"generated": _t(5)}, {}, None) == NOW - dt.timedelta(hours=5)


def test_verdict_memoised_per_publish(monkeypatch):
    t0 = NOW - dt.timedelta(hours=40)
    calls = []

    class _ES:
        @staticmethod
        def load_manifest(ev):
            calls.append(ev)
            return manifest(t0, *scaled(2.0)) if ev != "BAD" else None
    _set_module(monkeypatch, "eventstore", _ES)
    idx = {"E": index_entry("09000001", t0), "BAD": index_entry("09000002", t0)}
    for _ in range(3):
        assert len(ar.low_nse_candidates(idx, [], NOW)) == 1
    assert calls == ["E", "BAD", "BAD", "BAD"]                   # unreadable: retried
    idx["E"]["generated"] = _t(1)                                # re-published: judged again
    ar.low_nse_candidates(idx, [], NOW)
    assert calls.count("E") == 2
    del idx["E"]                                                 # rolled off the list
    ar.low_nse_candidates(idx, [], NOW)
    assert not any(k[0] == "E" for k in ar._verdicts)


def test_daily_cap():
    idx, mf = episodes({"A": ("09000001", 2.0)})
    done = [{"gauge": "09000009", "kind": "low_nse", "event": "Y", "ran": True, "at": _t(2)},
            {"gauge": "09000008", "kind": "oversim", "event": "Z", "ran": True, "at": _t(20)}]
    rec, why = ar.pick_candidate(idx, done, NOW, manifest=mf.get)
    assert rec is None and why.startswith("daily cap reached (2/2")
    done[1]["at"] = _t(25)                                       # rolled out of the 24 h window
    rec, _ = ar.pick_candidate(idx, done, NOW, manifest=mf.get)
    assert rec["event"] == "A"
    skips = [{"gauge": "09000007", "event": "S", "ran": False, "at": _t(1)}] * 5
    assert ar.pick_candidate(idx, skips, NOW, manifest=mf.get)[0] is not None


def test_misses_and_backlog_trigger_nothing(monkeypatch, tmp_path):
    for name in ("streaks", "miss_candidates", "oversim_candidates", "backlog_candidate",
                 "load_backlog", "backlog_skips"):
        assert not hasattr(ar, name), name
    assert not {"missed", "backlog"} & set(inspect.signature(ar.pick_candidate).parameters)
    # an outlet-Q miss on a hydrograph that scores well is no trigger
    t0 = NOW - dt.timedelta(hours=30)
    idx = {"A": index_entry("09000001", t0, trigger={"gauge": "09000001", "outlet_q_advisory": {
        "reason": "severe_underprediction", "frac": 0.2}})}
    mf = {"A": manifest(t0, *scaled(1.1))}
    rec, why = ar.pick_candidate(idx, [], NOW, manifest=mf.get)
    assert rec is None and why == "no eligible episode with NSE < 0.1"
    # the loop never reads events/missed.json (the fake raises if it does)
    es = _ES()
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, None, 1)
    assert picked == [] and ar.state["phase"].startswith(
        "idle — no eligible episode with NSE < 0.1 (0 episodes listed")


# --------------------------------------------------------------------------- #
# reservoir hold (PI 2026-10-27): no low-NSE episode calibrates a WESTERN
# >= 10 mm gauge until AUTOCAL_LAKES_ON=1; eastern gauges are never held
# --------------------------------------------------------------------------- #
def test_western_reservoir_gauges_held(monkeypatch):
    idx, mf = episodes({"W": ("09000004", 9.0), "W2": ("09000001", 3.0),
                        "E": ("07000003", 2.0)})
    RESV.update({"09000004": 267.0, "09000001": 11.0, "07000003": 40.0})
    held = []
    rec, why = ar.pick_candidate(idx, [], NOW, manifest=mf.get, held=held)
    assert rec["event"] == "E"                                   # east: never held
    assert "2 western reservoir gauge(s) held until AUTOCAL_LAKES_ON=1" in why
    assert sorted((h["kind"], h["gauge"], h["stor_mm"], h["west"], tuple(h["events"]))
                  for h in held) == [("low_nse", "09000001", 11.0, True, ("W2",)),
                                     ("low_nse", "09000004", 267.0, True, ("W",))]
    # nothing at all when every candidate is held
    held = []
    rec, why = ar.pick_candidate({"W": idx["W"]}, [], NOW, manifest=mf.get, held=held)
    assert rec is None and len(held) == 1
    assert why.startswith("no eligible episode with NSE < 0.1; 1 western reservoir gauge(s)")
    # the held list stays current while the daily cap is reached
    done = [{"gauge": "x", "ran": True, "at": _t(1)}, {"gauge": "y", "ran": True, "at": _t(2)}]
    held = []
    rec, why = ar.pick_candidate(idx, done, NOW, manifest=mf.get, held=held)
    assert rec is None and why.startswith("daily cap") and len(held) == 2
    # lakes on (flag + live module): the hold lifts, the worst western NSE leads
    monkeypatch.setattr(ar, "LAKES_ON", True)
    _lakes(monkeypatch)
    held = []
    rec, _ = ar.pick_candidate(idx, [], NOW, manifest=mf.get, held=held)
    assert rec["event"] == "W" and held == []


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
    h = [{"gauge": "10171000", "event": "2026101202_10171000", "kind": "low_nse",
          "west": True, "stor_mm": 266.53, "nse": -3.2}]
    ar._note_held(h)
    ar._note_held(list(h))                       # unchanged set: no second log line
    assert ar.state["held_reservoir"] == h and len(lines) == 1
    assert "10171000 (low_nse, 267 mm)" in lines[0]


def test_reservoir_table_reader(monkeypatch):
    monkeypatch.setattr(ar, "_resv", None)
    assert _reservoir_stor("10171000") == pytest.approx(266.53)   # Jordan R. (Utah Lake)
    assert _reservoir_stor("09371492") is None                     # Mud Creek: dam-free
    monkeypatch.setattr(ar, "_resv", None)
    monkeypatch.setattr(ar, "RESERVOIR_PATH", os.path.join(ROOT, "no", "such.json"))
    with pytest.raises(RuntimeError):                              # never fails open
        _reservoir_stor("10171000")
    monkeypatch.setattr(ar, "_resv", None)


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
    idx, mf = episodes({"A": ("09000001", 2.0)})
    held = []
    rec, why = ar.pick_candidate(idx, [], NOW, manifest=mf.get, held=held)
    assert rec is None and len(held) == 1
    assert "1 western reservoir gauge(s) held although AUTOCAL_LAKES_ON=1: hf_data.lakes" in why
    rec, why = ar.pick_candidate({}, [], NOW)
    assert rec is None and "reservoir hold kept although AUTOCAL_LAKES_ON=1" in why
    _lakes(monkeypatch)
    rec, why = ar.pick_candidate(idx, [], NOW, manifest=mf.get)
    assert rec["gauge"] == "09000001" and "hold" not in why


# --------------------------------------------------------------------------- #
# idle gate: yields ONLY to user-initiated work, fails closed (PI 2026-11-05:
# autocal takes no 2-D compute, so 2-D activity is no gate)
# --------------------------------------------------------------------------- #
TWO_D_BUSY = {"queue": {"unpublished": 3, "depth": 5}, "events": {"active": 4},
              "workers": {"active_claims": 2}, "tick": {"running": "run:09000001"}}


def _health(monkeypatch, payload=None, exc=None):
    """Fake `requests` for busy_reason: get(...).json() -> payload, or raise."""
    def get(url, timeout=None):
        if exc is not None:
            raise exc
        return types.SimpleNamespace(json=lambda: payload)
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=get))


def test_busy_reason_yields_only_to_user_work(monkeypatch):
    _health(monkeypatch, {"user_ops": {"sims": 0, "cals": 0}, **TWO_D_BUSY})
    assert ar.busy_reason() is None                              # 2-D activity: no gate
    _health(monkeypatch, {"user_ops": {"sims": 0, "cals": 0}})
    assert ar.busy_reason() is None
    _health(monkeypatch, {"user_ops": {"sims": 1, "cals": 0}})
    assert ar.busy_reason() == "user work running (1 sims, 0 cals)"
    _health(monkeypatch, {"user_ops": {"sims": 0, "cals": 2}, **TWO_D_BUSY})
    assert ar.busy_reason().startswith("user work running")


def test_busy_reason_fails_closed(monkeypatch):
    _health(monkeypatch, exc=ConnectionError("down"))
    assert ar.busy_reason() == "health unreadable (ConnectionError)"
    _health(monkeypatch, {"user_ops": {"error": "ImportError"}, **TWO_D_BUSY})
    assert ar.busy_reason() == "health user_ops unreadable (ImportError)"
    _health(monkeypatch, {"queue": {"unpublished": 0}})
    assert ar.busy_reason().startswith("health has no user_ops")
    _health(monkeypatch, ["not", "a", "dict"])
    assert ar.busy_reason().startswith("health has no user_ops")


REC = {"kind": "low_nse", "gauge": "09000004", "event": "2026102710_09000004",
       "events": ["2026102710_09000004"], "west": True, "status": "active",
       "t0": (NOW - dt.timedelta(hours=1)).strftime(HM),
       "t_end": (NOW + dt.timedelta(hours=11)).strftime(HM),
       "nse": -0.42371, "n_pairs": 20,
       # the flood's 20 paired hours: the first 20 of the 2-day test window
       "nse_span": [(NOW - dt.timedelta(hours=50)).strftime(HM),
                    (NOW - dt.timedelta(hours=31)).strftime(HM)],
       "obs_peak_m3s": 3.0, "sim_peak_m3s": 30.0}


TRIED = [{"id": "c1", "nse": -0.3}, {"id": "c2", "nse": 0.35}, {"id": "c3", "nse": None}]


def _calib_env(monkeypatch, series=(), fetch_error=None, done=None, rounds=1, tried=TRIED,
               errors=()) -> dict:
    """Fake obs / llm / calibrate for calibrate_one (2-day window); the obs
    fake asked NWIS for the whole window (from_store False), failing with
    the exception `fetch_error` when given, and answers every request (a
    refresh of the flood's paired hours too) with `series`. Each round
    tries `tried` (LLM candidates); the calibration swallowed `errors`
    (cand_errors). Returns what run_calibration was called with ("kw",
    "calls": its count)."""
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    monkeypatch.setattr(ar, "busy_reason", lambda: None)
    monkeypatch.setattr(ar, "log", lambda s: None)
    seen = {"calls": 0}

    def get_series(gid, a, b, info=None, refresh=False):
        info["from_store"] = False
        if fetch_error is not None:
            info.update({"fetch_error": str(fetch_error), "fetch_exc": fetch_error})
        return list(series)

    def run_calibration(*a, **k):
        seen.update({"kw": k, "calls": seen["calls"] + 1})
        k["cand_errors"].extend(errors)
        for i in range(rounds):
            yield "round", {"round": i + 1, "tried": [dict(t) for t in tried],
                            "best_nse": None, "source": "llm"}
        yield "done", done
    _set_module(monkeypatch, "obs", types.SimpleNamespace(get_series=get_series))
    _set_module(monkeypatch, "llm", types.SimpleNamespace(available=lambda: True))
    _set_module(monkeypatch, "calibrate", types.SimpleNamespace(run_calibration=run_calibration))
    return seen


def _full_series():
    a, b = ar.window(REC, NOW)
    return [(a + dt.timedelta(hours=i), 1.0)
            for i in range(int((b - a).total_seconds() // 3600) + 1)]


OK = {"baseline_nse": -0.4, "best_nse": 0.35, "best_params": {"wm": 2.0}, "saved": True,
      "improved": True, "rounds": 1, "llm_rounds": 1, "pushed": True, "model": "crest"}


def test_preemption_follows_the_same_rule(monkeypatch):
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    _calib_env(monkeypatch, series=_full_series(), done=OK, rounds=3)
    monkeypatch.setattr(ar, "busy_reason", _busy_reason)         # the real gate
    _health(monkeypatch, {"user_ops": {"sims": 0, "cals": 0}, **TWO_D_BUSY})
    assert ar.calibrate_one(REC, NOW)["ran"] is True             # 2-D work never preempts
    _health(monkeypatch, {"user_ops": {"sims": 1, "cals": 0}})
    with pytest.raises(ar.Preempted, match="user work running"):
        ar.calibrate_one(REC, NOW)
    _health(monkeypatch, exc=TimeoutError())
    with pytest.raises(ar.Preempted, match="health unreadable"):  # fails closed
        ar.calibrate_one(REC, NOW)


def test_record_fields(monkeypatch):
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    _calib_env(monkeypatch, series=_full_series(), done=OK)
    r = ar.calibrate_one(REC, NOW)
    assert {k: r[k] for k in ("kind", "gauge", "event", "events", "west", "status", "nse",
                              "n_pairs", "nse_max", "ran", "saved")} == {
        "kind": "low_nse", "gauge": "09000004", "event": REC["event"],
        "events": [REC["event"]], "west": True, "status": "active", "nse": -0.424,
        "n_pairs": 20, "nse_max": 0.1, "ran": True, "saved": True}
    assert r["baseline_nse"] == -0.4 and r["best_nse"] == 0.35 and r["n_runs"] == 3
    assert (r["n_nse"], r["rounds"], r["llm_rounds"], r["pushed"], r["model"]) == (
        2, 1, 1, True, "crest")
    assert r["window"][1] == (NOW - dt.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    assert r["obs_cover"] == 1.0 and ar._parse(r["finished"]) >= ar._parse(r["at"])
    assert r["flood_obs_h"] == {REC["event"]: [20, 20]} and "flood_refetched" not in r
    json.dumps(r)                                                # committed as JSON
    # a gauge candidate with several episodes: all of them in `events`
    many = {**REC, "event": "B", "events": ["A", "B"], "episode_nse": {"A": -0.1, "B": -0.4}}
    r = ar.calibrate_one(many, NOW)
    assert (r["event"], r["events"], r["episode_nse"]) == ("B", ["A", "B"], many["episode_nse"])
    _, handled, _ = ar._history([r], NOW)
    assert handled == {("09000004", "A"), ("09000004", "B")}
    # a skip carries no finish (it never ran)
    _calib_env(monkeypatch, series=_full_series()[:10], done=OK)
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and "finished" not in r and 0 < r["obs_cover"] < 0.5


def test_window_ends_at_the_episode_end():
    t0 = NOW - dt.timedelta(hours=40)
    idx, mf = episodes({"A": ("09000001", 2.0)}, t0=t0)
    c = ar.low_nse_candidates(idx, [], NOW, manifest=mf.get)[0]
    a, b = ar.window(c, NOW)
    assert b == t0 + dt.timedelta(hours=12) and b - a == dt.timedelta(days=183)
    assert ar.window({"t0": t0.strftime(HM)}, NOW)[1] == t0     # no t_end: its t0
    assert ar.window({}, NOW + dt.timedelta(minutes=37))[1] == NOW - dt.timedelta(hours=2)


# --------------------------------------------------------------------------- #
# outages, errors, history, pending commits
# --------------------------------------------------------------------------- #
from hf_data import eventstore as es_mod                       # noqa: E402


def _http_error(code: int):
    """requests' HTTPError as raise_for_status raises it."""
    requests = pytest.importorskip("requests")
    kind = "Server" if code >= 500 else "Client"
    return requests.HTTPError(f"{code} {kind} Error: x for url: https://x",
                              response=types.SimpleNamespace(status_code=code))


def _no_json():
    """What r.json() raises on a WAF or maintenance page."""
    requests = pytest.importorskip("requests")
    r = requests.Response()
    r.status_code, r._content = 200, b"<html>Request blocked</html>"
    try:
        r.json()
    except Exception as e:
        return e
    raise AssertionError("an HTML body parsed as JSON")


def test_fetch_outage_classification():
    requests = pytest.importorskip("requests")
    rx = requests.exceptions
    outages = (rx.ReadTimeout("HTTPSConnectionPool(host='waterservices.usgs.gov', "
                              "port=443): Read timed out. (read timeout=30)"),
               rx.ConnectTimeout("connect timed out"), rx.ConnectionError("Max retries"),
               rx.SSLError("certificate verify failed"),
               rx.ChunkedEncodingError("Connection broken: IncompleteRead"),
               ConnectionResetError("reset"), TimeoutError(),
               _no_json(), json.JSONDecodeError("Expecting value", "<html>", 0),
               _http_error(503), _http_error(500), _http_error(429), _http_error(403),
               _http_error(401))
    # the site's own: another 4xx, or its answer failing to parse
    own = (_http_error(404), _http_error(410), _http_error(400),
           IndexError("list index out of range"), KeyError("dateTime"),
           ValueError("Invalid isoformat string: '2026-13-01T00:00:00'"),
           TypeError("'NoneType' object is not subscriptable"))
    for e in outages:
        assert ar.fetch_outage(e), repr(e)
    for e in own:
        assert not ar.fetch_outage(e), repr(e)
    assert not ar.fetch_outage(None)


def test_fetch_outage_agrees_with_is_outage():
    for code in (400, 401, 403, 404, 410, 429, 500, 502, 503):
        e = _http_error(code)
        assert ar.fetch_outage(e) == ar.is_outage(e), code


def test_obs_fetch_outage_records_nothing_gauge_404_skips(monkeypatch):
    requests = pytest.importorskip("requests")
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    for err in (requests.exceptions.ReadTimeout("HTTPSConnectionPool(host='x', port=443): "
                                                "Read timed out."),
                _http_error(502), _http_error(403), _no_json()):
        _calib_env(monkeypatch, fetch_error=err)
        with pytest.raises(ar.Outage):
            ar.calibrate_one(REC, NOW)
    _calib_env(monkeypatch, fetch_error=_http_error(404))
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and r["reason"].startswith("obs cover 0%")
    assert r["kind"] == "low_nse" and r["nse"] == -0.424
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)  # a 404 may be the endpoint's
    # a site's answer failing to parse is its own error, not an outage that
    # would be retried every poll
    _calib_env(monkeypatch, fetch_error=IndexError("list index out of range"))
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and r["reason"].endswith("; fetch: list index out of range")
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)
    _calib_env(monkeypatch, series=(), fetch_error=None)        # NWIS 200, empty series
    r = ar.calibrate_one(REC, NOW)
    assert r["reason"].startswith("obs cover 0%") and ar.skip_ttl(r["reason"]) == dt.timedelta(
        hours=24)
    _calib_env(monkeypatch, series=_full_series()[:10], fetch_error=None)
    r = ar.calibrate_one(REC, NOW)
    assert r["reason"].startswith("obs cover") and ar.skip_ttl(r["reason"]) == dt.timedelta(
        days=30)


def _store_obs(monkeypatch, answers: list) -> list:
    """obs fake whose first call is served (in part) from the store: each
    call pops (series, NWIS answer) from `answers` — the answer an exception
    (the fetch failed), "empty" (no row) or None (rows); returns the
    refresh flag of every call."""
    calls = []

    def get_series(gid, a, b, info=None, refresh=False):
        calls.append(refresh)
        series, ans = answers.pop(0)
        info.update({"from_store": not refresh, "fetched_windows": 0, "empty_windows": 0})
        if isinstance(ans, BaseException):
            info.update({"fetch_error": str(ans), "fetch_exc": ans})
        else:
            info.update({"fetched_windows": 1, "empty_windows": int(ans == "empty")})
        return list(series)
    _set_module(monkeypatch, "obs", types.SimpleNamespace(get_series=get_series))
    return calls


def test_low_cover_from_the_store_is_asked_of_nwis_once_more(monkeypatch):
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    _calib_env(monkeypatch, done=OK)
    # the store's earlier (empty) answer said 0%: NWIS is asked again first
    calls = _store_obs(monkeypatch, [((), None), (_full_series(), None)])
    r = ar.calibrate_one(REC, NOW)
    assert calls == [False, True] and r["ran"] is True and r["obs_cover"] == 1.0
    assert r["obs_refetched"] is True
    # still too little: a skip on the fresh answer (a gauge short of obs)
    calls = _store_obs(monkeypatch, [(_full_series()[:10], None)] * 2)
    r = ar.calibrate_one(REC, NOW)
    assert calls == [False, True] and r["ran"] is False and 0 < r["obs_cover"] < 0.5
    assert r["reason"] == "obs cover 21% of the window (< 50%)" and r["obs_refetched"] is True
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(days=30)
    # NWIS answers no row for the whole window although the store holds
    # some in it: failing quietly, the short TTL
    calls = _store_obs(monkeypatch, [(_full_series()[:10], None),
                                     (_full_series()[:10], "empty")])
    r = ar.calibrate_one(REC, NOW)
    assert r["reason"] == f"obs cover 21% of the window (< 50%); {ar.QUIET_EMPTY}"
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)
    # the site's own fetch error on the part asked: asked once more as well
    calls = _store_obs(monkeypatch, [((), IndexError("list index out of range")),
                                     (_full_series(), None)])
    assert ar.calibrate_one(REC, NOW)["ran"] is True and calls == [False, True]
    # an outage is no reason to ask again: back off
    calls = _store_obs(monkeypatch, [((), _http_error(503))])
    with pytest.raises(ar.Outage):
        ar.calibrate_one(REC, NOW)
    assert calls == [False]
    # the whole window asked this call (_calib_env): decided as answered
    _calib_env(monkeypatch, series=(), done=OK)
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and "obs_refetched" not in r


def test_empty_nwis_answer_is_asked_again_on_the_next_try(monkeypatch, tmp_path):
    """The real obs store: NWIS 200 with an empty series is not cached as
    covered, so once the skip expires the gauge's window is asked again and
    calibrated; an empty answer next to an episode's rows on one side only
    is not trusted either (the flood's paired hours are asked once more
    before the skip), and the next try asks NWIS for it again."""
    from hf_data import obs, statecache
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path))
    nwis = {"up": False}
    asked = []

    def fetch(site, a, b):
        asked.append((a, b))
        if not nwis["up"]:
            return []
        return [(a + dt.timedelta(hours=i), 1.0)
                for i in range(int((b - a).total_seconds() // 3600) + 1)]
    monkeypatch.setattr(obs, "fetch_usgs_discharge", fetch)
    _calib_env(monkeypatch, done=OK)
    _set_module(monkeypatch, "obs", obs)
    a, b = ar.window(REC, NOW)
    s0, s1 = (ar._parse(x) for x in REC["nse_span"])
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and r["obs_cover"] == 0 and asked == [(a, b), (s0, s1)]
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24) and "obs_refetched" not in r
    assert r["flood_refetched"] is True and ar.QUIET_EMPTY in r["reason"]
    nwis["up"] = True                                           # after the skip's TTL
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is True and r["obs_cover"] == 1.0 and asked[2:] == [(a, b)]
    # an episode's rows in the store, then an empty answer next to them on
    # one side (a gap open towards the past: not trusted, not covered) —
    # the low cover is asked of NWIS once more, the flood's hours too
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path / "2"))
    obs.get_series(REC["gauge"], b - dt.timedelta(hours=6), b)
    nwis["up"] = False
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and r["obs_refetched"] is True and 0 < r["obs_cover"] < 0.5
    assert ar.QUIET_EMPTY in r["reason"] and ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)
    assert obs._load_store(REC["gauge"])[1] == [[b - dt.timedelta(hours=6), b]]
    nwis["up"] = True
    asked.clear()
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is True and "obs_refetched" not in r and asked == [(a, b - dt.timedelta(hours=6))]


def test_no_nse_is_not_a_run(monkeypatch):
    none = {**OK, "baseline_nse": None, "best_nse": None, "best_params": {}, "saved": False,
            "pushed": None}
    monkeypatch.setattr(ar, "WINDOW_D", 2.0)
    _calib_env(monkeypatch, series=_full_series(), done=none)
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and r["reason"] == "no EF5 run produced an NSE" and r["n_runs"] == 3
    _calib_env(monkeypatch, series=_full_series(), done={**none, "baseline_nse": 0.2})
    assert ar.calibrate_one(REC, NOW)["ran"] is True            # one NSE: it ran


def test_no_candidate_nse_is_not_a_run(monkeypatch):
    """Only the baseline scored: the calibration judged nothing (round
    payloads' tried[].nse, the baseline aside) — a skip with the error TTL,
    or an outage when every candidate failed for reasons not the gauge's."""
    requests = pytest.importorskip("requests")
    failed = [{"id": "c1", "nse": None}, {"id": "c2", "nse": float("nan")},
              {"id": "baseline", "nse": -0.4}]                  # a baseline never counts
    base_only = {**OK, "best_nse": -0.4, "saved": False, "improved": False, "pushed": None}
    _calib_env(monkeypatch, series=_full_series(), done=base_only, tried=failed, rounds=2)
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and r["reason"] == "no candidate run produced an NSE"
    assert (r["n_nse"], r["n_runs"]) == (0, 6) and "finished" not in r
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)
    _, handled, today = ar._history([r], NOW)
    assert not handled and today == 0
    # the gauge's own failures: named in the skip
    disk = OSError(28, "No space left on device")
    _calib_env(monkeypatch, series=_full_series(), done=base_only, tried=failed,
               errors=[disk] * 4)
    r = ar.calibrate_one(REC, NOW)
    assert r["reason"] == ("no candidate run produced an NSE (4 candidate run(s) failed; "
                           "last: OSError: [Errno 28] No space left on device)")
    # every failure an outage (network, Hub, memory): nothing recorded
    for errs in ([requests.ConnectionError("forcing download failed")] * 3,
                 [TimeoutError(), MemoryError()]):
        _calib_env(monkeypatch, series=_full_series(), done=base_only, tried=failed,
                   errors=errs)
        with pytest.raises(ar.Outage, match="no candidate run produced an NSE for 09000004"):
            ar.calibrate_one(REC, NOW)
    # the gauge's own error among them: a skip
    _calib_env(monkeypatch, series=_full_series(), done=base_only, tried=failed,
               errors=[requests.ConnectionError("x"), KeyError("wm")])
    assert ar.calibrate_one(REC, NOW)["ran"] is False
    # one candidate with an NSE, however poor: it ran
    _calib_env(monkeypatch, series=_full_series(), done=base_only,
               tried=failed + [{"id": "c3", "nse": -9.0}])
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is True and r["n_nse"] == 1


def test_no_llm_round_is_an_outage(monkeypatch):
    """AUTOCAL_REQUIRE_LLM: run_calibration is asked to stop before a
    heuristic-only search (nothing saved or pushed), and a calibration
    without an LLM round is an outage — nothing recorded, the gauge
    retried; a heuristic winner is never recorded."""
    stop = {**OK, "best_nse": -0.4, "saved": False, "improved": False, "pushed": None,
            "rounds": 0, "llm_rounds": 0,
            "llm_stopped": "LLM error: RateLimitError: Error code: 429 - insufficient_quota"}
    seen = _calib_env(monkeypatch, series=_full_series(), done=stop, rounds=0)
    with pytest.raises(ar.Outage, match=r"no LLM candidates for 09000004 \(LLM error: "
                                        r"RateLimitError"):
        ar.calibrate_one(REC, NOW)
    assert seen["kw"]["require_llm"] is True and seen["kw"]["save_vs_baseline"] is True
    # a payload that does not say: no proof of an LLM round
    _calib_env(monkeypatch, series=_full_series(),
               done={k: v for k, v in OK.items() if k != "llm_rounds"})
    with pytest.raises(ar.Outage, match="heuristic rounds only"):
        ar.calibrate_one(REC, NOW)
    # one LLM round is an AI calibration (the LLM failing later falls back)
    _calib_env(monkeypatch, series=_full_series(), done={**OK, "rounds": 4, "llm_rounds": 1})
    assert ar.calibrate_one(REC, NOW)["ran"] is True
    # AUTOCAL_REQUIRE_LLM=0: a heuristic search counts
    monkeypatch.setattr(ar, "REQUIRE_LLM", False)
    seen = _calib_env(monkeypatch, series=_full_series(), done={**OK, "llm_rounds": 0})
    assert ar.calibrate_one(REC, NOW)["ran"] is True and seen["kw"]["require_llm"] is False


def test_unpushed_winner_is_unfinished(monkeypatch):
    _calib_env(monkeypatch, series=_full_series(), done={**OK, "pushed": False})
    r = ar.calibrate_one(REC, NOW)
    assert (r["ran"], r["saved"], r["pushed"], r["model"]) == (True, True, False, "crest")
    assert "finished" not in r                                  # finished at the push
    # nothing saved: nothing to push, finished at once
    _calib_env(monkeypatch, series=_full_series(),
               done={**OK, "saved": False, "improved": False, "pushed": None})
    r = ar.calibrate_one(REC, NOW)
    assert r["pushed"] is None and ar._parse(r["finished"]) >= NOW


# --------------------------------------------------------------------------- #
# the flood's own observations
# --------------------------------------------------------------------------- #
def _flood_obs(monkeypatch, window_series, refresh_answer) -> list:
    """obs fake: the window's request answers `window_series` (asked of
    NWIS this call), a refresh of the flood's span answers `refresh_answer`
    (rows, or an exception: the fetch failed). Returns every request."""
    calls = []

    def get_series(gid, a, b, info=None, refresh=False):
        calls.append((a, b, refresh))
        info["from_store"] = False
        if not refresh:
            return list(window_series)
        if isinstance(refresh_answer, BaseException):
            info.update({"fetch_error": str(refresh_answer), "fetch_exc": refresh_answer})
            return []
        info["empty_windows"] = int(not refresh_answer)
        return list(refresh_answer)
    _set_module(monkeypatch, "obs", types.SimpleNamespace(get_series=get_series))
    return calls


def test_flood_obs_must_be_in_the_series(monkeypatch):
    """The runner paired obs over the episode's nse_span, so NWIS has them:
    most of those hours missing is asked of NWIS once more, and without them
    the gauge is not calibrated (QUIET_EMPTY, the error TTL)."""
    seen = _calib_env(monkeypatch, done=OK)
    a, b = ar.window(REC, NOW)
    s0, s1 = (ar._parse(x) for x in REC["nse_span"])
    full = _full_series()
    flood = [(t, q) for t, q in full if s0 <= t <= s1]
    rest = [(t, q) for t, q in full if not s0 <= t <= s1]       # 29 of 49 hours: cover 60%
    # NWIS answers the flood's hours when asked again: calibrated
    calls = _flood_obs(monkeypatch, rest, flood)
    r = ar.calibrate_one(REC, NOW)
    assert calls == [(a, b, False), (s0, s1, True)] and r["ran"] is True
    assert r["flood_refetched"] is True and r["flood_obs_h"] == {REC["event"]: [20, 20]}
    assert r["obs_cover"] == 1.0 and seen["calls"] == 1
    # still missing: never calibrated without the flood
    calls = _flood_obs(monkeypatch, rest, [])
    r = ar.calibrate_one(REC, NOW)
    assert r["ran"] is False and seen["calls"] == 1 and len(calls) == 2
    assert r["reason"] == (f"flood obs missing: {REC['event']} 0 of 20 paired hours; "
                           f"{ar.QUIET_EMPTY}")
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24) and "finished" not in r
    # a few of the flood's hours (fewer than half): the same
    calls = _flood_obs(monkeypatch, rest + flood[:9], flood[:9])
    assert ar.calibrate_one(REC, NOW)["reason"].startswith(
        f"flood obs missing: {REC['event']} 9 of 20")
    # half of them or more: no second request, calibrated
    calls = _flood_obs(monkeypatch, rest + flood[:10], AssertionError("not asked"))
    assert ar.calibrate_one(REC, NOW)["ran"] is True and len(calls) == 1
    # the refresh failing for a reason not the gauge's: an outage
    calls = _flood_obs(monkeypatch, rest, _http_error(503))
    with pytest.raises(ar.Outage, match="0 of the 20 paired hours"):
        ar.calibrate_one(REC, NOW)
    # the site's own fetch error: a skip naming it (error TTL)
    calls = _flood_obs(monkeypatch, rest, _http_error(404))
    r = ar.calibrate_one(REC, NOW)
    assert r["reason"].startswith(f"flood obs missing: {REC['event']} 0 of 20 paired hours; "
                                  f"fetch: 404 Client Error")
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)
    # too little obs for the window AND the flood missing: NWIS failing, not
    # a gauge short of obs — the error TTL, not SKIP_TTL_D
    calls = _flood_obs(monkeypatch, rest[-12:], [])
    r = ar.calibrate_one(REC, NOW)
    assert r["reason"].startswith("obs cover 25% of the window (< 50%); flood obs missing")
    assert r["reason"].endswith(ar.QUIET_EMPTY)
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)


def test_flood_spans():
    win = (NOW - dt.timedelta(days=183), NOW - dt.timedelta(hours=2))
    ta, tb = NOW - dt.timedelta(hours=100), NOW - dt.timedelta(hours=40)
    idx = {"A": index_entry("09000001", ta), "B": index_entry("09000001", tb)}
    mf = {"A": manifest(ta, *scaled(1.5)), "B": manifest(tb, *scaled(3.0))}
    mf["A"]["hydro"][3]["obs_q"] = None                          # 13 of A's 14 hours paired
    c = ar.low_nse_candidates(idx, [], NOW, manifest=mf.get)[0]
    assert c["episode_pairs"] == {"A": 13, "B": 14}
    assert c["episode_span"] == {"A": [(ta - dt.timedelta(hours=13)).strftime(HM),
                                       ta.strftime(HM)],
                                 "B": [(tb - dt.timedelta(hours=13)).strftime(HM),
                                       tb.strftime(HM)]}
    # every episode of the gauge, each with its own paired hours
    assert ar.flood_spans(c, *win) == [("A", ta - dt.timedelta(hours=13), ta, 13),
                                       ("B", tb - dt.timedelta(hours=13), tb, 14)]
    # clipped to the window (an active episode's last hours after its end)
    assert ar.flood_spans(c, win[0], tb - dt.timedelta(hours=3))[1] == (
        "B", tb - dt.timedelta(hours=13), tb - dt.timedelta(hours=3), 11)
    assert ar.flood_spans(c, win[0], ta - dt.timedelta(hours=20)) == []
    # a record without per-episode spans: its nse_span (the worst episode's)
    old = {k: v for k, v in c.items() if k not in ("episode_span", "episode_pairs")}
    assert ar.flood_spans(old, *win) == [("B", tb - dt.timedelta(hours=13), tb, 14)]
    # without one: the 2 days before its t0 to its t_end
    bare = {"event": "X", "t0": tb.strftime(HM), "t_end": (tb + dt.timedelta(hours=12)).strftime(HM),
            "n_pairs": 20, "nse_span": ["a", "b"]}
    assert ar.flood_spans(bare, *win) == [("X", tb - dt.timedelta(days=2),
                                           tb + dt.timedelta(hours=12), 20)]
    assert ar.flood_spans({"event": "Y"}, *win) == []


def test_flood_gap_open_towards_now_is_not_calibrated(monkeypatch, tmp_path):
    """The real obs store: it holds an earlier window of the gauge (an
    earlier calibration); NWIS answers 200 with an empty series for the
    rest, which ends with the flood. The window is covered well enough
    without it — the flood's hours decide: not calibrated, and asked again
    (not covered) once NWIS is back."""
    from hf_data import obs, statecache
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path))
    nwis = {"up": True}
    asked = []

    def fetch(site, a, b):
        asked.append((nwis["up"], a, b))
        if not nwis["up"]:
            return []
        return [(a - dt.timedelta(days=1) + dt.timedelta(hours=i), 3.0 + i % 5)
                for i in range(int((b - a).total_seconds() // 3600) + 49)]
    monkeypatch.setattr(obs, "fetch_usgs_discharge", fetch)
    seen = _calib_env(monkeypatch, done=OK)
    _set_module(monkeypatch, "obs", obs)
    monkeypatch.setattr(ar, "WINDOW_D", 30.0)
    end = NOW - dt.timedelta(days=3)
    rec = {**REC, "t0": (end - dt.timedelta(days=1)).strftime(HM), "t_end": end.strftime(HM),
           "status": "ended", "nse_span": [(end - dt.timedelta(hours=30)).strftime(HM),
                                           (end - dt.timedelta(hours=6)).strftime(HM)],
           "n_pairs": 25}
    a, b = ar.window(rec, NOW)
    obs.get_series(rec["gauge"], a - dt.timedelta(days=30), b - dt.timedelta(days=10))
    nwis["up"] = False
    r = ar.calibrate_one(rec, NOW)
    assert r["ran"] is False and r["obs_cover"] > 0.5 and seen["calls"] == 0
    assert r["reason"] == f"flood obs missing: {rec['event']} 0 of 25 paired hours; " \
                          f"{ar.QUIET_EMPTY}"
    assert [(x, y) for u, x, y in asked if not u] == [
        (b - dt.timedelta(days=10), b), tuple(ar._parse(t) for t in rec["nse_span"])]
    cov = obs._load_store(rec["gauge"])[1]
    assert cov == [[a - dt.timedelta(days=30), b - dt.timedelta(days=10)]]
    nwis["up"] = True
    r = ar.calibrate_one(rec, NOW)
    assert r["ran"] is True and seen["calls"] == 1 and r["flood_obs_h"] == {rec["event"]: [25, 25]}


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
            ar.run_candidate(REC, NOW)
    # the gauge's own calibration failing: a calibration that did not run
    # (an error-TTL skip), never re-raised to be picked again every poll
    own = (ValueError("no basin"), KeyError("wm"),
           requests.HTTPError("404", response=types.SimpleNamespace(status_code=404)))
    for e in own:
        assert not ar.is_outage(e), e
        monkeypatch.setattr(ar, "calibrate_one", raising(e))
        r = ar.run_candidate(REC, NOW)
        assert r["reason"] == f"calibration error: {type(e).__name__}: {e}"
        assert {k: r[k] for k in ("gauge", "event", "events", "kind", "at", "ran")} == {
            "gauge": "09000004", "event": REC["event"], "events": REC["events"],
            "kind": "low_nse", "at": NOW.strftime(TS), "ran": False}
        assert ar.skip_ttl(r["reason"]) == dt.timedelta(hours=24)
        json.dumps(r)                                           # shown on the status page
    with pytest.raises(ar.Preempted):                           # user work: nothing recorded
        monkeypatch.setattr(ar, "calibrate_one", raising(ar.Preempted("user work")))
        ar.run_candidate(REC, NOW)


class _ES:
    """Fake eventstore: index + manifests, autocal.json rows on a fake Hub,
    commits that may fail. Reading events/missed.json is an error."""

    def __init__(self, rows=(), ok=True, index=None, manifests=None):
        self.rows = None if rows is None else list(rows)
        self.ledger, self.ok, self.tried = {}, ok, []
        self.index, self.manifests = dict(index or {}), dict(manifests or {})

    def load_missed(self):
        raise AssertionError("autocal must not read events/missed.json")

    def load_index(self):
        return dict(self.index)

    def load_manifest(self, ev):
        return self.manifests.get(ev)

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
    on_hub = {"gauge": "09000003", "event": "C", "kind": "low_nse", "ran": True, "at": _t(3)}
    a = {"gauge": "09000004", "event": "A", "kind": "low_nse", "ran": True, "at": _t(2)}
    b = {"gauge": "09000001", "event": "E", "kind": "low_nse", "ran": False, "at": _t(1)}
    ar.state["pending"] = [on_hub, a, b]
    done = ar._flush_pending([on_hub])
    assert done == [on_hub, a, b] and ar.state["pending"] == [a, b]
    assert es.tried == [a]                                      # stops at the first failure
    # pending records count for the cap, the handled pairs and the cooldown
    # (a ran=False one is a skip: active_skips, never handled)
    last_run, handled, today = ar._history(done, NOW)
    assert today == 2 and ("09000004", "A") in handled and "09000004" in last_run
    assert ("09000001", "E") not in handled
    assert ar.active_skips(done, {}, NOW)["09000001"]["events"] == ["E"]
    es.ok = True
    assert ar._flush_pending([on_hub]) == [on_hub, a, b]
    assert ar.state["pending"] == [] and es.tried[1:] == [a, b]


class _Stop(BaseException):
    """Ends main() at a chosen sleep."""


def _run_main(monkeypatch, tmp_path, es, calibrate, n_sleeps, busy=lambda: None,
              clock=None):
    """Run main() on the fake eventstore until its n_sleeps-th sleep:
    (every sleep in s, the gauges handed to calibrate_one). `clock` ({"s":
    seconds}): main's utcnow runs on it and every sleep advances it, so
    skip TTLs lapse."""
    sleeps, picked = [], []

    def sleep(s):
        sleeps.append(s)
        if clock is not None:
            clock["s"] += s
        if len(sleeps) >= n_sleeps:
            raise _Stop
    if clock is not None:
        t0 = dt.datetime.utcnow()

        class FakeDT(dt.datetime):
            @classmethod
            def utcnow(cls):
                return t0 + dt.timedelta(seconds=clock["s"])
        monkeypatch.setattr(ar, "dt", types.SimpleNamespace(datetime=FakeDT,
                                                             timedelta=dt.timedelta))

    def cal(rec, now):
        picked.append(rec["gauge"])
        return calibrate(rec, now)
    monkeypatch.setenv("HF_TOKEN", "x")
    monkeypatch.setattr(ar, "time", types.SimpleNamespace(sleep=sleep, time=lambda: 1e9))
    monkeypatch.setattr(ar, "STATE_PATH", str(tmp_path / "state.json"))
    for k, v in {"busy_reason": busy, "IDLE_FOR_S": 0.0, "log": lambda s: None,
                 "calibrate_one": cal}.items():
        monkeypatch.setattr(ar, k, v)
    _set_module(monkeypatch, "eventstore", es)
    _set_module(monkeypatch, "llm", types.SimpleNamespace(available=lambda: True))
    with pytest.raises(_Stop):
        ar.main()
    return sleeps, picked


def _low_nse_es(**kw):
    """Three western low-NSE episodes: 09000004 worst, then 09000003, 09000002."""
    idx, mf = episodes({"A": ("09000004", 9.0), "B": ("09000003", 5.0),
                        "C": ("09000002", 3.0)})
    return _ES(index=idx, manifests=mf, **kw)


def _ran(rec, now):
    return {"gauge": rec["gauge"], "kind": "low_nse", "event": rec["event"], "ran": True,
            "at": now.strftime(TS), "baseline_nse": rec["nse"], "best_nse": 0.3}


def test_main_waits_when_history_unreadable(monkeypatch, tmp_path):
    es = _ES(rows=None)
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, None, 1)
    assert sleeps == [900.0] and picked == [] and es.tried == []
    assert ar.state["phase"].startswith("history unreadable")
    es = _ES()
    es.ledger = None                                            # the ledger alone unreadable
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, None, 1)
    assert picked == [] and ar.state["phase"].startswith("history unreadable")


def test_main_calibrates_during_2d_activity_and_yields_to_users(monkeypatch, tmp_path):
    _health(monkeypatch, {"user_ops": {"sims": 0, "cals": 0}, **TWO_D_BUSY})
    es = _low_nse_es()
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, _ran, 1, busy=_busy_reason)
    assert picked == ["09000004"] and es.rows[0]["kind"] == "low_nse"
    _health(monkeypatch, {"user_ops": {"sims": 2, "cals": 0}, **TWO_D_BUSY})
    sleeps, picked = _run_main(monkeypatch, tmp_path, _low_nse_es(), _ran, 1,
                               busy=_busy_reason)
    assert picked == [] and sleeps == [300]
    assert ar.state["phase"] == "yielding — user work running (2 sims, 0 cals)"


def test_main_keeps_failed_commit_pending(monkeypatch, tmp_path):
    es = _low_nse_es(ok=False)
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, _ran, 3)
    # poll 2 skips the pending 09000004 (handled + cooldown), poll 3 hits the
    # daily cap with two pending runs; every poll retried the oldest pending commit
    assert picked == ["09000004", "09000003"] and sleeps == [30, 30, 900.0]
    assert [r["gauge"] for r in ar.state["pending"]] == ["09000004", "09000003"]
    assert [r["gauge"] for r in es.tried] == ["09000004", "09000004", "09000003", "09000004"]
    assert ar.state["phase"].startswith("idle — daily cap reached (2/2")
    assert "2 pending" in ar.state["phase"]
    with open(tmp_path / "state.json") as fp:                  # survives a loop restart
        assert json.load(fp)["pending"] == ar.state["pending"]


def _empty_obs(rec, now):
    """calibrate_one's answer when NWIS answered 200 with an empty series."""
    return {"gauge": rec["gauge"], "kind": "low_nse", "event": rec["event"],
            "events": rec["events"], "ran": False, "at": now.strftime(TS),
            "reason": "obs cover 0% of the window (< 50%)"}


def test_main_skips_are_local_with_a_summary_and_a_run_guard(monkeypatch, tmp_path):
    idx, mf = episodes({"A": ("09000004", 9.0), "B": ("09000003", 5.0),
                        "C": ("09000002", 3.0), "D": ("09000001", 2.0)})
    es = _ES(index=idx, manifests=mf)
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, _empty_obs, 3)
    # three skips in a row: the loop pauses until the next poll
    assert picked == ["09000004", "09000003", "09000002"] and sleeps == [30, 30, 900.0]
    assert ar.state["phase"].startswith("paused — 3 calibrations in a row did not run "
                                        "(last: obs cover 0%")
    # no Hub record per skip, nothing handled: one summary record at most
    assert [r["kind"] for r in es.rows] == [ar.SKIPS_KIND] and es.rows[0]["gauge"] is None
    assert es.rows[0]["skipped"][0]["gauge"] == "09000004"
    assert sorted(ar.state["skipped"]) == ["09000002", "09000003", "09000004"]
    assert ar.state["skipped"]["09000004"]["events"] == ["A"] and ar.state["skips"] == 3
    # next poll: the one gauge left, then nothing until the skips' TTL
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, _empty_obs, 2)
    assert picked == ["09000001"] and sleeps == [30, 900.0]
    assert ar.state["phase"].startswith("idle — no eligible episode with NSE < 0.1; 4 gauge(s) "
                                        "wait out a skip")
    assert len(es.rows) == 1 and len(ar.state["skipped"]) == 4
    with open(tmp_path / "state.json") as fp:                  # survives a loop restart
        assert sorted(json.load(fp)["skipped"]) == sorted(ar.state["skipped"])
    # a fresh container (no local list): the Hub summary still holds 09000004
    ar.state.update({"skipped": {}, "skips_summarized": None})
    (tmp_path / "state.json").unlink()
    rows = list(es.rows)
    es = _ES(rows=rows, index=idx, manifests=mf)
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, _ran, 1)
    assert picked == ["09000003"]                               # 09000004 not re-probed


def test_main_ran_resets_the_skip_run(monkeypatch, tmp_path):
    idx, mf = episodes({"A": ("09000004", 9.0), "B": ("09000003", 5.0),
                        "C": ("09000002", 3.0), "D": ("09000001", 2.0)})
    es = _ES(index=idx, manifests=mf)
    out = iter([_empty_obs, _empty_obs, _ran, _empty_obs])
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, lambda r, n: next(out)(r, n), 4)
    assert len(picked) == 4 and sleeps == [30, 30, 30, 30]       # never paused
    assert [r["kind"] for r in es.rows] == [ar.SKIPS_KIND, "low_nse"]
    assert es.rows[1]["ran"] is True and es.rows[1]["gauge"] == "09000002"


def test_main_outage_records_nothing(monkeypatch, tmp_path):
    es = _low_nse_es()

    def down(rec, now):
        raise ar.Outage("obs fetch for x failed (timeout)")
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, down, 2)
    # nothing recorded, back off a poll each time; the episode is retried
    assert picked == ["09000004", "09000004"] and sleeps == [900.0, 900.0]
    assert es.rows == [] and es.tried == []
    assert ar.state["phase"].startswith("outage — obs fetch for x failed")
    assert "(2 in a row on 09000004; a skip at 4)" in ar.state["phase"]


def test_main_outage_strikes_skip_the_gauge(monkeypatch, tmp_path):
    """A fault only one gauge triggers that the outage rules misread must not
    pin the head of the queue: OUTAGE_STRIKES in a row make it a skip."""
    monkeypatch.setattr(ar, "OUTAGE_STRIKES", 3)
    es = _low_nse_es()

    def cal(rec, now):
        if rec["gauge"] == "09000004":
            raise ar.Outage("obs fetch for 09000004 failed (503), cover 0%")
        return _ran(rec, now)
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, cal, 4)
    # strikes 1 and 2 back off a poll each; the 3rd is a skip and the next
    # gauge gets its turn
    assert picked == ["09000004"] * 3 + ["09000003"] and sleeps == [900.0, 900.0, 30, 30]
    skip = ar.state["skipped"]["09000004"]
    assert skip["reason"].startswith("outage 3x in a row: obs fetch for 09000004 failed")
    assert skip["events"] == ["A"] and ar.skip_ttl(skip["reason"]) == dt.timedelta(seconds=900)
    assert ar.state["outage"] == {} and ar.state["skips"] == 1
    assert [(r["kind"], r["gauge"]) for r in es.rows] == [(ar.SKIPS_KIND, None),
                                                          ("low_nse", "09000003")]


def test_main_calibration_error_skips_the_gauge(monkeypatch, tmp_path):
    """An exception of the gauge's own calibration is a skip (error TTL) that
    counts toward the skip run — not an error shown and the same top-ranked
    gauge picked again every poll."""
    def cal(rec, now):
        if rec["gauge"] == "09000004":
            raise ValueError("deterministic failure in the baseline run")
        return _ran(rec, now)
    es = _low_nse_es()
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, cal, 2)
    assert picked == ["09000004", "09000003"] and sleeps == [30, 30]
    assert ar.state["skipped"]["09000004"]["reason"] == (
        "calibration error: ValueError: deterministic failure in the baseline run")
    assert es.rows[-1]["gauge"] == "09000003" and es.rows[-1]["ran"] is True
    # every gauge failing: the skip run pauses the loop
    ar.state.update({"skipped": {}, "skips_summarized": None})
    (tmp_path / "state.json").unlink()

    def broken(rec, now):
        raise KeyError("wm")
    sleeps, picked = _run_main(monkeypatch, tmp_path, _low_nse_es(), broken, 3)
    assert picked == ["09000004", "09000003", "09000002"] and sleeps == [30, 30, 900.0]
    assert ar.state["phase"].startswith("paused — 3 calibrations in a row did not run "
                                        "(last: calibration error: KeyError: 'wm')")


def test_main_keeps_an_unpushed_winner_until_it_is_on_crest_state(monkeypatch, tmp_path):
    """A winner saved on this Space whose push to CREST_state failed is
    unfinished: nothing recorded, the push retried every loop, the gauge
    counted as calibrated meanwhile; recorded (finished at the push) once
    the parameters are there."""
    pushes, up = [], {"ok": False}

    def push_remote(g, m):
        pushes.append((g, m))
        return up["ok"]
    _set_module(monkeypatch, "paramstore", types.SimpleNamespace(
        get=lambda g, m: None, push_remote=push_remote))

    def cal(rec, now):
        r = _ran(rec, now)
        if rec["gauge"] == "09000004":
            return {**r, "saved": True, "pushed": False, "model": "crest"}
        return {**r, "saved": False, "pushed": None, "finished": now.strftime(TS)}
    es = _low_nse_es()
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, cal, 2)
    # poll 2: the push fails again; 09000004 is handled and in its cooldown
    assert picked == ["09000004", "09000003"] and sleeps == [30, 30]
    assert [r["gauge"] for r in es.rows] == ["09000003"]          # nothing recorded for it
    assert [u["gauge"] for u in ar.state["unpushed"]] == ["09000004"]
    assert pushes == [("09000004", "crest")]
    with open(tmp_path / "state.json") as fp:                  # survives a loop restart
        assert [u["gauge"] for u in json.load(fp)["unpushed"]] == ["09000004"]
    # CREST_state answers again: pushed, finished, recorded; both count for the cap
    up["ok"] = True
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, cal, 1)
    assert picked == [] and ar.state["unpushed"] == [] and ar.state["pending"] == []
    (r4,) = [r for r in es.rows if r["gauge"] == "09000004"]
    assert r4["pushed"] is True and r4["push_tries"] == 3 and r4["ran"] is True
    assert ar._parse(r4["finished"]) >= ar._parse(r4["at"])
    assert ar.state["phase"].startswith("idle — daily cap reached (2/2")


def test_push_unpushed_never_raises(monkeypatch):
    monkeypatch.setattr(ar, "log", lambda s: None)

    def broken(g, m):
        raise OSError("params dir unwritable")
    _set_module(monkeypatch, "paramstore", types.SimpleNamespace(get=broken,
                                                                 push_remote=lambda g, m: True))
    u = {"gauge": "09000004", "model": "crest", "ran": True, "saved": True, "pushed": False,
         "at": _t(1)}
    ar.state["unpushed"] = [u, {**u, "gauge": "09000003", "model": None}]
    ar._push_unpushed()                                         # kept, retried next loop
    assert [(x["gauge"], x["push_tries"]) for x in ar.state["unpushed"]] == [
        ("09000004", 2), ("09000003", 2)] and ar.state["pending"] == []


def test_outage_strikes_are_counted_per_gauge(monkeypatch):
    monkeypatch.setattr(ar, "OUTAGE_STRIKES", 3)
    a, b = {**REC, "gauge": "09000004"}, {**REC, "gauge": "09000003", "event": "B"}
    o = ar.Outage("down")
    assert ar._outage(a, NOW, o) is None and ar._outage(b, NOW, o) is None
    assert ar._outage(a, NOW, o) is None             # b's outage between resets nothing
    assert ar.state["outage"]["09000004"]["n"] == 2 and ar.state["outage"]["09000003"]["n"] == 1
    r = ar._outage(a, NOW, o)
    assert r["ran"] is False and r["reason"] == "outage 3x in a row: down"
    assert "09000004" not in ar.state["outage"]
    assert ar.skip_ttl(r["reason"]) == dt.timedelta(seconds=900)
    # no further outage on the gauge for a day: it starts over
    assert ar._outage(b, NOW + dt.timedelta(hours=25), o) is None
    assert ar.state["outage"]["09000003"]["n"] == 1


def test_main_global_outage_keeps_the_order(monkeypatch, tmp_path):
    """NWIS down for every gauge: the strikes move from gauge to gauge and a
    strike skip lapses within a poll, so once NWIS is back the western
    worst-NSE gauge is calibrated first — a day-long strike skip left it
    behind the others after every global outage."""
    monkeypatch.setattr(ar, "OUTAGE_STRIKES", 3)
    clock = {"s": 0.0}

    def cal(rec, now):
        if clock["s"] < 3000:
            raise ar.Outage(f"obs fetch for {rec['gauge']} failed (503 Server Error), cover 0%")
        return _ran(rec, now)
    es = _low_nse_es()
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, cal, 7, clock=clock)
    assert picked == ["09000004"] * 3 + ["09000003", "09000004", "09000004", "09000003"]
    assert sleeps == [900.0, 900.0, 30, 900.0, 900.0, 30, 30]
    assert [r["gauge"] for r in es.rows if r.get("ran")] == ["09000004", "09000003"]
    assert ar.state["outage"] == {}


def test_main_two_faulty_gauges_do_not_pin_the_queue(monkeypatch, tmp_path):
    """Two gauges whose own faults look like outages alternate at the head:
    their strikes are counted per gauge, their short skips overlap and the
    third gauge gets its turn."""
    monkeypatch.setattr(ar, "OUTAGE_STRIKES", 2)
    clock = {"s": 0.0}

    def cal(rec, now):
        if rec["gauge"] in ("09000004", "09000003"):
            raise ar.Outage(f"ConnectionError: {rec['gauge']}'s forcing tile unreachable")
        return _ran(rec, now)
    es = _low_nse_es()
    sleeps, picked = _run_main(monkeypatch, tmp_path, es, cal, 7, clock=clock)
    assert picked == ["09000004", "09000004", "09000003", "09000004", "09000004", "09000003",
                      "09000002"]
    assert [r["gauge"] for r in es.rows if r.get("ran")] == ["09000002"]


def test_ledger_add():
    led = {}
    es_mod._ledger_add(led, {"gauge": "09000004", "ran": False, "at": _t(5)})
    es_mod._ledger_add(led, {"gauge": None, "kind": "backlog_skips", "ran": False})
    assert led == {}
    es_mod._ledger_add(led, {"gauge": "09000004", "ran": True, "at": _t(5)})
    es_mod._ledger_add(led, {"gauge": "09000004", "ran": True, "at": _t(50)})   # out of order
    assert led == {"09000004": {"first": _t(50), "last": _t(5), "n": 2}}
    # the latest finish is kept when records carry one
    es_mod._ledger_add(led, {"gauge": "09000004", "ran": True, "at": _t(3), "finished": _t(2)})
    es_mod._ledger_add(led, {"gauge": "09000004", "ran": True, "at": _t(60),
                             "finished": _t(59)})
    assert led == {"09000004": {"first": _t(60), "last": _t(3), "n": 4, "finished": _t(2)}}


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
    rec = {"gauge": "09000004", "kind": "low_nse", "event": "E", "ran": True, "at": _t(1),
           "nse": -0.4, "n_pairs": 20, "baseline_nse": 0.1, "best_nse": 0.4}
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


# --------------------------------------------------------------------------- #
# the retired western backlog: builder + committed files stay in the repo
# (the builder also writes the reservoir table the hold reads); autocal
# itself no longer reads the backlog
# --------------------------------------------------------------------------- #
def _builder():
    p = os.path.join(ROOT, "scripts", "build_autocal_backlog.py")
    spec = importlib.util.spec_from_file_location("build_autocal_backlog", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _committed_backlog() -> list:
    with open(BACKLOG_JSON, encoding="utf-8") as fp:
        return list(json.load(fp).get("gauges") or [])


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
    data = _committed_backlog()
    assert data, "hf_data/data/autocal_west_backlog.json missing or empty"
    keys = [(e["tier"], e["area_km2"], e["gauge"]) for e in data]
    assert keys == sorted(keys)
    assert {e["tier"] for e in data} <= {1, 2, 3, 4}
    assert all(set(e) >= {"gauge", "tier", "area_km2", "stor_mm", "donor", "reasons"}
               for e in data)
    assert len({e["gauge"] for e in data}) == len(data)


def test_committed_backlog_tiers_match_reservoir_table(monkeypatch):
    monkeypatch.setattr(ar, "_resv", None)
    for e in _committed_backlog():
        assert (_reservoir_stor(e["gauge"]) is not None) == (e["tier"] == 4), e
    monkeypatch.setattr(ar, "_resv", None)

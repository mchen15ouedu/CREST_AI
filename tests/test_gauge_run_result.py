"""A gauge run streams progress and ends with ONE typed result (ticket 07).

Driven through pipeline.run_gauge with the mock EF5; the remote data layer
(COG clips, multipliers, snow, control file) is stubbed so it runs offline."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime

import pytest

T0 = datetime(2025, 7, 3, 0, 0)
T1 = datetime(2025, 7, 3, 3, 0)               # 4 hourly steps, inclusive


def _drain(gen):
    """(events, final payload) — asserts the run ends with exactly one done."""
    evs = list(gen)
    dones = [p for k, p in evs if k == "done"]
    assert len(dones) == 1 and evs[-1][0] == "done"
    return evs, dones[0]


def test_ok_run_ends_with_typed_result(offline):
    from hf_data.runresult import GaugeRunResult, Outcome
    _, res = _drain(offline.run_gauge("05467000", T0, T1, use_mock=True,
                                      grids=False, warmup_days=0,
                                      workdir=None))
    assert isinstance(res, GaugeRunResult)
    assert res.outcome is Outcome.OK and res.ok
    assert res.gauge_id == "05467000"
    assert [r["time"] for r in res.rows] == ["2025-07-03 00:00", "2025-07-03 01:00",
                                            "2025-07-03 02:00", "2025-07-03 03:00"]


def test_ok_result_locates_record_and_states_with_provenance(offline):
    _, res = _drain(offline.run_gauge("05467000", T0, T1, use_mock=True,
                                      grids=False, warmup_days=0))
    assert res.cache_model == "crestphys" and res.state_model == "crestphys"
    assert res.record_key == "05467000_crestphys"
    assert res.state_key == "05467000_crestphys"
    assert os.path.isdir(res.state_dir)
    assert res.provenance == {
        "extent": {"extent": "basin", "source": "gages2", "table": "test-1",
                   "snap_rule": "nearest",
                   "bounds": [-90.975, 41.024, -90.157, 41.263]},
        "cut_ids": [], "lakes": {"on": False}}
    with open(res.record_path) as fh:                  # the pipeline stamped it
        rec = json.load(fh)
    assert len(rec["rows"]) == 4
    assert rec["extent"] == res.provenance["extent"]
    assert rec["cut_ids"] == [] and rec["lakes"] == {"on": False}


def test_repeat_run_is_served_from_cache(offline):
    from hf_data.runresult import Outcome
    _, first = _drain(offline.run_gauge("05467000", T0, T1, use_mock=True,
                                        grids=False, warmup_days=0))
    evs, again = _drain(offline.run_gauge("05467000", T0, T1, use_mock=True,
                                          grids=False, warmup_days=0))
    assert again.outcome is Outcome.CACHED and again.ok
    assert again.rows == first.rows
    assert again.record_path == first.record_path
    assert again.provenance == first.provenance


@pytest.mark.parametrize("gid,reason", [("15019990", "no_terrain"),
                                        ("01234567", "no_row")])
def test_outlet_without_basin_is_not_simulated(offline, gid, reason):
    from hf_data.runresult import Outcome
    _, res = _drain(offline.run_gauge(gid, T0, T1, use_mock=True, grids=False))
    assert res.outcome is Outcome.NO_BASIN and not res.ok
    assert res.reason == reason and res.rows == []
    assert res.record_path is None


def test_stopped_run_ends_cancelled(offline):
    from hf_data.runresult import Outcome
    stop = threading.Event()
    stop.set()
    evs, res = _drain(offline.run_gauge("05467000", T0, T1, use_mock=True,
                                        grids=False, warmup_days=0, cancel=stop))
    assert res.outcome is Outcome.CANCELLED and not res.ok


def test_run_stopped_mid_stream_ends_cancelled(offline):
    from hf_data.runresult import Outcome
    stop = threading.Event()
    long_end = datetime(2025, 7, 4, 0, 0)            # 25 mock steps (~4 s)
    for kind, payload in offline.run_gauge("05467000", T0, long_end, use_mock=True,
                                           grids=False, warmup_days=0, cancel=stop):
        if kind == "hydro":
            stop.set()                               # Stop once rows arrive
        if kind == "done":
            res = payload
    assert res.outcome is Outcome.CANCELLED
    assert 0 < len(res.rows) < 25


# ---- the chat demo's analyze() ends the same way ----------------------------
def _analyze(offline, monkeypatch, gid):
    import types
    monkeypatch.setattr(offline, "parse_query", lambda q, hours=48: types.SimpleNamespace(
        label="test", anchor=(41.2, -90.5), t_start=T0, t_end=T1))
    monkeypatch.setattr(offline, "select_outlet", lambda anchor: {
        "id": gid, "name": "Test Creek", "lat": 41.2, "lon": -90.5, "area": 448.6})
    return _drain(offline.analyze("test", use_mock=True, hours=3))


def test_analyze_without_a_basin_ends_with_a_typed_result(offline, monkeypatch):
    from hf_data.runresult import GaugeRunResult, Outcome
    _, res = _analyze(offline, monkeypatch, "15019990")
    assert isinstance(res, GaugeRunResult)
    assert res.outcome is Outcome.NO_BASIN and res.reason == "no_terrain" and not res.ok


def test_analyze_mock_run_ends_ok_with_its_rows(offline, monkeypatch):
    from hf_data.runresult import GaugeRunResult, Outcome
    evs, res = _analyze(offline, monkeypatch, "05467000")
    streamed = [r for k, p in evs if k == "hydro" for r in p["rows"]]
    assert isinstance(res, GaugeRunResult) and res.outcome is Outcome.OK and res.ok
    assert res.gauge_id == "05467000" and res.rows == streamed and len(res.rows) == 3


# ---- web job runner: the browser's gauge_done is derived from the result ----
def _job(offline, monkeypatch, gauge_ids, cancelled=False):
    from hf_data import llm, simjobs
    monkeypatch.setattr(simjobs, "USE_MOCK", True)
    monkeypatch.setattr(llm, "available", lambda: False)
    job = simjobs.SimJob(gauge_ids, T0, T1, {"warmup_days": 0})
    if cancelled:
        job.cancel.set()
    job._run()                                       # the worker body, inline
    return {e["gauge_id"]: e for e in job.events if e["kind"] == "gauge_done"}


def test_job_reports_outcomes_not_return_codes(offline, monkeypatch):
    done = _job(offline, monkeypatch, ["05467000", "15019990"])
    assert done["05467000"]["outcome"] == "ok" and done["05467000"]["n"] == 4
    assert done["15019990"]["outcome"] == "no_basin"
    assert done["15019990"]["reason"] == "no_terrain"
    assert not any("returncode" in e for e in done.values())
    # the browser reads `ok` from the result, it does not re-derive it
    assert done["05467000"]["ok"] is True and done["15019990"]["ok"] is False


def test_job_reports_cached_and_cancelled(offline, monkeypatch):
    _job(offline, monkeypatch, ["05467000"])
    assert _job(offline, monkeypatch, ["05467000"])["05467000"]["outcome"] == "cached"
    stopped = _job(offline, monkeypatch, ["05467000"], cancelled=True)
    assert stopped["05467000"]["outcome"] == "cancelled"

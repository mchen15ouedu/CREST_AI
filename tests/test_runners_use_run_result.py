"""Fleet, event simulation, routed nowcast and calibration consume the typed
gauge-run result (ticket 08): no return codes, no result-cache globbing, no
after-the-fact provenance patch.

Runs go through pipeline.run_gauge with the mock EF5 on the offline fixture
(tests/conftest.py); the Hub is a recorder."""
from __future__ import annotations

import functools
import json
from datetime import datetime

import pytest

T0 = datetime(2025, 7, 3, 0, 0)
T1 = datetime(2025, 7, 3, 3, 0)               # 4 hourly steps, one fleet chunk


@pytest.fixture
def mock_runs(offline, monkeypatch):
    """Every runner's run_gauge(use_mock=False) runs the mock EF5 instead."""
    real = offline.run_gauge

    @functools.wraps(real)
    def run_gauge(*a, **k):
        k["use_mock"] = True
        k.setdefault("warmup_days", 0)
        return real(*a, **k)
    monkeypatch.setattr(offline, "run_gauge", run_gauge)
    return offline


@pytest.fixture
def hub(monkeypatch):
    """Records what fleet_run commits to the fleet repo (file contents read at
    commit time: the local files are deleted afterwards)."""
    from hf_data import fleetstore
    from hf_data.hubdata import Dataset, MemoryHub, Put
    mem = MemoryHub()
    monkeypatch.setattr(fleetstore, "DATASET", Dataset("me/fleet", adapter=mem))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    commits = []
    real = mem.commit

    def commit(repo, ops, message):
        real(repo, ops, message)
        commits.append({"repo": repo, "message": message,
                        "files": {op.path: mem.files[op.path]
                                  for op in ops if isinstance(op, Put)}})
    mem.commit = commit
    return commits


@pytest.fixture
def packed(monkeypatch):
    """The mock EF5 saves no state grids: the bundle step records which state
    folder it was asked to pack and writes a stand-in bundle."""
    from hf_data import statebundle
    seen = []

    def pack_dir(state_dir, out_path=None):
        seen.append(state_dir)
        with open(out_path, "wb") as fh:
            fh.write(b"BUNDLE")
        return out_path
    monkeypatch.setattr(statebundle, "pack_dir", pack_dir)
    return seen


# ---- fleet runner -----------------------------------------------------------
def test_fleet_publishes_the_record_and_states_the_run_reported(mock_runs, hub, packed):
    import fleet_run
    from hf_data import statecache
    r = fleet_run._run_one_gauge("05467000", T0, T1, chunk_days=10)

    assert r["ok"] is True and r["key"] == "05467000_crestphys"
    assert r["rows"] == 4
    assert packed == [statecache.state_dir("05467000", "crestphys")]
    assert len(hub) == 1
    files = hub[0]["files"]
    assert set(files) == {"states/05467000_crestphys.pqf",
                          "results/05467000_crestphys.json"}
    assert files["states/05467000_crestphys.pqf"] == b"BUNDLE"
    rec = json.loads(files["results/05467000_crestphys.json"])
    assert [row["time"] for row in rec["rows"]] == [
        "2025-07-03 00:00", "2025-07-03 01:00", "2025-07-03 02:00", "2025-07-03 03:00"]
    # provenance stamped by the pipeline itself
    assert rec["extent"]["table"] == "test-1" and rec["extent"]["extent"] == "basin"
    assert rec["cut_ids"] == [] and rec["lakes"] == {"on": False}


def test_fleet_ignores_stale_records_of_other_variants(mock_runs, hub, packed):
    """A leftover record of another key is never what gets published (the old
    mtime glob would pick whichever record was written last)."""
    import os
    import fleet_run
    from hf_data import statecache
    stale = statecache.results_path("05467000", "crestphys-spd")
    os.makedirs(os.path.dirname(stale), exist_ok=True)
    r = fleet_run._run_one_gauge("05467000", T0, T1, chunk_days=10)
    with open(stale, "w") as fh:                       # newer than the real record
        json.dump({"rows": []}, fh)
    assert r["ok"]
    assert set(hub[0]["files"]) == {"states/05467000_crestphys.pqf",
                                    "results/05467000_crestphys.json"}


def test_fleet_reports_the_outcome_of_an_unsimulated_outlet(mock_runs, hub, packed):
    import fleet_run
    r = fleet_run._run_one_gauge("15019990", T0, T1, chunk_days=10)
    assert r["ok"] is False and r["outcome"] == "no_basin"
    assert "rc" not in r
    assert hub == [] and packed == []


def test_fleet_reports_a_failed_chunk_by_outcome(mock_runs, hub, packed, monkeypatch):
    import fleet_run
    from hf_data.runresult import GaugeRunResult, Outcome

    def failing(gid, *a, **k):
        yield ("status", "EF5 crashed")
        yield ("done", GaugeRunResult(Outcome.FAILED, gid, reason="ef5 rc=139"))
    monkeypatch.setattr(mock_runs, "run_gauge", failing)
    r = fleet_run._run_one_gauge("05467000", T0, T1, chunk_days=10)
    assert r["ok"] is False and r["outcome"] == "failed"
    assert r["at"] == "2025-07-03" and "ef5 rc=139" in r["last"]
    assert hub == []


# ---- the legacy done-dict view is gone ---------------------------------------
def test_run_result_has_no_legacy_dict_view():
    from hf_data.runresult import GaugeRunResult, Outcome
    assert not hasattr(GaugeRunResult(Outcome.OK, "05467000"), "get")


# ---- event simulation: exceptions come from outcome kinds --------------------
def test_event_run_without_a_basin_is_a_domain_error():
    from hf_data import eventsim, hucdomain
    from hf_data.runresult import GaugeRunResult, Outcome
    err = eventsim.run_error("05467000", GaugeRunResult(
        Outcome.NO_BASIN, "05467000", reason="no_terrain"))
    assert isinstance(err, hucdomain.DomainError) and "no_terrain" in str(err)


@pytest.mark.parametrize("outcome", ["failed", "no_state", "cut_failed", "cancelled"])
def test_event_run_that_did_not_simulate_is_a_runtime_error(outcome):
    from hf_data import eventsim, hucdomain
    from hf_data.runresult import GaugeRunResult, Outcome
    err = eventsim.run_error("05467000", GaugeRunResult(
        Outcome(outcome), "05467000", reason="why"))
    assert type(err) is RuntimeError
    assert outcome in str(err) and "why" in str(err)
    assert not isinstance(err, hucdomain.DomainError)


@pytest.mark.parametrize("outcome", ["ok", "cached"])
def test_event_run_that_simulated_is_no_error(outcome):
    from hf_data import eventsim
    from hf_data.runresult import GaugeRunResult, Outcome
    assert eventsim.run_error("05467000", GaugeRunResult(Outcome(outcome), "05467000")) is None


def test_event_run_that_never_finished_is_a_runtime_error():
    from hf_data import eventsim
    assert type(eventsim.run_error("05467000", None)) is RuntimeError


# ---- routed nowcast: keys and cut set from the Phase-A result ----------------
def _scripted(results, calls):
    """A run_gauge that ends each call with the next scripted result."""
    def run_gauge(gid, t_start, t_end, **k):
        calls.append(k)
        res = results.pop(0)
        for row in res.rows:
            yield ("hydro", {"rows": [row]})
        yield ("done", res)
    return run_gauge


@pytest.fixture
def point(offline, monkeypatch):
    from hf_data import fleetstore, virtualpoints
    monkeypatch.setattr(virtualpoints, "info", lambda gid: {
        "id": gid, "name": "Ungauged", "lat": 42.0, "lon": -122.0,
        "area": 15421.2, "virtual": True, "tz": "UTC"})
    monkeypatch.setattr(fleetstore, "ensure_local", lambda *a, **k: None)
    return offline


def test_routed_nowcast_forecasts_on_phase_a_state_and_cut_set(point, monkeypatch):
    from hf_data import routednow
    from hf_data.runresult import GaugeRunResult, Outcome
    gid = "V70520944"
    hist = GaugeRunResult(Outcome.OK, gid, rows=[{"time": "2025-07-03 00:00", "sim_q": 1.0}],
                          cache_model="crest-spd-p1", state_model="crest-spd",
                          provenance={"extent": {}, "cut_ids": ["11111111"],
                                      "lakes": {"on": False}})
    fcst = GaugeRunResult(Outcome.OK, gid, rows=[{"time": "2025-07-03 01:00", "sim_q": 2.5}],
                          cache_model="crest-spd-p1", state_model="crest-spd")
    calls = []
    monkeypatch.setattr(point, "run_gauge", _scripted([hist, fcst], calls))
    r = routednow.compute(gid, T0)
    assert r["ok"] and r["q"] == [2.5]
    assert r["cache_model"] == "crest-spd" and r["injected"] is True
    assert calls[1]["cut_ids"] == ["11111111"]


def test_routed_nowcast_skips_a_point_without_a_basin(point, monkeypatch):
    from hf_data import routednow
    from hf_data.runresult import GaugeRunResult, Outcome
    gid = "V70520944"
    calls = []
    monkeypatch.setattr(point, "run_gauge", _scripted(
        [GaugeRunResult(Outcome.NO_BASIN, gid, reason="no_terrain")], calls))
    r = routednow.compute(gid, T0)
    assert r["ok"] is False and r["no_basin"] is True and "no_terrain" in r["reason"]
    assert len(calls) == 1


# ---- calibration: a candidate run that did not simulate scores nothing -------
def test_calibration_scores_only_runs_that_simulated(mock_runs, monkeypatch):
    from hf_data import calibrate, llm, paramstore
    from hf_data.runresult import GaugeRunResult, Outcome
    monkeypatch.setattr(llm, "available", lambda: False)
    monkeypatch.setattr(paramstore, "maybe_save", lambda *a, **k: False)
    real = mock_runs.run_gauge
    n = []

    def first_ok_then_fail(gid, *a, **k):
        n.append(1)
        if len(n) == 1:                               # the baseline
            yield from real(gid, *a, **k)
            return
        yield ("hydro", {"rows": [{"time": "2025-07-03 00:00", "sim_q": 1.0,
                                   "obs_q": 1.0}]})
        yield ("done", GaugeRunResult(Outcome.FAILED, gid, reason="ef5 rc=139"))
    monkeypatch.setattr(mock_runs, "run_gauge", first_ok_then_fail)
    evs = list(calibrate._run_calibration("05467000", T0, T1, use_mock=True,
                                          rounds=1, k=2, ext_rounds=0))
    rounds = [p for kind, p in evs if kind == "round"]
    assert rounds[0]["tried"][0]["nse"] is not None              # baseline
    assert [t["nse"] for t in rounds[1]["tried"]] == [None, None]
    assert any("failed" in str(p) for kind, p in evs if kind == "status")


def test_calibration_stops_when_the_baseline_does_not_simulate(mock_runs, monkeypatch):
    from hf_data import calibrate, llm
    from hf_data.runresult import GaugeRunResult, Outcome
    monkeypatch.setattr(llm, "available", lambda: False)

    def failing(gid, *a, **k):
        yield ("done", GaugeRunResult(Outcome.FAILED, gid, reason="ef5 rc=139"))
    monkeypatch.setattr(mock_runs, "run_gauge", failing)
    evs = list(calibrate._run_calibration("05467000", T0, T1, use_mock=True,
                                          rounds=1, k=2, ext_rounds=0))
    kind, done = evs[-1]
    assert kind == "done" and "ef5 rc=139" in done["error"]
    assert not [p for k_, p in evs if k_ == "round"]

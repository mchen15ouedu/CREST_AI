"""hf_data.fleetstore owns the fleet repo layout on both sides: a run result
published through it is listed as done and fetched back by ensure_local.
Offline, over the in-memory dataset adapter."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from conftest import RES, write_grid
from hf_data import fleetstore as fs
from hf_data import statecache
from hf_data.hubdata import Dataset, MemoryHub
from hf_data.runresult import GaugeRunResult, Outcome

ROWS = [{"time": "2021-07-01 00:00", "sim_q": 1.5},
        {"time": "2021-07-01 01:00", "sim_q": 2.5}]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    h = MemoryHub()
    monkeypatch.setattr(fs, "DATASET", Dataset("me/fleet", adapter=h))
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path / "consumer"))
    monkeypatch.setattr(fs, "_seen", {})
    monkeypatch.setattr(fs, "_remote", {})
    monkeypatch.setenv("HF_TOKEN", "test")
    return h


def _run(tmp_path, gid, cache_model="crestphys-spd-p1", state_model="crestphys-spd"):
    """What a finished fleet run leaves behind: a row record + a state dir."""
    prod = tmp_path / "producer" / gid
    sdir = prod / "states"
    sdir.mkdir(parents=True)
    write_grid(str(sdir / "SM_20210701_0000.tif"), np.full((3, 3), 7.0), -97.0, 35.0)
    rec = prod / "record.json"
    rec.write_text(json.dumps({"gauge": gid, "model": cache_model, "rows": ROWS,
                               "state_times": ["2021-07-01 00:00"], "variant": "v1"}))
    return GaugeRunResult(Outcome.OK, gid, rows=ROWS, cache_model=cache_model,
                          state_model=state_model, record_path=str(rec),
                          state_dir=str(sdir))


def test_published_gauge_run_is_done_and_fetchable(hub, tmp_path):
    res = _run(tmp_path, "01234567")
    pub = fs.publish_run(res)
    assert pub.ok and pub.key == "01234567_crestphys-spd-p1" and pub.bytes > 0
    assert sorted(hub.files) == ["results/01234567_crestphys-spd-p1.json",
                                 "states/01234567_crestphys-spd.pqf"]
    assert len(hub.commits) == 1                      # record + bundle land together

    done = fs.done_ids()
    assert done.ok and done.value == {"01234567"}

    got = fs.ensure_local("1234567", "crestphys-spd", rows_model="crestphys-spd-p1",
                          variant="v1")
    assert got == "fetched"
    rec = statecache.load_record("1234567", "crestphys-spd-p1")
    assert [r["sim_q"] for r in rec["rows"]] == [1.5, 2.5]
    sdir = statecache.state_dir("1234567", "crestphys-spd")
    assert os.path.exists(os.path.join(sdir, "SM_20210701_0000.tif"))


def test_pour_point_hindcast_is_archived_under_vp_and_not_served(hub, tmp_path):
    """A pour point's multi-year hindcast must never overwrite (or be served
    in place of) the ungauged nowcast's rolling checkpoint of the same key."""
    res = _run(tmp_path, "V70520944", cache_model="crestphys-spd",
               state_model="crestphys-spd")
    assert fs.publish_run(res).ok
    assert sorted(hub.files) == ["vp/results/V70520944_crestphys-spd.json",
                                 "vp/states/V70520944_crestphys-spd.pqf"]
    assert fs.done_ids().value == {"V70520944"}
    assert fs.ensure_local("V70520944", "crestphys-spd") is None
    assert fs.has_remote("V70520944", "crestphys-spd") is False


def test_checkpoints_publish_in_one_commit_and_are_served_not_done(hub, tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path / "space"))
    pubs = []
    for vp in ("V70520944", "V70520945"):
        write_grid(os.path.join(statecache.state_dir(vp, "crest-spd"),
                                "SM_20210701_0000.tif"), np.full((3, 3), 1.0), -97.0, 35.0)
        with open(statecache.results_path(vp, "crest-spd"), "w") as fh:
            json.dump({"rows": ROWS, "state_times": [], "variant": None}, fh)
        pubs.append(fs.pack(vp, "crest-spd"))         # in a worker: plain bytes
    assert all(isinstance(p.bundle, bytes) for p in pubs)

    batch = fs.batch("ungauged checkpoints")
    for p in pubs:
        batch.add(p)
    assert batch.pending == 2 and batch.nbytes > 0
    assert batch.flush() == ["V70520944_crest-spd", "V70520945_crest-spd"]
    assert len(hub.commits) == 1 and batch.pending == 0
    assert hub.commits[0]["message"].startswith("ungauged checkpoints x2")

    assert fs.checkpoint_keys().value == {"V70520944_crest-spd", "V70520945_crest-spd"}
    assert fs.done_ids().value == set()              # a checkpoint is not a hindcast
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path / "consumer"))
    assert fs.ensure_local("V70520944", "crest-spd") == "fetched"


def _ungauged_runner(tmp_path, monkeypatch):
    import importlib.util
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "ungauged_runner_t", os.path.join(root, "ungauged_space", "runner.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "LOG", str(tmp_path / "ungauged.log"))
    return runner


def test_ungauged_runner_publishes_checkpoints_through_the_store(hub, tmp_path,
                                                                  monkeypatch):
    runner = _ungauged_runner(tmp_path, monkeypatch)
    hub.files["states/V70520900_crest-spd.pqf"] = b"old"
    runner._seed_uploaded_keys()
    assert runner._uploaded_keys == {"V70520900_crest-spd"}

    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path / "space"))
    write_grid(os.path.join(statecache.state_dir("V70520944", "crest-spd"),
                            "SM_20210701_0000.tif"), np.full((3, 3), 1.0), -97.0, 35.0)
    pub = runner._pack_fleet("V70520944", "crest-spd")   # in a spawned worker
    runner._queue_fleet(pub)                             # back in the parent
    assert runner._flush_fleet() == 1
    assert "states/V70520944_crest-spd.pqf" in hub.files
    assert "V70520944_crest-spd" in runner._uploaded_keys
    assert runner._flush_fleet() == 0 and len(hub.commits) == 1


def test_failed_batch_stays_queued_until_the_hub_is_back(hub, tmp_path):
    pub = fs.Publication("V70520944", "m", "m", b"bundle", b"{}")
    batch = fs.batch("ungauged checkpoints", max_pending=1)
    batch.add(pub)
    batch.add(fs.Publication("V70520945", "m", "m", b"bundle"))
    hub.outage = True
    assert batch.flush() == []
    assert batch.pending == 1                         # bounded while the Hub is down
    hub.outage = False
    assert batch.flush() == ["V70520944_m"]
    assert hub.files["states/V70520944_m.pqf"] == b"bundle"

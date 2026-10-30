"""Shared-store writes: one temp file per CALL (threads + processes write the
same forcing/param/obs stores), the obs store's cross-process lock, and the
ungauged runner's spawn pool (a forked worker inherits hf-xet mid-state)."""
from __future__ import annotations

import importlib.util
import os
import threading
from datetime import datetime

import numpy as np
import pytest

from conftest import ROOT, write_grid


def _tmp_left(d) -> list:
    return [p for p in os.listdir(d) if p.endswith(".tmp")]


def _threads(fn, n=8) -> list:
    errs = []

    def run(i):
        try:
            fn(i)
        except Exception as e:              # surfaced by the assert below
            errs.append(e)
    ts = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return errs


@pytest.fixture
def serial_replace(monkeypatch):
    """Windows refuses two os.replace()s onto one target in the same instant
    (Linux renames atomically); the tests are about the temp names."""
    replace, one = os.replace, threading.Lock()

    def rep(a, b):
        with one:
            replace(a, b)
    monkeypatch.setattr(os, "replace", rep)


# --- atomicfile ---------------------------------------------------------------
def test_atomic_path_unique_per_call(tmp_path):
    from hf_data.atomicfile import atomic_path
    target = tmp_path / "x.pqf"
    with atomic_path(str(target)) as a, atomic_path(str(target)) as b:
        assert a != b and os.path.dirname(a) == str(tmp_path)
        (tmp_path / os.path.basename(a)).write_text("a")
        (tmp_path / os.path.basename(b)).write_text("b")
    # b exits first, a last: the last writer wins, nothing left behind
    assert target.read_text() == "a" and _tmp_left(tmp_path) == []


def test_atomic_path_failure_keeps_old_file(tmp_path):
    from hf_data.atomicfile import atomic_path
    target = tmp_path / "x.pqf"
    target.write_text("old")
    with pytest.raises(RuntimeError):
        with atomic_path(str(target)) as tmp:
            with open(tmp, "w") as fh:
                fh.write("half")
            raise RuntimeError("writer died")
    assert target.read_text() == "old" and _tmp_left(tmp_path) == []


# --- forcing + param clips: threads of one process share the store ------------
def test_write_pqf_threads_same_hour(tmp_path, monkeypatch, serial_replace):
    import pyarrow.parquet as pq
    from hf_data import forcing
    path = str(tmp_path / "p.pqf")
    gate, temps, write = threading.Barrier(8), [], pq.write_table

    def overlapping_write(tbl, where, **kw):    # all 8 temp files exist at once
        write(tbl, where, **kw)
        temps.append(where)
        gate.wait(timeout=20)
    monkeypatch.setattr(pq, "write_table", overlapping_write)

    def w(i):
        forcing._write_pqf(path, np.full((4, 5), float(i), "float32"),
                           -90.0, 40.0, 0.01, -9999.0)
    assert _threads(w) == [] and len(set(temps)) == 8 and _tmp_left(tmp_path) == []
    a, xll, yll, cell, nod = forcing._read_pqf(open(path, "rb").read())
    assert a.shape == (4, 5) and len(set(a.ravel().tolist())) == 1   # one whole write


def test_param_clip_threads_share_store(tmp_path, serial_replace):
    import rasterio
    from hf_data import params
    src = str(tmp_path / "wm_usa.tif")
    write_grid(src, np.arange(100.0).reshape(10, 10), -90.0, 41.0, res=0.01)
    out_dir = tmp_path / "clip"
    out_dir.mkdir()
    out, skipped = {}, {}

    def w(i):
        params._clip_one_param(src, "wm.tif", (-89.97, 40.93, -89.93, 40.97),
                               str(out_dir), "wm", out, skipped)
    assert _threads(w) == [] and not skipped and _tmp_left(out_dir) == []
    with rasterio.open(out["wm"]) as ds:
        assert ds.read(1).shape == (4, 4) and ds.nodata == -9999.0


# --- obs store ----------------------------------------------------------------
@pytest.fixture
def obs(tmp_path, monkeypatch):
    from hf_data import obs, statecache
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path))
    return obs


def test_save_store_round_trip(obs, tmp_path):
    rows = {datetime(2026, 10, 1, h): float(h) for h in range(5)}
    cov = [[datetime(2026, 10, 1, 0), datetime(2026, 10, 1, 4)]]
    obs._save_store("01234567", rows, cov)
    assert obs._load_store("01234567") == (rows, cov)
    assert _tmp_left(tmp_path / "obs") == []


def test_save_store_failure_keeps_old_store(obs, tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    rows = {datetime(2026, 10, 1, 0): 1.0}
    obs._save_store("01234567", rows, [])

    def boom(tbl, where, **kw):
        with open(where, "wb") as fh:
            fh.write(b"PAR1 half")
        raise OSError("disk full")
    with monkeypatch.context() as m:
        m.setattr(pq, "write_table", boom)
        with pytest.raises(OSError):
            obs._save_store("01234567", {datetime(2026, 10, 2, 0): 9.0}, [])
    assert obs._load_store("01234567")[0] == rows
    assert _tmp_left(tmp_path / "obs") == []


def test_get_series_holds_cross_process_lock(obs, monkeypatch):
    from filelock import FileLock, Timeout
    held = []

    def fake_fetch(site, a, b):
        # a second, independent lock handle stands in for another process
        other = FileLock(obs._store_path(site) + ".lock", timeout=0)
        try:
            other.acquire()
            other.release()
            held.append(False)
        except Timeout:
            held.append(True)
        return [(datetime(2020, 1, 1, 1), 2.0)]
    monkeypatch.setattr(obs, "fetch_usgs_discharge", fake_fetch)
    got = obs.get_series("1234567", datetime(2020, 1, 1), datetime(2020, 1, 2))
    assert held == [True] and got == [(datetime(2020, 1, 1, 1), 2.0)]
    after = FileLock(obs._store_path("01234567") + ".lock", timeout=0)
    after.acquire()                                   # released on the way out
    after.release()


def test_persist_scan_skips_tmp_and_lock(tmp_path, monkeypatch):
    from hf_data import persist
    monkeypatch.setattr(persist, "CACHE_DIR", str(tmp_path))
    d = tmp_path / "obs"
    d.mkdir()
    for n in ("01234567.parquet", "01234567.parquet.lock", "01234567.parquet.k3x9.tmp"):
        (d / n).write_bytes(b"x")
    assert list(persist._scan()) == ["obs/01234567.parquet"]


# --- ungauged runner: workers are spawned, never forked -----------------------
def test_ungauged_pass_spawns_workers(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "ungauged_runner", os.path.join(ROOT, "ungauged_space", "runner.py"))
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    seen = {}

    class Pool:
        def __init__(self, max_workers=None, mp_context=None):
            seen["ctx"] = mp_context

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(runner, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(runner, "_all_points", lambda: [])
    monkeypatch.setattr(runner, "LOG", str(tmp_path / "ungauged.log"))
    runner.run_pass(datetime(2026, 10, 30, 12))
    assert seen["ctx"] is not None and seen["ctx"].get_start_method() == "spawn"

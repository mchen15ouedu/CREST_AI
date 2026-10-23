import os
from datetime import datetime

import numpy as np

from conftest import RES, write_grid

WEST, NORTH = -90.0, 41.0


def _nodes():
    m = np.zeros((4, 4), dtype=bool)
    m[1:3, 1:3] = True
    return m


def _tr():
    from rasterio.transform import from_origin
    return from_origin(WEST, NORTH, RES, RES)


def test_coverage(tmp_path):
    from hf_data import runstates
    a = np.ones((6, 6), dtype="float32")
    # same lattice, origin one cell west and one cell north: covers all nodes
    p = str(tmp_path / "full.tif")
    write_grid(p, a, WEST - RES, NORTH + RES)
    assert runstates.coverage(p, _nodes(), _tr()) == 1.0
    # one of the four nodes is nodata in the state (a clipped headwater)
    b = a.copy()
    b[2, 2] = -9999.0
    p2 = str(tmp_path / "part.tif")
    write_grid(p2, b, WEST - RES, NORTH + RES)
    assert runstates.coverage(p2, _nodes(), _tr()) == 0.75
    # a smaller raster (run rows -1..1) that misses the nodes' lower row
    p3 = str(tmp_path / "small.tif")
    write_grid(p3, np.ones((3, 6), dtype="float32"), WEST - RES, NORTH + RES)
    assert runstates.coverage(p3, _nodes(), _tr()) == 0.5
    # half a cell off the lattice, or another resolution: unusable
    p4 = str(tmp_path / "off.tif")
    write_grid(p4, a, WEST - RES / 2, NORTH)
    assert runstates.coverage(p4, _nodes(), _tr()) == 0.0
    p5 = str(tmp_path / "res.tif")
    write_grid(p5, a, WEST, NORTH, res=RES * 2)
    assert runstates.coverage(p5, _nodes(), _tr()) == 0.0


def test_stage_and_commit(tmp_path):
    from hf_data import runstates
    shared, run = tmp_path / "shared", tmp_path / "run"
    shared.mkdir()
    t = datetime(2026, 6, 30)
    for var in ("SM", "Q"):
        write_grid(str(shared / f"{var}_20260630_0000.tif"), np.ones((4, 4)), WEST, NORTH)
        write_grid(str(shared / f"{var}_20260620_0000.tif"), np.ones((4, 4)), WEST, NORTH)
    (shared / "gauge_relationships_20260630_0000.txt").write_text("rel")
    ok, cov = runstates.stage(str(shared), str(run), t, _nodes(), _tr())
    assert ok and cov == 1.0
    assert sorted(os.listdir(run)) == ["Q_20260630_0000.tif", "SM_20260630_0000.tif",
                                       "gauge_relationships_20260630_0000.txt"]
    # EF5 writes a new state; commit moves only that one back
    write_grid(str(run / "SM_20260710_0000.tif"), np.full((4, 4), 2.0), WEST, NORTH)
    assert runstates.commit(str(run), str(shared)) == 1
    assert (shared / "SM_20260710_0000.tif").exists()
    assert not (run / "SM_20260710_0000.tif").exists()


def test_stage_refuses_uncovering_state(tmp_path):
    from hf_data import runstates
    shared, run = tmp_path / "shared", tmp_path / "run"
    shared.mkdir()
    a = np.ones((4, 4), dtype="float32")
    a[1:3, 1:3] = -9999.0                 # the run's nodes are cold in this state
    write_grid(str(shared / "SM_20260630_0000.tif"), a, WEST, NORTH)
    ok, cov = runstates.stage(str(shared), str(run), datetime(2026, 6, 30),
                              _nodes(), _tr())
    assert not ok and cov == 0.0
    assert os.listdir(run) == []
    # no state at that time at all
    ok, cov = runstates.stage(str(shared), str(run), datetime(2026, 1, 1),
                              _nodes(), _tr())
    assert not ok


def test_stage_judges_water_grids_and_partial_cover(tmp_path):
    from hf_data import runstates
    shared = tmp_path / "shared"
    shared.mkdir()
    t = datetime(2026, 6, 30)
    write_grid(str(shared / "crestphys_SM_20260630_0000.tif"), np.ones((4, 4)), WEST, NORTH)
    snow = np.full((4, 4), -9999.0)           # snow17 filled from no pre-run here
    write_grid(str(shared / "snow17_wq_20260630_0000.tif"), snow, WEST, NORTH)
    ok, cov = runstates.stage(str(shared), str(tmp_path / "r1"), t, _nodes(), _tr())
    assert ok and cov == 1.0                  # snow17 grids don't veto the set
    # a half-covered water state: refused at MIN_COVER, accepted at PARTIAL_COVER
    half = np.ones((4, 4))
    half[2, :] = -9999.0
    write_grid(str(shared / "crestphys_SM_20260620_0000.tif"), half, WEST, NORTH)
    t2 = datetime(2026, 6, 20)
    assert runstates.stage(str(shared), str(tmp_path / "r2"), t2, _nodes(), _tr())[0] is False
    assert runstates.stage(str(shared), str(tmp_path / "r3"), t2, _nodes(), _tr(),
                           need=runstates.PARTIAL_COVER)[0] is True


def test_stage_unreadable_grid_is_not_staged(tmp_path):
    from hf_data import runstates
    shared, run = tmp_path / "shared", tmp_path / "run"
    shared.mkdir()
    (shared / "crestphys_SM_20260630_0000.tif").write_bytes(b"not a tiff")
    ok, cov = runstates.stage(str(shared), str(run), datetime(2026, 6, 30), _nodes(), _tr())
    assert not ok and cov == 0.0 and os.listdir(run) == []


def test_commit_no_clobber(tmp_path):
    from hf_data import runstates
    shared, run = tmp_path / "shared", tmp_path / "run"
    shared.mkdir()
    run.mkdir()
    (shared / "SM_20260710_0000.tif").write_bytes(b"hindcast")
    (run / "SM_20260710_0000.tif").write_bytes(b"nowcast")
    (run / "SM_20260711_0000.tif").write_bytes(b"new")
    assert runstates.commit(str(run), str(shared), clobber=False) == 1
    assert (shared / "SM_20260710_0000.tif").read_bytes() == b"hindcast"
    assert (shared / "SM_20260711_0000.tif").read_bytes() == b"new"


def test_nodes_of_speed_domain(tmp_path):
    from hf_data import runstates
    facc = np.full((4, 4), -9999.0)
    facc[1:3, 1:3] = 5.0
    write_grid(str(tmp_path / "facc_clip.tif"), facc, WEST, NORTH)
    m, tr = runstates.nodes_of(str(tmp_path))
    assert (m == _nodes()).all() and abs(tr.c - WEST) < 1e-12

"""Nowcast + fleet callers go through the forcing archive.

The archive is the one PQF decoder and the one owner of the forcing store key;
the operational nowcast writer reads its precip grids through it. Offline:
tars are served through the archive's `fetch` seam."""
from __future__ import annotations

from datetime import datetime

import numpy as np

from tests.test_forcing_archive import FakeStore, NODATA, pqf_bytes, tar_bytes

# MRMS-like grid: 4x5, 1-degree cells, lower-left (-100, 40); -3 = no coverage
MRMS = np.array([[0.0, 1.0, 2.0, 3.0, 4.0],
                 [5.0, -3.0, 7.0, 8.0, 9.0],
                 [10.0, 11.0, NODATA, 13.0, 14.0],
                 [15.0, 16.0, 17.0, 18.0, 19.0]], dtype="float32")
T = datetime(2026, 7, 20, 5)
MRMS_TAR = "mrms/2026/mrms_2026_07.tar"
MRMS_MEMBER = "mrms_corr_2026072005.pqf"


def test_read_without_a_box_returns_the_full_domain():
    from hf_data.forcing_archive import ForcingArchive
    arc = ForcingArchive(fetch=FakeStore({MRMS_TAR: tar_bytes({MRMS_MEMBER: pqf_bytes(MRMS)})}))
    g = arc.read("mrms", T)
    np.testing.assert_array_equal(g.values, MRMS)
    assert (g.xll, g.yll) == (-100.0, 40.0)


def test_forcing_store_key_is_the_bbox_at_three_decimals():
    from hf_data.forcing_archive import store_key
    assert store_key((-97.5, 35.25, -96.123456, 36.0)) == "m97p500_35p250_m96p123_36p000"


def test_forcing_store_dir_and_fleet_prune_use_the_archive_key(tmp_path, monkeypatch):
    import os
    from hf_data import forcing, statecache
    from hf_data.forcing_archive import store_key
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path))
    bbox = (-97.5, 35.25, -96.123456, 36.0)
    assert os.path.basename(forcing.store_dir("mrms", bbox)) == store_key(bbox)

    from fleet import fleet_run
    assert fleet_run._keys(bbox)[1] == store_key(bbox)


# ---- operational nowcast writer (scripts/run_nowcast_all.py) ---------------
def _writer(monkeypatch):
    import importlib.util
    import os
    import sys
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "run_nowcast_all.py")
    spec = importlib.util.spec_from_file_location("run_nowcast_all_t", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_nowcast_all_t", mod)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "MRMS_GRID", (-100.0, 40.0, 1.0, 4, 5, NODATA))
    return mod


# gauge A: 0.3-deg box inside cell (1, 2) -> 7.0
# gauge B: 1-deg box over rows 0-2, cols 0-2 -> mean of the >=0 cells
#          (0, 1, 2, 5, 7, 10, 11; -3 and nodata excluded) = 36/7
LON, LAT, AREA = np.array([-97.5, -98.5]), np.array([42.5, 42.5]), np.array([1.0, 8556.25])
EXPECTED = np.array([7.0, 36.0 / 7.0], dtype="float32")


def test_writer_box_means_from_the_pass2_tar_are_unchanged(monkeypatch):
    from hf_data.forcing_archive import ForcingArchive
    w = _writer(monkeypatch)
    arc = ForcingArchive(fetch=FakeStore({MRMS_TAR: tar_bytes({MRMS_MEMBER: pqf_bytes(MRMS)})}))
    a, from_pass2 = w._hour_grid(T, {}, arc, load_recent=None)
    assert from_pass2 is True
    np.testing.assert_allclose(w._box_means(a, w._grid_boxes(LON, LAT, AREA)), EXPECTED, rtol=1e-6)


def test_writer_prefers_the_recent_pass1_grid(monkeypatch):
    from hf_data.forcing_archive import ForcingArchive
    w = _writer(monkeypatch)
    store = FakeStore({})
    recent = {T: "mrms_recent/mrms1h_pass1_2026072005.pqf"}
    a, from_pass2 = w._hour_grid(T, recent, ForcingArchive(fetch=store),
                                 load_recent=lambda p: pqf_bytes(MRMS))
    assert from_pass2 is False and store.asked == []
    np.testing.assert_allclose(w._box_means(a, w._grid_boxes(LON, LAT, AREA)), EXPECTED, rtol=1e-6)


def test_writer_hour_missing_everywhere_is_none(monkeypatch):
    from hf_data.forcing_archive import ForcingArchive
    w = _writer(monkeypatch)
    a, _ = w._hour_grid(T, {}, ForcingArchive(fetch=FakeStore({})), load_recent=None)
    assert a is None


def test_writer_rejects_a_grid_off_the_mrms_lattice(monkeypatch):
    from hf_data.forcing_archive import ForcingArchive
    w = _writer(monkeypatch)
    arc = ForcingArchive(fetch=FakeStore({MRMS_TAR: tar_bytes({MRMS_MEMBER: pqf_bytes(MRMS[:3])})}))
    a, _ = w._hour_grid(T, {}, arc, load_recent=None)
    assert a is None

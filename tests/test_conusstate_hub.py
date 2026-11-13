"""hf_data.conusstate over the in-memory dataset adapter: the fleet's bundles
are listed, indexed and stitched through the dataset module, and an outage
serves the last good listing/index instead of an empty fleet. Offline."""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from conftest import RES, write_grid
from hf_data import conusstate as cs
from hf_data.hubdata import Dataset, MemoryHub

WEST, NORTH = -110.0, 38.0
T0 = datetime(2026, 10, 20, 16)
KEY = "09337500_crest"


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    from hf_data import statebundle
    sd = tmp_path / "s"
    sd.mkdir()
    write_grid(str(sd / "crest_SM_20261020_1600.tif"), np.full((5, 5), 0.5), WEST, NORTH)
    write_grid(str(sd / "kwr_IR_20261020_1600.tif"), np.full((5, 5), 0.1), WEST, NORTH)
    bpath = tmp_path / "b.pqf"
    statebundle.pack_dir(str(sd), out_path=str(bpath))
    h = MemoryHub({f"states/{KEY}.pqf": bpath.read_bytes(),
                   "states/V70520944_crest.pqf": b"ungauged: not mosaicked",
                   "results/09337500_crest.json": b"{}"})
    monkeypatch.setattr(cs, "DATASET", Dataset("me/fleet", adapter=h))
    monkeypatch.setattr(cs, "CELL", RES)
    monkeypatch.setattr(cs, "_areas", lambda: {})
    return h


def test_listing_index_and_stitch_through_the_dataset(fleet):
    assert cs.listing() == {"09337500": [KEY]}
    assert cs.load_index() == {}                            # no index yet
    idx = cs.refresh_index(workers=1, log=lambda m: None)
    assert idx[KEY]["n_times"] == 1 and idx[KEY]["t0"] == "20261020_1600"
    assert json.loads(fleet.files[cs.INDEX_FILE])["bundles"][KEY]["oid"] == idx[KEY]["oid"]
    assert cs.load_index(force=True)[KEY]["gauge"] == "09337500"
    assert cs.refresh_index(workers=1, log=lambda m: None) == idx
    assert len(fleet.commits) == 1                          # nothing new: no commit
    dst = {"xll": WEST, "yll": NORTH - 5 * RES, "nrows": 5, "ncols": 5}
    res = cs.stitch(dst, [KEY], T0, workers=1)
    assert sorted(res["grids"]) == ["crest_SM", "kwr_IR"] and res["coverage"] == 1.0
    assert float(res["grids"]["crest_SM"][2, 2]) == pytest.approx(0.5, abs=1e-3)


def test_outage_serves_the_last_good_listing_and_index(fleet):
    cs.refresh_index(workers=1, log=lambda m: None)
    assert cs.listing(force=True) and cs.load_index(force=True)
    fleet.outage = True
    assert cs.listing(force=True) == {"09337500": [KEY]}
    assert KEY in cs.load_index(force=True)
    with pytest.raises(RuntimeError):                       # no index built from nothing
        cs.refresh_index(workers=1, log=lambda m: None)


def test_index_404_is_an_empty_index(fleet):
    cs.refresh_index(workers=1, log=lambda m: None)
    assert cs.load_index(force=True)
    del fleet.files[cs.INDEX_FILE]
    assert cs.load_index(force=True) == {}

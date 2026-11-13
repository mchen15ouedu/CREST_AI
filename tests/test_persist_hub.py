"""hf_data.persist over the in-memory dataset adapter: restore pulls the
mirror into the cache (states back to EF5 tifs, local files win), backup
pushes changes and mirrors evictions in one commit. Offline."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from conftest import write_grid
from hf_data import persist
from hf_data.hubdata import Dataset, MemoryHub


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    h = MemoryHub()
    monkeypatch.setattr(persist, "DATASET", Dataset("me/state", adapter=h))
    monkeypatch.setattr(persist, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(persist, "_manifest", {})
    monkeypatch.setattr(persist, "_last", {})
    os.makedirs(tmp_path / "cache")
    return h


def _cache(*parts):
    return os.path.join(persist.CACHE_DIR, *parts)


def test_restore_pulls_the_mirror_and_keeps_local_files(mirror, tmp_path):
    tif = str(tmp_path / "g.tif")
    write_grid(tif, np.full((3, 4), 0.25), -100.0, 40.0)
    mirror.files["states/01234567_crest/crest_SM_20261020_1600.pqf"] = persist._tif_pqf_bytes(tif)
    mirror.files["params/01234567_crest.json"] = b'{"nse": 0.5}'
    mirror.files["users/a.json"] = b'{"remote": true}'
    mirror.files["frames/x.png"] = b"not persisted"
    os.makedirs(_cache("users"))
    with open(_cache("users", "a.json"), "w") as fh:
        fh.write('{"local": true}')
    rep = persist.restore()
    assert rep == {"restored": 2, "kept_local": 1, "failed": 0}
    assert json.load(open(_cache("params", "01234567_crest.json"))) == {"nse": 0.5}
    assert json.load(open(_cache("users", "a.json"))) == {"local": True}
    assert os.path.exists(_cache("states", "01234567_crest", "crest_SM_20261020_1600.tif"))
    assert not os.path.exists(_cache("frames"))
    assert persist.backup() == {"changed": 1, "deleted": 0}     # only the kept local file


def test_backup_pushes_changes_and_mirrors_evictions(mirror):
    mirror.files["results/old.json"] = b"{}"                  # evicted locally
    mirror.files["params/remote_winner.json"] = b"{}"         # append-only
    os.makedirs(_cache("results"))
    with open(_cache("results", "new.json"), "w") as fh:
        fh.write("{}")
    assert persist.backup() == {"changed": 1, "deleted": 1}
    assert len(mirror.commits) == 1
    assert set(mirror.files) == {"results/new.json", "params/remote_winner.json"}
    assert persist.backup() == {"changed": 0, "deleted": 0}   # nothing new: no commit
    assert len(mirror.commits) == 1


def test_backup_failure_is_reported_and_retried(mirror):
    os.makedirs(_cache("results"))
    with open(_cache("results", "new.json"), "w") as fh:
        fh.write("{}")
    mirror.outage = True
    with pytest.raises(RuntimeError):
        persist.backup()
    mirror.outage = False
    assert persist.backup() == {"changed": 1, "deleted": 0}   # not marked synced before


def test_without_write_access_everything_is_a_no_op(mirror):
    mirror.writable = False
    assert persist.restore() == {"enabled": False}
    assert persist.backup() == {"enabled": False}

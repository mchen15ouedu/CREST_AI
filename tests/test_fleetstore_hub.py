"""hf_data.fleetstore (read side) over the in-memory dataset adapter: a fleet
record is merged, a real 404 is remembered as "nothing there", and a Hub
outage is not. Offline."""
from __future__ import annotations

import json

import pytest

from hf_data import fleetstore as fs
from hf_data import statecache
from hf_data.hubdata import Dataset, MemoryHub

ROWS = {"rows": [{"time": "2026-10-01T00:00", "q": 1.0}],
        "state_times": [], "variant": None}


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    h = MemoryHub()
    monkeypatch.setattr(fs, "DATASET", Dataset("me/fleet", adapter=h))
    monkeypatch.setattr(statecache, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(fs, "_seen", {})
    monkeypatch.setattr(fs, "_remote", {})
    monkeypatch.setenv("HF_TOKEN", "test")
    return h


def test_fleet_record_is_fetched_then_cached(fleet):
    fleet.files["results/01234567_m.json"] = json.dumps(ROWS).encode()
    assert fs.ensure_local("1234567", "m") == "fetched"
    assert statecache.load_record("1234567", "m")["rows"][0]["q"] == 1.0
    assert fs.ensure_local("1234567", "m") == "cached"


def test_fleet_404_is_remembered_as_nothing_there(fleet):
    assert fs.ensure_local("1234567", "m") is None
    fleet.files["results/01234567_m.json"] = json.dumps(ROWS).encode()
    assert fs.ensure_local("1234567", "m") is None      # within TTL: not re-probed


def test_fleet_outage_is_not_nothing_there(fleet):
    fleet.files["results/01234567_m.json"] = json.dumps(ROWS).encode()
    fleet.outage = True
    assert fs.ensure_local("1234567", "m") is None      # run proceeds uncached ...
    assert fs.has_remote("1234567", "m") is False
    fleet.outage = False
    assert fs.ensure_local("1234567", "m") == "fetched"  # ... and the next one gets it
    fleet.files["states/01234567_m.pqf"] = b"x"
    assert fs.has_remote("1234567", "m") is True


def test_fleet_without_token_does_not_probe(fleet, monkeypatch):
    monkeypatch.delenv("HF_TOKEN")
    fleet.files["results/01234567_m.json"] = json.dumps(ROWS).encode()
    assert fs.ensure_local("1234567", "m") is None and not fs.has_remote("1234567", "m")

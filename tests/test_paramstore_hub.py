"""hf_data.paramstore over the in-memory dataset adapter: a newer winner in
the state dataset reaches this process, a push lands there, and an outage
neither loses the local record nor delays the next look. Offline."""
from __future__ import annotations

import json

import pytest

from hf_data import paramstore as ps
from hf_data.hubdata import Dataset, MemoryHub

OLD = {"wb": {"a": 1}, "kw": {}, "nse": 0.3, "source": "manual",
       "when": "2026-10-01 00:00 UTC", "window": None}
NEW = {**OLD, "nse": 0.6, "source": "autocal", "when": "2026-11-01 00:00 UTC"}


@pytest.fixture
def state(tmp_path, monkeypatch):
    h = MemoryHub()
    monkeypatch.setattr(ps, "DATASET", Dataset("me/state", adapter=h))
    monkeypatch.setattr(ps, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ps, "_remote_seen", {})
    monkeypatch.setenv("HF_TOKEN", "test")
    return h


def _put_local(rec):
    with open(ps._path("1234567", "crest"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh)


def test_newer_remote_winner_replaces_the_local_record(state):
    _put_local(OLD)
    state.files["params/01234567_crest.json"] = json.dumps(NEW).encode()
    assert ps.get("1234567", "crest")["source"] == "autocal"
    assert ps._local("1234567", "crest")["source"] == "autocal"   # written through


def test_outage_serves_local_and_retries_next_time(state):
    _put_local(OLD)
    state.files["params/01234567_crest.json"] = json.dumps(NEW).encode()
    state.outage = True
    assert ps.get("1234567", "crest")["source"] == "manual"
    state.outage = False
    assert ps.get("1234567", "crest")["source"] == "autocal"     # not held off for TTL


def test_push_remote_commits_the_local_record(state):
    assert ps.push_remote("1234567", "crest") is False            # nothing local
    _put_local(NEW)
    assert ps.push_remote("1234567", "crest") is True
    assert json.loads(state.files["params/01234567_crest.json"]) == NEW
    state.outage = True
    assert ps.push_remote("1234567", "crest") is False

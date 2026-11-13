"""hf_data.eventstore over the in-memory dataset adapter: what the store does
when a file is there, is a real 404, or the Hub is unreachable. Offline."""
from __future__ import annotations

import json

import pytest

from hf_data import eventstore as es
from hf_data.hubdata import Dataset, MemoryHub


@pytest.fixture
def hub(monkeypatch):
    h = MemoryHub()
    monkeypatch.setattr(es, "DATASET", Dataset("me/data", adapter=h))
    return h


def _json(hub, path):
    return json.loads(hub.files[path])


def test_autocal_log_missing_vs_unreachable(hub):
    assert es.load_autocal() == [] and es.load_autocal_done() == {}   # not created yet
    hub.files[es.AUTOCAL_FILE] = b'[{"gauge": "1"}]'
    hub.files[es.AUTOCAL_DONE_FILE] = b"[]"
    assert es.load_autocal() == [{"gauge": "1"}]
    assert es.load_autocal_done() is None                  # wrong type: unreadable
    hub.files[es.AUTOCAL_FILE] = b"[{"
    assert es.load_autocal() is None                       # corrupt JSON
    hub.outage = True
    assert es.load_autocal() is None and es.load_autocal_done() is None


def test_note_autocal_commits_record_and_ledger_together(hub, monkeypatch):
    rows = [{"gauge": "09000003", "ran": True, "at": "2026-10-01T00:00:00Z"},
            {"gauge": "09000002", "ran": False, "at": "2026-10-02T00:00:00Z"}]
    hub.files[es.AUTOCAL_FILE] = json.dumps(rows).encode()
    rec = {"gauge": "09000004", "kind": "backlog", "ran": True,
           "at": "2026-10-20T00:00:00Z", "baseline_nse": 0.1, "best_nse": 0.4}
    assert es.note_autocal(rec) is True
    assert len(hub.commits) == 1                            # one commit, both files
    assert _json(hub, es.AUTOCAL_FILE) == rows + [rec]
    assert _json(hub, es.AUTOCAL_DONE_FILE) == {           # first write: seeded from the log
        "09000003": {"first": rows[0]["at"], "last": rows[0]["at"], "n": 1},
        "09000004": {"first": rec["at"], "last": rec["at"], "n": 1}}
    skip = {**rec, "ran": False, "reason": "x"}
    assert es.note_autocal(skip) is True
    assert {op.path for op in hub.commits[-1]["ops"]} == {es.AUTOCAL_FILE}   # log only
    monkeypatch.setattr(es, "AUTOCAL_KEEP", 2)
    assert es.note_autocal(rec) is True
    assert len(_json(hub, es.AUTOCAL_FILE)) == 2           # the log is capped ...
    assert set(_json(hub, es.AUTOCAL_DONE_FILE)) == {"09000003", "09000004"}   # ... the ledger not
    assert _json(hub, es.AUTOCAL_DONE_FILE)["09000004"]["n"] == 2


def test_note_autocal_never_overwrites_history_it_could_not_read(hub):
    rec = {"gauge": "09000004", "ran": True, "at": "2026-10-20T00:00:00Z"}
    hub.files[es.AUTOCAL_FILE] = b"[{"                      # log unreadable
    assert es.note_autocal(rec) is False and hub.commits == []
    hub.files[es.AUTOCAL_FILE] = b"[]"
    hub.files[es.AUTOCAL_DONE_FILE] = b"[]"                 # ledger unreadable
    assert es.note_autocal(rec) is False and hub.commits == []
    assert es.note_autocal({**rec, "ran": False}) is True   # a skip needs the log only
    hub.outage = True
    assert es.note_autocal({**rec, "ran": False}) is False
    hub.outage = False
    hub.writable = False                                    # no token: nothing to do
    assert es.note_autocal({**rec, "ran": False}) is False
    assert len(hub.commits) == 1


def test_pinned_ids_missing_is_empty_but_an_outage_keeps_the_last_set(hub):
    assert es.pinned_ids(force=True) == frozenset()          # no pinned.json yet
    hub.files[es.PINNED_FILE] = b'{"ids": ["2026081015_07233650"], "note": "demo"}'
    assert es.pinned_ids(force=True) == {"2026081015_07233650"}
    hub.outage = True
    assert es.pinned_ids(force=True) == {"2026081015_07233650"}   # never unprotects


def test_pinned_ids_unknown_until_first_read(hub):
    hub.outage = True
    assert es.pinned_ids(force=True) is None                 # callers skip retention


def _event_dir(tmp_path, ev, frames):
    d = tmp_path / ev
    d.mkdir()
    for f in frames:
        (d / f"depth_{f}.tif").write_bytes(b"t")
        (d / f"depth_{f}.png").write_bytes(b"p")
    man = {"event_id": ev, "t0": "2026-10-20T12:00Z", "sim_start": "2026-10-20T06:00Z",
           "status": "active", "provenance": {"engine": "test"},
           "frames": [{"t": f"2026-10-20T{f}:00Z", "file": f"depth_{f}.tif",
                       "png": f"depth_{f}.png"} for f in frames]}
    return str(d), man


def test_publish_event_one_commit_with_files_and_index(hub, tmp_path):
    hub.files[es.INDEX_FILE] = b'{"2026101000_old": {"status": "ended"}}'
    ev = "2026102012_07233650"
    d, man = _event_dir(tmp_path, ev, ["12", "13"])
    assert es.publish_event(d, man) is True
    assert len(hub.commits) == 1
    assert {f"events/{ev}/depth_12.tif", f"events/{ev}/depth_13.png",
            f"events/{ev}/manifest.json"} <= set(hub.files)
    idx = _json(hub, es.INDEX_FILE)
    assert list(idx) == [ev, "2026101000_old"]               # newest first, old kept
    assert idx[ev]["n_frames"] == 2 and idx[ev]["engine"] == "test"
    assert es.load_index() == idx
    assert es.event_url(ev, "depth_12.png") == \
        f"https://huggingface.co/datasets/me/data/resolve/main/events/{ev}/depth_12.png"


def test_republish_carries_earlier_frames_and_aborts_if_prior_manifest_unreachable(
        hub, tmp_path):
    ev = "2026102012_07233650"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    d, man = _event_dir(tmp_path / "a", ev, ["08", "09"])
    assert es.publish_event(d, man) is True
    d2, man2 = _event_dir(tmp_path / "b", ev, ["10"])
    man2["sim_start"] = "2026-10-20T10:00Z"                  # re-simulated from 10Z on
    assert es.publish_event(d2, dict(man2)) is True
    m = _json(hub, f"events/{ev}/manifest.json")
    assert [f["file"] for f in m["frames"]] == ["depth_08.tif", "depth_09.tif", "depth_10.tif"]
    # the prior manifest exists but cannot be read: publishing now would drop
    # the carried frames, so the store waits
    n = len(hub.commits)
    hub.files[f"events/{ev}/manifest.json"] = b"{"
    assert es.publish_event(d2, dict(man2)) is False and len(hub.commits) == n


def test_writers_never_mistake_an_unreachable_index_for_an_empty_one(hub, tmp_path):
    hub.files[es.INDEX_FILE] = b"{"                          # unreadable index
    ev = "2026102012_07233650"
    d, man = _event_dir(tmp_path, ev, ["12"])
    assert es.load_index() == {}                             # readers: nothing to show
    assert es.publish_event(d, man) is False                 # writers: retry later
    assert es.mark_ended([ev]) is False
    assert es.drop_events([ev]) is False
    hub.files[es.PINNED_FILE] = b"[]"
    assert es.retention_sweep() == 0
    assert hub.commits == []


def test_publish_event_with_no_index_yet_starts_one(hub, tmp_path):
    ev = "2026102012_07233650"
    d, man = _event_dir(tmp_path, ev, ["12"])
    assert es.publish_event(d, man) is True
    assert list(_json(hub, es.INDEX_FILE)) == [ev]


def test_note_missed_appends_and_never_overwrites_an_unreadable_log(hub):
    rec = {"event": "E1", "at": "2026-10-20T12:00Z", "frac": 0.1,
           "sim_peak_m3s": 1.0, "obs_peak_m3s": 9.0}
    assert es.note_missed(rec) is True                       # no log yet: starts one
    assert es.note_missed({**rec, "at": "2026-10-20T13:00Z"}) is True
    rows = _json(hub, es.MISSED_FILE)
    assert len(rows) == 1 and rows[0]["n_checks"] == 2
    assert rows[0]["first_at"] == "2026-10-20T12:00Z"
    hub.outage = True
    assert es.note_missed(rec) is False
    hub.outage = False
    hub.files[es.MISSED_FILE] = b"[{"
    assert es.note_missed(rec) is False
    assert len(hub.commits) == 2

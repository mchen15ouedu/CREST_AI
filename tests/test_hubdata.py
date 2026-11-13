"""hf_data.hubdata: the one module that talks to the Hugging Face dataset repos.

Everything here runs offline: the in-memory adapter stands in for the Hub, and
the real adapter is exercised only against a faked hf_hub_download."""
from __future__ import annotations

import pytest

from hf_data import hubdata
from hf_data.hubdata import Dataset, MemoryHub


def test_read_json_answers_data_missing_or_unreachable():
    hub = MemoryHub({"a.json": b'{"x": 1}'})
    ds = Dataset("me/repo", adapter=hub)

    r = ds.read_json("a.json")
    assert r.ok and r.value == {"x": 1} and not r.missing and not r.unreachable

    r = ds.read_json("nope.json")
    assert r.missing and not r.ok and r.value is None

    hub.outage = True
    r = ds.read_json("a.json")
    assert r.unreachable and not r.missing and not r.ok
    r = ds.read_json("nope.json")
    assert r.unreachable and not r.missing        # an outage never reads as missing


def test_corrupt_or_wrong_type_json_is_not_data():
    ds = Dataset("me/repo", adapter=MemoryHub({"bad.json": b"[{", "list.json": b"[1]"}))
    assert ds.read_json("bad.json").unreachable
    assert ds.read_json("list.json", kind=dict).unreachable
    assert ds.read_json("list.json", kind=list).value == [1]


def test_commit_then_list_and_read_back(tmp_path):
    from hf_data.hubdata import Delete, Put
    hub = MemoryHub({"ev/1/a.tif": b"a", "ev/1/b.tif": b"b", "ev/2/c.tif": b"c"})
    ds = Dataset("me/repo", adapter=hub)
    local = tmp_path / "m.json"
    local.write_text('{"n": 2}')
    res = ds.commit([Put("idx.json", {"k": 1}),            # JSON value
                     Put("ev/3/m.json", str(local)),        # local file
                     Put("raw.bin", b"\x00\x01"),           # bytes
                     Delete("ev/1/a.tif"),
                     Delete("ev/2/", folder=True)], "msg")
    assert res and res.error is None
    assert ds.list_files().value == ["ev/1/b.tif", "ev/3/m.json", "idx.json", "raw.bin"]
    assert ds.list_files(prefix="ev/").value == ["ev/1/b.tif", "ev/3/m.json"]
    assert ds.read_json("ev/3/m.json").value == {"n": 2}
    assert ds.read_json("idx.json").value == {"k": 1}
    assert hub.commits[-1]["message"] == "msg"


def test_commit_failure_and_listing_outage_are_reported():
    from hf_data.hubdata import Put
    hub = MemoryHub({"a": b"1"})
    ds = Dataset("me/repo", adapter=hub)
    hub.outage = True
    assert ds.list_files().unreachable
    res = ds.commit([Put("b", b"2")], "msg")
    assert not res and "outage" in res.error
    hub.outage, hub.writable = False, False
    res = ds.commit([Put("b", b"2")], "msg")
    assert not res and "token" in res.error
    assert ds.list_files().value == ["a"]                   # nothing was written


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_commits_stop_at_the_hourly_limit_and_resume_as_the_window_slides():
    from hf_data.hubdata import CommitBudget, Put
    clock = _Clock()
    budget = CommitBudget(limit=3, window_s=3600, clock=clock)
    hub = MemoryHub()
    ds = Dataset("me/repo", adapter=hub, budget=budget)
    for i in range(3):
        assert ds.commit([Put(f"f{i}", b"x")], "m")
        clock.t += 60
    assert budget.remaining() == 0
    res = ds.commit([Put("f3", b"x")], "m")
    assert not res and "budget" in res.error and "f3" not in hub.files
    clock.t = 1000.0 + 3600 + 1                              # the first commit aged out
    assert budget.remaining() == 1
    assert ds.commit([Put("f3", b"x")], "m")
    # only commits that landed count against the limit
    hub.outage = True
    clock.t += 3600 * 2
    assert not ds.commit([Put("f4", b"x")], "m")
    assert budget.remaining() == 3


def test_batch_folds_writes_into_one_commit_per_interval():
    from hf_data.hubdata import Put
    clock = _Clock()
    hub = MemoryHub()
    ds = Dataset("me/repo", adapter=hub)
    batch = ds.batch("checkpoints", min_interval_s=600, max_ops=10, clock=clock)
    batch.add(Put("s/a", b"1"))
    batch.add(Put("s/b", b"1"))
    batch.add(Put("s/a", b"2"))                              # same path: latest wins
    assert batch.pending == 2
    assert batch.flush() is None and hub.commits == []      # first flush waits an interval
    clock.t += 600
    assert batch.flush()
    assert len(hub.commits) == 1 and hub.files == {"s/a": b"2", "s/b": b"1"}
    assert "checkpoints" in hub.commits[0]["message"] and batch.pending == 0
    # a full batch goes out at once; force flushes whatever is pending
    for i in range(10):
        batch.add(Put(f"s/{i}", b"x"))
    assert batch.flush() and len(hub.commits) == 2
    batch.add(Put("s/z", b"x"))
    assert batch.flush(force=True) and len(hub.commits) == 3
    # a failed commit keeps the ops queued for the next flush
    batch.add(Put("s/y", b"x"))
    hub.outage = True
    res = batch.flush(force=True)
    assert res is not None and not res and batch.pending == 1
    hub.outage = False
    assert batch.flush(force=True) and "s/y" in hub.files


def test_ttl_cache_serves_stale_during_an_outage_but_not_after_a_404():
    clock = _Clock()
    hub = MemoryHub({"pins.json": b'["a"]'})
    ds = Dataset("me/repo", adapter=hub)
    pins = ds.cached("pins.json", ttl_s=600, retry_s=60, kind=list, clock=clock)
    r = pins.get()
    assert r.ok and r.value == ["a"] and not r.stale
    hub.files["pins.json"] = b'["a", "b"]'
    clock.t += 599
    assert pins.get().value == ["a"]                         # within TTL: no Hub call
    clock.t += 2
    assert pins.get().value == ["a", "b"]                    # expired: re-read
    hub.outage = True
    clock.t += 601
    r = pins.get()
    assert r.ok and r.stale and r.value == ["a", "b"]        # outage: last good copy
    hub.outage = False
    hub.files["pins.json"] = b'["c"]'
    clock.t += 30
    assert pins.get().stale                                  # retry waits retry_s
    clock.t += 31
    r = pins.get()
    assert r.value == ["c"] and not r.stale
    del hub.files["pins.json"]
    clock.t += 601
    assert pins.get().missing                                # a real 404 is an answer
    hub.outage = True
    clock.t += 601
    assert pins.get().unreachable                            # nothing good to fall back on


def test_ttl_cache_custom_loader_reads_the_local_file():
    hub = MemoryHub({"x.txt": b"hello"})
    c = Dataset("me/repo", adapter=hub).cached(
        "x.txt", ttl_s=60, load=lambda p: open(p).read().upper())
    assert c.get().value == "HELLO"


def test_repo_identity_and_token_come_from_one_place(monkeypatch):
    monkeypatch.delenv("CREST_DATA_REPO", raising=False)
    monkeypatch.delenv("CREST_FEEDBACK_REPO", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hubdata.data().repo == "vincewin/CREST_data"
    assert hubdata.feedback().repo == "vincewin/CREST_data"
    assert not hubdata.data().can_write                    # no token: read-only
    monkeypatch.setenv("CREST_DATA_REPO", "me/data")
    monkeypatch.setenv("CREST_FEEDBACK_REPO", "me/fb")
    monkeypatch.setenv("HF_TOKEN", "tok")
    assert hubdata.data().repo == "me/data" and hubdata.feedback().repo == "me/fb"
    assert hubdata.data().can_write and hubdata.token() == "tok"
    assert hubdata.data().url("events/x/a.png") == \
        "https://huggingface.co/datasets/me/data/resolve/main/events/x/a.png"


def _fake_download(monkeypatch, tmp_path, files: dict, err=None, calls=None):
    """Fake huggingface_hub.hf_hub_download over {path: text}; absent = 404."""
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    def dl(repo, path, **kw):
        if calls is not None:
            calls.append((repo, path, kw))
        if err is not None:
            raise err
        if path not in files:
            raise EntryNotFoundError(path)
        p = tmp_path / path.replace("/", "_")
        p.write_text(files[path], encoding="utf-8")
        return str(p)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", dl)


def test_hub_adapter_only_a_real_404_is_missing(monkeypatch, tmp_path):
    from huggingface_hub.utils import LocalEntryNotFoundError
    monkeypatch.setenv("HF_TOKEN", "tok")
    calls = []
    ds = Dataset("me/repo", adapter=hubdata.HubAdapter())
    _fake_download(monkeypatch, tmp_path, {"a.json": "[1]"}, calls=calls)
    assert ds.read_json("a.json", fresh=True).value == [1]
    assert calls[-1][0] == "me/repo" and calls[-1][2]["token"] == "tok"
    assert calls[-1][2]["force_download"] is True and calls[-1][2]["repo_type"] == "dataset"
    assert ds.read_json("b.json").missing
    # LocalEntryNotFoundError subclasses EntryNotFoundError but means offline
    for e in (LocalEntryNotFoundError("offline"), OSError("disk"), RuntimeError("401")):
        _fake_download(monkeypatch, tmp_path, {}, err=e)
        assert ds.read_json("b.json").unreachable

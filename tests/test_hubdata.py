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


def test_concurrent_commits_never_overspend_the_budget():
    """Check-and-spend is one step: two writers racing for the last commit of
    the hour cannot both get it."""
    import threading
    from hf_data.hubdata import CommitBudget, Put

    class SlowHub(MemoryHub):
        def __init__(self):
            super().__init__()
            self.calls, self.inside, self.go = 0, threading.Event(), threading.Event()

        def commit(self, repo, ops, message):
            self.calls += 1
            self.inside.set()
            self.go.wait(timeout=2)
            super().commit(repo, ops, message)

    hub = SlowHub()
    ds = Dataset("me/repo", adapter=hub, budget=CommitBudget(limit=1))
    first = threading.Thread(target=ds.commit, args=([Put("a", b"1")], "m"))
    first.start()
    assert hub.inside.wait(timeout=2)
    second = ds.commit([Put("b", b"2")], "m")        # races the first one
    hub.go.set()
    first.join()
    assert not second and "budget" in second.error
    assert hub.calls == 1 and sorted(hub.files) == ["a"]


def test_a_failed_commit_gives_its_budget_back():
    from hf_data.hubdata import CommitBudget, Put
    hub = MemoryHub()
    budget = CommitBudget(limit=1)
    ds = Dataset("me/repo", adapter=hub, budget=budget)
    hub.outage = True
    assert not ds.commit([Put("a", b"1")], "m")
    assert budget.remaining() == 1
    hub.outage = False
    assert ds.commit([Put("a", b"1")], "m") and budget.remaining() == 0


def test_memory_hub_keeps_one_file_map_per_repo():
    from hf_data.hubdata import Put
    hub = MemoryHub({"a.json": b"[1]"})
    data = Dataset("me/data", adapter=hub)
    fleet = Dataset("me/fleet", adapter=hub)
    assert data.read_json("a.json").value == [1]           # seeded: the first repo used
    assert fleet.read_json("a.json").missing
    assert fleet.commit([Put("b.json", [2])], "m")
    assert fleet.list_files().value == ["b.json"]
    assert data.list_files().value == ["a.json"]
    assert sorted(hub.files) == ["a.json"]                  # .files = the first repo's map
    assert sorted(hub.repo_files("me/fleet")) == ["b.json"]


def test_batch_entries_group_ops_and_bound_the_queue_after_a_failure():
    """An entry (several ops under one key) is queued, replaced and committed
    as a unit; a failed flush keeps the oldest `max_pending` entries."""
    from hf_data.hubdata import Put
    hub = MemoryHub()
    ds = Dataset("me/repo", adapter=hub)
    batch = ds.batch("ckpts", min_interval_s=0, max_pending=1)
    batch.add([Put("s/a.pqf", b"1"), Put("r/a.json", b"{}")], key="a")
    batch.add([Put("s/b.pqf", b"22")], key="b")
    assert batch.pending == 2 and batch.nbytes == 5 and batch.keys() == ["a", "b"]
    hub.outage = True
    assert not batch.flush(force=True) and "outage" in batch.error
    assert batch.keys() == ["a"]                             # cut to max_pending
    hub.outage = False
    batch.add([Put("s/a.pqf", b"3")], key="a")               # replaces the whole entry
    assert batch.flush(force=True) and batch.error is None
    assert hub.files == {"s/a.pqf": b"3"} and hub.commits[0]["message"] == "ckpts x1"


def test_reads_at_a_revision_and_commits_checked_against_a_parent(tmp_path):
    """A read-modify-write pinned to head(): reads see that revision, and a
    commit whose parent is no longer the head answers moved (never lands,
    never spends budget)."""
    hub = MemoryHub({"index.json": {"a": 1}})
    ds = Dataset("me/data", adapter=hub)
    rev = ds.head().value
    assert ds.commit([hubdata.Put("index.json", {"a": 2})], "other writer")
    assert ds.read_json("index.json", revision=rev).value == {"a": 1}
    assert ds.read_json("index.json").value == {"a": 2}
    left = ds.budget.remaining()
    res = ds.commit([hubdata.Put("index.json", {"a": 3})], "stale", parent=rev)
    assert not res and res.moved and ds.budget.remaining() == left
    assert ds.read_json("index.json").value == {"a": 2}
    head = ds.head().value
    assert head != rev
    assert ds.commit([hubdata.Put("index.json", {"a": 3})], "fresh", parent=head)
    hub.outage = True
    assert ds.head().unreachable


def test_local_path_into_a_private_dir(tmp_path):
    ds = Dataset("me/data", adapter=MemoryHub({"events/e/manifest.json": b"{}"}))
    got = ds.local_path("events/e/manifest.json", local_dir=str(tmp_path))
    assert got.ok and got.value == str(tmp_path / "events" / "e" / "manifest.json")
    assert ds.local_path("events/e/none.json", local_dir=str(tmp_path)).missing


def _hub_http(code):
    """An HfHubHTTPError carrying status `code`, on huggingface_hub 0.x and 1.x."""
    from huggingface_hub.utils import HfHubHTTPError
    try:
        import httpx
        return HfHubHTTPError(f"{code}", response=httpx.Response(
            code, request=httpx.Request("POST", "https://huggingface.co/api/commit")))
    except Exception:
        import requests
        resp = requests.Response()
        resp.status_code = code
        return HfHubHTTPError(f"{code}", response=resp)


def test_hub_adapter_moved_parent_is_told_apart(monkeypatch):
    """create_commit's 409/412 against a parent_commit = the head moved;
    without a parent it is an ordinary failed commit."""
    import huggingface_hub
    monkeypatch.setenv("HF_TOKEN", "tok")
    seen = []

    class Api:
        def __init__(self, token=None):
            pass

        def create_commit(self, **kw):
            seen.append(kw)
            raise _hub_http(412)
    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    ds = Dataset("me/data", adapter=hubdata.HubAdapter(), budget=hubdata.CommitBudget())
    res = ds.commit([hubdata.Put("a.json", {})], "m", parent="abc")
    assert not res and res.moved and seen[-1]["parent_commit"] == "abc"
    res = ds.commit([hubdata.Put("a.json", {})], "m")
    assert not res and not res.moved and "parent_commit" not in seen[-1]

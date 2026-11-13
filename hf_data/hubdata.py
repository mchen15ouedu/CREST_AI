"""Access to the Hugging Face dataset repos (CREST_data and friends).

One module owns:
  * repo identity and token (env CREST_DATA_REPO / CREST_FEEDBACK_REPO, HF_TOKEN)
  * reads with three answers — data, missing (a real 404), or unreachable
    (network, auth, offline cache miss, unparseable content). An outage never
    reads as "missing".
  * listing
  * commits, metered against the account's 256 commits/hour limit, and a
    batcher that folds many small writes into one commit
  * TTL caching that serves the last good copy during an outage

Two adapters sit behind the Dataset interface: HubAdapter (the real Hub via
huggingface_hub) and MemoryHub (a dict, for tests)."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

DATA = "data"
MISSING = "missing"
UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class Read:
    """One read's answer. `value` is set only when status is DATA; `stale`
    marks a cached copy served because the Hub was unreachable."""
    status: str
    value: Any = None
    error: str | None = None
    stale: bool = False

    @property
    def ok(self) -> bool:
        return self.status == DATA

    @property
    def missing(self) -> bool:
        return self.status == MISSING

    @property
    def unreachable(self) -> bool:
        return self.status == UNREACHABLE


@dataclass(frozen=True)
class Put:
    """Write `path`. `content`: bytes, a local file path (str), or any other
    value, which is stored as JSON."""
    path: str
    content: Any

    def payload(self) -> bytes | str:
        """bytes, or the local file path for a str content."""
        c = self.content
        if isinstance(c, (bytes, str)):
            return c
        return json.dumps(c).encode("utf-8")


@dataclass(frozen=True)
class Delete:
    """Remove `path`; with folder=True everything under `path` (a prefix
    ending in '/')."""
    path: str
    folder: bool = False


@dataclass(frozen=True)
class Committed:
    """A commit's outcome; truthy on success."""
    ok: bool
    error: str | None = None

    def __bool__(self) -> bool:
        return self.ok


class NotFound(Exception):
    """Adapter signal: the repo answered, and the file is not there."""


class Unreachable(Exception):
    """Adapter signal: no trustworthy answer (network, auth, offline)."""


# --------------------------------------------------------------------------- #
# repo identity and token
# --------------------------------------------------------------------------- #
DEFAULT_DATA_REPO = "vincewin/CREST_data"


def data_repo() -> str:
    """The main dataset (forcing, events, states, ...)."""
    return os.environ.get("CREST_DATA_REPO", DEFAULT_DATA_REPO)


def feedback_repo() -> str:
    """The user-feedback dataset (defaults to the data repo)."""
    return os.environ.get("CREST_FEEDBACK_REPO") or data_repo()


DEFAULT_FLEET_REPO = "vincewin/CREST_fleet"


def fleet_repo() -> str:
    """The fleet's precomputed runs (private; separate from CREST_state, whose
    janitor mirrors local deletions)."""
    return os.environ.get("CREST_FLEET_REPO", DEFAULT_FLEET_REPO)


DEFAULT_STATE_REPO = "vincewin/CREST_state"


def state_repo() -> str:
    """The private cache mirror (persist.py: states, results, params, ...)."""
    return os.environ.get("CREST_STATE_REPO", DEFAULT_STATE_REPO)


def token() -> str | None:
    """The Hub token, or None (read-only, anonymous)."""
    return os.environ.get("HF_TOKEN") or None


def hf_api():
    """Escape hatch for code not yet on Dataset: an authenticated HfApi, or
    None without a token. New code should not need it."""
    tok = token()
    if tok is None:
        return None
    from huggingface_hub import HfApi
    return HfApi(token=tok)


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
@contextmanager
def _no_answer():
    """Adapter calls whose every failure means "the Hub gave no answer"."""
    try:
        yield
    except Exception as e:
        raise Unreachable(f"{type(e).__name__}: {e}") from e


class HubAdapter:
    """The real Hub through huggingface_hub. The token is read per call, so a
    Space that receives HF_TOKEN late still writes."""

    def download(self, repo: str, path: str, *, fresh: bool) -> str:
        import huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
        try:
            return huggingface_hub.hf_hub_download(
                repo, path, repo_type="dataset", token=token(),
                force_download=fresh)
        except LocalEntryNotFoundError as e:   # offline: subclass of the 404 below
            raise Unreachable(str(e)) from e
        except EntryNotFoundError as e:        # the repo answered: no such file
            raise NotFound(path) from e

    def can_write(self) -> bool:
        return token() is not None

    def _api(self):
        from huggingface_hub import HfApi
        return HfApi(token=token())

    def list_files(self, repo: str) -> list[str]:
        with _no_answer():
            return list(self._api().list_repo_files(repo, repo_type="dataset"))

    def exists(self, repo: str, path: str) -> bool:
        with _no_answer():
            return bool(self._api().get_paths_info(repo, [path], repo_type="dataset"))

    def file_info(self, repo: str) -> dict:
        with _no_answer():
            info = self._api().dataset_info(repo, files_metadata=True)
        return {s.rfilename: (s.blob_id, s.size) for s in info.siblings}

    def snapshot(self, repo: str, local_dir: str, patterns: list[str]) -> None:
        from huggingface_hub import snapshot_download
        with _no_answer():
            snapshot_download(repo_id=repo, repo_type="dataset", token=token(),
                              local_dir=local_dir, allow_patterns=patterns)

    def create(self, repo: str, private: bool) -> None:
        self._api().create_repo(repo, repo_type="dataset", private=private,
                                exist_ok=True)

    def open(self, repo: str, path: str):
        from huggingface_hub import HfFileSystem
        try:
            return HfFileSystem(token=token()).open(
                f"datasets/{repo}/{path}", "rb", block_size=1 << 20,
                cache_type="readahead")
        except FileNotFoundError as e:
            raise NotFound(path) from e

    def commit(self, repo: str, ops: list, message: str) -> None:
        import io
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        hub_ops = []
        for op in ops:
            if isinstance(op, Put):
                body = op.payload()
                hub_ops.append(CommitOperationAdd(
                    op.path, io.BytesIO(body) if isinstance(body, bytes) else body))
            else:
                hub_ops.append(CommitOperationDelete(op.path, is_folder=op.folder))
        self._api().create_commit(repo_id=repo, repo_type="dataset",
                                  operations=hub_ops, commit_message=message)


class MemoryHub:
    """In-memory dataset repos for tests: one {path: bytes} map per repo id.
    The seed `files` belong to the first repo the hub is asked about (or to
    `repo`, if given), and `.files` is that repo's map; every other repo starts
    empty (`repo_files(repo)`). Set `outage = True` to make every call
    unreachable."""

    def __init__(self, files: dict | None = None, repo: str | None = None):
        self._home = repo
        self._seed: dict[str, bytes] = {k: _as_bytes(v) for k, v in (files or {}).items()}
        self._repos: dict[str, dict[str, bytes]] = {}
        if repo is not None:
            self._repos[repo] = self._seed
        self.outage = False
        self.writable = True
        self.commits: list[dict] = []          # {"message", "ops", "repo"} per commit
        self._tmp = None

    @property
    def files(self) -> dict[str, bytes]:
        """The first repo's file map (the seed, before any repo was used)."""
        return self._seed if self._home is None else self._repos[self._home]

    def repo_files(self, repo: str) -> dict[str, bytes]:
        """The file map of `repo` (live: tests may edit it)."""
        if self._home is None:
            self._home, self._repos[repo] = repo, self._seed
        return self._repos.setdefault(repo, {})

    def can_write(self) -> bool:
        return self.writable

    def list_files(self, repo: str) -> list[str]:
        self._check()
        return sorted(self.repo_files(repo))

    def exists(self, repo: str, path: str) -> bool:
        self._check()
        return path in self.repo_files(repo)

    def file_info(self, repo: str) -> dict:
        import hashlib
        self._check()
        return {k: (hashlib.sha1(v).hexdigest(), len(v))
                for k, v in self.repo_files(repo).items()}

    def snapshot(self, repo: str, local_dir: str, patterns: list[str]) -> None:
        from fnmatch import fnmatch
        self._check()
        for k, v in self.repo_files(repo).items():
            if any(fnmatch(k, p) for p in patterns):
                p = os.path.join(local_dir, *k.split("/"))
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as fp:
                    fp.write(v)

    def create(self, repo: str, private: bool) -> None:
        self._check()

    def open(self, repo: str, path: str):
        import io
        self._check()
        files = self.repo_files(repo)
        if path not in files:
            raise NotFound(path)
        return io.BytesIO(files[path])

    def commit(self, repo: str, ops: list, message: str) -> None:
        self._check()
        live = self.repo_files(repo)
        files = dict(live)                     # all or nothing, like the Hub
        for op in ops:
            if isinstance(op, Put):
                body = op.payload()
                if isinstance(body, str):
                    with open(body, "rb") as fp:
                        body = fp.read()
                files[op.path] = body
            elif op.folder:
                files = {k: v for k, v in files.items() if not k.startswith(op.path)}
            else:
                files.pop(op.path, None)
        live.clear()                           # keep the map's identity
        live.update(files)
        self.commits.append({"message": message, "ops": list(ops), "repo": repo})

    def _check(self):
        if self.outage:
            raise Unreachable("memory hub: outage")

    def download(self, repo: str, path: str, *, fresh: bool) -> str:
        self._check()
        files = self.repo_files(repo)
        if path not in files:
            raise NotFound(path)
        if self._tmp is None:
            self._tmp = tempfile.mkdtemp(prefix="memhub_")
        p = os.path.join(self._tmp, repo.replace("/", "__"), path.replace("/", "__"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fp:
            fp.write(files[path])
        return p


def _as_bytes(v) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        return v.encode("utf-8")
    return json.dumps(v).encode("utf-8")


# --------------------------------------------------------------------------- #
# commit metering
# --------------------------------------------------------------------------- #
COMMITS_PER_HOUR = int(os.environ.get("HF_COMMITS_PER_HOUR", "256"))


class CommitBudget:
    """Sliding-window count of commits; `take()` reserves one (check and spend
    in one locked step) and refuses once `limit` fell inside the last
    `window_s`; `give_back()` returns a reservation whose commit did not land.
    Process-local: it keeps one process from spending the whole account
    limit, it cannot see other processes on the same token."""

    def __init__(self, limit: int = COMMITS_PER_HOUR, window_s: float = 3600.0,
                 clock=time.time):
        self.limit, self.window_s, self.clock = limit, window_s, clock
        self._stamps: list[float] = []
        self._lock = threading.Lock()

    def _trim(self, now):
        self._stamps = [t for t in self._stamps if now - t < self.window_s]

    def remaining(self) -> int:
        with self._lock:
            self._trim(self.clock())
            return max(0, self.limit - len(self._stamps))

    def take(self) -> float | None:
        """Reserve one commit: its stamp, or None when the budget is spent."""
        with self._lock:
            now = self.clock()
            self._trim(now)
            if len(self._stamps) >= self.limit:
                return None
            self._stamps.append(now)
            return now

    def give_back(self, stamp: float) -> None:
        """Undo a `take()` whose commit failed."""
        with self._lock:
            try:
                self._stamps.remove(stamp)
            except ValueError:                 # already aged out
                pass


_ACCOUNT_BUDGET = CommitBudget()          # every real-Hub Dataset in this process


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
class Dataset:
    """One dataset repo, reached through an adapter."""

    def __init__(self, repo: str, adapter=None, budget: CommitBudget | None = None):
        self.repo = repo
        if adapter is None:
            adapter = HubAdapter()
            budget = budget or _ACCOUNT_BUDGET
        self.adapter = adapter
        self.budget = budget or CommitBudget()

    @property
    def can_write(self) -> bool:
        """False when commits cannot succeed (no token)."""
        return self.adapter.can_write()

    def url(self, path: str) -> str:
        """Public resolve URL of `path` (what the browser fetches)."""
        return f"https://huggingface.co/datasets/{self.repo}/resolve/main/{path}"

    def local_path(self, path: str, fresh: bool = False) -> Read:
        """A local copy of `path` (value = filesystem path)."""
        try:
            return Read(DATA, self.adapter.download(self.repo, path, fresh=fresh))
        except NotFound:
            return Read(MISSING)
        except Exception as e:
            return _unreachable(e)

    def read_json(self, path: str, kind: type | None = None,
                  fresh: bool = False) -> Read:
        """Parsed JSON. Content that does not parse, or is not a `kind`, is
        no trustworthy answer: UNREACHABLE, never MISSING."""
        return self.read(path, _json_loader(kind), fresh=fresh)

    def read(self, path: str, load, fresh: bool = False) -> Read:
        """`load(local_path)` of `path`. A loader that raises (corrupt file,
        wrong type) makes the read UNREACHABLE, never MISSING."""
        r = self.local_path(path, fresh=fresh)
        if not r.ok:
            return r
        try:
            return Read(DATA, load(r.value))
        except Exception as e:
            return Read(UNREACHABLE, error=f"unreadable {path}: {type(e).__name__}: {e}")

    def cached(self, path: str, ttl_s: float, *, load=None, kind: type | None = None,
               retry_s: float | None = None, fresh: bool = False,
               clock=time.time) -> "Cached":
        """A TTL-cached read of `path` (JSON unless `load` is given) that
        serves the last good copy, marked stale, while the Hub is unreachable."""
        load = load or _json_loader(kind)
        return cached(lambda: self.read(path, load, fresh=fresh), ttl_s,
                      retry_s=retry_s, clock=clock)

    def exists(self, path: str) -> Read:
        """Whether `path` is in the repo, without downloading it (value =
        bool; UNREACHABLE when the Hub gave no answer)."""
        try:
            return Read(DATA, bool(self.adapter.exists(self.repo, path)))
        except Exception as e:
            return _unreachable(e)

    def file_info(self, prefix: str = "") -> Read:
        """{path: (blob id, size)} of the files under `prefix` — the blob id
        changes whenever the file is re-uploaded."""
        try:
            info = self.adapter.file_info(self.repo)
        except Exception as e:
            return _unreachable(e)
        return Read(DATA, {k: v for k, v in info.items() if k.startswith(prefix)})

    def snapshot(self, local_dir: str, patterns: list[str]) -> Read:
        """Download every file matching one of the glob `patterns` (e.g.
        "params/**") into `local_dir`, keeping repo paths (value = local_dir)."""
        try:
            self.adapter.snapshot(self.repo, local_dir, list(patterns))
        except Exception as e:
            return _unreachable(e)
        return Read(DATA, local_dir)

    def create(self, private: bool = True) -> Committed:
        """Create the repo if it does not exist yet. Never raises."""
        if not self.can_write:
            return Committed(False, "no HF token: read-only")
        try:
            self.adapter.create(self.repo, private)
        except Exception as e:
            return Committed(False, f"{type(e).__name__}: {e}")
        return Committed(True)

    def open(self, path: str) -> Read:
        """A binary file object over the remote file, read by range requests
        (for parquet footers and column chunks of big files) — value = the
        file object; the caller closes it."""
        try:
            return Read(DATA, self.adapter.open(self.repo, path))
        except NotFound:
            return Read(MISSING)
        except Exception as e:
            return _unreachable(e)

    def list_files(self, prefix: str = "") -> Read:
        """Sorted repo paths starting with `prefix` (DATA or UNREACHABLE)."""
        try:
            files = self.adapter.list_files(self.repo)
        except Exception as e:
            return _unreachable(e)
        return Read(DATA, sorted(f for f in files if f.startswith(prefix)))

    def commit(self, ops: list, message: str) -> Committed:
        """ONE commit of `ops` (Put / Delete). Never raises."""
        if not self.can_write:
            return Committed(False, "no HF token: read-only")
        stamp = self.budget.take()
        if stamp is None:
            return Committed(False, f"commit budget spent ({self.budget.limit}"
                                    f" per {self.budget.window_s:g} s)")
        try:
            self.adapter.commit(self.repo, list(ops), message)
        except Exception as e:
            self.budget.give_back(stamp)       # only commits that land count
            return Committed(False, f"{type(e).__name__}: {e}")
        return Committed(True)

    def batch(self, label: str, min_interval_s: float = 600.0,
              max_ops: int | None = 500, clock=time.time,
              max_pending: int | None = None) -> "CommitBatch":
        """A queue of writes committed together (see CommitBatch)."""
        return CommitBatch(self, label, min_interval_s, max_ops, clock, max_pending)


def _unreachable(e: Exception) -> Read:
    return Read(UNREACHABLE, error=f"{type(e).__name__}: {e}")


def _json_loader(kind: type | None):
    def load(local: str):
        with open(local, encoding="utf-8") as fp:
            data = json.load(fp)
        if kind is not None and not isinstance(data, kind):
            raise TypeError(f"not a {kind.__name__}")
        return data
    return load


def cached(fetch, ttl_s: float, retry_s: float | None = None,
           clock=time.time) -> "Cached":
    """A TTL cache over any `fetch() -> Read` (a read assembled from several
    files, say) with the same outage policy as Dataset.cached."""
    return Cached(fetch, ttl_s, ttl_s if retry_s is None else retry_s, clock)


class Cached:
    """See Dataset.cached. `get()` re-fetches after `ttl_s`; after an
    unreachable fetch it waits `retry_s` before trying again. A MISSING answer
    drops the cached copy (the file is gone, that is not an outage)."""

    def __init__(self, fetch, ttl_s, retry_s, clock):
        self.fetch, self.ttl_s, self.retry_s, self.clock = fetch, ttl_s, retry_s, clock
        self._serve: Read | None = None       # the answer until the next Hub read
        self._next = 0.0                      # clock time of the next Hub read
        self._lock = threading.Lock()

    def get(self, force: bool = False) -> Read:
        with self._lock:
            now = self.clock()
            if not force and self._serve is not None and now < self._next:
                return self._serve
            r = self.fetch()
            if not r.unreachable:
                self._serve, self._next = r, now + self.ttl_s
            else:
                if self._serve is not None and self._serve.ok:
                    r = Read(DATA, self._serve.value, stale=True)
                self._serve, self._next = r, now + self.retry_s
            return r

    def put(self, value) -> None:
        """Serve `value` (just written by this process) until the TTL runs out."""
        with self._lock:
            self._serve, self._next = Read(DATA, value), self.clock() + self.ttl_s


class CommitBatch:
    """Many small writes, few commits. Entries queue up under a key — one op
    keyed by its path, or several ops (one outlet's files, say) under a given
    key; a later entry of the same key replaces the earlier one whole — and
    `flush()` commits them all at once when `min_interval_s` passed since the
    batch started or last landed, or `max_ops` entries are queued, or `force`.
    A failed commit leaves the queue for the next flush, cut to the oldest
    `max_pending` entries when that is set (RAM stays bounded while the Hub
    is down); `error` says why it failed."""

    def __init__(self, ds: Dataset, label: str, min_interval_s: float,
                 max_ops: int | None, clock, max_pending: int | None = None):
        self.ds, self.label = ds, label
        self.min_interval_s, self.max_ops, self.clock = min_interval_s, max_ops, clock
        self.max_pending = max_pending
        self.error: str | None = None
        self._entries: dict[str, list] = {}
        self._last = clock()
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        return len(self._entries)

    @property
    def nbytes(self) -> int:
        """Bytes the queued Puts would upload."""
        n = 0
        for ops in list(self._entries.values()):
            for op in ops:
                if isinstance(op, Put):
                    body = op.payload()
                    n += len(body) if isinstance(body, bytes) else os.path.getsize(body)
        return n

    def keys(self) -> list[str]:
        """Queued entry keys, oldest first."""
        with self._lock:
            return list(self._entries)

    def add(self, op, key: str | None = None) -> None:
        """Queue one op (keyed by its path) or a list of ops under `key`."""
        ops = list(op) if isinstance(op, (list, tuple)) else [op]
        if key is None:
            if len(ops) != 1:
                raise ValueError("several ops need a key")
            key = ops[0].path
        with self._lock:
            self._entries.pop(key, None)          # re-insert: keeps add order
            self._entries[key] = ops

    def flush(self, force: bool = False) -> Committed | None:
        """None when nothing was due; otherwise the commit's outcome."""
        return self._flush(force)[0]

    def flush_keys(self, force: bool = False) -> list[str]:
        """Like flush; the keys that landed ([] when nothing did)."""
        res, keys = self._flush(force)
        return keys if res else []

    def _flush(self, force: bool):
        with self._lock:
            if not self._entries:
                return None, []
            due = (force
                   or (self.max_ops is not None and len(self._entries) >= self.max_ops)
                   or self.clock() - self._last >= self.min_interval_s)
            if not due:
                return None, []
            keys = list(self._entries)
            ops = [op for k in keys for op in self._entries[k]]
            res = self.ds.commit(ops, f"{self.label} x{len(keys)}")
            if res:
                self.error = None
                self._entries.clear()
                self._last = self.clock()
            else:
                self.error = res.error
                if self.max_pending is not None:
                    for k in keys[self.max_pending:]:
                        del self._entries[k]
            return res, keys


def data() -> Dataset:
    """The main dataset repo on the real Hub."""
    return Dataset(data_repo())


def feedback() -> Dataset:
    """The feedback dataset repo on the real Hub."""
    return Dataset(feedback_repo())


def fleet() -> Dataset:
    """The fleet dataset repo on the real Hub."""
    return Dataset(fleet_repo())


def state() -> Dataset:
    """The state (cache mirror) dataset repo on the real Hub."""
    return Dataset(state_repo())

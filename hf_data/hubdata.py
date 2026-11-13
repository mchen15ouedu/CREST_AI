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
        try:
            return list(self._api().list_repo_files(repo, repo_type="dataset"))
        except Exception as e:
            raise Unreachable(f"{type(e).__name__}: {e}") from e

    def exists(self, repo: str, path: str) -> bool:
        try:
            return bool(self._api().get_paths_info(repo, [path], repo_type="dataset"))
        except Exception as e:
            raise Unreachable(f"{type(e).__name__}: {e}") from e

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
    """In-memory dataset repo(s) for tests: {path: bytes} per repo id (one
    shared file map unless a repo is given its own). Set `outage = True` to
    make every call unreachable."""

    def __init__(self, files: dict | None = None):
        self.files: dict[str, bytes] = {k: _as_bytes(v) for k, v in (files or {}).items()}
        self.outage = False
        self.writable = True
        self.commits: list[dict] = []          # {"message", "ops"} per commit
        self._tmp = None

    def can_write(self) -> bool:
        return self.writable

    def list_files(self, repo: str) -> list[str]:
        self._check()
        return sorted(self.files)

    def exists(self, repo: str, path: str) -> bool:
        self._check()
        return path in self.files

    def commit(self, repo: str, ops: list, message: str) -> None:
        self._check()
        files = dict(self.files)               # all or nothing, like the Hub
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
        self.files = files
        self.commits.append({"message": message, "ops": list(ops)})

    def _check(self):
        if self.outage:
            raise Unreachable("memory hub: outage")

    def download(self, repo: str, path: str, *, fresh: bool) -> str:
        self._check()
        if path not in self.files:
            raise NotFound(path)
        if self._tmp is None:
            self._tmp = tempfile.mkdtemp(prefix="memhub_")
        p = os.path.join(self._tmp, path.replace("/", "__"))
        with open(p, "wb") as fp:
            fp.write(self.files[path])
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
    """Sliding-window count of commits that landed; `take()` refuses once
    `limit` commits fell inside the last `window_s`. Process-local: it keeps
    one process from spending the whole account limit, it cannot see other
    processes on the same token."""

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

    def spend(self) -> None:
        with self._lock:
            self._stamps.append(self.clock())


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
            return Read(UNREACHABLE, error=f"{type(e).__name__}: {e}")

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
            return Read(UNREACHABLE, error=f"{type(e).__name__}: {e}")

    def list_files(self, prefix: str = "") -> Read:
        """Sorted repo paths starting with `prefix` (DATA or UNREACHABLE)."""
        try:
            files = self.adapter.list_files(self.repo)
        except Exception as e:
            return Read(UNREACHABLE, error=f"{type(e).__name__}: {e}")
        return Read(DATA, sorted(f for f in files if f.startswith(prefix)))

    def commit(self, ops: list, message: str) -> Committed:
        """ONE commit of `ops` (Put / Delete). Never raises."""
        if not self.can_write:
            return Committed(False, "no HF token: read-only")
        if self.budget.remaining() <= 0:
            return Committed(False, f"commit budget spent ({self.budget.limit}"
                                    f" per {self.budget.window_s:g} s)")
        try:
            self.adapter.commit(self.repo, list(ops), message)
        except Exception as e:
            return Committed(False, f"{type(e).__name__}: {e}")
        self.budget.spend()
        return Committed(True)

    def batch(self, label: str, min_interval_s: float = 600.0,
              max_ops: int = 500, clock=time.time) -> "CommitBatch":
        """A queue of writes committed together (see CommitBatch)."""
        return CommitBatch(self, label, min_interval_s, max_ops, clock)


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


class CommitBatch:
    """Many small writes, few commits: ops queue up (a later Put/Delete of
    the same path replaces the earlier one) and `flush()` commits them all at
    once when `min_interval_s` passed since the batch started or last landed,
    or `max_ops` are queued, or `force`. A failed commit leaves the queue for
    the next flush."""

    def __init__(self, ds: Dataset, label: str, min_interval_s: float,
                 max_ops: int, clock):
        self.ds, self.label = ds, label
        self.min_interval_s, self.max_ops, self.clock = min_interval_s, max_ops, clock
        self._ops: dict[str, Any] = {}
        self._last = clock()
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        return len(self._ops)

    def add(self, op) -> None:
        with self._lock:
            self._ops.pop(op.path, None)          # re-insert: keeps add order
            self._ops[op.path] = op

    def flush(self, force: bool = False) -> Committed | None:
        """None when nothing was due; otherwise the commit's outcome."""
        with self._lock:
            if not self._ops:
                return None
            due = (force or len(self._ops) >= self.max_ops
                   or self.clock() - self._last >= self.min_interval_s)
            if not due:
                return None
            ops = list(self._ops.values())
            res = self.ds.commit(ops, f"{self.label} x{len(ops)}")
            if res:
                self._ops.clear()
                self._last = self.clock()
            return res


def data() -> Dataset:
    """The main dataset repo on the real Hub."""
    return Dataset(data_repo())


def feedback() -> Dataset:
    """The feedback dataset repo on the real Hub."""
    return Dataset(feedback_repo())


def fleet() -> Dataset:
    """The fleet dataset repo on the real Hub."""
    return Dataset(fleet_repo())

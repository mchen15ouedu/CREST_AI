"""The fleet store: CREST_FLEET_REPO's layout, its writers and its lazy fetch.

Two writers publish into the private dataset CREST_FLEET_REPO, both through
this module (publish_run / pack + publish / batch); the dashboard reads it
through ensure_local / has_remote. This module is the only place that knows
the layout:

  results/<gid>_<rows_model>.json    statecache record (rows + state_times + variant)
  states/<gid>_<state_model>.pqf     f16 state bundle (statebundle format)
  vp/results/..., vp/states/...      a pour point's HINDCAST (same names)

<gid> is the outlet id zero-padded to 8 (key()). rows_model = state_model
(+ "-<tag>" for a non-a-priori parameter set, see pipeline.config_tag); the
two are the same name for the a-priori runs that make up nearly all of the
fleet.

  HINDCAST    fleet/fleet_run.py: years of quick-run rows + 10-day states per
              outlet. Gauges under the bare prefix; pour points (V...) under
              vp/. done_ids() lists the outlets whose hindcast is complete.
  CHECKPOINT  ungauged_space/runner.py: a pour point's rolling 10-day
              checkpoint of the hourly routed nowcast, under the bare prefix,
              re-uploaded as it advances (batched: one commit per flush).
              checkpoint_keys() lists them.

The vp/ question — decided: a pour point's hindcast is NOT fetchable through
ensure_local / has_remote; they read the bare prefix only, as before. vp/
exists so that a 5-year hindcast never overwrites the rolling checkpoint of
the same key. The bare V... files are what routednow.compute fetches for its
cold-boot warm start (the newest checkpoint, close to t0); serving the vp/
hindcast there instead would warm-start from a state up to years old, and
merging its rows would mix two run configurations. The gauge pipeline never
asks the fleet for a virtual point (pipeline.py skips g["virtual"]), and
nothing else in the code base reads vp/. So vp/ is an archive, written and
listed (done_ids) but not served; serving it would need its own reader.

When a run touches a (gauge, model) this MERGES both into the local cache,
after which the normal plan()/warm-start machinery serves the user instantly:
fleet rows join a local record of the same run configuration (local rows win
where both exist), fleet state grids fill every time the local state dir
lacks (local grids are never overwritten). It used to fetch only into an
EMPTY cache — one earlier run of a gauge, or a record restored from
CREST_state, was enough to keep the fleet's 5 years out for good.

Deliberately a separate repo from CREST_state: persist.backup mirrors local
deletions to CREST_state, so fleet data there would be erased by the janitor.
Every read failure is a silent no-op — the run just proceeds uncached; writes
never raise (they return the commit's outcome). A real 404 is remembered for TTL_S ("the fleet has nothing here"); a Hub that
could not be reached is not, so the next run probes again.
"""
from __future__ import annotations

import glob
import json
import os
import threading
import time

from dataclasses import dataclass

from hf_data import hubdata, statecache
from hf_data.atomicfile import atomic_path

DATASET = hubdata.fleet()
# re-probe a key this often: the ungauged fleet re-uploads checkpoints, the
# janitor's LRU cap can evict fetched grids, and a gauge may finish later
TTL_S = float(os.environ.get("CREST_FLEET_TTL_S", "21600"))
MARK = ".fleet.json"

_seen: dict[str, float] = {}
_remote: dict[str, tuple[float, bool]] = {}
_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}


# --------------------------------------------------------------------------- #
# layout: the one place that knows where fleet outputs live in the repo
# --------------------------------------------------------------------------- #
HINDCAST = "hindcast"        # fleet/fleet_run.py: the multi-year pre-run of an outlet
CHECKPOINT = "checkpoint"    # ungauged_space: a pour point's rolling 10-day checkpoint
VP_PREFIX = "vp/"            # where a pour point's HINDCAST lives (see module doc)


def key(gauge, model) -> str:
    """<8-digit zero-padded id>_<model>: the name of a record or a bundle."""
    return f"{str(gauge).zfill(8)}_{model}"


def is_pour_point(gauge) -> bool:
    return str(gauge).upper().startswith("V")


def prefix(gauge, kind: str = CHECKPOINT) -> str:
    """'vp/' for a pour point's hindcast, '' for everything else."""
    return VP_PREFIX if kind == HINDCAST and is_pour_point(gauge) else ""


def results_path(gauge, model, kind: str = CHECKPOINT) -> str:
    return f"{prefix(gauge, kind)}results/{key(gauge, model)}.json"


def states_path(gauge, model, kind: str = CHECKPOINT) -> str:
    return f"{prefix(gauge, kind)}states/{key(gauge, model)}.pqf"


def _sig(path: str) -> str:
    """Identity of a downloaded fleet file (blob name + size)."""
    try:
        return f"{os.path.basename(os.path.realpath(path))}:{os.path.getsize(path)}"
    except OSError:
        return ""


def _merge_rows(gauge, rows_model, path: str, variant: str | None) -> int:
    """Fleet record -> local record. Returns the number of fleet rows that
    became newly available locally (0: nothing to do / kept the local one)."""
    with open(path) as fh:
        fl = json.load(fh)
    frows = fl.get("rows") or []
    if not frows:
        return 0
    sig = _sig(path)
    loc = statecache.load_record(gauge, rows_model)
    if loc and loc.get("fleet_sig") == sig:
        return 0                                   # this fleet file is already in
    states = sorted(set((loc or {}).get("state_times", []))
                    | set(fl.get("state_times", [])))
    if not loc or not loc.get("rows"):
        rows, var = frows, fl.get("variant")
    elif loc.get("variant") == fl.get("variant"):
        by_time = {r["time"]: r for r in frows}
        n0 = len(by_time)
        by_time.update({r["time"]: r for r in loc["rows"]})   # local rows win
        rows, var = sorted(by_time.values(), key=lambda r: r["time"]), loc.get("variant")
        if len(rows) == len(loc["rows"]) and n0 <= len(loc["rows"]):
            frows = []                             # fleet added no new hour
    elif variant is not None and fl.get("variant") == variant:
        # rows of two run configurations never mix: the run being set up
        # asks for the fleet's, so the fleet's replace the local ones
        rows, var = frows, fl.get("variant")
    else:
        return 0                                   # keep local; re-judged next probe
    rec = {"gauge": str(gauge).zfill(8), "model": rows_model, "rows": rows,
           "window": [rows[0]["time"], rows[-1]["time"]],
           "state_times": states, "variant": var, "fleet_sig": sig}
    dst = statecache.results_path(gauge, rows_model)
    with atomic_path(dst) as tmp, open(tmp, "w") as fh:
        json.dump(rec, fh)
    return len(frows)


def _merge_states(gauge, state_model, path: str) -> int:
    """Fleet bundle -> loose grids for every time the local dir lacks."""
    sdir = statecache.state_dir(gauge, state_model)
    mark = os.path.join(sdir, MARK)
    sig = _sig(path)
    n_tif = len(glob.glob(os.path.join(sdir, "*.tif")))
    try:
        with open(mark) as fh:
            m = json.load(fh)
        if m.get("sig") == sig and n_tif >= int(m.get("n", 0)):
            return 0                               # unpacked, nothing evicted since
    except Exception:
        pass
    from hf_data import statebundle
    n = statebundle.unpack(path, sdir)             # overwrite=False: local grids win
    try:
        with open(mark, "w") as fh:
            json.dump({"sig": sig, "n": len(glob.glob(os.path.join(sdir, "*.tif")))}, fh)
    except OSError:
        pass
    return n


def ensure_local(gauge, state_model, rows_model: str | None = None,
                 variant: str | None = None) -> str | None:
    """Make the fleet's rows + state grids for this gauge available locally.

    state_model: key of the EF5 state grids (model + scheme).
    rows_model:  key of the row record (defaults to state_model).
    variant:     run-configuration fingerprint of the run being set up —
                 decides which rows stay when local and fleet records disagree.
    Returns "fetched" (fleet data became newly available now), "cached" (the
    fleet's data is already in the local cache) or None (the fleet has
    nothing under these keys, or it could not be reached)."""
    if not hubdata.token():                        # private repo: no token, no fleet
        return None
    rows_model = rows_model or state_model
    k = key(gauge, state_model) + "|" + rows_model
    now = time.time()
    with _lock:
        klock = _key_locks.setdefault(k, threading.Lock())
    with klock:                    # two runs of one gauge: one fetch, one wait
        last = _seen.get(k)
        if last is not None and now - last < TTL_S:    # probed recently
            if not _remote.get(k, (0, False))[1]:
                return None                            # the fleet has nothing here
            rec = statecache.load_record(gauge, rows_model)
            if os.path.exists(os.path.join(
                    statecache.state_dir(gauge, state_model), MARK)) \
                    or (rec and rec.get("fleet_sig")):
                return "cached"
            # fetched earlier but evicted locally since -> merge again
        got = found = 0
        reached = True
        for path, merge in (
                (results_path(gauge, rows_model),
                 lambda p: _merge_rows(gauge, rows_model, p, variant)),
                (states_path(gauge, state_model),
                 lambda p: _merge_states(gauge, state_model, p))):
            r = DATASET.local_path(path)
            if r.unreachable:
                reached = False
            if not r.ok:
                continue
            found += 1
            try:
                got += 1 if merge(r.value) else 0
            except Exception:
                pass
        if reached:                                # an outage is not "nothing there"
            _seen[k] = now
            _remote[k] = (now, bool(found))
        if got:
            return "fetched"
        return "cached" if found else None


def has_remote(gauge, model) -> bool:
    """Does the fleet repo hold state grids under this key? (one cheap HEAD,
    remembered for TTL_S) — used to tell a full-basin run that the fleet's
    pre-run of this gauge exists under the speed scheme."""
    if not hubdata.token():
        return False
    k = "has|" + key(gauge, model)
    now = time.time()
    hit = _remote.get(k)
    if hit and now - hit[0] < TTL_S:
        return hit[1]
    r = DATASET.exists(states_path(gauge, model))
    if not r.ok:
        return False                               # unreachable: not remembered
    _remote[k] = (now, r.value)
    return r.value


# --------------------------------------------------------------------------- #
# write side
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Publication:
    """One outlet's packed outputs, ready to commit: the state bundle (bytes,
    or a local file path) and the row record (bytes, a path, or None to leave
    the repo's record as it is). Plain data: a worker process can build it and
    hand it to its parent."""
    gauge: str
    rows_model: str
    state_model: str
    bundle: bytes | str
    record: bytes | str | None = None
    kind: str = CHECKPOINT

    @property
    def key(self) -> str:
        return key(self.gauge, self.rows_model)

    @property
    def nbytes(self) -> int:
        return sum(len(x) if isinstance(x, bytes) else os.path.getsize(x)
                   for x in (self.bundle, self.record) if x is not None)

    def ops(self) -> list:
        ops = [hubdata.Put(states_path(self.gauge, self.state_model, self.kind),
                           self.bundle)]
        if self.record is not None:
            ops.append(hubdata.Put(results_path(self.gauge, self.rows_model, self.kind),
                                   self.record))
        return ops


@dataclass(frozen=True)
class Published:
    """publish_run's outcome; truthy on success."""
    ok: bool
    key: str | None = None
    bytes: int = 0
    error: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def pack(gauge, state_model, rows_model: str | None = None, *, kind: str = CHECKPOINT,
         state_dir: str | None = None, record_path: str | None = None,
         out_path: str | None = None) -> Publication | None:
    """Pack an outlet's state dir (default: the local cache's) + row record
    into a Publication; None when the dir holds no state grids. With out_path
    the bundle is written there (never held in RAM) and the record is passed
    by path; without, both are bytes. A missing record publishes the bundle
    alone."""
    from hf_data import statebundle
    rows_model = rows_model or state_model
    sdir = state_dir or statecache.state_dir(gauge, state_model)
    bundle = statebundle.pack_dir(sdir, out_path=out_path)
    if bundle is None:
        return None
    rec_path = record_path or statecache.results_path(gauge, rows_model)
    record = None
    if os.path.exists(rec_path):
        if out_path is None:
            with open(rec_path, "rb") as fh:
                record = fh.read()
        else:
            record = rec_path
    return Publication(str(gauge), rows_model, state_model, bundle, record, kind)


def publish(pubs, message: str | None = None) -> hubdata.Committed:
    """ONE commit of one or more Publications. Never raises."""
    pubs = [pubs] if isinstance(pubs, Publication) else list(pubs)
    ops = [op for p in pubs for op in p.ops()]
    if message is None:
        message = (f"fleet {prefix(pubs[0].gauge, pubs[0].kind)}{pubs[0].key}"
                   if len(pubs) == 1 else f"fleet x{len(pubs)}")
    return DATASET.commit(ops, message)


def publish_run(res, kind: str = HINDCAST) -> Published:
    """Publish a finished run (a GaugeRunResult): its state bundle and row
    record in one commit. The bundle is packed to a temp file, so a big
    basin's compressed bundle never sits in RAM."""
    import tempfile
    rk = res.record_key
    if not res.record_path or not os.path.exists(res.record_path):
        return Published(False, rk, error="no result record")
    fd, bpath = tempfile.mkstemp(prefix=f"fleet_{res.state_key}_", suffix=".pqf")
    os.close(fd)
    try:
        pub = pack(res.gauge_id, res.state_model, res.cache_model, kind=kind,
                   state_dir=res.state_dir, record_path=res.record_path,
                   out_path=bpath)
        if pub is None:
            return Published(False, rk, error="no state grids")
        size = os.path.getsize(bpath)
        c = publish(pub)
        return Published(c.ok, pub.key, size, c.error)
    finally:
        try:
            os.remove(bpath)
        except OSError:
            pass


def done_ids() -> hubdata.Read:
    """Ids of the outlets whose HINDCAST is fully published (record + bundle):
    a set (DATA), or UNREACHABLE — never an empty set for an outage. By id,
    not by key: rows are keyed by model+scheme+parameter tag, state grids by
    model+scheme only, so the two names differ for a calibrated gauge. A pour
    point's bare-prefix files are its rolling checkpoint, not a hindcast."""
    r = DATASET.list_files()
    if not r.ok:
        return r
    done = set()
    for pre in ("", VP_PREFIX):
        def ids(sub, pre=pre):
            head = pre + sub
            return {f[len(head):].split("_", 1)[0] for f in r.value if f.startswith(head)}
        both = ids("results/") & ids("states/")
        done |= {g for g in both if is_pour_point(g) == (pre == VP_PREFIX)}
    return hubdata.Read(hubdata.DATA, done)


def checkpoint_keys() -> hubdata.Read:
    """Keys of the pour-point checkpoints in the repo (bare states/V*.pqf),
    as a set (DATA), or UNREACHABLE."""
    r = DATASET.list_files("states/V")
    if not r.ok:
        return r
    return hubdata.Read(hubdata.DATA, {f[len("states/"):-len(".pqf")]
                                       for f in r.value if f.endswith(".pqf")})


class Batch:
    """Publications queued for ONE commit per flush: per-outlet commits blow
    the account's 256 commits/hour (and 429-starve every other publish). A
    later publication of the same outlet replaces the queued one; a failed
    flush keeps the queue for the next one, cut to the oldest `max_pending`
    so RAM stays bounded while the Hub is down (`error` says why)."""

    def __init__(self, label: str, max_pending: int = 400):
        self.label, self.max_pending = label, max_pending
        self.error: str | None = None
        self._q: dict[str, Publication] = {}
        self._lock = threading.Lock()

    @property
    def pending(self) -> int:
        return len(self._q)

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes for p in list(self._q.values()))

    def add(self, pub: Publication) -> None:
        with self._lock:
            self._q.pop(pub.key, None)
            self._q[pub.key] = pub

    def flush(self) -> list[str]:
        """Commit everything queued; the keys that landed ([] on failure or
        when nothing was queued)."""
        with self._lock:
            if not self._q:
                return []
            pubs = list(self._q.values())
            c = publish(pubs, f"{self.label} x{len(pubs)}")
            if not c:
                self.error = c.error
                for k in list(self._q)[self.max_pending:]:
                    del self._q[k]
                return []
            self.error = None
            self._q.clear()
            return [p.key for p in pubs]


def batch(label: str, max_pending: int = 400) -> Batch:
    """A Batch publishing to the fleet repo under commit message `label`."""
    return Batch(label, max_pending)

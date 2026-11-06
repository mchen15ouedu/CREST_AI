"""The hourly 2-D event tick as a long-running process: the Jetstream2 VM's
crest-events.service (fleet/jetstream2/; PI 2026-11-05: "we have jetstream2
VM as the workhorse"). It replaces the HF CREST_runner Space's
/api/event_tick, whose EVENT_TICK goes to 0 at the cutover: ticks never run
on both hosts.

What runs: eventsim.hourly_tick() of THIS checkout's hf_data - the call the
runner's /api/event_tick makes, with its own lock and stall watchdog (basin
outlines: EF5 extent and the capacity preflight from the basin outline, never
a box). server.py is never imported: its import starts persist backups, the
cache janitor and the merger supervisor, which the HF Spaces already run.
The janitor's cache bounds (datamgr.cleanup + compact_results) run here
between ticks instead, with the event_* EF5 work dirs eventsim leaves behind
(the Spaces lose them with /tmp; this disk keeps them).

When: a tick starts when the gauged nowcast's issue time
(nowcaststore.issue_t0()) advances - at most once per issue - and at least
EVENT_TICK_MAX_GAP_MIN (70) minutes after the previous tick started, so a
stalled nowcast feed still ends receded episodes and runs retention. One
tick at a time (a worker thread, plus eventsim's own _tick_lock); a lock file
in EVENT_TICK_STATE_DIR keeps a second daemon off the host. The last issue
and tick start survive restarts (event_tick_daemon.json there), so a restart
does not tick the same issue twice. A tick that falls due while the previous
one still runs is not started but noted as an attempt
(eventsim.note_tick_attempt, what the HF runner's busy branch did): a long
tick reads as busy in /api/health, not as a dead schedule.

Bounds: a tick busy longer than EVENT_TICK_MAX_BUSY_H (8) is wedged: the
daemon logs every thread's stack, publishes "wedged" (bounded) and exits 3;
systemd restarts it (KillMode=mixed reaps a leftover EF5 child, the saved
schedule keeps the issue). EF5's own run-time cap (CREST_RUN_TIMEOUT_S,
3 h in events.env) binds first on a runaway EF5 run - the tick records it
and moves on; this bound is for what no inner watchdog sees (a hung read, a
kill that did not take). The main loop also pings systemd's watchdog
(WatchdogSec=) on every pass.

Why a daemon and not a timer: eventsim keeps state in process memory across
ticks - the native-capacity suppression of oversize basins (retry after
EVENT_PREP_COOLDOWN_H), the CPU-timeout cooldown, conusstate's 24-h refresh
gate, the nowcast / threshold / parameter caches. A timer would start every
tick cold and re-run every deferred basin's preflight hourly.

Status: events/tick_status.json in CREST_data (hf_data/tickstatus.py) for the
Spaces' /api/health and /api/runner_status: one small commit at start, at
each tick start and end, every EVENT_TICK_STATUS_BEAT_MIN (15) minutes while
a tick runs (and when it stalls), hourly while idle and at stop - coalesced
(Publisher), since the 256 commits/h per account are shared with the merger,
the GPU worker and the fleet. A failed commit backs off and never stops a tick.

Guard: before each tick, EVENT_TICK_GUARD_URL (the CREST_runner Space) is
asked for /api/runner_status. While that Space runs its own tick (tick
running, or an attempt within 2 h, and not reading ours: tick_source
"remote") no tick starts here. A block is sticky: after one, ticks resume
only on a clear read - a Space that turns unreachable (restarting, say)
keeps the block, and "unreachable = clear" holds only while the guard never
blocked. The block survives restarts (the state file).

Stop: SIGTERM/SIGINT between ticks exits at once, and no tick starts after
the signal (a pass in flight checks again after its guard read). During a
tick the daemon waits up to EVENT_TICK_STOP_GRACE_S (600, counted from the
signal) for it - and for the finalize threads it dispatched - then publishes
"stopped" (an attempt and one retry, <= 60 s) and exits (a second signal
skips the rest of the grace). A grace that would not fit the unit's
TimeoutStopSec (EVENT_TICK_TIMEOUT_STOP_S) with that commit is cut. An
abandoned tick loses only its EF5 work: every Hub commit is atomic and the
next tick redoes the visit. 2-D solves run on the GPU worker and are never
touched.

Every Hub call has finite timeouts (scripts/merge_episodes.py
bound_hub_timeouts, huggingface_hub 0.x and 1.x).

    python scripts/event_tick_daemon.py            # the service (HF_TOKEN in env)
    python scripts/event_tick_daemon.py --check    # imports, EF5, reads; no tick, no write
HF_TOKEN (env, never printed): write on vincewin/CREST_data (+ the reads and
the CREST_fleet mosaic index named in fleet/jetstream2/README.md).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hf_data import eventsim, tickstatus  # noqa: E402

_TS = "%Y-%m-%dT%H:%M:%SZ"
MAX_GAP_MIN = float(os.environ.get("EVENT_TICK_MAX_GAP_MIN", "70"))
POLL_S = float(os.environ.get("EVENT_TICK_POLL_S", "60"))
STOP_GRACE_S = float(os.environ.get("EVENT_TICK_STOP_GRACE_S", "600"))
# a tick busy this long is wedged: status "wedged", exit, systemd restarts (0: no bound)
MAX_BUSY_H = float(os.environ.get("EVENT_TICK_MAX_BUSY_H", "8"))
# the unit's TimeoutStopSec (crest-events.service sets both): grace + the
# final commit must fit, or systemd SIGKILLs the daemon before "stopped"
TIMEOUT_STOP_S = float(os.environ.get("EVENT_TICK_TIMEOUT_STOP_S", "720"))
BEAT_MIN = float(os.environ.get("EVENT_TICK_STATUS_BEAT_MIN", "15"))
IDLE_BEAT_MIN = 60.0          # status refresh while no tick runs (none for this long)
GUARD_URL = os.environ.get("EVENT_TICK_GUARD_URL", "").strip().rstrip("/")
GUARD_RECENT_H = 2.0          # an HF tick attempt this recent means the HF tick is on
LOOP_S = 5.0                  # main loop granularity (status pump, stall check)
COALESCE_S = 20.0             # a status request waits this long for more to join it
MIN_COMMIT_S = 60.0           # and never commits sooner than this after the last one
BACKOFF_MIN_S, BACKOFF_MAX_S = 60.0, 900.0
FINAL_PUBLISH_S = 60.0        # the last commit ("stopped", "wedged"): attempt + one retry
FINAL_RETRY_PAUSE_S = 5.0     # between the two
STOP_SLACK_S = 60.0           # TimeoutStopSec beyond grace + final commit: the exit itself
JANITOR_S = 3600.0            # cache bounds between ticks, at most this often
EXIT_WEDGED = 3               # exit status after a wedged tick (Restart=always restarts)


def _iso(t: float | None) -> str | None:
    return None if t is None else dt.datetime.fromtimestamp(
        t, dt.timezone.utc).strftime(_TS)


def _epoch(s) -> float | None:
    try:
        return dt.datetime.strptime(str(s or ""), _TS).replace(
            tzinfo=dt.timezone.utc).timestamp()
    except ValueError:
        return None


def _issue_str(t) -> str | None:
    return None if t is None else t.strftime("%Y-%m-%dT%H:%MZ")


def _issue_dt(s):
    try:
        return dt.datetime.strptime(str(s or ""), "%Y-%m-%dT%H:%MZ")
    except ValueError:
        return None


def _log(s: str) -> None:
    print(f"{_iso(time.time())} event-tick-daemon: {s}", flush=True)


def _err(exc: BaseException) -> str:
    code = getattr(getattr(exc, "response", None), "status_code", None)
    return type(exc).__name__ + (f" (HTTP {code})" if code else "")


def _why_due(now: float, issue, last_issue, last_at, max_gap_s: float) -> str | None:
    """Why a tick is due at `now` after one at `last_at` for `last_issue`."""
    if last_at is None:
        return "start"
    if issue is not None and (last_issue is None or issue > last_issue):
        return "issue"
    if now - last_at >= max_gap_s:
        return "max_gap"
    return None


def sd_notify(msg: str) -> bool:
    """One datagram to systemd's NOTIFY_SOCKET (WATCHDOG=1 for WatchdogSec=);
    a no-op outside systemd. Never raises."""
    addr = os.environ.get("NOTIFY_SOCKET", "")
    if not addr:
        return False
    if addr[0] == "@":                         # abstract namespace
        addr = "\0" + addr[1:]
    try:
        import socket
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
        return True
    except (OSError, AttributeError):
        return False


def dump_stacks() -> None:
    """Every thread's stack to stderr (the journal). faulthandler prints code
    paths only, never frame locals. Never raises."""
    try:
        import faulthandler
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
    except Exception:
        pass


def _exit_now(rc: int) -> None:
    # an abandoned tick's threads (EF5 waits, thread pools) must not hold the
    # exit; systemd's KillMode=mixed then reaps any EF5 child it left
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def stop_grace(grace_s: float = STOP_GRACE_S, timeout_stop_s: float = TIMEOUT_STOP_S,
               log=_log) -> float:
    """The tick's stop grace, cut so that grace + the final status commit (an
    attempt and one retry, FINAL_PUBLISH_S) + STOP_SLACK_S fit the unit's
    TimeoutStopSec: past it systemd SIGKILLs the daemon before its "stopped"
    commit, and the Spaces go on reading a live tick host for 2 h."""
    cap = max(0.0, float(timeout_stop_s) - FINAL_PUBLISH_S - STOP_SLACK_S)
    if grace_s > cap:
        log(f"stop grace {grace_s:g} s cut to {cap:g} s: TimeoutStopSec {timeout_stop_s:g} s "
            f"must also hold the final status commit ({FINAL_PUBLISH_S:g} s) and "
            f"{STOP_SLACK_S:g} s for the exit")
        return cap
    return float(grace_s)


def bounds_line(max_busy_h: float = MAX_BUSY_H) -> str:
    """The tick's time bounds, for the startup log and --check. EF5's own
    caps (hf_data/runner.py) should bind before EVENT_TICK_MAX_BUSY_H: one
    event visit runs a warm-up and a main EF5, each up to the run cap."""
    from hf_data import runner
    cap_h = runner.RUN_TIMEOUT_S / 3600
    line = (f"bounds: EF5 run cap {cap_h:g} h (CREST_RUN_TIMEOUT_S), EF5 silent "
            f"{runner.STALL_TIMEOUT_S / 60:g} min (CREST_STALL_TIMEOUT_S); tick "
            + (f"{max_busy_h:g} h (EVENT_TICK_MAX_BUSY_H)" if max_busy_h > 0
               else "unbounded (EVENT_TICK_MAX_BUSY_H=0)"))
    if max_busy_h > 0 and 2 * cap_h >= max_busy_h:
        line += (" - WARNING: a warm-up and a main EF5 at the run cap outlast the tick "
                 "bound, which would then restart the daemon instead; set "
                 "CREST_RUN_TIMEOUT_S below half of it (events.env: 10800)")
    return line


class StopEvent(threading.Event):
    """The stop request. set() records when it came (monotonic): the stop
    grace counts from the signal, not from the end of the main-loop pass it
    interrupted."""

    def __init__(self):
        super().__init__()
        self.at = None

    def set(self):
        if self.at is None:
            self.at = time.monotonic()
        super().set()


class Publisher:
    """Coalesced status commits. request(reason) marks a reason; pump() makes
    ONE commit of the snapshot built at commit time once the burst settled
    (coalesce_s after its first request) and min_commit_s after the previous
    commit - so the daemon's start and its first tick start, or a short
    tick's start and end, are one commit. A failure keeps the reasons and
    backs off (1 min doubling to 15 min); nothing raises. request() never
    waits for a commit in flight (the stop's bounded pumps may leave one
    hanging)."""

    def __init__(self, build, upload, *, clock=time.time, coalesce_s=COALESCE_S,
                 min_commit_s=MIN_COMMIT_S, backoff=(BACKOFF_MIN_S, BACKOFF_MAX_S),
                 log=_log, enabled=True):
        self.build, self.upload, self.clock, self.log = build, upload, clock, log
        self.coalesce_s, self.min_commit_s = float(coalesce_s), float(min_commit_s)
        self.backoff, self.enabled = backoff, enabled
        self.pending: list[str] = []
        self.first_at = None
        self.last_commit = None
        self.retry_at = 0.0
        self.fails = self.commits = 0
        self.last_error = None
        self._mu = threading.Lock()            # pending reasons, counters
        self._commit = threading.Lock()        # one commit at a time

    def request(self, reason: str) -> None:
        with self._mu:
            if reason not in self.pending:
                self.pending.append(reason)
            if self.first_at is None:
                self.first_at = self.clock()

    def due(self, now: float) -> bool:
        return bool(self.pending) and now >= self.first_at + self.coalesce_s \
            and now >= self.retry_at \
            and (self.last_commit is None or now >= self.last_commit + self.min_commit_s)

    def pump(self, now: float | None = None, force: bool = False) -> bool:
        """Commit when due (force: now, whatever the coalescing and backoff).
        True when a commit was made."""
        with self._commit:
            with self._mu:
                now = self.clock() if now is None else now
                if not self.pending or not (force or self.due(now)):
                    return False
                reasons = list(self.pending)
            try:
                doc = self.build(reasons)
                if self.enabled:
                    self.upload(doc)
            except Exception as exc:
                with self._mu:
                    self.fails += 1
                    self.last_error = _err(exc)
                    wait = min(self.backoff[1], self.backoff[0] * 2 ** (self.fails - 1))
                    self.retry_at = now + wait
                self.log(f"status commit failed ({self.last_error}); retry in {wait:.0f} s")
                return False
            with self._mu:
                self.pending = [r for r in self.pending if r not in reasons]
                self.first_at = now if self.pending else None
                self.last_commit, self.retry_at, self.fails = now, 0.0, 0
                self.last_error = None
                self.commits += 1
            return True


def runner_guard(url: str, *, get=None, now=None, recent_h=GUARD_RECENT_H, log=_log,
                 sticky: bool = False):
    """None when the HF runner Space at `url` does not run its own tick, else
    the reason. Its /api/runner_status: tick_source "remote" (EVENT_TICK off,
    it shows OUR status) is clear; a running tick or an attempt within
    recent_h is not (code before tick_source: EVENT_TICK=0 restarts the Space,
    so last_attempt is None there). Unreadable (down, restarting, not a JSON
    object): clear - a Space that cannot be reached cannot receive the
    updater's tick ping either - unless `sticky` (the guard blocked the last
    tick): a Space restarting in the middle of its own ticking reads the
    same, so then only a clear read lifts the block."""
    if not url:
        return None
    try:
        if get is None:
            import requests
            get = requests.get
        r = get(f"{url}/api/runner_status", timeout=(10, 20))
        r.raise_for_status()
        body = r.json()
        if not isinstance(body, dict):
            raise ValueError("not a JSON object")
    except Exception as exc:
        if sticky:
            return (f"the HF runner ({url}) is unreadable ({_err(exc)}) after it blocked a "
                    f"tick: waiting for one clear read")
        log(f"guard: {url} unreadable ({_err(exc)}); not blocking")
        return None
    if body.get("tick_source") == "remote":
        return None
    if body.get("tick_source") == "local":
        # that Space's EVENT_TICK is 1: it ticks on the next updater ping,
        # even right after a restart (empty memory, no last_attempt yet)
        return (f"the HF runner ({url}) runs its own tick (tick_source local): "
                f"set EVENT_TICK=0 on that Space first")
    t = body.get("tick") or {}
    if t.get("running"):
        return f"the HF runner ({url}) is running its own tick (phase {t.get('phase')})"
    last = _epoch(t.get("last_attempt"))
    now = time.time() if now is None else now
    if last is not None and now - last <= recent_h * 3600:
        return (f"the HF runner ({url}) ticked {(now - last) / 3600:.1f} h ago: "
                f"set EVENT_TICK=0 on that Space first")
    return None


def finalize_threads() -> list:
    """Live threads running eventstore.finalize_episodes (the tick dispatches
    them: scoring, then one commit)."""
    from hf_data import eventstore
    return [t for t in threading.enumerate() if t.is_alive()
            and getattr(t, "_target", None) is eventstore.finalize_episodes]


def janitor(log=_log, ttl_h: float | None = None) -> dict:
    """The Spaces' hourly cache janitor (datamgr: run work dirs, LRU caps on
    forcing/frames/states, hub-cache eviction, result compaction), plus the
    event_* EF5 work dirs eventsim's run_one leaves in TMPDIR. Between ticks
    only, so nothing it removes is in use."""
    from hf_data import datamgr
    ttl_h = datamgr.WORKDIR_TTL_H if ttl_h is None else ttl_h
    rep = {}
    try:
        rep = datamgr.cleanup()
        rep["compact"] = datamgr.compact_results()
    except Exception as exc:
        rep["error"] = _err(exc)
    cutoff, n = time.time() - ttl_h * 3600, 0
    root = tempfile.gettempdir()
    for name in os.listdir(root):
        p = os.path.join(root, name)
        try:
            if name.startswith("event_") and os.path.isdir(p) and os.path.getmtime(p) < cutoff:
                shutil.rmtree(p, ignore_errors=True)
                n += 1
        except OSError:
            pass
    rep["event_workdirs"] = n
    log(f"janitor: {json.dumps(rep, default=str)[:400]}")
    return rep


def _summary(result) -> dict:
    if not isinstance(result, dict):
        return {}
    out = {k: result[k] for k in ("ran", "ended", "swept", "capacity_blocked", "skipped")
           if k in result}
    out["results"] = list(result.get("results") or [])[:10]
    return out


class Daemon:
    """The tick scheduler over injected parts (the tests fake them): tick()
    -> hourly_tick's result; issue() -> the nowcast issue datetime or None;
    guard(sticky) -> None or a reason not to tick (sticky: the guard blocked
    the last tick); tick_status() -> eventsim's progress dict;
    note_attempt() -> eventsim.note_tick_attempt; clock() -> epoch s; stop ->
    the stop request (no tick starts once it is set); exit_fn(rc) ends the
    process (a wedged tick); notify(msg) -> systemd's notify socket;
    dump() -> every thread's stack to the journal."""

    def __init__(self, *, tick=None, issue=None, guard=None, tick_status=None,
                 note_attempt=None, publisher=None, clock=time.time, state_path=None,
                 janitor=None, stop=None, exit_fn=None, notify=None, dump=None,
                 max_gap_s=MAX_GAP_MIN * 60, poll_s=POLL_S, grace_s=STOP_GRACE_S,
                 max_busy_s=MAX_BUSY_H * 3600, beat_s=BEAT_MIN * 60,
                 idle_beat_s=IDLE_BEAT_MIN * 60, janitor_s=JANITOR_S, loop_s=LOOP_S,
                 final_s=FINAL_PUBLISH_S, retry_pause_s=FINAL_RETRY_PAUSE_S, log=_log):
        self.tick_fn = tick or eventsim.hourly_tick
        self.issue_fn = issue
        self.guard_fn = guard
        self.tick_status_fn = tick_status or eventsim.tick_status
        self.note_attempt_fn = note_attempt or eventsim.note_tick_attempt
        self.clock, self.log, self.state_path = clock, log, state_path
        self.janitor_fn = janitor
        self.stop = stop if stop is not None else StopEvent()
        self.exit_fn = exit_fn or _exit_now
        self.notify = notify or sd_notify
        self.dump_fn = dump or dump_stacks
        self.max_gap_s, self.poll_s, self.grace_s = float(max_gap_s), float(poll_s), float(grace_s)
        self.max_busy_s = float(max_busy_s)
        self.beat_s, self.idle_beat_s = float(beat_s), float(idle_beat_s)
        self.janitor_s, self.loop_s = float(janitor_s), float(loop_s)
        self.final_s, self.retry_pause_s = float(final_s), float(retry_pause_s)
        self.publisher = publisher or Publisher(self.snapshot, tickstatus.publish,
                                                clock=clock, log=log)
        self.state = "idle"
        self.started = clock()
        self.last_issue = None
        self.last_start = self.last_end = None
        self.last_outcome = self.last_result = self.last_why = None
        self.ticks = 0
        self.blocked = None
        self.busy_attempts = 0
        self._busy_issue = self._busy_at = None
        self._thread = None
        self._done = threading.Event()
        self._result = self._error = None
        self._next_poll = 0.0
        self._beat_at = None
        self._stalled = False
        self._janitor_at, self._janitor_due = 0.0, False
        self._load_state()

    # ---- persisted schedule ------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path:
            return
        try:
            with open(self.state_path, encoding="utf-8") as fh:
                st = json.load(fh)
        except (OSError, ValueError):
            return
        self.last_issue = _issue_dt(st.get("last_issue"))
        self.last_start = _epoch(st.get("last_tick_started"))
        self.last_end = _epoch(st.get("last_tick_ended"))
        self.last_outcome = st.get("last_outcome")
        self.blocked = st.get("guard_blocked") or None     # a block is sticky
        if self.last_start is not None and self.last_start > self.clock():
            self.last_start = self.clock()     # a clock that went back: no endless wait

    def _save_state(self) -> None:
        if not self.state_path:
            return
        st = {"last_issue": _issue_str(self.last_issue),
              "last_tick_started": _iso(self.last_start),
              "last_tick_ended": _iso(self.last_end), "last_outcome": self.last_outcome,
              "guard_blocked": self.blocked}
        try:
            d = os.path.dirname(os.path.abspath(self.state_path))
            fd, tmp = tempfile.mkstemp(prefix=".event_tick_daemon.", dir=d)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(st, fh)
            os.replace(tmp, self.state_path)
        except OSError as exc:
            self.log(f"state file not written ({_err(exc)})")

    # ---- status --------------------------------------------------------------
    def info(self) -> dict:
        p = self.publisher
        return {"state": self.state, "started": _iso(self.started),
                "last_issue": _issue_str(self.last_issue),
                "last_tick_started": _iso(self.last_start),
                "last_tick_ended": _iso(self.last_end), "last_why": self.last_why,
                "last_outcome": self.last_outcome, "last_result": self.last_result,
                "ticks": self.ticks, "blocked": self.blocked,
                "busy_attempts": self.busy_attempts,
                "last_busy_attempt": _iso(self._busy_at),
                "max_gap_min": round(self.max_gap_s / 60, 1),
                "max_busy_h": round(self.max_busy_s / 3600, 2),
                "status_commits": p.commits, "status_error": p.last_error}

    def snapshot(self, reasons=()) -> dict:
        return tickstatus.build(self.info(), reasons)

    # ---- scheduling ------------------------------------------------------------
    def due(self, now: float, issue) -> str | None:
        """Why a tick is due now, or None."""
        return _why_due(now, issue, self.last_issue, self.last_start, self.max_gap_s)

    def ticking(self) -> bool:
        return self._thread is not None

    def _issue(self):
        if self.issue_fn is None:
            return None
        try:
            return self.issue_fn()
        except Exception as exc:
            self.log(f"nowcast issue unreadable ({_err(exc)})")
            return None

    def _guard(self) -> str | None:
        """The guard's reason not to tick, or None. A block is sticky: while
        self.blocked is set, the guard lifts it only on a clear read."""
        if self.guard_fn is None:
            return None
        try:
            return self.guard_fn(bool(self.blocked))
        except Exception as exc:
            if self.blocked:
                return (f"guard failed ({_err(exc)}) after it blocked a tick: waiting for "
                        f"one clear read")
            self.log(f"guard failed ({_err(exc)}); not blocking")
            return None

    def _start(self, now: float, issue, why: str) -> bool:
        reason = self._guard()
        if self.stop.is_set():                 # the signal came during the guard read
            self.log(f"tick due ({why}) but NOT started: stop requested")
            return False
        if reason:
            first = not self.blocked
            self.blocked = reason
            if first:                         # one log line and commit per blocked spell
                self.log(f"tick due ({why}) but NOT started: {reason}")
                self._save_state()            # sticky across restarts
                self.publisher.request("blocked")
            return False
        if self.blocked:
            self.log("guard clear: ticks resume" if self.guard_fn is not None
                     else "no guard configured (EVENT_TICK_GUARD_URL): the saved block is lifted")
            self.blocked = None
        self.last_start, self.last_why = now, why
        if issue is not None:
            self.last_issue = issue
        self._busy_issue = self._busy_at = None
        self.ticks += 1
        self.state = "ticking"
        self._save_state()
        self._done.clear()
        self._result = self._error = None
        self._stalled = False
        self._thread = threading.Thread(target=self._body, name="event-tick", daemon=True)
        self._thread.start()
        self._beat_at = now + self.beat_s
        self.log(f"tick #{self.ticks} started ({why}; nowcast issue {_issue_str(issue)})")
        self.publisher.request("tick_start")
        return True

    def _body(self) -> None:
        try:
            self._result = self.tick_fn()
        except BaseException as exc:            # recorded; the loop goes on
            self._error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            self._done.set()

    def _reap(self, now: float) -> bool:
        if self._thread is None or not self._done.is_set():
            return False
        self._thread.join(timeout=1.0)
        self._thread = None
        self.last_end = now
        if self._error:
            self.last_outcome, self.last_result = "exception", {"error": self._error}
        else:
            res = self._result if isinstance(self._result, dict) else {}
            try:
                outcome = self.tick_status_fn().get("outcome")
            except Exception:
                outcome = None
            self.last_outcome = "skipped" if res.get("skipped") else (outcome or "ok")
            self.last_result = _summary(res)
        if self.state == "ticking":
            self.state = "idle"
        self._save_state()
        took = now - (self.last_start or now)
        self.log(f"tick #{self.ticks} {self.last_outcome} after {took:.0f} s: "
                 f"{json.dumps(self.last_result, default=str)[:600]}")
        self.publisher.request("tick_end")
        return True

    def _note_busy(self, now: float, issue) -> bool:
        """A tick fell due while the previous one still runs: none starts
        (one at a time), but the attempt is stamped as the HF runner's busy
        branch did (eventsim.note_tick_attempt), so tick.last_attempt keeps
        saying "the schedule is alive" and only busy_h past HEALTH_WEDGE_MAX_H
        reads as wedged. Once per issue advance or max gap, like a start."""
        last_issue = self.last_issue if self._busy_issue is None else self._busy_issue
        last_at = self.last_start if self._busy_at is None else self._busy_at
        why = _why_due(now, issue, last_issue, last_at, self.max_gap_s)
        if not why:
            return False
        self._busy_at = now
        if issue is not None and (last_issue is None or issue > last_issue):
            self._busy_issue = issue
        self.busy_attempts += 1
        try:
            self.note_attempt_fn()
        except Exception as exc:
            self.log(f"tick attempt not noted ({_err(exc)})")
        self.log(f"tick due ({why}; nowcast issue {_issue_str(issue)}) while tick #{self.ticks} "
                 f"runs ({(now - (self.last_start or now)) / 3600:.1f} h): not started, "
                 f"attempt noted")
        return True

    def step(self, now: float | None = None) -> None:
        """One pass of the main loop. No tick starts once stop is set."""
        now = self.clock() if now is None else now
        self.notify("WATCHDOG=1")
        if self._reap(now):
            self._janitor_due = True
        if self._thread is None:
            if now >= self._next_poll and not self.stop.is_set():
                self._next_poll = now + self.poll_s
                issue = self._issue()
                why = self.due(now, issue)
                if why:
                    self._start(now, issue, why)
            if (self._thread is None and self._janitor_due and self.janitor_fn
                    and not self.stop.is_set()
                    and now >= self._janitor_at and not finalize_threads()):
                self._janitor_due, self._janitor_at = False, now + self.janitor_s
                try:
                    self.janitor_fn()
                except Exception as exc:
                    self.log(f"janitor failed ({_err(exc)})")
            p = self.publisher
            if (self._thread is None and not p.pending and p.last_commit is not None
                    and now - p.last_commit >= self.idle_beat_s):
                p.request("alive")
        else:
            busy = now - (self.last_start if self.last_start is not None else now)
            if self.max_busy_s > 0 and busy >= self.max_busy_s:
                self._wedged(busy)
                return
            if now >= self._next_poll and not self.stop.is_set():
                self._next_poll = now + self.poll_s
                self._note_busy(now, self._issue())
            if now >= self._beat_at:
                self._beat_at = now + self.beat_s
                self.publisher.request("heartbeat")
            try:
                stalled = bool(self.tick_status_fn().get("stalled"))
            except Exception:
                stalled = False
            if stalled and not self._stalled:
                self.log("tick stalled in a wait phase (eventsim watchdog dumped the stacks)")
                self.publisher.request("stalled")
            self._stalled = stalled
        self.publisher.pump(now)

    def _wedged(self, busy: float) -> None:
        """The running tick passed EVENT_TICK_MAX_BUSY_H: nothing inside it
        ended it (EF5's run-time and stall caps, the Hub and GDAL timeouts),
        so a thread is stuck where no timeout reaches. Its stacks go to the
        journal, "wedged" to the status file (bounded), and the process
        exits EXIT_WEDGED: systemd restarts the daemon, KillMode=mixed reaps
        a leftover EF5 child, and the saved schedule keeps the issue."""
        try:
            phase = self.tick_status_fn().get("phase")
        except Exception:
            phase = None
        self.state = "wedged"
        self.last_outcome = "wedged"
        self.last_result = {"busy_h": round(busy / 3600, 2), "phase": phase}
        self._save_state()
        self.log(f"tick #{self.ticks} WEDGED: busy {busy / 3600:.2f} h, past "
                 f"EVENT_TICK_MAX_BUSY_H={self.max_busy_s / 3600:g} (phase {phase}); thread "
                 f"stacks follow, then the 'wedged' status and exit {EXIT_WEDGED} for "
                 f"systemd to restart the daemon")
        self.dump_fn()
        self._final_publish("wedged")
        self.exit_fn(EXIT_WEDGED)

    # ---- run / stop --------------------------------------------------------------
    def run(self, stop: threading.Event | None = None,
            hurry: threading.Event | None = None) -> int:
        if stop is not None:
            self.stop = stop
        self.log(f"started: max gap {self.max_gap_s / 60:g} min, poll {self.poll_s:g} s, "
                 f"stop grace {self.grace_s:g} s, max busy {self.max_busy_s / 3600:g} h; "
                 f"last issue {_issue_str(self.last_issue)}, last tick started "
                 f"{_iso(self.last_start)}" + (f"; guard block kept: {self.blocked}"
                                               if self.blocked else ""))
        self.publisher.request("start")
        while not self.stop.is_set():
            self.step()
            self.stop.wait(self.loop_s)
        return self.shutdown(hurry)

    def _bounded_pump(self, until: float, force: bool = False):
        """publisher.pump in a thread, waited for until `until` (monotonic):
        a hung commit cannot hold the stop past its budget. The pump's
        result; None when it is still running."""
        out = []
        th = threading.Thread(target=lambda: out.append(self.publisher.pump(force=force)),
                              name="event-tick-status", daemon=True)
        th.start()
        th.join(max(0.0, until - time.monotonic()))
        return out[0] if out else None

    def _final_publish(self, reason: str) -> bool:
        """The last status commit ("stop", "wedged"), forced: an attempt and,
        when it fails, one retry retry_pause_s later, all within final_s (part
        of the unit's TimeoutStopSec). An attempt still hanging at the bound
        is abandoned: the process exits under it."""
        p = self.publisher
        p.request(reason)
        end = time.monotonic() + self.final_s
        for attempt in (1, 2):
            ok = self._bounded_pump(end, force=True)
            if ok or (ok is False and not p.pending):    # committed (here or by a pump before)
                return True
            if ok is None:
                self.log(f"final status commit ({reason}) still running after "
                         f"{self.final_s:g} s: exiting without it")
                return False
            if attempt == 2 or end - time.monotonic() <= self.retry_pause_s:
                break
            self.log(f"final status commit ({reason}) failed ({p.last_error}): one retry")
            time.sleep(self.retry_pause_s)
        self.log(f"final status commit ({reason}) failed ({p.last_error}): exiting without it")
        return False

    def shutdown(self, hurry: threading.Event | None = None) -> int:
        hurry = hurry or threading.Event()
        at = getattr(self.stop, "at", None)
        deadline = (time.monotonic() if at is None else at) + self.grace_s
        if self._thread is not None:
            try:
                phase = self.tick_status_fn().get("phase")
            except Exception:
                phase = None
            self.state = "stopping"
            left = max(0.0, deadline - time.monotonic())
            self.log(f"stop requested during tick #{self.ticks} (phase {phase}): "
                     f"waiting up to {left:.0f} s for it")
            self.publisher.request("stopping")
            while not self._done.is_set() and not hurry.is_set():
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._done.wait(min(self.loop_s, left))
                self.notify("WATCHDOG=1")
                if self.publisher.due(self.clock()):
                    self._bounded_pump(deadline)
            if not self._reap(self.clock()):
                self.log("grace over: exiting with the tick unfinished - its Hub commits "
                         "are atomic and the next tick redoes the visit")
        for th in finalize_threads():
            left = deadline - time.monotonic()
            if left <= 0 or hurry.is_set():
                break
            self.log(f"waiting up to {left:.0f} s for an episode finalize ({th.name})")
            th.join(left)
        self.state = "stopped"
        self._final_publish("stop")
        self.log(f"stopped ({self.ticks} tick(s) this run, "
                 f"{self.publisher.commits} status commit(s))")
        return 0


def install_signals(stop: threading.Event, hurry: threading.Event) -> None:
    """SIGTERM/SIGINT: the first stops after the running tick (within the
    grace), the second skips the rest of the grace."""
    def handler(signum, frame):
        # set from a helper thread: the main thread may be inside
        # Event.wait holding the Event's (non-reentrant) lock right now
        ev = hurry if stop.is_set() else stop
        threading.Thread(target=ev.set, daemon=True).start()
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, handler)


def acquire_host_lock(path: str):
    """The FileLock holding this host's daemon slot, or None when another
    daemon holds it."""
    from filelock import FileLock, Timeout
    lock = FileLock(path)
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return None
    return lock


def _hub_timeouts() -> str:
    try:
        from scripts import merge_episodes as me
    except ImportError:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import merge_episodes as me
    return me.bound_hub_timeouts()


def preflight() -> list[str]:
    """Why this process cannot tick (empty when it can): crestimap, which
    the Spaces' /api/event_tick also requires, and EF5 at ./EF5/bin/ef5
    (pipeline.run_gauge runs it from the working directory)."""
    bad = []
    try:
        import crestimap  # noqa: F401
    except ImportError as exc:
        bad.append(f"crestimap not importable ({exc})")
    if not os.access(os.path.join("EF5", "bin", "ef5"), os.X_OK):
        bad.append(f"no executable ./EF5/bin/ef5 under {os.getcwd()}")
    return bad


def check() -> int:
    """--check: everything short of a tick, read-only, no token needed."""
    ok = True
    print(_hub_timeouts())
    for name in ("numpy", "rasterio", "pyarrow", "osgeo.gdal", "torch", "crestimap",
                 "huggingface_hub"):
        try:
            mod = __import__(name, fromlist=["_"])
            print(f"import {name}: {getattr(mod, '__version__', 'ok')}")
        except Exception as exc:
            ok = False
            print(f"import {name}: FAILED ({_err(exc)})")
    for why in preflight():
        ok = False
        print(f"preflight: {why}")
    print(bounds_line())
    print(f"stop: grace {stop_grace(log=print):g} s + final status commit "
          f"{FINAL_PUBLISH_S:g} s within TimeoutStopSec {TIMEOUT_STOP_S:g} s")
    from hf_data import nowcaststore, provenance
    print(f"code: {provenance.stamp()}")
    print(f"nowcast issue t0: {_issue_str(nowcaststore.issue_t0())}")
    print(f"guard {GUARD_URL or '(none)'}: {runner_guard(GUARD_URL) or 'clear'}")
    doc, err = tickstatus.read_remote()
    print(f"{tickstatus.STATUS_FILE}: " + (f"host {doc.get('host')}, published "
          f"{doc.get('published_at')}, daemon {(doc.get('daemon') or {}).get('state')}"
          if doc else f"none ({err})"))
    print(f"HF_TOKEN in the environment: {'yes' if os.environ.get('HF_TOKEN') else 'no'}")
    print("check:", "OK" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="imports, EF5, nowcast/guard/status reads; no tick, no write")
    ap.add_argument("--no-publish", action="store_true",
                    help="run without committing events/tick_status.json")
    a = ap.parse_args(argv)
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    if a.check:
        return check()
    if not os.environ.get("HF_TOKEN"):
        _log("REFUSING TO START: HF_TOKEN is not set")
        return 2
    bad = preflight()
    if bad:
        _log("REFUSING TO START: " + "; ".join(bad))
        return 2
    state_dir = (os.environ.get("EVENT_TICK_STATE_DIR") or os.environ.get("CREST_CACHE_DIR")
                 or tempfile.gettempdir())
    os.makedirs(state_dir, exist_ok=True)
    lock = acquire_host_lock(os.path.join(state_dir, "event_tick_daemon.lock"))
    if lock is None:
        _log(f"REFUSING TO START: another daemon holds {state_dir}/event_tick_daemon.lock")
        return 2
    _log(_hub_timeouts())
    import faulthandler
    faulthandler.enable()        # the systemd watchdog's SIGABRT: every stack to the journal
    from hf_data import nowcaststore
    stop, hurry = StopEvent(), threading.Event()
    install_signals(stop, hurry)
    d = Daemon(issue=nowcaststore.issue_t0,
               guard=((lambda sticky: runner_guard(GUARD_URL, sticky=sticky))
                      if GUARD_URL else None),
               state_path=os.path.join(state_dir, "event_tick_daemon.json"),
               janitor=janitor, stop=stop, grace_s=stop_grace())
    d.publisher.enabled = not a.no_publish
    _log(f"host {tickstatus.host()}, CREST_AI {ROOT}, guard {GUARD_URL or '(none)'}, "
         f"status {'off' if a.no_publish else tickstatus.STATUS_FILE}")
    _log(bounds_line())
    return d.run(stop, hurry)


if __name__ == "__main__":
    _exit_now(main())

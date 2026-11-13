"""Always-on episode merger (PI 2026-11-03: the episode merge "has to run
automatically all the time", as a script on HF, not a step inside the hourly
tick).

On the runner Space only (EPISODE_MERGER=1; default off, so CREST_demo and
local runs never start it) the server runs scripts/merge_episodes.py --watch
--apply as a SEPARATE child process: its crashes, memory and slow merges stay
out of the tick, and it uses the Space's HF_TOKEN and env. A daemon
supervisor thread restarts the child whenever it exits (backoff 1 min
doubling to 30 min, back to 1 min after a child that ran 30 min). It polls
the child every POLL_S, and its watchdog treats a child whose status
heartbeat ('updated', written at least every --interval) is older than
max(HUNG_MIN_S, 6 x interval) as hung (a Hub call that never returns):
SIGTERM, SIGKILL after KILL_WAIT_S, restarted like an exit. Never two
children: one still alive after SIGKILL is polled, logged and killed again
every LINGER_LOG_S until it is gone. The child's stdout/stderr go to the
Space log (inherited); it writes its own status JSON (status_file(); while
that can't be written, fallback_status_file(), which the watchdog reads too;
when neither can, the child exits non-zero and is restarted), which status()
serves at GET /api/merger_status with {enabled, alive, pid, restarts,
last_exit} (+ start_error). At interpreter exit the child gets SIGTERM and
stops between groups, never mid-commit. start() never raises into
server.py: a failure is printed and reported. EPISODE_MERGER_ARGS: extra
script arguments, e.g. "--max-merges-per-hour 4" (not --status-file: the
status path is the supervisor's; start() refuses it).
"""
from __future__ import annotations

import atexit
import datetime
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "merge_episodes.py")
BACKOFF_MIN_S, BACKOFF_MAX_S = 60.0, 1800.0
STOP_WAIT_S = 8.0            # at exit: time the child gets to finish a commit
POLL_S = 30.0                # the supervisor looks at the child this often
HUNG_MIN_S = 7200.0          # watchdog: no heartbeat for max(this, 6 x interval) = hung
KILL_WAIT_S = 30.0           # watchdog: SIGTERM -> SIGKILL
LINGER_LOG_S = 600.0         # a child alive after SIGKILL: logged (and killed again) this often
INTERVAL_S = 300.0           # the script's --interval default


def enabled() -> bool:
    return os.environ.get("EPISODE_MERGER", "0").strip() == "1"


def status_file() -> str:
    """The script's default status path (passed to it explicitly too)."""
    return os.path.join(os.environ.get("CREST_CACHE_DIR") or tempfile.gettempdir(),
                        "merger_status.json")


def fallback_status_file() -> str:
    """Where the child writes its status while status_file() can't be
    written (the script's fallback_status_file)."""
    return os.path.join(tempfile.gettempdir(), "merger_status.fallback.json")


def _sets_status_file(arg: str) -> bool:
    """`arg` is --status-file (=value too), or an abbreviation argparse
    takes for it (--s ... --status-fil)."""
    opt = arg.split("=", 1)[0]
    return len(opt) >= 3 and "--status-file".startswith(opt)


def command() -> list:
    """The child's command line; ValueError naming EPISODE_MERGER_ARGS when
    shlex can't split it (an unclosed quote) or it sets --status-file (the
    watchdog reads the heartbeat at status_file(): another path would get a
    healthy child killed as hung)."""
    raw = os.environ.get("EPISODE_MERGER_ARGS", "")
    try:
        extra = shlex.split(raw)
    except ValueError as e:
        raise ValueError(f"EPISODE_MERGER_ARGS {raw!r} unparsable ({e})") from None
    bad = [a for a in extra if _sets_status_file(a)]
    if bad:
        raise ValueError(f"EPISODE_MERGER_ARGS {raw!r} sets {bad[0]}: the status file is "
                         f"the supervisor's ({status_file()}, from CREST_CACHE_DIR), its "
                         f"watchdog reads the heartbeat there; drop it")
    return [sys.executable, SCRIPT, "--watch", "--apply", "--status-file", status_file()] + extra


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(s) -> float | None:
    try:
        return datetime.datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except ValueError:
        return None


def _read(path) -> dict:
    try:
        with open(path, encoding="utf-8") as fp:
            st = json.load(fp)
    except (OSError, ValueError):
        return {}
    return st if isinstance(st, dict) else {}


def _child_status(paths, pid) -> dict:
    """Of the status files in `paths` (None skipped), the one `pid` wrote
    last ('updated'); {} when none is its."""
    best, at = {}, None
    for path in paths:
        st = _read(path) if path else {}
        if st.get("pid") != pid:
            continue
        t = _epoch(st.get("updated"))
        t = float("-inf") if t is None else t
        if at is None or t > at:
            best, at = st, t
    return best


class Supervisor:
    """Runs `cmd` as a child process and restarts it when it exits or its
    status heartbeat in `status_path` (None: no watchdog), or in
    `fallback_path` while the child can't write status_path, stops. Never
    two children: a child alive after its SIGKILL is waited for. popen,
    sleep, clock (monotonic) and wall (epoch, the status file's clock) are
    injectable (the tests run no real process)."""

    def __init__(self, cmd, popen=subprocess.Popen, sleep=None, clock=time.monotonic,
                 log=None, status_path=None, wall=time.time, fallback_path=None):
        self.cmd, self.popen, self.clock = list(cmd), popen, clock
        self.status_path, self.fallback_path, self.wall = status_path, fallback_path, wall
        self.stopping = threading.Event()
        self.sleep = sleep or self.stopping.wait
        self.log = log or (lambda s: print(s, flush=True))
        self.proc = None
        self.restarts = 0
        self.last_exit = None
        self.next_start_at = None
        self._lock = threading.Lock()

    def info(self) -> dict:
        with self._lock:
            p = self.proc
            return {"alive": p is not None and p.poll() is None,
                    "pid": p.pid if p is not None else None,
                    "restarts": self.restarts, "last_exit": self.last_exit,
                    "next_start_at": self.next_start_at}

    def run(self, runs=None) -> None:
        """Start, wait, back off, restart, until stop(); `runs` (tests): stop
        after that many children."""
        backoff, n = 0.0, 0
        while not self.stopping.is_set() and (runs is None or n < runs):
            if n:
                with self._lock:
                    self.restarts += 1
                    self.next_start_at = None
            n += 1
            t0, hung = self.clock(), None
            try:
                p = self.popen(self.cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
            except Exception as e:
                rc = f"start failed: {type(e).__name__}: {e}"
            else:
                with self._lock:
                    self.proc = p
                self.log(f"episode merger started (pid {p.pid})")
                rc, hung = self._wait(p, self.wall())
                if not self.stopping.is_set():
                    rc = self._gone(p, rc)   # never a new child while this one lives
            ran = self.clock() - t0
            with self._lock:
                self.last_exit = {"rc": rc, "at": _now(), "ran_s": round(ran, 1)}
                if hung:
                    self.last_exit["hung"] = hung
            if self.stopping.is_set():
                break
            # a child that ran a while was healthy: start the backoff over
            backoff = (BACKOFF_MIN_S if not backoff or ran >= BACKOFF_MAX_S
                       else min(BACKOFF_MAX_S, 2 * backoff))
            with self._lock:
                self.next_start_at = (datetime.datetime.now(datetime.timezone.utc)
                                      + datetime.timedelta(seconds=backoff)
                                      ).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.log(f"episode merger {'killed as hung' if hung else 'exited'} ({rc}) "
                     f"after {ran:.0f} s; restart in {backoff / 60:g} min")
            self.sleep(backoff)

    def _wait(self, p, started: float) -> tuple:
        """Poll the child every POLL_S until it exits: (rc, None); on stop():
        (rc or None, None). When hung(): SIGTERM, SIGKILL after KILL_WAIT_S:
        (rc, why)."""
        while True:
            rc = p.poll()
            if rc is not None or self.stopping.is_set():
                return rc, None
            try:
                why = self.hung(p.pid, started)
            except Exception:        # the watchdog must never end the supervision
                why = None
            if why:
                self.log(f"episode merger pid {p.pid} hung ({why}): terminating")
                return self._kill(p), why
            self.sleep(POLL_S)

    def hung(self, pid, started: float) -> str | None:
        """Why the child counts as hung, or None: the status file's 'updated'
        (only when it is this pid's, the later of status_path and
        fallback_path; else the child's start, wall clock) is older than
        max(HUNG_MIN_S, 6 x its --interval) and its state is not 'stopped'.
        No status_path: never."""
        if not self.status_path:
            return None
        st = _child_status((self.status_path, self.fallback_path), pid)
        if st.get("state") == "stopped":
            return None
        last = max(started, _epoch(st.get("updated")) or started)
        try:
            interval = float((st.get("config") or {}).get("interval_s") or INTERVAL_S)
        except (TypeError, ValueError, AttributeError):
            interval = INTERVAL_S
        limit = max(HUNG_MIN_S, 6 * interval)
        age = self.wall() - last
        if age <= limit:
            return None
        return f"no status heartbeat for {age / 60:.0f} min, limit {limit / 60:.0f}"

    def _kill(self, p):
        """SIGTERM (the child stops between groups if it can), SIGKILL after
        KILL_WAIT_S. The exit code, or why there is none."""
        for sig in ("terminate", "kill"):
            try:
                getattr(p, sig)()
                return p.wait(timeout=KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                if sig == "terminate":
                    self.log(f"episode merger pid {p.pid} ignored SIGTERM for "
                             f"{KILL_WAIT_S:g} s: killing")
            except Exception as e:
                return f"{sig} failed: {type(e).__name__}: {e}"
        return "still running after SIGKILL"

    def _gone(self, p, rc):
        """p's exit code once it is gone. A child still alive here (it
        outlived SIGKILL: stuck in the kernel, or a signal failed) is polled
        every POLL_S, logged and SIGKILLed again every LINGER_LOG_S, until it
        exits or stop() (then rc, what _wait returned): no new child starts
        beside it."""
        t0, said = self.clock(), None
        while True:
            try:
                done, err = p.poll(), ""
            except Exception as e:           # can't tell: count it alive
                done, err = None, f"; poll: {type(e).__name__}: {e}"
            if done is not None:
                if said is not None:
                    self.log(f"episode merger pid {p.pid} gone ({done}) after "
                             f"{self.clock() - t0:.0f} s more")
                return done
            if self.stopping.is_set():
                return rc
            now = self.clock()
            if said is None or now - said >= LINGER_LOG_S:
                said = now
                self.log(f"episode merger pid {p.pid} not gone ({rc}{err}; {now - t0:.0f} s "
                         f"on): SIGKILL again; no new child until it is")
                try:
                    p.kill()
                except Exception:
                    pass
            self.sleep(POLL_S)

    def stop(self, wait_s: float = STOP_WAIT_S) -> None:
        """No more restarts; SIGTERM the child, which stops between groups."""
        self.stopping.set()
        with self._lock:
            p = self.proc
        if p is None or p.poll() is not None:
            return
        try:
            p.terminate()
            p.wait(timeout=wait_s)
        except Exception:
            pass


_sup: Supervisor | None = None
_start_lock = threading.Lock()
_start_error: dict | None = None


def start() -> bool:
    """Idempotent; no-op unless EPISODE_MERGER=1. Never raises (server.py
    calls it at import): a failure is printed in one line, kept for status()
    and returns False."""
    global _sup, _start_error
    try:
        if not enabled():
            return False
        with _start_lock:
            if _sup is None:
                sup = Supervisor(command(), status_path=status_file(),
                                 fallback_path=fallback_status_file())
                threading.Thread(target=sup.run, name="episode-merger", daemon=True).start()
                _sup, _start_error = sup, None
                atexit.register(sup.stop)
        return True
    except Exception as e:
        _start_error = {"at": _now(), "error": f"{type(e).__name__}: {e}"[:500]}
        print(f"episode merger NOT started: {_start_error['error']}", flush=True)
        return False


def status() -> dict:
    """GET /api/merger_status: the child's status file (last pass, waiting
    groups, recent merges, last error; the fallback file instead while the
    running child writes there, its status_fallback says why) + {enabled,
    alive, pid, restarts, last_exit} from the supervisor, + start_error when
    start() failed."""
    try:
        with open(status_file(), encoding="utf-8") as fp:
            st = json.load(fp)
        st = st if isinstance(st, dict) else {}
    except FileNotFoundError:
        st = {}
    except Exception as e:
        st = {"status_error": f"{type(e).__name__}: {e}"}
    sup = _sup.info() if _sup is not None else {"alive": False, "pid": None,
                                                 "restarts": 0, "last_exit": None}
    if sup.get("pid") is not None:
        fb = _child_status([fallback_status_file()], sup["pid"])
        if fb and (st.get("pid") != sup["pid"]
                   or (_epoch(fb.get("updated")) or 0) > (_epoch(st.get("updated")) or 0)):
            st = fb
    out = {**st, "enabled": enabled(), **sup}
    if _start_error:
        out["start_error"] = _start_error
    return out
